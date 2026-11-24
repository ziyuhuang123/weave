#!/usr/bin/env python3
"""Two-node 16-rank benchmark: rccl_aiter vs TurboOverlapArm (mode 0/1/2).

Arms
----
* ``rccl_aiter``       official in-process RCCL all-to-all + aiter fused_moe
                       (moe_mode="l0"), the same baseline arm run_live_v5 uses.
* ``turbo_mode{0,1,2}` the EP16 512-thread ``4+4`` spatial-overlap kernel
                       ``stage_ep16_turbo_overlap`` (grid 80, block 512)
                       launched through the host adapter
                       ``libep16_turbo_host.so`` + ``TurboOverlapArm``.

Discipline inherited from the v5 live runner:
* exactly one global S per torchrun process (arena/QP teardown per cell);
* rccl_aiter runs first and every arm is checked against the independent FP64
  reference on EVERY element of the rank-local [t_local, hidden] output
  before any timing sample is taken;
* timing uses CUDA events recorded on the same stream as the measured work,
  in one forward and one reverse arm order, with global barriers between
  arms placed OUTSIDE the recorded event interval.

Turbo hard constraints (TURBO_ADAPTER_REPORT.md section 6) honoured here:
1. ``x`` is PRE-QUANTIZED FP8 e4m3fnuz [t_local][hidden] plus f32
   per-(1,128) ``scale_groups``.  The quantization decision copies
   ``ep_reference._quantize_act_per_128`` exactly (amax / 240, zero-amax
   rows get scale 1) so the measured path and the FP64 reference consume the
   same FP8 codes.
2. ``node_tokens`` is globally uniform (== t_local on this uniform-S grid) and
   >= every rank's t_local.
3. ``generation`` is strictly increasing, nonzero and identical on all 16
   ranks: a per-arm deterministic ledger plus dist barriers keep the sixteen
   host sequences in lock-step.
4. ``capacity`` = t_local (worst case: every local token lands one bucket on
   one destination); ``route_rows`` = WORLD*capacity*topk + 16*e_local
   (+ t_local*num_shared_experts when the v59 shared-experts ABI is used);
   ``sort_stats[1] <= route_rows`` is asserted after every launch.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _project_paths() -> tuple[Path, Path]:
    here = Path(__file__).resolve().parent
    project_root = here.parent
    for path in (project_root, project_root / "baseline"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return project_root, here


PROJECT_ROOT, EP16_DIR = _project_paths()
def weave_env(name: str, legacy: str | None = None, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if (value is None or value == "") and legacy:
        value = os.environ.get(legacy)
    return default if value is None or value == "" else value


def weave_int(name: str, legacy: str | None = None, default: int = 0) -> int:
    return int(weave_env(name, legacy, str(default)))



import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from ep_common import (  # noqa: E402
    BLOCK,
    FP8_MAX,
    S_VALUES,
    load_route_manifest,
    load_route_rank,
    model_spec,
    save_json,
)
from ep_harness import _load_weight_bundle, _make_buffers  # noqa: E402
from ep_reference import compute_reference, full_compare  # noqa: E402
from paper_step0 import run_official  # noqa: E402
from ep16.abba_timing import OFFICIAL_MODE, _event_metrics_us, _official_events  # noqa: E402
from ep16.mega_host import (  # noqa: E402
    GdaLibrary,
    MegaIpc,
    sha256_file,
    validate_service_id_base,
)
import ep16_turbo_runner as T  # noqa: E402


WORLD = 16
TURBO_MODES = tuple(int(x) for x in (weave_env("WEAVE_TURBO_MODES", "EP16_TURBO_MODES", "2") or "2").split(",") if x.strip())
REL_L2_GATE = 0.02
MAX_ROW_GATE = 0.05
SEED_BASE = 20261104
TIMED_ROUNDS = 2
TURBO_TIMEOUT_TICKS = 1_000_000_000


def _gpu_sync(sync_buf: torch.Tensor, rccl_group: Any, device: torch.device) -> None:
    torch.cuda.synchronize(device)
    dist.all_reduce(sync_buf, group=rccl_group)
    torch.cuda.synchronize(device)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.dtype):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _append_live_log(path: Path, record: Mapping[str, Any], rank: int) -> None:
    if int(rank) != 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), ensure_ascii=False,
                                default=_json_default, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


# ---------------------------------------------------------------------------
# turbo input preparation
# ---------------------------------------------------------------------------
def _prequant_fp8(x_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-quantize activations exactly like ep_reference._quantize_act_per_128.

    Returns (fp8 codes viewed as uint8 [t_local][hidden], f32 scales
    [t_local][hidden/128]).  The device entry guard requires
    payload_bytes == hidden and scale_groups == hidden/128, i.e. exactly this
    layout; the BF16 register-quantizing branch of the kernel is unreachable
    in this code object.
    """
    rows, hidden = int(x_bf16.shape[0]), int(x_bf16.shape[1])
    if hidden % BLOCK:
        raise ValueError(f"hidden={hidden} is not a multiple of {BLOCK}")
    blocks = x_bf16.float().reshape(rows, hidden // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1)
    scale = amax / float(FP8_MAX)
    scale = torch.where(amax == 0, torch.ones_like(scale), scale)
    codes = (blocks / scale[..., None]).to(torch.float8_e4m3fnuz)
    return (codes.reshape(rows, hidden).view(torch.uint8).contiguous(),
            scale.contiguous())


def _adaptive_slot_chunk(t_local: int) -> int:
    """Bulk-coalesced DISPATCH descriptor granularity.

    One descriptor now covers `slot_chunk` CONTIGUOUS bucket slots of a single
    peer, so a launch posts ~8 descriptors per QP instead of 2 * t_local.  The
    value targets ~8 chunks per QP on the whole paper grid while keeping the
    first chunk of a bucket postable after ~1/8 of its rows are staged.  The
    host clamps it to `capacity` and the plan validates power-of-two 1..64.
    """
    override = weave_int("WEAVE_SLOT_CHUNK", "EP16_SLOT_CHUNK", 0)
    if override:
        return max(1, min(64, override))
    k = 4
    while k * 8 < int(t_local) and k < 64:
        k *= 2
    return k


def _expert_map(n_experts: int, e_local: int) -> list[int]:
    """uint32 [n_experts] table: (dst_rank<<16) | local_id."""
    if n_experts != WORLD * e_local:
        raise ValueError(f"n_experts={n_experts} != {WORLD}*{e_local}")
    return [((e // e_local) << 16) | (e % e_local) for e in range(n_experts)]


class GenerationLedger:
    """Strictly increasing, nonzero, identical-on-all-ranks generation source.

    Every rank executes the same deterministic launch sequence (correctness
    probes, warmup, then forward/reverse rounds), and dist barriers separate
    the phases, so taking the next integer locally keeps all 16 ranks on the
    same value.
    """

    def __init__(self, start: int = 1) -> None:
        self._next = int(start)

    def take(self) -> int:
        value = self._next
        self._next += 1
        return value


# ---------------------------------------------------------------------------
# turbo arm construction
# ---------------------------------------------------------------------------
def _build_turbo_arm(*, name: str, mode: int, rank: int, ctx: Mapping[str, Any],
                     topk_ids_cpu, expert_map_cpu: Sequence[int],
                     tensors: Mapping[str, torch.Tensor],
                     device: torch.device, svcid_base: int,
                     host_lib: Path, co_path: Path, gda_path: Path,
                     timeout_ticks: int = TURBO_TIMEOUT_TICKS) -> dict[str, Any]:
    include_self = True
    spec = model_spec(str(ctx["model"]))
    n_experts = int(spec.experts)
    e_local = n_experts // WORLD
    t_local = int(ctx["t_local"])
    topk = int(ctx["topk"])
    capacity = t_local
    node_tokens = t_local
    route_rows = WORLD * capacity * topk + 16 * e_local

    shape = T.Ep16TurboShape()
    shape.rank, shape.world, shape.ranks_per_node, shape.channels = (
        int(rank), T.WORLD, T.RANKS_PER_NODE, T.CHANNELS)
    shape.hidden, shape.intermediate = int(ctx["hidden"]), int(ctx["intermediate"])
    shape.topk, shape.n_experts = topk, n_experts
    shape.capacity, shape.t_local = capacity, t_local
    shape.node_tokens, shape.mode = node_tokens, int(mode)
    shape.route_rows, shape.slot_chunk = route_rows, _adaptive_slot_chunk(t_local)
    # One metadata descriptor per peer keeps the per-QP WRITE budget flat
    # (cap*64 B <= 128 KiB for every point of the paper grid).
    shape.meta_chunk_slots = max(
        weave_int("WEAVE_META_CHUNK_SLOTS", "META_CHUNK_SLOTS", 256), t_local)
    shape.reserved[0] = weave_int("WEAVE_RET_CHUNK", "EP16_RET_CHUNK", 0)
    shape.flags = 1 if include_self else 0
    shape.enable_dispatch_time = 0
    shape.enable_scatter_time = 0

    host = T.TurboHostLibrary(host_lib)
    ipc = MegaIpc()
    gda = GdaLibrary(path=Path(gda_path))
    try:
        arm = T.TurboOverlapArm(
            name=name, host=host, co_path=str(co_path), ipc=ipc, gda=gda,
            shape=shape, gpu=int(device.index or 0),
            nic_index=(int(rank) % T.RANKS_PER_NODE) // 2,
            node_id_base=(int(rank) // T.RANKS_PER_NODE) * T.RANKS_PER_NODE,
            svcid_base=int(svcid_base),
            stream=int(torch.cuda.current_stream(device).cuda_stream),
            timeout_ticks=int(timeout_ticks), torch_module=torch)
        arm.plan()
        arm.allocate()
        arm.gda_open()
        record = arm.local_exchange_record()
        gathered: list[Any] = [None] * WORLD
        dist.all_gather_object(gathered, record)
        arm.connect(gathered)

        node_expected = node_rank_mask = node_send_expect = None
        if mode == 2:
            my_mask = T.token_dst_mask(topk_ids_cpu, expert_map_cpu,
                                       rank=rank, include_self=include_self)
            all_masks: list[Any] = [None] * WORLD
            dist.all_gather_object(all_masks, my_mask)
            my_node = int(rank) // T.RANKS_PER_NODE
            owned_source = T.canonical_xnode_ranks(int(rank))[int(rank) % T.RANKS_PER_NODE]
            node_expected, node_rank_mask = T.node_plan_from_masks(
                all_masks[owned_source], my_node, node_tokens)
            node_send_expect = T.node_send_expect_from_mask(
                my_mask, my_node, node_tokens)

        external = T.Ep16TurboExternal()
        external.x = int(tensors["x_fp8"].data_ptr())
        external.x_scale = int(tensors["x_scale"].data_ptr())
        external.topk_ids = int(tensors["topk_ids"].data_ptr())
        external.topk_weights = int(tensors["topk_weights"].data_ptr())
        external.expert_map = int(tensors["expert_map"].data_ptr())
        external.ck_b = int(tensors["w1_fp8"].data_ptr())
        external.ck_b_scale = int(tensors["w1_scale"].data_ptr())
        external.ck_w2 = int(tensors["w2_fp8"].data_ptr())
        external.ck_w2_scale = int(tensors["w2_scale"].data_ptr())
        external.final_output = int(tensors["out_bf16"].data_ptr())
        arm.finalize(external=external, node_expected=node_expected,
                     node_rank_mask=node_rank_mask,
                     node_send_expect=node_send_expect, load_module=True)
    except Exception:
        try:
            gda.close()
        except Exception:
            pass
        raise

    ledger = GenerationLedger()
    live = arm.as_live_arm(generation_ledger=ledger)
    live["schema"] = "ep16-turbo512-live-arm/1"
    live["co_path"] = str(Path(co_path).resolve())
    live["co_sha256"] = sha256_file(Path(co_path))
    live["host_lib_path"] = str(Path(host_lib).resolve())
    live["host_lib_sha256"] = sha256_file(Path(host_lib))
    live["gda_lib_path"] = str(Path(gda_path).resolve())
    live["gda_lib_sha256"] = sha256_file(Path(gda_path))
    live["svcid_range"] = validate_service_id_base(int(svcid_base))
    live["kernel_attributes"] = arm.kernel_attributes()
    return live


def _turbo_launch_state(arm_record: Mapping[str, Any], *, phase: str,
                        host_generation: int) -> dict[str, Any]:
    arm = arm_record["turbo"]
    local_experts = int(arm.shape.n_experts) // T.WORLD
    return {
        "schema": "ep16-turbo512-launch-state/1",
        "phase": str(phase),
        "rank": arm.rank,
        "host_generation": int(host_generation),
        "device_error": arm.device_error(),
        "plan_flags": arm.read_u32_region("PLAN_FLAGS", 8),
        "bucket_count": arm.read_u32_region("BUCKET_COUNT", T.WORLD),
        "source_bucket_rows": arm.read_u32_region("SOURCE_BUCKET_ROWS", T.WORLD),
        "num_valid_ids": arm.read_u32_region("NUM_VALID_IDS", 2),
        "sort_stats": arm.read_u32_region("SORT_STATS", 4),
        "compute_done": arm.read_u32_region("COMPUTE_DONE", 1),
        "ck_max_token_id": arm.read_u32_region("CK_MAX_TOKEN_ID", local_experts + 2),
        "plan_audit": arm.plan_audit(),
        "gda": arm.gda.report(require_pass=False),
    }


def _assert_route_rows_fit(arm_record: Mapping[str, Any]) -> dict[str, int]:
    arm = arm_record["turbo"]
    stats = arm.read_u32_region("SORT_STATS", 4)
    padded = int(stats[1])
    if padded > int(arm.shape.route_rows):
        raise RuntimeError(
            f"ck_route_rows={int(arm.shape.route_rows)} is too small: the "
            f"planner needed {padded} padded rows; raise route_rows and "
            "rebuild the arm")
    return {"padded_rows": padded, "route_rows": int(arm.shape.route_rows)}


def _check_correctness(actual: torch.Tensor, expected: torch.Tensor, *,
                       rank: int, arm_name: str, cell_dir: Path) -> dict[str, Any]:
    verify_dir = cell_dir / "correctness" / arm_name
    verify_dir.mkdir(parents=True, exist_ok=True)
    local = full_compare(actual, expected, rank, None, None,
                         rel_l2_gate=REL_L2_GATE, max_row_gate=MAX_ROW_GATE,
                         save_arrays=False)
    save_json(verify_dir / f"rank{rank}.json", local)
    gathered: list[Any] = [None] * WORLD
    dist.all_gather_object(gathered, local)
    per_rank = [row if isinstance(row, dict) else {} for row in gathered]
    rank_ids = [int(row.get("rank", -1)) for row in per_rank]
    complete = len(per_rank) == WORLD and set(rank_ids) == set(range(WORLD))
    all_elements = complete and all(
        row.get("all_elements_checked") == int(actual.numel()) and
        row.get("all_elements_compared") is True
        for row in per_rank)
    fp64 = complete and all(
        row.get("fp64_reference") is True and
        row.get("reference_dtype") == "torch.float64" and
        row.get("comparison_dtype") == "float64"
        for row in per_rank)
    passed = bool(complete and all_elements and fp64 and all(
        row.get("correct") is True and row.get("finite") is True
        for row in per_rank))
    receipt = {
        "schema": "ep16-turbo512-correctness/1",
        "arm": arm_name,
        "ok": passed,
        "world_size": WORLD,
        "rank_set_complete": complete,
        "all_elements_checked_on_every_rank": all_elements,
        "fp64_reference_on_every_rank": fp64,
        "rel_l2_gate": REL_L2_GATE,
        "max_row_rel_l2_gate": MAX_ROW_GATE,
        "all_elements_checked_global": sum(
            int(row.get("all_elements_checked", 0)) for row in per_rank),
        "per_rank": per_rank,
    }
    if rank == 0:
        save_json(verify_dir / "receipt.json", receipt)
    dist.barrier()
    return receipt


def _read_output(arm_record: Mapping[str, Any], *, device: torch.device,
                 generation: int) -> torch.Tensor:
    runtime = arm_record["runtime"]
    reader = runtime.get("read_output_fn")
    if not callable(reader):
        raise RuntimeError("turbo arm runtime has no read_output_fn hook")
    return reader(runtime, int(generation), device=device)


def _turbo_events(arm_record: Mapping[str, Any], generation: int
                  ) -> list[torch.cuda.Event]:
    """Two same-stream CUDA events bracketing exactly one launch()."""
    turbo = arm_record["turbo"]
    current = int(torch.cuda.current_stream(torch.device("cuda", turbo.gpu)).cuda_stream)
    if current != int(turbo.stream):
        raise RuntimeError(
            f"turbo arm {arm_record['name']} was built on stream "
            f"{int(turbo.stream)} but timing runs on stream {current}")
    events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    stream = torch.cuda.current_stream()
    events[0].record(stream)
    turbo.launch(int(generation), reset=False)
    events[1].record(stream)
    return events


def _timing_cell(*, ctx: dict[str, Any], weights: Mapping[str, torch.Tensor],
                 arms: Mapping[str, Mapping[str, Any]], rank: int,
                 device: torch.device, iters: int, warmup: int,
                 cell_dir: Path, live_log: Path, cell_name: str,
                 rccl_group: Any) -> dict[str, Any]:
    names = ["rccl_aiter", *(f"turbo_mode{m}" for m in TURBO_MODES)]
    order_forward = list(names)
    order_reverse = list(reversed(names))
    ledgers = {name: arms[name]["generation_ledger"]
               for name in names if name != "rccl_aiter"}
    arm_records: dict[str, dict[str, Any]] = {
        name: {"rounds": [], "samples_us": []} for name in names}
    last_generation: dict[str, int] = {}
    sync_buf = torch.zeros(1, dtype=torch.int32, device=device)

    for name in names:
        dist.barrier()
        if name == "rccl_aiter":
            for _ in range(int(warmup)):
                _gpu_sync(sync_buf, rccl_group, device)
                run_official(ctx, weights, moe_mode=OFFICIAL_MODE, timed=False)
                torch.cuda.synchronize(device)
        else:
            for _ in range(int(warmup)):
                arms[name]["turbo"].reset()
                _gpu_sync(sync_buf, rccl_group, device)
                arms[name]["turbo"].launch(ledgers[name].take(), reset=False)
                torch.cuda.synchronize(device)
    dist.barrier()

    err_buf = torch.zeros(1, dtype=torch.int32, device=device)
    for round_idx, order in enumerate((order_forward, order_reverse)):
        samples: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
        for sample_idx in range(int(iters)):
            events_by_arm: dict[str, Sequence[torch.cuda.Event]] = {}
            for name in order:
                if name != "rccl_aiter":
                    arms[name]["turbo"].reset()
                # Tight 4B GPU all_reduce sync OUTSIDE recorded CUDA events
                _gpu_sync(sync_buf, rccl_group, device)
                if name == "rccl_aiter":
                    events_by_arm[name] = _official_events(ctx, weights)
                else:
                    generation = int(ledgers[name].take())
                    last_generation[name] = generation
                    events_by_arm[name] = _turbo_events(arms[name], generation)
                # Global arm boundaries stay outside the recorded interval.
                events_by_arm[name][-1].synchronize()
                if name != "rccl_aiter":
                    error = int(arms[name]["turbo"].device_error())
                    err_buf.fill_(error)
                    dist.all_reduce(err_buf, op=dist.ReduceOp.MAX, group=rccl_group)
                    global_err = int(err_buf.item())
                    if global_err:
                        if error:
                            transport = arms[name]["gda"].report(require_pass=False)
                            failure = {
                                "rank": rank, "arm": name, "round": round_idx,
                                "sample": sample_idx,
                                "generation": last_generation[name],
                                "device_error": error, "transport": transport,
                            }
                            save_json(cell_dir / f"device_failure_rank{rank}.json",
                                      failure)
                            compact = {k: v for k, v in failure.items()
                                       if k != "transport"}
                            compact["gda_error"] = transport.get("gda_err")
                            print("EP16_TURBO_DEVICE_FAILURE " +
                                  json.dumps(compact), flush=True)
                        dist.barrier()
                        raise RuntimeError(
                            f"{name}: global device error {global_err} (local={error}) before next arm")
                dist.barrier()
            for name in order:
                events_by_arm[name][-1].synchronize()
            for name in names:
                events = events_by_arm[name]
                if name == "rccl_aiter":
                    values = _event_metrics_us(events)
                    total = float(values["total"])
                    values["total_us"] = total
                else:
                    total = float(events[0].elapsed_time(events[1])) * 1000.0
                    values = {
                        "total_us": total,
                        "event_total_us": total,
                        "kernel_launch_count": 1,
                        "event_metric_scope": (
                            "event[0] to event[1] around one "
                            "TurboOverlapArm.launch(generation): generation "
                            "576B context h2d + one "
                            "stage_ep16_turbo_overlap launch, same stream"),
                    }
                if not np.isfinite(float(values["total_us"])) or float(
                        values["total_us"]) <= 0.0:
                    raise RuntimeError(
                        f"invalid measured event total for {name}: {values}")
                values["sample"] = sample_idx
                values["round"] = round_idx
                samples[name].append(values)
        torch.cuda.synchronize(device)
        dist.barrier()
        for name in names:
            arm_records[name]["rounds"].append({
                "round": round_idx,
                "order": list(order),
                "samples": samples[name],
                "median_event_total_us": float(np.median(
                    [row["total_us"] for row in samples[name]])),
            })
            arm_records[name]["samples_us"].extend(
                float(row["total_us"]) for row in samples[name])
        save_json(cell_dir / f"timing_partial_rank{rank}_round{round_idx}.json",
                  arm_records)
        for name in names:
            if name == "rccl_aiter":
                continue
            generation = last_generation.get(name)
            if generation is None:
                raise RuntimeError(f"{name}: no timed launch generation")
            state = _turbo_launch_state(
                arms[name], phase=f"{name}_post_timing_round{round_idx}",
                host_generation=generation)
            save_json(cell_dir /
                      f"launch_state_{name}_post_timing_round{round_idx}_rank{rank}.json",
                      state)
            fit = _assert_route_rows_fit(arms[name])
            arm_records[name]["rounds"][-1]["route_rows_fit"] = fit
        _append_live_log(live_log, {
            "schema": "ep16-turbo512-timing-log/1", "cell": cell_name,
            "round": round_idx, "order": list(order),
            "samples_per_arm": int(iters),
        }, rank)
    dist.barrier()

    summary = {
        "schema": "ep16-turbo512-four-arm-timing/1",
        "rounds": TIMED_ROUNDS,
        "iters_per_arm_per_round": int(iters),
        "warmup_per_arm": int(warmup),
        "global_barrier_between_arms_outside_event_interval": True,
        "device_error_check_after_each_timed_turbo_launch": True,
        "orders": [order_forward, order_reverse],
        "event_total_definition": (
            "rccl_aiter: event[0] to event[10] around the full official "
            "operation; turbo arms: event[0] to event[1] around exactly one "
            "same-stream launch(generation)"),
        "arms": {},
    }
    for name in names:
        rows = arm_records[name]["samples_us"]
        summary["arms"][name] = {
            **arm_records[name],
            "median_event_total_us_all_samples": float(np.median(rows)),
            "min_event_total_us": float(min(rows)),
            "max_event_total_us": float(max(rows)),
            "sample_count": len(rows),
        }
    return summary


# ---------------------------------------------------------------------------
# cell
# ---------------------------------------------------------------------------
def _run_cell(*, model: str, global_s: int, rank: int, world: int,
              weights: dict[str, torch.Tensor], route_manifest: Path,
              weight_receipt: Mapping[str, Any], rccl_group: Any,
              run_dir: Path, live_log: Path, svcid_base: int,
              warmup: int, iters: int, device: torch.device,
              live_arms_dir: Path) -> dict[str, Any]:
    if world != WORLD:
        raise RuntimeError(f"turbo512 runner requires world_size={WORLD}, got {world}")
    cell_name = f"{model}_S{global_s}_EP{WORLD}"
    cell_dir = run_dir / "cells" / cell_name
    if rank == 0 and cell_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing cell: {cell_dir}")
    if rank == 0:
        cell_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    started = time.time()

    ctx, plan_stats = _make_buffers(
        model=model, global_s=global_s, rank=rank, world_size=world,
        route_manifest=route_manifest, seed_base=SEED_BASE,
        rccl_group=rccl_group, device=device)
    ctx["cell_dir"] = cell_dir
    save_json(cell_dir / f"plan_rank{rank}.json", plan_stats)

    expected, reference_receipt = compute_reference(
        model=model, global_s=global_s, rank=rank, world_size=world,
        route_manifest=route_manifest, weights_gpu=weights,
        seed_base=SEED_BASE, rccl_group=rccl_group, device=device)
    save_json(cell_dir / f"reference_receipt_rank{rank}.json", reference_receipt)

    spec = model_spec(model)
    n_experts = int(spec.experts)
    e_local = n_experts // WORLD
    t_local = int(ctx["t_local"])
    hidden = int(ctx["hidden"])

    # rccl_aiter correctness is always the first executed arm.
    official_out, _ = run_official(ctx, weights, moe_mode=OFFICIAL_MODE, timed=False)
    torch.cuda.synchronize(device)
    official_out = official_out.detach().clone()
    official_check = _check_correctness(
        official_out, expected, rank=rank, arm_name="rccl_aiter",
        cell_dir=cell_dir)
    _append_live_log(live_log, {
        "schema": "ep16-turbo512-arm-log/1", "cell": cell_name,
        "arm": "rccl_aiter", "stage": "correctness",
        "pass": official_check["ok"], "correctness": official_check,
    }, rank)
    if not official_check["ok"]:
        raise RuntimeError("rccl_aiter failed FP64 all-element correctness")
    del official_out
    torch.cuda.empty_cache()

    # turbo inputs: pre-quantized FP8 + per-(1,128) f32 scales.
    ids, route_w_cpu, route_meta = load_route_rank(
        load_route_manifest(route_manifest, require_complete=True),
        model=model, global_s=global_s, world_size=world, rank=rank,
        verify_file_sha256=True)
    topk_ids_cpu = np.ascontiguousarray(np.asarray(ids, dtype=np.int32))
    if topk_ids_cpu.shape != (t_local, int(ctx["topk"])):
        raise RuntimeError(f"route ids shape {topk_ids_cpu.shape} != "
                           f"({t_local}, {int(ctx['topk'])})")
    if not torch.is_tensor(route_w_cpu):
        route_w_cpu = torch.from_numpy(np.asarray(route_w_cpu))
    topk_weights_cpu = route_w_cpu.to(dtype=torch.float32).contiguous()
    expert_map_cpu = _expert_map(n_experts, e_local)

    x_fp8, x_scale = _prequant_fp8(ctx["x"])
    topk_ids = torch.from_numpy(topk_ids_cpu).to(device=device)
    topk_weights = topk_weights_cpu.to(device=device)
    expert_map = torch.tensor(expert_map_cpu, dtype=torch.int32,
                              device=device)
    out_bf16 = torch.empty((t_local, hidden), dtype=torch.bfloat16,
                           device=device)
    tensors = {
        "x_fp8": x_fp8, "x_scale": x_scale,
        "topk_ids": topk_ids, "topk_weights": topk_weights,
        "expert_map": expert_map,
        "w1_fp8": weights["w13_shuffled"].view(torch.uint8).contiguous(),
        "w1_scale": weights["sf13"].contiguous(),
        "w2_fp8": weights["w2_shuffled"].view(torch.uint8).contiguous(),
        "w2_scale": weights["sf2"].contiguous(),
        "out_bf16": out_bf16,
    }

    arms: dict[str, Any] = {}
    correctness_rows: dict[str, Any] = {"rccl_aiter": official_check}
    artifact_rows: dict[str, Any] = {}
    base_ranges: list[dict[str, Any]] = []
    timing_summary: dict[str, Any] = {}
    cell_record: dict[str, Any] = {}
    try:
        for mode in TURBO_MODES:
            name = f"turbo_mode{mode}"
            svcid = int(svcid_base) + (mode + 1) * 512
            service_range = validate_service_id_base(svcid)
            if base_ranges and service_range["first"] <= base_ranges[-1]["last_reserved"]:
                raise RuntimeError("turbo arms have overlapping service-ID blocks")
            base_ranges.append(service_range)
            arm = _build_turbo_arm(
                name=name, mode=mode, rank=rank, ctx=ctx,
                topk_ids_cpu=topk_ids_cpu, expert_map_cpu=expert_map_cpu,
                tensors=tensors, device=device, svcid_base=svcid,
                host_lib=live_arms_dir / f"libep16_turbo_host_mode{mode}.so",
                co_path=live_arms_dir / f"P_ep16_turbo_overlap_mode{mode}.co",
                gda_path=live_arms_dir / f"gda_turbo_mode{mode}.so")
            arms[name] = arm
            runtime = arm["runtime"]
            for other_name, other_arm in arms.items():
                if other_name == name:
                    continue
                other_runtime = other_arm["runtime"]
                if (int(runtime["control_ptr"]) == int(other_runtime["control_ptr"]) or
                        int(runtime["transport_arena"]) == int(other_runtime["transport_arena"]) or
                        int(runtime["packet_ptr"]) == int(other_runtime["packet_ptr"])):
                    raise RuntimeError(
                        f"runtime allocations alias: {name} and {other_name}")

            sync_buf = torch.zeros(1, dtype=torch.int32, device=device)
            arm["turbo"].reset()
            dist.barrier()
            _gpu_sync(sync_buf, rccl_group, device)
            ledger = arm["generation_ledger"]
            generation = int(ledger.take())
            arm["turbo"].launch(generation, reset=False)
            torch.cuda.synchronize(device)
            state = _turbo_launch_state(
                arm, phase=f"{name}_direct", host_generation=generation)
            save_json(cell_dir / f"launch_state_{name}_direct_rank{rank}.json", state)
            if int(state["device_error"]) != 0:
                raise RuntimeError(
                    f"{name}: device error {state['device_error']} on direct launch")
            fit = _assert_route_rows_fit(arm)
            output = _read_output(arm, device=device, generation=generation)
            check = _check_correctness(output, expected, rank=rank,
                                       arm_name=name, cell_dir=cell_dir)
            correctness_rows[name] = check
            artifact_rows[name] = {
                "code_object": arm["co_path"],
                "code_object_sha256": arm["co_sha256"],
                "host_library": arm["host_lib_path"],
                "host_library_sha256": arm["host_lib_sha256"],
                "gda_library": arm["gda_lib_path"],
                "gda_library_sha256": arm["gda_lib_sha256"],
                "kernel_name": T.KERNEL_NAME,
                "grid": T.GRID, "block": T.BLOCK,
                "kernel_attributes": arm["kernel_attributes"],
                "svcid_range": arm["svcid_range"],
                "turbo_receipt": arm["runtime"]["turbo_receipt"],
                "route_rows_fit": fit,
            }
            _append_live_log(live_log, {
                "schema": "ep16-turbo512-arm-log/1", "cell": cell_name,
                "arm": name, "stage": "direct_correctness",
                "pass": check["ok"], "generation": generation,
                "route_rows_fit": fit, "correctness": check,
            }, rank)
            if not check["ok"]:
                raise RuntimeError(f"{name} failed FP64 all-element correctness")
            del output

            # two extra probes: repeated monotonic launches reuse the same
            # arena/scratch and must reproduce the full output again.
            for probe_index in range(2):
                arm["turbo"].reset()
                dist.barrier()
                _gpu_sync(sync_buf, rccl_group, device)
                probe_generation = int(ledger.take())
                arm["turbo"].launch(probe_generation, reset=False)
                torch.cuda.synchronize(device)
                probe_state = _turbo_launch_state(
                    arm, phase=f"{name}_probe{probe_index}",
                    host_generation=probe_generation)
                save_json(cell_dir /
                          f"launch_state_{name}_probe{probe_index}_rank{rank}.json",
                          probe_state)
                if int(probe_state["device_error"]) != 0:
                    raise RuntimeError(
                        f"{name}: device error {probe_state['device_error']} "
                        f"on probe {probe_index}")
                _assert_route_rows_fit(arm)
                probe_output = _read_output(arm, device=device,
                                            generation=probe_generation)
                probe_check = _check_correctness(
                    probe_output, expected, rank=rank,
                    arm_name=f"{name}_probe{probe_index}", cell_dir=cell_dir)
                correctness_rows[f"{name}_probe{probe_index}"] = probe_check
                if not probe_check["ok"]:
                    raise RuntimeError(
                        f"{name}_probe{probe_index} full-output correctness failed")
                del probe_output

        dist.barrier()
        timing_summary = _timing_cell(
            ctx=ctx, weights=weights, arms=arms, rank=rank, device=device,
            iters=iters, warmup=warmup, cell_dir=cell_dir, live_log=live_log,
            cell_name=cell_name, rccl_group=rccl_group)
        # Verify the actual final timed output on every rank, without reset
        # or a replacement kernel launch between timing and this check.
        for post_name, post_arm in arms.items():
            post_out = _read_output(post_arm, device=device,
                                    generation=int(post_arm["turbo"].last_generation))
            post_check = _check_correctness(post_out, expected, rank=rank,
                arm_name=post_name+"_post_timing", cell_dir=cell_dir)
            correctness_rows[post_name+"_post_timing"] = post_check
            if not post_check["ok"]:
                raise RuntimeError(post_name+" failed post-timing FP64 full-output check")
            del post_out
        save_json(cell_dir / f"timing_rank{rank}.json", timing_summary)

        gathered_timing: list[Any] = [None] * WORLD
        dist.all_gather_object(
            gathered_timing,
            {name: {"median_event_total_us_all_samples":
                    float(summary["median_event_total_us_all_samples"]),
                    "min_event_total_us": float(summary["min_event_total_us"]),
                    "max_event_total_us": float(summary["max_event_total_us"]),
                    "sample_count": int(summary["sample_count"])}
             for name, summary in timing_summary["arms"].items()})
        if rank == 0:
            save_json(cell_dir / "timing_global.json", {
                "schema": "ep16-turbo512-timing-global/1",
                "cell": cell_name, "per_rank": gathered_timing})

        cell_record = {
            "schema": "ep16-turbo512-cell/1",
            "cell": cell_name,
            "model": model,
            "global_s": int(global_s),
            "t_local": t_local,
            "hidden": hidden,
            "intermediate": int(ctx["intermediate"]),
            "topk": int(ctx["topk"]),
            "n_experts": n_experts,
            "e_local": e_local,
            "capacity": t_local,
            "node_tokens": t_local,
            "route_rows": WORLD * t_local * int(ctx["topk"]) + 16 * e_local,
            "world_size": WORLD,
            "rank": rank,
            "seed_base": SEED_BASE,
            "duration_sec": round(time.time() - started, 3),
            "correctness": correctness_rows,
            "artifacts": artifact_rows,
            "weight_receipt": dict(weight_receipt),
            "timing": timing_summary,
        }
        save_json(cell_dir / f"cell_rank{rank}.json", cell_record)
        _append_live_log(live_log, {
            "schema": "ep16-turbo512-cell-log/1", "cell": cell_name,
            "pass": True, "duration_sec": cell_record["duration_sec"],
            "correctness_arms": sorted(correctness_rows),
            "timing_arms": sorted(timing_summary["arms"]),
        }, rank)
        return cell_record
    finally:
        # teardown: close arms in reverse construction order, barrier between
        # each close so QP teardown stays cross-rank ordered.
        dist.barrier()
        for name in sorted(arms, reverse=True):
            arm = arms.get(name)
            if arm is None:
                continue
            try:
                arm["turbo"].close()
            except Exception as exc:
                print(f"TURBO_CLOSE_ERROR rank={rank} arm={name}: {exc!r}",
                      flush=True)
            dist.barrier()


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _live_arm_paths(live_arms_dir: Path) -> dict[str, list[Path]]:
    """Required per-mode private copies under ``build/live_arms``.

    ``ep16_turbo_module_load`` keeps global module state (and GdaLibrary
    caches), so every arm needs its own on-disk ELF / code-object copy.
    """
    paths: dict[str, list[Path]] = {}
    for mode in TURBO_MODES:
        paths[f"mode{mode}"] = [
            live_arms_dir / f"libep16_turbo_host_mode{mode}.so",
            live_arms_dir / f"gda_turbo_mode{mode}.so",
            live_arms_dir / f"P_ep16_turbo_overlap_mode{mode}.co",
        ]
    return paths


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--s-values", required=True,
                        help="comma-separated global S values")
    parser.add_argument("--route-manifest", type=Path, required=True)
    parser.add_argument("--weight-cache", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--live-log", type=Path, required=True)
    parser.add_argument("--svcid-base", type=int, default=20000,
                        help="base service ID; arms use base+(mode+1)*512")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--live-arms-dir", type=Path,
                        default=PROJECT_ROOT / "build" / "live_arms")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.warmup < 1 or args.iters < 1:
        raise SystemExit("--warmup and --iters must be positive")
    try:
        s_values = [int(part.strip()) for part in args.s_values.split(",")
                    if part.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --s-values={args.s_values!r}") from exc
    if not s_values or len(set(s_values)) != len(s_values):
        raise SystemExit("--s-values must be nonempty and unique")
    if len(s_values) != 1:
        raise SystemExit(
            "run one --s-values value per torchrun process; the entry script "
            "invokes this runner once per cell so arena/GDA teardown completes "
            "before the next cell starts")
    if any(s not in S_VALUES or s % WORLD for s in s_values):
        raise SystemExit(f"S values must be in {S_VALUES} and divisible by {WORLD}")
    spec = model_spec(args.model)
    if int(spec.experts) % WORLD:
        raise SystemExit(
            f"model {args.model} has {spec.experts} experts, not divisible by {WORLD}")
    if not args.live_arms_dir.is_dir():
        raise SystemExit(f"--live-arms-dir {args.live_arms_dir} is not a directory")
    missing = [str(path) for paths in _live_arm_paths(args.live_arms_dir).values()
               for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing live-arm artifacts: " + ", ".join(missing))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} device_count={torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Aiter loads ROCm/RCCL components before any NCCL process group creation.
    os.environ.setdefault("AITER_LOG_LEVEL", "INFO")
    import aiter  # noqa: F401

    control_backend = os.environ.get("EP16_CONTROL_BACKEND", "gloo")
    dist.init_process_group(backend=control_backend,
                            timeout=timedelta(seconds=3600))
    rank = int(dist.get_rank())
    world = int(dist.get_world_size())
    try:
        if world != WORLD:
            raise RuntimeError(f"torchrun world {world} != required {WORLD}")
        rccl_group = dist.new_group(backend=os.environ.get("EP16_BACKEND", "nccl"))
        args.run_dir.mkdir(parents=True, exist_ok=True)
        route_manifest = load_route_manifest(
            args.route_manifest, require_complete=True)
        weights = None
        weight_receipt = None
        for loader_rank in range(world):
            if rank == loader_rank:
                weights, weight_receipt = _load_weight_bundle(
                    args.model, world, rank, args.weight_cache, device)
                torch.cuda.synchronize(device)
            dist.barrier()
        if weights is None or weight_receipt is None:
            raise RuntimeError(
                "serialized weight loading did not produce local weights")
        results = []
        for global_s in s_values:
            results.append(_run_cell(
                model=args.model, global_s=global_s, rank=rank, world=world,
                weights=weights, route_manifest=args.route_manifest,
                weight_receipt=weight_receipt, rccl_group=rccl_group,
                run_dir=args.run_dir, live_log=args.live_log,
                svcid_base=args.svcid_base, warmup=args.warmup,
                iters=args.iters, device=device,
                live_arms_dir=args.live_arms_dir))
        if rank == 0:
            save_json(args.run_dir / "turbo512_summary.json", {
                "schema": "ep16-turbo512-run/1",
                "model": args.model,
                "s_values": s_values,
                "world_size": WORLD,
                "control_backend": control_backend,
                "rccl_backend": os.environ.get("EP16_BACKEND", "nccl"),
                "aiter_path": str(aiter.__file__),
                "live_arms_dir": str(args.live_arms_dir),
                "svcid_base": int(args.svcid_base),
                "warmup": int(args.warmup),
                "iters": int(args.iters),
                "results": results,
            })
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
