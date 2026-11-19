#!/usr/bin/env python3
"""Deterministic inputs, routing, weights and statistics for the AMD EP baseline.

The benchmark currently runs EP8 on one MI308X node.  Rank/world-size and local
expert counts are parameters throughout so the same path can be launched as EP16.
The ShareGPT histogram files themselves contain eight source-rank rows; target
world sizes that are a multiple of eight are obtained by deterministic row splits.
"""
from __future__ import annotations

import datetime
import hashlib
import heapq
import json
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

FP8_MAX = 240.0  # torch.float8_e4m3fnuz MaxFinite
BLOCK = 128
HIST_SOURCE_WORLD_SIZE = 8
HIST_SOURCE_MAX_S = 20000
S_VALUES = [32, 128, 1024, 2048, 4096, 8192, 16000, 20000]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    experts: int
    topk: int
    hidden: int
    intermediate: int


MODEL_SPECS = {
    "qwen38max": ModelSpec("qwen38max", 512, 10, 8192, 2048),
    "minimax_m3": ModelSpec("minimax_m3", 128, 4, 6144, 3072),
    "Qwen3.5-397B-A17B": ModelSpec("Qwen3.5-397B-A17B", 512, 10, 4096, 1024),
    "Qwen3.5-122B-A10B": ModelSpec("Qwen3.5-122B-A10B", 256, 8, 3072, 1024),
    "DeepSeek-V4-Flash": ModelSpec("DeepSeek-V4-Flash", 256, 6, 4096, 2048),
    "Qwen3.5-35B-A3B": ModelSpec("Qwen3.5-35B-A3B", 256, 8, 2048, 512),
}
MODEL_ORDER = list(MODEL_SPECS)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") & ((1 << 63) - 1)


def save_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(tmp, path)


def check_disk_pressure(status_path: Path) -> dict:
    path = Path(status_path)
    if not path.exists():
        return {"present": False, "pressure": None}
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return {"present": True, "pressure": None, "error": repr(exc)}
    if data.get("pressure") is True:
        raise RuntimeError(f"disk_status pressure=true; stop compilation/output: {path}")
    return {"present": True, "pressure": bool(data.get("pressure")), "raw": data}


def model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; choices={MODEL_ORDER}") from exc


def local_tokens(global_s: int, world_size: int) -> int:
    if world_size < 1 or global_s % world_size:
        raise ValueError(f"S={global_s} is not divisible by EP={world_size}")
    return global_s // world_size


def hist_relative_path(model: str, source_s: int) -> str:
    return ("weave_v1/rebuttal/amd_mi308x/megamoe/mn_v2/"
            f"costmodel/bench_hist/{model}_S{source_s}.bin")


def hist_path(repo_root: Path, model: str, source_s: int) -> Path:
    root = Path(repo_root)
    staged = root / f"{model}_S{source_s}.bin"
    if staged.is_file():
        return staged
    return root / hist_relative_path(model, source_s)


def read_hist(path: Path, expected_experts: int,
              source_world_size: int = HIST_SOURCE_WORLD_SIZE) -> np.ndarray:
    raw = np.fromfile(Path(path), dtype="<i4")
    expected = int(source_world_size) * int(expected_experts)
    if raw.size != expected:
        raise ValueError(
            f"{path}: int32 count {raw.size} != {expected} "
            f"([{source_world_size}, {expected_experts}])")
    counts = raw.reshape(int(source_world_size), int(expected_experts)).copy()
    if np.any(counts < 0):
        raise ValueError(f"{path}: negative histogram count")
    return counts


def _largest_remainder_row(row: np.ndarray, target: int) -> np.ndarray:
    """Proportionally rescale one integer row to an exact integer total."""
    row = np.asarray(row, dtype=np.int64)
    total = int(row.sum())
    target = int(target)
    if total <= 0:
        raise ValueError("zero histogram row")
    if target < 0:
        raise ValueError("negative target")
    if target == 0:
        return np.zeros_like(row, dtype=np.int32)
    exact = row.astype(np.float64) * (float(target) / float(total))
    floor = np.floor(exact).astype(np.int64)
    remainder = target - int(floor.sum())
    if remainder < 0 or remainder > row.size:
        raise AssertionError((target, total, int(floor.sum()), remainder))
    if remainder:
        frac = exact - floor
        order = np.lexsort((np.arange(row.size), -frac))[:remainder]
        floor[order] += 1
    if int(floor.sum()) != target:
        raise AssertionError((int(floor.sum()), target))
    return floor.astype(np.int32)


def _split_hist_to_world(source: np.ndarray, world_size: int,
                         row_target: int) -> np.ndarray:
    """Split each EP8 source row into world_size/8 target source rows.

    Sequential largest-remainder splits preserve every expert's column total.
    Each resulting row has exactly row_target routes.  EP8 is returned unchanged.
    """
    src_world = source.shape[0]
    if world_size == src_world:
        if not np.all(source.sum(axis=1) == row_target):
            raise ValueError("source row total differs from target")
        return source.copy()
    if world_size % src_world:
        raise ValueError(
            f"target EP={world_size} must be a multiple of histogram EP={src_world}")
    factor = world_size // src_world
    out_rows: list[np.ndarray] = []
    for row in source:
        remaining = np.asarray(row, dtype=np.int64).copy()
        for j in range(factor):
            if j == factor - 1:
                part = remaining.astype(np.int32)
            else:
                # Allocate this target row, then leave the rest for later rows.
                part = _largest_remainder_row(remaining, row_target)
                remaining -= part.astype(np.int64)
            if int(part.sum()) != row_target:
                raise AssertionError((j, int(part.sum()), row_target))
            out_rows.append(part)
    out = np.stack(out_rows)
    if out.shape != (world_size, source.shape[1]):
        raise AssertionError(out.shape)
    if not np.array_equal(out.sum(axis=0), source.sum(axis=0)):
        raise AssertionError("world split changed expert column totals")
    return out


def derive_hist_for_s(source: np.ndarray, global_s: int, topk: int,
                      world_size: int = 8) -> tuple[np.ndarray, dict]:
    """Derive a missing route histogram from S20000 (HIST_DERIVED_ROUTE)."""
    if source.shape[0] != HIST_SOURCE_WORLD_SIZE:
        raise ValueError("derivation source must be the source hist world")
    target_row = local_tokens(global_s, world_size) * topk
    base = np.stack([_largest_remainder_row(row, target_row) for row in source])
    out = _split_hist_to_world(base, world_size, target_row)
    if not np.all(out.sum(axis=1) == target_row):
        raise AssertionError("derived row sums differ")
    if int(out.sum()) != global_s * topk:
        raise AssertionError("derived total differs")
    return out, {
        "label": "HIST_DERIVED_ROUTE",
        "source_S": HIST_SOURCE_MAX_S,
        "target_S": int(global_s),
        "source_world_size": int(source.shape[0]),
        "target_world_size": int(world_size),
        "method": (
            "per-source-rank proportional largest-remainder rescaling to exact row totals; "
            "then deterministic EP-world row split preserving expert column totals"
        ),
        "row_target": int(target_row),
        "total_routes": int(out.sum()),
    }


def load_cell_hist(repo_root: Path, model: str, global_s: int,
                   world_size: int = 8) -> tuple[np.ndarray, dict]:
    spec = model_spec(model)
    exact = hist_path(repo_root, model, global_s)
    if exact.is_file():
        source = read_hist(exact, spec.experts, HIST_SOURCE_WORLD_SIZE)
        meta = {
            "label": "SHAREGPT_HIST",
            "path": str(exact),
            "sha256": sha256_file(exact),
            "source_S": int(global_s),
            "target_S": int(global_s),
            "source_world_size": HIST_SOURCE_WORLD_SIZE,
            "target_world_size": int(world_size),
            "route_reconstruction": (
                "per-source expert marginals; deterministic min-load top-k assignment; "
                "not the unavailable original per-token route rows"),
        }
    else:
        source_path = hist_path(repo_root, model, HIST_SOURCE_MAX_S)
        if not source_path.is_file():
            raise FileNotFoundError(f"missing S20000 derivation source: {source_path}")
        source = read_hist(source_path, spec.experts, HIST_SOURCE_WORLD_SIZE)
        derived, derive_meta = derive_hist_for_s(source, global_s, spec.topk, world_size)
        source = derived
        meta = {
            **derive_meta,
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "route_reconstruction": (
                "per-source expert marginals; deterministic min-load top-k assignment; "
                "not the unavailable original per-token route rows"),
        }
    row_target = local_tokens(global_s, world_size) * spec.topk
    if world_size != HIST_SOURCE_WORLD_SIZE:
        source = _split_hist_to_world(source, world_size, row_target)
        meta["world_split_rows"] = int(world_size)
        meta["world_split_preserved_expert_columns"] = True
    row_sums = np.unique(source.sum(axis=1))
    if row_sums.tolist() != [row_target]:
        raise ValueError(f"hist row sums {row_sums.tolist()} != expected {row_target}")
    if int(source.sum()) != int(global_s) * spec.topk:
        raise ValueError(f"hist total {int(source.sum())} != S*K={int(global_s)*spec.topk}")
    meta.update({
        "row_sums": row_sums.tolist(),
        "local_tokens_per_rank": local_tokens(global_s, world_size),
        "experts": spec.experts,
        "topk": spec.topk,
        "route_count_sha256": hashlib.sha256(
            source.astype("<i4", copy=False).tobytes()).hexdigest(),
    })
    return source, meta


def make_topk_ids(counts: np.ndarray, tokens: int, topk: int) -> np.ndarray:
    """Build distinct per-token top-k experts with exact expert marginals.

    Experts are processed once in decreasing-count order.  Each occurrence takes
    the currently least-loaded eligible token.  A later occurrence of the same
    expert remains a different occurrence, so one token never receives duplicate
    experts.  The resulting bincount is asserted equal to ``counts``.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if int(counts.sum()) != tokens * topk:
        raise ValueError(f"counts sum {int(counts.sum())} != tokens*topk {tokens*topk}")
    if counts.size and int(counts.max()) > tokens:
        raise ValueError("an expert count exceeds token count")
    out = np.full((tokens, topk), -1, dtype=np.int32)
    load = np.zeros(tokens, dtype=np.int16)
    heap = [(0, t) for t in range(tokens)]
    heapq.heapify(heap)
    expert_order = np.lexsort((np.arange(counts.size), -counts))
    for expert in expert_order:
        n = int(counts[expert])
        chosen: list[tuple[int, int]] = []
        for _ in range(n):
            if not heap:
                raise RuntimeError("route heap exhausted")
            current, token = heapq.heappop(heap)
            if current != int(load[token]) or current >= topk:
                raise AssertionError((int(expert), current, int(load[token]), topk))
            out[token, current] = int(expert)
            load[token] += 1
            chosen.append((int(load[token]), token))
        for item in chosen:
            heapq.heappush(heap, item)
    if not np.all(load == topk):
        raise AssertionError(f"row loads not all {topk}: {np.unique(load).tolist()}")
    recovered = np.bincount(out.reshape(-1), minlength=counts.size)
    if not np.array_equal(recovered, counts):
        raise AssertionError("expert marginal mismatch")
    return out


def make_route_weights(global_s: int, source_rank: int, tokens: int, topk: int,
                       seed_base: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(stable_seed(seed_base, "route_weights", global_s, source_rank,
                              tokens, topk))
    logits = torch.randn((tokens, topk), generator=g, dtype=torch.float32)
    weights = torch.softmax(logits, dim=1)
    # Metadata transports 16 bits; reference and measured path both use these values.
    return weights.to(torch.bfloat16).to(torch.float16).to(torch.float32).contiguous()


def make_activation(global_s: int, source_rank: int, hidden: int,
                    seed_base: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(stable_seed(seed_base, "activation", global_s, source_rank, hidden))
    x32 = torch.randn((global_s, hidden), generator=g, dtype=torch.float32)
    return x32.to(torch.bfloat16).contiguous()


def _quantize_weight_box(rows: int, cols: int, generator: torch.Generator,
                         out: torch.Tensor, scale_out: torch.Tensor,
                         row_base: int = 0) -> None:
    if rows % BLOCK or cols % BLOCK:
        raise ValueError("weight dimensions must be 128-aligned")
    for rb in range(0, rows, BLOCK):
        raw = torch.randn((BLOCK, cols), generator=generator, dtype=torch.float32)
        blocks = raw.reshape(BLOCK, cols // BLOCK, BLOCK)
        amax = blocks.abs().amax(dim=(0, 2))
        scale = (amax * (1.0 / FP8_MAX)).clamp_min(1e-12)
        q = (blocks / scale[None, :, None]).to(torch.float8_e4m3fnuz).view(torch.uint8)
        out[row_base + rb:row_base + rb + BLOCK].copy_(q.reshape(BLOCK, cols))
        scale_out[row_base // BLOCK + rb // BLOCK].copy_(scale)


def generate_weights_cpu(spec: ModelSpec, rank: int, world_size: int,
                         seed_base: int) -> dict[str, torch.Tensor]:
    """Generate natural-layout randn FP8 weights and natural 128x128 scales."""
    if spec.experts % world_size:
        raise ValueError(f"E={spec.experts} not divisible by EP={world_size}")
    el = spec.experts // world_size
    h, inter = spec.hidden, spec.intermediate
    for value, label in ((h, "hidden"), (inter, "intermediate")):
        if value % BLOCK:
            raise ValueError(f"{label}={value} is not 128-aligned")
    seed = stable_seed(seed_base, "weights", spec.name, rank, world_size)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    w13 = torch.empty((el, 2 * inter, h), dtype=torch.uint8)
    sf13 = torch.empty((el, 2 * inter // BLOCK, h // BLOCK), dtype=torch.float32)
    w2 = torch.empty((el, h, inter), dtype=torch.uint8)
    sf2 = torch.empty((el, h // BLOCK, inter // BLOCK), dtype=torch.float32)
    for e in range(el):
        _quantize_weight_box(2 * inter, h, g, w13[e], sf13[e])
        _quantize_weight_box(h, inter, g, w2[e], sf2[e])
    return {
        "w13_u8": w13,
        "w2_u8": w2,
        "sf13": sf13,
        "sf2": sf2,
        "model": spec.name,
        "rank": int(rank),
        "world_size": int(world_size),
        "local_experts": int(el),
        "hidden": int(h),
        "intermediate": int(inter),
        "seed": int(seed),
        "generator": "torch.randn float32 -> per-128x128 dynamic e4m3fnuz quant",
    }


def save_weights_cpu(weights: dict[str, torch.Tensor], path: Path) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(weights, tmp)
    os.replace(tmp, path)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensor_sha256": {
            name: hashlib.sha256(weights[name].contiguous().numpy().tobytes()).hexdigest()
            for name in ("w13_u8", "w2_u8", "sf13", "sf2")
        },
    }


def load_weights_cpu(path: Path) -> dict[str, torch.Tensor]:
    return torch.load(Path(path), map_location="cpu", weights_only=True)


def weights_to_gpu(weights: dict[str, torch.Tensor],
                   device: torch.device) -> dict[str, torch.Tensor]:
    import sys
    from aiter.ops.shuffle import shuffle_weight
    def _stage(name: str) -> None:
        torch.cuda.synchronize(device)
        if os.environ.get("EP8_DEBUG", "0") == "1":
            print(f"[weight-stage] device={device} {name}", file=sys.stderr, flush=True)
    fp8 = torch.float8_e4m3fnuz
    _stage("shuffle_import")
    w13n = weights["w13_u8"].to(device=device).view(fp8)
    _stage("w13_copy")
    w2n = weights["w2_u8"].to(device=device).view(fp8)
    _stage("w2_copy")
    sf13 = weights["sf13"].to(device=device)
    sf2 = weights["sf2"].to(device=device)
    _stage("scales_copy")
    w13s = shuffle_weight(w13n.contiguous(), (16, 16))
    _stage("w13_shuffle")
    w2s = shuffle_weight(w2n.contiguous(), (16, 16))
    _stage("w2_shuffle")
    return {
        "w13_natural": w13n,
        "w2_natural": w2n,
        "w13_shuffled": w13s,
        "w2_shuffled": w2s,
        "sf13": sf13,
        "sf2": sf2,
    }


def _pack_meta_numpy(pos: np.ndarray, local_expert: np.ndarray,
                     weight: np.ndarray) -> np.ndarray:
    weight16 = torch.from_numpy(
        np.asarray(weight, dtype=np.float32)).to(torch.float16)
    wbits = weight16.view(torch.int16).to(torch.int64).numpy() & 0xFFFF
    pos = np.asarray(pos, dtype=np.int64)
    local_expert = np.asarray(local_expert, dtype=np.int64)
    if np.any(local_expert < 0) or np.any(local_expert >= 64):
        raise ValueError("local expert index does not fit the 6-bit metadata field")
    if np.any(pos < 0) or np.any(pos >= (1 << 42)):
        raise ValueError("token position does not fit packed metadata")
    meta = (pos << 22) | (local_expert << 16) | wbits
    return meta.astype(np.int64, copy=False)


def unpack_meta(meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = meta >> 22
    local_expert = (meta >> 16) & 0x3F
    weight = (meta & 0xFFFF).to(torch.int16).view(torch.float16).float()
    return pos, local_expert, weight


def build_ep_plan(topk_ids: np.ndarray, topk_weights: torch.Tensor, e_local: int,
                  world_size: int, device: torch.device) -> dict:
    """Build a selective-dedup static EP send plan (rank count is parameterized)."""
    if topk_ids.ndim != 2 or topk_ids.shape[1] < 1:
        raise ValueError("topk_ids must be [tokens, topk]")
    if tuple(topk_weights.shape) != tuple(topk_ids.shape):
        raise ValueError("topk weights shape mismatch")
    if e_local * world_size <= int(topk_ids.max(initial=0)):
        raise ValueError("expert id outside EP range")
    w_np = topk_weights.detach().cpu().numpy()
    tokens = topk_ids.shape[0]
    dst = topk_ids // e_local
    if np.any(dst < 0) or np.any(dst >= world_size):
        raise ValueError("expert destination outside EP range")
    send_tok, routes_meta, send_ct, send_cm = [], [], [], []
    for target in range(world_size):
        tok = np.flatnonzero(np.any(dst == target, axis=1))
        pos = np.full(tokens, -1, dtype=np.int64)
        pos[tok] = np.arange(tok.size, dtype=np.int64)
        rr, kk = np.nonzero(dst == target)
        local_e = topk_ids[rr, kk] - target * e_local
        meta = _pack_meta_numpy(pos[rr], local_e, w_np[rr, kk])
        send_tok.append(torch.from_numpy(tok.astype(np.int64, copy=False)).to(device))
        routes_meta.append(torch.from_numpy(meta).to(device))
        send_ct.append(int(tok.size))
        send_cm.append(int(rr.size))
    return {
        "send_tok": send_tok,
        "routes_meta": routes_meta,
        "send_ct": torch.tensor(send_ct, dtype=torch.int64, device=device),
        "send_cm": torch.tensor(send_cm, dtype=torch.int64, device=device),
        "send_ct_list": send_ct,
        "send_cm_list": send_cm,
        "send_tokens": int(sum(send_ct)),
        "send_routes": int(sum(send_cm)),
        "world_size": int(world_size),
        "local_experts": int(e_local),
    }


def round_stats(values: list[float] | tuple[float, ...]) -> dict:
    vals = [float(x) for x in values]
    if not vals:
        return {"n": 0}
    mean = statistics.mean(vals)
    cv = 0.0 if len(vals) == 1 or mean == 0 else (
        statistics.stdev(vals) / mean * 100.0)
    ordered = sorted(vals)
    return {
        "n": len(vals),
        "values_ms": vals,
        "median_ms": statistics.median(vals),
        "mean_ms": mean,
        "min_ms": min(vals),
        "max_ms": max(vals),
        "p10_ms": ordered[max(0, int(math.floor(0.10 * (len(ordered) - 1))))],
        "p90_ms": ordered[max(0, int(math.ceil(0.90 * (len(ordered) - 1))))],
        "inter_round_cv_pct": cv,
        "cv_formula": "100 * sample_stdev(round_slowest_rank_medians) / mean",
    }


def module_versions() -> dict:
    import importlib.metadata
    result: dict[str, Any] = {"utc": utc_now()}
    for dist_name in ("torch", "amd-aiter", "triton", "vllm", "numpy", "ninja"):
        try:
            result[dist_name] = importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            result[dist_name] = None
    try:
        import aiter
        result["aiter_module"] = aiter.__file__
        result["aiter_gfx"] = aiter.get_gfx()
        result["aiter_cu_num"] = aiter.get_cu_num()
    except Exception as exc:
        result["aiter_import_error"] = f"{type(exc).__name__}: {exc}"
    result["torch_version"] = torch.__version__
    result["torch_hip"] = torch.version.hip
    result["torch_cuda"] = torch.version.cuda
    try:
        result["torch_nccl_version"] = torch.cuda.nccl.version()
    except Exception as exc:
        result["torch_nccl_version_error"] = f"{type(exc).__name__}: {exc}"
    result["device_count"] = torch.cuda.device_count()
    result["devices"] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        result["devices"].append({
            "index": i,
            "name": props.name,
            "total_memory": props.total_memory,
            "gcn_arch": getattr(props, "gcnArchName", None),
            "compute_units": props.multi_processor_count,
        })
    env_keys = ("ROCM_PATH", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                "RCCL_DEBUG", "NCCL_DEBUG", "GPULOCK_GPUS", "AITER_CONFIG_FMOE")
    result["environment"] = {key: os.environ.get(key) for key in env_keys}
    return result
