#!/usr/bin/env python3
"""Run isolated SM90 MegaMoE correctness arms and compare full outputs.

Example (run on an idle Hopper host):
  python csrc/jit_kernels/weave_validate_sm90.py --checkout /path/to/weave --official-checkout /path/to/official --tokens 8 --seed 17 --repeats 3
  python csrc/jit_kernels/weave_validate_sm90.py --checkout /path/to/weave --official-checkout /path/to/official --world-size 8 --routing skew

The default mode starts each arm in a fresh Python process. Every arm receives
the same seed. Results are written as JSON plus complete tensor artifacts under
--output-dir. The FP64 oracle applies the mathematical MoE to the original
BF16 randn tensors; it is an accuracy reference, while arm-to-official checks
detect regressions under identical quantized inputs.
"""

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys


def absolute(path):
    return str(pathlib.Path(path).expanduser().resolve())


def check_gpu_idle():
    query = shutil.which("nvidia-smi")
    if query is None:
        raise RuntimeError("nvidia-smi is unavailable; refusing to start without the occupancy gate")
    result = subprocess.run(
        [query, "--query-compute-apps=pid,process_name,gpu_uuid", "--format=csv,noheader"],
        check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError("nvidia-smi occupancy query failed: " + result.stderr.strip())
    processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if processes:
        raise RuntimeError("GPU compute processes are active; wait until idle. No process will be killed:\\n" + "\\n".join(processes))


def check_official_checkout(path):
    root = pathlib.Path(path).expanduser().resolve()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    if not head.startswith("b64107f2"):
        raise RuntimeError(f"official checkout HEAD must be b64107f2, got {head}: {root}")
    if status:
        raise RuntimeError("official checkout is not clean; refusing to use a modified baseline:\n" + status)
    return root, head


def check_arm_policy(name, controls, log_text):
    if controls.get("WEAVE_SM90_ENABLE") != "1":
        return
    lines = [line for line in log_text.splitlines() if "WEAVE_SM90 policy:" in line]
    if not lines:
        raise RuntimeError(f"arm={name}: no WEAVE_SM90 policy line; runtime may have ignored the knobs")
    line = lines[0]
    result_lines = [line for line in log_text.splitlines() if line.startswith("WEAVE_SM90_RESULT ")]
    if not result_lines:
        raise RuntimeError(f"arm={name}: no validation result JSON is present in its log")
    result = json.loads(result_lines[-1].split(" ", 1)[1])
    expected_counts = {tuple(values) for values in result["expert_receive_counts_by_local_rank"]}
    counts_match = re.search(r"expert_counts=\[([0-9, ]*)\]", line)
    if not counts_match:
        raise RuntimeError(f"arm={name}: policy line is missing observed expert counts: {line}")
    actual_counts = tuple(int(value.strip()) for value in counts_match.group(1).split(",") if value.strip())
    if actual_counts not in expected_counts:
        raise RuntimeError(f"arm={name}: policy expert counts disagree with gathered routing: {line}")
    expected_warps = int(controls.get("WEAVE_SM90_COMM_WARPS", "0"))
    requested = re.search(r"requested_comm_warps=(\d+)", line)
    if not requested or int(requested.group(1)) != expected_warps:
        raise RuntimeError(f"arm={name}: requested comm warps do not match policy: {line}")
    if expected_warps:
        match = re.search(r"selected_dispatch_threads=(\d+)", line)
        if not match or int(match.group(1)) != expected_warps * 32:
            raise RuntimeError(f"arm={name}: forced comm warps do not match policy: {line}")
    for key in ("HEAD_ALL", "TAIL_ALL"):
        expected = controls.get("WEAVE_SM90_" + key)
        if expected is None:
            continue
        match = re.search(key.lower() + r"=(true|false|0|1)", line, re.IGNORECASE)
        if not match:
            raise RuntimeError(f"arm={name}: policy line is missing {key.lower()}: {line}")
        actual = match.group(1).lower() in ("true", "1")
        if actual != (expected == "1"):
            raise RuntimeError(f"arm={name}: forced {key.lower()} does not match policy: {line}")
    scatter = controls.get("WEAVE_SM90_SCATTER_COMM", "-1")
    if scatter in ("0", "1"):
        match = re.search(r"scatter_comm=(true|false|0|1)", line, re.IGNORECASE)
        if not match:
            raise RuntimeError(f"arm={name}: policy line is missing scatter_comm: {line}")
        actual = match.group(1).lower() in ("true", "1")
        if actual != (scatter == "1"):
            raise RuntimeError(f"arm={name}: forced scatter_comm does not match policy: {line}")


def check_spill_logs(name, log_text):
    matches = list(re.finditer(r"(\d+)\s+bytes?\s+spill stores.*?(\d+)\s+bytes?\s+spill loads",
                               log_text, re.IGNORECASE))
    if len(matches) < 2:
        raise RuntimeError(f"arm={name}: expected ptxas spill records for both SM90 phases, found {len(matches)}")
    for match in matches:
        stores, loads = map(int, match.groups())
        if stores or loads:
            raise RuntimeError(f"arm={name}: ptxas reported spill stores={stores}, loads={loads}")


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkout", required=True, help="DeepGEMM checkout path to run")
    p.add_argument("--official-checkout", help="clean official b64107f2 checkout")
    p.add_argument("--output-dir", default="./weave_sm90_validation", help="directory for tensor artifacts and JSON summaries")
    p.add_argument("--tokens", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--hidden", type=int, default=3072)
    p.add_argument("--intermediate", type=int, default=1536)
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--routing", choices=("randn", "skew", "masked"), default="randn")
    p.add_argument("--masked-ratio", type=float, default=0.25)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--arm", default="", help=argparse.SUPPRESS)
    p.add_argument("--output-file", default="", help=argparse.SUPPRESS)
    p.add_argument("--compare-file", default="", help=argparse.SUPPRESS)
    return p


def stats(actual, reference):
    import torch
    a = actual.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    b = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    d = a - b
    rms = torch.sqrt(torch.mean(d * d)).item()
    ref_rms = torch.sqrt(torch.mean(b * b)).item()
    return {
        "elements": int(a.numel()),
        "max_abs": torch.max(torch.abs(d)).item() if a.numel() else 0.0,
        "rms": rms,
        "relative_rms": rms / max(ref_rms, 1e-30),
        "reference_rms": ref_rms,
        "relative_l2": torch.linalg.vector_norm(d).item() / max(torch.linalg.vector_norm(b).item(), 1e-30),
    }


def fp64_reference(x_all, w1, w2, route_idx_all, route_weight_all, global_expert_start, activation_clamp, group):
    """FP64 MoE oracle over every rank's BF16 randn inputs and complete routing."""
    import torch
    tokens, hidden = x_all.shape
    out = torch.zeros((tokens, hidden), dtype=torch.float64, device=x_all.device)
    xd = x_all.to(torch.float64)
    for local_e in range(w1.shape[0]):
        global_e = global_expert_start + local_e
        slots = (route_idx_all == global_e).nonzero(as_tuple=False)
        if slots.numel() == 0:
            continue
        token_ids, topk_slots = slots[:, 0], slots[:, 1]
        h = xd[token_ids] @ w1[local_e].to(torch.float64).T
        gate, up = h.chunk(2, dim=-1)
        gate = gate.clamp(max=activation_clamp)
        up = up.clamp(min=-activation_clamp, max=activation_clamp)
        mid = torch.nn.functional.silu(gate) * up
        y = mid @ w2[local_e].to(torch.float64).T
        y *= route_weight_all[token_ids, topk_slots].to(torch.float64).unsqueeze(-1)
        out.index_add_(0, token_ids, y)
    if group.size() > 1:
        import torch.distributed as dist
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out


def worker(args):
    root = pathlib.Path(args.checkout).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError("checkout does not exist: " + str(root))
    sys.path.insert(0, str(root))
    import torch
    import torch.distributed as dist
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8
    deep_gemm_file = pathlib.Path(deep_gemm.__file__).resolve()
    try:
        deep_gemm_file.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"deep_gemm loaded from outside selected checkout: {deep_gemm_file}") from exc

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", str(args.world_size)))
    if world != args.world_size:
        raise RuntimeError(f"torch distributed WORLD_SIZE={world}, requested {args.world_size}")
    torch.cuda.set_device(local_rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world)
    group = dist.new_group(list(range(world)))
    if args.experts % world:
        raise ValueError("experts must be divisible by world-size")
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("SM90 validation requires a Hopper GPU (compute capability 9.x)")
    local_experts = args.experts // world
    expert_start = rank * local_experts
    # Each rank gets distinct source tokens/routes; expert weights are seeded
    # by global expert id so every checkout receives the same model.
    rank_seed = args.seed + rank * 1000003
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    x = torch.randn((args.tokens, args.hidden), device="cuda", dtype=torch.bfloat16)
    scores = torch.randn((args.tokens, args.experts), device="cuda", dtype=torch.float32)
    if args.routing == "skew":
        bias = torch.zeros(args.experts, device="cuda", dtype=torch.float32)
        bias[:max(1, args.experts // 8)] = 4.0
        scores += bias
    route_weight, route_idx = torch.topk(scores, args.topk, dim=-1, sorted=False)
    if args.routing == "masked":
        masked = torch.rand_like(route_weight) < args.masked_ratio
        route_idx.masked_fill_(masked, -1)
        route_weight.masked_fill_(masked, 0.0)
    w1_list, w2_list = [], []
    for e in range(expert_start, expert_start + local_experts):
        gen = torch.Generator(device="cuda").manual_seed(args.seed + 104729 * (e + 1))
        w1_list.append(torch.randn((2 * args.intermediate, args.hidden), generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05)
        w2_list.append(torch.randn((args.hidden, args.intermediate), generator=gen, device="cuda", dtype=torch.bfloat16) * 0.05)
    w1_bf16, w2_bf16 = torch.stack(w1_list), torch.stack(w2_list)
    x_parts = [torch.empty_like(x) for _ in range(world)]
    route_parts = [torch.empty_like(route_idx) for _ in range(world)]
    route_weight_parts = [torch.empty_like(route_weight) for _ in range(world)]
    dist.all_gather(x_parts, x, group=group)
    dist.all_gather(route_parts, route_idx, group=group)
    dist.all_gather(route_weight_parts, route_weight, group=group)
    x_all = torch.cat(x_parts, dim=0)
    route_idx_all = torch.cat(route_parts, dim=0)
    route_weight_all = torch.cat(route_weight_parts, dim=0)
    recv_counts = torch.bincount(
        route_idx_all[route_idx_all >= 0].reshape(-1), minlength=args.experts,
    )[expert_start:expert_start + local_experts]
    recv_count_parts = [torch.empty_like(recv_counts) for _ in range(world)]
    dist.all_gather(recv_count_parts, recv_counts, group=group)
    if os.environ.get("WEAVE_SM90_ENABLE") == "1":
        os.environ["WEAVE_SM90_EXPERT_COUNTS"] = ",".join(str(int(v)) for v in recv_counts.cpu().tolist())
    else:
        os.environ.pop("WEAVE_SM90_EXPERT_COUNTS", None)
    oracle_all = fp64_reference(x_all, w1_bf16, w2_bf16, route_idx_all,
                                route_weight_all, expert_start, 10.0, group)

    def quant_weights(weights):
        count, n, k = weights.shape
        fp8 = torch.empty((count, n, k), device="cuda", dtype=torch.float8_e4m3fn)
        sf = torch.empty((count, n // 128, k // 128), device="cuda", dtype=torch.float32)
        for e in range(count):
            blocks = weights[e].float().view(n // 128, 128, k // 128, 128)
            scales = blocks.abs().amax(dim=(1, 3)).clamp(min=1e-4) / 448.0
            fp8[e].copy_((blocks / scales[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k))
            sf[e].copy_(scales)
        return fp8, sf.contiguous()

    if args.hidden % 256 or args.intermediate % 128:
        raise ValueError("SM90 requires hidden multiple of 256 and intermediate multiple of 128")
    x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=128, use_packed_ue8m0=False)
    transformed = deep_gemm.transform_weights_for_mega_moe_sm90(quant_weights(w1_bf16), quant_weights(w2_bf16))
    buffer = deep_gemm.get_symm_buffer_for_sm90_mega_moe(
        group, args.experts, args.max_tokens, args.topk, args.hidden, args.intermediate,
    )
    y = torch.empty((args.tokens, args.hidden), device="cuda", dtype=torch.bfloat16)
    stats_out = torch.zeros(local_experts, device="cuda", dtype=torch.int32)
    outputs_by_repeat = []
    stats_by_repeat = []
    for _ in range(args.repeats):
        buffer.x[:args.tokens].copy_(x_fp8)
        buffer.x_sf[:args.tokens].copy_(x_sf)
        buffer.topk_idx[:args.tokens].copy_(route_idx)
        buffer.topk_weights[:args.tokens].copy_(route_weight)
        y.fill_(float("nan"))
        stats_out.zero_()
        deep_gemm.fp8_mega_moe(
            y, transformed[0], transformed[1], buffer,
            cumulative_local_expert_recv_stats=stats_out,
            recipe=(128, 128, 128), activation="swiglu", activation_clamp=10.0, fast_math=True,
        )
        torch.cuda.synchronize()
        output_parts = [torch.empty_like(y) for _ in range(world)]
        stats_parts = [torch.empty_like(stats_out) for _ in range(world)]
        dist.all_gather(output_parts, y, group=group)
        dist.all_gather(stats_parts, stats_out, group=group)
        outputs_by_repeat.append(torch.cat(output_parts, dim=0).cpu())
        stats_by_repeat.append([v.cpu() for v in stats_parts])
        dist.barrier(group=group)
    output_all = outputs_by_repeat[-1]
    local_digest = hashlib.sha256()
    for tensor in (x, route_idx, route_weight, w1_bf16, w2_bf16):
        local_digest.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    rank_digests = [None for _ in range(world)]
    dist.all_gather_object(rank_digests, local_digest.hexdigest(), group=group)
    ticket_hash = hashlib.sha256("".join(rank_digests).encode("ascii")).hexdigest()
    if rank == 0:
        oracle_stats_by_repeat = [stats(output, oracle_all) for output in outputs_by_repeat]
        repeated_bitwise_equal = all(
            torch.equal(outputs_by_repeat[0], output) for output in outputs_by_repeat[1:]
        )
        record = {
            "arm": args.arm, "checkout": str(root), "deep_gemm_file": str(deep_gemm_file),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True,
                                        capture_output=True, text=True).stdout.strip(),
            "env_controls": {key: os.environ.get(key) for key in (
                "WEAVE_SM90_ENABLE", "WEAVE_SM90_COMM_WARPS", "WEAVE_SM90_HEAD_ALL",
                "WEAVE_SM90_TAIL_ALL", "WEAVE_SM90_SCATTER_COMM", "WEAVE_SM90_EXPERT_COUNTS")},
            "seed": args.seed, "ticket_hash": ticket_hash,
            "shape": {"tokens": args.tokens, "hidden": args.hidden, "intermediate": args.intermediate,
                      "experts": args.experts, "topk": args.topk, "world_size": world,
                      "repeats": args.repeats, "routing_mode": args.routing,
                      "masked_ratio": args.masked_ratio if args.routing == "masked" else 0.0},
            "routing": {"indices": route_idx_all.cpu().tolist(), "weights": route_weight_all.cpu().tolist()},
            "expert_receive_counts_by_local_rank": [v.cpu().tolist() for v in recv_count_parts],
            "expert_receive_stats_by_repeat_rank": [
                [v.tolist() for v in stats_ranks] for stats_ranks in stats_by_repeat
            ],
            "fp64_oracle": oracle_stats_by_repeat[0],
            "fp64_oracle_by_repeat": oracle_stats_by_repeat,
            "output_shape": list(output_all.shape), "output_dtype": str(output_all.dtype),
            "output": outputs_by_repeat[0], "output_repeats": outputs_by_repeat,
            "oracle_output": oracle_all.cpu(),
            "output_all_finite": all(bool(torch.isfinite(output).all().item()) for output in outputs_by_repeat),
            "repeated_bitwise_equal": repeated_bitwise_equal,
        }
        if args.compare_file:
            baseline = torch.load(args.compare_file, map_location="cpu", weights_only=False)
            record["vs_official_by_repeat"] = [
                stats(output, reference)
                for output, reference in zip(outputs_by_repeat, baseline["output_repeats"])
            ]
            record["routing_equal_official"] = (
                record["routing"]["indices"] == baseline["routing"]["indices"]
                and record["routing"]["weights"] == baseline["routing"]["weights"]
            )
            record["ticket_hash_equal_official"] = ticket_hash == baseline["ticket_hash"]
            record["oracle_equal_official"] = torch.equal(record["oracle_output"], baseline["oracle_output"])
            record["no_worse_than_official"] = all(
                current["rms"] <= official["rms"] + 1e-12
                and current["max_abs"] <= official["max_abs"] + 1e-12
                for current, official in zip(oracle_stats_by_repeat, baseline["fp64_oracle_by_repeat"])
            )
            record["all_off_bitwise_equal"] = all(
                torch.equal(output, official)
                for output, official in zip(outputs_by_repeat, baseline["output_repeats"])
            )
        out_path = pathlib.Path(args.output_file).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, out_path)
        summary = {key: value for key, value in record.items()
                   if key not in ("output", "output_repeats", "oracle_output")}
        summary["artifact"] = str(out_path)
        summary_path = out_path.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        summary["summary_json"] = str(summary_path)
        print("WEAVE_SM90_RESULT " + json.dumps(summary, sort_keys=True), flush=True)
        if not record["output_all_finite"]:
            raise RuntimeError("kernel output contains NaN or Inf")
        if not record["repeated_bitwise_equal"]:
            raise RuntimeError("same-ticket repeated forwards are not bitwise equal")
        if args.compare_file and not record["no_worse_than_official"]:
            raise RuntimeError("FP64-oracle max_abs/rms regression versus the official output")
        for key in ("routing_equal_official", "ticket_hash_equal_official", "oracle_equal_official"):
            if args.compare_file and not record[key]:
                raise RuntimeError(key + " failed; this arm did not receive the same test ticket")
        if args.arm == "all_off" and not record["all_off_bitwise_equal"]:
            raise RuntimeError("all_off output is not bitwise equal to the independent official checkout")
    dist.barrier(group=group)
    buffer.destroy()
    dist.destroy_process_group()


ARMS = [
    ("official", {"WEAVE_SM90_ENABLE": "0"}),
    ("all_off", {"WEAVE_SM90_ENABLE": "0", "WEAVE_SM90_COMM_WARPS": "0", "WEAVE_SM90_HEAD_ALL": "0", "WEAVE_SM90_TAIL_ALL": "0", "WEAVE_SM90_SCATTER_COMM": "0"}),
    ("comm2", {"WEAVE_SM90_ENABLE": "1", "WEAVE_SM90_COMM_WARPS": "2", "WEAVE_SM90_HEAD_ALL": "0", "WEAVE_SM90_TAIL_ALL": "0", "WEAVE_SM90_SCATTER_COMM": "0"}),
    ("comm4", {"WEAVE_SM90_ENABLE": "1", "WEAVE_SM90_COMM_WARPS": "4", "WEAVE_SM90_HEAD_ALL": "0", "WEAVE_SM90_TAIL_ALL": "0", "WEAVE_SM90_SCATTER_COMM": "0"}),
    ("comm6", {"WEAVE_SM90_ENABLE": "1", "WEAVE_SM90_COMM_WARPS": "6", "WEAVE_SM90_HEAD_ALL": "0", "WEAVE_SM90_TAIL_ALL": "0", "WEAVE_SM90_SCATTER_COMM": "0"}),
    ("head_all", {"WEAVE_SM90_ENABLE": "1", "WEAVE_SM90_COMM_WARPS": "2", "WEAVE_SM90_HEAD_ALL": "1", "WEAVE_SM90_TAIL_ALL": "0", "WEAVE_SM90_SCATTER_COMM": "0"}),
    ("tail_all", {"WEAVE_SM90_ENABLE": "1", "WEAVE_SM90_COMM_WARPS": "2", "WEAVE_SM90_HEAD_ALL": "0", "WEAVE_SM90_TAIL_ALL": "1", "WEAVE_SM90_SCATTER_COMM": "0"}),
    ("scatter0", {"WEAVE_SM90_ENABLE": "1", "WEAVE_SM90_COMM_WARPS": "2", "WEAVE_SM90_HEAD_ALL": "0", "WEAVE_SM90_TAIL_ALL": "0", "WEAVE_SM90_SCATTER_COMM": "0"}),
    ("scatter1", {"WEAVE_SM90_ENABLE": "1", "WEAVE_SM90_COMM_WARPS": "2", "WEAVE_SM90_HEAD_ALL": "0", "WEAVE_SM90_TAIL_ALL": "0", "WEAVE_SM90_SCATTER_COMM": "1"}),
    ("full_auto", {"WEAVE_SM90_ENABLE": "1", "WEAVE_SM90_COMM_WARPS": "0", "WEAVE_SM90_HEAD_ALL": "1", "WEAVE_SM90_TAIL_ALL": "1", "WEAVE_SM90_SCATTER_COMM": "-1"}),
]


def orchestrate(args):
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3")
    if args.world_size < 1 or args.experts % args.world_size:
        raise ValueError("world-size must be positive and divide experts")
    if args.tokens < 1 or args.max_tokens < args.tokens:
        raise ValueError("require 1 <= tokens <= max-tokens")
    if args.hidden % 256 or args.intermediate % 128:
        raise ValueError("hidden must be divisible by 256 and intermediate by 128")
    if not 1 <= args.topk <= args.experts:
        raise ValueError("topk must be between 1 and experts")
    if not 0.0 <= args.masked_ratio <= 1.0:
        raise ValueError("masked-ratio must be between 0 and 1")
    check_gpu_idle()
    if not args.official_checkout:
        raise ValueError("--official-checkout is required in orchestrator mode")
    outdir = pathlib.Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    own_checkout = pathlib.Path(args.checkout).expanduser().resolve()
    official, official_head = check_official_checkout(args.official_checkout)
    if not own_checkout.is_dir():
        raise FileNotFoundError("weave checkout does not exist: " + str(own_checkout))
    print("Output directory: " + str(outdir), flush=True)
    arms = [("official", official, ARMS[0][1])] + [(name, own_checkout, env) for name, env in ARMS[1:]]
    baseline_file = None
    results = []
    for name, checkout, arm_env in arms:
        check_gpu_idle()
        arm_dir = outdir / name
        jit_dir = arm_dir / "jit"
        scratch_dir = arm_dir / "scratch"
        arm_dir.mkdir(parents=True, exist_ok=True)
        jit_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        artifact = outdir / (name + ".pt")
        log_file = outdir / (name + ".log")
        env = os.environ.copy()
        env.update(arm_env)
        env.pop("WEAVE_SM90_EXPERT_COUNTS", None)
        env.update({"DG_JIT_CHECK_NO_SPILLS": "1", "DG_JIT_DEBUG": "1",
                    "DG_PRINT_CONFIGS": "1", "DG_JIT_CACHE_DIR": str(jit_dir.resolve()),
                    "TMPDIR": str(scratch_dir.resolve()), "TMP": str(scratch_dir.resolve()),
                    "TEMP": str(scratch_dir.resolve())})
        if name == "official":
            for key in ("WEAVE_SM90_COMM_WARPS", "WEAVE_SM90_HEAD_ALL", "WEAVE_SM90_TAIL_ALL", "WEAVE_SM90_SCATTER_COMM"):
                env.pop(key, None)
        if args.world_size == 1:
            env.update({"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
                        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29517"})
        else:
            for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)
        env["PYTHONUNBUFFERED"] = "1"
        worker_args = [str(pathlib.Path(__file__).resolve()), "--worker",
                   "--checkout", str(checkout), "--output-dir", str(outdir),
                   "--tokens", str(args.tokens), "--max-tokens", str(args.max_tokens),
                   "--hidden", str(args.hidden), "--intermediate", str(args.intermediate),
                   "--experts", str(args.experts), "--topk", str(args.topk), "--seed", str(args.seed),
                   "--world-size", str(args.world_size), "--repeats", str(args.repeats),
                   "--routing", args.routing, "--masked-ratio", str(args.masked_ratio),
                   "--arm", name, "--output-file", str(artifact)]
        if baseline_file:
            worker_args += ["--compare-file", str(baseline_file)]
        if args.world_size > 1:
            launcher = shutil.which("torchrun")
            if launcher is None:
                raise RuntimeError("torchrun is required when world-size is greater than one")
            command = [launcher, "--standalone", "--nproc_per_node", str(args.world_size)] + worker_args
        else:
            command = [sys.executable] + worker_args
        print("Starting arm=" + name + " checkout=" + str(checkout) + " env=" + json.dumps(arm_env, sort_keys=True), flush=True)
        proc = subprocess.run(command, env=env, cwd=checkout, text=True, capture_output=True)
        full_log = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
        log_file.write_text(full_log, encoding="utf-8")
        print("Arm log: " + str(log_file), flush=True)
        print(proc.stdout, end="", flush=True)
        if proc.returncode:
            print(proc.stderr, file=sys.stderr, flush=True)
            raise RuntimeError("arm=" + name + " failed with exit code=" + str(proc.returncode))
        check_spill_logs(name, full_log)
        check_arm_policy(name, arm_env, full_log)
        if name == "official":
            baseline_file = artifact
        results.append({"arm": name, "artifact": str(artifact),
                        "summary_json": str(artifact.with_suffix(".json")),
                        "log": str(log_file),
                        "exit_code": proc.returncode})
    index = {"seed": args.seed, "official_head": official_head,
             "output_dir": str(outdir), "arms": results}
    index_file = outdir / "index.json"
    index_file.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print("Summary index: " + str(index_file), flush=True)


if __name__ == "__main__":
    parsed = make_parser().parse_args()
    if parsed.worker:
        worker(parsed)
    else:
        orchestrate(parsed)
