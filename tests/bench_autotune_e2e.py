#!/usr/bin/env python3
"""End-to-end check of the Weave autotune -> zero-env runtime auto-selection path.

For every (S_global, routing) cell, in ONE process group and with the SAME inputs:
  auto      weave_mega_moe() with ZERO WEAVE_* variables (calibrated profile + runtime rules)
  default   weave_mega_moe() with WEAVE_AUTOTUNE=0 (compiled defaults, nothing injected)
  cold      the architecture-family cold-start env (only when it differs from default)
  hand:*    hand-tuned arms (the winners of the earlier exhaustive public-knob sweeps on this
            architecture family, expressed as SM-count fractions); oracle = fastest hand arm
  official  official DeepGEMM (SM90: in-process WEAVE_FORCE_OFFICIAL=1; SM10x: measured by the
            separate upstream checkout script and merged afterwards)
Accuracy: every arm vs an FP32 reference over dequantized operands (all rows), plus vs the
auto arm.  Timing: palindromic multi-arm windows, slowest rank per window (CUDA-graph replay
with L2 flush on SM10x, aligned eager launches on SM90).

Usage (EP4 example):
  python tests/bench_autotune_e2e.py --model qwen3.8-2.4t --s-list 128,1024,4096,8192,16384 \
      --routings uniform,skew --num-processes 4 --out e2e.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("WEAVE_REPO_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def hand_arms(family, hw):
    from deep_gemm.mega.autotune import sm_fraction, SCATTER_COMM
    if family == "blackwell":
        return {
            "hand:dispatch_sms_0.5": {"WEAVE_DISPATCH_ACTIVE_SMS": str(sm_fraction(0.5, hw))},
            "hand:g96_16_8": {"WEAVE_GEMM1_CONFIG": "96:16", "WEAVE_GEMM2_CONFIG": "96:8"},
            "hand:g64_16_8": {"WEAVE_GEMM1_CONFIG": "64:16", "WEAVE_GEMM2_CONFIG": "64:8"},
            "hand:g128_canonical": {"WEAVE_GEMM1_CONFIG": "128:32:128:0:1", "WEAVE_GEMM2_CONFIG": "128:32:128:0:1"},
            "hand:v_head_1": {"WEAVE_V_HEAD": "1"},
            "hand:v_head_2": {"WEAVE_V_HEAD": "2"},
        }
    d85, d67 = sm_fraction(0.85, hw), sm_fraction(0.67, hw)
    return {
        "hand:comm_smem": dict(SCATTER_COMM),
        "hand:comm_smem_dw8": {**SCATTER_COMM, "WEAVE_DISPATCH_WARPS": "8"},
        "hand:comm_smem_g128": {**SCATTER_COMM, "WEAVE_GEMM1_CONFIG": "128", "WEAVE_GEMM2_CONFIG": "128"},
        "hand:comm_smem_g32_64": {**SCATTER_COMM, "WEAVE_GEMM1_CONFIG": "32", "WEAVE_GEMM2_CONFIG": "64"},
        "hand:comm_smem_d0.85": {**SCATTER_COMM, "WEAVE_DISPATCH_ACTIVE_SMS": str(d85)},
        "hand:comm_smem_dk0.67": {**SCATTER_COMM, "WEAVE_DISPATCH_ACTIVE_SMS": str(d67),
                                  "WEAVE_SCATTER_ACTIVE_SMS": str(d67)},
    }


def _clear_weave_env():
    """Drop manual WEAVE_* tuning variables (keeps WEAVE_AUTOTUNE_DIR / _FILE / WEAVE_REPO_ROOT ...)."""
    from deep_gemm.mega.autotune import manual_env_keys
    for k in manual_env_keys():
        os.environ.pop(k)


def worker(local_rank, args):
    from deep_gemm.utils.dist import init_dist
    rank, world, group = init_dist(local_rank, args.num_processes)
    _clear_weave_env()
    from deep_gemm.mega import autotune as at
    hw = at.query_fingerprint(world)
    fam = hw.family
    E, K, H, I = at.MODEL_PRESETS[args.model]
    mma = at.default_mma_type(fam)
    s_list = [int(s) for s in args.s_list.split(",")]
    buf_s = max([args.buffer_s] + s_list)  # same symmetric-buffer size as the autotune run
    wl = at.MoEWorkload(group, hw, E, K, H, I, mma, (buf_s + world - 1) // world, seed=args.seed)
    profile = at.profile_path(hw, E, K, H, I, 0, mma)
    graph_mode = fam == "blackwell"
    flush = torch.empty(int(2 * max(hw.l2_bytes, 64 << 20) // 4), dtype=torch.int32, device="cuda")
    sync_tok = torch.ones((1,), dtype=torch.int32, device="cuda")
    out = dict(schema="weave_autotune_e2e/v1", hw=hw.to_dict(), model=args.model,
               dims=dict(E=E, K=K, H=H, I=I), mma_type=mma, world=world,
               profile_present=profile.is_file(), profile_file=profile.name,
               timing=("cuda_graph_replay_l2_flush" if graph_mode else "aligned_eager_launch") +
                      f", palindromic windows x{args.rounds}, slowest rank per window",
               cells=[])
    if profile.is_file():
        out["profile_summary"] = {k: v for k, v in at.load_profile(profile).items()
                                  if k in ("hw", "model", "thresholds", "wall_seconds", "method")}

    def make_launch(kind):
        if kind == "auto":
            return lambda: wl.launch(use_weave_entry=True)
        if kind == "default":
            def f():
                os.environ["WEAVE_AUTOTUNE"] = "0"
                try:
                    wl.launch(use_weave_entry=True)
                finally:
                    os.environ.pop("WEAVE_AUTOTUNE", None)
            return f
        if kind == "official":
            return lambda: wl.launch()
        return lambda: wl.launch(use_weave_entry=True)   # explicit env -> selector source "manual"

    for s in s_list:
        for routing in args.routings.split(","):
            wl.set_inputs(s, routing)
            tokens = wl.tokens
            ref = wl.reference_rows(torch.arange(tokens, device="cuda"))
            arms = {"auto": ({}, "auto"), "default": ({}, "default")}
            cold = at.cold_start_env(fam, hw, s)
            if cold:
                arms["cold_start"] = (cold, "env")
            for n, e in hand_arms(fam, hw).items():
                arms[n] = (e, "env")
            if fam == "hopper" and args.official:
                arms["official"] = ({"WEAVE_FORCE_OFFICIAL": "1"}, "official")
            cell = dict(S=s, routing=routing, tokens_per_rank=tokens, arms={})
            # accuracy + warm-up (the auto arm's eager call also takes the routing decision)
            y_auto = None
            for name, (env, kind) in arms.items():
                launch = make_launch(kind)
                ok_compile = 1.0
                try:
                    with at.scoped_env(env):
                        wl.y.zero_()
                        launch()
                    torch.cuda.synchronize()
                except Exception as exc:
                    ok_compile = 0.0
                    msg = str(exc)[:300]
                agreed = at._all_reduce_scalar(ok_compile, "min", group)
                if agreed < 1.0:
                    cell["arms"][name] = dict(env=env, status="failed", error=msg if ok_compile < 1 else "other rank")
                    continue
                y = wl.y.clone()
                finite = at._all_reduce_scalar(1.0 if bool(torch.isfinite(y).all()) else 0.0, "min", group)
                cos = at._all_reduce_scalar(at._cos(y, ref), "min", group)
                yf, rf = y.float(), ref.float()
                rel = float(torch.linalg.vector_norm(yf - rf) / torch.linalg.vector_norm(rf).clamp_min(1e-30))
                rel = at._all_reduce_scalar(rel, "max", group)
                ent = dict(env=env, finite=finite >= 1.0, cos_vs_fp32=round(cos, 6), rel_l2_vs_fp32=round(rel, 6))
                if name == "auto":
                    sel = at.get_runtime_selector().last
                    ent["selection"] = dict(source=sel.source, bucket_S=sel.bucket_s, env=sel.env,
                                            rules_fired=sel.rules_fired) if sel else None
                    y_auto = y
                elif y_auto is not None:
                    ent["cos_vs_auto"] = round(at._all_reduce_scalar(at._cos(y, y_auto), "min", group), 6)
                ent["status"] = "ok" if (finite >= 1.0 and cos > args.cos_gate) else "accuracy_failed"
                cell["arms"][name] = ent
            names = [n for n, e in cell["arms"].items() if e["status"] == "ok"]
            # timing
            graphs = {}
            if graph_mode:
                for n in names:
                    env, kind = arms[n]
                    launch = make_launch(kind)
                    with at.scoped_env(env):
                        st = torch.cuda.Stream()
                        st.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(st):
                            for _ in range(2):
                                launch()
                        torch.cuda.current_stream().wait_stream(st)
                        torch.cuda.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g):
                            launch()
                        torch.cuda.synchronize()
                    graphs[n] = g

            def slot(n, iters):
                env, kind = arms[n]
                if graph_mode:
                    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                          for _ in range(iters)]
                    for i in range(iters):
                        flush.zero_()
                        ev[i][0].record()
                        graphs[n].replay()
                        ev[i][1].record()
                    torch.cuda.synchronize()
                    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in ev)
                launch = make_launch(kind)
                vals = []
                with at.scoped_env(env):
                    for _ in range(iters):
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        dist.barrier(group=group)
                        torch.cuda._sleep(at.EAGER_HEAD_START_CYCLES)
                        dist.all_reduce(sync_tok, group=group)
                        a.record()
                        launch()
                        b.record()
                        b.synchronize()
                        vals.append(a.elapsed_time(b) * 1e3)
                return statistics.median(vals)

            t_probe = at._all_reduce_scalar(slot(names[0], 1), "max", group)
            iters = int(min(20, max(3, 15000.0 / max(t_probe, 1.0)))) if graph_mode else \
                int(min(4, max(2, 8000.0 / max(t_probe, 1.0))))
            for _ in range(1):  # warm-up window
                for n in names:
                    slot(n, 1)
            samples = []
            for r in range(args.rounds):
                order = names if r % 2 == 0 else list(reversed(names))
                for n in order:
                    samples.append((n, slot(n, iters)))
            t = torch.tensor([v for _, v in samples], dtype=torch.float64, device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
            vals = t.tolist()
            for n in names:
                v = [x for (m, _), x in zip(samples, vals) if m == n]
                cell["arms"][n].update(median_us=round(statistics.median(v), 3), min_us=round(min(v), 3),
                                       windows_us=[round(x, 3) for x in v])
            hand = {n: cell["arms"][n]["median_us"] for n in names if n.startswith("hand:")}
            if hand:
                best = min(hand, key=hand.get)
                cell["oracle"] = dict(arm=best, median_us=hand[best])
            if "auto" in names:
                a = cell["arms"]["auto"]["median_us"]
                cell["auto_over_oracle"] = round(a / cell["oracle"]["median_us"], 5) if hand else None
                cell["auto_over_default"] = round(a / cell["arms"]["default"]["median_us"], 5) \
                    if "default" in names else None
                if "official" in names:
                    cell["official_over_auto"] = round(cell["arms"]["official"]["median_us"] / a, 5)
            out["cells"].append(cell)
            graphs.clear()
            if rank == 0:
                brief = {n: cell["arms"][n].get("median_us") for n in cell["arms"]}
                print(f"[e2e] S={s} {routing}: {json.dumps(brief)} oracle={cell.get('oracle')} "
                      f"auto/oracle={cell.get('auto_over_oracle')} sel={cell['arms'].get('auto', {}).get('selection')}",
                      flush=True)
                Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")

    # host-side selector overhead (eager, cached statistics) on the last cell
    n_calls = 2000
    sel = at.get_runtime_selector()
    t0 = time.perf_counter()
    for _ in range(n_calls):
        at.select_launch_env(wl.buffer, wl.y, mma)
    t1 = time.perf_counter()
    with at.scoped_env({"WEAVE_GEMM1_CONFIG": "0"}):
        pass
    t2 = time.perf_counter()
    for _ in range(n_calls):
        with at.scoped_env(sel.last.env if sel.last else {}):
            pass
    t3 = time.perf_counter()
    out["selector_host_overhead_us"] = dict(select_per_call=round((t1 - t0) / n_calls * 1e6, 3),
                                            scoped_env_per_call=round((t3 - t2) / n_calls * 1e6, 3),
                                            note="eager call, statistics cached (static routing); "
                                                 "CUDA-graph replay adds zero host overhead")
    if rank == 0:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("WROTE", args.out, flush=True)
    dist.barrier(group=group)
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--s-list", default="128,1024,4096,8192,16384")
    ap.add_argument("--routings", default="uniform,skew")
    ap.add_argument("--num-processes", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--cos-gate", type=float, default=0.99)
    ap.add_argument("--official", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buffer-s", type=int, default=0,
                    help="size the symmetric buffer for this S_global (default: max of --s-list)")
    ap.add_argument("--precompile-jobs", type=int, default=0,
                    help="JIT every fixed arm (+ the profile's bucket / rule-variant envs) before the timed "
                         "run, without launching kernels (0 = auto jobs, -1 = skip)")
    ap.add_argument("--precompile-only", action="store_true")
    args = ap.parse_args()
    if args.precompile_jobs >= 0:
        precompile_arms(args)
    if args.precompile_only:
        return
    torch.multiprocessing.spawn(worker, args=(args,), nprocs=args.num_processes)


def precompile_arms(args):
    from deep_gemm.mega import autotune as at
    _clear_weave_env()
    hw = at.query_fingerprint(args.num_processes)
    fam = hw.family
    E, K, H, I = at.MODEL_PRESETS[args.model]
    mma = at.default_mma_type(fam)
    s_list = [int(s) for s in args.s_list.split(",")]
    buf_s = max([args.buffer_s] + s_list)
    prof_path = at.profile_path(hw, E, K, H, I, 0, mma)
    prof = at.load_profile(prof_path) if prof_path.is_file() else None
    cells, seen = [], set()
    for s in s_list:
        envs = [{}, at.cold_start_env(fam, hw, s)] + list(hand_arms(fam, hw).values())
        if fam == "hopper" and args.official:
            envs.append({"WEAVE_FORCE_OFFICIAL": "1"})
        if prof is not None:
            b = at.select_bucket(prof, s)
            envs += [dict(b["env"])] + [dict(v) for v in b.get("verified_variants", [])]
        for env in envs:
            key = (s, at.env_key(env))
            if key not in seen:
                seen.add(key)
                cells.append(dict(S=s, env=env))
    print(f"[e2e] precompile {len(cells)} cells (profile {'found' if prof else 'absent'})", flush=True)
    at.precompile(hw, E, K, H, I, s_list, jobs=args.precompile_jobs, gpus=[0], ptxas_check=False,
                  buffer_s_buckets=[buf_s], cells=cells)


if __name__ == "__main__":
    main()
