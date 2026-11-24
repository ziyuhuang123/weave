#!/usr/bin/env python3
"""CPU-only static check: the fused SM90 (H20) MegaMoE kernel has no coarse
all-SM / cross-rank barrier between operators.

What is checked (source level, comments stripped):
  1. kFusedGridSyncIndex / kFusedPhaseBarrierIdx are gone, and the fused branch
     of for_each_selected_block_role only calls run_interleaved_l1_l2_blocks
     (no comm::grid_sync between GEMM1 and GEMM2).
  2. The fused kernel statically requires the barrier-free K-block interleave
     (modes 5/6/7) -- there is no all-L1 -> grid_sync -> all-L2 fallback.
  3. On the fused comm_owned SMEM engine (kSmemStreamCombineActive) the
     Scatter -> Combine handoff contains no comm::grid_sync and no
     kBeforeCombineReduceBarrierTag nvlink_barrier; the tail is ready-gated on
     per-(token, item) arrival counters.
  4. The Notify -> Pull handoff on the fused compact-notify engine
     (kGridFreeNotify) skips the Notify grid_sync and the
     kBeforeDispatchPullBarrierTag nvlink_barrier.
  5. Every remaining comm::grid_sync / comm::nvlink_barrier call site in the
     kernel is listed in an explicit allow-list with the reason it is not on
     the fused default path (or is the end-of-kernel workspace clean).
  6. The host resolver forces interleave 5/6/7 for every fused launch.

This is a static source check, not CUDA correctness evidence.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
KERNEL = ROOT / "deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh"
HOST = ROOT / "csrc/jit_kernels/heuristics/sm90_mega_moe.hpp"


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def block_after(text: str, start: str) -> str:
    """Return the brace-balanced block that starts at the first '{' after `start`."""
    i = text.index(start)
    j = text.index("{", i)
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j:k + 1]
    raise AssertionError("unbalanced block after " + start)


def main() -> int:
    raw = KERNEL.read_text(encoding="utf-8").replace("\r\n", "\n")
    src = strip_comments(raw)
    host = strip_comments(HOST.read_text(encoding="utf-8").replace("\r\n", "\n"))
    failures = []

    def check(cond, msg):
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            failures.append(msg)

    # 1. GEMM1 -> GEMM2
    check("kFusedGridSyncIndex" not in raw, "kFusedGridSyncIndex is 100% gone (incl. comments)")
    check("kFusedPhaseBarrierIdx" not in src, "kFusedPhaseBarrierIdx is gone")
    role = block_after(src, "const auto for_each_selected_block_role")
    fused = block_after(role, "if constexpr (MegaMoEPhase::is_fused)")
    check("grid_sync" not in fused and "nvlink_barrier" not in fused,
          "fused branch of for_each_selected_block_role has no grid_sync / nvlink_barrier")
    check(re.sub(r"\s+", "", fused) == "{run_interleaved_l1_l2_blocks(role_id,fused_tid,func);}",
          "fused branch only calls run_interleaved_l1_l2_blocks")
    interleave = block_after(src, "const auto run_interleaved_l1_l2_blocks")
    check("grid_sync" not in interleave and "nvlink_barrier" not in interleave,
          "run_interleaved_l1_l2_blocks contains no grid_sync / nvlink_barrier")

    # 2. no fallback
    check(re.search(r"DG_STATIC_ASSERT\(\s*not\s+MegaMoEPhase::is_fused\s+or\s+\(kAgyInterleaveActive\s+and\s+kAgyKbDepInterleave\)", src)
          is not None, "fused kernel static-asserts the barrier-free K-block interleave")

    # 3. Scatter -> Combine
    m = re.search(r"if constexpr \(kSmemStreamCombineActive\) \{\s*const uint32_t epi_warp_in_cta", src)
    check(m is not None, "epilogue Combine handoff block on the SMEM streaming engine found")
    if m:
        epi = block_after(src[m.start():], "if constexpr (kSmemStreamCombineActive)")
        check("grid_sync" not in epi and "nvlink_barrier" not in epi
              and "kBeforeCombineReduceBarrierTag" not in epi,
              "Scatter -> Combine handoff (epilogue) has no grid_sync / nvlink_barrier")
        check("kAllWarpCombineBarrierIdx" in epi and "run_stream_smem_tail_combine" in epi,
              "epilogue enters the tail via the CTA-local all-warp barrier")
    tail = block_after(src, "const auto run_stream_smem_tail_combine")
    check("ld_acq_sys(arrival_ptr)" in tail, "tail Combine is ready-gated on ld.acquire.sys arrival counters")
    check("grid_sync" not in tail and "nvlink_barrier" not in tail, "tail Combine has no barrier")
    check(re.search(r"constexpr bool kSmemStreamCombineActive\s*=\s*MegaMoEPhase::is_fused and kScatDispActive and \(not kHbmScatQueueActive\) and\s*\(not kCombineSyncChunkActive\);", src) is not None,
          "arrival counters are published independently of WEAVE_REDUCE_WARPS")
    check("WEAVE_OVERLAP_BAR2" not in src.split("#ifndef WEAVE_OVERLAP_BAR2")[-1].split("#endif", 1)[-1],
          "WEAVE_OVERLAP_BAR2 before-combine barrier variant removed from the kernel body")

    # 4. Notify -> Pull
    check("constexpr bool kGridFreeNotify" in src, "kGridFreeNotify defined")
    notify_gate = re.search(r"if constexpr \(not kGridFreeNotify\) \{(.*?)\} *\n", src, flags=re.S)
    check(notify_gate is not None and "kBeforeDispatchPullBarrierTag" in src[notify_gate.start():notify_gate.start() + 6000]
          and "grid_sync" in src[notify_gate.start():notify_gate.start() + 400],
          "Notify grid_sync and before-dispatch-pull nvlink_barrier are behind `not kGridFreeNotify`")
    check("(static_cast<uint32_t>(v >> 32) != kNumRanks)" in src, "receiver waits for per-expert kNumRanks readiness")
    # Without the Notify barrier, every reader of the per-(source rank, expert) counts must
    # itself acquire the expert's readiness word first (the rank-count cache is loaded by
    # dispatch warps 1.. in parallel with warp 0's readiness wait).
    rc = re.search(r"else if constexpr \(kAgyCacheRankCounts\) \{(.*?)set_rank_count", src, flags=re.S)
    check(rc is not None and "if constexpr (kGridFreeNotify)" in rc.group(1)
          and "ld_acq_sys(workspace.get_expert_recv_count_sum_ptr(e))" in rc.group(1)
          and rc.group(1).index("ld_acq_sys(workspace.get_expert_recv_count_sum_ptr(e))")
              < rc.group(1).index("get_expert_recv_count_ptr(r, e)"),
          "rank-count cache load is ordered after the per-expert readiness acquire")

    # 5. allow-list of remaining barrier call sites (by enclosing guard)
    sites = [mm.start() for mm in re.finditer(r"comm::(grid_sync|nvlink_barrier)<", src)]
    allowed = 0
    for pos in sites:
        ctx = src[max(0, pos - 2500):pos + 400]
        if "kAfterWorkspaceCleanBarrierTag" in src[pos:pos + 400]:
            allowed += 1          # end-of-kernel workspace clean (allowed)
        elif "End-of-kernel" in raw[raw.find(src[pos - 120:pos].strip()[-60:]) - 2500: raw.find(src[pos - 120:pos].strip()[-60:]) + 60] if src[pos - 120:pos].strip() else False:
            allowed += 1
        elif "if constexpr (not kGridFreeNotify)" in ctx:
            allowed += 1          # legacy Notify path (hetero / chunked / non-compact)
        elif "kCombineSyncChunkActive" in ctx:
            allowed += 1          # hbm_dedup chunked engine
        elif "else if constexpr (not kCombineSyncChunkActive)" in ctx or "kBeforeCombineReduceBarrierTag" in src[pos:pos + 400]:
            allowed += 1          # non-SMEM-engine fused / split-kernel Combine (not the comm_only+smem path)
        elif "kSmemStreamCombineActive" in ctx and "kDispatchGridSyncIndex" in src[pos:pos + 300]:
            allowed += 1          # end-of-kernel clean rendezvous after y is complete
        else:
            print("  unclassified barrier site:", src[pos:pos + 160].replace("\n", " "))
    check(allowed == len(sites), f"all {len(sites)} remaining barrier sites are allow-listed (classified {allowed})")

    # 6. host
    check("resolve_sm90_fused_interleave_mode" in host and "kSm90FusedDefaultInterleave = 7" in host,
          "host resolver forces the barrier-free interleave for fused launches")
    check(host.count("resolve_sm90_fused_interleave_mode(") >= 4,
          "JIT defines + both SMEM mirrors + config path use the same resolver")
    check("interleave_smem_scat" not in host and "kInterleaveSmemScat" not in src,
          "old CD-region scatter staging (kInterleaveSmemScat) removed on host and device")
    check(re.search(r"auto\* smem_scat_l2\s*=\s*reinterpret_cast<nv_bfloat16\*>\(smem_buffer\);", src) is not None,
          "L2 scatter staging is smem_buffer (disjoint from smem_cd_base)")

    print("FAILURES =", len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
