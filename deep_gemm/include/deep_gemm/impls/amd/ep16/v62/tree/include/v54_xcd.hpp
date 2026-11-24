#pragma once
// =============================================================================================================
// opus55 v54 (v54_qwen4) -- Fleet-inspired 4-XCD L2 partitioning + narrow-scope SC1/SC0 for the GEMM1->GEMM2
// hand-off on MI308X (gfx942, CDNA3).  EVERYTHING here is macro-gated: with V54_XCD_PART=V54_CACHE_S1=
// V54_CLEAN_SC1=0 the translation unit is identical to the v51_qwen4 base tree.
//
// Hardware premise (CDNA3 / MI308X): 80 CUs = 4 XCDs x 20 CUs.  Each XCD owns a PRIVATE, NON-COHERENT 4 MB L2
// ("GL1") above the shared 256 MB MALL.  A plain (write-back) global store lands in the producing XCD's L2 and
// reaches the MALL only on write-back / eviction; `buffer_inv sc1` invalidates the WHOLE 4 MB L2 of the issuing
// XCD (it is not address-ranged), so the v46 per-M-tile GEMM2 acquire also evicts every hot W1/W2 weight line.
//
// v54 lever 1 (V54_XCD_PART): the flat per-pass GEMM1/GEMM2 claim cursors become 4 per-XCD home queues.  The
// home of an M tile is `expert_id & 3` (descriptors are built expert-sorted by v46_build_pass, so a home queue
// is a set of WHOLE experts).  A compute group drives its own XCD's queue first and only then steals from
// (x+1)&3, (x+2)&3, (x+3)&3 -- with the SAME home filter, so a queue still hands out only its own tiles and no
// tile can ever be claimed twice.  Effects: (a) an expert's W1/W2 slices are re-read by one XCD only -> 4x less
// weight thrashing in the 4 MB L2; (b) GEMM1 and GEMM2 of a tile land on the same XCD unless stolen; (c) 4
// cursors instead of 1 -> 4x less atomic contention on the claim line.
//
// v54 lever 2 (V54_CACHE_S1): narrow the GEMM2 acquire.  The producer already publishes with a RELEASE agent
// atomic (= buffer_wbl2 sc1 + s_waitcnt vmcnt(0)), so the data is in the MALL before the done count is visible.
// A consumer on the SAME XCD as every producer of that tile therefore already has the newest line in its own
// L2 (its XCD's L2 is the coherence point for that XCD) and must only drop its per-CU L1 -> `buffer_inv sc0`.
// A consumer on a DIFFERENT XCD keeps the full `buffer_inv sc1`.  Same-XCD-ness is PROVEN, not assumed:
//   * home(mt) == my xcd  (the tile belongs to my home queue), AND
//   * V54_G1X[q][my xcd] == 0, i.e. no foreign XCD has ever claimed a GEMM1 item from my home queue in pass q.
//     Every GEMM1 item of a home-x tile is claimed from cursor (q,x); a claimant on another XCD is a stealer and
//     bumps that counter at CLAIM time.  The counter is monotone, so reading 0 AFTER the tile's done count
//     reached n1b proves all n1b producers ran on my XCD.
//   * V54_XMASK == 0xF: every compute CTA has OR-ed 1<<xcd into the mask before the plan rendezvous that gates
//     all GEMM work, so a degenerate / non-distinguishing XCD id (e.g. a register that reads back constant)
//     disables the narrow scope instead of silently corrupting the hand-off.
//
// plan_flags reservation (u32[1024], zeroed every launch).  Base map is in v46_sp.hpp; v48_tail owns 360..366,
// v48_plan owns 368..380, v46 hot cursors own 384..415, v46 control 448..479, plan words 480..511, CNT 512..767,
// OFF 768..1023, v46 sharded arrivals 272..335.  v54 adds ONLY words that no live path touches:
//   336..351  V54_G1X[4 passes][4 xcd]  foreign-GEMM1-claim counters (atomics; shares the 320..351 line with
//             V46_LS_SH, which is also atomics -> one access class per 128 B line is preserved).  The only other
//             user of this range was pf[340+b] in v41_mb_route_plan, which is called exclusively from the
//             !V46_SP branch of megamoe_amd.hip and is therefore not emitted in this TU.
//   352       V54_XMASK  OR of 1<<xcd over all compute CTAs (atomics; shares the 352..383 line with the v48
//             flags, also atomics).
//   416..447  V54_G1C[4][4] then V54_G2C[4][4]  per-XCD claim cursors.  This is one whole 128 B line
//             (416*4 = 13*128) holding atomics only, and it is the gap between the v46 hot line (384..415) and
//             the v46 control line (448..479).
// =============================================================================================================
#ifndef V54_XCD_PART
#define V54_XCD_PART 0
#endif
#ifndef V54_CACHE_S1
#define V54_CACHE_S1 0
#endif
#ifndef V54_CLEAN_SC1
#define V54_CLEAN_SC1 0
#endif
// 0 = home is `expert_id & 3` (Fleet's per-expert home).  1 = home is the contiguous quarter of the
// expert-sorted tile range (whole experts too, statically balanced in tile count, no descriptor load needed).
#ifndef V54_PART_MODE
#define V54_PART_MODE 0
#endif
// 0 = physical XCC id via s_getreg_b32 hwreg(HW_REG_XCC_ID); 1 = cta/20; 2 = cta%4 (proxies, see report).
#ifndef V54_XCD_SRC
#define V54_XCD_SRC 0
#endif
// same-XCD acquire instruction: 0 = `buffer_inv sc0` (L1 only, keeps the 4 MB L2 warm);
// 1 = `buffer_inv sc1` (identical to base -> CACHE_S1 becomes a pure no-op, used as the safety fallback).
#ifndef V54_S1_INV
#define V54_S1_INV 0
#endif
// v59 L3 (orthogonal to the Fleet levers above): GEMM1 writes stage1_out / stage1_scale with a system-scope
// write-through store (sc0 sc1) and GEMM2 reads A / sa with SC1, so the hand-off goes through the shared MALL
// and the GEMM2 acquire only drops the per-CU L1 (`buffer_inv sc0`).  Requires V54_CACHE_S1=0/V54_XCD_PART
// free; works with the global work-stealing claim path (V54_XCD_PART=0).
#ifndef V54_S1_BYPASS_MALL
#define V54_S1_BYPASS_MALL 0
#endif
#if V54_CACHE_S1 && !V54_XCD_PART
#error "V54_CACHE_S1 requires V54_XCD_PART (the same-XCD proof needs the home queues + steal counters)"
#endif
// v57: the coalesce scheduler and the v54 Fleet levers are mutually exclusive -- both claim
// plan_flags 416..447 (V55_G1CX vs V54_G1C/V54_G2C) and both rewrite the GEMM1 claim path.
#if defined(V55_COALESCE_SCHED) && V55_COALESCE_SCHED && (V54_XCD_PART || V54_CACHE_S1 || V54_CLEAN_SC1)
#error "V55_COALESCE_SCHED cannot be combined with V54_XCD_PART/V54_CACHE_S1/V54_CLEAN_SC1"
#endif

#if V54_XCD_PART || V54_CACHE_S1 || V54_CLEAN_SC1
#define V54_PFH   416u
#define V54_G1C   (V54_PFH + 0u)    // [4 passes][4 xcd] GEMM1 claim cursor (items = T*n1b)
#define V54_G2C   (V54_PFH + 16u)   // [4 passes][4 xcd] GEMM2 claim cursor (items = T*n2c)
#define V54_G1X   336u              // [4 passes][4 xcd] foreign-XCD GEMM1 claims taken from that home queue
#define V54_XMASK 352u              // OR of 1<<xcd over every compute CTA (must be 0xF for the narrow scope)

// Physical XCD of the wave issuing this instruction.  s_getreg_b32 writes an SGPR, and every wave of a CTA
// reads the same value (the CU's XCD), so the result is group-uniform without a readfirstlane.
__device__ __forceinline__ unsigned v54_xcd() {
#if V54_XCD_SRC == 0
  unsigned v;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(v));
  return v & 3u;
#elif V54_XCD_SRC == 1
  return __builtin_amdgcn_readfirstlane(blockIdx.x / 20u) & 3u;
#else
  return __builtin_amdgcn_readfirstlane(blockIdx.x & 3u);
#endif
}

// Home XCD of M tile `mtl` of a pass with T tiles.  Mode 0 needs the descriptor's expert id; mode 1 needs only
// mtl and T (descriptors are expert-sorted, so a contiguous quarter is a run of whole experts bar 3 boundaries).
__device__ __forceinline__ unsigned v54_home(unsigned expert_id, unsigned mtl, unsigned T) {
#if V54_PART_MODE == 0
  (void)mtl; (void)T;
  return expert_id & 3u;
#else
  (void)expert_id;
  const unsigned q1 = T >> 2u, q2 = T >> 1u, q3 = q1 + q2;
  return (mtl >= q3) ? 3u : ((mtl >= q2) ? 2u : ((mtl >= q1) ? 1u : 0u));
#endif
}

// Scan ONE home queue for its next claimable item.  Non-home tiles are skipped by CAS-jumping the cursor to the
// next tile boundary, so a queue only ever hands out tiles whose home == y, no matter who drives it.  MODE:
//   0 = GEMM1 claim (no readiness gate)
//   1 = GEMM2 whose GEMM1 is COMPLETE (s1d[mt] >= n1b)
//   2 = GEMM2 whose GEMM1 is fully CLAIMED (home G1 cursor >= (mtl+1)*n1b) -- the exact v46 phase-(3) test,
//       evaluated on the tile's home cursor instead of the old flat cursor.
// A blocked readiness gate returns NONE without advancing the cursor (same as the base `break`).
#define V54_BLOCKED 0xfffffffeu   // queue non-empty but its head tile is not ready yet (distinct from drained)

template <unsigned MODE>
__device__ __forceinline__ unsigned v54_scan(unsigned* cur, unsigned bound, unsigned nsub, unsigned y,
                                             unsigned T, unsigned B, const WeaveMTileDesc* tiles,
                                             const unsigned* s1d, unsigned n1b, const unsigned* g1c) {
  unsigned c = __hip_atomic_load(cur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  while (c < bound) {
    const unsigned mtl = c / nsub;
    const unsigned h = v54_home(tiles[B + mtl].expert_id, mtl, T);
    if (h != y) {
      const unsigned nc = (mtl + 1u) * nsub;
      if (__hip_atomic_compare_exchange_strong(cur, &c, nc, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT)) c = nc;
      continue;
    }
    if (MODE == 1u && __hip_atomic_load(s1d + B + mtl, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < n1b)
      return V54_BLOCKED;
    if (MODE == 2u && __hip_atomic_load(g1c + h, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (mtl + 1u) * n1b)
      return V54_BLOCKED;
    if (__hip_atomic_compare_exchange_strong(cur, &c, c + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT)) return c;
  }
  return V46_NONE;
}

// Own queue first, then steal round-robin (x+1)&3, (x+2)&3, (x+3)&3.  `from` receives the queue actually used,
// so the caller can tell a steal (from != xcd) from a home claim.  Fleet semantics: steal only once the OWN
// queue is EMPTY -- a merely readiness-blocked own head is not a reason to pull another XCD's tile over (that
// would defeat the L2 locality), and it is exactly the base v46 `break` behaviour.  Zero load-imbalance stall:
// a drained own queue always falls through to the other three.  One scan body per call site (the s loop is
// deliberately NOT unrolled) so the decision code stays the same size as the base flat-cursor version.
template <unsigned MODE>
__device__ __forceinline__ unsigned v54_claim(unsigned* curs, unsigned bound, unsigned nsub, unsigned T,
                                              unsigned B, const WeaveMTileDesc* tiles, unsigned xcd,
                                              unsigned n1b, const unsigned* s1d, const unsigned* g1c,
                                              unsigned& from) {
  from = xcd;
  #pragma clang loop unroll(disable)
  for (unsigned s = 0u; s < 4u; ++s) {
    const unsigned y = (xcd + s) & 3u;
    const unsigned c = v54_scan<MODE>(curs + y, bound, nsub, y, T, B, tiles, s1d, n1b, g1c);
    if (c == V54_BLOCKED) { if (s == 0u) return V46_NONE; continue; }
    if (c != V46_NONE) { from = y; return c; }
  }
  return V46_NONE;
}

// T is only needed by the contiguous-quarter home rule; mode 0 must not pay for a plan-word load.
#if V54_PART_MODE == 0
#define V54_TARG(pf, q) 0u
#else
#define V54_TARG(pf, q) ((pf)[V46_T + (q)])
#endif

// The narrow-scope predicate.  `pf` is plan_flags, `q` the pass, `expert_id`/`mtl`/`T` the tile's home inputs.
__device__ __forceinline__ bool v54_same_xcd(const unsigned* pf, unsigned q, unsigned expert_id,
                                             unsigned mtl, unsigned T, unsigned xcd) {
#if V54_CACHE_S1 && V54_S1_INV == 0
  if (v54_home(expert_id, mtl, T) != xcd) return false;
  if (__hip_atomic_load(pf + V54_XMASK, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 15u) return false;
  return __hip_atomic_load(pf + V54_G1X + q * 4u + xcd, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 0u;
#else
  (void)pf; (void)q; (void)expert_id; (void)mtl; (void)T; (void)xcd;
  return false;
#endif
}
#else
// All v54 levers off: the extra arguments become dead constants, so the TU is identical to the v51 base.
__device__ __forceinline__ unsigned v54_xcd() { return 0u; }
#define V54_TARG(pf, q) 0u
#endif  // V54_XCD_PART || V54_CACHE_S1 || V54_CLEAN_SC1
