#pragma once
// =============================================================================================================
// opus55 v46 SP  --  generalized 3-mode CU scheduler  +  single-plan 4-pass continuous overlap (s13)
//
// Grid = 80 CTAs x 8 waves.  Three intra-CU wave ratios coexist (compile time, N08 + N44 + N80 == 80):
//   CTA [0, N08)            0/8 : 8 comm waves                       cw = 8*cta + w
//   CTA [N08, N08+N44)      4/4 : w0..3 compute group g = cta-N08    w4..7 comm, cw = 8*N08 + 4*(cta-N08) + (w-4)
//   CTA [N08+N44, 80)       8/0 : two compute groups g = N44 + 2k + (w>=4)   (one group when !dual)
//   n_comp_groups = N44 + 2*N80 ;  W_comm = 8*N08 + 4*N44
// Comm-wave roles (cw space; every 4-aligned range lies in one CTA half, so 4-wave LDS barriers are legal):
//   cw 0..3   RDMA engine (ep16_gda_wave4, QP channel = cw)
//   cw 4..7   inter-node RX watcher, then combine (NR > Home > Scatter)
//   cw 8 .. 8+4*NRP-1      v46_plan_remote (NRP 4-wave groups), then combine
//   next NNR  dedicated NodeReduce + HomeCombine pool (no scatter)
//   rest NPULL front stage -> pull q0 (with compute) -> pull q3 -> pull q1 -> pull q2 -> combine (Scatter > NR > Home)
// Single plan, 4 passes (s13), ONE descriptor array, ONE row space, no grid barrier between passes:
//   q0 = local-node  sources, t <  TSL   rows [0,R0)              descs [0,T0)            (plan_local)
//   q3 = local-node  sources, t >= TSL   rows [R0,R0+R3)          descs [T0,T0+T3)        (plan_local)
//   -- 64-row / 256-byte gap: the remote section never shares a cache line with a local-section table entry --
//   q1 = remote-node sources, t <  TSR   rows [RL, RL+R1)         descs [DL, DL+T1)       (plan_remote)
//   q2 = remote-node sources, t >= TSR   rows [RL+R1, ..)         descs [DL+T1, ..)       (plan_remote)
//   TSL = align64(NT*P0PCT/100) (default 25%), TSR = align64(NT*P1PCT/100) (default 50%)
// Every pass owns its GEMM1 / GEMM2 cursors (bounded CAS), so no descriptor is ever parked or relocated.
// The Atrex loop: one decision per item by tid0 of the 4-wave group, broadcast through LDS (group-uniform).
// Streaming scatter: a token is claimed when bucket_done == nexp; gated ONLY on LOCAL_RDY (q0/q3) / P1_PLAN (q1/q2).
// =============================================================================================================
#ifndef V46_SP
#define V46_SP 0
#endif
#if V46_SP
#if !V44_COMBINE || !V45_TAIL || !V45_META || !V45_STAGE || (V45_PULL != 2)
#error "V46_SP requires V44_COMBINE=1 V45_TAIL=1 V45_META=1 V45_STAGE=1 V45_PULL=2"
#endif
#include "rb43/v43_util.hpp"   // vendored Racer B (race_ds1017_v43) helpers; v46_claim_lt is the per-thread form of v43_claim_bounded

// ---- official-knob driven role layout (see weave_21knobs.hpp) ---------------------------------------
// ---- V70 hardware-s_barrier 8-wave arm ---------------------------------------------------------------
// WEAVE_CU_WAVE_SPLIT=1 selects the pure 0-8 compute configuration explicitly; the official 0=auto
// value selects the SAME layout at compile time whenever fixed BM=128/128 stage tiles make the
// hardware contract hold (WEAVE_AUTO_PURE08).  When that layout is paired with those fixed tiles the
// kernel automatically routes every pure 8/0 compute CTA through ONE unified 8-wave group whose
// GEMM1/GEMM2 bodies synchronise with hardware s_barrier (hr1_gemm_8wave.hpp for BM=128,
// v70_gemm_bm64_8wave.hpp for BM=64).  V70_HR1_8W remains an
// internal assertion for build scripts that want an explicit compatibility check; it is not required
// to select the hardware path.
#ifndef V70_HR1_8W
#define V70_HR1_8W 0
#endif
#if V70_PATHB
#define V70_HW_SPLIT_OK 1
#define V70_HW_BM_OK 1
#define V70_8W_ACTIVE 1
#define V70_BM64_8W_ACTIVE 1
#else
#define V70_HW_SPLIT_OK (((WEAVE_CU_WAVE_SPLIT) == 1) || (((WEAVE_CU_WAVE_SPLIT) == 0) && WEAVE_AUTO_PURE08))
#define V70_HW_BM_OK (!(WEAVE_ADAPT_BM) && (((WEAVE_BLOCK_M_L1) == 128 && (WEAVE_BLOCK_M_L2) == 128) || \
                                          ((WEAVE_BLOCK_M_L1) == 64 && (WEAVE_BLOCK_M_L2) == 64)))
#define V70_BM64_8W_ACTIVE (V70_8W_ACTIVE && (WEAVE_BLOCK_M_L1) == 64 && (WEAVE_BLOCK_M_L2) == 64)
#if V70_HR1_8W
#if !V70_HW_SPLIT_OK
#error "V70_HR1_8W requires WEAVE_CU_WAVE_SPLIT=1 (pure 8/0 compute CTAs)"
#endif
#if !V70_HW_BM_OK
#error "V70_HR1_8W requires fixed WEAVE_BLOCK_M_L1/WEAVE_BLOCK_M_L2 in {64,128} with equal stage tiles and WEAVE_ADAPT_BM=0"
#endif
#endif
#define V70_8W_ACTIVE (V70_HW_SPLIT_OK && V70_HW_BM_OK)
#endif

#ifndef V46_N08
#define V46_N08 (WEAVE_LAYOUT_N08)
#endif
#ifndef V46_N44
#define V46_N44 (WEAVE_LAYOUT_N44)
#endif
#define V46_N80 (80u - (V46_N08) - (V46_N44))
#define V46_WCOMM (8u * (V46_N08) + 4u * (V46_N44))
#ifndef V46_NRP
#define V46_NRP (WEAVE_LAYOUT_NRP)
#endif
#define V46_CW_POOL (WEAVE_LAYOUT_CW_POOL)
#ifndef V46_NR_PCT
#define V46_NR_PCT 25u
#endif
#define V46_NNR (WEAVE_LAYOUT_NNR)
#define V46_NPULL (WEAVE_LAYOUT_NPULL)
#ifndef V46_P0PCT
#define V46_P0PCT (WEAVE_V_HEAD)   // official WEAVE_V_HEAD: q0 local-head split
#endif
#ifndef V46_P1PCT
#define V46_P1PCT 50u
#endif
#ifndef V62_DYN_BM
#define V62_DYN_BM (WEAVE_ADAPT_BM)   // official WEAVE_ADAPT_BM: no hidden fixed-BM override
#endif
#ifndef V62_BM48
#define V62_BM48 0
#endif
#ifndef V55_COALESCE_SCHED
#define V55_COALESCE_SCHED 0
#endif
#ifndef V55_N1G
#define V55_N1G 2u
#endif
#ifndef V55_N2G
#define V55_N2G 8u
#endif
#ifndef V55_SWIZZLE
#define V55_SWIZZLE 0
#endif
#ifndef V55_NOSPEC_G2
#define V55_NOSPEC_G2 0
#endif

// V70 Iron Rule 2 -- STRICT comm_only Scatter ownership.
// Compute/epilogue waves (W00..W03 of every pure-compute CTA) never issue a remote or local
// Scatter store.  They publish completed rows only through bucket_done and from there the
// Communication waves (W04..W07 on the 0/8 CTAs) own every Scatter copy.  The two dedup knobs
// remain live in the communication copy path (v45_bucket_data_fast): WEAVE_LOCAL_SCATTER_DEDUP
// selects the nexp==1 fast copy and WEAVE_DEDUP_MODE selects whether that fast copy is local-only.
// S1F direct completion is deliberately disabled here: its direct store and its completion
// publication are two halves of the same ownership transfer and must be disabled together.
#define V70_S1F_LOCAL_EN 0
#define V70_S1F_XNODE_EN 0
#define V70_S1F_EN 0
// Iron Rule 1/3/4 internal protocol words.  These live in reserved plan_flags space and do not
// change the public 21-knob surface or any runtime ABI.
#define V70_HEAD_RCNT   353u   // remote-head pull completions (all 8 compute waves)
#define V70_HEAD_READY  354u   // remote-head payload fully staged
#define V70_TAIL_START  355u   // diagnostic: first combine worker entry
#define V70_TAIL_END    356u   // diagnostic: last combine worker exit
#define V55_G1CX 416u  // [4][4] per-XCC GEMM1 claim cursors (free pf 416..447 when the v54 levers are off)
#ifndef V46_N2G
#define V46_N2G 4u
#endif
static_assert((V46_N08) + (V46_N44) <= 80u, "V46: N08 + N44 must be <= 80");
static_assert((V46_WCOMM) >= 16u, "V46: W_comm = 8*N08 + 4*N44 must be >= 16");
static_assert((V46_WCOMM) >= (V46_CW_POOL) + 2u, "V46: need >= 1 NR wave and >= 1 pull wave");
static_assert((V46_NPULL) >= 1u, "V46: NPULL >= 1");
static_assert((V46_P0PCT) <= 100u, "V46_P0PCT in 0..100");
static_assert((V46_P1PCT) <= 100u, "V46_P1PCT in 0..100");
#if V70_PATHB
// Path B (160x256, 2 CTAs/CU = 80 CUs): unified 3-CU-mode & spatial-knob layout.
// WEAVE_CU_WAVE_SPLIT: 0(auto)/2(4-4) -> 16 comm CTAs (CUs 0..15 run 4 comm + 4 compute);
//                      1(0-8)         ->  4 comm CTAs (76 pure 0-8 compute CUs + hybrid);
//                      3(8-0)         ->  8 comm CTAs on 4 dedicated 8-comm CUs (0..3 & 80..83), 76 pure 0-8 compute CUs.
#define V70_PB_NCOMM_BASE (((WEAVE_CU_WAVE_SPLIT) == 1) ? 4u : (((WEAVE_CU_WAVE_SPLIT) == 3) ? 8u : 16u))
#define V70_PB_NRP (((V70_PB_NCOMM_BASE) <= 4u) ? 1u : 2u)
#define V70_PB_NNR_RAW (((WEAVE_REDUCE_WARPS) == 0) ? 0u : ((((WEAVE_REDUCE_WARPS) + 3u) / 4u) > 2u ? 2u : (((WEAVE_REDUCE_WARPS) + 3u) / 4u)))
#define V70_PB_NNR_CTA (((V70_PB_NCOMM_BASE) >= 8u) ? (V70_PB_NNR_RAW) : 0u)
#define V70_PB_NPULL_DEF ((V70_PB_NCOMM_BASE) - 2u - (V70_PB_NRP) - (V70_PB_NNR_CTA))
#define V70_PB_NPULL_DW ((((WEAVE_DISPATCH_WARPS) > 0 && (WEAVE_DISPATCH_WARPS) <= 4) && (V70_PB_NPULL_DEF) >= 2u) ? ((V70_PB_NPULL_DEF) / 2u) : (V70_PB_NPULL_DEF))
#define V70_PB_NPULL_CTA (((WEAVE_DISPATCH_ACTIVE_SMS) > 0 && (WEAVE_DISPATCH_ACTIVE_SMS) <= 24) ? (unsigned)(WEAVE_DISPATCH_ACTIVE_SMS) : (((V70_PB_NPULL_DW) >= 1u) ? (V70_PB_NPULL_DW) : 1u))
#define V70_PB_NCOMM_CTA (2u + (V70_PB_NRP) + (V70_PB_NNR_CTA) + (V70_PB_NPULL_CTA))
#undef V46_WCOMM
#undef V46_NRP
#undef V46_NNR
#undef V46_NPULL
#define V46_WCOMM (4u * (V70_PB_NCOMM_CTA))
#define V46_NRP   (V70_PB_NRP)
#define V46_NNR   (4u * (V70_PB_NNR_CTA))
#define V46_NPULL (4u * (V70_PB_NPULL_CTA))
#undef V46_CW_POOL
#define V46_CW_POOL (8u + 4u * (V46_NRP))
#ifdef V70_PATHB_NCOMM_CTA
#undef V70_PATHB_NCOMM_CTA
#endif
#define V70_PATHB_NCOMM_CTA (V70_PB_NCOMM_CTA)
#ifdef V70_8W_ACTIVE
#undef V70_8W_ACTIVE
#endif
#define V70_8W_ACTIVE 1
#ifdef V70_BM64_8W_ACTIVE
#undef V70_BM64_8W_ACTIVE
#endif
#define V70_BM64_8W_ACTIVE 1
#endif
#define V70_REG_PULL_EFF (((WEAVE_REG_PULL_BYTES) == 0) ? 8192u : (unsigned)(WEAVE_REG_PULL_BYTES))
#define V70_READY_GRP_EFF (((WEAVE_READY_GROUP_SIZE) == 0) ? 16u : (unsigned)(WEAVE_READY_GROUP_SIZE))
#define V70_DEDUP_REMOTE_ROWS(n) (((WEAVE_DEDUP_MODE) == 0) ? (n) : 1u)
#define V70C_BM (((WEAVE_BLOCK_M_L2) == 32) ? 32u : 64u)
#define V70C_SH (((WEAVE_BLOCK_M_L2) == 32) ? 5u : 6u)


// ---- plan_flags map (u32[1024], zeroed every launch).  Live non-v46 words: 0, 200..215, 232, 250. ----
// s13: every 128-byte line holds ONE access class, so no plain-stored word shares a line with a hot atomic cursor:
//   384..415  hot claim / done cursors (atomics only)
//   448..479  control flags + arrival counters (atomics / st_agent)
//   480..511  plan-time words (plain stores by the pass builders, read only after the plan flag)
//   512..767  CNT[4][64]  per-pass expert counts, then row cursors      (one pass = two lines)
//   768..1023 OFF[4][64]  per-pass expert row offsets, LE + 1 <= 64
#define V46_NP 4u
#define V46_PFH 384u
#define V46_G1C       (V46_PFH + 0u)    // [4] GEMM1 claim cursor (items = T*n1b)
#define V46_G2C       (V46_PFH + 4u)    // [4] GEMM2 claim cursor (items = T*n2c)
#define V46_G1D       (V46_PFH + 8u)    // [4] GEMM1 items done
#define V46_G2D       (V46_PFH + 12u)   // [4] GEMM2 items done
#define V46_REM       (V46_PFH + 16u)   // [4] tokens with nexp>0 not yet scattered
#define V46_PF 448u
#define V46_META_L    (V46_PF + 0u)
#define V46_LH_CNT    (V46_PF + 1u)
#define V46_LBUILD    (V46_PF + 2u)
#define V46_LS_CNT    (V46_PF + 3u)
#define V46_LOCAL_RDY (V46_PF + 4u)
#define V46_P1_PLAN   (V46_PF + 5u)     // remote passes q1 + q2 planned and scattered
#define V46_P1_HCNT   (V46_PF + 6u)
#define V46_P1_BUILD  (V46_PF + 7u)
#define V46_P1_SCNT   (V46_PF + 8u)
#define V46_PULL_CNT  (V46_PF + 9u)     // [4]
#define V46_PULL_RDY  (V46_PF + 13u)    // [4]
#define V46_SDONE     (V46_PF + 17u)    // [4] pass scattered
#define V46_Q0_LBUILD (V46_PF + 21u)    // q0 descriptors + padding ready (head fast path)
#define V46_Q0_SCNT   (V46_PF + 22u)    // compute groups that finished q0 scatter
#define V46_Q0_RDY    (V46_PF + 23u)    // q0 ids/weights scattered: GEMM1 q0 may start
#define V46_Q0_NT     (V46_PW + 21u)    // q0 descriptor cursor after the head build
#define V46_Q0_ROW    (V46_PW + 22u)    // q0 row cursor after the head build
#define V46_PW 480u
#define V46_T         (V46_PW + 0u)     // [4] stage-2 descriptors of pass p (GEMM2 tile count)
#define V46_BASE      (V46_PW + 4u)     // [4] first stage-2 descriptor of pass p
#define V46_RBASE     (V46_PW + 8u)     // [4] first row of pass p
#define V46_NROW      (V46_PW + 12u)
#define V46_NDESC     (V46_PW + 13u)
#define V46_VROWS     (V46_PW + 14u)    // valid (unpadded) rows of the local passes (sort_stats[0] = local + remote)
// Independent stage-1 / stage-2 tile rows.  A stage-2 descriptor d is split into V46_NSUB implicit
// stage-1 sub-tiles d.row_start + s*L1 with min(L1, d.valid_rows - s*L1) valid rows.  The descriptor
// storage and the row space are unchanged, so the production RR/16 allocation is untouched.
#define V46_NSUB 1u
static_assert(V46_NSUB >= 1u && V46_NSUB <= 8u, "stage1 sub-tiles per stage2 tile must be 1..8");
#define V46_CNT(p)    (512u + 64u * (p))   // [4][64]
#define V46_OFF(p)    (768u + 64u * (p))   // [4][64]
#define V46_LE_MAX 63u
// error codes 950..979
#define V46_NONE 0xffffffffu

__device__ __forceinline__ unsigned v46_tsplit_pct(unsigned NT, unsigned pct) {
  const unsigned a = ((NT * pct) / 100u + 63u) & ~63u;
  return (a < NT) ? a : NT;
}
__device__ __forceinline__ unsigned v46_tsl(unsigned NT) { return v46_tsplit_pct(NT, V46_P0PCT); }
__device__ __forceinline__ unsigned v46_tsr(unsigned NT) { return v46_tsplit_pct(NT, V46_P1PCT); }
// q1 / q2 carry remote-node sources (planned by plan_remote), q0 / q3 local-node sources (planned by plan_local)
__device__ __forceinline__ bool v46_is_remote(unsigned p) { return p == 1u || p == 2u; }
__device__ __forceinline__ unsigned v46_n2g(unsigned n2b) {
  const unsigned g = (V46_N2G);
#if V55_COALESCE_SCHED
  if ((V55_N2G) >= 8u && n2b >= 32u && (n2b & 7u) == 0u) return 8u;
#endif
  if (g >= 4u && (n2b & 3u) == 0u) return 4u;
  if (g >= 2u && (n2b & 1u) == 0u) return 2u;
  return 1u;
}
#if V55_COALESCE_SCHED
__device__ __forceinline__ unsigned v55_xcc() {
  return __builtin_amdgcn_s_getreg((20 << 0) | (0 << 6) | (3 << 11)) & 0xfu;
}
__device__ __forceinline__ unsigned v55_claim_grp(unsigned* cur, unsigned bound, unsigned grp) {
  unsigned c = __hip_atomic_load(cur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  while (c + grp <= bound) {
    if (__hip_atomic_compare_exchange_strong(cur, &c, c + grp, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) return c;
  }
  while (c < bound) {
    if (__hip_atomic_compare_exchange_strong(cur, &c, c + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) return c;
  }
  return V46_NONE;
}
#endif
// token range of pass p: q0 [0,TSL)  q1 [0,TSR)  q2 [TSR,NT)  q3 [TSL,NT); both splits are 64-aligned, so every
// K41 payload chunk (16 / 32 / 64 tokens) and every 64-token scatter block lies in exactly one pass
__device__ __forceinline__ void v46_pass_range(unsigned p, unsigned NT, unsigned& tb, unsigned& te) {
  const unsigned TS = v46_is_remote(p) ? v46_tsr(NT) : v46_tsl(NT);
  if (p == 0u || p == 1u) { tb = 0u; te = TS; }
  else { tb = TS; te = NT; }
}
// LDS slot shift of this wave's 4-wave group (0 for waves 0..3, 32 for waves 4..7) as an SGPR value
// (opus55 v46 E9: every lane of a wave agrees, so the first lane's threadIdx decides; no VGPR select)
__device__ __forceinline__ unsigned v46_grp_sh() {
#if V70_PATHB
  return 0u;
#else
  return (__builtin_amdgcn_readfirstlane(threadIdx.x) >= 256u) ? 32u : 0u;
#endif
}
__device__ __forceinline__ unsigned v46_bcast(unsigned v, unsigned char* arena) {
#if V70_PATHB
  (void)arena;
  volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(
      reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 32760u));
  __syncthreads();
  if (threadIdx.x == 0u) sh[0] = v;
  __syncthreads();
  return __builtin_amdgcn_readfirstlane(sh[0]);
#else
  const unsigned grp_sh = v46_grp_sh();
  volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(
      reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65476u + grp_sh));
  weave_4wave_barrier_65440(arena);
  if ((threadIdx.x & 255u) == 0u) sh[0] = v;
  weave_4wave_barrier_65440(arena);
  return __builtin_amdgcn_readfirstlane(sh[0]);
#endif
}
// single-thread waits (agent scope); false on error / timeout (records `code` if no error yet)
__device__ __forceinline__ bool v46_wait_ge(const unsigned* p, unsigned target, const V968DispatchExt& env,
                                            const Ep16TurboContext& x, unsigned code) {
  const auto t0 = __builtin_amdgcn_s_memrealtime();
  unsigned sp = 0u;
  while (v42_ld_agent_rlx(p) < target) {
    if ((sp & 63u) == 0u) {
      if (ep16_load_relaxed((unsigned*)env.base.error)) return false;
      if (__builtin_amdgcn_s_memrealtime() - t0 > x.timeout_ticks) { v983_error((unsigned*)env.base.error, code); return false; }
    }
    if (sp < 64u) asm volatile("s_nop 0\n" ::: "memory");
    else __builtin_amdgcn_s_sleep(1);
    ++sp;
  }
  v42_acq_agent();
  return true;
}
__device__ __forceinline__ bool v46_wait_magic(const unsigned* p, const V968DispatchExt& env,
                                               const Ep16TurboContext& x, unsigned code) {
  const auto t0 = __builtin_amdgcn_s_memrealtime();
  unsigned sp = 0u;
  while (v42_ld_sys_rlx(p) != V809_PROGRESS_MAGIC) {
    if ((sp & 63u) == 0u) {
      if (ep16_load_relaxed((unsigned*)env.base.error)) return false;
      if (__builtin_amdgcn_s_memrealtime() - t0 > x.timeout_ticks) { v983_error((unsigned*)env.base.error, code); return false; }
    }
    if (sp < 128u) asm volatile("s_nop 0\n" ::: "memory");
    else __builtin_amdgcn_s_sleep(1);
    ++sp;
  }
  v42_acq_sys();
  return true;
}
// bounded claim: returns the claimed index, or V46_NONE once the cursor reached `bound` (never overshoots)
__device__ __forceinline__ unsigned v46_claim_lt(unsigned* cur, unsigned bound) {
  unsigned c = __hip_atomic_load(cur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  while (c < bound) {
    if (__hip_atomic_compare_exchange_strong(cur, &c, c + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT)) return c;
  }
  return V46_NONE;
}
// wave64 sum (all 64 lanes active) returned as an SGPR value.  opus55 v46 E10: BitMode ds_swizzle xor 1..16 inside
// each 32-lane half + readlane 0 / 32, i.e. no HIP __shfl lane-id arithmetic (whose `lane & ~63` term was CSE'd
// across the whole kernel and spilled around the Atrex loop)
__device__ __forceinline__ unsigned v46_wave_sum(unsigned v) {
  int s = static_cast<int>(v);
  s += __builtin_amdgcn_ds_swizzle(s, 0x041f);   // and 0x1f, or 0, xor 1
  s += __builtin_amdgcn_ds_swizzle(s, 0x081f);   // xor 2
  s += __builtin_amdgcn_ds_swizzle(s, 0x101f);   // xor 4
  s += __builtin_amdgcn_ds_swizzle(s, 0x201f);   // xor 8
  s += __builtin_amdgcn_ds_swizzle(s, 0x401f);   // xor 16
  return static_cast<unsigned>(__builtin_amdgcn_readlane(s, 0) + __builtin_amdgcn_readlane(s, 32));
}

// -------------------------------------------------------------------------------------------------------------
// descriptor builder for one pass (single thread).  BM rule and 16-row padding identical to v41_mb_route_plan.
// -------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ bool v46_build_pass(unsigned* pf, WeaveMTileDesc* descs, unsigned p, unsigned LE,
                                               unsigned& row, unsigned& nt, unsigned RR, unsigned& vsum) {
  const unsigned cap_desc = RR / 16u;
  if (LE > V46_LE_MAX) return false;   // OFF[p] holds LE + 1 <= 64 words
  pf[V46_BASE + p] = nt;
  pf[V46_RBASE + p] = row;
  bool ok = true;
  for (unsigned lid = 0u; lid < LE; ++lid) {
    const unsigned cnt = pf[V46_CNT(p) + lid];
    vsum += cnt;
    // V70: OFF[p] now carries the first descriptor of each expert.  This is
    // monotonic (experts are emitted in lid order) and lets the Atrex
    // scheduler recover per-expert tile counts without allocating new state.
    // The row end of the pass lives in OFF[p]+LE.
    pf[V46_OFF(p) + lid] = nt;
    unsigned rem = cnt;
    while (rem > 0u) {
#if WEAVE_ADAPT_BM
#if V70_R2X
      const unsigned bm = (rem > 48u) ? 64u : ((rem > 32u) ? 48u : ((rem > 16u) ? 32u : 16u));
#else
      const unsigned bm = (rem > 64u) ? 128u : ((rem > 32u) ? 64u : ((rem > 16u) ? 32u : 16u));
#endif
#else
      // Official WEAVE_BLOCK_M_L2: fixed stage-2 tile rows (the last tile of an expert is partial).
      const unsigned bm = (WEAVE_BLOCK_M_L2);
#endif
      const unsigned vr = (rem > bm) ? bm : rem;
      if (nt >= cap_desc) { ok = false; rem = 0u; break; }
      WeaveMTileDesc d{};
      d.expert_id = lid;
      d.row_start = row;
      d.bm = static_cast<uint16_t>(bm);
      d.valid_rows = static_cast<uint16_t>(vr);
      d.tile_idx = nt;
      descs[nt++] = d;
      row += ((vr + 15u) & ~15u);
      rem -= vr;
    }
  }
  pf[V46_OFF(p) + LE] = row;
  pf[V46_T + p] = nt - pf[V46_BASE + p];
  if (row > RR) ok = false;
  return ok;
}
// V70 padding from descriptor ranges.  The builder no longer stores row
// offsets: OFF[p]+lid is the first descriptor of expert lid and OFF[p]+LE is
// the row end.  Every descriptor owns [row_start+valid_rows, next_row_start),
// so the padded row holes are reconstructed exactly, including a final
// descriptor's 16-row tail.  Counts[p] is then rewritten to row cursors.
__device__ __forceinline__ void v46_pad_pass(const V968DispatchExt& env, const V983Ov1Ext& o, unsigned* pf,
                                             unsigned p, unsigned LE, unsigned tid, unsigned char* arena) {
  v42_acq_agent();
  const auto& e = env.base;
  const auto& ck = env.ck;
  auto* descs = reinterpret_cast<WeaveMTileDesc*>(ck.ck_a + 64u);
  auto* s1_ids = reinterpret_cast<int32_t*>(ck.ck_sorted_token_ids);
  auto* s2_ids = reinterpret_cast<int32_t*>(ck.ck_m64_sorted_ids);
  auto* r_rid = reinterpret_cast<unsigned*>(o.row_route_id);
  auto* r_wt = reinterpret_cast<float*>(o.stage2_weights);
  auto* row_dir = reinterpret_cast<unsigned*>(e.sorted_weights);
  const int32_t pad1 = static_cast<int32_t>(e.world * e.capacity);
  const int32_t pad2 = static_cast<int32_t>(ck.ck_route_rows);
  const unsigned base = pf[V46_BASE + p], T = pf[V46_T + p];
  const unsigned row_end = pf[V46_OFF(p) + LE];
  for (unsigned di = tid; di < T; di += 256u) {
    const WeaveMTileDesc d = descs[base + di];
    const unsigned beg = d.row_start + d.valid_rows;
    const unsigned end = (di + 1u < T) ? descs[base + di + 1u].row_start : row_end;
    for (unsigned r = beg; r < end; ++r) {
      s1_ids[r] = pad1;
      s2_ids[r] = pad2;
      r_rid[r] = 0xffffffffu;
      r_wt[r] = 0.0f;
#if V70_S1F_EN
      row_dir[r] = 0u;
#endif
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  for (unsigned lid = tid; lid < LE; lid += 256u) {
    const unsigned d0 = pf[V46_OFF(p) + lid];
    const unsigned d1 = pf[V46_OFF(p) + lid + 1u];
    pf[V46_CNT(p) + lid] = (d0 < d1 && d0 < base + T) ? descs[d0].row_start : row_end;
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
}
// one routed (src,t): count (hist) or scatter (ids) its local experts
__device__ __forceinline__ unsigned v46_tok_hist(const uint4& v0, const uint4& v1, unsigned topk, unsigned m,
                                                 unsigned LE, unsigned* cnt) {
  const uint4 vz = make_uint4(0u, 0u, 0u, 0u);
  unsigned nexp = 0u;
  #pragma unroll
  for (unsigned k = 0u; k < 12u; ++k) {
    if (k < topk) {
      const uint16_t em = v41_pick_u16(v0, v1, vz, k);
      if (em != 0xffffu && (em >> 12u) == m) {
        ++nexp;
        const unsigned lid = em & 0xfffu;
        if (lid < LE) __hip_atomic_fetch_add(cnt + lid, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
  return nexp;
}
__device__ __forceinline__ void v46_tok_scatter(const V968DispatchExt& env, const V983Ov1Ext& o,
                                                const uint4& v0, const uint4& v1, const uint4& v2,
                                                unsigned src, unsigned t, unsigned* cur) {
  const auto& e = env.base;
  const auto& ck = env.ck;
  const unsigned m = e.rank, LE = ck.ck_local_experts;
  const bool fp16 = (e.flags & 2u) != 0u;
  auto* s1_ids = reinterpret_cast<int32_t*>(ck.ck_sorted_token_ids);
  auto* s2_ids = reinterpret_cast<int32_t*>(ck.ck_m64_sorted_ids);
  auto* r_rid = reinterpret_cast<unsigned*>(o.row_route_id);
  auto* r_wt = reinterpret_cast<float*>(o.stage2_weights);
  auto* b_rows = reinterpret_cast<unsigned*>(o.bucket_route_rows);
  auto* row_dir = reinterpret_cast<unsigned*>(e.sorted_weights);
  (void)row_dir;
  const unsigned key = src * e.capacity + t;
  unsigned k_loc = 0u;
  unsigned last_row = 0xffffffffu;
  #pragma unroll
  for (unsigned k = 0u; k < 12u; ++k) {
    if (k < e.topk) {
      const uint16_t em = v41_pick_u16(v0, v1, v2, k);
      if (em != 0xffffu && (em >> 12u) == m) {
        const unsigned lid = em & 0xfffu;
        if (lid < LE) {
          const uint16_t wb = v41_pick_u16(v0, v1, v2, 12u + k);
          const unsigned row = __hip_atomic_fetch_add(cur + lid, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          s1_ids[row] = static_cast<int32_t>(key);
          s2_ids[row] = static_cast<int32_t>(row);
          r_rid[row] = (src * (e.capacity + 1u) + t) | (k_loc << 24u);
          r_wt[row] = weave_weight_value(wb, fp16);
          b_rows[static_cast<uint64_t>(key) * e.topk + k_loc] = row;
#if V70_S1F_EN
          row_dir[row] = 0u;
#endif
          last_row = row;
          ++k_loc;
        }
      }
    }
  }
#if V70_S1F_EN
  if (k_loc == 1u) {
    const bool intra = (src >> 3u) == (m >> 3u);
    if ((intra && V70_S1F_LOCAL_EN) || (!intra && V70_S1F_XNODE_EN))
      row_dir[last_row] = 0x80000000u | ((src & 0x0fu) << 16u) | (t & 0xffffu);
  }
#else
  (void)last_row;
#endif
}

// -------------------------------------------------------------------------------------------------------------
// v46_plan_local: grid-level over the n_comp_groups compute groups.  Needs ONLY the 8 local-node metas (xGMI).
//   (A) grp0: 8 local PROGRESS_MAGIC -> src_rows -> META_L
//   (B) all : scan + per-pass histogram + local_nexp + REM[0/3]            -> LH_CNT
//   (C) grp0: q0 + q3 descriptors, plan table, padding, cursors, 64-row / 256-B aligned remote start -> LBUILD
//   (D) all : stage1/stage2 ids, weights, row_route_id, bucket_route_rows  -> LS_CNT -> LOCAL_RDY (+ empty SDONE)
// -------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ bool v46_plan_local(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, unsigned grp, unsigned ncg FI_CTX_PARAM) {
  const auto& e = env.base;
  const auto& ck = env.ck;
  unsigned tid = threadIdx.x & 255u;
  asm volatile("" : "+v"(tid)); // opus55 E3
  const unsigned NT = x.node_tokens, m = e.rank, ln = m >> 3u;
  const unsigned LE = ck.ck_local_experts;
  const unsigned TS = v46_tsl(NT);
  const unsigned RR = ck.ck_route_rows;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* src_rows = reinterpret_cast<unsigned*>(e.source_bucket_rows);
  const auto* peer_prog_tbl = reinterpret_cast<const uint64_t*>(e.peer_progress_table);
  const auto* peer_meta_tbl = reinterpret_cast<const uint64_t*>(e.peer_meta_table);
  auto* local_nexp = reinterpret_cast<unsigned*>(o.home_slots);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  FI_BG(V1988_FI_CORE, 17u, grp);
  // (A)
  if (grp == 0u) {
    if (tid < 8u) {
      const unsigned src = ln * 8u + tid;
      if (v46_wait_magic(reinterpret_cast<const unsigned*>(peer_prog_tbl[src]), env, x, 950u)) {
        const uint64_t tl = v41_ld_sc1_u64(reinterpret_cast<const void*>(peer_meta_tbl[src]));
        src_rows[src] = static_cast<unsigned>(tl);
      }
    }
    __builtin_amdgcn_s_waitcnt(0);
    weave_4wave_barrier_65440(arena);
    if (tid == 0u) {
      __threadfence();
      if (trace) trace[1382u] = __builtin_amdgcn_s_memrealtime();   // v46: 8 local metas arrived
      v41_st_agent_u32(pf + V46_META_L, ep16_load_relaxed((unsigned*)e.error) ? 2u : 1u);
    }
  }
  unsigned ok = 1u;
  if (tid == 0u) ok = (v46_wait_ge(pf + V46_META_L, 1u, env, x, 951u) && v41_ld_agent_u32(pf + V46_META_L) == 1u) ? 1u : 0u;
  if (v46_bcast(ok, arena) == 0u) { FI_EG(V1988_FI_CORE, 17u, grp); return false; }

  // (B0) HEAD HISTOGRAM: q0 only.  The q3 histogram used to share this first
  // pass even though its descriptors cannot be built until the leader's tail.
  // Loading 4x fewer token records here removes the pre-GEMM1 q3 work from the
  // critical head path; the leader performs the q3 histogram concurrently with
  // the other compute groups' q0 GEMM1 below.
  const unsigned gtid = grp * 256u + tid, gstr = ncg * 256u;
  const unsigned ntok = 8u * NT;
  const unsigned nq0 = 8u * TS;
  unsigned r0 = 0u;
  if (nq0 != 0u) {
    for (unsigned j = gtid; j < nq0; j += gstr) {
      const unsigned so = j / TS, t = j - so * TS, src = ln * 8u + so;
      unsigned nexp = 0u;
      if (t < src_rows[src]) {
        const auto* tm = reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[src] + 64ull) + t;
        uint4 v0, v1;
        v41_ld2_sc1_u4(reinterpret_cast<const unsigned char*>(tm) + 0u,
                       reinterpret_cast<const unsigned char*>(tm) + 16u, v0, v1);
        nexp = v46_tok_hist(v0, v1, e.topk, m, LE, pf + V46_CNT(0u));
      }
      local_nexp[static_cast<uint64_t>(src) * NT + t] = nexp;
      if (nexp) ++r0;
    }
  }
  r0 = v46_wave_sum(r0);
  if ((threadIdx.x & 63u) == 0u) {
    if (r0) __hip_atomic_fetch_add(pf + V46_REM + 0u, r0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (tid == 0u) {
    __threadfence();
    __hip_atomic_fetch_add(pf + V46_LH_CNT, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  }

  // (C0) HEAD FAST PATH: grp0 builds ONLY q0, pads it and publishes Q0_LBUILD.  The q3 build,
  // q3 padding and the full local plan follow after every compute group has scattered q0.
  if (grp == 0u) {
    unsigned bok = 0u;
    if (tid == 0u) {
      if (v46_wait_ge(pf + V46_LH_CNT, ncg, env, x, 952u)) {
        auto* descs = reinterpret_cast<WeaveMTileDesc*>(ck.ck_a + 64u);
        unsigned q0_row = 0u, q0_nt = 0u, vsum0 = 0u;
        bool b = v46_build_pass(pf, descs, 0u, LE, q0_row, q0_nt, RR, vsum0);
        if (q0_row > RR) b = false;
        pf[V46_Q0_NT] = q0_nt;
        pf[V46_Q0_ROW] = q0_row;
        pf[V46_VROWS] = vsum0;
        if (!b) v983_error((unsigned*)e.error, 953u);
        bok = b ? 1u : 0u;
      }
    }
    bok = v46_bcast(bok, arena);
    if (bok) v46_pad_pass(env, o, pf, 0u, LE, tid, arena);
    if (tid == 0u) {
      __threadfence();
      if (trace && bok) trace[1387u] = __builtin_amdgcn_s_memrealtime();
      v41_st_agent_u32(pf + V46_Q0_LBUILD, bok ? 1u : 2u);
    }
  }
  ok = 1u;
  if (tid == 0u) ok = (v46_wait_ge(pf + V46_Q0_LBUILD, 1u, env, x, 955u) && v41_ld_agent_u32(pf + V46_Q0_LBUILD) == 1u) ? 1u : 0u;
  if (v46_bcast(ok, arena) == 0u) { FI_EG(V1988_FI_CORE, 17u, grp); return false; }

  // (D0) scatter q0 ids/weights only.  When the last compute group finishes, q0 GEMM1 becomes runnable.
  for (unsigned idx = gtid; idx < ntok; idx += gstr) {
    const unsigned so = idx / NT, t = idx - so * NT, src = ln * 8u + so;
    if (t >= TS || t >= src_rows[src]) continue;
    if (local_nexp[static_cast<uint64_t>(src) * NT + t] == 0u) continue;
    const auto* tm = reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[src] + 64ull) + t;
    uint4 v0, v1, v2;
    v41_ld3_sc1_u4(reinterpret_cast<const unsigned char*>(tm) + 0u,
                   reinterpret_cast<const unsigned char*>(tm) + 16u,
                   reinterpret_cast<const unsigned char*>(tm) + 32u, v0, v1, v2);
    v46_tok_scatter(env, o, v0, v1, v2, src, t, pf + V46_CNT(0u));
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (tid == 0u) {
    __threadfence();
    const unsigned d0 = __hip_atomic_fetch_add(pf + V46_Q0_SCNT, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (d0 == ncg) {
      if (v41_ld_agent_u32(pf + V46_REM + 0u) == 0u) v41_st_agent_u32(pf + V46_SDONE + 0u, 1u);
      v41_st_agent_u32(pf + V46_Q0_RDY, 1u);
      if (trace) { trace[1386u] = __builtin_amdgcn_s_memrealtime(); trace[17u] = trace[1386u]; }
    }
  }

  // All compute groups leave for GEMM1(q0) here.  The leader group runs v46_plan_local_tail after its
  // own q0 pull, so the q3 build/scatter overlaps head GEMM1 on the other compute groups.
  FI_EG(V1988_FI_CORE, 17u, grp);
  return true;
}

// -------------------------------------------------------------------------------------------------------------
// v46_plan_local_tail: leader compute group only, after its q0 pull.  Builds/pads/scatters q3 and publishes
// LBUILD (remote planning may start) and finally LOCAL_RDY (q3 GEMM1 and every local GEMM2 may start).
// -------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ bool v46_plan_local_tail(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena FI_CTX_PARAM) {
  const auto& e = env.base;
  const auto& ck = env.ck;
  const unsigned tid = threadIdx.x & 255u;
  const unsigned NT = x.node_tokens, m = e.rank, ln = m >> 3u;
  const unsigned LE = ck.ck_local_experts;
  const unsigned TS = v46_tsl(NT);
  const unsigned RR = ck.ck_route_rows;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* src_rows = reinterpret_cast<unsigned*>(e.source_bucket_rows);
  const auto* peer_meta_tbl = reinterpret_cast<const uint64_t*>(e.peer_meta_table);
  auto* local_nexp = reinterpret_cast<unsigned*>(o.home_slots);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  FI_BG(V1988_FI_CORE, 18u, 0u);

  // (B3) q3 histogram, leader 4-wave group only.  This runs after the leader has
  // pulled q0 and while the other compute groups execute q0 GEMM1.
  const unsigned nspan3 = NT - TS;
  if (nspan3 != 0u) {
    const unsigned nq3 = 8u * nspan3;
    unsigned r3 = 0u;
    for (unsigned j = tid; j < nq3; j += 256u) {
      const unsigned so = j / nspan3, t = TS + (j - so * nspan3), src = ln * 8u + so;
      unsigned nexp = 0u;
      if (t < src_rows[src]) {
        const auto* tm = reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[src] + 64ull) + t;
        uint4 v0, v1;
        v41_ld2_sc1_u4(reinterpret_cast<const unsigned char*>(tm) + 0u,
                       reinterpret_cast<const unsigned char*>(tm) + 16u, v0, v1);
        nexp = v46_tok_hist(v0, v1, e.topk, m, LE, pf + V46_CNT(3u));
      }
      local_nexp[static_cast<uint64_t>(src) * NT + t] = nexp;
      if (nexp) ++r3;
    }
    r3 = v46_wave_sum(r3);
    if ((threadIdx.x & 63u) == 0u && r3)
      __hip_atomic_fetch_add(pf + V46_REM + 3u, r3, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __builtin_amdgcn_s_waitcnt(0);
    weave_4wave_barrier_65440(arena);
    if (tid == 0u && trace) trace[1385u] = __builtin_amdgcn_s_memrealtime();
  }

  // (C3) q3 descriptors continue q0's row/descriptor cursors.
  unsigned bok = 0u;
  if (tid == 0u) {
    auto* plan = reinterpret_cast<WeaveTilePlanTable*>(ck.ck_a);
    auto* descs = reinterpret_cast<WeaveMTileDesc*>(ck.ck_a + 64u);
    unsigned row = pf[V46_Q0_ROW], nt = pf[V46_Q0_NT], vsum = pf[V46_VROWS];
    bool b = v46_build_pass(pf, descs, 3u, LE, row, nt, RR, vsum);
    // s13 R4: the remote section starts on a fresh 64-row and 256-B descriptor boundary.
    row = (row + 63u) & ~63u;
    {
      const uint64_t a0 = reinterpret_cast<uint64_t>(descs + nt);
      nt += static_cast<unsigned>((((a0 + 255ull) & ~255ull) - a0) >> 4u);
    }
    if (row > RR) b = false;
    pf[V46_NROW] = row;
    pf[V46_NDESC] = nt;
    pf[V46_VROWS] = vsum;
    plan->total_valid_rows = 0u;
    plan->total_padded_rows = RR;
    plan->total_m_tiles = nt;
    plan->pad_gather_slot = e.world * e.capacity;
    unsigned zero_v = 0u;
    asm volatile("" : "+v"(zero_v)); // opus55 E4
    for (unsigned bk = 0u; bk < 4u; ++bk) { plan->num_tiles[bk] = zero_v; plan->bucket_offset[bk] = zero_v; }
    if (!b) v983_error((unsigned*)e.error, 953u);
    bok = b ? 1u : 0u;
  }
  bok = v46_bcast(bok, arena);
  if (bok) v46_pad_pass(env, o, pf, 3u, LE, tid, arena);
  if (tid == 0u) {
    __threadfence();
    v41_st_agent_u32(pf + V46_LBUILD, bok ? 1u : 2u);
  }
  unsigned ok = 1u;
  if (tid == 0u) ok = (v46_wait_ge(pf + V46_LBUILD, 1u, env, x, 954u) && v41_ld_agent_u32(pf + V46_LBUILD) == 1u) ? 1u : 0u;
  if (v46_bcast(ok, arena) == 0u) { FI_EG(V1988_FI_CORE, 18u, 0u); return false; }

  // (D3) scatter the local tail (q3), then publish the full local plan.
  const unsigned ntok = 8u * NT;
  for (unsigned idx = tid; idx < ntok; idx += 256u) {
    const unsigned so = idx / NT, t = idx - so * NT, src = ln * 8u + so;
    if (t < TS || t >= src_rows[src]) continue;
    if (local_nexp[static_cast<uint64_t>(src) * NT + t] == 0u) continue;
    const auto* tm = reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[src] + 64ull) + t;
    uint4 v0, v1, v2;
    v41_ld3_sc1_u4(reinterpret_cast<const unsigned char*>(tm) + 0u,
                   reinterpret_cast<const unsigned char*>(tm) + 16u,
                   reinterpret_cast<const unsigned char*>(tm) + 32u, v0, v1, v2);
    v46_tok_scatter(env, o, v0, v1, v2, src, t, pf + V46_CNT(3u));
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (tid == 0u) {
    __threadfence();
    const unsigned d = __hip_atomic_fetch_add(pf + V46_LS_CNT, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (d == 1u) {
      if (v41_ld_agent_u32(pf + V46_REM + 3u) == 0u) v41_st_agent_u32(pf + V46_SDONE + 3u, 1u);
      v41_st_agent_u32(pf + V46_LOCAL_RDY, 1u);
      if (trace) { trace[1380u] = __builtin_amdgcn_s_memrealtime(); trace[1302u] = trace[1380u]; }
    }
  }
  FI_EG(V1988_FI_CORE, 18u, 0u);
  return true;
}


// -------------------------------------------------------------------------------------------------------------
// v46_plan_remote: NRP comm 4-wave groups (rg = 0 leads).  Waits ONLY for LBUILD and the 8 remote metas.
// Builds q1 (t < TSR) and q2 (t >= TSR) at the aligned descriptor NDESC / row NROW, scatters their ids, publishes
// P1_PLAN.  Loads are batched 4 tokens per thread (one s_waitcnt per batch of 4 sc1 loads).
// sort_stats (s13 instrumentation) = {valid rows q0..q3, padded row span incl. gaps, descriptor span, LE}.
// -------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void v46_plan_remote(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, unsigned rg, unsigned nrg FI_CTX_PARAM) {
  const auto& e = env.base;
  const auto& ck = env.ck;
  unsigned tid = threadIdx.x & 255u;
  asm volatile("" : "+v"(tid)); // opus55 E3
  const unsigned NT = x.node_tokens, m = e.rank, rn = (m >> 3u) ^ 1u;
  const unsigned LE = ck.ck_local_experts;
  const unsigned RR = ck.ck_route_rows;
  const unsigned TSR = v46_tsr(NT);
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* src_rows = reinterpret_cast<unsigned*>(e.source_bucket_rows);
  const auto* peer_prog_tbl = reinterpret_cast<const uint64_t*>(e.peer_progress_table);
  const auto* peer_meta_tbl = reinterpret_cast<const uint64_t*>(e.peer_meta_table);
  auto* local_nexp = reinterpret_cast<unsigned*>(o.home_slots);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  unsigned ok = 1u;
  if (tid == 0u) ok = (v46_wait_ge(pf + V46_LBUILD, 1u, env, x, 960u) && v41_ld_agent_u32(pf + V46_LBUILD) == 1u) ? 1u : 0u;
  if (tid < 8u) {
    const unsigned src = rn * 8u + tid;
    if (v46_wait_magic(reinterpret_cast<const unsigned*>(peer_prog_tbl[src]), env, x, 961u)) {
      const uint64_t tl = v41_ld_sc1_u64(reinterpret_cast<const void*>(peer_meta_tbl[src]));
      src_rows[src] = static_cast<unsigned>(tl);   // every rg writes the same value
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  ok = v46_bcast(ok && !ep16_load_relaxed((unsigned*)e.error) ? 1u : 0u, arena);
  v42_acq_agent();   // every wave: drop any L1 line of src_rows / local_nexp cached by a co-resident group
  if (!ok) return;
  FI_BG(V1988_FI_CORE, 18u, rg);
  if (rg == 0u && tid == 0u && trace) { trace[1301u] = __builtin_amdgcn_s_memrealtime(); }   // all 16 metas
  const unsigned G = nrg * 256u, g = rg * 256u + tid;
  const unsigned ntok = 8u * NT;
  // ---- histogram (4 tokens / thread / batch) ----
  unsigned r1 = 0u, r2 = 0u;
  for (unsigned b0 = g; b0 < ntok; b0 += 4u * G) {
    unsigned so[4], tt[4];
    bool live[4];
    const unsigned char* tp[4];
    #pragma unroll
    for (unsigned j = 0u; j < 4u; ++j) {
      const unsigned idx = b0 + j * G;
      const unsigned s = (idx < ntok) ? (idx / NT) : 0u;
      const unsigned t = (idx < ntok) ? (idx - s * NT) : 0u;
      so[j] = s; tt[j] = t;
      live[j] = (idx < ntok) && (t < src_rows[rn * 8u + s]);
      tp[j] = reinterpret_cast<const unsigned char*>(reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[rn * 8u + s] + 64ull) + (live[j] ? t : 0u));
    }
    uint4 a0, a1, a2, a3, c0, c1, c2, c3;
    v41_ld4_sc1_u4(tp[0], tp[1], tp[2], tp[3], a0, a1, a2, a3);
    v41_ld4_sc1_u4(tp[0] + 16u, tp[1] + 16u, tp[2] + 16u, tp[3] + 16u, c0, c1, c2, c3);
    #pragma unroll
    for (unsigned j = 0u; j < 4u; ++j) {
      const unsigned idx = b0 + j * G;
      if (idx >= ntok) continue;
      const uint4 va = (j == 0u) ? a0 : ((j == 1u) ? a1 : ((j == 2u) ? a2 : a3));
      const uint4 vc = (j == 0u) ? c0 : ((j == 1u) ? c1 : ((j == 2u) ? c2 : c3));
      const unsigned pj = (tt[j] < TSR) ? 1u : 2u;
      const unsigned nexp = live[j] ? v46_tok_hist(va, vc, e.topk, m, LE, pf + V46_CNT(pj)) : 0u;
      local_nexp[static_cast<uint64_t>(rn * 8u + so[j]) * NT + tt[j]] = nexp;
      if (nexp) { if (pj == 1u) ++r1; else ++r2; }
    }
  }
  r1 = v46_wave_sum(r1);
  r2 = v46_wave_sum(r2);
  if ((threadIdx.x & 63u) == 0u) {
    if (r1) __hip_atomic_fetch_add(pf + V46_REM + 1u, r1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (r2) __hip_atomic_fetch_add(pf + V46_REM + 2u, r2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (tid == 0u) {
    __threadfence();
    __hip_atomic_fetch_add(pf + V46_P1_HCNT, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  }
  // ---- leader builds q1 + q2 (the plan header line is NOT touched: it shares a line with q0's descriptors) ----
  if (rg == 0u) {
    unsigned bok = 0u;
    if (tid == 0u) {
      if (v46_wait_ge(pf + V46_P1_HCNT, nrg, env, x, 962u)) {
        auto* descs = reinterpret_cast<WeaveMTileDesc*>(ck.ck_a + 64u);
        unsigned row = pf[V46_NROW], nt = pf[V46_NDESC], vsum = 0u;
        bool b = v46_build_pass(pf, descs, 1u, LE, row, nt, RR, vsum);
        b = v46_build_pass(pf, descs, 2u, LE, row, nt, RR, vsum) && b;
        pf[V46_NROW] = row;
        pf[V46_NDESC] = nt;
        if (e.sort_stats) {
          auto* ss = reinterpret_cast<unsigned*>(e.sort_stats);
          ss[0] = pf[V46_VROWS] + vsum;
          ss[1] = row;
          ss[2] = nt;
          ss[3] = LE;
        }
        if (!b) v983_error((unsigned*)e.error, 963u);
        bok = b ? 1u : 0u;
      }
    }
    bok = v46_bcast(bok, arena);
    if (bok) {
      v46_pad_pass(env, o, pf, 1u, LE, tid, arena);
      v46_pad_pass(env, o, pf, 2u, LE, tid, arena);
    }
    if (tid == 0u) {
      __threadfence();
      v41_st_agent_u32(pf + V46_P1_BUILD, bok ? 1u : 2u);
    }
  }
  ok = 1u;
  if (tid == 0u) ok = (v46_wait_ge(pf + V46_P1_BUILD, 1u, env, x, 964u) && v41_ld_agent_u32(pf + V46_P1_BUILD) == 1u) ? 1u : 0u;
  if (v46_bcast(ok, arena) == 0u) { FI_EG(V1988_FI_CORE, 18u, rg); return; }
  // ---- scatter ids (4 tokens / thread / batch) ----
  for (unsigned b0 = g; b0 < ntok; b0 += 4u * G) {
    unsigned so[4], tt[4];
    bool live[4];
    const unsigned char* tp[4];
    #pragma unroll
    for (unsigned j = 0u; j < 4u; ++j) {
      const unsigned idx = b0 + j * G;
      const unsigned s = (idx < ntok) ? (idx / NT) : 0u;
      const unsigned t = (idx < ntok) ? (idx - s * NT) : 0u;
      so[j] = s; tt[j] = t;
      live[j] = (idx < ntok) && (t < src_rows[rn * 8u + s]) &&
                (local_nexp[static_cast<uint64_t>(rn * 8u + s) * NT + t] != 0u);
      tp[j] = reinterpret_cast<const unsigned char*>(reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[rn * 8u + s] + 64ull) + (live[j] ? t : 0u));
    }
    const uint64_t anyl = __ballot(live[0] || live[1] || live[2] || live[3]);
    if (anyl == 0ull) continue;
    uint4 a0, a1, a2, a3, c0, c1, c2, c3, w0, w1, w2, w3;
    v41_ld4_sc1_u4(tp[0], tp[1], tp[2], tp[3], a0, a1, a2, a3);
    v41_ld4_sc1_u4(tp[0] + 16u, tp[1] + 16u, tp[2] + 16u, tp[3] + 16u, c0, c1, c2, c3);
    v41_ld4_sc1_u4(tp[0] + 32u, tp[1] + 32u, tp[2] + 32u, tp[3] + 32u, w0, w1, w2, w3);
    #pragma unroll
    for (unsigned j = 0u; j < 4u; ++j) {
      if (!live[j]) continue;
      const uint4 va = (j == 0u) ? a0 : ((j == 1u) ? a1 : ((j == 2u) ? a2 : a3));
      const uint4 vc = (j == 0u) ? c0 : ((j == 1u) ? c1 : ((j == 2u) ? c2 : c3));
      const uint4 vw = (j == 0u) ? w0 : ((j == 1u) ? w1 : ((j == 2u) ? w2 : w3));
      v46_tok_scatter(env, o, va, vc, vw, rn * 8u + so[j], tt[j], pf + V46_CNT((tt[j] < TSR) ? 1u : 2u));
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (tid == 0u) {
    __threadfence();
    const unsigned d = __hip_atomic_fetch_add(pf + V46_P1_SCNT, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (d == nrg) {
      if (v41_ld_agent_u32(pf + V46_REM + 1u) == 0u) v41_st_agent_u32(pf + V46_SDONE + 1u, 1u);
      if (v41_ld_agent_u32(pf + V46_REM + 2u) == 0u) v41_st_agent_u32(pf + V46_SDONE + 2u, 1u);
      v41_st_agent_u32(pf + V46_P1_PLAN, 1u);
      if (trace) trace[1381u] = __builtin_amdgcn_s_memrealtime();
    }
  }
  FI_EG(V1988_FI_CORE, 18u, rg);
}

#include "v54_xcd.hpp"   // v58: Fleet 4-XCD home queues + narrow-scope acquire (macro-gated)
#include "v46_run.hpp"   // opus55 v46: pull / GEMM items / Atrex / 4-pass scatter / combine / role dispatch
#endif  // V46_SP
