#pragma once
// V62 Opus 5.5 High knobs (all default 0 = V61 code)
#ifndef V62_G2_SWZ
#define V62_G2_SWZ 0
#endif
// V70 Iron Rule 4: two-level cache schedule inside the megakernel.  This is an
// architectural internal constant (not a public WEAVE_* knob).
//   GEMM1: expert-local GROUP_SIZE_M=8 2D snake over (M tile, N group)
//   GEMM2: expert -> M tile -> all N groups (N-inner)
// Both flatten into four queues, each owning 20 contiguous slots of every
// 80-slot schedule block.  A CTA drains its physical XCD queue first and then
// steals from the other XCD queues, so one short XCD tail cannot idle the GPU.
#ifndef V70_TWO_LEVEL
#define V70_TWO_LEVEL 1
#endif
#define V70_XCD_N 4u
#ifndef V70_XCD_CHUNK
#define V70_XCD_CHUNK 20u
#endif
#ifndef V70_SWZ_GROUP_M
#define V70_SWZ_GROUP_M 8u
#endif
#define V62_G2CX 432u   // [4 passes][4 xcc] per-XCC GEMM2 claim cursors (pf 432..447, free unless V54_XCD_PART)
#define V70M_G2D 381u   // compact: completed M tiles (one publication per M tile)
#define V70M_G2T 382u   // compact: target completed M tiles (host-staged plan size)
#if V62_G2_SWZ && (V54_XCD_PART || !V55_COALESCE_SCHED || !V55_NOSPEC_G2)
#error "V62_G2_SWZ needs the C68 scheduler (V55_COALESCE_SCHED, V55_NOSPEC_G2, no V54_XCD_PART)"
#endif
#ifndef V62_NT0
#define V62_NT0 0
#endif
#if V62_NT0
// c in [0, T*cnt) -> tile mtl (return) and column group j, blocked M-inner order with block size V62_NT0
__device__ __forceinline__ unsigned v62_nt0_map(unsigned c, unsigned cnt, unsigned T, unsigned& j) {
  const unsigned per = (V62_NT0) * cnt;
  const unsigned blk = c / per;
  const unsigned r = c - blk * per;
  const unsigned rem = T - blk * (V62_NT0);
  const unsigned gb = (rem < (V62_NT0)) ? rem : (V62_NT0);
  j = r / gb;
  return blk * (V62_NT0) + (r - j * gb);
}
#endif
#ifndef V62_G1_STEAL
#define V62_G1_STEAL 0
#endif
#ifndef V62_G1_BODY
#define V62_G1_BODY 0
#endif
#ifndef V62_G1_BM32
#define V62_G1_BM32 0
#endif
#ifndef V62_DYN_BM
#define V62_DYN_BM (WEAVE_ADAPT_BM)
#endif
// Stage-2 runtime body selection is needed whenever the fixed BM2 is not 64 or adapt is on.
#define V70_G2_DYN_SELECT ((WEAVE_ADAPT_BM) || ((WEAVE_BLOCK_M_L2) != 64))
#ifndef V62_G1_SMALL
#define V62_G1_SMALL 1
#endif

// ---- official WEAVE_PRECISION_MODE / WEAVE_REG_BALANCE: body schedule selection ----------------------
// 0 keeps the V62 delivered schedule.  1 selects the lean kD3 / single-B-set stage-1 schedule.  2 forces
// the generic BK64 body for both stages (lowest register pressure, no BN512/BK128 fast body).
#if (WEAVE_PRECISION_MODE) == 2
  #undef V62_G2_BN256
  #define V62_G2_BN256 0
  #undef V62_G2_BN512
  #define V62_G2_BN512 0
  #undef V62_G2_BN384
  #define V62_G2_BN384 0
  #undef V62_G1_BODY
  #define V62_G1_BODY 0
  #undef V61_G1_1X4
  #define V61_G1_1X4 0
#elif (WEAVE_REG_BALANCE) == 2
  #undef V62_G2_BN256
  #define V62_G2_BN256 2
  #undef V62_G1_BODY
  #define V62_G1_BODY 1
#elif (WEAVE_REG_BALANCE) == 1 || (WEAVE_PRECISION_MODE) == 1
  #undef V62_G2_BN256
  #define V62_G2_BN256 3
  #undef V62_G1_BODY
  #define V62_G1_BODY 1
#endif
#ifndef V62_G2_BN256
#define V62_G2_BN256 0
#endif
#ifndef V62_BN256_BM32
#define V62_BN256_BM32 0
#endif
#ifndef V62_G2_BN512
#define V62_G2_BN512 0
#endif
#ifndef V62_BN512_BM32
#define V62_BN512_BM32 0
#endif
#ifndef V62_BN512_BM64
#define V62_BN512_BM64 0
#endif
#ifndef V62_G2_BN384
#define V62_G2_BN384 0
#endif
#ifndef V62_BN384_BM32
#define V62_BN384_BM32 0
#endif
#ifndef V62_BN384_BM64
#define V62_BN384_BM64 0
#endif
#ifndef V62_G1_BM48
#define V62_G1_BM48 0
#endif
#ifndef V62_BN512_BM16
#define V62_BN512_BM16 0
#endif
#ifndef V62_BN512_BM48
#define V62_BN512_BM48 0
#endif
#ifndef V62_BM48
#define V62_BM48 0
#endif
#if V62_DYN_BM && !V62_G2_BN256
#error "V62_DYN_BM needs V62_G2_BN256 (BM=16 GEMM2 tiles only exist as the bn256 body)"
#endif
// Forward declaration: GEMM2 readiness tests every implicit stage-1 sub-tile.
__device__ __forceinline__ bool v46_s1_ready(const unsigned* s1d, unsigned mt, unsigned n1b);

// V61 Opus 5.5 High: GEMM body selection for the BM=64 descriptors (both default 0 = V58 2x2 bodies)
#ifndef V61_G1_1X4
#define V61_G1_1X4 0
#endif
#ifndef V61_G2_1X4
#define V61_G2_1X4 0
#endif
// opus55 v46 SP part 2 (included at the end of v46_sp.hpp inside #if V46_SP):
//   v46_pull_pass, v46_gemm1_item / v46_gemm2_item, Atrex loop, 4-pass streaming scatter, combine loop, v46_main.

// -------------------------------------------------------------------------------------------------------------
// V70 Iron Rule 1: all eight compute waves participate in one bounded remote-head pull.
//
// Each compute wave receives a dense worker id 0..4*ncg-1 (two four-wave groups per pure-compute CTA),
// so W00..W07 of every CTA issue this helper and every wave owns a disjoint subset of one-token jobs.
// The bound is exactly WEAVE_V_HEAD percent of the remote pass, clamped to the q1 split.  This helper
// deliberately bypasses the q1 histogram: it copies the initial remote payload so q1 GEMM1 can consume
// it immediately, while the q1 tail remains the pull pool's job.  The q1 full-pass copy starts at this
// same bound, so no token is copied twice by the head protocol.
// -------------------------------------------------------------------------------------------------------------
#ifndef V70_XSHADOW
#define V70_XSHADOW 1
#endif
__device__ __forceinline__ unsigned char* v70_xshadow_pay(const V968DispatchExt& env) {
  const unsigned lo = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<unsigned>(env.ck.ck_a_scale)));
  const unsigned hi = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<unsigned>(env.ck.ck_a_scale >> 32u)));
  const uint64_t base = (static_cast<uint64_t>(hi) << 32u) | static_cast<uint64_t>(lo);
  const uint64_t rr = static_cast<uint64_t>(__builtin_amdgcn_readfirstlane(env.ck.ck_route_rows));
  return reinterpret_cast<unsigned char*>(base + (((rr + 64ull) * 4ull + 4095ull) & ~4095ull));
}
__device__ __forceinline__ float* v70_xshadow_scl(const V968DispatchExt& env) {
  const uint64_t cap = static_cast<uint64_t>(__builtin_amdgcn_readfirstlane(env.base.capacity));
  const uint64_t pb = static_cast<uint64_t>(__builtin_amdgcn_readfirstlane(env.base.payload_bytes));
  return reinterpret_cast<float*>(v70_xshadow_pay(env) + (((16ull * cap + 16ull) * pb + 4095ull) & ~4095ull));
}
#ifndef V70_R6_HEAD_MLP
#define V70_R6_HEAD_MLP 0
#endif
#ifndef V70_R6_HEAD_ACQ
#define V70_R6_HEAD_ACQ 0
#endif
__device__ __forceinline__ void v70_remote_head_copy(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned job0, unsigned stride, unsigned byte_limit FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned tsh_raw = v46_tsplit_pct(NT, (WEAVE_V_HEAD));
  const unsigned tsh = (tsh_raw < v46_tsr(NT)) ? tsh_raw : v46_tsr(NT);
  if (tsh != 0u && stride != 0u) {
    const unsigned m = e.rank, partner = m ^ 8u;
    const unsigned node = (m >> 3u) ^ 1u;
    const unsigned skip = m & 7u;
    const unsigned K41 = v41_kdisp(NT);
    const unsigned PB = e.payload_bytes, SG = e.scale_groups;
    const unsigned maxb = (byte_limit < PB) ? byte_limit : PB;
    const unsigned nb = 7u * tsh;
    const auto* peer_rdy_tbl = reinterpret_cast<const uint64_t*>(o.peer_payload_ready_table);
    const auto* peer_pay_tbl = reinterpret_cast<const uint64_t*>(e.peer_payload_table);
    const auto* peer_scl_tbl = reinterpret_cast<const uint64_t*>(e.peer_scale_table);
    auto* local_pay_base = reinterpret_cast<unsigned char*>(peer_pay_tbl[m]) -
                           static_cast<uint64_t>(m) * e.capacity * PB;
    auto* local_scl_base = reinterpret_cast<float*>(peer_scl_tbl[m]) -
                           static_cast<uint64_t>(m) * e.capacity * SG;
    const uint32_t pay_rng = e.capacity * PB, scl_rng = e.capacity * SG * 4u;
    const auto start = __builtin_amdgcn_s_memrealtime();
#if V70_R6_HEAD_ACQ
    {
      bool bad = false;
      for (unsigned jb = job0; jb < nb && !bad; jb += 64u * stride) {
        const unsigned myjob = jb + lane * stride;
        const unsigned* fp = nullptr;
        if (myjob < nb) {
          const unsigned r = myjob / tsh, t = myjob - r * tsh;
          const unsigned src = node * 8u + r + ((r >= skip) ? 1u : 0u);
          fp = reinterpret_cast<const unsigned*>(peer_rdy_tbl[src]) + (t / K41) * K41;
        }
        unsigned sp = 0u;
        for (;;) {
          const bool ok = (fp == nullptr) || (v42_ld_sys_rlx(fp) >= 1u);
          if (__ballot(ok ? 0 : 1) == 0ull) break;
          if ((sp & 63u) == 0u && (ep16_load_relaxed((unsigned*)e.error) ||
              (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks))) {
            if (lane == 0u) v983_error((unsigned*)e.error, 969u);
            bad = true; break;
          }
          ++sp;
          __builtin_amdgcn_s_sleep(1);
        }
      }
#if (V70_R6_HEAD_ACQ) != 2   // 2 = ablation only: no acquire at all
      v42_acq_sys();
#endif
    }
#endif
#if V70_R6_HEAD_MLP
    static_assert(V70_R6_HEAD_ACQ, "V70_R6_HEAD_MLP needs V70_R6_HEAD_ACQ");
    for (unsigned jb = job0; jb < nb; jb += 4u * stride) {
      if (ep16_load_relaxed((unsigned*)e.error)) break;
      WeaveI32x4 v[4][8];
      unsigned srcs[4], ts[4];
      #pragma unroll
      for (unsigned q = 0u; q < 4u; ++q) {
        const unsigned job = jb + q * stride;
        const bool live = job < nb;
        const unsigned r = live ? job / tsh : 0u;
        const unsigned t = live ? job - r * tsh : 0u;
        const unsigned src = __builtin_amdgcn_readfirstlane(node * 8u + r + ((r >= skip) ? 1u : 0u));
        srcs[q] = src; ts[q] = t;
        const WeaveI32x4 rsx = weave_make_buf_rsrc(reinterpret_cast<const void*>(peer_pay_tbl[src]), pay_rng);
        #pragma unroll
        for (unsigned k = 0u; k < 8u; ++k) {
          const unsigned off = k * 1024u + lane * 16u;
          const int32_t o = (live && off < maxb) ? static_cast<int32_t>(t * PB + off) : static_cast<int32_t>(V45_OOB);
          v[q][k] = weave_raw_buf_load_i32x4(rsx, o, 0, V44_SC1);
        }
      }
      #pragma unroll
      for (unsigned q = 0u; q < 4u; ++q) {
        const unsigned job = jb + q * stride;
        const bool live = job < nb;
        const unsigned src = srcs[q], t = ts[q];
        const WeaveI32x4 rdx = weave_make_buf_rsrc(local_pay_base + static_cast<uint64_t>(src) * e.capacity * PB, pay_rng);
        #pragma unroll
        for (unsigned k = 0u; k < 8u; ++k) {
          const unsigned off = k * 1024u + lane * 16u;
          const int32_t o = (live && off < maxb) ? static_cast<int32_t>(t * PB + off) : static_cast<int32_t>(V45_OOB);
          v45_raw_buf_store_i32x4(v[q][k], rdx, o, 0, (V70_E3_QUIET_HT) ? 16 : 0);
        }
        if (live && byte_limit >= PB) {
          const WeaveI32x4 rss = weave_make_buf_rsrc(reinterpret_cast<const void*>(peer_scl_tbl[src]), scl_rng);
          const WeaveI32x4 rds = weave_make_buf_rsrc(local_scl_base + static_cast<uint64_t>(src) * e.capacity * SG, scl_rng);
          for (unsigned g = lane; g < SG; g += 64u) {
            const int32_t off = static_cast<int32_t>((t * SG + g) * 4u);
            const float sf = weave_raw_buf_load_f32(rss, off, 0, V44_SC1);
            v45f_buf_store_f32(sf, rds, off, 0, (V70_E3_QUIET_HT) ? 16 : 0);
          }
        }
      }
    }
#else
    for (unsigned job = job0; job < nb; job += stride) {
      const unsigned r = job / tsh;
      const unsigned t = job - r * tsh;
      const unsigned src = __builtin_amdgcn_readfirstlane(node * 8u + r + ((r >= skip) ? 1u : 0u));
#if !V70_R6_HEAD_ACQ
      if (lane == 0u) {
        const auto* rdy_p = reinterpret_cast<const unsigned*>(peer_rdy_tbl[src]) + (t / K41) * K41;
        v45_spin_rdy(rdy_p, env, x, start, 969u);
      }
#endif
      __syncwarp();
      if (ep16_load_relaxed((unsigned*)e.error)) break;
      const WeaveI32x4 rsx = weave_make_buf_rsrc(reinterpret_cast<const void*>(peer_pay_tbl[src]), pay_rng);
      const WeaveI32x4 rdx = weave_make_buf_rsrc(local_pay_base + static_cast<uint64_t>(src) * e.capacity * PB, pay_rng);
      int32_t vo[8];
      #pragma unroll
      for (unsigned k = 0u; k < 8u; ++k) {
        const unsigned off = k * 1024u + lane * 16u;
        vo[k] = (off < maxb) ? static_cast<int32_t>(t * PB + off) : static_cast<int32_t>(V45_OOB);
      }
      WeaveI32x4 v[8];
      #pragma unroll
      for (unsigned k = 0u; k < 8u; ++k) v[k] = weave_raw_buf_load_i32x4(rsx, vo[k], 0, V44_SC1);
      #pragma unroll
      for (unsigned k = 0u; k < 8u; ++k) v45_raw_buf_store_i32x4(v[k], rdx, vo[k], 0, 0);
      if (byte_limit >= PB) {
        const WeaveI32x4 rss = weave_make_buf_rsrc(reinterpret_cast<const void*>(peer_scl_tbl[src]), scl_rng);
        const WeaveI32x4 rds = weave_make_buf_rsrc(local_scl_base + static_cast<uint64_t>(src) * e.capacity * SG, scl_rng);
        for (unsigned g = lane; g < SG; g += 64u) {
          const int32_t off = static_cast<int32_t>((t * SG + g) * 4u);
          const float sf = weave_raw_buf_load_f32(rss, off, 0, V44_SC1);
          v45f_buf_store_f32(sf, rds, off, 0, 0);
        }
      }
    }
#endif  // V70_R6_HEAD_MLP
  }
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __syncwarp();
}

// Publish exactly one head completion per compute worker.  Group 0 does its
// accounting immediately after its voluntary first job; group 1 does it after
// draining the remaining head jobs.  The last contributor alone releases q1.
#ifndef V70_R6_HEAD_PUB
#define V70_R6_HEAD_PUB 0
#endif
__device__ __forceinline__ void v70_remote_head_publish_n(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned nworkers, unsigned n FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  if (ep16_load_relaxed((unsigned*)e.error) == 0u && lane == 0u) {
    auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
#if V70_E3_QUIET_HT
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    const unsigned d = __hip_atomic_fetch_add(pf + V70_HEAD_RCNT, n, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT) + n;
#else
    __threadfence();
    const unsigned d = __hip_atomic_fetch_add(pf + V70_HEAD_RCNT, n, __ATOMIC_ACQ_REL,
                                               __HIP_MEMORY_SCOPE_AGENT) + n;
#endif
    if (d == nworkers) {
      v41_st_agent_u32(pf + V70_HEAD_READY, 1u);
      if (v42_ld_agent_rlx(pf + V46_PULL_CNT + 1u) >= (V46_NPULL)) {
        v41_st_agent_u32(pf + V46_PULL_RDY + 1u, 1u);
        auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
        if (trace) trace[1305u] = __builtin_amdgcn_s_memrealtime();
      }
    }
  }
  __syncwarp();
}
__device__ __forceinline__ void v70_remote_head_publish(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned nworkers FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  if (ep16_load_relaxed((unsigned*)e.error) == 0u && lane == 0u) {
    auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
    __threadfence();
    const unsigned d = __hip_atomic_fetch_add(pf + V70_HEAD_RCNT, 1u, __ATOMIC_ACQ_REL,
                                               __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (d == nworkers) {
      v41_st_agent_u32(pf + V70_HEAD_READY, 1u);
      if (v42_ld_agent_rlx(pf + V46_PULL_CNT + 1u) >= (V46_NPULL)) {
        v41_st_agent_u32(pf + V46_PULL_RDY + 1u, 1u);
        auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
        if (trace) trace[1305u] = __builtin_amdgcn_s_memrealtime();
      }
    }
  }
  __syncwarp();
}

// Iron Rule 1, first half: all eight waves issue the bounded remote pull at
// kernel entry.  Each wave copies the first 256 B of a disjoint head token; this
// is a genuine prefix pull, but it is short enough that W00..W03 can account it
// and immediately enter the q0 GEMM1 prelude.  W04..W07 later complete every
// token, including the prefix bytes already fetched, before publishing
// V70_HEAD_READY.  The 256 B touch keeps all eight waves active at t=0 without
// serialising the local-head GEMM behind a full 2 KiB remote copy.
__device__ __forceinline__ void v70_remote_head_begin(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned worker, unsigned nworkers, bool account_here FI_CTX_PARAM) {
  FI_BG(V1988_FI_CORE, 46u, worker);
  v70_remote_head_copy(env, o, x, worker, nworkers, 256u FI_CTX_ARG);
  if (account_here) v70_remote_head_publish(env, o, nworkers FI_CTX_ARG);
  FI_EG(V1988_FI_CORE, 46u, worker);
}

// Iron Rule 1, second half: W04..W07 complete all 7*tsh head tokens.  The dense
// second-half worker id starts at zero so this pass covers token jobs 0..895,
// not only the suffix left over from the all-wave prefix pass.
__device__ __forceinline__ void v70_remote_head_finish(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned worker, unsigned second_worker, unsigned nworkers, unsigned nsecond FI_CTX_PARAM) {
  FI_BG(V1988_FI_CORE, 46u, worker);
  v70_remote_head_copy(env, o, x, second_worker, nsecond, env.base.payload_bytes FI_CTX_ARG);
  v70_remote_head_publish(env, o, nworkers FI_CTX_ARG);
  FI_EG(V1988_FI_CORE, 46u, worker);
}

// -------------------------------------------------------------------------------------------------------------
// v46_pull_pass: intra-node xGMI pull of pass p (wave-level; v45_pull_copy_mb V45_PULL=2 body, 7 sources).
//   p=0/3 : the 7 local-node sources != m,  t in [tb,te), gate LBUILD
//   p=1/2 : the 7 remote-node sources != partner (staged at their local intermediaries), t in [tb,te), gate
//           P1_BUILD; the partner's chunks of [tb,te) arrive by RDMA directly into this rank's arena and are
//           only waited for.
// -------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void v46_pull_pass(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned p, unsigned worker, unsigned nworkers FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned K41 = v41_kdisp(NT);
  const unsigned m = e.rank;
  const unsigned partner = m ^ 8u;
  const bool is_rem = v46_is_remote(p);
  const unsigned node = is_rem ? ((m >> 3u) ^ 1u) : (m >> 3u);
  const unsigned skip = m & 7u;   // m (local passes) and partner (remote passes) have the same in-node index
  const unsigned PB = e.payload_bytes;
  const unsigned SG = e.scale_groups;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const auto start = __builtin_amdgcn_s_memrealtime();
  const auto* peer_rdy_tbl = reinterpret_cast<const uint64_t*>(o.peer_payload_ready_table);
#if (V46_DBG & 8)
  return;
#endif
  const unsigned pgate = is_rem ? V46_P1_BUILD : ((p == 0u) ? V46_Q0_LBUILD : V46_LBUILD);
  if (lane == 0u) v46_wait_ge(pf + pgate, 1u, env, x, 965u);
  __syncwarp();
  v42_acq_agent();
  if (ep16_load_relaxed((unsigned*)e.error)) return;
  unsigned tb, te;
  v46_pass_range(p, NT, tb, te);
#if !V70_R8_2WAVE
  if (p == 1u) {
    const unsigned head_end = v46_tsplit_pct(NT, (WEAVE_V_HEAD));
    if (tb < head_end) tb = (head_end < te) ? head_end : te;
  }
#endif
  const unsigned cb_beg = tb / K41;
  const unsigned cb_end = (te + K41 - 1u) / K41;
  const unsigned ncb = cb_end - cb_beg;
  const auto* peer_pay_tbl = reinterpret_cast<const uint64_t*>(e.peer_payload_table);
  const auto* peer_scl_tbl = reinterpret_cast<const uint64_t*>(e.peer_scale_table);
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
#if V70_XSHADOW
  auto* local_pay_base = v70_xshadow_pay(env);
  auto* local_scl_base = v70_xshadow_scl(env);
  constexpr unsigned kNSrc = 8u;
#else
  auto* local_pay_base = reinterpret_cast<unsigned char*>(peer_pay_tbl[m]) - static_cast<uint64_t>(m) * e.capacity * PB;
  auto* local_scl_base = reinterpret_cast<float*>(peer_scl_tbl[m]) - static_cast<uint64_t>(m) * e.capacity * SG;
  constexpr unsigned kNSrc = 7u;
#endif
  const unsigned BT = V45_PULL_BT;
  const unsigned bpc = (K41 + BT - 1u) / BT;
  const unsigned nb = ncb * kNSrc * bpc;
  const unsigned ns = (PB + 1023u) >> 10u;
  const uint32_t pay_rng = e.capacity * PB, scl_rng = e.capacity * SG * 4u;
  FI_BG(V1988_FI_CORE, 16u, p);
  for (unsigned bi = worker; bi < nb; bi += nworkers) {
    const unsigned ci = bi / (kNSrc * bpc);
    const unsigned rr = bi - ci * kNSrc * bpc;
    const unsigned r = rr / bpc;
    const unsigned sub = rr - r * bpc;
#if V70_XSHADOW
    const unsigned src = __builtin_amdgcn_readfirstlane(node * 8u + r);
#else
    const unsigned src = __builtin_amdgcn_readfirstlane(node * 8u + r + ((r >= skip) ? 1u : 0u));
#endif
    const unsigned toff = sub * BT;
    const unsigned tbb = __builtin_amdgcn_readfirstlane((cb_beg + ci) * K41 + toff);
    if (lane == 0u) {
      const auto* rdy_p = reinterpret_cast<const unsigned*>(peer_rdy_tbl[src]) + (cb_beg + ci) * K41;
      v45_spin_rdy(rdy_p, env, x, start, 966u);
    }
    __syncwarp();
    if (ep16_load_relaxed((unsigned*)e.error)) break;
    unsigned nexp_l = 0u;
    {
      const unsigned tl = tbb + lane;
      if (lane < BT && toff + lane < K41 && tl < te && tl < NT)
        nexp_l = local_nexp[static_cast<uint64_t>(src) * NT + tl];
    }
    uint64_t live = __ballot(nexp_l != 0u);
    if (live == 0ull) continue;
    const WeaveI32x4 rsx = weave_make_buf_rsrc(reinterpret_cast<const void*>(peer_pay_tbl[src]), pay_rng);
    const WeaveI32x4 rdx = weave_make_buf_rsrc(local_pay_base + static_cast<uint64_t>(src) * e.capacity * PB, pay_rng);
    for (unsigned hit = 0u; hit < 12u; ++hit) {
      uint64_t rem = ((WEAVE_DEDUP_MODE) == 0) ? __ballot(nexp_l > hit) : ((hit == 0u) ? live : 0ull);
      if (rem == 0ull) break;
      unsigned tcur = static_cast<unsigned>(__builtin_ctzll(rem)), s = 0u;
      while (rem) {
        constexpr unsigned RW = (V70_REG_PULL_EFF) / 1024u;
        int32_t vo[RW];
        #pragma unroll
        for (unsigned k = 0u; k < RW; ++k) {
          const unsigned off = s * 1024u + lane * 16u;
          vo[k] = (rem != 0ull && off < PB) ? static_cast<int32_t>((tbb + tcur) * PB + off) : static_cast<int32_t>(V45_OOB);
          if (rem != 0ull) {
            if (++s == ns) {
              s = 0u;
              rem &= rem - 1ull;
              tcur = rem ? static_cast<unsigned>(__builtin_ctzll(rem)) : 0u;
            }
          }
        }
        WeaveI32x4 v[RW];
        #pragma unroll
        for (unsigned k = 0u; k < RW; ++k) v[k] = weave_raw_buf_load_i32x4(rsx, vo[k], 0, V44_SC1);
        #pragma unroll
        for (unsigned k = 0u; k < RW; ++k) v45_raw_buf_store_i32x4(v[k], rdx, vo[k], 0, 0);
      }
    }
    {
      const WeaveI32x4 rss = weave_make_buf_rsrc(reinterpret_cast<const void*>(peer_scl_tbl[src]), scl_rng);
      const WeaveI32x4 rds = weave_make_buf_rsrc(local_scl_base + static_cast<uint64_t>(src) * e.capacity * SG, scl_rng);
      for (unsigned g0 = 0u; g0 < SG; g0 += 64u) {
        const unsigned gg = g0 + lane;
        uint64_t r2 = live;
        int32_t so[8];
        #pragma unroll
        for (unsigned k = 0u; k < 8u; ++k) {
          const unsigned tt = r2 ? static_cast<unsigned>(__builtin_ctzll(r2)) : 0u;
          so[k] = (r2 != 0ull && gg < SG) ? static_cast<int32_t>(((tbb + tt) * SG + gg) * 4u) : static_cast<int32_t>(V45_OOB);
          if (r2) r2 &= r2 - 1ull;
        }
        float f[8];
        #pragma unroll
        for (unsigned k = 0u; k < 8u; ++k) f[k] = weave_raw_buf_load_f32(rss, so[k], 0, V44_SC1);
        #pragma unroll
        for (unsigned k = 0u; k < 8u; ++k) v45f_buf_store_f32(f[k], rds, so[k], 0, 0);
      }
    }
  }
#if !V70_XSHADOW
  if (is_rem) {
    for (unsigned ci = worker; ci < ncb; ci += nworkers) {
      if (lane == 0u) {
        const auto* rdy_p = reinterpret_cast<const unsigned*>(peer_rdy_tbl[partner]) + (cb_beg + ci) * K41;
        v45_spin_rdy(rdy_p, env, x, start, 967u);
      }
      __syncwarp();
    }
  } else {
    // The compute groups no longer copy local payload themselves (head fast
    // path).  The pull must therefore certify that the self rank's payload
    // chunks covered by this pass are staged before any GEMM1 descriptor in
    // the pass can read the arena.  q0 waits only its own head chunks, so the
    // first GEMM1 tile does not wait for the full local payload stage.
    for (unsigned ci = worker; ci < ncb; ci += nworkers) {
      if (lane == 0u) {
        const auto* rdy_p = reinterpret_cast<const unsigned*>(peer_rdy_tbl[m]) + (cb_beg + ci) * K41;
        v45_spin_rdy(rdy_p, env, x, start, 968u);
      }
      __syncwarp();
    }
  }
#endif
  FI_EG(V1988_FI_CORE, 16u, p);
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __syncwarp();
  if (ep16_load_relaxed((unsigned*)e.error)) return;
  if (lane == 0u) {
    __threadfence();
    const unsigned d = __hip_atomic_fetch_add(pf + V46_PULL_CNT + p, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (d == nworkers) {
      // q1 has two disjoint producers: all eight compute waves copied [0,head) and the pull pool
      // copied [head,TSR).  PULL_RDY[1] is published only after both counters are complete.
      bool ready = true;
#if !V70_R8_2WAVE
      if (p == 1u)
        ready = v42_ld_agent_rlx(pf + V70_HEAD_READY) != 0u;
#endif
      if (ready) {
        v41_st_agent_u32(pf + V46_PULL_RDY + p, 1u);
        auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
        if (trace) trace[1304u + p] = __builtin_amdgcn_s_memrealtime();
      }
    }
  }
  __syncwarp();
}

// -------------------------------------------------------------------------------------------------------------
// s13 memory-ordering levels (compile time; s14 measured V46_R1=2 V46_R3=2):
//   V46_R1 (GEMM2 -> bucket_done -> scatter chain)
//     0 = s12: RELEASE row_n_done, RELAXED bucket_done, RELAXED CAS + acquire after the ballot
//     1 = formal acquire/release chain without redundant full fences: ACQ_REL row_n_done (its release part is the
//         buffer_wbl2 sc1 + s_waitcnt vmcnt(0) of a __threadfence), completing group acquire + RELEASE bucket_done,
//         ACQUIRE CAS + acquire after the ballot
//     2 = level 1 + the explicit __threadfence() before the row_n_done add and before the bucket_done adds
//   V46_R3 (scatter data -> arrival flag)
//     0 = s12: SC0|SC1 write-through data stores, s_waitcnt vmcnt(0), RELAXED system flag (ep16_bucket_v41_flag)
//     1 = RELEASE system-scope flags (v46_bucket_flag)
//     2 = level 1 + one __threadfence_system() per batch
// -------------------------------------------------------------------------------------------------------------
#ifndef V70_R6_SCAT_NOACQ
#define V70_R6_SCAT_NOACQ 0
#endif
#ifndef V70_R6_SCAT_RLX
#define V70_R6_SCAT_RLX 0
#endif
#ifndef V70_R6_NOACQ_G2DONE
#define V70_R6_NOACQ_G2DONE 0
#endif
#ifndef V70_R8_HYB
#define V70_R8_HYB 0
#endif
#ifndef V70_R8_CNT
#define V70_R8_CNT ((V70_R8_HYB) == 2)
#endif
#ifndef V70_R6_TAIL_ALL
#define V70_R6_TAIL_ALL 0
#endif
#ifndef V70_R6_SCAT_STEAL
#define V70_R6_SCAT_STEAL 0
#endif
#ifndef V46_R1
#define V46_R1 1
#endif
#ifndef V46_R3
#define V46_R3 0
#endif
#if V46_R1 >= 1
#define V46_RND_ORDER __ATOMIC_ACQ_REL
#define V46_BD_ORDER __ATOMIC_RELEASE
#define V46_CAS_ORDER __ATOMIC_ACQUIRE
#else
#define V46_RND_ORDER __ATOMIC_RELEASE
#define V46_BD_ORDER __ATOMIC_RELAXED
#define V46_CAS_ORDER __ATOMIC_RELAXED
#endif

// -------------------------------------------------------------------------------------------------------------
// single-item GEMM1 / GEMM2 (bodies of weave_v4_gemm1_direct / weave_v4_gemm2_direct restricted to BM in {32,64},
// use_bn64 = false, g2_base = 0, skip_rdy_wait = true; run = RR in both stages)
// -------------------------------------------------------------------------------------------------------------
// uniform 64-bit value -> SGPR pair.  readfirstlane returns int: each half is converted to unsigned BEFORE widening
// (a sign-extended low half would turn every arena base whose low 32 bits are >= 2^31 into 0xffffffff........)
__device__ __forceinline__ uint64_t v46_rfl_u64(uint64_t v) {
  const unsigned lo = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<unsigned>(v)));
  const unsigned hi = static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<unsigned>(v >> 32u)));
  return (static_cast<uint64_t>(hi) << 32u) | static_cast<uint64_t>(lo);
}
__device__ __forceinline__ void v46_gemm1_item_n(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned char* arena, unsigned mt, unsigned s1s,
    unsigned n1, unsigned max_n1g FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const V968CkExt& ck = env.ck;
  const unsigned tid = threadIdx.x & 255u;
  const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(ck.ck_a + 64u);
  const unsigned num_tokens = e.world * e.capacity;
  const unsigned run = ck.ck_route_rows;
  const int32_t* stage1_ids = reinterpret_cast<const int32_t*>(ck.ck_sorted_token_ids);
  const WeaveMTileDesc d_raw = tiles[mt];
  const unsigned expert_id = __builtin_amdgcn_readfirstlane(d_raw.expert_id);
  const unsigned vr2 = __builtin_amdgcn_readfirstlane(d_raw.valid_rows);
#if WEAVE_ADAPT_BM
  const unsigned row_start = __builtin_amdgcn_readfirstlane(d_raw.row_start);
  const unsigned bm = __builtin_amdgcn_readfirstlane(d_raw.bm);
  const unsigned valid_rows = vr2;
#else
  // Independent stage-1 tile: sub-tile s1s of the stage-2 descriptor, at WEAVE_BLOCK_M_L1 rows.
  const unsigned bm = (WEAVE_BLOCK_M_L1);
  const unsigned s1off = s1s * bm;
  const unsigned row_start = __builtin_amdgcn_readfirstlane(d_raw.row_start) + s1off;
  const unsigned valid_rows = (s1off < vr2) ? ((vr2 - s1off > bm) ? bm : (vr2 - s1off)) : 0u;
#endif
  unsigned rk = __builtin_amdgcn_readfirstlane(e.rank), cap = __builtin_amdgcn_readfirstlane(e.capacity);
  unsigned pbw = __builtin_amdgcn_readfirstlane(e.payload_bytes), sgw = __builtin_amdgcn_readfirstlane(e.scale_groups);
  asm volatile("" : "+s"(rk), "+s"(cap), "+s"(pbw), "+s"(sgw));
  const uint64_t pay_e = reinterpret_cast<const uint64_t*>(e.peer_payload_table)[rk];
  const uint64_t scl_e = reinterpret_cast<const uint64_t*>(e.peer_scale_table)[rk];
  const uint64_t pay_u = v46_rfl_u64(pay_e), scl_u = v46_rfl_u64(scl_e);
  const unsigned rc = rk * cap;
  const unsigned char* recv_payload = reinterpret_cast<const unsigned char*>(pay_u - static_cast<uint64_t>(rc) * pbw);
  const float* recv_scale = reinterpret_cast<const float*>(scl_u - static_cast<uint64_t>(rc) * sgw * 4ull);
  const unsigned g1_aux = (bm << 16u) | valid_rows;
  FI_BG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
  const int32_t* ids = stage1_ids + row_start;
#if V55_COALESCE_SCHED && (V46_NSUB == 1)
  const unsigned n1b_l = ck.ck_intermediate / 128u;
  unsigned n1g = (V55_N1G); if (n1g > n1b_l - n1) n1g = n1b_l - n1;
  if (n1g > max_n1g) n1g = max_n1g;
#else
  unsigned n1g = 1u; if (n1g > max_n1g) n1g = max_n1g;
#endif
  if (valid_rows != 0u) {
  #pragma unroll
  for (unsigned n1i = n1; n1i < n1 + n1g; ++n1i) {
#if WEAVE_ADAPT_BM
    if (bm == 16u) {
      weave_s1_1x4_v62<16u, (V62_G1_SMALL)>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
    } else if (bm == 32u) {
      weave_s1_1x4_v62<32u, (V62_G1_BM32)>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
    } else if (bm == 64u) {
#if V62_G1_BODY
      weave_s1_1x4_v62<64u, (V62_G1_BODY)>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
#elif V61_G1_1X4
      weave_s1_1x4_bk128_zerob<64u>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
#else
      weave_dispatch_stage1_bk64<64u>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale), nullptr,
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
#endif
    } else {
      weave_dispatch_stage1_bk64<128u>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale), nullptr,
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
    }
#else
    if constexpr ((WEAVE_BLOCK_M_L1) == 16) {
      weave_s1_1x4_v62<16u, (V62_G1_SMALL)>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
    } else if constexpr ((WEAVE_BLOCK_M_L1) == 32) {
      weave_s1_1x4_v62<32u, (V62_G1_BM32)>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
    } else if constexpr ((WEAVE_BLOCK_M_L1) == 64) {
#if V62_G1_BODY
      weave_s1_1x4_v62<64u, (V62_G1_BODY)>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
#elif V61_G1_1X4
      weave_s1_1x4_bk128_zerob<64u>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
#else
      weave_dispatch_stage1_bk64<64u>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale), nullptr,
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
#endif
    } else {
      weave_dispatch_stage1_bk64<128u>(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale), nullptr,
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
    }
#endif
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  }
  }
  FI_EG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
  if (tid == 0u) {
    const unsigned slot = mt * (V46_NSUB) + ((WEAVE_ADAPT_BM) ? 0u : s1s);
    // A zero-valid stage-1 sub-tile still owes GEMM2 n1b completion counts; publish them once, on n1==0.
    const unsigned n1b_all = ck.ck_intermediate / 128u;
    const unsigned done_inc = (valid_rows != 0u) ? n1g : ((n1 == 0u) ? n1b_all : 0u);
    if (done_inc)
      __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.stage1_tile_done) + slot, done_inc,
                             __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// Production wrapper: the normal four-wave claim covers V55_N1G blocks.
__device__ __forceinline__ void v46_gemm1_item(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned char* arena, unsigned mt, unsigned s1s,
    unsigned n1 FI_CTX_PARAM) {
  v46_gemm1_item_n(env, o, arena, mt, s1s, n1, (V55_N1G) FI_CTX_ARG);
}

__device__ __forceinline__ bool v46_gemm2_item(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, unsigned mt, unsigned cidx,
    unsigned n2g, unsigned& last_acq_mt, unsigned* compute_done_ptr,
    unsigned v54_q, unsigned v54_x, unsigned v54_T FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const V968CkExt& ck = env.ck;
  const unsigned tid = threadIdx.x & 255u;
  const unsigned grp_sh = v46_grp_sh();
  const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(ck.ck_a + 64u);
  const unsigned n1_blocks = ck.ck_intermediate / 128u;
  const unsigned n2_blocks = o.hidden / 128u;
  const int32_t* stage2_ids = reinterpret_cast<const int32_t*>(ck.ck_m64_sorted_ids);
  const float* weights = reinterpret_cast<const float*>(o.stage2_weights);
  const unsigned* row_route_id = reinterpret_cast<const unsigned*>(o.row_route_id);
  volatile WeaveBk64LocalU32* sh_done = reinterpret_cast<volatile WeaveBk64LocalU32*>(
      reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65480u + grp_sh));
  const unsigned run = ck.ck_route_rows;
  const WeaveMTileDesc d_raw = tiles[mt];
  const unsigned expert_id = __builtin_amdgcn_readfirstlane(d_raw.expert_id);
  const unsigned row_start = __builtin_amdgcn_readfirstlane(d_raw.row_start);
  const unsigned bm = __builtin_amdgcn_readfirstlane(d_raw.bm);
  const unsigned valid_rows = __builtin_amdgcn_readfirstlane(d_raw.valid_rows);
  FI_BG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt);
  if (mt != last_acq_mt) {
    volatile WeaveBk64LocalU32* sh_w = reinterpret_cast<volatile WeaveBk64LocalU32*>(
        reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65476u + grp_sh));
    weave_4wave_barrier_65440(arena);
    if (tid == 0u) {
      const unsigned* s1p = reinterpret_cast<const unsigned*>(o.stage1_tile_done) + mt * (V46_NSUB);
      unsigned backoff = 1u, spins = 0u, ok = 1u;
      unsigned s1_all = 0u;
      #pragma unroll
      for (unsigned s1i = 0u; s1i < (V46_NSUB); ++s1i)
        s1_all |= __hip_atomic_load(s1p + s1i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      while (!v46_s1_ready(reinterpret_cast<const unsigned*>(o.stage1_tile_done), mt, n1_blocks)) {
        if (spins < 128u) asm volatile("s_nop 0\n" ::: "memory");
        else v983_sleep(backoff);
        ++spins;
        if (spins >= V1895_WAIT_SPINS) {
          v983_error(reinterpret_cast<unsigned*>(e.error), 207u);
          ok = 0u;
          break;
        }
        if (backoff <= 64u) ++backoff;
      }
      (void)s1_all;
      sh_w[0] = ok;
#if V54_CACHE_S1 && V54_S1_INV == 0
      // v54: piggy-back the narrow-scope predicate on the existing ok broadcast (no extra barrier).  It is
      // evaluated AFTER the done count reached n1_blocks, which is what makes the monotone steal counter a
      // proof that every producer of this tile ran on this XCD.
      if (ok) {
        const unsigned* v54pf = reinterpret_cast<const unsigned*>(o.plan_flags);
        if (v54_same_xcd(v54pf, v54_q, expert_id, mt - v54pf[V46_BASE + v54_q], v54_T, v54_x)) sh_w[0] = 3u;
      }
#endif
    }
    weave_4wave_barrier_65440(arena);
    const unsigned v54_st = __builtin_amdgcn_readfirstlane(sh_w[0]);
    if (v54_st == 0u) { FI_EG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt); return false; }
#if V54_S1_BYPASS_MALL
    // v59 L3: every GEMM2 A / sa load is issued with the SC1 policy (stage1_out / stage1_scale are read
    // straight from the MALL), so only the per-CU L1 has to be dropped.  The 4 MB XCD L2 is never wiped and
    // stays 100% dedicated to the read-only MoE B weights.
    asm volatile("buffer_inv sc0\ns_waitcnt vmcnt(0)\n" ::: "memory");
#elif V54_CACHE_S1 && V54_S1_INV == 0
    // same XCD as every producer: the 4 MB L2 already holds the newest line, only the per-CU L1 must go.
    if (v54_st == 3u) asm volatile("buffer_inv sc0\ns_waitcnt vmcnt(0)\n" ::: "memory");
    else
    asm volatile("buffer_inv sc1\ns_waitcnt vmcnt(0)\n" ::: "memory");
#else
    asm volatile("buffer_inv sc1\ns_waitcnt vmcnt(0)\n" ::: "memory");
#endif
    last_acq_mt = mt;
  }
  FI_EG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt);
  const unsigned g2_aux = ((n2g & 0x7u) << 28u) | ((bm & 0xfffu) << 16u) | (valid_rows & 0xffffu);
  FI_BG(V1988_FI_CORE, V1988_EV_GEMM2_TILE, g2_aux);
  const int32_t* ids = stage2_ids + row_start;
  uint16_t* c_out = reinterpret_cast<uint16_t*>(o.gemm2_output);
  // S1F direct epilogue: v46_tok_scatter tags the unique route row of every nexp==1 bucket.
  // INTRA tags are enabled by WEAVE_LOCAL_SCATTER_DEDUP; cross-node tags by WEAVE_DEDUP_MODE 1/3.
  // Only pass row tags when v46_tok_scatter actually wrote them.  DEDUP_MODE alone is insufficient:
  // local dedup may be off and mode 2 deliberately leaves cross-node rows untagged.
#ifndef V70_S1F_EN
#define V70_S1F_EN 0
#endif
  const unsigned* row_dir = (V70_S1F_EN) ? reinterpret_cast<const unsigned*>(e.sorted_weights) : nullptr;
  const uint64_t* peer_tab = reinterpret_cast<const uint64_t*>(o.peer_return_payload_table);
  const uint64_t* node_tab = reinterpret_cast<const uint64_t*>(x.node_partial_table);
  const unsigned node_row_base = (e.rank & 7u) * x.node_tokens;
  const unsigned my_node = e.rank >> 3u;
  #pragma clang loop unroll(disable)
  for (unsigned g = 0u; g < n2g; ++g) {
    const unsigned n2 = cidx * n2g + g;
#if V70_G2_DYN_SELECT
    if (bm == 16u) {
#if V62_G2_BN512 && V62_BN512_BM16
      if (g + 3u < n2g) {
        weave_s2_1x4_bn512<16u, (V62_G2_BN512)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
        g += 3u;
      } else
#endif
      if (g + 1u < n2g) {
        weave_s2_1x4_bn256<16u, (V62_G2_BN256)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
        ++g;
      } else {
        // odd trailing 128-col block (no model has one: n2g is 4 or 8); the BM=32 body masks rows >= valid_rows
        weave_dispatch_stage2_bk64<32u>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
      }
    } else
#endif
#if V62_G2_BN512 && (V62_BN512_BM48 || V70_G2_DYN_SELECT)
#if V62_BN512_BM48
    if (bm == 48u) {
      if (g + 3u < n2g) {
        weave_s2_1x4_bn512<48u, (V62_G2_BN512)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
        g += 3u;
      } else {
        weave_s2_1x4_bn256<48u, (V62_G2_BN256)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
      }
    } else
#endif
    if (bm == 128u) {
      weave_dispatch_stage2_bk64<128u>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
    } else
#endif
    if (bm == 32u) {
#if V62_G2_BN512 && V62_BN512_BM32
      if (g + 3u < n2g) {
        weave_s2_1x4_bn512<32u, (V62_G2_BN512)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
        g += 3u;
      } else
#endif
#if V62_G2_BN384 && V62_BN384_BM32
      if (g + 2u < n2g) {
        weave_s2_1x4_bn384<32u, (V62_G2_BN384)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
        g += 2u;
      } else
#endif
#if V62_G2_BN256 && V62_BN256_BM32
      if (g + 1u < n2g) {
        weave_s2_1x4_bn256<32u, (V62_G2_BN256)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
        ++g;
      } else
#endif
      weave_dispatch_stage2_bk64<32u>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
    } else {
#if V62_G2_BN512 && V62_BN512_BM64
      // V62: four adjacent 128-col blocks per K loop when a full 512-column group remains
      if (g + 3u < n2g) {
        weave_s2_1x4_bn512<64u, (V62_G2_BN512)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
        g += 3u;
      } else
#endif
#if V62_G2_BN384 && V62_BN384_BM64
      if (g + 2u < n2g) {
        weave_s2_1x4_bn384<64u, (V62_G2_BN384)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
        g += 2u;
      } else
#endif
#if V62_G2_BN256
      // V62: two adjacent 128-col blocks (n2, n2+1) per K loop: one A tile / LDS stage / barrier per 256 cols
      if (g + 1u < n2g) {
        weave_s2_1x4_bn256<64u, (V62_G2_BN256)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
        ++g;
      } else
#endif
#if V61_G2_1X4 == 2
      // V61: GEMM2 1x4 with the distance-2 load pipeline (v61_s2_pf2.hpp)
      weave_s2_1x4_pf2<64u>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
#elif V61_G2_1X4
      weave_s2_1x4_bk128_zerob<64u>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
#else
      weave_dispatch_stage2_bk64<64u>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena, row_dir, peer_tab, node_tab, node_row_base, my_node);
#endif
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  FI_EG(V1988_FI_CORE, V1988_EV_GEMM2_TILE, g2_aux);
  if (tid == 0u) {
    // s13 R1: all 4 waves' GEMM2 stores completed (vmcnt(0) + barrier); release (L2 write-back) + count
#if V46_R1 >= 2
    __threadfence();
#endif
    sh_done[0] = __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.row_n_done) + mt, n2g,
                                        V46_RND_ORDER, __HIP_MEMORY_SCOPE_AGENT) + n2g;
  }
  weave_4wave_barrier_65440(arena);
  const unsigned completed = __builtin_amdgcn_readfirstlane(sh_done[0]);  // opus55 E5
  if (completed == n2_blocks) {
    // s13 R1: the completing group acquires the other groups' released column blocks (every wave), then
    // publishes each token's row count with RELEASE (consumer: the ACQUIRE CAS in v46_sweep_pass)
#if V46_R1 >= 2
    __threadfence();
#endif
#if V46_R1 >= 1
    v42_acq_agent();
#endif
    for (unsigned r = tid; r < valid_rows; r += 256u) {
      const unsigned row = row_start + r;
      const unsigned id = row_route_id[row];
      if (id == 0xffffffffu) continue;
      const unsigned g2p = id & 0xffffffu;
      const unsigned long long divmod = v983_divmod_u32(g2p, e.capacity + 1u);
      const unsigned src = (unsigned)(divmod >> 32);
      const unsigned slot = (unsigned)divmod;
      const unsigned key = src * e.capacity + slot;
      if (row_dir != nullptr && row_dir[row] != 0u) {
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
        if ((src >> 3u) == (e.rank >> 3u)) {
          auto* ph = reinterpret_cast<unsigned*>(
              reinterpret_cast<const uint64_t*>(o.peer_home_arrival_table)[src]);
          __hip_atomic_store(ph + slot, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        } else {
          auto* na = reinterpret_cast<unsigned*>(
              reinterpret_cast<const uint64_t*>(x.node_arrival_table)[src & 7u]);
          __hip_atomic_fetch_add(na + slot, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
        }
        ep16_st_agent_relaxed(reinterpret_cast<unsigned*>(o.bucket_sent) + key, 1u);
        __hip_atomic_store(reinterpret_cast<unsigned*>(o.bucket_done) + key, 0xffffffffu,
                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        auto* pfp = reinterpret_cast<unsigned*>(o.plan_flags);
        const unsigned p = ((src >> 3u) != (e.rank >> 3u))
            ? ((slot < v46_tsr(x.node_tokens)) ? 1u : 2u)
            : ((slot < v46_tsl(x.node_tokens)) ? 0u : 3u);
        const unsigned prev = __hip_atomic_fetch_sub(pfp + V46_REM + p, 1u, __ATOMIC_ACQ_REL,
                                                     __HIP_MEMORY_SCOPE_AGENT);
        if (prev == 1u) v41_st_agent_u32(pfp + V46_SDONE + p, 1u);
        continue;
      }
      __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.bucket_done) + key, 1u,
                             V46_BD_ORDER, __HIP_MEMORY_SCOPE_AGENT);
    }
    weave_4wave_barrier_65440(arena);
    if (tid == 0u && compute_done_ptr)
      __hip_atomic_fetch_add(compute_done_ptr, o.n2_blocks, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return true;
}

// -------------------------------------------------------------------------------------------------------------
// Atrex loop: unified GEMM1 <-> GEMM2 work loop of one 4-wave compute group.
// Decision word (tid0 -> LDS 65472 -> readfirstlane): act[31:30] pass[29:28] sub[27:20] mt[19:0]
//   act 0 idle, 1 GEMM1 (sub = n1 block), 2 GEMM2 (sub = claim index), 3 exit (pass 0) / error (pass 1)
// Priority: (1) GEMM2 whose M tile already finished GEMM1, passes 1,2,0,3 (frees bucket_done -> scatter early;
//               remote rows first: they feed NodeReduce + the inter-node return RDMA)
//           (2) GEMM1 item: q0, then q1, q2 (once planned + pulled), then q3 (one item per decision = non-blocking
//               steal: a pass that is not ready yet is skipped, never waited for)
//           (3) GEMM2 of an M tile whose GEMM1 items are all claimed (waits for a running group, never for a plan)
//           (4) exit when q1/q2 are planned and every pass's GEMM2 cursor reached T*n2c
// seen bits (tid0-private, LDS): 0 LOCAL_RDY, 1 P1_PLAN, 2+q PULL_RDY[q]
// -------------------------------------------------------------------------------------------------------------
#define V46_ACT_IDLE 0u
#define V46_ACT_G1 1u
#define V46_ACT_G2 2u
#define V46_ACT_END 3u
#define V46_WORD(a, p, s, mt) ((((a) & 3u) << 30u) | (((p) & 3u) << 28u) | (((s) & 0xffu) << 20u) | ((mt) & 0xfffffu))
#define V46_W_EXIT V46_WORD(V46_ACT_END, 0u, 0u, 0u)
#define V46_W_ERR  V46_WORD(V46_ACT_END, 1u, 0u, 0u)
// Idle marker used by compact scheduling once this CTA cannot claim more work.
// The Atrex loop upgrades it to EXIT only after the exact tile-completion count
// reaches the host-staged target.  Keeping that check outside v46_decide avoids
// extending the already saturated MFMA decision live ranges.
#define V46_W_CLAIMED V46_WORD(V46_ACT_IDLE, 2u, 0u, 0u)

// Official WEAVE_READY_QUEUE: the ready-GEMM2 claim order.  Mode 1 is the V62 remote-first baseline.
__device__ __forceinline__ unsigned v46_g2_pass(unsigned j) {
#if WEAVE_READY_QUEUE == 0
  return (j == 0u) ? 0u : ((j == 1u) ? 3u : ((j == 2u) ? 1u : 2u));   // local head, local tail, remote
#elif WEAVE_READY_QUEUE == 1
  return (j < 2u) ? (j + 1u) : ((j == 2u) ? 0u : 3u);                // remote first (baseline)
#elif WEAVE_READY_QUEUE == 2
  return (j == 0u) ? 2u : ((j == 1u) ? 1u : ((j == 2u) ? 3u : 0u));
#else
  return (j == 0u) ? 3u : ((j == 1u) ? 0u : ((j == 2u) ? 2u : 1u));
#endif
}
// V70 priority contract: q0 claims establish the local head immediately.  Once the
// remote plan is visible AND at least one q0 GEMM1 claim has been made, inter-node
// GEMM1 passes run before the remaining local tiles.  Every group makes one q0
// claim in its first decision, so the local head is not starved by a pre-staged
// remote plan.
__device__ __forceinline__ bool v46_g1_q0_started(unsigned* pf) {
#if V55_COALESCE_SCHED && V55_SWIZZLE && (V46_NSUB == 1)
  unsigned n = 0u;
  #pragma unroll
  for (unsigned x = 0u; x < 4u; ++x)
    n += v42_ld_agent_rlx(pf + V55_G1CX + x);
  return n != 0u;
#else
  return v42_ld_agent_rlx(pf + V46_G1C) != 0u;
#endif
}

__device__ __forceinline__ unsigned v46_g1_pass(unsigned j, bool remote_first) {
  if (!remote_first) return j;                    // q0, q1, q2, q3
  return (j == 0u) ? 1u : ((j == 1u) ? 2u : ((j == 2u) ? 0u : 3u));
}

// A stage-2 tile owns V46_NSUB implicit stage-1 sub-tiles; all of them must have published n1b blocks.
__device__ __forceinline__ bool v46_s1_ready(const unsigned* s1d, unsigned mt, unsigned n1b) {
#if V46_NSUB == 1
  return __hip_atomic_load(s1d + mt, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= n1b;
#else
  const unsigned base = mt * (V46_NSUB);
  #pragma unroll
  for (unsigned s = 0u; s < (V46_NSUB); ++s)
    if (__hip_atomic_load(s1d + base + s, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < n1b) return false;
  return true;
#endif
}
// V70 measurement helper: earliest-wins 64-bit stamp used by the head-latency trace slot.  The four
// per-XCC GEMM1 claim queues reach tile 0 concurrently, so a plain store could overwrite the first
// timestamp with a later one.
// R4: opaque wave-uniform copy of a pointer (SGPR pair); a null test on the result is evaluated at the use site.
template <class T>
__device__ __forceinline__ T* v70_fresh_sptr(T* p) {
  const uint64_t u = reinterpret_cast<uint64_t>(p);
  uint32_t lo = __builtin_amdgcn_readfirstlane((uint32_t)u), hi = __builtin_amdgcn_readfirstlane((uint32_t)(u >> 32));
  asm volatile("" : "+s"(lo), "+s"(hi));
  return reinterpret_cast<T*>((static_cast<uint64_t>(hi) << 32) | lo);
}
__device__ __forceinline__ void v70_trace_min(unsigned long long* slot, unsigned long long v) {
  auto* p = reinterpret_cast<unsigned long long*>(slot);
  unsigned long long old = *p;
  while ((old == 0ull || v < old) && !__hip_atomic_compare_exchange_weak(
             p, &old, v, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {}
}
// Latest-wins companion for a 64-bit device timestamp.  Only lane 0 writes these
// diagnostic slots, but several hundred waves can race, so a plain store could
// regress the latest-production timestamp.
__device__ __forceinline__ void v70_trace_max(unsigned long long* slot, unsigned long long v) {
  auto* p = reinterpret_cast<unsigned long long*>(slot);
  unsigned long long old = *p;
  while ((old == 0ull || v > old) && !__hip_atomic_compare_exchange_weak(
             p, &old, v, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {}
}

// ---- V70 two-level work-order helpers ---------------------------------------------------------------
// Local queue index -> global swizzled index.  A hardware XCD owns one
// contiguous V70_XCD_CHUNK=20 slice of every conceptual 80-slot wave.
__device__ __forceinline__ unsigned v70_xcd_slots(unsigned N, unsigned x) {
  const unsigned off = x * V70_XCD_CHUNK;
  if (N <= off) return 0u;
  const unsigned rem = N - off;
  return ((rem + V70_XCD_N * V70_XCD_CHUNK - 1u) / (V70_XCD_N * V70_XCD_CHUNK)) * V70_XCD_CHUNK;
}
__device__ __forceinline__ unsigned v70_xcd_work(unsigned l, unsigned x) {
  const unsigned block = l / V70_XCD_CHUNK;
  return block * (V70_XCD_N * V70_XCD_CHUNK) + x * V70_XCD_CHUNK + (l % V70_XCD_CHUNK);
}
// Map one global GEMM1 slot to (descriptor, n1 block).  The descriptor ranges
// come from OFF[p]+lid..OFF[p]+lid+1 written by v46_build_pass.
__device__ __forceinline__ void v70_map_g1(const unsigned* off, unsigned LE, unsigned dend,
                                           unsigned n1b, unsigned g, unsigned& mt, unsigned& n1) {
  const unsigned nn = n1b / (V55_N1G);
  unsigned lid = 0u;
  for (; lid + 1u < LE; ++lid) {
    const unsigned d0 = off[lid], d1 = off[lid + 1u];
    const unsigned wc = (d1 > d0) ? (d1 - d0) * nn : 0u;
    if (g < wc) break;
    g -= wc;
  }
  const unsigned d0 = off[lid], d1 = (lid + 1u < LE) ? off[lid + 1u] : dend;
  const unsigned mc = d1 - d0;
  const unsigned gm = (mc < V70_SWZ_GROUP_M) ? mc : V70_SWZ_GROUP_M;
  const unsigned per_group = gm * nn;
  const unsigned gi = (per_group != 0u) ? (g / per_group) : 0u;
  const unsigned rem = g - gi * per_group;
  const unsigned gb = ((gi + 1u) * V70_SWZ_GROUP_M < mc) ? V70_SWZ_GROUP_M
                                                          : (mc - gi * V70_SWZ_GROUP_M);
  unsigned j = (gb != 0u) ? (rem / gb) : 0u;
  const unsigned k = rem - j * gb;
  if (gi & 1u) j = (nn - 1u) - j;
  mt = d0 + gi * V70_SWZ_GROUP_M + k;
  n1 = j * (V55_N1G);
}
// GEMM2 "ni": expert, M tile, then every N group in increasing order.
__device__ __forceinline__ void v70_map_g2(const unsigned* off, unsigned LE, unsigned dend,
                                           unsigned n2c, unsigned g, unsigned& mt, unsigned& cidx) {
  unsigned lid = 0u;
  for (; lid + 1u < LE; ++lid) {
    const unsigned d0 = off[lid], d1 = off[lid + 1u];
    const unsigned wc = (d1 > d0) ? (d1 - d0) * n2c : 0u;
    if (g < wc) break;
    g -= wc;
  }
  const unsigned d0 = off[lid], d1 = (lid + 1u < LE) ? off[lid + 1u] : dend;
  const unsigned mtl = (n2c != 0u) ? (g / n2c) : 0u;
  cidx = g - mtl * n2c;
  mt = d0 + ((mtl < d1 - d0) ? mtl : 0u);
}

__device__ __forceinline__ bool v70_claim_work_xcd(unsigned* cur, unsigned bound, unsigned x,
                                                    unsigned N, unsigned& g) {
  unsigned l = __hip_atomic_load(cur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  while (l < bound) {
    g = v70_xcd_work(l, x);
    if (g >= N) {
      unsigned e = bound;
      __hip_atomic_compare_exchange_strong(cur, &l, e, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT);
      return false;
    }
    if (__hip_atomic_compare_exchange_strong(cur, &l, l + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT)) return true;
  }
  return false;
}

// Local-expert count for the claim maps.  Production always carries it in ck_local_experts; the single-GPU
// in-situ harness leaves that envelope field 0 (its synthetic combine tail depends on it) and passes the
// count in the otherwise-unused plan slot 359 instead.  Production never writes slot 359 (host-zeroed).
#define V70_LE_HOST 359u
__device__ __forceinline__ unsigned v70_claim_le(const V968DispatchExt& env, const unsigned* pf) {
  const unsigned le = env.ck.ck_local_experts;
  return le ? le : pf[V70_LE_HOST];
}

// ---- V70 MEGA80 (opus55 r3): pass-merged expert-major claim order (internal macro, no env knob) --------------
// Per-pass order streams every local expert's weight block once per pass (4 passes, ~1.6 BM64 tiles per expert
// per pass on 2.4T rank0).  When ALL passes are plan-visible and pulled, the four per-pass queues are replaced
// by ONE expert-major queue: expert lid -> GROUP_M swizzle over that expert's tiles of ALL passes -> n-block
// snake, so the ~6.6 tiles sharing one (expert, n-block) weight block are claimed back to back inside one
// 20-slot XCD chunk.  The mode is fixed ONCE per launch by a CAS on pf[V70M_MODE] (1 = per-pass, 2 = merged),
// so no item can be claimed from both orders.  If the passes are not all ready within V70M_WAIT ticks of the
// first decision, the launch permanently falls back to the per-pass order.  Completion accounting
// (stage1_tile_done, G1D/G2D per pass) is unchanged; the decision word carries the real pass q.
#ifndef V70_R2X_MERGE
#define V70_R2X_MERGE 0
#endif
#ifndef V70_R2X_NOPRE
#define V70_R2X_NOPRE 0
#endif
// Harness scheduler ablation: number of consecutive GEMM1 claims before one
// GEMM2 claim.  1 is the production alternating schedule; 255 drains GEMM1.
#ifndef V70_SCHED_DRAIN_G1
#define V70_SCHED_DRAIN_G1 0
#endif
#ifndef V70_SCHED_G1_RUN
#define V70_SCHED_G1_RUN (V70_SCHED_DRAIN_G1 ? 255u : 1u)
#endif
// Passes that join the merged queue.  q0 (local, plan-visible first) always stays on its per-pass cursor, so the
// head phase (prelude and the first Atrex decisions) can run q0 while the remote passes are still arriving,
// without any item being claimable from two orders.
// Two merged flavours, chosen once per launch from the GEMM1 work size (no host knob):
//   mode 2: all four passes merged.  Large launches: the prelude claims nothing, the first Atrex decisions idle
//           until every pass is pulled (head <= ~3% of the run on 2.4T), then one expert-major queue.
//   mode 3: q0 stays on its per-pass cursor (the PRE4 prelude consumes q0 while W04..W07 drain the head),
//           q1..q3 merged.  Small launches, where the head copy is a large fraction of the run.
// V70M_SMALL_WORK: sum_q T_q * n1_blocks * (K / 128) below which a launch is "small" (2.4T r0 ~2.2e5, 35B r0 ~1.1e4).
#ifndef V70M_SMALL_WORK
#define V70M_SMALL_WORK 100000u
#endif
#if V70_R2X_MERGE && !(V70_R2X_PRE4 || V70_R2X_NOPRE)
#error "V70_R2X_MERGE needs V70_R2X_PRE4 or V70_R2X_NOPRE (the old prelude does not take part in the mode CAS)"
#endif
// Exit the Atrex once every GEMM2 item is CLAIMED (not completed): the CTA moves straight into the combine
// tail, whose NodeReduce/Home visits already wait for row completion themselves.
#ifndef V70_R2X_EARLY_EXIT
#define V70_R2X_EARLY_EXIT 0
#endif
// Head prelude on the R2X group-0 body with a four-wave software barrier; W00..W03 keep claiming q0 GEMM1
// items until this CTA's W04..W07 have drained the remote head.
#ifndef V70_R2X_PRE4
#define V70_R2X_PRE4 0
#endif
#define V70_PRE4_FLAG 54784u    // LDS: number of W04..W07 waves that finished the head (reset at kernel entry)
#if V70_R2X_MERGE
#define V70M_MODE 357u          // launch-wide order mode: 0 undecided, 1 per-pass, 2 merged q0..q3, 3 merged q1..q3
#define V70M_T0   358u          // low 32 bits of s_memrealtime at the first decision (| 1)
#define V70M_G1C  360u          // [4] merged GEMM1 per-XCD cursors
#define V70M_G2C  364u          // [4] merged GEMM2 per-XCD cursors
#ifndef V70M_LDS
#define V70M_LDS  53248u        // LDS table: [0..64] tile prefix per expert, [65 + q*64 + lid] = (d0 << 10) | m
#endif
#ifndef V70M_WAIT
#define V70M_WAIT 30000u        // 300 us at 100 MHz
#endif
// ---- V70_R2X_COMPACT (opus55 r4): cross-pass expert row compaction ------------------------------------------
// Merged mode 4 = mode 2 (all four passes merged) + compacted tiles: the rows of one local expert lid of ALL
// passes are concatenated (q0 | q1 | q2 | q3 segments, each contiguous in the stage row space) and cut into
// 64-row tiles, so an expert pays ONE partial tail tile instead of up to four (2.4T rank0: 211 -> 161 tiles).
// A compacted tile is named by c = expert tile prefix + k and travels in the decision word as V70C_FLAG | c;
// its stage1_tile_done / row_n_done slot is c (host-zeroed like every descriptor slot, c < sum_q T_q).  The
// GEMM body reads its rows from the LDS row table R2X_RTAB (filled per tile), no new buffer, no ABI change.
// LDS table (same 53248 base): [0..64] compacted tile prefix per expert, [65 + q*64 + lid] = (rs >> 4) << 17 | cnt.
#ifndef V70_R2X_COMPACT
#define V70_R2X_COMPACT 0
#endif
#if V70_R2X_COMPACT && !(V70_R2X_EARLY_EXIT)
#error "V70_R2X_COMPACT needs V70_R2X_EARLY_EXIT=1"
#endif
#define V70C_FLAG 0x10000u
__device__ __forceinline__ unsigned v70m_mask(unsigned md) { return (md == 2u || md == 4u) ? 0xFu : 0xEu; }
// launch-wide size class (identical in every CTA)
__device__ __forceinline__ bool v70m_small(const V968DispatchExt& env, const unsigned* pf) {
  const unsigned T = pf[V46_T] + pf[V46_T + 1u] + pf[V46_T + 2u] + pf[V46_T + 3u];
  const unsigned long long w = static_cast<unsigned long long>(T) * (env.ck.ck_intermediate / 128u) *
                               (env.ck.ck_hidden / 128u);
  return w < (V70M_SMALL_WORK);
}
// R4: explicit LDS address space -> ds_read/ds_write with a 32-bit offset.  The generic volatile pointer used
// before was lowered to flat_* accesses with a 64-bit (src_shared_base:offset) VGPR pair that the production
// stage kernel spilled to scratch (2 x 8-byte slots, 12 VGPR spills, private 20).
__device__ __forceinline__ volatile WeaveBk64LocalU32* v70m_tab() {
  return reinterpret_cast<volatile WeaveBk64LocalU32*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + V70M_LDS));
}
// One full wave (64 lanes) builds the table; lane = local expert id.
__device__ __forceinline__ void v70m_build(const unsigned* pf, unsigned LE, unsigned mask) {
  const unsigned lane = threadIdx.x & 63u;
  volatile WeaveBk64LocalU32* tb = v70m_tab();
  v42_acq_agent();
  unsigned M = 0u;
  #pragma unroll
  for (unsigned q = 0u; q < V46_NP; ++q) {
    unsigned d0 = 0u, m = 0u;
    if ((mask >> q) & 1u) if (lane < LE) {
      d0 = pf[V46_OFF(q) + lane];
      const unsigned d1 = (lane + 1u < LE) ? pf[V46_OFF(q) + lane + 1u] : (pf[V46_BASE + q] + pf[V46_T + q]);
      m = (d1 > d0) ? (d1 - d0) : 0u;
      if (m > 1023u) m = 1023u;
    }
    tb[65u + q * 64u + lane] = (d0 << 10u) | m;
    M += m;
  }
  unsigned sc = M;
  #pragma unroll
  for (unsigned d = 1u; d < 64u; d <<= 1u) {
    const unsigned t = __shfl_up(sc, d);
    if (lane >= d) sc += t;
  }
  tb[lane + 1u] = sc;
  if (lane == 0u) { tb[0] = 0u; tb[321u] = 0u; }
}
#ifndef V70_R8_2WAVE
#define V70_R8_2WAVE 0
#endif
#if V70_R2X_COMPACT
__device__ __forceinline__ unsigned v70c_pass_tiles(
    const WeaveMTileDesc* descs, const unsigned* pf, unsigned LE, unsigned q, unsigned lane) {
  if (lane >= LE) return 0u;
  const unsigned d0 = pf[V46_OFF(q) + lane];
  const unsigned d1 = (lane + 1u < LE) ? pf[V46_OFF(q) + lane + 1u] : (pf[V46_BASE + q] + pf[V46_T + q]);
  if (d1 <= d0) return 0u;
  const unsigned rs = descs[d0].row_start;
  const WeaveMTileDesc dl = descs[d1 - 1u];
  const unsigned cnt = (dl.row_start - rs) + dl.valid_rows;
  return (cnt + ((V70C_BM) - 1u)) >> (V70C_SH);
}
__device__ __forceinline__ void v70c_build_3g(
    const V968DispatchExt& env, const unsigned* pf, unsigned LE, unsigned rg) {
  const unsigned lane = threadIdx.x & 63u;
  volatile WeaveBk64LocalU32* tb = v70m_tab();
  const auto* descs = reinterpret_cast<const WeaveMTileDesc*>(env.ck.ck_a + 64u);
  const unsigned mask = (rg == 0u) ? 0x1u : ((rg == 1u) ? 0x6u : 0x8u);
  unsigned prior_sc = 0u;
  if (rg >= 1u) prior_sc += v70c_pass_tiles(descs, pf, LE, 0u, lane);
  if (rg == 1u) prior_sc += v70c_pass_tiles(descs, pf, LE, 3u, lane);
  const unsigned slot_base = (rg == 0u) ? 0u : v46_wave_sum(prior_sc);
  unsigned n = 0u, bad = 0u;
  #pragma unroll
  for (unsigned q = 0u; q < V46_NP; ++q) {
    unsigned pk = 0u;
    if (((mask >> q) & 1u) && lane < LE) {
      const unsigned d0 = pf[V46_OFF(q) + lane];
      const unsigned d1 = (lane + 1u < LE) ? pf[V46_OFF(q) + lane + 1u] : (pf[V46_BASE + q] + pf[V46_T + q]);
      if (d1 > d0) {
        const unsigned rs = descs[d0].row_start;
        const WeaveMTileDesc dl = descs[d1 - 1u];
        const unsigned cnt = (dl.row_start - rs) + dl.valid_rows;
        if ((rs & 15u) != 0u || (rs >> 4u) >= (1u << 15u) || cnt >= (1u << 17u)) bad = 1u;
        pk = ((rs >> 4u) << 17u) | (cnt & 0x1ffffu);
        n += cnt;
      }
    }
    tb[65u + q * 64u + lane] = pk;
  }
  if (bad) v983_error(reinterpret_cast<unsigned*>(env.base.error), 976u);
  unsigned sc = (n + ((V70C_BM) - 1u)) >> (V70C_SH);
  #pragma unroll
  for (unsigned d = 1u; d < 64u; d <<= 1u) {
    const unsigned t = __shfl_up(sc, d);
    if (lane >= d) sc += t;
  }
  tb[lane + 1u] = sc;
  if (lane == 0u) { tb[0] = 0u; tb[321u] = slot_base; }
}
// One full wave builds the compacted table; lane = local expert id.  Per-pass segment = [rs, rs + cnt) with
// rs = row_start of the expert's first descriptor of that pass and cnt = (last.row_start - rs) + last.valid_rows
// (every non-last descriptor of an expert is full, so its rows are contiguous).
__device__ __forceinline__ void v70c_build(const V968DispatchExt& env, const unsigned* pf, unsigned LE) {
  const unsigned lane = threadIdx.x & 63u;
  volatile WeaveBk64LocalU32* tb = v70m_tab();
  const auto* descs = reinterpret_cast<const WeaveMTileDesc*>(env.ck.ck_a + 64u);
  v42_acq_agent();
  unsigned n = 0u, bad = 0u;
  #pragma unroll
  for (unsigned q = 0u; q < V46_NP; ++q) {
    unsigned pk = 0u;
    if (lane < LE) {
      const unsigned d0 = pf[V46_OFF(q) + lane];
      const unsigned d1 = (lane + 1u < LE) ? pf[V46_OFF(q) + lane + 1u] : (pf[V46_BASE + q] + pf[V46_T + q]);
      if (d1 > d0) {
        const unsigned rs = descs[d0].row_start;
        const WeaveMTileDesc dl = descs[d1 - 1u];
        const unsigned cnt = (dl.row_start - rs) + dl.valid_rows;
        if ((rs & 15u) != 0u || (rs >> 4u) >= (1u << 15u) || cnt >= (1u << 17u)) bad = 1u;
        pk = ((rs >> 4u) << 17u) | (cnt & 0x1ffffu);
        n += cnt;
      }
    }
    tb[65u + q * 64u + lane] = pk;
  }
  if (bad) v983_error(reinterpret_cast<unsigned*>(env.base.error), 976u);
  unsigned sc = (n + ((V70C_BM) - 1u)) >> (V70C_SH);
  #pragma unroll
  for (unsigned d = 1u; d < 64u; d <<= 1u) {
    const unsigned t = __shfl_up(sc, d);
    if (lane >= d) sc += t;
  }
  tb[lane + 1u] = sc;
  if (lane == 0u) { tb[0] = 0u; tb[321u] = 0u; }
}
// compacted slot g -> (compacted tile c, n-block group j): expert -> GROUP_M swizzle over its tiles -> n snake
__device__ __forceinline__ void v70c_map(unsigned LE, unsigned nn, unsigned g, unsigned& c, unsigned& j) {
  volatile WeaveBk64LocalU32* tb = v70m_tab();
  unsigned lo = 0u, hi = LE;
  while (hi - lo > 1u) {
    const unsigned mid = (lo + hi) >> 1u;
    if (tb[mid] * nn <= g) lo = mid; else hi = mid;
  }
  const unsigned lid = lo;
  const unsigned M = tb[lid + 1u] - tb[lid];
  g -= tb[lid] * nn;
  const unsigned gm = (M < V70_SWZ_GROUP_M) ? M : V70_SWZ_GROUP_M;
  const unsigned per_group = gm * nn;
  const unsigned gi = (per_group != 0u) ? (g / per_group) : 0u;
  const unsigned rem = g - gi * per_group;
  const unsigned gb = ((gi + 1u) * V70_SWZ_GROUP_M < M) ? V70_SWZ_GROUP_M : (M - gi * V70_SWZ_GROUP_M);
  j = (gb != 0u) ? (rem / gb) : 0u;
  const unsigned k = gi * V70_SWZ_GROUP_M + (rem - j * gb);
  if ((gi + lid) & 1u) j = (nn - 1u) - j;
  c = tb[lid] + k;
}
#endif
// merged slot g -> (pass q, descriptor mt, n-block group j)
__device__ __forceinline__ void v70m_map(unsigned LE, unsigned nn, unsigned g, unsigned& q, unsigned& mt,
                                         unsigned& j, unsigned mask) {
  volatile WeaveBk64LocalU32* tb = v70m_tab();
  unsigned lo = 0u, hi = LE;
  while (hi - lo > 1u) {
    const unsigned mid = (lo + hi) >> 1u;
    if (tb[mid] * nn <= g) lo = mid; else hi = mid;
  }
  const unsigned lid = lo;
  const unsigned M = tb[lid + 1u] - tb[lid];
  g -= tb[lid] * nn;
  const unsigned gm = (M < V70_SWZ_GROUP_M) ? M : V70_SWZ_GROUP_M;
  const unsigned per_group = gm * nn;
  const unsigned gi = (per_group != 0u) ? (g / per_group) : 0u;
  const unsigned rem = g - gi * per_group;
  const unsigned gb = ((gi + 1u) * V70_SWZ_GROUP_M < M) ? V70_SWZ_GROUP_M : (M - gi * V70_SWZ_GROUP_M);
  j = (gb != 0u) ? (rem / gb) : 0u;
  unsigned k = gi * V70_SWZ_GROUP_M + (rem - j * gb);
  if ((gi + lid) & 1u) j = (nn - 1u) - j;
  q = 0u; mt = 0u;
  #pragma unroll
  for (unsigned qq = 0u; qq < V46_NP; ++qq) {
    if (!((mask >> qq) & 1u)) continue;
    const unsigned pk = tb[65u + qq * 64u + lid];
    const unsigned m = pk & 1023u;
    if (k < m) { q = qq; mt = (pk >> 10u) + k; break; }
    k -= m;
  }
}
#endif  // V70_R2X_MERGE
// every GEMM2 item of a claim sequence with per-XCD cursors cur[0..3] over N slots has been claimed
__device__ __forceinline__ bool v70_xcd_all_claimed(const unsigned* cur, unsigned N) {
  #pragma unroll
  for (unsigned xx = 0u; xx < V70_XCD_N; ++xx) {
    const unsigned l = __hip_atomic_load(cur + xx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    if (l < v70_xcd_slots(N, xx) && v70_xcd_work(l, xx) < N) return false;
  }
  return true;
}

__device__ __forceinline__ unsigned v46_decide(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned& seen, unsigned n1b_in, unsigned n2c_in,
    unsigned v54_x = 0u) {
  unsigned n1b = __builtin_amdgcn_readfirstlane(n1b_in), n2c = __builtin_amdgcn_readfirstlane(n2c_in);
  asm volatile("" : "+s"(n1b), "+s"(n2c));   // opus55 v46 E6: re-defined per decision -> no hoisted reciprocal
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const unsigned* s1d = reinterpret_cast<const unsigned*>(o.stage1_tile_done);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  if (ep16_load_relaxed((unsigned*)env.base.error)) return V46_W_ERR;
  // HEAD FAST PATH: q0 becomes runnable as soon as its own descriptors, padding and id scatter are
  // published (V46_Q0_RDY), without waiting for q3 or the full local plan.
  if ((seen & 0x100u) == 0u) {
    if (v42_ld_agent_rlx(pf + V46_Q0_RDY) != 0u) {
      v42_acq_agent();
      seen |= 0x100u;
      if (trace) trace[1384u] = __builtin_amdgcn_s_memrealtime();
    }
  }
  if ((seen & 1u) == 0u) {
    if (v42_ld_agent_rlx(pf + V46_LOCAL_RDY) != 0u) { v42_acq_agent(); seen |= 1u; }
  }
  const bool local_full = (seen & 1u) != 0u;
  const bool q0_ready = (seen & 0x100u) != 0u;
  if (!local_full && !q0_ready) return V46_WORD(V46_ACT_IDLE, 0u, 0u, 0u);
  if ((seen & 2u) == 0u && v42_ld_agent_rlx(pf + V46_P1_PLAN) != 0u) { v42_acq_agent(); seen |= 2u; }
  #pragma unroll
  for (unsigned q = 0u; q < V46_NP; ++q)
    if ((seen & (4u << q)) == 0u && v42_ld_agent_rlx(pf + V46_PULL_RDY + q) != 0u) { v42_acq_agent(); seen |= 4u << q; }
  const bool remote_g1_first = ((seen & 2u) != 0u) && v46_g1_q0_started(pf);
#if V70_TWO_LEVEL
  // Iron Rule 4 scheduler.  Each decision prefers the stage that lost the
  // previous round, so a continuously ready GEMM2 queue cannot drain all of L2
  // before GEMM1 (or vice versa).  Own-XCD work is attempted first, followed by
  // the dynamic tail steal from the other three XCD queues.
  {
    const unsigned xcd = v55_xcc() & 3u;
    const unsigned LE = v70_claim_le(env, pf);
    const unsigned nn1 = n1b / (V55_N1G);
    if ((n1b % (V55_N1G)) != 0u) return V46_W_ERR;
    const auto g1_ok = [&](unsigned q) -> bool {
      if (v46_is_remote(q) && (seen & 2u) == 0u) return false;
      if ((seen & (4u << q)) == 0u) return false;
      if (q == 0u && !q0_ready) return false;
      if (q == 3u && (!local_full || (seen & 4u) == 0u)) return false;
      return pf[V46_T + q] != 0u;
    };
    const auto g2_ok = [&](unsigned q) -> bool {
      if (v46_is_remote(q) && (seen & 2u) == 0u) return false;
      if (q != 0u && !v46_is_remote(q) && !local_full) return false;
      if (q == 0u && !q0_ready) return false;
      return pf[V46_T + q] != 0u;
    };
    // G2 queue head: inspect readiness before advancing the contiguous cursor.
    const auto g2_take = [&](unsigned q, unsigned xx) -> unsigned {
      const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q];
      const unsigned N = T * n2c;
      const unsigned bound = v70_xcd_slots(N, xx);
      unsigned* cur = pf + V62_G2CX + q * 4u + xx;
      unsigned l = __hip_atomic_load(cur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      while (l < bound) {
        const unsigned g = v70_xcd_work(l, xx);
        if (g >= N) break;
        unsigned mt, cidx;
        v70_map_g2(pf + V46_OFF(q), LE, B + T, n2c, g, mt, cidx);
        if (!v46_s1_ready(s1d, mt, n1b)) break;
        if (__hip_atomic_compare_exchange_strong(cur, &l, l + 1u, __ATOMIC_RELAXED,
                                                 __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
          return V46_WORD(V46_ACT_G2, q, cidx, mt);
      }
      return V46_NONE;
    };
    const auto g1_take = [&](unsigned q, unsigned xx) -> unsigned {
      const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q];
      const unsigned N = T * nn1;
      const unsigned bound = v70_xcd_slots(N, xx);
      unsigned* cur = pf + V55_G1CX + q * 4u + xx;
      unsigned g = 0u;
      if (!v70_claim_work_xcd(cur, bound, xx, N, g)) return V46_NONE;
      unsigned mt, n1;
      v70_map_g1(pf + V46_OFF(q), LE, B + T, n1b, g, mt, n1);
      if (mt == B && trace) v70_trace_min(&trace[1308u + q], __builtin_amdgcn_s_memrealtime());
      return V46_WORD(V46_ACT_G1, q, n1, mt);
    };
    const auto try_g1 = [&](bool steal) -> unsigned {
      #pragma unroll
      for (unsigned j = 0u; j < V46_NP; ++j) {
        const unsigned q = v46_g1_pass(j, remote_g1_first);
        if (!g1_ok(q)) continue;
        if (!steal) {
          const unsigned w = g1_take(q, xcd);
          if (w != V46_NONE) return w;
        } else {
          #pragma unroll
          for (unsigned dx = 1u; dx < V70_XCD_N; ++dx) {
            const unsigned w = g1_take(q, (xcd + dx) & 3u);
            if (w != V46_NONE) return w;
          }
        }
      }
      return V46_NONE;
    };
    const auto try_g2 = [&](bool steal) -> unsigned {
      #pragma unroll
      for (unsigned j = 0u; j < V46_NP; ++j) {
        const unsigned q = v46_g2_pass(j);
        if (!g2_ok(q)) continue;
        if (!steal) {
          const unsigned w = g2_take(q, xcd);
          if (w != V46_NONE) return w;
        } else {
          #pragma unroll
          for (unsigned dx = 1u; dx < V70_XCD_N; ++dx) {
            const unsigned w = g2_take(q, (xcd + dx) & 3u);
            if (w != V46_NONE) return w;
          }
        }
      }
      return V46_NONE;
    };
#if V70_R2X_MERGE
    {
      unsigned md = __hip_atomic_load(pf + V70M_MODE, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      // Passes outside the merged mask (q0 in mode 3) use their own per-pass cursors.  In mode 0 nothing is
      // claimable from a per-pass cursor (a large launch merges q0 too).
      const unsigned pmask = (md == 3u) ? 0xEu : 0xFu;
      const auto pq_take = [&](bool g1, bool steal) -> unsigned {
        #pragma unroll
        for (unsigned q = 0u; q < V46_NP; ++q) {
          if ((pmask >> q) & 1u) continue;
          if (g1 ? !g1_ok(q) : !g2_ok(q)) continue;
          #pragma unroll
          for (unsigned dx = (steal ? 1u : 0u); dx < (steal ? V70_XCD_N : 1u); ++dx) {
            const unsigned xx = (xcd + dx) & 3u;
            const unsigned w = g1 ? g1_take(q, xx) : g2_take(q, xx);
            if (w != V46_NONE) return w;
          }
        }
        return V46_NONE;
      };
      if (md == 0u) {
        bool all = local_full && (seen & 2u) != 0u && q0_ready;
        #pragma unroll
        for (unsigned q = 0u; q < V46_NP; ++q)
          if (pf[V46_T + q] != 0u && (seen & (4u << q)) == 0u) all = false;
        unsigned want = all ? ((V70_R2X_COMPACT) ? 4u : 2u) : 0u;
        if (!all) {
          const unsigned now = static_cast<unsigned>(__builtin_amdgcn_s_memrealtime()) | 1u;
          unsigned t0 = 0u;
          __hip_atomic_compare_exchange_strong(pf + V70M_T0, &t0, now, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
          if (t0 != 0u && now - t0 >= V70M_WAIT) want = 1u;   // permanent per-pass fallback
        }
        if (want != 0u) {
          unsigned e0 = 0u;
          __hip_atomic_compare_exchange_strong(pf + V70M_MODE, &e0, want, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
          md = (e0 == 0u) ? want : e0;
        }
      }
      if (md != 1u) {
        // First merged decision of this CTA: ask wave 0 to build the LDS table (needs 64 lanes).
        const bool mg = (md >= 2u);
        const unsigned mmask = v70m_mask(md);
        if (mg && (seen & 0x400u) == 0u) { seen |= 0x400u; return V46_WORD(V46_ACT_IDLE, 1u, 0u, 0u); }
        const unsigned NT = mg ? v70m_tab()[LE] : 0u;
        const auto g1m = [&](unsigned xx) -> unsigned {
          const unsigned N = NT * nn1;
          unsigned g = 0u;
          if (N == 0u || !v70_claim_work_xcd(pf + V70M_G1C + xx, v70_xcd_slots(N, xx), xx, N, g)) return V46_NONE;
#if V70_R2X_COMPACT
          if (md == 4u) {
            unsigned c, jc;
            v70c_map(LE, nn1, g, c, jc);
            return V46_WORD(V46_ACT_G1, 0u, jc * (V55_N1G), V70C_FLAG | c);
          }
#endif
          unsigned q, mt, j;
          v70m_map(LE, nn1, g, q, mt, j, mmask);
          if (mt == pf[V46_BASE + q] && trace) v70_trace_min(&trace[1308u + q], __builtin_amdgcn_s_memrealtime());
          return V46_WORD(V46_ACT_G1, q, j * (V55_N1G), mt);
        };
        const auto g2m = [&](unsigned xx) -> unsigned {
          const unsigned N = NT * n2c;
          const unsigned bound = v70_xcd_slots(N, xx);
          unsigned* cur = pf + V70M_G2C + xx;
          unsigned l = __hip_atomic_load(cur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          while (l < bound) {
            const unsigned g = v70_xcd_work(l, xx);
            if (g >= N) break;
            unsigned q, mt, cidx;
#if V70_R2X_COMPACT
            if (md == 4u) {
              v70c_map(LE, n2c, g, mt, cidx);
              if (!v46_s1_ready(s1d, mt, n1b)) break;
              if (__hip_atomic_compare_exchange_strong(cur, &l, l + 1u, __ATOMIC_RELAXED,
                                                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
                return V46_WORD(V46_ACT_G2, 0u, cidx, V70C_FLAG | mt);
              continue;
            }
#endif
            v70m_map(LE, n2c, g, q, mt, cidx, mmask);
            if (!v46_s1_ready(s1d, mt, n1b)) break;
            if (__hip_atomic_compare_exchange_strong(cur, &l, l + 1u, __ATOMIC_RELAXED,
                                                     __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
              return V46_WORD(V46_ACT_G2, q, cidx, mt);
          }
          return V46_NONE;
        };
        const auto stage = [&](bool g1, bool steal) -> unsigned {
          unsigned w = (md == 3u) ? pq_take(g1, steal) : V46_NONE;
          #pragma unroll
          for (unsigned dx = (steal ? 1u : 0u); dx < (steal ? V70_XCD_N : 1u) && w == V46_NONE; ++dx)
            w = g1 ? g1m((xcd + dx) & 3u) : g2m((xcd + dx) & 3u);
          return w;
        };
#if V70_SCHED_G1_RUN > 1
        const unsigned g1_run = (seen >> 16u) & 0xffu;
        const bool pg1 = g1_run < (V70_SCHED_G1_RUN);
#else
        const bool pg1 = (seen & 0x200u) != 0u;
#endif
        unsigned w = stage(pg1, false);
        if (w == V46_NONE) w = stage(!pg1, false);
        if (w == V46_NONE) w = stage(pg1, true);
        if (w == V46_NONE) w = stage(!pg1, true);
        if (w != V46_NONE) {
#if V70_SCHED_G1_RUN > 1
          if ((w >> 30u) == V46_ACT_G1) {
            const unsigned r = ((seen >> 16u) & 0xffu) + 1u;
            seen = (seen & ~0xff0000u) | ((r < (V70_SCHED_G1_RUN) ? r : (V70_SCHED_G1_RUN)) << 16u);
            seen |= 0x200u;
          } else {
            seen &= ~(0xff0000u | 0x200u);
          }
#else
          if ((w >> 30u) == V46_ACT_G1) seen &= ~0x200u; else seen |= 0x200u;
#endif
          return w;
        }
        if (!mg) return V46_WORD(V46_ACT_IDLE, 0u, 0u, 0u);
#if V70_R2X_EARLY_EXIT && V70_R2X_COMPACT
        // Compact mode names tiles by the derived compact id, so the per-pass
        // G2D counters are intentionally inert.  The exact tile-completion gate
        // runs in the outer Atrex loop, outside this register-critical body.
        return V46_W_CLAIMED;
#else
        bool done = true;
#if V70_R2X_EARLY_EXIT
        if (!v70_xcd_all_claimed(pf + V70M_G2C, NT * n2c)) done = false;
        #pragma unroll
        for (unsigned q = 0u; q < V46_NP; ++q)
          if (!((mmask >> q) & 1u) && !v70_xcd_all_claimed(pf + V62_G2CX + q * 4u, pf[V46_T + q] * n2c))
            done = false;
#else
        #pragma unroll
        for (unsigned q = 0u; q < V46_NP; ++q)
          if (__hip_atomic_load(pf + V46_G2D + q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
              pf[V46_T + q] * n2c) done = false;
#endif
        return done ? V46_W_EXIT : V46_WORD(V46_ACT_IDLE, 0u, 0u, 0u);
# endif
      }
    }
#endif
    // Bit 9 means "prefer GEMM1 on the next decision".  After a successful claim
    // flip it, so both stages alternate instead of repeatedly retrying the last
    // stage.  The old code stored "last stage" and then tested it as a preference,
    // which drained nearly all GEMM1 before GEMM2 could start.
    const bool prefer_g1 = (seen & 0x200u) != 0u;
    unsigned w = V46_NONE;
    if (prefer_g1) {
      w = try_g1(false); if (w == V46_NONE) w = try_g2(false);
      if (w == V46_NONE) w = try_g1(true);
      if (w == V46_NONE) w = try_g2(true);
    } else {
      w = try_g2(false); if (w == V46_NONE) w = try_g1(false);
      if (w == V46_NONE) w = try_g2(true);
      if (w == V46_NONE) w = try_g1(true);
    }
    if (w != V46_NONE) {
      if ((w >> 30u) == V46_ACT_G1) seen &= ~0x200u;
      else seen |= 0x200u;
      return w;
    }
    // Exit only after every claimed GEMM2 item has completed (V70_R2X_EARLY_EXIT: once every item is claimed).
    if (local_full && (seen & 2u)) {
#if V70_R2X_EARLY_EXIT && V70_R2X_COMPACT
      return V46_W_CLAIMED;
#else
      bool all = true;
      #pragma unroll
      for (unsigned q = 0u; q < V46_NP; ++q)
#if V70_R2X_EARLY_EXIT
        if (!v70_xcd_all_claimed(pf + V62_G2CX + q * 4u, pf[V46_T + q] * n2c)) all = false;
#else
        if (__hip_atomic_load(pf + V46_G2D + q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
            pf[V46_T + q] * n2c) all = false;
#endif
      if (all) return V46_W_EXIT;
#endif
    }
    return V46_WORD(V46_ACT_IDLE, 0u, 0u, 0u);
  }
#elif V54_XCD_PART && (V46_NSUB != 1)
#error "V54_XCD_PART claim queues do not carry stage-1 sub-tiles"
#elif V54_XCD_PART
  // ---- v54 Fleet 4-XCD home queues: identical phase order / pass order / readiness gates as the base, but
  // every flat cursor is replaced by 4 per-XCD cursors driven own-first with a clean steal fallback. ----
  const bool q3ok = true;   // v46 blood line has no V48 Q3R gate; folds every !q3ok test away
  const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(env.ck.ck_a + 64u);
  const unsigned xcd = v54_x;
  unsigned from = 0u;
  // (1) ready GEMM2 heads, pass order 1, 2, 0, 3
  #pragma unroll
  for (unsigned j = 0u; j < V46_NP; ++j) {
    const unsigned q = v46_g2_pass(j);
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
    if (!v46_is_remote(q) && !local_full) continue;
    const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q];
    const unsigned c = v54_claim<1u>(pf + V54_G2C + q * 4u, T * n2c, n2c, T, B, tiles, xcd,
                                     n1b, s1d, pf + V54_G1C + q * 4u, from);
    if (c != V46_NONE) {
      const unsigned mtl = c / n2c;
      return V46_WORD(V46_ACT_G2, q, c - mtl * n2c, B + mtl);
    }
  }
  // (2) GEMM1 items: q0 head bootstrap, then q1/q2 before remaining local tiles.
  #pragma unroll
  for (unsigned j = 0u; j < V46_NP; ++j) {
    const unsigned q = v46_g1_pass(j, remote_g1_first);
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
    if ((seen & (4u << q)) == 0u) continue;
    if (q == 0u && !q0_ready) continue;            // head fast path only needs q0
    if (q == 3u && !local_full) continue;          // q3 waits for the full local plan
    if (q == 3u && (seen & 4u) == 0u) continue;   // own-rank staging is certified by PULL_RDY[0]
    const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q];
    const unsigned c = v54_claim<0u>(pf + V54_G1C + q * 4u, T * n1b, n1b, T, B, tiles, xcd,
                                     n1b, s1d, pf + V54_G1C + q * 4u, from);
    if (c != V46_NONE) {
      // A foreign XCD taking a GEMM1 item out of home queue `from` invalidates the same-XCD proof for that
      // queue in this pass, so the narrow-scope acquire must fall back to buffer_inv sc1 from now on.
      if (from != xcd)
        __hip_atomic_fetch_add(pf + V54_G1X + q * 4u + from, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (c == 0u && trace) trace[1308u + q] = __builtin_amdgcn_s_memrealtime();
      const unsigned mtl = c / n1b;
      return V46_WORD(V46_ACT_G1, q, c - mtl * n1b, B + mtl);
    }
  }
  // (3) GEMM2 of an M tile whose GEMM1 items are all claimed, pass order 1, 2, 0, 3
  #pragma unroll
  for (unsigned j = 0u; j < V46_NP; ++j) {
    const unsigned q = v46_g2_pass(j);
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
    if (!v46_is_remote(q) && !local_full) continue;
    const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q];
    const unsigned c = v54_claim<2u>(pf + V54_G2C + q * 4u, T * n2c, n2c, T, B, tiles, xcd,
                                     n1b, s1d, pf + V54_G1C + q * 4u, from);
    if (c != V46_NONE) {
      const unsigned mtl = c / n2c;
      return V46_WORD(V46_ACT_G2, q, c - mtl * n2c, B + mtl);
    }
  }
  // (4) exit test: every one of the 16 GEMM2 home queues must have reached its pass bound
  if (local_full && (seen & 2u)) {
    bool all = true;
    #pragma unroll
    for (unsigned q = 0u; q < V46_NP; ++q) {
      const unsigned bd = pf[V46_T + q] * n2c;
      #pragma unroll
      for (unsigned y = 0u; y < 4u; ++y)
        if (__hip_atomic_load(pf + V54_G2C + q * 4u + y, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < bd) all = false;
    }
    if (all) return V46_W_EXIT;
  }
  return V46_WORD(V46_ACT_IDLE, 0u, 0u, 0u);
#else
  (void)v54_x;
#if V62_G2_SWZ
  const bool v62_swz = (n2c >= 4u);
  const unsigned v62_xcc = v55_xcc() & 3u;
  // V62: ready GEMM2 head of XCC queue xx of pass q (V46_NONE if empty / head not ready)
  auto v62_g2_take = [&](unsigned q, unsigned xx) -> unsigned {
    const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q];
    const unsigned cnt_x = (n2c + 3u - xx) / 4u;
    unsigned* cur = pf + V62_G2CX + q * 4u + xx;
    unsigned c = __hip_atomic_load(cur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    while (c < T * cnt_x) {
#if V62_NT0
      unsigned j_ = 0u;
      const unsigned mtl = v62_nt0_map(c, cnt_x, T, j_);
#else
      const unsigned mtl = c / cnt_x;
      const unsigned j_ = c - mtl * cnt_x;
#endif
      if (!v46_s1_ready(s1d, B + mtl, n1b)) break;
      if (__hip_atomic_compare_exchange_strong(cur, &c, c + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
        return V46_WORD(V46_ACT_G2, q, j_ * 4u + xx, B + mtl);
    }
    return V46_NONE;
  };
#endif
  // (1) ready GEMM2 heads, pass order 1, 2, 0, 3
  #pragma unroll
  for (unsigned j = 0u; j < V46_NP; ++j) {
    const unsigned q = v46_g2_pass(j);
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
    if (q != 0u && !v46_is_remote(q) && !local_full) continue;   // q3 waits for the full local plan
    if (q == 0u && !q0_ready) continue;
#if V62_G2_SWZ
    if (v62_swz) {
      const unsigned w = v62_g2_take(q, v62_xcc);
      if (w != V46_NONE) return w;
#if V62_G2_SWZ >= 2
      // SWZ=2: work-conserving -- a ready head of another XCC queue is taken before any GEMM1 item
      #pragma unroll
      for (unsigned dx = 1u; dx < 4u; ++dx) {
        const unsigned w2 = v62_g2_take(q, (v62_xcc + dx) & 3u);
        if (w2 != V46_NONE) return w2;
      }
#endif
      continue;
    }
#endif
    const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q], bound = T * n2c;
    unsigned* cur = pf + V46_G2C + q;
    unsigned c = __hip_atomic_load(cur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    while (c < bound) {
#if V62_NT0
      unsigned j_ = 0u;
      const unsigned mtl = v62_nt0_map(c, n2c, T, j_);
      if (!v46_s1_ready(s1d, B + mtl, n1b)) break;
      if (__hip_atomic_compare_exchange_strong(cur, &c, c + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
        return V46_WORD(V46_ACT_G2, q, j_, B + mtl);
#else
      const unsigned mtl = c / n2c;
      if (!v46_s1_ready(s1d, B + mtl, n1b)) break;
      if (__hip_atomic_compare_exchange_strong(cur, &c, c + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
        return V46_WORD(V46_ACT_G2, q, c - mtl * n2c, B + mtl);
#endif
    }
  }
  // (2) GEMM1 items: q0 head bootstrap, then q1/q2 before remaining local tiles.
  #pragma unroll
  for (unsigned j = 0u; j < V46_NP; ++j) {
    const unsigned q = v46_g1_pass(j, remote_g1_first);
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
    if ((seen & (4u << q)) == 0u) continue;
    if (q == 0u && !q0_ready) continue;            // head fast path only needs q0
    if (q == 3u && !local_full) continue;          // q3 waits for the full local plan
    if (q == 3u && (seen & 4u) == 0u) continue;   // own-rank staging is certified by PULL_RDY[0]
    const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q];
#if V55_COALESCE_SCHED && (V46_NSUB == 1)
#if V55_SWIZZLE
    const unsigned n1g_per_mt = n1b / (V55_N1G);
    if (n1g_per_mt >= 4u) {
      const unsigned xcc = v55_xcc();
      const unsigned cnt_x = (n1g_per_mt + 3u - xcc) / 4u;
      const unsigned c = v46_claim_lt(pf + V55_G1CX + q * 4u + xcc, T * cnt_x);
      if (c != V46_NONE) {
#if V62_NT0
        unsigned j_ = 0u;
        const unsigned mtl = v62_nt0_map(c, cnt_x, T, j_);
        const unsigned n1grp = j_ * 4u + xcc;
#else
        const unsigned mtl = c / cnt_x;
        const unsigned n1grp = (c % cnt_x) * 4u + xcc;
#endif
        if (mtl == 0u && trace) v70_trace_min(&trace[1308u + q], __builtin_amdgcn_s_memrealtime());
        return V46_WORD(V46_ACT_G1, q, n1grp * (V55_N1G), B + mtl);
      }
      continue;
    }
#endif
    {
      const unsigned c = v55_claim_grp(pf + V46_G1C + q, T * n1b, (V55_N1G));
      if (c != V46_NONE) {
        if (c == 0u && trace) trace[1308u + q] = __builtin_amdgcn_s_memrealtime();
        const unsigned mtl = c / n1b;
        return V46_WORD(V46_ACT_G1, q, c - mtl * n1b, B + mtl);
      }
    }
#else
    const unsigned c = v46_claim_lt(pf + V46_G1C + q, T * n1b * (V46_NSUB));
    if (c != V46_NONE) {
      if (c == 0u && trace) trace[1308u + q] = __builtin_amdgcn_s_memrealtime();
      const unsigned g1pm = n1b * (V46_NSUB);
      const unsigned mtl = c / g1pm, rem = c - mtl * g1pm;
      const unsigned s1s = rem / n1b;
      return V46_WORD(V46_ACT_G1, q, rem - s1s * n1b, ((s1s) << 17u) | (B + mtl));
    }
#endif
  }
#if V62_G2_SWZ
  // (3s) V62: steal ready GEMM2 heads from the other XCC queues (only when no GEMM1 item was available)
  if (v62_swz) {
    #pragma unroll
    for (unsigned j = 0u; j < V46_NP; ++j) {
      const unsigned q = v46_g2_pass(j);
      if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
      #pragma unroll
      for (unsigned dx = 1u; dx < 4u; ++dx) {
        const unsigned w = v62_g2_take(q, (v62_xcc + dx) & 3u);
        if (w != V46_NONE) return w;
      }
    }
  }
#endif
#if V62_G1_STEAL && V55_COALESCE_SCHED && V55_SWIZZLE && (V46_NSUB == 1)
  // (3g) V62: steal GEMM1 items from the other XCC queues (pass tail only: own queue (2) was empty)
  {
    const unsigned n1g_per_mt = n1b / (V55_N1G);
    if (n1g_per_mt >= 4u) {
      const unsigned xcc0 = v55_xcc();
      #pragma unroll
      for (unsigned j = 0u; j < V46_NP; ++j) {
        const unsigned q = v46_g1_pass(j, remote_g1_first);
        if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
        if ((seen & (4u << q)) == 0u) continue;
        if (q == 0u && !q0_ready) continue;            // head fast path: q0 ids/weights must be scattered
        if (q == 3u && !local_full) continue;          // q3 waits for the full local plan
        if (q == 3u && (seen & 4u) == 0u) continue;
        const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q];
        #pragma unroll
        for (unsigned dx = 1u; dx < 4u; ++dx) {
          const unsigned xx = (xcc0 + dx) & 3u;
          const unsigned cnt_x = (n1g_per_mt + 3u - xx) / 4u;
          const unsigned c = v46_claim_lt(pf + V55_G1CX + q * 4u + xx, T * cnt_x);
          if (c != V46_NONE) {
#if V62_NT0
            unsigned j_ = 0u;
            const unsigned mtl = v62_nt0_map(c, cnt_x, T, j_);
            const unsigned n1grp = j_ * 4u + xx;
#else
            const unsigned mtl = c / cnt_x;
            const unsigned n1grp = (c % cnt_x) * 4u + xx;
#endif
            return V46_WORD(V46_ACT_G1, q, n1grp * (V55_N1G), B + mtl);
          }
        }
      }
    }
  }
#endif
#if !V55_NOSPEC_G2
  // (3) GEMM2 of an M tile whose GEMM1 items are all claimed, pass order 1, 2, 0, 3
  #pragma unroll
  for (unsigned j = 0u; j < V46_NP; ++j) {
    const unsigned q = v46_g2_pass(j);
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
    if (q != 0u && !v46_is_remote(q) && !local_full) continue;
    if (q == 0u && !q0_ready) continue;
    const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q], bound = T * n2c;
    unsigned* cur = pf + V46_G2C + q;
    unsigned c = __hip_atomic_load(cur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    while (c < bound) {
      const unsigned mtl = c / n2c;
      if (__hip_atomic_load(pf + V46_G1C + q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (mtl + 1u) * n1b * (V46_NSUB)) break;
      if (__hip_atomic_compare_exchange_strong(cur, &c, c + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
        return V46_WORD(V46_ACT_G2, q, c - mtl * n2c, B + mtl);
    }
  }
#endif
  // (4) exit test
  if (local_full && (seen & 2u)) {
    bool all = true;
#if V62_G2_SWZ
    if (v62_swz) {
      #pragma unroll
      for (unsigned q = 0u; q < V46_NP; ++q) {
        #pragma unroll
        for (unsigned xx = 0u; xx < 4u; ++xx)
          if (__hip_atomic_load(pf + V62_G2CX + q * 4u + xx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) <
              pf[V46_T + q] * ((n2c + 3u - xx) / 4u)) all = false;
      }
    } else {
      #pragma unroll
      for (unsigned q = 0u; q < V46_NP; ++q)
        if (__hip_atomic_load(pf + V46_G2C + q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < pf[V46_T + q] * n2c) all = false;
    }
#else
    #pragma unroll
    for (unsigned q = 0u; q < V46_NP; ++q)
      if (__hip_atomic_load(pf + V46_G2C + q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < pf[V46_T + q] * n2c) all = false;
#endif
    if (all) return V46_W_EXIT;
  }
  return V46_WORD(V46_ACT_IDLE, 0u, 0u, 0u);
#endif
}

// -------------------------------------------------------------------------------------------------------------
// Iron Rule 1 prelude: the W00..W03 half of each pure-compute CTA claims one
// q0 GEMM1 item through the production two-level cursor and executes it with
// the validated four-wave body.  The other half is still draining the bounded
// remote head.  Both halves then rendezvous at the unified s_barrier body.
// -------------------------------------------------------------------------------------------------------------
#if V70_R2X && V70_R2X_PRE4
#include "v70_r2_8wave.hpp"
// V70_R2X_PRE4 prelude: identical claim / accounting to the four-wave prelude below, but (1) the GEMM1 item runs
// on the fused R2X group-0 body (LDS [0, 26368)) with the W00..W03 software barrier, and (2) W00..W03 keep
// claiming q0 items until this CTA's four head-drain waves have published (LDS counter V70_PRE4_FLAG == 4).
__device__ __forceinline__ bool v70_head_g1_prelude(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena FI_CTX_PARAM) {
  const unsigned tid = threadIdx.x & 255u;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(
      reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65472u));
  volatile unsigned* flag = reinterpret_cast<volatile unsigned*>(g_v1262_phase_arena + V70_PRE4_FLAG);
  const unsigned n1b = env.ck.ck_intermediate / 128u;
  bool q0_ok = true;
  for (unsigned it = 0u;; ++it) {
    if (tid == 0u) {
      unsigned word = V46_NONE;
      if (it == 0u)
        q0_ok = v46_wait_ge(pf + V46_Q0_RDY, 1u, env, x, 977u) && v41_ld_agent_u32(pf + V46_Q0_RDY) == 1u;
#if V70_R2X_MERGE
      // Large launch: leave q0 to the all-pass merged queue (mode 2).  Small launch: pin mode 3 before the
      // first per-pass q0 claim so the merged queue never contains q0.
      if (it == 0u && q0_ok) {
        if (!v70m_small(env, pf)) q0_ok = false;
        else {
          unsigned e0 = 0u;
          __hip_atomic_compare_exchange_strong(pf + V70M_MODE, &e0, 3u, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
          if (e0 != 0u && e0 != 3u) q0_ok = false;
        }
      }
#endif
      if (q0_ok && (it == 0u || *flag < 4u)) {
        const unsigned LE = v70_claim_le(env, pf);
        const unsigned T = pf[V46_T], B = pf[V46_BASE];
        if ((n1b % (V55_N1G)) != 0u) {
          v983_error(reinterpret_cast<unsigned*>(env.base.error), 978u);
        } else {
          const unsigned N = T * (n1b / (V55_N1G));
          const unsigned xx = v55_xcc() & 3u;
          unsigned g = 0u;
          if (v70_claim_work_xcd(pf + V55_G1CX + xx, v70_xcd_slots(N, xx), xx, N, g)) {
            unsigned mt = 0u, n1 = 0u;
            v70_map_g1(pf + V46_OFF(0u), LE, B + T, n1b, g, mt, n1);
            if (mt == B && trace) v70_trace_min(&trace[1308u], __builtin_amdgcn_s_memrealtime());
            word = V46_WORD(V46_ACT_G1, 0u, n1, mt);
          }
        }
      }
      sh[0] = word;
    }
    weave_4wave_barrier_65440(arena);
    const unsigned w = __builtin_amdgcn_readfirstlane(sh[0]);
    if (w == V46_NONE) break;
    const unsigned sub = (w >> 20u) & 0xffu;
    const unsigned mt = w & 0x1ffffu;
    const V964DispatchExt& e = env.base;
    const V968CkExt& ck = env.ck;
    const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(ck.ck_a + 64u);
    const WeaveMTileDesc d_raw = tiles[mt];
    const unsigned expert_id = __builtin_amdgcn_readfirstlane(d_raw.expert_id);
    const unsigned valid_rows = __builtin_amdgcn_readfirstlane(d_raw.valid_rows);
    const unsigned row_start = __builtin_amdgcn_readfirstlane(d_raw.row_start);
    const unsigned run = ck.ck_route_rows;
    unsigned rk = __builtin_amdgcn_readfirstlane(e.rank), cap = __builtin_amdgcn_readfirstlane(e.capacity);
    unsigned pbw = __builtin_amdgcn_readfirstlane(e.payload_bytes), sgw = __builtin_amdgcn_readfirstlane(e.scale_groups);
    asm volatile("" : "+s"(rk), "+s"(cap), "+s"(pbw), "+s"(sgw));
    const uint64_t pay_u = v46_rfl_u64(reinterpret_cast<const uint64_t*>(e.peer_payload_table)[rk]);
    const uint64_t scl_u = v46_rfl_u64(reinterpret_cast<const uint64_t*>(e.peer_scale_table)[rk]);
    const unsigned rc = rk * cap;
    const unsigned char* recv_payload = reinterpret_cast<const unsigned char*>(pay_u - static_cast<uint64_t>(rc) * pbw);
    const float* recv_scale = reinterpret_cast<const float*>(scl_u - static_cast<uint64_t>(rc) * sgw * 4ull);
    const int32_t* ids = reinterpret_cast<const int32_t*>(ck.ck_sorted_token_ids);
    unsigned g1inc = 0u;
    if (valid_rows != 0u) {
      // R4: wave 0 fills the LDS row table, the four-wave barrier publishes it to W00..W03.
      r2x::rtab_fill(tid, ids, nullptr, (int)valid_rows, [&](int r) { return (int)row_start + r; });
      weave_4wave_barrier_65440(arena);
      #pragma clang loop unroll(disable)
      for (unsigned j = 0u; j < (V55_N1G) && sub + j < n1b; ++j) {
        const unsigned g1_aux = (mt << 16u) | ((sub + j) & 0xffffu);
        FI_BG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
        r2x::unit<1, 1>(g_v1262_phase_arena, (int)tid, (int)valid_rows, (int)(sub + j),
                        (int)expert_id, recv_payload, reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
                        reinterpret_cast<const float*>(ck.ck_b_scale), reinterpret_cast<uint8_t*>(ck.ck_quant_output),
                        reinterpret_cast<float*>(ck.ck_quant_scale), nullptr, ck.ck_hidden, ck.ck_intermediate, run,
                        nullptr, nullptr, nullptr, 0u, 0u);
        FI_EG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
        ++g1inc;
      }
    }
    if (tid == 0u) {
      const unsigned done_inc = (valid_rows != 0u) ? g1inc : ((sub == 0u) ? n1b : 0u);
      if (done_inc)
        __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.stage1_tile_done) + mt * (V46_NSUB), done_inc,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      const unsigned d = __hip_atomic_fetch_add(pf + V46_G1D, g1inc, __ATOMIC_RELAXED,
                                                __HIP_MEMORY_SCOPE_AGENT) + g1inc;
      if (d == pf[V46_T] * n1b && trace) trace[1312u] = __builtin_amdgcn_s_memrealtime();
    }
    weave_4wave_barrier_65440(arena);
  }
  return true;
}
// W04..W07: one LDS increment per wave once its head drain has been published.
__device__ __forceinline__ void v70_pre4_signal() {
  if ((threadIdx.x & 63u) == 0u)
    __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(g_v1262_phase_arena + V70_PRE4_FLAG), 1u,
                           __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP);
}
// Kernel entry, all eight waves present: reset the PRE4 counter and both four-wave barrier counter sets.
__device__ __forceinline__ void v70_pre4_reset() {
  const unsigned t = threadIdx.x;
  if (t < 8u) reinterpret_cast<volatile unsigned*>(g_v1262_phase_arena + 65440u)[t] = 0u;
  if (t == 8u) *reinterpret_cast<volatile unsigned*>(g_v1262_phase_arena + V70_PRE4_FLAG) = 0u;
  __syncthreads();
}
#else
__device__ __forceinline__ void v70_pre4_signal() {}
__device__ __forceinline__ void v70_pre4_reset() {}
__device__ __forceinline__ bool v70_head_g1_prelude(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena FI_CTX_PARAM) {
  const unsigned tid = threadIdx.x & 255u;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(
      reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65472u));
  if (tid == 0u) {
    unsigned word = V46_NONE;
    const bool q0_ok = v46_wait_ge(pf + V46_Q0_RDY, 1u, env, x, 977u) &&
                       v41_ld_agent_u32(pf + V46_Q0_RDY) == 1u;
    if (q0_ok) {
      const unsigned LE = v70_claim_le(env, pf);
      const unsigned T = pf[V46_T], B = pf[V46_BASE];
      const unsigned n1b = env.ck.ck_intermediate / 128u;
      if ((n1b % (V55_N1G)) != 0u) {
        v983_error(reinterpret_cast<unsigned*>(env.base.error), 978u);
      } else {
        const unsigned nn1 = n1b / (V55_N1G);
        const unsigned N = T * nn1;
        const unsigned xx = v55_xcc() & 3u;
        const unsigned bound = v70_xcd_slots(N, xx);
        unsigned* cur = pf + V55_G1CX + xx;
        unsigned g = 0u;
        if (v70_claim_work_xcd(cur, bound, xx, N, g)) {
          unsigned mt = 0u, n1 = 0u;
          v70_map_g1(pf + V46_OFF(0u), LE, B + T, n1b, g, mt, n1);
          if (mt == B && trace) v70_trace_min(&trace[1308u], __builtin_amdgcn_s_memrealtime());
          word = V46_WORD(V46_ACT_G1, 0u, n1, mt);
        }
      }
    }
    sh[0] = word;
  }
  weave_4wave_barrier_65440(arena);
  const unsigned w = __builtin_amdgcn_readfirstlane(sh[0]);
  if (w != V46_NONE) {
    const unsigned sub = (w >> 20u) & 0xffu;
    const unsigned mt = w & 0x1ffffu;
    const unsigned n1b = env.ck.ck_intermediate / 128u;
    const V964DispatchExt& e = env.base;
    const V968CkExt& ck = env.ck;
    const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(ck.ck_a + 64u);
    const WeaveMTileDesc d_raw = tiles[mt];
    const unsigned expert_id = __builtin_amdgcn_readfirstlane(d_raw.expert_id);
    const unsigned valid_rows = __builtin_amdgcn_readfirstlane(d_raw.valid_rows);
    const unsigned row_start = __builtin_amdgcn_readfirstlane(d_raw.row_start);
    const unsigned num_tokens = e.world * e.capacity;
    const unsigned run = ck.ck_route_rows;
    const unsigned rk = __builtin_amdgcn_readfirstlane(e.rank);
    const unsigned cap = __builtin_amdgcn_readfirstlane(e.capacity);
    const unsigned pbw = __builtin_amdgcn_readfirstlane(e.payload_bytes);
    const unsigned sgw = __builtin_amdgcn_readfirstlane(e.scale_groups);
    const uint64_t pay_e = reinterpret_cast<const uint64_t*>(e.peer_payload_table)[rk];
    const uint64_t scl_e = reinterpret_cast<const uint64_t*>(e.peer_scale_table)[rk];
    const uint64_t pay_u = v46_rfl_u64(pay_e), scl_u = v46_rfl_u64(scl_e);
    const unsigned rc = rk * cap;
    const unsigned char* recv_payload = reinterpret_cast<const unsigned char*>(pay_u - static_cast<uint64_t>(rc) * pbw);
    const float* recv_scale = reinterpret_cast<const float*>(scl_u - static_cast<uint64_t>(rc) * sgw * 4ull);
    const int32_t* ids = reinterpret_cast<const int32_t*>(ck.ck_sorted_token_ids) + row_start;
    unsigned g1inc = 0u;
    if (valid_rows != 0u) {
      for (unsigned j = 0u; j < (V55_N1G) && sub + j < n1b; ++j) {
        // Validated deep-prefetch 400T wave body: unlike the old zerob schedule,
        // its K-loop has no in-loop vmcnt(0) drain.
        const unsigned g1_aux = (mt << 16u) | ((sub + j) & 0xffffu);
        FI_BG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
        if constexpr ((WEAVE_BLOCK_M_L1) == 64u) {
          weave_s1_400t_2x2<64u>(ids, (int32_t)expert_id, recv_payload,
              reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
              reinterpret_cast<const float*>(ck.ck_b_scale),
              reinterpret_cast<uint8_t*>(ck.ck_quant_output),
              reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, sub + j,
              num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
        } else {
          weave_s1_400t_2x2<128u>(ids, (int32_t)expert_id, recv_payload,
              reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
              reinterpret_cast<const float*>(ck.ck_b_scale),
              reinterpret_cast<uint8_t*>(ck.ck_quant_output),
              reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, sub + j,
              num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
        }
        FI_EG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
        ++g1inc;
      }
    }
    if (tid == 0u) {
      // This prelude bypasses v46_gemm1_item_n's wrapper.  Publish the exact
      // stage-1 tile completion GEMM2 waits on, including the zero-row case.
      const unsigned done_inc = (valid_rows != 0u) ? g1inc : ((sub == 0u) ? n1b : 0u);
      if (done_inc)
        __hip_atomic_fetch_add(
            reinterpret_cast<unsigned*>(o.stage1_tile_done) + mt * (V46_NSUB), done_inc,
            __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
      const unsigned d = __hip_atomic_fetch_add(pf + V46_G1D, g1inc, __ATOMIC_RELAXED,
                                                __HIP_MEMORY_SCOPE_AGENT) + g1inc;
      if (d == pf[V46_T] * n1b && trace) trace[1312u] = __builtin_amdgcn_s_memrealtime();
    }
  }
  weave_4wave_barrier_65440(arena);
  return w != V46_WORD(V46_ACT_END, 1u, 0u, 0u);
}
#endif  // V70_R2X_PRE4

__device__ __forceinline__ bool v46_atrex(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned char* arena FI_CTX_PARAM) {
  const unsigned tid = threadIdx.x & 255u;
  const unsigned grp_sh = v46_grp_sh();
  volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(
      reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65472u + grp_sh));
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  const unsigned n1b = env.ck.ck_intermediate / 128u;
  const unsigned n2b = o.hidden / 128u;
  const unsigned n2g = v46_n2g(n2b);
#if V55_COALESCE_SCHED
  const unsigned n2c = (n2g == 8u) ? (n2b >> 3u) : ((n2g == 4u) ? (n2b >> 2u) : ((n2g == 2u) ? (n2b >> 1u) : n2b));
#else
  const unsigned n2c = (n2g == 4u) ? (n2b >> 2u) : ((n2g == 2u) ? (n2b >> 1u) : n2b);
#endif
  // opus55 v46 E9: the readiness bits `seen` are private to tid0 and live in LDS (65488 + grp_sh, the slot of the
  // unused rdy-wait cache), so no divergent (tid0-only) loop-carried VGPR crosses the GEMM1/GEMM2 bodies
  volatile WeaveBk64LocalU32* seen_sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(
      reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65488u + grp_sh));
  if (tid == 0u) seen_sh[0] = 0u;
  unsigned last_acq_mt = 0xffffffffu;
  for (;;) {
    weave_4wave_barrier_65440(arena);
    if (tid == 0u) {
      unsigned seen = seen_sh[0];
      unsigned w = v46_decide(env, o, seen, n1b, n2c, v54_xcd());
      seen_sh[0] = seen;
#if V70_R2X_EARLY_EXIT && V70_R2X_COMPACT
      if (w == V46_W_CLAIMED &&
          __hip_atomic_load(pf + V70M_G2D, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >=
          __hip_atomic_load(pf + V70M_G2T, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
        w = V46_W_EXIT;
#endif
      if ((w >> 30u) == V46_ACT_IDLE && trace &&
          __builtin_amdgcn_s_memrealtime() - trace[16u + blockIdx.x * 16u] > 4ull * x.timeout_ticks) {
        v983_error((unsigned*)env.base.error, 970u);
        w = V46_W_ERR;
      }
      sh[0] = w;
    }
    weave_4wave_barrier_65440(arena);
    const unsigned w = __builtin_amdgcn_readfirstlane(sh[0]);
    const unsigned act = w >> 30u;
    if (act == V46_ACT_END) return ((w >> 28u) & 3u) == 0u;
    if (act == V46_ACT_IDLE) { __builtin_amdgcn_s_sleep(2); continue; }
    const unsigned q = (w >> 28u) & 3u, sub = (w >> 20u) & 0xffu;
    const unsigned mt = w & 0x1ffffu, s1s = (w >> 17u) & 7u;
    if (act == V46_ACT_G1) {
      v46_gemm1_item(env, o, arena, mt, s1s, sub FI_CTX_ARG);
      if (tid == 0u) {
#if V55_COALESCE_SCHED
        unsigned g1inc = (V55_N1G); if (g1inc > n1b - sub) g1inc = n1b - sub;
#else
        const unsigned g1inc = 1u;
#endif
        const unsigned d = __hip_atomic_fetch_add(pf + V46_G1D + q, g1inc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + g1inc;
        if (d == pf[V46_T + q] * n1b * (V46_NSUB) && trace) trace[1312u + q] = __builtin_amdgcn_s_memrealtime();
      }
    } else {
      if (!v46_gemm2_item(env, o, x, arena, mt, sub, n2g, last_acq_mt, (unsigned*)x.compute_done,
                          q, v54_xcd(), V54_TARG(pf, q) FI_CTX_ARG)) return false;
      if (tid == 0u) {
        const unsigned d = __hip_atomic_fetch_add(pf + V46_G2D + q, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
        if (d == pf[V46_T + q] * n2c && trace) trace[1316u + q] = __builtin_amdgcn_s_memrealtime();
      }
    }
  }
}

// -------------------------------------------------------------------------------------------------------------
// s13 R3: v46-private copy of ep16_bucket_v41_flag with RELEASE (system scope) arrival flags.  The caller has
// issued s_waitcnt vmcnt(0) and ONE __threadfence_system() after all data stores of the batch.
// -------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void v46_bucket_flag(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned src, unsigned t) {
  const auto& e = env.base;
  const unsigned key = src * e.capacity + t;
  if ((src >> 3u) == (e.rank >> 3u)) {
    auto* ph = reinterpret_cast<unsigned*>(reinterpret_cast<const uint64_t*>(o.peer_home_arrival_table)[src]);
    __hip_atomic_store(ph + t, 1u, (V70_R6_SCAT_RLX) ? __ATOMIC_RELAXED : __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  } else {
    auto* na = reinterpret_cast<unsigned*>(reinterpret_cast<const uint64_t*>(x.node_arrival_table)[src & 7u]);
    __hip_atomic_fetch_add(na + t, 1u, (V70_R6_SCAT_RLX) ? __ATOMIC_RELAXED : __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  ep16_st_agent_relaxed(reinterpret_cast<unsigned*>(o.bucket_sent) + key, 1u);
}

// -------------------------------------------------------------------------------------------------------------
// 4-pass streaming scatter (generalizes v45_sweep_mb).  Block = (64-token chunk c, in-node source so), chunk-major.
// A token is claimed by CAS bucket_done: nexp -> 0xffffffff, i.e. only after all its expert rows left GEMM2.
// st bits: 0 = pass plan seen, 1 = local mask empty, 2 = pass globally scattered (SDONE)
// -------------------------------------------------------------------------------------------------------------
#ifndef V46_STREAM_CLAIM
#define V46_STREAM_CLAIM (WEAVE_READY_GROUP_SIZE)
#endif
__device__ __forceinline__ unsigned v46_sweep_pass(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned p,
    uint64_t& m0, uint64_t& m1, unsigned& pos, unsigned& st, unsigned worker, unsigned& seed FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  if (st & 4u) return 0u;
  if (v42_ld_agent_rlx(pf + V46_SDONE + p) != 0u) { st |= 4u; return 0u; }
  unsigned tb, te;
  v46_pass_range(p, NT, tb, te);
  const unsigned nblk = __builtin_amdgcn_readfirstlane(((te - tb + 63u) >> 6u) * 8u);
  if ((st & 1u) == 0u) {
    const unsigned gate = (p == 0u) ? V46_Q0_RDY : (v46_is_remote(p) ? V46_P1_PLAN : V46_LOCAL_RDY);
    if (v42_ld_agent_rlx(pf + gate) == 0u) return 8u;
    v42_acq_agent();
    m0 = v44_lowmask(nblk < 64u ? nblk : 64u);
    m1 = v44_lowmask(nblk > 64u ? nblk - 64u : 0u);
    pos = nblk ? ((worker + (WEAVE_SCATTER_OWNER)) % nblk) : 0u;
    st |= 1u;
  }
  if (!(m0 | m1)) { st |= 2u; return 8u; }
  const unsigned blk = __builtin_amdgcn_readfirstlane(v44_next_blk(m0, m1, nblk, pos));
  const unsigned c = blk >> 3u, so = blk & 7u;
  const unsigned node = v46_is_remote(p) ? ((e.rank >> 3u) ^ 1u) : (e.rank >> 3u);
  const unsigned src = node * 8u + so;
  const unsigned t = tb + c * 64u + lane;
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
  unsigned* bdp = reinterpret_cast<unsigned*>(o.bucket_done) + (src * e.capacity + (t < te ? t : tb));
  unsigned nexp = 0u, bd = 0xffffffffu;
  if (t < te) {
    nexp = local_nexp[static_cast<uint64_t>(src) * NT + t];
    if (nexp != 0u) bd = ep16_load_agent_relaxed(bdp);
  }
  const uint64_t open = v42_ballot_eq((nexp != 0u && bd != 0xffffffffu) ? 1u : 0u, 1u);
  if (open == 0ull) { v44_clear_blk(m0, m1, blk); pos = blk + 1u; return 8u; }
  const bool cand = (nexp != 0u && bd == nexp);
  const uint64_t cm = v42_ballot_eq(cand ? 1u : 0u, 1u);
  if (cm == 0ull) { pos = blk + 1u; return 8u; }
  const unsigned pc = __builtin_amdgcn_readfirstlane((unsigned)__builtin_popcountll(cm));
  const unsigned rot = __builtin_amdgcn_readfirstlane(seed % pc);
  seed += 1u;
  unsigned won = 0u;
  if (cand) {
    const unsigned rk = v44_mbcnt(cm);
    if (((rk + pc - rot) % pc) < V46_STREAM_CLAIM) {
      unsigned exp_bd = nexp;
      // s13 R1: ACQUIRE on success pairs with the RELEASE bucket_done increments of v46_gemm2_item
      won = __hip_atomic_compare_exchange_strong(bdp, &exp_bd, 0xffffffffu, V46_CAS_ORDER, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT) ? 1u : 0u;
    }
  }
  const uint64_t mr = v42_ballot_eq(won, 1u);
  if (mr == 0ull) { pos = blk; return 8u; }
  v42_acq_agent();
  uint64_t m2 = mr;
  while (m2) {
    const unsigned l = (unsigned)__builtin_ctzll(m2);
    m2 &= (m2 - 1ull);
    const unsigned t2 = tb + c * 64u + l, n2 = __builtin_amdgcn_readlane(nexp, l);   // opus55 v46 E10
    FI_BG(V1988_FI_CORE, 9u, src * e.capacity + t2);
    V45_BUCKET_DATA(env, o, x, src, t2, n2);
    FI_EG(V1988_FI_CORE, 9u, src * e.capacity + t2);
  }
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __syncwarp();
#if V46_R3 >= 2
  __threadfence_system();   // s13 R3: once per batch, before any arrival flag of the batch
#endif
#if V46_R3 >= 1
  if (won) v46_bucket_flag(env, o, x, src, t);
#else
  if (won) ep16_bucket_v41_flag(env, o, x, src, t);
#endif
  __syncwarp();
  if (lane == 0u) {
    const unsigned gn = (unsigned)__builtin_popcountll(mr);
    const unsigned prev = __hip_atomic_fetch_sub(pf + V46_REM + p, gn, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (prev == gn) {
      v41_st_agent_u32(pf + V46_SDONE + p, 1u);
      auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
      if (trace) trace[1320u + p] = __builtin_amdgcn_s_memrealtime();
    }
  }
  __syncwarp();
  if ((open & ~mr) == 0ull) { v44_clear_blk(m0, m1, blk); pos = blk + 1u; }
  else pos = blk;
  return 1u;
}

// =============================================================================================================
// V70 Iron Rule 3: zero-CAS Combine.  NodeReduce and Home ownership is a deterministic stripe over the
// 636 combine-capable waves (worker-4).  Each token has exactly one owner, so the per-token atomic_fetch_or
// claim word is unnecessary.  Per-peer arrival flags remain the readiness contract.  Scatter ownership is
// likewise deterministic, but only the 92 communication workers that are not GDA engines may enter it.
// =============================================================================================================
#ifndef V70_NSCAT_RAW
#define V70_NSCAT_MAX ((V46_WCOMM) - 8u)
#define V70_NSCAT_SW (((WEAVE_SCATTER_WARPS) > 0 && (WEAVE_SCATTER_WARPS) >= 8 && (V70_NSCAT_MAX) >= 8u) ? ((V70_NSCAT_MAX) / 2u) : (V70_NSCAT_MAX))
#define V70_NSCAT_REQ (((WEAVE_SCATTER_ACTIVE_SMS) > 0) ? ((unsigned)(WEAVE_SCATTER_ACTIVE_SMS) * 4u) : (V70_NSCAT_SW))
#define V70_NSCAT_RAW (((V70_NSCAT_REQ) > (V70_NSCAT_MAX)) ? (V70_NSCAT_MAX) : (((V70_NSCAT_REQ) < 4u) ? 4u : (V70_NSCAT_REQ)))
#endif
#ifdef V70_NSCAT
#undef V70_NSCAT
#endif
#define V70_NSCAT (V70_NSCAT_RAW)
static_assert(V70_NSCAT > 0u, "V70 needs communication Scatter owners");

__device__ __forceinline__ unsigned v70_scat_rank(unsigned worker) {
  // Dense Scatter rank over cw 4..CW_POOL-1 (20 owners: RX 4..7 + planner 8..23),
  // the NNR reduce pool (10 owners), and cw CW_POOL+NNR..WCOMM-1 (62 pull owners).
  // cw 0..3 are RDMA SQ/CQ engines and never issue Scatter.  Opening the NNR pool
  // balances the late Scatter drain across every communication wave; compute and
  // epilogue waves remain strictly excluded.
  if (worker < 8u || worker >= (V46_WCOMM)) return (V70_NSCAT);
  return worker - 8u;
}

__device__ __forceinline__ unsigned v70_visit_nr_owned(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned t FI_CTX_PARAM) {
  const unsigned NT = x.node_tokens;
  if (t >= NT) return 0u;
  auto* nrdy = reinterpret_cast<unsigned*>(x.node_ready);
#if V70_R8_2WAVE
  if (ep16_load_agent_relaxed(nrdy + t) != 0u) return 0u;
#else
  if (ep16_load_relaxed(nrdy + t) != 0u) return 0u;
#endif
  if (v42_ld_sys_rlx(reinterpret_cast<const unsigned*>(x.node_arrival) + t) <
      reinterpret_cast<const unsigned*>(x.node_expected)[t]) return 0u;
#if V70_E3_QUIET_HT
  asm volatile("buffer_inv sc0\ns_waitcnt vmcnt(0)\n" ::: "memory");
#else
  v42_acq_sys();
#endif
  FI_BG(V1988_FI_CORE, 41u, t);
  V45_NODE_REDUCE(env, o, x, t);
  FI_EG(V1988_FI_CORE, 41u, t);
  if (unsigned long long* tr_f = v70_fresh_sptr(reinterpret_cast<unsigned long long*>(o.trace))) {
    auto* tr = tr_f;
    const unsigned long long now = __builtin_amdgcn_s_memrealtime();
    if (tr[1340u] == 0ull) tr[1340u] = now;
    tr[1341u] = now;
  }
  return 1u;
}

__device__ __forceinline__ unsigned v70_visit_home_owned(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned t FI_CTX_PARAM) {
  const auto& e = env.base;
  if (t >= e.t_local) return 0u;
  auto* hdone = reinterpret_cast<unsigned*>(o.home_done);
  if (ep16_load_agent_relaxed(hdone + t) != 0u) return 0u;
  if (!v44_home_ready_rlx(env, o, x, t)) return 0u;
#if V70_E3_QUIET_HT
  v42_acq_sys();
#else
  v42_acq_agent();
#endif
  FI_BG(V1988_FI_CORE, 12u, t);
  V44_HOME(env, o, x, t);
  FI_EG(V1988_FI_CORE, 12u, t);
  if (unsigned long long* tr_f = v70_fresh_sptr(reinterpret_cast<unsigned long long*>(o.trace))) {
    auto* tr = tr_f;
    const unsigned long long now = __builtin_amdgcn_s_memrealtime();
    if (tr[1342u] == 0ull) tr[1342u] = now;
    tr[1343u] = now;
  }
  return 1u;
}

// One Scatter owner scans its deterministic slots for one pass.  Inter-node passes are called first by
// the caller.  Ownership is exclusive, so bucket_done is a readiness value instead of a CAS mutex.
__device__ __forceinline__ unsigned v70_scatter_owned_pass(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned p, unsigned rank, unsigned& pos, unsigned steal_n FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned m = e.rank;
  const unsigned node = v46_is_remote(p) ? ((m >> 3u) ^ 1u) : (m >> 3u);
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  if (v42_ld_agent_rlx(pf + V46_SDONE + p) != 0u) return 0u;
  unsigned tb, te;
  v46_pass_range(p, NT, tb, te);
  const unsigned span = (te > tb) ? (te - tb) : 0u;
  // H9 2-Node 16-GPU fix: cover all 8 source ranks (including m and remote partner m^8) so SDONE always retires!
  const unsigned total = 8u * span;
  if (total == 0u) return 0u;
  const unsigned sstride = steal_n ? steal_n : (V70_NSCAT);
  const unsigned me = steal_n ? rank : (rank % (V70_NSCAT));
  const unsigned limit = (V70_READY_GRP_EFF);
  if (pos < me || pos > total) pos = me;
  unsigned attempts = 0u, did = 0u, slot = pos;
  for (; slot < total && attempts < limit; slot += sstride) {
    ++attempts;
    const unsigned r = slot / span;
    const unsigned toff = slot - r * span;
    const unsigned t = tb + toff;
    const unsigned src = node * 8u + r;
    const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
    const unsigned nexp = local_nexp[static_cast<uint64_t>(src) * NT + t];
    if (nexp == 0u) continue;
    unsigned* bdp = reinterpret_cast<unsigned*>(o.bucket_done) + (src * e.capacity + t);
    if (ep16_load_agent_relaxed(bdp) != nexp) continue;
#if V70_R6_SCAT_STEAL
    {
      unsigned won = 0u;
      if (lane == 0u) {
        unsigned ex = nexp;
        won = __hip_atomic_compare_exchange_strong(bdp, &ex, 0xffffffffu, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                   __HIP_MEMORY_SCOPE_AGENT) ? 1u : 0u;
      }
      if (__builtin_amdgcn_readfirstlane(won) == 0u) continue;
    }
#else
    // Exclusive owner: claim without CAS.  GEMM2's RELEASE publication makes the rows visible.
    __hip_atomic_store(bdp, 0xffffffffu, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
#if V70_R6_SCAT_NOACQ
    asm volatile("buffer_inv sc0\ns_waitcnt vmcnt(0)\n" ::: "memory");
#else
    v42_acq_agent();
#endif
    FI_BG(V1988_FI_CORE, 9u, src * e.capacity + t);
    V45_BUCKET_DATA(env, o, x, src, t, nexp);
    // Per-tile publish: one drain and one release flag for the complete token tile.
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __syncwarp();
#if V46_R3 >= 2
    __threadfence_system();
#endif
    if (lane == 0u) v46_bucket_flag(env, o, x, src, t);
    FI_EG(V1988_FI_CORE, 9u, src * e.capacity + t);
    __syncwarp();
    if (lane == 0u) {
#if V70_R8_CNT
      __hip_atomic_fetch_add(pf + 369u, nexp, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
      const unsigned prev = __hip_atomic_fetch_sub(pf + V46_REM + p, 1u,
                                                   (V70_R6_SCAT_RLX) ? __ATOMIC_RELAXED : __ATOMIC_ACQ_REL,
                                                   __HIP_MEMORY_SCOPE_AGENT);
      if (prev == 1u) {
        v41_st_agent_u32(pf + V46_SDONE + p, 1u);
        auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
        if (auto* trf = v70_fresh_sptr(trace)) trf[1320u + p] = __builtin_amdgcn_s_memrealtime();
      }
    }
    __syncwarp();
    did = 1u;
  }
  pos = (slot < total) ? slot : me;
  return did;
}

__device__ __forceinline__ bool v70_local_work_pending(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned rank, unsigned nworkers, bool do_bucket) {
  const auto& e = env.base;
  const unsigned NT = x.node_tokens;
  const auto* nrdy = reinterpret_cast<const unsigned*>(x.node_ready);
  for (unsigned t = rank; t < NT; t += nworkers)
#if V70_R8_2WAVE
    if (ep16_load_agent_relaxed(nrdy + t) == 0u) return true;
#else
    if (ep16_load_relaxed(nrdy + t) == 0u) return true;
#endif
  const unsigned* hdone = reinterpret_cast<const unsigned*>(o.home_done);
  for (unsigned t = rank; t < e.t_local; t += nworkers)
    if (ep16_load_agent_relaxed(hdone + t) == 0u) return true;
  if (do_bucket) {
    auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
    #pragma unroll
    for (unsigned p = 0u; p < 4u; ++p)
      if (v42_ld_agent_rlx(pf + V46_SDONE + p) == 0u) return true;
  }
  return false;
}

__device__ __forceinline__ bool v70_combine_tail(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned worker, unsigned nworkers, unsigned prio, bool do_bucket FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned NT = x.node_tokens;
  const unsigned lane = threadIdx.x & 63u;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  const unsigned rank = (worker >= 8u) ? (worker - 8u) : nworkers;
  (void)prio;   // V70 is always inter-node-first: q1/q2 Scatter precedes NodeReduce/Home.
  if (rank >= nworkers) return true;
#if V70_E3_PRIO
  __builtin_amdgcn_s_setprio(0);
#endif
  FI_BG(V1988_FI_CORE, 48u, worker);
#if V70_R8_2WAVE
  v42_acq_agent();
#endif
  if (lane == 0u && trace) {
    v70_trace_min(&trace[3000u], __builtin_amdgcn_s_memrealtime());
    v41_st_agent_u32(pf + V70_TAIL_START, 1u);
  }
  const unsigned scatter_rank = v70_scat_rank(worker);
  const bool scatter_owner = ((WEAVE_SCATTER_OWNER) != 0) && do_bucket && scatter_rank < V70_NSCAT;
  constexpr unsigned kNC = (V46_WCOMM) - 8u;
#if V70_R6_SCAT_STEAL || ((WEAVE_SCATTER_OWNER) == 0)
  const bool stealer = do_bucket && !scatter_owner && rank >= kNC && nworkers > kNC;
  const unsigned srank = stealer ? rank - kNC : 0u, nsteal = stealer ? nworkers - kNC : 0u;
#else
  const bool stealer = false;
  const unsigned srank = 0u, nsteal = 0u;
#endif
  const unsigned sp0 = stealer ? srank : scatter_rank;
  unsigned scpos[4] = {sp0, sp0, sp0, sp0};
  const auto start = __builtin_amdgcn_s_memrealtime();
  for (;;) {
    if (!v70_local_work_pending(env, o, x, rank, nworkers, scatter_owner || stealer)) break;
    unsigned did = 0u;
    // Inter-node RDMA returns are streamed first.  q1/q2 Scatter feeds the peer's NodeReduce and the
    // inter-node return engine while the local q0/q3 GEMM2 work is still draining.
    if (scatter_owner) {
      did |= v70_scatter_owned_pass(env, o, x, 1u, scatter_rank, scpos[1], 0u FI_CTX_ARG);
      did |= v70_scatter_owned_pass(env, o, x, 2u, scatter_rank, scpos[2], 0u FI_CTX_ARG);
      did |= v70_scatter_owned_pass(env, o, x, 0u, scatter_rank, scpos[0], 0u FI_CTX_ARG);
      did |= v70_scatter_owned_pass(env, o, x, 3u, scatter_rank, scpos[3], 0u FI_CTX_ARG);
    }
    if (stealer) {
      did |= v70_scatter_owned_pass(env, o, x, 1u, srank, scpos[1], nsteal FI_CTX_ARG);
      did |= v70_scatter_owned_pass(env, o, x, 2u, srank, scpos[2], nsteal FI_CTX_ARG);
      did |= v70_scatter_owned_pass(env, o, x, 0u, srank, scpos[0], nsteal FI_CTX_ARG);
      did |= v70_scatter_owned_pass(env, o, x, 3u, srank, scpos[3], nsteal FI_CTX_ARG);
    }
    // Every owned token is visited once.  Multiple tokens per worker are possible for small grids.
    for (unsigned t = rank; t < NT; t += nworkers) {
      did |= v70_visit_nr_owned(env, o, x, t FI_CTX_ARG);
      if (did && !(V70_E3_QUIET_HT)) break;   // remote-return priority
    }
    if ((V70_E3_QUIET_HT) || !(did & 1u)) {
      for (unsigned t = rank; t < e.t_local; t += nworkers) {
        did |= v70_visit_home_owned(env, o, x, t FI_CTX_ARG);
        if (did && !(V70_E3_QUIET_HT)) break;
      }
    }
    if (!did) {
      unsigned stop = 0u;
      if (lane == 0u) {
        stop = ep16_load_relaxed(reinterpret_cast<unsigned*>(e.error));
        if (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks) {
          v983_error(reinterpret_cast<unsigned*>(e.error), 914u);
          stop = 1u;
        }
      }
      if (__builtin_amdgcn_readlane(stop, 0)) break;
      __builtin_amdgcn_s_sleep((V70_R8_2WAVE) ? 6 : 2);
    }
  }
  if (lane == 0u && trace) {
    v70_trace_max(&trace[3001u], __builtin_amdgcn_s_memrealtime());
    v41_st_agent_u32(pf + V70_TAIL_END, 1u);
  }
  FI_EG(V1988_FI_CORE, 48u, worker);
  return ep16_load_relaxed(reinterpret_cast<unsigned*>(e.error)) == 0u;
}

// -------------------------------------------------------------------------------------------------------------
// v46_combine_loop: prio 0 = NodeReduce > Home > Scatter (v45 order), prio 1 = Scatter > NodeReduce > Home.
// Scatter pass order q1 > q2 > q0 > q3 (the remote passes feed NodeReduce + the inter-node return RDMA).
// -------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void v46_combine_loop(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned worker, unsigned prio, bool do_bucket, bool do_node_red, bool do_home FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const unsigned nb_nr = __builtin_amdgcn_readfirstlane((NT + 63u) >> 6u);
  const unsigned nb_h = __builtin_amdgcn_readfirstlane((e.t_local + 63u) >> 6u);
  if (nb_nr > 16u || nb_h > 128u) {   // NT <= 1024 keeps every scatter pass within 128 blocks
    if (lane == 0u) v983_error(reinterpret_cast<unsigned*>(e.error), 975u);
    return;
  }
  uint64_t nr0 = 0ull, nr1 = 0ull, h0 = 0ull, h1 = 0ull;
  if (do_node_red) { nr0 = v44_lowmask(nb_nr < 64u ? nb_nr : 64u); nr1 = v44_lowmask(nb_nr > 64u ? nb_nr - 64u : 0u); }
  if (do_home)     { h0 = v44_lowmask(nb_h < 64u ? nb_h : 64u);    h1 = v44_lowmask(nb_h > 64u ? nb_h - 64u : 0u); }
  unsigned pos_nr = nb_nr ? (worker % nb_nr) : 0u;
  unsigned pos_h = nb_h ? (worker % nb_h) : 0u;
  unsigned seed = __builtin_amdgcn_readfirstlane(worker * 7u);
#if (V46_DBG & 2)
  return;
#endif
  bool scat_live = do_bucket;
  // Official WEAVE_READY_QUEUE=1 is the remote/return-first queue: every
  // combine worker drains ready scatter rows (including S1F direct returns)
  // before NodeReduce/Home.  The other queue modes preserve the caller's
  // phase order, so the knob changes real tail execution rather than only the
  // GEMM2 claim permutation.
  unsigned eprio = prio;
#if WEAVE_READY_QUEUE == 1
  eprio = 1u;
#endif
#if WEAVE_SM0_STREAM_SCHED > 0
  // SM0 stream bias: 1 -> worker 0, 2 -> workers 0..1, 3 -> workers 0..3
  // keep the return/S1F scatter stream saturated; all other workers retain the
  // caller's priority so this is independently observable from READY_QUEUE.
  constexpr unsigned kSm0StreamN = 1u << ((WEAVE_SM0_STREAM_SCHED) - 1u);
  eprio = (worker < kSm0StreamN) ? 1u : prio;
#endif
  // sweep state per pass: a = q1, b = q2, c = q0, d = q3
  uint64_t a0 = 0ull, a1 = 0ull, b0 = 0ull, b1 = 0ull, c0 = 0ull, c1 = 0ull, d0 = 0ull, d1 = 0ull;
  unsigned ap = 0u, as = 0u, bp = 0u, bs = 0u, cp = 0u, cs = 0u, dp = 0u, ds = 0u;
  unsigned sseed = __builtin_amdgcn_readfirstlane(worker * 13u);
  bool pf0_seen = false;
  const auto start = __builtin_amdgcn_s_memrealtime();
  for (;;) {
    if (!(nr0 | nr1 | h0 | h1) && !scat_live) return;
    unsigned stop = 0u;
    if (lane == 0u) {
      stop = ep16_load_relaxed(reinterpret_cast<unsigned*>(e.error));
      if (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks) {
        v983_error(reinterpret_cast<unsigned*>(e.error), 914u);
        stop = 1u;
      }
    }
    if (__builtin_amdgcn_readlane(stop, 0)) return;   // opus55 v46 E10 (lane 0 wrote stop)
    unsigned did = 0u;
    if (eprio && scat_live) {
      unsigned r = __builtin_amdgcn_readfirstlane(v46_sweep_pass(env, o, x, 1u, a0, a1, ap, as, worker, sseed FI_CTX_ARG));
      if (!(r & 1u)) r |= __builtin_amdgcn_readfirstlane(v46_sweep_pass(env, o, x, 2u, b0, b1, bp, bs, worker, sseed FI_CTX_ARG));
      if (!(r & 1u)) r |= __builtin_amdgcn_readfirstlane(v46_sweep_pass(env, o, x, 0u, c0, c1, cp, cs, worker, sseed FI_CTX_ARG));
      if (!(r & 1u)) r |= __builtin_amdgcn_readfirstlane(v46_sweep_pass(env, o, x, 3u, d0, d1, dp, ds, worker, sseed FI_CTX_ARG));
      if ((as & 4u) && (bs & 4u) && (cs & 4u) && (ds & 4u)) scat_live = false;
      if (r & 1u) continue;
    }
    if (nr0 | nr1) {
      const unsigned blk = __builtin_amdgcn_readfirstlane(v44_next_blk(nr0, nr1, nb_nr, pos_nr));
      const unsigned r = __builtin_amdgcn_readfirstlane(v45_visit_nr(env, o, x, blk, seed FI_CTX_ARG));
      seed += 1u;
      if (r & 2u) v44_clear_blk(nr0, nr1, blk);
      if (r & 1u) { pos_nr = blk; continue; }
      pos_nr = blk + 1u;
    }
    if ((h0 | h1) && !pf0_seen && v42_ld_agent_rlx(pf + 0u) != 0u) { v42_acq_agent(); pf0_seen = true; }
    if ((h0 | h1) && pf0_seen) {
      const unsigned blk = __builtin_amdgcn_readfirstlane(v44_next_blk(h0, h1, nb_h, pos_h));
      const unsigned r = __builtin_amdgcn_readfirstlane(v45_visit_home(env, o, x, blk, seed FI_CTX_ARG));
      seed += 1u;
      if (r & 2u) v44_clear_blk(h0, h1, blk);
      if (r & 1u) { pos_h = blk; continue; }
      pos_h = blk + 1u;
    }
    if (!eprio && scat_live) {
      unsigned r = __builtin_amdgcn_readfirstlane(v46_sweep_pass(env, o, x, 1u, a0, a1, ap, as, worker, sseed FI_CTX_ARG));
      if (!(r & 1u)) r |= __builtin_amdgcn_readfirstlane(v46_sweep_pass(env, o, x, 2u, b0, b1, bp, bs, worker, sseed FI_CTX_ARG));
      if (!(r & 1u)) r |= __builtin_amdgcn_readfirstlane(v46_sweep_pass(env, o, x, 0u, c0, c1, cp, cs, worker, sseed FI_CTX_ARG));
      if (!(r & 1u)) r |= __builtin_amdgcn_readfirstlane(v46_sweep_pass(env, o, x, 3u, d0, d1, dp, ds, worker, sseed FI_CTX_ARG));
      did |= r & 1u;
      if ((as & 4u) && (bs & 4u) && (cs & 4u) && (ds & 4u)) scat_live = false;
    }
    if (!did) __builtin_amdgcn_s_sleep(2);
  }
}

#include "v70_v46_8wave.hpp"   // V70: hardware-s_barrier unified 8-wave Atrex for pure 8/0 CTAs

// -------------------------------------------------------------------------------------------------------------
// v46_main: role dispatch of one wave (called from stage_ep16_turbo_overlap under #if V46_SP)
// -------------------------------------------------------------------------------------------------------------
#ifndef V46_DBG
#define V46_DBG 0
#endif
__device__ __forceinline__ void v46_main(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, unsigned cta, unsigned wave FI_CTX_PARAM) {
  const auto& e = env.base;
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  const unsigned lane = threadIdx.x & 63u;
  const bool dual = (e.t_local >= 16u);
  const unsigned N08 = V46_N08, N44 = V46_N44;
  const unsigned ncg = N44 + (dual ? 2u : 1u) * (V46_N80);
  const unsigned NPULL = V46_NPULL, NNR = V46_NNR;
  const unsigned nwf = NPULL + 4u * ncg;   // front-stage and P0-pull workers
  const unsigned ncomb = 4u * ncg + (V46_WCOMM - 8u);   // dense combine workers: cw 8..WCOMM-1 + all compute waves
  unsigned cw = V46_NONE, grp = V46_NONE;
  if (cta < N08) {
    cw = cta * 8u + wave;
  } else if (cta < N08 + N44) {
    if (wave >= 4u) cw = 8u * N08 + 4u * (cta - N08) + (wave - 4u);
    else grp = cta - N08;
  } else {
    const unsigned k = cta - N08 - N44;
    if (dual) grp = N44 + 2u * k + ((wave >= 4u) ? 1u : 0u);
    else if (wave < 4u) grp = N44 + k;
  }
  cw = __builtin_amdgcn_readfirstlane(cw);
  grp = __builtin_amdgcn_readfirstlane(grp);
  if (cw != V46_NONE) {
    if (cw < 4u) {
      ep16_gda_wave4(env, o, x, cw FI_CTX_ARG);
      return;
    }
    if (cw < 8u) {
      FI_BG(V1988_FI_CORE, 45u, cw);
      v41_combine_loop(env, o, x, cw - 4u, 4u, true, false, false, false FI_CTX_ARG);
      FI_EG(V1988_FI_CORE, 45u, cw);
    } else if (cw < V46_CW_POOL) {
#if !(V46_DBG & 4)
      v46_plan_remote(env, o, x, arena, (cw - 8u) >> 2u, V46_NRP FI_CTX_ARG);
#endif
      v70_combine_tail(env, o, x, cw, ncomb, 0u, true FI_CTX_ARG);
    } else if (cw < V46_CW_POOL + NNR) {
      // The NNR pool owns NodeReduce/Home and also participates in the
      // communication-only Scatter drain, always after inter-node passes.
      v70_combine_tail(env, o, x, cw, ncomb, 0u, true FI_CTX_ARG);
    } else {
      const unsigned pw = cw - V46_CW_POOL - NNR;
      // pull order q0 (with the compute groups) -> q3 (local, gated only on LBUILD) -> q1 -> q2: the remote pulls
      // are bounded by the remote RDMA arrival, so pulling the local tail first never delays them
      v45_front_stage(env, o, x, pw, nwf, pw, NPULL FI_CTX_ARG);
      v46_pull_pass(env, o, x, 0u, pw, nwf FI_CTX_ARG);
      v46_pull_pass(env, o, x, 3u, pw, NPULL FI_CTX_ARG);
      v46_pull_pass(env, o, x, 1u, pw, NPULL FI_CTX_ARG);
      v46_pull_pass(env, o, x, 2u, pw, NPULL FI_CTX_ARG);
      v70_combine_tail(env, o, x, cw, ncomb, 1u, true FI_CTX_ARG);
    }
    if (lane == 0u && trace) trace[16u + cta * 16u + 5u] = v983_clock();
    return;
  }
  if (grp == V46_NONE) return;
#if V54_CACHE_S1 && V54_S1_INV == 0
  // v54: one OR per compute CTA, before the front stage and therefore before the plan rendezvous that gates
  // every GEMM item.  If the XCD id does not distinguish all four XCDs the mask stays != 0xF and the
  // narrow-scope acquire is disabled globally (correctness guard, see v54_xcd.hpp).
  if (threadIdx.x == 0u)
    __hip_atomic_fetch_or(reinterpret_cast<unsigned*>(o.plan_flags) + V54_XMASK, 1u << v54_xcd(),
                          __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
#endif
  const unsigned wk = NPULL + grp * 4u + (wave & 3u);
  const unsigned head_worker = grp * 4u + (wave & 3u);
  const unsigned head_workers = 4u * ncg;
  const unsigned head_second_workers = 2u * ncg;
  const unsigned head_second_worker = (grp >> 1u) * 4u + (wave & 3u);
  // V70_R2X_NOPRE: skip the four-wave q0 prelude; all eight waves copy the head, then enter the R2X
  // 8-wave Atrex together (the prelude's old 4-wave body held half of every CTA for ~500 us).
  const bool pure_head_split = dual && ((V46_N44) == 0u) && !(V70_R2X_NOPRE);
  // IRON RULE 1: all W00..W07 contribute one remote token at entry.  On the
  // pure-compute layout W00..W03 account and immediately leave; W04..W07 drain
  // the remaining WEAVE_V_HEAD range after publishing the q0 plan.  This keeps
  // every CTA's q0 rendezvous off the tail of the head copy while the other
  // four waves are still issuing remote pulls.
  if (pure_head_split) {
    v70_pre4_reset();
    v70_remote_head_begin(env, o, x, head_worker, head_workers, wave < 4u FI_CTX_ARG);
  } else {
    FI_BG(V1988_FI_CORE, 46u, head_worker);
    v70_remote_head_copy(env, o, x, head_worker, head_workers, env.base.payload_bytes FI_CTX_ARG);
    v70_remote_head_publish(env, o, head_workers FI_CTX_ARG);
    FI_EG(V1988_FI_CORE, 46u, head_worker);
  }
  // HEAD: compute groups contribute only to the 48 B/token local meta record.
  // Payload staging stays on the persistent pull workers, so this group can
  // reach q0 planning and GEMM1 instead of copying every local row first.
  v45_front_stage<false>(env, o, x, wk, nwf, 0u, 1u FI_CTX_ARG);
#if (V46_DBG & 16)
  return;
#endif
  if (!v46_plan_local(env, o, x, arena, grp, ncg FI_CTX_ARG)) return;
  v46_pull_pass(env, o, x, 0u, wk, nwf FI_CTX_ARG);
  if (pure_head_split) {
    if (wave < 4u) {
      // W00..W03: q0 is ready while W04..W07 are still in the head copy.
      // Consume one q0 GEMM1 item with the four-wave body, then the leader
      // builds q3 before the unified eight-wave rendezvous.
      if (!v70_head_g1_prelude(env, o, x, arena FI_CTX_ARG)) return;
      if (grp == 0u && !v46_plan_local_tail(env, o, x, arena FI_CTX_ARG)) return;
    } else {
      v70_remote_head_finish(env, o, x, head_worker, head_second_worker,
                             head_workers, head_second_workers FI_CTX_ARG);
      v70_pre4_signal();
    }
  } else if (grp == 0u && !v46_plan_local_tail(env, o, x, arena FI_CTX_ARG)) {
    return;
  }
#if (V46_DBG & 1)
  const bool ok = true;
#elif V70_8W_ACTIVE && ((V46_N44) == 0)
  // Pure-compute layout: every CTA outside the 0/8 comm prefix is an 8/0
  // compute CTA.  The 8-wave path requires the dual group (t_local >= 16);
  // rejecting a single-group launch here keeps every s_barrier reached by all
  // eight waves.  Compiling out the unreachable 4-wave matmul also prevents the
  // allocator from merging its live ranges into the 8-wave kernel body.
  if (!dual) {
    v983_error(reinterpret_cast<unsigned*>(e.error), 976u);
    return;
  }
  const bool ok = v70_atrex8(env, o, x, arena FI_CTX_ARG);
#elif V70_8W_ACTIVE
  // Mixed layout: 8/0 CTAs use the unified hardware-s_barrier body.  A 4/4 CTA
  // has only W00..W03 compute threads here; W04..W07 already took their comm
  // role above.  V70_R2X_44 gives those four waves the full sequential R2 item
  // with its own four-wave software rendezvous.
  const bool is_8w = dual && cta >= (V46_N08 + V46_N44);
#if V70_R2X_44
  const bool ok = is_8w
      ? v70_atrex8(env, o, x, arena FI_CTX_ARG)
      : v70_atrex4_r2(env, o, x, arena FI_CTX_ARG);
#else
  const bool ok = is_8w
      ? v70_atrex8(env, o, x, arena FI_CTX_ARG)
      : v46_atrex(env, o, x, arena FI_CTX_ARG);
#endif
#else
  const bool ok = v46_atrex(env, o, x, arena FI_CTX_ARG);
#endif
  if ((threadIdx.x & 255u) == 0u && trace) {
    trace[16u + cta * 16u + 9u] = trace[1312u];
    trace[16u + cta * 16u + 10u] = v983_clock();
  }
  if (!ok) return;
  // IRON RULE 3: every one of the eight compute waves enters the zero-CAS
  // private-shard Combine tail.  These workers are never Scatter owners.
  v70_combine_tail(env, o, x, V46_WCOMM + grp * 4u + (wave & 3u), ncomb, 0u, true FI_CTX_ARG);
  if ((threadIdx.x & 255u) == 0u && trace) trace[16u + cta * 16u + 11u] = v983_clock();
}

#include "v70_v46_8wave.hpp"
#include "v70_r8.hpp"
#if V70_PATHB
// =============================================================================================================
// v46_main_pathb: Production 2-Node 16xMI308X Path B (160 CTAs x 256 threads, 32 KiB dynamic LDS).
// Supports ALL THREE MI308X CU wave-split modes with zero deadlock and full compute saturation:
//   WEAVE_CU_WAVE_SPLIT = 0 (auto) / 2 (4-4): NCOMM_CTA comm CTAs on CUs [0..NCOMM_CTA) paired with compute CTAs
//   WEAVE_CU_WAVE_SPLIT = 1 (0-8):            4 comm CTAs (GDA/RX/plan_remote/pull) + 156 compute CTAs (76 pure 0-8 CUs)
//   WEAVE_CU_WAVE_SPLIT = 3 (8-0):            4 dedicated 8-comm CUs (cta 0..3 & 80..83) + 76 pure 0-8 compute CUs
// Deadlock-free ordering: compute CTAs run v45_front_stage<false> FIRST (publishing local meta PROGRESS_MAGIC)
// before plan_local -> pull q0 -> plan_local_tail(grp 0) -> v70_remote_head_copy -> v70_atrex4_r8 -> v70_combine_tail.
// =============================================================================================================
__device__ __forceinline__ void v46_main_pathb(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, unsigned cta, unsigned wave FI_CTX_PARAM) {
  const auto& e = env.base;
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  const unsigned lane = threadIdx.x & 63u;
  constexpr unsigned NCTA = 160u;
  constexpr unsigned NCOMM_CTA = V70_PB_NCOMM_CTA;
  constexpr unsigned NCG = NCTA - NCOMM_CTA;
  constexpr unsigned NWCOMM = 4u * NCOMM_CTA;
  constexpr unsigned NRP = V46_NRP;
  constexpr unsigned NNR_CTA = V70_PB_NNR_CTA;
  constexpr unsigned NPULL = V46_NPULL;
  constexpr unsigned nwf = NPULL + 4u * NCG;
  constexpr unsigned ncomb = 4u * NCG + (NWCOMM - 8u);

  unsigned cidx = V46_NONE, grp = V46_NONE;
#if (WEAVE_CU_WAVE_SPLIT) == 3
  // 8-0 mode: pair comm CTAs on the same physical CUs (cta 0..K-1 and cta 80..80+(NCOMM-K)-1)
  constexpr unsigned K0 = (NCOMM_CTA + 1u) / 2u;
  constexpr unsigned K1 = NCOMM_CTA - K0;
  if (cta < K0) {
    cidx = cta;
  } else if (cta >= 80u && cta < 80u + K1) {
    cidx = K0 + (cta - 80u);
  } else if (cta < 80u) {
    grp = cta - K0;
  } else {
    grp = (80u - K0) + (cta - (80u + K1));
  }
#else
  if (cta < NCOMM_CTA) cidx = cta;
  else grp = cta - NCOMM_CTA;
#endif
  cidx = __builtin_amdgcn_readfirstlane(cidx);
  grp = __builtin_amdgcn_readfirstlane(grp);

  // Zero VGPRs live across v70_atrex4_r8: keep role/idx strictly in SGPRs and recompute tail_w/tail_prio after GEMM!
  unsigned do_gemm_u = 0u, gemm_role = 0u, gemm_idx = 0u;
  if (cidx != V46_NONE) {
    const unsigned cw = __builtin_amdgcn_readfirstlane(cidx * 4u + wave);
    if (cidx == 0u) {
      ep16_gda_wave4(env, o, x, wave FI_CTX_ARG);
      return;
    }
    if (cidx == 1u) {
      FI_BG(V1988_FI_CORE, 45u, cw);
      v41_combine_loop(env, o, x, wave, 4u, true, false, false, false FI_CTX_ARG);
      FI_EG(V1988_FI_CORE, 45u, cw);
      if (lane == 0u && trace) trace[16u + cta * 16u + 5u] = v983_clock();
      return;
    } else if (cidx < 2u + NRP) {
      v46_plan_remote(env, o, x, arena, cidx - 2u, NRP FI_CTX_ARG);
      do_gemm_u = ((V70_R8_HYB_EFF) > 0 && cidx >= (V70_R8_HYB_LO)) ? 1u : 0u; gemm_role = 1u; gemm_idx = cidx;
    } else if (cidx < 2u + NRP + NNR_CTA) {
      // Dedicated streaming Reduce CTA (WEAVE_REDUCE_WARPS > 0): skips GEMM and runs NR/Home during Middle!
    } else {
      const unsigned pw = __builtin_amdgcn_readfirstlane((cidx - (2u + NRP + NNR_CTA)) * 4u + wave);
      v45_front_stage(env, o, x, pw, nwf, pw, NPULL FI_CTX_ARG);
      v46_pull_pass(env, o, x, 0u, pw, nwf FI_CTX_ARG);
      v46_pull_pass(env, o, x, 3u, pw, NPULL FI_CTX_ARG);
      v46_pull_pass(env, o, x, 1u, pw, NPULL FI_CTX_ARG);
      v46_pull_pass(env, o, x, 2u, pw, NPULL FI_CTX_ARG);
      do_gemm_u = ((V70_R8_HYB_EFF) > 0 && cidx >= (V70_R8_HYB_LO)) ? 1u : 0u; gemm_role = 1u; gemm_idx = cidx;
    }
  } else {
    // Compute CTA (grp in [0, NCG)): stage local meta FIRST (zero RDMA circular wait!), then plan_local + q0 pull + head pull
    const unsigned wk = __builtin_amdgcn_readfirstlane(NPULL + grp * 4u + wave);
    const unsigned head_worker = __builtin_amdgcn_readfirstlane(grp * 4u + wave);
    const unsigned head_workers = 4u * NCG;
    v45_front_stage<false>(env, o, x, wk, nwf, 0u, 1u FI_CTX_ARG);
    if (!v46_plan_local(env, o, x, arena, grp, NCG FI_CTX_ARG)) return;
    v46_pull_pass(env, o, x, 0u, wk, nwf FI_CTX_ARG);
    if (grp == 0u && !v46_plan_local_tail(env, o, x, arena FI_CTX_ARG)) return;
#if !V70_R8_2WAVE
    FI_BG(V1988_FI_CORE, 46u, head_worker);
    v70_remote_head_copy(env, o, x, head_worker, head_workers, env.base.payload_bytes FI_CTX_ARG);
    __syncthreads();
    if (wave == 0u) v70_remote_head_publish_n(env, o, head_workers, 4u FI_CTX_ARG);
    FI_EG(V1988_FI_CORE, 46u, head_worker);
#else
    __syncthreads();
#endif
    if (threadIdx.x == 0u && trace) trace[16u + cta * 16u + 8u] = v983_clock();
    do_gemm_u = 1u; gemm_role = 0u; gemm_idx = grp;
  }
  do_gemm_u = __builtin_amdgcn_readfirstlane(do_gemm_u);
  gemm_role = __builtin_amdgcn_readfirstlane(gemm_role);
  gemm_idx = __builtin_amdgcn_readfirstlane(gemm_idx);
  if (do_gemm_u) {
    const bool ok = v70_atrex4_r8(env, o, x, arena, gemm_role, gemm_idx FI_CTX_ARG);
    if (gemm_role == 0u && threadIdx.x == 0u && trace) {
      trace[16u + blockIdx.x * 16u + 9u] = trace[1312u];
      trace[16u + blockIdx.x * 16u + 10u] = v983_clock();
    }
    if (!ok) return;
  }
  // Recompute tail worker id purely from SGPRs (blockIdx.x + readfirstlane(threadIdx.x>>6)) -> ZERO VGPRs live across GEMM!
  const unsigned cta2 = blockIdx.x;
  const unsigned w2 = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6u);
  unsigned tail_w = 0u, tail_prio = 0u;
  bool is_comm2 = false;
#if (WEAVE_CU_WAVE_SPLIT) == 3
  if (cta2 < K0) { is_comm2 = true; tail_w = cta2 * 4u + w2; }
  else if (cta2 >= 80u && cta2 < 80u + K1) { is_comm2 = true; tail_w = (K0 + cta2 - 80u) * 4u + w2; }
  else { const unsigned g2 = (cta2 < 80u) ? (cta2 - K0) : ((80u - K0) + (cta2 - (80u + K1))); tail_w = NWCOMM + g2 * 4u + w2; }
#else
  if (cta2 < NCOMM_CTA) { is_comm2 = true; tail_w = cta2 * 4u + w2; }
  else { tail_w = NWCOMM + (cta2 - NCOMM_CTA) * 4u + w2; }
#endif
  tail_prio = is_comm2 ? 1u : 0u;
  v70_combine_tail(env, o, x, __builtin_amdgcn_readfirstlane(tail_w), ncomb, __builtin_amdgcn_readfirstlane(tail_prio), true FI_CTX_ARG);
  if (is_comm2) {
    if ((threadIdx.x & 63u) == 0u && trace) trace[16u + cta2 * 16u + 5u] = v983_clock();
  } else if (threadIdx.x == 0u && trace) {
    trace[16u + cta2 * 16u + 11u] = v983_clock();
  }
}
#endif

