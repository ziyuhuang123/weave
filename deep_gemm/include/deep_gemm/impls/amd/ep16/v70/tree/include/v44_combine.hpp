// v44_combine.hpp -- opus55 v44 C1: unified dynamic combine worker loop.
//
// Replaces the static (worker, stride) 64-item slice ownership of v41_combine_loop.
// With stride = 592 compute waves, only t_local/64 (=16 for 1024 tokens) waves owned a
// NodeReduce / HomeCombine slice; every other wave saw pending==0 and returned, so the
// whole reduce/home tail ran on ~40 waves (r80 397b: NR 36 waves, Home 42 waves).
//
//   * Stage B (scatter): per-MB atomic cursor over a CHUNK-MAJOR item order
//       item = chunk(64 tokens) x src(remote node first) x token-in-chunk,
//     claimed V44_SCAT_CLAIM items at a time, started once MB b's GEMM2 is globally
//     complete (pf[48+b]); claim exclusivity is still enforced by the bucket_done CAS.
//   * Stage C (node reduce) / Stage D (home combine): every wave visits ALL 64-token
//     blocks (cyclic, rotated by worker id), claims at most V44_TOK_CLAIM ready tokens
//     per visit, and drops a block from its private bitmap only once all 64 tokens are
//     globally complete.  A wave returns only when all three stages are globally done.
//   * Per iteration the order is C (NodeReduce, unlocks RDMA return) -> D -> B.
#pragma once
#ifndef V44_COMBINE
#define V44_COMBINE 0
#endif
#if V44_COMBINE
#ifndef V44_SCAT_CLAIM
#define V44_SCAT_CLAIM 8u
#endif
#ifndef V44_TOK_CLAIM
#define V44_TOK_CLAIM 1u
#endif
#define V44_PF_SCAT_CUR 600u   // pf[600+b]: chunk-major scatter cursor of MB b (plan_flags is zeroed per launch)
#ifndef V44_QUIET
#define V44_QUIET 0
#endif
// opus55 v44 Q1 (V44_QUIET=1): the combine loop polls its gate flags with relaxed loads and issues ONE acquire
// the first time a flag is seen set (cached in a register), instead of an acquire (buffer_inv sc1) on every
// iteration of every polling wave; a successful token claim is followed by s_waitcnt vmcnt(0) only, because
// node_partial / return_recv_payload / node_recv live in the MTYPE-UC arena (no cache can hold stale lines).
#if V44_QUIET
#define V44_CLAIM_ACQ() asm volatile("s_waitcnt vmcnt(0)" ::: "memory")
#else
#define V44_CLAIM_ACQ() v42_acq_sys()
#endif

__device__ __forceinline__ unsigned v44_mbcnt(uint64_t m) {
  return __builtin_amdgcn_mbcnt_hi(static_cast<unsigned>(m >> 32u),
                                   __builtin_amdgcn_mbcnt_lo(static_cast<unsigned>(m), 0u));
}
__device__ __forceinline__ uint64_t v44_ballot_nz(unsigned v) {
  return __builtin_amdgcn_uicmp(v, 0u, 33u);  // 33 == ICMP_NE
}
__device__ __forceinline__ uint64_t v44_lowmask(unsigned n) {  // n in [0,64]
  return (n >= 64u) ? ~0ull : ((n == 0u) ? 0ull : ((1ull << n) - 1ull));
}
// next set bit at or after pos (cyclic) in the 128-bit (m0,m1) mask of nblk blocks; caller guarantees m0|m1 != 0
__device__ __forceinline__ unsigned v44_next_blk(uint64_t m0, uint64_t m1, unsigned nblk, unsigned pos) {
  if (pos >= nblk) pos = 0u;
  if (pos < 64u) {
    const uint64_t a = m0 & ~v44_lowmask(pos);
    if (a) return (unsigned)__builtin_ctzll(a);
    if (m1) return 64u + (unsigned)__builtin_ctzll(m1);
    return (unsigned)__builtin_ctzll(m0);
  }
  const uint64_t b = m1 & ~v44_lowmask(pos - 64u);
  if (b) return 64u + (unsigned)__builtin_ctzll(b);
  if (m0) return (unsigned)__builtin_ctzll(m0);
  return 64u + (unsigned)__builtin_ctzll(m1);
}
__device__ __forceinline__ void v44_clear_blk(uint64_t& m0, uint64_t& m1, unsigned blk) {
  if (blk < 64u) m0 &= ~(1ull << blk); else m1 &= ~(1ull << (blk - 64u));
}

#ifndef V44_FASTTOK
#define V44_FASTTOK 0
#endif
#if V44_FASTTOK
// opus55 v44 F1: per-token NodeReduce / HomeCombine with compiler-tracked sc1 buffer loads.
// The v41/v42 helpers end every 4-load batch with an inline-asm s_waitcnt vmcnt(0) (24 serialized
// round trips per 4096-wide token).  Here all <=8 partial rows of TWO 512-column slices are issued
// before the first use; the compiler inserts the (partial) vmcnt waits.  Arithmetic order per
// element is identical to the original (rows accumulated in q = 0..7 order, then node_recv).
#define V44_SC1 16   // CPol::SC1 on gfx940+ (same coherence as the inline "global_load ... sc1")
__device__ __forceinline__ uint4 v44_bld(WeaveI32x4 rs, uint32_t voff) {
  const WeaveI32x4 v = weave_raw_buf_load_i32x4(rs, static_cast<int32_t>(voff), 0, V44_SC1);
  return make_uint4((unsigned)v[0], (unsigned)v[1], (unsigned)v[2], (unsigned)v[3]);
}
__device__ __forceinline__ uint4 v44_bld_sys(WeaveI32x4 rs, uint32_t voff) {
  const WeaveI32x4 v = weave_raw_buf_load_i32x4(rs, static_cast<int32_t>(voff), 0, 17);
  return make_uint4((unsigned)v[0], (unsigned)v[1], (unsigned)v[2], (unsigned)v[3]);
}
__device__ __forceinline__ uint4 v44_pack_bf16x8(const float* s) {
  uint4 o;
  o.x = weave_comm_bf16(s[0]) | (static_cast<unsigned>(weave_comm_bf16(s[1])) << 16u);
  o.y = weave_comm_bf16(s[2]) | (static_cast<unsigned>(weave_comm_bf16(s[3])) << 16u);
  o.z = weave_comm_bf16(s[4]) | (static_cast<unsigned>(weave_comm_bf16(s[5])) << 16u);
  o.w = weave_comm_bf16(s[6]) | (static_cast<unsigned>(weave_comm_bf16(s[7])) << 16u);
  return o;
}

__device__ __forceinline__ void v44_node_reduce_fast(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned t) {
  const unsigned lane = threadIdx.x & 63u;
  const unsigned H = o.hidden, NT = x.node_tokens;
  const unsigned mask = __builtin_amdgcn_readfirstlane(reinterpret_cast<const unsigned*>(x.node_rank_mask)[t]);
  const uint32_t rsb = NT * H * 2u;                         // bytes per local-rank partial plane
  const WeaveI32x4 rp = weave_make_buf_rsrc(reinterpret_cast<const void*>(x.node_partial), 8u * rsb);
  const uint32_t tb = t * H * 2u;
  unsigned col = lane * 8u;
  for (; col + 512u < H; col += 1024u) {
    uint4 a[8], b[8];
    #pragma unroll
    for (unsigned q = 0u; q < 8u; ++q) {
      a[q] = make_uint4(0u, 0u, 0u, 0u); b[q] = a[q];
      if ((mask >> q) & 1u) {
        a[q] = v44_bld(rp, q * rsb + tb + col * 2u);
        b[q] = v44_bld(rp, q * rsb + tb + (col + 512u) * 2u);
      }
    }
    float sa[8] = {}, sb[8] = {};
    #pragma unroll
    for (unsigned q = 0u; q < 8u; ++q) {
      v42_acc_bf16x8(sa, a[q], ((mask >> q) & 1u) != 0u);
      v42_acc_bf16x8(sb, b[q], ((mask >> q) & 1u) != 0u);
    }
    ep16_st_sys_u4(reinterpret_cast<uint16_t*>(x.node_output) + static_cast<uint64_t>(t) * H + col, v44_pack_bf16x8(sa));
    ep16_st_sys_u4(reinterpret_cast<uint16_t*>(x.node_output) + static_cast<uint64_t>(t) * H + col + 512u, v44_pack_bf16x8(sb));
  }
  for (; col < H; col += 512u) {
    uint4 a[8];
    #pragma unroll
    for (unsigned q = 0u; q < 8u; ++q) {
      a[q] = make_uint4(0u, 0u, 0u, 0u);
      if ((mask >> q) & 1u) a[q] = v44_bld(rp, q * rsb + tb + col * 2u);
    }
    float sa[8] = {};
    #pragma unroll
    for (unsigned q = 0u; q < 8u; ++q) v42_acc_bf16x8(sa, a[q], ((mask >> q) & 1u) != 0u);
    ep16_st_sys_u4(reinterpret_cast<uint16_t*>(x.node_output) + static_cast<uint64_t>(t) * H + col, v44_pack_bf16x8(sa));
  }
  ep16_vmcnt0();
  __syncwarp();
  if (lane == 0u) ep16_st_sys_relaxed(reinterpret_cast<unsigned*>(x.node_ready) + t, 1u);
  __syncwarp();
  (void)env;
}

#ifndef V70_R6_HOME_MLP
#define V70_R6_HOME_MLP 0
#endif
__device__ __forceinline__ void v44_home_fast(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned t) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned H = o.hidden;
  const unsigned hmask = __builtin_amdgcn_readfirstlane(reinterpret_cast<const unsigned*>(o.home_expected)[t]);
  const unsigned node_base = (e.rank >> 3u) * 8u;
  const uint32_t rsb = e.capacity * H * 2u;                  // bytes per source-rank return plane
  const WeaveI32x4 rr = weave_make_buf_rsrc(reinterpret_cast<const uint16_t*>(o.return_recv_payload) +
                                            static_cast<uint64_t>(node_base) * e.capacity * H, 8u * rsb);
  const WeaveI32x4 rn = weave_make_buf_rsrc(reinterpret_cast<const void*>(x.node_recv), x.node_tokens * H * 2u);
  const bool has_n = (hmask & 256u) != 0u;
  const uint32_t tb = t * H * 2u;
  uint16_t* outp = reinterpret_cast<uint16_t*>(o.final_output) + static_cast<uint64_t>(t) * H;
  unsigned col = lane * 8u;
#if V70_R6_HOME_MLP
  for (; col + 1536u < H; col += 2048u) {
    uint4 a[4][8], na[4];
    #pragma unroll
    for (unsigned c = 0u; c < 4u; ++c) {
      na[c] = make_uint4(0u, 0u, 0u, 0u);
      #pragma unroll
      for (unsigned q = 0u; q < 8u; ++q) {
        a[c][q] = make_uint4(0u, 0u, 0u, 0u);
        if ((hmask >> q) & 1u) a[c][q] = v44_bld(rr, q * rsb + tb + (col + c * 512u) * 2u);
      }
      if (has_n) na[c] = v44_bld_sys(rn, tb + (col + c * 512u) * 2u);
    }
    #pragma unroll
    for (unsigned c = 0u; c < 4u; ++c) {
      float sa[8] = {};
      #pragma unroll
      for (unsigned q = 0u; q < 8u; ++q) v42_acc_bf16x8(sa, a[c][q], ((hmask >> q) & 1u) != 0u);
      if (has_n) {
        const unsigned wa[4] = {na[c].x, na[c].y, na[c].z, na[c].w};
        #pragma unroll
        for (unsigned k = 0u; k < 4u; ++k) {
          sa[2u * k]      += weave_comm_bf16_val(static_cast<uint16_t>(wa[k] & 65535u));
          sa[2u * k + 1u] += weave_comm_bf16_val(static_cast<uint16_t>(wa[k] >> 16u));
        }
      }
      *reinterpret_cast<uint4*>(outp + col + c * 512u) = v44_pack_bf16x8(sa);
    }
  }
#endif
  for (; col + 512u < H; col += 1024u) {
    uint4 a[8], b[8], na = make_uint4(0u, 0u, 0u, 0u), nb = na;
    #pragma unroll
    for (unsigned q = 0u; q < 8u; ++q) {
      a[q] = make_uint4(0u, 0u, 0u, 0u); b[q] = a[q];
      if ((hmask >> q) & 1u) {
        a[q] = v44_bld(rr, q * rsb + tb + col * 2u);
        b[q] = v44_bld(rr, q * rsb + tb + (col + 512u) * 2u);
      }
    }
    if (has_n) { na = v44_bld_sys(rn, tb + col * 2u); nb = v44_bld_sys(rn, tb + (col + 512u) * 2u); }
    float sa[8] = {}, sb[8] = {};
    #pragma unroll
    for (unsigned q = 0u; q < 8u; ++q) {
      v42_acc_bf16x8(sa, a[q], ((hmask >> q) & 1u) != 0u);
      v42_acc_bf16x8(sb, b[q], ((hmask >> q) & 1u) != 0u);
    }
    if (has_n) {   // identical to ep16_home_v41: plain fp32 add of the node_recv row
      const unsigned wa[4] = {na.x, na.y, na.z, na.w}, wb[4] = {nb.x, nb.y, nb.z, nb.w};
      #pragma unroll
      for (unsigned k = 0u; k < 4u; ++k) {
        sa[2u * k]      += weave_comm_bf16_val(static_cast<uint16_t>(wa[k] & 65535u));
        sa[2u * k + 1u] += weave_comm_bf16_val(static_cast<uint16_t>(wa[k] >> 16u));
        sb[2u * k]      += weave_comm_bf16_val(static_cast<uint16_t>(wb[k] & 65535u));
        sb[2u * k + 1u] += weave_comm_bf16_val(static_cast<uint16_t>(wb[k] >> 16u));
      }
    }
    *reinterpret_cast<uint4*>(outp + col) = v44_pack_bf16x8(sa);
    *reinterpret_cast<uint4*>(outp + col + 512u) = v44_pack_bf16x8(sb);
  }
  for (; col < H; col += 512u) {
    uint4 a[8], na = make_uint4(0u, 0u, 0u, 0u);
    #pragma unroll
    for (unsigned q = 0u; q < 8u; ++q) {
      a[q] = make_uint4(0u, 0u, 0u, 0u);
      if ((hmask >> q) & 1u) a[q] = v44_bld(rr, q * rsb + tb + col * 2u);
    }
    if (has_n) na = v44_bld_sys(rn, tb + col * 2u);
    float sa[8] = {};
    #pragma unroll
    for (unsigned q = 0u; q < 8u; ++q) v42_acc_bf16x8(sa, a[q], ((hmask >> q) & 1u) != 0u);
    if (has_n) {
      const unsigned wa[4] = {na.x, na.y, na.z, na.w};
      #pragma unroll
      for (unsigned k = 0u; k < 4u; ++k) {
        sa[2u * k]      += weave_comm_bf16_val(static_cast<uint16_t>(wa[k] & 65535u));
        sa[2u * k + 1u] += weave_comm_bf16_val(static_cast<uint16_t>(wa[k] >> 16u));
      }
    }
    *reinterpret_cast<uint4*>(outp + col) = v44_pack_bf16x8(sa);
  }
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __syncwarp();
  if (lane == 0u) ep16_st_agent_relaxed(reinterpret_cast<unsigned*>(o.home_done) + t, 1u);
  __syncwarp();
}
#define V44_NODE_REDUCE(env, o, x, t) v44_node_reduce_fast(env, o, x, t)
#define V44_HOME(env, o, x, t) v44_home_fast(env, o, x, t)
#else
#define V44_NODE_REDUCE(env, o, x, t) ep16_node_reduce(env, o, x, t, 0u, 1u)
#define V44_HOME(env, o, x, t) ep16_home_v41(env, o, x, t)
#endif

// Stage C visit of one 64-token block. Returns bit0 = did work, bit1 = block globally complete.
// r82 fix (C1b): a token whose claim bit (parts word bit 31) is already set is "owned elsewhere" and is
// NOT a candidate (r81 bug: every wave retried the lowest in-flight token, i.e. one token per block at a
// time globally).  Readiness is polled with relaxed loads; one acquire fence follows a successful claim.
__device__ __forceinline__ unsigned v44_visit_nr(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned blk FI_CTX_PARAM) {
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned t = blk * 64u + lane;
  auto* nrdy = reinterpret_cast<unsigned*>(x.node_ready);
  unsigned st = 0u;  // 0 complete / out of range, 1 ready & unclaimed, 2 not ready or owned by another wave
  if (t < NT && !ep16_load_relaxed(nrdy + t)) {
    st = 2u;
    if ((ep16_load_relaxed(nrdy + NT + t) & 0x80000000u) == 0u &&
        v42_ld_sys_rlx(reinterpret_cast<const unsigned*>(x.node_arrival) + t) >=
            reinterpret_cast<const unsigned*>(x.node_expected)[t])
      st = 1u;
  }
  const uint64_t cand = v42_ballot_eq(st, 1u);
  unsigned mine = 0u;
  if (st == 1u && v44_mbcnt(cand) < V44_TOK_CLAIM) {
    mine = ((__hip_atomic_fetch_or(nrdy + NT + t, 0x80000000u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) &
             0x80000000u) == 0u) ? 1u : 0u;
  }
  uint64_t mm = v42_ballot_eq(mine, 1u);
  const uint64_t open = v44_ballot_nz(st) & ~mm;
  const unsigned did = mm ? 1u : 0u;
  if (mm) V44_CLAIM_ACQ();   // pairs with the producers' system-scope release of node_arrival
  while (mm) {
    const unsigned l = (unsigned)__builtin_ctzll(mm);
    mm &= (mm - 1ull);
    const unsigned t2 = blk * 64u + l;
    FI_BG(V1988_FI_CORE, 41u, t2);
    V44_NODE_REDUCE(env, o, x, t2);
    FI_EG(V1988_FI_CORE, 41u, t2);
  }
  return did | (open ? 0u : 2u);
}

// relaxed twin of ep16_home_ready_v41 (the acquire is issued once by the claiming wave)
__device__ __forceinline__ bool v44_home_ready_rlx(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned t) {
  const auto& e = env.base;
  const unsigned hmask = reinterpret_cast<const unsigned*>(o.home_expected)[t];
  const unsigned node_base = (e.rank >> 3u) * 8u;
  for (unsigned j = 0u; j < 8u; ++j) {
    if (hmask & (1u << j)) {
      if (!v42_ld_sys_rlx(reinterpret_cast<const unsigned*>(x.return_ready) +
                          static_cast<uint64_t>(node_base + j) * e.capacity + t))
        return false;
    }
  }
  if (hmask & 256u) {
    if (!v42_ld_agent_rlx(reinterpret_cast<const unsigned*>(x.node_recv_ready) + t)) return false;
  }
  return true;
}

// Stage D visit of one 64-token block (home tokens t < t_local).
__device__ __forceinline__ unsigned v44_visit_home(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned blk FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned t = blk * 64u + lane;
  auto* hdone = reinterpret_cast<unsigned*>(o.home_done);
  unsigned st = 0u;
  if (t < e.t_local && !ep16_load_agent_relaxed(hdone + t)) {
    st = 2u;
    if ((ep16_load_agent_relaxed(hdone + e.t_local + t) & 0x80000000u) == 0u && v44_home_ready_rlx(env, o, x, t))
      st = 1u;
  }
  const uint64_t cand = v42_ballot_eq(st, 1u);
  unsigned mine = 0u;
  if (st == 1u && v44_mbcnt(cand) < V44_TOK_CLAIM) {
    mine = ((__hip_atomic_fetch_or(hdone + e.t_local + t, 0x80000000u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) &
             0x80000000u) == 0u) ? 1u : 0u;
  }
  uint64_t mm = v42_ballot_eq(mine, 1u);
  const uint64_t open = v44_ballot_nz(st) & ~mm;
  const unsigned did = mm ? 1u : 0u;
  if (mm) V44_CLAIM_ACQ();   // pairs with the system-scope release of return_ready / node_recv_ready
  while (mm) {
    const unsigned l = (unsigned)__builtin_ctzll(mm);
    mm &= (mm - 1ull);
    const unsigned t2 = blk * 64u + l;
    FI_BG(V1988_FI_CORE, 12u, t2);
    V44_HOME(env, o, x, t2);
    FI_EG(V1988_FI_CORE, 12u, t2);
  }
  return did | (open ? 0u : 2u);
}

// Stage B: one chunk-major cursor claim. Returns bit0 = did work, bit1 = all MBs' scatter complete.
__device__ __forceinline__ unsigned v44_scatter_step(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned& g2_seen, unsigned& sc_seen FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned M = v41_num_mb(NT);
  const unsigned my_node = e.rank >> 3u;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
  unsigned all_done = 1u, did = 0u;
  for (unsigned b = 0u; b < M; ++b) {
#if V44_QUIET
    if ((sc_seen >> b) & 1u) continue;
    if (v42_ld_agent_rlx(pf + 96u + b) != 0u) { sc_seen |= 1u << b; continue; }  // completion only, no data read
#else
    if (v42_ld_agent_nz_acq(pf + 96u + b) != 0u) continue;  // MB b scatter complete
#endif
    all_done = 0u;
    if (did) break;
#if V44_QUIET
    if (((g2_seen >> b) & 1u) == 0u) {
      if (v42_ld_agent_rlx(pf + 48u + b) == 0u) continue;  // MB b GEMM2 not globally complete yet
      v42_acq_agent();  // once per MB per wave
      g2_seen |= 1u << b;
    }
#else
    if (v42_ld_agent_nz_acq(pf + 48u + b) == 0u) continue;  // MB b GEMM2 not globally complete yet
#endif
    unsigned cb_beg, cb_end, t_beg, t_end;
    v41_mb_range(NT, b, M, cb_beg, cb_end, t_beg, t_end);
    const unsigned nt_b = t_end - t_beg;
    const unsigned total = ((nt_b + 63u) >> 6u) * 1024u;
    unsigned c0 = 0u;
    if (lane == 0u) {
      c0 = ep16_load_agent_relaxed(pf + V44_PF_SCAT_CUR + b);   // skip the RMW once the MB is fully claimed
      if (c0 < total)
        c0 = __hip_atomic_fetch_add(pf + V44_PF_SCAT_CUR + b, V44_SCAT_CLAIM, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    c0 = __builtin_amdgcn_readfirstlane(c0);
    if (c0 >= total) continue;  // MB b fully claimed; other waves are finishing it
    did = 1u;
    unsigned ready = 0u, my_src = 0u, my_t = 0u, my_nexp = 0u;
    const unsigned it = c0 + lane;
    if (lane < V44_SCAT_CLAIM && it < total) {
      const unsigned c = it >> 10u, r = it & 1023u, so = r >> 6u, tt = r & 63u;
      const unsigned t = t_beg + c * 64u + tt;
      if (t < t_end) {
        my_src = (so < 8u) ? (((my_node ^ 1u) << 3u) + so) : ((my_node << 3u) + (so - 8u));
        my_t = t;
        my_nexp = local_nexp[static_cast<uint64_t>(my_src) * NT + t];
        if (my_nexp != 0u) {
          unsigned* bdp = reinterpret_cast<unsigned*>(o.bucket_done) + (my_src * e.capacity + t);
          unsigned sp = 0u;
          for (;;) {
            unsigned exp_bd = my_nexp;
            if (__hip_atomic_compare_exchange_strong(bdp, &exp_bd, 0xffffffffu, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                     __HIP_MEMORY_SCOPE_AGENT)) { ready = 1u; break; }
            if (exp_bd == 0xffffffffu) break;          // owned elsewhere (cannot happen with the cursor)
            if (++sp > (1u << 20u)) { v983_error(reinterpret_cast<unsigned*>(e.error), 944u); break; }
            __builtin_amdgcn_s_sleep(1);
          }
        }
      }
    }
    uint64_t mr = v42_ballot_eq(ready, 1u);
    while (mr) {   // V43G MTB: <=4 claimed tokens per vmcnt(0) + one MB-counter atomic
      unsigned gsrc[4] = {0u, 0u, 0u, 0u}, gtt[4] = {0u, 0u, 0u, 0u}, gn = 0u;
      #pragma unroll
      for (unsigned g = 0u; g < 4u; ++g) {
        if (mr != 0ull) {
          const unsigned l = (unsigned)__builtin_ctzll(mr);
          mr &= (mr - 1ull);
          const unsigned s2 = __shfl(my_src, l), t2 = __shfl(my_t, l), n2 = __shfl(my_nexp, l);
          FI_BG(V1988_FI_CORE, 9u, s2 * e.capacity + t2);
          ep16_bucket_v41_data(env, o, x, s2, t2, n2);
          FI_EG(V1988_FI_CORE, 9u, s2 * e.capacity + t2);
          gsrc[g] = s2; gtt[g] = t2; gn = g + 1u;
        }
      }
      asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
      __syncwarp();
      if (lane == 0u) {
        #pragma unroll
        for (unsigned g = 0u; g < 4u; ++g)
          if (g < gn) ep16_bucket_v41_flag(env, o, x, gsrc[g], gtt[g]);
        const unsigned prev = __hip_atomic_fetch_sub(pf + 80u + b, gn, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (prev == gn) {
          v41_st_agent_u32(pf + 96u + b, 1u);
          auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
          if (trace) trace[1320u + b] = __builtin_amdgcn_s_memrealtime(); // t_scat_done[b]
        }
      }
      __syncwarp();
    }
  }
  return did | (all_done << 1u);
}

__device__ __forceinline__ void v44_combine_loop(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned worker, bool do_bucket, bool do_node_red, bool do_home FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const unsigned nb_nr = __builtin_amdgcn_readfirstlane((NT + 63u) >> 6u);
  const unsigned nb_h = __builtin_amdgcn_readfirstlane((e.t_local + 63u) >> 6u);
  if (nb_nr > 128u || nb_h > 128u) {
    if (lane == 0u) v983_error(reinterpret_cast<unsigned*>(e.error), 945u);
    return;
  }
  uint64_t nr0 = 0ull, nr1 = 0ull, h0 = 0ull, h1 = 0ull;
  if (do_node_red) { nr0 = v44_lowmask(nb_nr < 64u ? nb_nr : 64u); nr1 = v44_lowmask(nb_nr > 64u ? nb_nr - 64u : 0u); }
  if (do_home)     { h0 = v44_lowmask(nb_h < 64u ? nb_h : 64u);    h1 = v44_lowmask(nb_h > 64u ? nb_h - 64u : 0u); }
  unsigned pos_nr = nb_nr ? (worker % nb_nr) : 0u;
  unsigned pos_h = nb_h ? (worker % nb_h) : 0u;
  bool scat_live = do_bucket;
  unsigned g2_seen = 0u, sc_seen = 0u;
  bool pf0_seen = false;
  const auto start = __builtin_amdgcn_s_memrealtime();

  for (;;) {
    unsigned did = 0u;
    // C. Node Reduce first: completes 64-token return chunks -> inter-node RDMA return fires early
    if (nr0 | nr1) {
      const unsigned blk = __builtin_amdgcn_readfirstlane(v44_next_blk(nr0, nr1, nb_nr, pos_nr));
      const unsigned r = __builtin_amdgcn_readfirstlane(v44_visit_nr(env, o, x, blk FI_CTX_ARG));
      did |= r & 1u;
      if (r & 2u) v44_clear_blk(nr0, nr1, blk);
      pos_nr = blk + 1u;
    }
    // D. Home Combine
#if V44_QUIET
    if ((h0 | h1) && !pf0_seen && v42_ld_agent_rlx(pf + 0u) != 0u) { v42_acq_agent(); pf0_seen = true; }
    if ((h0 | h1) && pf0_seen) {
#else
    if ((h0 | h1) && v42_ld_agent_nz_acq(pf + 0u) != 0u) {
#endif
      const unsigned blk = __builtin_amdgcn_readfirstlane(v44_next_blk(h0, h1, nb_h, pos_h));
      const unsigned r = __builtin_amdgcn_readfirstlane(v44_visit_home(env, o, x, blk FI_CTX_ARG));
      did |= r & 1u;
      if (r & 2u) v44_clear_blk(h0, h1, blk);
      pos_h = blk + 1u;
    }
    // B. Chunk-major scatter claim
    if (scat_live) {
      const unsigned r = __builtin_amdgcn_readfirstlane(v44_scatter_step(env, o, x, g2_seen, sc_seen FI_CTX_ARG));
      did |= r & 1u;
      if (r & 2u) scat_live = false;
    }
    if (!(nr0 | nr1 | h0 | h1) && !scat_live) return;
    unsigned stop = 0u;
    if (lane == 0u) {
      stop = ep16_load_relaxed(reinterpret_cast<unsigned*>(e.error));
      if (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks) {
        v983_error(reinterpret_cast<unsigned*>(e.error), 914u);
        stop = 1u;
      }
    }
    if (__shfl(stop, 0u)) return;
    if (!did) __builtin_amdgcn_s_sleep(2);
  }
}
#endif  // V44_COMBINE

// ---------------------------------------------------------------------------------------------
// opus55 v44 G2: per-MB gather of A into the cached row-sorted shadow (all compute groups).
// One release (L2 writeback) + one agent acquire (buffer_inv sc1) per group per MB.
// ---------------------------------------------------------------------------------------------
#include "v44_cka.hpp"
#if V44_CKA
__device__ __forceinline__ void v44_cka_st16(void* p, WeaveI32x4 v) {
  auto* q = (__attribute__((address_space(1))) uint64_t*)(p);
  __hip_atomic_store(q, static_cast<uint64_t>(static_cast<uint32_t>(v[0])) | (static_cast<uint64_t>(static_cast<uint32_t>(v[1])) << 32u),
                     __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  __hip_atomic_store(q + 1, static_cast<uint64_t>(static_cast<uint32_t>(v[2])) | (static_cast<uint64_t>(static_cast<uint32_t>(v[3])) << 32u),
                     __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void v44_cka_st4(void* p, unsigned v) {
  __hip_atomic_store((__attribute__((address_space(1))) unsigned*)(p), v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
#define V44_PF_CKA_CNT 640u   // pf[640+b]: groups done with the MB b gather
#define V44_PF_CKA_RDY 648u   // pf[648+b]: MB b shadow published
__device__ __forceinline__ bool v44_cka_gather(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, unsigned grp_id, unsigned n_comp_groups, unsigned b) {
  const auto& e = env.base;
  const auto& ck = env.ck;
  unsigned tid = threadIdx.x & 255u;
  asm volatile("" : "+v"(tid));
  const unsigned lane = tid & 63u;
  const unsigned wv = __builtin_amdgcn_readfirstlane(tid >> 6u);
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const auto* plan = reinterpret_cast<const volatile WeaveTilePlanTable*>(ck.ck_a);
  const unsigned rows = __builtin_amdgcn_readfirstlane(plan->total_padded_rows);
  const unsigned pad = __builtin_amdgcn_readfirstlane(plan->pad_gather_slot);
  const unsigned RR = ck.ck_route_rows, PB = e.payload_bytes, SG = e.scale_groups;
  if (PB != o.hidden || SG * 128u != o.hidden || rows > RR || (PB & 1023u) != 0u) {
    if (tid == 0u) v983_error(reinterpret_cast<unsigned*>(e.error), 946u);
    return false;
  }
  const int32_t* s1 = reinterpret_cast<const int32_t*>(ck.ck_sorted_token_ids);
  const unsigned char* rp = reinterpret_cast<const unsigned char*>(
      reinterpret_cast<const uint64_t*>(e.peer_payload_table)[e.rank]) - static_cast<uint64_t>(e.rank) * e.capacity * PB;
  const float* rsc = reinterpret_cast<const float*>(
      reinterpret_cast<const uint64_t*>(e.peer_scale_table)[e.rank]) - static_cast<uint64_t>(e.rank) * e.capacity * SG;
  const WeaveI32x4 ra = weave_make_buf_rsrc(rp, e.world * e.capacity * PB);
  const WeaveI32x4 rs = weave_make_buf_rsrc(rsc, e.world * e.capacity * SG * 4u);
  unsigned char* da = v44_cka_a(env);
  float* ds = v44_cka_s(env);
  int32_t* di = v44_cka_ids(env);
  const unsigned nw = n_comp_groups * 4u;
  for (unsigned r = grp_id * 4u + wv; r < rows; r += nw) {
    const unsigned key = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(s1[r]));
    const bool ok = key < pad;
    if (lane == 0u) v44_cka_st4(di + r, ok ? r : RR);
    if (!ok) continue;
    unsigned char* dst = da + static_cast<uint64_t>(r) * PB;
    const uint32_t sb = key * PB;
    unsigned off = lane * 16u;
    // r83 fix (K2): system-scope write-through stores (global_store_dwordx2 sc0 sc1), mirroring the
    // GEMM1->GEMM2 hand-off (writer bypasses its XCD L2, reader issues buffer_inv sc1); r82 used plain
    // write-back stores + one __threadfence per group and failed FP64 on all 16 ranks.
    for (; off + 3072u < PB; off += 4096u) {
      const WeaveI32x4 v0 = weave_raw_buf_load_i32x4(ra, static_cast<int32_t>(sb + off), 0, 16);
      const WeaveI32x4 v1 = weave_raw_buf_load_i32x4(ra, static_cast<int32_t>(sb + off + 1024u), 0, 16);
      const WeaveI32x4 v2 = weave_raw_buf_load_i32x4(ra, static_cast<int32_t>(sb + off + 2048u), 0, 16);
      const WeaveI32x4 v3 = weave_raw_buf_load_i32x4(ra, static_cast<int32_t>(sb + off + 3072u), 0, 16);
      v44_cka_st16(dst + off, v0);
      v44_cka_st16(dst + off + 1024u, v1);
      v44_cka_st16(dst + off + 2048u, v2);
      v44_cka_st16(dst + off + 3072u, v3);
    }
    for (; off < PB; off += 1024u) {
      v44_cka_st16(dst + off, weave_raw_buf_load_i32x4(ra, static_cast<int32_t>(sb + off), 0, 16));
    }
    for (unsigned g = lane; g < SG; g += 64u) {
      v44_cka_st4(ds + static_cast<uint64_t>(r) * SG + g,
                  __builtin_bit_cast(unsigned, weave_raw_buf_load_f32(rs, static_cast<int32_t>((key * SG + g) * 4u), 0, 16)));
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (tid == 0u) {
    __threadfence();  // release this XCD's L2 (shadow rows + ids) once per group
    const unsigned d = __hip_atomic_fetch_add(pf + V44_PF_CKA_CNT + b, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (d == n_comp_groups) v41_st_agent_u32(pf + V44_PF_CKA_RDY + b, 1u);
    const auto start = __builtin_amdgcn_s_memrealtime();
    while (v42_ld_agent_rlx(pf + V44_PF_CKA_RDY + b) == 0u) {
      if (ep16_load_relaxed(reinterpret_cast<unsigned*>(e.error))) break;
      if (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks) { v983_error(reinterpret_cast<unsigned*>(e.error), 947u); break; }
      __builtin_amdgcn_s_sleep(1);
    }
    v42_acq_agent();  // ONE agent acquire (buffer_inv sc1) per group per MB
  }
  weave_4wave_barrier_65440(arena);
#if V44_CKA_VERIFY == 2
  // r87 diagnostic (mechanism of the r85 961 ids mismatch).  Counters land in pf[960..991], which the runner
  // prints for every rank on a device error (V43C_DIAG scan_last = pf[960:976] hex, scan_sp = pf[976:992]).
  //   pf[960+8b+0..5] first mismatch {r+1, got(plain), want, got(system-coherent sc0 sc1 load), key, own?}
  //   pf[976+4b+0..3] ids: cross mismatches (plain), of which coherent load == want, of which got == 0, own mismatches
  //   pf[984+4b+0..3] data: cross rows checked, cross mismatches (plain), cross mismatches (coherent), own mismatches
  {
    using AS1U = __attribute__((address_space(1))) unsigned;
    const WeaveI32x4 sa_ = weave_make_buf_rsrc(da, RR * PB);
    const unsigned vg = (grp_id + (n_comp_groups >> 1u)) % n_comp_groups;
    unsigned c_x = 0u, c_coh = 0u, c_zero = 0u, c_own = 0u, d_n = 0u, d_x = 0u, d_xc = 0u, d_own = 0u;
    for (unsigned pass = 0u; pass < 2u; ++pass) {
      const unsigned gsel = pass ? grp_id : vg;
      for (unsigned r = gsel * 4u + wv; r < rows; r += nw) {
        const unsigned key = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(s1[r]));
        const unsigned want = (key < pad) ? r : RR;
        unsigned rv = r;
        asm volatile("" : "+v"(rv));
        const unsigned got = ((const AS1U*)(di))[rv];
        if (got != want) {
          const unsigned coh = __hip_atomic_load(((AS1U*)(di)) + rv, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
          if (pass) { ++c_own; } else { ++c_x; if (coh == want) ++c_coh; if (got == 0u) ++c_zero; }
          if (lane == 0u && atomicCAS(pf + 960u + b * 8u, 0u, r + 1u) == 0u) {
            pf[960u + b * 8u + 1u] = got; pf[960u + b * 8u + 2u] = want; pf[960u + b * 8u + 3u] = coh;
            pf[960u + b * 8u + 4u] = key; pf[960u + b * 8u + 5u] = pass;
          }
        }
        if (key >= pad) continue;
        const unsigned o0 = lane * 16u;
        const WeaveI32x4 a0 = weave_raw_buf_load_i32x4(sa_, static_cast<int32_t>(r * PB + o0), 0, 0);
        const WeaveI32x4 s0 = weave_raw_buf_load_i32x4(ra, static_cast<int32_t>(key * PB + o0), 0, 16);
        const WeaveI32x4 c0 = weave_raw_buf_load_i32x4(sa_, static_cast<int32_t>(r * PB + o0), 0, 17);  // sc0 sc1
        const bool bad_p = a0[0] != s0[0] || a0[1] != s0[1] || a0[2] != s0[2] || a0[3] != s0[3];
        const bool bad_c = c0[0] != s0[0] || c0[1] != s0[1] || c0[2] != s0[2] || c0[3] != s0[3];
        const bool any_p = __ballot(bad_p) != 0ull, any_c = __ballot(bad_c) != 0ull;
        if (pass) { if (any_p) ++d_own; } else { ++d_n; if (any_p) ++d_x; if (any_c) ++d_xc; }
      }
    }
    if (lane == 0u) {
      if (c_x)   atomicAdd(pf + 976u + b * 4u + 0u, c_x);
      if (c_coh) atomicAdd(pf + 976u + b * 4u + 1u, c_coh);
      if (c_zero) atomicAdd(pf + 976u + b * 4u + 2u, c_zero);
      if (c_own) atomicAdd(pf + 976u + b * 4u + 3u, c_own);
      if (d_n)   atomicAdd(pf + 984u + b * 4u + 0u, d_n);
      if (d_x)   atomicAdd(pf + 984u + b * 4u + 1u, d_x);
      if (d_xc)  atomicAdd(pf + 984u + b * 4u + 2u, d_xc);
      if (d_own) atomicAdd(pf + 984u + b * 4u + 3u, d_own);
    }
    __builtin_amdgcn_s_waitcnt(0);
    weave_4wave_barrier_65440(arena);
    if (tid == 0u) {
      __threadfence();
      const unsigned d = __hip_atomic_fetch_add(pf + 704u + b, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
      if (d == n_comp_groups) {
        const unsigned bad = __hip_atomic_load(pf + 976u + b * 4u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) |
                             __hip_atomic_load(pf + 976u + b * 4u + 3u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) |
                             __hip_atomic_load(pf + 984u + b * 4u + 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) |
                             __hip_atomic_load(pf + 984u + b * 4u + 3u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (bad) v983_error(reinterpret_cast<unsigned*>(e.error), 965u + b);
      }
    }
    weave_4wave_barrier_65440(arena);
  }
#elif V44_CKA_VERIFY
  // r85 diagnostic: every group re-reads rows gathered by ANOTHER group (other XCD with high probability)
  // through the same cached buffer path GEMM1 uses (aux=0) and compares them with the UC source.
  //   961 = ids mismatch, 962 = A-row bytes mismatch (first + last 1 KB of the row), 963 = scale mismatch.
  {
    const WeaveI32x4 sa_ = weave_make_buf_rsrc(da, RR * PB);
    const WeaveI32x4 ss_ = weave_make_buf_rsrc(ds, RR * SG * 4u);
    const unsigned vg = (grp_id + (n_comp_groups >> 1u)) % n_comp_groups;
    unsigned bad = 0u;
    for (unsigned r = vg * 4u + wv; r < rows; r += nw) {
      const unsigned key = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(s1[r]));
      const bool ok = key < pad;
      unsigned rv = r;
      asm volatile("" : "+v"(rv));
      const unsigned got = static_cast<unsigned>(((const __attribute__((address_space(1))) int32_t*)(di))[rv]);
      if (got != (ok ? r : RR)) bad |= 1u;
      if (!ok) continue;
      const unsigned o0 = lane * 16u, o1 = PB - 1024u + lane * 16u;
      const WeaveI32x4 a0 = weave_raw_buf_load_i32x4(sa_, static_cast<int32_t>(r * PB + o0), 0, 0);
      const WeaveI32x4 b0 = weave_raw_buf_load_i32x4(ra, static_cast<int32_t>(key * PB + o0), 0, 16);
      const WeaveI32x4 a1 = weave_raw_buf_load_i32x4(sa_, static_cast<int32_t>(r * PB + o1), 0, 0);
      const WeaveI32x4 b1 = weave_raw_buf_load_i32x4(ra, static_cast<int32_t>(key * PB + o1), 0, 16);
      if (a0[0] != b0[0] || a0[1] != b0[1] || a0[2] != b0[2] || a0[3] != b0[3] ||
          a1[0] != b1[0] || a1[1] != b1[1] || a1[2] != b1[2] || a1[3] != b1[3]) bad |= 2u;
      if (lane < SG) {
        const float sv = weave_raw_buf_load_f32(ss_, static_cast<int32_t>((r * SG + lane) * 4u), 0, 0);
        const float sr = weave_raw_buf_load_f32(rs, static_cast<int32_t>((key * SG + lane) * 4u), 0, 16);
        if (__builtin_bit_cast(unsigned, sv) != __builtin_bit_cast(unsigned, sr)) bad |= 4u;
      }
    }
    const bool e1 = __ballot((bad & 1u) != 0u) != 0ull;
    const bool e2 = __ballot((bad & 2u) != 0u) != 0ull;
    const bool e4 = __ballot((bad & 4u) != 0u) != 0ull;
    if (lane == 0u && (e1 || e2 || e4))
      v983_error(reinterpret_cast<unsigned*>(e.error), e1 ? 961u : (e2 ? 962u : 963u));
    weave_4wave_barrier_65440(arena);
  }
#endif
  return true;
}
#endif  // V44_CKA
