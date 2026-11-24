// v45_tail.hpp -- opus55 v45 Milestone 1 (r92_v45a): tail serialization fixes on top of the v44 final
// combine loop (CbFQS2 = V44_COMBINE + V44_FASTTOK + V44_QUIET + V44_SAFE_ST=2).
//
// Everything here is compiled only with -DV45_TAIL=1; with V45_TAIL=0 the header defines only the
// V45_COMBINE_LOOP alias (= v44_combine_loop), so a build without -DV45_* is byte-identical to v44 final.
//
//   T1 (claim):    NodeReduce / HomeCombine visits pick one ready token per try among the first
//                  V45_SEL_WIN ready candidates of the 64-token block, the pick index being a per-wave,
//                  per-visit seed (v44: every wave picked the LOWEST ready token -> 1 winner per block per
//                  round, mean concurrency 1.3).  A lost fetch_or retries the next candidate immediately
//                  (up to V45_TRIES), never sleeps while ready work exists.
//   T2 (priority): one loop iteration runs C (NodeReduce); if C did work the wave re-visits the same block
//                  at once (drain) and skips D/B; otherwise D (Home), with the same drain rule; only when
//                  neither found work it runs one B (Scatter) claim.  v44 ran C,D,B unconditionally, so a
//                  wave spent ~110 us in B for every 9 us of C (scatter duty 63 %, NodeReduce 2.3 %).
//   T3 (scatter):  (V45_SCAT_FAST) a claimed batch (<= V45_SCAT_CLAIM cursor items) is copied with
//                  compiler-tracked buffer loads (sc1, same policy as the inline "global_load ... sc1") and
//                  compiler-tracked buffer stores (sc0 sc1, same write-through policy as the inline
//                  "global_store_dwordx4 ... sc0 sc1"), i.e. no s_waitcnt vmcnt(0) between the loads and the
//                  stores of a token and none between tokens; ONE s_waitcnt vmcnt(0) per batch, then the
//                  owning lanes publish their flags in parallel and lane 0 does one pf[80+b] decrement.
//                  Per-element arithmetic order of the multi-expert sum is identical to ep16_bucket_v41_data.
//   T4 (reduce):   (V45_NR_FAST) v44_node_reduce_fast with the two inline-asm sc0 sc1 stores per 1024 columns
//                  (each ending in s_waitcnt vmcnt(0) under SAFE_ST=2) replaced by compiler-tracked sc0 sc1
//                  buffer stores; one vmcnt(0) before node_ready, as before.  Arithmetic unchanged.
// No LDS, no new arrays indexed at run time, no s_barrier.  Compiler-generated stores are hazard-tracked by
// LLVM (no inline-asm store-data VGPR hazard), so CDNA3 store safety does not depend on V44_ASM_TAIL here.
#pragma once
#ifndef V45_TAIL
#define V45_TAIL 0
#endif
#if V45_TAIL
#if !V44_COMBINE || !V44_FASTTOK || !V44_QUIET
#error "V45_TAIL requires -DV44_COMBINE=1 -DV44_FASTTOK=1 -DV44_QUIET=1"
#endif
#ifndef V45_TRIES
#define V45_TRIES 3u
#endif
#ifndef V45_SEL_WIN
#define V45_SEL_WIN 16u
#endif
#ifndef V45_SCAT_FAST
#define V45_SCAT_FAST 1
#endif
#ifndef V45_NR_FAST
#define V45_NR_FAST 1
#endif
#ifndef V45_SCAT_CLAIM
#define V45_SCAT_CLAIM 8u
#endif
#ifndef V45_PRIO
#define V45_PRIO 1
#endif
static_assert(V45_SCAT_CLAIM >= 1u && V45_SCAT_CLAIM <= 64u, "V45_SCAT_CLAIM in 1..64");

#define V45_SC01 17   // CPol SC0|SC1 on gfx940+ (aux bit0 = sc0, bit4 = sc1)
__device__ void v45_raw_buf_store_i32x4(WeaveI32x4 vdata, WeaveI32x4 srsrc, int32_t voffset, int32_t soffset,
                                        int32_t aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");
__device__ __forceinline__ WeaveI32x4 v45_i4(uint4 v) {
  WeaveI32x4 r;
  r[0] = static_cast<int32_t>(v.x); r[1] = static_cast<int32_t>(v.y);
  r[2] = static_cast<int32_t>(v.z); r[3] = static_cast<int32_t>(v.w);
  return r;
}
__device__ __forceinline__ void v45_bst(WeaveI32x4 rs, uint32_t voff, uint4 v) {
  v45_raw_buf_store_i32x4(v45_i4(v), rs, static_cast<int32_t>(voff), 0, V45_SC01);
}

// ---------------------------------------------------------------------------------------------
// T4: NodeReduce with compiler-tracked sc0 sc1 stores (arithmetic identical to v44_node_reduce_fast)
// ---------------------------------------------------------------------------------------------
#if V45_NR_FAST
__device__ __forceinline__ void v45_node_reduce_fast(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned t) {
  const unsigned lane = threadIdx.x & 63u;
  const unsigned H = o.hidden, NT = x.node_tokens;
  const unsigned mask = __builtin_amdgcn_readfirstlane(reinterpret_cast<const unsigned*>(x.node_rank_mask)[t]);
  const uint32_t rsb = NT * H * 2u;
  const WeaveI32x4 rp = weave_make_buf_rsrc(reinterpret_cast<const void*>(x.node_partial), 8u * rsb);
  const WeaveI32x4 ro = weave_make_buf_rsrc(reinterpret_cast<uint16_t*>(x.node_output) + static_cast<uint64_t>(t) * H, H * 2u);
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
    v45_bst(ro, col * 2u, v44_pack_bf16x8(sa));
    v45_bst(ro, (col + 512u) * 2u, v44_pack_bf16x8(sb));
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
    v45_bst(ro, col * 2u, v44_pack_bf16x8(sa));
  }
  ep16_vmcnt0();
  __syncwarp();
  if (lane == 0u) ep16_st_sys_relaxed(reinterpret_cast<unsigned*>(x.node_ready) + t, 1u);
  __syncwarp();
  (void)env;
}
#define V45_NODE_REDUCE(env, o, x, t) v45_node_reduce_fast(env, o, x, t)
#else
#define V45_NODE_REDUCE(env, o, x, t) V44_NODE_REDUCE(env, o, x, t)
#endif

// ---------------------------------------------------------------------------------------------
// T1: rotated single-token claim with immediate retry
// ---------------------------------------------------------------------------------------------
// st: 0 = complete / out of range, 1 = ready & unclaimed, 2 = not ready or owned by another wave.
// Returns the ballot of tokens this wave now owns (0 or one bit).  `cand` is lane-uniform.
__device__ __forceinline__ uint64_t v45_claim_one(unsigned* claim_word, unsigned st, unsigned seed) {
  const unsigned lane = threadIdx.x & 63u;
  uint64_t cand = v42_ballot_eq(st, 1u);
  #pragma unroll 1
  for (unsigned tr = 0u; tr < V45_TRIES && cand != 0ull; ++tr) {
    const unsigned pc = __builtin_amdgcn_readfirstlane((unsigned)__builtin_popcountll(cand));
    const unsigned win = (pc < V45_SEL_WIN) ? pc : V45_SEL_WIN;
    const unsigned sel = __builtin_amdgcn_readfirstlane((seed + tr) % win);
    const bool pick = (((cand >> lane) & 1ull) != 0ull) && (v44_mbcnt(cand) == sel);
    unsigned won = 0u;
    if (pick)
      won = ((__hip_atomic_fetch_or(claim_word, 0x80000000u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) &
              0x80000000u) == 0u) ? 1u : 0u;
    const uint64_t w = v42_ballot_eq(won, 1u);
    if (w) return w;
    cand &= ~v42_ballot_eq(pick ? 1u : 0u, 1u);   // that token is owned elsewhere -> next candidate now
  }
  return 0ull;
}

__device__ __forceinline__ unsigned v45_visit_nr(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned blk, unsigned seed FI_CTX_PARAM) {
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned t = blk * 64u + lane;
  auto* nrdy = reinterpret_cast<unsigned*>(x.node_ready);
  unsigned st = 0u;
  if (t < NT && !ep16_load_relaxed(nrdy + t)) {
    st = 2u;
    if ((ep16_load_relaxed(nrdy + NT + t) & 0x80000000u) == 0u &&
        v42_ld_sys_rlx(reinterpret_cast<const unsigned*>(x.node_arrival) + t) >=
            reinterpret_cast<const unsigned*>(x.node_expected)[t])
      st = 1u;
  }
  uint64_t mm = v45_claim_one(nrdy + NT + (t < NT ? t : 0u), st, seed);
  const uint64_t open = v44_ballot_nz(st) & ~mm;
  const unsigned did = mm ? 1u : 0u;
  if (mm) V44_CLAIM_ACQ();
  while (mm) {
    const unsigned l = (unsigned)__builtin_ctzll(mm);
    mm &= (mm - 1ull);
    const unsigned t2 = blk * 64u + l;
    FI_BG(V1988_FI_CORE, 41u, t2);
    V45_NODE_REDUCE(env, o, x, t2);
    FI_EG(V1988_FI_CORE, 41u, t2);
  }
  return did | (open ? 0u : 2u);
}

__device__ __forceinline__ unsigned v45_visit_home(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned blk, unsigned seed FI_CTX_PARAM) {
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
  uint64_t mm = v45_claim_one(hdone + e.t_local + (t < e.t_local ? t : 0u), st, seed);
  const uint64_t open = v44_ballot_nz(st) & ~mm;
  const unsigned did = mm ? 1u : 0u;
  if (mm) V44_CLAIM_ACQ();
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

// ---------------------------------------------------------------------------------------------
// T3: scatter data half with compiler-tracked loads/stores (no wait inside; caller drains once per batch)
// ---------------------------------------------------------------------------------------------
#if V45_SCAT_FAST
__device__ __forceinline__ void v45_bucket_data_fast(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned src, unsigned t, unsigned nexp) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned H = o.hidden;
  const unsigned key = src * e.capacity + t;
  const auto* rows = reinterpret_cast<const unsigned*>(o.bucket_route_rows) + static_cast<uint64_t>(key) * e.topk;
  const unsigned my_node = e.rank >> 3u;
  const bool same_node = ((src >> 3u) == my_node);
  uint16_t* dst_base = nullptr;
  if (same_node) {
    dst_base = reinterpret_cast<uint16_t*>(reinterpret_cast<const uint64_t*>(o.peer_return_payload_table)[src]) +
               static_cast<uint64_t>(t) * H;
  } else {
    dst_base = reinterpret_cast<uint16_t*>(reinterpret_cast<const uint64_t*>(x.node_partial_table)[src & 7u]) +
               (static_cast<uint64_t>(e.rank & 7u) * x.node_tokens + t) * H;
  }
  const WeaveI32x4 rd = weave_make_buf_rsrc(dst_base, H * 2u);
  const uint16_t* g2o = reinterpret_cast<const uint16_t*>(o.gemm2_output);
  if (nexp == 1u) {
    // pure copy: 8 x 16 B per lane in flight (8 KB per wave), out-of-range slices are buffer-OOB no-ops
    const unsigned r0 = __builtin_amdgcn_readfirstlane(rows[0]);
    const WeaveI32x4 rg = weave_make_buf_rsrc(g2o + static_cast<uint64_t>(r0) * H, H * 2u);
    for (unsigned c0 = 0u; c0 < H; c0 += 4096u) {
      uint4 v[8];
      #pragma unroll
      for (unsigned s = 0u; s < 8u; ++s) v[s] = v44_bld(rg, (c0 + s * 512u + lane * 8u) * 2u);
      #pragma unroll
      for (unsigned s = 0u; s < 8u; ++s) v45_bst(rd, (c0 + s * 512u + lane * 8u) * 2u, v[s]);
    }
  } else {
    // weighted rows were already applied by GEMM2; plain fp32 sum in k = 0..nexp-1 order (as v41)
    for (unsigned col = lane * 8u; col < H; col += 1024u) {
      float sa[8] = {}, sb[8] = {};
      for (unsigned k = 0u; k < nexp; k += 4u) {
        uint4 a[4], b[4];
        #pragma unroll
        for (unsigned q = 0u; q < 4u; ++q) {
          const unsigned kk = (k + q < nexp) ? k + q : k;
          const unsigned rr = __builtin_amdgcn_readfirstlane(rows[kk]);
          const WeaveI32x4 rg = weave_make_buf_rsrc(g2o + static_cast<uint64_t>(rr) * H, H * 2u);
          a[q] = v44_bld(rg, col * 2u);
          b[q] = v44_bld(rg, (col + 512u) * 2u);
        }
        #pragma unroll
        for (unsigned q = 0u; q < 4u; ++q) {
          v42_acc_bf16x8(sa, a[q], (q == 0u) || (k + q < nexp));
          v42_acc_bf16x8(sb, b[q], (q == 0u) || (k + q < nexp));
        }
      }
      v45_bst(rd, col * 2u, v44_pack_bf16x8(sa));
      v45_bst(rd, (col + 512u) * 2u, v44_pack_bf16x8(sb));   // OOB no-op when col + 512 >= H
    }
  }
}
#define V45_BUCKET_DATA(env, o, x, s, t, n) v45_bucket_data_fast(env, o, x, s, t, n)
#else
#define V45_BUCKET_DATA(env, o, x, s, t, n) ep16_bucket_v41_data(env, o, x, s, t, n)
#endif

// Stage B: one chunk-major cursor claim (same cursor, gating and CAS as v44_scatter_step, V44_QUIET path);
// the whole claimed batch shares ONE s_waitcnt vmcnt(0) and ONE pf[80+b] decrement.
__device__ __forceinline__ unsigned v45_scatter_step(
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
    if ((sc_seen >> b) & 1u) continue;
    if (v42_ld_agent_rlx(pf + 96u + b) != 0u) { sc_seen |= 1u << b; continue; }
    all_done = 0u;
    if (did) break;
    if (((g2_seen >> b) & 1u) == 0u) {
      if (v42_ld_agent_rlx(pf + 48u + b) == 0u) continue;  // MB b GEMM2 not globally complete yet
      v42_acq_agent();
      g2_seen |= 1u << b;
    }
    unsigned cb_beg, cb_end, t_beg, t_end;
    v41_mb_range(NT, b, M, cb_beg, cb_end, t_beg, t_end);
    const unsigned nt_b = t_end - t_beg;
    const unsigned total = ((nt_b + 63u) >> 6u) * 1024u;
    unsigned c0 = 0u;
    if (lane == 0u) {
      c0 = ep16_load_agent_relaxed(pf + V44_PF_SCAT_CUR + b);
      if (c0 < total)
        c0 = __hip_atomic_fetch_add(pf + V44_PF_SCAT_CUR + b, V45_SCAT_CLAIM, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    c0 = __builtin_amdgcn_readfirstlane(c0);
    if (c0 >= total) continue;
    did = 1u;
    unsigned ready = 0u, my_src = 0u, my_t = 0u, my_nexp = 0u;
    const unsigned it = c0 + lane;
    if (lane < V45_SCAT_CLAIM && it < total) {
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
            if (exp_bd == 0xffffffffu) break;
            if (++sp > (1u << 20u)) { v983_error(reinterpret_cast<unsigned*>(e.error), 944u); break; }
            __builtin_amdgcn_s_sleep(1);
          }
        }
      }
    }
    const uint64_t mr = v42_ballot_eq(ready, 1u);
    if (mr == 0ull) continue;
    uint64_t m2 = mr;
    while (m2) {
      const unsigned l = (unsigned)__builtin_ctzll(m2);
      m2 &= (m2 - 1ull);
      const unsigned s2 = __shfl(my_src, l), t2 = __shfl(my_t, l), n2 = __shfl(my_nexp, l);
      FI_BG(V1988_FI_CORE, 9u, s2 * e.capacity + t2);
      V45_BUCKET_DATA(env, o, x, s2, t2, n2);
      FI_EG(V1988_FI_CORE, 9u, s2 * e.capacity + t2);
    }
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");   // every data store of the batch has completed
    __syncwarp();
    if (ready) ep16_bucket_v41_flag(env, o, x, my_src, my_t);   // owning lanes publish in parallel
    __syncwarp();
    if (lane == 0u) {
      const unsigned gn = (unsigned)__builtin_popcountll(mr);
      const unsigned prev = __hip_atomic_fetch_sub(pf + 80u + b, gn, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      if (prev == gn) {
        v41_st_agent_u32(pf + 96u + b, 1u);
        auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
        if (trace) trace[1320u + b] = __builtin_amdgcn_s_memrealtime();
      }
    }
    __syncwarp();
  }
  return did | (all_done << 1u);
}

// ---------------------------------------------------------------------------------------------
// T5 (V45_STREAM): non-blocking streaming scatter sweep (opus55 v45 M3).
//   MB b becomes sweepable at pf[32+b] (its route plan has published local_nexp, the zeroed bucket_done
//   counters and pf[80+b]) instead of pf[48+b] (all of MB b's GEMM2 done).  A token (src,t) is claimed only
//   when bucket_done == nexp, i.e. all its local expert rows have left the GEMM2 epilogue, by the same
//   CAS nexp -> 0xffffffff that gives v44 its exclusivity; a not-yet-finished token is skipped (never spun on).
//   Blocks are 64 tokens of one source, in the v44 cursor order (chunk-major, remote-node sources first);
//   a block leaves the wave's private 128-bit mask once every live token in it has been claimed.  After a
//   successful claim the wave issues an agent acquire (pairs with the GEMM2 epilogue's release on
//   row_n_done), then copies with V45_BUCKET_DATA, one vmcnt(0), parallel flags, one pf[80+b] decrement.
// ---------------------------------------------------------------------------------------------
#ifndef V45_STREAM
#define V45_STREAM 0
#endif
#if V45_STREAM
#ifndef V45_STREAM_CLAIM
#define V45_STREAM_CLAIM 16u
#endif
// st bits: 0 = MB plan seen (masks valid), 1 = local mask empty, 2 = MB globally scattered (pf[96+b])
__device__ __forceinline__ unsigned v45_sweep_mb(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned b,
    uint64_t& m0, uint64_t& m1, unsigned& pos, unsigned& st, unsigned worker, unsigned& seed FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned M = v41_num_mb(NT);
  const unsigned my_node = e.rank >> 3u;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  if (st & 4u) return 0u;
  if (v42_ld_agent_rlx(pf + 96u + b) != 0u) { st |= 4u; return 0u; }
  unsigned cb_beg, cb_end, t_beg, t_end;
  v41_mb_range(NT, b, M, cb_beg, cb_end, t_beg, t_end);
  const unsigned nblk = __builtin_amdgcn_readfirstlane(((t_end - t_beg + 63u) >> 6u) * 16u);
  if ((st & 1u) == 0u) {
    if (v42_ld_agent_rlx(pf + 32u + b) == 0u) return 8u;      // MB b route plan not published yet
    v42_acq_agent();
    m0 = v44_lowmask(nblk < 64u ? nblk : 64u);
    m1 = v44_lowmask(nblk > 64u ? nblk - 64u : 0u);
    pos = nblk ? (worker % nblk) : 0u;
    st |= 1u;
  }
  if (!(m0 | m1)) { st |= 2u; return 8u; }
  const unsigned blk = __builtin_amdgcn_readfirstlane(v44_next_blk(m0, m1, nblk, pos));
  const unsigned c = blk >> 4u, so = blk & 15u;
  const unsigned src = (so < 8u) ? (((my_node ^ 1u) << 3u) + so) : ((my_node << 3u) + (so - 8u));
  const unsigned t = t_beg + c * 64u + lane;
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
  unsigned* bdp = reinterpret_cast<unsigned*>(o.bucket_done) + (src * e.capacity + (t < t_end ? t : t_beg));
  unsigned nexp = 0u, bd = 0xffffffffu;
  if (t < t_end) {
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
    if (((rk + pc - rot) % pc) < V45_STREAM_CLAIM) {
      unsigned exp_bd = nexp;
      won = __hip_atomic_compare_exchange_strong(bdp, &exp_bd, 0xffffffffu, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                 __HIP_MEMORY_SCOPE_AGENT) ? 1u : 0u;
    }
  }
  const uint64_t mr = v42_ballot_eq(won, 1u);
  if (mr == 0ull) { pos = blk; return 8u; }                    // lost every CAS: re-visit this block next time
  v42_acq_agent();
  uint64_t m2 = mr;
  while (m2) {
    const unsigned l = (unsigned)__builtin_ctzll(m2);
    m2 &= (m2 - 1ull);
    const unsigned t2 = t_beg + c * 64u + l, n2 = __shfl(nexp, l);
    FI_BG(V1988_FI_CORE, 9u, src * e.capacity + t2);
    V45_BUCKET_DATA(env, o, x, src, t2, n2);
    FI_EG(V1988_FI_CORE, 9u, src * e.capacity + t2);
  }
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");   // every data store of the batch has completed
  __syncwarp();
  if (won) ep16_bucket_v41_flag(env, o, x, src, t);      // owning lanes publish in parallel
  __syncwarp();
  if (lane == 0u) {
    const unsigned gn = (unsigned)__builtin_popcountll(mr);
    const unsigned prev = __hip_atomic_fetch_sub(pf + 80u + b, gn, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (prev == gn) {
      v41_st_agent_u32(pf + 96u + b, 1u);
      auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
      if (trace) trace[1320u + b] = __builtin_amdgcn_s_memrealtime();
    }
  }
  __syncwarp();
  if ((open & ~mr) == 0ull) { v44_clear_blk(m0, m1, blk); pos = blk + 1u; }
  else pos = blk;                                          // drain the rest of this block next
  return 1u;
}
// returns bit0 = did work, bit1 = scatter globally complete for every MB
__device__ __forceinline__ unsigned v45_scatter_sweep(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    uint64_t& a0, uint64_t& a1, unsigned& ap, unsigned& as, uint64_t& b0, uint64_t& b1, unsigned& bp, unsigned& bs,
    unsigned worker, unsigned& seed FI_CTX_PARAM) {
  const unsigned M = v41_num_mb(x.node_tokens);
  unsigned r = v45_sweep_mb(env, o, x, 0u, a0, a1, ap, as, worker, seed FI_CTX_ARG);
  if (!(r & 1u) && M > 1u) r |= v45_sweep_mb(env, o, x, 1u, b0, b1, bp, bs, worker, seed FI_CTX_ARG);
  if (M > 1u && !(bs & 4u) && (r & 1u) == 0u) {  // refresh MB1 global-done bit even when MB0 did work
    auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
    if (v42_ld_agent_rlx(pf + 97u) != 0u) bs |= 4u;
  }
  const bool all = (as & 4u) && (M < 2u || (bs & 4u));
  return (r & 1u) | (all ? 2u : 0u);
}
#endif  // V45_STREAM

// ---------------------------------------------------------------------------------------------
// T2: priority loop  C (drain) > D (drain) > B
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void v45_combine_loop(
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
  unsigned seed = __builtin_amdgcn_readfirstlane(worker * 7u);
  bool scat_live = do_bucket;
  unsigned g2_seen = 0u, sc_seen = 0u;
#if V45_STREAM
  uint64_t sa0 = 0ull, sa1 = 0ull, sb0 = 0ull, sb1 = 0ull;
  unsigned sap = 0u, sas = 0u, sbp = 0u, sbs = 0u;
  unsigned sseed = __builtin_amdgcn_readfirstlane(worker * 13u);
#endif
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
    if (__shfl(stop, 0u)) return;
    unsigned did = 0u;
    // C. NodeReduce (unlocks the inter-node RDMA return)
    if (nr0 | nr1) {
      const unsigned blk = __builtin_amdgcn_readfirstlane(v44_next_blk(nr0, nr1, nb_nr, pos_nr));
      const unsigned r = __builtin_amdgcn_readfirstlane(v45_visit_nr(env, o, x, blk, seed FI_CTX_ARG));
      seed += 1u;
      if (r & 2u) v44_clear_blk(nr0, nr1, blk);
      if (r & 1u) {
        did = 1u;
        pos_nr = blk;            // drain this block first
#if V45_PRIO
        continue;
#endif
      } else {
        pos_nr = blk + 1u;
      }
    }
    // D. Home Combine
    if ((h0 | h1) && !pf0_seen && v42_ld_agent_rlx(pf + 0u) != 0u) { v42_acq_agent(); pf0_seen = true; }
    if ((h0 | h1) && pf0_seen) {
      const unsigned blk = __builtin_amdgcn_readfirstlane(v44_next_blk(h0, h1, nb_h, pos_h));
      const unsigned r = __builtin_amdgcn_readfirstlane(v45_visit_home(env, o, x, blk, seed FI_CTX_ARG));
      seed += 1u;
      if (r & 2u) v44_clear_blk(h0, h1, blk);
      if (r & 1u) {
        did = 1u;
        pos_h = blk;
#if V45_PRIO
        continue;
#endif
      } else {
        pos_h = blk + 1u;
      }
    }
    // B. Chunk-major scatter claim
    if (scat_live) {
#if V45_STREAM
      const unsigned r = __builtin_amdgcn_readfirstlane(
          v45_scatter_sweep(env, o, x, sa0, sa1, sap, sas, sb0, sb1, sbp, sbs, worker, sseed FI_CTX_ARG));
#else
      const unsigned r = __builtin_amdgcn_readfirstlane(v45_scatter_step(env, o, x, g2_seen, sc_seen FI_CTX_ARG));
#endif
      did |= r & 1u;
      if (r & 2u) scat_live = false;
    }
    if (!did) __builtin_amdgcn_s_sleep(2);
  }
}
#define V45_COMBINE_LOOP v45_combine_loop
#else   // !V45_TAIL
#define V45_COMBINE_LOOP v44_combine_loop
#endif  // V45_TAIL
