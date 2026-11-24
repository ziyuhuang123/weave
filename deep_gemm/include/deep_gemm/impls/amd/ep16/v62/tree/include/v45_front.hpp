// v45_front.hpp -- opus55 v45 Milestone 2 (r93_v45b): front protocol (meta stage, payload stage, xGMI pull).
//
// Everything here is compiled only with -DV45_META=1 / -DV45_STAGE=1 / -DV45_PULL=1|2; with all three at 0 the
// header is empty, so a build without -DV45_* stays byte-identical to v44 final.
//
//   F1 (V45_META):  the local 48 B/token meta record (V41TokenMeta) is produced by ALL front workers (32 stage
//                   waves of CTA 4..11 + 4*n_comp_groups compute waves, which are idle until meta16 anyway)
//                   instead of the 4 waves of CTA 0..3 w5.  16 lanes per token (4 tokens per wave), lane k < topk
//                   owns (t, k): ids/weights load -> ONE dependent expert_map load (depth 2; v41: depth ~4*topk
//                   per lane).  Lanes 0..11 store the 12 words of the record (sc0 sc1, compiler-tracked buffer
//                   store), home_mask is an OR-reduction over the 16 lanes.  Same bytes as v41_stage_local_meta.
//                   The last of the n_meta workers publishes PROGRESS_MAGIC + pf[0] exactly like v41 (pf[250]).
//   F2 (V45_STAGE): the local payload/scale staging is done by the same front workers, one token per wave per
//                   round (<= 8 x 16 B loads in flight per lane, compiler-tracked; sc0 sc1 buffer stores), and
//                   chunk cj is published by the LAST TOKEN of cj (per-token fetch_add on pf[200+cj]; v41: a
//                   32-wave rendezvous per chunk, chunks strictly serialised).  Store policy (write-through,
//                   s_waitcnt vmcnt(0) before the publishing atomic) is the one v41 relies on.
//   F3 (V45_PULL):  the MB pull copy uses compiler-tracked sc1 buffer loads, 8 x 16 B per lane in flight across
//                   the live tokens of an 8-token batch (v41: inline asm with s_waitcnt vmcnt(0) inside each
//                   4 x 16 B group and a dependent local_nexp load per token).  The (src == m, src == partner)
//                   skip is done at batch-index level.  V45_PULL=1 keeps the MB rendezvous (pf[420+b]);
//                   V45_PULL=2 removes it: each batch waits only for ITS source chunk ready word (chunk-major
//                   batch order), the partner's chunks of MB b are waited by workers < ncb before the pf[260+b]
//                   count, so pf[16+b] keeps its meaning (all MB b data local).
// No LDS, no s_barrier, no new run-time indexed arrays.
#pragma once
#ifndef V45_META
#define V45_META 0
#endif
#ifndef V45_STAGE
#define V45_STAGE 0
#endif
#ifndef V45_PULL
#define V45_PULL 0
#endif
#if V45_META || V45_STAGE || V45_PULL
#if !V45_TAIL
#define V45_SC01 17   // CPol SC0|SC1 on gfx940+
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
#endif
__device__ void v45f_buf_store_i32(int32_t vdata, WeaveI32x4 srsrc, int32_t voffset, int32_t soffset,
                                   int32_t aux) __asm("llvm.amdgcn.raw.buffer.store.i32");
__device__ void v45f_buf_store_f32(float vdata, WeaveI32x4 srsrc, int32_t voffset, int32_t soffset,
                                   int32_t aux) __asm("llvm.amdgcn.raw.buffer.store.f32");
#define V45_OOB 0x80000000u   // voffset >= num_records: load returns 0, store is dropped
#endif

// ---------------------------------------------------------------------------------------------
// F1 + F2: front stage (meta, then payload) -- called by every front worker 0..nw-1
// ---------------------------------------------------------------------------------------------
#if V45_META || V45_STAGE
__device__ __forceinline__ unsigned v45_n_meta(unsigned NT, unsigned nw) {
  const unsigned q = (NT + 3u) >> 2u;
  return (q < nw) ? q : nw;
}
__device__ __forceinline__ void v45_front_stage(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned worker, unsigned nw FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned m = e.rank;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  FI_BG(V1988_FI_CORE, 15u, worker);
#if V45_META
  const unsigned n_meta = v45_n_meta(NT, nw);
  if (worker < n_meta) {
    const unsigned g = lane >> 4u, k = lane & 15u;
    const unsigned my_node = m >> 3u;
    const bool include_self = (e.flags & 1u) != 0u;
    const bool fp16 = (e.flags & 2u) != 0u;
    const unsigned topk = e.topk;
    const auto* ids = reinterpret_cast<const int32_t*>(e.topk_ids);
    const auto* weights = reinterpret_cast<const float*>(e.topk_weights);
    const auto* expert_map = reinterpret_cast<const uint32_t*>(e.expert_map);
    auto* home_mask = reinterpret_cast<unsigned*>(o.home_expected);
    const uint64_t MR = static_cast<uint64_t>(V809_META_HEADER) + static_cast<uint64_t>(e.capacity) * V809_META_STRIDE;
    auto* dst_region = reinterpret_cast<unsigned char*>(e.recv_meta) + static_cast<uint64_t>(m) * MR;
    const WeaveI32x4 rs = weave_make_buf_rsrc(dst_region + 64u, NT * 48u);
    if (worker == 0u && lane == 0u) v41_st_sys_u64(dst_region, static_cast<uint64_t>(e.t_local));
    for (unsigned tb = worker * 4u; tb < NT; tb += nw * 4u) {
      const unsigned t = tb + g;
      unsigned ev = 0xffffu, wv = 0u, hb = 0u;
      if (t < NT && t < e.t_local && k < topk && k < 12u) {
        const int exp_id = ids[static_cast<uint64_t>(t) * topk + k];
        if (exp_id >= 0 && static_cast<unsigned>(exp_id) < e.n_experts) {
          const unsigned mapped = expert_map[exp_id];
          const unsigned dst = mapped >> 16u;
          const unsigned lid = mapped & 0xffffu;
          if (dst < 16u && (include_self || dst != m)) {
            ev = ((dst & 0xfu) << 12u) | (lid & 0xfffu);
            wv = weave_comm_weight_bits(weights[static_cast<uint64_t>(t) * topk + k], fp16);
            hb = ((dst >> 3u) == my_node) ? (1u << (dst & 7u)) : 256u;
          }
        }
      }
      hb |= __shfl_xor(hb, 1, 16); hb |= __shfl_xor(hb, 2, 16);
      hb |= __shfl_xor(hb, 4, 16); hb |= __shfl_xor(hb, 8, 16);
      const unsigned j = (k < 6u) ? k : (k - 6u);           // word k: k<6 -> eid pair j, else wbits pair j
      const unsigned l0 = g * 16u + 2u * j;
      const unsigned e_lo = __shfl(ev, l0), e_hi = __shfl(ev, l0 + 1u);
      const unsigned w_lo = __shfl(wv, l0), w_hi = __shfl(wv, l0 + 1u);
      const unsigned word = (k < 6u) ? ((e_lo & 0xffffu) | (e_hi << 16u)) : ((w_lo & 0xffffu) | (w_hi << 16u));
      if (t < NT) {
        if (k < 12u) v45f_buf_store_i32(static_cast<int32_t>(word), rs, static_cast<int32_t>(t * 48u + k * 4u), 0, V45_SC01);
#if V46_SP
        if (k == 0u && t < e.t_local) __hip_atomic_store(home_mask + t, hb, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);  // opus55 v46 s13 R2
#else
        if (k == 0u && t < e.t_local) home_mask[t] = hb;
#endif
      }
    }
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __syncwarp();
    if (lane == 0u) {
#if V46_SP
      __threadfence();   // opus55 v46 s13 R2: meta + home_mask stores of this wave before the pf[250] count
#endif
      const unsigned done = __hip_atomic_fetch_add(pf + 250u, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
      if (done == n_meta) {
        auto* my_prog = reinterpret_cast<unsigned*>(reinterpret_cast<const uint64_t*>(e.peer_progress_table)[m]);
        v41_st_sys_u32(my_prog, V809_PROGRESS_MAGIC);
        v41_st_agent_u32(pf + 0u, 1u);
        if (trace) trace[1300u] = __builtin_amdgcn_s_memrealtime(); // t_local_meta_ready
      }
    }
    __syncwarp();
  }
#endif
#if V45_STAGE
  {
    const unsigned K41 = v41_kdisp(NT);
    const unsigned nch = (NT + K41 - 1u) / K41;
    const unsigned PB = e.payload_bytes;
    const unsigned SG = e.scale_groups;
    const auto* src_x = reinterpret_cast<const unsigned char*>(e.x);
    const auto* src_s = reinterpret_cast<const float*>(e.x_scale);
    auto* dst_x = reinterpret_cast<unsigned char*>(reinterpret_cast<const uint64_t*>(e.peer_payload_table)[m]);
    auto* dst_s = reinterpret_cast<float*>(reinterpret_cast<const uint64_t*>(e.peer_scale_table)[m]);
    auto* my_rdy = reinterpret_cast<unsigned*>(reinterpret_cast<const uint64_t*>(o.peer_payload_ready_table)[m]);
    const WeaveI32x4 rdx = weave_make_buf_rsrc(dst_x, e.capacity * PB);
    const WeaveI32x4 rds = weave_make_buf_rsrc(dst_s, e.capacity * SG * 4u);
#if V45_META
    const unsigned rot = v45_n_meta(NT, nw) % nw;   // meta workers take their payload tokens last
#else
    const unsigned rot = 0u;
#endif
    const unsigned wr = (worker + nw - rot) % nw;
    for (unsigned t = wr; t < NT; t += nw) {
      if (t < e.t_local) {
        const auto* sx = src_x + static_cast<uint64_t>(t) * PB;
        for (unsigned o0 = 0u; o0 < PB; o0 += 8192u) {
          uint4 v[8];
          #pragma unroll
          for (unsigned s = 0u; s < 8u; ++s) {
            const unsigned off = o0 + s * 1024u + lane * 16u;
            v[s] = (off < PB) ? *reinterpret_cast<const uint4*>(sx + off) : make_uint4(0u, 0u, 0u, 0u);
          }
          #pragma unroll
          for (unsigned s = 0u; s < 8u; ++s) {
            const unsigned off = o0 + s * 1024u + lane * 16u;
            v45_bst(rdx, (off < PB) ? (t * PB + off) : V45_OOB, v[s]);
          }
        }
        for (unsigned g = lane; g < SG; g += 64u) {
          v45f_buf_store_f32(src_s[static_cast<uint64_t>(t) * SG + g], rds, static_cast<int32_t>((t * SG + g) * 4u), 0, V45_SC01);
        }
      }
    }
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __syncwarp();
    if (lane == 0u) {
      for (unsigned t = wr; t < NT; t += nw) {
        const unsigned cj = t / K41;
        const unsigned t0 = cj * K41;
        const unsigned ln = (K41 < (NT - t0)) ? K41 : (NT - t0);
        const unsigned d = __hip_atomic_fetch_add(pf + 200u + cj, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
        if (d == ln) {
          v41_st_sys_u32(my_rdy + t0, 1u);
          const unsigned c = __hip_atomic_fetch_add(pf + 232u, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
          if (trace) {
            const unsigned long long now = __builtin_amdgcn_s_memrealtime();
            if (cj == 0u) trace[1401u] = now;                      // v45: first payload chunk published
            if (c == nch) trace[16u + 4u * 16u + 2u] = now;        // pay_stage := ALL chunks published
          }
        }
      }
    }
    __syncwarp();
  }
#endif
  FI_EG(V1988_FI_CORE, 15u, worker);
}
#endif

// ---------------------------------------------------------------------------------------------
// F3: MLP xGMI pull copy of microbatch b
// ---------------------------------------------------------------------------------------------
#if V45_PULL
#ifndef V45_PULL_BT
#define V45_PULL_BT 8u
#endif
static_assert(V45_PULL_BT >= 1u && V45_PULL_BT <= 8u, "V45_PULL_BT in 1..8");
#define V45_PULL_COPY v45_pull_copy_mb
__device__ __forceinline__ bool v45_spin_rdy(const unsigned* p, const V968DispatchExt& env, const Ep16TurboContext& x,
                                             unsigned long long start, unsigned code) {
  unsigned sp = 0u;
  while (v42_ld_sys_rlx(p) < 1u) {
    if ((sp & 63u) == 0u && (ep16_load_relaxed((unsigned*)env.base.error) ||
        (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks))) {
      v983_error((unsigned*)env.base.error, code);
      return false;
    }
    if (sp < 64u) asm volatile("s_nop 0\n" ::: "memory");
    else __builtin_amdgcn_s_sleep(1);
    ++sp;
  }
  v42_acq_sys();
  return true;
}
__device__ __forceinline__ void v45_pull_copy_mb(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned b, unsigned worker, unsigned nworkers FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned K41 = v41_kdisp(NT);
  const unsigned M = v41_num_mb(NT);
  const unsigned m = e.rank;
  const unsigned partner = m ^ 8u;
  const unsigned PB = e.payload_bytes;
  const unsigned SG = e.scale_groups;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const auto start = __builtin_amdgcn_s_memrealtime();
  const auto* peer_rdy_tbl = reinterpret_cast<const uint64_t*>(o.peer_payload_ready_table);

  if (lane == 0u) {
#if V45_PULL == 1
    const unsigned gate = 420u + b;   // v42: MB rendezvous of the 32 pull_wait workers
    const unsigned code = 913u;
#else
    const unsigned gate = 1u;         // v45: only the all-16 scan (local_nexp valid)
    const unsigned code = 911u;
#endif
    while (v42_ld_agent_rlx(pf + gate) == 0u) {
      if (ep16_load_relaxed((unsigned*)e.error) ||
          (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks)) {
        v983_error((unsigned*)e.error, code);
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    } v42_acq_agent();
  }
  __syncwarp();
  if (ep16_load_relaxed((unsigned*)e.error)) return;

  const auto* peer_pay_tbl = reinterpret_cast<const uint64_t*>(e.peer_payload_table);
  const auto* peer_scl_tbl = reinterpret_cast<const uint64_t*>(e.peer_scale_table);
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
  auto* local_pay_base = reinterpret_cast<unsigned char*>(peer_pay_tbl[m]) - static_cast<uint64_t>(m) * e.capacity * PB;
  auto* local_scl_base = reinterpret_cast<float*>(peer_scl_tbl[m]) - static_cast<uint64_t>(m) * e.capacity * SG;

  unsigned cb_beg, cb_end, t_beg, t_end;
  v41_mb_range(NT, b, M, cb_beg, cb_end, t_beg, t_end);
  const unsigned ncb = cb_end - cb_beg;
  const unsigned BT = V45_PULL_BT;
  const unsigned bpc = (K41 + BT - 1u) / BT;          // batches per (source, chunk)
  const unsigned nb = ncb * 14u * bpc;                 // chunk-major, 14 sources (not m, not partner)
  const unsigned lo_s = (m < partner) ? m : partner, hi_s = (m < partner) ? partner : m;
  const unsigned ns = (PB + 1023u) >> 10u;             // 1 KB rows per token
  const uint32_t pay_rng = e.capacity * PB, scl_rng = e.capacity * SG * 4u;
  FI_BG(V1988_FI_CORE, 16u, b);
  for (unsigned bi = worker; bi < nb; bi += nworkers) {
    const unsigned ci = bi / (14u * bpc);
    const unsigned rr = bi - ci * 14u * bpc;
    const unsigned r = rr / bpc;
    const unsigned sub = rr - r * bpc;
    unsigned src = r;
    if (src >= lo_s) ++src;
    if (src >= hi_s) ++src;
    src = __builtin_amdgcn_readfirstlane(src);
    const unsigned toff = sub * BT;                       // token offset inside the chunk
    const unsigned tb = __builtin_amdgcn_readfirstlane(t_beg + ci * K41 + toff);
#if V45_PULL == 2
    if (lane == 0u) {
      const auto* rdy_p = reinterpret_cast<const unsigned*>(peer_rdy_tbl[src]) + (cb_beg + ci) * K41;
      v45_spin_rdy(rdy_p, env, x, start, 912u);
    }
    __syncwarp();
    if (ep16_load_relaxed((unsigned*)e.error)) break;
#endif
    bool need = false;
    {
      const unsigned tl = tb + lane;
      if (lane < BT && toff + lane < K41 && tl < t_end && tl < NT)
        need = local_nexp[static_cast<uint64_t>(src) * NT + tl] != 0u;
    }
    uint64_t live = __ballot(need);
    if (live == 0ull) continue;
    const WeaveI32x4 rsx = weave_make_buf_rsrc(reinterpret_cast<const void*>(peer_pay_tbl[src]), pay_rng);
    const WeaveI32x4 rdx = weave_make_buf_rsrc(local_pay_base + static_cast<uint64_t>(src) * e.capacity * PB, pay_rng);
    // payload: flat (token, 1 KB row) jobs, 8 per round, all loads of a round before its stores
    {
      uint64_t rem = live;
      unsigned tcur = static_cast<unsigned>(__builtin_ctzll(rem)), s = 0u;
      while (rem) {
        int32_t vo[8];
        #pragma unroll
        for (unsigned k = 0u; k < 8u; ++k) {
          const unsigned off = s * 1024u + lane * 16u;
          vo[k] = (rem != 0ull && off < PB) ? static_cast<int32_t>((tb + tcur) * PB + off) : static_cast<int32_t>(V45_OOB);
          if (rem != 0ull) {
            if (++s == ns) {
              s = 0u;
              rem &= rem - 1ull;
              tcur = rem ? static_cast<unsigned>(__builtin_ctzll(rem)) : 0u;
            }
          }
        }
        WeaveI32x4 v[8];
        #pragma unroll
        for (unsigned k = 0u; k < 8u; ++k) v[k] = weave_raw_buf_load_i32x4(rsx, vo[k], 0, V44_SC1);
        #pragma unroll
        for (unsigned k = 0u; k < 8u; ++k) v45_raw_buf_store_i32x4(v[k], rdx, vo[k], 0, 0);
      }
    }
    // scale: up to 8 live tokens x 64 groups per round
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
          so[k] = (r2 != 0ull && gg < SG) ? static_cast<int32_t>(((tb + tt) * SG + gg) * 4u) : static_cast<int32_t>(V45_OOB);
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
#if V45_PULL == 2
  // the partner's MB b chunks arrive by RDMA directly into this rank's arena (no copy); pf[16+b] certifies them
  for (unsigned ci = worker; ci < ncb; ci += nworkers) {
    if (lane == 0u) {
      const auto* rdy_p = reinterpret_cast<const unsigned*>(peer_rdy_tbl[partner]) + (cb_beg + ci) * K41;
      v45_spin_rdy(rdy_p, env, x, start, 912u);
    }
    __syncwarp();
  }
#endif
  FI_EG(V1988_FI_CORE, 16u, b);
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __syncwarp();
  if (ep16_load_relaxed((unsigned*)e.error)) return;
  if (lane == 0u) {
    __threadfence(); // publish the locally-cached pulled payload to all 8 XCDs
    const unsigned d = __hip_atomic_fetch_add(pf + 260u + b, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (d == nworkers) {
      v41_st_agent_u32(pf + 16u + b, 1u);
      auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
      if (trace) trace[1304u + b] = __builtin_amdgcn_s_memrealtime();
    }
  }
  __syncwarp();
}
#else
#define V45_PULL_COPY v42_pull_copy_mb
#endif
