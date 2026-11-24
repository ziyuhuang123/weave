#pragma once
#include "v44_safe_st.hpp"  // opus55 v44 S1
#ifndef V44_BM128
#define V44_BM128 0
#endif
#ifndef V44_BM128_MINWORK
#define V44_BM128_MINWORK 1u
#endif
#include <hip/hip_runtime.h>
#include <stdint.h>

struct V41TokenMeta {
  uint16_t eid[12];    // ((dst & 0xfu) << 12u) | (lid & 0xfffu), or 0xffffu if unused
  uint16_t wbits[12];
};
static_assert(sizeof(V41TokenMeta) == 48, "V41TokenMeta size");

__device__ __forceinline__ unsigned v41_kdisp(unsigned nt) {
  if (nt >= 1024u) return 64u;
  if (nt >= 256u)  return 32u;
  if (nt >= 64u)   return 16u;
  const unsigned k = (nt + 3u) >> 2u;
  return k ? k : 1u;
}
__device__ __forceinline__ unsigned v41_num_mb(unsigned nt) {
  // V42 D4: M=2 instead of 4 -> expert weights are re-read from HBM 2x not 4x,
  // M-tiles stay at BM=128 instead of collapsing to BM=64, and MB0/MB1 write
  // DISJOINT halves of gemm2_output so the ping-pong wait disappears entirely.
  const unsigned k = v41_kdisp(nt);
  const unsigned nch = (nt + k - 1u) / k;
  if (nch >= 4u) return 2u;
  return 1u;
}
__device__ __forceinline__ void v41_mb_range(unsigned nt, unsigned b, unsigned M,
                                             unsigned& cb_beg, unsigned& cb_end,
                                             unsigned& t_beg, unsigned& t_end) {
  const unsigned k = v41_kdisp(nt);
  const unsigned nch = (nt + k - 1u) / k;
  cb_beg = (b * nch) / M;
  cb_end = ((b + 1u) * nch) / M;
  t_beg = cb_beg * k;
  t_end = cb_end * k;
  if (t_end > nt) t_end = nt;
}
__device__ __forceinline__ unsigned v41_mb_of_token(unsigned nt, unsigned t, unsigned M) {
  const unsigned k = v41_kdisp(nt);
  const unsigned nch = (nt + k - 1u) / k;
  const unsigned cj = t / k;
  unsigned b = (cj * M) / nch;
  if (b >= M) b = M - 1u;
  return b;
}

__device__ __forceinline__ uint64_t v42_ballot_eq(unsigned v, unsigned cmp) {
  return __builtin_amdgcn_uicmp(v, cmp, 32u); // 32 == ICMP_EQ
}

using v41_u32x4 = unsigned __attribute__((ext_vector_type(4)));
// CRITICAL CDNA3 (gfx942) SAFETY: LLVM SIInsertWaitcnts cannot see VGPR reads inside inline
// asm global_store_*! Without s_waitcnt vmcnt(0), an immediately following VALU overwriting
// the store-data VGPR corrupts lanes (lane&15) in {12..15} of wave64!
// Always pair single stores with s_waitcnt vmcnt(0), and batch 2/3/4 stores into ONE asm block!
__device__ __forceinline__ void v41_st_sys_u4(void* ptr, uint4 v) {
  v41_u32x4 r{v.x, v.y, v.z, v.w};
#if V44_SAFE_ST == 1
  (void)r; v44_st_sys_u4(ptr, v.x, v.y, v.z, v.w);
#else
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n" V44_ASM_TAIL :: "v"(ptr), "v"(r) : "memory");
#endif
}
__device__ __forceinline__ void v41_st2_sys_u4(void* p0, uint4 v0, void* p1, uint4 v1) {
  v41_u32x4 r0{v0.x, v0.y, v0.z, v0.w}, r1{v1.x, v1.y, v1.z, v1.w};
#if V44_SAFE_ST == 1
  (void)r0; (void)r1; v44_st_sys_u4(p0, v0.x, v0.y, v0.z, v0.w); v44_st_sys_u4(p1, v1.x, v1.y, v1.z, v1.w);
#else
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n"
               "global_store_dwordx4 %2, %3, off sc0 sc1\n"
               V44_ASM_TAIL
               :: "v"(p0), "v"(r0), "v"(p1), "v"(r1) : "memory");
#endif
}
__device__ __forceinline__ void v41_st3_sys_u4(void* p0, uint4 v0, void* p1, uint4 v1, void* p2, uint4 v2) {
  v41_u32x4 r0{v0.x, v0.y, v0.z, v0.w}, r1{v1.x, v1.y, v1.z, v1.w}, r2{v2.x, v2.y, v2.z, v2.w};
#if V44_SAFE_ST == 1
  (void)r0; (void)r1; (void)r2; v44_st_sys_u4(p0, v0.x, v0.y, v0.z, v0.w); v44_st_sys_u4(p1, v1.x, v1.y, v1.z, v1.w);
  v44_st_sys_u4(p2, v2.x, v2.y, v2.z, v2.w);
#else
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n"
               "global_store_dwordx4 %2, %3, off sc0 sc1\n"
               "global_store_dwordx4 %4, %5, off sc0 sc1\n"
               V44_ASM_TAIL
               :: "v"(p0), "v"(r0), "v"(p1), "v"(r1), "v"(p2), "v"(r2) : "memory");
#endif
}
__device__ __forceinline__ void v41_st4_sys_u4(void* p0, uint4 v0, void* p1, uint4 v1, void* p2, uint4 v2, void* p3, uint4 v3) {
  v41_u32x4 r0{v0.x, v0.y, v0.z, v0.w}, r1{v1.x, v1.y, v1.z, v1.w};
  v41_u32x4 r2{v2.x, v2.y, v2.z, v2.w}, r3{v3.x, v3.y, v3.z, v3.w};
#if V44_SAFE_ST == 1
  (void)r0; (void)r1; (void)r2; (void)r3; v44_st_sys_u4(p0, v0.x, v0.y, v0.z, v0.w); v44_st_sys_u4(p1, v1.x, v1.y, v1.z, v1.w);
  v44_st_sys_u4(p2, v2.x, v2.y, v2.z, v2.w); v44_st_sys_u4(p3, v3.x, v3.y, v3.z, v3.w);
#else
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n"
               "global_store_dwordx4 %2, %3, off sc0 sc1\n"
               "global_store_dwordx4 %4, %5, off sc0 sc1\n"
               "global_store_dwordx4 %6, %7, off sc0 sc1\n"
               V44_ASM_TAIL
               :: "v"(p0), "v"(r0), "v"(p1), "v"(r1), "v"(p2), "v"(r2), "v"(p3), "v"(r3) : "memory");
#endif
}
__device__ __forceinline__ void v41_st_sys_f32(void* ptr, float v) {
#if V44_SAFE_ST == 1
  v44_st_sys_f32(ptr, v);
#else
  asm volatile("global_store_dword %0, %1, off sc0 sc1\n" V44_ASM_TAIL :: "v"(ptr), "v"(v) : "memory");
#endif
}
__device__ __forceinline__ void v41_st_sys_u64(void* ptr, uint64_t v) {
#if V44_SAFE_ST == 1
  v44_st_sys_u64(ptr, v);
#else
  asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1\n" V44_ASM_TAIL :: "v"(ptr), "v"(v) : "memory");
#endif
}
__device__ __forceinline__ uint4 v41_ld_sc1_u4(const void* ptr) {
  v41_u32x4 r;
  asm volatile("global_load_dwordx4 %0, %1, off sc1\ns_waitcnt vmcnt(0)\n"
               : "=&v"(r) : "v"(ptr) : "memory");
  return make_uint4(r[0], r[1], r[2], r[3]);
}
__device__ __forceinline__ void v41_ld2_sc1_u4(const void* p0, const void* p1, uint4& o0, uint4& o1) {
  v41_u32x4 r0, r1;
  asm volatile("global_load_dwordx4 %0, %2, off sc1\n"
               "global_load_dwordx4 %1, %3, off sc1\n"
               "s_waitcnt vmcnt(0)\n"
               : "=&v"(r0), "=&v"(r1) : "v"(p0), "v"(p1) : "memory");
  o0 = make_uint4(r0[0], r0[1], r0[2], r0[3]);
  o1 = make_uint4(r1[0], r1[1], r1[2], r1[3]);
}
__device__ __forceinline__ void v41_ld3_sc1_u4(const void* p0, const void* p1, const void* p2, uint4& o0, uint4& o1, uint4& o2) {
  v41_u32x4 r0, r1, r2;
  asm volatile("global_load_dwordx4 %0, %3, off sc1\n"
               "global_load_dwordx4 %1, %4, off sc1\n"
               "global_load_dwordx4 %2, %5, off sc1\n"
               "s_waitcnt vmcnt(0)\n"
               : "=&v"(r0), "=&v"(r1), "=&v"(r2) : "v"(p0), "v"(p1), "v"(p2) : "memory");
  o0 = make_uint4(r0[0], r0[1], r0[2], r0[3]);
  o1 = make_uint4(r1[0], r1[1], r1[2], r1[3]);
  o2 = make_uint4(r2[0], r2[1], r2[2], r2[3]);
}
__device__ __forceinline__ void v41_ld4_sc1_u4(const void* p0, const void* p1, const void* p2, const void* p3,
                                               uint4& o0, uint4& o1, uint4& o2, uint4& o3) {
  v41_u32x4 r0, r1, r2, r3;
  asm volatile("global_load_dwordx4 %0, %4, off sc1\n"
               "global_load_dwordx4 %1, %5, off sc1\n"
               "global_load_dwordx4 %2, %6, off sc1\n"
               "global_load_dwordx4 %3, %7, off sc1\n"
               "s_waitcnt vmcnt(0)\n"
               : "=&v"(r0), "=&v"(r1), "=&v"(r2), "=&v"(r3)
               : "v"(p0), "v"(p1), "v"(p2), "v"(p3) : "memory");
  o0 = make_uint4(r0[0], r0[1], r0[2], r0[3]);
  o1 = make_uint4(r1[0], r1[1], r1[2], r1[3]);
  o2 = make_uint4(r2[0], r2[1], r2[2], r2[3]);
  o3 = make_uint4(r3[0], r3[1], r3[2], r3[3]);
}
__device__ __forceinline__ float v41_ld_sc1_f32(const void* ptr) {
  float r;
  asm volatile("global_load_dword %0, %1, off sc1\ns_waitcnt vmcnt(0)\n"
               : "=&v"(r) : "v"(ptr) : "memory");
  return r;
}
__device__ __forceinline__ uint64_t v41_ld_sc1_u64(const void* ptr) {
  uint64_t r;
  asm volatile("global_load_dwordx2 %0, %1, off sc1\ns_waitcnt vmcnt(0)\n"
               : "=&v"(r) : "v"(ptr) : "memory");
  return r;
}
__device__ __forceinline__ unsigned v41_ld_agent_u32(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ unsigned v41_ld_sys_u32(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void v41_st_agent_u32(unsigned* p, unsigned v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void v41_st_sys_u32(unsigned* p, unsigned v) {
  // V42F F1: every caller already stored its data with sc0|sc1 + s_waitcnt vmcnt(0);
  // a relaxed system store (global_store sc0 sc1) needs no buffer_wbl2 sc0 sc1.
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory"); // V42G safety net
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// ---- V42F: relaxed polling + fence-based acquire (no cache maintenance per spin) ----
__device__ __forceinline__ unsigned v42_ld_sys_rlx(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);   // global_load sc0 sc1
}
__device__ __forceinline__ unsigned v42_ld_agent_rlx(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);    // global_load sc1
}
__device__ __forceinline__ void v42_acq_sys() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); }
__device__ __forceinline__ void v42_acq_agent() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); }
__device__ __forceinline__ unsigned v42_ld_sys_nz_acq(const unsigned* p) {
  const unsigned v = v42_ld_sys_rlx(p);
  if (v != 0u) v42_acq_sys();
  return v;
}
__device__ __forceinline__ bool v42_ld_sys_ge_acq(const unsigned* p, unsigned target) {
  const bool ok = v42_ld_sys_rlx(p) >= target;
  if (ok) v42_acq_sys();
  return ok;
}
__device__ __forceinline__ unsigned v42_ld_agent_nz_acq(const unsigned* p) {
  const unsigned v = v42_ld_agent_rlx(p);
  if (v != 0u) v42_acq_agent();
  return v;
}


__device__ __forceinline__ uint16_t v41_pick_u16(uint4 v0, uint4 v1, uint4 v2, unsigned idx) {
  // Compile-time constant idx in 0..23 -> zero stack / scratch!
  switch (idx) {
    case 0u:  return v0.x & 0xffffu; case 1u:  return v0.x >> 16u;
    case 2u:  return v0.y & 0xffffu; case 3u:  return v0.y >> 16u;
    case 4u:  return v0.z & 0xffffu; case 5u:  return v0.z >> 16u;
    case 6u:  return v0.w & 0xffffu; case 7u:  return v0.w >> 16u;
    case 8u:  return v1.x & 0xffffu; case 9u:  return v1.x >> 16u;
    case 10u: return v1.y & 0xffffu; case 11u: return v1.y >> 16u;
    case 12u: return v1.z & 0xffffu; case 13u: return v1.z >> 16u;
    case 14u: return v1.w & 0xffffu; case 15u: return v1.w >> 16u;
    case 16u: return v2.x & 0xffffu; case 17u: return v2.x >> 16u;
    case 18u: return v2.y & 0xffffu; case 19u: return v2.y >> 16u;
    case 20u: return v2.z & 0xffffu; case 21u: return v2.z >> 16u;
    case 22u: return v2.w & 0xffffu; default:  return v2.w >> 16u;
  }
}

// ---------------------------------------------------------------------------
// 1. Local Meta Stage (CTAs 0..3, w5 = 4 waves = 256 threads -> ~1.5 us!)
//    Stages rank m's own 48B/tok V41TokenMeta into RECV_META[m] (sc0 sc1),
//    populates 9-bit home_expected[t] mask, releases RECV_PROGRESS[m].meta_ready!
// ---------------------------------------------------------------------------
__device__ __forceinline__ void v41_stage_local_meta(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x) {
  const auto& e = env.base;
  const unsigned w_id = blockIdx.x; // 0..3
  const unsigned lane = threadIdx.x & 63u;
  const unsigned gtid = w_id * 64u + lane;
  const unsigned NT = x.node_tokens;
  const unsigned m = e.rank;
  const unsigned my_node = m >> 3u;
  const bool include_self = (e.flags & 1u) != 0u;
  const bool fp16 = (e.flags & 2u) != 0u;

  const auto* ids = reinterpret_cast<const int32_t*>(e.topk_ids);
  const auto* weights = reinterpret_cast<const float*>(e.topk_weights);
  const auto* expert_map = reinterpret_cast<const uint32_t*>(e.expert_map);
  auto* home_mask = reinterpret_cast<unsigned*>(o.home_expected);

  const uint64_t MR = static_cast<uint64_t>(V809_META_HEADER) + static_cast<uint64_t>(e.capacity) * V809_META_STRIDE;
  auto* dst_region = reinterpret_cast<unsigned char*>(e.recv_meta) + static_cast<uint64_t>(m) * MR;
  auto* dst_meta = reinterpret_cast<V41TokenMeta*>(dst_region + 64u);

  if (gtid == 0u) {
    v41_st_sys_u64(dst_region, static_cast<uint64_t>(e.t_local));
  }
  for (unsigned t = gtid; t < NT; t += 256u) {
    uint32_t ep0 = 0xffffffffu, ep1 = 0xffffffffu, ep2 = 0xffffffffu;
    uint32_t ep3 = 0xffffffffu, ep4 = 0xffffffffu, ep5 = 0xffffffffu;
    uint32_t wp0 = 0u, wp1 = 0u, wp2 = 0u, wp3 = 0u, wp4 = 0u, wp5 = 0u;
    unsigned hmask = 0u;
    if (t < e.t_local) {
      #pragma unroll
      for (unsigned k = 0u; k < 12u; ++k) {
        if (k < e.topk) {
          const int exp_id = ids[static_cast<uint64_t>(t) * e.topk + k];
          if (exp_id >= 0 && static_cast<unsigned>(exp_id) < e.n_experts) {
            const unsigned mapped = expert_map[exp_id];
            const unsigned dst = mapped >> 16u;
            const unsigned lid = mapped & 0xffffu;
            if (dst < 16u && (include_self || dst != m)) {
              const uint32_t ev = ((dst & 0xfu) << 12u) | (lid & 0xfffu);
              const uint32_t wv = weave_comm_weight_bits(weights[static_cast<uint64_t>(t) * e.topk + k], fp16);
              if ((dst >> 3u) == my_node) hmask |= (1u << (dst & 7u));
              else hmask |= 256u;
              if (k == 0u)  { ep0 = (ep0 & 0xffff0000u) | ev; wp0 = (wp0 & 0xffff0000u) | wv; }
              else if (k == 1u)  { ep0 = (ep0 & 0x0000ffffu) | (ev << 16u); wp0 = (wp0 & 0x0000ffffu) | (wv << 16u); }
              else if (k == 2u)  { ep1 = (ep1 & 0xffff0000u) | ev; wp1 = (wp1 & 0xffff0000u) | wv; }
              else if (k == 3u)  { ep1 = (ep1 & 0x0000ffffu) | (ev << 16u); wp1 = (wp1 & 0x0000ffffu) | (wv << 16u); }
              else if (k == 4u)  { ep2 = (ep2 & 0xffff0000u) | ev; wp2 = (wp2 & 0xffff0000u) | wv; }
              else if (k == 5u)  { ep2 = (ep2 & 0x0000ffffu) | (ev << 16u); wp2 = (wp2 & 0x0000ffffu) | (wv << 16u); }
              else if (k == 6u)  { ep3 = (ep3 & 0xffff0000u) | ev; wp3 = (wp3 & 0xffff0000u) | wv; }
              else if (k == 7u)  { ep3 = (ep3 & 0x0000ffffu) | (ev << 16u); wp3 = (wp3 & 0x0000ffffu) | (wv << 16u); }
              else if (k == 8u)  { ep4 = (ep4 & 0xffff0000u) | ev; wp4 = (wp4 & 0xffff0000u) | wv; }
              else if (k == 9u)  { ep4 = (ep4 & 0x0000ffffu) | (ev << 16u); wp4 = (wp4 & 0x0000ffffu) | (wv << 16u); }
              else if (k == 10u) { ep5 = (ep5 & 0xffff0000u) | ev; wp5 = (wp5 & 0xffff0000u) | wv; }
              else               { ep5 = (ep5 & 0x0000ffffu) | (ev << 16u); wp5 = (wp5 & 0x0000ffffu) | (wv << 16u); }
            }
          }
        }
      }
      home_mask[t] = hmask;
    }
    auto* d4 = reinterpret_cast<uint4*>(dst_meta + t);
    v41_st3_sys_u4(d4 + 0, make_uint4(ep0, ep1, ep2, ep3),
                   d4 + 1, make_uint4(ep4, ep5, wp0, wp1),
                   d4 + 2, make_uint4(wp2, wp3, wp4, wp5));
  }
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __syncwarp();
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  if (lane == 0u) {
    // V42 D3: 48B/token meta was written with sc0|sc1 + s_waitcnt vmcnt(0).
    const unsigned done = __hip_atomic_fetch_add(pf + 250u, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (done == 4u) {
      auto* my_prog = reinterpret_cast<unsigned*>(reinterpret_cast<const uint64_t*>(e.peer_progress_table)[m]);
      v41_st_sys_u32(my_prog, V809_PROGRESS_MAGIC);
      v41_st_agent_u32(pf + 0u, 1u);
      auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
      if (trace) trace[1300u] = __builtin_amdgcn_s_memrealtime(); // t_local_meta_ready
    }
  }
  __syncwarp();
}

// ---------------------------------------------------------------------------
// 2. Local Payload Stage (CTAs 4..11, w4..w7 = 32 waves = 2048 threads -> ~6 us!)
//    Stages rank m's own x & x_scale chunk-by-chunk into RECV_PAYLOAD/SCALE[m] (sc0 sc1),
//    releasing RECV_PAYLOAD_READY[m][t0] = 1 after each chunk so RDMA fires at t~3us!
// ---------------------------------------------------------------------------
__device__ __forceinline__ void v41_stage_local_payload(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned worker /* 0..31 */) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned K41 = v41_kdisp(NT);
  const unsigned nch = (NT + K41 - 1u) / K41;
  const unsigned m = e.rank;
  const unsigned PB = e.payload_bytes;
  const unsigned SG = e.scale_groups;

  const auto* src_x = reinterpret_cast<const unsigned char*>(e.x);
  const auto* src_s = reinterpret_cast<const float*>(e.x_scale);
  auto* dst_x = reinterpret_cast<unsigned char*>(reinterpret_cast<const uint64_t*>(e.peer_payload_table)[m]);
  auto* dst_s = reinterpret_cast<float*>(reinterpret_cast<const uint64_t*>(e.peer_scale_table)[m]);
  auto* my_rdy = reinterpret_cast<unsigned*>(reinterpret_cast<const uint64_t*>(o.peer_payload_ready_table)[m]);
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);

  for (unsigned cj = 0u; cj < nch; ++cj) {
    const unsigned t0 = cj * K41;
    const unsigned ln = (K41 < (NT - t0)) ? K41 : (NT - t0);
    for (unsigned i = worker; i < ln; i += 32u) {
      const unsigned t = t0 + i;
      if (t < e.t_local) {
        const auto* sx = src_x + static_cast<uint64_t>(t) * PB;
        auto* dx = dst_x + static_cast<uint64_t>(t) * PB;
        unsigned off = lane * 16u;
        for (; off + 3072u < PB; off += 4096u) {
          const uint4 v0 = *reinterpret_cast<const uint4*>(sx + off);
          const uint4 v1 = *reinterpret_cast<const uint4*>(sx + off + 1024u);
          const uint4 v2 = *reinterpret_cast<const uint4*>(sx + off + 2048u);
          const uint4 v3 = *reinterpret_cast<const uint4*>(sx + off + 3072u);
          v41_st4_sys_u4(dx + off, v0, dx + off + 1024u, v1, dx + off + 2048u, v2, dx + off + 3072u, v3);
        }
        for (; off + 1024u < PB; off += 2048u) {
          const uint4 v0 = *reinterpret_cast<const uint4*>(sx + off);
          const uint4 v1 = *reinterpret_cast<const uint4*>(sx + off + 1024u);
          v41_st2_sys_u4(dx + off, v0, dx + off + 1024u, v1);
        }
        for (; off < PB; off += 1024u) {
          v41_st_sys_u4(dx + off, *reinterpret_cast<const uint4*>(sx + off));
        }
        const auto* ss = src_s + static_cast<uint64_t>(t) * SG;
        auto* ds = dst_s + static_cast<uint64_t>(t) * SG;
        for (unsigned g = lane; g < SG; g += 64u) {
          v41_st_sys_f32(ds + g, ss[g]);
        }
      }
    }
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __syncwarp();
    if (lane == 0u) {
      // V42 D3: sc0|sc1 stores + s_waitcnt vmcnt(0) above already published the chunk.
      const unsigned d = __hip_atomic_fetch_add(pf + 200u + cj, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
      if (d == 32u) {
        v41_st_sys_u32(my_rdy + t0, 1u);
      }
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// 3. Intra-Node xGMI Pull (CTAs 4..11, w4..w7 = 32 waves)
//    Pulls ONLY needed tokens (local_nexp[src*NT+t] > 0) for MB b=0..M-1 from the
//    7 intra-node neighbor holders into local CACHED RECV_PAYLOAD/SCALE[src][t]!
// ---------------------------------------------------------------------------
__device__ __forceinline__ void v42_pull_wait_mb(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned b, unsigned worker) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned K41 = v41_kdisp(NT);
  const unsigned M = v41_num_mb(NT);
  const unsigned m = e.rank;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const auto* peer_rdy_tbl = reinterpret_cast<const uint64_t*>(o.peer_payload_ready_table);
  const auto start = __builtin_amdgcn_s_memrealtime();

  if (lane == 0u) {
    while (v42_ld_agent_rlx(pf + 1u) < 1u) {
      if (ep16_load_relaxed((unsigned*)e.error) ||
          (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks)) {
        v983_error((unsigned*)e.error, 911u);
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    } v42_acq_agent();
  }
  __syncwarp();

  unsigned cb_beg, cb_end, t_beg, t_end;
  v41_mb_range(NT, b, M, cb_beg, cb_end, t_beg, t_end);
  const unsigned ncb = cb_end - cb_beg;
  for (unsigned w = worker; w < 16u * ncb; w += 32u) {
    const unsigned src = w / ncb;
    const unsigned ci = w - src * ncb;
    if (src == m) continue;
    const unsigned t0 = (cb_beg + ci) * K41;
    if (lane == 0u) {
      const auto* rdy_p = reinterpret_cast<const unsigned*>(peer_rdy_tbl[src]) + t0;
      unsigned sp = 0u;
      while (v42_ld_sys_rlx(rdy_p) < 1u) {
        if ((sp & 63u) == 0u && (ep16_load_relaxed((unsigned*)e.error) ||
            (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks))) {
          v983_error((unsigned*)e.error, 912u);
          break;
        }
        if (sp < 64u) asm volatile("s_nop 0\n" ::: "memory");
        else __builtin_amdgcn_s_sleep(1);
        ++sp;
      } v42_acq_sys();
    }
    __syncwarp();
  }
  if (lane == 0u) {
    const unsigned d = __hip_atomic_fetch_add(pf + 400u + b, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (d == 32u) v41_st_agent_u32(pf + 420u + b, 1u);
  }
  __syncwarp();
}

__device__ __forceinline__ void v42_pull_copy_mb(
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

  if (lane == 0u) {
    while (v42_ld_agent_rlx(pf + 420u + b) == 0u) {
      if (ep16_load_relaxed((unsigned*)e.error) ||
          (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks)) {
        v983_error((unsigned*)e.error, 913u);
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
  const unsigned per_src = (cb_end - cb_beg) * K41;
  const unsigned total = 16u * per_src;
  FI_BG(V1988_FI_CORE, 16u, b);
  for (unsigned w = worker; w < total; w += nworkers) {
    const unsigned src = w / per_src;
    if (src == m || src == partner) continue;
    const unsigned t = t_beg + (w - src * per_src);
    if (t >= t_end || t >= NT) continue;
    if (local_nexp[static_cast<uint64_t>(src) * NT + t] == 0u) continue;
    const auto* sx = reinterpret_cast<const unsigned char*>(peer_pay_tbl[src]) + static_cast<uint64_t>(t) * PB;
    auto* dx = local_pay_base + (static_cast<uint64_t>(src) * e.capacity + t) * PB;
    unsigned off = lane * 16u;
    for (; off + 3072u < PB; off += 4096u) {
      uint4 v0, v1, v2, v3;
      v41_ld4_sc1_u4(sx + off, sx + off + 1024u, sx + off + 2048u, sx + off + 3072u, v0, v1, v2, v3);
      *reinterpret_cast<uint4*>(dx + off) = v0;
      *reinterpret_cast<uint4*>(dx + off + 1024u) = v1;
      *reinterpret_cast<uint4*>(dx + off + 2048u) = v2;
      *reinterpret_cast<uint4*>(dx + off + 3072u) = v3;
    }
    for (; off + 1024u < PB; off += 2048u) {
      uint4 v0, v1;
      v41_ld2_sc1_u4(sx + off, sx + off + 1024u, v0, v1);
      *reinterpret_cast<uint4*>(dx + off) = v0;
      *reinterpret_cast<uint4*>(dx + off + 1024u) = v1;
    }
    for (; off < PB; off += 1024u) {
      *reinterpret_cast<uint4*>(dx + off) = v41_ld_sc1_u4(sx + off);
    }
    const auto* ss = reinterpret_cast<const float*>(peer_scl_tbl[src]) + static_cast<uint64_t>(t) * SG;
    auto* ds = local_scl_base + (static_cast<uint64_t>(src) * e.capacity + t) * SG;
    for (unsigned g = lane; g < SG; g += 64u) {
      ds[g] = v41_ld_sc1_f32(ss + g);
    }
  }
  FI_EG(V1988_FI_CORE, 16u, b);
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __syncwarp();
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

__device__ __forceinline__ void v41_xgmi_pull_loop(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned worker /* 0..31 */ FI_CTX_PARAM) {
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

  // Wait for all16_meta_scanned (pf[1] == 1)
  if (lane == 0u) {
    while (v42_ld_agent_rlx(pf + 1u) < 1u) {
      if (ep16_load_relaxed((unsigned*)e.error) || (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks)) {
        v983_error((unsigned*)e.error, 911u);
        break;
      }
      __builtin_amdgcn_s_sleep(1);
    } v42_acq_agent();
  }
  __syncwarp();
  if (ep16_load_relaxed((unsigned*)e.error)) return;

  const auto* peer_pay_tbl = reinterpret_cast<const uint64_t*>(e.peer_payload_table);
  const auto* peer_scl_tbl = reinterpret_cast<const uint64_t*>(e.peer_scale_table);
  const auto* peer_rdy_tbl = reinterpret_cast<const uint64_t*>(o.peer_payload_ready_table);
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);

  auto* local_pay_base = reinterpret_cast<unsigned char*>(peer_pay_tbl[m]) - static_cast<uint64_t>(m) * e.capacity * PB;
  auto* local_scl_base = reinterpret_cast<float*>(peer_scl_tbl[m]) - static_cast<uint64_t>(m) * e.capacity * SG;

  for (unsigned b = 0u; b < M; ++b) {
    unsigned cb_beg, cb_end, t_beg, t_end;
    v41_mb_range(NT, b, M, cb_beg, cb_end, t_beg, t_end);
    const unsigned ncb = cb_end - cb_beg;
    // Step A: Local-node neighbor holders FIRST (already ready at t~6us!), then remote-node holders!
    const unsigned my_node = m >> 3u;
    for (unsigned pass = 0u; pass < 2u; ++pass) {
      const unsigned node_sel = (pass == 0u) ? my_node : (my_node ^ 1u);
      for (unsigned ci = 0u; ci < ncb; ++ci) {
        const unsigned cj = cb_beg + ci;
        const unsigned t0 = cj * K41;
        const unsigned ln = (K41 < (NT - t0)) ? K41 : (NT - t0);
        for (unsigned j = 0u; j < 8u; ++j) {
          const unsigned src = node_sel * 8u + j;
          if (src == m) continue; // already staged locally!
          // Wait for holder(src)'s chunk readiness flag
          const auto* rdy_p = reinterpret_cast<const unsigned*>(peer_rdy_tbl[src]) + t0;
          if (lane == 0u) {
            unsigned sp = 0u;
            while (v42_ld_sys_rlx(rdy_p) < 1u) {
              if ((sp & 63u) == 0u && (ep16_load_relaxed((unsigned*)e.error) || (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks))) {
                v983_error((unsigned*)e.error, 912u);
                break;
              }
              if (sp < 64u) asm volatile("s_nop 0\n" ::: "memory");
              else __builtin_amdgcn_s_sleep(1);
              ++sp;
            } v42_acq_sys();
          }
          __syncwarp();
          if (ep16_load_relaxed((unsigned*)e.error)) return;
          if (src == partner) continue; // already RDMA-landed in local RECV_PAYLOAD[partner]!

          // Pull ONLY needed tokens of (src, [t0..t0+ln)) over 346.5 GB/s xGMI (4x pipelined loads = 4KB/wave in flight!)
          FI_BG(V1988_FI_CORE, 16u, (b << 16u) | (src << 8u) | (cj & 0xffu));
          const auto* sp_base = reinterpret_cast<const unsigned char*>(peer_pay_tbl[src]);
          const auto* ss_base = reinterpret_cast<const float*>(peer_scl_tbl[src]);
          auto* dp_base = local_pay_base + static_cast<uint64_t>(src) * e.capacity * PB;
          auto* ds_base = local_scl_base + static_cast<uint64_t>(src) * e.capacity * SG;
          for (unsigned i = worker; i < ln; i += 32u) {
            const unsigned t = t0 + i;
            if (local_nexp[static_cast<uint64_t>(src) * NT + t] == 0u) continue;
            const auto* sx = sp_base + static_cast<uint64_t>(t) * PB;
            auto* dx = dp_base + static_cast<uint64_t>(t) * PB;
            unsigned off = lane * 16u;
            for (; off + 3072u < PB; off += 4096u) {
              uint4 v0, v1, v2, v3;
              v41_ld4_sc1_u4(sx + off, sx + off + 1024u, sx + off + 2048u, sx + off + 3072u, v0, v1, v2, v3);
              *reinterpret_cast<uint4*>(dx + off) = v0;
              *reinterpret_cast<uint4*>(dx + off + 1024u) = v1;
              *reinterpret_cast<uint4*>(dx + off + 2048u) = v2;
              *reinterpret_cast<uint4*>(dx + off + 3072u) = v3;
            }
            for (; off + 1024u < PB; off += 2048u) {
              uint4 v0, v1;
              v41_ld2_sc1_u4(sx + off, sx + off + 1024u, v0, v1);
              *reinterpret_cast<uint4*>(dx + off) = v0;
              *reinterpret_cast<uint4*>(dx + off + 1024u) = v1;
            }
            for (; off < PB; off += 1024u) {
              *reinterpret_cast<uint4*>(dx + off) = v41_ld_sc1_u4(sx + off);
            }
            const auto* ss = ss_base + static_cast<uint64_t>(t) * SG;
            auto* ds = ds_base + static_cast<uint64_t>(t) * SG;
            for (unsigned g = lane; g < SG; g += 64u) {
              ds[g] = v41_ld_sc1_f32(ss + g);
            }
          }
          FI_EG(V1988_FI_CORE, 16u, (b << 16u) | (src << 8u) | (cj & 0xffu));
        }
      }
    }
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    __syncwarp();
    if (lane == 0u) {
      __threadfence(); // flush local L2 pulled data to 256MB MALL for all 8 XCDs!
      const unsigned done = __hip_atomic_fetch_add(pf + 260u + b, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
      if (done == 32u) {
        v41_st_agent_u32(pf + 16u + b, 1u);
        auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
        if (trace) trace[1304u + b] = __builtin_amdgcn_s_memrealtime(); // t_pull_done[b]
      }
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// 4. V41 Combine Primitives: ep16_bucket_v41 + ep16_home_ready_v41 + ep16_home_v41
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ep16_bucket_v41(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned src, unsigned t, unsigned nexp) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned key = src * e.capacity + t;
  const auto* rows = reinterpret_cast<const unsigned*>(o.bucket_route_rows) + static_cast<uint64_t>(key) * e.topk;
  const unsigned my_node = e.rank >> 3u;
  const bool same_node = ((src >> 3u) == my_node);

  uint16_t* dst_base = nullptr;
  if (same_node) {
    dst_base = reinterpret_cast<uint16_t*>(reinterpret_cast<const uint64_t*>(o.peer_return_payload_table)[src]) +
               static_cast<uint64_t>(t) * o.hidden;
  } else {
    dst_base = reinterpret_cast<uint16_t*>(reinterpret_cast<const uint64_t*>(x.node_partial_table)[src & 7u]) +
               (static_cast<uint64_t>(e.rank & 7u) * x.node_tokens + t) * o.hidden;
  }

  if (nexp == 1u) {
    const uint16_t* src_row = reinterpret_cast<const uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(rows[0]) * o.hidden;
    for (unsigned col = lane * 8u; col < o.hidden; col += 1024u) {
      if (col + 512u < o.hidden) {
        uint4 v0, v1;
        ep16_ld2_agent_u4(src_row + col, src_row + col + 512u, v0, v1);
        v41_st2_sys_u4(dst_base + col, v0, dst_base + col + 512u, v1);
      } else {
        v41_st_sys_u4(dst_base + col, ep16_ld_agent_u4(src_row + col));
      }
    }
  } else {
    for (unsigned col = lane * 8u; col < o.hidden; col += 512u) {
      float sum[8] = {};
      const uint16_t* g2o = reinterpret_cast<const uint16_t*>(o.gemm2_output) + col;
      for (unsigned k = 0u; k < nexp; k += 4u) { // V42H G2: 4 expert rows per s_waitcnt
        const unsigned k1 = (k + 1u < nexp) ? k + 1u : k;
        const unsigned k2 = (k + 2u < nexp) ? k + 2u : k;
        const unsigned k3 = (k + 3u < nexp) ? k + 3u : k;
        uint4 a0, a1, a2, a3;
        v42_ld4_agent_u4(g2o + static_cast<uint64_t>(rows[k]) * o.hidden, g2o + static_cast<uint64_t>(rows[k1]) * o.hidden,
                         g2o + static_cast<uint64_t>(rows[k2]) * o.hidden, g2o + static_cast<uint64_t>(rows[k3]) * o.hidden,
                         a0, a1, a2, a3);
        v42_acc_bf16x8(sum, a0, true);
        v42_acc_bf16x8(sum, a1, k + 1u < nexp);
        v42_acc_bf16x8(sum, a2, k + 2u < nexp);
        v42_acc_bf16x8(sum, a3, k + 3u < nexp);
      }
      uint4 o0;
      auto* p0 = reinterpret_cast<unsigned*>(&o0);
      #pragma unroll
      for (unsigned j = 0u; j < 4u; ++j) {
        p0[j] = weave_comm_bf16(sum[j * 2u]) | (static_cast<unsigned>(weave_comm_bf16(sum[j * 2u + 1u])) << 16u);
      }
      v41_st_sys_u4(dst_base + col, o0);
    }
  }
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __syncwarp();
  if (lane == 0u) {
    // V42 D3: the bf16 partial row was stored with sc0|sc1 + s_waitcnt vmcnt(0);
    // a system-scope L2 writeback here runs once PER KEY (~3.3k times per rank).
    if (same_node) {
      auto* ph = reinterpret_cast<unsigned*>(reinterpret_cast<const uint64_t*>(o.peer_home_arrival_table)[src]);
      v41_st_sys_u32(ph + t, 1u);
    } else {
      auto* na = reinterpret_cast<unsigned*>(reinterpret_cast<const uint64_t*>(x.node_arrival_table)[src & 7u]);
      __hip_atomic_fetch_add(na + t, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    ep16_st_agent_relaxed(reinterpret_cast<unsigned*>(o.bucket_sent) + key, 1u);
    // Decrement micro-batch remaining scatter counter
    const unsigned M = v41_num_mb(x.node_tokens);
    const unsigned b = v41_mb_of_token(x.node_tokens, t, M);
    auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
    const unsigned prev = __hip_atomic_fetch_sub(pf + 80u + b, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (prev == 1u) {
      v41_st_agent_u32(pf + 96u + b, 1u);
      auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
      if (trace) trace[1320u + b] = __builtin_amdgcn_s_memrealtime(); // t_scat_done[b]
    }
  }
  __syncwarp();
}

// V43G MTB: data half of ep16_bucket_v41 (no completion wait, no flags)
__device__ __forceinline__ void ep16_bucket_v41_data(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned src, unsigned t, unsigned nexp) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned key = src * e.capacity + t;
  const auto* rows = reinterpret_cast<const unsigned*>(o.bucket_route_rows) + static_cast<uint64_t>(key) * e.topk;
  const unsigned my_node = e.rank >> 3u;
  const bool same_node = ((src >> 3u) == my_node);

  uint16_t* dst_base = nullptr;
  if (same_node) {
    dst_base = reinterpret_cast<uint16_t*>(reinterpret_cast<const uint64_t*>(o.peer_return_payload_table)[src]) +
               static_cast<uint64_t>(t) * o.hidden;
  } else {
    dst_base = reinterpret_cast<uint16_t*>(reinterpret_cast<const uint64_t*>(x.node_partial_table)[src & 7u]) +
               (static_cast<uint64_t>(e.rank & 7u) * x.node_tokens + t) * o.hidden;
  }

  if (nexp == 1u) {
    const uint16_t* src_row = reinterpret_cast<const uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(rows[0]) * o.hidden;
    for (unsigned col = lane * 8u; col < o.hidden; col += 1024u) {
      if (col + 512u < o.hidden) {
        uint4 v0, v1;
        ep16_ld2_agent_u4(src_row + col, src_row + col + 512u, v0, v1);
        v41_st2_sys_u4(dst_base + col, v0, dst_base + col + 512u, v1);
      } else {
        v41_st_sys_u4(dst_base + col, ep16_ld_agent_u4(src_row + col));
      }
    }
  } else {
    for (unsigned col = lane * 8u; col < o.hidden; col += 512u) {
      float sum[8] = {};
      const uint16_t* g2o = reinterpret_cast<const uint16_t*>(o.gemm2_output) + col;
      for (unsigned k = 0u; k < nexp; k += 4u) { // V42H G2: 4 expert rows per s_waitcnt
        const unsigned k1 = (k + 1u < nexp) ? k + 1u : k;
        const unsigned k2 = (k + 2u < nexp) ? k + 2u : k;
        const unsigned k3 = (k + 3u < nexp) ? k + 3u : k;
        uint4 a0, a1, a2, a3;
        v42_ld4_agent_u4(g2o + static_cast<uint64_t>(rows[k]) * o.hidden, g2o + static_cast<uint64_t>(rows[k1]) * o.hidden,
                         g2o + static_cast<uint64_t>(rows[k2]) * o.hidden, g2o + static_cast<uint64_t>(rows[k3]) * o.hidden,
                         a0, a1, a2, a3);
        v42_acc_bf16x8(sum, a0, true);
        v42_acc_bf16x8(sum, a1, k + 1u < nexp);
        v42_acc_bf16x8(sum, a2, k + 2u < nexp);
        v42_acc_bf16x8(sum, a3, k + 3u < nexp);
      }
      uint4 o0;
      auto* p0 = reinterpret_cast<unsigned*>(&o0);
      #pragma unroll
      for (unsigned j = 0u; j < 4u; ++j) {
        p0[j] = weave_comm_bf16(sum[j * 2u]) | (static_cast<unsigned>(weave_comm_bf16(sum[j * 2u + 1u])) << 16u);
      }
      v41_st_sys_u4(dst_base + col, o0);
    }
  }
}

// V43G MTB: lane-0 readiness flags of one scattered token. Caller has already issued
// s_waitcnt vmcnt(0) after ALL data stores of the batch.
__device__ __forceinline__ void ep16_bucket_v41_flag(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned src, unsigned t) {
  const auto& e = env.base;
  const unsigned key = src * e.capacity + t;
  if ((src >> 3u) == (e.rank >> 3u)) {
    auto* ph = reinterpret_cast<unsigned*>(reinterpret_cast<const uint64_t*>(o.peer_home_arrival_table)[src]);
    __hip_atomic_store(ph + t, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  } else {
    auto* na = reinterpret_cast<unsigned*>(reinterpret_cast<const uint64_t*>(x.node_arrival_table)[src & 7u]);
    __hip_atomic_fetch_add(na + t, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  ep16_st_agent_relaxed(reinterpret_cast<unsigned*>(o.bucket_sent) + key, 1u);
}

__device__ __forceinline__ bool ep16_home_ready_v41(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned t) {
  const auto& e = env.base;
  const unsigned hmask = reinterpret_cast<const unsigned*>(o.home_expected)[t];
  const unsigned node_base = (e.rank >> 3u) * 8u;
  for (unsigned j = 0u; j < 8u; ++j) {
    if (hmask & (1u << j)) {
      const unsigned r = node_base + j;
      if (!v42_ld_sys_nz_acq(reinterpret_cast<const unsigned*>(x.return_ready) + static_cast<uint64_t>(r) * e.capacity + t))
        return false;
    }
  }
  if (hmask & 256u) {
    if (!v42_ld_agent_nz_acq(reinterpret_cast<const unsigned*>(x.node_recv_ready) + t))
      return false;
  }
  return true;
}

__device__ __forceinline__ void ep16_home_v41(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned t) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned hmask = reinterpret_cast<const unsigned*>(o.home_expected)[t];
  const unsigned node_base = (e.rank >> 3u) * 8u;
  for (unsigned col = lane * 8u; col < o.hidden; col += 512u) {
    float sum[8] = {};
    { // V42H G2: 4 intra-node return rows per s_waitcnt
      const uint16_t* rb = reinterpret_cast<const uint16_t*>(o.return_recv_payload) +
                           (static_cast<uint64_t>(node_base) * e.capacity + t) * o.hidden + col;
      const uint64_t rs = static_cast<uint64_t>(e.capacity) * o.hidden;
      #pragma unroll
      for (unsigned q = 0u; q < 8u; q += 4u) {
        if (!((hmask >> q) & 15u)) continue;
        uint4 a0, a1, a2, a3;
        v42_ld4_agent_u4(rb + (q + 0u) * rs, rb + (q + 1u) * rs, rb + (q + 2u) * rs, rb + (q + 3u) * rs, a0, a1, a2, a3);
        v42_acc_bf16x8(sum, a0, ((hmask >> (q + 0u)) & 1u) != 0u);
        v42_acc_bf16x8(sum, a1, ((hmask >> (q + 1u)) & 1u) != 0u);
        v42_acc_bf16x8(sum, a2, ((hmask >> (q + 2u)) & 1u) != 0u);
        v42_acc_bf16x8(sum, a3, ((hmask >> (q + 3u)) & 1u) != 0u);
      }
    }
    if (hmask & 256u) {
      const uint4 v = ep16_ld_agent_u4(reinterpret_cast<const uint16_t*>(x.node_recv) +
                                       static_cast<uint64_t>(t) * o.hidden + col);
      const unsigned w[4] = {v.x, v.y, v.z, v.w};
      #pragma unroll
      for (unsigned k = 0u; k < 4u; ++k) {
        sum[2u * k]     += weave_comm_bf16_val(static_cast<uint16_t>(w[k] & 65535u));
        sum[2u * k + 1u] += weave_comm_bf16_val(static_cast<uint16_t>(w[k] >> 16u));
      }
    }
    uint4 out_v;
    auto* ow = reinterpret_cast<unsigned*>(&out_v);
    #pragma unroll
    for (unsigned k = 0u; k < 4u; ++k) {
      ow[k] = weave_comm_bf16(sum[2u * k]) | (static_cast<unsigned>(weave_comm_bf16(sum[2u * k + 1u])) << 16u);
    }
    *reinterpret_cast<uint4*>(reinterpret_cast<uint16_t*>(o.final_output) +
                              static_cast<uint64_t>(t) * o.hidden + col) = out_v;
  }
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __syncwarp();
  if (lane == 0u) {
    ep16_st_agent_relaxed(reinterpret_cast<unsigned*>(o.home_done) + t, 1u);
  }
  __syncwarp();
}

// ---------------------------------------------------------------------------
// 5. Unified V41 Combine Worker Loop (shared by RX Watcher, Return Overlap waves,
//    post-Pull xGMI waves, AND post-GEMM reclaimed Compute waves!)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void v41_combine_loop(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned worker, unsigned stride, bool do_rx, bool do_bucket, bool do_node_red, bool do_home FI_CTX_PARAM) {
  const auto& e = env.base;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned NT = x.node_tokens;
  const unsigned M = v41_num_mb(NT);
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
  const auto start = __builtin_amdgcn_s_memrealtime();

  for (;;) {
    unsigned did_work = 0u;
    unsigned pending = 0u;

    // A. Inter-Node RX Flag Drain: 64 lanes test 64 jobs per iteration (<1us reaction!)
    if (do_rx) {
      for (unsigned base = 0u; base < x.rx_count; base += 64u) {
        const unsigned i = base + lane;
        unsigned rdy = 0u;
        if (i < x.rx_count) {
          const auto& r = reinterpret_cast<const Ep16RxJob*>(x.rx_jobs)[i];
          const unsigned cnt = (r.reserved & EP16_RXJ_COUNT_MASK) ? (r.reserved & EP16_RXJ_COUNT_MASK) : 1u;
          auto* pub = reinterpret_cast<unsigned*>(r.publish);
          const bool already = ep16_load_relaxed(pub + (cnt - 1u)) >= r.value;
          if (!already) {
            if (__hip_atomic_load(reinterpret_cast<const uint64_t*>(r.flag), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == x.generation) {
              if (cnt == 1u) {
                v41_st_sys_u32(pub, r.value);
              } else {
                for (unsigned k = 0u; k < cnt; ++k) ep16_st_agent_relaxed(pub + k, r.value);
                __threadfence();
              }
              rdy = 1u;
              { // V42 diag: when RX job 0/1 (partner meta flag, first dispatch chunk) land
                auto* _tr = reinterpret_cast<unsigned long long*>(o.trace);
                if (_tr && i < 2u) _tr[1330u + i] = __builtin_amdgcn_s_memrealtime();
              }
            } else {
              rdy = 2u;
            }
          }
        }
        if (v42_ballot_eq(rdy, 1u)) did_work = 1u;
        pending += (unsigned)__builtin_popcountll(v42_ballot_eq(rdy, 2u));
      }
    }

    // B. Intra-Node xGMI Scatter Push & Cross-Node Partial Push (across all ready MBs!)
    if (do_bucket) {
      for (unsigned b = 0u; b < M; ++b) {
        if (v42_ld_agent_nz_acq(pf + 96u + b) != 0u) continue; // MB b scatter complete!
        if (v42_ld_agent_nz_acq(pf + 32u + b) == 0u) { ++pending; continue; } // MB b plan not ready yet
        unsigned cb_beg, cb_end, t_beg, t_end;
        v41_mb_range(NT, b, M, cb_beg, cb_end, t_beg, t_end);
        const unsigned nt_b = t_end - t_beg;
        const unsigned nkeys = 16u * nt_b;
        for (unsigned base = worker * 64u; base < nkeys; base += stride * 64u) {
          const unsigned idx = base + lane;
          unsigned ready = 0u, my_src = 0u, my_t = 0u, my_nexp = 0u;
          if (idx < nkeys) {
            my_src = idx / nt_b;
            my_t = t_beg + (idx - my_src * nt_b);
            my_nexp = local_nexp[static_cast<uint64_t>(my_src) * NT + my_t];
            if (my_nexp != 0u) {
              const unsigned key = my_src * e.capacity + my_t;
              if (!ep16_load_relaxed(reinterpret_cast<unsigned*>(o.bucket_sent) + key)) {
                const unsigned bd = ep16_load_agent_relaxed(reinterpret_cast<unsigned*>(o.bucket_done) + key);
                if (bd == my_nexp) {
                  unsigned exp_bd = my_nexp;
                  if (__hip_atomic_compare_exchange_strong(reinterpret_cast<unsigned*>(o.bucket_done) + key,
                                                           &exp_bd, 0xffffffffu, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                                           __HIP_MEMORY_SCOPE_AGENT)) {
                    ready = 1u;
                  } else {
                    ready = ep16_load_relaxed(reinterpret_cast<unsigned*>(o.bucket_sent) + key) ? 0u : 2u;
                  }
                } else {
                  ready = 2u;
                }
              }
            }
          }
          pending += (unsigned)__builtin_popcountll(v42_ballot_eq(ready, 2u));
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
              const unsigned bmb = v41_mb_of_token(NT, gtt[0], M);
              const unsigned prev = __hip_atomic_fetch_sub(pf + 80u + bmb, gn, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
              if (prev == gn) {
                v41_st_agent_u32(pf + 96u + bmb, 1u);
                auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
                if (trace) trace[1320u + bmb] = __builtin_amdgcn_s_memrealtime(); // t_scat_done[b]
              }
            }
            __syncwarp();
            did_work = 1u;
          }
        }
      }
    }

    // C. Node Reduce (Mode-2 cross-node 1.0x dedup -> triggers RDMA Return IMMEDIATELY per MB!)
    if (do_node_red) {
      for (unsigned base = worker * 64u; base < NT; base += stride * 64u) {
        const unsigned t = base + lane;
        unsigned ready = 0u;
        if (t < NT && !ep16_load_relaxed(reinterpret_cast<unsigned*>(x.node_ready) + t)) {
          auto* parts = reinterpret_cast<unsigned*>(x.node_ready) + NT;
          if (v42_ld_sys_ge_acq(reinterpret_cast<const unsigned*>(x.node_arrival) + t,
              reinterpret_cast<const unsigned*>(x.node_expected)[t])) {
            if ((__hip_atomic_fetch_or(parts + t, 0x80000000u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) & 0x80000000u) == 0u) {
              ready = 1u;
            } else {
              ready = ep16_load_relaxed(reinterpret_cast<unsigned*>(x.node_ready) + t) ? 0u : 2u;
            }
          } else {
            ready = 2u;
          }
        }
        pending += (unsigned)__builtin_popcountll(v42_ballot_eq(ready, 2u));
        uint64_t mr = v42_ballot_eq(ready, 1u);
        while (mr) {
          const unsigned l = (unsigned)__builtin_ctzll(mr);
          mr &= (mr - 1ull);
          const unsigned t2 = base + l;
          FI_BG(V1988_FI_CORE, 41u, t2);
          ep16_node_reduce(env, o, x, t2, 0u, 1u);
          FI_EG(V1988_FI_CORE, 41u, t2);
          did_work = 1u;
        }
      }
    }

    // D. Home Combine (intra-node returns + remote Mode-2 node_recv -> final_output)
    if (do_home) {
      if (v42_ld_agent_nz_acq(pf + 0u) == 0u) {
        ++pending;
      } else {
        for (unsigned base = worker * 64u; base < e.t_local; base += stride * 64u) {
          const unsigned t = base + lane;
          unsigned ready = 0u;
          if (t < e.t_local && !ep16_load_agent_relaxed(reinterpret_cast<unsigned*>(o.home_done) + t)) {
            auto* parts = reinterpret_cast<unsigned*>(o.home_done) + e.t_local;
            if (ep16_home_ready_v41(env, o, x, t)) {
              if ((__hip_atomic_fetch_or(parts + t, 0x80000000u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) & 0x80000000u) == 0u) {
                ready = 1u;
              } else {
                ready = ep16_load_agent_relaxed(reinterpret_cast<unsigned*>(o.home_done) + t) ? 0u : 2u;
              }
            } else {
              ready = 2u;
            }
          }
          pending += (unsigned)__builtin_popcountll(v42_ballot_eq(ready, 2u));
          uint64_t mr = v42_ballot_eq(ready, 1u);
          while (mr) {
            const unsigned l = (unsigned)__builtin_ctzll(mr);
            mr &= (mr - 1ull);
            const unsigned t2 = base + l;
            FI_BG(V1988_FI_CORE, 12u, t2);
            ep16_home_v41(env, o, x, t2);
            FI_EG(V1988_FI_CORE, 12u, t2);
            did_work = 1u;
          }
        }
      }
    }

    if (!pending) return;
    unsigned stop = 0u;
    if (lane == 0u) {
      stop = ep16_load_relaxed(reinterpret_cast<unsigned*>(e.error));
      if (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks) {
        v983_error(reinterpret_cast<unsigned*>(e.error), 914u);
        stop = 1u;
      }
    }
    if (__shfl(stop, 0u)) return;
    if (!did_work) __builtin_amdgcn_s_sleep(1);
  }
}

// ---------------------------------------------------------------------------
// 6. Compute-Side Meta Wait + All-16 Scan (runs ONCE on all n_comp_groups -> ~20us!)
//    Populates local_nexp[src*NT + t] in o.home_slots & source_bucket_rows[src] = src_t_local!
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool v41_meta_wait_and_scan(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, unsigned grp_id, unsigned n_comp_groups) {
  const auto& e = env.base;
  unsigned tid = threadIdx.x & 255u;
  asm volatile("" : "+v"(tid)); // opus55 E3
  const unsigned NT = x.node_tokens;
  const unsigned m = e.rank;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* src_rows = reinterpret_cast<unsigned*>(e.source_bucket_rows);
  const auto* peer_prog_tbl = reinterpret_cast<const uint64_t*>(e.peer_progress_table);
  const auto* peer_meta_tbl = reinterpret_cast<const uint64_t*>(e.peer_meta_table);
  auto* local_nexp = reinterpret_cast<unsigned*>(o.home_slots);
  const auto start = __builtin_amdgcn_s_memrealtime();

  if (grp_id == 0u) {
    if (tid < 16u) {
      const auto* p_rdy = reinterpret_cast<const unsigned*>(peer_prog_tbl[tid]);
      unsigned sp = 0u;
      while (v42_ld_sys_rlx(p_rdy) != V809_PROGRESS_MAGIC) {
        if ((sp & 63u) == 0u && (ep16_load_relaxed((unsigned*)e.error) || (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks))) {
          v983_error((unsigned*)e.error, 915u);
          break;
        }
        if (sp < 128u) asm volatile("s_nop 0\n" ::: "memory");
        else __builtin_amdgcn_s_sleep(1);
        ++sp;
      } v42_acq_sys();
      if (!ep16_load_relaxed((unsigned*)e.error)) {
        const uint64_t tl = v41_ld_sc1_u64(reinterpret_cast<const void*>(peer_meta_tbl[tid]));
        src_rows[tid] = static_cast<unsigned>(tl);
      }
    }
    __builtin_amdgcn_s_waitcnt(0);
    weave_4wave_barrier_65440(arena);
    if (tid == 0u) {
      auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
      if (trace) trace[1301u] = __builtin_amdgcn_s_memrealtime(); // t_all16_meta_arrived!
      __threadfence();
      v41_st_agent_u32(pf + 5u, ep16_load_relaxed((unsigned*)e.error) ? 2u : 1u);
    }
  }
  // All n_comp_groups wait pf[5]
  if (tid == 0u) {
    while (v42_ld_agent_rlx(pf + 5u) == 0u) {
      if (ep16_load_relaxed((unsigned*)e.error)) break;
      __builtin_amdgcn_s_sleep(1);
    } v42_acq_agent();
  }
  weave_4wave_barrier_65440(arena);
  if (ep16_load_relaxed((unsigned*)e.error) || v41_ld_agent_u32(pf + 5u) != 1u) return false;

  // All n_comp_groups * 256 threads scan 16*NT tokens in ONE pass (~4us!)
  const unsigned total_tok = 16u * NT;
  const unsigned gtid = grp_id * 256u + tid;
  const unsigned gstr = n_comp_groups * 256u;
  for (unsigned idx = gtid; idx < total_tok; idx += gstr) {
    const unsigned src = idx / NT;
    const unsigned t = idx - src * NT;
    unsigned nexp = 0u;
    if (t < src_rows[src]) {
      const auto* tm = reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[src] + 64ull) + t;
      // Pipelined coherent sc1 read of first 32 bytes (covers all 12 uint16_t eids = 24 bytes!)
      uint4 v0, v1;
      v41_ld2_sc1_u4(reinterpret_cast<const unsigned char*>(tm) + 0u,
                     reinterpret_cast<const unsigned char*>(tm) + 16u, v0, v1);
      const uint4 vz = make_uint4(0u, 0u, 0u, 0u);
      #pragma unroll
      for (unsigned k = 0u; k < 12u; ++k) {
        if (k < e.topk) {
          const uint16_t em = v41_pick_u16(v0, v1, vz, k);
          if (em != 0xffffu && (em >> 12u) == m) ++nexp;
        }
      }
    }
    local_nexp[idx] = nexp;
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (tid == 0u) {
    __threadfence();
    const unsigned done = __hip_atomic_fetch_add(pf + 280u, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (done == n_comp_groups) {
      v41_st_agent_u32(pf + 1u, 1u); // unleashes Intra-Node xGMI Pull immediately!
      auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
      if (trace) { trace[1302u] = __builtin_amdgcn_s_memrealtime(); trace[17u] = trace[1302u]; } // compat rp_tot end!
    }
    while (v42_ld_agent_rlx(pf + 1u) == 0u) { __builtin_amdgcn_s_sleep(1); } v42_acq_agent();
  }
  weave_4wave_barrier_65440(arena);
  return true;
}

// ---------------------------------------------------------------------------
// 7. Ultra-Fast Per-Microbatch Route Plan (runs in ~6us on all n_comp_groups!)
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool v41_mb_route_plan(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, unsigned grp_id, unsigned n_comp_groups,
    unsigned b, unsigned M, unsigned& out_g2_base) {
  const auto& e = env.base;
  const auto& ck = env.ck;
  unsigned tid = threadIdx.x & 255u;
  asm volatile("" : "+v"(tid)); // opus55 E3
  const unsigned NT = x.node_tokens;
  const unsigned m = e.rank;
  const unsigned local_experts = ck.ck_local_experts;
  const bool fp16 = (e.flags & 2u) != 0u;

  unsigned cb_beg, cb_end, t_beg, t_end;
  v41_mb_range(NT, b, M, cb_beg, cb_end, t_beg, t_end);
  const unsigned nt_b = t_end - t_beg;
  const unsigned nkeys = 16u * nt_b;

  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const auto* src_rows = reinterpret_cast<const unsigned*>(e.source_bucket_rows);
  const auto* peer_meta_tbl = reinterpret_cast<const uint64_t*>(e.peer_meta_table);
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
  auto* exp_counts = reinterpret_cast<unsigned*>(e.expert_counts);
  auto* exp_cursors = reinterpret_cast<unsigned*>(e.expert_cursors);
  auto* exp_offsets = reinterpret_cast<unsigned*>(e.expert_offsets);
  auto* plan = reinterpret_cast<WeaveTilePlanTable*>(ck.ck_a);
  auto* descs = reinterpret_cast<WeaveMTileDesc*>(ck.ck_a + 64u);

  // Step 1: Zero expert_counts & mb_scatter_rem[b] on grp 0
  if (grp_id == 0u) {
    for (unsigned lid = tid; lid < local_experts; lid += 256u) {
      exp_counts[lid] = 0u;
    }
    if (tid == 0u) pf[80u + b] = 0u;
    __builtin_amdgcn_s_waitcnt(0);
    weave_4wave_barrier_65440(arena);
    if (tid == 0u) {
      __threadfence();
      v41_st_agent_u32(pf + 340u + b, 1u);
    }
  }
  if (tid == 0u) { while (v42_ld_agent_rlx(pf + 340u + b) == 0u) asm volatile("s_nop 0\n" ::: "memory"); v42_acq_agent(); }
  weave_4wave_barrier_65440(arena);

  // Step 2: All n_comp_groups histogram tokens in MB b + count non-empty buckets!
  const unsigned gtid = grp_id * 256u + tid;
  const unsigned gstr = n_comp_groups * 256u;
  unsigned my_nbuckets = 0u;
  for (unsigned idx = gtid; idx < nkeys; idx += gstr) {
    const unsigned src = idx / nt_b;
    const unsigned t = t_beg + (idx - src * nt_b);
    if (t >= src_rows[src]) continue;
    const unsigned nexp = local_nexp[static_cast<uint64_t>(src) * NT + t];
    if (nexp == 0u) continue;
    ++my_nbuckets;
    const auto* tm = reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[src] + 64ull) + t;
    uint4 v0, v1;
    v41_ld2_sc1_u4(reinterpret_cast<const unsigned char*>(tm) + 0u,
                   reinterpret_cast<const unsigned char*>(tm) + 16u, v0, v1);
    const uint4 vz = make_uint4(0u, 0u, 0u, 0u);
    #pragma unroll
    for (unsigned k = 0u; k < 12u; ++k) {
      if (k < e.topk) {
        const uint16_t em = v41_pick_u16(v0, v1, vz, k);
        if (em != 0xffffu && (em >> 12u) == m) {
          const unsigned lid = em & 0xfffu;
          if (lid < local_experts) __hip_atomic_fetch_add(exp_counts + lid, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
  }
  if (my_nbuckets) __hip_atomic_fetch_add(pf + 80u + b, my_nbuckets, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (tid == 0u) {
    __threadfence();
    __hip_atomic_fetch_add(pf + 112u + b, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  }

  // Step 3: Grp 0 builds expert offsets, expert-major WeaveMTileDesc[], zeroes s1_done/row_n_done
  if (grp_id == 0u) {
    if (tid == 0u) {
      while (v42_ld_agent_rlx(pf + 112u + b) < n_comp_groups) asm volatile("s_nop 0\n" ::: "memory"); v42_acq_agent();
      unsigned cur_row = 0u;
      unsigned total_tiles = 0u;
      unsigned total_valid = 0u;
#if V44_BM128
      // opus55 v44 G1: BM=128 tiles, unless that leaves fewer GEMM1 work items than compute groups
      unsigned use128 = 1u;
      {
        unsigned nt128 = 0u;
        for (unsigned lid = 0u; lid < local_experts; ++lid) {
          unsigned rem = exp_counts[lid];
          while (rem > 0u) {
            const unsigned bm = (rem > 96u) ? 128u : ((rem > 32u) ? 64u : 32u);
            rem -= (rem > bm) ? bm : rem;
            ++nt128;
          }
        }
        if (nt128 * (ck.ck_intermediate / 128u) < V44_BM128_MINWORK * n_comp_groups) use128 = 0u;
        pf[620u + b] = use128 + 2u * nt128;  // diagnostic only
      }
#endif
      for (unsigned lid = 0u; lid < local_experts; ++lid) {
        const unsigned cnt = exp_counts[lid];
        exp_offsets[lid] = cur_row;
        exp_cursors[lid] = cur_row;
        total_valid += cnt;
        unsigned rem = cnt;
        while (rem > 0u) {
#if V44_BM128
          unsigned bm = (use128 && rem > 96u) ? 128u : ((rem > 32u) ? 64u : 32u);  // opus55 v44 G1
#else
          unsigned bm = (rem > 32u) ? 64u : 32u;
#endif
          unsigned vr = (rem > bm) ? bm : rem;
          WeaveMTileDesc d{};
          d.expert_id = lid;
          d.row_start = cur_row;
          d.bm = static_cast<uint16_t>(bm);
          d.valid_rows = static_cast<uint16_t>(vr);
          d.tile_idx = total_tiles;
          descs[total_tiles++] = d;
          cur_row += ((vr + 15u) & ~15u);
          rem -= vr;
        }
      }
      exp_offsets[local_experts] = cur_row;
      plan->total_valid_rows = total_valid;
      plan->total_padded_rows = cur_row;
      plan->total_m_tiles = total_tiles;
      plan->pad_gather_slot = e.world * e.capacity;
      unsigned zero_v = 0u;
      asm volatile("" : "+v"(zero_v)); // opus55 E4
      for (unsigned bk = 0u; bk < 4u; ++bk) { plan->num_tiles[bk] = zero_v; plan->bucket_offset[bk] = zero_v; }
      const unsigned half_rr = ck.ck_route_rows >> 1u;
      const unsigned g2_base = (M >= 2u && cur_row <= half_rr) ? ((b & 1u) * half_rr) : 0u; // V42 D4
      pf[300u + b] = g2_base;
    }
    __builtin_amdgcn_s_waitcnt(0);
    weave_4wave_barrier_65440(arena);
    const unsigned nt_m = plan->total_m_tiles;
    auto* s1_done = reinterpret_cast<unsigned*>(o.stage1_tile_done);
    auto* rn_done = reinterpret_cast<unsigned*>(o.row_n_done);
    for (unsigned mt = tid; mt < nt_m; mt += 256u) {
      s1_done[mt] = 0u;
      rn_done[mt] = 0u;
    }
    // Fill padding rows
    const unsigned pad_gather = plan->pad_gather_slot;
    const unsigned padded_rows = plan->total_padded_rows;
    auto* s1_ids = reinterpret_cast<int32_t*>(ck.ck_sorted_token_ids);
    auto* s2_ids = reinterpret_cast<int32_t*>(ck.ck_m64_sorted_ids);
    auto* r_rid = reinterpret_cast<unsigned*>(o.row_route_id);
    auto* r_wt = reinterpret_cast<float*>(o.stage2_weights);
    for (unsigned lid = tid; lid < local_experts; lid += 256u) {
      const unsigned beg = exp_offsets[lid], end = exp_offsets[lid + 1u];
      for (unsigned r = beg + exp_counts[lid]; r < end; ++r) {
        s1_ids[r] = static_cast<int32_t>(pad_gather);
        s2_ids[r] = static_cast<int32_t>(padded_rows);
        r_rid[r] = 0xffffffffu;
        r_wt[r] = 0.0f;
      }
    }
    __builtin_amdgcn_s_waitcnt(0);
    weave_4wave_barrier_65440(arena);
    if (tid == 0u) {
      __threadfence();
      v41_st_agent_u32(pf + 310u + b, 1u);
    }
  }
  if (tid == 0u) { while (v42_ld_agent_rlx(pf + 310u + b) == 0u) asm volatile("s_nop 0\n" ::: "memory"); v42_acq_agent(); }
  weave_4wave_barrier_65440(arena);

  out_g2_base = pf[300u + b];
  const unsigned g2_base = out_g2_base;

  // Step 4: All n_comp_groups scatter stage1_ids, stage2_ids, stage2_weights, row_route_id, bucket_route_rows!
  auto* s1_ids = reinterpret_cast<int32_t*>(ck.ck_sorted_token_ids);
  auto* s2_ids = reinterpret_cast<int32_t*>(ck.ck_m64_sorted_ids);
  auto* r_rid = reinterpret_cast<unsigned*>(o.row_route_id);
  auto* r_wt = reinterpret_cast<float*>(o.stage2_weights);
  auto* b_rows = reinterpret_cast<unsigned*>(o.bucket_route_rows);

  for (unsigned idx = gtid; idx < nkeys; idx += gstr) {
    const unsigned src = idx / nt_b;
    const unsigned t = t_beg + (idx - src * nt_b);
    if (t >= src_rows[src]) continue;
    if (local_nexp[static_cast<uint64_t>(src) * NT + t] == 0u) continue;
    const auto* tm = reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[src] + 64ull) + t;
    uint4 v0, v1, v2;
    v41_ld3_sc1_u4(reinterpret_cast<const unsigned char*>(tm) + 0u,
                   reinterpret_cast<const unsigned char*>(tm) + 16u,
                   reinterpret_cast<const unsigned char*>(tm) + 32u, v0, v1, v2);

    const unsigned key = src * e.capacity + t;
    unsigned k_loc = 0u;
    #pragma unroll
    for (unsigned k = 0u; k < 12u; ++k) {
      if (k < e.topk) {
        const uint16_t em = v41_pick_u16(v0, v1, v2, k);
        if (em != 0xffffu && (em >> 12u) == m) {
          const unsigned lid = em & 0xfffu;
          if (lid < local_experts) {
            const uint16_t wb = v41_pick_u16(v0, v1, v2, 12u + k);
            const unsigned row = __hip_atomic_fetch_add(exp_cursors + lid, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            s1_ids[row] = static_cast<int32_t>(key);
            s2_ids[row] = static_cast<int32_t>(row);
            r_rid[row] = (src * (e.capacity + 1u) + t) | (k_loc << 24u);
            r_wt[row] = weave_weight_value(wb, fp16);
            b_rows[static_cast<uint64_t>(key) * e.topk + k_loc] = g2_base + row;
            ++k_loc;
          }
        }
      }
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (tid == 0u) {
    __threadfence();
    const unsigned done = __hip_atomic_fetch_add(pf + 128u + b, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (done == n_comp_groups) {
      if (pf[80u + b] == 0u) v41_st_agent_u32(pf + 96u + b, 1u);
      v41_st_agent_u32(pf + 32u + b, 1u);
    }
    while (v42_ld_agent_rlx(pf + 32u + b) == 0u) asm volatile("s_nop 0\n" ::: "memory"); v42_acq_agent();
  }
  weave_4wave_barrier_65440(arena);
  return true;
}
