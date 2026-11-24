#pragma once
// V62 Opus 5.5 High knobs (all default 0 = V61 code)
#ifndef V62_G2_SWZ
#define V62_G2_SWZ 0
#endif
#define V62_G2CX 432u   // [4 passes][4 xcc] per-XCC GEMM2 claim cursors (pf 432..447, free unless V54_XCD_PART)
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
#define V62_DYN_BM 0
#endif
#ifndef V62_G1_SMALL
#define V62_G1_SMALL 1
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
  if (lane == 0u) v46_wait_ge(pf + (is_rem ? V46_P1_BUILD : V46_LBUILD), 1u, env, x, 965u);
  __syncwarp();
  if (ep16_load_relaxed((unsigned*)e.error)) return;
  unsigned tb, te;
  v46_pass_range(p, NT, tb, te);
  const unsigned cb_beg = tb / K41;
  const unsigned cb_end = (te + K41 - 1u) / K41;
  const unsigned ncb = cb_end - cb_beg;
  const auto* peer_pay_tbl = reinterpret_cast<const uint64_t*>(e.peer_payload_table);
  const auto* peer_scl_tbl = reinterpret_cast<const uint64_t*>(e.peer_scale_table);
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
  auto* local_pay_base = reinterpret_cast<unsigned char*>(peer_pay_tbl[m]) - static_cast<uint64_t>(m) * e.capacity * PB;
  auto* local_scl_base = reinterpret_cast<float*>(peer_scl_tbl[m]) - static_cast<uint64_t>(m) * e.capacity * SG;
  const unsigned BT = V45_PULL_BT;
  const unsigned bpc = (K41 + BT - 1u) / BT;
  const unsigned nb = ncb * 7u * bpc;
  const unsigned ns = (PB + 1023u) >> 10u;
  const uint32_t pay_rng = e.capacity * PB, scl_rng = e.capacity * SG * 4u;
  FI_BG(V1988_FI_CORE, 16u, p);
  for (unsigned bi = worker; bi < nb; bi += nworkers) {
    const unsigned ci = bi / (7u * bpc);
    const unsigned rr = bi - ci * 7u * bpc;
    const unsigned r = rr / bpc;
    const unsigned sub = rr - r * bpc;
    const unsigned src = __builtin_amdgcn_readfirstlane(node * 8u + r + ((r >= skip) ? 1u : 0u));
    const unsigned toff = sub * BT;
    const unsigned tbb = __builtin_amdgcn_readfirstlane((cb_beg + ci) * K41 + toff);
    if (lane == 0u) {
      const auto* rdy_p = reinterpret_cast<const unsigned*>(peer_rdy_tbl[src]) + (cb_beg + ci) * K41;
      v45_spin_rdy(rdy_p, env, x, start, 966u);
    }
    __syncwarp();
    if (ep16_load_relaxed((unsigned*)e.error)) break;
    bool need = false;
    {
      const unsigned tl = tbb + lane;
      if (lane < BT && toff + lane < K41 && tl < te && tl < NT)
        need = local_nexp[static_cast<uint64_t>(src) * NT + tl] != 0u;
    }
    uint64_t live = __ballot(need);
    if (live == 0ull) continue;
    const WeaveI32x4 rsx = weave_make_buf_rsrc(reinterpret_cast<const void*>(peer_pay_tbl[src]), pay_rng);
    const WeaveI32x4 rdx = weave_make_buf_rsrc(local_pay_base + static_cast<uint64_t>(src) * e.capacity * PB, pay_rng);
    {
      uint64_t rem = live;
      unsigned tcur = static_cast<unsigned>(__builtin_ctzll(rem)), s = 0u;
      while (rem) {
        int32_t vo[8];
        #pragma unroll
        for (unsigned k = 0u; k < 8u; ++k) {
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
        WeaveI32x4 v[8];
        #pragma unroll
        for (unsigned k = 0u; k < 8u; ++k) v[k] = weave_raw_buf_load_i32x4(rsx, vo[k], 0, V44_SC1);
        #pragma unroll
        for (unsigned k = 0u; k < 8u; ++k) v45_raw_buf_store_i32x4(v[k], rdx, vo[k], 0, 0);
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
  if (is_rem) {
    for (unsigned ci = worker; ci < ncb; ci += nworkers) {
      if (lane == 0u) {
        const auto* rdy_p = reinterpret_cast<const unsigned*>(peer_rdy_tbl[partner]) + (cb_beg + ci) * K41;
        v45_spin_rdy(rdy_p, env, x, start, 967u);
      }
      __syncwarp();
    }
  }
  FI_EG(V1988_FI_CORE, 16u, p);
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __syncwarp();
  if (ep16_load_relaxed((unsigned*)e.error)) return;
  if (lane == 0u) {
    __threadfence();
    const unsigned d = __hip_atomic_fetch_add(pf + V46_PULL_CNT + p, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (d == nworkers) {
      v41_st_agent_u32(pf + V46_PULL_RDY + p, 1u);
      auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
      if (trace) trace[1304u + p] = __builtin_amdgcn_s_memrealtime();
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
__device__ __forceinline__ void v46_gemm1_item(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned char* arena, unsigned mt, unsigned n1 FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const V968CkExt& ck = env.ck;
  const unsigned tid = threadIdx.x & 255u;
  const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(ck.ck_a + 64u);
  const unsigned num_tokens = e.world * e.capacity;
  const unsigned run = ck.ck_route_rows;
  const int32_t* stage1_ids = reinterpret_cast<const int32_t*>(ck.ck_sorted_token_ids);
  const WeaveMTileDesc d_raw = tiles[mt];
  const unsigned expert_id = __builtin_amdgcn_readfirstlane(d_raw.expert_id);
  const unsigned row_start = __builtin_amdgcn_readfirstlane(d_raw.row_start);
  const unsigned bm = __builtin_amdgcn_readfirstlane(d_raw.bm);
  const unsigned valid_rows = __builtin_amdgcn_readfirstlane(d_raw.valid_rows);
  // opus55 v46 E8: rank / capacity / row geometry re-defined per item behind an opaque SGPR copy, so the receive
  // arena bases are rebuilt in SALU inside the item: no hoisted or CSE'd 64-bit VGPR pair (zext(rank),
  // -rank*cap*PB, -rank*cap*SG) stays live across the Atrex loop and its GEMM1/GEMM2 bodies
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
#if V55_COALESCE_SCHED
  const unsigned n1b_l = ck.ck_intermediate / 128u;
  unsigned n1g = (V55_N1G); if (n1g > n1b_l - n1) n1g = n1b_l - n1;
#else
  const unsigned n1g = 1u;
#endif
  #pragma unroll
  for (unsigned n1i = n1; n1i < n1 + n1g; ++n1i) {
#if V62_DYN_BM
  if (bm == 16u) {
    weave_s1_1x4_v62<16u, (V62_G1_SMALL)>(ids, (int32_t)expert_id, recv_payload,
        reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
        reinterpret_cast<const float*>(ck.ck_b_scale),
        reinterpret_cast<uint8_t*>(ck.ck_quant_output),
        reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
        num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
  } else
#endif
#if V62_G1_BM32
  if (bm == 32u) {
    weave_s1_1x4_v62<32u, (V62_G1_BM32)>(ids, (int32_t)expert_id, recv_payload,
        reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
        reinterpret_cast<const float*>(ck.ck_b_scale),
        reinterpret_cast<uint8_t*>(ck.ck_quant_output),
        reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
        num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
  } else
#endif
#if V62_G1_BM48
  if (bm == 48u) {
    weave_s1_1x4_v62<48u, (V62_G1_BM48)>(ids, (int32_t)expert_id, recv_payload,
        reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
        reinterpret_cast<const float*>(ck.ck_b_scale),
        reinterpret_cast<uint8_t*>(ck.ck_quant_output),
        reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
        num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
  } else
#endif
  if (bm == 32u) {
    weave_dispatch_stage1_bk64<32u>(ids, (int32_t)expert_id, recv_payload,
        reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
        reinterpret_cast<const float*>(ck.ck_b_scale), nullptr,
        reinterpret_cast<uint8_t*>(ck.ck_quant_output),
        reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
        num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
  } else {
#if V62_G1_BODY
    weave_s1_1x4_v62<64u, (V62_G1_BODY)>(ids, (int32_t)expert_id, recv_payload,
        reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
        reinterpret_cast<const float*>(ck.ck_b_scale),
        reinterpret_cast<uint8_t*>(ck.ck_quant_output),
        reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
        num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
#elif V61_G1_1X4
    // V61: 1x4 weight-once BK128 body (each of the 4 waves owns 64 rows x 32 cols, B read once per tile)
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
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  }
  FI_EG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
  if (tid == 0u)
#if V55_COALESCE_SCHED
    __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.stage1_tile_done) + mt, n1g, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#else
    __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.stage1_tile_done) + mt, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#endif
}

__device__ __forceinline__ bool v46_gemm2_item(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned char* arena, unsigned mt, unsigned cidx,
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
      const unsigned* s1p = reinterpret_cast<const unsigned*>(o.stage1_tile_done) + mt;
      unsigned backoff = 1u, spins = 0u, ok = 1u;
      while (__hip_atomic_load(s1p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < n1_blocks) {
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
  #pragma clang loop unroll(disable)
  for (unsigned g = 0u; g < n2g; ++g) {
    const unsigned n2 = cidx * n2g + g;
#if V62_DYN_BM
    if (bm == 16u) {
#if V62_G2_BN512 && V62_BN512_BM16
      if (g + 3u < n2g) {
        weave_s2_1x4_bn512<16u, (V62_G2_BN512)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
        g += 3u;
      } else
#endif
      if (g + 1u < n2g) {
        weave_s2_1x4_bn256<16u, (V62_G2_BN256)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
        ++g;
      } else {
        // odd trailing 128-col block (no model has one: n2g is 4 or 8); the BM=32 body masks rows >= valid_rows
        weave_dispatch_stage2_bk64<32u>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
      }
    } else
#endif
#if V62_G2_BN512 && V62_BN512_BM48
    if (bm == 48u) {
      if (g + 3u < n2g) {
        weave_s2_1x4_bn512<48u, (V62_G2_BN512)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
        g += 3u;
      } else {
        weave_s2_1x4_bn256<48u, (V62_G2_BN256)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
      }
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
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
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
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
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
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
        ++g;
      } else
#endif
      weave_dispatch_stage2_bk64<32u>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
    } else {
#if V62_G2_BN512 && V62_BN512_BM64
      // V62: four adjacent 128-col blocks per K loop when a full 512-column group remains
      if (g + 3u < n2g) {
        weave_s2_1x4_bn512<64u, (V62_G2_BN512)>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
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
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
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
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
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
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
#elif V61_G2_1X4
      weave_s2_1x4_bk128_zerob<64u>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
#else
      weave_dispatch_stage2_bk64<64u>(ids, (int32_t)expert_id,
          reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<const uint8_t*>(o.ck_w2),
          reinterpret_cast<const float*>(ck.ck_quant_scale),
          reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
          c_out, row_start, valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
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
      const unsigned id = row_route_id[row_start + r];
      if (id == 0xffffffffu) continue;
      const unsigned g2p = id & 0xffffffu;
      const unsigned long long divmod = v983_divmod_u32(g2p, e.capacity + 1u);
      const unsigned src = (unsigned)(divmod >> 32);
      const unsigned slot = (unsigned)divmod;
      __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.bucket_done) + (src * e.capacity + slot), 1u,
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

__device__ __forceinline__ unsigned v46_decide(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned& seen, unsigned n1b_in, unsigned n2c_in,
    unsigned v54_x = 0u) {
  unsigned n1b = __builtin_amdgcn_readfirstlane(n1b_in), n2c = __builtin_amdgcn_readfirstlane(n2c_in);
  asm volatile("" : "+s"(n1b), "+s"(n2c));   // opus55 v46 E6: re-defined per decision -> no hoisted reciprocal
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const unsigned* s1d = reinterpret_cast<const unsigned*>(o.stage1_tile_done);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  if (ep16_load_relaxed((unsigned*)env.base.error)) return V46_W_ERR;
  if ((seen & 1u) == 0u) {
    if (v42_ld_agent_rlx(pf + V46_LOCAL_RDY) == 0u) return V46_WORD(V46_ACT_IDLE, 0u, 0u, 0u);
    v42_acq_agent();
    seen |= 1u;
  }
  if ((seen & 2u) == 0u && v42_ld_agent_rlx(pf + V46_P1_PLAN) != 0u) { v42_acq_agent(); seen |= 2u; }
  #pragma unroll
  for (unsigned q = 0u; q < V46_NP; ++q)
    if ((seen & (4u << q)) == 0u && v42_ld_agent_rlx(pf + V46_PULL_RDY + q) != 0u) { v42_acq_agent(); seen |= 4u << q; }
#if V54_XCD_PART
  // ---- v54 Fleet 4-XCD home queues: identical phase order / pass order / readiness gates as the base, but
  // every flat cursor is replaced by 4 per-XCD cursors driven own-first with a clean steal fallback. ----
  const bool q3ok = true;   // v46 blood line has no V48 Q3R gate; folds every !q3ok test away
  const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(env.ck.ck_a + 64u);
  const unsigned xcd = v54_x;
  unsigned from = 0u;
  // (1) ready GEMM2 heads, pass order 1, 2, 0, 3
  #pragma unroll
  for (unsigned j = 0u; j < V46_NP; ++j) {
    const unsigned q = (j < 2u) ? (j + 1u) : ((j == 2u) ? 0u : 3u);
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
    const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q];
    const unsigned c = v54_claim<1u>(pf + V54_G2C + q * 4u, T * n2c, n2c, T, B, tiles, xcd,
                                     n1b, s1d, pf + V54_G1C + q * 4u, from);
    if (c != V46_NONE) {
      const unsigned mtl = c / n2c;
      return V46_WORD(V46_ACT_G2, q, c - mtl * n2c, B + mtl);
    }
  }
  // (2) GEMM1 items, pass order 0, 1, 2, 3
  #pragma unroll
  for (unsigned q = 0u; q < V46_NP; ++q) {
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
    if ((seen & (4u << q)) == 0u) continue;
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
    const unsigned q = (j < 2u) ? (j + 1u) : ((j == 2u) ? 0u : 3u);
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
    const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q];
    const unsigned c = v54_claim<2u>(pf + V54_G2C + q * 4u, T * n2c, n2c, T, B, tiles, xcd,
                                     n1b, s1d, pf + V54_G1C + q * 4u, from);
    if (c != V46_NONE) {
      const unsigned mtl = c / n2c;
      return V46_WORD(V46_ACT_G2, q, c - mtl * n2c, B + mtl);
    }
  }
  // (4) exit test: every one of the 16 GEMM2 home queues must have reached its pass bound
  if (seen & 2u) {
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
      if (__hip_atomic_load(s1d + B + mtl, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < n1b) break;
      if (__hip_atomic_compare_exchange_strong(cur, &c, c + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
        return V46_WORD(V46_ACT_G2, q, j_ * 4u + xx, B + mtl);
    }
    return V46_NONE;
  };
#endif
  // (1) ready GEMM2 heads, pass order 1, 2, 0, 3
  #pragma unroll
  for (unsigned j = 0u; j < V46_NP; ++j) {
    const unsigned q = (j < 2u) ? (j + 1u) : ((j == 2u) ? 0u : 3u);
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
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
      if (__hip_atomic_load(s1d + B + mtl, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < n1b) break;
      if (__hip_atomic_compare_exchange_strong(cur, &c, c + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
        return V46_WORD(V46_ACT_G2, q, j_, B + mtl);
#else
      const unsigned mtl = c / n2c;
      if (__hip_atomic_load(s1d + B + mtl, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < n1b) break;
      if (__hip_atomic_compare_exchange_strong(cur, &c, c + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
        return V46_WORD(V46_ACT_G2, q, c - mtl * n2c, B + mtl);
#endif
    }
  }
  // (2) GEMM1 items, pass order 0, 1, 2, 3
  #pragma unroll
  for (unsigned q = 0u; q < V46_NP; ++q) {
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
    if ((seen & (4u << q)) == 0u) continue;
    if (q == 3u && (seen & 4u) == 0u) continue;   // own-rank staging is certified by PULL_RDY[0]
    const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q];
#if V55_COALESCE_SCHED
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
    const unsigned c = v46_claim_lt(pf + V46_G1C + q, T * n1b);
    if (c != V46_NONE) {
      if (c == 0u && trace) trace[1308u + q] = __builtin_amdgcn_s_memrealtime();
      const unsigned mtl = c / n1b;
      return V46_WORD(V46_ACT_G1, q, c - mtl * n1b, B + mtl);
    }
#endif
  }
#if V62_G2_SWZ
  // (3s) V62: steal ready GEMM2 heads from the other XCC queues (only when no GEMM1 item was available)
  if (v62_swz) {
    #pragma unroll
    for (unsigned j = 0u; j < V46_NP; ++j) {
      const unsigned q = (j < 2u) ? (j + 1u) : ((j == 2u) ? 0u : 3u);
      if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
      #pragma unroll
      for (unsigned dx = 1u; dx < 4u; ++dx) {
        const unsigned w = v62_g2_take(q, (v62_xcc + dx) & 3u);
        if (w != V46_NONE) return w;
      }
    }
  }
#endif
#if V62_G1_STEAL && V55_COALESCE_SCHED && V55_SWIZZLE
  // (3g) V62: steal GEMM1 items from the other XCC queues (pass tail only: own queue (2) was empty)
  {
    const unsigned n1g_per_mt = n1b / (V55_N1G);
    if (n1g_per_mt >= 4u) {
      const unsigned xcc0 = v55_xcc();
      #pragma unroll
      for (unsigned q = 0u; q < V46_NP; ++q) {
        if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
        if ((seen & (4u << q)) == 0u) continue;
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
    const unsigned q = (j < 2u) ? (j + 1u) : ((j == 2u) ? 0u : 3u);
    if (v46_is_remote(q) && (seen & 2u) == 0u) continue;
    const unsigned T = pf[V46_T + q], B = pf[V46_BASE + q], bound = T * n2c;
    unsigned* cur = pf + V46_G2C + q;
    unsigned c = __hip_atomic_load(cur, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    while (c < bound) {
      const unsigned mtl = c / n2c;
      if (__hip_atomic_load(pf + V46_G1C + q, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (mtl + 1u) * n1b) break;
      if (__hip_atomic_compare_exchange_strong(cur, &c, c + 1u, __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
        return V46_WORD(V46_ACT_G2, q, c - mtl * n2c, B + mtl);
    }
  }
#endif
  // (4) exit test
  if (seen & 2u) {
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
    const unsigned q = (w >> 28u) & 3u, sub = (w >> 20u) & 0xffu, mt = w & 0xfffffu;
    if (act == V46_ACT_G1) {
      v46_gemm1_item(env, o, arena, mt, sub FI_CTX_ARG);
      if (tid == 0u) {
#if V55_COALESCE_SCHED
        unsigned g1inc = (V55_N1G); if (g1inc > n1b - sub) g1inc = n1b - sub;
#else
        const unsigned g1inc = 1u;
#endif
        const unsigned d = __hip_atomic_fetch_add(pf + V46_G1D + q, g1inc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + g1inc;
        if (d == pf[V46_T + q] * n1b && trace) trace[1312u + q] = __builtin_amdgcn_s_memrealtime();
      }
    } else {
      if (!v46_gemm2_item(env, o, arena, mt, sub, n2g, last_acq_mt, (unsigned*)x.compute_done,
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
    __hip_atomic_store(ph + t, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  } else {
    auto* na = reinterpret_cast<unsigned*>(reinterpret_cast<const uint64_t*>(x.node_arrival_table)[src & 7u]);
    __hip_atomic_fetch_add(na + t, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  ep16_st_agent_relaxed(reinterpret_cast<unsigned*>(o.bucket_sent) + key, 1u);
}

// -------------------------------------------------------------------------------------------------------------
// 4-pass streaming scatter (generalizes v45_sweep_mb).  Block = (64-token chunk c, in-node source so), chunk-major.
// A token is claimed by CAS bucket_done: nexp -> 0xffffffff, i.e. only after all its expert rows left GEMM2.
// st bits: 0 = pass plan seen, 1 = local mask empty, 2 = pass globally scattered (SDONE)
// -------------------------------------------------------------------------------------------------------------
#ifndef V46_STREAM_CLAIM
#define V46_STREAM_CLAIM 16u
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
    if (v42_ld_agent_rlx(pf + (v46_is_remote(p) ? V46_P1_PLAN : V46_LOCAL_RDY)) == 0u) return 8u;
    v42_acq_agent();
    m0 = v44_lowmask(nblk < 64u ? nblk : 64u);
    m1 = v44_lowmask(nblk > 64u ? nblk - 64u : 0u);
    pos = nblk ? (worker % nblk) : 0u;
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
    if (prio && scat_live) {
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
    if (!prio && scat_live) {
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
      v46_combine_loop(env, o, x, cw, 0u, true, true, true FI_CTX_ARG);
    } else if (cw < V46_CW_POOL) {
#if !(V46_DBG & 4)
      v46_plan_remote(env, o, x, arena, (cw - 8u) >> 2u, V46_NRP FI_CTX_ARG);
#endif
      v46_combine_loop(env, o, x, cw, 0u, true, true, true FI_CTX_ARG);
    } else if (cw < V46_CW_POOL + NNR) {
      v46_combine_loop(env, o, x, cw, 0u, false, true, true FI_CTX_ARG);
    } else {
      const unsigned pw = cw - V46_CW_POOL - NNR;
      // pull order q0 (with the compute groups) -> q3 (local, gated only on LBUILD) -> q1 -> q2: the remote pulls
      // are bounded by the remote RDMA arrival, so pulling the local tail first never delays them
      v45_front_stage(env, o, x, pw, nwf FI_CTX_ARG);
      v46_pull_pass(env, o, x, 0u, pw, nwf FI_CTX_ARG);
      v46_pull_pass(env, o, x, 3u, pw, NPULL FI_CTX_ARG);
      v46_pull_pass(env, o, x, 1u, pw, NPULL FI_CTX_ARG);
      v46_pull_pass(env, o, x, 2u, pw, NPULL FI_CTX_ARG);
      v46_combine_loop(env, o, x, cw, 1u, true, true, true FI_CTX_ARG);
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
  v45_front_stage(env, o, x, wk, nwf FI_CTX_ARG);
#if (V46_DBG & 16)
  return;
#endif
  if (!v46_plan_local(env, o, x, arena, grp, ncg FI_CTX_ARG)) return;
  v46_pull_pass(env, o, x, 0u, wk, nwf FI_CTX_ARG);
#if (V46_DBG & 1)
  const bool ok = true;
#else
  const bool ok = v46_atrex(env, o, x, arena FI_CTX_ARG);
#endif
  if ((threadIdx.x & 255u) == 0u && trace) {
    trace[16u + cta * 16u + 9u] = trace[1312u];
    trace[16u + cta * 16u + 10u] = v983_clock();
  }
  if (!ok) return;
  v46_combine_loop(env, o, x, V46_WCOMM + grp * 4u + (wave & 3u), 0u, true, true, true FI_CTX_ARG);
  if ((threadIdx.x & 255u) == 0u && trace) trace[16u + cta * 16u + 11u] = v983_clock();
}
