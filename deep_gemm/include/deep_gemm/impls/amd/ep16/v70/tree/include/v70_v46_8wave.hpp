#pragma once
// =============================================================================================================
// V70: hardware-s_barrier 8-wave Atrex for PURE 8/0 compute CTAs.
//
// Enabled only for the official WEAVE_CU_WAVE_SPLIT==1 layout (all non-dispatch CTAs are 0-8 pure compute)
// and only when the compile-time arm sets V70_HR1_8W=1.  The 8-wave bodies are the validated HR1 MI308X
// kernels (hr1_gemm_8wave.hpp): BM=128 x BN=128 cooperative GEMM1/GEMM2, exactly ONE s_barrier per K step,
// full 64 KB LDS shared by all 512 threads, zero software spin barrier in the K loop.
//
// PLUG-AND-PLAY: v70_at_atrex() is the single dispatch point.  Replacing hr1_s1_8wave / hr1_s2_8wave with a
// newer Horse 1 / Horse 2 / Opus 5.5 hardware-barrier body requires no change here: the item wrappers below
// publish stage1_tile_done / row_n_done / bucket_done / S1F arrivals with the production protocol.
//
// Deadlock-freedom: only dual==true (e.t_local >= 16) 8/0 CTAs enter v70_atrex8, and on those CTAs all eight
// waves are compute (v46_main assigns no comm role).  Every hr1_barrier() is at a point all 512 threads reach.
// 0/8 and 4/4 CTAs keep the 4-wave software path and never execute an s_barrier.
// =============================================================================================================

#include "hr1_gemm_8wave.hpp"
#include "v70_gemm_bm64_8wave.hpp"
#include "v70_r2_8wave.hpp"
#if V70_R2X && !(V70_BM64_8W_ACTIVE)
#error "V70_R2X (fused R2 body) requires the BM64/64 pure 8/0 hardware-barrier layout"
#endif

#if V70_8W_ACTIVE

#ifndef V70_S1_ACQ
#define V70_S1_ACQ 1
#endif
#ifndef V70_R2X_G2SW
#define V70_R2X_G2SW 0
#endif
#ifndef V70_PATHB
#define V70_PATHB 0
#endif
#ifndef V70_PROFILE_PHASES
#define V70_PROFILE_PHASES 0   // harness-only per-CTA phase accounting
#endif
#ifndef V70_ATREX_1BAR
#define V70_ATREX_1BAR 0       // publish the next decision across one four-wave barrier
#endif
#if V70_PATHB
// Path B: the work unit ends at 26368 B, followed by the compact claim table at
// 26624 and the row table at 28672.  Keep every live control slot below 32 KiB.
#define V70_8W_SH_WORD 30720u
#define V70_8W_SH_W    30724u
#define V70_8W_SH_DONE 30728u
#define V70_8W_SH_LAM  30732u
#define V70_8W_SH_SEEN 30736u
// A 256-thread CTA is exactly one four-wave group, so workgroup s_barrier is the
// native four-wave rendezvous.  Path B has no software LDS barrier in the K loop.
#define V70_R2X_4W_SW 0
#else
#define V70_8W_SH_WORD 65472u   // Atrex decision word
#define V70_8W_SH_W    65476u   // GEMM2 readiness broadcast
#define V70_8W_SH_DONE 65480u   // row_n_done broadcast
#define V70_8W_SH_LAM  65484u   // last acquired GEMM2 M tile
#define V70_8W_SH_SEEN 65488u   // v46_decide readiness bits
#define V70_R2X_4W_SW 1
#endif

#if V70_R2X_COMPACT
// Compacted tile c (V70_R2X_COMPACT, merged mode 4): expert lid, valid rows (uniform) and the absolute stage row of
// local row r (per lane).  Reads the per-CTA LDS table built by v70c_build: [0..64] tile prefix per expert,
// [65 + q*64 + lid] = (rs >> 4) << 17 | cnt per pass segment.  Segments are concatenated in pass order q0..q3.
__device__ __forceinline__ unsigned v70c_lid(unsigned LE, unsigned c) {
  volatile WeaveBk64LocalU32* tb = v70m_tab();
  unsigned lo = 0u, hi = LE;
  while (hi - lo > 1u) {
    const unsigned mid = (lo + hi) >> 1u;
    if (__builtin_amdgcn_readfirstlane(tb[mid]) <= c) lo = mid; else hi = mid;
  }
  return lo;
}
__device__ __forceinline__ unsigned v70c_valid(unsigned lid, unsigned c) {
  volatile WeaveBk64LocalU32* tb = v70m_tab();
  unsigned n = 0u;
  #pragma unroll
  for (unsigned q = 0u; q < V46_NP; ++q) n += __builtin_amdgcn_readfirstlane(tb[65u + q * 64u + lid]) & 0x1ffffu;
  const unsigned t0 = (c - __builtin_amdgcn_readfirstlane(tb[lid])) << (V70C_SH);
  return (n > t0) ? ((n - t0 >= (V70C_BM)) ? (V70C_BM) : n - t0) : 0u;
}
__device__ __forceinline__ int v70c_pos(unsigned lid, unsigned c, int r) {
  volatile WeaveBk64LocalU32* tb = v70m_tab();
  const int g = (int)((c - tb[lid]) << (V70C_SH)) + r;
  int acc = 0, p = 0;
  #pragma unroll
  for (unsigned q = 0u; q < V46_NP; ++q) {
    const unsigned pk = tb[65u + q * 64u + lid];
    const int rs = (int)((pk >> 17u) << 4u), cnt = (int)(pk & 0x1ffffu);
    if (g >= acc && g < acc + cnt) p = rs + (g - acc);
    acc += cnt;
  }
  return p;
}
#endif

__device__ __forceinline__ volatile WeaveBk64LocalU32* v70_8w_sh(unsigned off) {
  return reinterpret_cast<volatile WeaveBk64LocalU32*>(
      reinterpret_cast<uintptr_t>(g_v1262_phase_arena + off));
}

// -------------------------------------------------------------------------------------------------------------
// GEMM1 item, unified 8-wave group.  Honours the production V55 coalesced claim (n1g n1-blocks per item),
// WEAVE_BLOCK_M_L1 sub-tiling, WEAVE_ADAPT_BM tiles and the exact stage1_tile_done publication.
// -------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ void v70_gemm1_item8(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned char* arena,
    unsigned mt, unsigned s1s, unsigned n1 FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const V968CkExt& ck = env.ck;
  const unsigned tid = threadIdx.x;
  const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(ck.ck_a + 64u);
  const unsigned num_tokens = e.world * e.capacity;
  const unsigned run = ck.ck_route_rows;
  const int32_t* stage1_ids = reinterpret_cast<const int32_t*>(ck.ck_sorted_token_ids);
#if V70_R2X_COMPACT
  // merged mode 4: mt = V70C_FLAG | compacted tile c; no descriptor is read
  const bool cmp = (mt & V70C_FLAG) != 0u;
  const unsigned cidx_t = mt & 0xffffu;
  WeaveMTileDesc d_raw;
  if (cmp) {
    const unsigned lid = v70c_lid(v70_claim_le(env, reinterpret_cast<const unsigned*>(o.plan_flags)), cidx_t);
    d_raw = WeaveMTileDesc{lid, 0u, 64u, (uint16_t)v70c_valid(lid, cidx_t), 0u};
  } else {
    d_raw = tiles[mt];
  }
#else
  const WeaveMTileDesc d_raw = tiles[mt];
#endif
  const unsigned expert_id = __builtin_amdgcn_readfirstlane(d_raw.expert_id);
  const unsigned vr2 = __builtin_amdgcn_readfirstlane(d_raw.valid_rows);
#if WEAVE_ADAPT_BM
  const unsigned row_start = __builtin_amdgcn_readfirstlane(d_raw.row_start);
  const unsigned bm = __builtin_amdgcn_readfirstlane(d_raw.bm);
  const unsigned valid_rows = vr2;
#else
  const unsigned bm = (WEAVE_ADAPT_BM) ? (V70C_BM) : (WEAVE_BLOCK_M_L1);
  const unsigned s1off = s1s * bm;
  const unsigned row_start = __builtin_amdgcn_readfirstlane(d_raw.row_start) + s1off;
  const unsigned valid_rows = (s1off < vr2) ? ((vr2 - s1off > bm) ? bm : (vr2 - s1off)) : 0u;
#endif
#if V70_XSHADOW
  const unsigned char* recv_payload = v70_xshadow_pay(env);
  const float* recv_scale = v70_xshadow_scl(env);
#else
  unsigned rk = __builtin_amdgcn_readfirstlane(e.rank), cap = __builtin_amdgcn_readfirstlane(e.capacity);
  unsigned pbw = __builtin_amdgcn_readfirstlane(e.payload_bytes), sgw = __builtin_amdgcn_readfirstlane(e.scale_groups);
  asm volatile("" : "+s"(rk), "+s"(cap), "+s"(pbw), "+s"(sgw));
  const uint64_t pay_e = reinterpret_cast<const uint64_t*>(e.peer_payload_table)[rk];
  const uint64_t scl_e = reinterpret_cast<const uint64_t*>(e.peer_scale_table)[rk];
  const uint64_t pay_u = v46_rfl_u64(pay_e), scl_u = v46_rfl_u64(scl_e);
  const unsigned rc = rk * cap;
  const unsigned char* recv_payload = reinterpret_cast<const unsigned char*>(pay_u - static_cast<uint64_t>(rc) * pbw);
  const float* recv_scale = reinterpret_cast<const float*>(scl_u - static_cast<uint64_t>(rc) * sgw * 4ull);
#endif
  const unsigned g1_aux = (bm << 16u) | valid_rows;
  FI_BG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
  const int32_t* ids = stage1_ids + row_start;
#if V55_COALESCE_SCHED && (V46_NSUB == 1)
  const unsigned n1b_l = ck.ck_intermediate / 128u;
  unsigned n1g = (V55_N1G); if (n1g > n1b_l - n1) n1g = n1b_l - n1;
#else
  const unsigned n1g = 1u;
#endif
#if V70_R2X
  if (valid_rows != 0u) {
    // Two 4-wave groups, one n1 block each (V55 coalesced claim: n1g = 2 for every supported model).
    const unsigned grp = __builtin_amdgcn_readfirstlane(tid >> 8u);
    uint8_t* glds = g_v1262_phase_arena + grp * R2X_GRP_LDS;
    const int nk1 = (int)(ck.ck_hidden >> 7u);
    // R4: wave 0 fills the LDS row table of this tile, the CTA barrier publishes it to both groups.
#if V70_R2X_COMPACT
    if (cmp) r2x::rtab_fill(tid, stage1_ids, nullptr, (int)valid_rows,
                            [&](int r) { return v70c_pos(expert_id, cidx_t, r); });
    else
#endif
    r2x::rtab_fill(tid, stage1_ids, nullptr, (int)valid_rows, [&](int r) { return (int)row_start + r; });
    hr1_barrier();
    #pragma clang loop unroll(disable)
    for (unsigned b0 = n1; b0 < n1 + n1g; b0 += 2u) {
      const unsigned nb = b0 + grp;
      // Re-define the per-item scalars every iteration: otherwise LICM hoists the per-row id / weight
      // addresses out of this loop and keeps them live (spilled) across the whole K-loop body.
      const unsigned char* pay_l = recv_payload; const float* rsc_l = recv_scale;
      unsigned vr_l = valid_rows, ex_l = expert_id;
      asm volatile("" : "+s"(pay_l), "+s"(rsc_l), "+s"(vr_l), "+s"(ex_l));
      if (nb < n1 + n1g)
        r2x::unit<1>(glds, (int)(tid & 255u), (int)vr_l, (int)nb, (int)ex_l,
                     pay_l, reinterpret_cast<const uint8_t*>(ck.ck_b), rsc_l,
                     reinterpret_cast<const float*>(ck.ck_b_scale), reinterpret_cast<uint8_t*>(ck.ck_quant_output),
                     reinterpret_cast<float*>(ck.ck_quant_scale), nullptr, ck.ck_hidden, ck.ck_intermediate, run,
                     nullptr, nullptr, nullptr, 0u, 0u);
      else
        r2x::idle<0>(nk1 + 2);
    }
  }
#else
  if (valid_rows != 0u) {
    #pragma unroll
    for (unsigned n1i = n1; n1i < n1 + n1g; ++n1i) {
      // V70_HR1_8W enforces fixed 128-row stage tiles, so every descriptor takes the cooperative body.
#if V70_BM64_8W_ACTIVE
      v70_s1_8wave_bm64(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
#else
      hr1_s1_8wave(ids, (int32_t)expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), row_start, valid_rows, n1i,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
#endif
    }
  }
#endif  // V70_R2X
  __builtin_amdgcn_s_waitcnt(0);
  hr1_barrier();                                          // all 512 rejoin
  FI_EG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
  if (tid == 0u) {
    const unsigned slot = (mt & 0xffffu) * (V46_NSUB) + ((WEAVE_ADAPT_BM) ? 0u : s1s);
    const unsigned n1b_all = ck.ck_intermediate / 128u;
    const unsigned done_inc = (valid_rows != 0u) ? n1g : ((n1 == 0u) ? n1b_all : 0u);
    if (done_inc)
      __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.stage1_tile_done) + slot, done_inc,
                             (V70_E3_SC1_NOBW) ? __ATOMIC_RELAXED : __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
}

// -------------------------------------------------------------------------------------------------------------
// GEMM2 item, unified 8-wave group.  Includes the production S1F direct epilogue, the V54 MALL bypass and the
// exact row_n_done / bucket_done / compute_done publication.
// -------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ bool v70_gemm2_item8(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned char* arena,
    unsigned mt, unsigned cidx, unsigned n2g, unsigned* compute_done_ptr FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const V968CkExt& ck = env.ck;
  const unsigned tid = threadIdx.x;
  volatile WeaveBk64LocalU32* sh_done = v70_8w_sh(V70_8W_SH_DONE);
  volatile WeaveBk64LocalU32* sh_lam  = v70_8w_sh(V70_8W_SH_LAM);
  const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(ck.ck_a + 64u);
  const unsigned n1_blocks = ck.ck_intermediate / 128u;
  const unsigned n2_blocks = o.hidden / 128u;
  const int32_t* stage2_ids = reinterpret_cast<const int32_t*>(ck.ck_m64_sorted_ids);
  const float* weights = reinterpret_cast<const float*>(o.stage2_weights);
  const unsigned* row_route_id = reinterpret_cast<const unsigned*>(o.row_route_id);
  const unsigned run = ck.ck_route_rows;
#if V70_R2X_COMPACT
  // merged mode 4: mt = V70C_FLAG | compacted tile c; no descriptor is read
  const bool cmp = (mt & V70C_FLAG) != 0u;
  const unsigned cidx_t = mt & 0xffffu;
  WeaveMTileDesc d_raw;
  if (cmp) {
    const unsigned lid = v70c_lid(v70_claim_le(env, reinterpret_cast<const unsigned*>(o.plan_flags)), cidx_t);
    d_raw = WeaveMTileDesc{lid, 0u, 64u, (uint16_t)v70c_valid(lid, cidx_t), 0u};
  } else {
    d_raw = tiles[mt];
  }
#else
  const WeaveMTileDesc d_raw = tiles[mt];
#endif
  const unsigned expert_id = __builtin_amdgcn_readfirstlane(d_raw.expert_id);
  const unsigned row_start = __builtin_amdgcn_readfirstlane(d_raw.row_start);
  const unsigned bm = __builtin_amdgcn_readfirstlane(d_raw.bm);
  const unsigned valid_rows = __builtin_amdgcn_readfirstlane(d_raw.valid_rows);
#if V70_R2X_COMPACT
  const unsigned mts = (mt & 0xffffu) + (cmp ? __builtin_amdgcn_readfirstlane(v70m_tab()[321u]) : 0u);
#else
  const unsigned mts = mt & 0xffffu;
#endif   // done-counter slot (descriptor or compacted tile)
  FI_BG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt);
  const unsigned last_acq_mt = __builtin_amdgcn_readfirstlane(sh_lam[0]);
  const unsigned lam_key = (mt & V70C_FLAG) ? (V70C_FLAG | mts) : mt;
  if (lam_key != last_acq_mt) {
    volatile WeaveBk64LocalU32* sh_w = v70_8w_sh(V70_8W_SH_W);
    hr1_barrier();
    if (tid == 0u) {
      const unsigned* s1p = reinterpret_cast<const unsigned*>(o.stage1_tile_done) + mts * (V46_NSUB);
      unsigned backoff = 1u, spins = 0u, ok = 1u;
      while (!v46_s1_ready(reinterpret_cast<const unsigned*>(o.stage1_tile_done), mts, n1_blocks)) {
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
      (void)s1p;
      sh_w[0] = ok;
    }
    hr1_barrier();
    if (__builtin_amdgcn_readfirstlane(sh_w[0]) == 0u) {
      FI_EG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt);
      return false;
    }
#if V70_S1_ACQ == 0
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
#elif V70_S1_ACQ == 2 || !V54_S1_BYPASS_MALL
    asm volatile("buffer_inv sc1\ns_waitcnt vmcnt(0)\n" ::: "memory");
#else
    asm volatile("buffer_inv sc0\ns_waitcnt vmcnt(0)\n" ::: "memory");
#endif
    if (tid == 0u) sh_lam[0] = lam_key;
    hr1_barrier();
  }
  FI_EG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt);
  const unsigned g2_aux = ((n2g & 0x7u) << 28u) | ((bm & 0xfffu) << 16u) | (valid_rows & 0xffffu);
  FI_BG(V1988_FI_CORE, V1988_EV_GEMM2_TILE, g2_aux);
  const int32_t* ids = stage2_ids + row_start;
  uint16_t* c_out = reinterpret_cast<uint16_t*>(o.gemm2_output);
#ifndef V70_S1F_EN
#define V70_S1F_EN 0
#endif
#if V70_R2X
  if (valid_rows != 0u) {
    // Two 4-wave groups, 256 output columns per R2 work unit: chunks c0+grp of the n2g x 128 claim (n2g in {4,8}).
    if ((n2g & 1u) && tid == 0u) v983_error(reinterpret_cast<unsigned*>(e.error), 977u);
    const unsigned grp = __builtin_amdgcn_readfirstlane(tid >> 8u);
    uint8_t* glds = g_v1262_phase_arena + grp * R2X_GRP_LDS;
    const int nk2 = (int)(ck.ck_intermediate >> 7u);
    const unsigned nch = n2g >> 1u;
    const unsigned c_base = (cidx * n2g) >> 1u;
    // R4: wave 0 fills the LDS row table of this tile, the CTA barrier publishes it to both groups.
#if V70_R2X_COMPACT
    if (cmp) r2x::rtab_fill(tid, stage2_ids, weights, (int)valid_rows,
                            [&](int r) { return v70c_pos(expert_id, cidx_t, r); });
    else
#endif
    r2x::rtab_fill(tid, stage2_ids, weights, (int)valid_rows, [&](int r) { return (int)row_start + r; });
    hr1_barrier();
    #pragma clang loop unroll(disable)
    for (unsigned c0 = 0u; c0 < nch; c0 += 2u) {
      const unsigned c = c0 + grp;
      // Same anti-LICM re-definition as GEMM1 (the chunk loop runs twice for n2g = 8).
      const unsigned* row_dir = (V70_S1F_EN) ? reinterpret_cast<const unsigned*>(e.sorted_weights) : nullptr;
      const uint64_t* peer_tab = reinterpret_cast<const uint64_t*>(o.peer_return_payload_table);
      const uint64_t* node_tab = reinterpret_cast<const uint64_t*>(x.node_partial_table);
      unsigned node_row_base = (e.rank & 7u) * x.node_tokens;
      unsigned my_node = e.rank >> 3u;
      const float* wts_l = weights;
      unsigned vr_l = valid_rows, ex_l = expert_id;
      asm volatile("" : "+s"(wts_l), "+s"(vr_l), "+s"(ex_l), "+s"(row_dir),
                        "+s"(peer_tab), "+s"(node_tab), "+s"(node_row_base), "+s"(my_node));
      if (c < nch)
        r2x::unit<0, V70_R2X_G2SW>(glds, (int)(tid & 255u), (int)vr_l, (int)(c_base + c), (int)ex_l,
                     reinterpret_cast<const uint8_t*>(ck.ck_quant_output), reinterpret_cast<const uint8_t*>(o.ck_w2),
                     reinterpret_cast<const float*>(ck.ck_quant_scale), reinterpret_cast<const float*>(o.ck_w2_scale),
                     c_out, nullptr, wts_l, ck.ck_intermediate, o.hidden, run,
                     row_dir, peer_tab, node_tab, node_row_base, my_node);
      else
        r2x::idle<V70_R2X_G2SW>(nk2 + 1);
    }
  }
#else
  #pragma clang loop unroll(disable)
  for (unsigned g = 0u; g < n2g; ++g) {
    const unsigned n2 = cidx * n2g + g;
    // Fixed 128-row stage-2 tiles: one cooperative 8-wave body per 128-column block.
    // Compute first: no S1F or dense-output address is in scope across its K-loop.
#if V70_BM64_8W_ACTIVE
    v70_s2_8wave_compute_bm64(ids, (int32_t)expert_id,
        reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
        reinterpret_cast<const uint8_t*>(o.ck_w2),
        reinterpret_cast<const float*>(ck.ck_quant_scale),
        reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
        valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
#else
    hr1_s2_8wave_compute(ids, (int32_t)expert_id,
        reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
        reinterpret_cast<const uint8_t*>(o.ck_w2),
        reinterpret_cast<const float*>(ck.ck_quant_scale),
        reinterpret_cast<const float*>(o.ck_w2_scale), weights + row_start,
        valid_rows, n2, run, o.hidden, ck.ck_intermediate, arena);
#endif
    // Start the store-address live ranges only after the K-loop has finished.
    asm volatile("" ::: "memory");
    const unsigned* row_dir = (V70_S1F_EN)
        ? reinterpret_cast<const unsigned*>(e.sorted_weights) : nullptr;
    const uint64_t* peer_tab = reinterpret_cast<const uint64_t*>(o.peer_return_payload_table);
    const uint64_t* node_tab = reinterpret_cast<const uint64_t*>(x.node_partial_table);
    const unsigned node_row_base = (e.rank & 7u) * x.node_tokens;
    const unsigned my_node = e.rank >> 3u;
#if V70_BM64_8W_ACTIVE
    v70_s2_8wave_store_bm64(c_out, row_start, valid_rows, n2, run, o.hidden,
                            row_dir, peer_tab, node_tab, node_row_base, my_node);
#else
    hr1_s2_8wave_store(c_out, row_start, valid_rows, n2, run, o.hidden,
                       row_dir, peer_tab, node_tab, node_row_base, my_node);
#endif
  }
#endif  // V70_R2X
  __builtin_amdgcn_s_waitcnt(0);
  hr1_barrier();
  // Re-materialise only when the epilogue publishes S1F completions; this is after all GEMM2 K-loops.
  const unsigned* s1f_row_dir = (V70_S1F_EN)
      ? reinterpret_cast<const unsigned*>(e.sorted_weights) : nullptr;
  FI_EG(V1988_FI_CORE, V1988_EV_GEMM2_TILE, g2_aux);
  if (tid == 0u) {
    sh_done[0] = __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.row_n_done) + mts, n2g,
                                        __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT) + n2g;
  }
  hr1_barrier();
  const unsigned completed = __builtin_amdgcn_readfirstlane(sh_done[0]);
  if (completed == n2_blocks) {
#if V46_R1 >= 2
    __threadfence();
#endif
#if V46_R1 >= 1
    v42_acq_agent();
#endif
    for (unsigned r = tid; r < valid_rows; r += 512u) {
#if V70_R2X
      const unsigned row = (unsigned)r2x::rtab()[r].x;   // R4: same LDS row table as the GEMM2 body (valid <= 64)
#else
      const unsigned row = row_start + r;
#endif
      const unsigned id = row_route_id[row];
      if (id == 0xffffffffu) continue;
      const unsigned g2p = id & 0xffffffu;
      const unsigned long long divmod = v983_divmod_u32(g2p, e.capacity + 1u);
      const unsigned src = (unsigned)(divmod >> 32);
      const unsigned slot = (unsigned)divmod;
      const unsigned key = src * e.capacity + slot;
      if (s1f_row_dir != nullptr && s1f_row_dir[row] != 0u) {
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
    hr1_barrier();
    if (tid == 0u && compute_done_ptr) {
      __hip_atomic_fetch_add(compute_done_ptr, o.n2_blocks, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#if V70_R2X_COMPACT
      __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.plan_flags) + V70M_G2D, 1u,
                             __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
    }
  }
  return true;
}


// =============================================================================================================
// V70_R2X_44: one native four-wave R2X compute group on W00..W03 of a 4/4 CTA.  W04..W07 are ordinary
// communication workers and never enter these functions.  Every intra-item rendezvous therefore uses the
// four-wave software barrier at LDS 65440; no CTA-wide s_barrier exists in this path.
//
// A 4/4 CU runs the same standalone R2 geometry as the 8/0 pair: group 0 executes the complete R2 work item
// sequentially (both GEMM1 n1 sub-blocks; all GEMM2 256-column chunks).  Keeping the item boundary unchanged
// lets both layouts share the exact claim cursors, completion counters and per-tile work accounting.
// =============================================================================================================
#ifndef V70_R2X_44
#define V70_R2X_44 0
#endif
#ifndef V70_R6_GUARD
#define V70_R6_GUARD 0
#endif
#ifndef V70R8_RREL
#define V70R8_RREL 368u
#define V70R8_RSCT 369u
#endif
#ifndef V70_R8_CNT
#define V70_R8_CNT ((V70_R8_HYB) == 2)
#endif
#if V70_R2X && V70_R2X_44

__device__ __forceinline__ void v70_gemm1_item4(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned char* arena,
    unsigned mt, unsigned s1s, unsigned n1 FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const V968CkExt& ck = env.ck;
  const unsigned tid = threadIdx.x;
  const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(ck.ck_a + 64u);
  const int32_t* stage1_ids = reinterpret_cast<const int32_t*>(ck.ck_sorted_token_ids);
#if V70_R2X_COMPACT
  const bool cmp = (mt & V70C_FLAG) != 0u;
  const unsigned cidx_t = mt & 0xffffu;
  WeaveMTileDesc d_raw;
  if (cmp) {
    const unsigned lid = v70c_lid(v70_claim_le(env, reinterpret_cast<const unsigned*>(o.plan_flags)), cidx_t);
    d_raw = WeaveMTileDesc{lid, 0u, 64u, (uint16_t)v70c_valid(lid, cidx_t), 0u};
  } else {
    d_raw = tiles[mt];
  }
#else
  const WeaveMTileDesc d_raw = tiles[mt];
#endif
  const unsigned expert_id = __builtin_amdgcn_readfirstlane(d_raw.expert_id);
  const unsigned vr2 = __builtin_amdgcn_readfirstlane(d_raw.valid_rows);

#if V70_R6_GUARD
  {
    const unsigned le_g = v70_claim_le(env, reinterpret_cast<const unsigned*>(o.plan_flags));
    if (expert_id >= le_g || vr2 > 64u) {
      if (tid == 0u) {
        auto* tr_g = reinterpret_cast<unsigned long long*>(o.trace);
        const unsigned k_g = __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.plan_flags) + 1000u, 1u,
                                                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) & 3u;
        if (tr_g) {
          tr_g[3600u + 8u * k_g + 0u] = 1u; tr_g[3600u + 8u * k_g + 1u] = mt;
          tr_g[3600u + 8u * k_g + 2u] = expert_id; tr_g[3600u + 8u * k_g + 3u] = vr2;
          tr_g[3600u + 8u * k_g + 4u] = blockIdx.x; tr_g[3600u + 8u * k_g + 5u] = le_g;
          tr_g[3600u + 8u * k_g + 6u] = reinterpret_cast<const unsigned*>(o.plan_flags)[357];
          tr_g[3600u + 8u * k_g + 7u] = __builtin_amdgcn_s_memrealtime();
        }
        v983_error(reinterpret_cast<unsigned*>(e.error), 990u);
      }
      return;
    }
  }
#endif
  const unsigned bm = (WEAVE_ADAPT_BM) ? (V70C_BM) : (WEAVE_BLOCK_M_L1);
  const unsigned s1off = s1s * bm;
  const unsigned row_start = __builtin_amdgcn_readfirstlane(d_raw.row_start) + s1off;
  const unsigned valid_rows = (s1off < vr2) ? ((vr2 - s1off > bm) ? bm : (vr2 - s1off)) : 0u;
#if V70_XSHADOW
  const unsigned char* recv_payload = v70_xshadow_pay(env);
  const float* recv_scale = v70_xshadow_scl(env);
#else
  unsigned rk = __builtin_amdgcn_readfirstlane(e.rank), cap = __builtin_amdgcn_readfirstlane(e.capacity);
  unsigned pbw = __builtin_amdgcn_readfirstlane(e.payload_bytes), sgw = __builtin_amdgcn_readfirstlane(e.scale_groups);
  asm volatile("" : "+s"(rk), "+s"(cap), "+s"(pbw), "+s"(sgw));
  const uint64_t pay_e = reinterpret_cast<const uint64_t*>(e.peer_payload_table)[rk];
  const uint64_t scl_e = reinterpret_cast<const uint64_t*>(e.peer_scale_table)[rk];
  const uint64_t pay_u = v46_rfl_u64(pay_e), scl_u = v46_rfl_u64(scl_e);
  const unsigned rc = rk * cap;
  const unsigned char* recv_payload = reinterpret_cast<const unsigned char*>(pay_u - static_cast<uint64_t>(rc) * pbw);
  const float* recv_scale = reinterpret_cast<const float*>(scl_u - static_cast<uint64_t>(rc) * sgw * 4ull);
#endif
  const unsigned g1_aux = (bm << 16u) | valid_rows;
  FI_BG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
  const unsigned n1b_l = ck.ck_intermediate / 128u;
  unsigned n1g = (V55_N1G); if (n1g > n1b_l - n1) n1g = n1b_l - n1;
  unsigned g1inc = 0u;
  if constexpr (!(WEAVE_ADAPT_BM) && ((WEAVE_BLOCK_M_L1) == 32) && ((V70C_BM) == 64)) {
    if (vr2 != 0u) {
      const unsigned nsub_in = __builtin_amdgcn_readfirstlane((vr2 > 32u) ? 2u : 1u);
      #pragma clang loop unroll(disable)
      for (unsigned sp_i = 0u; sp_i < nsub_in; ++sp_i) {
        if (sp_i != 0u) r2x::bar<V70_R2X_4W_SW>();
        const unsigned roff = sp_i * 32u;
        const unsigned vr_s = __builtin_amdgcn_readfirstlane((vr2 - roff > 32u) ? 32u : (vr2 - roff));
#if V70_R2X_COMPACT
        if (cmp) r2x::rtab_fill(tid, stage1_ids, nullptr, (int)vr_s,
                                [&](int r) { return v70c_pos(expert_id, cidx_t, (int)roff + r); });
        else
#endif
        r2x::rtab_fill(tid, stage1_ids, nullptr, (int)vr_s, [&](int r) { return (int)(row_start + roff) + r; });
        r2x::bar<V70_R2X_4W_SW>();
        #pragma clang loop unroll(disable)
        for (unsigned j = 0u; j < n1g; ++j) {
#if V70_E3_PRIO
          if ((blockIdx.x >= 80u) ^ ((j & 1u) != 0u)) __builtin_amdgcn_s_setprio(2);
          else __builtin_amdgcn_s_setprio(1);
#endif
          const unsigned char* pay_l = recv_payload; const float* rsc_l = recv_scale;
          unsigned vr_l = vr_s, ex_l = expert_id;
          asm volatile("" : "+s"(pay_l), "+s"(rsc_l), "+s"(vr_l), "+s"(ex_l));
          r2x::unit<1, V70_R2X_4W_SW>(g_v1262_phase_arena, (int)tid, (int)vr_l, (int)(n1 + j), (int)ex_l,
                          pay_l, reinterpret_cast<const uint8_t*>(ck.ck_b), rsc_l,
                          reinterpret_cast<const float*>(ck.ck_b_scale), reinterpret_cast<uint8_t*>(ck.ck_quant_output),
                          reinterpret_cast<float*>(ck.ck_quant_scale), nullptr, ck.ck_hidden, ck.ck_intermediate,
                          ck.ck_route_rows, nullptr, nullptr, nullptr, 0u, 0u);
          if (sp_i == 0u) ++g1inc;
        }
      }
    }
  } else if (valid_rows != 0u) {
#if V70_R2X_COMPACT
    if (cmp) r2x::rtab_fill(tid, stage1_ids, nullptr, (int)valid_rows,
                            [&](int r) { return v70c_pos(expert_id, cidx_t, (int)s1off + r); });
    else
#endif
    r2x::rtab_fill(tid, stage1_ids, nullptr, (int)valid_rows, [&](int r) { return (int)row_start + r; });
    r2x::bar<V70_R2X_4W_SW>();
    #pragma clang loop unroll(disable)
    for (unsigned j = 0u; j < n1g; ++j) {
#if V70_E3_PRIO
      if ((blockIdx.x >= 80u) ^ ((j & 1u) != 0u)) __builtin_amdgcn_s_setprio(2);
      else __builtin_amdgcn_s_setprio(1);
#endif
      const unsigned char* pay_l = recv_payload; const float* rsc_l = recv_scale;
      unsigned vr_l = valid_rows, ex_l = expert_id;
      asm volatile("" : "+s"(pay_l), "+s"(rsc_l), "+s"(vr_l), "+s"(ex_l));
      r2x::unit<1, V70_R2X_4W_SW>(g_v1262_phase_arena, (int)tid, (int)vr_l, (int)(n1 + j), (int)ex_l,
                      pay_l, reinterpret_cast<const uint8_t*>(ck.ck_b), rsc_l,
                      reinterpret_cast<const float*>(ck.ck_b_scale), reinterpret_cast<uint8_t*>(ck.ck_quant_output),
                      reinterpret_cast<float*>(ck.ck_quant_scale), nullptr, ck.ck_hidden, ck.ck_intermediate,
                      ck.ck_route_rows, nullptr, nullptr, nullptr, 0u, 0u);
      ++g1inc;
    }
  }
#if V70_E3_3BAR
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
#else
  __builtin_amdgcn_s_waitcnt(0);
#endif
  r2x::bar<V70_R2X_4W_SW>();
  FI_EG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
  if (tid == 0u) {
#if V70_R2X_COMPACT
    const unsigned mts_g1 = (mt & 0xffffu) + (cmp ? __builtin_amdgcn_readfirstlane(v70m_tab()[321u]) : 0u);
#else
    const unsigned mts_g1 = mt & 0xffffu;
#endif
    const unsigned slot = mts_g1 * (V46_NSUB) + s1s;
    const unsigned n1b_all = ck.ck_intermediate / 128u;
    const unsigned done_inc = (valid_rows != 0u) ? g1inc : ((n1 == 0u) ? n1b_all : 0u);
    if (done_inc)
      __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.stage1_tile_done) + slot, done_inc,
                             (V70_E3_SC1_NOBW) ? __ATOMIC_RELAXED : __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  }
}

__device__ __forceinline__ bool v70_gemm2_item4(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x, unsigned char* arena,
    unsigned mt, unsigned cidx, unsigned n2g, unsigned* compute_done_ptr FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const V968CkExt& ck = env.ck;
  const unsigned tid = threadIdx.x;
  volatile WeaveBk64LocalU32* sh_done = v70_8w_sh(V70_8W_SH_DONE);
  volatile WeaveBk64LocalU32* sh_lam  = v70_8w_sh(V70_8W_SH_LAM);
  const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(ck.ck_a + 64u);
  const unsigned n1_blocks = ck.ck_intermediate / 128u;
  const unsigned n2_blocks = o.hidden / 128u;
  const int32_t* stage2_ids = reinterpret_cast<const int32_t*>(ck.ck_m64_sorted_ids);
  const float* weights = reinterpret_cast<const float*>(o.stage2_weights);
  const unsigned* row_route_id = reinterpret_cast<const unsigned*>(o.row_route_id);
  const unsigned run = ck.ck_route_rows;
#if V70_R2X_COMPACT
  const bool cmp = (mt & V70C_FLAG) != 0u;
  const unsigned cidx_t = mt & 0xffffu;
  WeaveMTileDesc d_raw;
  if (cmp) {
    const unsigned lid = v70c_lid(v70_claim_le(env, reinterpret_cast<const unsigned*>(o.plan_flags)), cidx_t);
    d_raw = WeaveMTileDesc{lid, 0u, 64u, (uint16_t)v70c_valid(lid, cidx_t), 0u};
  } else {
    d_raw = tiles[mt];
  }
#else
  const WeaveMTileDesc d_raw = tiles[mt];
#endif
  const unsigned expert_id = __builtin_amdgcn_readfirstlane(d_raw.expert_id);
  const unsigned row_start = __builtin_amdgcn_readfirstlane(d_raw.row_start);
  const unsigned bm = __builtin_amdgcn_readfirstlane(d_raw.bm);
  const unsigned valid_rows = __builtin_amdgcn_readfirstlane(d_raw.valid_rows);
#if V70_R2X_COMPACT
  const unsigned mts = (mt & 0xffffu) + (cmp ? __builtin_amdgcn_readfirstlane(v70m_tab()[321u]) : 0u);
#else
  const unsigned mts = mt & 0xffffu;
#endif

#if V70_R6_GUARD
  {
    const unsigned le_g = v70_claim_le(env, reinterpret_cast<const unsigned*>(o.plan_flags));
    if (expert_id >= le_g || valid_rows > 64u) {
      if (tid == 0u) {
        auto* tr_g = reinterpret_cast<unsigned long long*>(o.trace);
        const unsigned k_g = __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.plan_flags) + 1000u, 1u,
                                                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) & 3u;
        if (tr_g) {
          tr_g[3600u + 8u * k_g + 0u] = 2u; tr_g[3600u + 8u * k_g + 1u] = mt;
          tr_g[3600u + 8u * k_g + 2u] = expert_id; tr_g[3600u + 8u * k_g + 3u] = valid_rows;
          tr_g[3600u + 8u * k_g + 4u] = blockIdx.x; tr_g[3600u + 8u * k_g + 5u] = le_g;
          tr_g[3600u + 8u * k_g + 6u] = reinterpret_cast<const unsigned*>(o.plan_flags)[357];
          tr_g[3600u + 8u * k_g + 7u] = __builtin_amdgcn_s_memrealtime();
        }
        v983_error(reinterpret_cast<unsigned*>(e.error), 990u);
      }
      return false;
    }
  }
#endif
#if !(V70_E3_3BAR && V70_E3_SC1_NOBW)
  FI_BG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt);
  const unsigned last_acq_mt = __builtin_amdgcn_readfirstlane(sh_lam[0]);
  const unsigned lam_key = (mt & V70C_FLAG) ? (V70C_FLAG | mts) : mt;
  if (lam_key != last_acq_mt) {
    volatile WeaveBk64LocalU32* sh_w = v70_8w_sh(V70_8W_SH_W);
    r2x::bar<V70_R2X_4W_SW>();
    if (tid == 0u) {
      unsigned backoff = 1u, spins = 0u, ok = 1u;
      while (!v46_s1_ready(reinterpret_cast<const unsigned*>(o.stage1_tile_done), mts, n1_blocks)) {
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
    }
    r2x::bar<V70_R2X_4W_SW>();
    if (__builtin_amdgcn_readfirstlane(sh_w[0]) == 0u) {
      FI_EG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt);
      return false;
    }
#if V70_S1_ACQ == 0
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
#elif V70_S1_ACQ == 2 || !V54_S1_BYPASS_MALL
    asm volatile("buffer_inv sc1\ns_waitcnt vmcnt(0)\n" ::: "memory");
#else
    asm volatile("buffer_inv sc0\ns_waitcnt vmcnt(0)\n" ::: "memory");
#endif
    if (tid == 0u) sh_lam[0] = lam_key;
    r2x::bar<V70_R2X_4W_SW>();
  }
  FI_EG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt);
#endif
  const unsigned g2_aux = ((n2g & 0x7u) << 28u) | ((bm & 0xfffu) << 16u) | (valid_rows & 0xffffu);
  FI_BG(V1988_FI_CORE, V1988_EV_GEMM2_TILE, g2_aux);
  uint16_t* c_out = reinterpret_cast<uint16_t*>(o.gemm2_output);
  const unsigned nk2 = (int)(ck.ck_intermediate >> 7);
  const unsigned nch = n2g >> 1u;
  const unsigned c_base = (cidx * n2g) >> 1u;
  if ((n2g & 1u) && tid == 0u) v983_error(reinterpret_cast<unsigned*>(e.error), 977u);
  if (valid_rows != 0u) {
#if V70_R2X_COMPACT
    if (cmp) r2x::rtab_fill(tid, stage2_ids, weights, (int)valid_rows,
                            [&](int r) { return v70c_pos(expert_id, cidx_t, r); });
    else
#endif
    r2x::rtab_fill(tid, stage2_ids, weights, (int)valid_rows, [&](int r) { return (int)row_start + r; });
    r2x::bar<V70_R2X_4W_SW>();
    #pragma clang loop unroll(disable)
    for (unsigned c = 0u; c < nch; ++c) {
#if V70_E3_PRIO
      if ((blockIdx.x >= 80u) ^ ((c & 1u) != 0u)) __builtin_amdgcn_s_setprio(2);
      else __builtin_amdgcn_s_setprio(1);
#endif
      const unsigned* row_dir = (V70_S1F_EN) ? reinterpret_cast<const unsigned*>(e.sorted_weights) : nullptr;
      const uint64_t* peer_tab = reinterpret_cast<const uint64_t*>(o.peer_return_payload_table);
      const uint64_t* node_tab = reinterpret_cast<const uint64_t*>(x.node_partial_table);
      unsigned node_row_base = (e.rank & 7u) * x.node_tokens;
      unsigned my_node = e.rank >> 3u;
      const float* wts_l = weights;
      unsigned vr_l = valid_rows, ex_l = expert_id;
      asm volatile("" : "+s"(wts_l), "+s"(vr_l), "+s"(ex_l), "+s"(row_dir),
                        "+s"(peer_tab), "+s"(node_tab), "+s"(node_row_base), "+s"(my_node));
      r2x::unit<0, V70_R2X_4W_SW>(g_v1262_phase_arena, (int)tid, (int)vr_l, (int)(c_base + c), (int)ex_l,
                      reinterpret_cast<const uint8_t*>(ck.ck_quant_output), reinterpret_cast<const uint8_t*>(o.ck_w2),
                      reinterpret_cast<const float*>(ck.ck_quant_scale), reinterpret_cast<const float*>(o.ck_w2_scale),
                      c_out, nullptr, wts_l, ck.ck_intermediate, o.hidden, run,
                      row_dir, peer_tab, node_tab, node_row_base, my_node);
    }
  }
#if V70_E3_3BAR
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
#else
  __builtin_amdgcn_s_waitcnt(0);
#endif
  r2x::bar<V70_R2X_4W_SW>();
  const unsigned* s1f_row_dir = (V70_S1F_EN)
      ? reinterpret_cast<const unsigned*>(e.sorted_weights) : nullptr;
  FI_EG(V1988_FI_CORE, V1988_EV_GEMM2_TILE, g2_aux);
  if (tid == 0u) {
    sh_done[0] = __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.row_n_done) + mts, n2g,
                                        (V70_E3_SC1_NOBW) ? __ATOMIC_RELAXED : __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT) + n2g;
  }
  r2x::bar<V70_R2X_4W_SW>();
  const unsigned completed = __builtin_amdgcn_readfirstlane(sh_done[0]);
  if (completed == n2_blocks) {
#if V46_R1 >= 2
    __threadfence();
#endif
#if V46_R1 >= 1 && !(V70_R6_NOACQ_G2DONE)
    v42_acq_agent();
#endif
    for (unsigned r = tid; r < valid_rows; r += 256u) {
      const unsigned row = (unsigned)r2x::rtab()[r].x;
      const unsigned id = row_route_id[row];
      if (id == 0xffffffffu) continue;
      const unsigned g2p = id & 0xffffffu;
      const unsigned long long divmod = v983_divmod_u32(g2p, e.capacity + 1u);
      const unsigned src = (unsigned)(divmod >> 32);
      const unsigned slot = (unsigned)divmod;
      const unsigned key = src * e.capacity + slot;
      if (s1f_row_dir != nullptr && s1f_row_dir[row] != 0u) {
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
                             (V70_E3_SC1_NOBW) ? __ATOMIC_RELAXED : V46_BD_ORDER, __HIP_MEMORY_SCOPE_AGENT);
    }
    r2x::bar<V70_R2X_4W_SW>();
#if V70_R8_CNT
    if (tid == 0u) __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.plan_flags) + V70R8_RREL, valid_rows,
                                          __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
    if (tid == 0u && compute_done_ptr) {
      __hip_atomic_fetch_add(compute_done_ptr, o.n2_blocks, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#if V70_R2X_COMPACT
      __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.plan_flags) + V70M_G2D, 1u,
                             __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
    }
  }
  return true;
}

__device__ __forceinline__ bool v70_atrex4_r2(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena FI_CTX_PARAM) {
  const unsigned tid = threadIdx.x;
  volatile WeaveBk64LocalU32* sh = v70_8w_sh(V70_8W_SH_WORD);
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
#if V70_PROFILE_PHASES
  unsigned long long prof_g1 = 0ull, prof_g2 = 0ull, prof_ovh = 0ull, prof_idle = 0ull;
  unsigned long long prof_wait = 0ull, prof_og2 = 0ull;
  unsigned long long prof_first_g1 = 0ull, prof_first_g2 = 0ull;
  unsigned long long prof_last_g1 = 0ull, prof_last_g2 = 0ull;
  unsigned prof_ng1 = 0u, prof_ng2 = 0u;
#endif
  unsigned n1b = __builtin_amdgcn_readfirstlane(env.ck.ck_intermediate / 128u);
  unsigned n2b = __builtin_amdgcn_readfirstlane(o.hidden / 128u);
  unsigned n2g = __builtin_amdgcn_readfirstlane(v46_n2g(n2b));
  unsigned n2c = (n2g == 8u) ? (n2b >> 3u) : ((n2g == 4u) ? (n2b >> 2u) : ((n2g == 2u) ? (n2b >> 1u) : n2b));
  asm volatile("" : "+s"(n1b), "+s"(n2b), "+s"(n2g), "+s"(n2c));
  volatile WeaveBk64LocalU32* seen_sh = v70_8w_sh(V70_8W_SH_SEEN);
  volatile WeaveBk64LocalU32* lam_sh  = v70_8w_sh(V70_8W_SH_LAM);
  if (tid == 0u) { seen_sh[0] = 0u; lam_sh[0] = 0xffffffffu; }
#if V70_ATREX_1BAR
#define V70_STORE_DECISION() do { \
    if (tid == 0u) { \
      unsigned seen = seen_sh[0]; \
      unsigned w = v46_decide(env, o, seen, n1b, n2c, v54_xcd()); \
      seen_sh[0] = seen; \
      if (w == V46_W_CLAIMED && \
          __hip_atomic_load(pf + V70M_G2D, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= \
          __hip_atomic_load(pf + V70M_G2T, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) \
        w = V46_W_EXIT; \
      if ((w >> 30u) == V46_ACT_IDLE && trace && \
          __builtin_amdgcn_s_memrealtime() - trace[16u + blockIdx.x * 16u] > 4ull * x.timeout_ticks) { \
        v983_error((unsigned*)env.base.error, 970u); \
        w = V46_W_ERR; \
      } \
      sh[0] = w; \
    } \
  } while (0)
  r2x::bar<V70_R2X_4W_SW>();
  V70_STORE_DECISION();
  r2x::bar<V70_R2X_4W_SW>();
#endif
  for (;;) {
#if !V70_ATREX_1BAR
#if V70_PROFILE_PHASES
    const unsigned long long prof_t0 = __builtin_amdgcn_s_memrealtime();
#endif
    r2x::bar<V70_R2X_4W_SW>();
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
    r2x::bar<V70_R2X_4W_SW>();
#if V70_PROFILE_PHASES
    const unsigned long long prof_t1 = __builtin_amdgcn_s_memrealtime();
    prof_ovh += prof_t1 - prof_t0;
    {
      const unsigned pa = __builtin_amdgcn_readfirstlane(sh[0]) >> 30u;
      if (pa == V46_ACT_IDLE) prof_wait += prof_t1 - prof_t0;
      else if (pa == V46_ACT_G2) prof_og2 += prof_t1 - prof_t0;
    }
#endif
#endif
    const unsigned w = __builtin_amdgcn_readfirstlane(sh[0]);
    const unsigned act = w >> 30u;
#if V70_PROFILE_PHASES
    if (act == V46_ACT_END) {
      if (tid == 0u && trace) {
        const unsigned b = 16u + blockIdx.x * 16u;
        trace[b + 1u] = prof_g1; trace[b + 2u] = prof_g2;
        trace[b + 3u] = prof_ovh; trace[b + 4u] = prof_first_g1;
        trace[b + 6u] = prof_first_g2; trace[b + 7u] = prof_last_g1;
        trace[b + 9u] = prof_last_g2; trace[b + 11u] = prof_idle;
        trace[b + 14u] = prof_ng1; trace[b + 15u] = prof_ng2;
        trace[b + 5u] = prof_wait; trace[b + 13u] = prof_og2;
      }
      return ((w >> 28u) & 3u) == 0u;
    }
#else
    if (act == V46_ACT_END) return ((w >> 28u) & 3u) == 0u;
#endif
#if V70_R2X_MERGE
    if (act == V46_ACT_IDLE && ((w >> 28u) & 3u) == 1u) {
      if (tid < 64u) {
        const unsigned md = __hip_atomic_load(pf + V70M_MODE, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#if V70_R2X_COMPACT
        if (md == 4u) v70c_build(env, pf, v70_claim_le(env, pf)); else
#endif
        v70m_build(pf, v70_claim_le(env, pf), v70m_mask(md));
      }
#if V70_ATREX_1BAR
      V70_STORE_DECISION();
      r2x::bar<V70_R2X_4W_SW>();
#endif
      continue;
    }
#endif
    if (act == V46_ACT_IDLE) {
#if V70_PROFILE_PHASES
      const unsigned long long ti = __builtin_amdgcn_s_memrealtime();
      __builtin_amdgcn_s_sleep(2);
      prof_idle += __builtin_amdgcn_s_memrealtime() - ti;
#else
      __builtin_amdgcn_s_sleep(2);
#endif
#if V70_ATREX_1BAR
      V70_STORE_DECISION();
      r2x::bar<V70_R2X_4W_SW>();
#endif
      continue;
    }
    const unsigned q = (w >> 28u) & 3u, sub = (w >> 20u) & 0xffu;
    const unsigned mt = w & 0x1ffffu, s1s = (w >> 17u) & 7u;
    if (act == V46_ACT_G1) {
#if V70_PROFILE_PHASES
      const unsigned long long tw = __builtin_amdgcn_s_memrealtime();
      if (!prof_first_g1) prof_first_g1 = tw;
#endif
      v70_gemm1_item4(env, o, arena, mt, s1s, sub FI_CTX_ARG);
#if V70_PROFILE_PHASES
      prof_last_g1 = __builtin_amdgcn_s_memrealtime();
      prof_g1 += prof_last_g1 - tw;
      ++prof_ng1;
#endif
      if (tid == 0u && (mt & 0x10000u) == 0u) {
        unsigned g1inc = (V55_N1G); if (g1inc > n1b - sub) g1inc = n1b - sub;
        const unsigned d = __hip_atomic_fetch_add(pf + V46_G1D + q, g1inc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + g1inc;
        if (d == pf[V46_T + q] * n1b * (V46_NSUB) && trace) trace[1312u + q] = __builtin_amdgcn_s_memrealtime();
      }
    } else {
#if V70_PROFILE_PHASES
      const unsigned long long tw = __builtin_amdgcn_s_memrealtime();
      if (!prof_first_g2) prof_first_g2 = tw;
#endif
      if (!v70_gemm2_item4(env, o, x, arena, mt, sub, n2g, (unsigned*)x.compute_done FI_CTX_ARG)) return false;
#if V70_PROFILE_PHASES
      prof_last_g2 = __builtin_amdgcn_s_memrealtime();
      prof_g2 += prof_last_g2 - tw;
      ++prof_ng2;
#endif
      if (tid == 0u && (mt & V70C_FLAG) == 0u) {
        const unsigned d = __hip_atomic_fetch_add(pf + V46_G2D + q, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
        if (d == pf[V46_T + q] * n2c && trace) trace[1316u + q] = __builtin_amdgcn_s_memrealtime();
      }
    }
#if V70_ATREX_1BAR
    V70_STORE_DECISION();
    r2x::bar<V70_R2X_4W_SW>();
#endif
  }
#if V70_ATREX_1BAR
#undef V70_STORE_DECISION
#endif
}
#endif  // V70_R2X && V70_R2X_44

#ifndef V70_R6_SCHED
#define V70_R6_SCHED 0
#endif
#if V70_R2X && V70_R2X_44 && V70_R6_SCHED
#if !(V70_R2X_COMPACT && V70_R2X_MERGE && V70_R2X_EARLY_EXIT)
#error "V70_R6_SCHED needs the compacted merged work lists (V70_R2X_COMPACT/MERGE/EARLY_EXIT)"
#endif
// tid 0 only: wait until every pass of the plan is visible and pulled (the md==0 condition of v46_decide).
__device__ __forceinline__ bool v70_r6_plan_ready(const V983Ov1Ext& o, const unsigned* pf) {
  if (v42_ld_agent_rlx(pf + V46_LOCAL_RDY) == 0u) return false;
  if (v42_ld_agent_rlx(pf + V46_Q0_RDY) == 0u) return false;
  if (v42_ld_agent_rlx(pf + V46_P1_PLAN) == 0u) return false;
  #pragma unroll
  for (unsigned q = 0u; q < V46_NP; ++q)
    if (v42_ld_agent_rlx(pf + V46_T + q) != 0u && v42_ld_agent_rlx(pf + V46_PULL_RDY + q) == 0u) return false;
  return true;
}
__device__ __forceinline__ bool v70_atrex4_r6(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena FI_CTX_PARAM) {
  const unsigned tid = threadIdx.x;
  volatile WeaveBk64LocalU32* sh = v70_8w_sh(V70_8W_SH_WORD);
  volatile WeaveBk64LocalU32* lam_sh = v70_8w_sh(V70_8W_SH_LAM);
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
#if V70_PROFILE_PHASES
  unsigned long long prof_g1 = 0ull, prof_g2 = 0ull, prof_ovh = 0ull, prof_idle = 0ull;
  unsigned long long prof_first_g1 = 0ull, prof_first_g2 = 0ull, prof_last_g1 = 0ull, prof_last_g2 = 0ull;
  unsigned long long prof_wait = 0ull;
  unsigned prof_ng1 = 0u, prof_ng2 = 0u;
#endif
  unsigned n1b = __builtin_amdgcn_readfirstlane(env.ck.ck_intermediate / 128u);
  unsigned n2b = __builtin_amdgcn_readfirstlane(o.hidden / 128u);
  unsigned n2g = __builtin_amdgcn_readfirstlane(v46_n2g(n2b));
  unsigned n2c = (n2g == 8u) ? (n2b >> 3u) : ((n2g == 4u) ? (n2b >> 2u) : ((n2g == 2u) ? (n2b >> 1u) : n2b));
  asm volatile("" : "+s"(n1b), "+s"(n2b), "+s"(n2g), "+s"(n2c));
  // ---- one-time plan gate + LDS table ----
  {
#if V70_PROFILE_PHASES
    const unsigned long long tw0 = __builtin_amdgcn_s_memrealtime();
#endif
    if (tid == 0u) {
      lam_sh[0] = 0xffffffffu;
      unsigned ok = 1u, spins = 0u;
      while (!v70_r6_plan_ready(o, pf)) {
        __builtin_amdgcn_s_sleep(2);
        if (ep16_load_relaxed((unsigned*)env.base.error) != 0u) { ok = 0u; break; }
        if (++spins > (1u << 22)) { v983_error((unsigned*)env.base.error, 971u); ok = 0u; break; }
      }
      v42_acq_agent();
      if (ok && blockIdx.x == (V70_PATHB_NCOMM_CTA)) {
        unsigned e0 = 0u;
        __hip_atomic_compare_exchange_strong(pf + V70M_MODE, &e0, 4u, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT);
      }
      sh[0] = ok;
    }
    r2x::bar<V70_R2X_4W_SW>();
    if (__builtin_amdgcn_readfirstlane(sh[0]) == 0u) return false;
    if (tid < 64u) v70c_build(env, pf, v70_claim_le(env, pf));
    r2x::bar<V70_R2X_4W_SW>();
#if V70_PROFILE_PHASES
    prof_wait += __builtin_amdgcn_s_memrealtime() - tw0;
#endif
  }
  unsigned stage = 0u, xmask = 0u;   // tid 0 private claim state
  for (;;) {
#if V70_PROFILE_PHASES
    const unsigned long long prof_t0 = __builtin_amdgcn_s_memrealtime();
#endif
    r2x::bar<V70_R2X_4W_SW>();
    if (tid == 0u) {
      const unsigned LE = v70_claim_le(env, pf);
      const unsigned NT = v70m_tab()[LE];
      const unsigned nn1 = n1b / (V55_N1G);
      const unsigned xcd = v55_xcc() & 3u;
      unsigned w = V46_W_EXIT;
      for (;;) {
        const unsigned N = (stage == 0u) ? NT * nn1 : NT * n2c;
        unsigned* cur = pf + ((stage == 0u) ? V70M_G1C : V70M_G2C);
        unsigned got = 0xffffffffu;
        #pragma unroll
        for (unsigned dx = 0u; dx < V70_XCD_N; ++dx) {
          const unsigned xx = (xcd + dx) & 3u;
          if ((xmask >> xx) & 1u) continue;
          const unsigned bound = v70_xcd_slots(N, xx);
          const unsigned l = __hip_atomic_fetch_add(cur + xx, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          const unsigned g = v70_xcd_work(l, xx);
          if (l < bound && g < N) { got = g; break; }
          xmask |= 1u << xx;
        }
        if (got != 0xffffffffu) {
          unsigned c, j;
          if (stage == 0u) {
            v70c_map(LE, nn1, got, c, j);
            w = V46_WORD(V46_ACT_G1, 0u, j * (V55_N1G), V70C_FLAG | c);
          } else {
            v70c_map(LE, n2c, got, c, j);
            w = V46_WORD(V46_ACT_G2, 0u, j, V70C_FLAG | c);
          }
          break;
        }
        if (stage == 0u) { stage = 1u; xmask = 0u; continue; }
        break;
      }
      if (ep16_load_relaxed((unsigned*)env.base.error) != 0u) w = V46_W_ERR;
      sh[0] = w;
    }
    r2x::bar<V70_R2X_4W_SW>();
    const unsigned w = __builtin_amdgcn_readfirstlane(sh[0]);
    const unsigned act = w >> 30u;
#if V70_PROFILE_PHASES
    prof_ovh += __builtin_amdgcn_s_memrealtime() - prof_t0;
    if (act == V46_ACT_END) {
      if (tid == 0u && trace) {
        const unsigned b = 16u + blockIdx.x * 16u;
        trace[b + 1u] = prof_g1; trace[b + 2u] = prof_g2;
        trace[b + 3u] = prof_ovh; trace[b + 4u] = prof_first_g1;
        trace[b + 6u] = prof_first_g2; trace[b + 7u] = prof_last_g1;
        trace[b + 9u] = prof_last_g2; trace[b + 11u] = prof_idle;
        trace[b + 14u] = prof_ng1; trace[b + 15u] = prof_ng2;
        unsigned hw_id_, xcc_id_;  // R6_HWID
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(hw_id_));
        asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xcc_id_));
        trace[b + 5u] = prof_wait; trace[b + 13u] = ((unsigned long long)xcc_id_ << 32) | hw_id_;
      }
      return ((w >> 28u) & 3u) == 0u;
    }
#else
    if (act == V46_ACT_END) return ((w >> 28u) & 3u) == 0u;
#endif
    const unsigned sub = (w >> 20u) & 0xffu, mt = w & 0x1ffffu, s1s = (w >> 17u) & 7u;
    if (act == V46_ACT_G1) {
#if V70_PROFILE_PHASES
      const unsigned long long tw = __builtin_amdgcn_s_memrealtime();
      if (!prof_first_g1) prof_first_g1 = tw;
#endif
      v70_gemm1_item4(env, o, arena, mt, s1s, sub FI_CTX_ARG);
#if V70_PROFILE_PHASES
      prof_last_g1 = __builtin_amdgcn_s_memrealtime();
      prof_g1 += prof_last_g1 - tw;
      ++prof_ng1;
#endif
    } else {
#if V70_PROFILE_PHASES
      const unsigned long long tw = __builtin_amdgcn_s_memrealtime();
      if (!prof_first_g2) prof_first_g2 = tw;
#endif
      if (!v70_gemm2_item4(env, o, x, arena, mt, sub, n2g, (unsigned*)x.compute_done FI_CTX_ARG)) return false;
#if V70_PROFILE_PHASES
      prof_last_g2 = __builtin_amdgcn_s_memrealtime();
      prof_g2 += prof_last_g2 - tw;
      ++prof_ng2;
#endif
    }
  }
}
#endif  // V70_R6_SCHED
#include "v70_r8.hpp"

// -------------------------------------------------------------------------------------------------------------
// Atrex loop, ONE unified 8-wave group (all 512 threads).  Same decision function, same global bounded-CAS
// cursors and exit test as the 4-wave version; only group width and barrier primitive change.
// -------------------------------------------------------------------------------------------------------------
__device__ __forceinline__ bool v70_atrex8(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena FI_CTX_PARAM) {
  const unsigned tid = threadIdx.x;
  volatile WeaveBk64LocalU32* sh = v70_8w_sh(V70_8W_SH_WORD);
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
  unsigned n1b = __builtin_amdgcn_readfirstlane(env.ck.ck_intermediate / 128u);
  unsigned n2b = __builtin_amdgcn_readfirstlane(o.hidden / 128u);
  unsigned n2g = __builtin_amdgcn_readfirstlane(v46_n2g(n2b));
#if V55_COALESCE_SCHED
  unsigned n2c = (n2g == 8u) ? (n2b >> 3u) : ((n2g == 4u) ? (n2b >> 2u) : ((n2g == 2u) ? (n2b >> 1u) : n2b));
#else
  unsigned n2c = (n2g == 4u) ? (n2b >> 2u) : ((n2g == 2u) ? (n2b >> 1u) : n2b);
#endif
  asm volatile("" : "+s"(n1b), "+s"(n2b), "+s"(n2g), "+s"(n2c));
  volatile WeaveBk64LocalU32* seen_sh = v70_8w_sh(V70_8W_SH_SEEN);
  volatile WeaveBk64LocalU32* lam_sh  = v70_8w_sh(V70_8W_SH_LAM);
  if (tid == 0u) { seen_sh[0] = 0u; lam_sh[0] = 0xffffffffu; }
  for (;;) {
    hr1_barrier();
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
    hr1_barrier();
    const unsigned w = __builtin_amdgcn_readfirstlane(sh[0]);
    const unsigned act = w >> 30u;
    if (act == V46_ACT_END) return ((w >> 28u) & 3u) == 0u;
#if V70_R2X_MERGE
    // q == 1 on an IDLE word: the launch just entered merged order; wave 0 builds the LDS claim table.
    // The next hr1_barrier (lgkmcnt(0) + s_barrier) publishes it before tid 0 decides again.
    if (act == V46_ACT_IDLE && ((w >> 28u) & 3u) == 1u) {
      if (tid < 64u) {
        const unsigned md = __hip_atomic_load(pf + V70M_MODE, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#if V70_R2X_COMPACT
        if (md == 4u) v70c_build(env, pf, v70_claim_le(env, pf)); else
#endif
        v70m_build(pf, v70_claim_le(env, pf), v70m_mask(md));
      }
      continue;
    }
#endif
    if (act == V46_ACT_IDLE) { __builtin_amdgcn_s_sleep(2); continue; }
    const unsigned q = (w >> 28u) & 3u, sub = (w >> 20u) & 0xffu;
    const unsigned mt = w & 0x1ffffu, s1s = (w >> 17u) & 7u;
    if (act == V46_ACT_G1) {
      v70_gemm1_item8(env, o, arena, mt, s1s, sub FI_CTX_ARG);
      if (tid == 0u && (mt & 0x10000u) == 0u) {
#if V55_COALESCE_SCHED
        unsigned g1inc = (V55_N1G); if (g1inc > n1b - sub) g1inc = n1b - sub;
#else
        const unsigned g1inc = 1u;
#endif
        const unsigned d = __hip_atomic_fetch_add(pf + V46_G1D + q, g1inc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + g1inc;
        if (d == pf[V46_T + q] * n1b * (V46_NSUB) && trace) trace[1312u + q] = __builtin_amdgcn_s_memrealtime();
      }
    } else {
      if (!v70_gemm2_item8(env, o, x, arena, mt, sub, n2g, (unsigned*)x.compute_done FI_CTX_ARG)) return false;
      if (tid == 0u && (mt & V70C_FLAG) == 0u) {
        const unsigned d = __hip_atomic_fetch_add(pf + V46_G2D + q, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) + 1u;
        if (d == pf[V46_T + q] * n2c && trace) trace[1316u + q] = __builtin_amdgcn_s_memrealtime();
      }
    }
  }
}

#endif  // V70_8W_ACTIVE
