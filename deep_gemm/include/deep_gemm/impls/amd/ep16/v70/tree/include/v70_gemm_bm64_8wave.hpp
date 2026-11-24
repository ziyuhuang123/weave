#pragma once
// ============================================================================
// V70: cooperative hardware-s_barrier GEMM1/GEMM2 for fixed BM=64, BN=128.
//
// The BM=128 HR1 body assigns eight waves as 2 M bands x 4 N bands.  BM=64
// keeps all eight waves active by assigning them as 4 M bands x 2 N bands:
//   wave: wm = wave >> 1 (0..3), wn = wave & 1 (0..1)
//   each wave: one 16-row M tile x four 16-column N tiles = 16 x 64 outputs.
//
// Every K step has exactly one workgroup-wide s_barrier.  A global load is
// issued exactly once per CTA and consumed from the full 64 KB LDS arena.
// ============================================================================

#include "hr1_gemm_8wave.hpp"

#define V70_BM64 64u
#define V70_BM64_BN 128u
#define V70_BM64_BK 64u

// GEMM1 LDS map (two stages):
//   [0, 8192)       A: 2 x 64 x 64 B
//   [8192, 40960)   B: gate/up, 2 x 16 x 1024 B
//   [40960, 41216)  token ids
//   [41216, 41728)  A scales: 2 x 64 f32
// The 16 KiB BF16 epilogue reuses [0, 16384).
#define V70_BM64_S1_A0    0u
#define V70_BM64_S1_ASTR  4096u
#define V70_BM64_S1_B0    8192u
#define V70_BM64_S1_BSTR  16384u
#define V70_BM64_S1_TOK   40960u
#define V70_BM64_S1_SA    41216u
#define V70_BM64_S1_EPI   0u

// GEMM2 LDS map:
//   [0, 8192)       A: 2 x 64 x 64 B
//   [8192, 24576)   B: two stages x 8 x 1024 B
//   [24576, 24832)  token ids
//   [24832, 25344)  A scales
#define V70_BM64_S2_A0    0u
#define V70_BM64_S2_ASTR  4096u
#define V70_BM64_S2_B0    8192u
#define V70_BM64_S2_BSTR  8192u
#define V70_BM64_S2_TOK   24576u
#define V70_BM64_S2_SA    24832u
#define V70_BM64_S2_EPI   0u

__device__ __forceinline__ void v70_bm64_s1_k64(
    const WeaveLdsU8* a_p, const WeaveLdsF32* sa_p,
    const WeaveLdsU8* b_p, float sb_g, float sb_u,
    WeaveAcc4 (&gate_acc)[4], WeaveAcc4 (&up_acc)[4]) {
  __builtin_amdgcn_sched_barrier(0);
  // A 16-byte LDS fragment contains both K32 halves for this 16-row band: av[0]
  // is K[0,32), av[1] is K[32,64).  Each wave owns one M band, so the second
  // K half must reuse av, not load the next 16-row band at +1024 B.
  const WeaveI64x2 av = weave_lds_load_i64x2(a_p);
  const float sab_g = sa_p[0] * sb_g;
  const float sab_u = sa_p[0] * sb_u;

  // Four independent N tiles are issued before either K=32 half is consumed a
  // second time.  This gives the MFMA dependency a four-instruction distance
  // while keeping the accumulator set to 4 x 4 VGPRs per matrix.
  WeaveI64x2 bg[4];
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) bg[n] = weave_lds_load_i64x2(b_p + n * 1024u);
  WeaveFloat4 tg[4];
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) {
    tg[n] = {0, 0, 0, 0};
    tg[n] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[n][0], av[0], tg[n], 0, 0, 0);
  }
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n)
    tg[n] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[n][1], av[1], tg[n], 0, 0, 0);
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) weave_acc_pk_fma4(gate_acc[n], tg[n], sab_g);

  WeaveI64x2 bu[4];
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) bu[n] = weave_lds_load_i64x2(b_p + 8192u + n * 1024u);
  WeaveFloat4 tu[4];
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) {
    tu[n] = {0, 0, 0, 0};
    tu[n] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[n][0], av[0], tu[n], 0, 0, 0);
  }
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n)
    tu[n] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[n][1], av[1], tu[n], 0, 0, 0);
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) weave_acc_pk_fma4(up_acc[n], tu[n], sab_u);
}

__device__ __forceinline__ void v70_bm64_s2_k64(
    const WeaveLdsU8* a_p, const WeaveLdsF32* sa_p,
    const WeaveLdsU8* b_p, float sb, WeaveAcc4 (&acc)[4]) {
  __builtin_amdgcn_sched_barrier(0);
  // A 16-byte LDS fragment contains both K32 halves for this 16-row band: av[0]
  // is K[0,32), av[1] is K[32,64).  Each wave owns one M band, so the second
  // K half must reuse av, not load the next 16-row band at +1024 B.
  const WeaveI64x2 av = weave_lds_load_i64x2(a_p);
  const float sab = sa_p[0] * sb;
  WeaveI64x2 b[4];
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) b[n] = weave_lds_load_i64x2(b_p + n * 1024u);
  WeaveFloat4 t[4];
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) {
    t[n] = {0, 0, 0, 0};
    t[n] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[n][0], av[0], t[n], 0, 0, 0);
  }
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n)
    t[n] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b[n][1], av[1], t[n], 0, 0, 0);
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) weave_acc_pk_fma4(acc[n], t[n], sab);
}

__device__ __forceinline__ void v70_s1_8wave_bm64(
    const int32_t* __restrict__ s_ids_tile, int32_t expert_id,
    const uint8_t* __restrict__ a_base, const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base, const float* __restrict__ b_scale_base,
    uint8_t* __restrict__ quant_out_base, float* __restrict__ quant_scale_base,
    unsigned row_offset, unsigned valid_rows, unsigned tile_n,
    unsigned num_tokens, unsigned route_rows, unsigned N, unsigned K,
    unsigned char* arena) {
  (void)arena;
  WeaveLdsU8* const lds = reinterpret_cast<WeaveLdsU8*>(
      reinterpret_cast<uintptr_t>(HR1_LDS_ARENA));
  const unsigned tid = threadIdx.x;
  const unsigned wave = tid >> 6u;
  const unsigned wm = wave >> 1u;              // 0..3 (16 rows each)
  const unsigned wn = wave & 1u;               // 0..1  (64 cols each)
  const unsigned lane = tid & 63u;
  const unsigned m_lane = lane & 15u;
  const unsigned k_grp = lane >> 4u;

  const unsigned num_k128 = __builtin_amdgcn_readfirstlane(K >> 7u);
  const unsigned num_k64 = __builtin_amdgcn_readfirstlane(K >> 6u);
  const unsigned n_blocks = __builtin_amdgcn_readfirstlane(N >> 7u);
  const unsigned exp_u = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
  const uint32_t nk_bytes = __builtin_amdgcn_readfirstlane(N * K);
  const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(exp_u) *
      (static_cast<uint64_t>(nk_bytes) * 2ull);
  const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * num_k128;
  const float* bs_gate_ptr = b_scale_base + static_cast<uint64_t>(exp_u) * expert_bs_stride
      + static_cast<uint64_t>(tile_n) * num_k128;
  const uint32_t bs_up_stride = __builtin_amdgcn_readfirstlane(n_blocks * num_k128);
  WeaveConstF32As4* bs_gate_s = weave_make_as4_f32(bs_gate_ptr);

  const uint32_t a_limit = __builtin_amdgcn_readfirstlane(num_tokens * K);
  const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(num_tokens * num_k128 * 4u);
  const uint32_t b_limit = __builtin_amdgcn_readfirstlane(nk_bytes * 2u);
  const WeaveI32x4 a_rsrc = weave_make_buf_rsrc(a_base, a_limit);
  const WeaveI32x4 sa_rsrc = weave_make_buf_rsrc(a_scale_base, sa_limit);
  const WeaveI32x4 b_rsrc = weave_make_buf_rsrc(b_gate_exp, b_limit);
  const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);

  WeaveLdsU32* const lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + V70_BM64_S1_TOK);
  const WeaveI32x4 ids_rsrc = weave_make_buf_rsrc(s_ids_tile, V70_BM64 * 4u);
  if (tid < V70_BM64) {
    const uint32_t tok = static_cast<uint32_t>(
        hr1_raw_buf_load_i32(ids_rsrc, static_cast<int32_t>(tid * 4u), 0, 0)) & 0xffffffu;
    lds_tok[tid] = (tid < valid_rows) ? tok : num_tokens;
  }
  hr1_barrier();

  const unsigned a_row = (tid >> 2u) & (V70_BM64 - 1u);
  const bool a_owner = tid < (V70_BM64 * 4u);
  const unsigned a_chk = tid & 3u;
  const uint32_t tok_a = lds_tok[a_row];
  const uint32_t a_voff = tok_a * K + a_chk * 16u;
  WeaveLdsU8* const a_wr = lds + V70_BM64_S1_A0 + a_row * 64u
      + ((a_chk ^ ((a_row >> 1u) & 3u)) * 16u);
  const uint32_t sa_voff = lds_tok[tid & (V70_BM64 - 1u)] * num_k128 * 4u;

  const unsigned b_blk = tid >> 5u;
  const unsigned b_in = (tid & 31u) * 32u;
  const uint32_t b_voff = (tile_n * 8u + (b_blk & 7u)) * b_nstride + b_in
      + ((b_blk >> 3u) * nk_bytes);
  WeaveLdsU8* const b_wr = lds + V70_BM64_S1_B0 + b_blk * 1024u + b_in;

  const unsigned r0 = wm * 16u + m_lane;
  WeaveLdsU8* const a_rd = lds + V70_BM64_S1_A0 + r0 * 64u
      + ((k_grp ^ ((m_lane >> 1u) & 3u)) * 16u);
  const WeaveLdsF32* const sa_rd = reinterpret_cast<const WeaveLdsF32*>(
      lds + V70_BM64_S1_SA) + r0;
  WeaveLdsU8* const b_rd = lds + V70_BM64_S1_B0 + (wn * 4u) * 1024u + lane * 16u;

  WeaveAcc4 gate_acc[4], up_acc[4];
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) {
    weave_acc_zero(gate_acc[n]);
    weave_acc_zero(up_acc[n]);
  }

  WeaveI64x2 ra[2], rb0, rb1;
  float rsa;
  auto issue_b = [&](unsigned kk) {
    const uint32_t soff_b = kk * 1024u;
    rb0 = weave_buf_load_i64x2(b_rsrc, b_voff, soff_b);
    rb1 = weave_buf_load_i64x2(b_rsrc, b_voff + 16u, soff_b);
    rsa = weave_buf_load_f32(sa_rsrc, sa_voff, (kk >> 1u) * 4u);
  };
  auto issue_a = [&](unsigned kk, auto slotv) {
    constexpr unsigned slot = decltype(slotv)::value;
    ra[slot] = weave_buf_load_i64x2(a_rsrc, a_voff, kk * 64u);
  };
  auto commit = [&](auto slotv, auto stv) {
    constexpr unsigned slot = decltype(slotv)::value;
    constexpr unsigned st = decltype(stv)::value;
    if (a_owner) weave_lds_store_i64x2(a_wr + st * V70_BM64_S1_ASTR, ra[slot]);
    weave_lds_store_i64x2(b_wr + st * V70_BM64_S1_BSTR, rb0);
    weave_lds_store_i64x2(b_wr + st * V70_BM64_S1_BSTR + 16u, rb1);
    if (tid < V70_BM64)
      reinterpret_cast<WeaveLdsF32*>(lds + V70_BM64_S1_SA + st * (V70_BM64 * 4u))[tid] = rsa;
  };

  issue_b(0u); issue_a(0u, hr1_ic<0>{});
  __builtin_amdgcn_s_waitcnt(0x0000);
  commit(hr1_ic<0>{}, hr1_ic<0>{});
  issue_b(1u); issue_a(1u, hr1_ic<1>{});
  __builtin_amdgcn_s_waitcnt(0x0000);
  commit(hr1_ic<1>{}, hr1_ic<1>{});
  hr1_barrier();

  float sb_g_cur = bs_gate_s[0];
  float sb_u_cur = bs_gate_s[bs_up_stride];
  auto step = [&](unsigned k64, auto curv, auto nxtv) {
    constexpr unsigned cur = decltype(curv)::value;
    constexpr unsigned nxt = decltype(nxtv)::value;
    const unsigned kn = (k64 + 1u < num_k64) ? (k64 + 1u) : (num_k64 - 1u);
    const unsigned kl = (k64 + 2u < num_k64) ? (k64 + 2u) : (num_k64 - 1u);
    issue_b(kn);
    issue_a(kl, curv);
    const float sb_g_nxt = bs_gate_s[kn >> 1u];
    const float sb_u_nxt = bs_gate_s[bs_up_stride + (kn >> 1u)];
    WeaveLdsU8* const ap = a_rd + cur * V70_BM64_S1_ASTR;
    const WeaveLdsF32* const sp = sa_rd + cur * V70_BM64;
    WeaveLdsU8* const bp = b_rd + cur * V70_BM64_S1_BSTR;
    v70_bm64_s1_k64(ap, sp, bp, sb_g_cur, sb_u_cur, gate_acc, up_acc);
    hr1_wait_vmcnt1();
    commit(nxtv, nxtv);
    sb_g_cur = sb_g_nxt;
    sb_u_cur = sb_u_nxt;
    hr1_barrier();
  };

  const unsigned n_pairs = num_k64 >> 1;
  #pragma clang loop unroll(disable)
  for (unsigned p = 0u; p < n_pairs; ++p) {
    const unsigned k64 = p * 2u;
    step(k64, hr1_ic<0>{}, hr1_ic<1>{});
    step(k64 + 1u, hr1_ic<1>{}, hr1_ic<0>{});
  }

  WeaveLdsU8* const epi = lds + V70_BM64_S1_EPI;
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) {
    const unsigned r = wm * 16u + m_lane;
    const unsigned col = (wn * 4u + n) * 16u + k_grp * 4u;
    const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
        weave_bk64_silu(gate_acc[n].lo[0]) * up_acc[n].lo[0])) |
        (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
        weave_bk64_silu(gate_acc[n].lo[1]) * up_acc[n].lo[1])) << 16u);
    const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
        weave_bk64_silu(gate_acc[n].hi[0]) * up_acc[n].hi[0])) |
        (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
        weave_bk64_silu(gate_acc[n].hi[1]) * up_acc[n].hi[1])) << 16u);
    const unsigned pair = ((col >> 3u) ^ (r & 7u)) * 16u + (col & 7u) * 2u;
    *reinterpret_cast<WeaveLdsU2*>(epi + r * 256u + pair) = make_uint2(p0, p1);
  }
  hr1_barrier();

  const uint32_t q_limit = __builtin_amdgcn_readfirstlane(route_rows * n_blocks * 128u);
  const uint32_t qs_limit = __builtin_amdgcn_readfirstlane(route_rows * n_blocks * 4u);
  const WeaveI32x4 q_rsrc = weave_make_buf_rsrc(quant_out_base, q_limit);
  const WeaveI32x4 qs_rsrc = weave_make_buf_rsrc(quant_scale_base, qs_limit);
  unsigned q_seed = tid;
  asm volatile("" : "+v"(q_seed));
  const unsigned q_row = q_seed >> 4u;
  const unsigned q_lane = q_seed & 15u;
  #pragma clang loop unroll(disable)
  for (unsigned r0_i = 0u; r0_i < V70_BM64; r0_i += 32u) {
    const unsigned r = r0_i + q_row;
    const unsigned dense_row = row_offset + r;
    const bool valid = (r < V70_BM64) && (r < valid_rows) &&
        (dense_row < route_rows) && (lds_tok[r] < num_tokens);
    uint4 raw = make_uint4(0, 0, 0, 0);
    if (valid)
      raw = *reinterpret_cast<const WeaveLdsU4*>(epi + r * 256u + ((q_lane ^ (r & 7u)) * 16u));
    hr1_quant_bf16x8_store_buf(raw, valid, q_lane, dense_row * n_blocks + tile_n,
                               q_rsrc, qs_rsrc);
  }
  hr1_barrier();
}

__device__ __forceinline__ void v70_s2_8wave_compute_bm64(
    const int32_t* __restrict__ s2_ids_tile, int32_t expert_id,
    const uint8_t* __restrict__ a_base, const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base, const float* __restrict__ b_scale_base,
    const float* __restrict__ weights_tile, unsigned valid_rows, unsigned tile_n,
    unsigned route_rows, unsigned N, unsigned K, unsigned char* arena) {
  (void)arena;
  WeaveLdsU8* const lds = reinterpret_cast<WeaveLdsU8*>(
      reinterpret_cast<uintptr_t>(HR1_LDS_ARENA));
  const unsigned tid = threadIdx.x;
  const unsigned wave = tid >> 6u;
  const unsigned wm = wave >> 1u;
  const unsigned wn = wave & 1u;
  const unsigned lane = tid & 63u;
  const unsigned m_lane = lane & 15u;
  const unsigned k_grp = lane >> 4u;

  const unsigned num_k128 = __builtin_amdgcn_readfirstlane(K >> 7u);
  const unsigned num_k64 = __builtin_amdgcn_readfirstlane(K >> 6u);
  const unsigned n_blocks = __builtin_amdgcn_readfirstlane(N >> 7u);
  const unsigned exp_u = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
  const uint32_t nk_bytes = __builtin_amdgcn_readfirstlane(N * K);
  const uint8_t* b_exp = b_base + static_cast<uint64_t>(exp_u) * nk_bytes;
  const float* bs_ptr = b_scale_base +
      (static_cast<uint64_t>(exp_u) * n_blocks + tile_n) * num_k128;
  WeaveConstF32As4* bs_s = weave_make_as4_f32(bs_ptr);

  const uint32_t a_limit = __builtin_amdgcn_readfirstlane(route_rows * K);
  const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(route_rows * num_k128 * 4u);
  const WeaveI32x4 a_rsrc = weave_make_buf_rsrc(a_base, a_limit);
  const WeaveI32x4 sa_rsrc = weave_make_buf_rsrc(a_scale_base, sa_limit);
  const WeaveI32x4 b_rsrc = weave_make_buf_rsrc(b_exp, nk_bytes);
  const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);

  WeaveLdsU32* const lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + V70_BM64_S2_TOK);
  if (tid < V70_BM64)
    lds_tok[tid] = (tid < valid_rows)
        ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu)
        : route_rows;
  hr1_barrier();

  const unsigned a_row = (tid >> 2u) & (V70_BM64 - 1u);
  const bool a_owner = tid < (V70_BM64 * 4u);
  const unsigned a_chk = tid & 3u;
  const uint32_t tok_a = lds_tok[a_row];
  const uint32_t a_voff = (tok_a < route_rows) ? (tok_a * K + a_chk * 16u) : a_limit;
  WeaveLdsU8* const a_wr = lds + V70_BM64_S2_A0 + a_row * 64u
      + ((a_chk ^ ((a_row >> 1u) & 3u)) * 16u);
  const uint32_t sa_voff = (tid < V70_BM64)
      ? ((lds_tok[tid] < route_rows) ? (lds_tok[tid] * num_k128 * 4u) : sa_limit)
      : sa_limit;

  const unsigned b_blk = tid >> 6u;
  const unsigned b_in = (tid & 63u) * 16u;
  const uint32_t b_voff = (tile_n * 8u + b_blk) * b_nstride + b_in;
  WeaveLdsU8* const b_wr = lds + V70_BM64_S2_B0 + b_blk * 1024u + b_in;

  const unsigned r0 = wm * 16u + m_lane;
  WeaveLdsU8* const a_rd = lds + V70_BM64_S2_A0 + r0 * 64u
      + ((k_grp ^ ((m_lane >> 1u) & 3u)) * 16u);
  const WeaveLdsF32* const sa_rd = reinterpret_cast<const WeaveLdsF32*>(
      lds + V70_BM64_S2_SA) + r0;
  WeaveLdsU8* const b_rd = lds + V70_BM64_S2_B0 + (wn * 4u) * 1024u + lane * 16u;

  WeaveAcc4 acc[4];
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) weave_acc_zero(acc[n]);

  WeaveI64x2 ra[2], rb[2];
  float rsa[2];
  auto issue = [&](unsigned kk, auto slotv) {
    constexpr unsigned slot = decltype(slotv)::value;
    ra[slot] = weave_buf_load_i64x2(a_rsrc, a_voff, kk * 64u);
    rb[slot] = weave_buf_load_i64x2(b_rsrc, b_voff, kk * 1024u);
    rsa[slot] = weave_buf_load_f32(sa_rsrc, sa_voff, (kk >> 1u) * 4u);
  };
  auto commit = [&](auto slotv, auto stv) {
    constexpr unsigned slot = decltype(slotv)::value;
    constexpr unsigned st = decltype(stv)::value;
    if (a_owner) weave_lds_store_i64x2(a_wr + st * V70_BM64_S2_ASTR, ra[slot]);
    weave_lds_store_i64x2(b_wr + st * V70_BM64_S2_BSTR, rb[slot]);
    if (tid < V70_BM64)
      reinterpret_cast<WeaveLdsF32*>(lds + V70_BM64_S2_SA + st * (V70_BM64 * 4u))[tid] = rsa[slot];
  };

  issue(0u, hr1_ic<0>{}); issue(1u, hr1_ic<1>{});
  __builtin_amdgcn_s_waitcnt(0x0000);
  commit(hr1_ic<0>{}, hr1_ic<0>{});
  commit(hr1_ic<1>{}, hr1_ic<1>{});
  hr1_barrier();

  float sb_cur = bs_s[0];
  auto step = [&](unsigned k64, auto curv, auto nxtv) {
    constexpr unsigned cur = decltype(curv)::value;
    constexpr unsigned nxt = decltype(nxtv)::value;
    const unsigned kl = (k64 + 2u < num_k64) ? (k64 + 2u) : (num_k64 - 1u);
    issue(kl, curv);
    const unsigned kn = (k64 + 1u < num_k64) ? (k64 + 1u) : (num_k64 - 1u);
    const float sb_nxt = bs_s[kn >> 1u];
    WeaveLdsU8* const ap = a_rd + cur * V70_BM64_S2_ASTR;
    const WeaveLdsF32* const sp = sa_rd + cur * V70_BM64;
    WeaveLdsU8* const bp = b_rd + cur * V70_BM64_S2_BSTR;
    v70_bm64_s2_k64(ap, sp, bp, sb_cur, acc);
    hr1_wait_vmcnt3();
    commit(nxtv, nxtv);
    sb_cur = sb_nxt;
    hr1_barrier();
  };
  const unsigned n_pairs = num_k64 >> 1;
  #pragma clang loop unroll(disable)
  for (unsigned p = 0u; p < n_pairs; ++p) {
    const unsigned k64 = p * 2u;
    step(k64, hr1_ic<0>{}, hr1_ic<1>{});
    step(k64 + 1u, hr1_ic<1>{}, hr1_ic<0>{});
  }

  WeaveLdsU8* const epi = lds + V70_BM64_S2_EPI;
  const unsigned r = wm * 16u + m_lane;
  const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
  #pragma unroll
  for (unsigned n = 0u; n < 4u; ++n) {
    const unsigned col = (wn * 4u + n) * 16u + k_grp * 4u;
    const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[n].lo[0] * weight)) |
        (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[n].lo[1] * weight)) << 16u);
    const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[n].hi[0] * weight)) |
        (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[n].hi[1] * weight)) << 16u);
    const unsigned pair = ((col >> 3u) ^ (r & 7u)) * 16u + (col & 7u) * 2u;
    *reinterpret_cast<WeaveLdsU2*>(epi + r * 256u + pair) = make_uint2(p0, p1);
  }
  hr1_barrier();
}

__device__ __forceinline__ void v70_s2_8wave_store_bm64(
    uint16_t* __restrict__ c_out_base, unsigned row_offset, unsigned valid_rows,
    unsigned tile_n, unsigned route_rows, unsigned N,
    const unsigned* row_dir = nullptr, const uint64_t* peer_tab = nullptr,
    const uint64_t* node_tab = nullptr, unsigned node_row_base = 0u,
    unsigned my_node = 0u) {
  WeaveLdsU8* const lds = reinterpret_cast<WeaveLdsU8*>(
      reinterpret_cast<uintptr_t>(HR1_LDS_ARENA));
  const unsigned tid = threadIdx.x;
  const unsigned c_row = tid >> 4u;
  const unsigned c_lane = tid & 15u;
  const WeaveLdsU32* const lds_tok = reinterpret_cast<const WeaveLdsU32*>(
      lds + V70_BM64_S2_TOK);
  const WeaveLdsU8* const epi = lds + V70_BM64_S2_EPI;
  #pragma clang loop unroll(disable)
  for (unsigned r0_i = 0u; r0_i < V70_BM64; r0_i += 32u) {
    const unsigned r = r0_i + c_row;
    const uint32_t tok_m = lds_tok[r];
    if (r < V70_BM64 && r < valid_rows && tok_m < route_rows &&
        (row_offset + r) < route_rows) {
      const uint4 u = *reinterpret_cast<const WeaveLdsU4*>(
          epi + r * 256u + ((c_lane ^ (r & 7u)) * 16u));
      uint16_t* direct_p = weave_direct_row_dst(row_dir, peer_tab, node_tab,
                                                node_row_base, my_node, tok_m, N);
      if (direct_p != nullptr) {
        direct_p += static_cast<uint64_t>(tile_n) * 128u + c_lane * 8u;
        weave_gstore_u4_wt(direct_p, u);
      } else {
        weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N +
                        tile_n * 128u + c_lane * 8u, u);
      }
    }
  }
  hr1_barrier();
}
