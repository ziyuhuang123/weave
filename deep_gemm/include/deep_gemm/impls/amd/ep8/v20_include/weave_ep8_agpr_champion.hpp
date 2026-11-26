#pragma once

// ============================================================================
// Weave EP8 1x4 Cooperative-LDS + Full-Prefetch + Native MFMA Champion Engine
// Target: AMD Instinct MI308X (gfx942, CDNA3, 80 CUs, Strict 1 CTA/CU = 80x512)
//
// Architectural pillars:
//   1. Zero-Duplicate-Traffic `1x4` Wave Decomposition across ALL BM tiers
//      (`BM in {16, 32, 64, 128}`, `1.0x` B weight HBM loads, `1.0x` A loads).
//   2. Single-Pass `BM=128 x BN=128` Stage-1 & Stage-2 in `<= 17.5 KB` LDS
//      (`BK=64` ping-pong, `< 32 KB` per 4-wave half-CTA) -> ALL 8 waves/CU
//      execute native `BM=128` tiles with ZERO `split_128` chopping to `BM=64`,
//      halving expert weight HBM traffic at `S >= 1024`!
//   3. Full 512-cycle `B_next + A_next` HBM3 software prefetch BEFORE MFMAs
//      in Stage-1 `BM=64` & `BM=128` and Stage-2 `BM=64` & `BM=128` (eliminating
//      `base_v20`'s unprefetched Stage-1 weight stalls).
//   4. `BN=256` Split-B Stage-2 (`weave_ep8_s2_agpr_bn256` for `BM in {16,32,64}`)
//      computing `(tile_n, tile_n + 1)` in ONE K-loop (`0.5x` A loads & barriers),
//      plus 3-stage (`K=384`) deep-pipelined `BN=64` Stage-2 for `S=16`.
//   5. Unified 4-SGPR `srsrc` (`llvm.amdgcn.raw.buffer.load.v4i32`), hardware
//      `src2 = 0` MFMA zero-init, and unclumped `WeaveAgprAcc4` (`2x v_pk_fma_f32`).
// ============================================================================

#ifndef WEAVE_MFMA_VGPR_DEST
#define WEAVE_MFMA_VGPR_DEST 1
#endif
#include <deep_gemm/impls/amd/include/weave_agpr_gemm_gfx942.hpp>

static __device__ __forceinline__ void weave_ep8_lds_st_i64x2(WeaveLdsU8* p, WeaveAgprI64x2 v)
{
    *reinterpret_cast<WeaveAgprI64x2 __attribute__((address_space(3)))*>(p) = v;
}

static __device__ __forceinline__ WeaveAgprI64x2 weave_ep8_lds_ld_i64x2(const WeaveLdsU8* p)
{
    return *reinterpret_cast<const WeaveAgprI64x2 __attribute__((address_space(3)))*>(p);
}

// Component-wise SReg_32 buffer descriptor: zero loop-carried SReg_128 (0 size=16 frame slots in LLVM 22!)
// AND 100% hardware raw.buffer.load OOB clamping (ZERO basic-block branches or vmcnt(0) stalls!)
struct WeaveEp8ABuf {
    int32_t w0;
    int32_t w1;
    int32_t w2;
};

using WeaveEp8BBuf = WeaveEp8ABuf;

static __device__ __forceinline__ WeaveEp8ABuf weave_ep8_make_abuf(const void* ptr, uint32_t lim)
{
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    WeaveEp8ABuf r;
    r.w0 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(static_cast<uint32_t>(addr)));
    r.w1 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(static_cast<uint32_t>(addr >> 32u)));
    r.w2 = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(lim));
    return r;
}

static __device__ __forceinline__ WeaveEp8BBuf weave_ep8_make_bbuf(const void* ptr, uint32_t lim)
{
    return weave_ep8_make_abuf(ptr, lim);
}

static __device__ __forceinline__ WeaveAgprI64x2 weave_ep8_ld_i64x2(
    WeaveEp8ABuf r, uint32_t voff, uint32_t soff)
{
    const WeaveAgprI32x4 d = {r.w0, r.w1, r.w2, 0x00020000};
    const WeaveAgprI32x4 v = weave_agpr_raw_buf_load_v4i32(
        d, static_cast<int32_t>(voff), static_cast<int32_t>(soff), 0);
    return __builtin_bit_cast(WeaveAgprI64x2, v);
}

static __device__ __forceinline__ float weave_ep8_ld_f32(
    WeaveEp8ABuf r, uint32_t voff, uint32_t soff)
{
    const WeaveAgprI32x4 d = {r.w0, r.w1, r.w2, 0x00020000};
    return weave_agpr_raw_buf_load_f32(
        d, static_cast<int32_t>(voff), static_cast<int32_t>(soff), 0);
}

static __device__ __forceinline__ void weave_ep8_quant_bf16x8_store(
    uint4 raw,
    bool valid,
    unsigned q_lane,
    uint64_t dense_group,
    uint8_t* __restrict__ quant_out_base,
    float* __restrict__ quant_scale_base)
{
    const float v0 = __uint_as_float(raw.x << 16u);
    const float v1 = __uint_as_float(raw.x & 0xffff0000u);
    const float v2 = __uint_as_float(raw.y << 16u);
    const float v3 = __uint_as_float(raw.y & 0xffff0000u);
    const float v4 = __uint_as_float(raw.z << 16u);
    const float v5 = __uint_as_float(raw.z & 0xffff0000u);
    const float v6 = __uint_as_float(raw.w << 16u);
    const float v7 = __uint_as_float(raw.w & 0xffff0000u);

    float abs_max = 1.0e-10f;
    if (valid) {
        abs_max = fmaxf(abs_max, fmaxf(fabsf(v0), fabsf(v1)));
        abs_max = fmaxf(abs_max, fmaxf(fabsf(v2), fabsf(v3)));
        abs_max = fmaxf(abs_max, fmaxf(fabsf(v4), fabsf(v5)));
        abs_max = fmaxf(abs_max, fmaxf(fabsf(v6), fabsf(v7)));
    }
    #pragma unroll
    for (unsigned delta = 1u; delta < 16u; delta <<= 1u) {
        abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffffffffffffull, abs_max, static_cast<int>(delta), 16));
    }
    const float fp8_max = 240.0f;
    const float scale = abs_max * (1.0f / 240.0f);
    if (valid && q_lane == 0u) weave_as1_arr_mut(quant_scale_base)[dense_group] = scale;
    if (valid) {
        const float inv_s = 1.0f / scale;
        const uint32_t p0 = __builtin_amdgcn_cvt_pk_fp8_f32(
            __builtin_amdgcn_fmed3f(v0 * inv_s, fp8_max, -fp8_max),
            __builtin_amdgcn_fmed3f(v1 * inv_s, fp8_max, -fp8_max), 0, 0);
        const uint32_t p1 = __builtin_amdgcn_cvt_pk_fp8_f32(
            __builtin_amdgcn_fmed3f(v2 * inv_s, fp8_max, -fp8_max),
            __builtin_amdgcn_fmed3f(v3 * inv_s, fp8_max, -fp8_max), 0, 0);
        const uint32_t p2 = __builtin_amdgcn_cvt_pk_fp8_f32(
            __builtin_amdgcn_fmed3f(v4 * inv_s, fp8_max, -fp8_max),
            __builtin_amdgcn_fmed3f(v5 * inv_s, fp8_max, -fp8_max), 0, 0);
        const uint32_t p3 = __builtin_amdgcn_cvt_pk_fp8_f32(
            __builtin_amdgcn_fmed3f(v6 * inv_s, fp8_max, -fp8_max),
            __builtin_amdgcn_fmed3f(v7 * inv_s, fp8_max, -fp8_max), 0, 0);
        weave_gstore_u2(quant_out_base + dense_group * 128ull + q_lane * 8u,
                        make_uint2(p0 | (p1 << 16u), p2 | (p3 << 16u)));
    }
}

// ============================================================================
// 1. STAGE 1 SMALL (BM=16, 32): 2-Stage (K=256, 4 k64 Halves) Deep Pipeline
// ============================================================================
template <unsigned kMXDL>
static __device__ __forceinline__ void weave_ep8_s1_small_half_mfma(
    const WeaveAgprI64x2 (&av)[kMXDL],
    const WeaveAgprI64x2 (&bg)[2],
    const WeaveAgprI64x2 (&bu)[2],
    const float (&sa_m)[kMXDL],
    float sb_g,
    float sb_u,
    WeaveAgprAcc4 (&gate_acc)[kMXDL][2],
    WeaveAgprAcc4 (&up_acc)[kMXDL][2])
{
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const float sab_g = sa_m[mx] * sb_g;
        const float sab_u = sa_m[mx] * sb_u;
        WeaveAgprF32x4 tg0, tg1, tu0, tu1;

        weave_mfma_f8_z(tg0, bg[0][0], av[mx][0]);
        weave_mfma_f8_z(tg1, bg[1][0], av[mx][0]);
        weave_mfma_f8_a(tg0, bg[0][1], av[mx][1]);
        weave_mfma_f8_a(tg1, bg[1][1], av[mx][1]);

        weave_mfma_f8_z(tu0, bu[0][0], av[mx][0]);
        weave_mfma_f8_z(tu1, bu[1][0], av[mx][0]);
        weave_agpr_prom_acc4(gate_acc[mx][0], tg0, sab_g);
        weave_mfma_f8_a(tu0, bu[0][1], av[mx][1]);
        weave_mfma_f8_a(tu1, bu[1][1], av[mx][1]);
        weave_agpr_prom_acc4(gate_acc[mx][1], tg1, sab_g);

        weave_agpr_prom_acc4(up_acc[mx][0], tu0, sab_u);
        weave_agpr_prom_acc4(up_acc[mx][1], tu1, sab_u);
    }
}

template <unsigned kMXDL>
static __device__ __forceinline__ void weave_ep8_s1_small_k128_mfma(
    const WeaveAgprI64x2 (&av0)[kMXDL],
    const WeaveAgprI64x2 (&av1)[kMXDL],
    const WeaveAgprI64x2 (&bg0)[2],
    const WeaveAgprI64x2 (&bu0)[2],
    const WeaveAgprI64x2 (&bg1)[2],
    const WeaveAgprI64x2 (&bu1)[2],
    const float (&sa_m)[kMXDL],
    float sb_g,
    float sb_u,
    WeaveAgprAcc4 (&gate_acc)[kMXDL][2],
    WeaveAgprAcc4 (&up_acc)[kMXDL][2])
{
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const float sab_g = sa_m[mx] * sb_g;
        const float sab_u = sa_m[mx] * sb_u;
        WeaveAgprF32x4 tg0, tg1, tu0, tu1;

        weave_mfma_f8_z(tg0, bg0[0][0], av0[mx][0]);
        weave_mfma_f8_z(tg1, bg0[1][0], av0[mx][0]);
        weave_mfma_f8_a(tg0, bg0[0][1], av0[mx][1]);
        weave_mfma_f8_a(tg1, bg0[1][1], av0[mx][1]);
        weave_mfma_f8_a(tg0, bg1[0][0], av1[mx][0]);
        weave_mfma_f8_a(tg1, bg1[1][0], av1[mx][0]);
        weave_mfma_f8_a(tg0, bg1[0][1], av1[mx][1]);
        weave_mfma_f8_a(tg1, bg1[1][1], av1[mx][1]);

        weave_mfma_f8_z(tu0, bu0[0][0], av0[mx][0]);
        weave_mfma_f8_z(tu1, bu0[1][0], av0[mx][0]);
        weave_mfma_f8_a(tu0, bu0[0][1], av0[mx][1]);
        weave_mfma_f8_a(tu1, bu0[1][1], av0[mx][1]);
        weave_agpr_prom_acc4(gate_acc[mx][0], tg0, sab_g);
        weave_agpr_prom_acc4(gate_acc[mx][1], tg1, sab_g);
        weave_mfma_f8_a(tu0, bu1[0][0], av1[mx][0]);
        weave_mfma_f8_a(tu1, bu1[1][0], av1[mx][0]);
        weave_mfma_f8_a(tu0, bu1[0][1], av1[mx][1]);
        weave_mfma_f8_a(tu1, bu1[1][1], av1[mx][1]);

        weave_agpr_prom_acc4(up_acc[mx][0], tu0, sab_u);
        weave_agpr_prom_acc4(up_acc[mx][1], tu1, sab_u);
    }
}

template <unsigned kBM>
static __device__ __forceinline__ void weave_ep8_s1_small_deep(
    const int32_t* __restrict__ s_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    uint8_t* __restrict__ quant_out_base,
    float* __restrict__ quant_scale_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned num_tokens,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    static_assert(kBM == 16u || kBM == 32u);
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kMXDL = kBM / 16u;
    constexpr unsigned kTokOff = 16384u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows)
            ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu)
            : num_tokens;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = __builtin_amdgcn_readfirstlane(K >> 7u);
    const unsigned num_k64  = __builtin_amdgcn_readfirstlane(K >> 6u);
    const unsigned n_blocks = __builtin_amdgcn_readfirstlane(N >> 7u);
    const unsigned exp_u    = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
    const uint32_t nk_bytes = __builtin_amdgcn_readfirstlane(N * K);
    const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(exp_u) * (static_cast<uint64_t>(nk_bytes) * 2ull);

    const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * num_k128;
    const float* bs_gate_ptr = b_scale_base + static_cast<uint64_t>(exp_u) * expert_bs_stride
        + static_cast<uint64_t>(tile_n) * num_k128;
    const uint32_t bs_up_stride = __builtin_amdgcn_readfirstlane(n_blocks * num_k128);
    WeaveAgprConstAs4* bs_gate_s = weave_agpr_make_as4_f32(bs_gate_ptr);

    const uint32_t a_limit  = __builtin_amdgcn_readfirstlane(num_tokens * K);
    const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(num_tokens * num_k128 * 4u);
    const uint32_t b_limit  = __builtin_amdgcn_readfirstlane(nk_bytes * 2u);
    const WeaveEp8ABuf a_rsrc  = weave_ep8_make_abuf(a_base, a_limit);
    const WeaveEp8ABuf sa_rsrc = weave_ep8_make_abuf(a_scale_base, sa_limit);
    const WeaveEp8BBuf b_rsrc  = weave_ep8_make_bbuf(b_gate_exp, b_limit);

    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);
    const uint32_t b_voff0 = ((tile_n * 8u + wave_idx * 2u) * num_k64 * 64u + lane) * 16u;

    WeaveAgprAcc4 gate_acc[kMXDL][2];
    WeaveAgprAcc4 up_acc[kMXDL][2];
    uint32_t a_voff[kMXDL];
    uint32_t sa_voff[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        weave_agpr_acc_zero(gate_acc[mx][0]); weave_agpr_acc_zero(gate_acc[mx][1]);
        weave_agpr_acc_zero(up_acc[mx][0]);   weave_agpr_acc_zero(up_acc[mx][1]);
        const unsigned r = mx * 16u + m_lane;
        const uint32_t tok = lds_tok[r];
        const bool ok = (r < valid_rows) && (tok < num_tokens);
        a_voff[mx]  = ok ? (tok * K + k_grp * 16u) : a_limit;
        sa_voff[mx] = ok ? (tok * num_k128 * 4u) : sa_limit;
    }

    auto load_s1_half = [&](unsigned k64_idx, WeaveAgprI64x2 (&av)[kMXDL],
                            WeaveAgprI64x2 (&bg)[2], WeaveAgprI64x2 (&bu)[2]) {
        const uint32_t ak = k64_idx * 64u;
        const uint32_t bk = k64_idx * 1024u;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av[mx] = weave_ep8_ld_i64x2(a_rsrc, a_voff[mx], ak);
        }
        bg[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk);
        bg[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk + b_nstride);
        bu[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + bk);
        bu[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + bk + b_nstride);
    };

    auto load_s1_scales = [&](unsigned k128_idx, float (&sa)[kMXDL], float& sbg, float& sbu) {
        const uint32_t sak = k128_idx * 4u;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            sa[mx] = weave_ep8_ld_f32(sa_rsrc, sa_voff[mx], sak);
        }
        sbg = bs_gate_s[k128_idx];
        sbu = bs_gate_s[bs_up_stride + k128_idx];
    };

    if (num_k128 >= 2u) {
        WeaveAgprI64x2 av0_c[kMXDL], av1_c[kMXDL], bg0_c[2], bu0_c[2], bg1_c[2], bu1_c[2];
        WeaveAgprI64x2 av0_n[kMXDL], av1_n[kMXDL], bg0_n[2], bu0_n[2], bg1_n[2], bu1_n[2];
        float sa_c[kMXDL], sa_n[kMXDL], sbg_c, sbu_c, sbg_n, sbu_n;

        load_s1_half(0u, av0_c, bg0_c, bu0_c);
        load_s1_scales(0u, sa_c, sbg_c, sbu_c);
        load_s1_half(1u, av1_c, bg1_c, bu1_c);
        load_s1_half(2u, av0_n, bg0_n, bu0_n);
        load_s1_scales(1u, sa_n, sbg_n, sbu_n);
        load_s1_half(3u, av1_n, bg1_n, bu1_n);

        #pragma clang loop unroll(disable)
        for (unsigned k128 = 0u; k128 + 2u < num_k128; ++k128) {
            const unsigned nn = k128 + 2u;
            weave_ep8_s1_small_k128_mfma<kMXDL>(av0_c, av1_c, bg0_c, bu0_c, bg1_c, bu1_c, sa_c, sbg_c, sbu_c, gate_acc, up_acc);
            load_s1_half(nn * 2u + 0u, av0_c, bg0_c, bu0_c);
            float sa_nn[kMXDL], sbg_nn, sbu_nn;
            load_s1_scales(nn, sa_nn, sbg_nn, sbu_nn);
            load_s1_half(nn * 2u + 1u, av1_c, bg1_c, bu1_c);

            #pragma unroll
            for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                const WeaveAgprI64x2 t0 = av0_c[mx]; av0_c[mx] = av0_n[mx]; av0_n[mx] = t0;
                const WeaveAgprI64x2 t1 = av1_c[mx]; av1_c[mx] = av1_n[mx]; av1_n[mx] = t1;
                sa_c[mx] = sa_n[mx]; sa_n[mx] = sa_nn[mx];
            }
            #pragma unroll
            for (unsigned nx = 0u; nx < 2u; ++nx) {
                const WeaveAgprI64x2 g0 = bg0_c[nx]; bg0_c[nx] = bg0_n[nx]; bg0_n[nx] = g0;
                const WeaveAgprI64x2 u0 = bu0_c[nx]; bu0_c[nx] = bu0_n[nx]; bu0_n[nx] = u0;
                const WeaveAgprI64x2 g1 = bg1_c[nx]; bg1_c[nx] = bg1_n[nx]; bg1_n[nx] = g1;
                const WeaveAgprI64x2 u1 = bu1_c[nx]; bu1_c[nx] = bu1_n[nx]; bu1_n[nx] = u1;
            }
            sbg_c = sbg_n; sbu_c = sbu_n;
            sbg_n = sbg_nn; sbu_n = sbu_nn;
        }
        weave_ep8_s1_small_k128_mfma<kMXDL>(av0_c, av1_c, bg0_c, bu0_c, bg1_c, bu1_c, sa_c, sbg_c, sbu_c, gate_acc, up_acc);
        weave_ep8_s1_small_k128_mfma<kMXDL>(av0_n, av1_n, bg0_n, bu0_n, bg1_n, bu1_n, sa_n, sbg_n, sbu_n, gate_acc, up_acc);
    } else {
        WeaveAgprI64x2 av0[kMXDL], av1[kMXDL], bg0[2], bu0[2], bg1[2], bu1[2];
        float sa[kMXDL], sbg, sbu;
        load_s1_half(0u, av0, bg0, bu0);
        load_s1_scales(0u, sa, sbg, sbu);
        load_s1_half(1u, av1, bg1, bu1);
        weave_ep8_s1_small_k128_mfma<kMXDL>(av0, av1, bg0, bu0, bg1, bu1, sa, sbg, sbu, gate_acc, up_acc);
    }

    WeaveLdsU16* lds_q = reinterpret_cast<WeaveLdsU16*>(lds);
    const unsigned q_row = tid >> 4u;
    const unsigned q_lane = tid & 15u;
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + m_lane;
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx].lo[0]) * up_acc[mx][nx].lo[0])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx].lo[1]) * up_acc[mx][nx].lo[1])) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx].hi[0]) * up_acc[mx][nx].hi[0])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx].hi[1]) * up_acc[mx][nx].hi[1])) << 16u);
            *reinterpret_cast<WeaveLdsU2*>(lds_q + r * 128u + col) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + q_row;
        const unsigned dense_row = row_offset + r;
        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (valid) raw = *reinterpret_cast<const WeaveLdsU4*>(lds_q + r * 128u + q_lane * 8u);
        weave_ep8_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

// ============================================================================
// 2. STAGE 1 BM=64 (1x4 Cooperative A-LDS + Full 512-Cycle B_next & A_next Prefetch)
//    LDS = 17,664 B (< 32 KB); 1.0x A load, 1.0x B load, 0 unprefetched stalls!
// ============================================================================
static __device__ __forceinline__ void weave_ep8_s1_1x4_bm64(
    const int32_t* __restrict__ s_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    uint8_t* __restrict__ quant_out_base,
    float* __restrict__ quant_scale_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned num_tokens,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kStageBytes = 8192u; // 64 * 128
    constexpr unsigned kSaBaseOff  = 16384u;
    constexpr unsigned kTokOff     = 17408u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < 64u) {
        lds_tok[tid] = (tid < valid_rows)
            ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu)
            : num_tokens;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = __builtin_amdgcn_readfirstlane(K >> 7u);
    const unsigned num_k64  = __builtin_amdgcn_readfirstlane(K >> 6u);
    const unsigned n_blocks = __builtin_amdgcn_readfirstlane(N >> 7u);
    const unsigned exp_u    = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
    const uint32_t nk_bytes = __builtin_amdgcn_readfirstlane(N * K);
    const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(exp_u) * (static_cast<uint64_t>(nk_bytes) * 2ull);

    const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * num_k128;
    const float* bs_gate_ptr = b_scale_base + static_cast<uint64_t>(exp_u) * expert_bs_stride
        + static_cast<uint64_t>(tile_n) * num_k128;
    const uint32_t bs_up_stride = __builtin_amdgcn_readfirstlane(n_blocks * num_k128);
    WeaveAgprConstAs4* bs_gate_s = weave_agpr_make_as4_f32(bs_gate_ptr);

    const uint32_t a_limit  = __builtin_amdgcn_readfirstlane(num_tokens * K);
    const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(num_tokens * num_k128 * 4u);
    const uint32_t b_limit  = __builtin_amdgcn_readfirstlane(nk_bytes * 2u);
    const WeaveEp8ABuf a_rsrc  = weave_ep8_make_abuf(a_base, a_limit);
    const WeaveEp8ABuf sa_rsrc = weave_ep8_make_abuf(a_scale_base, sa_limit);
    const WeaveEp8BBuf b_rsrc  = weave_ep8_make_bbuf(b_gate_exp, b_limit);

    const unsigned a_r32 = tid >> 3u;
    const unsigned a_c16 = tid & 7u;
    uint32_t a_voff[2];
    #pragma unroll
    for (unsigned ap = 0u; ap < 2u; ++ap) {
        const unsigned r = ap * 32u + a_r32;
        const uint32_t tok = lds_tok[r];
        a_voff[ap] = ((r < valid_rows) && (tok < num_tokens)) ? (tok * K + a_c16 * 16u) : a_limit;
    }
    const bool sa_active = (tid < 64u);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : num_tokens;
    const uint32_t sa_voff = (sa_active && (tid < valid_rows) && (tok_sa < num_tokens))
        ? (tok_sa * num_k128 * 4u) : sa_limit;

    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);
    const uint32_t b_voff0 = ((tile_n * 8u + wave_idx * 2u) * num_k64 * 64u + lane) * 16u;

    WeaveAgprAcc4 gate_acc[4][2];
    WeaveAgprAcc4 up_acc[4][2];
    #pragma unroll
    for (unsigned mx = 0u; mx < 4u; ++mx) {
        weave_agpr_acc_zero(gate_acc[mx][0]); weave_agpr_acc_zero(gate_acc[mx][1]);
        weave_agpr_acc_zero(up_acc[mx][0]);   weave_agpr_acc_zero(up_acc[mx][1]);
    }

    // Prologue: kb = 0
    {
        const WeaveAgprI64x2 a0 = weave_ep8_ld_i64x2(a_rsrc, a_voff[0], 0u);
        const WeaveAgprI64x2 a1 = weave_ep8_ld_i64x2(a_rsrc, a_voff[1], 0u);
        const float sa0 = weave_ep8_ld_f32(sa_rsrc, sa_voff, 0u);
        const unsigned r0 = a_r32, r1 = 32u + a_r32;
        weave_ep8_lds_st_i64x2(lds + r0 * 128u + (a_c16 ^ (r0 & 7u)) * 16u, a0);
        weave_ep8_lds_st_i64x2(lds + r1 * 128u + (a_c16 ^ (r1 & 7u)) * 16u, a1);
        if (sa_active) reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff)[tid] = sa0;
    }
    WeaveAgprI64x2 bg0[2], bu0[2], bg1[2], bu1[2];
    bg0[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, 0u);
    bg0[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, b_nstride);
    bu0[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes);
    bu0[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + b_nstride);
    bg1[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, 1024u);
    bg1[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, 1024u + b_nstride);
    bu1[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + 1024u);
    bu1[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + 1024u + b_nstride);
    float sb_g_cur = bs_gate_s[0];
    float sb_u_cur = bs_gate_s[bs_up_stride];
    weave_4wave_lds_barrier_65440(arena);

    #pragma clang loop unroll(disable)
    for (unsigned kb = 0u; kb + 1u < num_k128; ++kb) {
        const unsigned nxt_kb = kb + 1u;
        const uint32_t ak_off = nxt_kb * 128u;
        const uint32_t bk_off = nxt_kb * 2048u;
        const WeaveLdsU8* lds_a_cur = lds + (kb & 1u) * kStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + (kb & 1u) * 512u);
        WeaveLdsU8* lds_a_nxt = lds + (nxt_kb & 1u) * kStageBytes;

        // Prefetch BOTH A_next AND ALL of B_next BEFORE the 32 MFMAs (512 full cycles of HBM3 flight!)
        const WeaveAgprI64x2 a_n0 = weave_ep8_ld_i64x2(a_rsrc, a_voff[0], ak_off);
        const WeaveAgprI64x2 a_n1 = weave_ep8_ld_i64x2(a_rsrc, a_voff[1], ak_off);
        const float sa_nxt = weave_ep8_ld_f32(sa_rsrc, sa_voff, nxt_kb * 4u);
        const WeaveAgprI64x2 bg0_n0 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk_off);
        const WeaveAgprI64x2 bg0_n1 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk_off + b_nstride);
        const WeaveAgprI64x2 bu0_n0 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + bk_off);
        const WeaveAgprI64x2 bu0_n1 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + bk_off + b_nstride);
        const WeaveAgprI64x2 bg1_n0 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk_off + 1024u);
        const WeaveAgprI64x2 bg1_n1 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk_off + 1024u + b_nstride);
        const WeaveAgprI64x2 bu1_n0 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + bk_off + 1024u);
        const WeaveAgprI64x2 bu1_n1 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + bk_off + 1024u + b_nstride);
        const float sb_g_nxt = bs_gate_s[nxt_kb];
        const float sb_u_nxt = bs_gate_s[bs_up_stride + nxt_kb];

        #pragma unroll
        for (unsigned mx = 0u; mx < 4u; ++mx) {
            const unsigned rm = mx * 16u + m_lane;
            const WeaveAgprI64x2 av0 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
            const WeaveAgprI64x2 av1 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
            const float sa_val = lds_sa_cur[rm];
            const float sab_g = sa_val * sb_g_cur;
            const float sab_u = sa_val * sb_u_cur;

            WeaveAgprF32x4 tg0, tg1, tu0, tu1;
            weave_mfma_f8_z(tg0, bg0[0][0], av0[0]);
            weave_mfma_f8_z(tg1, bg0[1][0], av0[0]);
            weave_mfma_f8_a(tg0, bg0[0][1], av0[1]);
            weave_mfma_f8_a(tg1, bg0[1][1], av0[1]);
            weave_mfma_f8_a(tg0, bg1[0][0], av1[0]);
            weave_mfma_f8_a(tg1, bg1[1][0], av1[0]);
            weave_mfma_f8_a(tg0, bg1[0][1], av1[1]);
            weave_mfma_f8_a(tg1, bg1[1][1], av1[1]);

            weave_mfma_f8_z(tu0, bu0[0][0], av0[0]);
            weave_mfma_f8_z(tu1, bu0[1][0], av0[0]);
            weave_mfma_f8_a(tu0, bu0[0][1], av0[1]);
            weave_mfma_f8_a(tu1, bu0[1][1], av0[1]);
            weave_agpr_prom_acc4(gate_acc[mx][0], tg0, sab_g);
            weave_agpr_prom_acc4(gate_acc[mx][1], tg1, sab_g);
            weave_mfma_f8_a(tu0, bu1[0][0], av1[0]);
            weave_mfma_f8_a(tu1, bu1[1][0], av1[0]);
            weave_mfma_f8_a(tu0, bu1[0][1], av1[1]);
            weave_mfma_f8_a(tu1, bu1[1][1], av1[1]);

            weave_agpr_prom_acc4(up_acc[mx][0], tu0, sab_u);
            weave_agpr_prom_acc4(up_acc[mx][1], tu1, sab_u);
        }

        bg0[0] = bg0_n0; bg0[1] = bg0_n1;
        bu0[0] = bu0_n0; bu0[1] = bu0_n1;
        bg1[0] = bg1_n0; bg1[1] = bg1_n1;
        bu1[0] = bu1_n0; bu1[1] = bu1_n1;
        const unsigned r0 = a_r32, r1 = 32u + a_r32;
        weave_ep8_lds_st_i64x2(lds_a_nxt + r0 * 128u + (a_c16 ^ (r0 & 7u)) * 16u, a_n0);
        weave_ep8_lds_st_i64x2(lds_a_nxt + r1 * 128u + (a_c16 ^ (r1 & 7u)) * 16u, a_n1);
        if (sa_active) reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + (nxt_kb & 1u) * 512u)[tid] = sa_nxt;
        sb_g_cur = sb_g_nxt;
        sb_u_cur = sb_u_nxt;
        weave_4wave_lds_barrier_65440(arena);
    }

    // Epilogue K-step: kb = num_k128 - 1
    {
        const unsigned cur_st = (num_k128 - 1u) & 1u;
        const WeaveLdsU8* lds_a_cur = lds + cur_st * kStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u);
        #pragma unroll
        for (unsigned mx = 0u; mx < 4u; ++mx) {
            const unsigned rm = mx * 16u + m_lane;
            const WeaveAgprI64x2 av0 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
            const WeaveAgprI64x2 av1 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
            const float sa_val = lds_sa_cur[rm];
            const float sab_g = sa_val * sb_g_cur;
            const float sab_u = sa_val * sb_u_cur;

            WeaveAgprF32x4 tg0, tg1, tu0, tu1;
            weave_mfma_f8_z(tg0, bg0[0][0], av0[0]);
            weave_mfma_f8_z(tg1, bg0[1][0], av0[0]);
            weave_mfma_f8_a(tg0, bg0[0][1], av0[1]);
            weave_mfma_f8_a(tg1, bg0[1][1], av0[1]);
            weave_mfma_f8_a(tg0, bg1[0][0], av1[0]);
            weave_mfma_f8_a(tg1, bg1[1][0], av1[0]);
            weave_mfma_f8_a(tg0, bg1[0][1], av1[1]);
            weave_mfma_f8_a(tg1, bg1[1][1], av1[1]);

            weave_mfma_f8_z(tu0, bu0[0][0], av0[0]);
            weave_mfma_f8_z(tu1, bu0[1][0], av0[0]);
            weave_mfma_f8_a(tu0, bu0[0][1], av0[1]);
            weave_mfma_f8_a(tu1, bu0[1][1], av0[1]);
            weave_agpr_prom_acc4(gate_acc[mx][0], tg0, sab_g);
            weave_agpr_prom_acc4(gate_acc[mx][1], tg1, sab_g);
            weave_mfma_f8_a(tu0, bu1[0][0], av1[0]);
            weave_mfma_f8_a(tu1, bu1[1][0], av1[0]);
            weave_mfma_f8_a(tu0, bu1[0][1], av1[1]);
            weave_mfma_f8_a(tu1, bu1[1][1], av1[1]);

            weave_agpr_prom_acc4(up_acc[mx][0], tu0, sab_u);
            weave_agpr_prom_acc4(up_acc[mx][1], tu1, sab_u);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    WeaveLdsU16* lds_q = reinterpret_cast<WeaveLdsU16*>(lds);
    #pragma unroll
    for (unsigned mx = 0u; mx < 4u; ++mx) {
        const unsigned r = mx * 16u + m_lane;
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx].lo[0]) * up_acc[mx][nx].lo[0])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx].lo[1]) * up_acc[mx][nx].lo[1])) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx].hi[0]) * up_acc[mx][nx].hi[0])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx].hi[1]) * up_acc[mx][nx].hi[1])) << 16u);
            *reinterpret_cast<WeaveLdsU2*>(lds_q + r * 128u + col) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned q_row = tid >> 4u;
    const unsigned q_lane = tid & 15u;
    #pragma clang loop unroll(disable)
    for (unsigned r0 = 0u; r0 < 64u; r0 += 16u) {
        const unsigned r = r0 + q_row;
        const unsigned dense_row = row_offset + r;
        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (valid) raw = *reinterpret_cast<const WeaveLdsU4*>(lds_q + r * 128u + q_lane * 8u);
        weave_ep8_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

// ============================================================================
// 3. STAGE 1 BM=128: SINGLE-PASS 1x4 Cooperative A-LDS (BK=64 Ping-Pong)
//    LDS = 17,920 B (< 32 KB -> ALL 8 waves/CU run native BM=128, 0 split_128!),
//    1.0x A load, 1.0x B load, 512-cycle full B_next + A_next HBM3 prefetch!
// ============================================================================
static __device__ __forceinline__ void weave_ep8_s1_1x4_bm128(
    const int32_t* __restrict__ s_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    uint8_t* __restrict__ quant_out_base,
    float* __restrict__ quant_scale_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned num_tokens,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kStageBytes = 8192u;  // 128 * 64 B
    constexpr unsigned kSaBaseOff  = 16384u; // 2 * 512 B = 1024 B
    constexpr unsigned kTokOff     = 17408u; // 128 * 4 B = 512 B (total 17,920 B!)

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < 128u) {
        lds_tok[tid] = (tid < valid_rows)
            ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu)
            : num_tokens;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = __builtin_amdgcn_readfirstlane(K >> 7u);
    const unsigned num_k64  = __builtin_amdgcn_readfirstlane(K >> 6u);
    const unsigned n_blocks = __builtin_amdgcn_readfirstlane(N >> 7u);
    const unsigned exp_u    = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
    const uint32_t nk_bytes = __builtin_amdgcn_readfirstlane(N * K);
    const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(exp_u) * (static_cast<uint64_t>(nk_bytes) * 2ull);

    const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * num_k128;
    const float* bs_gate_ptr = b_scale_base + static_cast<uint64_t>(exp_u) * expert_bs_stride
        + static_cast<uint64_t>(tile_n) * num_k128;
    const uint32_t bs_up_stride = __builtin_amdgcn_readfirstlane(n_blocks * num_k128);
    WeaveAgprConstAs4* bs_gate_s = weave_agpr_make_as4_f32(bs_gate_ptr);

    const uint32_t a_limit  = __builtin_amdgcn_readfirstlane(num_tokens * K);
    const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(num_tokens * num_k128 * 4u);
    const uint32_t b_limit  = __builtin_amdgcn_readfirstlane(nk_bytes * 2u);
    const WeaveEp8ABuf a_rsrc  = weave_ep8_make_abuf(a_base, a_limit);
    const WeaveEp8ABuf sa_rsrc = weave_ep8_make_abuf(a_scale_base, sa_limit);
    const WeaveEp8BBuf b_rsrc  = weave_ep8_make_bbuf(b_gate_exp, b_limit);

    const unsigned a_r64 = tid >> 2u;
    const unsigned a_c16 = tid & 3u;
    uint32_t a_voff[2];
    #pragma unroll
    for (unsigned ap = 0u; ap < 2u; ++ap) {
        const unsigned r = ap * 64u + a_r64;
        const uint32_t tok = lds_tok[r];
        a_voff[ap] = ((r < valid_rows) && (tok < num_tokens)) ? (tok * K + a_c16 * 16u) : a_limit;
    }
    const bool sa_active = (tid < 128u);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : num_tokens;
    const uint32_t sa_voff = (sa_active && (tid < valid_rows) && (tok_sa < num_tokens))
        ? (tok_sa * num_k128 * 4u) : sa_limit;

    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);
    const uint32_t b_voff0 = ((tile_n * 8u + wave_idx * 2u) * num_k64 * 64u + lane) * 16u;

    WeaveAgprAcc4 gate_acc[8][2];
    WeaveAgprAcc4 up_acc[8][2];
    #pragma unroll
    for (unsigned mx = 0u; mx < 8u; ++mx) {
        weave_agpr_acc_zero(gate_acc[mx][0]); weave_agpr_acc_zero(gate_acc[mx][1]);
        weave_agpr_acc_zero(up_acc[mx][0]);   weave_agpr_acc_zero(up_acc[mx][1]);
    }

    // Prologue: k64 = 0
    {
        const WeaveAgprI64x2 a0 = weave_ep8_ld_i64x2(a_rsrc, a_voff[0], 0u);
        const WeaveAgprI64x2 a1 = weave_ep8_ld_i64x2(a_rsrc, a_voff[1], 0u);
        const float sa0 = weave_ep8_ld_f32(sa_rsrc, sa_voff, 0u);
        const unsigned r0 = a_r64, r1 = 64u + a_r64;
        weave_ep8_lds_st_i64x2(lds + r0 * 64u + (a_c16 ^ ((r0 >> 1u) & 3u)) * 16u, a0);
        weave_ep8_lds_st_i64x2(lds + r1 * 64u + (a_c16 ^ ((r1 >> 1u) & 3u)) * 16u, a1);
        if (sa_active) reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff)[tid] = sa0;
    }
    WeaveAgprI64x2 bg_cur[2], bu_cur[2];
    bg_cur[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, 0u);
    bg_cur[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, b_nstride);
    bu_cur[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes);
    bu_cur[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + b_nstride);
    float sb_g_cur = bs_gate_s[0];
    float sb_u_cur = bs_gate_s[bs_up_stride];
    weave_4wave_lds_barrier_65440(arena);

    #pragma clang loop unroll(disable)
    for (unsigned k64 = 0u; k64 + 1u < num_k64; ++k64) {
        const unsigned nxt_k64 = k64 + 1u;
        const uint32_t ak_off = nxt_k64 * 64u;
        const uint32_t bk_off = nxt_k64 * 1024u;
        const bool need_next_scale = ((nxt_k64 & 1u) == 0u);
        const unsigned nxt_k128 = nxt_k64 >> 1u;

        const WeaveLdsU8* lds_a_cur = lds + (k64 & 1u) * kStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(
            lds + kSaBaseOff + ((k64 >> 1u) & 1u) * 512u);
        WeaveLdsU8* lds_a_nxt = lds + (nxt_k64 & 1u) * kStageBytes;

        // Staggered full prefetch: issue A_next + Bu_next BEFORE Gate (512 cycles flight),
        // then reload Bg_cur IN-PLACE right after Gate (256+ cycles flight, saves 16 VGPRs -> 0 spills!)
        const WeaveAgprI64x2 a_n0 = weave_ep8_ld_i64x2(a_rsrc, a_voff[0], ak_off);
        const WeaveAgprI64x2 a_n1 = weave_ep8_ld_i64x2(a_rsrc, a_voff[1], ak_off);
        const WeaveAgprI64x2 bu_n0 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + bk_off);
        const WeaveAgprI64x2 bu_n1 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, nk_bytes + bk_off + b_nstride);
        float sa_nxt = 0.0f, sb_g_nxt = sb_g_cur, sb_u_nxt = sb_u_cur;
        if (need_next_scale) {
            sa_nxt = weave_ep8_ld_f32(sa_rsrc, sa_voff, nxt_k128 * 4u);
            sb_g_nxt = bs_gate_s[nxt_k128];
            sb_u_nxt = bs_gate_s[bs_up_stride + nxt_k128];
        }

        {
            WeaveAgprF32x4 t0a, t1a, t0b, t1b;
            float sab_a = 0.0f, sab_b = 0.0f;
            #pragma unroll
            for (unsigned m2 = 0u; m2 < 8u; m2 += 2u) {
                {
                    const unsigned rm = m2 * 16u + m_lane;
                    const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                        lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                    sab_a = lds_sa_cur[rm] * sb_g_cur;
                    weave_mfma_f8_z(t0a, bg_cur[0][0], av[0]);
                    weave_mfma_f8_z(t1a, bg_cur[1][0], av[0]);
                    weave_mfma_f8_a(t0a, bg_cur[0][1], av[1]);
                    weave_mfma_f8_a(t1a, bg_cur[1][1], av[1]);
                    if (m2 > 0u) {
                        weave_agpr_prom_acc4(gate_acc[m2 - 1u][0], t0b, sab_b);
                        weave_agpr_prom_acc4(gate_acc[m2 - 1u][1], t1b, sab_b);
                    }
                }
                {
                    const unsigned rm = (m2 + 1u) * 16u + m_lane;
                    const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                        lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                    sab_b = lds_sa_cur[rm] * sb_g_cur;
                    weave_mfma_f8_z(t0b, bg_cur[0][0], av[0]);
                    weave_mfma_f8_z(t1b, bg_cur[1][0], av[0]);
                    weave_mfma_f8_a(t0b, bg_cur[0][1], av[1]);
                    weave_mfma_f8_a(t1b, bg_cur[1][1], av[1]);
                    weave_agpr_prom_acc4(gate_acc[m2][0], t0a, sab_a);
                    weave_agpr_prom_acc4(gate_acc[m2][1], t1a, sab_a);
                }
            }
            asm volatile("" : "+v"(bg_cur[0]), "+v"(bg_cur[1]));
            bg_cur[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk_off);
            bg_cur[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk_off + b_nstride);

            #pragma unroll
            for (unsigned m2 = 0u; m2 < 8u; m2 += 2u) {
                {
                    const unsigned rm = m2 * 16u + m_lane;
                    const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                        lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                    sab_a = lds_sa_cur[rm] * sb_u_cur;
                    weave_mfma_f8_z(t0a, bu_cur[0][0], av[0]);
                    weave_mfma_f8_z(t1a, bu_cur[1][0], av[0]);
                    weave_mfma_f8_a(t0a, bu_cur[0][1], av[1]);
                    weave_mfma_f8_a(t1a, bu_cur[1][1], av[1]);
                    if (m2 == 0u) {
                        weave_agpr_prom_acc4(gate_acc[7][0], t0b, sab_b);
                        weave_agpr_prom_acc4(gate_acc[7][1], t1b, sab_b);
                    } else {
                        weave_agpr_prom_acc4(up_acc[m2 - 1u][0], t0b, sab_b);
                        weave_agpr_prom_acc4(up_acc[m2 - 1u][1], t1b, sab_b);
                    }
                }
                {
                    const unsigned rm = (m2 + 1u) * 16u + m_lane;
                    const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                        lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                    sab_b = lds_sa_cur[rm] * sb_u_cur;
                    weave_mfma_f8_z(t0b, bu_cur[0][0], av[0]);
                    weave_mfma_f8_z(t1b, bu_cur[1][0], av[0]);
                    weave_mfma_f8_a(t0b, bu_cur[0][1], av[1]);
                    weave_mfma_f8_a(t1b, bu_cur[1][1], av[1]);
                    weave_agpr_prom_acc4(up_acc[m2][0], t0a, sab_a);
                    weave_agpr_prom_acc4(up_acc[m2][1], t1a, sab_a);
                }
            }
            weave_agpr_prom_acc4(up_acc[7][0], t0b, sab_b);
            weave_agpr_prom_acc4(up_acc[7][1], t1b, sab_b);
        }

        bu_cur[0] = bu_n0; bu_cur[1] = bu_n1;
        const unsigned r0 = a_r64, r1 = 64u + a_r64;
        weave_ep8_lds_st_i64x2(lds_a_nxt + r0 * 64u + (a_c16 ^ ((r0 >> 1u) & 3u)) * 16u, a_n0);
        weave_ep8_lds_st_i64x2(lds_a_nxt + r1 * 64u + (a_c16 ^ ((r1 >> 1u) & 3u)) * 16u, a_n1);
        if (need_next_scale) {
            if (sa_active) {
                reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + (nxt_k128 & 1u) * 512u)[tid] = sa_nxt;
            }
            sb_g_cur = sb_g_nxt;
            sb_u_cur = sb_u_nxt;
        }
        weave_4wave_lds_barrier_65440(arena);
    }

    // Final K=64 step: k64 = num_k64 - 1
    {
        const unsigned last_k64 = num_k64 - 1u;
        const WeaveLdsU8* lds_a_cur = lds + (last_k64 & 1u) * kStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(
            lds + kSaBaseOff + ((last_k64 >> 1u) & 1u) * 512u);
        WeaveAgprF32x4 t0a, t1a, t0b, t1b;
        float sab_a = 0.0f, sab_b = 0.0f;
        #pragma unroll
        for (unsigned m2 = 0u; m2 < 8u; m2 += 2u) {
            {
                const unsigned rm = m2 * 16u + m_lane;
                const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                    lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                sab_a = lds_sa_cur[rm] * sb_g_cur;
                weave_mfma_f8_z(t0a, bg_cur[0][0], av[0]);
                weave_mfma_f8_z(t1a, bg_cur[1][0], av[0]);
                weave_mfma_f8_a(t0a, bg_cur[0][1], av[1]);
                weave_mfma_f8_a(t1a, bg_cur[1][1], av[1]);
                if (m2 > 0u) {
                    weave_agpr_prom_acc4(gate_acc[m2 - 1u][0], t0b, sab_b);
                    weave_agpr_prom_acc4(gate_acc[m2 - 1u][1], t1b, sab_b);
                }
            }
            {
                const unsigned rm = (m2 + 1u) * 16u + m_lane;
                const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                    lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                sab_b = lds_sa_cur[rm] * sb_g_cur;
                weave_mfma_f8_z(t0b, bg_cur[0][0], av[0]);
                weave_mfma_f8_z(t1b, bg_cur[1][0], av[0]);
                weave_mfma_f8_a(t0b, bg_cur[0][1], av[1]);
                weave_mfma_f8_a(t1b, bg_cur[1][1], av[1]);
                weave_agpr_prom_acc4(gate_acc[m2][0], t0a, sab_a);
                weave_agpr_prom_acc4(gate_acc[m2][1], t1a, sab_a);
            }
        }
        #pragma unroll
        for (unsigned m2 = 0u; m2 < 8u; m2 += 2u) {
            {
                const unsigned rm = m2 * 16u + m_lane;
                const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                    lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                sab_a = lds_sa_cur[rm] * sb_u_cur;
                weave_mfma_f8_z(t0a, bu_cur[0][0], av[0]);
                weave_mfma_f8_z(t1a, bu_cur[1][0], av[0]);
                weave_mfma_f8_a(t0a, bu_cur[0][1], av[1]);
                weave_mfma_f8_a(t1a, bu_cur[1][1], av[1]);
                if (m2 == 0u) {
                    weave_agpr_prom_acc4(gate_acc[7][0], t0b, sab_b);
                    weave_agpr_prom_acc4(gate_acc[7][1], t1b, sab_b);
                } else {
                    weave_agpr_prom_acc4(up_acc[m2 - 1u][0], t0b, sab_b);
                    weave_agpr_prom_acc4(up_acc[m2 - 1u][1], t1b, sab_b);
                }
            }
            {
                const unsigned rm = (m2 + 1u) * 16u + m_lane;
                const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                    lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                sab_b = lds_sa_cur[rm] * sb_u_cur;
                weave_mfma_f8_z(t0b, bu_cur[0][0], av[0]);
                weave_mfma_f8_z(t1b, bu_cur[1][0], av[0]);
                weave_mfma_f8_a(t0b, bu_cur[0][1], av[1]);
                weave_mfma_f8_a(t1b, bu_cur[1][1], av[1]);
                weave_agpr_prom_acc4(up_acc[m2][0], t0a, sab_a);
                weave_agpr_prom_acc4(up_acc[m2][1], t1a, sab_a);
            }
        }
        weave_agpr_prom_acc4(up_acc[7][0], t0b, sab_b);
        weave_agpr_prom_acc4(up_acc[7][1], t1b, sab_b);
    }
    weave_4wave_lds_barrier_65440(arena);

    // 2-pass epilogue in lds[0..16383] (leaves lds_tok at 17408 intact!)
    WeaveLdsU16* lds_q = reinterpret_cast<WeaveLdsU16*>(lds);
    const unsigned q_row = tid >> 4u;
    const unsigned q_lane = tid & 15u;
    auto run_ep_s1 = [&]<unsigned kEp>() {
        #pragma unroll
        for (unsigned m_i = 0u; m_i < 4u; ++m_i) {
            constexpr unsigned kMxBase = kEp * 4u;
            const unsigned mx = kMxBase + m_i;
            const unsigned lds_r = m_i * 16u + m_lane;
            #pragma unroll
            for (unsigned nx = 0u; nx < 2u; ++nx) {
                const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
                const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx].lo[0]) * up_acc[mx][nx].lo[0])) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx].lo[1]) * up_acc[mx][nx].lo[1])) << 16u);
                const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx].hi[0]) * up_acc[mx][nx].hi[0])) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx].hi[1]) * up_acc[mx][nx].hi[1])) << 16u);
                *reinterpret_cast<WeaveLdsU2*>(lds_q + lds_r * 128u + col) = make_uint2(p0, p1);
            }
        }
        weave_4wave_lds_barrier_65440(arena);

        #pragma clang loop unroll(disable)
        for (unsigned r0 = 0u; r0 < 64u; r0 += 16u) {
            const unsigned lds_r = r0 + q_row;
            const unsigned r = kEp * 64u + lds_r;
            const unsigned dense_row = row_offset + r;
            const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
            uint4 raw = make_uint4(0, 0, 0, 0);
            if (valid) raw = *reinterpret_cast<const WeaveLdsU4*>(lds_q + lds_r * 128u + q_lane * 8u);
            weave_ep8_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
        }
        weave_4wave_lds_barrier_65440(arena);
    };
    run_ep_s1.template operator()<0u>();
    run_ep_s1.template operator()<1u>();
    asm volatile("" ::: "memory");
}

// ============================================================================
// 4. STAGE 2 BN=64 (BM=16, 32, 64): 3-Stage (K=384) / 2-Stage (K=256) Pipeline
// ============================================================================
template <unsigned kBM>
static __device__ __forceinline__ void weave_ep8_s2_agpr_bn64(
    const int32_t* __restrict__ s2_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    const float* __restrict__ weights_tile,
    uint16_t* __restrict__ c_out_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n64,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    static_assert(kBM == 16u || kBM == 32u || kBM == 64u);
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kMXDL = kBM / 16u;
    constexpr unsigned kTokOff = 16384u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows)
            ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu)
            : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = __builtin_amdgcn_readfirstlane(K >> 7u);
    const unsigned num_k64  = __builtin_amdgcn_readfirstlane(K >> 6u);
    const unsigned n_blocks128 = __builtin_amdgcn_readfirstlane(N >> 7u);
    const unsigned tile_n128 = __builtin_amdgcn_readfirstlane(tile_n64 >> 1u);
    const unsigned exp_u = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
    const uint64_t nk_bytes = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(exp_u) * nk_bytes;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(exp_u) * n_blocks128 + tile_n128) * num_k128;
    WeaveAgprConstAs4* bs_s = weave_agpr_make_as4_f32(bs_row);

    const uint32_t a_limit  = __builtin_amdgcn_readfirstlane(route_rows * K);
    const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(route_rows * num_k128 * 4u);
    const uint32_t b_limit  = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(nk_bytes));
    const WeaveEp8ABuf a_rsrc  = weave_ep8_make_abuf(a_base, a_limit);
    const WeaveEp8ABuf sa_rsrc = weave_ep8_make_abuf(a_scale_base, sa_limit);
    const WeaveEp8BBuf b_rsrc  = weave_ep8_make_bbuf(b_exp, b_limit);

    const uint32_t b_voff0 = ((tile_n64 * 4u + wave_idx) * num_k64 * 64u + lane) * 16u;

    WeaveAgprAcc4 acc[kMXDL];
    uint32_t a_voff[kMXDL];
    uint32_t sa_voff[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        weave_agpr_acc_zero(acc[mx]);
        const unsigned r = mx * 16u + m_lane;
        const uint32_t tok = lds_tok[r];
        const bool ok = (r < valid_rows) && (tok < route_rows);
        a_voff[mx]  = ok ? (tok * K + k_grp * 16u) : a_limit;
        sa_voff[mx] = ok ? (tok * num_k128 * 4u) : sa_limit;
    }

    auto load_half = [&](unsigned k64_idx, WeaveAgprI64x2 (&av)[kMXDL], WeaveAgprI64x2& bc) {
        const uint32_t ak = k64_idx * 64u;
        const uint32_t bk = k64_idx * 1024u;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av[mx] = weave_ep8_ld_i64x2(a_rsrc, a_voff[mx], ak);
        }
        bc = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk);
    };

    auto load_scales = [&](unsigned k128_idx, float (&sa)[kMXDL], float& sb) {
        const uint32_t sak = k128_idx * 4u;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            sa[mx] = weave_ep8_ld_f32(sa_rsrc, sa_voff[mx], sak);
        }
        sb = bs_s[k128_idx];
    };

    auto mfma_half = [&](const WeaveAgprI64x2 (&av)[kMXDL], const WeaveAgprI64x2& bc,
                         const float (&sa)[kMXDL], float sb) {
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            WeaveAgprF32x4 t0;
            weave_mfma_f8_z(t0, bc[0], av[mx][0]);
            weave_mfma_f8_a(t0, bc[1], av[mx][1]);
            weave_agpr_prom_acc4(acc[mx], t0, sa[mx] * sb);
        }
    };

    auto mfma_k128 = [&](const WeaveAgprI64x2 (&av0)[kMXDL], const WeaveAgprI64x2 (&av1)[kMXDL],
                         const WeaveAgprI64x2& b0, const WeaveAgprI64x2& b1,
                         const float (&sa)[kMXDL], float sb) {
        WeaveAgprF32x4 t[kMXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            weave_mfma_f8_z(t[mx], b0[0], av0[mx][0]);
            weave_mfma_f8_a(t[mx], b0[1], av0[mx][1]);
            weave_mfma_f8_a(t[mx], b1[0], av1[mx][0]);
            weave_mfma_f8_a(t[mx], b1[1], av1[mx][1]);
        }
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            weave_agpr_prom_acc4(acc[mx], t[mx], sa[mx] * sb);
        }
    };

    if constexpr (kBM <= 32u) {
        if (num_k128 >= 3u) {
            WeaveAgprI64x2 av0_0[kMXDL], av1_0[kMXDL], b0_0, b1_0;
            WeaveAgprI64x2 av0_1[kMXDL], av1_1[kMXDL], b0_1, b1_1;
            WeaveAgprI64x2 av0_2[kMXDL], av1_2[kMXDL], b0_2, b1_2;
            float sa_0[kMXDL], sa_1[kMXDL], sa_2[kMXDL], sb_0, sb_1, sb_2;

            load_half(0u, av0_0, b0_0); load_scales(0u, sa_0, sb_0); load_half(1u, av1_0, b1_0);
            load_half(2u, av0_1, b0_1); load_scales(1u, sa_1, sb_1); load_half(3u, av1_1, b1_1);
            load_half(4u, av0_2, b0_2); load_scales(2u, sa_2, sb_2); load_half(5u, av1_2, b1_2);

            #pragma clang loop unroll(disable)
            for (unsigned kb = 0u; kb + 3u < num_k128; ++kb) {
                const unsigned nn = kb + 3u;
                mfma_k128(av0_0, av1_0, b0_0, b1_0, sa_0, sb_0);
                load_half(nn * 2u + 0u, av0_0, b0_0);
                float sa_nn[kMXDL], sb_nn;
                load_scales(nn, sa_nn, sb_nn);
                load_half(nn * 2u + 1u, av1_0, b1_0);

                #pragma unroll
                for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                    const WeaveAgprI64x2 t0 = av0_0[mx]; av0_0[mx] = av0_1[mx]; av0_1[mx] = av0_2[mx]; av0_2[mx] = t0;
                    const WeaveAgprI64x2 t1 = av1_0[mx]; av1_0[mx] = av1_1[mx]; av1_1[mx] = av1_2[mx]; av1_2[mx] = t1;
                    sa_0[mx] = sa_1[mx]; sa_1[mx] = sa_2[mx]; sa_2[mx] = sa_nn[mx];
                }
                const WeaveAgprI64x2 q0 = b0_0; b0_0 = b0_1; b0_1 = b0_2; b0_2 = q0;
                const WeaveAgprI64x2 q1 = b1_0; b1_0 = b1_1; b1_1 = b1_2; b1_2 = q1;
                sb_0 = sb_1; sb_1 = sb_2; sb_2 = sb_nn;
            }
            mfma_k128(av0_0, av1_0, b0_0, b1_0, sa_0, sb_0);
            mfma_k128(av0_1, av1_1, b0_1, b1_1, sa_1, sb_1);
            mfma_k128(av0_2, av1_2, b0_2, b1_2, sa_2, sb_2);
        } else {
            #pragma clang loop unroll(disable)
            for (unsigned kb = 0u; kb < num_k128; ++kb) {
                WeaveAgprI64x2 av0[kMXDL], av1[kMXDL], b0, b1;
                float sa[kMXDL], sb;
                load_half(kb * 2u + 0u, av0, b0); load_scales(kb, sa, sb); load_half(kb * 2u + 1u, av1, b1);
                mfma_k128(av0, av1, b0, b1, sa, sb);
            }
        }
    } else {
        if ((num_k128 >= 2u) && ((num_k128 & 1u) == 0u)) {
            WeaveAgprI64x2 av0_a[kMXDL], av1_a[kMXDL], b0_a, b1_a; float sa_a[kMXDL], sb_a;
            WeaveAgprI64x2 av0_b[kMXDL], av1_b[kMXDL], b0_b, b1_b; float sa_b[kMXDL], sb_b;
            load_half(0u, av0_a, b0_a); load_scales(0u, sa_a, sb_a); load_half(1u, av1_a, b1_a);
            load_half(2u, av0_b, b0_b); load_scales(1u, sa_b, sb_b); load_half(3u, av1_b, b1_b);

            #pragma clang loop unroll(disable)
            for (unsigned kb = 0u; kb + 2u < num_k128; kb += 2u) {
                mfma_k128(av0_a, av1_a, b0_a, b1_a, sa_a, sb_a);
                load_half((kb + 2u) * 2u + 0u, av0_a, b0_a); load_scales(kb + 2u, sa_a, sb_a); load_half((kb + 2u) * 2u + 1u, av1_a, b1_a);

                mfma_k128(av0_b, av1_b, b0_b, b1_b, sa_b, sb_b);
                load_half((kb + 3u) * 2u + 0u, av0_b, b0_b); load_scales(kb + 3u, sa_b, sb_b); load_half((kb + 3u) * 2u + 1u, av1_b, b1_b);
            }
            mfma_k128(av0_a, av1_a, b0_a, b1_a, sa_a, sb_a);
            mfma_k128(av0_b, av1_b, b0_b, b1_b, sa_b, sb_b);
        } else {
            #pragma clang loop unroll(disable)
            for (unsigned kb = 0u; kb < num_k128; ++kb) {
                WeaveAgprI64x2 av0[kMXDL], av1[kMXDL], b0, b1;
                float sa[kMXDL], sb;
                load_half(kb * 2u + 0u, av0, b0); load_scales(kb, sa, sb); load_half(kb * 2u + 1u, av1, b1);
                mfma_k128(av0, av1, b0, b1, sa, sb);
            }
        }
    }

    WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
    const unsigned h_row = tid >> 3u;
    const unsigned h_lane = tid & 7u;
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + m_lane;
        const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
        const unsigned col = wave_idx * 16u + k_grp * 4u;
        const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx].lo[0] * weight)) |
            (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx].lo[1] * weight)) << 16u);
        const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx].hi[0] * weight)) |
            (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx].hi[1] * weight)) << 16u);
        const unsigned c8 = (col >> 2u) ^ (r & 7u);
        *reinterpret_cast<WeaveLdsU2*>(lds_c + r * 64u + c8 * 4u) = make_uint2(p0, p1);
    }
    weave_4wave_lds_barrier_65440(arena);

    #pragma clang loop unroll(disable)
    for (unsigned r0 = 0u; r0 < kBM; r0 += 32u) {
        const unsigned r = r0 + h_row;
        if (r < kBM) {
            const uint32_t tok_m = lds_tok[r];
            if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
                const unsigned c8_0 = (h_lane * 2u + 0u) ^ (r & 7u);
                const unsigned c8_1 = (h_lane * 2u + 1u) ^ (r & 7u);
                const uint2 u0 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 64u + c8_0 * 4u);
                const uint2 u1 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 64u + c8_1 * 4u);
                weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n64 * 64u + h_lane * 8u,
                                make_uint4(u0.x, u0.y, u1.x, u1.y));
            }
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

// ============================================================================
// 5. STAGE 2 BN=256 (BM=16, 32, 64): 1x4 Split-B Computing (tile_n, tile_n+1)
//    Halves A traffic & K-loop barriers! Proven +11%..+12% faster on S=128!
// ============================================================================
template <unsigned kBM>
static __device__ __forceinline__ void weave_ep8_s2_agpr_bn256(
    const int32_t* __restrict__ s2_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    const float* __restrict__ weights_tile,
    uint16_t* __restrict__ c_out_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    static_assert(kBM == 16u || kBM == 32u || kBM == 64u);
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kMXDL = kBM / 16u;
    constexpr unsigned kAPasses = (kBM + 31u) / 32u;
    constexpr unsigned kAStageBytes = kBM * 128u;
    constexpr unsigned kSaBaseOff = 16384u;
    constexpr unsigned kTokOff = 17408u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows)
            ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu)
            : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned nk = __builtin_amdgcn_readfirstlane(K >> 7u);
    const unsigned num_k64 = __builtin_amdgcn_readfirstlane(K >> 6u);
    const unsigned n_blocks = __builtin_amdgcn_readfirstlane(N >> 7u);
    const unsigned exp_u = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
    const uint64_t nk_bytes = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(exp_u) * nk_bytes;
    const float* bs_row0 = b_scale_base + (static_cast<uint64_t>(exp_u) * n_blocks + tile_n) * nk;
    WeaveAgprConstAs4* bs0_s = weave_agpr_make_as4_f32(bs_row0);

    const uint32_t a_limit  = __builtin_amdgcn_readfirstlane(route_rows * K);
    const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(route_rows * nk * 4u);
    const uint32_t b_limit  = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(nk_bytes));
    const WeaveEp8ABuf a_rsrc  = weave_ep8_make_abuf(a_base, a_limit);
    const WeaveEp8ABuf sa_rsrc = weave_ep8_make_abuf(a_scale_base, sa_limit);
    const WeaveEp8BBuf b_rsrc  = weave_ep8_make_bbuf(b_exp, b_limit);

    const unsigned a_r32 = tid >> 3u;
    const unsigned a_c16 = tid & 7u;
    const bool a_thr = (kBM >= 32u) || (a_r32 < kBM);
    uint32_t a_voff[kAPasses];
    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 32u + a_r32;
        const uint32_t tok = a_thr ? lds_tok[(r < kBM) ? r : 0u] : route_rows;
        const bool ok = a_thr && (r < valid_rows) && (tok < route_rows);
        a_voff[ap] = ok ? (tok * K + a_c16 * 16u) : a_limit;
    }
    const bool sa_active = (tid < kBM);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : route_rows;
    const uint32_t sa_voff = ((tid < valid_rows) && sa_active && (tok_sa < route_rows))
        ? (tok_sa * nk * 4u) : sa_limit;

    WeaveAgprAcc4 acc[kMXDL][2][2];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        #pragma unroll
        for (unsigned h = 0u; h < 2u; ++h) {
            weave_agpr_acc_zero(acc[mx][h][0]);
            weave_agpr_acc_zero(acc[mx][h][1]);
        }
    }

    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);
    const uint32_t b_half_soff = __builtin_amdgcn_readfirstlane(8u * b_nstride);
    const uint32_t b_voff0 = ((tile_n * 8u + wave_idx * 2u) * num_k64 * 64u + lane) * 16u;

    WeaveAgprI64x2 ar[kAPasses];
    float sar, sb0r, sb1r;
    WeaveAgprI64x2 b0_0, b0_1, b0_2, b0_3;
    WeaveAgprI64x2 b1_0, b1_1, b1_2, b1_3;

    auto issue_a = [&](unsigned step) {
        const unsigned s = (step < nk) ? step : (nk - 1u);
        const uint32_t ak = s * 128u;
        const uint32_t sak = s * 4u;
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) {
            ar[ap] = weave_ep8_ld_i64x2(a_rsrc, a_voff[ap], ak);
        }
        sar = weave_ep8_ld_f32(sa_rsrc, sa_voff, sak);
    };
    auto issue_b0 = [&](unsigned step) {
        const unsigned s = (step < nk) ? step : (nk - 1u);
        const uint32_t bk = s * 2048u;
        b0_0 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk);
        b0_1 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk + b_nstride);
        b0_2 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk + 1024u);
        b0_3 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk + 1024u + b_nstride);
        sb0r = bs0_s[s];
    };
    auto issue_b1 = [&](unsigned step) {
        const unsigned s = (step < nk) ? step : (nk - 1u);
        const uint32_t bk = b_half_soff + s * 2048u;
        b1_0 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk);
        b1_1 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk + b_nstride);
        b1_2 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk + 1024u);
        b1_3 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk + 1024u + b_nstride);
        sb1r = bs0_s[nk + s];
    };
    auto stage_a = [&](unsigned step) {
        WeaveLdsU8* dst = lds + (step & 1u) * kAStageBytes;
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) {
            const unsigned r = ap * 32u + a_r32;
            if (a_thr) weave_ep8_lds_st_i64x2(dst + r * 128u + (a_c16 ^ (r & 7u)) * 16u, ar[ap]);
        }
        if (sa_active) {
            reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + (step & 1u) * 512u)[tid] = sar;
        }
    };

    issue_a(0u);
    issue_b0(0u);
    issue_b1(0u);
    stage_a(0u);
    weave_4wave_lds_barrier_65440(arena);

    #pragma clang loop unroll(disable)
    for (unsigned kb = 0u; kb < nk; ++kb) {
        issue_a(kb + 1u);
        const WeaveLdsU8* lds_a_cur = lds + (kb & 1u) * kAStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + (kb & 1u) * 512u);
        WeaveAgprI64x2 av0[kMXDL], av1[kMXDL];
        float sa[kMXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned rm = mx * 16u + m_lane;
            av0[mx] = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
            av1[mx] = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
            sa[mx] = lds_sa_cur[rm];
        }

        WeaveAgprF32x4 t10[kMXDL], t11[kMXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const float sab0 = sa[mx] * sb0r;
            WeaveAgprF32x4 t00, t01;
            weave_mfma_f8_z(t00, b0_0[0], av0[mx][0]);
            weave_mfma_f8_z(t01, b0_1[0], av0[mx][0]);
            weave_mfma_f8_a(t00, b0_0[1], av0[mx][1]);
            weave_mfma_f8_a(t01, b0_1[1], av0[mx][1]);
            weave_mfma_f8_a(t00, b0_2[0], av1[mx][0]);
            weave_mfma_f8_a(t01, b0_3[0], av1[mx][0]);
            weave_mfma_f8_a(t00, b0_2[1], av1[mx][1]);
            weave_mfma_f8_a(t01, b0_3[1], av1[mx][1]);

            weave_mfma_f8_z(t10[mx], b1_0[0], av0[mx][0]);
            weave_mfma_f8_z(t11[mx], b1_1[0], av0[mx][0]);
            weave_mfma_f8_a(t10[mx], b1_0[1], av0[mx][1]);
            weave_mfma_f8_a(t11[mx], b1_1[1], av0[mx][1]);
            weave_agpr_prom_acc4(acc[mx][0][0], t00, sab0);
            weave_agpr_prom_acc4(acc[mx][0][1], t01, sab0);
            weave_mfma_f8_a(t10[mx], b1_2[0], av1[mx][0]);
            weave_mfma_f8_a(t11[mx], b1_3[0], av1[mx][0]);
            weave_mfma_f8_a(t10[mx], b1_2[1], av1[mx][1]);
            weave_mfma_f8_a(t11[mx], b1_3[1], av1[mx][1]);
        }
        const float sb1_cur = sb1r;
        issue_b0(kb + 1u);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const float sab1 = sa[mx] * sb1_cur;
            weave_agpr_prom_acc4(acc[mx][1][0], t10[mx], sab1);
            weave_agpr_prom_acc4(acc[mx][1][1], t11[mx], sab1);
        }
        issue_b1(kb + 1u);

        stage_a(kb + 1u);
        weave_4wave_lds_barrier_65440(arena);
    }

    const unsigned c_row = tid >> 4u;
    const unsigned c_lane = tid & 15u;
    WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
    auto run_ep_bn256 = [&]<unsigned kH>() {
        weave_4wave_lds_barrier_65440(arena);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned r = mx * 16u + m_lane;
            const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
            #pragma unroll
            for (unsigned nx = 0u; nx < 2u; ++nx) {
                const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
                const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][kH][nx].lo[0] * weight)) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][kH][nx].lo[1] * weight)) << 16u);
                const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][kH][nx].hi[0] * weight)) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][kH][nx].hi[1] * weight)) << 16u);
                const unsigned c8 = (col >> 2u) ^ (r & 7u);
                *reinterpret_cast<WeaveLdsU2*>(lds_c + r * 128u + c8 * 4u) = make_uint2(p0, p1);
            }
        }
        weave_4wave_lds_barrier_65440(arena);
        #pragma clang loop unroll(disable)
        for (unsigned r0 = 0u; r0 < kBM; r0 += 16u) {
            const unsigned r = r0 + c_row;
            const uint32_t tok_m = lds_tok[r];
            if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
                const unsigned c8_0 = (c_lane * 2u + 0u) ^ (r & 7u);
                const unsigned c8_1 = (c_lane * 2u + 1u) ^ (r & 7u);
                const uint2 u0 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_0 * 4u);
                const uint2 u1 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_1 * 4u);
                weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + (tile_n + kH) * 128u + c_lane * 8u,
                                make_uint4(u0.x, u0.y, u1.x, u1.y));
            }
        }
    };
    run_ep_bn256.template operator()<0u>();
    run_ep_bn256.template operator()<1u>();
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

// ============================================================================
// 6. STAGE 2 BN=128 SMALL (BM=16, 32): 3-Stage (K=384) Deep Pipeline
// ============================================================================
template <unsigned kBM>
static __device__ __forceinline__ void weave_ep8_s2_small_deep(
    const int32_t* __restrict__ s2_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    const float* __restrict__ weights_tile,
    uint16_t* __restrict__ c_out_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    static_assert(kBM == 16u || kBM == 32u);
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kMXDL = kBM / 16u;
    constexpr unsigned kTokOff = 16384u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows)
            ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu)
            : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = __builtin_amdgcn_readfirstlane(K >> 7u);
    const unsigned num_k64  = __builtin_amdgcn_readfirstlane(K >> 6u);
    const unsigned n_blocks = __builtin_amdgcn_readfirstlane(N >> 7u);
    const unsigned exp_u = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
    const uint64_t nk_bytes = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(exp_u) * nk_bytes;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(exp_u) * n_blocks + tile_n) * num_k128;
    WeaveAgprConstAs4* bs_s = weave_agpr_make_as4_f32(bs_row);

    const uint32_t a_limit  = __builtin_amdgcn_readfirstlane(route_rows * K);
    const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(route_rows * num_k128 * 4u);
    const uint32_t b_limit  = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(nk_bytes));
    const WeaveEp8ABuf a_rsrc  = weave_ep8_make_abuf(a_base, a_limit);
    const WeaveEp8ABuf sa_rsrc = weave_ep8_make_abuf(a_scale_base, sa_limit);
    const WeaveEp8BBuf b_rsrc  = weave_ep8_make_bbuf(b_exp, b_limit);

    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);
    const uint32_t b_voff0 = ((tile_n * 8u + wave_idx * 2u) * num_k64 * 64u + lane) * 16u;

    WeaveAgprAcc4 acc[kMXDL][2];
    uint32_t a_voff[kMXDL];
    uint32_t sa_voff[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        weave_agpr_acc_zero(acc[mx][0]); weave_agpr_acc_zero(acc[mx][1]);
        const unsigned r = mx * 16u + m_lane;
        const uint32_t tok = lds_tok[r];
        const bool ok = (r < valid_rows) && (tok < route_rows);
        a_voff[mx]  = ok ? (tok * K + k_grp * 16u) : a_limit;
        sa_voff[mx] = ok ? (tok * num_k128 * 4u) : sa_limit;
    }

    auto load_half = [&](unsigned k64_idx, WeaveAgprI64x2 (&av)[kMXDL], WeaveAgprI64x2 (&bc)[2]) {
        const uint32_t ak = k64_idx * 64u;
        const uint32_t bk = k64_idx * 1024u;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av[mx] = weave_ep8_ld_i64x2(a_rsrc, a_voff[mx], ak);
        }
        bc[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk);
        bc[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk + b_nstride);
    };
    auto load_scales = [&](unsigned k128_idx, float (&sa)[kMXDL], float& sb) {
        const uint32_t sak = k128_idx * 4u;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            sa[mx] = weave_ep8_ld_f32(sa_rsrc, sa_voff[mx], sak);
        }
        sb = bs_s[k128_idx];
    };
    auto mfma_half = [&](const WeaveAgprI64x2 (&av)[kMXDL], const WeaveAgprI64x2 (&bc)[2],
                         const float (&sa)[kMXDL], float sb) {
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const float sab = sa[mx] * sb;
            WeaveAgprF32x4 t0, t1;
            weave_mfma_f8_z(t0, bc[0][0], av[mx][0]);
            weave_mfma_f8_z(t1, bc[1][0], av[mx][0]);
            weave_mfma_f8_a(t0, bc[0][1], av[mx][1]);
            weave_mfma_f8_a(t1, bc[1][1], av[mx][1]);
            weave_agpr_prom_acc4(acc[mx][0], t0, sab);
            weave_agpr_prom_acc4(acc[mx][1], t1, sab);
        }
    };
    auto mfma_k128 = [&](const WeaveAgprI64x2 (&av0)[kMXDL], const WeaveAgprI64x2 (&av1)[kMXDL],
                         const WeaveAgprI64x2 (&b0)[2], const WeaveAgprI64x2 (&b1)[2],
                         const float (&sa)[kMXDL], float sb) {
        WeaveAgprF32x4 t0[kMXDL], t1[kMXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            weave_mfma_f8_z(t0[mx], b0[0][0], av0[mx][0]);
            weave_mfma_f8_z(t1[mx], b0[1][0], av0[mx][0]);
            weave_mfma_f8_a(t0[mx], b0[0][1], av0[mx][1]);
            weave_mfma_f8_a(t1[mx], b0[1][1], av0[mx][1]);
            weave_mfma_f8_a(t0[mx], b1[0][0], av1[mx][0]);
            weave_mfma_f8_a(t1[mx], b1[1][0], av1[mx][0]);
            weave_mfma_f8_a(t0[mx], b1[0][1], av1[mx][1]);
            weave_mfma_f8_a(t1[mx], b1[1][1], av1[mx][1]);
        }
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const float sab = sa[mx] * sb;
            weave_agpr_prom_acc4(acc[mx][0], t0[mx], sab);
            weave_agpr_prom_acc4(acc[mx][1], t1[mx], sab);
        }
    };

    if (num_k128 >= 3u) {
        WeaveAgprI64x2 av0_0[kMXDL], av1_0[kMXDL], b0_0[2], b1_0[2];
        WeaveAgprI64x2 av0_1[kMXDL], av1_1[kMXDL], b0_1[2], b1_1[2];
        WeaveAgprI64x2 av0_2[kMXDL], av1_2[kMXDL], b0_2[2], b1_2[2];
        float sa_0[kMXDL], sa_1[kMXDL], sa_2[kMXDL], sb_0, sb_1, sb_2;

        load_half(0u, av0_0, b0_0); load_scales(0u, sa_0, sb_0); load_half(1u, av1_0, b1_0);
        load_half(2u, av0_1, b0_1); load_scales(1u, sa_1, sb_1); load_half(3u, av1_1, b1_1);
        load_half(4u, av0_2, b0_2); load_scales(2u, sa_2, sb_2); load_half(5u, av1_2, b1_2);

        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u; kb + 3u < num_k128; ++kb) {
            const unsigned nn = kb + 3u;
            mfma_k128(av0_0, av1_0, b0_0, b1_0, sa_0, sb_0);
            load_half(nn * 2u + 0u, av0_0, b0_0);
            float sa_nn[kMXDL], sb_nn;
            load_scales(nn, sa_nn, sb_nn);
            load_half(nn * 2u + 1u, av1_0, b1_0);

            #pragma unroll
            for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                const WeaveAgprI64x2 t0 = av0_0[mx]; av0_0[mx] = av0_1[mx]; av0_1[mx] = av0_2[mx]; av0_2[mx] = t0;
                const WeaveAgprI64x2 t1 = av1_0[mx]; av1_0[mx] = av1_1[mx]; av1_1[mx] = av1_2[mx]; av1_2[mx] = t1;
                sa_0[mx] = sa_1[mx]; sa_1[mx] = sa_2[mx]; sa_2[mx] = sa_nn[mx];
            }
            #pragma unroll
            for (unsigned nx = 0u; nx < 2u; ++nx) {
                const WeaveAgprI64x2 q0 = b0_0[nx]; b0_0[nx] = b0_1[nx]; b0_1[nx] = b0_2[nx]; b0_2[nx] = q0;
                const WeaveAgprI64x2 q1 = b1_0[nx]; b1_0[nx] = b1_1[nx]; b1_1[nx] = b1_2[nx]; b1_2[nx] = q1;
            }
            sb_0 = sb_1; sb_1 = sb_2; sb_2 = sb_nn;
        }
        mfma_k128(av0_0, av1_0, b0_0, b1_0, sa_0, sb_0);
        mfma_k128(av0_1, av1_1, b0_1, b1_1, sa_1, sb_1);
        mfma_k128(av0_2, av1_2, b0_2, b1_2, sa_2, sb_2);
    } else {
        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u; kb < num_k128; ++kb) {
            WeaveAgprI64x2 av0[kMXDL], av1[kMXDL], b0[2], b1[2];
            float sa[kMXDL], sb;
            load_half(kb * 2u + 0u, av0, b0); load_scales(kb, sa, sb); load_half(kb * 2u + 1u, av1, b1);
            mfma_k128(av0, av1, b0, b1, sa, sb);
        }
    }

    WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
    const unsigned c_row = tid >> 4u;
    const unsigned c_lane = tid & 15u;
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + m_lane;
        const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].lo[0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].lo[1] * weight)) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].hi[0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].hi[1] * weight)) << 16u);
            const unsigned c8 = (col >> 2u) ^ (r & 7u);
            *reinterpret_cast<WeaveLdsU2*>(lds_c + r * 128u + c8 * 4u) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + c_row;
        const uint32_t tok_m = lds_tok[r];
        if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
            const unsigned c8_0 = (c_lane * 2u + 0u) ^ (r & 7u);
            const unsigned c8_1 = (c_lane * 2u + 1u) ^ (r & 7u);
            const uint2 u0 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_0 * 4u);
            const uint2 u1 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_1 * 4u);
            weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n * 128u + c_lane * 8u,
                            make_uint4(u0.x, u0.y, u1.x, u1.y));
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

// ============================================================================
// 7. STAGE 2 BM=64 & BM=128 (1x4 Cooperative A-LDS, <= 17.9 KB LDS -> 0 split_128!)
// ============================================================================
static __device__ __forceinline__ void weave_ep8_s2_1x4_bm64(
    const int32_t* __restrict__ s2_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    const float* __restrict__ weights_tile,
    uint16_t* __restrict__ c_out_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kStageBytes = 8192u;
    constexpr unsigned kSaBaseOff  = 16384u;
    constexpr unsigned kTokOff     = 17408u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < 64u) {
        lds_tok[tid] = (tid < valid_rows)
            ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu)
            : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = __builtin_amdgcn_readfirstlane(K >> 7u);
    const unsigned num_k64  = __builtin_amdgcn_readfirstlane(K >> 6u);
    const unsigned n_blocks = __builtin_amdgcn_readfirstlane(N >> 7u);
    const unsigned exp_u = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
    const uint64_t nk_bytes = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(exp_u) * nk_bytes;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(exp_u) * n_blocks + tile_n) * num_k128;
    WeaveAgprConstAs4* bs_s = weave_agpr_make_as4_f32(bs_row);

    const uint32_t a_limit  = __builtin_amdgcn_readfirstlane(route_rows * K);
    const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(route_rows * num_k128 * 4u);
    const uint32_t b_limit  = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(nk_bytes));
    const WeaveEp8ABuf a_rsrc  = weave_ep8_make_abuf(a_base, a_limit);
    const WeaveEp8ABuf sa_rsrc = weave_ep8_make_abuf(a_scale_base, sa_limit);
    const WeaveEp8BBuf b_rsrc  = weave_ep8_make_bbuf(b_exp, b_limit);

    const unsigned a_r32 = tid >> 3u;
    const unsigned a_c16 = tid & 7u;
    uint32_t a_voff[2];
    #pragma unroll
    for (unsigned ap = 0u; ap < 2u; ++ap) {
        const unsigned r = ap * 32u + a_r32;
        const uint32_t tok = lds_tok[r];
        a_voff[ap] = ((r < valid_rows) && (tok < route_rows)) ? (tok * K + a_c16 * 16u) : a_limit;
    }
    const bool sa_active = (tid < 64u);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : route_rows;
    const uint32_t sa_voff = (sa_active && (tid < valid_rows) && (tok_sa < route_rows))
        ? (tok_sa * num_k128 * 4u) : sa_limit;

    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);
    const uint32_t b_voff0 = ((tile_n * 8u + wave_idx * 2u) * num_k64 * 64u + lane) * 16u;

    WeaveAgprAcc4 acc[4][2];
    #pragma unroll
    for (unsigned mx = 0u; mx < 4u; ++mx) {
        weave_agpr_acc_zero(acc[mx][0]); weave_agpr_acc_zero(acc[mx][1]);
    }

    {
        const WeaveAgprI64x2 a0 = weave_ep8_ld_i64x2(a_rsrc, a_voff[0], 0u);
        const WeaveAgprI64x2 a1 = weave_ep8_ld_i64x2(a_rsrc, a_voff[1], 0u);
        const float sa0 = weave_ep8_ld_f32(sa_rsrc, sa_voff, 0u);
        const unsigned r0 = a_r32, r1 = 32u + a_r32;
        weave_ep8_lds_st_i64x2(lds + r0 * 128u + (a_c16 ^ (r0 & 7u)) * 16u, a0);
        weave_ep8_lds_st_i64x2(lds + r1 * 128u + (a_c16 ^ (r1 & 7u)) * 16u, a1);
        if (sa_active) reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff)[tid] = sa0;
    }
    WeaveAgprI64x2 bw0[2], bw1[2];
    bw0[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, 0u);
    bw0[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, b_nstride);
    bw1[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, 1024u);
    bw1[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, 1024u + b_nstride);
    float sb_cur = bs_s[0];
    weave_4wave_lds_barrier_65440(arena);

    #pragma clang loop unroll(disable)
    for (unsigned kb = 0u; kb + 1u < num_k128; ++kb) {
        const unsigned nxt_kb = kb + 1u;
        const uint32_t ak_off = nxt_kb * 128u;
        const uint32_t bk_off = nxt_kb * 2048u;
        const WeaveLdsU8* lds_a_cur = lds + (kb & 1u) * kStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + (kb & 1u) * 512u);
        WeaveLdsU8* lds_a_nxt = lds + (nxt_kb & 1u) * kStageBytes;

        const WeaveAgprI64x2 a_n0 = weave_ep8_ld_i64x2(a_rsrc, a_voff[0], ak_off);
        const WeaveAgprI64x2 a_n1 = weave_ep8_ld_i64x2(a_rsrc, a_voff[1], ak_off);
        const float sa_nxt = weave_ep8_ld_f32(sa_rsrc, sa_voff, nxt_kb * 4u);
        const WeaveAgprI64x2 bw0_n0 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk_off);
        const WeaveAgprI64x2 bw0_n1 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk_off + b_nstride);
        const WeaveAgprI64x2 bw1_n0 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk_off + 1024u);
        const WeaveAgprI64x2 bw1_n1 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk_off + 1024u + b_nstride);
        const float sb_nxt = bs_s[nxt_kb];

        {
            WeaveAgprF32x4 t0a, t1a, t0b, t1b;
            float sab_a = 0.0f, sab_b = 0.0f;
            #pragma unroll
            for (unsigned m2 = 0u; m2 < 4u; m2 += 2u) {
                {
                    const unsigned rm = m2 * 16u + m_lane;
                    const WeaveAgprI64x2 av0 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
                    const WeaveAgprI64x2 av1 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
                    sab_a = lds_sa_cur[rm] * sb_cur;
                    weave_mfma_f8_z(t0a, bw0[0][0], av0[0]);
                    weave_mfma_f8_z(t1a, bw0[1][0], av0[0]);
                    weave_mfma_f8_a(t0a, bw0[0][1], av0[1]);
                    weave_mfma_f8_a(t1a, bw0[1][1], av0[1]);
                    if (m2 > 0u) {
                        weave_agpr_prom_acc4(acc[m2 - 1u][0], t0b, sab_b);
                        weave_agpr_prom_acc4(acc[m2 - 1u][1], t1b, sab_b);
                    }
                    weave_mfma_f8_a(t0a, bw1[0][0], av1[0]);
                    weave_mfma_f8_a(t1a, bw1[1][0], av1[0]);
                    weave_mfma_f8_a(t0a, bw1[0][1], av1[1]);
                    weave_mfma_f8_a(t1a, bw1[1][1], av1[1]);
                }
                {
                    const unsigned rm = (m2 + 1u) * 16u + m_lane;
                    const WeaveAgprI64x2 av0 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
                    const WeaveAgprI64x2 av1 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
                    sab_b = lds_sa_cur[rm] * sb_cur;
                    weave_mfma_f8_z(t0b, bw0[0][0], av0[0]);
                    weave_mfma_f8_z(t1b, bw0[1][0], av0[0]);
                    weave_mfma_f8_a(t0b, bw0[0][1], av0[1]);
                    weave_mfma_f8_a(t1b, bw0[1][1], av0[1]);
                    weave_agpr_prom_acc4(acc[m2][0], t0a, sab_a);
                    weave_agpr_prom_acc4(acc[m2][1], t1a, sab_a);
                    weave_mfma_f8_a(t0b, bw1[0][0], av1[0]);
                    weave_mfma_f8_a(t1b, bw1[1][0], av1[0]);
                    weave_mfma_f8_a(t0b, bw1[0][1], av1[1]);
                    weave_mfma_f8_a(t1b, bw1[1][1], av1[1]);
                }
            }
            weave_agpr_prom_acc4(acc[3][0], t0b, sab_b);
            weave_agpr_prom_acc4(acc[3][1], t1b, sab_b);
        }

        bw0[0] = bw0_n0; bw0[1] = bw0_n1;
        bw1[0] = bw1_n0; bw1[1] = bw1_n1;
        const unsigned r0 = a_r32, r1 = 32u + a_r32;
        weave_ep8_lds_st_i64x2(lds_a_nxt + r0 * 128u + (a_c16 ^ (r0 & 7u)) * 16u, a_n0);
        weave_ep8_lds_st_i64x2(lds_a_nxt + r1 * 128u + (a_c16 ^ (r1 & 7u)) * 16u, a_n1);
        if (sa_active) reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + (nxt_kb & 1u) * 512u)[tid] = sa_nxt;
        sb_cur = sb_nxt;
        weave_4wave_lds_barrier_65440(arena);
    }

    {
        const unsigned cur_st = (num_k128 - 1u) & 1u;
        const WeaveLdsU8* lds_a_cur = lds + cur_st * kStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u);
        WeaveAgprF32x4 t0a, t1a, t0b, t1b;
        float sab_a = 0.0f, sab_b = 0.0f;
        #pragma unroll
        for (unsigned m2 = 0u; m2 < 4u; m2 += 2u) {
            {
                const unsigned rm = m2 * 16u + m_lane;
                const WeaveAgprI64x2 av0 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
                const WeaveAgprI64x2 av1 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
                sab_a = lds_sa_cur[rm] * sb_cur;
                weave_mfma_f8_z(t0a, bw0[0][0], av0[0]);
                weave_mfma_f8_z(t1a, bw0[1][0], av0[0]);
                weave_mfma_f8_a(t0a, bw0[0][1], av0[1]);
                weave_mfma_f8_a(t1a, bw0[1][1], av0[1]);
                if (m2 > 0u) {
                    weave_agpr_prom_acc4(acc[m2 - 1u][0], t0b, sab_b);
                    weave_agpr_prom_acc4(acc[m2 - 1u][1], t1b, sab_b);
                }
                weave_mfma_f8_a(t0a, bw1[0][0], av1[0]);
                weave_mfma_f8_a(t1a, bw1[1][0], av1[0]);
                weave_mfma_f8_a(t0a, bw1[0][1], av1[1]);
                weave_mfma_f8_a(t1a, bw1[1][1], av1[1]);
            }
            {
                const unsigned rm = (m2 + 1u) * 16u + m_lane;
                const WeaveAgprI64x2 av0 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
                const WeaveAgprI64x2 av1 = weave_ep8_lds_ld_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
                sab_b = lds_sa_cur[rm] * sb_cur;
                weave_mfma_f8_z(t0b, bw0[0][0], av0[0]);
                weave_mfma_f8_z(t1b, bw0[1][0], av0[0]);
                weave_mfma_f8_a(t0b, bw0[0][1], av0[1]);
                weave_mfma_f8_a(t1b, bw0[1][1], av0[1]);
                weave_agpr_prom_acc4(acc[m2][0], t0a, sab_a);
                weave_agpr_prom_acc4(acc[m2][1], t1a, sab_a);
                weave_mfma_f8_a(t0b, bw1[0][0], av1[0]);
                weave_mfma_f8_a(t1b, bw1[1][0], av1[0]);
                weave_mfma_f8_a(t0b, bw1[0][1], av1[1]);
                weave_mfma_f8_a(t1b, bw1[1][1], av1[1]);
            }
        }
        weave_agpr_prom_acc4(acc[3][0], t0b, sab_b);
        weave_agpr_prom_acc4(acc[3][1], t1b, sab_b);
    }
    weave_4wave_lds_barrier_65440(arena);

    WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
    #pragma unroll
    for (unsigned mx = 0u; mx < 4u; ++mx) {
        const unsigned r = mx * 16u + m_lane;
        const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].lo[0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].lo[1] * weight)) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].hi[0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].hi[1] * weight)) << 16u);
            const unsigned c8 = (col >> 2u) ^ (r & 7u);
            *reinterpret_cast<WeaveLdsU2*>(lds_c + r * 128u + c8 * 4u) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned c_row = tid >> 4u;
    const unsigned c_lane = tid & 15u;
    #pragma clang loop unroll(disable)
    for (unsigned r0 = 0u; r0 < 64u; r0 += 16u) {
        const unsigned r = r0 + c_row;
        const uint32_t tok_m = lds_tok[r];
        if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
            const unsigned c8_0 = (c_lane * 2u + 0u) ^ (r & 7u);
            const unsigned c8_1 = (c_lane * 2u + 1u) ^ (r & 7u);
            const uint2 u0 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_0 * 4u);
            const uint2 u1 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_1 * 4u);
            weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n * 128u + c_lane * 8u,
                            make_uint4(u0.x, u0.y, u1.x, u1.y));
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

static __device__ __forceinline__ void weave_ep8_s2_1x4_bm128(
    const int32_t* __restrict__ s2_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    const float* __restrict__ weights_tile,
    uint16_t* __restrict__ c_out_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kStageBytes = 8192u;  // 128 * 64 B
    constexpr unsigned kSaBaseOff  = 16384u; // 1024 B
    constexpr unsigned kTokOff     = 17408u; // 512 B (total 17,920 B < 32 KB!)

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < 128u) {
        lds_tok[tid] = (tid < valid_rows)
            ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu)
            : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = __builtin_amdgcn_readfirstlane(K >> 7u);
    const unsigned num_k64  = __builtin_amdgcn_readfirstlane(K >> 6u);
    const unsigned n_blocks = __builtin_amdgcn_readfirstlane(N >> 7u);
    const unsigned exp_u = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
    const uint64_t nk_bytes = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(exp_u) * nk_bytes;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(exp_u) * n_blocks + tile_n) * num_k128;
    WeaveAgprConstAs4* bs_s = weave_agpr_make_as4_f32(bs_row);

    const uint32_t a_limit  = __builtin_amdgcn_readfirstlane(route_rows * K);
    const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(route_rows * num_k128 * 4u);
    const uint32_t b_limit  = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(nk_bytes));
    const WeaveEp8ABuf a_rsrc  = weave_ep8_make_abuf(a_base, a_limit);
    const WeaveEp8ABuf sa_rsrc = weave_ep8_make_abuf(a_scale_base, sa_limit);
    const WeaveEp8BBuf b_rsrc  = weave_ep8_make_bbuf(b_exp, b_limit);

    const unsigned a_r64 = tid >> 2u;
    const unsigned a_c16 = tid & 3u;
    uint32_t a_voff[2];
    #pragma unroll
    for (unsigned ap = 0u; ap < 2u; ++ap) {
        const unsigned r = ap * 64u + a_r64;
        const uint32_t tok = lds_tok[r];
        a_voff[ap] = ((r < valid_rows) && (tok < route_rows)) ? (tok * K + a_c16 * 16u) : a_limit;
    }
    const bool sa_active = (tid < 128u);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : route_rows;
    const uint32_t sa_voff = (sa_active && (tid < valid_rows) && (tok_sa < route_rows))
        ? (tok_sa * num_k128 * 4u) : sa_limit;

    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);
    const uint32_t b_voff0 = ((tile_n * 8u + wave_idx * 2u) * num_k64 * 64u + lane) * 16u;

    WeaveAgprAcc4 acc[8][2];
    #pragma unroll
    for (unsigned mx = 0u; mx < 8u; ++mx) {
        weave_agpr_acc_zero(acc[mx][0]); weave_agpr_acc_zero(acc[mx][1]);
    }

    // 2-step (512-cycle) B prefetch + 1-step A-LDS double buffer!
    {
        const WeaveAgprI64x2 a0 = weave_ep8_ld_i64x2(a_rsrc, a_voff[0], 0u);
        const WeaveAgprI64x2 a1 = weave_ep8_ld_i64x2(a_rsrc, a_voff[1], 0u);
        const float sa0 = weave_ep8_ld_f32(sa_rsrc, sa_voff, 0u);
        const unsigned r0 = a_r64, r1 = 64u + a_r64;
        weave_ep8_lds_st_i64x2(lds + r0 * 64u + (a_c16 ^ ((r0 >> 1u) & 3u)) * 16u, a0);
        weave_ep8_lds_st_i64x2(lds + r1 * 64u + (a_c16 ^ ((r1 >> 1u) & 3u)) * 16u, a1);
        if (sa_active) reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff)[tid] = sa0;
    }
    WeaveAgprI64x2 bw_c[2], bw_n[2];
    bw_c[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, 0u);
    bw_c[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, b_nstride);
    bw_n[0] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, 1024u);
    bw_n[1] = weave_ep8_ld_i64x2(b_rsrc, b_voff0, 1024u + b_nstride);
    float sb_cur = bs_s[0];
    weave_4wave_lds_barrier_65440(arena);

    #pragma clang loop unroll(disable)
    for (unsigned k64 = 0u; k64 + 1u < num_k64; ++k64) {
        const unsigned nxt_k64 = k64 + 1u;
        const unsigned nn_k64  = k64 + 2u;
        const bool need_next_scale = ((nxt_k64 & 1u) == 0u);
        const unsigned nxt_k128 = nxt_k64 >> 1u;

        const WeaveLdsU8* lds_a_cur = lds + (k64 & 1u) * kStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(
            lds + kSaBaseOff + ((k64 >> 1u) & 1u) * 512u);
        WeaveLdsU8* lds_a_nxt = lds + (nxt_k64 & 1u) * kStageBytes;

        const WeaveAgprI64x2 a_n0 = weave_ep8_ld_i64x2(a_rsrc, a_voff[0], nxt_k64 * 64u);
        const WeaveAgprI64x2 a_n1 = weave_ep8_ld_i64x2(a_rsrc, a_voff[1], nxt_k64 * 64u);
        WeaveAgprI64x2 bw_nn0 = bw_n[0], bw_nn1 = bw_n[1];
        if (nn_k64 < num_k64) {
            const uint32_t bk2 = nn_k64 * 1024u;
            bw_nn0 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk2);
            bw_nn1 = weave_ep8_ld_i64x2(b_rsrc, b_voff0, bk2 + b_nstride);
        }
        float sa_nxt = 0.0f, sb_nxt = sb_cur;
        if (need_next_scale) {
            sa_nxt = weave_ep8_ld_f32(sa_rsrc, sa_voff, nxt_k128 * 4u);
            sb_nxt = bs_s[nxt_k128];
        }

        {
            WeaveAgprF32x4 t0a, t1a, t0b, t1b;
            float sab_a = 0.0f, sab_b = 0.0f;
            #pragma unroll
            for (unsigned m2 = 0u; m2 < 8u; m2 += 2u) {
                {
                    const unsigned rm = m2 * 16u + m_lane;
                    const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                        lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                    sab_a = lds_sa_cur[rm] * sb_cur;
                    weave_mfma_f8_z(t0a, bw_c[0][0], av[0]);
                    weave_mfma_f8_z(t1a, bw_c[1][0], av[0]);
                    weave_mfma_f8_a(t0a, bw_c[0][1], av[1]);
                    weave_mfma_f8_a(t1a, bw_c[1][1], av[1]);
                    if (m2 > 0u) {
                        weave_agpr_prom_acc4(acc[m2 - 1u][0], t0b, sab_b);
                        weave_agpr_prom_acc4(acc[m2 - 1u][1], t1b, sab_b);
                    }
                }
                {
                    const unsigned rm = (m2 + 1u) * 16u + m_lane;
                    const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                        lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                    sab_b = lds_sa_cur[rm] * sb_cur;
                    weave_mfma_f8_z(t0b, bw_c[0][0], av[0]);
                    weave_mfma_f8_z(t1b, bw_c[1][0], av[0]);
                    weave_mfma_f8_a(t0b, bw_c[0][1], av[1]);
                    weave_mfma_f8_a(t1b, bw_c[1][1], av[1]);
                    weave_agpr_prom_acc4(acc[m2][0], t0a, sab_a);
                    weave_agpr_prom_acc4(acc[m2][1], t1a, sab_a);
                }
            }
            weave_agpr_prom_acc4(acc[7][0], t0b, sab_b);
            weave_agpr_prom_acc4(acc[7][1], t1b, sab_b);
        }

        bw_c[0] = bw_n[0]; bw_c[1] = bw_n[1];
        bw_n[0] = bw_nn0;  bw_n[1] = bw_nn1;
        const unsigned r0 = a_r64, r1 = 64u + a_r64;
        weave_ep8_lds_st_i64x2(lds_a_nxt + r0 * 64u + (a_c16 ^ ((r0 >> 1u) & 3u)) * 16u, a_n0);
        weave_ep8_lds_st_i64x2(lds_a_nxt + r1 * 64u + (a_c16 ^ ((r1 >> 1u) & 3u)) * 16u, a_n1);
        if (need_next_scale) {
            if (sa_active) {
                reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + (nxt_k128 & 1u) * 512u)[tid] = sa_nxt;
            }
            sb_cur = sb_nxt;
        }
        weave_4wave_lds_barrier_65440(arena);
    }

    {
        const unsigned last_k64 = num_k64 - 1u;
        const WeaveLdsU8* lds_a_cur = lds + (last_k64 & 1u) * kStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(
            lds + kSaBaseOff + ((last_k64 >> 1u) & 1u) * 512u);
        WeaveAgprF32x4 t0a, t1a, t0b, t1b;
        float sab_a = 0.0f, sab_b = 0.0f;
        #pragma unroll
        for (unsigned m2 = 0u; m2 < 8u; m2 += 2u) {
            {
                const unsigned rm = m2 * 16u + m_lane;
                const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                    lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                sab_a = lds_sa_cur[rm] * sb_cur;
                weave_mfma_f8_z(t0a, bw_c[0][0], av[0]);
                weave_mfma_f8_z(t1a, bw_c[1][0], av[0]);
                weave_mfma_f8_a(t0a, bw_c[0][1], av[1]);
                weave_mfma_f8_a(t1a, bw_c[1][1], av[1]);
                if (m2 > 0u) {
                    weave_agpr_prom_acc4(acc[m2 - 1u][0], t0b, sab_b);
                    weave_agpr_prom_acc4(acc[m2 - 1u][1], t1b, sab_b);
                }
            }
            {
                const unsigned rm = (m2 + 1u) * 16u + m_lane;
                const WeaveAgprI64x2 av = weave_ep8_lds_ld_i64x2(
                    lds_a_cur + rm * 64u + (k_grp ^ ((rm >> 1u) & 3u)) * 16u);
                sab_b = lds_sa_cur[rm] * sb_cur;
                weave_mfma_f8_z(t0b, bw_c[0][0], av[0]);
                weave_mfma_f8_z(t1b, bw_c[1][0], av[0]);
                weave_mfma_f8_a(t0b, bw_c[0][1], av[1]);
                weave_mfma_f8_a(t1b, bw_c[1][1], av[1]);
                weave_agpr_prom_acc4(acc[m2][0], t0a, sab_a);
                weave_agpr_prom_acc4(acc[m2][1], t1a, sab_a);
            }
        }
        weave_agpr_prom_acc4(acc[7][0], t0b, sab_b);
        weave_agpr_prom_acc4(acc[7][1], t1b, sab_b);
    }
    weave_4wave_lds_barrier_65440(arena);

    WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
    const unsigned c_row = tid >> 4u;
    const unsigned c_lane = tid & 15u;
    auto run_ep_s2 = [&]<unsigned kEp>() {
        #pragma unroll
        for (unsigned m_i = 0u; m_i < 4u; ++m_i) {
            constexpr unsigned kMxBase = kEp * 4u;
            const unsigned mx = kMxBase + m_i;
            const unsigned lds_r = m_i * 16u + m_lane;
            const unsigned r = kEp * 64u + lds_r;
            const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
            #pragma unroll
            for (unsigned nx = 0u; nx < 2u; ++nx) {
                const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
                const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].lo[0] * weight)) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].lo[1] * weight)) << 16u);
                const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].hi[0] * weight)) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].hi[1] * weight)) << 16u);
                const unsigned c8 = (col >> 2u) ^ (lds_r & 7u);
                *reinterpret_cast<WeaveLdsU2*>(lds_c + lds_r * 128u + c8 * 4u) = make_uint2(p0, p1);
            }
        }
        weave_4wave_lds_barrier_65440(arena);

        #pragma clang loop unroll(disable)
        for (unsigned r0 = 0u; r0 < 64u; r0 += 16u) {
            const unsigned lds_r = r0 + c_row;
            const unsigned r = kEp * 64u + lds_r;
            const uint32_t tok_m = lds_tok[r];
            if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
                const unsigned c8_0 = (c_lane * 2u + 0u) ^ (lds_r & 7u);
                const unsigned c8_1 = (c_lane * 2u + 1u) ^ (lds_r & 7u);
                const uint2 u0 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + lds_r * 128u + c8_0 * 4u);
                const uint2 u1 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + lds_r * 128u + c8_1 * 4u);
                weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n * 128u + c_lane * 8u,
                                make_uint4(u0.x, u0.y, u1.x, u1.y));
            }
        }
        weave_4wave_lds_barrier_65440(arena);
    };
    run_ep_s2.template operator()<0u>();
    run_ep_s2.template operator()<1u>();
    asm volatile("" ::: "memory");
}

// Unified Stage-1 & Stage-2 Dispatchers for EP8 Champion (`kBM in {16, 32, 64, 128}`)
template <unsigned kBM>
static __device__ __forceinline__ void weave_ep8_dispatch_stage1_agpr(
    const int32_t* __restrict__ s_ids,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    uint8_t* __restrict__ quant_out_base,
    float* __restrict__ quant_scale_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned num_tokens,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    if constexpr (kBM == 16u) {
        weave_dispatch_stage1_bk64<16u>(
            s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            nullptr, quant_out_base, quant_scale_base, row_offset, valid_rows, tile_n,
            num_tokens, route_rows, N, K, arena);
    } else if constexpr (kBM == 32u) {
        weave_ep8_s1_small_deep<32u>(
            s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            quant_out_base, quant_scale_base, row_offset, valid_rows, tile_n,
            num_tokens, route_rows, N, K, arena);
    } else if constexpr (kBM == 64u) {
        weave_ep8_s1_1x4_bm64(
            s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            quant_out_base, quant_scale_base, row_offset, valid_rows, tile_n,
            num_tokens, route_rows, N, K, arena);
    } else {
        static_assert(kBM == 128u);
        const unsigned v0 = (valid_rows < 64u) ? valid_rows : 64u;
        weave_ep8_s1_1x4_bm64(
            s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            quant_out_base, quant_scale_base, row_offset, v0, tile_n,
            num_tokens, route_rows, N, K, arena);
        if (valid_rows > 64u) {
            const unsigned rem = valid_rows - 64u;
            const int32_t* ids1 = s_ids + 64u;
            const unsigned rs1 = row_offset + 64u;
            if (rem > 32u) {
                weave_ep8_s1_1x4_bm64(
                    ids1, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                    quant_out_base, quant_scale_base, rs1, rem, tile_n,
                    num_tokens, route_rows, N, K, arena);
            } else if (rem > 16u) {
                weave_ep8_s1_small_deep<32u>(
                    ids1, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                    quant_out_base, quant_scale_base, rs1, rem, tile_n,
                    num_tokens, route_rows, N, K, arena);
            } else {
                weave_dispatch_stage1_bk64<16u>(
                    ids1, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                    nullptr, quant_out_base, quant_scale_base, rs1, rem, tile_n,
                    num_tokens, route_rows, N, K, arena);
            }
        }
    }
}

template <unsigned kBM>
static __device__ __forceinline__ void weave_ep8_dispatch_stage2_agpr(
    const int32_t* __restrict__ s2_ids,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    const float* __restrict__ weights,
    uint16_t* __restrict__ c_out_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    if constexpr (kBM == 16u) {
        weave_dispatch_stage2_bk64<16u>(
            s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            weights, c_out_base, row_offset, valid_rows, tile_n, route_rows, N, K, arena);
    } else if constexpr (kBM == 32u) {
        weave_ep8_s2_small_deep<32u>(
            s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            weights, c_out_base, row_offset, valid_rows, tile_n, route_rows, N, K, arena);
    } else if constexpr (kBM == 64u) {
        weave_ep8_s2_1x4_bm64(
            s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            weights, c_out_base, row_offset, valid_rows, tile_n, route_rows, N, K, arena);
    } else {
        static_assert(kBM == 128u);
        const unsigned v0 = (valid_rows < 64u) ? valid_rows : 64u;
        weave_ep8_s2_1x4_bm64(
            s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            weights, c_out_base, row_offset, v0, tile_n, route_rows, N, K, arena);
        if (valid_rows > 64u) {
            const unsigned rem = valid_rows - 64u;
            const int32_t* ids1 = s2_ids + 64u;
            const unsigned rs1 = row_offset + 64u;
            const float* w1 = weights + 64u;
            if (rem > 32u) {
                weave_ep8_s2_1x4_bm64(
                    ids1, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                    w1, c_out_base, rs1, rem, tile_n, route_rows, N, K, arena);
            } else if (rem > 16u) {
                weave_ep8_s2_small_deep<32u>(
                    ids1, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                    w1, c_out_base, rs1, rem, tile_n, route_rows, N, K, arena);
            } else {
                weave_dispatch_stage2_bk64<16u>(
                    ids1, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                    w1, c_out_base, rs1, rem, tile_n, route_rows, N, K, arena);
            }
        }
    }
}
