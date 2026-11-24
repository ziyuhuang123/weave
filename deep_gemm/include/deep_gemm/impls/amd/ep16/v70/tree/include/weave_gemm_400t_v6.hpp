#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>
#include "ck/ck.hpp"

// ============================================================================
// Weave 400-TFLOPS Zero-Spill CDNA3 (gfx942) 2x2 Wave-Local GEMM1 & GEMM2 (v6)
// Upgrades over v5 (71fab27b):
//   1. Unified b_rsrc for Gate+Up (b_up_soff = nk_bytes) -> saves 4 SGPRs,
//      eliminating SGPR-to-VGPR spill cascade in S1 BM=128!
//   2. S1 BM=128 uses 16-VGPR alternating tA/tB (instead of 32-VGPR tg/tu/qg/qu)
//      + kNoCseLds barrier -> ZERO VGPR spills & ZERO scratch across ALL 4 kernels!
//   3. sc1 global stores on S1 quantized output & scale (keeps L1/L2 clean,
//      coherently published to 256MB MALL for GEMM2!).
// ============================================================================

using WeaveI32x4 = int32_t __attribute__((ext_vector_type(4)));
using WeaveF32x2 = float   __attribute__((ext_vector_type(2)));
using WeaveConstF32As4 = const float __attribute__((address_space(4)));

__device__ WeaveI32x4 weave_raw_buf_load_i32x4(
    WeaveI32x4 srsrc, int32_t voffset, int32_t soffset, int32_t glc_slc)
    __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device__ float weave_raw_buf_load_f32(
    WeaveI32x4 srsrc, int32_t voffset, int32_t soffset, int32_t glc_slc)
    __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ __forceinline__ WeaveI32x4 weave_make_buf_rsrc(const void* ptr, uint32_t num_bytes)
{
    union {
        struct {
            const void* p;
            uint32_t range;
            uint32_t cfg;
        } f;
        WeaveI32x4 v;
    } u;
    u.f.p = ptr;
    u.f.range = num_bytes;
    u.f.cfg = static_cast<uint32_t>(CK_BUFFER_RESOURCE_3RD_DWORD);
    WeaveI32x4 out;
    out[0] = __builtin_amdgcn_readfirstlane(u.v[0]);
    out[1] = __builtin_amdgcn_readfirstlane(u.v[1]);
    out[2] = __builtin_amdgcn_readfirstlane(u.v[2]);
    out[3] = __builtin_amdgcn_readfirstlane(u.v[3]);
    return out;
}

__device__ __forceinline__ WeaveConstF32As4* weave_make_as4_f32(const float* p)
{
    const uint64_t addr = reinterpret_cast<uint64_t>(p);
    const uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr));
    const uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr >> 32));
    return reinterpret_cast<WeaveConstF32As4*>(
        static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32));
}

__device__ __forceinline__ WeaveI64x2 weave_buf_load_i64x2(
    WeaveI32x4 rsrc, uint32_t voff, uint32_t soff)
{
    const WeaveI32x4 v = weave_raw_buf_load_i32x4(
        rsrc, static_cast<int32_t>(voff), static_cast<int32_t>(soff), 0);
    return __builtin_bit_cast(WeaveI64x2, v);
}

__device__ __forceinline__ float weave_buf_load_f32(
    WeaveI32x4 rsrc, uint32_t voff, uint32_t soff)
{
    return weave_raw_buf_load_f32(
        rsrc, static_cast<int32_t>(voff), static_cast<int32_t>(soff), 0);
}

// v59 L3 (V54_S1_BYPASS_MALL): GEMM2 A / A-scale come from stage1_out / stage1_scale, which GEMM1 wrote
// write-through to the MALL.  Loading them with the SC1 cache policy makes the read bypass the consuming
// XCD's private 4 MB L2, so no `buffer_inv sc1` (whole-L2 wipe) is needed before GEMM2 and the 4 MB L2 stays
// dedicated to the read-only MoE B weights.  B loads above are deliberately left cached (policy 0).
#ifndef V54_S1_BYPASS_MALL
#define V54_S1_BYPASS_MALL 0
#endif
#ifndef WEAVE_CPOL_SC1
#define WEAVE_CPOL_SC1 16
#endif
__device__ __forceinline__ WeaveI64x2 weave_s2_a_load_i64x2(
    WeaveI32x4 rsrc, uint32_t voff, uint32_t soff)
{
#if V54_S1_BYPASS_MALL
    const WeaveI32x4 v = weave_raw_buf_load_i32x4(
        rsrc, static_cast<int32_t>(voff), static_cast<int32_t>(soff), WEAVE_CPOL_SC1);
    return __builtin_bit_cast(WeaveI64x2, v);
#else
    return weave_buf_load_i64x2(rsrc, voff, soff);
#endif
}
__device__ __forceinline__ float weave_s2_sa_load(
    WeaveI32x4 rsrc, uint32_t voff, uint32_t soff)
{
#if V54_S1_BYPASS_MALL
    return weave_raw_buf_load_f32(
        rsrc, static_cast<int32_t>(voff), static_cast<int32_t>(soff), WEAVE_CPOL_SC1);
#else
    return weave_buf_load_f32(rsrc, voff, soff);
#endif
}

__device__ __forceinline__ void weave_lds_store_i64x2(WeaveLdsU8* p, WeaveI64x2 v)
{
    *reinterpret_cast<WeaveI64x2 __attribute__((address_space(3)))*>(p) = v;
}

struct WeaveAcc4 {
    WeaveF32x2 lo;
    WeaveF32x2 hi;
};

__device__ __forceinline__ void weave_acc_zero(WeaveAcc4& a)
{
    a.lo = {0.0f, 0.0f};
    a.hi = {0.0f, 0.0f};
}

__device__ __forceinline__ void weave_acc_pk_fma4(WeaveAcc4& acc, WeaveFloat4 t, float s)
{
    const WeaveF32x2 s2 = {s, s};
    const WeaveF32x2 t0 = {t[0], t[1]};
    const WeaveF32x2 t1 = {t[2], t[3]};
    acc.lo = __builtin_elementwise_fma(t0, s2, acc.lo);
    acc.hi = __builtin_elementwise_fma(t1, s2, acc.hi);
}

// ----------------------------------------------------------------------------
// Helper A (Zero-Spill 16-temp VGPR version): Gate+Up fused K=64 for 2 M-rows
// Uses ONLY 16 temp MFMA VGPRs (tA0, tA1, tB0, tB1) by folding gate during up
// and folding row0-up during row1-gate (4 MFMAs apart -> 0 wait/nop!).
// ----------------------------------------------------------------------------
template <unsigned kMBase>
__device__ __forceinline__ void weave_400t_s1_mhalf_k64(
    WeaveLdsU8* lds_rd0,
    WeaveLdsU8* lds_wr0,
    const WeaveLdsF32* lds_sa_rd,
    const WeaveI64x2 (&bg)[2],
    const WeaveI64x2 (&bu)[2],
    float sb_g,
    float sb_u,
    const WeaveI64x2 (&a_wr)[2],
    bool do_wr,
    WeaveAcc4 (&gate_acc)[4][2],
    WeaveAcc4 (&up_acc)[4][2])
{
    __builtin_amdgcn_sched_barrier(0);
    WeaveLdsU8* rd_p = lds_rd0;
    const WeaveLdsF32* sa_p = lds_sa_rd;
    asm volatile("" : "+v"(rd_p), "+v"(sa_p));

    const WeaveI64x2 av0 = weave_lds_load_i64x2(rd_p);
    const float sa0 = sa_p[(kMBase + 0u) * 16u];
    const float sab_g0 = sa0 * sb_g;
    const float sab_u0 = sa0 * sb_u;

    // Row 0 Gate (tA0, tA1)
    WeaveFloat4 tA0 = {0, 0, 0, 0}, tA1 = {0, 0, 0, 0};
    WeaveFloat4 tB0 = {0, 0, 0, 0}, tB1 = {0, 0, 0, 0};
    tA0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][0], av0[0], tA0, 0, 0, 0);
    tA1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][0], av0[0], tA1, 0, 0, 0);
    tA0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][1], av0[1], tA0, 0, 0, 0);
    tA1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][1], av0[1], tA1, 0, 0, 0);

    // Row 0 Up (tB0, tB1) while prefetching av1 & folding Row 0 Gate!
    const WeaveI64x2 av1 = weave_lds_load_i64x2(rd_p + 1024u);
    const float sa1 = sa_p[(kMBase + 1u) * 16u];
    const float sab_g1 = sa1 * sb_g;
    const float sab_u1 = sa1 * sb_u;

    tB0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][0], av0[0], tB0, 0, 0, 0);
    tB1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][0], av0[0], tB1, 0, 0, 0);
    weave_acc_pk_fma4(gate_acc[kMBase + 0u][0], tA0, sab_g0);
    weave_acc_pk_fma4(gate_acc[kMBase + 0u][1], tA1, sab_g0);
    tB0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][1], av0[1], tB0, 0, 0, 0);
    tB1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][1], av0[1], tB1, 0, 0, 0);
    if (do_wr) weave_lds_store_i64x2(lds_wr0, a_wr[0]);

    // Row 1 Gate (reuse tA0, tA1) while folding Row 0 Up!
    tA0 = {0, 0, 0, 0}; tA1 = {0, 0, 0, 0};
    tA0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][0], av1[0], tA0, 0, 0, 0);
    tA1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][0], av1[0], tA1, 0, 0, 0);
    weave_acc_pk_fma4(up_acc[kMBase + 0u][0], tB0, sab_u0);
    weave_acc_pk_fma4(up_acc[kMBase + 0u][1], tB1, sab_u0);
    tA0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][1], av1[1], tA0, 0, 0, 0);
    tA1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][1], av1[1], tA1, 0, 0, 0);

    // Row 1 Up (reuse tB0, tB1) while folding Row 1 Gate!
    tB0 = {0, 0, 0, 0}; tB1 = {0, 0, 0, 0};
    tB0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][0], av1[0], tB0, 0, 0, 0);
    tB1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][0], av1[0], tB1, 0, 0, 0);
    weave_acc_pk_fma4(gate_acc[kMBase + 1u][0], tA0, sab_g1);
    weave_acc_pk_fma4(gate_acc[kMBase + 1u][1], tA1, sab_g1);
    tB0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][1], av1[1], tB0, 0, 0, 0);
    tB1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][1], av1[1], tB1, 0, 0, 0);
    if (do_wr) weave_lds_store_i64x2(lds_wr0 + 1024u, a_wr[1]);

    weave_acc_pk_fma4(up_acc[kMBase + 1u][0], tB0, sab_u1);
    weave_acc_pk_fma4(up_acc[kMBase + 1u][1], tB1, sab_u1);
}

// ----------------------------------------------------------------------------
// Helper B: N-pair K=64 micro-kernel for S1 BM=64 and S2 BM=64/128
// ----------------------------------------------------------------------------
template <
    unsigned kMXDL,
    unsigned kNBase,
    bool kNoCseLds,
    unsigned kWrBegin,
    unsigned kWrEnd,
    unsigned kWrLen>
__device__ __forceinline__ void weave_400t_pair_k64(
    WeaveLdsU8* lds_rd0,
    WeaveLdsU8* lds_wr0,
    const WeaveLdsF32* lds_sa_rd,
    const WeaveI64x2& b0,
    const WeaveI64x2& b1,
    float sb,
    const WeaveI64x2 (&a_wr)[kWrLen],
    WeaveAcc4 (&acc)[kMXDL][4])
{
    __builtin_amdgcn_sched_barrier(0);
    WeaveLdsU8* rd_p = lds_rd0;
    const WeaveLdsF32* sa_p = lds_sa_rd;
    if constexpr (kNoCseLds) {
        asm volatile("" : "+v"(rd_p), "+v"(sa_p));
    }

    WeaveI64x2 av_cur = weave_lds_load_i64x2(rd_p);
    float sab_cur = sa_p[0] * sb;

    WeaveFloat4 tA0 = {0, 0, 0, 0}, tA1 = {0, 0, 0, 0};
    WeaveFloat4 tB0 = {0, 0, 0, 0}, tB1 = {0, 0, 0, 0};
    float sab_prev = 0.0f;

    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        WeaveI64x2 av_nxt;
        float sab_nxt = 0.0f;
        if (mx + 1u < kMXDL) {
            av_nxt = weave_lds_load_i64x2(rd_p + (mx + 1u) * 1024u);
            sab_nxt = sa_p[(mx + 1u) * 16u] * sb;
        }
        const int64_t a0 = av_cur[0];
        const int64_t a1 = av_cur[1];

        if ((mx & 1u) == 0u) {
            tA0 = {0, 0, 0, 0}; tA1 = {0, 0, 0, 0};
            tA0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0[0], a0, tA0, 0, 0, 0);
            tA1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1[0], a0, tA1, 0, 0, 0);
            if (mx > 0u) {
                weave_acc_pk_fma4(acc[mx - 1u][kNBase + 0u], tB0, sab_prev);
                weave_acc_pk_fma4(acc[mx - 1u][kNBase + 1u], tB1, sab_prev);
            }
            tA0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0[1], a1, tA0, 0, 0, 0);
            tA1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1[1], a1, tA1, 0, 0, 0);
        } else {
            tB0 = {0, 0, 0, 0}; tB1 = {0, 0, 0, 0};
            tB0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0[0], a0, tB0, 0, 0, 0);
            tB1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1[0], a0, tB1, 0, 0, 0);
            weave_acc_pk_fma4(acc[mx - 1u][kNBase + 0u], tA0, sab_prev);
            weave_acc_pk_fma4(acc[mx - 1u][kNBase + 1u], tA1, sab_prev);
            tB0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0[1], a1, tB0, 0, 0, 0);
            tB1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1[1], a1, tB1, 0, 0, 0);
        }

        if constexpr (kWrBegin < kWrEnd) {
            if (mx >= kWrBegin && mx < kWrEnd) {
                weave_lds_store_i64x2(lds_wr0 + mx * 1024u, a_wr[mx - kWrBegin]);
            }
        }

        sab_prev = sab_cur;
        if (mx + 1u < kMXDL) {
            av_cur = av_nxt;
            sab_cur = sab_nxt;
        }
    }
    weave_acc_pk_fma4(acc[kMXDL - 1u][kNBase + 0u], tB0, sab_prev);
    weave_acc_pk_fma4(acc[kMXDL - 1u][kNBase + 1u], tB1, sab_prev);
}

// ============================================================================
// STAGE 1: BM = 128 (2-Pass 128x64, Unified b_rsrc, 64 acc VGPRs, 25.5KB LDS)
//          AND BM = 64 (1-Pass 64x128, 64 acc VGPRs, 17.5KB LDS)
// ============================================================================
template <unsigned kBM>
__device__ __forceinline__ void weave_s1_400t_2x2(
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
    static_assert(kBM == 64u || kBM == 128u);
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned wm = wave_idx >> 1u;
    const unsigned wn = wave_idx & 1u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;
    const unsigned ld_r = lane >> 2u;
    const unsigned ld_c16 = lane & 3u;

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
    WeaveConstF32As4* bs_gate_s = weave_make_as4_f32(bs_gate_ptr);

    const uint32_t a_limit  = __builtin_amdgcn_readfirstlane(num_tokens * K);
    const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(num_tokens * num_k128 * 4u);
    const uint32_t b_limit  = __builtin_amdgcn_readfirstlane(nk_bytes * 2u);

    const WeaveI32x4 a_rsrc  = weave_make_buf_rsrc(a_base, a_limit);
    const WeaveI32x4 sa_rsrc = weave_make_buf_rsrc(a_scale_base, sa_limit);
    const WeaveI32x4 b_rsrc  = weave_make_buf_rsrc(b_gate_exp, b_limit);
    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);

    if constexpr (kBM == 128u) {
        constexpr unsigned kTokOff   = 8192u;
        constexpr unsigned kSaLdsOff = 8704u;
        constexpr unsigned kHalf0Off = 9728u;

        WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
        if (tid < 128u) {
            lds_tok[tid] = (tid < valid_rows)
                ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu)
                : num_tokens;
        }
        weave_4wave_lds_barrier_65440(arena);

        uint32_t a_voff[4];
        #pragma unroll
        for (unsigned mx = 0u; mx < 4u; ++mx) {
            const unsigned r_ld = wm * 64u + mx * 16u + ld_r;
            const uint32_t tok_ld = lds_tok[r_ld];
            a_voff[mx] = ((r_ld < valid_rows) && (tok_ld < num_tokens))
                ? (tok_ld * K + ld_c16 * 16u) : a_limit;
        }
        const unsigned r_sa = wm * 64u + lane;
        const uint32_t tok_sa = lds_tok[r_sa];
        const uint32_t sa_voff = ((r_sa < valid_rows) && (tok_sa < num_tokens))
            ? (tok_sa * num_k128 * 4u) : sa_limit;

        WeaveLdsU8* const lds_wr0 = lds + wave_idx * 2048u
            + ld_r * 64u + (ld_c16 ^ ((ld_r >> 1u) & 3u)) * 16u;
        WeaveLdsU8* const lds_rd0 = lds + wave_idx * 2048u
            + m_lane * 64u + (k_grp ^ ((m_lane >> 1u) & 3u)) * 16u;
        WeaveLdsF32* const lds_sa_wr = reinterpret_cast<WeaveLdsF32*>(lds + kSaLdsOff + wave_idx * 256u) + lane;
        const WeaveLdsF32* const lds_sa_rd = reinterpret_cast<const WeaveLdsF32*>(lds + kSaLdsOff + wave_idx * 256u) + m_lane;
        WeaveLdsU16* const lds_half0 = reinterpret_cast<WeaveLdsU16*>(lds + kHalf0Off);

        #pragma clang loop unroll(disable)
        for (unsigned n_half = 0u; n_half < 2u; ++n_half) {
            const uint32_t b_voff0 = ((tile_n * 8u + n_half * 4u + wn * 2u) * num_k64 * 64u + lane) * 16u;

            WeaveAcc4 gate_acc[4][2];
            WeaveAcc4 up_acc[4][2];
            #pragma unroll
            for (unsigned mx = 0u; mx < 4u; ++mx) {
                weave_acc_zero(gate_acc[mx][0]); weave_acc_zero(gate_acc[mx][1]);
                weave_acc_zero(up_acc[mx][0]);   weave_acc_zero(up_acc[mx][1]);
            }

            // opus55 S1_128 deep prefetch (2-deep register rings, +24 VGPRs in this loop only):
            //   A staging : aX = r23 rows (written in m_half 0), aY = r01 rows (written in m_half 1)
            //   B weights : P = kh0 set (bgP/buP), Q = kh1 set (bgQ/buQ), each reloaded in place
            //               right after its last use, with the chunk needed 2 kh later.
            // Every A/B chunk is still loaded exactly once from the identical (rsrc, voffset, soffset),
            // every LDS write stores the identical data at the identical program point, and every
            // accumulator receives the identical MFMA/pk_fma sequence -> bit-identical results.
            // Only the issue point of each buffer_load moves earlier: A ~1 quarter (16 MFMAs),
            // B ~2.5 quarters (40 MFMAs) -> removes the 4 in-loop s_waitcnt vmcnt(0).
            WeaveI64x2 bgP[2], buP[2], bgQ[2], buQ[2];
            WeaveI64x2 aX[2], aY[2];

            {
                const WeaveI64x2 am0_0 = weave_buf_load_i64x2(a_rsrc, a_voff[0], 0u);
                const WeaveI64x2 am0_1 = weave_buf_load_i64x2(a_rsrc, a_voff[1], 0u);
                aX[0] = weave_buf_load_i64x2(a_rsrc, a_voff[2], 0u);
                aX[1] = weave_buf_load_i64x2(a_rsrc, a_voff[3], 0u);
                const float sa0 = weave_buf_load_f32(sa_rsrc, sa_voff, 0u);
                bgP[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, 0u);
                bgP[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_nstride);
                buP[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes);
                buP[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_nstride);
                aY[0] = weave_buf_load_i64x2(a_rsrc, a_voff[0], 64u);
                aY[1] = weave_buf_load_i64x2(a_rsrc, a_voff[1], 64u);
                bgQ[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, 1024u);
                bgQ[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, 1024u + b_nstride);
                buQ[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + 1024u);
                buQ[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + 1024u + b_nstride);
                weave_lds_store_i64x2(lds_wr0,         am0_0);
                weave_lds_store_i64x2(lds_wr0 + 1024u, am0_1);
                *lds_sa_wr = sa0;
                __builtin_amdgcn_s_waitcnt(0xc07f);
            }
            float sb_g_cur = bs_gate_s[0];
            float sb_u_cur = bs_gate_s[bs_up_stride];

            // m_half = 1 (rows 2,3) of one kh with B set (bg, bu).  do_wr: write a_src (the r01 rows of
            // the following kh) into this wave's LDS staging slots at the SAME two points as before;
            // load_next_b: reload the SAME B set in place (right after its last use) with chunk next_b_soff.
            auto s1_m1_and_reload_b = [&](WeaveI64x2 (&bg)[2], WeaveI64x2 (&bu)[2],
                                          const WeaveI64x2 (&a_src)[2],
                                          uint32_t next_b_soff, bool do_wr, bool load_next_b) {
                WeaveLdsU8* rd_p = lds_rd0;
                const WeaveLdsF32* sa_p = lds_sa_rd;
                asm volatile("" : "+v"(rd_p), "+v"(sa_p));
                const WeaveI64x2 av0 = weave_lds_load_i64x2(rd_p);
                const float sa0 = sa_p[2u * 16u];
                const float sab_g0 = sa0 * sb_g_cur;
                const float sab_u0 = sa0 * sb_u_cur;

                WeaveFloat4 tA0 = {0, 0, 0, 0}, tA1 = {0, 0, 0, 0};
                WeaveFloat4 tB0 = {0, 0, 0, 0}, tB1 = {0, 0, 0, 0};
                tA0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][0], av0[0], tA0, 0, 0, 0);
                tA1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][0], av0[0], tA1, 0, 0, 0);
                tA0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][1], av0[1], tA0, 0, 0, 0);
                tA1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][1], av0[1], tA1, 0, 0, 0);

                const WeaveI64x2 av1 = weave_lds_load_i64x2(rd_p + 1024u);
                const float sa1 = sa_p[3u * 16u];
                const float sab_g1 = sa1 * sb_g_cur;
                const float sab_u1 = sa1 * sb_u_cur;

                tB0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][0], av0[0], tB0, 0, 0, 0);
                tB1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][0], av0[0], tB1, 0, 0, 0);
                weave_acc_pk_fma4(gate_acc[2][0], tA0, sab_g0);
                weave_acc_pk_fma4(gate_acc[2][1], tA1, sab_g0);
                tB0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][1], av0[1], tB0, 0, 0, 0);
                tB1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][1], av0[1], tB1, 0, 0, 0);
                if (do_wr) weave_lds_store_i64x2(lds_wr0, a_src[0]);

                tA0 = {0, 0, 0, 0}; tA1 = {0, 0, 0, 0};
                tA0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][0], av1[0], tA0, 0, 0, 0);
                tA1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][0], av1[0], tA1, 0, 0, 0);
                weave_acc_pk_fma4(up_acc[2][0], tB0, sab_u0);
                weave_acc_pk_fma4(up_acc[2][1], tB1, sab_u0);
                tA0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][1], av1[1], tA0, 0, 0, 0);
                tA1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][1], av1[1], tA1, 0, 0, 0);
                // bg is now DEAD -> reload next bg in-place while Row 1 Up runs!
                if (load_next_b) {
                    bg[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, next_b_soff);
                    bg[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, next_b_soff + b_nstride);
                }

                tB0 = {0, 0, 0, 0}; tB1 = {0, 0, 0, 0};
                tB0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][0], av1[0], tB0, 0, 0, 0);
                tB1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][0], av1[0], tB1, 0, 0, 0);
                weave_acc_pk_fma4(gate_acc[3][0], tA0, sab_g1);
                weave_acc_pk_fma4(gate_acc[3][1], tA1, sab_g1);
                tB0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][1], av1[1], tB0, 0, 0, 0);
                tB1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][1], av1[1], tB1, 0, 0, 0);
                if (do_wr) weave_lds_store_i64x2(lds_wr0 + 1024u, a_src[1]);
                if (load_next_b) {
                    bu[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + next_b_soff);
                    bu[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + next_b_soff + b_nstride);
                }
                weave_acc_pk_fma4(up_acc[3][0], tB0, sab_u1);
                weave_acc_pk_fma4(up_acc[3][1], tB1, sab_u1);
            };

            uint32_t a_soff = 0u, b_soff = 0u, sa_soff = 0u;

            #pragma clang loop unroll(disable)
            for (unsigned kb = 0u; kb + 1u < num_k128; ++kb) {
                const uint32_t nxt_kb = kb + 1u;

                // Q0 (kh0, m_half 0): B=P(kb,kh0); write aX=A(r23,kb,kh0); refill aX <- A(r23,kb,kh1)
                weave_400t_s1_mhalf_k64<0u>(
                    lds_rd0, lds_wr0, lds_sa_rd, bgP, buP, sb_g_cur, sb_u_cur, aX, true, gate_acc, up_acc);
                aX[0] = weave_buf_load_i64x2(a_rsrc, a_voff[2], a_soff + 64u);
                aX[1] = weave_buf_load_i64x2(a_rsrc, a_voff[3], a_soff + 64u);
                const float sa_nxt = weave_buf_load_f32(sa_rsrc, sa_voff, sa_soff + 4u);
                const float sb_g_nxt = bs_gate_s[nxt_kb];
                const float sb_u_nxt = bs_gate_s[bs_up_stride + nxt_kb];
                asm volatile("" ::: "memory");

                // Q1 (kh0, m_half 1): B=P; write aY=A(r01,kb,kh1); P <- B(kb+1,kh0) in place;
                //                     refill aY <- A(r01,kb+1,kh0)
                s1_m1_and_reload_b(bgP, buP, aY, b_soff + 2048u, true, true);
                aY[0] = weave_buf_load_i64x2(a_rsrc, a_voff[0], a_soff + 128u);
                aY[1] = weave_buf_load_i64x2(a_rsrc, a_voff[1], a_soff + 128u);
                asm volatile("" ::: "memory");

                // Q2 (kh1, m_half 0): B=Q(kb,kh1); write aX=A(r23,kb,kh1); refill aX <- A(r23,kb+1,kh0)
                weave_400t_s1_mhalf_k64<0u>(
                    lds_rd0, lds_wr0, lds_sa_rd, bgQ, buQ, sb_g_cur, sb_u_cur, aX, true, gate_acc, up_acc);
                aX[0] = weave_buf_load_i64x2(a_rsrc, a_voff[2], a_soff + 128u);
                aX[1] = weave_buf_load_i64x2(a_rsrc, a_voff[3], a_soff + 128u);
                asm volatile("" ::: "memory");

                // Q3 (kh1, m_half 1): B=Q; write aY=A(r01,kb+1,kh0); Q <- B(kb+1,kh1) in place;
                //                     refill aY <- A(r01,kb+1,kh1)
                s1_m1_and_reload_b(bgQ, buQ, aY, b_soff + 3072u, true, true);
                aY[0] = weave_buf_load_i64x2(a_rsrc, a_voff[0], a_soff + 192u);
                aY[1] = weave_buf_load_i64x2(a_rsrc, a_voff[1], a_soff + 192u);

                a_soff  += 128u;
                b_soff  += 2048u;
                sa_soff += 4u;
                *lds_sa_wr = sa_nxt;
                sb_g_cur = sb_g_nxt;
                sb_u_cur = sb_u_nxt;
                asm volatile("" ::: "memory");
            }

            {
                // last kb: P = B(kh0) and Q = B(kh1) are already resident -> no B reloads;
                // only the A(r23,kh1) refill is still needed (A(r01,kh1) already sits in aY).
                weave_400t_s1_mhalf_k64<0u>(
                    lds_rd0, lds_wr0, lds_sa_rd, bgP, buP, sb_g_cur, sb_u_cur, aX, true, gate_acc, up_acc);
                aX[0] = weave_buf_load_i64x2(a_rsrc, a_voff[2], a_soff + 64u);
                aX[1] = weave_buf_load_i64x2(a_rsrc, a_voff[3], a_soff + 64u);
                asm volatile("" ::: "memory");

                s1_m1_and_reload_b(bgP, buP, aY, 0u, true, false);
                asm volatile("" ::: "memory");

                weave_400t_s1_mhalf_k64<0u>(
                    lds_rd0, lds_wr0, lds_sa_rd, bgQ, buQ, sb_g_cur, sb_u_cur, aX, true, gate_acc, up_acc);
                asm volatile("" ::: "memory");

                s1_m1_and_reload_b(bgQ, buQ, aY, 0u, false, false);
            }

            if (n_half == 0u) {
                #pragma unroll
                for (unsigned mx = 0u; mx < 4u; ++mx) {
                    const unsigned r = wm * 64u + mx * 16u + m_lane;
                    #pragma unroll
                    for (unsigned nx = 0u; nx < 2u; ++nx) {
                        const unsigned col = (wn * 2u + nx) * 16u + k_grp * 4u;
                        const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                            weave_bk64_silu(gate_acc[mx][nx].lo[0]) * up_acc[mx][nx].lo[0])) |
                            (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                            weave_bk64_silu(gate_acc[mx][nx].lo[1]) * up_acc[mx][nx].lo[1])) << 16u);
                        const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                            weave_bk64_silu(gate_acc[mx][nx].hi[0]) * up_acc[mx][nx].hi[0])) |
                            (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                            weave_bk64_silu(gate_acc[mx][nx].hi[1]) * up_acc[mx][nx].hi[1])) << 16u);
                        *reinterpret_cast<WeaveLdsU2*>(lds_half0 + r * 64u + col) = make_uint2(p0, p1);
                    }
                }
                weave_4wave_lds_barrier_65440(arena);
            } else {
                weave_4wave_lds_barrier_65440(arena);
                WeaveLdsU16* const lds_half1 = reinterpret_cast<WeaveLdsU16*>(lds);
                const unsigned q_row = tid >> 4u;
                const unsigned q_lane = tid & 15u;

                #pragma unroll
                for (unsigned ep = 0u; ep < 2u; ++ep) {
                    const unsigned mx_base = ep * 2u;
                    #pragma unroll
                    for (unsigned m_i = 0u; m_i < 2u; ++m_i) {
                        const unsigned mx = mx_base + m_i;
                        const unsigned lds_r = wm * 32u + m_i * 16u + m_lane;
                        #pragma unroll
                        for (unsigned nx = 0u; nx < 2u; ++nx) {
                            const unsigned col = (wn * 2u + nx) * 16u + k_grp * 4u;
                            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                                weave_bk64_silu(gate_acc[mx][nx].lo[0]) * up_acc[mx][nx].lo[0])) |
                                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                                weave_bk64_silu(gate_acc[mx][nx].lo[1]) * up_acc[mx][nx].lo[1])) << 16u);
                            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                                weave_bk64_silu(gate_acc[mx][nx].hi[0]) * up_acc[mx][nx].hi[0])) |
                                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                                weave_bk64_silu(gate_acc[mx][nx].hi[1]) * up_acc[mx][nx].hi[1])) << 16u);
                            *reinterpret_cast<WeaveLdsU2*>(lds_half1 + lds_r * 64u + col) = make_uint2(p0, p1);
                        }
                    }
                    weave_4wave_lds_barrier_65440(arena);

                    #pragma clang loop unroll(disable)
                    for (unsigned r0_i = 0u; r0_i < 64u; r0_i += 16u) {
                        const unsigned lds_r = r0_i + q_row;
                        const unsigned r = (lds_r < 32u) ? (ep * 32u + lds_r) : (64u + ep * 32u + (lds_r - 32u));
                        const unsigned dense_row = row_offset + r;
                        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
                        uint4 raw = make_uint4(0, 0, 0, 0);
                        if (valid) {
                            const WeaveLdsU16* src_p = (q_lane < 8u)
                                ? (lds_half0 + r * 64u + q_lane * 8u)
                                : (lds_half1 + lds_r * 64u + (q_lane - 8u) * 8u);
                            raw = *reinterpret_cast<const WeaveLdsU4*>(src_p);
                        }
                        weave_quant_bf16x8_store(
                            raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n,
                            quant_out_base, quant_scale_base);
                    }
                    weave_4wave_lds_barrier_65440(arena);
                }
            }
        }
        asm volatile("" ::: "memory");
        return;
    }

    // BM=64 1-PASS (64x128, 2x2 Waves, Unified b_rsrc, 64 acc VGPRs, 17.5KB LDS, ZERO SPILL!)
    constexpr unsigned kTokOff   = 16384u;
    constexpr unsigned kSaLdsOff = 16896u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < 64u) {
        lds_tok[tid] = (tid < valid_rows)
            ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu)
            : num_tokens;
    }
    weave_4wave_lds_barrier_65440(arena);

    uint32_t a_voff[2];
    #pragma unroll
    for (unsigned mx = 0u; mx < 2u; ++mx) {
        const unsigned r_ld = wm * 32u + mx * 16u + ld_r;
        const uint32_t tok_ld = lds_tok[r_ld];
        a_voff[mx] = ((r_ld < valid_rows) && (tok_ld < num_tokens))
            ? (tok_ld * K + ld_c16 * 16u) : a_limit;
    }
    const unsigned r_sa = wm * 32u + lane;
    const bool sa_in_m = (lane < 32u);
    const uint32_t tok_sa = sa_in_m ? lds_tok[r_sa] : num_tokens;
    const uint32_t sa_voff = (sa_in_m && (r_sa < valid_rows) && (tok_sa < num_tokens))
        ? (tok_sa * num_k128 * 4u) : sa_limit;

    const uint32_t b_voff0 = ((tile_n * 8u + wn * 4u) * num_k64 * 64u + lane) * 16u;

    WeaveLdsU8* const lds_wr0 = lds + wave_idx * 4096u
        + ld_r * 64u + (ld_c16 ^ ((ld_r >> 1u) & 3u)) * 16u;
    WeaveLdsU8* const lds_rd0 = lds + wave_idx * 4096u
        + m_lane * 64u + (k_grp ^ ((m_lane >> 1u) & 3u)) * 16u;
    WeaveLdsF32* const lds_sa_wr = reinterpret_cast<WeaveLdsF32*>(lds + kSaLdsOff + wave_idx * 256u) + lane;
    const WeaveLdsF32* const lds_sa_rd = reinterpret_cast<const WeaveLdsF32*>(lds + kSaLdsOff + wave_idx * 256u) + m_lane;

    WeaveAcc4 gate_acc[2][4];
    WeaveAcc4 up_acc[2][4];
    #pragma unroll
    for (unsigned mx = 0u; mx < 2u; ++mx) {
        #pragma unroll
        for (unsigned nx = 0u; nx < 4u; ++nx) {
            weave_acc_zero(gate_acc[mx][nx]);
            weave_acc_zero(up_acc[mx][nx]);
        }
    }

    WeaveI64x2 bp0[2], bp1[2], a_pair[2], dummy_a[2];
    a_pair[0] = weave_buf_load_i64x2(a_rsrc, a_voff[0], 0u);
    a_pair[1] = weave_buf_load_i64x2(a_rsrc, a_voff[1], 0u);
    const float sa0 = weave_buf_load_f32(sa_rsrc, sa_voff, 0u);
    bp0[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, 0u);
    bp0[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_nstride);
    float sb_g_cur = bs_gate_s[0];
    float sb_u_cur = bs_gate_s[bs_up_stride];
    weave_lds_store_i64x2(lds_wr0,         a_pair[0]);
    weave_lds_store_i64x2(lds_wr0 + 1024u, a_pair[1]);
    *lds_sa_wr = sa0;
    __builtin_amdgcn_s_waitcnt(0xc07f);

    uint32_t a_soff = 0u, b_soff = 0u, sa_soff = 0u;
    #pragma clang loop unroll(disable)
    for (unsigned kb = 0u; kb + 1u < num_k128; ++kb) {
        const uint32_t nxt_kb = kb + 1u;
        const uint32_t b_soff_h1 = b_soff + 1024u;

        bp1[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff + b_nstride * 2u);
        bp1[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff + b_nstride * 3u);
        a_pair[0] = weave_buf_load_i64x2(a_rsrc, a_voff[0], a_soff + 64u);
        a_pair[1] = weave_buf_load_i64x2(a_rsrc, a_voff[1], a_soff + 64u);
        const float sa_nxt = weave_buf_load_f32(sa_rsrc, sa_voff, sa_soff + 4u);
        const float sb_g_nxt = bs_gate_s[nxt_kb];
        const float sb_u_nxt = bs_gate_s[bs_up_stride + nxt_kb];
        weave_400t_pair_k64<2u, 0u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp0[0], bp0[1], sb_g_cur, dummy_a, gate_acc);

        bp0[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff);
        bp0[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff + b_nstride);
        weave_400t_pair_k64<2u, 2u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp1[0], bp1[1], sb_g_cur, dummy_a, gate_acc);

        bp1[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff + b_nstride * 2u);
        bp1[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff + b_nstride * 3u);
        weave_400t_pair_k64<2u, 0u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp0[0], bp0[1], sb_u_cur, dummy_a, up_acc);

        bp0[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1);
        bp0[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride);
        weave_400t_pair_k64<2u, 2u, false, 0u, 2u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp1[0], bp1[1], sb_u_cur, a_pair, up_acc);
        asm volatile("" ::: "memory");

        bp1[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride * 2u);
        bp1[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride * 3u);
        a_pair[0] = weave_buf_load_i64x2(a_rsrc, a_voff[0], a_soff + 128u);
        a_pair[1] = weave_buf_load_i64x2(a_rsrc, a_voff[1], a_soff + 128u);
        weave_400t_pair_k64<2u, 0u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp0[0], bp0[1], sb_g_cur, dummy_a, gate_acc);

        bp0[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff_h1);
        bp0[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff_h1 + b_nstride);
        weave_400t_pair_k64<2u, 2u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp1[0], bp1[1], sb_g_cur, dummy_a, gate_acc);

        bp1[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff_h1 + b_nstride * 2u);
        bp1[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff_h1 + b_nstride * 3u);
        weave_400t_pair_k64<2u, 0u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp0[0], bp0[1], sb_u_cur, dummy_a, up_acc);

        a_soff  += 128u;
        b_soff  += 2048u;
        sa_soff += 4u;

        bp0[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff);
        bp0[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff + b_nstride);
        weave_400t_pair_k64<2u, 2u, false, 0u, 2u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp1[0], bp1[1], sb_u_cur, a_pair, up_acc);
        *lds_sa_wr = sa_nxt;
        sb_g_cur = sb_g_nxt;
        sb_u_cur = sb_u_nxt;
        asm volatile("" ::: "memory");
    }
    {
        const uint32_t b_soff_h1 = b_soff + 1024u;
        bp1[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff + b_nstride * 2u);
        bp1[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff + b_nstride * 3u);
        a_pair[0] = weave_buf_load_i64x2(a_rsrc, a_voff[0], a_soff + 64u);
        a_pair[1] = weave_buf_load_i64x2(a_rsrc, a_voff[1], a_soff + 64u);
        weave_400t_pair_k64<2u, 0u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp0[0], bp0[1], sb_g_cur, dummy_a, gate_acc);

        bp0[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff);
        bp0[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff + b_nstride);
        weave_400t_pair_k64<2u, 2u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp1[0], bp1[1], sb_g_cur, dummy_a, gate_acc);

        bp1[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff + b_nstride * 2u);
        bp1[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff + b_nstride * 3u);
        weave_400t_pair_k64<2u, 0u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp0[0], bp0[1], sb_u_cur, dummy_a, up_acc);

        bp0[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1);
        bp0[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride);
        weave_400t_pair_k64<2u, 2u, false, 0u, 2u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp1[0], bp1[1], sb_u_cur, a_pair, up_acc);
        asm volatile("" ::: "memory");

        bp1[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride * 2u);
        bp1[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride * 3u);
        weave_400t_pair_k64<2u, 0u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp0[0], bp0[1], sb_g_cur, dummy_a, gate_acc);

        bp0[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff_h1);
        bp0[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff_h1 + b_nstride);
        weave_400t_pair_k64<2u, 2u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp1[0], bp1[1], sb_g_cur, dummy_a, gate_acc);

        bp1[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff_h1 + b_nstride * 2u);
        bp1[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, nk_bytes + b_soff_h1 + b_nstride * 3u);
        weave_400t_pair_k64<2u, 0u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp0[0], bp0[1], sb_u_cur, dummy_a, up_acc);
        weave_400t_pair_k64<2u, 2u, false, 0u, 0u, 2u>(lds_rd0, lds_wr0, lds_sa_rd, bp1[0], bp1[1], sb_u_cur, dummy_a, up_acc);
    }

    weave_4wave_lds_barrier_65440(arena);
    WeaveLdsU16* lds_q = reinterpret_cast<WeaveLdsU16*>(lds);
    #pragma unroll
    for (unsigned mx = 0u; mx < 2u; ++mx) {
        const unsigned r = wm * 32u + mx * 16u + m_lane;
        #pragma unroll
        for (unsigned nx = 0u; nx < 4u; ++nx) {
            const unsigned col = (wn * 4u + nx) * 16u + k_grp * 4u;
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
    for (unsigned r0_i = 0u; r0_i < 64u; r0_i += 16u) {
        const unsigned r = r0_i + q_row;
        const unsigned dense_row = row_offset + r;
        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (valid) raw = *reinterpret_cast<const WeaveLdsU4*>(lds_q + r * 128u + q_lane * 8u);
        weave_quant_bf16x8_store(
            raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n,
            quant_out_base, quant_scale_base);
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

// ============================================================================
// STAGE 2: BM in {64, 128}, BN = 128, BK = 128 (2 x K=64), 2x2 Waves,
//          ZERO K-Loop Barriers, 17.5KB LDS, vmcnt(13) Deep Prefetch, 0 Spills!
// ============================================================================
template <unsigned kBM>
__device__ __forceinline__ void weave_s2_400t_2x2(
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
    unsigned char* arena,
    const unsigned* row_dir = nullptr,
    const uint64_t* peer_tab = nullptr,
    const uint64_t* node_tab = nullptr,
    unsigned node_row_base = 0u,
    unsigned my_node = 0u)
{
    static_assert(kBM == 64u || kBM == 128u);
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned wm = wave_idx >> 1u;
    const unsigned wn = wave_idx & 1u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;
    const unsigned ld_r = lane >> 2u;
    const unsigned ld_c16 = lane & 3u;

    constexpr unsigned kHalfM = kBM >> 1u;
    constexpr unsigned kMXDL = kHalfM >> 4u;
    constexpr bool kNoCse = (kBM == 128u);
    constexpr unsigned kTokOff = 16384u;
    constexpr unsigned kSaLdsOff = 16896u;

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
    const unsigned exp_u    = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
    const uint64_t nk_bytes = static_cast<uint64_t>(N) * static_cast<uint64_t>(K);
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(exp_u) * nk_bytes;
    const float* bs_ptr = b_scale_base
        + (static_cast<uint64_t>(exp_u) * n_blocks + tile_n) * num_k128;
    WeaveConstF32As4* bs_s = weave_make_as4_f32(bs_ptr);

    const uint32_t a_limit  = __builtin_amdgcn_readfirstlane(route_rows * K);
    const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(route_rows * num_k128 * 4u);
    const uint32_t b_limit  = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(nk_bytes));

    const WeaveI32x4 a_rsrc  = weave_make_buf_rsrc(a_base, a_limit);
    const WeaveI32x4 sa_rsrc = weave_make_buf_rsrc(a_scale_base, sa_limit);
    const WeaveI32x4 b_rsrc  = weave_make_buf_rsrc(b_exp, b_limit);

    uint32_t a_voff[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r_ld = wm * kHalfM + mx * 16u + ld_r;
        const uint32_t tok_ld = lds_tok[r_ld];
        a_voff[mx] = ((r_ld < valid_rows) && (tok_ld < route_rows))
            ? (tok_ld * K + ld_c16 * 16u) : a_limit;
    }
    const unsigned r_sa = wm * kHalfM + lane;
    const bool sa_in_m = (lane < kHalfM);
    const uint32_t tok_sa = sa_in_m ? lds_tok[r_sa] : route_rows;
    const uint32_t sa_voff = (sa_in_m && (r_sa < valid_rows) && (tok_sa < route_rows))
        ? (tok_sa * num_k128 * 4u) : sa_limit;

    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);
    const uint32_t b_voff0 = ((tile_n * 8u + wn * 4u) * num_k64 * 64u + lane) * 16u;

    WeaveLdsU8* const lds_wr0 = lds + wave_idx * 4096u
        + ld_r * 64u + (ld_c16 ^ ((ld_r >> 1u) & 3u)) * 16u;
    WeaveLdsU8* const lds_rd0 = lds + wave_idx * 4096u
        + m_lane * 64u + (k_grp ^ ((m_lane >> 1u) & 3u)) * 16u;
    WeaveLdsF32* const lds_sa_wr = reinterpret_cast<WeaveLdsF32*>(lds + kSaLdsOff + wave_idx * 256u) + lane;
    const WeaveLdsF32* const lds_sa_rd = reinterpret_cast<const WeaveLdsF32*>(lds + kSaLdsOff + wave_idx * 256u) + m_lane;

    WeaveAcc4 acc[kMXDL][4];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        #pragma unroll
        for (unsigned nx = 0u; nx < 4u; ++nx) weave_acc_zero(acc[mx][nx]);
    }

    WeaveI64x2 bw0[4], bw1[4];
    WeaveI64x2 a_ready[kMXDL], a_inflight[kMXDL];

    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        a_inflight[mx] = weave_s2_a_load_i64x2(a_rsrc, a_voff[mx], 0u);
    }
    const float sa0 = weave_s2_sa_load(sa_rsrc, sa_voff, 0u);
    bw0[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, 0u);
    bw0[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_nstride);
    bw0[2] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_nstride * 2u);
    bw0[3] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_nstride * 3u);
    float sb_cur = bs_s[0];

    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        a_ready[mx] = weave_s2_a_load_i64x2(a_rsrc, a_voff[mx], 64u);
    }
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        weave_lds_store_i64x2(lds_wr0 + mx * 1024u, a_inflight[mx]);
    }
    *lds_sa_wr = sa0;
    __builtin_amdgcn_s_waitcnt(0xc07f);

    uint32_t a_soff = 0u, b_soff = 0u, sa_soff = 0u;
    WeaveI64x2 dummy_a[1];

    #pragma clang loop unroll(disable)
    for (unsigned kb = 0u; kb + 1u < num_k128; ++kb) {
        const uint32_t nxt_kb = kb + 1u;
        const uint32_t b_soff_h1 = b_soff + 1024u;

        bw1[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1);
        bw1[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            a_inflight[mx] = weave_s2_a_load_i64x2(a_rsrc, a_voff[mx], a_soff + 128u);
        }
        const float sa_nxt = weave_s2_sa_load(sa_rsrc, sa_voff, sa_soff + 4u);
        const float sb_nxt = bs_s[nxt_kb];

        weave_400t_pair_k64<kMXDL, 0u, kNoCse, 0u, 0u, 1>(
            lds_rd0, lds_wr0, lds_sa_rd, bw0[0], bw0[1], sb_cur, dummy_a, acc);

        bw1[2] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride * 2u);
        bw1[3] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride * 3u);
        weave_400t_pair_k64<kMXDL, 2u, kNoCse, 0u, kMXDL, kMXDL>(
            lds_rd0, lds_wr0, lds_sa_rd, bw0[2], bw0[3], sb_cur, a_ready, acc);
        asm volatile("" ::: "memory");

        a_soff  += 128u;
        b_soff  += 2048u;
        sa_soff += 4u;

        bw0[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff);
        bw0[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff + b_nstride);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            a_ready[mx] = weave_s2_a_load_i64x2(a_rsrc, a_voff[mx], a_soff + 64u);
        }
        weave_400t_pair_k64<kMXDL, 0u, kNoCse, 0u, 0u, 1>(
            lds_rd0, lds_wr0, lds_sa_rd, bw1[0], bw1[1], sb_cur, dummy_a, acc);

        bw0[2] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff + b_nstride * 2u);
        bw0[3] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff + b_nstride * 3u);
        weave_400t_pair_k64<kMXDL, 2u, kNoCse, 0u, kMXDL, kMXDL>(
            lds_rd0, lds_wr0, lds_sa_rd, bw1[2], bw1[3], sb_cur, a_inflight, acc);

        *lds_sa_wr = sa_nxt;
        sb_cur = sb_nxt;
        asm volatile("" ::: "memory");
    }

    {
        const uint32_t b_soff_h1 = b_soff + 1024u;
        bw1[0] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1);
        bw1[1] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride);
        weave_400t_pair_k64<kMXDL, 0u, kNoCse, 0u, 0u, 1>(
            lds_rd0, lds_wr0, lds_sa_rd, bw0[0], bw0[1], sb_cur, dummy_a, acc);

        bw1[2] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride * 2u);
        bw1[3] = weave_buf_load_i64x2(b_rsrc, b_voff0, b_soff_h1 + b_nstride * 3u);
        weave_400t_pair_k64<kMXDL, 2u, kNoCse, 0u, kMXDL, kMXDL>(
            lds_rd0, lds_wr0, lds_sa_rd, bw0[2], bw0[3], sb_cur, a_ready, acc);
        asm volatile("" ::: "memory");

        weave_400t_pair_k64<kMXDL, 0u, kNoCse, 0u, 0u, 1>(
            lds_rd0, lds_wr0, lds_sa_rd, bw1[0], bw1[1], sb_cur, dummy_a, acc);
        weave_400t_pair_k64<kMXDL, 2u, kNoCse, 0u, 0u, 1>(
            lds_rd0, lds_wr0, lds_sa_rd, bw1[2], bw1[3], sb_cur, dummy_a, acc);
    }

    weave_4wave_lds_barrier_65440(arena);
    WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
    const unsigned c_row = tid >> 4u;
    const unsigned c_lane = tid & 15u;

    constexpr unsigned kEpiPasses = (kBM == 128u) ? 2u : 1u;
    #pragma unroll
    for (unsigned ep = 0u; ep < kEpiPasses; ++ep) {
        const unsigned mx_base = ep * 2u;
        #pragma unroll
        for (unsigned m_i = 0u; m_i < 2u; ++m_i) {
            const unsigned mx = mx_base + m_i;
            const unsigned r = wm * kHalfM + mx * 16u + m_lane;
            const unsigned lds_r = wm * 32u + m_i * 16u + m_lane;
            const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
            #pragma unroll
            for (unsigned nx = 0u; nx < 4u; ++nx) {
                const unsigned col = (wn * 4u + nx) * 16u + k_grp * 4u;
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
        for (unsigned r0_i = 0u; r0_i < 64u; r0_i += 16u) {
            const unsigned lds_r = r0_i + c_row;
            const unsigned r = (kBM == 128u)
                ? ((lds_r < 32u) ? (ep * 32u + lds_r) : (64u + ep * 32u + (lds_r - 32u)))
                : lds_r;
            const uint32_t tok_m = lds_tok[r];
            if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
                const unsigned c8_0 = (c_lane * 2u + 0u) ^ (lds_r & 7u);
                const unsigned c8_1 = (c_lane * 2u + 1u) ^ (lds_r & 7u);
                const uint2 u0 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + lds_r * 128u + c8_0 * 4u);
                const uint2 u1 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + lds_r * 128u + c8_1 * 4u);
                uint16_t* dst_p = c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n * 128u + c_lane * 8u;
                uint16_t* direct_p = weave_direct_row_dst(row_dir, peer_tab, node_tab, node_row_base, my_node, tok_m, N);
                if (direct_p != nullptr) {
                    direct_p += static_cast<uint64_t>(tile_n) * 128u + c_lane * 8u;
                    weave_gstore_u4_wt(direct_p, make_uint4(u0.x, u0.y, u1.x, u1.y));
                } else {
                    weave_gstore_u4(dst_p, make_uint4(u0.x, u0.y, u1.x, u1.y));
                }
            }
        }
        weave_4wave_lds_barrier_65440(arena);
    }
    asm volatile("" ::: "memory");
}
