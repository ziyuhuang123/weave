#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>
#include "ck/ck.hpp"
#include "ck/utility/synchronization.hpp"
#include "official_m16_m32_configs_v1262.hpp"
#include "gemm2_configs_v1010_hip.hpp"
#include "v956_stage1_body.hpp"
#include "v1262_subwave_barrier.hpp"

// ============================================================================
// 1. Zero-SBarrier 4-Wave (waves 0..3: tid < 256) Hardware LDS Barrier
// Uses 16 bytes at arena + 65440..65455 for waves 0..3.
// Contiguous LDS arena 0..65439 = 65,440 bytes reserved for GEMM ping-pong!
// Waves 4..7 in 512-thread workgroup run untouched (zero s_barrier instructions).
// ============================================================================
#ifndef WEAVE_4WAVE_BARRIER_DEFINED
#define WEAVE_4WAVE_BARRIER_DEFINED
__device__ __forceinline__ void weave_4wave_barrier_65440()
{
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    __threadfence_block();
    const unsigned tid = threadIdx.x;
    if (tid < 256u && (tid & 63u) == 0u) {
        const unsigned w = tid >> 6u;
        volatile unsigned* sbar = reinterpret_cast<volatile unsigned*>(g_v1262_phase_arena + 65440u);
        const unsigned next = sbar[w] + 1u;
        sbar[w] = next;
        __builtin_amdgcn_s_waitcnt(0xc07f);
        while (sbar[0] < next || sbar[1] < next || sbar[2] < next || sbar[3] < next) {
            // Tight poll without s_sleep: ~8 cycles across 4 SIMDs on same CU!
        }
    }
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    __threadfence_block();
}
#endif

// ============================================================================
// 2. High-Performance Math and Bit-Conversion Primitives
// ============================================================================
using WeaveFloat4 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float weave_fast_silu(float x) {
    float y = 0.0f;
    ck::tensor_operation::element_wise::Silu{}(y, x);
    return y;
}

__device__ __forceinline__ uint16_t weave_f32_to_bf16_bits(float x) {
    ck::bhalf_t b = ck::type_convert<ck::bhalf_t>(x);
    return ck::bit_cast<uint16_t>(b);
}

__device__ __forceinline__ float weave_bf16_bits_to_f32(uint16_t u) {
    return ck::type_convert<float>(ck::bit_cast<ck::bhalf_t>(u));
}

__device__ __forceinline__ float weave_round_bf16(float x) {
    return weave_bf16_bits_to_f32(weave_f32_to_bf16_bits(x));
}

// ============================================================================
// 3. Composable Kernel 4-Wave (256-Thread) Configurations (Tiers M16, M32, M64)
// ============================================================================
template <int MPerBlock_, int NPerBlock_, int KPerBlock_, int MWaves_, int NWaves_,
          ck::BlockGemmPipelineVersion PipeVer_, int ActOP_, bool GateUp_, bool MulWeight_,
          int CShuffleMXdl_ = (MPerBlock_ / (16 * MWaves_))>
struct MakeAgy256MoeOp {
    static constexpr unsigned BlockSize = 256;
    static constexpr unsigned MPerBlock = MPerBlock_;
    static constexpr unsigned NPerBlock = NPerBlock_;
    static constexpr unsigned KPerBlock = KPerBlock_;
    static constexpr ck::BlockGemmPipelineVersion PipeVer = PipeVer_;

    static constexpr ck::index_t MNPerXDL = 16;
    static_assert(MWaves_ * NWaves_ * 64 == 256);
    static constexpr ck::index_t MXDL = MPerBlock_ / (MNPerXDL * MWaves_);
    static constexpr ck::index_t NXDL = NPerBlock_ / (MNPerXDL * NWaves_);
    static constexpr ck::index_t K0 = KPerBlock_ / 16;
    static constexpr ck::index_t AK1Threads = (256 / K0 > MPerBlock_) ? ((256 / K0) / MPerBlock_) : 1;
    static constexpr ck::index_t AMThreads = 256 / (K0 * AK1Threads);
    static constexpr ck::index_t AScalarPerVec = 16 / AK1Threads;
    static constexpr ck::index_t BK0_N = 256 / K0;

    using Op = ck::tensor_operation::device::DeviceMoeGemmBlockScale<
        Row, Col, V1262DsLayout, Row,
        V1262A0, V1262A1, V1262B0, V1262B1, V1262Ds, V1262E,
        V1262Acc, F32,
        V1262Pass, V1262Pass, V1262CElementOp, V1262GemmSpec,
        256, V1262ScaleBlockM, V1262ScaleBlockN, V1262ScaleBlockK,
        MPerBlock_, NPerBlock_, KPerBlock_,
        16, 16,
        16, 16,
        MXDL, NXDL,
        ck::Sequence<K0, AMThreads, AK1Threads>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
        2, AScalarPerVec, AScalarPerVec, false,
        ck::Sequence<K0, BK0_N, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
        2, 16, 16, false,
        CShuffleMXdl_, NXDL, ck::Sequence<1, 16, 1, 16>, ck::Sequence<2, 1, 1, 1>,
        ck::BlockGemmPipelineScheduler::Intrawave,
        PipeVer_,
        ActOP_, false, GateUp_, false, MulWeight_,
        int32_t, V1262A0, V1262A0, V1262A0, V1262A0, false>;
    using Grid = typename Op::GridwiseGemm64;
};

// ============================================================================
// 4. Custom BK=64 2-LDS Ping-Pong Stage 1 Kernel (BM in {64, 128, 256})
// Fuses Silu(Gate)*Up + 1x128 FP8 Blockscale Quantize with 0 Scratch Alloca!
// ============================================================================
template <unsigned kBM>
__device__ __noinline__ void weave_custom_s1_f8_256t(
    const int32_t* __restrict__ s_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    uint16_t* __restrict__ stage1_tmp_bf16,
    uint8_t* __restrict__ quant_out_base,
    float* __restrict__ quant_scale_base,
    unsigned row_offset,
    unsigned tile_n,
    unsigned num_tokens,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    static_assert(kBM == 16u || kBM == 32u || kBM == 64u || kBM == 128u || kBM == 256u);
    const unsigned tid = threadIdx.x;
    if (tid >= 256u) return;

    const unsigned wave = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kSubNPasses = (kBM == 256u) ? 2u : 1u;
    constexpr unsigned kBN = 128u / kSubNPasses;
    constexpr unsigned kNXDL = kBN / 64u;
    constexpr unsigned kMXDL = kBM / 16u;

    constexpr unsigned kRPass = (kBM >= 64u) ? (kBM / 64u) : 1u;
    constexpr unsigned kLdsStageBytes = kBM * 64u;
    uint32_t* lds_tok = reinterpret_cast<uint32_t*>(arena + 32768u);

    if (tid < kBM) {
        lds_tok[tid] = static_cast<uint32_t>(s_ids_tile[tid]) & 0xffffffu;
    }
    weave_4wave_barrier_65440();

    const unsigned num_k64 = K >> 6u;
    const unsigned k_groups = K >> 7u;
    const unsigned n_blocks = N >> 7u;
    const uint64_t expert_b_bytes = (uint64_t)N * (uint64_t)K * 2ull;
    const uint8_t* b_gate_exp = b_base + (uint64_t)(uint32_t)expert_id * expert_b_bytes;
    const uint8_t* b_up_exp   = b_gate_exp + (uint64_t)N * (uint64_t)K;

    const uint64_t expert_bs_stride = (uint64_t)n_blocks * 2ull * k_groups;
    const float* bs_gate_row = b_scale_base + (uint64_t)(uint32_t)expert_id * expert_bs_stride + (uint64_t)tile_n * k_groups;
    const float* bs_up_row   = bs_gate_row + (uint64_t)n_blocks * k_groups;

    const unsigned a_r_sub = tid >> 2u;
    const unsigned a_c16   = tid & 3u;

    #pragma clang loop unroll(disable)
    for (unsigned sub_n = 0u; sub_n < kSubNPasses; ++sub_n) {
        WeaveFloat4 gate_acc[kMXDL][kNXDL];
        WeaveFloat4 up_acc[kMXDL][kNXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            #pragma unroll
            for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                gate_acc[mx][nx] = {0.f, 0.f, 0.f, 0.f};
                up_acc[mx][nx]   = {0.f, 0.f, 0.f, 0.f};
            }
        }

        // Prologue
        {
            uint8_t* lds_dst = arena;
            #pragma unroll
            for (unsigned rp = 0u; rp < kRPass; ++rp) {
                const unsigned r = rp * 64u + a_r_sub;
                if constexpr (kBM < 64u) {
                    if (r < kBM) {
                        const uint32_t tok = lds_tok[r];
                        const uint4 v = (tok < num_tokens)
                            ? *reinterpret_cast<const uint4*>(a_base + (uint64_t)tok * K + a_c16 * 16u)
                            : make_uint4(0u, 0u, 0u, 0u);
                        const unsigned c16_swz = a_c16 ^ ((r >> 1u) & 3u);
                        *reinterpret_cast<uint4*>(lds_dst + r * 64u + c16_swz * 16u) = v;
                    }
                } else {
                    const uint32_t tok = lds_tok[r];
                    const uint4 v = (tok < num_tokens)
                        ? *reinterpret_cast<const uint4*>(a_base + (uint64_t)tok * K + a_c16 * 16u)
                        : make_uint4(0u, 0u, 0u, 0u);
                    const unsigned c16_swz = a_c16 ^ ((r >> 1u) & 3u);
                    *reinterpret_cast<uint4*>(lds_dst + r * 64u + c16_swz * 16u) = v;
                }
            }
        }
        weave_4wave_barrier_65440();

        #pragma clang loop unroll(disable)
        for (unsigned k64 = 0u; k64 < num_k64; ++k64) {
            uint4 a_prefetch[kRPass];
            const unsigned next_k64 = k64 + 1u;
            if (next_k64 < num_k64) {
                const unsigned k_byte = next_k64 * 64u + a_c16 * 16u;
                #pragma unroll
                for (unsigned rp = 0u; rp < kRPass; ++rp) {
                    const unsigned r = rp * 64u + a_r_sub;
                    if constexpr (kBM < 64u) {
                        if (r < kBM) {
                            const uint32_t tok = lds_tok[r];
                            a_prefetch[rp] = (tok < num_tokens)
                                ? *reinterpret_cast<const uint4*>(a_base + (uint64_t)tok * K + k_byte)
                                : make_uint4(0u, 0u, 0u, 0u);
                        } else {
                            a_prefetch[rp] = make_uint4(0u, 0u, 0u, 0u);
                        }
                    } else {
                        const uint32_t tok = lds_tok[r];
                        a_prefetch[rp] = (tok < num_tokens)
                            ? *reinterpret_cast<const uint4*>(a_base + (uint64_t)tok * K + k_byte)
                            : make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            }

            const unsigned ks = k64 >> 1u;
            const float sb_g = bs_gate_row[ks];
            const float sb_u = bs_up_row[ks];

            uint4 bg_vec[kNXDL];
            uint4 bu_vec[kNXDL];
            #pragma unroll
            for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                const unsigned col16_in_tile = sub_n * 4u + wave * kNXDL + nx;
                const unsigned n16_global = tile_n * 8u + col16_in_tile;
                const uint64_t b_off = (((uint64_t)n16_global * num_k64 + k64) * 64ull + lane) * 16ull;
                bg_vec[nx] = *reinterpret_cast<const uint4*>(b_gate_exp + b_off);
                bu_vec[nx] = *reinterpret_cast<const uint4*>(b_up_exp   + b_off);
            }

            const uint8_t* lds_cur = arena + (k64 & 1u) * kLdsStageBytes;
            #pragma unroll
            for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                const unsigned r = mx * 16u + m_lane;
                const unsigned c16_swz = k_grp ^ ((r >> 1u) & 3u);
                const uint4 a_vec = *reinterpret_cast<const uint4*>(lds_cur + r * 64u + c16_swz * 16u);
                const int64_t a0 = (int64_t)a_vec.x | ((int64_t)a_vec.y << 32);
                const int64_t a1 = (int64_t)a_vec.z | ((int64_t)a_vec.w << 32);

                const uint32_t tok_m = lds_tok[r];
                const float sa = (tok_m < num_tokens) ? a_scale_base[(uint64_t)tok_m * k_groups + ks] : 0.0f;
                const float sab_g = sa * sb_g;
                const float sab_u = sa * sb_u;

                #pragma unroll
                for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                    const int64_t bg0 = (int64_t)bg_vec[nx].x | ((int64_t)bg_vec[nx].y << 32);
                    const int64_t bg1 = (int64_t)bg_vec[nx].z | ((int64_t)bg_vec[nx].w << 32);
                    WeaveFloat4 tmp = {0.f, 0.f, 0.f, 0.f};
                    tmp = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg0, a0, tmp, 0, 0, 0);
                    tmp = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg1, a1, tmp, 0, 0, 0);
                    gate_acc[mx][nx] += tmp * sab_g;

                    const int64_t bu0 = (int64_t)bu_vec[nx].x | ((int64_t)bu_vec[nx].y << 32);
                    const int64_t bu1 = (int64_t)bu_vec[nx].z | ((int64_t)bu_vec[nx].w << 32);
                    tmp = {0.f, 0.f, 0.f, 0.f};
                    tmp = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu0, a0, tmp, 0, 0, 0);
                    tmp = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu1, a1, tmp, 0, 0, 0);
                    up_acc[mx][nx] += tmp * sab_u;
                }
            }

            if (next_k64 < num_k64) {
                uint8_t* lds_nxt = arena + (next_k64 & 1u) * kLdsStageBytes;
                #pragma unroll
                for (unsigned rp = 0u; rp < kRPass; ++rp) {
                    const unsigned r = rp * 64u + a_r_sub;
                    if constexpr (kBM < 64u) {
                        if (r < kBM) {
                            const unsigned c16_swz = a_c16 ^ ((r >> 1u) & 3u);
                            *reinterpret_cast<uint4*>(lds_nxt + r * 64u + c16_swz * 16u) = a_prefetch[rp];
                        }
                    } else {
                        const unsigned c16_swz = a_c16 ^ ((r >> 1u) & 3u);
                        *reinterpret_cast<uint4*>(lds_nxt + r * 64u + c16_swz * 16u) = a_prefetch[rp];
                    }
                }
                weave_4wave_barrier_65440();
            }
        }

        // Epilogue: All rows into LDS as BF16 in ONE pass (Zero intermediate barriers!)
        weave_4wave_barrier_65440();
        if constexpr (kBM <= 128u) {
            uint16_t* lds_bf16 = reinterpret_cast<uint16_t*>(arena);
            #pragma unroll
            for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                const unsigned r = mx * 16u + m_lane;
                #pragma unroll
                for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                    const unsigned col = (wave * kNXDL + nx) * 16u + k_grp * 4u;
                    const uint32_t p0 = (uint32_t)weave_f32_to_bf16_bits(weave_fast_silu(gate_acc[mx][nx][0]) * up_acc[mx][nx][0]) |
                                        ((uint32_t)weave_f32_to_bf16_bits(weave_fast_silu(gate_acc[mx][nx][1]) * up_acc[mx][nx][1]) << 16u);
                    const uint32_t p1 = (uint32_t)weave_f32_to_bf16_bits(weave_fast_silu(gate_acc[mx][nx][2]) * up_acc[mx][nx][2]) |
                                        ((uint32_t)weave_f32_to_bf16_bits(weave_fast_silu(gate_acc[mx][nx][3]) * up_acc[mx][nx][3]) << 16u);
                    *reinterpret_cast<uint2*>(lds_bf16 + r * 128u + col) = make_uint2(p0, p1);
                    if (stage1_tmp_bf16 != nullptr && (row_offset + r < route_rows)) {
                        *reinterpret_cast<uint2*>(stage1_tmp_bf16 + (uint64_t)(row_offset + r) * N + tile_n * 128u + col) = make_uint2(p0, p1);
                    }
                }
            }
            weave_4wave_barrier_65440();

            // Quantize with ZERO stack alloca
            const float fp8_max = ck_tile::type_convert<float>(ck_tile::numeric<ck_tile::fp8_t>::max());
            const float inv_fp8_max = 1.0f / fp8_max;
            constexpr unsigned kRowsPerPass = 16u; // 16 rows per pass
            constexpr unsigned kQPasses = (kBM >= 16u) ? (kBM / kRowsPerPass) : 1u;

            const unsigned q_row_sub = tid >> 4u; // 0..15
            const unsigned q_lane    = tid & 15u; // 0..15

            #pragma unroll
            for (unsigned qp = 0u; qp < kQPasses; ++qp) {
                const unsigned r = qp * kRowsPerPass + q_row_sub;
                const unsigned dense_row = row_offset + r;
                const uint32_t tok_q = lds_tok[r];
                const bool valid = (dense_row < route_rows) && (tok_q < num_tokens);

                const uint4* src_p = reinterpret_cast<const uint4*>(lds_bf16 + r * 128u + q_lane * 8u);
                const uint4 v4 = *src_p;
                const uint32_t lo0 = v4.x;
                const uint32_t hi0 = v4.y;
                const uint32_t lo1 = v4.z;
                const uint32_t hi1 = v4.w;

                const float f0 = weave_bf16_bits_to_f32((uint16_t)lo0);
                const float f1 = weave_bf16_bits_to_f32((uint16_t)(lo0 >> 16u));
                const float f2 = weave_bf16_bits_to_f32((uint16_t)hi0);
                const float f3 = weave_bf16_bits_to_f32((uint16_t)(hi0 >> 16u));
                const float f4 = weave_bf16_bits_to_f32((uint16_t)lo1);
                const float f5 = weave_bf16_bits_to_f32((uint16_t)(lo1 >> 16u));
                const float f6 = weave_bf16_bits_to_f32((uint16_t)hi1);
                const float f7 = weave_bf16_bits_to_f32((uint16_t)(hi1 >> 16u));

                float abs_max = 1.0e-10f;
                if (valid) {
                    abs_max = fmaxf(abs_max, fabsf(f0));
                    abs_max = fmaxf(abs_max, fabsf(f1));
                    abs_max = fmaxf(abs_max, fabsf(f2));
                    abs_max = fmaxf(abs_max, fabsf(f3));
                    abs_max = fmaxf(abs_max, fabsf(f4));
                    abs_max = fmaxf(abs_max, fabsf(f5));
                    abs_max = fmaxf(abs_max, fabsf(f6));
                    abs_max = fmaxf(abs_max, fabsf(f7));
                }
                #pragma unroll
                for (unsigned delta = 1u; delta < 16u; delta <<= 1u) {
                    abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffffffffffffull, abs_max, (int)delta, 16));
                }
                const float scale = abs_max * inv_fp8_max;
                const uint64_t dense_group = (uint64_t)dense_row * n_blocks + tile_n;
                if (valid && q_lane == 0u) quant_scale_base[dense_group] = scale;

                if (valid) {
                    const float inv_scale = 1.0f / scale;
                    const uint16_t pk0 = __builtin_amdgcn_cvt_pk_fp8_f32(
                        __builtin_amdgcn_fmed3f(f0 * inv_scale, fp8_max, -fp8_max),
                        __builtin_amdgcn_fmed3f(f1 * inv_scale, fp8_max, -fp8_max), 0, 0);
                    const uint16_t pk1 = __builtin_amdgcn_cvt_pk_fp8_f32(
                        __builtin_amdgcn_fmed3f(f2 * inv_scale, fp8_max, -fp8_max),
                        __builtin_amdgcn_fmed3f(f3 * inv_scale, fp8_max, -fp8_max), 0, 0);
                    const uint16_t pk2 = __builtin_amdgcn_cvt_pk_fp8_f32(
                        __builtin_amdgcn_fmed3f(f4 * inv_scale, fp8_max, -fp8_max),
                        __builtin_amdgcn_fmed3f(f5 * inv_scale, fp8_max, -fp8_max), 0, 0);
                    const uint16_t pk3 = __builtin_amdgcn_cvt_pk_fp8_f32(
                        __builtin_amdgcn_fmed3f(f6 * inv_scale, fp8_max, -fp8_max),
                        __builtin_amdgcn_fmed3f(f7 * inv_scale, fp8_max, -fp8_max), 0, 0);

                    uint8_t* qdst = quant_out_base + dense_group * 128ull + q_lane * 8u;
                    *reinterpret_cast<uint2*>(qdst) = make_uint2((uint32_t)pk0 | ((uint32_t)pk1 << 16u),
                                                                 (uint32_t)pk2 | ((uint32_t)pk3 << 16u));
                }
            }
        } else {
            // kBM == 256: 2 subpasses of BN=64
            if (sub_n == 0u) {
                #pragma unroll
                for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                    const unsigned r = mx * 16u + m_lane;
                    const uint32_t tok_m = lds_tok[r];
                    if (tok_m < num_tokens) {
                        const unsigned col_base = wave * 16u + k_grp * 4u;
                        uint16_t* dst = stage1_tmp_bf16 + (uint64_t)(row_offset + r) * N + tile_n * 128u + col_base;
                        const uint32_t p0 = (uint32_t)weave_f32_to_bf16_bits(weave_fast_silu(gate_acc[mx][0][0]) * up_acc[mx][0][0]) |
                                            ((uint32_t)weave_f32_to_bf16_bits(weave_fast_silu(gate_acc[mx][0][1]) * up_acc[mx][0][1]) << 16u);
                        const uint32_t p1 = (uint32_t)weave_f32_to_bf16_bits(weave_fast_silu(gate_acc[mx][0][2]) * up_acc[mx][0][2]) |
                                            ((uint32_t)weave_f32_to_bf16_bits(weave_fast_silu(gate_acc[mx][0][3]) * up_acc[mx][0][3]) << 16u);
                        *reinterpret_cast<uint2*>(dst) = make_uint2(p0, p1);
                    }
                }
                weave_4wave_barrier_65440();
            } else {
                float* lds_q = reinterpret_cast<float*>(arena);
                const float fp8_max = ck_tile::type_convert<float>(ck_tile::numeric<ck_tile::fp8_t>::max());
                const float inv_fp8_max = 1.0f / fp8_max;
                const unsigned q_row = tid >> 4u;
                const unsigned q_lane = tid & 15u;

                #pragma unroll
                for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                    const unsigned col_base_hi = wave * 16u + k_grp * 4u;
                    float* dst_row = lds_q + m_lane * 64u + col_base_hi;
                    #pragma unroll
                    for (unsigned t = 0u; t < 4u; ++t) {
                        dst_row[t] = weave_round_bf16(weave_fast_silu(gate_acc[mx][0][t]) * up_acc[mx][0][t]);
                    }
                    weave_4wave_barrier_65440();

                    const unsigned r = mx * 16u + q_row;
                    const unsigned dense_row = row_offset + r;
                    const uint32_t tok_q = lds_tok[r];
                    const bool valid = (dense_row < route_rows) && (tok_q < num_tokens);

                    float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
                    float v4 = 0.f, v5 = 0.f, v6 = 0.f, v7 = 0.f;
                    if (valid) {
                        if (q_lane < 8u) {
                            const uint16_t* src_lo = stage1_tmp_bf16 + (uint64_t)dense_row * N + tile_n * 128u + q_lane * 8u;
                            v0 = weave_bf16_bits_to_f32(src_lo[0]); v1 = weave_bf16_bits_to_f32(src_lo[1]);
                            v2 = weave_bf16_bits_to_f32(src_lo[2]); v3 = weave_bf16_bits_to_f32(src_lo[3]);
                            v4 = weave_bf16_bits_to_f32(src_lo[4]); v5 = weave_bf16_bits_to_f32(src_lo[5]);
                            v6 = weave_bf16_bits_to_f32(src_lo[6]); v7 = weave_bf16_bits_to_f32(src_lo[7]);
                        } else {
                            const float* src_hi = lds_q + q_row * 64u + (q_lane - 8u) * 8u;
                            v0 = src_hi[0]; v1 = src_hi[1]; v2 = src_hi[2]; v3 = src_hi[3];
                            v4 = src_hi[4]; v5 = src_hi[5]; v6 = src_hi[6]; v7 = src_hi[7];
                        }
                    }
                    float abs_max = 1.0e-10f;
                    if (valid) {
                        abs_max = fmaxf(abs_max, fabsf(v0)); abs_max = fmaxf(abs_max, fabsf(v1));
                        abs_max = fmaxf(abs_max, fabsf(v2)); abs_max = fmaxf(abs_max, fabsf(v3));
                        abs_max = fmaxf(abs_max, fabsf(v4)); abs_max = fmaxf(abs_max, fabsf(v5));
                        abs_max = fmaxf(abs_max, fabsf(v6)); abs_max = fmaxf(abs_max, fabsf(v7));
                    }
                    #pragma unroll
                    for (unsigned delta = 1u; delta < 16u; delta <<= 1u) {
                        abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffffffffffffull, abs_max, (int)delta, 16));
                    }
                    const float scale = abs_max * inv_fp8_max;
                    const uint64_t dense_group = (uint64_t)dense_row * n_blocks + tile_n;
                    if (valid && q_lane == 0u) quant_scale_base[dense_group] = scale;
                    if (valid) {
                        const float inv_scale = 1.0f / scale;
                        uint8_t* qdst = quant_out_base + dense_group * 128ull + q_lane * 8u;
                        const uint16_t pk0 = __builtin_amdgcn_cvt_pk_fp8_f32(
                            __builtin_amdgcn_fmed3f(v0 * inv_scale, fp8_max, -fp8_max),
                            __builtin_amdgcn_fmed3f(v1 * inv_scale, fp8_max, -fp8_max), 0, 0);
                        const uint16_t pk1 = __builtin_amdgcn_cvt_pk_fp8_f32(
                            __builtin_amdgcn_fmed3f(v2 * inv_scale, fp8_max, -fp8_max),
                            __builtin_amdgcn_fmed3f(v3 * inv_scale, fp8_max, -fp8_max), 0, 0);
                        const uint16_t pk2 = __builtin_amdgcn_cvt_pk_fp8_f32(
                            __builtin_amdgcn_fmed3f(v4 * inv_scale, fp8_max, -fp8_max),
                            __builtin_amdgcn_fmed3f(v5 * inv_scale, fp8_max, -fp8_max), 0, 0);
                        const uint16_t pk3 = __builtin_amdgcn_cvt_pk_fp8_f32(
                            __builtin_amdgcn_fmed3f(v6 * inv_scale, fp8_max, -fp8_max),
                            __builtin_amdgcn_fmed3f(v7 * inv_scale, fp8_max, -fp8_max), 0, 0);
                        *reinterpret_cast<uint2*>(qdst)      = make_uint2((uint32_t)pk0 | ((uint32_t)pk1 << 16u),
                                                                          (uint32_t)pk2 | ((uint32_t)pk3 << 16u));
                    }
                    weave_4wave_barrier_65440();
                }
            }
        }
    }
}

// ============================================================================
// 5. Custom BK=64 2-LDS Ping-Pong Stage 2 Kernel (BM in {64, 128, 256})
// Single-Pass BN=128 with Direct Coalesced BF16 Global Store + MulRoutedWeight
// ============================================================================
template <unsigned kBM>
__device__ __noinline__ void weave_custom_s2_f8_256t(
    const int32_t* __restrict__ s2_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    const float* __restrict__ weights_tile,
    uint16_t* __restrict__ c_out_base,
    unsigned row_offset,
    unsigned tile_n,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    static_assert(kBM == 16u || kBM == 32u || kBM == 64u || kBM == 128u || kBM == 256u);
    const unsigned tid = threadIdx.x;
    if (tid >= 256u) return;

    const unsigned wave = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kNXDL = 2u;
    constexpr unsigned kMXDL = kBM / 16u;
    constexpr unsigned kRPass = (kBM >= 64u) ? (kBM / 64u) : 1u;
    constexpr unsigned kLdsStageBytes = kBM * 64u;
    uint32_t* lds_tok = reinterpret_cast<uint32_t*>(arena + 32768u);

    if (tid < kBM) {
        lds_tok[tid] = static_cast<uint32_t>(s2_ids_tile[tid]) & 0xffffffu;
    }
    weave_4wave_barrier_65440();

    const unsigned num_k64 = K >> 6u;
    const unsigned k_groups = K >> 7u;
    const unsigned n_blocks = N >> 7u;
    const uint8_t* b_exp = b_base + (uint64_t)(uint32_t)expert_id * (uint64_t)N * (uint64_t)K;
    const float* bs_row  = b_scale_base + ((uint64_t)(uint32_t)expert_id * n_blocks + tile_n) * k_groups;

    const unsigned a_r_sub = tid >> 2u;
    const unsigned a_c16   = tid & 3u;

    WeaveFloat4 acc[kMXDL][kNXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) {
            acc[mx][nx] = {0.f, 0.f, 0.f, 0.f};
        }
    }

    // Prologue
    {
        uint8_t* lds_dst = arena;
        #pragma unroll
        for (unsigned rp = 0u; rp < kRPass; ++rp) {
            const unsigned r = rp * 64u + a_r_sub;
            if constexpr (kBM < 64u) {
                if (r < kBM) {
                    const uint32_t tok = lds_tok[r];
                    const uint4 v = (tok < route_rows)
                        ? *reinterpret_cast<const uint4*>(a_base + (uint64_t)tok * K + a_c16 * 16u)
                        : make_uint4(0u, 0u, 0u, 0u);
                    const unsigned c16_swz = a_c16 ^ ((r >> 1u) & 3u);
                    *reinterpret_cast<uint4*>(lds_dst + r * 64u + c16_swz * 16u) = v;
                }
            } else {
                const uint32_t tok = lds_tok[r];
                const uint4 v = (tok < route_rows)
                    ? *reinterpret_cast<const uint4*>(a_base + (uint64_t)tok * K + a_c16 * 16u)
                    : make_uint4(0u, 0u, 0u, 0u);
                const unsigned c16_swz = a_c16 ^ ((r >> 1u) & 3u);
                *reinterpret_cast<uint4*>(lds_dst + r * 64u + c16_swz * 16u) = v;
            }
        }
    }
    weave_4wave_barrier_65440();

    #pragma clang loop unroll(disable)
    for (unsigned k64 = 0u; k64 < num_k64; ++k64) {
        uint4 a_prefetch[kRPass];
        const unsigned next_k64 = k64 + 1u;
        if (next_k64 < num_k64) {
            const unsigned k_byte = next_k64 * 64u + a_c16 * 16u;
            #pragma unroll
            for (unsigned rp = 0u; rp < kRPass; ++rp) {
                const unsigned r = rp * 64u + a_r_sub;
                if constexpr (kBM < 64u) {
                    if (r < kBM) {
                        const uint32_t tok = lds_tok[r];
                        a_prefetch[rp] = (tok < route_rows)
                            ? *reinterpret_cast<const uint4*>(a_base + (uint64_t)tok * K + k_byte)
                            : make_uint4(0u, 0u, 0u, 0u);
                    } else {
                        a_prefetch[rp] = make_uint4(0u, 0u, 0u, 0u);
                    }
                } else {
                    const uint32_t tok = lds_tok[r];
                    a_prefetch[rp] = (tok < route_rows)
                        ? *reinterpret_cast<const uint4*>(a_base + (uint64_t)tok * K + k_byte)
                        : make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }

        const unsigned ks = k64 >> 1u;
        const float sb = bs_row[ks];

        uint4 b_vec[kNXDL];
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) {
            const unsigned n16_global = tile_n * 8u + wave * 2u + nx;
            const uint64_t b_off = (((uint64_t)n16_global * num_k64 + k64) * 64ull + lane) * 16ull;
            b_vec[nx] = *reinterpret_cast<const uint4*>(b_exp + b_off);
        }

        const uint8_t* lds_cur = arena + (k64 & 1u) * kLdsStageBytes;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned r = mx * 16u + m_lane;
            const unsigned c16_swz = k_grp ^ ((r >> 1u) & 3u);
            const uint4 a_vec = *reinterpret_cast<const uint4*>(lds_cur + r * 64u + c16_swz * 16u);
            const int64_t a0 = (int64_t)a_vec.x | ((int64_t)a_vec.y << 32);
            const int64_t a1 = (int64_t)a_vec.z | ((int64_t)a_vec.w << 32);

            const uint32_t tok_m = lds_tok[r];
            const float sa = (tok_m < route_rows) ? a_scale_base[(uint64_t)tok_m * k_groups + ks] : 0.0f;
            const float sab = sa * sb;

            #pragma unroll
            for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                const int64_t b0 = (int64_t)b_vec[nx].x | ((int64_t)b_vec[nx].y << 32);
                const int64_t b1 = (int64_t)b_vec[nx].z | ((int64_t)b_vec[nx].w << 32);
                WeaveFloat4 tmp = {0.f, 0.f, 0.f, 0.f};
                tmp = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0, a0, tmp, 0, 0, 0);
                tmp = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1, a1, tmp, 0, 0, 0);
                acc[mx][nx] += tmp * sab;
            }
        }

        if (next_k64 < num_k64) {
            uint8_t* lds_nxt = arena + (next_k64 & 1u) * kLdsStageBytes;
            #pragma unroll
            for (unsigned rp = 0u; rp < kRPass; ++rp) {
                const unsigned r = rp * 64u + a_r_sub;
                if constexpr (kBM < 64u) {
                    if (r < kBM) {
                        const unsigned c16_swz = a_c16 ^ ((r >> 1u) & 3u);
                        *reinterpret_cast<uint4*>(lds_nxt + r * 64u + c16_swz * 16u) = a_prefetch[rp];
                    }
                } else {
                    const unsigned c16_swz = a_c16 ^ ((r >> 1u) & 3u);
                    *reinterpret_cast<uint4*>(lds_nxt + r * 64u + c16_swz * 16u) = a_prefetch[rp];
                }
            }
            weave_4wave_barrier_65440();
        }
    }

    // Direct Coalesced BF16 Global Store with ZERO STACK ALLOCA!
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + m_lane;
        const uint32_t tok_m = lds_tok[r];
        if (tok_m < route_rows) {
            const float w = weights_tile[r];
            #pragma unroll
            for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                const unsigned col = tile_n * 128u + (wave * 2u + nx) * 16u + k_grp * 4u;
                const uint32_t p0 = (uint32_t)weave_f32_to_bf16_bits(acc[mx][nx][0] * w) |
                                    ((uint32_t)weave_f32_to_bf16_bits(acc[mx][nx][1] * w) << 16u);
                const uint32_t p1 = (uint32_t)weave_f32_to_bf16_bits(acc[mx][nx][2] * w) |
                                    ((uint32_t)weave_f32_to_bf16_bits(acc[mx][nx][3] * w) << 16u);
                *reinterpret_cast<uint2*>(c_out_base + (uint64_t)tok_m * N + col) = make_uint2(p0, p1);
            }
        }
    }
    weave_4wave_barrier_65440();
}

// ============================================================================
// 6. Zero-Spill 4-Wave CK Wrappers for Tiers BM=32 and BM=16
// Each tier is isolated in __device__ __noinline__ function to prevent
// compiler SROA failure and aggregate frame register explosion!
// ============================================================================
// Stage 1 CK M32 256T (4 compute waves: waves 0..3)
__device__ __noinline__ void weave_ck_s1_m32_256t(
    const int32_t* __restrict__ s_ids,
    const int32_t* __restrict__ e_ids,
    const int32_t* __restrict__ max_id,
    const void* __restrict__ a,
    const void* __restrict__ b,
    void* __restrict__ c,
    const float* __restrict__ as,
    const float* __restrict__ bs,
    int rows, int topk, int N, int K,
    unsigned tile_n,
    unsigned char* arena)
{
    using Cfg = MakeAgy256MoeOp<32, 128, 128, 2, 2, ck::BlockGemmPipelineVersion::v1, 1, true, false>;
    v927_fused::PackedArgument<typename Cfg::Grid> arg(
        s_ids, e_ids, max_id,
        (const V1262A0*)a, (const V1262B0*)b, (V1262E*)c,
        as, bs, rows, topk, rows * topk, N, K, K, K, N);
    Cfg::Grid::template RunAt<true, ck::InMemoryDataOperationEnum::Set, ck::TailNumber::Even>(
        arg.p_sorted_token_ids, arg.p_sorted_expert_ids, arg.p_max_token_id,
        arg.p_a_grid, arg.p_b_grid, arg.p_ds_grid, arg.p_c_grid,
        arg.p_a_scale_grid, arg.p_b_scale_grid,
        arena, arg,
        arg.a_element_op, arg.b_element_op, arg.c_element_op,
        tile_n, 0);
}

// Stage 1 CK M16 256T (4 compute waves: waves 0..3)
__device__ __noinline__ void weave_ck_s1_m16_256t(
    const int32_t* __restrict__ s_ids,
    const int32_t* __restrict__ e_ids,
    const int32_t* __restrict__ max_id,
    const void* __restrict__ a,
    const void* __restrict__ b,
    void* __restrict__ c,
    const float* __restrict__ as,
    const float* __restrict__ bs,
    int rows, int topk, int N, int K,
    unsigned tile_n,
    unsigned char* arena)
{
    using Cfg = MakeAgy256MoeOp<16, 128, 128, 1, 4, ck::BlockGemmPipelineVersion::v1, 1, true, false>;
    v927_fused::PackedArgument<typename Cfg::Grid> arg(
        s_ids, e_ids, max_id,
        (const V1262A0*)a, (const V1262B0*)b, (V1262E*)c,
        as, bs, rows, topk, rows * topk, N, K, K, K, N);
    Cfg::Grid::template RunAt<true, ck::InMemoryDataOperationEnum::Set, ck::TailNumber::Even>(
        arg.p_sorted_token_ids, arg.p_sorted_expert_ids, arg.p_max_token_id,
        arg.p_a_grid, arg.p_b_grid, arg.p_ds_grid, arg.p_c_grid,
        arg.p_a_scale_grid, arg.p_b_scale_grid,
        arena, arg,
        arg.a_element_op, arg.b_element_op, arg.c_element_op,
        tile_n, 0);
}

// Stage 2 CK M32 256T (4 compute waves: waves 0..3)
__device__ __noinline__ void weave_ck_s2_m32_256t(
    const int32_t* __restrict__ s_ids,
    const int32_t* __restrict__ e_ids,
    const int32_t* __restrict__ max_id,
    const void* __restrict__ a,
    const void* __restrict__ b,
    void* __restrict__ c,
    const float* __restrict__ as,
    const float* __restrict__ bs,
    const float* __restrict__ ds,
    int rows, int topk, int N, int K,
    unsigned tile_n,
    unsigned char* arena)
{
    using Cfg = MakeAgy256MoeOp<32, 128, 128, 2, 2, ck::BlockGemmPipelineVersion::v1, 0, false, true>;
    v927_fused::PackedArgument<typename Cfg::Grid> arg(
        s_ids, e_ids, max_id,
        (const V1262A0*)a, (const V1262B0*)b, (V1262E*)c,
        as, bs, rows, topk, rows * topk, N, K, K, K, N);
    arg.p_ds_grid = decltype(arg.p_ds_grid)(ds);
    Cfg::Grid::template RunAt<true, ck::InMemoryDataOperationEnum::Set, ck::TailNumber::Even>(
        arg.p_sorted_token_ids, arg.p_sorted_expert_ids, arg.p_max_token_id,
        arg.p_a_grid, arg.p_b_grid, arg.p_ds_grid, arg.p_c_grid,
        arg.p_a_scale_grid, arg.p_b_scale_grid,
        arena, arg,
        arg.a_element_op, arg.b_element_op, arg.c_element_op,
        tile_n, 0);
}

// Stage 2 CK M16 256T (4 compute waves: waves 0..3)
__device__ __noinline__ void weave_ck_s2_m16_256t(
    const int32_t* __restrict__ s_ids,
    const int32_t* __restrict__ e_ids,
    const int32_t* __restrict__ max_id,
    const void* __restrict__ a,
    const void* __restrict__ b,
    void* __restrict__ c,
    const float* __restrict__ as,
    const float* __restrict__ bs,
    const float* __restrict__ ds,
    int rows, int topk, int N, int K,
    unsigned tile_n,
    unsigned char* arena)
{
    using Cfg = MakeAgy256MoeOp<16, 128, 128, 1, 4, ck::BlockGemmPipelineVersion::v1, 0, false, true>;
    v927_fused::PackedArgument<typename Cfg::Grid> arg(
        s_ids, e_ids, max_id,
        (const V1262A0*)a, (const V1262B0*)b, (V1262E*)c,
        as, bs, rows, topk, rows * topk, N, K, K, K, N);
    arg.p_ds_grid = decltype(arg.p_ds_grid)(ds);
    Cfg::Grid::template RunAt<true, ck::InMemoryDataOperationEnum::Set, ck::TailNumber::Even>(
        arg.p_sorted_token_ids, arg.p_sorted_expert_ids, arg.p_max_token_id,
        arg.p_a_grid, arg.p_b_grid, arg.p_ds_grid, arg.p_c_grid,
        arg.p_a_scale_grid, arg.p_b_scale_grid,
        arena, arg,
        arg.a_element_op, arg.b_element_op, arg.c_element_op,
        tile_n, 0);
}

// ============================================================================
// 7. Zero-Spill 5-Tier Dispatchers (Tiers: 256, 128, 64, 32, 16)
// ============================================================================
// Stage 1 Dispatcher
__device__ __forceinline__ void weave_dispatch_stage1_5tier(
    unsigned m_tier,
    const int32_t* __restrict__ s_ids,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    uint16_t* __restrict__ stage1_tmp_bf16,
    uint8_t* __restrict__ quant_out_base,
    float* __restrict__ quant_scale_base,
    unsigned row_offset,
    unsigned tile_n,
    unsigned num_tokens,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    if (m_tier == 256u) {
        weave_custom_s1_f8_256t<256u>(s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                                      stage1_tmp_bf16, quant_out_base, quant_scale_base,
                                      row_offset, tile_n, num_tokens, route_rows, N, K, arena);
    } else if (m_tier == 128u) {
        weave_custom_s1_f8_256t<128u>(s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                                      stage1_tmp_bf16, quant_out_base, quant_scale_base,
                                      row_offset, tile_n, num_tokens, route_rows, N, K, arena);
    } else if (m_tier == 64u) {
        weave_custom_s1_f8_256t<64u>(s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                                     stage1_tmp_bf16, quant_out_base, quant_scale_base,
                                     row_offset, tile_n, num_tokens, route_rows, N, K, arena);
    } else if (m_tier == 32u) {
        weave_custom_s1_f8_256t<32u>(s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                                     stage1_tmp_bf16, quant_out_base, quant_scale_base,
                                     row_offset, tile_n, num_tokens, route_rows, N, K, arena);
    } else if (m_tier == 16u) {
        weave_custom_s1_f8_256t<16u>(s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                                     stage1_tmp_bf16, quant_out_base, quant_scale_base,
                                     row_offset, tile_n, num_tokens, route_rows, N, K, arena);
    }
}

// Stage 2 Dispatcher
__device__ __forceinline__ void weave_dispatch_stage2_5tier(
    unsigned m_tier,
    const int32_t* __restrict__ s2_ids,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    const float* __restrict__ weights,
    uint16_t* __restrict__ c_out_base,
    unsigned row_offset,
    unsigned tile_n,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    if (m_tier == 256u) {
        weave_custom_s2_f8_256t<256u>(s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                                      weights, c_out_base, row_offset, tile_n, route_rows, N, K, arena);
    } else if (m_tier == 128u) {
        weave_custom_s2_f8_256t<128u>(s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                                      weights, c_out_base, row_offset, tile_n, route_rows, N, K, arena);
    } else if (m_tier == 64u) {
        weave_custom_s2_f8_256t<64u>(s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                                     weights, c_out_base, row_offset, tile_n, route_rows, N, K, arena);
    } else if (m_tier == 32u) {
        weave_custom_s2_f8_256t<32u>(s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                                     weights, c_out_base, row_offset, tile_n, route_rows, N, K, arena);
    } else if (m_tier == 16u) {
        weave_custom_s2_f8_256t<16u>(s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
                                     weights, c_out_base, row_offset, tile_n, route_rows, N, K, arena);
    }
}
