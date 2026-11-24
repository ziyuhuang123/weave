#pragma once
// ============================================================================
// HR1 / RACE#1 : 8-wave cooperative FP8 GEMM1+GEMM2 for PURE-COMPUTE (8/0) CTAs
//                AMD CDNA3 (gfx942), ROCm 7.2, 512 threads / workgroup.
//
// WHY THIS EXISTS
// ---------------
// gemm_bk64_winner.hpp + weave_gemm_400t_v6.hpp are written for a 4-compute-wave
// group: they mask `tid = threadIdx.x & 255u` and take only HALF of the 64 KB LDS
// (`weave_lds_ptr` -> +32768 for tid>=256).  On an 8/0 CTA (cta >= N08+N44) that
// makes the CU run TWO INDEPENDENT 4-wave groups, each re-loading the same A rows
// (wn redundancy) and the same B weights (wm redundancy) from global memory, and
// each re-loading A once per n_half.  Measured consequence on qwen38max S=16384:
// RK03 GEMM = 12,757.6 us ~= 163 TFLOPS = 34% of the 478.4 TFLOPS FP8 peak.
//
// On an 8/0 CTA ALL 8 waves are compute, so:
//   * `s_barrier` is strictly per-workgroup -> legal and deadlock-free, because
//     every one of the 512 threads enters the same code path (proof below);
//   * the FULL 65536 B LDS is available -> A and B can be staged COOPERATIVELY
//     (each byte loaded from global exactly ONCE per CTA per K-step);
//   * the wave grid becomes 2(M) x 4(N) over ONE BM=128 x BN=128 tile instead of
//     two separate BM=128 x BN=64 tiles.
//
// DEADLOCK-FREEDOM PROOF (8/0 CTAs)
// --------------------------------
// 1. These entry points are only ever called under `cta >= V46_N08 + V46_N44`
//    (see hr1_dispatch_* and the megakernel guard).  On such a CTA v46_sp.hpp
//    assigns w0..7 ALL to compute; no wave runs a comm loop.
// 2. Every hr1_barrier() call site is at function scope inside a loop whose trip
//    count is a wave-UNIFORM value (num_k64 = K>>6, readfirstlane'd), or in the
//    straight-line prologue/epilogue.  There is no `if (tid < X)` around any
//    barrier and no early `return` between two barriers.
// 3. All 512 threads of the workgroup are resident: LDS = 50688 B <= 65536 B and
//    VGPRs <= 256 => exactly 1 workgroup / CU, 8 waves, 2 waves / SIMD.
// 4. 0/8 and 4/4 CTAs never execute this code, so they can never be required to
//    arrive at an 8/0 CTA's barrier.  s_barrier has no cross-workgroup semantics
//    on gfx942 (it is the workgroup barrier, not a grid barrier).
// => every s_barrier is reached by exactly the 8 waves of its own workgroup.
//
// LDS MAP (S1, gate+up, BM=128 BN=128 BK=64, 2 stages)
//   [0      , 16384)  A stages   2 x 128 rows x 64 B
//   [16384  , 49152)  B stages   2 x (8 n16 gate + 8 n16 up) x 1024 B
//   [49152  , 49664)  token ids  128 x u32
//   [49664  , 50688)  A scales   2 x 128 x f32
//   epilogue reuses [0, 32768) as 128 rows x 128 cols BF16 (pair-swizzled)
//   TOTAL 50688 B <= 65536 B.
//
// K-LOOP: 3 buffer loads issued per thread per step into a 2-deep register ring,
//         consumed one step later with `s_waitcnt vmcnt(3)` -> the 3 newest loads
//         stay in flight.  ZERO vmcnt(0) and ZERO software spin barriers in the
//         K-loop.  Exactly ONE hardware s_barrier per K-step.
// ============================================================================

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <type_traits>

#include "gemm_bk64_winner.hpp"
#include "weave_gemm_400t_v6.hpp"

// Compile-time ring index.  The K-loop is unrolled by 2 so every register-ring
// slot and every LDS stage offset is a CONSTANT.  With a runtime index
// (ra[k64&1]) the backend forms a loop-carried phi of the whole ring and the
// waitcnt-insertion pass cannot attribute a value to one specific buffer_load,
// so it conservatively emits `s_waitcnt vmcnt(0)` inside the K-loop.  Constant
// indices keep the ring in plain SSA and let vmcnt(4)/vmcnt(3) survive.
template <unsigned N> using hr1_ic = std::integral_constant<unsigned, N>;

#ifndef HR1_LDS_ARENA
#define HR1_LDS_ARENA g_v1262_phase_arena
#endif

// ---------------------------------------------------------------------------
// The one and only barrier primitive used by the 8-wave path.
// lgkmcnt(0) is emitted by the backend's waitcnt insertion pass before s_barrier;
// we additionally fence the LDS writes explicitly so the ordering is not left to
// the optimiser.  vmcnt is NOT drained here (0xc07f == vmcnt(63) expcnt(7)
// lgkmcnt(0) on gfx942), which is what keeps the global prefetch in flight.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void hr1_barrier()
{
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_s_barrier();
}

// Partial VMEM drain: keep the N most recent buffer loads in flight.
// gfx942 s_waitcnt SIMM16: vmcnt[3:0]=lo, vmcnt[15:14]=hi, expcnt[6:4], lgkmcnt[13:8].
// expcnt=7 (no wait), lgkmcnt=0, vmcnt hi=0  =>  imm = 0x0070 | N.
// S1 issues 4 loads/thread/step (A, Bgate, Bup, Ascale) -> vmcnt(4) = 0x0074.
// S2 issues 3 loads/thread/step (A, B, Ascale)          -> vmcnt(3) = 0x0073.
__device__ __forceinline__ void hr1_wait_vmcnt4()
{
    __builtin_amdgcn_s_waitcnt(0x0074);
}
__device__ __forceinline__ void hr1_wait_vmcnt3()
{
    __builtin_amdgcn_s_waitcnt(0x0073);
}
// S1: B+scale use a 1-deep ring and A(k+2) is issued last, so only the single
// newest load stays in flight.
__device__ __forceinline__ void hr1_wait_vmcnt1()
{
    __builtin_amdgcn_s_waitcnt(0x0071);
}

#define HR1_BM 128u
#define HR1_BN 128u
#define HR1_BK 64u

// Raw buffer STORES.  The epilogue publishes through an SGPR buffer resource
// instead of a 64-bit address_space(1) pointer: a generic global pointer costs
// 2 live VGPRs per output tensor, and those VGPRs would have to survive the
// whole K-loop (which already needs all 256), which is what caused the folded
// spills.  An SGPR rsrc + 32-bit voffset costs zero VGPRs.
__device__ void hr1_raw_buf_store_u2(WeaveRawU2 data, WeaveI32x4 rsrc,
                                     int32_t voffset, int32_t soffset, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.store.v2i32");
__device__ void hr1_raw_buf_store_f32(float data, WeaveI32x4 rsrc,
                                      int32_t voffset, int32_t soffset, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.store.f32");
__device__ int32_t hr1_raw_buf_load_i32(WeaveI32x4 rsrc,
                                        int32_t voffset, int32_t soffset, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.i32");

// Same numerics as weave_quant_bf16x8_store (per-128-group absmax, /240 scale,
// med3 clamp, cvt_pk_fp8_f32) but publishing via buffer resources.
__device__ __forceinline__ void hr1_quant_bf16x8_store_buf(
    uint4 raw, bool valid, unsigned q_lane, uint32_t dense_group,
    WeaveI32x4 q_rsrc, WeaveI32x4 qs_rsrc)
{
    const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
    float abs_max = 1.0e-10f;
    if (valid) {
        #pragma unroll
        for (unsigned i = 0u; i < 8u; ++i)
            abs_max = fmaxf(abs_max, fabsf(weave_bk64_bf16_bits_to_f32(h[i])));
    }
    #pragma unroll
    for (unsigned delta = 1u; delta < 16u; delta <<= 1u)
        abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffffffffffffull, abs_max, static_cast<int>(delta), 16));
    const float scale = abs_max * (1.0f / 240.0f);
    if (valid && q_lane == 0u)
        hr1_raw_buf_store_f32(scale, qs_rsrc, static_cast<int32_t>(dense_group * 4u), 0, 0);
    if (valid) {
        const float inv_scale = 1.0f / scale;
        const float fp8_max = 240.0f;
        uint16_t packed[4];
        #pragma unroll
        for (unsigned pair = 0u; pair < 4u; ++pair) {
            const float f0 = __builtin_amdgcn_fmed3f(weave_bk64_bf16_bits_to_f32(h[pair * 2u]) * inv_scale, fp8_max, -fp8_max);
            const float f1 = __builtin_amdgcn_fmed3f(weave_bk64_bf16_bits_to_f32(h[pair * 2u + 1u]) * inv_scale, fp8_max, -fp8_max);
            packed[pair] = __builtin_amdgcn_cvt_pk_fp8_f32(f0, f1, 0, 0);
        }
        WeaveRawU2 v;
        v[0] = static_cast<uint32_t>(packed[0]) | (static_cast<uint32_t>(packed[1]) << 16u);
        v[1] = static_cast<uint32_t>(packed[2]) | (static_cast<uint32_t>(packed[3]) << 16u);
        hr1_raw_buf_store_u2(v, q_rsrc, static_cast<int32_t>(dense_group * 128u + q_lane * 8u), 0, 0);
    }
}

// S1 LDS offsets
#define HR1_S1_A0     0u
#define HR1_S1_ASTR   8192u
#define HR1_S1_B0     16384u
#define HR1_S1_BSTR   16384u
#define HR1_S1_TOK    49152u
#define HR1_S1_SA     49664u
#define HR1_S1_TOTAL  50688u
// S2 LDS offsets (B is only 8 n16 blocks, no up matrix)
#define HR1_S2_A0     0u
#define HR1_S2_ASTR   8192u
#define HR1_S2_B0     16384u
#define HR1_S2_BSTR   8192u
#define HR1_S2_TOK    32768u
#define HR1_S2_SA     33280u
#define HR1_S2_TOTAL  34304u

// ---------------------------------------------------------------------------
// MFMA micro-kernel: 2 M-tiles x 2 N-tiles, gate+up fused, K=64.
// Uses only TWO temp accumulators (8 VGPRs) instead of the four (16 VGPRs) in
// weave_400t_s1_mhalf_k64: gate is folded before up reuses the same temps.
// Consecutive writes to one temp are 2 MFMAs apart (32 cycles) and the pk_fma
// consumer is 1 MFMA (16 cycles) after the last write, so no stall is introduced.
// The 8 saved VGPRs are what let S1 keep its 64 gate+up accumulator VGPRs plus
// the cooperative prefetch ring inside the 256-VGPR budget with ZERO spills.
// ---------------------------------------------------------------------------
template <unsigned kMBase>
__device__ __forceinline__ void hr1_s1_mhalf_k64(
    const WeaveLdsU8* rd_p,
    const WeaveLdsF32* sa_p,
    const WeaveLdsU8* bp,
    float sb_g,
    float sb_u,
    WeaveAcc4 (&gate_acc)[4][2],
    WeaveAcc4 (&up_acc)[4][2])
{
    // B is read from LDS INSIDE the helper, and GATE for BOTH rows is folded
    // before UP is loaded.  That makes bg[2] dead before bu[2] is born, so the
    // peak B live set is 4 VGPRs instead of 8 -- the 4 VGPRs that were the last
    // folded spills in S1.  B LDS read count is unchanged (4 per helper).
    const WeaveI64x2 av0 = weave_lds_load_i64x2(rd_p);
    const WeaveI64x2 av1 = weave_lds_load_i64x2(rd_p + 1024u);
    const float sab_g0 = sa_p[(kMBase + 0u) * 16u] * sb_g;
    const float sab_g1 = sa_p[(kMBase + 1u) * 16u] * sb_g;

    WeaveFloat4 t0 = {0, 0, 0, 0}, t1 = {0, 0, 0, 0};
    {
        const WeaveI64x2 bg[2] = {weave_lds_load_i64x2(bp), weave_lds_load_i64x2(bp + 1024u)};
        t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][0], av0[0], t0, 0, 0, 0);
        t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][0], av0[0], t1, 0, 0, 0);
        t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][1], av0[1], t0, 0, 0, 0);
        t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][1], av0[1], t1, 0, 0, 0);
        weave_acc_pk_fma4(gate_acc[kMBase + 0u][0], t0, sab_g0);
        weave_acc_pk_fma4(gate_acc[kMBase + 0u][1], t1, sab_g0);

        t0 = {0, 0, 0, 0}; t1 = {0, 0, 0, 0};
        t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][0], av1[0], t0, 0, 0, 0);
        t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][0], av1[0], t1, 0, 0, 0);
        t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[0][1], av1[1], t0, 0, 0, 0);
        t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg[1][1], av1[1], t1, 0, 0, 0);
        weave_acc_pk_fma4(gate_acc[kMBase + 1u][0], t0, sab_g1);
        weave_acc_pk_fma4(gate_acc[kMBase + 1u][1], t1, sab_g1);
    }
    {
        const float sab_u0 = sa_p[(kMBase + 0u) * 16u] * sb_u;
        const float sab_u1 = sa_p[(kMBase + 1u) * 16u] * sb_u;
        const WeaveI64x2 bu[2] = {weave_lds_load_i64x2(bp + 8192u), weave_lds_load_i64x2(bp + 9216u)};
        t0 = {0, 0, 0, 0}; t1 = {0, 0, 0, 0};
        t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][0], av0[0], t0, 0, 0, 0);
        t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][0], av0[0], t1, 0, 0, 0);
        t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][1], av0[1], t0, 0, 0, 0);
        t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][1], av0[1], t1, 0, 0, 0);
        weave_acc_pk_fma4(up_acc[kMBase + 0u][0], t0, sab_u0);
        weave_acc_pk_fma4(up_acc[kMBase + 0u][1], t1, sab_u0);

        t0 = {0, 0, 0, 0}; t1 = {0, 0, 0, 0};
        t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][0], av1[0], t0, 0, 0, 0);
        t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][0], av1[0], t1, 0, 0, 0);
        t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[0][1], av1[1], t0, 0, 0, 0);
        t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu[1][1], av1[1], t1, 0, 0, 0);
        weave_acc_pk_fma4(up_acc[kMBase + 1u][0], t0, sab_u1);
        weave_acc_pk_fma4(up_acc[kMBase + 1u][1], t1, sab_u1);
    }
}

// ---------------------------------------------------------------------------
// MFMA micro-kernel: 4 M-tiles x 2 N-tiles, single matrix (GEMM2), K=64.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void hr1_s2_pair_k64(
    const WeaveLdsU8* rd_p,
    const WeaveLdsF32* sa_p,
    const WeaveI64x2& b0,
    const WeaveI64x2& b1,
    float sb,
    WeaveAcc4 (&acc)[4][2])
{
    __builtin_amdgcn_sched_barrier(0);
    WeaveI64x2 av_cur = weave_lds_load_i64x2(rd_p);
    float sab_cur = sa_p[0] * sb;

    // Two direct-output accumulators instead of the old alternating A/B pair.
    // This releases eight VGPRs in the full 8-wave megakernel K loop while
    // preserving the same four M-tile MFMA order and accumulation precision.
    #pragma unroll
    for (unsigned mx = 0u; mx < 4u; ++mx) {
        WeaveI64x2 av_nxt;
        float sab_nxt = 0.0f;
        if (mx + 1u < 4u) {
            av_nxt = weave_lds_load_i64x2(rd_p + (mx + 1u) * 1024u);
            sab_nxt = sa_p[(mx + 1u) * 16u] * sb;
        }
        WeaveFloat4 t0 = {0, 0, 0, 0}, t1 = {0, 0, 0, 0};
        t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0[0], av_cur[0], t0, 0, 0, 0);
        t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1[0], av_cur[0], t1, 0, 0, 0);
        t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0[1], av_cur[1], t0, 0, 0, 0);
        t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1[1], av_cur[1], t1, 0, 0, 0);
        weave_acc_pk_fma4(acc[mx][0], t0, sab_cur);
        weave_acc_pk_fma4(acc[mx][1], t1, sab_cur);
        if (mx + 1u < 4u) { av_cur = av_nxt; sab_cur = sab_nxt; }
    }
}

// ===========================================================================
// Shared microbench argument block (also used by hr1_isa_probe.hip and the
// host launcher hr1_bench_host.cpp).
// ===========================================================================
#include "hr1_args.h"

// ===========================================================================
// GEMM1 (gate/up + SiLU + FP8 requant), 8-wave cooperative, BM=128 BN=128 BK=64
// Drop-in signature-compatible with weave_s1_400t_2x2<128u>.
// ===========================================================================
__device__ __forceinline__ void hr1_s1_8wave(
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
    (void)arena;
    WeaveLdsU8* const lds = reinterpret_cast<WeaveLdsU8*>(
        reinterpret_cast<uintptr_t>(HR1_LDS_ARENA));
    static_assert(HR1_S1_TOTAL <= 65536u, "HR1 S1 LDS budget");

    const unsigned tid  = threadIdx.x;                 // 0..511, NOT masked
    const unsigned wave = tid >> 6u;                   // 0..7
    const unsigned wm   = wave >> 2u;                  // 0..1  (M band, 64 rows)
    const unsigned wn   = wave & 3u;                   // 0..3  (N band, 32 cols)
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp  = lane >> 4u;

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

    // ---- token ids -------------------------------------------------------
    // The 64-bit generic pointer is converted to an SGPR buffer resource here.
    // A generic pointer otherwise survives most of the 256-VGPR K loop and was
    // one of the final folded spills in the production megakernel.
    WeaveLdsU32* const lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + HR1_S1_TOK);
    const WeaveI32x4 ids_rsrc = weave_make_buf_rsrc(s_ids_tile, HR1_BM * 4u);
    if (tid < HR1_BM) {
        const uint32_t tok = static_cast<uint32_t>(
            hr1_raw_buf_load_i32(ids_rsrc, static_cast<int32_t>(tid * 4u), 0, 0)) & 0xffffffu;
        lds_tok[tid] = (tid < valid_rows) ? tok : num_tokens;
    }
    hr1_barrier();

    // ---- cooperative A load geometry: thread t -> row t>>2, 16B chunk t&3 --
    // No `valid ? off : limit` select is needed: invalid rows carry
    // tok == num_tokens, so tok*K + ... is already >= the buffer range and the
    // raw buffer load returns 0 by itself.  Dropping the select keeps `a_limit`
    // / `sa_limit` in SGPRs instead of burning a VGPR on a v_cndmask operand
    // (that single VGPR was the last folded spill in the S1 prologue).
    const unsigned a_row  = tid >> 2u;                 // 0..127
    const unsigned a_chk  = tid & 3u;
    const uint32_t tok_a  = lds_tok[a_row];
    const uint32_t a_voff = tok_a * K + a_chk * 16u;
    WeaveLdsU8* const a_wr = lds + HR1_S1_A0 + a_row * 64u
        + ((a_chk ^ ((a_row >> 1u) & 3u)) * 16u);
    // tid>=128 mirrors row (tid&127): a real address, so every wave issues the
    // same number of loads, and the value is simply not stored.
    const uint32_t sa_voff = lds_tok[tid & (HR1_BM - 1u)] * num_k128 * 4u;

    // ---- cooperative B load geometry: thread t -> block t>>5 (0..15) ------
    //      blocks 0..7 = gate n16, blocks 8..15 = up n16; 32 B per thread.
    const unsigned b_blk = tid >> 5u;                  // 0..15
    const unsigned b_in  = (tid & 31u) * 32u;          // 0..992
    const uint32_t b_voff = (tile_n * 8u + (b_blk & 7u)) * b_nstride + b_in
        + ((b_blk >> 3u) * nk_bytes);
    WeaveLdsU8* const b_wr = lds + HR1_S1_B0 + b_blk * 1024u + b_in;

    // ---- per-wave MFMA read pointers -------------------------------------
    const unsigned r0 = wm * 64u + m_lane;
    WeaveLdsU8* const a_rd = lds + HR1_S1_A0 + r0 * 64u
        + ((k_grp ^ ((m_lane >> 1u) & 3u)) * 16u);
    const WeaveLdsF32* const sa_rd = reinterpret_cast<const WeaveLdsF32*>(lds + HR1_S1_SA) + r0;
    WeaveLdsU8* const b_rd = lds + HR1_S1_B0 + (wn * 2u) * 1024u + lane * 16u;

    WeaveAcc4 gate_acc[4][2], up_acc[4][2];
    #pragma unroll
    for (unsigned mx = 0u; mx < 4u; ++mx) {
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            weave_acc_zero(gate_acc[mx][nx]);
            weave_acc_zero(up_acc[mx][nx]);
        }
    }

    // ---- 2-deep register ring for the cooperative prefetch ---------------
    WeaveI64x2 ra[2];      // A: 16 B / thread
    // B (expert weights) is L2/MALL-resident and reused across M-tiles, so it
    // only needs a 1-deep ring; A comes from the uncached recv_payload and keeps
    // the 2-deep ring.  This costs 9 fewer live VGPRs than a 2-deep B ring,
    // which is exactly the headroom S1 needs (64 gate+up accumulator VGPRs).
    WeaveI64x2 rb0, rb1;         // B: 2 x 16 B / thread, single slot
    float rsa;                   // A scale, single slot

    // Issue order matters: A(k+2) is issued LAST so that `vmcnt(1)` leaves it
    // in flight while completing A(k+1), B(k+1) and sa(k+1).
    auto issue_b = [&](unsigned kk) {
        const uint32_t soff_b = kk * 1024u;
        rb0 = weave_buf_load_i64x2(b_rsrc, b_voff, soff_b);
        rb1 = weave_buf_load_i64x2(b_rsrc, b_voff + 16u, soff_b);
        // Issued by ALL 512 threads (OOB-clamped to sa_limit for tid>=128) so
        // the per-wave outstanding-load count is uniform.
        rsa = weave_buf_load_f32(sa_rsrc, sa_voff, (kk >> 1u) * 4u);
    };
    auto issue_a = [&](unsigned kk, auto slotv) {
        constexpr unsigned slot = decltype(slotv)::value;
        ra[slot] = weave_buf_load_i64x2(a_rsrc, a_voff, kk * 64u);
    };
    auto commit = [&](auto slotv, auto stv) {
        constexpr unsigned slot = decltype(slotv)::value;
        constexpr unsigned st = decltype(stv)::value;
        weave_lds_store_i64x2(a_wr + st * HR1_S1_ASTR, ra[slot]);
        weave_lds_store_i64x2(b_wr + st * HR1_S1_BSTR, rb0);
        weave_lds_store_i64x2(b_wr + st * HR1_S1_BSTR + 16u, rb1);
        if (tid < HR1_BM) {
            reinterpret_cast<WeaveLdsF32*>(lds + HR1_S1_SA + st * 512u)[tid] = rsa;
        }
    };

    // prologue: stages 0 and 1 resident before the first MFMA
    issue_b(0u); issue_a(0u, hr1_ic<0>{});
    __builtin_amdgcn_s_waitcnt(0x0000);   // vmcnt(0): prologue only, outside K-loop
    commit(hr1_ic<0>{}, hr1_ic<0>{});
    issue_b(1u); issue_a(1u, hr1_ic<1>{});
    __builtin_amdgcn_s_waitcnt(0x0000);   // vmcnt(0): prologue only, outside K-loop
    commit(hr1_ic<1>{}, hr1_ic<1>{});
    hr1_barrier();

    float sb_g_cur = bs_gate_s[0];
    float sb_u_cur = bs_gate_s[bs_up_stride];

    // One K-step.  CUR/NXT are compile-time so the ring stays in SSA.
    auto step = [&](unsigned k64, auto curv, auto nxtv) {
        constexpr unsigned cur = decltype(curv)::value;
        constexpr unsigned nxt = decltype(nxtv)::value;
        const unsigned kn = (k64 + 1u < num_k64) ? (k64 + 1u) : (num_k64 - 1u);
        // clamped so the loop stays branch-free and vmcnt(1) is valid on every
        // iteration, including the last two.
        const unsigned kl = (k64 + 2u < num_k64) ? (k64 + 2u) : (num_k64 - 1u);
        issue_b(kn);
        issue_a(kl, curv);

        // Weave scales are per 128 K elements; kn is a 64-element step, so both k64 halves of a
        // 128 group must read the SAME scale (the A-scale path above already uses kk>>1).
        const float sb_g_nxt = bs_gate_s[kn >> 1u];
        const float sb_u_nxt = bs_gate_s[bs_up_stride + (kn >> 1u)];

        WeaveLdsU8* const ap = a_rd + cur * HR1_S1_ASTR;
        const WeaveLdsF32* const sp = sa_rd + cur * 128u;
        WeaveLdsU8* const bp = b_rd + cur * HR1_S1_BSTR;
        hr1_s1_mhalf_k64<0u>(ap,        sp, bp, sb_g_cur, sb_u_cur, gate_acc, up_acc);
        hr1_s1_mhalf_k64<2u>(ap + 2048u, sp, bp, sb_g_cur, sb_u_cur, gate_acc, up_acc);

        // Drain everything except the single newest load (A for k64+2).
        hr1_wait_vmcnt1();
        commit(nxtv, nxtv);
        sb_g_cur = sb_g_nxt;
        sb_u_cur = sb_u_nxt;
        hr1_barrier();
    };

    // BRANCH-FREE loop tail.  num_k64 = K/64 is EVEN for every supported model
    // (H and I are all multiples of 128).  Expressing the trip count as
    // num_k64>>1 PAIRS (rather than `k64 < num_k64; k64 += 2`) tells the backend
    // there is no remainder, so it does not peel a cold last iteration that
    // would split the software pipeline and inflate the preheader live set.
    const unsigned n_pairs = num_k64 >> 1;
    #pragma clang loop unroll(disable)
    for (unsigned p = 0u; p < n_pairs; ++p) {
        const unsigned k64 = p * 2u;
        step(k64,      hr1_ic<0>{}, hr1_ic<1>{});
        step(k64 + 1u, hr1_ic<1>{}, hr1_ic<0>{});
    }

    // ---- epilogue: SiLU(gate)*up -> BF16 in LDS (pair-swizzled) -----------
    WeaveLdsU8* const epi = lds;   // reuses [0, 32768)
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
            const unsigned pair = ((col >> 3u) ^ (r & 7u)) * 16u + (col & 7u) * 2u;
            *reinterpret_cast<WeaveLdsU2*>(epi + r * 256u + pair) = make_uint2(p0, p1);
        }
    }
    hr1_barrier();

    // Buffer resources are built HERE (after the K-loop) so no 64-bit output
    // pointer has to stay live across the 256-VGPR K-loop body.
    const uint32_t q_limit  = __builtin_amdgcn_readfirstlane(route_rows * n_blocks * 128u);
    const uint32_t qs_limit = __builtin_amdgcn_readfirstlane(route_rows * n_blocks * 4u);
    const WeaveI32x4 q_rsrc  = weave_make_buf_rsrc(quant_out_base, q_limit);
    const WeaveI32x4 qs_rsrc = weave_make_buf_rsrc(quant_scale_base, qs_limit);

    // Re-derive the LDS row address from a value the compiler must treat as
    // modified here.  Without this barrier-side clobber clang hoists the
    // per-lane `epi + r*256 + ...` address above the epilogue barrier, keeping
    // one more VGPR live across the 256-VGPR K body (the last folded spill).
    unsigned q_seed = tid;
    asm volatile("" : "+v"(q_seed));
    const unsigned q_row = q_seed >> 4u;   // 0..31
    const unsigned q_lane = q_seed & 15u;
    #pragma clang loop unroll(disable)
    for (unsigned r0_i = 0u; r0_i < HR1_BM; r0_i += 32u) {
        const unsigned r = r0_i + q_row;
        const unsigned dense_row = row_offset + r;
        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
        uint4 raw = make_uint4(0, 0, 0, 0);
        if (valid) {
            raw = *reinterpret_cast<const WeaveLdsU4*>(epi + r * 256u + ((q_lane ^ (r & 7u)) * 16u));
        }
        hr1_quant_bf16x8_store_buf(raw, valid, q_lane, dense_row * n_blocks + tile_n,
                                   q_rsrc, qs_rsrc);
    }
    hr1_barrier();
}

// ===========================================================================
// GEMM2 (down proj + routing weight + BF16 store), 8-wave cooperative.
// Drop-in signature-compatible with weave_s2_400t_2x2<128u>.
// ===========================================================================
__device__ __forceinline__ void hr1_s2_8wave_compute(
    const int32_t* __restrict__ s2_ids_tile,
    int32_t expert_id,
    const uint8_t* __restrict__ a_base,
    const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base,
    const float* __restrict__ b_scale_base,
    const float* __restrict__ weights_tile,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    (void)arena;
    WeaveLdsU8* const lds = reinterpret_cast<WeaveLdsU8*>(
        reinterpret_cast<uintptr_t>(HR1_LDS_ARENA));
    static_assert(HR1_S2_TOTAL <= 65536u, "HR1 S2 LDS budget");

    const unsigned tid  = threadIdx.x;
    const unsigned wave = tid >> 6u;
    const unsigned wm   = wave >> 2u;
    const unsigned wn   = wave & 3u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp  = lane >> 4u;

    const unsigned num_k128 = __builtin_amdgcn_readfirstlane(K >> 7u);
    const unsigned num_k64  = __builtin_amdgcn_readfirstlane(K >> 6u);
    const unsigned n_blocks = __builtin_amdgcn_readfirstlane(N >> 7u);
    const unsigned exp_u    = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(expert_id));
    const uint32_t nk_bytes = __builtin_amdgcn_readfirstlane(N * K);
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(exp_u) * nk_bytes;
    const float* bs_ptr = b_scale_base
        + (static_cast<uint64_t>(exp_u) * n_blocks + tile_n) * num_k128;
    WeaveConstF32As4* bs_s = weave_make_as4_f32(bs_ptr);

    const uint32_t a_limit  = __builtin_amdgcn_readfirstlane(route_rows * K);
    const uint32_t sa_limit = __builtin_amdgcn_readfirstlane(route_rows * num_k128 * 4u);
    const WeaveI32x4 a_rsrc  = weave_make_buf_rsrc(a_base, a_limit);
    const WeaveI32x4 sa_rsrc = weave_make_buf_rsrc(a_scale_base, sa_limit);
    const WeaveI32x4 b_rsrc  = weave_make_buf_rsrc(b_exp, nk_bytes);
    const uint32_t b_nstride = __builtin_amdgcn_readfirstlane(num_k64 * 1024u);

    WeaveLdsU32* const lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + HR1_S2_TOK);
    if (tid < HR1_BM) {
        lds_tok[tid] = (tid < valid_rows)
            ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu)
            : route_rows;
    }
    hr1_barrier();

    const unsigned a_row = tid >> 2u;
    const unsigned a_chk = tid & 3u;
    const uint32_t tok_a = lds_tok[a_row];
    const uint32_t a_voff = (tok_a < route_rows) ? (tok_a * K + a_chk * 16u) : a_limit;
    WeaveLdsU8* const a_wr = lds + HR1_S2_A0 + a_row * 64u
        + ((a_chk ^ ((a_row >> 1u) & 3u)) * 16u);
    const uint32_t sa_voff = (tid < HR1_BM)
        ? ((lds_tok[tid] < route_rows) ? (lds_tok[tid] * num_k128 * 4u) : sa_limit)
        : sa_limit;

    // B: 8192 B / 512 threads = 16 B each; block = tid>>6 (0..7)
    const unsigned b_blk = tid >> 6u;
    const unsigned b_in  = (tid & 63u) * 16u;
    const uint32_t b_voff = (tile_n * 8u + b_blk) * b_nstride + b_in;
    WeaveLdsU8* const b_wr = lds + HR1_S2_B0 + b_blk * 1024u + b_in;

    const unsigned r0 = wm * 64u + m_lane;
    WeaveLdsU8* const a_rd = lds + HR1_S2_A0 + r0 * 64u
        + ((k_grp ^ ((m_lane >> 1u) & 3u)) * 16u);
    const WeaveLdsF32* const sa_rd = reinterpret_cast<const WeaveLdsF32*>(lds + HR1_S2_SA) + r0;
    WeaveLdsU8* const b_rd = lds + HR1_S2_B0 + (wn * 2u) * 1024u + lane * 16u;

    WeaveAcc4 acc[4][2];
    #pragma unroll
    for (unsigned mx = 0u; mx < 4u; ++mx) {
        weave_acc_zero(acc[mx][0]);
        weave_acc_zero(acc[mx][1]);
    }

    WeaveI64x2 ra[2], rb[2];
    float rsa[2];
    auto issue = [&](unsigned kk, auto slotv) {
        constexpr unsigned slot = decltype(slotv)::value;
        ra[slot]  = weave_buf_load_i64x2(a_rsrc, a_voff, kk * 64u);
        rb[slot]  = weave_buf_load_i64x2(b_rsrc, b_voff, kk * 1024u);
        // issued by all 512 threads (OOB-clamped) -> exactly 3 outstanding/wave.
        rsa[slot] = weave_buf_load_f32(sa_rsrc, sa_voff, (kk >> 1u) * 4u);
    };
    auto commit = [&](auto slotv, auto stv) {
        constexpr unsigned slot = decltype(slotv)::value;
        constexpr unsigned st = decltype(stv)::value;
        weave_lds_store_i64x2(a_wr + st * HR1_S2_ASTR, ra[slot]);
        weave_lds_store_i64x2(b_wr + st * HR1_S2_BSTR, rb[slot]);
        if (tid < HR1_BM) {
            reinterpret_cast<WeaveLdsF32*>(lds + HR1_S2_SA + st * 512u)[tid] = rsa[slot];
        }
    };

    issue(0u, hr1_ic<0>{});
    issue(1u, hr1_ic<1>{});
    __builtin_amdgcn_s_waitcnt(0x0000);   // prologue only
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
        // Same per-128 scale contract as S1: kn is a 64-element step, the table is per 128.
        const float sb_nxt = bs_s[kn >> 1u];

        WeaveLdsU8* const ap = a_rd + cur * HR1_S2_ASTR;
        const WeaveLdsF32* const sp = sa_rd + cur * 128u;
        WeaveLdsU8* const bp = b_rd + cur * HR1_S2_BSTR;
        const WeaveI64x2 b0 = weave_lds_load_i64x2(bp);
        const WeaveI64x2 b1 = weave_lds_load_i64x2(bp + 1024u);
        hr1_s2_pair_k64(ap, sp, b0, b1, sb_cur, acc);

        hr1_wait_vmcnt3();
        commit(nxtv, nxtv);
        sb_cur = sb_nxt;
        hr1_barrier();
    };
    // BRANCH-FREE loop tail (num_k64 = K/64 is even for every supported model).
    const unsigned n_pairs = num_k64 >> 1;
    #pragma clang loop unroll(disable)
    for (unsigned p = 0u; p < n_pairs; ++p) {
        const unsigned k64 = p * 2u;
        step(k64,      hr1_ic<0>{}, hr1_ic<1>{});
        step(k64 + 1u, hr1_ic<1>{}, hr1_ic<0>{});
    }

    WeaveLdsU8* const epi = lds;
    #pragma unroll
    for (unsigned mx = 0u; mx < 4u; ++mx) {
        const unsigned r = wm * 64u + mx * 16u + m_lane;
        const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const unsigned col = (wn * 2u + nx) * 16u + k_grp * 4u;
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].lo[0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].lo[1] * weight)) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].hi[0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx].hi[1] * weight)) << 16u);
            const unsigned pair = ((col >> 3u) ^ (r & 7u)) * 16u + (col & 7u) * 2u;
            *reinterpret_cast<WeaveLdsU2*>(epi + r * 256u + pair) = make_uint2(p0, p1);
        }
    }
    hr1_barrier();
}

// Store half of GEMM2, deliberately a separate function.  The S1F table bases
// and the dense output base are first materialised HERE, after the 256-VGPR
// K-loop.  Keeping those 64-bit addresses out of the K-loop live range is what
// removes the spilled-pointer pressure seen when the HR1 body was folded into
// the production megakernel.  The BF16+weight epilogue is already resident in
// LDS at [0,32768); GEMM2's token ids remain at HR1_S2_TOK.
__device__ __forceinline__ void hr1_s2_8wave_store(
    uint16_t* __restrict__ c_out_base,
    unsigned row_offset,
    unsigned valid_rows,
    unsigned tile_n,
    unsigned route_rows,
    unsigned N,
    const unsigned* row_dir = nullptr,
    const uint64_t* peer_tab = nullptr,
    const uint64_t* node_tab = nullptr,
    unsigned node_row_base = 0u,
    unsigned my_node = 0u)
{
    WeaveLdsU8* const lds = reinterpret_cast<WeaveLdsU8*>(
        reinterpret_cast<uintptr_t>(HR1_LDS_ARENA));
    const unsigned tid  = threadIdx.x;
    // The BF16 epilogue tile is 128 rows x 128 columns = 256 B per row.  512 threads cover it as
    // 32 rows x (16 lanes x 8 values): c_lane must span all 16 16-byte lanes of the 256-B row.
    // (The HR1 prototype used 64 rows x 8 lanes = 64 columns and silently dropped half the tile.)
    const unsigned c_row = tid >> 4u;      // 0..31
    const unsigned c_lane = tid & 15u;     // 0..15
    const WeaveLdsU32* const lds_tok =
        reinterpret_cast<const WeaveLdsU32*>(lds + HR1_S2_TOK);
    const WeaveLdsU8* const epi = lds;
    #pragma clang loop unroll(disable)
    for (unsigned r0_i = 0u; r0_i < HR1_BM; r0_i += 32u) {
        const unsigned r = r0_i + c_row;
        const uint32_t tok_m = lds_tok[r];
        if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
            const uint4 u = *reinterpret_cast<const WeaveLdsU4*>(
                epi + r * 256u + ((c_lane ^ (r & 7u)) * 16u));
            uint16_t* direct_p = weave_direct_row_dst(row_dir, peer_tab, node_tab,
                                                      node_row_base, my_node, tok_m, N);
            if (direct_p != nullptr) {
                direct_p += static_cast<uint64_t>(tile_n) * 128u + c_lane * 8u;
                weave_gstore_u4_wt(direct_p, u);
            } else {
                weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N
                                + tile_n * 128u + c_lane * 8u, u);
            }
        }
    }
    hr1_barrier();
}
