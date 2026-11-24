#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "ck/utility/type_convert.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#ifndef WEAVE_BK64_ARENA_DECLARED
#define WEAVE_BK64_ARENA_DECLARED
__shared__ __align__(16) unsigned char g_v1262_phase_arena[65536];
#endif

using WeaveBk64LocalU32 = unsigned __attribute__((address_space(3)));
using WeaveLdsU8  = unsigned char __attribute__((address_space(3)));
using WeaveLdsU16 = uint16_t      __attribute__((address_space(3)));
using WeaveLdsU32 = uint32_t      __attribute__((address_space(3)));
using WeaveLdsF32 = float         __attribute__((address_space(3)));
using WeaveLdsU2  = uint2         __attribute__((address_space(3)));
using WeaveLdsU4  = uint4         __attribute__((address_space(3)));

using WeaveRawU4 = uint32_t __attribute__((ext_vector_type(4)));
using WeaveRawU2 = uint32_t __attribute__((ext_vector_type(2)));
using WeaveI64x2 = int64_t  __attribute__((ext_vector_type(2)));
using WeaveLdsI64x2     = const WeaveI64x2 __attribute__((address_space(3)));
using WeaveGlobRawU4    = const WeaveRawU4 __attribute__((address_space(1)));
using WeaveGlobI64x2    = const WeaveI64x2 __attribute__((address_space(1)));
using WeaveGlobOutRawU4 = WeaveRawU4       __attribute__((address_space(1)));
using WeaveGlobOutRawU2 = WeaveRawU2       __attribute__((address_space(1)));

__device__ __forceinline__ uint4 weave_gload_u4(const void* p) {
    const WeaveRawU4 r = *reinterpret_cast<WeaveGlobRawU4*>(reinterpret_cast<uintptr_t>(p));
    return make_uint4(r.x, r.y, r.z, r.w);
}
__device__ __forceinline__ WeaveI64x2 weave_gload_i64x2(const void* p) {
    return *reinterpret_cast<WeaveGlobI64x2*>(reinterpret_cast<uintptr_t>(p));
}
__device__ __forceinline__ WeaveI64x2 weave_lds_load_i64x2(const WeaveLdsU8* p) {
    return *reinterpret_cast<WeaveLdsI64x2*>(p);
}
__device__ __forceinline__ void weave_gstore_u4(void* p, uint4 v) {
    *reinterpret_cast<WeaveGlobOutRawU4*>(reinterpret_cast<uintptr_t>(p)) = WeaveRawU4{v.x, v.y, v.z, v.w};
}
__device__ __forceinline__ void weave_gstore_u2(void* p, uint2 v) {
    *reinterpret_cast<WeaveGlobOutRawU2*>(reinterpret_cast<uintptr_t>(p)) = WeaveRawU2{v.x, v.y};
}
template <typename T>
__device__ __forceinline__ const T __attribute__((address_space(1)))* weave_as1_arr(const T* p) {
    return reinterpret_cast<const T __attribute__((address_space(1)))*>(reinterpret_cast<uintptr_t>(p));
}
template <typename T>
__device__ __forceinline__ T __attribute__((address_space(1)))* weave_as1_arr_mut(T* p) {
    return reinterpret_cast<T __attribute__((address_space(1)))*>(reinterpret_cast<uintptr_t>(p));
}

// ----------------------------------------------------------------------------
// v59 L3 (V54_S1_BYPASS_MALL): GEMM1 publishes stage1_out / stage1_scale with a
// system-scope write-through store (sc0 sc1) so the bytes land in the shared MALL
// instead of staying dirty in the producing XCD's private 4 MB L2, and GEMM2 reads
// them back with sc1 loads (bypassing any stale line in the consuming XCD's L2).
// With the macro at 0 every helper below collapses to the pristine v46 code.
// ----------------------------------------------------------------------------
#ifndef V54_S1_BYPASS_MALL
#define V54_S1_BYPASS_MALL 0
#endif
// v60: keep the write-through (sc0 sc1) stores and every buffer-resource SC1 load, but do
// NOT use the hand-written flat-address global_load sc1 in the BM<=32 direct-register body
// (it faults).  Set to 1 only to reproduce the v59 L3 crash.
#ifndef V60_L3_SMALL_RAW_SC1
#define V60_L3_SMALL_RAW_SC1 0
#endif
// raw buffer-load cache policy `SC1` (CPol::SC1 on gfx940+); identical value to V44_SC1.
#define WEAVE_CPOL_SC1 16
#define WEAVE_AS1_GPTR(T, p) ((__attribute__((address_space(1))) T*)(p))
#if V54_S1_BYPASS_MALL
__device__ __forceinline__ void weave_gstore_u2_wt(void* p, uint2 v) {
    __hip_atomic_store(WEAVE_AS1_GPTR(uint64_t, p),
                       static_cast<uint64_t>(v.x) | (static_cast<uint64_t>(v.y) << 32u),
                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void weave_gstore_f32_wt(void* p, float v) {
    __hip_atomic_store(WEAVE_AS1_GPTR(float, p), v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ uint4 weave_gload_u4_sc1(const void* p) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    WeaveRawU4 r;
    asm volatile("global_load_dwordx4 %0, %1, off sc1\n" : "=v"(r) : "v"(a) : "memory");
    return make_uint4(r.x, r.y, r.z, r.w);
}
__device__ __forceinline__ float weave_gload_f32_sc1(const float* p) {
    const uint64_t a = reinterpret_cast<uint64_t>(p);
    float r;
    asm volatile("global_load_dword %0, %1, off sc1\n" : "=v"(r) : "v"(a) : "memory");
    return r;
}
#endif

__device__ __forceinline__ WeaveLdsU8* weave_lds_ptr(unsigned char* arena)
{
    (void)arena;
    const unsigned grp_off = (threadIdx.x >= 256u) ? 32768u : 0u;
    WeaveLdsU8* p = reinterpret_cast<WeaveLdsU8*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + grp_off));
    asm volatile("" : "+v"(p));
    return p;
}

#ifndef V62_BAR_DIAG
#define V62_BAR_DIAG 0
#endif
#ifndef V62_BAR_MODE
#define V62_BAR_MODE 0
#endif
// V62 BAR_MODE>=1: one 32-bit LDS phase counter per 4-wave group (65424 / 65428).  Each lane-0 does one
// ds_add_u32 and derives the phase target from the returned old value; the wait is a single dword poll.
// BAR_MODE>=2: the GEMM2 BN256 K loop uses split arrive/wait so global load issue hides the spin.
// BAR_MODE>=3: same split schedule for the GEMM1 1x4 K loop.
__device__ __forceinline__ unsigned weave_4wave_bar2_off()
{
    const unsigned raw_tid = threadIdx.x;
    return (raw_tid >= 256u) ? 65428u : 65424u;
}
__device__ __forceinline__ unsigned weave_4wave_arrive_65440()
{
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    unsigned tgt = 0u;
    if ((threadIdx.x & 63u) == 0u) {
        WeaveBk64LocalU32* ctr = reinterpret_cast<WeaveBk64LocalU32*>(
            reinterpret_cast<uintptr_t>(g_v1262_phase_arena + weave_4wave_bar2_off()));
        const unsigned old = atomicAdd(ctr, 1u);
        tgt = ((old >> 2u) + 1u) << 2u;
    }
    return __builtin_amdgcn_readfirstlane(tgt);
}
__device__ __forceinline__ void weave_4wave_wait_65440(unsigned tgt)
{
    if ((threadIdx.x & 63u) == 0u) {
        volatile WeaveBk64LocalU32* ctr = reinterpret_cast<volatile WeaveBk64LocalU32*>(
            reinterpret_cast<uintptr_t>(g_v1262_phase_arena + weave_4wave_bar2_off()));
        while (*ctr < tgt) { asm volatile("s_nop 0" ::: "memory"); }
    }
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    asm volatile("" ::: "memory");
}
__device__ __forceinline__ void weave_4wave_lds_barrier_65440(unsigned char* arena)
{
    (void)arena;
#if V62_BAR_MODE
    weave_4wave_wait_65440(weave_4wave_arrive_65440());
    return;
#endif
#if V62_BAR_DIAG
    // DIAGNOSTIC ONLY: no cross-wave rendezvous (timing ceiling probe; results are not valid).
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    asm volatile("" ::: "memory");
    return;
#endif
    asm volatile("" ::: "memory");
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    const unsigned raw_tid = threadIdx.x;
    if ((raw_tid & 63u) == 0u) {
        const unsigned wave_idx = (raw_tid >> 6u) & 3u;
        const unsigned bar_off = (raw_tid >= 256u) ? 65456u : 65440u;
        volatile WeaveBk64LocalU32* counters = reinterpret_cast<volatile WeaveBk64LocalU32*>(
            reinterpret_cast<uintptr_t>(g_v1262_phase_arena + bar_off));
        const unsigned next_count = counters[wave_idx] + 1u;
        counters[wave_idx] = next_count;
        __builtin_amdgcn_s_waitcnt(0xc07f);
        for (;;) {
            asm volatile("" ::: "memory");
            const uint4 c = *reinterpret_cast<const WeaveLdsU4*>(
                reinterpret_cast<uintptr_t>(g_v1262_phase_arena + bar_off));
            const unsigned m01 = (c.x < c.y) ? c.x : c.y; const unsigned m23 = (c.z < c.w) ? c.z : c.w; if (((m01 < m23) ? m01 : m23) >= next_count) break;
            asm volatile("s_nop 0" ::: "memory");
        }
    }
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    asm volatile("" ::: "memory");
}

#ifndef WEAVE_4WAVE_BARRIER_DEFINED
#define WEAVE_4WAVE_BARRIER_DEFINED
__device__ __forceinline__ void weave_4wave_barrier_65440(unsigned char* arena)
{
    (void)arena;
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    __threadfence_block();
    const unsigned raw_tid = threadIdx.x;
    if ((raw_tid & 63u) == 0u) {
        const unsigned wave_idx = (raw_tid >> 6u) & 3u;
        const unsigned bar_off = (raw_tid >= 256u) ? 65456u : 65440u;
        volatile WeaveBk64LocalU32* counters = reinterpret_cast<volatile WeaveBk64LocalU32*>(
            reinterpret_cast<uintptr_t>(g_v1262_phase_arena + bar_off));
        const unsigned next_count = counters[wave_idx] + 1u;
        counters[wave_idx] = next_count;
        __builtin_amdgcn_s_waitcnt(0xc07f);
        for (;;) {
            asm volatile("" ::: "memory");
            const uint4 c = *reinterpret_cast<const WeaveLdsU4*>(
                reinterpret_cast<uintptr_t>(g_v1262_phase_arena + bar_off));
            const unsigned m01 = (c.x < c.y) ? c.x : c.y; const unsigned m23 = (c.z < c.w) ? c.z : c.w; if (((m01 < m23) ? m01 : m23) >= next_count) break;
            asm volatile("s_nop 0" ::: "memory");
        }
    }
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    __threadfence_block();
}

__device__ __forceinline__ void weave_4wave_barrier_65440()
{
    weave_4wave_barrier_65440(g_v1262_phase_arena);
}

__device__ __forceinline__ void weave_4wave_barrier_reset_65440(unsigned char* arena)
{
    (void)arena;
    const unsigned raw_tid = threadIdx.x;
    if ((raw_tid & 63u) == 0u) {
        const unsigned bar_off = (raw_tid >= 256u) ? 65456u : 65440u;
        reinterpret_cast<volatile WeaveBk64LocalU32*>(
            reinterpret_cast<uintptr_t>(g_v1262_phase_arena + bar_off))[(raw_tid >> 6u) & 3u] = 0u;
        if ((raw_tid & 63u) == 0u)
            *reinterpret_cast<volatile WeaveBk64LocalU32*>(
                reinterpret_cast<uintptr_t>(g_v1262_phase_arena + ((raw_tid >= 256u) ? 65428u : 65424u))) = 0u;
    }
    __threadfence_block();
}
#endif

using WeaveFloat4 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float weave_bk64_silu(float value)
{
    float result = 0.0f;
    ck::tensor_operation::element_wise::Silu{}(result, value);
    return result;
}

__device__ __forceinline__ uint16_t weave_bk64_f32_to_bf16_bits(float value)
{
    return ck::bit_cast<uint16_t>(ck::type_convert<ck::bhalf_t>(value));
}

__device__ __forceinline__ float weave_bk64_bf16_bits_to_f32(uint16_t bits)
{
    return ck::type_convert<float>(ck::bit_cast<ck::bhalf_t>(bits));
}

__device__ __forceinline__ int64_t weave_pack_i64(uint32_t lo, uint32_t hi)
{
    return static_cast<int64_t>(lo) | (static_cast<int64_t>(hi) << 32);
}

__device__ __forceinline__ void weave_quant_bf16x8_store(
    uint4 raw,
    bool valid,
    unsigned q_lane,
    uint64_t dense_group,
    uint8_t* __restrict__ quant_out_base,
    float* __restrict__ quant_scale_base)
{
    const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
    float abs_max = 1.0e-10f;
    if (valid) {
        #pragma unroll
        for (unsigned i = 0u; i < 8u; ++i) {
            abs_max = fmaxf(abs_max, fabsf(weave_bk64_bf16_bits_to_f32(h[i])));
        }
    }
    #pragma unroll
    for (unsigned delta = 1u; delta < 16u; delta <<= 1u) {
        abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffffffffffffull, abs_max, static_cast<int>(delta), 16));
    }
    const float fp8_max = 240.0f;
    const float scale = abs_max * (1.0f / 240.0f);
#if V54_S1_BYPASS_MALL
    if (valid && q_lane == 0u) weave_gstore_f32_wt(quant_scale_base + dense_group, scale);
#else
    if (valid && q_lane == 0u) weave_as1_arr_mut(quant_scale_base)[dense_group] = scale;
#endif
    if (valid) {
        const float inv_scale = 1.0f / scale;
        uint16_t packed[4];
        #pragma unroll
        for (unsigned pair = 0u; pair < 4u; ++pair) {
            const float f0 = __builtin_amdgcn_fmed3f(weave_bk64_bf16_bits_to_f32(h[pair * 2u]) * inv_scale, fp8_max, -fp8_max);
            const float f1 = __builtin_amdgcn_fmed3f(weave_bk64_bf16_bits_to_f32(h[pair * 2u + 1u]) * inv_scale, fp8_max, -fp8_max);
            packed[pair] = __builtin_amdgcn_cvt_pk_fp8_f32(f0, f1, 0, 0);
        }
#if V54_S1_BYPASS_MALL
        weave_gstore_u2_wt(quant_out_base + dense_group * 128ull + q_lane * 8u, make_uint2(
            static_cast<uint32_t>(packed[0]) | (static_cast<uint32_t>(packed[1]) << 16u),
            static_cast<uint32_t>(packed[2]) | (static_cast<uint32_t>(packed[3]) << 16u)));
#else
        weave_gstore_u2(quant_out_base + dense_group * 128ull + q_lane * 8u, make_uint2(
            static_cast<uint32_t>(packed[0]) | (static_cast<uint32_t>(packed[1]) << 16u),
            static_cast<uint32_t>(packed[2]) | (static_cast<uint32_t>(packed[3]) << 16u)));
#endif
    }
}

// ============================================================================
// Low-VGPR K=64 MFMA Step Reading Both A & B from LDS (0 Bank Conflicts!)
// ============================================================================
template <unsigned kMXBegin, unsigned kMXEnd, unsigned kNXDL>
__device__ __forceinline__ void weave_lds_s1_k64_mfma_range(
    const WeaveLdsU8* lds_a,
    const WeaveLdsU8* lds_bg,
    const WeaveLdsU8* lds_bu,
    const WeaveLdsF32* lds_sa,
    unsigned row_base_w,
    unsigned b_chunk_base,
    unsigned lane,
    unsigned m_lane,
    unsigned k_grp,
    float sb_g,
    float sb_u,
    WeaveFloat4 (*gate_acc)[kNXDL],
    WeaveFloat4 (*up_acc)[kNXDL])
{
    static_assert((kNXDL % 2u) == 0u);
    constexpr unsigned kCountM = kMXEnd - kMXBegin;
    int64_t a0_m[kCountM], a1_m[kCountM];
    float sab_g_m[kCountM], sab_u_m[kCountM];
    #pragma unroll
    for (unsigned i = 0u; i < kCountM; ++i) {
        const unsigned mx = kMXBegin + i;
        const unsigned r = row_base_w + mx * 16u + m_lane;
        const unsigned swz = k_grp ^ ((r >> 1u) & 3u);
        const uint4 av = *reinterpret_cast<const WeaveLdsU4*>(lds_a + r * 64u + swz * 16u);
        const float sa = lds_sa[r];
        a0_m[i] = weave_pack_i64(av.x, av.y);
        a1_m[i] = weave_pack_i64(av.z, av.w);
        sab_g_m[i] = sa * sb_g;
        sab_u_m[i] = sa * sb_u;
    }

    #pragma unroll
    for (unsigned n0 = 0u; n0 < kNXDL; n0 += 2u) {
        const uint4 bgv0 = *reinterpret_cast<const WeaveLdsU4*>(lds_bg + (b_chunk_base + n0 + 0u) * 1024u + lane * 16u);
        const uint4 bgv1 = *reinterpret_cast<const WeaveLdsU4*>(lds_bg + (b_chunk_base + n0 + 1u) * 1024u + lane * 16u);
        const uint4 buv0 = *reinterpret_cast<const WeaveLdsU4*>(lds_bu + (b_chunk_base + n0 + 0u) * 1024u + lane * 16u);
        const uint4 buv1 = *reinterpret_cast<const WeaveLdsU4*>(lds_bu + (b_chunk_base + n0 + 1u) * 1024u + lane * 16u);

        const int64_t bg0_lo = weave_pack_i64(bgv0.x, bgv0.y);
        const int64_t bg0_hi = weave_pack_i64(bgv0.z, bgv0.w);
        const int64_t bg1_lo = weave_pack_i64(bgv1.x, bgv1.y);
        const int64_t bg1_hi = weave_pack_i64(bgv1.z, bgv1.w);
        const int64_t bu0_lo = weave_pack_i64(buv0.x, buv0.y);
        const int64_t bu0_hi = weave_pack_i64(buv0.z, buv0.w);
        const int64_t bu1_lo = weave_pack_i64(buv1.x, buv1.y);
        const int64_t bu1_hi = weave_pack_i64(buv1.z, buv1.w);

        #pragma unroll
        for (unsigned i = 0u; i < kCountM; ++i) {
            const unsigned mx = kMXBegin + i;
            WeaveFloat4 tg0 = {0,0,0,0}, tg1 = {0,0,0,0}, tu0 = {0,0,0,0}, tu1 = {0,0,0,0};
            tg0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg0_lo, a0_m[i], tg0, 0, 0, 0);
            tg1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg1_lo, a0_m[i], tg1, 0, 0, 0);
            tu0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu0_lo, a0_m[i], tu0, 0, 0, 0);
            tu1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu1_lo, a0_m[i], tu1, 0, 0, 0);

            tg0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg0_hi, a1_m[i], tg0, 0, 0, 0);
            tg1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg1_hi, a1_m[i], tg1, 0, 0, 0);
            tu0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu0_hi, a1_m[i], tu0, 0, 0, 0);
            tu1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu1_hi, a1_m[i], tu1, 0, 0, 0);

            gate_acc[mx][n0 + 0u] += tg0 * sab_g_m[i];
            gate_acc[mx][n0 + 1u] += tg1 * sab_g_m[i];
            up_acc[mx][n0 + 0u]   += tu0 * sab_u_m[i];
            up_acc[mx][n0 + 1u]   += tu1 * sab_u_m[i];
        }
    }
}

template <unsigned kMXBegin, unsigned kMXEnd, unsigned kNXDL>
__device__ __forceinline__ void weave_lds_s2_k64_mfma_range(
    const WeaveLdsU8* lds_a,
    const WeaveLdsU8* lds_b,
    const WeaveLdsF32* lds_sa,
    unsigned row_base_w,
    unsigned b_chunk_base,
    unsigned lane,
    unsigned m_lane,
    unsigned k_grp,
    float sb,
    WeaveFloat4 (*acc)[kNXDL])
{
    static_assert((kNXDL % 2u) == 0u);
    #pragma unroll
    for (unsigned n0 = 0u; n0 < kNXDL; n0 += 2u) {
        const uint4 bv0 = *reinterpret_cast<const WeaveLdsU4*>(lds_b + (b_chunk_base + n0 + 0u) * 1024u + lane * 16u);
        const uint4 bv1 = *reinterpret_cast<const WeaveLdsU4*>(lds_b + (b_chunk_base + n0 + 1u) * 1024u + lane * 16u);
        const int64_t b0_lo = weave_pack_i64(bv0.x, bv0.y);
        const int64_t b0_hi = weave_pack_i64(bv0.z, bv0.w);
        const int64_t b1_lo = weave_pack_i64(bv1.x, bv1.y);
        const int64_t b1_hi = weave_pack_i64(bv1.z, bv1.w);

        #pragma unroll
        for (unsigned mx = kMXBegin; mx < kMXEnd; ++mx) {
            const unsigned r = row_base_w + mx * 16u + m_lane;
            const unsigned swz = k_grp ^ ((r >> 1u) & 3u);
            const uint4 av = *reinterpret_cast<const WeaveLdsU4*>(lds_a + r * 64u + swz * 16u);
            const float sa = lds_sa[r];
            const int64_t a0 = weave_pack_i64(av.x, av.y);
            const int64_t a1 = weave_pack_i64(av.z, av.w);

            WeaveFloat4 t0 = {0,0,0,0}, t1 = {0,0,0,0};
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0_lo, a0, t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1_lo, a0, t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0_hi, a1, t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1_hi, a1, t1, 0, 0, 0);

            const float sab = sa * sb;
            acc[mx][n0 + 0u] += t0 * sab;
            acc[mx][n0 + 1u] += t1 * sab;
        }
    }
}
// ============================================================================
// TIER 1 & TIER 2: BM = 16 and BM = 32 (1x4 Wave Layout, ZERO K-Loop Barriers!)
// Each wave reads A directly into registers (shared CU L1 TCP hit across 4 waves)
// -> 0 ds_read/ds_write and 0 weave_4wave_lds_barrier_65440 in the entire K-loop!
// ============================================================================
template <unsigned kMXDL>
__device__ __forceinline__ void weave_s1_direct_reg_k128_mfma(
    const uint4 (&av0)[kMXDL],
    const uint4 (&av1)[kMXDL],
    const uint4 (&bg0)[2],
    const uint4 (&bu0)[2],
    const uint4 (&bg1)[2],
    const uint4 (&bu1)[2],
    const float (&sa_m)[kMXDL],
    float sb_g,
    float sb_u,
    WeaveFloat4 (&gate_acc)[kMXDL][2],
    WeaveFloat4 (&up_acc)[kMXDL][2])
{
    const int64_t bg00_lo = weave_pack_i64(bg0[0].x, bg0[0].y);
    const int64_t bg00_hi = weave_pack_i64(bg0[0].z, bg0[0].w);
    const int64_t bg01_lo = weave_pack_i64(bg0[1].x, bg0[1].y);
    const int64_t bg01_hi = weave_pack_i64(bg0[1].z, bg0[1].w);
    const int64_t bu00_lo = weave_pack_i64(bu0[0].x, bu0[0].y);
    const int64_t bu00_hi = weave_pack_i64(bu0[0].z, bu0[0].w);
    const int64_t bu01_lo = weave_pack_i64(bu0[1].x, bu0[1].y);
    const int64_t bu01_hi = weave_pack_i64(bu0[1].z, bu0[1].w);

    const int64_t bg10_lo = weave_pack_i64(bg1[0].x, bg1[0].y);
    const int64_t bg10_hi = weave_pack_i64(bg1[0].z, bg1[0].w);
    const int64_t bg11_lo = weave_pack_i64(bg1[1].x, bg1[1].y);
    const int64_t bg11_hi = weave_pack_i64(bg1[1].z, bg1[1].w);
    const int64_t bu10_lo = weave_pack_i64(bu1[0].x, bu1[0].y);
    const int64_t bu10_hi = weave_pack_i64(bu1[0].z, bu1[0].w);
    const int64_t bu11_lo = weave_pack_i64(bu1[1].x, bu1[1].y);
    const int64_t bu11_hi = weave_pack_i64(bu1[1].z, bu1[1].w);

    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const int64_t a00 = weave_pack_i64(av0[mx].x, av0[mx].y);
        const int64_t a01 = weave_pack_i64(av0[mx].z, av0[mx].w);
        const int64_t a10 = weave_pack_i64(av1[mx].x, av1[mx].y);
        const int64_t a11 = weave_pack_i64(av1[mx].z, av1[mx].w);

        WeaveFloat4 tg0 = {0,0,0,0}, tg1 = {0,0,0,0}, tu0 = {0,0,0,0}, tu1 = {0,0,0,0};
        tg0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg00_lo, a00, tg0, 0, 0, 0);
        tg1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg01_lo, a00, tg1, 0, 0, 0);
        tu0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu00_lo, a00, tu0, 0, 0, 0);
        tu1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu01_lo, a00, tu1, 0, 0, 0);

        tg0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg00_hi, a01, tg0, 0, 0, 0);
        tg1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg01_hi, a01, tg1, 0, 0, 0);
        tu0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu00_hi, a01, tu0, 0, 0, 0);
        tu1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu01_hi, a01, tu1, 0, 0, 0);

        tg0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg10_lo, a10, tg0, 0, 0, 0);
        tg1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg11_lo, a10, tg1, 0, 0, 0);
        tu0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu10_lo, a10, tu0, 0, 0, 0);
        tu1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu11_lo, a10, tu1, 0, 0, 0);

        tg0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg10_hi, a11, tg0, 0, 0, 0);
        tg1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg11_hi, a11, tg1, 0, 0, 0);
        tu0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu10_hi, a11, tu0, 0, 0, 0);
        tu1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu11_hi, a11, tu1, 0, 0, 0);

        const float sab_g = sa_m[mx] * sb_g;
        const float sab_u = sa_m[mx] * sb_u;
        gate_acc[mx][0] += tg0 * sab_g;
        gate_acc[mx][1] += tg1 * sab_g;
        up_acc[mx][0]   += tu0 * sab_u;
        up_acc[mx][1]   += tu1 * sab_u;
    }
}

template <unsigned kMXDL>
__device__ __forceinline__ void weave_s1_direct_reg_half_mfma(
    const uint4 (&av)[kMXDL],
    const uint4 (&bg)[2],
    const uint4 (&bu)[2],
    const float (&sa_m)[kMXDL],
    float sb_g,
    float sb_u,
    WeaveFloat4 (&gate_acc)[kMXDL][2],
    WeaveFloat4 (&up_acc)[kMXDL][2])
{
    const int64_t bg0_lo = weave_pack_i64(bg[0].x, bg[0].y);
    const int64_t bg0_hi = weave_pack_i64(bg[0].z, bg[0].w);
    const int64_t bg1_lo = weave_pack_i64(bg[1].x, bg[1].y);
    const int64_t bg1_hi = weave_pack_i64(bg[1].z, bg[1].w);
    const int64_t bu0_lo = weave_pack_i64(bu[0].x, bu[0].y);
    const int64_t bu0_hi = weave_pack_i64(bu[0].z, bu[0].w);
    const int64_t bu1_lo = weave_pack_i64(bu[1].x, bu[1].y);
    const int64_t bu1_hi = weave_pack_i64(bu[1].z, bu[1].w);

    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const int64_t a0 = weave_pack_i64(av[mx].x, av[mx].y);
        const int64_t a1 = weave_pack_i64(av[mx].z, av[mx].w);

        WeaveFloat4 tg0 = {0,0,0,0}, tg1 = {0,0,0,0}, tu0 = {0,0,0,0}, tu1 = {0,0,0,0};
        tg0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg0_lo, a0, tg0, 0, 0, 0);
        tg1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg1_lo, a0, tg1, 0, 0, 0);
        tu0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu0_lo, a0, tu0, 0, 0, 0);
        tu1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu1_lo, a0, tu1, 0, 0, 0);

        tg0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg0_hi, a1, tg0, 0, 0, 0);
        tg1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg1_hi, a1, tg1, 0, 0, 0);
        tu0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu0_hi, a1, tu0, 0, 0, 0);
        tu1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu1_hi, a1, tu1, 0, 0, 0);

        const float sab_g = sa_m[mx] * sb_g;
        const float sab_u = sa_m[mx] * sb_u;
        gate_acc[mx][0] += tg0 * sab_g;
        gate_acc[mx][1] += tg1 * sab_g;
        up_acc[mx][0]   += tu0 * sab_u;
        up_acc[mx][1]   += tu1 * sab_u;
    }
}

template <unsigned kBM>
__device__ __forceinline__ void weave_s1_small_1x4_v10(
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
    constexpr unsigned kTokenCacheOffset = 16384u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokenCacheOffset);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu) : num_tokens;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint64_t expert_b_bytes = static_cast<uint64_t>(N) * K * 2ull;
    const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_b_bytes;
    const uint8_t* b_up_exp = b_gate_exp + static_cast<uint64_t>(N) * K;
    const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * num_k128;
    const float* bs_gate_row = b_scale_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_bs_stride + static_cast<uint64_t>(tile_n) * num_k128;
    const float* bs_up_row = bs_gate_row + static_cast<uint64_t>(n_blocks) * num_k128;

    const unsigned n16_base = tile_n * 8u + wave_idx * 2u;

    WeaveFloat4 gate_acc[kMXDL][2];
    WeaveFloat4 up_acc[kMXDL][2];
    uint32_t a_m_off[kMXDL];
    uint32_t sa_m_off[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        gate_acc[mx][0] = {0,0,0,0}; gate_acc[mx][1] = {0,0,0,0};
        up_acc[mx][0]   = {0,0,0,0}; up_acc[mx][1]   = {0,0,0,0};
        const unsigned r = mx * 16u + m_lane;
        const uint32_t tok = lds_tok[r];
        const bool ok = (r < valid_rows) && (tok < num_tokens);
        a_m_off[mx]  = ok ? (tok * K + k_grp * 16u) : 0xffffffffu;
        sa_m_off[mx] = ok ? (tok * num_k128) : 0xffffffffu;
    }

    // Prologue: k128 = 0, Half 0
    uint4 av0_cur[kMXDL];
    float sa_cur[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        av0_cur[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx]) : make_uint4(0,0,0,0);
        sa_cur[mx]  = (sa_m_off[mx] != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_m_off[mx]] : 0.0f;
    }
    uint4 bg0[2], bu0[2];
    #pragma unroll
    for (unsigned nx = 0u; nx < 2u; ++nx) {
        const uint64_t boff = (static_cast<uint64_t>(n16_base + nx) * num_k64 * 64ull + lane) * 16ull;
        bg0[nx] = weave_gload_u4(b_gate_exp + boff);
        bu0[nx] = weave_gload_u4(b_up_exp + boff);
    }
    float sb_gate_cur = weave_as1_arr(bs_gate_row)[0];
    float sb_up_cur   = weave_as1_arr(bs_up_row)[0];

    // 100% BARRIER-FREE K-LOOP! Waves 0..3 stream independently at full HBM3 speed!
    #pragma clang loop unroll(disable)
    for (unsigned k128 = 0u; k128 + 1u < num_k128; ++k128) {
        const unsigned k64_odd = k128 * 2u + 1u;
        const unsigned next_k128 = k128 + 1u;
        const uint32_t ak_odd  = k128 * 128u + 64u;
        const uint32_t ak_next = next_k128 * 128u;

        uint4 av1_cur[kMXDL], bg1[2], bu1[2];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av1_cur[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak_odd) : make_uint4(0,0,0,0);
        }
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const uint64_t boff1 = ((static_cast<uint64_t>(n16_base + nx) * num_k64 + k64_odd) * 64ull + lane) * 16ull;
            bg1[nx] = weave_gload_u4(b_gate_exp + boff1);
            bu1[nx] = weave_gload_u4(b_up_exp + boff1);
        }

        weave_s1_direct_reg_half_mfma<kMXDL>(av0_cur, bg0, bu0, sa_cur, sb_gate_cur, sb_up_cur, gate_acc, up_acc);

        uint4 av0_next[kMXDL];
        float sa_next[kMXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av0_next[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak_next) : make_uint4(0,0,0,0);
            sa_next[mx]  = (sa_m_off[mx] != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_m_off[mx] + next_k128] : 0.0f;
        }
        const unsigned next_k64_even = next_k128 * 2u;
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const uint64_t boff_n = ((static_cast<uint64_t>(n16_base + nx) * num_k64 + next_k64_even) * 64ull + lane) * 16ull;
            bg0[nx] = weave_gload_u4(b_gate_exp + boff_n);
            bu0[nx] = weave_gload_u4(b_up_exp + boff_n);
        }
        const float sb_gate_next = weave_as1_arr(bs_gate_row)[next_k128];
        const float sb_up_next   = weave_as1_arr(bs_up_row)[next_k128];

        weave_s1_direct_reg_half_mfma<kMXDL>(av1_cur, bg1, bu1, sa_cur, sb_gate_cur, sb_up_cur, gate_acc, up_acc);

        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av0_cur[mx] = av0_next[mx];
            sa_cur[mx]  = sa_next[mx];
        }
        sb_gate_cur = sb_gate_next;
        sb_up_cur   = sb_up_next;
    }

    {
        const unsigned k128 = num_k128 - 1u;
        const unsigned k64_odd = k128 * 2u + 1u;
        const uint32_t ak_odd = k128 * 128u + 64u;
        uint4 av1_cur[kMXDL], bg1[2], bu1[2];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av1_cur[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak_odd) : make_uint4(0,0,0,0);
        }
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const uint64_t boff1 = ((static_cast<uint64_t>(n16_base + nx) * num_k64 + k64_odd) * 64ull + lane) * 16ull;
            bg1[nx] = weave_gload_u4(b_gate_exp + boff1);
            bu1[nx] = weave_gload_u4(b_up_exp + boff1);
        }
        weave_s1_direct_reg_half_mfma<kMXDL>(av0_cur, bg0, bu0, sa_cur, sb_gate_cur, sb_up_cur, gate_acc, up_acc);
        weave_s1_direct_reg_half_mfma<kMXDL>(av1_cur, bg1, bu1, sa_cur, sb_gate_cur, sb_up_cur, gate_acc, up_acc);
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
                weave_bk64_silu(gate_acc[mx][nx][0]) * up_acc[mx][nx][0])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][1]) * up_acc[mx][nx][1])) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][2]) * up_acc[mx][nx][2])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][3]) * up_acc[mx][nx][3])) << 16u);
            *reinterpret_cast<WeaveLdsU2*>(lds_q + r * 128u + col) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + q_row;
        const unsigned dense_row = row_offset + r;
        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
        uint4 raw = make_uint4(0,0,0,0);
        if (valid) { raw = *reinterpret_cast<const WeaveLdsU4*>(lds_q + r * 128u + q_lane * 8u); }
        weave_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

template <unsigned kBM>
__device__ __forceinline__ void weave_s2_small_1x4_v10(
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
    constexpr unsigned kTokenCacheOffset = 16384u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokenCacheOffset);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu) : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * N * K;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * n_blocks + tile_n) * num_k128;

    const unsigned n16_base = tile_n * 8u + wave_idx * 2u;

    WeaveFloat4 acc[kMXDL][2];
    uint32_t a_m_off[kMXDL];
    uint32_t sa_m_off[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        acc[mx][0] = {0,0,0,0}; acc[mx][1] = {0,0,0,0};
        const unsigned r = mx * 16u + m_lane;
        const uint32_t tok = lds_tok[r];
        const bool ok = (r < valid_rows) && (tok < route_rows);
        a_m_off[mx]  = ok ? (tok * K + k_grp * 16u) : 0xffffffffu;
        sa_m_off[mx] = ok ? (tok * num_k128) : 0xffffffffu;
    }

    uint4 av0_cur[kMXDL];
    float sa_cur[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        av0_cur[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx]) : make_uint4(0,0,0,0);
        sa_cur[mx]  = (sa_m_off[mx] != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_m_off[mx]] : 0.0f;
    }
    uint4 b0_cur[2];
    #pragma unroll
    for (unsigned nx = 0u; nx < 2u; ++nx) {
        const uint64_t boff = (static_cast<uint64_t>(n16_base + nx) * num_k64 * 64ull + lane) * 16ull;
        b0_cur[nx] = weave_gload_u4(b_exp + boff);
    }
    float sb_cur = weave_as1_arr(bs_row)[0];

    auto s2_half = [&](const uint4 (&av)[kMXDL], const uint4 (&bc)[2], const float (&sam)[kMXDL], float sbc) {
        const int64_t b0_lo = weave_pack_i64(bc[0].x, bc[0].y);
        const int64_t b0_hi = weave_pack_i64(bc[0].z, bc[0].w);
        const int64_t b1_lo = weave_pack_i64(bc[1].x, bc[1].y);
        const int64_t b1_hi = weave_pack_i64(bc[1].z, bc[1].w);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const int64_t a0 = weave_pack_i64(av[mx].x, av[mx].y);
            const int64_t a1 = weave_pack_i64(av[mx].z, av[mx].w);
            WeaveFloat4 t0 = {0,0,0,0}, t1 = {0,0,0,0};
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0_lo, a0, t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1_lo, a0, t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0_hi, a1, t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1_hi, a1, t1, 0, 0, 0);
            const float sab = sam[mx] * sbc;
            acc[mx][0] += t0 * sab;
            acc[mx][1] += t1 * sab;
        }
    };

    #pragma clang loop unroll(disable)
    for (unsigned k128 = 0u; k128 + 1u < num_k128; ++k128) {
        const unsigned k64_odd = k128 * 2u + 1u;
        const unsigned next_k128 = k128 + 1u;
        const uint32_t ak_odd  = k128 * 128u + 64u;
        const uint32_t ak_next = next_k128 * 128u;

        uint4 av1_cur[kMXDL], b1_cur[2];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av1_cur[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak_odd) : make_uint4(0,0,0,0);
        }
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const uint64_t boff1 = ((static_cast<uint64_t>(n16_base + nx) * num_k64 + k64_odd) * 64ull + lane) * 16ull;
            b1_cur[nx] = weave_gload_u4(b_exp + boff1);
        }

        s2_half(av0_cur, b0_cur, sa_cur, sb_cur);

        uint4 av0_next[kMXDL];
        float sa_next[kMXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av0_next[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak_next) : make_uint4(0,0,0,0);
            sa_next[mx]  = (sa_m_off[mx] != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_m_off[mx] + next_k128] : 0.0f;
        }
        const unsigned next_k64_even = next_k128 * 2u;
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const uint64_t boff_n = ((static_cast<uint64_t>(n16_base + nx) * num_k64 + next_k64_even) * 64ull + lane) * 16ull;
            b0_cur[nx] = weave_gload_u4(b_exp + boff_n);
        }
        const float sb_next = weave_as1_arr(bs_row)[next_k128];

        s2_half(av1_cur, b1_cur, sa_cur, sb_cur);

        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av0_cur[mx] = av0_next[mx];
            sa_cur[mx]  = sa_next[mx];
        }
        sb_cur = sb_next;
    }

    {
        const unsigned k128 = num_k128 - 1u;
        const unsigned k64_odd = k128 * 2u + 1u;
        const uint32_t ak_odd = k128 * 128u + 64u;
        uint4 av1_cur[kMXDL], b1_cur[2];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av1_cur[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak_odd) : make_uint4(0,0,0,0);
        }
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const uint64_t boff1 = ((static_cast<uint64_t>(n16_base + nx) * num_k64 + k64_odd) * 64ull + lane) * 16ull;
            b1_cur[nx] = weave_gload_u4(b_exp + boff1);
        }
        s2_half(av0_cur, b0_cur, sa_cur, sb_cur);
        s2_half(av1_cur, b1_cur, sa_cur, sb_cur);
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
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][1] * weight)) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][2] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][3] * weight)) << 16u);
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
            weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n * 128u + c_lane * 8u, make_uint4(u0.x, u0.y, u1.x, u1.y));
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

// ============================================================================
template <unsigned kBM>
__device__ __forceinline__ void weave_s1_small_1x4_deep(
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
    constexpr unsigned kTokenCacheOffset = 16384u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokenCacheOffset);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu) : num_tokens;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint64_t expert_b_bytes = static_cast<uint64_t>(N) * K * 2ull;
    const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_b_bytes;
    const uint8_t* b_up_exp = b_gate_exp + static_cast<uint64_t>(N) * K;
    const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * num_k128;
    const float* bs_gate_row = b_scale_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_bs_stride + static_cast<uint64_t>(tile_n) * num_k128;
    const float* bs_up_row = bs_gate_row + static_cast<uint64_t>(n_blocks) * num_k128;

    const unsigned n16_base = tile_n * 8u + wave_idx * 2u;

    WeaveFloat4 gate_acc[kMXDL][2];
    WeaveFloat4 up_acc[kMXDL][2];
    uint32_t a_m_off[kMXDL];
    uint32_t sa_m_off[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        gate_acc[mx][0] = {0,0,0,0}; gate_acc[mx][1] = {0,0,0,0};
        up_acc[mx][0]   = {0,0,0,0}; up_acc[mx][1]   = {0,0,0,0};
        const unsigned r = mx * 16u + m_lane;
        const uint32_t tok = lds_tok[r];
        const bool ok = (r < valid_rows) && (tok < num_tokens);
        a_m_off[mx]  = ok ? (tok * K + k_grp * 16u) : 0xffffffffu;
        sa_m_off[mx] = ok ? (tok * num_k128) : 0xffffffffu;
    }

    auto load_s1_half = [&](unsigned k64_idx, uint4 (&av)[kMXDL], uint4 (&bg)[2], uint4 (&bu)[2]) {
        const uint32_t ak = k64_idx * 64u;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak) : make_uint4(0,0,0,0);
        }
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const uint64_t boff = ((static_cast<uint64_t>(n16_base + nx) * num_k64 + k64_idx) * 64ull + lane) * 16ull;
            bg[nx] = weave_gload_u4(b_gate_exp + boff);
            bu[nx] = weave_gload_u4(b_up_exp + boff);
        }
    };

    auto load_s1_scales = [&](unsigned k128_idx, float (&sa)[kMXDL], float& sbg, float& sbu) {
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            sa[mx] = (sa_m_off[mx] != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_m_off[mx] + k128_idx] : 0.0f;
        }
        sbg = weave_as1_arr(bs_gate_row)[k128_idx];
        sbu = weave_as1_arr(bs_up_row)[k128_idx];
    };

    if (num_k128 >= 2u) {
        // 2-k128 (K=256, 4 k64 stages = >3 MB across GPU!) deep software pipeline
        uint4 av0_c[kMXDL], av1_c[kMXDL], bg0_c[2], bu0_c[2], bg1_c[2], bu1_c[2];
        uint4 av0_n[kMXDL], av1_n[kMXDL], bg0_n[2], bu0_n[2], bg1_n[2], bu1_n[2];
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
            weave_s1_direct_reg_half_mfma<kMXDL>(av0_c, bg0_c, bu0_c, sa_c, sbg_c, sbu_c, gate_acc, up_acc);
            load_s1_half(nn * 2u + 0u, av0_c, bg0_c, bu0_c);
            float sa_nn[kMXDL], sbg_nn, sbu_nn;
            load_s1_scales(nn, sa_nn, sbg_nn, sbu_nn);

            weave_s1_direct_reg_half_mfma<kMXDL>(av1_c, bg1_c, bu1_c, sa_c, sbg_c, sbu_c, gate_acc, up_acc);
            load_s1_half(nn * 2u + 1u, av1_c, bg1_c, bu1_c);

            #pragma unroll
            for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                const uint4 t0 = av0_c[mx]; av0_c[mx] = av0_n[mx]; av0_n[mx] = t0;
                const uint4 t1 = av1_c[mx]; av1_c[mx] = av1_n[mx]; av1_n[mx] = t1;
                sa_c[mx] = sa_n[mx]; sa_n[mx] = sa_nn[mx];
            }
            #pragma unroll
            for (unsigned nx = 0u; nx < 2u; ++nx) {
                const uint4 g0 = bg0_c[nx]; bg0_c[nx] = bg0_n[nx]; bg0_n[nx] = g0;
                const uint4 u0 = bu0_c[nx]; bu0_c[nx] = bu0_n[nx]; bu0_n[nx] = u0;
                const uint4 g1 = bg1_c[nx]; bg1_c[nx] = bg1_n[nx]; bg1_n[nx] = g1;
                const uint4 u1 = bu1_c[nx]; bu1_c[nx] = bu1_n[nx]; bu1_n[nx] = u1;
            }
            sbg_c = sbg_n; sbu_c = sbu_n;
            sbg_n = sbg_nn; sbu_n = sbu_nn;
        }

        weave_s1_direct_reg_k128_mfma<kMXDL>(av0_c, av1_c, bg0_c, bu0_c, bg1_c, bu1_c, sa_c, sbg_c, sbu_c, gate_acc, up_acc);
        weave_s1_direct_reg_k128_mfma<kMXDL>(av0_n, av1_n, bg0_n, bu0_n, bg1_n, bu1_n, sa_n, sbg_n, sbu_n, gate_acc, up_acc);
    } else {
        uint4 av0[kMXDL], av1[kMXDL], bg0[2], bu0[2], bg1[2], bu1[2];
        float sa[kMXDL], sbg, sbu;
        load_s1_half(0u, av0, bg0, bu0);
        load_s1_scales(0u, sa, sbg, sbu);
        load_s1_half(1u, av1, bg1, bu1);
        weave_s1_direct_reg_half_mfma<kMXDL>(av0, bg0, bu0, sa, sbg, sbu, gate_acc, up_acc);
        weave_s1_direct_reg_half_mfma<kMXDL>(av1, bg1, bu1, sa, sbg, sbu, gate_acc, up_acc);
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
                weave_bk64_silu(gate_acc[mx][nx][0]) * up_acc[mx][nx][0])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][1]) * up_acc[mx][nx][1])) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][2]) * up_acc[mx][nx][2])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][3]) * up_acc[mx][nx][3])) << 16u);
            *reinterpret_cast<WeaveLdsU2*>(lds_q + r * 128u + col) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + q_row;
        const unsigned dense_row = row_offset + r;
        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
        uint4 raw = make_uint4(0,0,0,0);
        if (valid) { raw = *reinterpret_cast<const WeaveLdsU4*>(lds_q + r * 128u + q_lane * 8u); }
        weave_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

template <unsigned kBM>
__device__ __forceinline__ void weave_s2_small_1x4_deep(
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
    constexpr unsigned kTokenCacheOffset = 16384u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokenCacheOffset);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu) : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * N * K;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * n_blocks + tile_n) * num_k128;

    const unsigned n16_base = tile_n * 8u + wave_idx * 2u;

    WeaveFloat4 acc[kMXDL][2];
    uint32_t a_m_off[kMXDL];
    uint32_t sa_m_off[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        acc[mx][0] = {0,0,0,0}; acc[mx][1] = {0,0,0,0};
        const unsigned r = mx * 16u + m_lane;
        const uint32_t tok = lds_tok[r];
        const bool ok = (r < valid_rows) && (tok < route_rows);
        a_m_off[mx]  = ok ? (tok * K + k_grp * 16u) : 0xffffffffu;
        sa_m_off[mx] = ok ? (tok * num_k128) : 0xffffffffu;
    }

    auto s2_half = [&](const uint4 (&av)[kMXDL], const uint4 (&bc)[2], const float (&sam)[kMXDL], float sbc) {
        const int64_t b0_lo = weave_pack_i64(bc[0].x, bc[0].y);
        const int64_t b0_hi = weave_pack_i64(bc[0].z, bc[0].w);
        const int64_t b1_lo = weave_pack_i64(bc[1].x, bc[1].y);
        const int64_t b1_hi = weave_pack_i64(bc[1].z, bc[1].w);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const int64_t a0 = weave_pack_i64(av[mx].x, av[mx].y);
            const int64_t a1 = weave_pack_i64(av[mx].z, av[mx].w);
            WeaveFloat4 t0 = {0,0,0,0}, t1 = {0,0,0,0};
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0_lo, a0, t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1_lo, a0, t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0_hi, a1, t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1_hi, a1, t1, 0, 0, 0);
            const float sab = sam[mx] * sbc;
            acc[mx][0] += t0 * sab;
            acc[mx][1] += t1 * sab;
        }
    };

    auto load_s2_half = [&](unsigned k64_idx, uint4 (&av)[kMXDL], uint4 (&bc)[2]) {
        const uint32_t ak = k64_idx * 64u;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak) : make_uint4(0,0,0,0);
        }
        #pragma unroll
        for (unsigned nx = 0u; nx < 2u; ++nx) {
            const uint64_t boff = ((static_cast<uint64_t>(n16_base + nx) * num_k64 + k64_idx) * 64ull + lane) * 16ull;
            bc[nx] = weave_gload_u4(b_exp + boff);
        }
    };

    auto load_s2_scales = [&](unsigned k128_idx, float (&sa)[kMXDL], float& sb) {
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            sa[mx] = (sa_m_off[mx] != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_m_off[mx] + k128_idx] : 0.0f;
        }
        sb = weave_as1_arr(bs_row)[k128_idx];
    };

    if (num_k128 >= 3u) {
        // 3-k128 (K=384, 6 k64 halves = >3.84 MB in flight across GPU!) deep software pipeline
        uint4 av0_0[kMXDL], av1_0[kMXDL], b0_0[2], b1_0[2];
        uint4 av0_1[kMXDL], av1_1[kMXDL], b0_1[2], b1_1[2];
        uint4 av0_2[kMXDL], av1_2[kMXDL], b0_2[2], b1_2[2];
        float sa_0[kMXDL], sa_1[kMXDL], sa_2[kMXDL], sb_0, sb_1, sb_2;

        load_s2_half(0u, av0_0, b0_0); load_s2_scales(0u, sa_0, sb_0); load_s2_half(1u, av1_0, b1_0);
        load_s2_half(2u, av0_1, b0_1); load_s2_scales(1u, sa_1, sb_1); load_s2_half(3u, av1_1, b1_1);
        load_s2_half(4u, av0_2, b0_2); load_s2_scales(2u, sa_2, sb_2); load_s2_half(5u, av1_2, b1_2);

        #pragma clang loop unroll(disable)
        for (unsigned k128 = 0u; k128 + 3u < num_k128; ++k128) {
            const unsigned nn = k128 + 3u;
            s2_half(av0_0, b0_0, sa_0, sb_0);
            load_s2_half(nn * 2u + 0u, av0_0, b0_0);
            float sa_nn[kMXDL], sb_nn;
            load_s2_scales(nn, sa_nn, sb_nn);

            s2_half(av1_0, b1_0, sa_0, sb_0);
            load_s2_half(nn * 2u + 1u, av1_0, b1_0);

            #pragma unroll
            for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                const uint4 t0 = av0_0[mx]; av0_0[mx] = av0_1[mx]; av0_1[mx] = av0_2[mx]; av0_2[mx] = t0;
                const uint4 t1 = av1_0[mx]; av1_0[mx] = av1_1[mx]; av1_1[mx] = av1_2[mx]; av1_2[mx] = t1;
                sa_0[mx] = sa_1[mx]; sa_1[mx] = sa_2[mx]; sa_2[mx] = sa_nn[mx];
            }
            #pragma unroll
            for (unsigned nx = 0u; nx < 2u; ++nx) {
                const uint4 q0 = b0_0[nx]; b0_0[nx] = b0_1[nx]; b0_1[nx] = b0_2[nx]; b0_2[nx] = q0;
                const uint4 q1 = b1_0[nx]; b1_0[nx] = b1_1[nx]; b1_1[nx] = b1_2[nx]; b1_2[nx] = q1;
            }
            sb_0 = sb_1; sb_1 = sb_2; sb_2 = sb_nn;
        }

        weave_s2_direct_reg_k128_mfma<kMXDL, 2u>(av0_0, av1_0, b0_0, b1_0, sa_0, sb_0, acc);
        weave_s2_direct_reg_k128_mfma<kMXDL, 2u>(av0_1, av1_1, b0_1, b1_1, sa_1, sb_1, acc);
        weave_s2_direct_reg_k128_mfma<kMXDL, 2u>(av0_2, av1_2, b0_2, b1_2, sa_2, sb_2, acc);
    } else {
        #pragma clang loop unroll(disable)
        for (unsigned k128 = 0u; k128 < num_k128; ++k128) {
            uint4 av0[kMXDL], av1[kMXDL], b0[2], b1[2];
            float sa[kMXDL], sb;
            load_s2_half(k128 * 2u + 0u, av0, b0);
            load_s2_scales(k128, sa, sb);
            load_s2_half(k128 * 2u + 1u, av1, b1);
            weave_s2_direct_reg_k128_mfma<kMXDL, 2u>(av0, av1, b0, b1, sa, sb, acc);
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
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][1] * weight)) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][2] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][3] * weight)) << 16u);
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
            weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n * 128u + c_lane * 8u, make_uint4(u0.x, u0.y, u1.x, u1.y));
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

// ============================================================================
// TIER 3 & TIER 4: TRUE SINGLE-PASS BM = 64 & BM = 128 (2x2 Wave Layout!)
// Computes FULL kBM x 128 in ONE SINGLE PASS (0 n_half loop, 1.0x A load, 1.0x B load)!
// ============================================================================
__device__ __forceinline__ void weave_s1_true_bm64_singlepass(
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
    constexpr unsigned kBM = 64u;
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kMXDL = kBM / 32u;          // 2 for BM=64 (64 acc VGPRs), 4 for BM=128 (128 acc VGPRs)
    constexpr unsigned kNXDL = 4u;                 // 64 N-cols per wave -> 128 N-cols across 2x2 waves!
    constexpr unsigned kAPasses = kBM / 64u;       // 1 for BM=64, 2 for BM=128
    constexpr unsigned kAStageBytes = kBM * 64u;   // 4096 or 8192
    constexpr unsigned kBStageBytes = 16384u;      // 8 gate (8192B) + 8 up (8192B)
    constexpr unsigned kBBaseOff = 2u * kAStageBytes; // 8192 or 16384; + 32768 = 40960 or 49152
    constexpr unsigned kSaBaseOff = 49152u;
    constexpr unsigned kTokOff = 50176u;

    const unsigned mw = wave_idx >> 1u;
    const unsigned nw = wave_idx & 1u;
    const unsigned row_base_w = mw * (kBM / 2u);
    const unsigned b_chunk_w  = nw * 4u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu) : num_tokens;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint64_t expert_b_bytes = static_cast<uint64_t>(N) * K * 2ull;
    const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_b_bytes;
    const uint8_t* b_up_exp = b_gate_exp + static_cast<uint64_t>(N) * K;
    const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * num_k128;
    const float* bs_gate_row = b_scale_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_bs_stride + static_cast<uint64_t>(tile_n) * num_k128;
    const float* bs_up_row = bs_gate_row + static_cast<uint64_t>(n_blocks) * num_k128;

    const unsigned a_r64 = tid >> 2u;
    const unsigned a_c16 = tid & 3u;
    uint32_t a_roff[kAPasses];
    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 64u + a_r64;
        const uint32_t tok = lds_tok[r];
        a_roff[ap] = ((r < valid_rows) && (tok < num_tokens)) ? (tok * K + a_c16 * 16u) : 0xffffffffu;
    }
    const bool sa_active = (tid < kBM);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : num_tokens;
    const uint32_t sa_roff = ((tid < valid_rows) && sa_active && (tok_sa < num_tokens)) ? (tok_sa * num_k128) : 0xffffffffu;

    WeaveFloat4 gate_acc[kMXDL][kNXDL];
    WeaveFloat4 up_acc[kMXDL][kNXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) {
            gate_acc[mx][nx] = {0,0,0,0};
            up_acc[mx][nx]   = {0,0,0,0};
        }
    }

    const unsigned n16_b0 = tile_n * 8u + wave_idx;
    const unsigned n16_b1 = tile_n * 8u + 4u + wave_idx;

    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 64u + a_r64;
        const uint4 av = (a_roff[ap] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[ap]) : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<WeaveLdsU4*>(lds + r * 64u + (a_c16 ^ ((r >> 1u) & 3u)) * 16u) = av;
    }
    {
        const uint64_t boff0 = (static_cast<uint64_t>(n16_b0) * num_k64 * 64ull + lane) * 16ull;
        const uint64_t boff1 = (static_cast<uint64_t>(n16_b1) * num_k64 * 64ull + lane) * 16ull;
        *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + wave_idx * 1024u + lane * 16u) = weave_gload_u4(b_gate_exp + boff0);
        *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + (4u + wave_idx) * 1024u + lane * 16u) = weave_gload_u4(b_gate_exp + boff1);
        *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + 8192u + wave_idx * 1024u + lane * 16u) = weave_gload_u4(b_up_exp + boff0);
        *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + 8192u + (4u + wave_idx) * 1024u + lane * 16u) = weave_gload_u4(b_up_exp + boff1);
    }
    if (sa_active) {
        reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff)[tid] = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff] : 0.0f;
    }
    float sb_g_cur = weave_as1_arr(bs_gate_row)[0];
    float sb_u_cur = weave_as1_arr(bs_up_row)[0];
    weave_4wave_lds_barrier_65440(arena);

    #pragma clang loop unroll(disable)
    for (unsigned k64 = 0u; k64 + 1u < num_k64; ++k64) {
        const unsigned next_k64 = k64 + 1u;
        const unsigned next_sidx = next_k64 >> 1u;
        const uint32_t ak_off = next_k64 * 64u;

        constexpr unsigned kMXHalf = kMXDL / 2u;
        const unsigned cur_st = k64 & 1u;
        const unsigned nxt_st = next_k64 & 1u;
        const WeaveLdsU8* lds_a_cur  = lds + cur_st * kAStageBytes;
        const WeaveLdsU8* lds_bg_cur = lds + kBBaseOff + cur_st * kBStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u);
        WeaveLdsU8* lds_a_nxt  = lds + nxt_st * kAStageBytes;
        WeaveLdsU8* lds_bg_nxt = lds + kBBaseOff + nxt_st * kBStageBytes;

        const uint64_t boff0_n = ((static_cast<uint64_t>(n16_b0) * num_k64 + next_k64) * 64ull + lane) * 16ull;
        const uint64_t boff1_n = ((static_cast<uint64_t>(n16_b1) * num_k64 + next_k64) * 64ull + lane) * 16ull;

        const uint4 an0 = (a_roff[0] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[0] + ak_off) : make_uint4(0u, 0u, 0u, 0u);
        const uint4 bg0_next = weave_gload_u4(b_gate_exp + boff0_n);
        const uint4 bg1_next = weave_gload_u4(b_gate_exp + boff1_n);
        __builtin_amdgcn_sched_barrier(0);

        weave_lds_s1_k64_mfma_range<0u, kMXHalf, kNXDL>(
            lds_a_cur, lds_bg_cur, lds_bg_cur + 8192u, lds_sa_cur,
            row_base_w, b_chunk_w, lane, m_lane, k_grp, sb_g_cur, sb_u_cur, gate_acc, up_acc);

        *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + a_r64 * 64u + (a_c16 ^ ((a_r64 >> 1u) & 3u)) * 16u) = an0;
        *reinterpret_cast<WeaveLdsU4*>(lds_bg_nxt + wave_idx * 1024u + lane * 16u) = bg0_next;
        *reinterpret_cast<WeaveLdsU4*>(lds_bg_nxt + (4u + wave_idx) * 1024u + lane * 16u) = bg1_next;
        __builtin_amdgcn_sched_barrier(0);

        uint4 an1 = make_uint4(0u, 0u, 0u, 0u);
        if constexpr (kAPasses > 1u) {
            an1 = (a_roff[1] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[1] + ak_off) : make_uint4(0u, 0u, 0u, 0u);
        }
        const uint4 bu0_next = weave_gload_u4(b_up_exp + boff0_n);
        const uint4 bu1_next = weave_gload_u4(b_up_exp + boff1_n);
        const float sa_next = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff + next_sidx] : 0.0f;
        const float sb_g_next = weave_as1_arr(bs_gate_row)[next_sidx];
        const float sb_u_next = weave_as1_arr(bs_up_row)[next_sidx];
        __builtin_amdgcn_sched_barrier(0);

        weave_lds_s1_k64_mfma_range<kMXHalf, kMXDL, kNXDL>(
            lds_a_cur, lds_bg_cur, lds_bg_cur + 8192u, lds_sa_cur,
            row_base_w, b_chunk_w, lane, m_lane, k_grp, sb_g_cur, sb_u_cur, gate_acc, up_acc);

        if constexpr (kAPasses > 1u) {
            const unsigned r1 = 64u + a_r64;
            *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + r1 * 64u + (a_c16 ^ ((r1 >> 1u) & 3u)) * 16u) = an1;
        }
        *reinterpret_cast<WeaveLdsU4*>(lds_bg_nxt + 8192u + wave_idx * 1024u + lane * 16u) = bu0_next;
        *reinterpret_cast<WeaveLdsU4*>(lds_bg_nxt + 8192u + (4u + wave_idx) * 1024u + lane * 16u) = bu1_next;
        if (sa_active) {
            reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + nxt_st * 512u)[tid] = sa_next;
        }
        sb_g_cur = sb_g_next;
        sb_u_cur = sb_u_next;
        weave_4wave_lds_barrier_65440(arena);
    }

    {
        const unsigned cur_st = (num_k64 - 1u) & 1u;
        const WeaveLdsU8* lds_bg_cur = lds + kBBaseOff + cur_st * kBStageBytes;
        weave_lds_s1_k64_mfma_range<0u, kMXDL, kNXDL>(
            lds + cur_st * kAStageBytes, lds_bg_cur, lds_bg_cur + 8192u,
            reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u),
            row_base_w, b_chunk_w, lane, m_lane, k_grp,
            sb_g_cur, sb_u_cur, gate_acc, up_acc);
    }
    weave_4wave_lds_barrier_65440(arena);

    // Single-pass epilogue: kBM * 128 * 2B <= 32768B in lds[0..32768) (lds_tok at 50176 untouched!)
    WeaveLdsU16* lds_q = reinterpret_cast<WeaveLdsU16*>(lds);
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = row_base_w + mx * 16u + m_lane;
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) {
            const unsigned col = (b_chunk_w + nx) * 16u + k_grp * 4u;
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][0]) * up_acc[mx][nx][0])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][1]) * up_acc[mx][nx][1])) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][2]) * up_acc[mx][nx][2])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][3]) * up_acc[mx][nx][3])) << 16u);
            *reinterpret_cast<WeaveLdsU2*>(lds_q + r * 128u + col) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned q_row = tid >> 4u;
    const unsigned q_lane = tid & 15u;
    #pragma clang loop unroll(disable)
    for (unsigned r0 = 0u; r0 < kBM; r0 += 16u) {
        const unsigned r = r0 + q_row;
        const unsigned dense_row = row_offset + r;
        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
        uint4 raw = make_uint4(0,0,0,0);
        if (valid) { raw = *reinterpret_cast<const WeaveLdsU4*>(lds_q + r * 128u + q_lane * 8u); }
        weave_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

// TIER 3 & TIER 4: TRUE BM = 64 (2x2) & TRUE BM = 128 (4x1 Stage1 / 2x2 Stage2)
// ============================================================================
template <unsigned kBM>
__device__ __forceinline__ void weave_s1_true_bm64_bm128(
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
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kMXDL = 2u;
    constexpr unsigned kNXDL = (kBM == 128u) ? 4u : 2u;
    constexpr unsigned kAPasses = kBM / 64u;
    constexpr unsigned kAStageBytes = kBM * 64u;
    constexpr unsigned kBStageBytes = 8192u;
    constexpr unsigned kBBaseOff = 2u * kAStageBytes;
    constexpr unsigned kHalf0Off = 32768u;
    constexpr unsigned kSaBaseOff = 49152u;
    constexpr unsigned kTokOff = 50176u;

    const unsigned row_base_w = (kBM == 128u) ? (wave_idx * 32u) : ((wave_idx >> 1u) * 32u);
    const unsigned b_chunk_w  = (kBM == 128u) ? 0u : ((wave_idx & 1u) * 2u);

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu) : num_tokens;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint64_t expert_b_bytes = static_cast<uint64_t>(N) * K * 2ull;
    const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_b_bytes;
    const uint8_t* b_up_exp = b_gate_exp + static_cast<uint64_t>(N) * K;
    const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * num_k128;
    const float* bs_gate_row = b_scale_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_bs_stride + static_cast<uint64_t>(tile_n) * num_k128;
    const float* bs_up_row = bs_gate_row + static_cast<uint64_t>(n_blocks) * num_k128;

    const unsigned a_r64 = tid >> 2u;
    const unsigned a_c16 = tid & 3u;
    uint32_t a_roff[kAPasses];
    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 64u + a_r64;
        const uint32_t tok = lds_tok[r];
        a_roff[ap] = ((r < valid_rows) && (tok < num_tokens)) ? (tok * K + a_c16 * 16u) : 0xffffffffu;
    }
    const bool sa_active = (tid < kBM);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : num_tokens;
    const uint32_t sa_roff = ((tid < valid_rows) && sa_active && (tok_sa < num_tokens)) ? (tok_sa * num_k128) : 0xffffffffu;

    WeaveLdsU16* lds_half0 = reinterpret_cast<WeaveLdsU16*>(lds + kHalf0Off);

    #pragma clang loop unroll(disable)
    for (unsigned n_half = 0u; n_half < 2u; ++n_half) {
        WeaveFloat4 gate_acc[kMXDL][kNXDL];
        WeaveFloat4 up_acc[kMXDL][kNXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            #pragma unroll
            for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                gate_acc[mx][nx] = {0,0,0,0};
                up_acc[mx][nx]   = {0,0,0,0};
            }
        }

        const unsigned n16_b = tile_n * 8u + n_half * 4u + wave_idx;

        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) {
            const unsigned r = ap * 64u + a_r64;
            const uint4 av = (a_roff[ap] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[ap]) : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<WeaveLdsU4*>(lds + r * 64u + (a_c16 ^ ((r >> 1u) & 3u)) * 16u) = av;
        }
        {
            const uint64_t boff0 = (static_cast<uint64_t>(n16_b) * num_k64 * 64ull + lane) * 16ull;
            *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + wave_idx * 1024u + lane * 16u) =
                weave_gload_u4(b_gate_exp + boff0);
            *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + 4096u + wave_idx * 1024u + lane * 16u) =
                weave_gload_u4(b_up_exp + boff0);
        }
        if (sa_active) {
            reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff)[tid] = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff] : 0.0f;
        }
        float sb_g_cur = weave_as1_arr(bs_gate_row)[0];
        float sb_u_cur = weave_as1_arr(bs_up_row)[0];
        weave_4wave_lds_barrier_65440(arena);

        #pragma clang loop unroll(disable)
        for (unsigned k64 = 0u; k64 + 1u < num_k64; ++k64) {
            const unsigned next_k64 = k64 + 1u;
            const unsigned next_sidx = next_k64 >> 1u;

            uint4 a_next[kAPasses];
            const uint32_t ak_off = next_k64 * 64u;
            #pragma unroll
            for (unsigned ap = 0u; ap < kAPasses; ++ap) {
                a_next[ap] = (a_roff[ap] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[ap] + ak_off) : make_uint4(0u, 0u, 0u, 0u);
            }
            const uint64_t boff_n = ((static_cast<uint64_t>(n16_b) * num_k64 + next_k64) * 64ull + lane) * 16ull;
            const uint4 bg_next = weave_gload_u4(b_gate_exp + boff_n);
            const uint4 bu_next = weave_gload_u4(b_up_exp + boff_n);
            const float sa_next = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff + next_sidx] : 0.0f;
            const float sb_g_next = weave_as1_arr(bs_gate_row)[next_sidx];
            const float sb_u_next = weave_as1_arr(bs_up_row)[next_sidx];

            const unsigned cur_st = k64 & 1u;
            const WeaveLdsU8* lds_bg_cur = lds + kBBaseOff + cur_st * kBStageBytes;
            weave_lds_s1_k64_mfma_range<0u, kMXDL, kNXDL>(
                lds + cur_st * kAStageBytes, lds_bg_cur, lds_bg_cur + 4096u,
                reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u),
                row_base_w, b_chunk_w, lane, m_lane, k_grp,
                sb_g_cur, sb_u_cur, gate_acc, up_acc);

            const unsigned nxt_st = next_k64 & 1u;
            WeaveLdsU8* lds_a_nxt  = lds + nxt_st * kAStageBytes;
            WeaveLdsU8* lds_bg_nxt = lds + kBBaseOff + nxt_st * kBStageBytes;
            #pragma unroll
            for (unsigned ap = 0u; ap < kAPasses; ++ap) {
                const unsigned r = ap * 64u + a_r64;
                *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + r * 64u + (a_c16 ^ ((r >> 1u) & 3u)) * 16u) = a_next[ap];
            }
            *reinterpret_cast<WeaveLdsU4*>(lds_bg_nxt + wave_idx * 1024u + lane * 16u) = bg_next;
            *reinterpret_cast<WeaveLdsU4*>(lds_bg_nxt + 4096u + wave_idx * 1024u + lane * 16u) = bu_next;
            if (sa_active) {
                reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + nxt_st * 512u)[tid] = sa_next;
            }
            sb_g_cur = sb_g_next;
            sb_u_cur = sb_u_next;
            weave_4wave_lds_barrier_65440(arena);
        }

        {
            const unsigned cur_st = (num_k64 - 1u) & 1u;
            const WeaveLdsU8* lds_bg_cur = lds + kBBaseOff + cur_st * kBStageBytes;
            weave_lds_s1_k64_mfma_range<0u, kMXDL, kNXDL>(
                lds + cur_st * kAStageBytes, lds_bg_cur, lds_bg_cur + 4096u,
                reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u),
                row_base_w, b_chunk_w, lane, m_lane, k_grp,
                sb_g_cur, sb_u_cur, gate_acc, up_acc);
        }
        weave_4wave_lds_barrier_65440(arena);

        WeaveLdsU16* dst_half = (n_half == 0u) ? lds_half0 : reinterpret_cast<WeaveLdsU16*>(lds);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned r = row_base_w + mx * 16u + m_lane;
            #pragma unroll
            for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                const unsigned col = (b_chunk_w + nx) * 16u + k_grp * 4u;
                const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx][0]) * up_acc[mx][nx][0])) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx][1]) * up_acc[mx][nx][1])) << 16u);
                const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx][2]) * up_acc[mx][nx][2])) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx][3]) * up_acc[mx][nx][3])) << 16u);
                *reinterpret_cast<WeaveLdsU2*>(dst_half + r * 64u + col) = make_uint2(p0, p1);
            }
        }
        weave_4wave_lds_barrier_65440(arena);
    }

    const WeaveLdsU16* lds_half1 = reinterpret_cast<const WeaveLdsU16*>(lds);
    const unsigned q_row = tid >> 4u;
    const unsigned q_lane = tid & 15u;

    #pragma clang loop unroll(disable)
    for (unsigned r0 = 0u; r0 < kBM; r0 += 16u) {
        const unsigned r = r0 + q_row;
        const unsigned dense_row = row_offset + r;
        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
        const WeaveLdsU16* src_p = (q_lane < 8u)
            ? (lds_half0 + r * 64u + q_lane * 8u)
            : (lds_half1 + r * 64u + (q_lane - 8u) * 8u);
        uint4 raw = make_uint4(0,0,0,0); if (valid) { raw = *reinterpret_cast<const WeaveLdsU4*>(src_p); }
        weave_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}



// ============================================================================
// ZERO-COPY UNROLLED REGISTER PIPELINES FOR BM=16 & BM=32 (ZERO v_mov_b32!)
// Eliminates same-iteration s_waitcnt vmcnt(0) stalls caused by register rotation!
// ============================================================================
template <unsigned kBM>
__device__ __forceinline__ void weave_s1_small_1x4_zc2(
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
    constexpr unsigned kTokenCacheOffset = 16384u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokenCacheOffset);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu) : num_tokens;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint64_t expert_b_bytes = static_cast<uint64_t>(N) * K * 2ull;
    const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_b_bytes;
    const uint8_t* b_up_exp = b_gate_exp + static_cast<uint64_t>(N) * K;
    const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * num_k128;
    const float* bs_gate_row = b_scale_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_bs_stride + static_cast<uint64_t>(tile_n) * num_k128;
    const float* bs_up_row = bs_gate_row + static_cast<uint64_t>(n_blocks) * num_k128;

    const unsigned n16_base = tile_n * 8u + wave_idx * 2u;
    uint64_t b_col_off[2];
    b_col_off[0] = (static_cast<uint64_t>(n16_base + 0u) * num_k64 * 64ull + lane) * 16ull;
    b_col_off[1] = (static_cast<uint64_t>(n16_base + 1u) * num_k64 * 64ull + lane) * 16ull;

    WeaveFloat4 gate_acc[kMXDL][2];
    WeaveFloat4 up_acc[kMXDL][2];
    uint32_t a_m_off[kMXDL];
    uint32_t sa_m_off[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        gate_acc[mx][0] = {0,0,0,0}; gate_acc[mx][1] = {0,0,0,0};
        up_acc[mx][0]   = {0,0,0,0}; up_acc[mx][1]   = {0,0,0,0};
        const unsigned r = mx * 16u + m_lane;
        const uint32_t tok = lds_tok[r];
        const bool ok = (r < valid_rows) && (tok < num_tokens);
        a_m_off[mx]  = ok ? (tok * K + k_grp * 16u) : 0xffffffffu;
        sa_m_off[mx] = ok ? (tok * num_k128) : 0xffffffffu;
    }

    auto load_k128 = [&](unsigned kb, uint4 (&av0)[kMXDL], uint4 (&av1)[kMXDL],
                         uint4 (&bg0)[2], uint4 (&bu0)[2], uint4 (&bg1)[2], uint4 (&bu1)[2],
                         float (&sa)[kMXDL], float& sbg, float& sbu) {
        const uint32_t ak = kb * 128u;
        const uint64_t bk = static_cast<uint64_t>(kb) * 2048ull;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av0[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak) : make_uint4(0,0,0,0);
            av1[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak + 64u) : make_uint4(0,0,0,0);
            sa[mx]  = (sa_m_off[mx] != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_m_off[mx] + kb] : 0.0f;
        }
        bg0[0] = weave_gload_u4(b_gate_exp + b_col_off[0] + bk);
        bg0[1] = weave_gload_u4(b_gate_exp + b_col_off[1] + bk);
        bu0[0] = weave_gload_u4(b_up_exp   + b_col_off[0] + bk);
        bu0[1] = weave_gload_u4(b_up_exp   + b_col_off[1] + bk);
        bg1[0] = weave_gload_u4(b_gate_exp + b_col_off[0] + bk + 1024ull);
        bg1[1] = weave_gload_u4(b_gate_exp + b_col_off[1] + bk + 1024ull);
        bu1[0] = weave_gload_u4(b_up_exp   + b_col_off[0] + bk + 1024ull);
        bu1[1] = weave_gload_u4(b_up_exp   + b_col_off[1] + bk + 1024ull);
        sbg = weave_as1_arr(bs_gate_row)[kb];
        sbu = weave_as1_arr(bs_up_row)[kb];
    };

    if constexpr (kBM == 16u) {
        if ((num_k128 >= 2u) && ((num_k128 & 1u) == 0u)) {
            uint4 av0_a[1], av1_a[1], bg0_a[2], bu0_a[2], bg1_a[2], bu1_a[2];
            uint4 av0_b[1], av1_b[1], bg0_b[2], bu0_b[2], bg1_b[2], bu1_b[2];
            float sa_a[1], sa_b[1], sbg_a, sbu_a, sbg_b, sbu_b;

            load_k128(0u, av0_a, av1_a, bg0_a, bu0_a, bg1_a, bu1_a, sa_a, sbg_a, sbu_a);
            load_k128(1u, av0_b, av1_b, bg0_b, bu0_b, bg1_b, bu1_b, sa_b, sbg_b, sbu_b);

            #pragma clang loop unroll(disable)
            for (unsigned kb = 0u; kb + 2u < num_k128; kb += 2u) {
                weave_s1_direct_reg_k128_mfma<1u>(av0_a, av1_a, bg0_a, bu0_a, bg1_a, bu1_a, sa_a, sbg_a, sbu_a, gate_acc, up_acc);
                load_k128(kb + 2u, av0_a, av1_a, bg0_a, bu0_a, bg1_a, bu1_a, sa_a, sbg_a, sbu_a);

                weave_s1_direct_reg_k128_mfma<1u>(av0_b, av1_b, bg0_b, bu0_b, bg1_b, bu1_b, sa_b, sbg_b, sbu_b, gate_acc, up_acc);
                load_k128(kb + 3u, av0_b, av1_b, bg0_b, bu0_b, bg1_b, bu1_b, sa_b, sbg_b, sbu_b);
            }
            weave_s1_direct_reg_k128_mfma<1u>(av0_a, av1_a, bg0_a, bu0_a, bg1_a, bu1_a, sa_a, sbg_a, sbu_a, gate_acc, up_acc);
            weave_s1_direct_reg_k128_mfma<1u>(av0_b, av1_b, bg0_b, bu0_b, bg1_b, bu1_b, sa_b, sbg_b, sbu_b, gate_acc, up_acc);
        } else {
            #pragma clang loop unroll(disable)
            for (unsigned kb = 0u; kb < num_k128; ++kb) {
                uint4 av0[1], av1[1], bg0[2], bu0[2], bg1[2], bu1[2];
                float sa[1], sbg, sbu;
                load_k128(kb, av0, av1, bg0, bu0, bg1, bu1, sa, sbg, sbu);
                weave_s1_direct_reg_k128_mfma<1u>(av0, av1, bg0, bu0, bg1, bu1, sa, sbg, sbu, gate_acc, up_acc);
            }
        }
    } else {
        // Zero-Copy 2-Half K=128 Pipeline for BM=32 Stage 1 (only 80 VGPRs total -> ZERO spills, ZERO v_mov_b32!)
        uint4 av0[2], bg0[2], bu0[2], av1[2], bg1[2], bu1[2];
        float sa[2], sbg, sbu;
        #pragma unroll
        for (unsigned mx = 0u; mx < 2u; ++mx) {
            av0[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx]) : make_uint4(0,0,0,0);
            sa[mx]  = (sa_m_off[mx] != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_m_off[mx]] : 0.0f;
        }
        bg0[0] = weave_gload_u4(b_gate_exp + b_col_off[0]);
        bg0[1] = weave_gload_u4(b_gate_exp + b_col_off[1]);
        bu0[0] = weave_gload_u4(b_up_exp   + b_col_off[0]);
        bu0[1] = weave_gload_u4(b_up_exp   + b_col_off[1]);
        sbg = weave_as1_arr(bs_gate_row)[0];
        sbu = weave_as1_arr(bs_up_row)[0];

        #pragma unroll
        for (unsigned mx = 0u; mx < 2u; ++mx) {
            av1[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + 64u) : make_uint4(0,0,0,0);
        }
        bg1[0] = weave_gload_u4(b_gate_exp + b_col_off[0] + 1024ull);
        bg1[1] = weave_gload_u4(b_gate_exp + b_col_off[1] + 1024ull);
        bu1[0] = weave_gload_u4(b_up_exp   + b_col_off[0] + 1024ull);
        bu1[1] = weave_gload_u4(b_up_exp   + b_col_off[1] + 1024ull);

        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u; kb + 1u < num_k128; ++kb) {
            const unsigned nkb = kb + 1u;
            const uint32_t nak = nkb * 128u;
            const uint64_t nbk = static_cast<uint64_t>(nkb) * 2048ull;

            weave_s1_direct_reg_half_mfma<2u>(av0, bg0, bu0, sa, sbg, sbu, gate_acc, up_acc);
            #pragma unroll
            for (unsigned mx = 0u; mx < 2u; ++mx) {
                av0[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + nak) : make_uint4(0,0,0,0);
            }
            bg0[0] = weave_gload_u4(b_gate_exp + b_col_off[0] + nbk);
            bg0[1] = weave_gload_u4(b_gate_exp + b_col_off[1] + nbk);
            bu0[0] = weave_gload_u4(b_up_exp   + b_col_off[0] + nbk);
            bu0[1] = weave_gload_u4(b_up_exp   + b_col_off[1] + nbk);

            weave_s1_direct_reg_half_mfma<2u>(av1, bg1, bu1, sa, sbg, sbu, gate_acc, up_acc);
            #pragma unroll
            for (unsigned mx = 0u; mx < 2u; ++mx) {
                av1[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + nak + 64u) : make_uint4(0,0,0,0);
                sa[mx]  = (sa_m_off[mx] != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_m_off[mx] + nkb] : 0.0f;
            }
            bg1[0] = weave_gload_u4(b_gate_exp + b_col_off[0] + nbk + 1024ull);
            bg1[1] = weave_gload_u4(b_gate_exp + b_col_off[1] + nbk + 1024ull);
            bu1[0] = weave_gload_u4(b_up_exp   + b_col_off[0] + nbk + 1024ull);
            bu1[1] = weave_gload_u4(b_up_exp   + b_col_off[1] + nbk + 1024ull);
            sbg = weave_as1_arr(bs_gate_row)[nkb];
            sbu = weave_as1_arr(bs_up_row)[nkb];
        }
        weave_s1_direct_reg_k128_mfma<2u>(av0, av1, bg0, bu0, bg1, bu1, sa, sbg, sbu, gate_acc, up_acc);
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
                weave_bk64_silu(gate_acc[mx][nx][0]) * up_acc[mx][nx][0])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][1]) * up_acc[mx][nx][1])) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][2]) * up_acc[mx][nx][2])) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                weave_bk64_silu(gate_acc[mx][nx][3]) * up_acc[mx][nx][3])) << 16u);
            *reinterpret_cast<WeaveLdsU2*>(lds_q + r * 128u + col) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + q_row;
        const unsigned dense_row = row_offset + r;
        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
        uint4 raw = make_uint4(0,0,0,0);
        if (valid) { raw = *reinterpret_cast<const WeaveLdsU4*>(lds_q + r * 128u + q_lane * 8u); }
        weave_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

template <unsigned kBM>
__device__ __forceinline__ void weave_s2_small_1x4_zc(
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
    constexpr unsigned kTokenCacheOffset = 16384u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokenCacheOffset);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu) : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * N * K;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * n_blocks + tile_n) * num_k128;

    const unsigned n16_base = tile_n * 8u + wave_idx * 2u;
    uint64_t b_col_off[2];
    b_col_off[0] = (static_cast<uint64_t>(n16_base + 0u) * num_k64 * 64ull + lane) * 16ull;
    b_col_off[1] = (static_cast<uint64_t>(n16_base + 1u) * num_k64 * 64ull + lane) * 16ull;

    WeaveFloat4 acc[kMXDL][2];
    uint32_t a_m_off[kMXDL];
    uint32_t sa_m_off[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        acc[mx][0] = {0,0,0,0}; acc[mx][1] = {0,0,0,0};
        const unsigned r = mx * 16u + m_lane;
        const uint32_t tok = lds_tok[r];
        const bool ok = (r < valid_rows) && (tok < route_rows);
        a_m_off[mx]  = ok ? (tok * K + k_grp * 16u) : 0xffffffffu;
        sa_m_off[mx] = ok ? (tok * num_k128) : 0xffffffffu;
    }

    auto load_k128 = [&](unsigned kb, uint4 (&av0)[kMXDL], uint4 (&av1)[kMXDL],
                         uint4 (&b0)[2], uint4 (&b1)[2], float (&sa)[kMXDL], float& sb) {
        const uint32_t ak = kb * 128u;
        const uint64_t bk = static_cast<uint64_t>(kb) * 2048ull;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
#if V60_L3_SMALL_RAW_SC1   // OFF: the raw flat `global_load ... sc1` below page-faults on silicon
            av0[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4_sc1(a_base + a_m_off[mx] + ak) : make_uint4(0,0,0,0);
            av1[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4_sc1(a_base + a_m_off[mx] + ak + 64u) : make_uint4(0,0,0,0);
            sa[mx]  = (sa_m_off[mx] != 0xffffffffu) ? weave_gload_f32_sc1(a_scale_base + sa_m_off[mx] + kb) : 0.0f;
#else
            av0[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak) : make_uint4(0,0,0,0);
            av1[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak + 64u) : make_uint4(0,0,0,0);
            sa[mx]  = (sa_m_off[mx] != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_m_off[mx] + kb] : 0.0f;
#endif
        }
        b0[0] = weave_gload_u4(b_exp + b_col_off[0] + bk);
        b0[1] = weave_gload_u4(b_exp + b_col_off[1] + bk);
        b1[0] = weave_gload_u4(b_exp + b_col_off[0] + bk + 1024ull);
        b1[1] = weave_gload_u4(b_exp + b_col_off[1] + bk + 1024ull);
        sb = weave_as1_arr(bs_row)[kb];
    };

    if constexpr (kBM == 16u) {
        // 4-Stage Zero-Copy Unrolled K=512 Pipeline (104 VGPRs total, 18 global_load_dwordx4/wave in flight!)
        if ((num_k128 >= 4u) && ((num_k128 & 3u) == 0u)) {
            uint4 av0_0[1], av1_0[1], b0_0[2], b1_0[2]; float sa_0[1], sb_0;
            uint4 av0_1[1], av1_1[1], b0_1[2], b1_1[2]; float sa_1[1], sb_1;
            uint4 av0_2[1], av1_2[1], b0_2[2], b1_2[2]; float sa_2[1], sb_2;
            uint4 av0_3[1], av1_3[1], b0_3[2], b1_3[2]; float sa_3[1], sb_3;

            load_k128(0u, av0_0, av1_0, b0_0, b1_0, sa_0, sb_0);
            load_k128(1u, av0_1, av1_1, b0_1, b1_1, sa_1, sb_1);
            load_k128(2u, av0_2, av1_2, b0_2, b1_2, sa_2, sb_2);
            load_k128(3u, av0_3, av1_3, b0_3, b1_3, sa_3, sb_3);

            #pragma clang loop unroll(disable)
            for (unsigned kb = 0u; kb + 4u < num_k128; kb += 4u) {
                weave_s2_direct_reg_k128_mfma<1u, 2u>(av0_0, av1_0, b0_0, b1_0, sa_0, sb_0, acc);
                load_k128(kb + 4u, av0_0, av1_0, b0_0, b1_0, sa_0, sb_0);

                weave_s2_direct_reg_k128_mfma<1u, 2u>(av0_1, av1_1, b0_1, b1_1, sa_1, sb_1, acc);
                load_k128(kb + 5u, av0_1, av1_1, b0_1, b1_1, sa_1, sb_1);

                weave_s2_direct_reg_k128_mfma<1u, 2u>(av0_2, av1_2, b0_2, b1_2, sa_2, sb_2, acc);
                load_k128(kb + 6u, av0_2, av1_2, b0_2, b1_2, sa_2, sb_2);

                weave_s2_direct_reg_k128_mfma<1u, 2u>(av0_3, av1_3, b0_3, b1_3, sa_3, sb_3, acc);
                load_k128(kb + 7u, av0_3, av1_3, b0_3, b1_3, sa_3, sb_3);
            }
            weave_s2_direct_reg_k128_mfma<1u, 2u>(av0_0, av1_0, b0_0, b1_0, sa_0, sb_0, acc);
            weave_s2_direct_reg_k128_mfma<1u, 2u>(av0_1, av1_1, b0_1, b1_1, sa_1, sb_1, acc);
            weave_s2_direct_reg_k128_mfma<1u, 2u>(av0_2, av1_2, b0_2, b1_2, sa_2, sb_2, acc);
            weave_s2_direct_reg_k128_mfma<1u, 2u>(av0_3, av1_3, b0_3, b1_3, sa_3, sb_3, acc);
        } else {
            #pragma clang loop unroll(disable)
            for (unsigned kb = 0u; kb < num_k128; ++kb) {
                uint4 av0[1], av1[1], b0[2], b1[2]; float sa[1], sb;
                load_k128(kb, av0, av1, b0, b1, sa, sb);
                weave_s2_direct_reg_k128_mfma<1u, 2u>(av0, av1, b0, b1, sa, sb, acc);
            }
        }
    } else {
        // 2-Stage Zero-Copy Unrolled K=256 Pipeline for BM=32 (80 VGPRs total!)
        if ((num_k128 >= 2u) && ((num_k128 & 1u) == 0u)) {
            uint4 av0_a[2], av1_a[2], b0_a[2], b1_a[2]; float sa_a[2], sb_a;
            uint4 av0_b[2], av1_b[2], b0_b[2], b1_b[2]; float sa_b[2], sb_b;
            load_k128(0u, av0_a, av1_a, b0_a, b1_a, sa_a, sb_a);
            load_k128(1u, av0_b, av1_b, b0_b, b1_b, sa_b, sb_b);

            #pragma clang loop unroll(disable)
            for (unsigned kb = 0u; kb + 2u < num_k128; kb += 2u) {
                weave_s2_direct_reg_k128_mfma<2u, 2u>(av0_a, av1_a, b0_a, b1_a, sa_a, sb_a, acc);
                load_k128(kb + 2u, av0_a, av1_a, b0_a, b1_a, sa_a, sb_a);

                weave_s2_direct_reg_k128_mfma<2u, 2u>(av0_b, av1_b, b0_b, b1_b, sa_b, sb_b, acc);
                load_k128(kb + 3u, av0_b, av1_b, b0_b, b1_b, sa_b, sb_b);
            }
            weave_s2_direct_reg_k128_mfma<2u, 2u>(av0_a, av1_a, b0_a, b1_a, sa_a, sb_a, acc);
            weave_s2_direct_reg_k128_mfma<2u, 2u>(av0_b, av1_b, b0_b, b1_b, sa_b, sb_b, acc);
        } else {
            #pragma clang loop unroll(disable)
            for (unsigned kb = 0u; kb < num_k128; ++kb) {
                uint4 av0[2], av1[2], b0[2], b1[2]; float sa[2], sb;
                load_k128(kb, av0, av1, b0, b1, sa, sb);
                weave_s2_direct_reg_k128_mfma<2u, 2u>(av0, av1, b0, b1, sa, sb, acc);
            }
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
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][1] * weight)) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][2] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][3] * weight)) << 16u);
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
            weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n * 128u + c_lane * 8u, make_uint4(u0.x, u0.y, u1.x, u1.y));
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

template <unsigned kBM>
__device__ __forceinline__ void weave_s1_small_dispatch(
    const int32_t* __restrict__ s_ids, int32_t expert_id,
    const uint8_t* __restrict__ a_base, const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base, const float* __restrict__ b_scale_base,
    uint8_t* __restrict__ quant_out_base, float* __restrict__ quant_scale_base,
    unsigned row_offset, unsigned valid_rows, unsigned tile_n,
    unsigned num_tokens, unsigned route_rows, unsigned N, unsigned K, unsigned char* arena)
{
    weave_s1_small_1x4_zc2<kBM>(s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
        quant_out_base, quant_scale_base, row_offset, valid_rows, tile_n, num_tokens, route_rows, N, K, arena);
}

template <unsigned kBM>
__device__ __forceinline__ void weave_s2_small_dispatch(
    const int32_t* __restrict__ s2_ids, int32_t expert_id,
    const uint8_t* __restrict__ a_base, const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base, const float* __restrict__ b_scale_base,
    const float* __restrict__ weights, uint16_t* __restrict__ c_out_base,
    unsigned row_offset, unsigned valid_rows, unsigned tile_n,
    unsigned route_rows, unsigned N, unsigned K, unsigned char* arena)
{
    weave_s2_small_1x4_zc<kBM>(s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
        weights, c_out_base, row_offset, valid_rows, tile_n, route_rows, N, K, arena);
}

template <unsigned kBM>
__device__ __forceinline__ void weave_s1_true_bm64_dispatch(
    const int32_t* __restrict__ s_ids, int32_t expert_id,
    const uint8_t* __restrict__ a_base, const uint8_t* __restrict__ b_base,
    const float* __restrict__ a_scale_base, const float* __restrict__ b_scale_base,
    uint8_t* __restrict__ quant_out_base, float* __restrict__ quant_scale_base,
    unsigned row_offset, unsigned valid_rows, unsigned tile_n,
    unsigned num_tokens, unsigned route_rows, unsigned N, unsigned K, unsigned char* arena)
{
    if constexpr (kBM == 64u) {
        weave_s1_true_bm64_singlepass(s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            quant_out_base, quant_scale_base, row_offset, valid_rows, tile_n, num_tokens, route_rows, N, K, arena);
    } else {
        weave_s1_true_bm64_bm128<128u>(s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            quant_out_base, quant_scale_base, row_offset, valid_rows, tile_n, num_tokens, route_rows, N, K, arena);
    }
}
template <unsigned kBM>
__device__ __forceinline__ void weave_s2_true_bm64_bm128(
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
    static_assert(kBM == 64u || kBM == 128u);
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kMXDL = kBM / 32u;
    constexpr unsigned kNXDL = 4u;
    constexpr unsigned kAPasses = kBM / 64u;
    constexpr unsigned kAStageBytes = kBM * 64u;
    constexpr unsigned kBStageBytes = 8192u;
    constexpr unsigned kBBaseOff = 2u * kAStageBytes;
    constexpr unsigned kSaBaseOff = (kBM == 64u) ? 16384u : 32768u;
    constexpr unsigned kTokOff = (kBM == 64u) ? 17408u : 33792u;

    const unsigned mw = wave_idx >> 1u;
    const unsigned nw = wave_idx & 1u;
    const unsigned row_base_w = mw * (kBM / 2u);
    const unsigned b_chunk_w  = nw * 4u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu) : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * N * K;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * n_blocks + tile_n) * num_k128;

    const unsigned a_r64 = tid >> 2u;
    const unsigned a_c16 = tid & 3u;
    uint32_t a_roff[kAPasses];
    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 64u + a_r64;
        const uint32_t tok = lds_tok[r];
        a_roff[ap] = ((r < valid_rows) && (tok < route_rows)) ? (tok * K + a_c16 * 16u) : 0xffffffffu;
    }
    const bool sa_active = (tid < kBM);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : route_rows;
    const uint32_t sa_roff = ((tid < valid_rows) && sa_active && (tok_sa < route_rows)) ? (tok_sa * num_k128) : 0xffffffffu;

    WeaveFloat4 acc[kMXDL][kNXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) acc[mx][nx] = {0,0,0,0};
    }

    const unsigned n16_b0 = tile_n * 8u + wave_idx;
    const unsigned n16_b1 = tile_n * 8u + 4u + wave_idx;

    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 64u + a_r64;
        const uint4 av = (a_roff[ap] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[ap]) : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<WeaveLdsU4*>(lds + r * 64u + (a_c16 ^ ((r >> 1u) & 3u)) * 16u) = av;
    }
    *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + wave_idx * 1024u + lane * 16u) =
        weave_gload_u4(b_exp + (static_cast<uint64_t>(n16_b0) * num_k64 * 64ull + lane) * 16ull);
    *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + (4u + wave_idx) * 1024u + lane * 16u) =
        weave_gload_u4(b_exp + (static_cast<uint64_t>(n16_b1) * num_k64 * 64ull + lane) * 16ull);
    if (sa_active) {
        reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff)[tid] = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff] : 0.0f;
    }
    float sb_cur = weave_as1_arr(bs_row)[0];
    weave_4wave_lds_barrier_65440(arena);

    #pragma clang loop unroll(disable)
    for (unsigned k64 = 0u; k64 + 1u < num_k64; ++k64) {
        const unsigned next_k64 = k64 + 1u;
        const unsigned next_sidx = next_k64 >> 1u;

        uint4 a_next[kAPasses];
        const uint32_t ak_off = next_k64 * 64u;
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) {
            a_next[ap] = (a_roff[ap] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[ap] + ak_off) : make_uint4(0u, 0u, 0u, 0u);
        }
        const uint4 b0_next = weave_gload_u4(
            b_exp + ((static_cast<uint64_t>(n16_b0) * num_k64 + next_k64) * 64ull + lane) * 16ull);
        const uint4 b1_next = weave_gload_u4(
            b_exp + ((static_cast<uint64_t>(n16_b1) * num_k64 + next_k64) * 64ull + lane) * 16ull);
        const float sa_next = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff + next_sidx] : 0.0f;
        const float sb_next = weave_as1_arr(bs_row)[next_sidx];

        const unsigned cur_st = k64 & 1u;
        weave_lds_s2_k64_mfma_range<0u, kMXDL, kNXDL>(
            lds + cur_st * kAStageBytes,
            lds + kBBaseOff + cur_st * kBStageBytes,
            reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u),
            row_base_w, b_chunk_w, lane, m_lane, k_grp, sb_cur, acc);

        const unsigned nxt_st = next_k64 & 1u;
        WeaveLdsU8* lds_a_nxt = lds + nxt_st * kAStageBytes;
        WeaveLdsU8* lds_b_nxt = lds + kBBaseOff + nxt_st * kBStageBytes;
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) {
            const unsigned r = ap * 64u + a_r64;
            *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + r * 64u + (a_c16 ^ ((r >> 1u) & 3u)) * 16u) = a_next[ap];
        }
        *reinterpret_cast<WeaveLdsU4*>(lds_b_nxt + wave_idx * 1024u + lane * 16u) = b0_next;
        *reinterpret_cast<WeaveLdsU4*>(lds_b_nxt + (4u + wave_idx) * 1024u + lane * 16u) = b1_next;
        if (sa_active) {
            reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + nxt_st * 512u)[tid] = sa_next;
        }
        sb_cur = sb_next;
        weave_4wave_lds_barrier_65440(arena);
    }

    {
        const unsigned cur_st = (num_k64 - 1u) & 1u;
        weave_lds_s2_k64_mfma_range<0u, kMXDL, kNXDL>(
            lds + cur_st * kAStageBytes,
            lds + kBBaseOff + cur_st * kBStageBytes,
            reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u),
            row_base_w, b_chunk_w, lane, m_lane, k_grp, sb_cur, acc);
    }

    weave_4wave_lds_barrier_65440(arena);
    WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = row_base_w + mx * 16u + m_lane;
        const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) {
            const unsigned col = (b_chunk_w + nx) * 16u + k_grp * 4u;
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][1] * weight)) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][2] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][3] * weight)) << 16u);
            const unsigned c8 = (col >> 2u) ^ (r & 7u);
            *reinterpret_cast<WeaveLdsU2*>(lds_c + r * 128u + c8 * 4u) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned c_row = tid >> 4u;
    const unsigned c_lane = tid & 15u;
    #pragma clang loop unroll(disable)
    for (unsigned r0 = 0u; r0 < kBM; r0 += 16u) {
        const unsigned r = r0 + c_row;
        const uint32_t tok_m = lds_tok[r];
        if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
            const unsigned c8_0 = (c_lane * 2u + 0u) ^ (r & 7u);
            const unsigned c8_1 = (c_lane * 2u + 1u) ^ (r & 7u);
            const uint2 u0 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_0 * 4u);
            const uint2 u1 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_1 * 4u);
            weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n * 128u + c_lane * 8u, make_uint4(u0.x, u0.y, u1.x, u1.y));
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

// ============================================================================
// TIER 5: TRUE BM = 256 (4x1 M-Stacked Wave Layout: Wave w=0..3 owns 64 rows!)
// Split-A prefetch reuses 8 VGPRs across mx=0..1 and mx=2..3 -> ZERO VGPR SPILL!
// ============================================================================
__device__ __forceinline__ void weave_s1_true_bm256(
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

    constexpr unsigned kMXDL = 4u;
    constexpr unsigned kNXDL = 2u;
    constexpr unsigned kAStageBytes = 16384u;
    constexpr unsigned kBBaseOff = 32768u;
    constexpr unsigned kBStageBytes = 4096u;
    constexpr unsigned kSaBaseOff = 40960u;
    constexpr unsigned kTokOff = 43008u;

    const unsigned row_base_w = wave_idx * 64u;
    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu) : num_tokens;
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint64_t expert_b_bytes = static_cast<uint64_t>(N) * K * 2ull;
    const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_b_bytes;
    const uint8_t* b_up_exp = b_gate_exp + static_cast<uint64_t>(N) * K;
    const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * num_k128;
    const float* bs_gate_row = b_scale_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_bs_stride + static_cast<uint64_t>(tile_n) * num_k128;
    const float* bs_up_row = bs_gate_row + static_cast<uint64_t>(n_blocks) * num_k128;

    const unsigned a_r64 = tid >> 2u;
    const unsigned a_c16 = tid & 3u;
    uint32_t a_roff[4];
    #pragma unroll
    for (unsigned ap = 0u; ap < 4u; ++ap) {
        const unsigned r = ap * 64u + a_r64;
        const uint32_t tok = lds_tok[r];
        a_roff[ap] = ((r < valid_rows) && (tok < num_tokens)) ? (tok * K + a_c16 * 16u) : 0xffffffffu;
    }
    const uint32_t tok_sa = lds_tok[tid];
    const uint32_t sa_roff = ((tid < valid_rows) && (tok_sa < num_tokens)) ? (tok_sa * num_k128) : 0xffffffffu;

    const uint8_t* b_my_exp = (wave_idx < 2u) ? b_gate_exp : b_up_exp;
    const unsigned b_my_nx  = wave_idx & 1u;

    #pragma clang loop unroll(disable)
    for (unsigned sub_n = 0u; sub_n < 4u; ++sub_n) {
        WeaveFloat4 gate_acc[kMXDL][kNXDL];
        WeaveFloat4 up_acc[kMXDL][kNXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            gate_acc[mx][0] = {0,0,0,0}; gate_acc[mx][1] = {0,0,0,0};
            up_acc[mx][0]   = {0,0,0,0}; up_acc[mx][1]   = {0,0,0,0};
        }

        const unsigned n16_my = tile_n * 8u + sub_n * 2u + b_my_nx;

        #pragma unroll
        for (unsigned ap = 0u; ap < 4u; ++ap) {
            const unsigned r = ap * 64u + a_r64;
            const uint4 av = (a_roff[ap] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[ap]) : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<WeaveLdsU4*>(lds + r * 64u + (a_c16 ^ ((r >> 1u) & 3u)) * 16u) = av;
        }
        *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + wave_idx * 1024u + lane * 16u) =
            weave_gload_u4(b_my_exp + (static_cast<uint64_t>(n16_my) * num_k64 * 64ull + lane) * 16ull);
        reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff)[tid] = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff] : 0.0f;
        float sb_g_cur = weave_as1_arr(bs_gate_row)[0];
        float sb_u_cur = weave_as1_arr(bs_up_row)[0];
        weave_4wave_lds_barrier_65440(arena);

        #pragma clang loop unroll(disable)
        for (unsigned k64 = 0u; k64 + 1u < num_k64; ++k64) {
            const unsigned next_k64 = k64 + 1u;
            const unsigned next_sidx = next_k64 >> 1u;
            const uint32_t ak_off = next_k64 * 64u;
            const unsigned cur_st = k64 & 1u;
            const unsigned nxt_st = next_k64 & 1u;
            const WeaveLdsU8* lds_a_cur  = lds + cur_st * kAStageBytes;
            const WeaveLdsU8* lds_bg_cur = lds + kBBaseOff + cur_st * kBStageBytes;
            const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 1024u);
            WeaveLdsU8* lds_a_nxt = lds + nxt_st * kAStageBytes;

            // Half-A #0 (ap=0,1) + B_next hidden behind mx=0..1 (256 cycles)!
            const uint4 an0 = (a_roff[0] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[0] + ak_off) : make_uint4(0,0,0,0);
            const uint4 an1 = (a_roff[1] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[1] + ak_off) : make_uint4(0,0,0,0);
            const uint4 b_next = weave_gload_u4(
                b_my_exp + ((static_cast<uint64_t>(n16_my) * num_k64 + next_k64) * 64ull + lane) * 16ull);

            weave_lds_s1_k64_mfma_range<0u, 2u, kNXDL>(
                lds_a_cur, lds_bg_cur, lds_bg_cur + 2048u, lds_sa_cur,
                row_base_w, 0u, lane, m_lane, k_grp, sb_g_cur, sb_u_cur, gate_acc, up_acc);

            *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + (0u + a_r64) * 64u + (a_c16 ^ (((0u + a_r64) >> 1u) & 3u)) * 16u) = an0;
            *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + (64u + a_r64) * 64u + (a_c16 ^ (((64u + a_r64) >> 1u) & 3u)) * 16u) = an1;

            // Half-A #1 (ap=2,3) + scales hidden behind mx=2..3 (256 cycles), reusing an0/an1 VGPRs!
            const uint4 an2 = (a_roff[2] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[2] + ak_off) : make_uint4(0,0,0,0);
            const uint4 an3 = (a_roff[3] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[3] + ak_off) : make_uint4(0,0,0,0);
            const float sa_next = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff + next_sidx] : 0.0f;
            const float sb_g_next = weave_as1_arr(bs_gate_row)[next_sidx];
            const float sb_u_next = weave_as1_arr(bs_up_row)[next_sidx];

            weave_lds_s1_k64_mfma_range<2u, 4u, kNXDL>(
                lds_a_cur, lds_bg_cur, lds_bg_cur + 2048u, lds_sa_cur,
                row_base_w, 0u, lane, m_lane, k_grp, sb_g_cur, sb_u_cur, gate_acc, up_acc);

            *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + (128u + a_r64) * 64u + (a_c16 ^ (((128u + a_r64) >> 1u) & 3u)) * 16u) = an2;
            *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + (192u + a_r64) * 64u + (a_c16 ^ (((192u + a_r64) >> 1u) & 3u)) * 16u) = an3;
            *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + nxt_st * kBStageBytes + wave_idx * 1024u + lane * 16u) = b_next;
            reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + nxt_st * 1024u)[tid] = sa_next;
            sb_g_cur = sb_g_next;
            sb_u_cur = sb_u_next;
            weave_4wave_lds_barrier_65440(arena);
        }

        {
            const unsigned cur_st = (num_k64 - 1u) & 1u;
            const WeaveLdsU8* lds_bg_cur = lds + kBBaseOff + cur_st * kBStageBytes;
            weave_lds_s1_k64_mfma_range<0u, 4u, kNXDL>(
                lds + cur_st * kAStageBytes, lds_bg_cur, lds_bg_cur + 2048u,
                reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 1024u),
                row_base_w, 0u, lane, m_lane, k_grp, sb_g_cur, sb_u_cur, gate_acc, up_acc);
        }
        weave_4wave_lds_barrier_65440(arena);

        WeaveLdsU16* lds_sub = reinterpret_cast<WeaveLdsU16*>(lds);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned r = row_base_w + mx * 16u + m_lane;
            #pragma unroll
            for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                const unsigned col = nx * 16u + k_grp * 4u;
                const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx][0]) * up_acc[mx][nx][0])) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx][1]) * up_acc[mx][nx][1])) << 16u);
                const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx][2]) * up_acc[mx][nx][2])) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][nx][3]) * up_acc[mx][nx][3])) << 16u);
                *reinterpret_cast<WeaveLdsU2*>(lds_sub + r * 32u + col) = make_uint2(p0, p1);
            }
        }
        weave_4wave_lds_barrier_65440(arena);

        if (sub_n < 3u) {
            const unsigned s_row = tid >> 2u;
            const unsigned s_lane = tid & 3u;
            #pragma clang loop unroll(disable)
            for (unsigned r0 = 0u; r0 < 256u; r0 += 64u) {
                const unsigned r = r0 + s_row;
                if (r < valid_rows && (row_offset + r) < route_rows) {
                    const uint4 val = *reinterpret_cast<const WeaveLdsU4*>(lds_sub + r * 32u + s_lane * 8u);
                    weave_gstore_u4(stage1_tmp_bf16 + static_cast<uint64_t>(row_offset + r) * N + tile_n * 128u + sub_n * 32u + s_lane * 8u, val);
                }
            }
            weave_4wave_lds_barrier_65440(arena);
        }
    }

    const WeaveLdsU16* lds_sub3 = reinterpret_cast<const WeaveLdsU16*>(lds);
    const unsigned q_row = tid >> 4u;
    const unsigned q_lane = tid & 15u;

    #pragma clang loop unroll(disable)
    for (unsigned r0 = 0u; r0 < 256u; r0 += 16u) {
        const unsigned r = r0 + q_row;
        const unsigned dense_row = row_offset + r;
        const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
        uint4 raw = make_uint4(0,0,0,0);
        if (valid) {
            if (q_lane < 12u) {
                raw = weave_gload_u4(stage1_tmp_bf16 + static_cast<uint64_t>(dense_row) * N + tile_n * 128u + q_lane * 8u);
            } else {
                raw = *reinterpret_cast<const WeaveLdsU4*>(lds_sub3 + r * 32u + (q_lane - 12u) * 8u);
            }
        }
        weave_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

__device__ __forceinline__ void weave_s2_true_bm256(
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

    constexpr unsigned kMXDL = 4u;
    constexpr unsigned kNXDL = 4u;
    constexpr unsigned kAStageBytes = 16384u;
    constexpr unsigned kBBaseOff = 32768u;
    constexpr unsigned kBStageBytes = 4096u;
    constexpr unsigned kSaBaseOff = 40960u;
    constexpr unsigned kTokOff = 43008u;

    const unsigned row_base_w = wave_idx * 64u;
    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu) : route_rows;
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * N * K;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * n_blocks + tile_n) * num_k128;

    const unsigned a_r64 = tid >> 2u;
    const unsigned a_c16 = tid & 3u;
    uint32_t a_roff[4];
    #pragma unroll
    for (unsigned ap = 0u; ap < 4u; ++ap) {
        const unsigned r = ap * 64u + a_r64;
        const uint32_t tok = lds_tok[r];
        a_roff[ap] = ((r < valid_rows) && (tok < route_rows)) ? (tok * K + a_c16 * 16u) : 0xffffffffu;
    }
    const uint32_t tok_sa = lds_tok[tid];
    const uint32_t sa_roff = ((tid < valid_rows) && (tok_sa < route_rows)) ? (tok_sa * num_k128) : 0xffffffffu;

    #pragma clang loop unroll(disable)
    for (unsigned n_half = 0u; n_half < 2u; ++n_half) {
        WeaveFloat4 acc[kMXDL][kNXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            #pragma unroll
            for (unsigned nx = 0u; nx < kNXDL; ++nx) acc[mx][nx] = {0,0,0,0};
        }

        const unsigned n16_my = tile_n * 8u + n_half * 4u + wave_idx;

        #pragma unroll
        for (unsigned ap = 0u; ap < 4u; ++ap) {
            const unsigned r = ap * 64u + a_r64;
            const uint4 av = (a_roff[ap] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[ap]) : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<WeaveLdsU4*>(lds + r * 64u + (a_c16 ^ ((r >> 1u) & 3u)) * 16u) = av;
        }
        *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + wave_idx * 1024u + lane * 16u) =
            weave_gload_u4(b_exp + (static_cast<uint64_t>(n16_my) * num_k64 * 64ull + lane) * 16ull);
        reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff)[tid] = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff] : 0.0f;
        float sb_cur = weave_as1_arr(bs_row)[0];
        weave_4wave_lds_barrier_65440(arena);

        #pragma clang loop unroll(disable)
        for (unsigned k64 = 0u; k64 + 1u < num_k64; ++k64) {
            const unsigned next_k64 = k64 + 1u;
            const unsigned next_sidx = next_k64 >> 1u;
            const uint32_t ak_off = next_k64 * 64u;
            const unsigned cur_st = k64 & 1u;
            const unsigned nxt_st = next_k64 & 1u;
            const WeaveLdsU8* lds_a_cur = lds + cur_st * kAStageBytes;
            const WeaveLdsU8* lds_b_cur = lds + kBBaseOff + cur_st * kBStageBytes;
            const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 1024u);
            WeaveLdsU8* lds_a_nxt = lds + nxt_st * kAStageBytes;

            const uint4 an0 = (a_roff[0] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[0] + ak_off) : make_uint4(0,0,0,0);
            const uint4 an1 = (a_roff[1] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[1] + ak_off) : make_uint4(0,0,0,0);
            const uint4 b_next = weave_gload_u4(
                b_exp + ((static_cast<uint64_t>(n16_my) * num_k64 + next_k64) * 64ull + lane) * 16ull);

            weave_lds_s2_k64_mfma_range<0u, 2u, kNXDL>(
                lds_a_cur, lds_b_cur, lds_sa_cur, row_base_w, 0u, lane, m_lane, k_grp, sb_cur, acc);

            *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + (0u + a_r64) * 64u + (a_c16 ^ (((0u + a_r64) >> 1u) & 3u)) * 16u) = an0;
            *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + (64u + a_r64) * 64u + (a_c16 ^ (((64u + a_r64) >> 1u) & 3u)) * 16u) = an1;

            const uint4 an2 = (a_roff[2] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[2] + ak_off) : make_uint4(0,0,0,0);
            const uint4 an3 = (a_roff[3] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[3] + ak_off) : make_uint4(0,0,0,0);
            const float sa_next = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff + next_sidx] : 0.0f;
            const float sb_next = weave_as1_arr(bs_row)[next_sidx];

            weave_lds_s2_k64_mfma_range<2u, 4u, kNXDL>(
                lds_a_cur, lds_b_cur, lds_sa_cur, row_base_w, 0u, lane, m_lane, k_grp, sb_cur, acc);

            *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + (128u + a_r64) * 64u + (a_c16 ^ (((128u + a_r64) >> 1u) & 3u)) * 16u) = an2;
            *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + (192u + a_r64) * 64u + (a_c16 ^ (((192u + a_r64) >> 1u) & 3u)) * 16u) = an3;
            *reinterpret_cast<WeaveLdsU4*>(lds + kBBaseOff + nxt_st * kBStageBytes + wave_idx * 1024u + lane * 16u) = b_next;
            reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + nxt_st * 1024u)[tid] = sa_next;
            sb_cur = sb_next;
            weave_4wave_lds_barrier_65440(arena);
        }

        {
            const unsigned cur_st = (num_k64 - 1u) & 1u;
            weave_lds_s2_k64_mfma_range<0u, 4u, kNXDL>(
                lds + cur_st * kAStageBytes,
                lds + kBBaseOff + cur_st * kBStageBytes,
                reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 1024u),
                row_base_w, 0u, lane, m_lane, k_grp, sb_cur, acc);
        }
        weave_4wave_lds_barrier_65440(arena);

        WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned r = row_base_w + mx * 16u + m_lane;
            const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
            #pragma unroll
            for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                const unsigned col = nx * 16u + k_grp * 4u;
                const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][0] * weight)) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][1] * weight)) << 16u);
                const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][2] * weight)) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][3] * weight)) << 16u);
                const unsigned c8 = (col >> 2u) ^ (r & 7u);
                *reinterpret_cast<WeaveLdsU2*>(lds_c + r * 64u + c8 * 4u) = make_uint2(p0, p1);
            }
        }
        weave_4wave_lds_barrier_65440(arena);

        const unsigned h_row = tid >> 3u;
        const unsigned h_lane = tid & 7u;
        #pragma unroll
        for (unsigned r0 = 0u; r0 < 256u; r0 += 32u) {
            const unsigned r = r0 + h_row;
            const uint32_t tok_m = lds_tok[r];
            if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
                const unsigned c8_0 = (h_lane * 2u + 0u) ^ (r & 7u);
                const unsigned c8_1 = (h_lane * 2u + 1u) ^ (r & 7u);
                const uint2 u0 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 64u + c8_0 * 4u);
                const uint2 u1 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 64u + c8_1 * 4u);
                weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n * 128u + n_half * 64u + h_lane * 8u, make_uint4(u0.x, u0.y, u1.x, u1.y));
            }
        }
        weave_4wave_lds_barrier_65440(arena);
    }
    asm volatile("" ::: "memory");
}


template <unsigned kMXDL, unsigned kNXDL>
__device__ __forceinline__ void weave_s2_direct_reg_k128_mfma(
    const uint4 (&av0)[kMXDL],
    const uint4 (&av1)[kMXDL],
    const uint4 (&b0)[kNXDL],
    const uint4 (&b1)[kNXDL],
    const float (&sa_m)[kMXDL],
    float sb,
    WeaveFloat4 (&acc)[kMXDL][kNXDL])
{
    int64_t b0_lo[kNXDL], b0_hi[kNXDL], b1_lo[kNXDL], b1_hi[kNXDL];
    #pragma unroll
    for (unsigned nx = 0u; nx < kNXDL; ++nx) {
        b0_lo[nx] = weave_pack_i64(b0[nx].x, b0[nx].y);
        b0_hi[nx] = weave_pack_i64(b0[nx].z, b0[nx].w);
        b1_lo[nx] = weave_pack_i64(b1[nx].x, b1[nx].y);
        b1_hi[nx] = weave_pack_i64(b1[nx].z, b1[nx].w);
    }
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const int64_t a00 = weave_pack_i64(av0[mx].x, av0[mx].y);
        const int64_t a01 = weave_pack_i64(av0[mx].z, av0[mx].w);
        const int64_t a10 = weave_pack_i64(av1[mx].x, av1[mx].y);
        const int64_t a11 = weave_pack_i64(av1[mx].z, av1[mx].w);
        const float sab = sa_m[mx] * sb;
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) {
            WeaveFloat4 t = {0,0,0,0};
            t = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0_lo[nx], a00, t, 0, 0, 0);
            t = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0_hi[nx], a01, t, 0, 0, 0);
            t = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1_lo[nx], a10, t, 0, 0, 0);
            t = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1_hi[nx], a11, t, 0, 0, 0);
            acc[mx][nx] += t * sab;
        }
    }
}

// ============================================================================
// AITER-INSPIRED 1x4 WAVE LAYOUT + BK=128 DOUBLE-BUFFERED A-LDS + ZERO B IN LDS!
// - BM=64  Stage 1: 1-pass 64x128  (kMXDL=4, kNXDL=2 -> 64 acc VGPRs, 16KB A-LDS, 0 B-LDS, 1 barrier/K128!)
// - BM=128 Stage 1: 2-pass 128x64  (kMXDL=8, kNXDL=1 -> 64 acc VGPRs, 32KB A-LDS, 0 B-LDS, 1 barrier/K128, 0 global tmp!)
// - BM=64  Stage 2: 1-pass 64x128  (kMXDL=4, kNXDL=2 -> 32 acc VGPRs, 16KB A-LDS, 0 B-LDS, 1 barrier/K128!)
// - BM=128 Stage 2: 1-pass 128x128 (kMXDL=8, kNXDL=2 -> 64 acc VGPRs, 32KB A-LDS, 0 B-LDS, 1 barrier/K128!)
// ============================================================================
template <unsigned kBM>
__device__ __forceinline__ void weave_s1_1x4_bk128_zerob(
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
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kSubNPasses = (kBM == 64u) ? 1u : 2u;
    constexpr unsigned kMXDL = kBM / 16u;
    constexpr unsigned kNXDL = (kBM == 64u) ? 2u : 1u;
    constexpr unsigned kAPasses = kBM / 32u;
    constexpr unsigned kAStageBytes = kBM * 128u;
    constexpr unsigned kSaBaseOff = (kBM == 64u) ? 16384u : 32768u;
    constexpr unsigned kTokOff = (kBM == 64u) ? 17408u : 33792u;
    constexpr unsigned kHalf0Off = 48128u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s_ids_tile)[tid]) & 0xffffffu) : num_tokens;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint64_t expert_b_bytes = static_cast<uint64_t>(N) * K * 2ull;
    const uint8_t* b_gate_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_b_bytes;
    const uint8_t* b_up_exp = b_gate_exp + static_cast<uint64_t>(N) * K;
    const uint64_t expert_bs_stride = static_cast<uint64_t>(n_blocks) * 2ull * num_k128;
    const float* bs_gate_row = b_scale_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * expert_bs_stride + static_cast<uint64_t>(tile_n) * num_k128;
    const float* bs_up_row = bs_gate_row + static_cast<uint64_t>(n_blocks) * num_k128;

    const unsigned a_r32 = tid >> 3u;
    const unsigned a_c16 = tid & 7u;
    uint32_t a_roff[kAPasses];
    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 32u + a_r32;
        const uint32_t tok = lds_tok[r];
        a_roff[ap] = ((r < valid_rows) && (tok < num_tokens)) ? (tok * K + a_c16 * 16u) : 0xffffffffu;
    }
    const bool sa_active = (tid < kBM);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : num_tokens;
    const uint32_t sa_roff = ((tid < valid_rows) && sa_active && (tok_sa < num_tokens)) ? (tok_sa * num_k128) : 0xffffffffu;

    WeaveLdsU16* lds_half0 = reinterpret_cast<WeaveLdsU16*>(lds + kHalf0Off);

    #pragma clang loop unroll(disable)
    for (unsigned n_half = 0u; n_half < kSubNPasses; ++n_half) {
        WeaveFloat4 gate_acc[kMXDL][kNXDL];
        WeaveFloat4 up_acc[kMXDL][kNXDL];
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            #pragma unroll
            for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                gate_acc[mx][nx] = {0,0,0,0};
                up_acc[mx][nx]   = {0,0,0,0};
            }
        }

        uint64_t b_col_off[kNXDL];
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) {
            const unsigned n16 = (kBM == 64u)
                ? (tile_n * 8u + wave_idx * 2u + nx)
                : (tile_n * 8u + n_half * 4u + wave_idx);
            b_col_off[nx] = (static_cast<uint64_t>(n16) * num_k64 * 64ull + lane) * 16ull;
        }

        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) {
            const unsigned r = ap * 32u + a_r32;
            const uint4 av = (a_roff[ap] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[ap]) : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<WeaveLdsU4*>(lds + r * 128u + (a_c16 ^ (r & 7u)) * 16u) = av;
        }
        if (sa_active) {
            reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff)[tid] = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff] : 0.0f;
        }
        WeaveI64x2 bg0[kNXDL], bu0[kNXDL], bg1[kNXDL], bu1[kNXDL];
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) {
            bg0[nx] = weave_gload_i64x2(b_gate_exp + b_col_off[nx]);
            bu0[nx] = weave_gload_i64x2(b_up_exp   + b_col_off[nx]);
            bg1[nx] = weave_gload_i64x2(b_gate_exp + b_col_off[nx] + 1024ull);
            bu1[nx] = weave_gload_i64x2(b_up_exp   + b_col_off[nx] + 1024ull);
        }
        float sb_g_cur = weave_as1_arr(bs_gate_row)[0];
        float sb_u_cur = weave_as1_arr(bs_up_row)[0];
        weave_4wave_lds_barrier_65440(arena);

        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u; kb + 1u < num_k128; ++kb) {
            const unsigned next_kb = kb + 1u;
            const uint32_t ak_off = next_kb * 128u;
            const uint64_t bk_off = static_cast<uint64_t>(next_kb) * 2048ull;
            const unsigned cur_st = kb & 1u;
            const unsigned nxt_st = next_kb & 1u;
            const WeaveLdsU8* lds_a_cur = lds + cur_st * kAStageBytes;
            const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u);
            WeaveLdsU8* lds_a_nxt = lds + nxt_st * kAStageBytes;

            // Prefetch A_next BEFORE the 32 MFMAs (512 cycles of HBM3 flight time, 0 extra B_next VGPRs -> 0 spills!)
            uint4 a_next[kAPasses];
            #pragma unroll
            for (unsigned ap = 0u; ap < kAPasses; ++ap) {
                a_next[ap] = (a_roff[ap] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[ap] + ak_off) : make_uint4(0u, 0u, 0u, 0u);
            }
            const float sa_next = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff + next_kb] : 0.0f;

            // In-place 2-phase W1 prefetch (nx=0 -> reload nx=0 -> nx=1 hides 512 cycles of HBM3! 0 extra VGPRs!)
            #pragma unroll
            for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                #pragma unroll
                for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                    const unsigned rm = mx * 16u + m_lane;
                    const WeaveI64x2 av0 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
                    const WeaveI64x2 av1 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
                    const float sa_val = lds_sa_cur[rm];
                    const float sab_g = sa_val * sb_g_cur;
                    const float sab_u = sa_val * sb_u_cur;
                    WeaveFloat4 tg = {0,0,0,0}, tu = {0,0,0,0};
                    tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg0[nx][0], av0[0], tg, 0, 0, 0);
                    tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu0[nx][0], av0[0], tu, 0, 0, 0);
                    tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg0[nx][1], av0[1], tg, 0, 0, 0);
                    tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu0[nx][1], av0[1], tu, 0, 0, 0);
                    tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg1[nx][0], av1[0], tg, 0, 0, 0);
                    tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu1[nx][0], av1[0], tu, 0, 0, 0);
                    tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg1[nx][1], av1[1], tg, 0, 0, 0);
                    tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu1[nx][1], av1[1], tu, 0, 0, 0);
                    gate_acc[mx][nx] += tg * sab_g;
                    up_acc[mx][nx]   += tu * sab_u;
                }
                bg0[nx] = weave_gload_i64x2(b_gate_exp + b_col_off[nx] + bk_off);
                bu0[nx] = weave_gload_i64x2(b_up_exp   + b_col_off[nx] + bk_off);
                bg1[nx] = weave_gload_i64x2(b_gate_exp + b_col_off[nx] + bk_off + 1024ull);
                bu1[nx] = weave_gload_i64x2(b_up_exp   + b_col_off[nx] + bk_off + 1024ull);
            }
            const float sb_g_next = weave_as1_arr(bs_gate_row)[next_kb];
            const float sb_u_next = weave_as1_arr(bs_up_row)[next_kb];

            #pragma unroll
            for (unsigned ap = 0u; ap < kAPasses; ++ap) {
                const unsigned r = ap * 32u + a_r32;
                *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + r * 128u + (a_c16 ^ (r & 7u)) * 16u) = a_next[ap];
            }
            if (sa_active) {
                reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + nxt_st * 512u)[tid] = sa_next;
            }
            sb_g_cur = sb_g_next;
            sb_u_cur = sb_u_next;
            weave_4wave_lds_barrier_65440(arena);
        }

        {
            const unsigned cur_st = (num_k128 - 1u) & 1u;
            const WeaveLdsU8* lds_a_cur = lds + cur_st * kAStageBytes;
            const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u);
            #pragma unroll
            for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                const unsigned rm = mx * 16u + m_lane;
                const WeaveI64x2 av0 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
                const WeaveI64x2 av1 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
                const float sa_val = lds_sa_cur[rm];
                const float sab_g = sa_val * sb_g_cur;
                const float sab_u = sa_val * sb_u_cur;
                #pragma unroll
                for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                    WeaveFloat4 tg = {0,0,0,0}, tu = {0,0,0,0};
                    tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg0[nx][0], av0[0], tg, 0, 0, 0);
                    tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu0[nx][0], av0[0], tu, 0, 0, 0);
                    tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg0[nx][1], av0[1], tg, 0, 0, 0);
                    tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu0[nx][1], av0[1], tu, 0, 0, 0);
                    tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg1[nx][0], av1[0], tg, 0, 0, 0);
                    tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu1[nx][0], av1[0], tu, 0, 0, 0);
                    tg = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bg1[nx][1], av1[1], tg, 0, 0, 0);
                    tu = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bu1[nx][1], av1[1], tu, 0, 0, 0);
                    gate_acc[mx][nx] += tg * sab_g;
                    up_acc[mx][nx]   += tu * sab_u;
                }
            }
        }
        weave_4wave_lds_barrier_65440(arena);

        if constexpr (kBM == 64u) {
            WeaveLdsU16* lds_q = reinterpret_cast<WeaveLdsU16*>(lds);
            #pragma unroll
            for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                const unsigned r = mx * 16u + m_lane;
                #pragma unroll
                for (unsigned nx = 0u; nx < kNXDL; ++nx) {
                    const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
                    const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                        weave_bk64_silu(gate_acc[mx][nx][0]) * up_acc[mx][nx][0])) |
                        (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                        weave_bk64_silu(gate_acc[mx][nx][1]) * up_acc[mx][nx][1])) << 16u);
                    const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                        weave_bk64_silu(gate_acc[mx][nx][2]) * up_acc[mx][nx][2])) |
                        (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                        weave_bk64_silu(gate_acc[mx][nx][3]) * up_acc[mx][nx][3])) << 16u);
                    *reinterpret_cast<WeaveLdsU2*>(lds_q + r * 128u + col) = make_uint2(p0, p1);
                }
            }
            weave_4wave_lds_barrier_65440(arena);
        } else {
            WeaveLdsU16* dst_half = (n_half == 0u) ? lds_half0 : reinterpret_cast<WeaveLdsU16*>(lds);
            #pragma unroll
            for (unsigned mx = 0u; mx < kMXDL; ++mx) {
                const unsigned r = mx * 16u + m_lane;
                const unsigned col = wave_idx * 16u + k_grp * 4u;
                const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][0][0]) * up_acc[mx][0][0])) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][0][1]) * up_acc[mx][0][1])) << 16u);
                const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][0][2]) * up_acc[mx][0][2])) |
                    (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(
                    weave_bk64_silu(gate_acc[mx][0][3]) * up_acc[mx][0][3])) << 16u);
                *reinterpret_cast<WeaveLdsU2*>(dst_half + r * 64u + col) = make_uint2(p0, p1);
            }
            weave_4wave_lds_barrier_65440(arena);
        }
    }

    unsigned q_tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(q_tid));
    const unsigned q_row = q_tid >> 4u;
    const unsigned q_lane = q_tid & 15u;
    if constexpr (kBM == 64u) {
        const WeaveLdsU16* lds_q = reinterpret_cast<const WeaveLdsU16*>(lds);
        #pragma clang loop unroll(disable)
        for (unsigned r0 = 0u; r0 < 64u; r0 += 16u) {
            const unsigned r = r0 + q_row;
            const unsigned dense_row = row_offset + r;
            const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
            uint4 raw = make_uint4(0,0,0,0);
            if (valid) { raw = *reinterpret_cast<const WeaveLdsU4*>(lds_q + r * 128u + q_lane * 8u); }
            weave_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
        }
    } else {
        const WeaveLdsU16* lds_half1 = reinterpret_cast<const WeaveLdsU16*>(lds);
        #pragma clang loop unroll(disable)
        for (unsigned r0 = 0u; r0 < 128u; r0 += 16u) {
            const unsigned r = r0 + q_row;
            const unsigned dense_row = row_offset + r;
            const bool valid = (r < valid_rows) && (dense_row < route_rows) && (lds_tok[r] < num_tokens);
            const WeaveLdsU16* src_p = (q_lane < 8u)
                ? (lds_half0 + r * 64u + q_lane * 8u)
                : (lds_half1 + r * 64u + (q_lane - 8u) * 8u);
            uint4 raw = make_uint4(0,0,0,0);
            if (valid) { raw = *reinterpret_cast<const WeaveLdsU4*>(src_p); }
            weave_quant_bf16x8_store(raw, valid, q_lane, static_cast<uint64_t>(dense_row) * n_blocks + tile_n, quant_out_base, quant_scale_base);
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

template <unsigned kBM>
__device__ __forceinline__ void weave_s2_1x4_bk128_zerob(
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
    static_assert(kBM == 64u || kBM == 128u);
    asm volatile("" ::: "memory");
    unsigned tid = threadIdx.x & 255u;
    asm volatile("" : "+v"(tid));
    WeaveLdsU8* const lds = weave_lds_ptr(arena);

    const unsigned wave_idx = tid >> 6u;
    const unsigned lane = tid & 63u;
    const unsigned m_lane = lane & 15u;
    const unsigned k_grp = lane >> 4u;

    constexpr unsigned kMXDL = kBM / 16u;
    constexpr unsigned kNXDL = 2u;
    constexpr unsigned kAPasses = kBM / 32u;
    constexpr unsigned kAStageBytes = kBM * 128u;
    constexpr unsigned kSaBaseOff = (kBM == 64u) ? 16384u : 32768u;
    constexpr unsigned kTokOff = (kBM == 64u) ? 17408u : 33792u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokOff);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu) : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks = N >> 7u;
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * N * K;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * n_blocks + tile_n) * num_k128;

    const unsigned a_r32 = tid >> 3u;
    const unsigned a_c16 = tid & 7u;
    uint32_t a_roff[kAPasses];
    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 32u + a_r32;
        const uint32_t tok = lds_tok[r];
        a_roff[ap] = ((r < valid_rows) && (tok < route_rows)) ? (tok * K + a_c16 * 16u) : 0xffffffffu;
    }
    const bool sa_active = (tid < kBM);
    const uint32_t tok_sa = sa_active ? lds_tok[tid] : route_rows;
    const uint32_t sa_roff = ((tid < valid_rows) && sa_active && (tok_sa < route_rows)) ? (tok_sa * num_k128) : 0xffffffffu;

    WeaveFloat4 acc[kMXDL][kNXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) acc[mx][nx] = {0,0,0,0};
    }

    uint64_t b_col_off[2];
    b_col_off[0] = (static_cast<uint64_t>(tile_n * 8u + wave_idx * 2u + 0u) * num_k64 * 64ull + lane) * 16ull;
    b_col_off[1] = (static_cast<uint64_t>(tile_n * 8u + wave_idx * 2u + 1u) * num_k64 * 64ull + lane) * 16ull;

    #pragma unroll
    for (unsigned ap = 0u; ap < kAPasses; ++ap) {
        const unsigned r = ap * 32u + a_r32;
        const uint4 av = (a_roff[ap] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[ap]) : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<WeaveLdsU4*>(lds + r * 128u + (a_c16 ^ (r & 7u)) * 16u) = av;
    }
    if (sa_active) {
        reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff)[tid] = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff] : 0.0f;
    }
    WeaveI64x2 bw0[2], bw1[2];
    bw0[0] = weave_gload_i64x2(b_exp + b_col_off[0]);
    bw0[1] = weave_gload_i64x2(b_exp + b_col_off[1]);
    bw1[0] = weave_gload_i64x2(b_exp + b_col_off[0] + 1024ull);
    bw1[1] = weave_gload_i64x2(b_exp + b_col_off[1] + 1024ull);
    float sb_cur = weave_as1_arr(bs_row)[0];
    weave_4wave_lds_barrier_65440(arena);

    #pragma clang loop unroll(disable)
    for (unsigned kb = 0u; kb + 1u < num_k128; ++kb) {
        const unsigned next_kb = kb + 1u;
        const uint32_t ak_off = next_kb * 128u;
        const uint64_t bk_off = static_cast<uint64_t>(next_kb) * 2048ull;
        const unsigned cur_st = kb & 1u;
        const unsigned nxt_st = next_kb & 1u;
        const WeaveLdsU8* lds_a_cur = lds + cur_st * kAStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u);
        WeaveLdsU8* lds_a_nxt = lds + nxt_st * kAStageBytes;

        // Prefetch ALL of A_next AND ALL of B_next BEFORE the 16..32 MFMAs (only 72..112 VGPRs total -> 0 spills!)
        uint4 a_next[kAPasses];
        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) {
            a_next[ap] = (a_roff[ap] != 0xffffffffu) ? weave_gload_u4(a_base + a_roff[ap] + ak_off) : make_uint4(0u, 0u, 0u, 0u);
        }
        const WeaveI64x2 bw0_n0 = weave_gload_i64x2(b_exp + b_col_off[0] + bk_off);
        const WeaveI64x2 bw0_n1 = weave_gload_i64x2(b_exp + b_col_off[1] + bk_off);
        const WeaveI64x2 bw1_n0 = weave_gload_i64x2(b_exp + b_col_off[0] + bk_off + 1024ull);
        const WeaveI64x2 bw1_n1 = weave_gload_i64x2(b_exp + b_col_off[1] + bk_off + 1024ull);
        const float sa_next = (sa_roff != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_roff + next_kb] : 0.0f;
        const float sb_next = weave_as1_arr(bs_row)[next_kb];

        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned rm = mx * 16u + m_lane;
            const WeaveI64x2 av0 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
            const WeaveI64x2 av1 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
            const float sab = lds_sa_cur[rm] * sb_cur;
            WeaveFloat4 t0 = {0,0,0,0}, t1 = {0,0,0,0};
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw0[0][0], av0[0], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw0[1][0], av0[0], t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw0[0][1], av0[1], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw0[1][1], av0[1], t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw1[0][0], av1[0], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw1[1][0], av1[0], t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw1[0][1], av1[1], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw1[1][1], av1[1], t1, 0, 0, 0);
            acc[mx][0] += t0 * sab;
            acc[mx][1] += t1 * sab;
        }

        bw0[0] = bw0_n0; bw0[1] = bw0_n1;
        bw1[0] = bw1_n0; bw1[1] = bw1_n1;

        #pragma unroll
        for (unsigned ap = 0u; ap < kAPasses; ++ap) {
            const unsigned r = ap * 32u + a_r32;
            *reinterpret_cast<WeaveLdsU4*>(lds_a_nxt + r * 128u + (a_c16 ^ (r & 7u)) * 16u) = a_next[ap];
        }
        if (sa_active) {
            reinterpret_cast<WeaveLdsF32*>(lds + kSaBaseOff + nxt_st * 512u)[tid] = sa_next;
        }
        sb_cur = sb_next;
        weave_4wave_lds_barrier_65440(arena);
    }

    {
        const unsigned cur_st = (num_k128 - 1u) & 1u;
        const WeaveLdsU8* lds_a_cur = lds + cur_st * kAStageBytes;
        const WeaveLdsF32* lds_sa_cur = reinterpret_cast<const WeaveLdsF32*>(lds + kSaBaseOff + cur_st * 512u);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const unsigned rm = mx * 16u + m_lane;
            const WeaveI64x2 av0 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + (k_grp ^ (rm & 7u)) * 16u);
            const WeaveI64x2 av1 = weave_lds_load_i64x2(lds_a_cur + rm * 128u + ((4u + k_grp) ^ (rm & 7u)) * 16u);
            const float sab = lds_sa_cur[rm] * sb_cur;
            WeaveFloat4 t0 = {0,0,0,0}, t1 = {0,0,0,0};
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw0[0][0], av0[0], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw0[1][0], av0[0], t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw0[0][1], av0[1], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw0[1][1], av0[1], t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw1[0][0], av1[0], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw1[1][0], av1[0], t1, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw1[0][1], av1[1], t0, 0, 0, 0);
            t1 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(bw1[1][1], av1[1], t1, 0, 0, 0);
            acc[mx][0] += t0 * sab;
            acc[mx][1] += t1 * sab;
        }
    }

    weave_4wave_lds_barrier_65440(arena);
    WeaveLdsU16* lds_c = reinterpret_cast<WeaveLdsU16*>(lds);
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        const unsigned r = mx * 16u + m_lane;
        const float weight = (r < valid_rows) ? weave_as1_arr(weights_tile)[r] : 0.0f;
        #pragma unroll
        for (unsigned nx = 0u; nx < kNXDL; ++nx) {
            const unsigned col = (wave_idx * 2u + nx) * 16u + k_grp * 4u;
            const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][0] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][1] * weight)) << 16u);
            const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][2] * weight)) |
                (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][nx][3] * weight)) << 16u);
            const unsigned c8 = (col >> 2u) ^ (r & 7u);
            *reinterpret_cast<WeaveLdsU2*>(lds_c + r * 128u + c8 * 4u) = make_uint2(p0, p1);
        }
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned c_row = tid >> 4u;
    const unsigned c_lane = tid & 15u;
    #pragma clang loop unroll(disable)
    for (unsigned r0 = 0u; r0 < kBM; r0 += 16u) {
        const unsigned r = r0 + c_row;
        const uint32_t tok_m = lds_tok[r];
        if (r < valid_rows && tok_m < route_rows && (row_offset + r) < route_rows) {
            const unsigned c8_0 = (c_lane * 2u + 0u) ^ (r & 7u);
            const unsigned c8_1 = (c_lane * 2u + 1u) ^ (r & 7u);
            const uint2 u0 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_0 * 4u);
            const uint2 u1 = *reinterpret_cast<const WeaveLdsU2*>(lds_c + r * 128u + c8_1 * 4u);
            weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n * 128u + c_lane * 8u, make_uint4(u0.x, u0.y, u1.x, u1.y));
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}

#include "v62_s1_1x4.hpp"  // V62 GEMM1 1x4 BM 16/32/64 body (V62_G1_BODY / V62_G1_BM32 / V62_DYN_BM)
#include "v62_s2_bn256.hpp"  // V62 GEMM2 1x4 BN=256 body (used only when V62_G2_BN256 != 0)
#include "v62_s2_bn512.hpp"  // V62 GEMM2 1x4 BN=512 body (used only when V62_G2_BN512 != 0)
#include "v62_s2_bn384.hpp"  // V62 GEMM2 1x4 BN=384 body (used only when V62_G2_BN384 != 0)
#include "v61_s2_pf2.hpp"  // V61 GEMM2 1x4 distance-2 pipeline (used only when V61_G2_1X4 == 2)
#include "weave_gemm_400t_v6.hpp"

template <unsigned kBM>
__device__ __forceinline__ void weave_dispatch_stage1_bk64(
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
    unsigned valid_rows,
    unsigned tile_n,
    unsigned num_tokens,
    unsigned route_rows,
    unsigned N,
    unsigned K,
    unsigned char* arena)
{
    if constexpr (kBM == 16u || kBM == 32u) {
        (void)stage1_tmp_bf16;
        weave_s1_small_dispatch<kBM>(s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            quant_out_base, quant_scale_base, row_offset, valid_rows, tile_n, num_tokens, route_rows, N, K, arena);
    } else if constexpr (kBM == 64u || kBM == 128u) {
        (void)stage1_tmp_bf16;
        weave_s1_400t_2x2<kBM>(s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            quant_out_base, quant_scale_base, row_offset, valid_rows, tile_n, num_tokens, route_rows, N, K, arena);
    } else {
        static_assert(kBM == 256u);
        weave_s1_true_bm256(s_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            stage1_tmp_bf16, quant_out_base, quant_scale_base, row_offset, valid_rows, tile_n, num_tokens, route_rows, N, K, arena);
    }
}

template <unsigned kBM>
__device__ __forceinline__ void weave_dispatch_stage2_bk64(
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
    if constexpr (kBM == 16u || kBM == 32u) {
        weave_s2_small_dispatch<kBM>(s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            weights, c_out_base, row_offset, valid_rows, tile_n, route_rows, N, K, arena);
    } else if constexpr (kBM == 64u || kBM == 128u) {
        weave_s2_400t_2x2<kBM>(s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            weights, c_out_base, row_offset, valid_rows, tile_n, route_rows, N, K, arena);
    } else {
        static_assert(kBM == 256u);
        weave_s2_true_bm256(s2_ids, expert_id, a_base, b_base, a_scale_base, b_scale_base,
            weights, c_out_base, row_offset, valid_rows, tile_n, route_rows, N, K, arena);
    }
}

// ============================================================================
// Barrier-Free BN=64 Stage-2 Kernel for BM in {16, 32, 64} (1x4 Waves, kNXDL=1!)
// Doubles CTA parallelism across 80 CUs at S<=1024 AND makes BM=64 S2 100% barrier-free!
// ============================================================================
template <unsigned kBM>
__device__ __forceinline__ void weave_s2_small_1x4_bn64(
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
    constexpr unsigned kTokenCacheOffset = 16384u;

    WeaveLdsU32* lds_tok = reinterpret_cast<WeaveLdsU32*>(lds + kTokenCacheOffset);
    if (tid < kBM) {
        lds_tok[tid] = (tid < valid_rows) ? (static_cast<uint32_t>(weave_as1_arr(s2_ids_tile)[tid]) & 0xffffffu) : route_rows;
    }
    weave_4wave_lds_barrier_65440(arena);

    const unsigned num_k128 = K >> 7u;
    const unsigned num_k64 = K >> 6u;
    const unsigned n_blocks128 = N >> 7u;
    const unsigned tile_n128 = tile_n64 >> 1u;
    const uint8_t* b_exp = b_base + static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * N * K;
    const float* bs_row = b_scale_base + (static_cast<uint64_t>(static_cast<uint32_t>(expert_id)) * n_blocks128 + tile_n128) * num_k128;

    const unsigned n16_col = tile_n64 * 4u + wave_idx;

    WeaveFloat4 acc[kMXDL];
    uint32_t a_m_off[kMXDL];
    uint32_t sa_m_off[kMXDL];
    #pragma unroll
    for (unsigned mx = 0u; mx < kMXDL; ++mx) {
        acc[mx] = {0,0,0,0};
        const unsigned r = mx * 16u + m_lane;
        const uint32_t tok = lds_tok[r];
        const bool ok = (r < valid_rows) && (tok < route_rows);
        a_m_off[mx]  = ok ? (tok * K + k_grp * 16u) : 0xffffffffu;
        sa_m_off[mx] = ok ? (tok * num_k128) : 0xffffffffu;
    }

    auto s2_bn64_half = [&](const uint4 (&av)[kMXDL], const uint4& bc, const float (&sam)[kMXDL], float sbc) {
        const int64_t b_lo = weave_pack_i64(bc.x, bc.y);
        const int64_t b_hi = weave_pack_i64(bc.z, bc.w);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const int64_t a0 = weave_pack_i64(av[mx].x, av[mx].y);
            const int64_t a1 = weave_pack_i64(av[mx].z, av[mx].w);
            WeaveFloat4 t0 = {0,0,0,0};
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b_lo, a0, t0, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b_hi, a1, t0, 0, 0, 0);
            acc[mx] += t0 * (sam[mx] * sbc);
        }
    };

    auto load_half = [&](unsigned k64_idx, uint4 (&av)[kMXDL], uint4& bc) {
        const uint32_t ak = k64_idx * 64u;
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            av[mx] = (a_m_off[mx] != 0xffffffffu) ? weave_gload_u4(a_base + a_m_off[mx] + ak) : make_uint4(0,0,0,0);
        }
        const uint64_t boff = ((static_cast<uint64_t>(n16_col) * num_k64 + k64_idx) * 64ull + lane) * 16ull;
        bc = weave_gload_u4(b_exp + boff);
    };

    auto load_scales = [&](unsigned k128_idx, float (&sa)[kMXDL], float& sb) {
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            sa[mx] = (sa_m_off[mx] != 0xffffffffu) ? weave_as1_arr(a_scale_base)[sa_m_off[mx] + k128_idx] : 0.0f;
        }
        sb = weave_as1_arr(bs_row)[k128_idx];
    };

    auto s2_bn64_k128 = [&](const uint4 (&av0)[kMXDL], const uint4 (&av1)[kMXDL],
                            const uint4& b0, const uint4& b1, const float (&sam)[kMXDL], float sbc) {
        const int64_t b0_lo = weave_pack_i64(b0.x, b0.y), b0_hi = weave_pack_i64(b0.z, b0.w);
        const int64_t b1_lo = weave_pack_i64(b1.x, b1.y), b1_hi = weave_pack_i64(b1.z, b1.w);
        #pragma unroll
        for (unsigned mx = 0u; mx < kMXDL; ++mx) {
            const int64_t a00 = weave_pack_i64(av0[mx].x, av0[mx].y), a01 = weave_pack_i64(av0[mx].z, av0[mx].w);
            const int64_t a10 = weave_pack_i64(av1[mx].x, av1[mx].y), a11 = weave_pack_i64(av1[mx].z, av1[mx].w);
            WeaveFloat4 t0 = {0,0,0,0};
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0_lo, a00, t0, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b0_hi, a01, t0, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1_lo, a10, t0, 0, 0, 0);
            t0 = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b1_hi, a11, t0, 0, 0, 0);
            acc[mx] += t0 * (sam[mx] * sbc);
        }
    };
    if ((num_k128 >= 2u) && ((num_k128 & 1u) == 0u)) {
        uint4 av0_a[kMXDL], av1_a[kMXDL], b0_a, b1_a; float sa_a[kMXDL], sb_a;
        uint4 av0_b[kMXDL], av1_b[kMXDL], b0_b, b1_b; float sa_b[kMXDL], sb_b;
        load_half(0u, av0_a, b0_a); load_scales(0u, sa_a, sb_a); load_half(1u, av1_a, b1_a);
        load_half(2u, av0_b, b0_b); load_scales(1u, sa_b, sb_b); load_half(3u, av1_b, b1_b);

        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u; kb + 2u < num_k128; kb += 2u) {
            s2_bn64_k128(av0_a, av1_a, b0_a, b1_a, sa_a, sb_a);
            load_half((kb + 2u) * 2u + 0u, av0_a, b0_a); load_scales(kb + 2u, sa_a, sb_a); load_half((kb + 2u) * 2u + 1u, av1_a, b1_a);

            s2_bn64_k128(av0_b, av1_b, b0_b, b1_b, sa_b, sb_b);
            load_half((kb + 3u) * 2u + 0u, av0_b, b0_b); load_scales(kb + 3u, sa_b, sb_b); load_half((kb + 3u) * 2u + 1u, av1_b, b1_b);
        }
        s2_bn64_k128(av0_a, av1_a, b0_a, b1_a, sa_a, sb_a);
        s2_bn64_k128(av0_b, av1_b, b0_b, b1_b, sa_b, sb_b);
    } else {
        #pragma clang loop unroll(disable)
        for (unsigned kb = 0u; kb < num_k128; ++kb) {
            uint4 av0[kMXDL], av1[kMXDL], b0, b1;
            float sa[kMXDL], sb;
            load_half(kb * 2u + 0u, av0, b0); load_scales(kb, sa, sb); load_half(kb * 2u + 1u, av1, b1);
            s2_bn64_k128(av0, av1, b0, b1, sa, sb);
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
        const uint32_t p0 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][0] * weight)) |
            (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][1] * weight)) << 16u);
        const uint32_t p1 = static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][2] * weight)) |
            (static_cast<uint32_t>(weave_bk64_f32_to_bf16_bits(acc[mx][3] * weight)) << 16u);
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
                weave_gstore_u4(c_out_base + static_cast<uint64_t>(tok_m) * N + tile_n64 * 64u + h_lane * 8u, make_uint4(u0.x, u0.y, u1.x, u1.y));
            }
        }
    }
    weave_4wave_lds_barrier_65440(arena);
    asm volatile("" ::: "memory");
}
