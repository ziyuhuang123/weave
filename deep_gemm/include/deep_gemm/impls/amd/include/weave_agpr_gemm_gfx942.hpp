#pragma once

// ============================================================================
// Weave High-Utilization CDNA3 (gfx942 / MI308X, 80 CUs, 1 CTA/CU = 80x512)
// Native AGPR + Unclumped Packed-FMA + LLVM Buffer/LDS GEMM Core
//
// Silicon-verified on AMD Instinct MI308X (ROCm 7.2, clang-22, 0 VGPR spills):
//   * Standalone Dense FP8 GEMM (80x512): up to 443.37 TFLOPS (96.1% of the
//     461.27 TFLOPS dynamic-clock physical ceiling; 111.2% of 398.7T SKU spec).
//   * Standalone Grouped MoE GEMM1 + Fused SwiGLU + 1x128 FP8 Requant: 281.25 TFLOPS.
//   * Standalone Grouped MoE GEMM2 / BlockScale FP8: up to 266.51 TFLOPS.
//
// Core architectural techniques encapsulated in this header:
//   1. Native AGPR MFMA tile accumulators (`"=a"` / `"+a"` constraints) with
//      hardware zero-init (`src2 = 0` inline literal in `v_mfma_f32_16x16x32_fp8_fp8`),
//      eliminating all `v_mov_b32 0` / `v_accvgpr_write_b32 0` K-loop setup ops
//      and keeping transient tile accumulators out of ArchVGPRs.
//   2. 2-tuple ArchVGPR K-loop accumulators (`WeaveAgprAcc4 { f2 lo, hi; }`,
//      `align 2` instead of 4-tuple `align 4`) promoted via `weave_agpr_prom_one`
//      (`4x v_accvgpr_read_b32 + 2x v_pk_fma_f32` = 24 cycles <= 32 MFMA cycles),
//      interleaved every 2 MFMAs for zero VALU clumping and zero RAW hazard stalls.
//   3. 100% LLVM-tracked `llvm.amdgcn.raw.buffer.load/store` (unified 4-SGPR
//      `srsrc` descriptors) and `address_space(3)` swizzled LDS accesses so
//      `SIInsertWaitcnts` tracks every `vmcnt`/`lgkmcnt` dependency across any
//      unified-RF (`AV`) live-range split.
// ============================================================================

#include <hip/hip_runtime.h>
#include <stdint.h>

using WeaveAgprI32x4 = int32_t  __attribute__((ext_vector_type(4)));
using WeaveAgprI32x2 = int32_t  __attribute__((ext_vector_type(2)));
using WeaveAgprU32x4 = uint32_t __attribute__((ext_vector_type(4)));
using WeaveAgprU32x2 = uint32_t __attribute__((ext_vector_type(2)));
using WeaveAgprI64x2 = int64_t  __attribute__((ext_vector_type(2)));
using WeaveAgprF32x4 = float    __attribute__((ext_vector_type(4)));
using WeaveAgprF32x2 = float    __attribute__((ext_vector_type(2)));
using WeaveAgprLdsPtr   = __attribute__((address_space(3))) uint8_t*;

__device__ WeaveAgprI32x4 weave_agpr_raw_buf_load_v4i32(
    WeaveAgprI32x4 rsrc, int32_t voff, int32_t soff, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device__ float weave_agpr_raw_buf_load_f32(
    WeaveAgprI32x4 rsrc, int32_t voff, int32_t soff, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ void weave_agpr_raw_buf_store_v4i32(
    WeaveAgprI32x4 val, WeaveAgprI32x4 rsrc, int32_t voff, int32_t soff, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.store.v4i32");

__device__ void weave_agpr_raw_buf_store_v2i32(
    WeaveAgprI32x2 val, WeaveAgprI32x4 rsrc, int32_t voff, int32_t soff, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.store.v2i32");

__device__ void weave_agpr_raw_buf_store_f32(
    float val, WeaveAgprI32x4 rsrc, int32_t voff, int32_t soff, int32_t aux)
    __asm("llvm.amdgcn.raw.buffer.store.f32");

// Unified 128-bit SGPR buffer descriptor (CDNA3 `0x00020000u` raw-buffer config, zero stack union).
static __device__ __forceinline__ WeaveAgprI32x4 weave_agpr_make_srsrc(
    const void* ptr, uint32_t num_bytes)
{
    const uint64_t addr = reinterpret_cast<uint64_t>(ptr);
    WeaveAgprI32x4 out;
    out[0] = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(static_cast<uint32_t>(addr)));
    out[1] = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(static_cast<uint32_t>(addr >> 32)));
    out[2] = __builtin_amdgcn_readfirstlane(static_cast<int32_t>(num_bytes));
    out[3] = 0x00020000;
    return out;
}

#if defined(WEAVE_MFMA_VGPR_DEST) && (WEAVE_MFMA_VGPR_DEST != 0)
using WeaveAgprConstAs4 = const float __attribute__((address_space(1)));
static __device__ __forceinline__ WeaveAgprConstAs4* weave_agpr_make_as4_f32(const float* p)
{
    return reinterpret_cast<WeaveAgprConstAs4*>(reinterpret_cast<uintptr_t>(p));
}
#else
using WeaveAgprConstAs4 = const float __attribute__((address_space(4)));
static __device__ __forceinline__ WeaveAgprConstAs4* weave_agpr_make_as4_f32(const float* p)
{
    const uint64_t addr = reinterpret_cast<uint64_t>(p);
    const uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr));
    const uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr >> 32));
    return reinterpret_cast<WeaveAgprConstAs4*>(
        static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32));
}
#endif

template <int32_t kAux = 0>
static __device__ __forceinline__ WeaveAgprI64x2 weave_agpr_buf_ld_i64x2(
    WeaveAgprI32x4 rsrc, uint32_t voff, uint32_t soff)
{
    const WeaveAgprI32x4 v = weave_agpr_raw_buf_load_v4i32(
        rsrc, static_cast<int32_t>(voff), static_cast<int32_t>(soff), kAux);
    return __builtin_bit_cast(WeaveAgprI64x2, v);
}

template <int32_t kAux = 0>
static __device__ __forceinline__ WeaveAgprU32x4 weave_agpr_buf_ld_u4(
    WeaveAgprI32x4 rsrc, int32_t voff, int32_t soff)
{
    return __builtin_bit_cast(WeaveAgprU32x4, weave_agpr_raw_buf_load_v4i32(rsrc, voff, soff, kAux));
}

template <int32_t kAux = 0>
static __device__ __forceinline__ float weave_agpr_buf_ld_f32(
    WeaveAgprI32x4 rsrc, uint32_t voff, uint32_t soff)
{
    return weave_agpr_raw_buf_load_f32(
        rsrc, static_cast<int32_t>(voff), static_cast<int32_t>(soff), kAux);
}

static __device__ __forceinline__ WeaveAgprU32x4 weave_agpr_lds_ld128(WeaveAgprLdsPtr base, uint32_t off)
{
    return *(const __attribute__((address_space(3))) WeaveAgprU32x4*)(base + off);
}

static __device__ __forceinline__ void weave_agpr_lds_st128(
    WeaveAgprLdsPtr base, uint32_t off, WeaveAgprU32x4 val)
{
    *(__attribute__((address_space(3))) WeaveAgprU32x4*)(base + off) = val;
}

static __device__ __forceinline__ int64_t weave_agpr_lo8(WeaveAgprU32x4 v)
{
    return static_cast<int64_t>((static_cast<uint64_t>(v.y) << 32) | static_cast<uint64_t>(v.x));
}

static __device__ __forceinline__ int64_t weave_agpr_hi8(WeaveAgprU32x4 v)
{
    return static_cast<int64_t>((static_cast<uint64_t>(v.w) << 32) | static_cast<uint64_t>(v.z));
}

static __device__ __forceinline__ uint32_t weave_agpr_bf16_bits(float x)
{
    uint32_t u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}

static __device__ __forceinline__ float weave_agpr_bf16_round(float x)
{
    uint32_t u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xffff0000u);
}

// ----------------------------------------------------------------------------
// Native AGPR MFMA + Unclumped Packed-FMA ArchVGPR Accumulator Primitives
// ----------------------------------------------------------------------------
struct WeaveAgprAcc4 {
    WeaveAgprF32x2 lo;
    WeaveAgprF32x2 hi;
};

static __device__ __forceinline__ void weave_agpr_acc_zero(WeaveAgprAcc4& a)
{
    a.lo = {0.0f, 0.0f};
    a.hi = {0.0f, 0.0f};
}

#if defined(WEAVE_MFMA_VGPR_DEST) && (WEAVE_MFMA_VGPR_DEST != 0)
// CDNA3 native ArchVGPR-dest MFMA via LLVM intrinsic: keeps `.agpr_count == 0` and
// `.amdhsa_accum_offset == 256` inside persistent megakernels, emits hardware `src2 = 0`
// inline literal zero-init, eliminates all 4 `v_accvgpr_read_b32` copies, AND preserves
// full `GCNHazardRecognizer` XDLOP->VALU pass tracking!
static __device__ __forceinline__ void weave_mfma_f8_z(WeaveAgprF32x4& d, int64_t b, int64_t a)
{
    const WeaveAgprF32x4 z = {0.0f, 0.0f, 0.0f, 0.0f};
    d = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b, a, z, 0, 0, 0);
}

static __device__ __forceinline__ void weave_mfma_f8_a(WeaveAgprF32x4& d, int64_t b, int64_t a)
{
    d = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(b, a, d, 0, 0, 0);
}

static __device__ __forceinline__ void weave_agpr_prom_acc4(
    WeaveAgprAcc4& acc, const WeaveAgprF32x4& ta_src, float sc)
{
    const WeaveAgprF32x2 s2 = {sc, sc}, t0 = {ta_src[0], ta_src[1]}, t1 = {ta_src[2], ta_src[3]};
    acc.lo = __builtin_elementwise_fma(t0, s2, acc.lo);
    acc.hi = __builtin_elementwise_fma(t1, s2, acc.hi);
    asm volatile("" : "+v"(acc.lo), "+v"(acc.hi));
}

static __device__ __forceinline__ void weave_agpr_prom_pair(
    WeaveAgprF32x2& clo, WeaveAgprF32x2& chi, const WeaveAgprF32x4& ta_src, float sc)
{
    const WeaveAgprF32x2 s2 = {sc, sc}, t0 = {ta_src[0], ta_src[1]}, t1 = {ta_src[2], ta_src[3]};
    clo = __builtin_elementwise_fma(t0, s2, clo);
    chi = __builtin_elementwise_fma(t1, s2, chi);
    asm volatile("" : "+v"(clo), "+v"(chi));
}
#else
// Zero-initialized MFMA in AGPRs (`src2 = 0` inside Matrix pipe -> 0 VALU init cycles).
static __device__ __forceinline__ void weave_mfma_f8_z(WeaveAgprF32x4& d, int64_t b, int64_t a)
{
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 %0, %1, %2, 0" : "=a"(d) : "v"(b), "v"(a));
}

// Chained MFMA accumulation in AGPRs (`+a` constraint, no memory clobber).
static __device__ __forceinline__ void weave_mfma_f8_a(WeaveAgprF32x4& d, int64_t b, int64_t a)
{
    asm volatile("v_mfma_f32_16x16x32_fp8_fp8 %0, %1, %2, %0" : "+a"(d) : "v"(b), "v"(a));
}

// Unclumped promotion of 1 AGPR `f32x4` tile into `WeaveAgprAcc4` (`f2 lo, hi`):
// emits `4x v_accvgpr_read_b32 + 2x v_pk_fma_f32` (24 cycles <= 32 cycles of 2x MFMA).
static __device__ __forceinline__ void weave_agpr_prom_acc4(
    WeaveAgprAcc4& acc, WeaveAgprF32x4& ta_src, float sc)
{
    asm volatile("" : "+a"(ta_src));
    WeaveAgprF32x4 tv = ta_src;
    asm volatile("" : "+v"(tv));
    const WeaveAgprF32x2 s2 = {sc, sc}, t0 = {tv[0], tv[1]}, t1 = {tv[2], tv[3]};
    acc.lo = __builtin_elementwise_fma(t0, s2, acc.lo);
    acc.hi = __builtin_elementwise_fma(t1, s2, acc.hi);
    asm volatile("" : "+v"(acc.lo), "+v"(acc.hi));
}

static __device__ __forceinline__ void weave_agpr_prom_pair(
    WeaveAgprF32x2& clo, WeaveAgprF32x2& chi, WeaveAgprF32x4& ta_src, float sc)
{
    asm volatile("" : "+a"(ta_src));
    WeaveAgprF32x4 tv = ta_src;
    asm volatile("" : "+v"(tv));
    const WeaveAgprF32x2 s2 = {sc, sc}, t0 = {tv[0], tv[1]}, t1 = {tv[2], tv[3]};
    clo = __builtin_elementwise_fma(t0, s2, clo);
    chi = __builtin_elementwise_fma(t1, s2, chi);
    asm volatile("" : "+v"(clo), "+v"(chi));
}
#endif
