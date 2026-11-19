#pragma once

// Exact Aiter gfx942 per_1x128 blockscale bodies used by the token-dependent
// branches in this carrier:
//   token <= 32: CK two-stage with M16 stage1 and M16 stage2
//   token >  32: one-stage ASM M32 (32x128 / 32x256 body selection follows)
// The M16 body is 256 threads / four waves.  The persistent kernel around it
// is 512 threads; only waves 0..3 enter this body and use the v1262 subgroup
// barrier.  The M32 body remains the existing 512-thread official CK carrier.

#include "gemm_moe_ck2stages_common_blockscale.cuh"
#include <cstdint>

using V1262A0 = F8;
using V1262B0 = F8;
using V1262Acc = F32;
using V1262E = B16;
using V1262CElementOp = MulABScaleExpertWeightA8W8blkscale;
using V1262A1 = F32;
using V1262B1 = F32;
using V1262Ds = ck::Tuple<F32>;
using V1262DsLayout = ck::Tuple<Row>;
using V1262Pass = ck::tensor_operation::element_wise::PassThrough;
constexpr auto V1262GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
constexpr ck::index_t V1262ScaleBlockM = 1;
constexpr ck::index_t V1262ScaleBlockN = 128;
constexpr ck::index_t V1262ScaleBlockK = 128;

// Generated instance:
// moe_ck2stages_gemm1_256x16x128x256_1x4_..._V1_..._silu_F8_F8_B16.cu
using V1262Stage1M16Op = ck::tensor_operation::device::DeviceMoeGemmBlockScale<
    Row, Col, V1262DsLayout, Row,
    V1262A0, V1262A1, V1262B0, V1262B1, V1262Ds, V1262E,
    V1262Acc, F32,
    V1262Pass, V1262Pass, V1262CElementOp, V1262GemmSpec,
    256, V1262ScaleBlockM, V1262ScaleBlockN, V1262ScaleBlockK,
    16, 128, 256,
    16, 16,
    16, 16,
    1, 2,
    ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    1, 2, ck::Sequence<1, 16, 1, 16>, ck::Sequence<2, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    1, false, true, false, false,
    int32_t, V1262A0, V1262A0, V1262A0, V1262A0, false>;

// Generated instance:
// moe_ck2stages_gemm2_256x16x128x256_1x4_..._V1_..._MulRoutedWeight1_F8_F8_B16.cu
using V1262Stage2M16Op = ck::tensor_operation::device::DeviceMoeGemmBlockScale<
    Row, Col, V1262DsLayout, Row,
    V1262A0, V1262A1, V1262B0, V1262B1, V1262Ds, V1262E,
    V1262Acc, F32,
    V1262Pass, V1262Pass, V1262CElementOp, V1262GemmSpec,
    256, V1262ScaleBlockM, V1262ScaleBlockN, V1262ScaleBlockK,
    16, 128, 256,
    16, 16,
    16, 16,
    1, 2,
    ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    ck::Sequence<16, 16, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    1, 2, ck::Sequence<1, 16, 1, 16>, ck::Sequence<2, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    0, false, false, false, true,
    int32_t, V1262A0, V1262A0, V1262A0, V1262A0, false>;

using V1262Grid1M16 = typename V1262Stage1M16Op::GridwiseGemm64;
using V1262Grid2M16 = typename V1262Stage2M16Op::GridwiseGemm64;

struct V1262Stage1M16 {
    static constexpr unsigned BlockSize = 256;
    static constexpr unsigned MPerBlock = 16;
    static constexpr unsigned NPerBlock = 128;
    static constexpr unsigned PhysicalMPerBlock = 16;
    static constexpr bool Uses2Lds = false;
    static constexpr unsigned SecondLdsOffset = 0;
    static constexpr unsigned SecondLdsSpan = 0;
    using Grid = V1262Grid1M16;
};

struct V1262Stage2M16 {
    static constexpr unsigned BlockSize = 256;
    static constexpr unsigned MPerBlock = 16;
    static constexpr unsigned NPerBlock = 128;
    static constexpr unsigned PhysicalMPerBlock = 16;
    static constexpr bool Uses2Lds = false;
    static constexpr unsigned SecondLdsOffset = 0;
    static constexpr unsigned SecondLdsSpan = 0;
    using Grid = V1262Grid2M16;
};

static_assert(V1262Grid1M16::MaxBlockSize == 256);
static_assert(V1262Grid2M16::MaxBlockSize == 256);
static_assert(V1262Stage1M16Op::NXdlPerWave64 == 2);
static_assert(V1262Stage2M16Op::NXdlPerWave64 == 2);
