#pragma once
#include "official_m16_m32_configs_v1262.hpp"

// Stage1 M16: 256 threads (1x4 waves), BM=16, BN=128, BK=256, GateUp=true (SiLU)
// Matches official V1262Stage1M16Op in official_m16_m32_configs_v1262.hpp
using AgyStage1M16 = V1262Stage1M16;
using AgyStage2M16 = V1262Stage2M16;

// Stage1 M32: 256 threads (1x4 waves), BM=32, BN=128, BK=128, GateUp=true (SiLU)
// A/B cluster: <8, 32, 1> (8*16=128=BK, 32<=32=BM), CShuffle: (2, 2) -> 32*128*4 = 16384 B <= 32768 B
using AgyStage1M32Op = ck::tensor_operation::device::DeviceMoeGemmBlockScale<
    Row, Col, V1262DsLayout, Row,
    V1262A0, V1262A1, V1262B0, V1262B1, V1262Ds, V1262E,
    V1262Acc, F32,
    V1262Pass, V1262Pass, V1262CElementOp, V1262GemmSpec,
    256, V1262ScaleBlockM, V1262ScaleBlockN, V1262ScaleBlockK,
    32, 128, 128,
    16, 16,
    16, 16,
    2, 2,
    ck::Sequence<8, 32, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    ck::Sequence<8, 32, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    2, 2, ck::Sequence<1, 16, 1, 16>, ck::Sequence<2, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    1, false, true, false, false,
    int32_t, V1262A0, V1262A0, V1262A0, V1262A0, false>;

// Stage2 M32: 256 threads (1x4 waves), BM=32, BN=128, BK=128, MulRoutedWeight=true
using AgyStage2M32Op = ck::tensor_operation::device::DeviceMoeGemmBlockScale<
    Row, Col, V1262DsLayout, Row,
    V1262A0, V1262A1, V1262B0, V1262B1, V1262Ds, V1262E,
    V1262Acc, F32,
    V1262Pass, V1262Pass, V1262CElementOp, V1262GemmSpec,
    256, V1262ScaleBlockM, V1262ScaleBlockN, V1262ScaleBlockK,
    32, 128, 128,
    16, 16,
    16, 16,
    2, 2,
    ck::Sequence<8, 32, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    ck::Sequence<8, 32, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    2, 2, ck::Sequence<1, 16, 1, 16>, ck::Sequence<2, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    0, false, false, false, true,
    int32_t, V1262A0, V1262A0, V1262A0, V1262A0, false>;

struct AgyStage1M32 {
    static constexpr unsigned BlockSize = 256;
    static constexpr unsigned MPerBlock = 32;
    static constexpr unsigned NPerBlock = 128;
    using Grid = typename AgyStage1M32Op::GridwiseGemm64;
};

struct AgyStage2M32 {
    static constexpr unsigned BlockSize = 256;
    static constexpr unsigned MPerBlock = 32;
    static constexpr unsigned NPerBlock = 128;
    using Grid = typename AgyStage2M32Op::GridwiseGemm64;
};

// Stage1 M64: 256 threads (1x4 waves), BM=64, BN=128, BK=128, GateUp=true (SiLU)
// CShuffle: (2, 2) -> 32*128*4 = 16384 B <= 32768 B
using AgyStage1M64Op = ck::tensor_operation::device::DeviceMoeGemmBlockScale<
    Row, Col, V1262DsLayout, Row,
    V1262A0, V1262A1, V1262B0, V1262B1, V1262Ds, V1262E,
    V1262Acc, F32,
    V1262Pass, V1262Pass, V1262CElementOp, V1262GemmSpec,
    256, V1262ScaleBlockM, V1262ScaleBlockN, V1262ScaleBlockK,
    64, 128, 128,
    16, 16,
    16, 16,
    4, 2,
    ck::Sequence<8, 32, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    ck::Sequence<8, 32, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    2, 2, ck::Sequence<1, 16, 1, 16>, ck::Sequence<2, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    1, false, true, false, false,
    int32_t, V1262A0, V1262A0, V1262A0, V1262A0, false>;

// Stage2 M64x256: 256 threads (1x4 waves), BM=64, BN=256, BK=128, MulRoutedWeight=true
// CShuffle: (2, 4) -> 32*256*4 = 32768 B <= 32768 B (8x fewer tiles than 16x128!)
using AgyStage2M64N256Op = ck::tensor_operation::device::DeviceMoeGemmBlockScale<
    Row, Col, V1262DsLayout, Row,
    V1262A0, V1262A1, V1262B0, V1262B1, V1262Ds, V1262E,
    V1262Acc, F32,
    V1262Pass, V1262Pass, V1262CElementOp, V1262GemmSpec,
    256, V1262ScaleBlockM, V1262ScaleBlockN, V1262ScaleBlockK,
    64, 256, 128,
    16, 16,
    16, 16,
    4, 4,
    ck::Sequence<8, 32, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    ck::Sequence<8, 32, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
    2, 16, 16, false,
    2, 4, ck::Sequence<1, 16, 1, 16>, ck::Sequence<2, 1, 1, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1,
    0, false, false, false, true,
    int32_t, V1262A0, V1262A0, V1262A0, V1262A0, false>;

struct AgyStage1M64 {
    static constexpr unsigned BlockSize = 256;
    static constexpr unsigned MPerBlock = 64;
    static constexpr unsigned NPerBlock = 128;
    using Grid = typename AgyStage1M64Op::GridwiseGemm64;
};

struct AgyStage2M64 {
    static constexpr unsigned BlockSize = 256;
    static constexpr unsigned MPerBlock = 64;
    static constexpr unsigned NPerBlock = 256;
    using Grid = typename AgyStage2M64N256Op::GridwiseGemm64;
};

static_assert(AgyStage1M32::Grid::MaxBlockSize == 256);
static_assert(AgyStage2M32::Grid::MaxBlockSize == 256);
static_assert(AgyStage1M64::Grid::MaxBlockSize == 256);
static_assert(AgyStage2M64::Grid::MaxBlockSize == 256);
