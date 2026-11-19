#pragma once

// v1010 GEMM2: exact formally accepted v964 Uniform512 CK body. Both arms use the exact Aiter/CK blockscale stage2
// M64xN256xK128 body; only wave geometry changes for the matched comparison.
#include <cstdint>
#include "ck/tensor_operation/gpu/device/impl/device_moe_gemm_blockscale.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace v1010_gemm2 {
using F8 = ck::f8_t;
using B16 = ck::bhalf_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using Pass = ck::tensor_operation::element_wise::PassThrough;
using Ds = ck::Tuple<F32>;

struct ScaleCast {
    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const {
        (void)d;
        e = ck::type_convert<E>(c);
    }
};

template <int BlockSize_, int MPerBlock_, int NPerBlock_,
          int MWaves_, int NWaves_, int MXdl_, int NXdl_,
          int AK1, int BK1, int CShuffleM, int CShuffleN>
struct Config {
    static constexpr int BlockSize = BlockSize_;
    static constexpr int MPerBlock = MPerBlock_;
    static constexpr int NPerBlock = NPerBlock_;
    static constexpr int MWaves = MWaves_;
    static constexpr int NWaves = NWaves_;
    static constexpr int MXdl = MXdl_;
    static constexpr int NXdl = NXdl_;
    static constexpr bool DualLds = false;
    static constexpr int KPerBlock = 128;
    static constexpr int LdsBytes = MPerBlock * NPerBlock * 4;
    static_assert(BlockSize == MWaves * NWaves * 64);
    static_assert(MXdl == MPerBlock / (16 * MWaves));
    static_assert(NXdl == NPerBlock / (16 * NWaves));
    static_assert(LdsBytes > 0 && LdsBytes <= 65536);

    using Op = ck::tensor_operation::device::DeviceMoeGemmBlockScale<
        Row, Col, ck::Tuple<Row>, Row,
        F8, F32, F8, F32, Ds, B16, F32, F32,
        Pass, Pass, ScaleCast,
        ck::tensor_operation::device::GemmSpecialization::Default,
        BlockSize, 1, 128, 128,
        MPerBlock, NPerBlock, KPerBlock,
        16, 16,
        16, 16,
        MXdl, NXdl,
        ck::Sequence<8, MPerBlock / (MPerBlock > 64 ? 2 : 1), 1>,
        ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
        2, 16, 16, false,
        ck::Sequence<8, NPerBlock / 4, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
        2, 16, 16, false,
        MXdl, NXdl, ck::Sequence<1, CShuffleM, 1, CShuffleN>, ck::Sequence<2, 1, 1, 1>,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v1,
        0, false, false, false, true,
        std::int32_t, F8, F8, F8, F8, false>;
    using Grid = typename Op::GridwiseGemm64;
    using Argument = typename Grid::Argument;
};

// Matched official 4-wave reference: the exact Aiter/CK stage2 geometry used
// by the previous row-4 sweep for the same 64x256 tile.
using Official256 = Config<256, 64, 256, 1, 4, 4, 4, 32, 32, 4, 64>;
// Candidate OV1 resource: 8 waves, four MXDL waves by two NXDL lanes.
using Uniform512 = Config<512, 64, 256, 1, 8, 4, 2, 64, 64, 8, 64>;
// v1050 runtime-selected small-token tile: two MXDL waves by two NXDL lanes.
using DynamicM32 = Config<512, 32, 256, 1, 8, 2, 2, 64, 64, 4, 64>;
// v1050 runtime-selected large-token tile: two M waves by four N waves.
using DynamicM128 = Config<512, 128, 128, 2, 4, 4, 2, 64, 64, 8, 64>;

static_assert(Official256::Op::GridwiseGemm64::MaxBlockSize == 256);
static_assert(Uniform512::Op::GridwiseGemm64::MaxBlockSize == 512);
static_assert(DynamicM32::Op::GridwiseGemm64::MaxBlockSize == 512);
static_assert(DynamicM128::Op::GridwiseGemm64::MaxBlockSize == 512);

constexpr int kLdsBytes = 65536;

}  // namespace v964_gemm2
