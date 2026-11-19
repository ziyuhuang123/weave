#pragma once
#include "official_m16_m32_configs_v1262.hpp"

template <int MPerBlock_, int NPerBlock_, int KPerBlock_, int MWaves_, int NWaves_,
          ck::BlockGemmPipelineVersion PipeVer_, int ActOP_, bool GateUp_, bool MulWeight_>
struct MakeAgy512MoeOp {
    static constexpr unsigned BlockSize = 512;
    static constexpr unsigned MPerBlock = MPerBlock_;
    static constexpr unsigned NPerBlock = NPerBlock_;
    static constexpr unsigned KPerBlock = KPerBlock_;
    static constexpr ck::BlockGemmPipelineVersion PipeVer = PipeVer_;

    static constexpr ck::index_t MNPerXDL = 16;
    static_assert(MWaves_ * NWaves_ * 64 == 512);
    static constexpr ck::index_t MXDL = MPerBlock_ / (MNPerXDL * MWaves_);
    static constexpr ck::index_t NXDL = NPerBlock_ / (MNPerXDL * NWaves_);
    static_assert(MXDL >= 1 && NXDL >= 1);
    static constexpr ck::index_t K0 = KPerBlock_ / 16;
    static constexpr ck::index_t K0_MN = 512 / K0;
    static_assert(K0_MN <= MPerBlock_ && K0_MN <= NPerBlock_);

    using Op = ck::tensor_operation::device::DeviceMoeGemmBlockScale<
        Row, Col, V1262DsLayout, Row,
        V1262A0, V1262A1, V1262B0, V1262B1, V1262Ds, V1262E,
        V1262Acc, F32,
        V1262Pass, V1262Pass, V1262CElementOp, V1262GemmSpec,
        512, V1262ScaleBlockM, V1262ScaleBlockN, V1262ScaleBlockK,
        MPerBlock_, NPerBlock_, KPerBlock_,
        16, 16,
        16, 16,
        MXDL, NXDL,
        ck::Sequence<K0, K0_MN, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
        2, 16, 16, false,
        ck::Sequence<K0, K0_MN, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>,
        2, 16, 16, false,
        MXDL, NXDL, ck::Sequence<1, 32, 1, 16>, ck::Sequence<2, 1, 1, 1>,
        ck::BlockGemmPipelineScheduler::Intrawave,
        PipeVer_,
        ActOP_, false, GateUp_, false, MulWeight_,
        int32_t, V1262A0, V1262A0, V1262A0, V1262A0, false>;
    using Grid = typename Op::GridwiseGemm64;
};

// Tier 1 (BM = 32): 512T, 32x128x256, MWaves=2, NWaves=4, PipelineVersion::v1
using Agy512S1M32 = MakeAgy512MoeOp<32, 128, 256, 2, 4, ck::BlockGemmPipelineVersion::v1, 1, true, false>;
using Agy512S2M32 = MakeAgy512MoeOp<32, 128, 256, 2, 4, ck::BlockGemmPipelineVersion::v1, 0, false, true>;

// Tier 2 (BM = 64): 512T, 64x128x128, MWaves=1, NWaves=8, PipelineVersion::v3 (2-LDS Ping-Pong)
using Agy512S1M64 = MakeAgy512MoeOp<64, 128, 128, 1, 8, ck::BlockGemmPipelineVersion::v3, 1, true, false>;
using Agy512S2M64 = MakeAgy512MoeOp<64, 128, 128, 1, 8, ck::BlockGemmPipelineVersion::v3, 0, false, true>;

// Tier 3 (BM = 128): 512T, 128x128x128, MWaves=2, NWaves=4, PipelineVersion::v3 (2-LDS Ping-Pong)
using Agy512S1M128 = MakeAgy512MoeOp<128, 128, 128, 2, 4, ck::BlockGemmPipelineVersion::v3, 1, true, false>;
using Agy512S2M128 = MakeAgy512MoeOp<128, 128, 128, 2, 4, ck::BlockGemmPipelineVersion::v3, 0, false, true>;

// Verify K-loop tail parity across all 6 target models: K in {512, 1024, 2048, 3072, 4096, 6144, 8192}
static_assert(Agy512S1M32::Grid::CalculateHasMainKBlockLoop(2048));
static_assert(Agy512S1M64::Grid::CalculateHasMainKBlockLoop(2048));
static_assert(Agy512S1M128::Grid::CalculateHasMainKBlockLoop(2048));
static_assert(Agy512S1M64::Grid::CalculateKBlockLoopTailNum(2048) == ck::TailNumber::Even);
static_assert(Agy512S1M128::Grid::CalculateKBlockLoopTailNum(2048) == ck::TailNumber::Even);
static_assert(Agy512S2M64::Grid::CalculateKBlockLoopTailNum(512) == ck::TailNumber::Even);
static_assert(Agy512S2M128::Grid::CalculateKBlockLoopTailNum(512) == ck::TailNumber::Even);
static_assert(Agy512S2M32::Grid::CalculateKBlockLoopTailNum(512) == ck::TailNumber::Even);

template <typename Cfg, typename Arg>
__device__ __forceinline__ void weave_run_ck_512_tile(
    Arg& arg, unsigned char* arena, unsigned tile_n, unsigned tile_m) {
  using Grid = typename Cfg::Grid;
  if constexpr (Cfg::PipeVer == ck::BlockGemmPipelineVersion::v3) {
    if constexpr (!Grid::CalculateHasMainKBlockLoop(512)) {
      if (!Grid::CalculateHasMainKBlockLoop(arg.K)) {
        Grid::template Run_2LdsAt<false, ck::InMemoryDataOperationEnum::Set, ck::TailNumber::Even>(
            arg.p_sorted_token_ids, arg.p_sorted_expert_ids, arg.p_max_token_id,
            arg.p_a_grid, arg.p_b_grid, arg.p_ds_grid, arg.p_c_grid,
            arg.p_a_scale_grid, arg.p_b_scale_grid,
            arena, arena + 32768u, arg,
            arg.a_element_op, arg.b_element_op, arg.c_element_op,
            tile_n, tile_m);
        return;
      }
    }
    Grid::template Run_2LdsAt<true, ck::InMemoryDataOperationEnum::Set, ck::TailNumber::Even>(
        arg.p_sorted_token_ids, arg.p_sorted_expert_ids, arg.p_max_token_id,
        arg.p_a_grid, arg.p_b_grid, arg.p_ds_grid, arg.p_c_grid,
        arg.p_a_scale_grid, arg.p_b_scale_grid,
        arena, arena + 32768u, arg,
        arg.a_element_op, arg.b_element_op, arg.c_element_op,
        tile_n, tile_m);
  } else {
    if constexpr (!Grid::CalculateHasMainKBlockLoop(512)) {
      if (!Grid::CalculateHasMainKBlockLoop(arg.K)) {
        Grid::template RunAt<false, ck::InMemoryDataOperationEnum::Set, ck::TailNumber::Even>(
            arg.p_sorted_token_ids, arg.p_sorted_expert_ids, arg.p_max_token_id,
            arg.p_a_grid, arg.p_b_grid, arg.p_ds_grid, arg.p_c_grid,
            arg.p_a_scale_grid, arg.p_b_scale_grid,
            arena, arg,
            arg.a_element_op, arg.b_element_op, arg.c_element_op,
            tile_n, tile_m);
        return;
      }
    }
    Grid::template RunAt<true, ck::InMemoryDataOperationEnum::Set, ck::TailNumber::Even>(
        arg.p_sorted_token_ids, arg.p_sorted_expert_ids, arg.p_max_token_id,
        arg.p_a_grid, arg.p_b_grid, arg.p_ds_grid, arg.p_c_grid,
        arg.p_a_scale_grid, arg.p_b_scale_grid,
        arena, arg,
        arg.a_element_op, arg.b_element_op, arg.c_element_op,
        tile_n, tile_m);
  }
}
