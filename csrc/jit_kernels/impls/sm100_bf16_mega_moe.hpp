#pragma once

#include <format>
#include <sstream>
#include <torch/python.h>

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../../runtime/runtime.hpp"
#include "../../utils/exception.hpp"
#include "../heuristics/mega_moe.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

static void sm100_bf16_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l2_acts,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l2_acts,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;

    // Heuristics
    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, 0, MmaKind::BF16);

    // Make tensormap
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       config.block_n / 2, config.store_block_m_l1,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode);
    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_weights,
        hidden, shared_intermediate_hidden * 2,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_n / 2, config.store_block_m_l1,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_weights,
        shared_intermediate_hidden, hidden,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l2_weights;

    // Stats can be optional
    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const auto num_sms = runtime->get_num_sms();

    // Unified knob mirror.  The host resolver already validated these; the
    // defines below are what the kernel translation unit consumes (CTA-0
    // cost model via weave::default_weave_knobs, the scheduler's L2-kblock /
    // L2-ratio knobs, the dispatch pull throttle and the ring / thread-layout
    // static asserts).
    const auto& knobs = config.knobs;
    // One shared store height: asymmetric L1/L2 store requests cannot be
    // honored by the single STORE_BLOCK_M template and are rejected.
    DG_HOST_ASSERT(config.store_block_m_l2 == config.store_block_m_l1);
    std::ostringstream knob_macros;
    // Single locality domain: this backend has no SM-domain routing.
    knob_macros << "#define WEAVE_LOCALITY_DOMAINS 1\n";
    knob_macros << "#define WEAVE_PRECISION_MODE " << knobs.precision_mode << "u\n";
    if (knobs.override_v_head != weave::kAutoU32)
        knob_macros << "#define WEAVE_V_HEAD " << knobs.override_v_head << "u\n";
    knob_macros << "#define WEAVE_DISPATCH_WARPS " << (config.num_dispatch_threads / 32) << "\n";
    knob_macros << "#define WEAVE_REDUCE_WARPS " << knobs.reduce_warps << "\n";
    knob_macros << "#define WEAVE_SM0_STREAM_SCHED " << knobs.sm0_stream_sched << "\n";
    knob_macros << "#define WEAVE_L2_KBLOCK_DEP " << knobs.l2_kblock_dep << "\n";
    knob_macros << "#define WEAVE_RING_BLOCKS " << knobs.ring_blocks << "\n";
    knob_macros << "#define WEAVE_EARLY_STREAM_REDUCE " << knobs.early_stream_reduce << "\n";
    knob_macros << "#define WEAVE_TMEM_PIPELINE_MODE " << config.tmem_pipeline_mode << "\n";
    knob_macros << "#define WEAVE_HEAD_PULL_CHUNK " << knobs.head_pull_chunk << "\n";
    knob_macros << "#define WEAVE_LOCAL_SCATTER_DEDUP " << knobs.local_scatter_dedup << "\n";
    knob_macros << "#define WEAVE_ADAPT_BM " << knobs.adapt_bm << "\n";
    knob_macros << "#define WEAVE_BLOCK_M_L1 " << knobs.block_m_l1 << "\n";
    knob_macros << "#define WEAVE_BLOCK_M_L2 " << knobs.block_m_l2 << "\n";
    knob_macros << "#define WEAVE_STORE_BLOCK_M_L1 " << config.store_block_m_l1 << "\n";
    knob_macros << "#define WEAVE_DISPATCH_ACTIVE_SMS " << knobs.dispatch_active_sms << "\n";
    knob_macros << "#define WEAVE_SCATTER_OWNER " << knobs.scatter_owner << "\n";
    knob_macros << "#define WEAVE_SCATTER_STAGING " << knobs.scatter_staging << "\n";
    knob_macros << "#define WEAVE_COMBINE_SYNC_CHUNK " << knobs.combine_sync_chunk << "\n";
    knob_macros << "#define WEAVE_EPI_REG_STAGES " << weave::resolve_epi_reg_stages(
        knobs, static_cast<uint32_t>(config.block_m), config.tmem_pipeline_mode) << "\n";
    knob_macros << "#define WEAVE_SCHED_L2_RATIO " << knobs.sched_l2_ratio << "\n";
    knob_macros << "#define WEAVE_REG_BALANCE " << knobs.reg_balance << "\n";

    // Compile
    const auto kernel = jit->compile("sm100_bf16_mega_moe", std::format(R"(
{}
#include <deep_gemm/impls/sm100_bf16_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {}, {}, {},
        {},
        {},
        {},
        {},
        {},
        {}, {}, {},
        {}, {},
        {},
        {}
    >);
}};
)", knob_macros.str(),
        num_max_tokens_per_rank,
        hidden, intermediate_hidden,
        num_experts, num_shared_experts,
        num_topk,
        config.block_m, config.block_n, config.block_k,
        config.store_block_m_l1,
        config.num_ring_tokens,
        config.num_stages,
        config.num_bytes_per_pull,
        config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
        num_sms, num_ranks,
        to_string(activation_clamp),
        fast_math ? "true" : "false"));

    // Launch
    jit->launch(
        kernel, {
            .num_smem_bytes = config.smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads, 1, 1),
            .cluster_dim = dim3(2, 1, 1),
        },
        y.data_ptr(),
        cumulative_local_expert_recv_stats_ptr,
        num_tokens,
        layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        tensor_map_l1_acts,
        tensor_map_l1_weights,
        tensor_map_l1_output,
        tensor_map_l2_acts,
        tensor_map_l2_weights,
        tensor_map_shared_l1_acts,
        tensor_map_shared_l1_weights,
        tensor_map_shared_l1_output,
        tensor_map_shared_l2_acts,
        tensor_map_shared_l2_weights);
}

} // namespace deep_gemm
