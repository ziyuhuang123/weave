#pragma once

#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <pybind11/functional.h>
#include <c10/cuda/CUDAFunctions.h>

#include <deep_jit/utils/env.hpp>
#include "../runtime/runtime.hpp"
#include "../jit_kernels/impls/sm90_fp8_mega_moe.hpp"
#include "../utils/layout.hpp"

namespace deep_gemm::mega {

static constexpr int kSM90MegaMoETokenAlignment = 128;

static int get_token_alignment_for_sm90_mega_moe() {
    return kSM90MegaMoETokenAlignment;
}

static std::tuple<int64_t, std::function<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>(const torch::Tensor&)>>
get_symm_buffer_size_for_sm90_mega_moe(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const bool& use_fp8_dispatch, const std::string& activation) {
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(num_experts % num_ranks == 0);
    if (not use_fp8_dispatch)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE currently supports FP8 dispatch only");
    if (activation != "swiglu")
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE currently supports the swiglu activation only");
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0 and
                   num_max_tokens_per_rank % kSM90MegaMoETokenAlignment == 0);
    if (hidden <= 0 or hidden % 256 != 0)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires hidden to be a positive multiple of 256 for combine vectorization");
    if (intermediate_hidden <= 0 or intermediate_hidden % 128 != 0)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires intermediate_hidden to be a positive multiple of 128");

    // Workspace bytes
    const auto num_max_pool_tokens = layout::nv_moe::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts / num_ranks);
    const auto workspace = layout::nv_moe::Workspace(
        nullptr, num_ranks, num_experts, num_max_tokens_per_rank, num_topk,
        num_max_pool_tokens, hidden);

    // Layouts
    const auto fp8_token_layout = layout::Data(hidden);
    const auto bf16_token_layout = layout::Data(hidden * 2);
    const auto fp8_intermediate_token_layout = layout::Data(intermediate_hidden);
    // Pool SFs are MN-major; their 128-token outer stride remains TMA-aligned
    // even when one logical token's SF byte count is not.
    const auto fp8_sf_layout = layout::Data(hidden / 32, false);
    const auto fp8_intermediate_sf_layout = layout::Data(intermediate_hidden / 16, false);
    const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
    const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
    const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

    // Input buffers
    const auto input_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_tokens_per_rank,
        workspace.get_end_ptr());
    const auto input_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_tokens_per_rank,
        input_token_buffer.get_end_ptr());
    const auto input_topk_idx_buffer = layout::Buffer(
        input_topk_idx_layout, 1, num_max_tokens_per_rank,
        input_sf_buffer.get_end_ptr());
    const auto input_topk_weights_buffer = layout::Buffer(
        input_topk_weights_layout, 1, num_max_tokens_per_rank,
        input_topk_idx_buffer.get_end_ptr());

    // BLOCK_M=64 is the worst case for the allocated SF pool capacity.
    const auto num_max_padded_sf_pool_tokens =
        layout::nv_moe::get_num_sf_ring_tokens(num_max_pool_tokens, 64);

    // L1 input buffer
    const auto l1_token_buffer = layout::Buffer(
        fp8_token_layout, 1, num_max_pool_tokens,
        input_topk_weights_buffer.get_end_ptr());
    const auto l1_sf_buffer = layout::Buffer(
        fp8_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l1_token_buffer.get_end_ptr());
    const auto l1_topk_weights_buffer = layout::Buffer(
        l1_topk_weights_layout, 1, num_max_pool_tokens,
        l1_sf_buffer.get_end_ptr());

    // L2 input buffer
    const auto l2_token_buffer = layout::Buffer(
        fp8_intermediate_token_layout, 1, num_max_pool_tokens,
        l1_topk_weights_buffer.get_end_ptr());
    const auto l2_sf_buffer = layout::Buffer(
        fp8_intermediate_sf_layout, 1, num_max_padded_sf_pool_tokens,
        l2_token_buffer.get_end_ptr());

    // Combine input buffer: BF16 tokens for cross-rank combine
    const auto combine_token_buffer = layout::Buffer(
        bf16_token_layout, num_topk, num_max_tokens_per_rank,
        l2_sf_buffer.get_end_ptr());

    // `x_sf` is K-major; pool scale factors are M-major.
    auto slice_input_buffers = [=](const torch::Tensor& buffer) {
        auto x = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_token_buffer.base)),
            {num_max_tokens_per_rank, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto x_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_sf_buffer.base)),
            {num_max_tokens_per_rank, hidden / 128},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto topk_idx = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_idx_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kInt64).device(buffer.device()));
        auto topk_weights = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(input_topk_weights_buffer.base)),
            {num_max_tokens_per_rank, num_topk},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l1_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_token_buffer.base)),
            {num_max_pool_tokens, hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l1_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l1_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, hidden / 128},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        auto l2_acts = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_token_buffer.base)),
            {num_max_pool_tokens, intermediate_hidden},
            torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(buffer.device()));
        auto l2_acts_sf = torch::from_blob(
            math::advance_ptr(buffer.data_ptr(), reinterpret_cast<int64_t>(l2_sf_buffer.base)),
            {num_max_padded_sf_pool_tokens, intermediate_hidden / 64},
            {1, num_max_padded_sf_pool_tokens},
            torch::TensorOptions().dtype(torch::kFloat32).device(buffer.device()));
        return std::make_tuple(x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf);
    };
    return {reinterpret_cast<int64_t>(combine_token_buffer.get_end_ptr()), slice_input_buffers};
}

// SM90 (Hopper) FP8 MegaMoE entry point.
//
// Mirrors `fp8_fp4_mega_moe` but expects FP8 (e4m3) weights with per-128 channel
// float scale factors. Top-level routing (which entry to call) is the caller's
// responsibility (see `deep_gemm/mega/__init__.py`).
static void fp8_mega_moe(
    const torch::Tensor& y,
    const std::tuple<torch::Tensor, torch::Tensor>& l1_weights_tuple,
    const std::tuple<torch::Tensor, torch::Tensor>& l2_weights_tuple,
    const std::optional<torch::Tensor>& cumulative_local_expert_recv_stats,
    const torch::Tensor& sym_buffer,
    const std::vector<int64_t>& sym_buffer_ptrs, const int& rank_idx,
    const int& num_max_tokens_per_rank,
    const int& num_experts, const int& num_topk,
    const std::tuple<int, int, int>& recipe,
    const std::string& activation,
    const std::optional<float>& activation_clamp_opt,
    const bool& fast_math
) {
    const auto [l1_weights, l1_weights_sf] = l1_weights_tuple;
    const auto [l2_weights, l2_weights_sf] = l2_weights_tuple;

    DG_HOST_ASSERT(y.dim() == 2);
    DG_HOST_ASSERT(y.is_cuda() and y.scalar_type() == torch::kBFloat16 and y.is_contiguous());
    DG_HOST_ASSERT(y.get_device() == c10::cuda::current_device());
    DG_HOST_ASSERT(l1_weights.dim() == 3);
    DG_HOST_ASSERT(y.size(1) == l1_weights.size(2));
    const auto output_device = y.device();
    const auto is_local_cuda_tensor = [&output_device](const torch::Tensor& tensor) {
        return tensor.is_cuda() and tensor.device() == output_device;
    };
    DG_HOST_ASSERT(is_local_cuda_tensor(l1_weights) and is_local_cuda_tensor(l2_weights));
    DG_HOST_ASSERT(is_local_cuda_tensor(l1_weights_sf) and is_local_cuda_tensor(l2_weights_sf));
    DG_HOST_ASSERT(is_local_cuda_tensor(sym_buffer) and sym_buffer.is_contiguous());
    if (cumulative_local_expert_recv_stats.has_value())
        DG_HOST_ASSERT(is_local_cuda_tensor(cumulative_local_expert_recv_stats.value()));
    DG_HOST_ASSERT(sym_buffer_ptrs.size() <= layout::kNumMaxRanks);
    DG_HOST_ASSERT(rank_idx >= 0 and static_cast<size_t>(rank_idx) < sym_buffer_ptrs.size());
    DG_HOST_ASSERT(sym_buffer_ptrs[rank_idx] == reinterpret_cast<int64_t>(sym_buffer.data_ptr()));

    // Architecture check
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major != 9)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires a compute capability 9.x GPU");

    // Config checks: SM90 uses block (128, 128) float SF for weights,
    // per-token per-128-K float SF for activations.
    const auto num_tokens = static_cast<int>(y.size(0));
    const auto [rm, rn, rk] = recipe;
    if (rm != 128 or rn != 128 or rk != 128)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires recipe=(128, 128, 128)");
    if (activation != "swiglu")
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE currently supports the swiglu activation only");

    // Activation checks
    const auto activation_clamp =
        activation_clamp_opt.value_or(std::numeric_limits<float>::infinity());
    DG_HOST_ASSERT(activation_clamp >= 0);

    // Tensor checks: SM90 weights must be FP8 e4m3, K-major
    DG_HOST_ASSERT(get_major_type_ab(l1_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(get_major_type_ab(l2_weights) == cute::UMMA::Major::K);
    DG_HOST_ASSERT(l1_weights.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(l2_weights.scalar_type() == torch::kFloat8_e4m3fn);
    const auto [num_experts_per_rank, intermediate_hidden_2, hidden] = get_shape<3>(l1_weights);
    const auto [num_experts_per_rank_, hidden_, intermediate_hidden] = get_shape<3>(l2_weights);
    DG_HOST_ASSERT(num_tokens <= num_max_tokens_per_rank);
    DG_HOST_ASSERT(num_experts_per_rank == num_experts_per_rank_);
    DG_HOST_ASSERT(hidden == hidden_);
    DG_HOST_ASSERT(intermediate_hidden_2 == 2 * intermediate_hidden);
    DG_HOST_ASSERT(l1_weights.is_contiguous() and l2_weights.is_contiguous());

    // The combine warp consumes eight values per lane, so hidden must satisfy
    // at least its base 256-element vector width. The selected launch config is
    // checked against the exact one/two/four-chunk layout before JIT compilation.
    if (hidden <= 0 or hidden % 256 != 0)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires hidden to be a positive multiple of 256 for combine vectorization");
    if (intermediate_hidden <= 0 or intermediate_hidden % 128 != 0)
        DG_HOST_UNREACHABLE("SM90 FP8 MegaMoE requires intermediate_hidden to be a positive multiple of 128");

    // Weight SFs are raw global-memory loads in natural MN-major order.
    constexpr int kGranMN = 128, kGranK = 128;
    check_sf_layout(l1_weights_sf, intermediate_hidden * 2, hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);
    check_sf_layout(l2_weights_sf, hidden, intermediate_hidden, kGranMN, kGranK,
                    num_experts_per_rank, false, true, torch::kFloat);
    if (not l1_weights_sf.is_contiguous() or not l2_weights_sf.is_contiguous())
        DG_HOST_UNREACHABLE(
            "SM90 FP8 MegaMoE weight scale factors must use contiguous natural layouts");

    // Check stats counter
    if (cumulative_local_expert_recv_stats.has_value()) {
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->scalar_type() == torch::kInt);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->numel() ==
                       num_experts_per_rank);
        DG_HOST_ASSERT(cumulative_local_expert_recv_stats->is_contiguous());
    }

    // Check buffer bytes
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts_ = num_experts_per_rank * num_ranks;
    const auto [num_required_bytes, slice] = get_symm_buffer_size_for_sm90_mega_moe(
        num_ranks, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden,
        true, activation);
    DG_HOST_ASSERT(sym_buffer.nbytes() >= static_cast<size_t>(num_required_bytes));
    DG_HOST_ASSERT(num_experts == num_experts_);

    // Already registered tensors
    const auto [x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf] = slice(sym_buffer);

    const int agy_prof_mode = agy_prof::get_mode();
    uint64_t* const agy_prof_dev_ptr = (agy_prof_mode > 0 and y.is_cuda())
        ? agy_prof::begin_invocation(y.get_device(), agy_prof_mode)
        : nullptr;

    const auto fast_num_max_pool_tokens = layout::nv_moe::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const auto fast_padded_sf_tokens =
        layout::nv_moe::get_num_sf_ring_tokens(fast_num_max_pool_tokens, 64);
    const Sm90MoeHeuristicInput fast_heur_input {
        runtime->get_num_sms(),
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden,
        fast_padded_sf_tokens
    };
    const auto agy_spec = get_agy_sm90_spec(fast_heur_input);
    if (agy_spec.enabled and agy_spec.get_int("fast_host", 0) > 0) {
        if (try_launch_agy_sm90_fast_plan(
                agy_spec, y, l1_weights, l2_weights,
                l1_weights_sf, l2_weights_sf,
                cumulative_local_expert_recv_stats,
                sym_buffer, sym_buffer_ptrs,
                rank_idx, num_max_tokens_per_rank,
                num_experts_per_rank, num_tokens, num_topk,
                hidden, intermediate_hidden,
                activation_clamp, fast_math,
                agy_prof_mode, agy_prof_dev_ptr)) {
            static const bool kCommDebug = deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG", 0) != 0;
            if (kCommDebug)
                sym_buffer.zero_();
            if (agy_prof_mode > 0)
                agy_prof::maybe_auto_dump(y.get_device(), agy_prof_mode);
            return;
        }
    }

    sm90_fp8_mega_moe(y,
                     l1_acts, l1_acts_sf,
                     l2_acts, l2_acts_sf,
                     l1_weights, l2_weights,
                     l1_weights_sf, l2_weights_sf,
                     cumulative_local_expert_recv_stats,
                     sym_buffer_ptrs,
                     rank_idx, num_max_tokens_per_rank,
                     num_experts_per_rank,
                     num_tokens, num_topk,
                     hidden, intermediate_hidden,
                     activation_clamp, fast_math,
                     &sym_buffer,
                     agy_prof_mode,
                     agy_prof_dev_ptr);

    if (deep_jit::get_env<int>("DG_COMM_KERNEL_DEBUG", 0))
        sym_buffer.zero_();
    if (agy_prof_mode > 0)
        agy_prof::maybe_auto_dump(y.get_device(), agy_prof_mode);
}


static void fp8_mega_moe_compile_only(
    const int& num_ranks, const int& num_experts,
    const int& num_max_tokens_per_rank, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const int& num_tokens, const float& activation_clamp, const bool& fast_math
) {
    DG_HOST_ASSERT(num_ranks > 0 and num_experts % num_ranks == 0);
    DG_HOST_ASSERT(num_max_tokens_per_rank > 0 and
                   num_max_tokens_per_rank % kSM90MegaMoETokenAlignment == 0);
    DG_HOST_ASSERT(hidden > 0 and hidden % 256 == 0);
    DG_HOST_ASSERT(intermediate_hidden > 0 and intermediate_hidden % 128 == 0);
    const auto num_experts_per_rank = num_experts / num_ranks;
    const auto num_max_pool_tokens = layout::nv_moe::get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
    const auto num_padded_sf_pool_tokens =
        layout::nv_moe::get_num_sf_ring_tokens(num_max_pool_tokens, 64);
    const Sm90MoeHeuristicInput heuristic_input {
        runtime->get_num_sms(),
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk,
        hidden, intermediate_hidden,
        num_padded_sf_pool_tokens
    };
    const auto launch_config = select_mega_moe_sm90(heuristic_input);
    const auto make_args = [&](const SM90FP8MegaMoERuntime::KernelPhase phase) {
        SM90FP8MegaMoERuntime::Args args{};
        args.num_max_tokens_per_rank = num_max_tokens_per_rank;
        args.hidden = hidden;
        args.intermediate_hidden = intermediate_hidden;
        args.num_experts = num_experts;
        args.num_topk = num_topk;
        args.num_ranks = num_ranks;
        args.activation_clamp = activation_clamp;
        args.fast_math = fast_math;
        args.bf16_scaled_accum = launch_config.numerical.bf16_scaled_accum;
        args.l1_nmajor_for_fused = launch_config.l1.nmajor_schedule;
        args.kernel_phase = phase;
        args.config = phase == SM90FP8MegaMoERuntime::KernelPhase::Linear1 ? launch_config.l1 : launch_config.l2;
        args.num_tokens = num_tokens;
        args.agy_prof_mode = agy_prof::get_env_mode();
        return args;
    };
    if (launch_config.fused_single_kernel) {
        SM90FP8MegaMoERuntime::compile_only(
            "sm90_fp8_mega_moe_fused_impl", make_args(SM90FP8MegaMoERuntime::KernelPhase::Fused));
    } else {
        SM90FP8MegaMoERuntime::compile_only(
            "sm90_fp8_mega_moe_l1_impl", make_args(SM90FP8MegaMoERuntime::KernelPhase::Linear1));
        SM90FP8MegaMoERuntime::compile_only(
            "sm90_fp8_mega_moe_l2_impl", make_args(SM90FP8MegaMoERuntime::KernelPhase::Linear2));
    }
}

static void register_sm90_apis(pybind11::module_& m) {
    m.def("get_token_alignment_for_sm90_mega_moe", &get_token_alignment_for_sm90_mega_moe);
    m.def("get_symm_buffer_size_for_sm90_mega_moe", &get_symm_buffer_size_for_sm90_mega_moe);
    m.def("fp8_mega_moe", &fp8_mega_moe);
    m.def("fp8_mega_moe_compile_only", &fp8_mega_moe_compile_only);
    m.def("agy_prof_set_mode", &agy_prof::set_mode_override);
    m.def("agy_prof_get_mode", &agy_prof::get_mode);
    m.def("agy_prof_reset", &agy_prof::reset_current_device);
    m.def("agy_prof_snapshot_bytes", &agy_prof::snapshot_bytes);
    m.def("agy_prof_last_host_meta_json", &agy_prof::last_host_meta_json);
}


} // namespace deep_gemm::mega
