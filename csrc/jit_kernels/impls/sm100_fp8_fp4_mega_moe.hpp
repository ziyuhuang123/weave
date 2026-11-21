#pragma once

#include <torch/python.h>
#include <fstream>
#include <sstream>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "runtime_utils.hpp"

#include <deep_gemm/layout/mega_moe.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>

#include "../heuristics/mega_moe.hpp"

namespace deep_gemm {

class SM100FP8FP4MegaMoERuntime final : public LaunchRuntime<SM100FP8FP4MegaMoERuntime> {
public:
    struct Args {
        // Templated arguments
        int num_max_tokens_per_rank;
        int hidden, intermediate_hidden;
        int num_experts, num_shared_experts, num_topk;
        int num_ranks;
        float activation_clamp;
        bool fast_math;
        bool d3_has_extra_tmas;
        MegaMoEConfig config;

        // Runtime arguments
        void* y;
        int64_t* dg_prof_buf;
        uint32_t dg_prof_max_records;
        const int32_t* dg_prof_sm_map;
        uint32_t dg_prof_capture_slots;
        uint32_t dg_prof_warp_mask;
        int profile_mode;
        int* cumulative_local_expert_recv_stats;
        int num_tokens;
        const uint8_t* sm_locality_domains;
        layout::SymBuffer<> sym_buffer_ptrs;

        // Tensormap
        CUtensorMap tensor_map_l1_acts;
        CUtensorMap tensor_map_l1_acts_sf;
        CUtensorMap tensor_map_l1_weights;
        CUtensorMap tensor_map_l1_weights_sf;
        CUtensorMap tensor_map_l1_output;
        CUtensorMap tensor_map_l1_output_m16;
        CUtensorMap tensor_map_l1_output_m8;
        CUtensorMap tensor_map_l2_acts;
        CUtensorMap tensor_map_l2_acts_sf;
        CUtensorMap tensor_map_l2_weights;
        CUtensorMap tensor_map_l2_weights_sf;
        CUtensorMap tensor_map_l1_acts_m16;
        CUtensorMap tensor_map_l2_acts_m16;
        CUtensorMap tensor_map_l1_acts_m8;
        CUtensorMap tensor_map_l2_acts_m8;
        CUtensorMap tensor_map_shared_l1_acts;
        CUtensorMap tensor_map_shared_l1_acts_sf;
        CUtensorMap tensor_map_shared_l1_weights;
        CUtensorMap tensor_map_shared_l1_weights_sf;
        CUtensorMap tensor_map_shared_l1_output;
        CUtensorMap tensor_map_shared_l2_acts;
        CUtensorMap tensor_map_shared_l2_acts_sf;
        CUtensorMap tensor_map_shared_l2_weights;
        CUtensorMap tensor_map_shared_l2_weights_sf;

        // Launch configs
        LaunchArgs launch_args;
    };

    static std::string generate_d4_static_table(const Args& args, const int d4_static) {
        if (d4_static == 0)
            return "";
        const int el = args.num_experts / args.num_ranks;
        const auto path = get_weave_env<std::string>("WEAVE_STATIC_EXPERT_ORDER_FILE",
                          "DG_TUNE_D4_STATIC_FILE", "DG_ISOLATED_D4_STATIC_FILE", "");
        if (path.empty())
            return "";
        std::ifstream fin(path);
        DG_HOST_ASSERT(fin.good());
        std::vector<std::vector<int>> table(args.num_ranks, std::vector<int>(el, -1));
        int found_ranks = 0;
        std::string line;
        while (std::getline(fin, line)) {
            if (line.empty() or line[0] == '#')
                continue;
            std::istringstream iss(line);
            int file_e = 0, r = 0;
            std::string csv;
            if (!(iss >> file_e >> r >> csv))
                continue;
            if (file_e != args.num_experts)
                continue;
            DG_HOST_ASSERT(0 <= r and r < args.num_ranks);
            std::vector<int> vals;
            std::stringstream ss(csv);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty())
                    vals.push_back(std::stoi(item));
            }
            DG_HOST_ASSERT(static_cast<int>(vals.size()) == el);
            table[r] = vals;
            ++ found_ranks;
        }
        DG_HOST_ASSERT(found_ranks == args.num_ranks);
        std::ostringstream oss;
        oss << "#define DG_ISOLATED_D4_HAS_STATIC_TABLE 1\n";
        oss << "namespace deep_gemm::sched {\n";
        oss << "__device__ __constant__ unsigned char kD4StaticExpertOrder[" << args.num_ranks << "][" << el << "] = {\n";
        for (int r = 0; r < args.num_ranks; ++ r) {
            oss << "    {";
            for (int j = 0; j < el; ++ j) {
                if (j > 0) oss << ", ";
                oss << table[r][j];
            }
            oss << "}" << (r + 1 < args.num_ranks ? "," : "") << "\n";
        }
        oss << "};\n} // namespace deep_gemm::sched\n";
        return oss.str();
    }

    static std::string generate_impl(const Args& args) {
        const auto& knobs = args.config.knobs;
        // Axis 1 (D1)
        const int d1_g = get_weave_env<int>("WEAVE_READY_GROUP_SIZE", "DG_TUNE_D1_G", "DG_ISOLATED_D1_G", 0);
        // Axis 2 (D2)
        const int d2_mode = get_weave_env<int>("WEAVE_WARP_SPLIT_MODE", "DG_TUNE_D2_MODE", "DG_ISOLATED_D2_MODE", 0);
        const int d2_reg_pull_bytes = get_weave_env<int>("WEAVE_REG_PULL_BYTES", "DG_TUNE_D2_REG_PULL_BYTES", "DG_ISOLATED_D2_REG_PULL_BYTES", 0);
        const int d2_d8_sm_limit = get_weave_env<int>("WEAVE_HETERO_SM_LIMIT", "DG_ISOLATED_D2_D8_SM_LIMIT", nullptr, 74);
        const int d2_disp_regs = get_weave_env<int>("WEAVE_DISP_REGS", "DG_ISOLATED_D2_DISP_REGS", nullptr, 0);
        const int d2_epi_regs = get_weave_env<int>("WEAVE_EPI_REGS", "DG_ISOLATED_D2_EPI_REGS", nullptr, 0);
        // Axis 3 (D3)
        const int d3_bm = get_weave_env<int>("WEAVE_BLOCK_M", "DG_TUNE_D3_BM", "DG_D3_BM", 0);
        const int d3_bn = get_weave_env<int>("WEAVE_BLOCK_N", "DG_TUNE_D3_BN", "DG_D3_BN", 0);
        const int d3_bk = get_weave_env<int>("WEAVE_BLOCK_K", "DG_TUNE_D3_BK", "DG_D3_BK", 0);
        const int d3_store_m = get_weave_env<int>("WEAVE_STORE_BLOCK_M", "DG_TUNE_D3_STORE_M", "DG_D3_STORE_M", args.config.store_block_m_l1);
        const int d3_store_m_l2 = get_weave_env<int>("WEAVE_STORE_BLOCK_M_L2", "DG_TUNE_D3_STORE_M_L2", "DG_D3_STORE_M_L2", args.config.store_block_m_l2);
        const int d3_dyn_epi = get_weave_env<int>("WEAVE_DYN_EPILOGUE", "DG_TUNE_D3_DYN_EPI", "DG_D3_DYN_EPI", 0);
        const int d3_dyn_tma_a = get_weave_env<int>("WEAVE_DYN_TMA_A", "DG_TUNE_D3_DYN_TMA_A", "DG_D3_DYN_TMA_A", 0);
        const int d3_overlap_smem = get_weave_env<int>("WEAVE_OVERLAP_EXPERT_SMEM", "DG_TUNE_D3_OVERLAP_SMEM", "DG_D3_OVERLAP_EXPERT_SMEM", 0);
        const int d3_expert_step_mode = get_weave_env<int>("WEAVE_EXPERT_STEP_MODE", "DG_TUNE_D3_EXPERT_STEP_MODE", "DG_D3_EXPERT_STEP_MODE",
                                                          (knobs.adapt_bm != 0u) ? 1 : 0);
        const int d3_accum_stages = get_weave_env<int>("WEAVE_ACCUM_STAGES", "DG_D3_ACCUM_STAGES", "DG_MEGA_ACCUM_STAGES", 2);
        const int d3_tma_store_stages = get_weave_env<int>("WEAVE_TMA_STORE_STAGES", "DG_D3_TMA_STORE_STAGES", "DG_MEGA_TMA_STORE_STAGES", 2);
        const int d3_sched_stages = get_weave_env<int>("WEAVE_SCHED_STAGES", "DG_TUNE_D3_SCHED_STAGES", "DG_D3_SCHED_STAGES", 2);
        // Axis 4 (D4)
        const int d4_epw = get_weave_env<int>("WEAVE_EXPERTS_PER_WAVE", "DG_TUNE_D4_EPW", "DG_ISOLATED_D4_EPW", 0);
        const int d4_static = get_weave_env<int>("WEAVE_STATIC_EXPERT_ORDER", "DG_TUNE_D4_STATIC", "DG_ISOLATED_D4_STATIC", 0);
        const int d4_readyq = weave_knobs::named_env(
            "WEAVE_READY_QUEUE", {"formula", "sm0_fifo"},
            get_weave_env<int>("DG_TUNE_D4_READYQ", "DG_ISOLATED_D4_READYQ", nullptr, 0));
        const int d4_mode = get_weave_env<int>("WEAVE_SCHED_MODE", "DG_TUNE_D4_MODE", "DG_ISOLATED_D4_MODE", 0);
        const std::string d4_static_code = generate_d4_static_table(args, d4_static);
        // Axis 5 (D5): public WEAVE_DEDUP_MODE is binary (0 = off, 1 = remote pull dedup + local fanout).
        const int public_dedup = get_env<int>("WEAVE_DEDUP_MODE", -1);
        DG_HOST_ASSERT(public_dedup == -1 or public_dedup == 0 or public_dedup == 1);
        const int d5_dedup = (public_dedup >= 0)
            ? (public_dedup == 1 ? 2 : 0)
            : get_weave_env<int>("DG_TUNE_D5_DEDUP", "DG_ISOLATED_D5_DEDUP", nullptr, 0);
        const int d5_sched = get_weave_env<int>("WEAVE_DEDUP_SCHED", "DG_TUNE_D5_SCHED", "DG_ISOLATED_D5_SCHED", (d5_dedup == 2) ? 1 : 0);
        // Axis 6 (D6)
        const int d6_g = get_weave_env<int>("WEAVE_COMBINE_GROUP_TILES", "DG_TUNE_D6_G", "DG_ISOLATED_D6_G",
                                           (knobs.early_stream_reduce != 0u) ? 2 : 0);
        const int d6_rw = get_weave_env<int>("WEAVE_COMBINE_REDUCE_WARPS", "DG_TUNE_D6_REDUCE_WARPS", "DG_ISOLATED_D6_REDUCE_WARPS", 0);
        // Axis 7 (D7)
        const int d7_scatter_mode = get_weave_env<int>("WEAVE_SCATTER_MODE", "DG_TUNE_D7_SCATTER_MODE", nullptr, -1);
        // Derived from the strict binary ownership + staging knobs unless an
        // explicit WEAVE_SCATTER_MODE pins the executor:
        //   epilogue_only            -> 0 (epilogue issues remote stores)
        //   comm_only + smem         -> 1 (SMEM handoff to dispatch warps)
        //   comm_only + hbm_dedup    -> 3 (bf16x2 dedup + HBM scatter queue)
        const int derived_scatter_mode =
            knobs.scatter_owner == 0u ? 0 : (knobs.scatter_staging == 2u ? 3 : 1);
        int d7_isolated_scatter = get_weave_env<int>("WEAVE_SCATTER_MODE", "DG_ISOLATED_D7_SCATTER", nullptr,
                                                     (d7_scatter_mode >= 0) ? d7_scatter_mode : derived_scatter_mode);
        if (knobs.local_scatter_dedup != 0u)
            d7_isolated_scatter = 3;  // local bf16x2 dedup + HBM scatter queue
        DG_HOST_ASSERT(d7_isolated_scatter >= 0 and d7_isolated_scatter <= 3);
        // The emitted ownership/staging always agree with the final executor,
        // so an explicit WEAVE_SCATTER_MODE cannot produce a mixed owner.
        const int emitted_scatter_owner = (d7_isolated_scatter == 1 or d7_isolated_scatter == 3) ? 1 : 0;
        const int emitted_scatter_staging = d7_isolated_scatter == 3 ? 2 : (d7_isolated_scatter == 1 ? 1 : 0);
        DG_HOST_ASSERT(knobs.scatter_owner == 0u or emitted_scatter_owner == 1);
        const int d7_smem_alias = get_weave_env<int>("WEAVE_SCATTER_SMEM_ALIAS", "DG_TUNE_D7_SMEM_ALIAS", nullptr, 1);
        const int d7_warps = get_weave_env<int>("WEAVE_SCATTER_WARPS", "DG_TUNE_D7_WARPS", "DG_TUNE_D7_SCATTER_WARPS", get_env<int>("DG_SCATTER_WARPS", 0));
        const int d7_handoff_wait = get_weave_env<int>("WEAVE_SCATTER_HANDOFF_WAIT", "DG_TUNE_D7_HANDOFF_WAIT", nullptr, 1);
        const int d7_active_sms = get_weave_env<int>("WEAVE_SCATTER_ACTIVE_SMS", "DG_D7_ACTIVE_SMS", nullptr,
                                                    static_cast<int>(knobs.scatter_active_sms));
        const int enable_multicast = get_weave_env<int>("WEAVE_ENABLE_MULTICAST", nullptr, nullptr, 1);

        // Locality-domain routing is only valid when every queue can be split
        // into two equal halves; otherwise the build silently keeps one domain.
        const int num_l1_clusters = (args.intermediate_hidden * 2) / (args.config.block_n * 2);
        const int num_l2_clusters = args.hidden / (args.config.block_n * 2);
        const int num_shared_l1_clusters = args.num_shared_experts > 0
            ? (args.intermediate_hidden * 2 * args.num_shared_experts) / (args.config.block_n * 2) : 0;
        const int requested_locality_domains =
            get_weave_env<int>("WEAVE_LOCALITY_DOMAINS", "DG_LOCALITY_DOMAINS", nullptr, 0);
        const bool use_locality_domains = requested_locality_domains != 0 and
            num_l1_clusters % 2 == 0 and num_l2_clusters % 2 == 0 and num_shared_l1_clusters % 2 == 0;
        if (requested_locality_domains != 0 and not use_locality_domains)
            printf("WEAVE_LOCALITY_DOMAINS disabled: L1/L2 cluster counts (%d, %d) do not split evenly\n",
                   num_l1_clusters, num_l2_clusters);

        std::ostringstream d7_macros;
        d7_macros << "#define WEAVE_SCATTER_MODE " << d7_isolated_scatter << "\n";
        d7_macros << "#define WEAVE_SCATTER_SMEM_ALIAS " << d7_smem_alias << "\n";
        d7_macros << "#define WEAVE_SCATTER_WARPS " << d7_warps << "\n";
        d7_macros << "#define WEAVE_SCATTER_HANDOFF_WAIT " << d7_handoff_wait << "\n";
        d7_macros << "#define WEAVE_SCATTER_ACTIVE_SMS " << d7_active_sms << "\n";

        // Unified knob mirror.  The host resolver already validated these; the
        // defines below are what the kernel translation unit consumes.
        const int knobs_locality_domains = use_locality_domains
            ? static_cast<int>(weave::resolve_locality_domains(
                  knobs, weave::HardwareArch::B300_SM100))
            : 1;
        std::ostringstream knob_macros;
        knob_macros << "#define WEAVE_LOCALITY_DOMAINS " << knobs_locality_domains << "\n";
        knob_macros << "#define WEAVE_DISPATCH_WARPS " << (args.config.num_dispatch_threads / 32) << "\n";
        knob_macros << "#define WEAVE_SM0_STREAM_SCHED " << knobs.sm0_stream_sched << "\n";
        knob_macros << "#define WEAVE_L2_KBLOCK_DEP " << knobs.l2_kblock_dep << "\n";
        knob_macros << "#define WEAVE_RING_BLOCKS " << knobs.ring_blocks << "\n";
        knob_macros << "#define WEAVE_EARLY_STREAM_REDUCE " << knobs.early_stream_reduce << "\n";
        knob_macros << "#define WEAVE_TMEM_PIPELINE_MODE " << args.config.tmem_pipeline_mode << "\n";
        knob_macros << "#define WEAVE_HEAD_PULL_CHUNK " << knobs.head_pull_chunk << "\n";
        knob_macros << "#define WEAVE_LOCAL_SCATTER_DEDUP " << knobs.local_scatter_dedup << "\n";
        knob_macros << "#define WEAVE_ADAPT_BM " << knobs.adapt_bm << "\n";
        knob_macros << "#define WEAVE_BLOCK_M_L1 " << knobs.block_m_l1 << "\n";
        knob_macros << "#define WEAVE_BLOCK_M_L2 " << knobs.block_m_l2 << "\n";
        knob_macros << "#define WEAVE_STORE_BLOCK_M_L1 " << args.config.store_block_m_l1 << "\n";
        knob_macros << "#define WEAVE_DISPATCH_ACTIVE_SMS " << knobs.dispatch_active_sms << "\n";
        knob_macros << "#define WEAVE_SCATTER_OWNER " << emitted_scatter_owner << "\n";
        knob_macros << "#define WEAVE_SCATTER_STAGING " << emitted_scatter_staging << "\n";
        knob_macros << "#define WEAVE_COMBINE_SYNC_CHUNK " << knobs.combine_sync_chunk << "\n";
        uint32_t eff_epi_reg_stages = weave::resolve_epi_reg_stages(
            knobs, static_cast<uint32_t>(d3_bm > 0 ? d3_bm : args.config.block_m),
            static_cast<uint32_t>(args.config.tmem_pipeline_mode));
        // The two-stage register prefetch needs one 128-column N subtile and a
        // static epilogue; never auto-select it outside that envelope.
        const int eff_block_n = d3_bn > 0 ? d3_bn : args.config.block_n;
        if (knobs.epi_reg_stages == 0u and (eff_block_n > 128 or d3_dyn_epi != 0))
            eff_epi_reg_stages = 1u;
        knob_macros << "#define WEAVE_EPI_REG_STAGES " << eff_epi_reg_stages << "\n";
        knob_macros << "#define WEAVE_SCHED_L2_RATIO " << knobs.sched_l2_ratio << "\n";
        knob_macros << "#define WEAVE_REG_BALANCE " << knobs.reg_balance << "\n";

        return fmt::format(R"(
{}
#define WEAVE_READY_GROUP_SIZE {}
#define WEAVE_WARP_SPLIT_MODE {}
#define WEAVE_REG_PULL_BYTES {}u
#define WEAVE_HETERO_SM_LIMIT {}u
#define WEAVE_DISP_REGS {}u
#define WEAVE_EPI_REGS {}u
#define WEAVE_BLOCK_M {}
#define WEAVE_BLOCK_N {}
#define WEAVE_BLOCK_K {}
#define WEAVE_STORE_BLOCK_M {}
#define WEAVE_STORE_BLOCK_M_L2 {}
#define WEAVE_DYN_EPILOGUE {}
#define WEAVE_DYN_TMA_A {}
#define WEAVE_OVERLAP_EXPERT_SMEM {}
#define WEAVE_EXPERT_STEP_MODE {}
#define WEAVE_ACCUM_STAGES {}u
#define WEAVE_TMA_STORE_STAGES {}u
#define WEAVE_SCHED_STAGES {}u
#define DG_D3_HAS_EXTRA_TMAS {}
#define WEAVE_EXPERTS_PER_WAVE {}
#define WEAVE_STATIC_EXPERT_ORDER {}
#define WEAVE_READY_QUEUE {}
#define WEAVE_SCHED_MODE {}
{}
#define WEAVE_DEDUP_MODE {}
#define WEAVE_DEDUP_SCHED {}
#define WEAVE_COMBINE_GROUP_TILES {}u
#define WEAVE_COMBINE_REDUCE_WARPS {}u
{}
#define WEAVE_SHARED_L2_EARLY_BLOCKS {}u
#define WEAVE_ENABLE_MULTICAST {}
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
        {},
        {}, {},
        {}, {},
        {},
        {}, {}, {},
        {},
        {}, {},
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
)",
            knob_macros.str(),
            d1_g,
            d2_mode,
            d2_reg_pull_bytes,
            d2_d8_sm_limit, d2_disp_regs, d2_epi_regs,
            d3_bm, d3_bn, d3_bk, d3_store_m, d3_store_m_l2,
            d3_dyn_epi,
            d3_dyn_tma_a,
            d3_overlap_smem,
            d3_expert_step_mode,
            d3_accum_stages, d3_tma_store_stages, d3_sched_stages,
            args.d3_has_extra_tmas ? 1 : 0,
            d4_epw,
            d4_static,
            d4_readyq,
            d4_mode,
            d4_static_code,
            d5_dedup, d5_sched,
            d6_g, d6_rw,
            d7_macros.str(),
            get_weave_env<int>("WEAVE_SHARED_L2_EARLY_BLOCKS", "DG_SHARED_L2_EARLY_BLOCKS", nullptr, 10),
            enable_multicast,
            args.num_max_tokens_per_rank,
            args.hidden, args.intermediate_hidden,
            args.num_experts, args.num_shared_experts,
            args.num_topk,
            args.config.block_m, args.config.block_n, args.config.block_k,
            args.config.store_block_m_l1,
            args.config.sf_block_m, args.config.sf_block_n,
            args.config.num_ring_tokens,
            args.config.num_sf_ring_tokens,
            args.config.num_stages,
            args.config.num_bytes_per_pull,
            args.config.num_dispatch_threads, args.config.num_non_epilogue_threads, args.config.num_epilogue_threads,
            args.launch_args.grid_dim.first, args.num_ranks,
            to_string(args.activation_clamp),
            args.fast_math ? "true" : "false");
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        if (not args.d3_has_extra_tmas) {
            DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
                args.y,
                args.cumulative_local_expert_recv_stats,
                args.num_tokens,
                args.sym_buffer_ptrs,
                args.tensor_map_l1_acts,
                args.tensor_map_l1_acts_sf,
                args.tensor_map_l1_weights,
                args.tensor_map_l1_weights_sf,
                args.tensor_map_l1_output,
                args.tensor_map_l2_acts,
                args.tensor_map_l2_acts_sf,
                args.tensor_map_l2_weights,
                args.tensor_map_l2_weights_sf,
                args.tensor_map_shared_l1_acts,
                args.tensor_map_shared_l1_acts_sf,
                args.tensor_map_shared_l1_weights,
                args.tensor_map_shared_l1_weights_sf,
                args.tensor_map_shared_l1_output,
                args.tensor_map_shared_l2_acts,
                args.tensor_map_shared_l2_acts_sf,
                args.tensor_map_shared_l2_weights,
                args.tensor_map_shared_l2_weights_sf,
                args.sm_locality_domains
            ));
            return;
        }
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.y,
            args.cumulative_local_expert_recv_stats,
            args.num_tokens,
            args.sym_buffer_ptrs,
            args.tensor_map_l1_acts,
            args.tensor_map_l1_acts_sf,
            args.tensor_map_l1_weights,
            args.tensor_map_l1_weights_sf,
            args.tensor_map_l1_output,
            args.tensor_map_l1_output_m8,
            args.tensor_map_l2_acts,
            args.tensor_map_l2_acts_sf,
            args.tensor_map_l2_weights,
            args.tensor_map_l2_weights_sf,
            args.tensor_map_l1_acts_m16,
            args.tensor_map_l2_acts_m16,
            args.tensor_map_l1_acts_m8,
            args.tensor_map_l2_acts_m8,
            args.tensor_map_shared_l1_acts,
            args.tensor_map_shared_l1_acts_sf,
            args.tensor_map_shared_l1_weights,
            args.tensor_map_shared_l1_weights_sf,
            args.tensor_map_shared_l1_output,
            args.tensor_map_shared_l2_acts,
            args.tensor_map_shared_l2_acts_sf,
            args.tensor_map_shared_l2_weights,
            args.tensor_map_shared_l2_weights_sf,
            args.sm_locality_domains
        ));
    }
};

static void sm100_fp8_fp4_mega_moe(
    const torch::Tensor& y,
    const torch::Tensor& l1_acts, const torch::Tensor& l1_acts_sf,
    const torch::Tensor& l2_acts, const torch::Tensor& l2_acts_sf,
    const torch::Tensor& shared_l1_acts, const torch::Tensor& shared_l1_acts_sf,
    const torch::Tensor& shared_l2_acts, const torch::Tensor& shared_l2_acts_sf,
    const torch::Tensor& l1_weights, const torch::Tensor& l2_weights,
    const torch::Tensor& l1_weights_sf, const torch::Tensor& l2_weights_sf,
    const torch::Tensor& shared_l1_weights, const torch::Tensor& shared_l2_weights,
    const torch::Tensor& shared_l1_weights_sf, const torch::Tensor& shared_l2_weights_sf,
    const std::optional<torch::Tensor> cumulative_local_expert_recv_stats,
    const std::vector<int64_t>& sym_buffer_ptrs,
    const torch::Tensor& sm_locality_domains,
    const int& rank_idx, const int& num_max_tokens_per_rank,
    const int& num_experts_per_rank,
    const int& num_shared_experts,
    const int& num_tokens, const int& num_topk,
    const int& hidden, const int& intermediate_hidden,
    const float& activation_clamp,
    const bool& fast_math,
    // NOTE(h20_fix): align with apis/mega_moe.hpp call site which forwards the
    // situ activation knobs. The SM100 FP8xFP4 kernel does not yet consume situ
    // (pre-existing gap); these parameters preserve the call signature while the
    // profiler parameters below keep their defaults. swiglu behavior is unchanged.
    const bool& use_situ,
    const float& situ_beta,
    const float& situ_linear_beta,
    const std::optional<torch::Tensor>& profiler_buf = std::nullopt,
    const std::optional<torch::Tensor>& profiler_sm_map = std::nullopt,
    const int64_t& profiler_capture_slots = 0,
    const int64_t& profiler_max_records = 0,
    const std::optional<int64_t>& profile_mode = std::nullopt,
    const int64_t& profiler_warp_mask = (1ll << 20) - 1
) {
    const auto num_ranks = static_cast<int>(sym_buffer_ptrs.size());
    const auto num_experts = num_experts_per_rank * num_ranks;
    const auto num_ring_tokens = static_cast<int>(l1_acts.size(0));
    const auto num_sf_ring_tokens = static_cast<int>(l1_acts_sf.size(0));
    const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
    const auto num_sms = device_runtime->get_num_sms();

    const int d3_dyn_epi = get_weave_env<int>("WEAVE_DYN_EPILOGUE", "DG_TUNE_D3_DYN_EPI", "DG_D3_DYN_EPI", 0);
    const int d3_dyn_tma_a = get_weave_env<int>("WEAVE_DYN_TMA_A", "DG_TUNE_D3_DYN_TMA_A", "DG_D3_DYN_TMA_A", 0);
    const bool d3_has_extra_tmas = (d3_dyn_epi == 2) or (d3_dyn_tma_a > 0);

    const int selected_profile_mode = profile_mode.has_value()
        ? static_cast<int>(*profile_mode) : 0;
    DG_HOST_ASSERT(0 <= selected_profile_mode and selected_profile_mode <= 2);
    int64_t* dg_prof_buf_ptr = nullptr;
    const int32_t* dg_prof_sm_map_ptr = nullptr;
    uint32_t dg_prof_capture_slots = 0;
    uint32_t dg_prof_max_records = 0;

    const auto config = get_mega_moe_config(
        num_ranks, num_experts, num_experts_per_rank,
        num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
        num_ring_tokens, num_sf_ring_tokens,
        MmaKind::MXFP8FP4);

    constexpr int kGranK = 32;
    const int sf_smem_outer_dim = config.block_k / (kGranK * 4);
    const int l1_out_subtile_n = 64;
    const auto tensor_map_l1_acts = make_tma_2d_desc(l1_acts,
                                                     hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l1_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l1_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_acts_sf,
                                                        config.num_sf_ring_tokens, hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l1_weights = make_tma_2d_desc(l1_weights,
                                                        hidden, num_experts_per_rank * intermediate_hidden * 2,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l1_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l1_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l1_weights_sf,
                                                           intermediate_hidden * 2, hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                           sf_smem_outer_dim);
    const auto tensor_map_l1_output = make_tma_2d_desc(l2_acts,
                                                       intermediate_hidden, config.num_ring_tokens,
                                                       l1_out_subtile_n, config.store_block_m_l1,
                                                       static_cast<int>(l2_acts.stride(-2)),
                                                       config.swizzle_acts_mode / 2);
    const auto tensor_map_l1_output_m16 = tensor_map_l1_output;
    const auto tensor_map_l1_output_m8 = d3_has_extra_tmas ? make_tma_2d_desc(
        l2_acts,
        intermediate_hidden, config.num_ring_tokens,
        l1_out_subtile_n, 8,
        static_cast<int>(l2_acts.stride(-2)),
        config.swizzle_acts_mode / 2) : tensor_map_l1_output;

    const auto tensor_map_l2_acts = make_tma_2d_desc(l2_acts,
                                                     intermediate_hidden, config.num_ring_tokens,
                                                     config.block_k, config.load_block_m,
                                                     static_cast<int>(l2_acts.stride(-2)),
                                                     config.swizzle_acts_mode);
    const auto tensor_map_l2_acts_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_acts_sf,
                                                        config.num_sf_ring_tokens, intermediate_hidden,
                                                        config.sf_block_m, kGranK,
                                                        1, 0, 0, false,
                                                        sf_smem_outer_dim);
    const auto tensor_map_l2_weights = make_tma_2d_desc(l2_weights,
                                                        intermediate_hidden, num_experts_per_rank * hidden,
                                                        config.block_k, config.load_block_n,
                                                        static_cast<int>(l2_weights.stride(-2)),
                                                        config.swizzle_weights_mode);
    const auto tensor_map_l2_weights_sf = make_tma_sf_desc(cute::UMMA::Major::MN, l2_weights_sf,
                                                           hidden, intermediate_hidden,
                                                           config.block_n, kGranK,
                                                           num_experts_per_rank, 0, 0, false,
                                                           sf_smem_outer_dim);

    const int m16_load_m = std::min(config.load_block_m, 16);
    const int m8_load_m = std::min(config.load_block_m, 8);
    const auto tensor_map_l1_acts_m16 = d3_has_extra_tmas ? make_tma_2d_desc(
        l1_acts, hidden, config.num_ring_tokens,
        config.block_k, m16_load_m, static_cast<int>(l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_l2_acts_m16 = d3_has_extra_tmas ? make_tma_2d_desc(
        l2_acts, intermediate_hidden, config.num_ring_tokens,
        config.block_k, m16_load_m, static_cast<int>(l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_l1_acts_m8 = d3_has_extra_tmas ? make_tma_2d_desc(
        l1_acts, hidden, config.num_ring_tokens,
        config.block_k, m8_load_m, static_cast<int>(l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_l2_acts_m8 = d3_has_extra_tmas ? make_tma_2d_desc(
        l2_acts, intermediate_hidden, config.num_ring_tokens,
        config.block_k, m8_load_m, static_cast<int>(l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;

    const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_acts,
        hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l1_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l1_acts;
    const auto tensor_map_shared_l1_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_acts_sf,
        static_cast<int>(shared_l1_acts_sf.size(0)), hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_acts_sf;
    const auto tensor_map_shared_l1_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l1_weights,
        hidden, shared_intermediate_hidden * 2,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l1_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l1_weights;
    const auto tensor_map_shared_l1_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l1_weights_sf,
        shared_intermediate_hidden * 2, hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l1_weights_sf;
    const auto tensor_map_shared_l1_output = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        l1_out_subtile_n, config.store_block_m_l1,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode / 2) : tensor_map_l1_output;
    const auto tensor_map_shared_l2_acts = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_acts,
        shared_intermediate_hidden, num_max_tokens_per_rank,
        config.block_k, config.load_block_m,
        static_cast<int>(shared_l2_acts.stride(-2)),
        config.swizzle_acts_mode) : tensor_map_l2_acts;
    const auto tensor_map_shared_l2_acts_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_acts_sf,
        static_cast<int>(shared_l2_acts_sf.size(0)), shared_intermediate_hidden,
        config.sf_block_m, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_acts_sf;
    const auto tensor_map_shared_l2_weights = num_shared_experts > 0 ? make_tma_2d_desc(
        shared_l2_weights,
        shared_intermediate_hidden, hidden,
        config.block_k, config.load_block_n,
        static_cast<int>(shared_l2_weights.stride(-2)),
        config.swizzle_weights_mode) : tensor_map_l2_weights;
    const auto tensor_map_shared_l2_weights_sf = num_shared_experts > 0 ? make_tma_sf_desc(
        cute::UMMA::Major::MN, shared_l2_weights_sf,
        hidden, shared_intermediate_hidden,
        config.block_n, kGranK,
        1, 0, 0, false,
        sf_smem_outer_dim) : tensor_map_l2_weights_sf;

    int* cumulative_local_expert_recv_stats_ptr = nullptr;
    if (cumulative_local_expert_recv_stats.has_value())
        cumulative_local_expert_recv_stats_ptr = cumulative_local_expert_recv_stats->data_ptr<int>();

    const SM100FP8FP4MegaMoERuntime::Args args = {
        .num_max_tokens_per_rank = num_max_tokens_per_rank,
        .hidden = hidden, .intermediate_hidden = intermediate_hidden,
        .num_experts = num_experts, .num_shared_experts = num_shared_experts,
        .num_topk = num_topk,
        .num_ranks = num_ranks,
        .activation_clamp = activation_clamp,
        .fast_math = fast_math,
        .d3_has_extra_tmas = d3_has_extra_tmas,
        .config = config,
        .y = y.data_ptr(),
        .dg_prof_buf = dg_prof_buf_ptr,
        .dg_prof_max_records = dg_prof_max_records,
        .dg_prof_sm_map = dg_prof_sm_map_ptr,
        .dg_prof_capture_slots = dg_prof_capture_slots,
        .dg_prof_warp_mask = static_cast<uint32_t>(profiler_warp_mask),
        .profile_mode = selected_profile_mode,
        .cumulative_local_expert_recv_stats = cumulative_local_expert_recv_stats_ptr,
        .num_tokens = num_tokens,
        .sm_locality_domains = sm_locality_domains.data_ptr<uint8_t>(),
        .sym_buffer_ptrs = layout::SymBuffer<>(sym_buffer_ptrs, rank_idx),
        .tensor_map_l1_acts = tensor_map_l1_acts,
        .tensor_map_l1_acts_sf = tensor_map_l1_acts_sf,
        .tensor_map_l1_weights = tensor_map_l1_weights,
        .tensor_map_l1_weights_sf = tensor_map_l1_weights_sf,
        .tensor_map_l1_output = tensor_map_l1_output,
        .tensor_map_l1_output_m16 = tensor_map_l1_output_m16,
        .tensor_map_l1_output_m8 = tensor_map_l1_output_m8,
        .tensor_map_l2_acts = tensor_map_l2_acts,
        .tensor_map_l2_acts_sf = tensor_map_l2_acts_sf,
        .tensor_map_l2_weights = tensor_map_l2_weights,
        .tensor_map_l2_weights_sf = tensor_map_l2_weights_sf,
        .tensor_map_l1_acts_m16 = tensor_map_l1_acts_m16,
        .tensor_map_l2_acts_m16 = tensor_map_l2_acts_m16,
        .tensor_map_l1_acts_m8 = tensor_map_l1_acts_m8,
        .tensor_map_l2_acts_m8 = tensor_map_l2_acts_m8,
        .tensor_map_shared_l1_acts = tensor_map_shared_l1_acts,
        .tensor_map_shared_l1_acts_sf = tensor_map_shared_l1_acts_sf,
        .tensor_map_shared_l1_weights = tensor_map_shared_l1_weights,
        .tensor_map_shared_l1_weights_sf = tensor_map_shared_l1_weights_sf,
        .tensor_map_shared_l1_output = tensor_map_shared_l1_output,
        .tensor_map_shared_l2_acts = tensor_map_shared_l2_acts,
        .tensor_map_shared_l2_acts_sf = tensor_map_shared_l2_acts_sf,
        .tensor_map_shared_l2_weights = tensor_map_shared_l2_weights,
        .tensor_map_shared_l2_weights_sf = tensor_map_shared_l2_weights_sf,
        .launch_args = LaunchArgs(num_sms,
                                  config.num_dispatch_threads + config.num_non_epilogue_threads + config.num_epilogue_threads,
                                  config.smem_size, 2)
    };

    const auto code = SM100FP8FP4MegaMoERuntime::generate(args);
    const auto runtime = compiler->build("sm100_fp8_fp4_mega_moe", code);
    if (get_env<int>("DG_JIT_COMPILE_ONLY", 0))
        return;
    SM100FP8FP4MegaMoERuntime::launch(runtime, args);
}

} // namespace deep_gemm
