#pragma once
// =============================================================================
// Python-visible schema and defaults for the unified Weave knob space.
//
// The same WeaveKnobs request is used by the CTA-0 policy rail and by
// the host dispatchers.  This binding exposes the public names, their category
// (Spatial / Temporal / Implementation) and the compiled defaults so tuning
// scripts and the documentation generator share one source of truth.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include <deep_gemm/scheduler/weave_knobs.cuh>

#include "../jit_kernels/heuristics/weave_knob_config.hpp"

namespace deep_gemm::weave_api {

namespace py = pybind11;

struct KnobSchemaEntry {
    const char* name;
    const char* category;
    const char* values;
    const char* mechanism;
};

static const std::vector<KnobSchemaEntry>& knob_schema_entries() {
    static const std::vector<KnobSchemaEntry> entries = {
        // Spatial (strictly SM-level & Warp/Wave-level hardware resource ratios)
        {"WEAVE_DISPATCH_ACTIVE_SMS", "Spatial",
         "0 = auto/all SMs; D = only D SMs run dispatch pull",
         "SM-level dispatch allocation."},
        {"WEAVE_SCATTER_ACTIVE_SMS", "Spatial",
         "0 = auto/all SMs; K = only K SMs drain the scatter queue",
         "SM-level scatter allocation (independent from dispatch SMs)."},
        {"WEAVE_SM0_STREAM_SCHED", "Spatial",
         "0 = off; 1 = SM 0 communication warps become the scheduler lane",
         "Dedicated SM 0 streaming scheduler."},
        {"WEAVE_DISPATCH_WARPS", "Spatial",
         "0 = auto; 4 or 8 = dispatch warps per CTA",
         "Per-SM dispatch communication warp ratio."},
        {"WEAVE_SCATTER_WARPS", "Spatial",
         "0 = automatic; 4 or 8 = scatter warps",
         "Per-SM scatter communication warp ratio."},
        {"WEAVE_REG_BALANCE", "Spatial",
         "0 = default (96/72/168); 1 = rebalanced (64/56/192) at E/rank > 64",
         "Per-SM register file ratio across comm, MMA/TMA, and epilogue warps."},
        {"WEAVE_AMD_8WAVE_MIN_TOK", "Spatial",
         "N = minimum local tokens per CU-expert; 0 = dual-group default",
         "MI308X per-CU 8-wave vs 4-wave dual-group ratio floor."},
        {"WEAVE_DUALGROUP_MIN_TOKENS", "Spatial",
         "N = second compute-group threshold",
         "MI308X per-CU second compute-wave group threshold."},
        // Temporal (Head / Middle / Tail overlap, role switching, interleaving & sync granularity)
        {"WEAVE_OVERLAP_PIPELINE", "Temporal",
         "0 = serialized; 1 = compute-communication overlap",
         "Cross-stage compute-communication overlap."},
        {"WEAVE_V_HEAD", "Temporal",
         "0 = automatic; N = opening head-pull budget in tokens",
         "Head schedule: cooperative all-warp opening pull budget."},
        {"WEAVE_HEAD_PULL_CHUNK", "Temporal",
         "0 = one opening prefix; N = tokens per head-pull chunk",
         "Head schedule: opening head-pull chunk granularity."},
        {"WEAVE_READY_GROUP_SIZE", "Temporal",
         "0 = per-expert readiness; N = group-granularity publication",
         "Head schedule: fine-grained readiness publication granularity."},
        {"WEAVE_READY_QUEUE", "Temporal",
         "0 = preset expert order; 1 = global FIFO ready queue",
         "Head/Middle schedule: arrival-ordered global FIFO ready queue."},
        {"WEAVE_SCATTER_OWNER", "Temporal",
         "0 = epilogue_only; 1 = comm_only",
         "Middle schedule: strict binary scatter role ownership."},
        {"WEAVE_L2_KBLOCK_DEP", "Temporal",
         "0 = default L1-first order; 1 = tight 1 L2 <-> 1 L1 alternation",
         "Middle schedule: 64-bit XOR K-block L1-to-L2 interleaving."},
        {"WEAVE_SCHED_L2_RATIO", "Temporal",
         "1..4 = L2 tasks per L1 task in interleaved scheduling",
         "Middle schedule: L2-to-L1 interleaved wave ratio."},
        {"WEAVE_SCHED_MODE", "Temporal",
         "0 = descending load; 1 = L2 tile swizzle; 2 = expert-major",
         "Middle schedule: expert/tile ordering key."},
        {"WEAVE_EXPERTS_PER_WAVE", "Temporal",
         "0 = auto; N = LPT scheduling window",
         "Middle schedule: wave window size for load pairing."},
        {"WEAVE_COMBINE_SYNC_CHUNK", "Temporal",
         "0 = global barrier; 16, 32, 64, 128 = chunked dedup combine sync",
         "Tail schedule: G-token chunked cross-GPU combine sync."},
        {"WEAVE_EARLY_STREAM_REDUCE", "Temporal",
         "0 = barrier before combine; 1 = per-token cnt==topk early reduce",
         "Tail schedule: barrier-free streaming reduce."},
        {"WEAVE_ALL_WAVE_COMBINE", "Temporal",
         "0 = compute waves only; 1 = all waves join the tail combine",
         "Tail schedule: cooperative all-idle-warp tail combine."},
        // Implementation (Single-op optimizations, independent GEMM1/GEMM2/Dispatch/Scatter search,
        // per-M profile lookup, TMEM 1.75 & 2-stage Reg pipelines, staging paths & dedup)
        {"WEAVE_PRECISION_MODE", "Implementation",
         "0 = fp8xfp4; 1 = fp4xfp4; 2 = fp8xfp8; 3 = bf16",
         "Single-op numeric precision and tensor-core mode."},
        {"WEAVE_ADAPT_BM", "Implementation",
         "0 = fixed BM; 1 = per-expert M-bucketed profile lookup",
         "Runtime per-expert M-bucketed (Mainloop + Epilogue) profile lookup."},
        {"WEAVE_BM_TIER_MODE", "Implementation",
         "0/1/2 = per-expert BM ladder mode",
         "MI308X offline-calibrated per-expert M-bucket ladder."},
        {"WEAVE_BLOCK_M_L1", "Implementation",
         "0 = auto; 16, 32, 64, 80, 96, 128, 192, 240, 256",
         "Independent GEMM1 mainloop block-M parameter."},
        {"WEAVE_STORE_BLOCK_M_L1", "Implementation",
         "0 = auto; positive multiple of 8",
         "Independent GEMM1 epilogue store-M parameter."},
        {"WEAVE_BLOCK_M_L2", "Implementation",
         "0 = auto; 16, 32, 64, 80, 96, 128, 192, 240, 256",
         "Independent GEMM2 mainloop block-M parameter."},
        {"WEAVE_STORE_BLOCK_M_L2", "Implementation",
         "0 = auto; positive multiple of 8",
         "Independent GEMM2 epilogue store-M parameter."},
        {"WEAVE_TMEM_PIPELINE_MODE", "Implementation",
         "0 = auto; 100 = 1.0; 175 = 1.75; 200 = 2.0",
         "Single-op TMEM accumulator pipeline mode (1.0 / 1.75 overlap / 2.0)."},
        {"WEAVE_EPI_REG_STAGES", "Implementation",
         "0 = auto; 1 = single register buffer; 2 = two register stages",
         "Single-op 2-stage register epilogue prefetch pipeline."},
        {"WEAVE_RING_BLOCKS", "Implementation",
         "0 = auto live-pool ring; N = forced ring block count",
         "Intermediate activation ring-buffer slot reuse."},
        {"WEAVE_REG_PULL_BYTES", "Implementation",
         "0 = SMEM staging; 1024, 2048, 4096 = register-direct pull width",
         "Independent Dispatch operator pull packet width."},
        {"WEAVE_DEDUP_MODE", "Implementation",
         "0 = off; 1 = unique buckets; 2 = local pre-reduce",
         "Independent Dispatch operator input deduplication."},
        {"WEAVE_SCATTER_STAGING", "Implementation",
         "0 = reg_2stage; 1 = smem; 2 = hbm_dedup",
         "Independent Scatter operator staging implementation."},
        {"WEAVE_LOCAL_SCATTER_DEDUP", "Implementation",
         "0 = full-precision local accumulation; 1 = bf16x2 L2 atomic dedup",
         "Independent Scatter operator local deduplication."},
        {"WEAVE_SCATTER_SMEM_ALIAS", "Implementation",
         "0 = dedicated scatter SMEM; 1 = alias the dispatch dead zone",
         "Scatter SMEM buffer aliasing."},
        {"WEAVE_SCATTER_MODE", "Implementation",
         "0 = epilogue inline; 1 = SMEM handoff; 2 = local dedup first; 3 = local dedup + HBM queue",
         "Low-level scatter data-path code (derived on SM103)."},
        {"WEAVE_ENABLE_COST_MODEL", "Implementation",
         "0 = static policy; 1 = CTA-0 runtime cost model",
         "CTA-0 runtime policy selector."},
        {"WEAVE_ABLATION_PRESET", "Implementation",
         "0 = full Weave; 1 = -S; 2 = -S-T; 3 = -S-T-C",
         "Progressive ablation selector."},
        {"WEAVE_PROFILER_LEVEL", "Implementation",
         "0 = production; 1 = phase events; 2 = full events",
         "Instrumentation level."},
    };
    return entries;
}

static std::pair<weave::HardwareArch, weave::PrecisionMode> arch_for_mma_type(
        const std::string& mma_type) {
    if (mma_type == "fp8" or mma_type == "fp8xfp8")
        return {weave::HardwareArch::H20_SM90, weave::PrecisionMode::FP8xFP8};
    if (mma_type == "bf16")
        return {weave::HardwareArch::B300_SM100, weave::PrecisionMode::BF16};
    if (mma_type == "fp4xfp4" or mma_type == "nvfp4")
        return {weave::HardwareArch::B300_SM100, weave::PrecisionMode::NVFP4};
    if (mma_type == "mi308x")
        return {weave::HardwareArch::MI308X_GFX942, weave::PrecisionMode::FP8xFP8};
    return {weave::HardwareArch::B300_SM100, weave::PrecisionMode::FP8xFP4};
}

static std::vector<std::string> weave_knob_names() {
    std::vector<std::string> names;
    for (const auto& entry : knob_schema_entries())
        names.emplace_back(entry.name);
    return names;
}

static py::dict weave_knob_defaults(const std::string& mma_type) {
    const auto [arch, precision] = arch_for_mma_type(mma_type);
    const auto knobs = weave::default_weave_knobs(arch, precision);
    py::dict out;
    out["WEAVE_PRECISION_MODE"] = knobs.precision_mode;
    out["WEAVE_DISPATCH_WARPS"] = knobs.dispatch_warps;
    out["WEAVE_SCATTER_ACTIVE_SMS"] = knobs.scatter_active_sms;
    out["WEAVE_SM0_STREAM_SCHED"] = knobs.sm0_stream_sched;
    out["WEAVE_AMD_8WAVE_MIN_TOK"] = knobs.amd_8wave_min_tok;
    out["WEAVE_L2_KBLOCK_DEP"] = knobs.l2_kblock_dep;
    out["WEAVE_RING_BLOCKS"] = knobs.ring_blocks;
    out["WEAVE_EARLY_STREAM_REDUCE"] = knobs.early_stream_reduce;
    out["WEAVE_TMEM_PIPELINE_MODE"] = knobs.tmem_pipeline_mode;
    out["WEAVE_HEAD_PULL_CHUNK"] = knobs.head_pull_chunk;
    out["WEAVE_ADAPT_BM"] = knobs.adapt_bm;
    out["WEAVE_BLOCK_M_L1"] = knobs.block_m_l1;
    out["WEAVE_BLOCK_M_L2"] = knobs.block_m_l2;
    out["WEAVE_STORE_BLOCK_M_L1"] = knobs.store_block_m_l1;
    out["WEAVE_STORE_BLOCK_M_L2"] = knobs.store_block_m_l2;
    out["WEAVE_LOCAL_SCATTER_DEDUP"] = knobs.local_scatter_dedup;
    out["WEAVE_SCATTER_OWNER"] = knobs.scatter_owner;
    out["WEAVE_SCATTER_STAGING"] = knobs.scatter_staging;
    out["WEAVE_EPI_REG_STAGES"] = knobs.epi_reg_stages;
    out["WEAVE_DISPATCH_ACTIVE_SMS"] = knobs.dispatch_active_sms;
    out["WEAVE_COMBINE_SYNC_CHUNK"] = knobs.combine_sync_chunk;
    out["WEAVE_SCHED_L2_RATIO"] = knobs.sched_l2_ratio;
    out["WEAVE_REG_BALANCE"] = knobs.reg_balance;
    out["WEAVE_ABLATION_PRESET"] = static_cast<uint32_t>(WEAVE_ABLATION_PRESET);
    out["WEAVE_ENABLE_COST_MODEL"] = knobs.enable_runtime_cost_model;
    out["WEAVE_V_HEAD"] = static_cast<uint32_t>(WEAVE_V_HEAD);
    out["WEAVE_SCATTER_MODE"] = static_cast<uint32_t>(WEAVE_SCATTER_MODE);
    out["WEAVE_DUALGROUP_MIN_TOKENS"] = static_cast<uint32_t>(WEAVE_DUALGROUP_MIN_TOKENS);
    out["WEAVE_BM_TIER_MODE"] = static_cast<uint32_t>(WEAVE_BM_TIER_MODE);
    return out;
}

static py::list weave_knob_schema() {
    py::list out;
    for (const auto& entry : knob_schema_entries()) {
        py::dict row;
        row["name"] = entry.name;
        row["category"] = entry.category;
        row["values"] = entry.values;
        row["mechanism"] = entry.mechanism;
        out.append(row);
    }
    return out;
}

static void register_apis(py::module_& m) {
    auto submodule = m.def_submodule("weave_knobs",
        "Unified Spatial / Temporal / Implementation knob space");
    submodule.def("names", &weave_knob_names);
    submodule.def("schema", &weave_knob_schema);
    submodule.def("defaults", &weave_knob_defaults,
                  py::arg("mma_type") = "fp8xfp4");
}

}  // namespace deep_gemm::weave_api
