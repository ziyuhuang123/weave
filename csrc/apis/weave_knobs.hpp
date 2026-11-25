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
        // The public surface is exactly 14 knobs: Spatial 5, Temporal 3,
        // Implementation 6, in the same order as deep_gemm/mega/knobs.py
        // (tests/check_weave_knob_schema.py compares the two lists).
        // Six former knobs (WEAVE_REDUCE_WARPS, WEAVE_COMBINE_SYNC_CHUNK,
        // WEAVE_REG_BALANCE, WEAVE_LOCAL_SCATTER_DEDUP, WEAVE_SM0_STREAM_SCHED,
        // WEAVE_READY_QUEUE) are retired from the public surface: they stay
        // internal compile-time macros (default 0) and internal environment
        // diagnostics, and are deliberately absent from this list.
        // WEAVE_BLOCK_M_L1 / WEAVE_BLOCK_M_L2 are legacy environment aliases of
        // WEAVE_GEMM1_CONFIG / WEAVE_GEMM2_CONFIG.  WEAVE_ADAPT_GEMM_CONFIG (and
        // its alias WEAVE_ADAPT_BM, plus DG_ADAPT_BM) is accepted by the host
        // for backward compatibility but is NOT a public knob: host shape-adaptive
        // GEMM tuple selection is always on when GEMM1/GEMM2 config are 0.
        //
        // Spatial (strictly SM-level & Warp/Wave-level hardware resource ratios)
        {"WEAVE_DISPATCH_ACTIVE_SMS", "Spatial",
         "0 = auto/all SMs; D = only D SMs run dispatch pull (independent of the scatter SM count)",
         "SM-level dispatch allocation."},
        {"WEAVE_SCATTER_ACTIVE_SMS", "Spatial",
         "0 = auto/all SMs; K = only K SMs drain the scatter queue (independent of the dispatch SM count)",
         "SM-level scatter allocation (independent from dispatch SMs)."},
        {"WEAVE_DISPATCH_WARPS", "Spatial",
         "0 = arch default; 4 or 8 = dispatch warps per CTA",
         "Per-SM dispatch communication warp ratio."},
        {"WEAVE_SCATTER_WARPS", "Spatial",
         "0 = arch default; 4 or 8 = scatter warps",
         "Per-SM scatter communication warp ratio."},
        {"WEAVE_CU_WAVE_SPLIT", "Spatial",
         "0 = auto; 1 = 0-8; 2 = 4-4; 3 = 8-0 (MI308X per-CU comm-compute wave split; H20 uses E, M-E or 7-W expert-wave syntax)",
         "Per-unit wave split: MI308X per-CU (comm - compute) wavefront ratio across 0-8, 4-4, and 8-0; H20 L1/L2 expert-wave split."},
        // Temporal (Head / Middle / Tail overlap, task-queue dispatch & sync granularity)
        {"WEAVE_V_HEAD", "Temporal",
         "0 = automatic; N = head quota (SM103: N helper rounds of num_SMs tokens; H20: N head tokens; MI308X: claim quota)",
         "Head schedule: quota of opening pull tasks that helping warps complete before switching to GEMM."},
        {"WEAVE_READY_GROUP_SIZE", "Temporal",
         "0 = full-expert readiness; N = 1, 2, 4, 8, 16 group-granularity publication",
         "Head schedule: fine-grained readiness publication granularity."},
        {"WEAVE_SCATTER_OWNER", "Temporal",
         "0 = epilogue_only; 1 = comm_only",
         "Middle schedule: strict binary scatter role ownership."},
        // Implementation (Independent GEMM1 / GEMM2 / Dispatch / Scatter operator configs:
        // M -> blockM synchronously locks blockN/K, mainloop_stages, epi_M/N/K, epi_stages)
        {"WEAVE_PRECISION_MODE", "Implementation",
         "0 = fp8xfp4; 1 = fp4xfp4; 2 = fp8xfp8; 3 = bf16",
         "Single-op numeric precision and tensor-core mode."},
        {"WEAVE_GEMM1_CONFIG", "Implementation",
         "0 = auto (shape-adaptive tuple from (M_expected, N, K)); BM or BM[:STORE_BM[:BK[:STAGES[:EPI]]]] (EPI in 0..4)",
         "Independent GEMM1 config selector (locks blockN/K, mainloop stages, epi M/N/K & multi-stage epilogue)."},
        {"WEAVE_GEMM2_CONFIG", "Implementation",
         "0 = follow GEMM1/auto; same grammar as WEAVE_GEMM1_CONFIG (decoupled from GEMM1)",
         "Independent GEMM2 config selector (locks blockN/K, mainloop stages, epi M/N/K & multi-stage epilogue)."},
        {"WEAVE_REG_PULL_BYTES", "Implementation",
         "0 = SMEM/LDS staging; 1024, 2048, 4096 (32-lane warp) or 8192 (64-lane wavefront) = bytes per pull chunk",
         "Independent Dispatch operator per-warp/wavefront register-direct pull chunk width."},
        {"WEAVE_DEDUP_MODE", "Implementation",
         "0 = off; 1 = pull each unique (src_rank, token) once + local multi-expert fanout",
         "Independent Dispatch operator remote token pull deduplication."},
        {"WEAVE_SCATTER_STAGING", "Implementation",
         "0 = reg; 1 = smem; 2 = hbm",
         "Independent Scatter operator staging path (multi-stage register-direct, shared memory, or HBM buffer)."},
    };
    return entries;
}

static std::pair<weave::HardwareArch, weave::PrecisionMode> arch_for_mma_type(
        const std::string& mma_type) {
    if (mma_type == "fp8" or mma_type == "fp8xfp8")
        return {weave::HardwareArch::H20_SM90, weave::PrecisionMode::FP8xFP8};
    if (mma_type == "bf16" or mma_type == "bf16xbf16")
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
    // Exactly the 14 public names, in schema order.  The internal struct field
    // names (adapt_bm, block_m_l1, block_m_l2, ...) are unchanged.
    out["WEAVE_DISPATCH_ACTIVE_SMS"] = knobs.dispatch_active_sms;
    out["WEAVE_SCATTER_ACTIVE_SMS"] = knobs.scatter_active_sms;
    out["WEAVE_DISPATCH_WARPS"] = knobs.dispatch_warps;
    out["WEAVE_SCATTER_WARPS"] = 0u;
    out["WEAVE_CU_WAVE_SPLIT"] = knobs.cu_wave_split;
    out["WEAVE_V_HEAD"] = static_cast<uint32_t>(WEAVE_V_HEAD);
    out["WEAVE_READY_GROUP_SIZE"] = 0u;
    out["WEAVE_SCATTER_OWNER"] = knobs.scatter_owner;
    out["WEAVE_PRECISION_MODE"] = knobs.precision_mode;
    out["WEAVE_GEMM1_CONFIG"] = knobs.block_m_l1;
    out["WEAVE_GEMM2_CONFIG"] = knobs.block_m_l2;
    out["WEAVE_REG_PULL_BYTES"] = 0u;
    out["WEAVE_DEDUP_MODE"] = 0u;
    out["WEAVE_SCATTER_STAGING"] = knobs.scatter_staging;
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
