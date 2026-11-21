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
         "0 = CTA 0 comm warps pull/scatter normally; 1 = allocate CTA 0 comm warps as dedicated scheduler",
         "CTA-0 communication-warp scheduler allocation."},
        {"WEAVE_DISPATCH_WARPS", "Spatial",
         "0 = auto; 4 or 8 = dispatch warps per CTA",
         "Per-SM dispatch communication warp ratio."},
        {"WEAVE_SCATTER_WARPS", "Spatial",
         "0 = automatic; 4 or 8 = scatter warps",
         "Per-SM scatter communication warp ratio."},
        {"WEAVE_REDUCE_WARPS", "Spatial",
         "0 = off; 2, 4, 8 = dedicated fine-grained streaming Reduce warps in Middle",
         "Per-SM dedicated Reduce warp allocation overlapping Middle GEMM/Scatter."},
        {"WEAVE_CU_WAVE_SPLIT", "Spatial",
         "0 = auto; 1 = 0-8; 2 = 4-4; 3 = 8-0",
         "MI308X per-CU (comm - compute) wavefront ratio across 0-8, 4-4, and 8-0."},
        {"WEAVE_REG_BALANCE", "Spatial",
         "0 = default (96/72/168); 1 = rebalanced (64/56/192) at E/rank > 64",
         "Per-SM register file ratio across comm, MMA/TMA, and epilogue warps."},
        // Temporal (Head / Middle / Tail overlap, task-queue dispatch & sync granularity)
        {"WEAVE_V_HEAD", "Temporal",
         "0 = automatic; N = pull tasks (tokens) completed by helping warps before exiting to GEMM",
         "Head schedule: quota of opening pull tasks that helping warps complete before switching to GEMM."},
        {"WEAVE_READY_GROUP_SIZE", "Temporal",
         "0 = full-expert readiness; N = group-granularity publication",
         "Head schedule: fine-grained readiness publication granularity."},
        {"WEAVE_READY_QUEUE", "Temporal",
         "0 = formula (static heuristic formula); 1 = sm0_fifo (SM 0 dynamic global FIFO queue)",
         "Head/Middle task dispatch: static heuristic formula vs SM 0 dynamic global queue."},
        {"WEAVE_SCATTER_OWNER", "Temporal",
         "0 = epilogue_only; 1 = comm_only",
         "Middle schedule: strict binary scatter role ownership."},
        {"WEAVE_COMBINE_SYNC_CHUNK", "Temporal",
         "0 = global barrier; 16, 32, 64, 128 = chunked dedup combine sync",
         "Tail schedule: G-token chunked cross-GPU combine sync (all idle warps join Combine by default)."},
        // Implementation (Independent GEMM1 / GEMM2 / Dispatch / Scatter operator configs:
        // M -> blockM synchronously locks blockN/K, mainloop_stages, epi_M/N/K, epi_stages)
        {"WEAVE_PRECISION_MODE", "Implementation",
         "0 = fp8xfp4; 1 = fp4xfp4; 2 = fp8xfp8; 3 = bf16",
         "Single-op numeric precision and tensor-core mode."},
        {"WEAVE_ADAPT_BM", "Implementation",
         "0 = fixed profile; 1 = per-expert M-bucketed offline profile lookup",
         "Runtime per-expert M -> blockM profile lookup."},
        {"WEAVE_BLOCK_M_L1", "Implementation",
         "0 = auto from M; 16, 32, 64, 80, 96, 128, 192, 240, 256",
         "Independent GEMM1 profile selector (locks blockN/K, mainloop stages, epi M/N/K & stages)."},
        {"WEAVE_BLOCK_M_L2", "Implementation",
         "0 = auto from M; 16, 32, 64, 80, 96, 128, 192, 240, 256",
         "Independent GEMM2 profile selector (locks blockN/K, mainloop stages, epi M/N/K & stages)."},
        {"WEAVE_REG_PULL_BYTES", "Implementation",
         "0 = SMEM staging; 1024, 2048, 4096 = per-warp bytes per pull chunk (8, 16, 32 regs/thread)",
         "Independent Dispatch operator per-warp register-direct pull chunk width."},
        {"WEAVE_DEDUP_MODE", "Implementation",
         "0 = off; 1 = pull each unique (src_rank, token) once + local multi-expert fanout",
         "Independent Dispatch operator remote token pull deduplication."},
        {"WEAVE_SCATTER_STAGING", "Implementation",
         "0 = reg_2stage; 1 = smem; 2 = hbm_dedup",
         "Independent Scatter operator staging implementation."},
        {"WEAVE_LOCAL_SCATTER_DEDUP", "Implementation",
         "0 = direct per-hit scatter; 1 = bf16x2 local merge before send",
         "Independent Scatter operator local deduplication."},
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
    out["WEAVE_DISPATCH_ACTIVE_SMS"] = knobs.dispatch_active_sms;
    out["WEAVE_SCATTER_ACTIVE_SMS"] = knobs.scatter_active_sms;
    out["WEAVE_SM0_STREAM_SCHED"] = knobs.sm0_stream_sched;
    out["WEAVE_DISPATCH_WARPS"] = knobs.dispatch_warps;
    out["WEAVE_SCATTER_WARPS"] = 0u;
    out["WEAVE_REDUCE_WARPS"] = knobs.reduce_warps;
    out["WEAVE_CU_WAVE_SPLIT"] = knobs.cu_wave_split;
    out["WEAVE_REG_BALANCE"] = knobs.reg_balance;
    out["WEAVE_V_HEAD"] = static_cast<uint32_t>(WEAVE_V_HEAD);
    out["WEAVE_READY_GROUP_SIZE"] = 0u;
    out["WEAVE_READY_QUEUE"] = static_cast<uint32_t>(
        deep_gemm::weave_knobs::named_env("WEAVE_READY_QUEUE", {"formula", "sm0_fifo"}, 0));
    out["WEAVE_SCATTER_OWNER"] = knobs.scatter_owner;
    out["WEAVE_COMBINE_SYNC_CHUNK"] = knobs.combine_sync_chunk;
    out["WEAVE_ADAPT_BM"] = knobs.adapt_bm;
    out["WEAVE_BLOCK_M_L1"] = knobs.block_m_l1;
    out["WEAVE_BLOCK_M_L2"] = knobs.block_m_l2;
    out["WEAVE_REG_PULL_BYTES"] = 0u;
    out["WEAVE_DEDUP_MODE"] = 0u;
    out["WEAVE_SCATTER_STAGING"] = knobs.scatter_staging;
    out["WEAVE_LOCAL_SCATTER_DEDUP"] = knobs.local_scatter_dedup;
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
