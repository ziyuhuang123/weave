#pragma once
// =============================================================================
// Host-side resolution of the unified Weave knob space.
//
// This header is the single host entry point that turns the WEAVE_* environment
// into a WeaveKnobs request.  Every field of WeaveKnobs is either read by the
// CTA-0 cost model or copied into a JIT define by a backend dispatcher; there
// are no inert fields.  Backends that cannot implement a knob reject the
// request at parse time instead of silently ignoring it.
// =============================================================================

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

#include <deep_gemm/scheduler/weave_knobs.cuh>

#include "../../utils/exception.hpp"
#include "../../utils/system.hpp"

namespace deep_gemm::weave_knobs {

static inline int bounded_env(const char* primary, const char* legacy,
                              const int default_value, const int lo,
                              const int hi) {
    const int value = get_weave_env<int>(primary, legacy, nullptr, default_value);
    DG_HOST_ASSERT(value >= lo and value <= hi);
    return value;
}

static inline bool is_candidate_block_m(const uint32_t block_m) {
    if (block_m == 0u)
        return true;
    for (const uint32_t candidate : {16u, 32u, 64u, 80u, 96u, 128u, 192u, 240u, 256u})
        if (block_m == candidate)
            return true;
    return false;
}

// Named enum knobs accept either the documented name or its integer code.
// Returns `fallback` when the variable is unset or empty.
static inline int named_env(const char* name,
                            std::initializer_list<const char*> names,
                            const int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr or *raw == '\0')
        return fallback;
    int code = 0;
    for (const char* candidate : names) {
        if (std::strcmp(raw, candidate) == 0)
            return code;
        ++code;
    }
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    DG_HOST_ASSERT(end != raw and *end == '\0' and value >= 0 and
                   value < static_cast<long>(names.size()));
    return static_cast<int>(value);
}

// Whole per-GEMM config tuple (SM103 FP8xFP4 MegaMoE):
//   WEAVE_GEMM1_CONFIG / WEAVE_GEMM2_CONFIG = "BM[:STORE_BM[:BK[:STAGES[:EPI_REG_STAGES]]]]"
// (',' is accepted as a separator too).  BM is the phase's M tile, STORE_BM
// the phase's epilogue TMA-store height; BK, smem STAGES and EPI_REG_STAGES are
// launch-wide (one kernel runs both GEMMs) and must agree when both phases set
// them.  A bare integer is the legacy WEAVE_BLOCK_M_L1 / _L2 spelling (BM only).
// The legacy names stay accepted as aliases: WEAVE_BLOCK_M_L1 -> GEMM1,
// WEAVE_BLOCK_M_L2 -> GEMM2.  WEAVE_ADAPT_BM -> WEAVE_ADAPT_GEMM_CONFIG is an alias of an
// accepted-but-not-public name (see adapt_gemm_config_env below).
struct GemmPhaseConfig {
    bool set = false;
    int bm = 0, store_bm = 0, bk = 0, stages = 0, epi_reg_stages = 0;
};

static inline GemmPhaseConfig gemm_phase_config_env(const char* primary, const char* legacy) {
    GemmPhaseConfig cfg;
    const char* raw = std::getenv(primary);
    if (raw == nullptr or *raw == '\0') {
        raw = legacy != nullptr ? std::getenv(legacy) : nullptr;
        if (raw == nullptr or *raw == '\0')
            return cfg;
    }
    int fields[5] = {0, 0, 0, 0, 0};
    int n = 0;
    const char* p = raw;
    while (true) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        DG_HOST_ASSERT(end != p and v >= 0 and v <= 4096 and n < 5);
        fields[n ++] = static_cast<int>(v);
        if (*end == '\0')
            break;
        DG_HOST_ASSERT(*end == ':' or *end == ',');
        p = end + 1;
    }
    cfg.bm = fields[0];
    cfg.store_bm = fields[1];
    cfg.bk = fields[2];
    cfg.stages = fields[3];
    cfg.epi_reg_stages = fields[4];
    cfg.set = cfg.bm != 0 or n > 1;
    DG_HOST_ASSERT(is_candidate_block_m(static_cast<uint32_t>(cfg.bm)));
    DG_HOST_ASSERT(cfg.store_bm == 0 or (cfg.store_bm % 8 == 0 and (cfg.bm == 0 or cfg.store_bm <= cfg.bm)));
    DG_HOST_ASSERT(cfg.bk == 0 or cfg.bk == 128 or cfg.bk == 256);
    DG_HOST_ASSERT(cfg.stages == 0 or (cfg.stages >= 2 and cfg.stages <= 16));
    DG_HOST_ASSERT(cfg.epi_reg_stages >= 0 and cfg.epi_reg_stages <= 2);
    return cfg;
}

static inline GemmPhaseConfig gemm1_config_env() {
    return gemm_phase_config_env("WEAVE_GEMM1_CONFIG", "WEAVE_BLOCK_M_L1");
}
static inline GemmPhaseConfig gemm2_config_env() {
    return gemm_phase_config_env("WEAVE_GEMM2_CONFIG", "WEAVE_BLOCK_M_L2");
}

// WEAVE_ADAPT_GEMM_CONFIG (legacy alias WEAVE_ADAPT_BM, plus DG_ADAPT_BM): NOT a public
// knob (the public surface is 14 knobs), kept for backward compatibility.  -1 = unset
// (adaptive default: host shape-adaptive tuple selection is always on), 0 = fixed
// canonical tuple (internal legacy/diagnostic override), 1 = adaptive host (M, N, K)
// selector.  A pinned WEAVE_GEMM1_CONFIG / WEAVE_GEMM2_CONFIG tuple wins over it.
static inline int adapt_gemm_config_env() {
    for (const char* name : {"WEAVE_ADAPT_GEMM_CONFIG", "WEAVE_ADAPT_BM", "DG_ADAPT_BM"}) {
        const char* raw = std::getenv(name);
        if (raw == nullptr or *raw == '\0')
            continue;
        const int v = get_env<int>(name, 0);
        DG_HOST_ASSERT(v == 0 or v == 1);
        return v;
    }
    return -1;
}

static inline void validate(const weave::WeaveKnobs& knobs, const weave::HardwareArch arch);
static inline int precision_mode_env(const int fallback);

static inline weave::WeaveKnobs from_env(
        const weave::HardwareArch arch,
        const weave::PrecisionMode precision) {
    weave::WeaveKnobs knobs = weave::default_weave_knobs(arch, precision);

    // Spatial
    knobs.dispatch_warps = static_cast<uint32_t>(bounded_env(
        "WEAVE_DISPATCH_WARPS", "DG_DISPATCH_WARPS",
        static_cast<int>(knobs.dispatch_warps), 0, 8));
    knobs.reduce_warps = static_cast<uint32_t>(bounded_env(
        "WEAVE_REDUCE_WARPS", "DG_REDUCE_WARPS",
        static_cast<int>(knobs.reduce_warps), 0, 8));
    knobs.scatter_active_sms = static_cast<uint32_t>(bounded_env(
        "WEAVE_SCATTER_ACTIVE_SMS", "DG_SCATTER_ACTIVE_SMS",
        static_cast<int>(knobs.scatter_active_sms), 0, 1024));
    knobs.locality_domains = static_cast<uint32_t>(bounded_env(
        "WEAVE_LOCALITY_DOMAINS", "DG_LOCALITY_DOMAINS",
        static_cast<int>(knobs.locality_domains), 0, 4));
    const int ready_queue_mode = named_env(
        "WEAVE_READY_QUEUE", {"formula", "sm0_fifo"}, 0);
    const bool sm0_explicit = std::getenv("WEAVE_SM0_STREAM_SCHED") != nullptr or
                              std::getenv("DG_SM0_STREAM_SCHED") != nullptr;
    knobs.sm0_stream_sched = static_cast<uint32_t>(bounded_env(
        "WEAVE_SM0_STREAM_SCHED", "DG_SM0_STREAM_SCHED",
        static_cast<int>(knobs.sm0_stream_sched), 0, 1));
    if (not sm0_explicit and ready_queue_mode == 1)
        knobs.sm0_stream_sched = 1u;
    knobs.cu_wave_split = static_cast<uint32_t>(named_env(
        "WEAVE_CU_WAVE_SPLIT", {"auto", "0-8", "4-4", "8-0"},
        static_cast<int>(knobs.cu_wave_split)));
    knobs.amd_8wave_min_tok = static_cast<uint32_t>(bounded_env(
        "WEAVE_AMD_8WAVE_MIN_TOK", "DG_AMD_8WAVE_MIN_TOK",
        static_cast<int>(knobs.amd_8wave_min_tok), 0, 65536));
    knobs.dispatch_active_sms = static_cast<uint32_t>(bounded_env(
        "WEAVE_DISPATCH_ACTIVE_SMS", nullptr,
        static_cast<int>(knobs.dispatch_active_sms), 0, 1024));

    // Temporal
    const int v_head_env = get_weave_env<int>("WEAVE_V_HEAD", "DG_V_HEAD", nullptr, -1);
    DG_HOST_ASSERT(v_head_env >= -1 and v_head_env <= 65536);
    if (v_head_env > 0) {
        knobs.override_v_head = static_cast<uint32_t>(v_head_env);
        knobs.override_mask |= weave::kOverrideVHead;
    } else if (v_head_env == 0) {
        knobs.override_v_head = weave::kAutoU32;
        knobs.override_mask &= ~static_cast<uint32_t>(weave::kOverrideVHead);
    }
    knobs.l2_kblock_dep = static_cast<uint32_t>(bounded_env(
        "WEAVE_L2_KBLOCK_DEP", "DG_L2_KBLOCK_DEP",
        static_cast<int>(knobs.l2_kblock_dep), 0, 1));
    knobs.ring_blocks = static_cast<uint32_t>(bounded_env(
        "WEAVE_RING_BLOCKS", "DG_RING_BLOCKS",
        static_cast<int>(knobs.ring_blocks), 0, 1 << 20));
    knobs.early_stream_reduce = static_cast<uint32_t>(bounded_env(
        "WEAVE_EARLY_STREAM_REDUCE", "DG_EARLY_STREAM_REDUCE",
        static_cast<int>(knobs.early_stream_reduce), 0, 1));
    knobs.tmem_pipeline_mode = static_cast<uint32_t>(bounded_env(
        "WEAVE_TMEM_PIPELINE_MODE", "DG_TMEM_PIPELINE_MODE",
        static_cast<int>(knobs.tmem_pipeline_mode), 0, 200));
    knobs.head_pull_chunk = static_cast<uint32_t>(bounded_env(
        "WEAVE_HEAD_PULL_CHUNK", "DG_HEAD_PULL_CHUNK",
        static_cast<int>(knobs.head_pull_chunk), 0, 65536));
    knobs.combine_sync_chunk = static_cast<uint32_t>(bounded_env(
        "WEAVE_COMBINE_SYNC_CHUNK", nullptr,
        static_cast<int>(knobs.combine_sync_chunk), 0, 896));
    knobs.epi_reg_stages = static_cast<uint32_t>(bounded_env(
        "WEAVE_EPI_REG_STAGES", nullptr,
        static_cast<int>(knobs.epi_reg_stages), 0, 2));
    knobs.sched_l2_ratio = static_cast<uint32_t>(bounded_env(
        "WEAVE_SCHED_L2_RATIO", nullptr,
        static_cast<int>(knobs.sched_l2_ratio), 1, 4));

    // Implementation
    // Precision is the dispatched backend's; the env (aliases accepted) only selects it in Python.
    (void) precision_mode_env(-1);  // rejects unknown WEAVE_PRECISION_MODE spellings everywhere
    knobs.precision_mode = static_cast<uint32_t>(precision);
    {
        const int adapt = adapt_gemm_config_env();
        knobs.adapt_bm = adapt >= 0 ? static_cast<uint32_t>(adapt) : knobs.adapt_bm;
    }
    knobs.block_m_l1 = static_cast<uint32_t>(bounded_env(
        "WEAVE_BLOCK_M_L1", "DG_BLOCK_M_L1",
        static_cast<int>(knobs.block_m_l1), 0, 256));
    knobs.block_m_l2 = static_cast<uint32_t>(bounded_env(
        "WEAVE_BLOCK_M_L2", "DG_BLOCK_M_L2",
        static_cast<int>(knobs.block_m_l2), 0, 256));
    knobs.store_block_m_l1 = static_cast<uint32_t>(bounded_env(
        "WEAVE_STORE_BLOCK_M_L1", "DG_STORE_BLOCK_M_L1",
        static_cast<int>(knobs.store_block_m_l1), 0, 256));
    knobs.store_block_m_l2 = static_cast<uint32_t>(bounded_env(
        "WEAVE_STORE_BLOCK_M_L2", "DG_STORE_BLOCK_M_L2",
        static_cast<int>(knobs.store_block_m_l2), 0, 256));
    // New whole-tuple spellings win over the legacy per-field names.
    {
        const auto g1 = gemm_phase_config_env("WEAVE_GEMM1_CONFIG", nullptr);
        const auto g2 = gemm_phase_config_env("WEAVE_GEMM2_CONFIG", nullptr);
        if (g1.set) {
            knobs.block_m_l1 = static_cast<uint32_t>(g1.bm);
            if (g1.store_bm != 0)
                knobs.store_block_m_l1 = static_cast<uint32_t>(g1.store_bm);
        }
        if (g2.set) {
            knobs.block_m_l2 = static_cast<uint32_t>(g2.bm);
            if (g2.store_bm != 0)
                knobs.store_block_m_l2 = static_cast<uint32_t>(g2.store_bm);
        }
    }
    knobs.local_scatter_dedup = static_cast<uint32_t>(bounded_env(
        "WEAVE_LOCAL_SCATTER_DEDUP", "DG_LOCAL_SCATTER_DEDUP",
        static_cast<int>(knobs.local_scatter_dedup), 0, 1));
    knobs.reg_balance = static_cast<uint32_t>(bounded_env(
        "WEAVE_REG_BALANCE", nullptr,
        static_cast<int>(knobs.reg_balance), 0, 1));

    // Scatter ownership and staging.  Local dedup without an explicit staging
    // request selects the HBM dedup path (comm-owned, D7 mode 3) unless the
    // epilogue explicitly owns the scatter, in which case the epilogue performs
    // the local dedup itself (D7 mode 2).  HBM dedup always enables the bf16x2
    // dedup, and any staged path is owned by the communication warps.
    const bool owner_set = std::getenv("WEAVE_SCATTER_OWNER") != nullptr;
    const uint32_t requested_owner = static_cast<uint32_t>(named_env(
        "WEAVE_SCATTER_OWNER", {"epilogue_only", "comm_only"},
        static_cast<int>(knobs.scatter_owner)));
    const bool epilogue_owner_pinned = owner_set and requested_owner == 0u;
    const bool staging_set = std::getenv("WEAVE_SCATTER_STAGING") != nullptr;
    knobs.scatter_staging = static_cast<uint32_t>(named_env(
        "WEAVE_SCATTER_STAGING", {"reg_2stage", "smem", "hbm_dedup"},
        static_cast<int>(knobs.scatter_staging)));
    if (not staging_set and knobs.local_scatter_dedup != 0u and not epilogue_owner_pinned)
        knobs.scatter_staging = 2u;
    if (knobs.scatter_staging == 2u)
        knobs.local_scatter_dedup = 1u;
    knobs.scatter_owner = requested_owner;
    if (not owner_set and (knobs.scatter_staging != 0u or knobs.scatter_active_sms != 0u))
        knobs.scatter_owner = 1u;
    // Epilogue-owned local dedup (D7 mode 2) is only selected by an explicit
    // WEAVE_SCATTER_OWNER=epilogue_only; local dedup on a non-HBM staging with
    // a defaulted owner stays a rejected mixed request.
    if (knobs.local_scatter_dedup != 0u and knobs.scatter_owner == 0u)
        DG_HOST_ASSERT(epilogue_owner_pinned);

    // Legacy primary spellings map onto the phase-specific fields.
    const int legacy_store_m = get_weave_env<int>(
        "WEAVE_STORE_BLOCK_M", "DG_STORE_BLOCK_M", nullptr, 0);
    if (legacy_store_m > 0 and knobs.store_block_m_l1 == 0u)
        knobs.store_block_m_l1 = static_cast<uint32_t>(legacy_store_m);
    const int legacy_block_m = get_weave_env<int>(
        "WEAVE_BLOCK_M", "DG_BLOCK_M", nullptr, 0);
    if (legacy_block_m > 0 and knobs.block_m_l1 == 0u)
        knobs.block_m_l1 = static_cast<uint32_t>(legacy_block_m);

    // Four public knobs are consumed directly by the backend JIT define blocks
    // instead of the CTA-0 ABI.  Validate them here so every public WEAVE_*
    // name is fail-closed at the single host entry point.
    const int scatter_warps = get_weave_env<int>(
        "WEAVE_SCATTER_WARPS", "DG_TUNE_D7_WARPS", "DG_TUNE_D7_SCATTER_WARPS",
        get_env<int>("DG_SCATTER_WARPS", 0));
    DG_HOST_ASSERT(scatter_warps == 0 or scatter_warps == 4 or scatter_warps == 8);
    const int reg_pull_bytes = get_weave_env<int>(
        "WEAVE_REG_PULL_BYTES", "DG_TUNE_D2_REG_PULL_BYTES",
        "DG_ISOLATED_D2_REG_PULL_BYTES", 0);
    DG_HOST_ASSERT(reg_pull_bytes == 0 or reg_pull_bytes == 1024 or
                   reg_pull_bytes == 2048 or reg_pull_bytes == 4096);
    const int ready_group_size = get_weave_env<int>(
        "WEAVE_READY_GROUP_SIZE", "DG_TUNE_D1_G", "DG_ISOLATED_D1_G", 0);
    DG_HOST_ASSERT(ready_group_size >= 0);
    knobs.ready_group_size = static_cast<uint32_t>(ready_group_size);
    // Unset WEAVE_DEDUP_MODE leaves the Tier-3 dedup verdict to the live mu
    // gate (kAutoU32); an explicit 0/1 pins it.  The structural pull path is
    // still resolved by the backend dispatcher from the same env.
    const bool dedup_set = std::getenv("WEAVE_DEDUP_MODE") != nullptr;
    const int dedup_mode = get_weave_env<int>("WEAVE_DEDUP_MODE", nullptr, nullptr, 0);
    DG_HOST_ASSERT(dedup_mode == 0 or dedup_mode == 1);
    knobs.dedup_mode = dedup_set ? static_cast<uint32_t>(dedup_mode)
                                 : weave::kAutoU32;

    validate(knobs, arch);
    return knobs;
}

static inline void validate(const weave::WeaveKnobs& knobs,
                            const weave::HardwareArch arch) {
    DG_HOST_ASSERT(knobs.dispatch_warps == 0u or
                   knobs.dispatch_warps == 4u or knobs.dispatch_warps == 8u);
    DG_HOST_ASSERT(knobs.reduce_warps == 0u or knobs.reduce_warps == 1u or
                   knobs.reduce_warps == 2u or knobs.reduce_warps == 4u or
                   knobs.reduce_warps == 8u);
    DG_HOST_ASSERT(knobs.sm0_stream_sched <= 1u);
    DG_HOST_ASSERT(knobs.cu_wave_split <= 3u);
    DG_HOST_ASSERT(knobs.l2_kblock_dep <= 1u);
    DG_HOST_ASSERT(knobs.early_stream_reduce <= 1u);
    DG_HOST_ASSERT(knobs.adapt_bm <= 1u);
    DG_HOST_ASSERT(knobs.local_scatter_dedup <= 1u);
    DG_HOST_ASSERT(knobs.locality_domains == 0u or knobs.locality_domains == 1u or
                   knobs.locality_domains == 2u or knobs.locality_domains == 4u);
    if (knobs.locality_domains == 2u)
        DG_HOST_ASSERT(arch == weave::HardwareArch::B300_SM100);
    if (knobs.locality_domains == 4u)
        DG_HOST_ASSERT(arch == weave::HardwareArch::MI308X_GFX942);
    DG_HOST_ASSERT(is_candidate_block_m(knobs.block_m_l1));
    DG_HOST_ASSERT(is_candidate_block_m(knobs.block_m_l2));
    DG_HOST_ASSERT(knobs.store_block_m_l1 == 0u or knobs.store_block_m_l1 % 8u == 0u);
    DG_HOST_ASSERT(knobs.store_block_m_l2 == 0u or knobs.store_block_m_l2 % 8u == 0u);
    DG_HOST_ASSERT(knobs.tmem_pipeline_mode == 0u or
                   knobs.tmem_pipeline_mode == 100u or
                   knobs.tmem_pipeline_mode == 175u or
                   knobs.tmem_pipeline_mode == 200u);
    if (knobs.tmem_pipeline_mode == 175u)
        DG_HOST_ASSERT(arch == weave::HardwareArch::B300_SM100);
    // Strict binary scatter ownership: staged paths need a comm owner, and a
    // throttled scatter SM subset only exists when comm warps own the scatter.
    DG_HOST_ASSERT(knobs.scatter_owner <= 1u);
    DG_HOST_ASSERT(knobs.scatter_staging <= 2u);
    // The bf16x2 local dedup has exactly two executors.  Comm-owned, it runs on
    // the HBM dedup staging path (D7 mode 3); accepting reg_2stage / smem there
    // would make the host SMEM sizing disagree with the emitted kernel macro.
    // Epilogue-owned (D7 mode 2, B300/SM100 FP8xFP4 only), the epilogue merges
    // rows itself on reg_2stage and publishes them from the last contributor,
    // which the fine-grained D6 combine cannot track: the kernel static-asserts
    // mode 2 against any D6 group (COMBINE_SYNC_CHUNK, EARLY_STREAM_REDUCE or
    // spatial REDUCE_WARPS), so reject those mixes on the host instead.
    if (knobs.local_scatter_dedup != 0u and knobs.scatter_owner != 0u)
        DG_HOST_ASSERT(knobs.scatter_staging == 2u);
    if (knobs.local_scatter_dedup != 0u and knobs.scatter_owner == 0u) {
        DG_HOST_ASSERT(arch == weave::HardwareArch::B300_SM100);
        DG_HOST_ASSERT(knobs.scatter_staging == 0u);
        DG_HOST_ASSERT(knobs.combine_sync_chunk == 0u and knobs.early_stream_reduce == 0u and
                       knobs.reduce_warps == 0u);
    }
    if (knobs.scatter_staging != 0u)
        DG_HOST_ASSERT(knobs.scatter_owner == 1u);
    if (knobs.scatter_active_sms != 0u)
        DG_HOST_ASSERT(knobs.scatter_owner == 1u);
    DG_HOST_ASSERT(knobs.combine_sync_chunk == 0u or knobs.combine_sync_chunk == 16u or
                   knobs.combine_sync_chunk == 32u or knobs.combine_sync_chunk == 64u or
                   knobs.combine_sync_chunk == 128u or knobs.combine_sync_chunk == 896u);
    if (knobs.combine_sync_chunk != 0u and arch != weave::HardwareArch::B300_SM100)
        DG_HOST_ASSERT(knobs.local_scatter_dedup == 1u);
    DG_HOST_ASSERT(knobs.epi_reg_stages <= 2u);
    DG_HOST_ASSERT(knobs.sched_l2_ratio >= 1u and knobs.sched_l2_ratio <= 4u);
    DG_HOST_ASSERT(knobs.reg_balance <= 1u);
}

// WEAVE_PRECISION_MODE (then the legacy WEAVE_KNOB_PRECISION_MODE) with the
// spellings of deep_gemm/mega/precision.py, case-insensitive and blank-trimmed:
//   0/fp8xfp4, 1/fp4xfp4/nvfp4, 2/fp8xfp8/fp8, 3/bf16/bf16xbf16.
// Unset, empty or "auto" returns `fallback`; any other value is rejected.  The
// env value never overrides the precision of the backend being launched: it is
// the Python dispatcher's selector, and an explicit mma_type wins over it.
static inline int precision_mode_env(const int fallback) {
    const char* raw = std::getenv("WEAVE_PRECISION_MODE");
    if (raw == nullptr)
        raw = std::getenv("WEAVE_KNOB_PRECISION_MODE");
    if (raw == nullptr)
        return fallback;
    std::string value(raw);
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return fallback;
    value = value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
    for (auto& c : value)
        if (c >= 'A' and c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    if (value == "auto")
        return fallback;
    struct Spelling { const char* name; int code; };
    static constexpr Spelling kSpellings[] = {
        {"0", 0}, {"fp8xfp4", 0},
        {"1", 1}, {"fp4xfp4", 1}, {"nvfp4", 1},
        {"2", 2}, {"fp8xfp8", 2}, {"fp8", 2},
        {"3", 3}, {"bf16", 3}, {"bf16xbf16", 3},
    };
    for (const auto& [name, code] : kSpellings)
        if (value == name)
            return code;
    DG_HOST_UNREACHABLE(std::string("WEAVE_PRECISION_MODE='") + raw +
                        "' is invalid; expected 0/fp8xfp4, 1/fp4xfp4 (nvfp4), "
                        "2/fp8xfp8 (fp8), 3/bf16 (bf16xbf16) or auto");
}

static inline uint32_t resolve_tmem_pipeline_mode(
        const weave::WeaveKnobs& knobs, const uint32_t block_m) {
    if (knobs.tmem_pipeline_mode != 0u)
        return knobs.tmem_pipeline_mode;
    // Auto: 2.0 stages below the 256-row tile, 1.75 overlapping at 256.
    return block_m >= 256u ? 175u : 200u;
}

static inline const char* tmem_pipeline_mode_name(const uint32_t mode) {
    return mode == 100u ? "1.0" : (mode == 175u ? "1.75" : "2.0");
}

}  // namespace deep_gemm::weave_knobs
