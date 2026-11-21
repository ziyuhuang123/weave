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

static inline void validate(const weave::WeaveKnobs& knobs,
                            const weave::HardwareArch arch);

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
        static_cast<int>(knobs.combine_sync_chunk), 0, 128));
    knobs.epi_reg_stages = static_cast<uint32_t>(bounded_env(
        "WEAVE_EPI_REG_STAGES", nullptr,
        static_cast<int>(knobs.epi_reg_stages), 0, 2));
    knobs.sched_l2_ratio = static_cast<uint32_t>(bounded_env(
        "WEAVE_SCHED_L2_RATIO", nullptr,
        static_cast<int>(knobs.sched_l2_ratio), 1, 4));

    // Implementation
    knobs.precision_mode = static_cast<uint32_t>(named_env(
        "WEAVE_PRECISION_MODE", {"fp8xfp4", "fp4xfp4", "fp8xfp8", "bf16"},
        static_cast<int>(knobs.precision_mode)));
    knobs.adapt_bm = static_cast<uint32_t>(bounded_env(
        "WEAVE_ADAPT_BM", "DG_ADAPT_BM",
        static_cast<int>(knobs.adapt_bm), 0, 1));
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
    knobs.local_scatter_dedup = static_cast<uint32_t>(bounded_env(
        "WEAVE_LOCAL_SCATTER_DEDUP", "DG_LOCAL_SCATTER_DEDUP",
        static_cast<int>(knobs.local_scatter_dedup), 0, 1));
    knobs.reg_balance = static_cast<uint32_t>(bounded_env(
        "WEAVE_REG_BALANCE", nullptr,
        static_cast<int>(knobs.reg_balance), 0, 1));

    // Scatter ownership and staging.  Local dedup without an explicit staging
    // request selects the HBM dedup path; HBM dedup always enables the bf16x2
    // dedup; and any staged path is owned by the communication warps.
    const bool staging_set = std::getenv("WEAVE_SCATTER_STAGING") != nullptr;
    knobs.scatter_staging = static_cast<uint32_t>(named_env(
        "WEAVE_SCATTER_STAGING", {"reg_2stage", "smem", "hbm_dedup"},
        static_cast<int>(knobs.scatter_staging)));
    if (not staging_set and knobs.local_scatter_dedup != 0u)
        knobs.scatter_staging = 2u;
    if (knobs.scatter_staging == 2u)
        knobs.local_scatter_dedup = 1u;
    const bool owner_set = std::getenv("WEAVE_SCATTER_OWNER") != nullptr;
    knobs.scatter_owner = static_cast<uint32_t>(named_env(
        "WEAVE_SCATTER_OWNER", {"epilogue_only", "comm_only"},
        static_cast<int>(knobs.scatter_owner)));
    if (not owner_set and (knobs.scatter_staging != 0u or knobs.scatter_active_sms != 0u))
        knobs.scatter_owner = 1u;

    // Legacy primary spellings map onto the phase-specific fields.
    const int legacy_store_m = get_weave_env<int>(
        "WEAVE_STORE_BLOCK_M", "DG_STORE_BLOCK_M", nullptr, 0);
    if (legacy_store_m > 0 and knobs.store_block_m_l1 == 0u)
        knobs.store_block_m_l1 = static_cast<uint32_t>(legacy_store_m);
    const int legacy_block_m = get_weave_env<int>(
        "WEAVE_BLOCK_M", "DG_BLOCK_M", nullptr, 0);
    if (legacy_block_m > 0 and knobs.block_m_l1 == 0u)
        knobs.block_m_l1 = static_cast<uint32_t>(legacy_block_m);

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
    if (knobs.scatter_staging != 0u)
        DG_HOST_ASSERT(knobs.scatter_owner == 1u);
    if (knobs.scatter_active_sms != 0u)
        DG_HOST_ASSERT(knobs.scatter_owner == 1u);
    DG_HOST_ASSERT(knobs.combine_sync_chunk == 0u or knobs.combine_sync_chunk == 16u or
                   knobs.combine_sync_chunk == 32u or knobs.combine_sync_chunk == 64u or
                   knobs.combine_sync_chunk == 128u);
    if (knobs.combine_sync_chunk != 0u and arch != weave::HardwareArch::B300_SM100)
        DG_HOST_ASSERT(knobs.local_scatter_dedup == 1u);
    DG_HOST_ASSERT(knobs.epi_reg_stages <= 2u);
    DG_HOST_ASSERT(knobs.sched_l2_ratio >= 1u and knobs.sched_l2_ratio <= 4u);
    DG_HOST_ASSERT(knobs.reg_balance <= 1u);
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
