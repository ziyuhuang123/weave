#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <iostream>
#include <string>
#include <array>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <unordered_set>
#include <vector>

#include "../../utils/exception.hpp"
#include "../../utils/system.hpp"

#include <deep_jit/utils/env.hpp>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/layout/nv_moe_workspace.cuh>
#include <deep_gemm/layout/sm90_mega_moe.cuh>
#include <deep_gemm/scheduler/weave_knobs.cuh>

#include "../../utils/math.hpp"
#include "sm90.hpp"

namespace deep_gemm {

// SM90 uses register-resident WGMMA accumulators and FP8 weights with per-128
// float scale factors; it has no TMEM, FP4, cluster MMA, or TMA multicast.

struct MegaMoESM90Config {
    // Block tiling (no STORE_BLOCK_M / SF_BLOCK_M concept on SM90)
    int block_m, block_n, block_k;

    // Pool capacity, allocated SF capacity, and per-config logical SF stride.
    int num_max_pool_tokens;
    int num_padded_sf_pool_tokens, sf_pool_stride_tokens;

    int num_experts_per_wave;

    int num_sms;

    int num_stages, smem_size;

    // Thread layout: dispatch + non-epilogue (TMA) + epilogue (math)
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;

    // Scheduler and epilogue modes selected for this phase.
    bool direct_l2_scatter, nmajor_schedule, one_warp_cleanup, swap_ab;

    friend std::ostream& operator << (std::ostream& os, const MegaMoESM90Config& config) {
        os << "MegaMoESM90Config("
           << "block_m=" << config.block_m << ", block_n=" << config.block_n << ", block_k=" << config.block_k
           << ", num_max_pool_tokens=" << config.num_max_pool_tokens
           << ", num_padded_sf_pool_tokens=" << config.num_padded_sf_pool_tokens
           << ", sf_pool_stride_tokens=" << config.sf_pool_stride_tokens
           << ", num_experts_per_wave=" << config.num_experts_per_wave
           << ", num_sms=" << config.num_sms
           << ", num_stages=" << config.num_stages << ", smem_size=" << config.smem_size
           << ", num_dispatch_threads=" << config.num_dispatch_threads
           << ", num_non_epilogue_threads=" << config.num_non_epilogue_threads
           << ", num_epilogue_threads=" << config.num_epilogue_threads
           << ", direct_l2_scatter=" << config.direct_l2_scatter
           << ", nmajor_schedule=" << config.nmajor_schedule
           << ", one_warp_cleanup=" << config.one_warp_cleanup
           << ", swap_ab=" << config.swap_ab << ")";
        return os;
    }
};

enum class Sm90MoeHardwareProfile {
    LowSm,
    HighSm
};

struct Sm90MoeHeuristicInput {
    int launch_num_sms;

    int num_ranks, num_experts, num_experts_per_rank;
    int num_max_tokens_per_rank, num_tokens, num_topk;
    int hidden, intermediate_hidden;
    int num_padded_sf_pool_tokens;
};

struct Sm90MoeNumericalConfig {
    bool bf16_scaled_accum = false;
};

struct Sm90MoeLaunchConfig {
    MegaMoESM90Config l1;
    MegaMoESM90Config l2;
    Sm90MoeNumericalConfig numerical;
    bool fused_single_kernel = false;
};

static Sm90MoeHardwareProfile classify_sm90_moe_hardware(
    const int launch_num_sms) {
    DG_HOST_ASSERT(launch_num_sms > 0);
    return launch_num_sms < 100
        ? Sm90MoeHardwareProfile::LowSm
        : Sm90MoeHardwareProfile::HighSm;
}

enum class Sm90MoeShapeFamily {
    Generic,
    Compact,
    Wide
};

struct Sm90MoeLoad {
    int64_t routed_tokens;
    int64_t local_experts;

    bool valid() const {
        return routed_tokens >= 0 and local_experts > 0;
    }

    bool less_than(const int64_t value) const {
        return routed_tokens < value * local_experts;
    }

    bool less_equal(const int64_t value) const {
        return routed_tokens <= value * local_experts;
    }

    bool greater_than(const int64_t value) const {
        return routed_tokens > value * local_experts;
    }

    bool greater_equal(const int64_t value) const {
        return routed_tokens >= value * local_experts;
    }

    bool in_open_closed(const int64_t low, const int64_t high) const {
        return greater_than(low) and less_equal(high);
    }

    bool in_closed(const int64_t low, const int64_t high) const {
        return greater_equal(low) and less_equal(high);
    }
};

static Sm90MoeLoad get_sm90_moe_load(const Sm90MoeHeuristicInput& input) {
    const Sm90MoeLoad load {
        static_cast<int64_t>(input.num_tokens) * input.num_topk,
        input.num_experts_per_rank,
    };
    DG_HOST_ASSERT(load.valid());
    return load;
}

struct AgySm90KnobSpec {
    bool enabled = false;
    std::string raw;
    uint64_t raw_hash = 0;
    std::vector<std::pair<std::string, std::string>> kv;

    bool has(const char* k) const {
        for (const auto& p : kv)
            if (p.first == k) return true;
        return false;
    }

    int get_int(const char* k, int def_val) const {
        for (const auto& p : kv) {
            if (p.first == k) {
                char* end = nullptr;
                long v = std::strtol(p.second.c_str(), &end, 10);
                DG_HOST_ASSERT(end != p.second.c_str() and *end == '\0');
                return static_cast<int>(v);
            }
        }
        return def_val;
    }

    bool legacy_scatter_role_disp() const {
        if (not enabled) return false;
        for (const auto& p : kv) {
            if (p.first == "scat_disp") {
                char* end = nullptr;
                long v = std::strtol(p.second.c_str(), &end, 10);
                DG_HOST_ASSERT(end != p.second.c_str() and *end == '\0');
                return v > 0;
            }
            if (p.first == "scatter_role") {
                if (p.second == "1" or p.second == "disp" or p.second == "reuse_dispatch_warps")
                    return true;
                if (p.second == "0" or p.second == "epi" or p.second == "epi_warps")
                    return false;
                DG_HOST_ASSERT(false and "Invalid scatter_role knob");
            }
        }
        return false;
    }

    int resolved_scatter_staging() const {
        if (not enabled) return 0;
        const bool local_dedup = get_int("local_dedup", 0) > 0;
        const bool hbm_scat = get_int("hbm_scat", get_int("hbm_scatter_queue", 0)) > 0;
        const int scatter_sms = get_int("scatter_sms", get_int("scatter_active_sms", 0));
        const bool legacy_disp = legacy_scatter_role_disp();
        const int def_staging = (local_dedup or hbm_scat or (scatter_sms > 0 and not legacy_disp))
            ? 2 : (legacy_disp ? 1 : 0);
        const int staging = get_int("scatter_staging", def_staging);
        DG_HOST_ASSERT(staging >= 0 and staging <= 2);
        // The HBM queue / local dedup switches select hbm_dedup; an explicit
        // conflicting staging would make host SMEM sizing disagree with the
        // device layout, so reject it here.
        DG_HOST_ASSERT(staging == 2 or not (local_dedup or hbm_scat));
        // A throttled scatter SM subset without the legacy dispatch role uses
        // the HBM queue on the device unless smem staging is explicit.
        DG_HOST_ASSERT(not (scatter_sms > 0 and not legacy_disp and staging == 0));
        return staging;
    }

    int resolved_scatter_owner() const {
        if (not enabled) return 0;
        const int staging = resolved_scatter_staging();
        const int scatter_sms = get_int("scatter_sms", get_int("scatter_active_sms", 0));
        const bool legacy_disp = legacy_scatter_role_disp();
        const int def_owner = (staging != 0 or legacy_disp or scatter_sms > 0) ? 1 : 0;
        const int owner = get_int("scatter_owner", def_owner);
        DG_HOST_ASSERT(owner == 0 or owner == 1);
        DG_HOST_ASSERT(staging == 0 or owner == 1);
        DG_HOST_ASSERT(owner == 0 or staging != 0);
        DG_HOST_ASSERT(scatter_sms == 0 or owner == 1);
        return owner;
    }

    bool is_scatter_role_disp() const {
        return enabled and resolved_scatter_owner() == 1;
    }
};

static AgySm90KnobSpec parse_agy_sm90_spec_string(const char* raw_str) {
    AgySm90KnobSpec spec;
    std::string combined = (raw_str != nullptr) ? std::string(raw_str) : std::string();
    auto append_env_knob = [&](const char* key, const char* val) {
        if (val == nullptr or *val == '\0' or (val[0] == '0' and val[1] == '\0'))
            return;
        const std::string needle = std::string(key) + "=";
        if (combined.find(needle) != std::string::npos)
            return;
        if (not combined.empty())
            combined += ",";
        combined += needle + std::string(val);
    };
    auto append_weave_env = [&](const char* key, const char* primary,
                                  const char* legacy1 = nullptr,
                                  const char* legacy2 = nullptr) {
        const char* val = std::getenv(primary);
        if ((val == nullptr or *val == '\0') and legacy1 != nullptr)
            val = std::getenv(legacy1);
        if ((val == nullptr or *val == '\0') and legacy2 != nullptr)
            val = std::getenv(legacy2);
        append_env_knob(key, val);
    };
    append_weave_env("meta_g", "WEAVE_READY_GROUP_SIZE", "AGY_META_GROUP_G");
    append_weave_env("tile_sync", "AGY_TILE_SYNC_G", "AGY_TILE_SYNC_G");
    append_weave_env("fine_comb_g", "AGY_FINE_COMB_G", "AGY_FINE_COMB_G");
    append_weave_env("early_comb", "AGY_EARLY_COMB", "AGY_EARLY_COMB");
    append_weave_env("comb_warps", "WEAVE_REDUCE_WARPS", "AGY_COMB_WARPS");
    append_weave_env("v_head", "WEAVE_V_HEAD");
    append_weave_env("reg_balance", "WEAVE_REG_BALANCE");
    append_weave_env("scatter_warps", "WEAVE_SCATTER_WARPS");

    // Public H20 knobs: WEAVE_* wins; archived AGY_* spellings are fallback.
    append_weave_env("mcast", "WEAVE_MCAST", "AGY_MCAST");
    append_weave_env("swizzle_tiles", "WEAVE_SWIZZLE_TILES", "AGY_SWIZZLE_TILES");
    append_weave_env("dedup_disp", "WEAVE_DEDUP_MODE", "WEAVE_DEDUP_DISPATCH", "AGY_DEDUP_DISP");
    append_weave_env("dedup_scat", "WEAVE_DEDUP_SCATTER", "AGY_DEDUP_SCAT");
    append_weave_env("reg_comb", "WEAVE_REG_COMBINE", "AGY_REG_COMB");
    append_weave_env("all_idle_comb", "WEAVE_ALL_IDLE_COMBINE", "AGY_ALL_IDLE_COMB");
    append_weave_env("dyn", "WEAVE_DYN_SCHED", "AGY_DYN_SCHED");
    append_weave_env("interleave_l1_l2", "WEAVE_INTERLEAVE_L1_L2", "AGY_INTERLEAVE_L1_L2");
    append_weave_env("scatter_role", "WEAVE_SCATTER_ROLE_DISP", "AGY_SCATTER_ROLE_DISP");
    append_weave_env("d_mask_mode", "WEAVE_HETERO_MASK_MODE", "AGY_D_MASK_MODE");
    append_weave_env("pipe_wgmma", "WEAVE_PIPE_WGMMA", "AGY_PIPE_WGMMA");
    append_weave_env("swap_pipe", "WEAVE_SWAP_PIPE", "AGY_SWAP_PIPE");
    append_weave_env("fast_pull", "WEAVE_FAST_PULL", "AGY_FAST_PULL");
    append_weave_env("adapt_bm", "WEAVE_ADAPT_BM", "AGY_ADAPT_BM");
    append_weave_env("hbm_scat", "WEAVE_HBM_SCATTER_QUEUE", "AGY_HBM_SCAT");
    append_weave_env("scatter_sms", "WEAVE_SCATTER_ACTIVE_SMS", "AGY_SCATTER_ACTIVE_SMS");
    append_weave_env("dispatch_warps", "WEAVE_DISPATCH_WARPS", "AGY_DISPATCH_WARPS");
    append_weave_env("sm0_sched", "WEAVE_SM0_STREAM_SCHED", "AGY_SM0_STREAM_SCHED");
    append_weave_env("l2_kblock_dep", "WEAVE_L2_KBLOCK_DEP", "AGY_L2_KBLOCK_DEP");
    append_weave_env("ring_blocks", "WEAVE_RING_BLOCKS", "AGY_RING_BLOCKS");
    append_weave_env("early_stream", "WEAVE_EARLY_STREAM_REDUCE", "AGY_EARLY_STREAM_REDUCE");
    append_weave_env("head_pull_chunk", "WEAVE_HEAD_PULL_CHUNK", "AGY_HEAD_PULL_CHUNK");
    append_weave_env("local_dedup", "WEAVE_LOCAL_SCATTER_DEDUP", "AGY_LOCAL_SCATTER_DEDUP");
    append_weave_env("dispatch_sms", "WEAVE_DISPATCH_ACTIVE_SMS");
    append_weave_env("comb_sync_chunk", "WEAVE_COMBINE_SYNC_CHUNK");
    append_weave_env("epi_reg_stages", "WEAVE_EPI_REG_STAGES");
    append_weave_env("l1_bm", "WEAVE_BLOCK_M_L1");
    append_weave_env("l2_bm", "WEAVE_BLOCK_M_L2");
    append_weave_env("l1_bn", "WEAVE_BLOCK_N_L1");
    append_weave_env("l2_bn", "WEAVE_BLOCK_N_L2");
    append_weave_env("l1_wgs", "WEAVE_WAGS_L1");
    append_weave_env("l2_wgs", "WEAVE_WAGS_L2");
    append_weave_env("inline_roles", "WEAVE_INLINE_ROLES");
    // Named enum knobs: the spec carries the integer code.
    auto append_named_env = [&](const char* key, const char* primary,
                                std::initializer_list<const char*> names) {
        const char* val = std::getenv(primary);
        if (val == nullptr or *val == '\0')
            return;
        int code = 0;
        for (const char* candidate : names) {
            if (std::strcmp(val, candidate) == 0) {
                const std::string code_str = std::to_string(code);
                // Code 0 is the default and is never appended (see append_env_knob).
                append_env_knob(key, code_str.c_str());
                return;
            }
            ++code;
        }
        append_env_knob(key, val);
    };
    append_named_env("scatter_owner", "WEAVE_SCATTER_OWNER", {"epilogue_only", "comm_only"});
    append_named_env("scatter_staging", "WEAVE_SCATTER_STAGING", {"reg_2stage", "smem", "hbm_dedup"});
    append_named_env("ready_queue", "WEAVE_READY_QUEUE", {"formula", "sm0_fifo"});
    if (const char* rq = std::getenv("WEAVE_READY_QUEUE");
        rq != nullptr and (std::strcmp(rq, "0") == 0 or std::strcmp(rq, "formula") == 0)) {
        append_env_knob("ready_queue_formula", "1");
    }
    if (combined.empty())
        return spec;
    spec.enabled = true;
    spec.raw = std::move(combined);
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char c : spec.raw) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ull;
    }
    spec.raw_hash = hash;
    size_t pos = 0;
    while (pos < spec.raw.size()) {
        size_t comma = spec.raw.find(',', pos);
        std::string item = comma == std::string::npos ?
            spec.raw.substr(pos) : spec.raw.substr(pos, comma - pos);
        if (not item.empty()) {
            size_t eq = item.find('=');
            DG_HOST_ASSERT(eq != std::string::npos and eq > 0 and eq + 1 < item.size());
            std::string k_name = item.substr(0, eq);
            if (k_name == "dedup")
                k_name = "dedup_disp";
            spec.kv.emplace_back(std::move(k_name), item.substr(eq + 1));
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return spec;
}

// Single 1-line switch to enable __AUTO__ regime selection out-of-the-box when
// both AGY_FORCE_OFFICIAL and AGY_SM90_CFG are unset.
#ifndef AGY_ENABLE_AUTO_REGIME
#define AGY_ENABLE_AUTO_REGIME 0
#endif

struct AgyAutoRegimeSelector {
    static constexpr bool kEnableAutoDefault = (AGY_ENABLE_AUTO_REGIME != 0);

    // R0: Ultra-sparse decode (routed_tokens <= local_experts or num_tokens <= 4)
    static constexpr const char* kDecodeHighDup =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=16,bn=128,bk=128,wgs=2,swap=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,dedup_scat=1,reg_comb=1";
    static constexpr const char* kDecodeLowDup =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=16,bn=128,bk=128,wgs=2,swap=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,reg_comb=1";

    // R1: Small-batch / short-prefill (local_experts < routed_tokens <= 8 * local_experts)
    static constexpr const char* kSmallCompactHighDup =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=16,bn=128,bk=128,wgs=2,swap=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,dedup_scat=1,reg_comb=1";
    static constexpr const char* kSmallCompactLowDup =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=16,bn=128,bk=128,wgs=2,swap=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,reg_comb=1";
    static constexpr const char* kSmallWideHighDup =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=16,bn=256,bk=128,wgs=4,swap=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,dedup_scat=1,reg_comb=1";
    static constexpr const char* kSmallWideLowDup =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=32,bn=256,bk=128,wgs=4,swap=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,reg_comb=1";

    // R2: Medium prefill / crossover (8 * local_experts < routed_tokens <= 64 * local_experts)
    static constexpr const char* kMidShallowK =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=64,bn=256,bk=128,wgs=2,disp_warps=6,bf16_acc=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,dedup_scat=1,reg_comb=1";
    static constexpr const char* kMidSwapBm32HighDup =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=32,bn=256,bk=128,wgs=4,swap=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,dedup_scat=1,reg_comb=1";
    static constexpr const char* kMidSwapBm32LowDup =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=32,bn=256,bk=128,wgs=4,swap=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,reg_comb=1";

    // R3: Compute-bound / heavy prefill (routed_tokens > 64 * local_experts)
    static constexpr const char* kLargeShallowKBm128 =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=128,bn=128,bk=128,wgs=2,disp_warps=6,bf16_acc=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,dedup_scat=1,reg_comb=1";
    static constexpr const char* kLargeFusedBn256HighDup =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=64,bn=256,bk=128,wgs=2,disp_warps=6,bf16_acc=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,dedup_scat=1,reg_comb=1";
    static constexpr const char* kLargeFusedBn256LowDup =
        "fast_host=1,fused=1,compact=1,dyn=1,alias=1,bm=64,bn=256,bk=128,wgs=2,disp_warps=6,bf16_acc=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,reg_comb=1";
    static constexpr const char* kLargeSplitL1Bn512 =
        "fast_host=1,dyn=1,alias=1,l2_nodisp=1,bm=64,l1_bn=512,l1_wgs=4,l2_bn=256,l2_wgs=2,bk=128,disp_warps=4,bf16_acc=1,direct_l2=1,l2_nmajor=1,cleanup=1,stages=0,pipe_wgmma=1,fast_pull=1,dedup_disp=1,reg_comb=1";

    // Conservative fallback when BN=256 divisibility is unavailable
    static constexpr const char* kFallbackBn128 =
        "fast_host=1,fused=1,alias=1,bm=64,bn=128,bk=128,wgs=1,disp_warps=4,stages=0,pipe_wgmma=1,fast_pull=1,reg_comb=1";
};

static AgySm90KnobSpec make_agy_auto_sm90_spec(
    const Sm90MoeHeuristicInput& input,
    const bool force_enable = false) {
    if (not AgyAutoRegimeSelector::kEnableAutoDefault and not force_enable)
        return {};
    if (input.launch_num_sms < 2 or input.num_ranks < 2 or
        input.num_experts_per_rank <= 0 or
        input.num_topk <= 0 or input.num_topk > 32 or
        input.hidden <= 0 or input.hidden > 8192 or input.hidden % 256 != 0 or
        input.intermediate_hidden <= 0 or input.intermediate_hidden % 128 != 0 or
        input.num_experts <= 0 or input.num_experts > 2048)
        return {};

    const bool can_bn256 =
        ((2 * input.intermediate_hidden) % 256 == 0) and (input.hidden % 256 == 0);
    if (not can_bn256)
        return parse_agy_sm90_spec_string(AgyAutoRegimeSelector::kFallbackBn128);

    const int64_t routed_tokens =
        static_cast<int64_t>(input.num_tokens) * input.num_topk;
    const int64_t local_experts =
        static_cast<int64_t>(input.num_experts_per_rank);
    const bool high_dup_topk = (input.num_topk * 2) > input.num_ranks;
    const bool shallow_k = input.intermediate_hidden <= 512;
    const bool wide_expert = input.intermediate_hidden >= 2048;

    const char* selected_cfg = nullptr;
    if (input.num_tokens <= 4 or routed_tokens <= local_experts) {
        selected_cfg = high_dup_topk
            ? AgyAutoRegimeSelector::kDecodeHighDup
            : AgyAutoRegimeSelector::kDecodeLowDup;
    } else if (routed_tokens <= 8 * local_experts) {
        if (not wide_expert) {
            selected_cfg = high_dup_topk
                ? AgyAutoRegimeSelector::kSmallCompactHighDup
                : AgyAutoRegimeSelector::kSmallCompactLowDup;
        } else {
            selected_cfg = high_dup_topk
                ? AgyAutoRegimeSelector::kSmallWideHighDup
                : AgyAutoRegimeSelector::kSmallWideLowDup;
        }
    } else if (routed_tokens <= 64 * local_experts) {
        if (shallow_k) {
            selected_cfg = AgyAutoRegimeSelector::kMidShallowK;
        } else {
            selected_cfg = high_dup_topk
                ? AgyAutoRegimeSelector::kMidSwapBm32HighDup
                : AgyAutoRegimeSelector::kMidSwapBm32LowDup;
        }
    } else {
        if (shallow_k) {
            selected_cfg = AgyAutoRegimeSelector::kLargeShallowKBm128;
        } else {
            selected_cfg = high_dup_topk
                ? AgyAutoRegimeSelector::kLargeFusedBn256HighDup
                : AgyAutoRegimeSelector::kLargeFusedBn256LowDup;
        }
    }
    return parse_agy_sm90_spec_string(selected_cfg);
}

static bool has_any_public_weave_sm90_env() {
    for (const char* name : {
            "WEAVE_DISPATCH_ACTIVE_SMS", "WEAVE_SCATTER_ACTIVE_SMS",
            "WEAVE_SM0_STREAM_SCHED", "WEAVE_DISPATCH_WARPS",
            "WEAVE_SCATTER_WARPS", "WEAVE_REDUCE_WARPS", "WEAVE_REG_BALANCE",
            "WEAVE_V_HEAD", "WEAVE_READY_GROUP_SIZE", "WEAVE_READY_QUEUE",
            "WEAVE_SCATTER_OWNER", "WEAVE_COMBINE_SYNC_CHUNK",
            "WEAVE_ADAPT_BM", "WEAVE_BLOCK_M_L1", "WEAVE_BLOCK_M_L2",
            "WEAVE_DEDUP_MODE", "WEAVE_SCATTER_STAGING", "WEAVE_LOCAL_SCATTER_DEDUP"}) {
        const char* v = std::getenv(name);
        if (v != nullptr and *v != '\0')
            return true;
    }
    return false;
}

static AgySm90KnobSpec get_agy_sm90_spec(const Sm90MoeHeuristicInput* input = nullptr) {
    const char* force_off = std::getenv("WEAVE_FORCE_OFFICIAL");
    if (force_off == nullptr or *force_off == '\0')
        force_off = std::getenv("AGY_FORCE_OFFICIAL");
    if (force_off != nullptr and *force_off != '\0' and *force_off != '0')
        return {};
    const char* env_val = std::getenv("WEAVE_H20_CONFIG");
    if (env_val == nullptr or *env_val == '\0')
        env_val = std::getenv("AGY_SM90_CFG");
    if (env_val != nullptr and *env_val != '\0') {
        if (std::strcmp(env_val, "auto=1") == 0 or std::strcmp(env_val, "AUTO") == 0)
            return input != nullptr ? make_agy_auto_sm90_spec(*input, true) : AgySm90KnobSpec{};
        return parse_agy_sm90_spec_string(env_val);
    }
    if (has_any_public_weave_sm90_env()) {
        // Pure unified WEAVE_* tuning path: start from the neutral fused SM90
        // MegaMoE skeleton (no forced dedup/scatter/bm opinion) and apply the
        // explicit WEAVE_* environment variables directly.
        static constexpr const char* kUnifiedWeaveSm90Skeleton =
            "fast_host=1,fused=1,interleave=5,epw=8,compact=1,dyn=1,alias=1,"
            "stages=0,swap_pipe=1,fast_swap_epi=1,rank_cache=1,fast_clean=1,"
            "fast_pull=1,dedup_tier=1,reg_comb=1,fast_comb=1,all_idle_comb=1";
        return parse_agy_sm90_spec_string(kUnifiedWeaveSm90Skeleton);
    }
    if (input != nullptr)
        return make_agy_auto_sm90_spec(*input, false);
    return {};
}

static AgySm90KnobSpec get_agy_sm90_spec(const Sm90MoeHeuristicInput& input) {
    return get_agy_sm90_spec(&input);
}

static std::string get_agy_sm90_tag_suffix(const AgySm90KnobSpec& spec) {
    if (not spec.enabled) return "";
    return std::format("_agy_{:016x}", spec.raw_hash);
}

static std::string get_agy_sm90_tag_suffix() {
    return get_agy_sm90_tag_suffix(get_agy_sm90_spec());
}

static std::string get_agy_sm90_jit_defines(const AgySm90KnobSpec& spec) {
    if (not spec.enabled) return "";
    const bool is_fused = spec.get_int("fused", 0) > 0;
    std::string defs;
    if (spec.get_int("alias", 0) > 0)
        defs += "#define AGY_ALIAS_DISPATCH_SMEM 1\n";
    if (spec.get_int("l2_nodisp", 0) > 0 and not is_fused)
        defs += "#define AGY_L2_OMIT_DISPATCH_SMEM 1\n";
    if (spec.get_int("full_nswap", 0) > 0)
        defs += "#define AGY_FULL_NSWAP 1\n";
    const bool hbm_scat = spec.get_int("hbm_scat", spec.get_int("hbm_scatter_queue", 0)) > 0;
    const int scatter_sms = spec.get_int(
        "scatter_sms", spec.get_int("scatter_active_sms", 0));
    if (scatter_sms > 0)
        defs += std::format("#define WEAVE_SCATTER_ACTIVE_SMS {}\n", scatter_sms);
    const int dispatch_warps = spec.get_int("dispatch_warps", 0);
    if (dispatch_warps > 0) {
        DG_HOST_ASSERT(dispatch_warps == 4 or dispatch_warps == 8);
        defs += std::format("#define WEAVE_DISPATCH_WARPS {}\n", dispatch_warps);
    }
    if (spec.get_int("sm0_sched", 0) > 0 or spec.get_int("ready_queue", 0) > 0)
        defs += "#define WEAVE_SM0_STREAM_SCHED 1\n";
    const int v_head = spec.get_int("v_head", 0);
    if (v_head > 0)
        defs += std::format("#define WEAVE_V_HEAD {}u\n", v_head);
    const int l2_kblock_dep = spec.get_int("l2_kblock_dep", 0);
    if (l2_kblock_dep > 0)
        defs += "#define WEAVE_L2_KBLOCK_DEP 1\n";
    const int ring_blocks = spec.get_int("ring_blocks", 0);
    if (ring_blocks > 0)
        defs += std::format("#define WEAVE_RING_BLOCKS {}\n", ring_blocks);
    if (spec.get_int("early_stream", 0) > 0)
        defs += "#define WEAVE_EARLY_STREAM_REDUCE 1\n";
    const int head_pull_chunk = spec.get_int("head_pull_chunk", 0);
    if (head_pull_chunk > 0)
        defs += std::format("#define WEAVE_HEAD_PULL_CHUNK {}\n", head_pull_chunk);
    const bool local_dedup = spec.get_int("local_dedup", 0) > 0;
    const int staging = spec.resolved_scatter_staging();
    const int owner = spec.resolved_scatter_owner();
    const bool use_hbm_dedup = local_dedup or staging == 2;
    if (use_hbm_dedup)
        defs += "#define WEAVE_LOCAL_SCATTER_DEDUP 1\n";
    if (staging == 2)
        defs += "#define WEAVE_HBM_SCATTER_QUEUE 1\n";
    (void)hbm_scat;
    defs += std::format("#define WEAVE_SCATTER_OWNER {}\n", owner);
    defs += std::format("#define WEAVE_SCATTER_STAGING {}\n", staging);
    const int comb_sync_chunk = spec.get_int("comb_sync_chunk", 0);
    DG_HOST_ASSERT(comb_sync_chunk == 0 or comb_sync_chunk == 16 or comb_sync_chunk == 32 or
                   comb_sync_chunk == 64 or comb_sync_chunk == 128);
    if (comb_sync_chunk > 0 and use_hbm_dedup)
        defs += std::format("#define WEAVE_COMBINE_SYNC_CHUNK {}\n", comb_sync_chunk);
    const int dispatch_sms = spec.get_int("dispatch_sms", 0);
    DG_HOST_ASSERT(dispatch_sms >= 0);
    if (dispatch_sms > 0)
        defs += std::format("#define WEAVE_DISPATCH_ACTIVE_SMS {}\n", dispatch_sms);
    const int epi_reg_stages = spec.get_int("epi_reg_stages", 0);
    DG_HOST_ASSERT(epi_reg_stages >= 0 and epi_reg_stages <= 2);
    if (epi_reg_stages > 0)
        defs += std::format("#define WEAVE_EPI_REG_STAGES {}\n", epi_reg_stages);
    if (spec.has("inline_roles"))
        defs += std::format("#define WEAVE_INLINE_ROLES {}\n", spec.get_int("inline_roles", 1));
    if (spec.get_int("adapt_bm", 0) > 0) {
        defs += "#define WEAVE_ADAPT_BM 1\n";
        if (spec.get_int("full_nswap", 0) == 0)
            defs += "#define AGY_FULL_NSWAP 1\n";
    }
    if (is_fused)
        defs += "#define AGY_FUSED_1KERNEL 1\n";
    if (spec.get_int("compact", 0) > 0)
        defs += "#define AGY_COMPACT_NOTIFY 1\n";
    const bool force_formula = spec.get_int("ready_queue_formula", 0) > 0;
    if ((not force_formula) and
        (spec.get_int("dyn", 0) > 0 or spec.get_int("ready_queue", 0) > 0 or spec.get_int("adapt_bm", 0) > 0))
        defs += "#define WEAVE_DYN_SCHED 1\n";
    if (spec.get_int("pipe_wgmma", 0) > 0)
        defs += "#define WEAVE_PIPE_WGMMA 1\n";
    if (spec.get_int("swap_pipe", 0) > 0)
        defs += "#define WEAVE_SWAP_PIPE 1\n";
    if (spec.get_int("fast_pull", 0) > 0)
        defs += "#define WEAVE_FAST_PULL 1\n";
    if (spec.get_int("dedup_disp", 0) > 0)
        defs += "#define WEAVE_DEDUP_DISPATCH 1\n";
    if (spec.get_int("dedup_scat", 0) > 0 or use_hbm_dedup or scatter_sms > 0)
        defs += "#define WEAVE_DEDUP_SCATTER 1\n";
    if (spec.get_int("fast_comb", 0) > 0)
        defs += "#define AGY_FAST_COMB 1\n";
    const int reg_comb = spec.get_int("reg_comb", 0);
    if (reg_comb > 0)
        defs += std::format("#define WEAVE_REG_COMBINE {}\n", reg_comb);
    if (spec.get_int("fast_swap_epi", 0) > 0 or spec.get_int("swap_epi", 0) > 0)
        defs += "#define AGY_FAST_SWAP_EPI 1\n";
    if (spec.get_int("fast_swap_l2", 0) > 0)
        defs += "#define AGY_FAST_SWAP_L2 1\n";
    const int swizzle_val = spec.get_int("swizzle", spec.get_int("swizzle_tiles", 0));
    const int group_m_val = spec.get_int(
        "group_m", swizzle_val > 1 ? swizzle_val : (swizzle_val > 0 ? 4 : 0));
    const int l1_group_m = spec.get_int("l1_group_m", group_m_val);
    const int l2_group_m = spec.get_int("l2_group_m", group_m_val);
    if (swizzle_val > 0 or l1_group_m > 0 or l2_group_m > 0) {
        defs += "#define WEAVE_SWIZZLE_TILES 1\n";
        defs += std::format("#define AGY_GROUP_M {}\n", group_m_val > 0 ? group_m_val : 4);
        defs += std::format("#define AGY_L1_GROUP_M {}\n", l1_group_m);
        defs += std::format("#define AGY_L2_GROUP_M {}\n", l2_group_m);
    }
    if (spec.get_int("rank_cache", 0) > 0 or spec.get_int("cache_rank_counts", 0) > 0)
        defs += "#define AGY_CACHE_RANK_COUNTS 1\n";
    if (spec.get_int("fast_clean", 0) > 0)
        defs += "#define AGY_FAST_CLEAN 1\n";
    if (spec.get_int("static_stride", 0) > 0)
        defs += "#define AGY_STATIC_STRIDE 1\n";
    const int mcast_val = spec.get_int("mcast", 1);
    if (mcast_val == 2 and is_fused) {
        defs += "#define WEAVE_MCAST 2\n";
        if (spec.get_int("dyn", 0) == 0 and spec.get_int("adapt_bm", 0) == 0)
            defs += "#define WEAVE_DYN_SCHED 1\n";
    }

    // MD P1: meta_g + tile_sync
    const int meta_g = spec.get_int("meta_g", 0);
    if (meta_g > 0) {
        DG_HOST_ASSERT(meta_g == 1 or meta_g == 2 or meta_g == 4 or meta_g == 8 or meta_g == 16);
        defs += std::format("#define AGY_META_GROUP_G {}\n", meta_g);
    }
    const int tile_sync = spec.get_int("tile_sync", 0);
    if (tile_sync > 0) {
        DG_HOST_ASSERT(tile_sync == 1 or tile_sync == 2 or tile_sync == 4 or tile_sync == 8 or tile_sync == 16);
        defs += std::format("#define AGY_TILE_SYNC_G {}\n", tile_sync);
    }

    // MD P2: d_mask_mode + register reconfig
    int d_mask_mode = spec.get_int("d_mask_mode", 0);
    if (d_mask_mode == 0 and spec.get_int("md_p2_auto", 0) > 0 and not spec.has("d_mask_mode"))
        d_mask_mode = 1;
    if (d_mask_mode > 0)
        defs += std::format("#define WEAVE_HETERO_MASK_MODE {}\n", d_mask_mode);
    const int reg_bal = spec.get_int("reg_balance", 0);
    const int front_regs = spec.get_int("front_regs", reg_bal > 0 ? 40 : 0);
    if (front_regs > 0)
        defs += std::format("#define AGY_FRONT_REGS {}\n", front_regs);
    const int non_epi_regs = spec.get_int("non_epi_regs", reg_bal > 0 ? 40 : 0);
    if (non_epi_regs > 0)
        defs += std::format("#define AGY_NON_EPI_REGS {}\n", non_epi_regs);
    const int math_regs = spec.get_int("math_regs", 0);
    if (math_regs > 0)
        defs += std::format("#define AGY_MATH_REGS {}\n", math_regs);

    // MD P4: hot_cold + swizzle_l1/l2
    const int hc_base = spec.get_int("hot_cold", 0);
    const int hc_l1 = spec.get_int("l1_hot_cold", hc_base);
    const int hc_l2 = is_fused ? hc_l1 : spec.get_int("l2_hot_cold", hc_base);
    DG_HOST_ASSERT(hc_l1 >= 0 and hc_l1 <= 2 and hc_l2 >= 0 and hc_l2 <= 2);
    if (hc_l1 > 0)
        defs += std::format("#define WEAVE_HOT_COLD_MODE {}\n", hc_l1);
    if (hc_l2 > 0)
        defs += std::format("#define WEAVE_HOT_COLD_L2 {}\n", hc_l2);
    const int swz_base = spec.has("swizzle") ? spec.get_int("swizzle", 0) : spec.get_int("group_m", 0);
    const int swz_l1 = spec.get_int("l1_swizzle", swz_base);
    const int swz_l2 = spec.get_int("l2_swizzle", swz_base);
    if (swz_l1 > 1)
        defs += std::format("#define AGY_SWIZZLE_L1 {}\n", swz_l1);
    if (swz_l2 > 1)
        defs += std::format("#define AGY_SWIZZLE_L2 {}\n", swz_l2);

    // MD P5: dedup_tier + dedup_stats
    if (spec.get_int("dedup_tier", 0) > 0)
        defs += "#define AGY_DEDUP_TIER 1\n";
    if (spec.get_int("dedup_stats", 0) > 0)
        defs += "#define AGY_DEDUP_STATS 1\n";

    // MD P6: fine_comb_g + early_comb + comb_warps + all_idle_comb
    int fine_comb_g = spec.get_int("fine_comb_g", spec.get_int("comb_g", 0));
    if (fine_comb_g == 0 and comb_sync_chunk > 0 and not use_hbm_dedup)
        fine_comb_g = comb_sync_chunk / 16;
    const int early_comb = spec.get_int("early_comb", spec.get_int("early_stream", 0));
    const int comb_warps = spec.get_int("comb_warps", 0);
    const int all_idle_comb = spec.get_int("all_idle_comb", 0);
    if (early_comb > 0 and fine_comb_g == 0)
        fine_comb_g = 2;
    if (fine_comb_g > 0) {
        DG_HOST_ASSERT(fine_comb_g == 1 or fine_comb_g == 2 or fine_comb_g == 4 or
                       fine_comb_g == 8 or fine_comb_g == 16);
        defs += std::format("#define AGY_FINE_COMB_G {}\n", fine_comb_g);
    }
    if (early_comb > 0)
        defs += "#define AGY_EARLY_COMB 1\n";
    if (comb_warps > 0)
        defs += std::format("#define AGY_COMB_WARPS {}\n", comb_warps);
    if (all_idle_comb > 0)
        defs += "#define WEAVE_ALL_IDLE_COMBINE 1\n";

    // Expert-Granularity / Expert-Per-Wave / B300 L2KBlockDependency Interleaved L1+L2 atomicAdd (1..6)
    int interleave_mode = spec.get_int("interleave", spec.get_int("interleave_l1_l2", 0));
    if (interleave_mode == 0 and spec.get_int("l2_kblock_dep", 0) > 0)
        interleave_mode = 5;
    if (interleave_mode > 0 and is_fused) {
        DG_HOST_ASSERT(interleave_mode >= 1 and interleave_mode <= 6);
        defs += std::format("#define WEAVE_INTERLEAVE_L1_L2 {}\n", interleave_mode);
        if (spec.get_int("dyn", 0) == 0 and spec.get_int("adapt_bm", 0) == 0)
            defs += "#define WEAVE_DYN_SCHED 1\n";
    }

    // MD P7: scatter_role=disp / scat_disp=1 (unlocked to run simultaneously with interleave=1..6!)
    if (spec.is_scatter_role_disp())
        defs += "#define WEAVE_SCATTER_ROLE_DISP 1\n";

    // Unified single-header kernel: NO AGY_ISO_Px multi-header split!
    return defs;
}

static std::string get_agy_sm90_jit_defines() {
    return get_agy_sm90_jit_defines(get_agy_sm90_spec());
}

static Sm90MoeShapeFamily classify_sm90_moe_shape(
    const int hidden, const int intermediate_hidden,
    const AgySm90KnobSpec* agy_opt = nullptr) {
    const auto agy = agy_opt != nullptr ? *agy_opt : get_agy_sm90_spec();
    if (agy.enabled and agy.get_int("tune", 0) > 0) {
        if (hidden >= 5120 and hidden <= 8192 and
            intermediate_hidden >= 1536 and intermediate_hidden <= 4096)
            return Sm90MoeShapeFamily::Wide;
        if (hidden >= 2048 and hidden < 5120 and
            intermediate_hidden >= 512 and intermediate_hidden < 2560)
            return Sm90MoeShapeFamily::Compact;
    }
    if (hidden >= 3072 and hidden < 5120 and
        intermediate_hidden >= 1536 and intermediate_hidden < 2560)
        return Sm90MoeShapeFamily::Compact;
    if (hidden >= 5120 and hidden <= 8192 and
        intermediate_hidden >= 2560 and intermediate_hidden <= 4096)
        return Sm90MoeShapeFamily::Wide;
    return Sm90MoeShapeFamily::Generic;
}

static int get_sm90_moe_weave_int_env(const char* primary, const char* legacy,
                                             const int default_value) {
    const int value = get_weave_env<int>(primary, legacy, nullptr, default_value);
    DG_HOST_ASSERT(value >= 0 and value <= 1000000);
    return value;
}

static int get_sm90_moe_int_env(const char* name, const int default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr or *value == '\0')
        return default_value;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    DG_HOST_ASSERT(end != value and *end == '\0' and parsed >= 0 and parsed <= 1000000);
    return static_cast<int>(parsed);
}

// Padding-waste criterion for swap-AB.  swap_ab moves the token dimension from
// WGMMA's fixed M=64 onto its flexible N dimension (granularity 8), so the only
// recoverable waste is the M padding:
//   n            = routed_tokens / local_experts
//   blocks       = ceil(n / block_m)
//   used_slots   = block_m*(blocks-1) + align_up(n - block_m*(blocks-1), 8)
//   padded_slots = block_m * blocks
//   pad_frac     = 1 - used_slots / padded_slots
// Measured on 8xH20 (H=8192, I=2048, E=512, topk=10, 11 token counts):
//   pad_frac >= 0.167 -> always a win (-2.21% .. -32.70%)
//   pad_frac == 0     -> always a loss (+0.15% .. +2.63%)
// Therefore the gate belongs on pad_frac, not on the official fixed "n <= 24".
static double get_sm90_moe_swapab_pad_frac(const Sm90MoeLoad& load, const int block_m) {
    if (not load.valid() or load.local_experts <= 0 or block_m <= 0)
        return 0.0;
    const double n = static_cast<double>(load.routed_tokens) /
                     static_cast<double>(load.local_experts);
    if (n <= 0.0)
        return 0.0;
    const double bm = static_cast<double>(block_m);
    const double blocks = std::ceil(n / bm);
    const double head = bm * (blocks - 1.0);
    const double tail_slots = std::min(bm, std::ceil((n - head) / 8.0) * 8.0);
    const double used = head + tail_slots;
    const double padded = bm * blocks;
    return padded > 0.0 ? 1.0 - used / padded : 0.0;
}

// pad_frac gate: only decides whether the wide tile (BN=256 + 4 epilogue WGs)
// is admissible.  Thresholds and the token floor come from the measured
// 8-point sweep documented above.
static bool get_sm90_moe_swapab_pad_ok(const Sm90MoeLoad& load, const int block_m) {
    const int pad_pct = get_sm90_moe_weave_int_env("WEAVE_SWAPAB_PAD_PCT", "DG_MOE_SWAPAB_PAD_PCT", 25);
    const int min_load = get_sm90_moe_weave_int_env("WEAVE_SWAPAB_MIN_LOAD", "DG_MOE_SWAPAB_MIN_LOAD", 8);
    const double pad_frac = get_sm90_moe_swapab_pad_frac(load, block_m);
    return load.greater_equal(min_load) and
           pad_frac * 100.0 >= static_cast<double>(pad_pct);
}

// Effective epilogue warpgroup count for the wide tile.  A request for 4 WGs
// still has to pass the pad criterion; otherwise it falls back to 2 WGs (the
// official BN=128 shape), so n<8 or low pad_frac remains bit-identical.
static int get_sm90_moe_effective_swapab_epi_wg(const Sm90MoeLoad& load, const int block_m) {
    const int req = get_sm90_moe_weave_int_env("WEAVE_SWAPAB_EPI_WG", "DG_MOE_SWAPAB_EPI_WG", 2);
    DG_HOST_ASSERT(req == 2 or req == 4);
    if (req != 4)
        return req;
    if (get_sm90_moe_weave_int_env("WEAVE_SWAPAB_AUTO", "DG_MOE_SWAPAB_AUTO", 0) == 0)
        return 4;
    return get_sm90_moe_swapab_pad_ok(load, block_m) ? 4 : 2;
}

static bool should_use_swap_ab_for_mega_moe_sm90(
    const Sm90MoeLoad& load,
    const Sm90MoeShapeFamily shape_family,
    const int block_m,
    const int num_epilogue_threads) {
    const int default_max_load =
        shape_family == Sm90MoeShapeFamily::Compact ? 24 : 16;
    // Knob: swap-AB ceiling on tokens-per-expert.  The default reproduces the
    // official DeepGEMM decision exactly.
    const int max_load =
        get_sm90_moe_weave_int_env("WEAVE_SWAPAB_MAX_LOAD", "DG_MOE_SWAPAB_MAX_LOAD", default_max_load);
    // 4-warpgroup epilogue (512 threads) also satisfies the split-N shape:
    // each warpgroup still owns 64 columns of the weight N dimension, which is
    // what WGMMA's fixed M=64 requires after the A/B swap.
    const bool decode_split_n_path =
        block_m == 64 and
        (num_epilogue_threads == 256 or num_epilogue_threads == 512);
    // AUTO mode replaces the official fixed ceiling with the padding-waste
    // criterion.  0 = off (bit-identical to official), 1 = on.
    const int swapab_auto = get_sm90_moe_weave_int_env("WEAVE_SWAPAB_AUTO", "DG_MOE_SWAPAB_AUTO", 0);
    if (swapab_auto != 0) {
        // Real measured lesson (S=16 +56.08%, S=32 +58.24%): returning false
        // here does NOT mean "fall back to official".  The official path keeps
        // swap_ab enabled for n<=default_max_load (24); disabling it degrades
        // the slot from 8 back to 64 and multiplies compute 8x.  AUTO must be
        // a superset of the official decision.  pad_frac only gates the wide
        // tile, via get_sm90_moe_effective_swapab_epi_wg.
        const bool official_ok = load.greater_than(0) and
                                 load.less_equal(default_max_load);
        const bool pad_ok = get_sm90_moe_swapab_pad_ok(load, block_m);
        return decode_split_n_path and (official_ok or pad_ok);
    }
    return decode_split_n_path and load.greater_than(0) and
           load.less_equal(max_load);
}

static int get_generic_num_experts_per_wave_for_mega_moe_sm90(
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n,
    const int& num_sms) {
    const float expected_tokens_per_expert =
        static_cast<float>(num_tokens) * num_topk / num_experts_per_rank;
    if (expected_tokens_per_expert < 1)
        return num_experts_per_rank;

    constexpr int kImbalanceFactor = 2;
    const int num_m_blocks = ceil_div(
        static_cast<int>(std::ceil(expected_tokens_per_expert)), block_m);
    const int num_n_blocks = (2 * intermediate_hidden) / block_n;
    const int num_l1_blocks_per_expert = num_m_blocks * num_n_blocks;
    const int min_num_experts_per_wave = num_l1_blocks_per_expert > 0
        ? ceil_div(kImbalanceFactor * num_sms, num_l1_blocks_per_expert) : 1;
    if (min_num_experts_per_wave >= num_experts_per_rank)
        return num_experts_per_rank;
    if (num_l1_blocks_per_expert >= num_sms)
        return min_num_experts_per_wave;

    const int max_num_experts_per_wave = std::min(
        num_experts_per_rank, min_num_experts_per_wave * 2);
    int best_num_experts_per_wave = min_num_experts_per_wave;
    float best_tail_ratio = -1.0f;
    for (int num_experts_per_wave = min_num_experts_per_wave;
         num_experts_per_wave <= max_num_experts_per_wave;
         ++num_experts_per_wave) {
        const int remainder = num_experts_per_rank % num_experts_per_wave;
        const float tail_ratio = remainder == 0
            ? 1.0f : static_cast<float>(remainder) / num_experts_per_wave;
        if (tail_ratio > best_tail_ratio) {
            best_tail_ratio = tail_ratio;
            best_num_experts_per_wave = num_experts_per_wave;
        }
    }
    return best_num_experts_per_wave;
}

static int get_num_experts_per_wave_for_mega_moe_sm90(
    const Sm90MoeLoad& load,
    const int& num_experts_per_rank, const int& num_tokens, const int& num_topk,
    const int& intermediate_hidden, const int& block_m, const int& block_n, const int& num_sms) {
    if (block_m == 64 and (load.less_than(1) or load.greater_than(4))) {
        return num_experts_per_rank;
    }
    return get_generic_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, num_tokens, num_topk,
        intermediate_hidden, block_m, block_n, num_sms);
}

static int normalize_num_experts_per_wave_for_mega_moe_sm90(
    const int num_experts_per_rank,
    const int requested_num_experts_per_wave) {
    DG_HOST_ASSERT(num_experts_per_rank > 0);
    DG_HOST_ASSERT(requested_num_experts_per_wave > 0 and
                   requested_num_experts_per_wave <= num_experts_per_rank);
    if (num_experts_per_rank % requested_num_experts_per_wave == 0)
        return requested_num_experts_per_wave;

    // The tuned schedules use exact divisors. For a generic, previously unseen
    // expert count, round upward to the first divisor so the fallback remains
    // legal without reducing the amount of work available in a wave.
    for (int candidate = requested_num_experts_per_wave + 1;
         candidate < num_experts_per_rank; ++ candidate) {
        if (num_experts_per_rank % candidate == 0)
            return candidate;
    }
    return num_experts_per_rank;
}

static std::pair<int, int> get_pipeline_config_for_mega_moe_sm90(
    const int& smem_capacity,
    const int& num_experts, const int& hidden,
    const int& block_m, const int& block_n, const int& block_k,
    const int& num_dispatch_warps, const int& num_epilogue_warps,
    const bool& direct_l2_scatter_enabled = false,
    const int& default_num_stages = 0,
    const bool& swap_ab = false,
    const bool& require_exact_default_stages = false,
    const bool& is_linear2_phase = false,
    const int& num_ranks = 8,
    const AgySm90KnobSpec* agy_opt = nullptr) {
    constexpr int kSmemAlignment = 1024;

    // Dispatch region (same as SM100, or unioned / omitted in split L2 when AGY_SM90_CFG requests)
    const int smem_expert_count_size = align(
        num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers_size = align(
        static_cast<int>(layout::Buffer(layout::Data(hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const auto agy = agy_opt != nullptr ? *agy_opt : get_agy_sm90_spec();
    const bool is_fused = agy.enabled and agy.get_int("fused", 0) > 0;
    int smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size;
    if (agy.enabled) {
        if (agy.get_int("l2_nodisp", 0) > 0 and not is_fused) {
            smem_dispatch_size = is_linear2_phase ? 0 : std::max(smem_expert_count_size, smem_send_buffers_size);
        } else if (agy.get_int("alias", 0) > 0) {
            smem_dispatch_size = std::max(smem_expert_count_size, smem_send_buffers_size);
        }
    }

    // C/D output region: max of L1 FP8 (single-buffered, BLOCK_N/2 post-SwiGLU)
    // and L2 BF16, then 1024-byte aligned (matches kernel's SMEM_CD_SIZE).
    const auto num_epilogue_warpgroups = num_epilogue_warps / 4;
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        block_m, block_n, num_epilogue_warpgroups);
    const int wg_block_m = static_cast<int>(wg_layout.block_m);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    const int smem_cd_l1 = num_epilogue_warpgroups * wg_block_m * (wg_block_n / 2);  // 1 byte/elem (FP8)
    const int interleave_pipe = (agy.enabled and is_fused) ? agy.get_int("interleave", agy.get_int("interleave_l1_l2", 0)) : 0;
    const int hc_pipe = agy.enabled ? agy.get_int("l1_hot_cold", agy.get_int("hot_cold", 0)) : 0;
    const bool use_scat_disp = agy.enabled and agy.is_scatter_role_disp();
    const bool use_hbm_scat = agy.enabled and agy.resolved_scatter_staging() == 2;
    const bool direct_l2_scatter =
        direct_l2_scatter_enabled and not use_scat_disp and not swap_ab and wg_block_n == 128;
    const bool use_fast_swap_l2 = agy.enabled and agy.get_int("fast_swap_l2", 0) > 0;
    const int l2_smem_stride_n =
        (use_fast_swap_l2 and swap_ab and wg_block_n == 64) ? (wg_block_n + 8) : wg_block_n;
    const int raw_cd_l2 = direct_l2_scatter ? 0 :
        num_epilogue_warpgroups * wg_block_m * l2_smem_stride_n *
            static_cast<int>(sizeof(nv_bfloat16));
    const bool full_l2_smem_staging = (not use_scat_disp) or (use_hbm_scat and is_fused);
    const int smem_cd_l2 = full_l2_smem_staging
        ? raw_cd_l2
        : std::max(0, raw_cd_l2 - smem_dispatch_size);
    const int smem_cd_swap_l1 = swap_ab
        ? block_m * (block_n / 2) *
              (static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(uint8_t)))
        : 0;
    const int eff_cd_l1 = (agy.enabled and not is_fused and is_linear2_phase) ? 0 : smem_cd_l1;
    const int eff_cd_l2 = (agy.enabled and not is_fused and not is_linear2_phase) ? 0 : smem_cd_l2;
    const int smem_cd = align(
        std::max(std::max(eff_cd_l1, eff_cd_l2), smem_cd_swap_l1),
        kSmemAlignment);

    // SF on SM90:
    //   * SFA per stage must hold one aligned BLOCK_M-float vector for every
    //     per-64-K L2 scale group (two for BK128, four for BK256)
    //   * SFB is loaded directly from global by the math warpgroup (block-(128,128)
    //     weight quantization), so no SMEM is reserved for it.
    const int smem_sfa_half_stride_bytes = align(block_m * static_cast<int>(sizeof(float)), 128);
    const int smem_sfa_per_stage =
        (block_k / 64) * smem_sfa_half_stride_bytes;
    // Per-stage: A tile + B tile + SFA tile. Weight SF is loaded directly.
    const int smem_per_stage = block_m * block_k + block_n * block_k +
                               smem_sfa_per_stage;

    // Barriers (8 bytes each):
    //   * dispatch: num_dispatch_warps
    //   * GEMM full + empty: 2 * num_stages
    //   * combine: 2 * num_epilogue_warps
    const bool use_dyn_sched = agy.enabled and (agy.get_int("dyn", 0) > 0 or agy.get_int("adapt_bm", 0) > 0 or agy.get_int("mcast", 1) == 2 or agy.get_int("static_stride", 0) > 0 or interleave_pipe > 0 or hc_pipe > 0);
    const bool use_rank_cache = agy.enabled and
        (agy.get_int("rank_cache", 0) > 0 or agy.get_int("cache_rank_counts", 0) > 0) and
        (not is_linear2_phase or is_fused);
    const bool use_expert_cache = agy.enabled and
        (agy.get_int("compact", 0) > 0 or use_dyn_sched or use_rank_cache);
    const bool use_dedup_disp = agy.enabled and agy.get_int("dedup_disp", 0) > 0 and
                                (not is_linear2_phase or is_fused);
    const int num_experts_per_rank = std::max(1, num_experts / std::max(1, num_ranks));
    const int expert_cache_base_bytes =
        (7 * num_experts_per_rank + 19) * static_cast<int>(sizeof(uint32_t));
    const int expert_cache_rank_bytes = use_rank_cache
        ? (num_experts_per_rank * num_ranks * static_cast<int>(sizeof(uint16_t)))
        : 0;
    const int smem_expert_cache_size = use_expert_cache
        ? align(expert_cache_base_bytes + expert_cache_rank_bytes, 128)
        : 0;
    const int smem_dedup_fifo_size = use_dedup_disp ? (num_dispatch_warps * 16 * 32) : 0;

    const bool use_meta_ring = agy.enabled and agy.get_int("meta_g", 0) > 0 and
                               (not is_linear2_phase or is_fused);
    const bool use_tsync_state = agy.enabled and is_fused and
                                 (agy.get_int("tile_sync", 0) > 0) and (not use_meta_ring);
    const bool use_scat_disp_bars = use_scat_disp and (is_linear2_phase or is_fused);
    const int smem_barriers_fixed =
        (num_dispatch_warps + 2 * num_epilogue_warps +
         (use_dyn_sched ? 16 : 0) +
         (use_meta_ring ? 14 : (use_tsync_state ? 2 : 0)) +
         (use_scat_disp_bars ? 4 : 0)) * 8;
    const int smem_barriers_per_stage = 2 * 8;

    const int smem_fixed = smem_dispatch_size + smem_cd +
                           smem_expert_cache_size + smem_dedup_fifo_size +
                           smem_barriers_fixed;

    const int max_num_stages = (smem_capacity - smem_fixed) /
                               (smem_per_stage + smem_barriers_per_stage);
    if (max_num_stages < 2)
        return {0, 0};
    const bool prefer_bn256_n_tile = block_n == 256;
    if (require_exact_default_stages and default_num_stages > max_num_stages)
        return {0, 0};
    const int preferred_num_stages = default_num_stages > 0
        ? std::min(default_num_stages, max_num_stages)
        : (prefer_bn256_n_tile ? std::min(4, max_num_stages) : 0);
    const int num_stages = preferred_num_stages > 0 ?
        preferred_num_stages : max_num_stages;
    if (num_stages < 2 or num_stages > max_num_stages)
        return {0, 0};
    return {num_stages,
            smem_fixed + num_stages * (smem_per_stage + smem_barriers_per_stage)};
}

// Mirror the device kernel's pre-barrier alias region so invalid combine
// vectorization is rejected before NVRTC sees a shape specialization.
static uint32_t get_sm90_moe_pre_barrier_smem_size_for_combine(
    const Sm90MoeHeuristicInput& input,
    const MegaMoESM90Config& config) {
    constexpr int kSmemAlignment = 1024;
    const int num_dispatch_warps = config.num_dispatch_threads / 32;
    const int num_epilogue_warpgroups = config.num_epilogue_threads / 128;
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        config.block_m, config.block_n, num_epilogue_warpgroups);

    const int smem_expert_count = align(
        input.num_experts * static_cast<int>(sizeof(uint32_t)), kSmemAlignment);
    const int smem_send_buffers = align(
        static_cast<int>(layout::Buffer(
            layout::Data(input.hidden), num_dispatch_warps, 1).get_num_bytes()),
        kSmemAlignment);
    const auto agy = get_agy_sm90_spec(input);
    const bool is_fused = agy.enabled and agy.get_int("fused", 0) > 0;
    const bool use_scat_disp = agy.enabled and agy.is_scatter_role_disp();
    const bool use_hbm_scat = agy.enabled and agy.resolved_scatter_staging() == 2;
    const bool use_fast_swap_l2 = agy.enabled and agy.get_int("fast_swap_l2", 0) > 0;
    int smem_disp = smem_expert_count + smem_send_buffers;
    if (agy.enabled) {
        if (agy.get_int("l2_nodisp", 0) > 0 and not is_fused) {
            smem_disp = 0;
        } else if (agy.get_int("alias", 0) > 0) {
            smem_disp = std::max(smem_expert_count, smem_send_buffers);
        }
    }
    const int wg_bm_c = static_cast<int>(wg_layout.block_m);
    const int wg_bn_c = static_cast<int>(wg_layout.block_n);
    const int l2_smem_stride_n =
        (use_fast_swap_l2 and config.swap_ab and wg_bn_c == 64) ? (wg_bn_c + 8) : wg_bn_c;
    const int raw_cd_l2 = (config.direct_l2_scatter and not use_scat_disp) ? 0 :
        num_epilogue_warpgroups * wg_bm_c *
        l2_smem_stride_n * static_cast<int>(sizeof(nv_bfloat16));
    const bool full_l2_smem_staging = (not use_scat_disp) or (use_hbm_scat and is_fused);
    const int smem_cd_l2 = full_l2_smem_staging ? raw_cd_l2 : std::max(0, raw_cd_l2 - smem_disp);
    const int smem_cd_swap_l1 = config.swap_ab ?
        config.block_m * (config.block_n / 2) *
            (static_cast<int>(sizeof(float)) + static_cast<int>(sizeof(uint8_t))) : 0;
    const int smem_gemm = config.num_stages *
        (config.block_m * config.block_k + config.block_n * config.block_k);
    const int smem_cd_l1_fused = is_fused
        ? (num_epilogue_warpgroups * wg_bm_c * (wg_bn_c / 2))
        : 0;
    const int smem_cd_actual = align(
        std::max(std::max(smem_cd_l1_fused, smem_cd_l2), smem_cd_swap_l1), kSmemAlignment);
    return static_cast<uint32_t>(smem_disp + smem_cd_actual + smem_gemm);
}

static bool supports_sm90_moe_bf16_scaled_accum(
    const MegaMoESM90Config& config,
    const bool agy_enabled_hint = false) {
    const int num_epilogue_warpgroups = config.num_epilogue_threads / 128;
    const bool agy_enabled = agy_enabled_hint or get_agy_sm90_spec().enabled;
    if (config.swap_ab) {
        if (agy_enabled) {
            return (config.block_m == 16 or config.block_m == 32 or config.block_m == 64) and
                   (config.block_n / num_epilogue_warpgroups == 64) and
                   (config.block_k == 128 or config.block_k == 256) and
                   (num_epilogue_warpgroups == 2 or num_epilogue_warpgroups == 4);
        }
        return config.block_m == 64 and config.block_n == 128 and
               config.block_k == 128 and num_epilogue_warpgroups == 2;
    }
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        config.block_m, config.block_n, num_epilogue_warpgroups);
    const int wg_block_m = static_cast<int>(wg_layout.block_m);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    return (agy_enabled ? (wg_block_m == 64) : (config.block_m == 64)) and
           (wg_block_n == 128 or wg_block_n == 256) and
           (config.block_k == 128 or config.block_k == 256);
}

static bool is_sm90_moe_phase_config_legal(
    const Sm90MoeHeuristicInput& input,
    const MegaMoESM90Config& config,
    const int phase_n,
    const bool require_exact_pipeline,
    const bool is_linear2_phase = false,
    const AgySm90KnobSpec* agy_opt = nullptr) {
    const auto agy = agy_opt != nullptr ? *agy_opt : get_agy_sm90_spec(input);
    const bool agy_enabled = agy.enabled;
    const bool legal_bm = (config.block_m == 64 or config.block_m == 128) or
                          (agy_enabled and config.swap_ab and (config.block_m == 32 or config.block_m == 16));
    if ((not legal_bm) or
        (config.block_n != 128 and config.block_n != 256 and config.block_n != 512) or
        (config.block_k != 128 and config.block_k != 256))
        return false;
    if (config.sf_pool_stride_tokens > input.num_padded_sf_pool_tokens)
        return false;
    if (phase_n % config.block_n != 0 or
        input.hidden % config.block_k != 0 or
        input.intermediate_hidden % config.block_k != 0)
        return false;
    if (config.num_sms <= 0 or config.num_sms > input.launch_num_sms)
        return false;
    if (config.num_experts_per_wave <= 0 or
        config.num_experts_per_wave > input.num_experts_per_rank or
        input.num_experts_per_rank % config.num_experts_per_wave != 0)
        return false;
    const bool legal_disp_threads =
        config.num_dispatch_threads == 64 or
        config.num_dispatch_threads == 128 or
        config.num_dispatch_threads == 192 or
        config.num_dispatch_threads == 256;
    if ((not legal_disp_threads) or
        ((config.num_dispatch_threads + config.num_non_epilogue_threads) % 128 != 0) or
        (config.num_non_epilogue_threads != 64 and config.num_non_epilogue_threads != 128) or
        config.num_epilogue_threads <= 0 or config.num_epilogue_threads % 128 != 0 or
        config.num_dispatch_threads + config.num_non_epilogue_threads +
            config.num_epilogue_threads > 1024)
        return false;

    const int num_epilogue_warpgroups = config.num_epilogue_threads / 128;
    const auto wg_layout = layout::get_sm90_moe_warpgroup_layout(
        config.block_m, config.block_n, num_epilogue_warpgroups);
    const int split_m = static_cast<int>(wg_layout.split_m);
    const int split_n_count = static_cast<int>(wg_layout.split_n_count);
    if (split_m <= 0 or split_n_count <= 0 or
        config.block_m % split_m != 0 or config.block_n % split_n_count != 0)
        return false;
    const int wg_block_m = static_cast<int>(wg_layout.block_m);
    const int wg_block_n = static_cast<int>(wg_layout.block_n);
    const bool legal_wg_bm = (wg_block_m == 64) or
                             (agy_enabled and config.swap_ab and (wg_block_m == 32 or wg_block_m == 16));
    if ((not legal_wg_bm) or
        (wg_block_n != 64 and wg_block_n != 128 and wg_block_n != 256))
        return false;
    // WG_BLOCK_N=64 is supported only by the swap-AB path.
    if (wg_block_n == 64 and not config.swap_ab)
        return false;
    const int epi_wg_knob = get_sm90_moe_weave_int_env("WEAVE_SWAPAB_EPI_WG", "DG_MOE_SWAPAB_EPI_WG", 2);
    const bool swap_ab_wide = config.swap_ab and (agy_enabled or epi_wg_knob == 4) and
                              config.block_n == 256 and
                              num_epilogue_warpgroups == 4;
    if (config.block_m == 64 and config.block_n == 256 and
        num_epilogue_warpgroups != 2 and not swap_ab_wide)
        return false;
    const int bk_knob = get_sm90_moe_weave_int_env("WEAVE_BLOCK_K", "DG_MOE_BLOCK_K", 128);
    const bool bk_ok = config.block_k == 128 or config.block_k == bk_knob;
    if (config.swap_ab) {
        if (agy_enabled) {
            if (not ((config.block_m == 16 or config.block_m == 32 or config.block_m == 64) and
                     wg_block_n == 64 and
                     (config.block_k == 128 or config.block_k == 256) and
                     ((config.block_n == 128 and num_epilogue_warpgroups == 2) or
                      swap_ab_wide)))
                return false;
        } else {
            if (not (config.block_m == 64 and bk_ok and
                     ((config.block_n == 128 and num_epilogue_warpgroups == 2) or
                      swap_ab_wide)))
                return false;
        }
    }
    if (config.direct_l2_scatter and (config.swap_ab or wg_block_n != 128))
        return false;
    if (config.smem_size <= 0 or config.smem_size > SM90ArchSpec::smem_capacity)
        return false;
    if (not require_exact_pipeline)
        return true;

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        input.num_experts, input.hidden,
        config.block_m, config.block_n, config.block_k,
        config.num_dispatch_threads / 32, config.num_epilogue_threads / 32,
        config.direct_l2_scatter,
        config.num_stages,
        config.swap_ab,
        true,
        is_linear2_phase,
        input.num_ranks,
        &agy);
    return num_stages == config.num_stages and smem_size == config.smem_size;
}

static bool is_sm90_moe_launch_config_legal(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeLaunchConfig& config,
    const bool require_exact_pipeline = true,
    const AgySm90KnobSpec* agy_opt = nullptr) {
    const auto agy = agy_opt != nullptr ? *agy_opt : get_agy_sm90_spec(input);
    if (config.l1.num_sms != input.launch_num_sms or
        config.l2.num_sms != input.launch_num_sms)
        return false;
    if (config.l1.block_m != config.l2.block_m)
        return false;
    if (config.fused_single_kernel) {
        if (config.l1.block_n != config.l2.block_n or
            config.l1.block_k != config.l2.block_k or
            config.l1.num_stages != config.l2.num_stages or
            config.l1.smem_size != config.l2.smem_size or
            config.l1.num_dispatch_threads != config.l2.num_dispatch_threads or
            config.l1.num_non_epilogue_threads != config.l2.num_non_epilogue_threads or
            config.l1.num_epilogue_threads != config.l2.num_epilogue_threads or
            config.l1.swap_ab != config.l2.swap_ab or
            config.l1.num_experts_per_wave != config.l2.num_experts_per_wave)
            return false;
    }
    if (not is_sm90_moe_phase_config_legal(
            input, config.l1, 2 * input.intermediate_hidden,
            require_exact_pipeline, false, &agy) or
        not is_sm90_moe_phase_config_legal(
            input, config.l2, input.hidden, require_exact_pipeline, true, &agy))
        return false;
    // The four-math-warpgroup BN512 L1 path relies on packed BF16 partial
    // accumulation to stay within the CTA register budget.
    if (config.l1.block_n == 512 and
        not config.numerical.bf16_scaled_accum)
        return false;
    return not config.numerical.bf16_scaled_accum or
           (supports_sm90_moe_bf16_scaled_accum(config.l1, agy.enabled) and
            supports_sm90_moe_bf16_scaled_accum(config.l2, agy.enabled));
}

static MegaMoESM90Config make_generic_mega_moe_config_sm90(
    const Sm90MoeHeuristicInput& input,
    const bool conservative = false,
    const AgySm90KnobSpec* agy_opt = nullptr) {
    const int block_m = 64;
    const int num_max_pool_tokens = layout::nv_moe::get_num_max_pool_tokens(
        input.num_ranks, input.num_max_tokens_per_rank,
        input.num_topk, input.num_experts_per_rank);
    const int block_k = 128;
    const auto load = get_sm90_moe_load(input);
    const auto shape_family = classify_sm90_moe_shape(
        input.hidden, input.intermediate_hidden, agy_opt);

    const bool prefer_swap_ab_block =
        not conservative and
        should_use_swap_ab_for_mega_moe_sm90(
            load, shape_family, block_m, 256);
    const int eff_epi_wg = get_sm90_moe_effective_swapab_epi_wg(load, block_m);
    // kSwapSFSubBlocks: BLOCK_N=256 selects two 64-column activation-SF
    // sub-blocks; SWAPAB_FINE_LADDER selects the 8/16/24/32/40/48/56/64
    // token ladder at both L1 and L2 dispatch points.
    const int block_n =
        (prefer_swap_ab_block and eff_epi_wg == 4) ? 256 :
        (prefer_swap_ab_block ? 128 : (conservative ? 128 : 256));
    const bool prefer_swap_ab_shape = prefer_swap_ab_block and
        (block_n == 128 or eff_epi_wg == 4);
    const int swapab_epi_wg = eff_epi_wg;
    DG_HOST_ASSERT(swapab_epi_wg == 2 or swapab_epi_wg == 4);
    const int num_epilogue_warpgroups =
        (prefer_swap_ab_block and swapab_epi_wg == 4) ? 4 :
        ((block_n == 256 or prefer_swap_ab_shape) ? 2 : 1);
    DG_HOST_ASSERT(block_m % num_epilogue_warpgroups == 0);
    DG_HOST_ASSERT(block_m != 128 or
                   (block_n == 128 and num_epilogue_warpgroups == 2));
    // The 4-warpgroup epilogue is gated by DG_MOE_SWAPAB_EPI_WG and validated
    // by the full-tensor correctness gate, not by this assert.
    DG_HOST_ASSERT(block_m != 64 or block_n != 256 or
                   num_epilogue_warpgroups == 2 or
                   (num_epilogue_warpgroups == 4 and
                    get_sm90_moe_weave_int_env("WEAVE_SWAPAB_EPI_WG", "DG_MOE_SWAPAB_EPI_WG", 2) == 4));
    const int num_epilogue_threads = num_epilogue_warpgroups * 128;
    const bool swap_ab =
        not conservative and
        (block_n == 128 or
         get_sm90_moe_weave_int_env("WEAVE_SWAPAB_EPI_WG", "DG_MOE_SWAPAB_EPI_WG", 2) == 4) and
        should_use_swap_ab_for_mega_moe_sm90(
            load, shape_family, block_m, num_epilogue_threads);

    const bool compact_frontend = block_n >= 256 or swap_ab;
    const int num_dispatch_warps = compact_frontend ? 2 : 4;
    DG_HOST_ASSERT(not compact_frontend or num_dispatch_warps == 2);
    const int num_dispatch_threads = num_dispatch_warps * 32;
    const int num_non_epilogue_threads = compact_frontend ? 64 : 128;
    DG_HOST_ASSERT((num_dispatch_threads + num_non_epilogue_threads) % 128 == 0);

    constexpr bool direct_l2_scatter = false;
    constexpr bool l2_nmajor_schedule = false;
    constexpr bool one_warp_cleanup = false;

    const int requested_num_experts_per_wave = conservative ?
        get_generic_num_experts_per_wave_for_mega_moe_sm90(
            input.num_experts_per_rank, input.num_tokens, input.num_topk,
            input.intermediate_hidden, block_m, block_n, input.launch_num_sms) :
        get_num_experts_per_wave_for_mega_moe_sm90(
            load,
            input.num_experts_per_rank, input.num_tokens, input.num_topk,
            input.intermediate_hidden, block_m, block_n, input.launch_num_sms);
    const int num_experts_per_wave =
        normalize_num_experts_per_wave_for_mega_moe_sm90(
            input.num_experts_per_rank, requested_num_experts_per_wave);
    DG_HOST_ASSERT(num_experts_per_wave > 0 and
                   num_experts_per_wave <= input.num_experts_per_rank and
                   input.num_experts_per_rank % num_experts_per_wave == 0);

    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        input.num_experts, input.hidden,
        block_m, block_n, block_k,
        num_dispatch_threads / 32, num_epilogue_threads / 32,
        direct_l2_scatter,
        0,
        swap_ab,
        false,
        false,
        input.num_ranks,
        agy_opt);
    DG_HOST_ASSERT(num_stages >= 2 and smem_size > 0);
    const int sf_pool_stride_tokens =
        layout::nv_moe::get_num_sf_ring_tokens(num_max_pool_tokens, block_m);
    return {
        block_m, block_n, block_k,
        num_max_pool_tokens, input.num_padded_sf_pool_tokens, sf_pool_stride_tokens,
        num_experts_per_wave,
        input.launch_num_sms,
        num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads,
        direct_l2_scatter, l2_nmajor_schedule, one_warp_cleanup, swap_ab
    };
}

static bool select_sm90_moe_high_sm_bf16_scaled_accum(
    const Sm90MoeShapeFamily shape_family,
    const Sm90MoeLoad& load) {
    if (shape_family == Sm90MoeShapeFamily::Compact) {
        return load.less_equal(3) or
               (load.greater_than(6) and load.less_equal(384));
    }
    if (shape_family == Sm90MoeShapeFamily::Wide) {
        return load.less_equal(4) or load.greater_than(8);
    }
    return false;
}

struct Sm90MoePhaseTuning {
    int block_m, block_n, block_k;
    int num_experts_per_wave, num_stages;
    int num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads;
    bool direct_l2_scatter, nmajor_schedule, one_warp_cleanup, swap_ab;
};

struct Sm90MoeScheduleTuning {
    bool selected;
    Sm90MoePhaseTuning l1, l2;
};

static Sm90MoePhaseTuning make_sm90_moe_phase_tuning(
    const MegaMoESM90Config& config) {
    return {
        config.block_m, config.block_n, config.block_k,
        config.num_experts_per_wave, config.num_stages,
        config.num_dispatch_threads, config.num_non_epilogue_threads,
        config.num_epilogue_threads,
        config.direct_l2_scatter, config.nmajor_schedule,
        config.one_warp_cleanup, config.swap_ab,
    };
}

static void set_sm90_moe_specialized_phase_tuning(
    Sm90MoePhaseTuning& tuning,
    const int block_n,
    const int block_k,
    const int num_experts_per_wave,
    const int num_stages) {
    tuning.block_m = 64;
    tuning.block_n = block_n;
    tuning.block_k = block_k;
    tuning.num_experts_per_wave = num_experts_per_wave;
    tuning.num_stages = num_stages;
    tuning.num_dispatch_threads = block_n >= 256 ? 64 : 128;
    tuning.num_non_epilogue_threads = block_n >= 256 ? 64 : 128;
    tuning.num_epilogue_threads = block_n >= 256 ? block_n : 128;
    tuning.direct_l2_scatter = false;
    tuning.nmajor_schedule = false;
    tuning.one_warp_cleanup = false;
    tuning.swap_ab = false;
}

static int derive_sm90_moe_tuned_epw(
    const int num_experts_per_rank,
    const int target_num_experts_per_wave) {
    const int requested = std::min(
        num_experts_per_rank, target_num_experts_per_wave);
    return normalize_num_experts_per_wave_for_mega_moe_sm90(
        num_experts_per_rank, requested);
}

static Sm90MoeScheduleTuning make_sm90_moe_schedule_tuning(
    const Sm90MoeLaunchConfig& generic) {
    return {
        false,
        make_sm90_moe_phase_tuning(generic.l1),
        make_sm90_moe_phase_tuning(generic.l2),
    };
}

static bool uses_sm90_moe_bn256_frontend(
    const Sm90MoePhaseTuning& tuning) {
    return tuning.block_m == 64 and tuning.block_n == 256;
}

static Sm90MoeScheduleTuning select_sm90_moe_low_sm_tuning(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeShapeFamily shape_family,
    const Sm90MoeLoad& load,
    Sm90MoeScheduleTuning tuning) {
    if (shape_family == Sm90MoeShapeFamily::Compact and
        load.greater_than(8) and load.less_than(16)) {
        const int tuned_epw = derive_sm90_moe_tuned_epw(
            input.num_experts_per_rank, 16);
        tuning.selected = true;
        tuning.l1.num_experts_per_wave = tuned_epw;
        tuning.l2.num_experts_per_wave = tuned_epw;
    }

    if (not uses_sm90_moe_bn256_frontend(tuning.l1) or
        not uses_sm90_moe_bn256_frontend(tuning.l2))
        return tuning;

    int target_epw = 0;
    int num_stages = 4;
    bool direct_l2_scatter = false;
    bool l2_nmajor = false;
    bool one_warp_cleanup = false;
    bool range_selected = false;

    if (shape_family == Sm90MoeShapeFamily::Compact) {
        if (load.in_open_closed(16, 32)) {
            range_selected = true;
            num_stages = 5;
            direct_l2_scatter = true;
            one_warp_cleanup = true;
        } else if (load.greater_than(48) and load.less_than(96)) {
            range_selected = true;
            direct_l2_scatter = true;
            num_stages = 4;
        } else if (load.in_closed(96, 160)) {
            range_selected = true;
            target_epw = 8;
            num_stages = 5;
            direct_l2_scatter = true;
            one_warp_cleanup = true;
        } else if (load.in_open_closed(160, 384)) {
            range_selected = true;
            target_epw = 16;
            num_stages = 5;
            direct_l2_scatter = true;
            l2_nmajor = load.greater_equal(256);
        } else if (load.greater_than(384)) {
            range_selected = true;
            target_epw = 32;
            num_stages = 5;
            direct_l2_scatter = true;
            l2_nmajor = true;
        }
    } else if (shape_family == Sm90MoeShapeFamily::Wide) {
        if (load.in_open_closed(16, 48)) {
            range_selected = true;
            target_epw = 16;
        } else if (load.in_open_closed(48, 96)) {
            range_selected = true;
            target_epw = input.num_experts_per_rank;
            num_stages = 5;
            direct_l2_scatter = true;
            one_warp_cleanup = false;
        } else if (load.in_open_closed(96, 192)) {
            range_selected = true;
            target_epw = input.num_experts_per_rank;
            num_stages = 5;
            direct_l2_scatter = true;
            one_warp_cleanup = true;
        } else if (load.in_open_closed(192, 384)) {
            range_selected = true;
            target_epw = input.num_experts_per_rank;
            direct_l2_scatter = true;
        } else if (load.greater_than(384)) {
            range_selected = true;
            target_epw = input.num_experts_per_rank;
            num_stages = 5;
            direct_l2_scatter = true;
        }
    }

    if (not range_selected)
        return tuning;

    tuning.selected = true;
    if (target_epw > 0) {
        const int tuned_epw = derive_sm90_moe_tuned_epw(
            input.num_experts_per_rank, target_epw);
        tuning.l1.num_experts_per_wave = tuned_epw;
        tuning.l2.num_experts_per_wave = tuned_epw;
    }
    tuning.l1.num_stages = tuning.l2.num_stages = num_stages;
    tuning.l1.direct_l2_scatter = tuning.l2.direct_l2_scatter =
        direct_l2_scatter;
    tuning.l1.nmajor_schedule = false;
    tuning.l2.nmajor_schedule = l2_nmajor;
    tuning.l1.one_warp_cleanup = tuning.l2.one_warp_cleanup =
        one_warp_cleanup;
    return tuning;
}

static Sm90MoeScheduleTuning select_sm90_moe_high_sm_tuning(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeShapeFamily shape_family,
    const Sm90MoeLoad& load,
    Sm90MoeScheduleTuning tuning) {
    int l1_block_n = 256, l1_block_k = 128, l1_epw = 0, l1_stages = 0;
    int l2_block_n = 256, l2_block_k = 128, l2_epw = 0, l2_stages = 0;
    bool l1_nmajor = false, one_warp_cleanup = false;
    bool specialized = false;

    if (shape_family == Sm90MoeShapeFamily::Compact) {
        if (load.in_open_closed(12, 32)) {
            specialized = true;
            l1_epw = l2_epw = 4;
            l1_stages = l2_stages = 3;
        } else if (load.in_open_closed(128, 256)) {
            specialized = true;
            l1_epw = l2_epw = 32;
            l1_stages = l2_stages = 4;
        } else if (load.greater_than(1024)) {
            specialized = true;
            l1_epw = l2_epw = 32;
            l1_stages = l2_stages = 4;
        }
    } else if (shape_family == Sm90MoeShapeFamily::Wide) {
        if (load.in_open_closed(8, 24)) {
            specialized = true;
            l1_block_n = 512;
            l1_epw = l2_epw = 16;
            l1_stages = 2;
            l2_stages = 3;
        } else if (load.in_open_closed(24, 48)) {
            specialized = true;
            l1_block_k = 256;
            l1_epw = 8;
            l2_epw = 48;
            l1_stages = 2;
            l2_stages = 3;
        } else if (load.in_open_closed(48, 96)) {
            specialized = true;
            l1_block_n = 512;
            l1_epw = l2_epw = 16;
            l1_stages = 2;
            l2_stages = 3;
            l1_nmajor = true;
        } else if (load.in_open_closed(96, 192)) {
            specialized = true;
            l1_block_n = 512;
            l1_epw = l2_epw = 16;
            l1_stages = 2;
            l2_stages = 3;
            l1_nmajor = true;
            one_warp_cleanup = true;
        } else if (load.greater_than(192)) {
            specialized = true;
            l1_block_n = 512;
            l1_epw = l2_epw = 16;
            l1_stages = 2;
            l2_stages = 3;
            one_warp_cleanup = true;
        }
    }
    if (not specialized)
        return tuning;

    l1_epw = derive_sm90_moe_tuned_epw(
        input.num_experts_per_rank, l1_epw);
    l2_epw = derive_sm90_moe_tuned_epw(
        input.num_experts_per_rank, l2_epw);
    tuning.selected = true;
    set_sm90_moe_specialized_phase_tuning(
        tuning.l1, l1_block_n, l1_block_k, l1_epw, l1_stages);
    set_sm90_moe_specialized_phase_tuning(
        tuning.l2, l2_block_n, l2_block_k, l2_epw, l2_stages);
    tuning.l1.nmajor_schedule = l1_nmajor;
    tuning.l2.nmajor_schedule = true;
    tuning.l2.one_warp_cleanup = one_warp_cleanup;
    return tuning;
}

static Sm90MoeScheduleTuning select_sm90_moe_tuning(
    const Sm90MoeHeuristicInput& input,
    const Sm90MoeHardwareProfile hardware_profile,
    const Sm90MoeShapeFamily shape_family,
    const Sm90MoeLoad& load,
    const Sm90MoeLaunchConfig& generic) {
    auto tuning = make_sm90_moe_schedule_tuning(generic);
    if (hardware_profile == Sm90MoeHardwareProfile::LowSm) {
        return select_sm90_moe_low_sm_tuning(
            input, shape_family, load, tuning);
    }
    if (hardware_profile == Sm90MoeHardwareProfile::HighSm) {
        return select_sm90_moe_high_sm_tuning(
            input, shape_family, load, tuning);
    }
    return tuning;
}

static bool try_materialize_sm90_moe_phase_tuning(
    const Sm90MoeHeuristicInput& input,
    MegaMoESM90Config& config,
    const Sm90MoePhaseTuning& tuning,
    const bool is_linear2_phase = false,
    const AgySm90KnobSpec* agy_opt = nullptr) {
    config.block_m = tuning.block_m;
    config.block_n = tuning.block_n;
    config.block_k = tuning.block_k;
    config.num_experts_per_wave = tuning.num_experts_per_wave;
    config.num_sms = input.launch_num_sms;
    config.num_dispatch_threads = tuning.num_dispatch_threads;
    config.num_non_epilogue_threads = tuning.num_non_epilogue_threads;
    config.num_epilogue_threads = tuning.num_epilogue_threads;
    config.direct_l2_scatter = tuning.direct_l2_scatter;
    config.nmajor_schedule = tuning.nmajor_schedule;
    config.one_warp_cleanup = tuning.one_warp_cleanup;
    config.swap_ab = tuning.swap_ab;
    config.sf_pool_stride_tokens = layout::nv_moe::get_num_sf_ring_tokens(
        config.num_max_pool_tokens, std::max(config.block_m, 64));

    const auto agy = agy_opt != nullptr ? *agy_opt : get_agy_sm90_spec(input);
    const bool strict_stages = not agy.enabled;
    const auto [num_stages, smem_size] = get_pipeline_config_for_mega_moe_sm90(
        SM90ArchSpec::smem_capacity,
        input.num_experts, input.hidden,
        config.block_m, config.block_n, config.block_k,
        config.num_dispatch_threads / 32, config.num_epilogue_threads / 32,
        config.direct_l2_scatter,
        tuning.num_stages,
        config.swap_ab,
        strict_stages,
        is_linear2_phase,
        input.num_ranks,
        &agy);
    if (num_stages == 0 or smem_size == 0)
        return false;
    config.num_stages = num_stages;
    config.smem_size = smem_size;
    return true;
}

static bool try_apply_sm90_moe_tuning(
    const Sm90MoeHeuristicInput& input,
    Sm90MoeScheduleTuning tuning,
    Sm90MoeLaunchConfig& config,
    const AgySm90KnobSpec* agy_opt = nullptr) {
    if (not tuning.selected)
        return true;
    return try_materialize_sm90_moe_phase_tuning(input, config.l1, tuning.l1, false, agy_opt) and
           try_materialize_sm90_moe_phase_tuning(input, config.l2, tuning.l2, true, agy_opt);
}

static Sm90MoeLaunchConfig select_mega_moe_sm90(
    const Sm90MoeHeuristicInput& input) {
    const auto agy = get_agy_sm90_spec(input);
    const AgySm90KnobSpec official_agy{};
    const auto hardware_profile = classify_sm90_moe_hardware(input.launch_num_sms);
    const auto shape_family = classify_sm90_moe_shape(
        input.hidden, input.intermediate_hidden, &agy);
    const auto load = get_sm90_moe_load(input);

    auto generic = make_generic_mega_moe_config_sm90(input, false, &official_agy);
    Sm90MoeLaunchConfig result {
        generic, generic,
        {}
    };
    result.l1.nmajor_schedule = false;

    // Shapes that cannot use the BN256 frontend fall back to the conservative
    // BN128 route before hardware-specific range tuning is applied.
    if (not is_sm90_moe_launch_config_legal(input, result, false, &official_agy)) {
        generic = make_generic_mega_moe_config_sm90(input, true, &official_agy);
        result.l1 = result.l2 = generic;
        result.l1.nmajor_schedule = false;
    }
    DG_HOST_ASSERT(is_sm90_moe_launch_config_legal(input, result, false, &official_agy));

    auto candidate = result;
    candidate.numerical.bf16_scaled_accum =
        hardware_profile == Sm90MoeHardwareProfile::HighSm and
        select_sm90_moe_high_sm_bf16_scaled_accum(shape_family, load);
    const auto tuning = select_sm90_moe_tuning(
        input, hardware_profile, shape_family, load, result);
    if (try_apply_sm90_moe_tuning(input, tuning, candidate, &official_agy) and
        is_sm90_moe_launch_config_legal(input, candidate, tuning.selected, &official_agy)) {
        result = candidate;
    }

    if (agy.enabled) {
        const int req_l1_bm = agy.get_int("l1_bm", 0);
        const int req_l2_bm = agy.get_int("l2_bm", 0);
        const bool decoupled_bm = (req_l1_bm > 0 and req_l2_bm > 0 and req_l1_bm != req_l2_bm);
        const bool is_fused = (agy.get_int("fused", 0) > 0) and not decoupled_bm;
        result.fused_single_kernel = is_fused;
        auto apply_phase = [&](MegaMoESM90Config& cfg, bool is_l2) {
            if (agy.get_int("adapt_bm", 0) > 0) {
                if (not agy.has("bm")) cfg.block_m = 64;
                if (not agy.has("swap")) cfg.swap_ab = true;
            }
            const char* pfx_bm = is_l2 ? "l2_bm" : "l1_bm";
            const char* pfx_bn = is_l2 ? "l2_bn" : "l1_bn";
            const char* pfx_bk = is_l2 ? "l2_bk" : "l1_bk";
            const char* pfx_wgs = is_l2 ? "l2_wgs" : "l1_wgs";
            const char* pfx_st = is_l2 ? "l2_stages" : "l1_stages";
            const char* pfx_epw = is_l2 ? "l2_epw" : "l1_epw";
            const char* pfx_swap = is_l2 ? "l2_swap" : "l1_swap";

            if (agy.has("bm")) cfg.block_m = agy.get_int("bm", cfg.block_m);
            if (agy.has(pfx_bm)) {
                cfg.block_m = agy.get_int(pfx_bm, cfg.block_m);
                // Synchronous M -> blockM profile lock when bn/swap/wgs are not overridden:
                const bool can_bn256 = ((2 * input.intermediate_hidden) % 256 == 0) and (input.hidden % 256 == 0);
                if (not agy.has("swap") and not agy.has(pfx_swap))
                    cfg.swap_ab = (cfg.block_m <= 32);
                if (not agy.has("bn") and not agy.has(pfx_bn))
                    cfg.block_n = (cfg.block_m >= 128 or not can_bn256) ? 128 : 256;
                if (not agy.has("bk") and not agy.has(pfx_bk))
                    cfg.block_k = 128;
            }
            if (agy.has("bn")) cfg.block_n = agy.get_int("bn", cfg.block_n);
            if (agy.has(pfx_bn)) cfg.block_n = agy.get_int(pfx_bn, cfg.block_n);
            if (agy.has("bk")) cfg.block_k = agy.get_int("bk", cfg.block_k);
            if (agy.has(pfx_bk)) cfg.block_k = agy.get_int(pfx_bk, cfg.block_k);

            if (agy.has("swap")) cfg.swap_ab = agy.get_int("swap", cfg.swap_ab ? 1 : 0) > 0;
            if (agy.has(pfx_swap)) cfg.swap_ab = agy.get_int(pfx_swap, cfg.swap_ab ? 1 : 0) > 0;

            int wgs = cfg.num_epilogue_threads / 128;
            if (agy.has("wgs")) wgs = agy.get_int("wgs", wgs);
            if (agy.has(pfx_wgs)) wgs = agy.get_int(pfx_wgs, wgs);
            else if (not agy.has("wgs")) {
                if (cfg.swap_ab and (cfg.block_n / 64 == 2 or cfg.block_n / 64 == 4))
                    wgs = cfg.block_n / 64;
                else if (not cfg.swap_ab and cfg.block_m == 64 and cfg.block_n >= 256)
                    wgs = (cfg.block_n == 512) ? 4 : 2;
                else if (not cfg.swap_ab and cfg.block_m == 128 and cfg.block_n == 128)
                    wgs = 2;
            }
            if (not cfg.swap_ab and cfg.block_m == 64 and cfg.block_n == 512 and not agy.has(pfx_wgs))
                wgs = 4;
            cfg.num_epilogue_threads = wgs * 128;
            if (agy.has("epi_warps")) {
                const int ew = agy.get_int("epi_warps", wgs * 4);
                DG_HOST_ASSERT(ew > 0 and (ew % 4) == 0);
                wgs = ew / 4;
                cfg.num_epilogue_threads = wgs * 128;
            }

            const char* pfx_disp = is_l2 ? "l2_disp_warps" : "l1_disp_warps";
            const int req_d_mask = agy.get_int("d_mask_mode", 0);
            const bool compact_fe = cfg.block_n >= 256 or cfg.swap_ab or wgs == 4;
            int disp_warps = compact_fe ? 2 : 4;
            if (req_d_mask > 0 and (not is_l2 or is_fused) and
                not agy.has("disp_warps") and not agy.has(pfx_disp)) {
                disp_warps = 6;
            }
            if (agy.has("dispatch_warps"))
                disp_warps = agy.get_int("dispatch_warps", disp_warps);
            if (agy.has("scatter_warps") and is_l2)
                disp_warps = agy.get_int("scatter_warps", disp_warps);
            if (agy.has("disp_warps"))
                disp_warps = agy.get_int("disp_warps", disp_warps);
            if (agy.has(pfx_disp))
                disp_warps = agy.get_int(pfx_disp, disp_warps);
            if (wgs == 4 and disp_warps > 2) {
                if (agy.has("dispatch_warps") or agy.has("scatter_warps") or agy.has("disp_warps")) {
                    wgs = 2;
                    cfg.num_epilogue_threads = wgs * 128;
                } else {
                    disp_warps = 2;
                }
            }

            int non_epi_warps = ((disp_warps % 4) == 2) ? 2 : 4;
            if (agy.has("non_epi_warps"))
                non_epi_warps = agy.get_int("non_epi_warps", non_epi_warps);

            cfg.num_dispatch_threads = disp_warps * 32;
            cfg.num_non_epilogue_threads = non_epi_warps * 32;

            if (agy.has("direct_l2"))
                cfg.direct_l2_scatter = agy.get_int("direct_l2", 0) > 0;
            const int ilv_req = is_fused ? agy.get_int("interleave", agy.get_int("interleave_l1_l2", 0)) : 0;
            if ((agy.get_int("dedup_scat", 0) > 0 and ilv_req == 0) or agy.is_scatter_role_disp())
                cfg.direct_l2_scatter = false;
            if (cfg.swap_ab or (cfg.block_n / wgs) != 128)
                cfg.direct_l2_scatter = false;

            if (not is_l2 and agy.has("l1_nmajor"))
                cfg.nmajor_schedule = agy.get_int("l1_nmajor", 0) > 0;
            if (is_l2 and agy.has("l2_nmajor"))
                cfg.nmajor_schedule = agy.get_int("l2_nmajor", 0) > 0;
            if (agy.has("cleanup"))
                cfg.one_warp_cleanup = agy.get_int("cleanup", 0) > 0;

            if (agy.has("epw"))
                cfg.num_experts_per_wave = normalize_num_experts_per_wave_for_mega_moe_sm90(
                    input.num_experts_per_rank,
                    std::clamp(agy.get_int("epw", cfg.num_experts_per_wave), 1, input.num_experts_per_rank));
            if (agy.has(pfx_epw))
                cfg.num_experts_per_wave = normalize_num_experts_per_wave_for_mega_moe_sm90(
                    input.num_experts_per_rank,
                    std::clamp(agy.get_int(pfx_epw, cfg.num_experts_per_wave), 1, input.num_experts_per_rank));

            cfg.sf_pool_stride_tokens = layout::nv_moe::get_num_sf_ring_tokens(
                cfg.num_max_pool_tokens, std::max(cfg.block_m, 64));

            bool explicit_stages = agy.has("stages") or agy.has(pfx_st);
            int req_stages = cfg.num_stages;
            if (agy.has("stages")) req_stages = agy.get_int("stages", req_stages);
            if (agy.has(pfx_st)) req_stages = agy.get_int(pfx_st, req_stages);
            if (explicit_stages and req_stages == 0) {
                const int phase_k = is_fused
                    ? std::max(input.hidden, input.intermediate_hidden)
                    : (is_l2 ? input.intermediate_hidden : input.hidden);
                const int k_tiles = std::max(2, phase_k / cfg.block_k);
                const int max_auto_st = agy.get_int("max_st", 6);
                const int target_st = std::min(k_tiles, max_auto_st);
                const auto [max_st, max_smem] = get_pipeline_config_for_mega_moe_sm90(
                    SM90ArchSpec::smem_capacity,
                    input.num_experts, input.hidden,
                    cfg.block_m, cfg.block_n, cfg.block_k,
                    cfg.num_dispatch_threads / 32, cfg.num_epilogue_threads / 32,
                    cfg.direct_l2_scatter,
                    target_st,
                    cfg.swap_ab,
                    false,
                    is_l2,
                    input.num_ranks,
                    &agy);
                DG_HOST_ASSERT(max_st >= 2 and max_smem > 0);
                cfg.num_stages = max_st;
                cfg.smem_size = max_smem;
            } else {
                const auto [st, smem] = get_pipeline_config_for_mega_moe_sm90(
                    SM90ArchSpec::smem_capacity,
                    input.num_experts, input.hidden,
                    cfg.block_m, cfg.block_n, cfg.block_k,
                    cfg.num_dispatch_threads / 32, cfg.num_epilogue_threads / 32,
                    cfg.direct_l2_scatter,
                    req_stages,
                    cfg.swap_ab,
                    false,
                    is_l2,
                    input.num_ranks,
                    &agy);
                DG_HOST_ASSERT(st >= 2 and smem > 0);
                cfg.num_stages = st;
                cfg.smem_size = smem;
            }
        };
        apply_phase(result.l1, false);
        apply_phase(result.l2, true);
        if (is_fused) {
            // Ensure identical CTA pipeline geometry between L1 and L2 in Fused mode
            result.l1.direct_l2_scatter = result.l2.direct_l2_scatter;
            result.l1.one_warp_cleanup = result.l2.one_warp_cleanup;
            result.l2.block_m = result.l1.block_m;
            result.l2.block_n = result.l1.block_n;
            result.l2.block_k = result.l1.block_k;
            result.l2.swap_ab = result.l1.swap_ab;
            result.l2.num_dispatch_threads = result.l1.num_dispatch_threads;
            result.l2.num_non_epilogue_threads = result.l1.num_non_epilogue_threads;
            result.l2.num_epilogue_threads = result.l1.num_epilogue_threads;
            result.l2.num_experts_per_wave = result.l1.num_experts_per_wave;
            const auto [f_st, f_smem] = get_pipeline_config_for_mega_moe_sm90(
                SM90ArchSpec::smem_capacity,
                input.num_experts, input.hidden,
                result.l1.block_m, result.l1.block_n, result.l1.block_k,
                result.l1.num_dispatch_threads / 32, result.l1.num_epilogue_threads / 32,
                result.l2.direct_l2_scatter,
                std::min(result.l1.num_stages, result.l2.num_stages),
                result.l1.swap_ab,
                false,
                false,
                input.num_ranks,
                &agy);
            DG_HOST_ASSERT(f_st >= 2 and f_smem > 0);
            result.l1.num_stages = result.l2.num_stages = f_st;
            result.l1.smem_size = result.l2.smem_size = f_smem;
        }
        if (agy.get_int("adapt_bm", 0) > 0 and not agy.has("bf16_acc"))
            result.numerical.bf16_scaled_accum = true;
        if (agy.has("bf16_acc"))
            result.numerical.bf16_scaled_accum = agy.get_int("bf16_acc", 0) > 0;
        if (not is_sm90_moe_launch_config_legal(input, result, true, &agy)) {
            std::cerr << "[AGY_ILLEGAL_CFG] agy=" << agy.raw
                      << " fused=" << result.fused_single_kernel
                      << " l1_legal=" << is_sm90_moe_phase_config_legal(input, result.l1, 2 * input.intermediate_hidden, true, false, &agy)
                      << " l2_legal=" << is_sm90_moe_phase_config_legal(input, result.l2, input.hidden, true, true, &agy)
                      << " padded_sf=" << input.num_padded_sf_pool_tokens
                      << " l1=" << result.l1
                      << " l2=" << result.l2
                      << " bf16_acc=" << result.numerical.bf16_scaled_accum
                      << std::endl;
        }
        DG_HOST_ASSERT(is_sm90_moe_launch_config_legal(input, result, true, &agy) and
                       "AGY_SM90_CFG produced an illegal launch configuration");
    }

    if (deep_jit::get_env<int>("DG_JIT_DEBUG", 0) or deep_jit::get_env<int>("DG_PRINT_CONFIGS", 0)) {
        const auto key = std::format(
            "Sm90MoeLaunchConfig(num_ranks={}, num_experts={}, hidden={}, intermediate_hidden={}, num_max_tokens_per_rank={}, num_tokens={}, num_topk={}, agy={})",
            input.num_ranks, input.num_experts, input.hidden,
            input.intermediate_hidden, input.num_max_tokens_per_rank,
            input.num_tokens, input.num_topk, agy.raw);
        static std::unordered_set<std::string> printed;
        if (printed.count(key) == 0) {
            std::cout << key << ": profile=" << static_cast<int>(hardware_profile)
                      << ", shape_family=" << static_cast<int>(shape_family)
                      << ", l1=" << result.l1
                      << ", l2=" << result.l2
                      << ", bf16_scaled_accum=" << result.numerical.bf16_scaled_accum
                      << std::endl;
            printed.insert(key);
        }
    }
    return result;
}

} // namespace deep_gemm
