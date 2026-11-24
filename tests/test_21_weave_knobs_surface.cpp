// =============================================================================
// tests/test_21_weave_knobs_surface.cpp
//
// Standalone CPU-host verification of the unified WEAVE_* surface: the 14 public
// knobs (Spatial 5, Temporal 3, Implementation 6), the legacy aliases (WEAVE_BLOCK_M_L1 /
// WEAVE_BLOCK_M_L2, plus WEAVE_ADAPT_BM of the accepted-but-not-public
// WEAVE_ADAPT_GEMM_CONFIG: host shape-adaptive tuple selection is always on), and the
// internal pins of the six retired knobs.
// (The file keeps its historical test_21_* name.)
// No GPU, no CUDA runtime, no PyTorch and no JIT compiler is required: the test
// exercises the *real* repository headers
//
//   * deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh   (ABI + macros)
//   * csrc/jit_kernels/heuristics/weave_knob_config.hpp       (host from_env)
//   * csrc/jit_kernels/heuristics/sm90_mega_moe.hpp           (SM90 spec parser,
//     included verbatim as a generated slice when WEAVE_SURFACE_SM90_SPEC_SLICE
//     is defined; see tests/test_21_weave_knobs_surface.py)
//
// Build (the four documented include dirs, plus the host stand-ins for the
// CUDA / torch / CUTLASS headers that a CPU-only checkout does not have):
//
//   g++ -std=c++20 -I deep_gemm/include -I csrc \
//       -I third-party/cutlass/include -I third-party/deep_jit/include \
//       -I tests/weave_surface_host tests/test_21_weave_knobs_surface.cpp -o t
//
// Modes:
//   (no args)                     run the full behavioural suite
//   --resolve ARCH PREC [K=V...]  print the resolved WeaveKnobs as JSON
//   --sm90-spec [K=V...]          print the parsed SM90 knob spec as JSON
//   --sm90-resolved [K=V...]      print the resolved SM90 staging/owner as JSON
//   --quiet                       only print failures and the summary
//
// ARCH is sm103 | h20 | mi308x; PREC is fp8xfp4 | fp4xfp4 | fp8xfp8 | bf16.
// A rejected request prints {"error": "..."} and exits with status 3, so the
// Python driver can assert fail-closed behaviour without a GPU.
//
// tests/test_21_weave_knobs_surface.py generates the SM90 slice, builds this
// file, runs the full suite and repeats the knob matrix through the CLI
// contract above.  Define WEAVE_SURFACE_SHARED to also export
// weave_surface_entry() for in-process loading by that driver.
// =============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#include <deep_gemm/scheduler/weave_knobs.cuh>
#include <jit_kernels/heuristics/weave_knob_config.hpp>

#ifdef WEAVE_SURFACE_SM90_SPEC_SLICE
#include "weave_sm90_spec_slice.hpp"
#endif

namespace {

#if defined(_WIN32)
// MSVC / MinGW expose only the _putenv_s spelling; the empty value unsets.
int host_setenv(const char* name, const char* value, int) {
    return _putenv_s(name, value);
}
int host_unsetenv(const char* name) {
    return _putenv_s(name, "");
}
#else
int host_setenv(const char* name, const char* value, int overwrite) {
    return ::setenv(name, value, overwrite);
}
int host_unsetenv(const char* name) {
    return ::unsetenv(name);
}
#endif

int g_checks = 0;
int g_failures = 0;
bool g_quiet = false;
std::string g_section;

void section(const char* name) {
    g_section = name;
    if (not g_quiet)
        std::printf("\n== %s ==\n", name);
}

void check(bool ok, const std::string& label) {
    ++g_checks;
    if (not ok) {
        ++g_failures;
        std::printf("  [FAIL] %s :: %s\n", g_section.c_str(), label.c_str());
    }
}

// Every environment name read by from_env(), including the archived DG_*
// aliases, so each case starts from a pristine environment.
const std::vector<std::string>& all_env_names() {
    static const std::vector<std::string> names = {
        "WEAVE_DISPATCH_ACTIVE_SMS", "WEAVE_SCATTER_ACTIVE_SMS",
        "WEAVE_SM0_STREAM_SCHED", "WEAVE_DISPATCH_WARPS", "WEAVE_SCATTER_WARPS",
        "WEAVE_REDUCE_WARPS", "WEAVE_CU_WAVE_SPLIT", "WEAVE_REG_BALANCE",
        "WEAVE_V_HEAD", "WEAVE_READY_GROUP_SIZE", "WEAVE_READY_QUEUE",
        "WEAVE_SCATTER_OWNER", "WEAVE_COMBINE_SYNC_CHUNK", "WEAVE_PRECISION_MODE",
        "WEAVE_ADAPT_BM", "WEAVE_BLOCK_M_L1", "WEAVE_BLOCK_M_L2",
        "WEAVE_REG_PULL_BYTES", "WEAVE_DEDUP_MODE", "WEAVE_SCATTER_STAGING",
        "WEAVE_LOCAL_SCATTER_DEDUP",
        "WEAVE_ADAPT_GEMM_CONFIG", "WEAVE_GEMM1_CONFIG", "WEAVE_GEMM2_CONFIG",
        "WEAVE_LOCALITY_DOMAINS", "WEAVE_AMD_8WAVE_MIN_TOK", "WEAVE_L2_KBLOCK_DEP",
        "WEAVE_RING_BLOCKS", "WEAVE_EARLY_STREAM_REDUCE", "WEAVE_TMEM_PIPELINE_MODE",
        "WEAVE_HEAD_PULL_CHUNK", "WEAVE_EPI_REG_STAGES", "WEAVE_SCHED_L2_RATIO",
        "WEAVE_STORE_BLOCK_M", "WEAVE_STORE_BLOCK_M_L1", "WEAVE_STORE_BLOCK_M_L2",
        "WEAVE_BLOCK_M", "WEAVE_FORCE_OFFICIAL", "WEAVE_H20_CONFIG",
        "WEAVE_KNOB_PRECISION_MODE",
        "DG_DISPATCH_WARPS", "DG_REDUCE_WARPS", "DG_SCATTER_ACTIVE_SMS",
        "DG_LOCALITY_DOMAINS", "DG_SM0_STREAM_SCHED", "DG_AMD_8WAVE_MIN_TOK",
        "DG_V_HEAD", "DG_L2_KBLOCK_DEP", "DG_RING_BLOCKS", "DG_EARLY_STREAM_REDUCE",
        "DG_TMEM_PIPELINE_MODE", "DG_HEAD_PULL_CHUNK", "DG_ADAPT_BM",
        "DG_BLOCK_M_L1", "DG_BLOCK_M_L2", "DG_STORE_BLOCK_M_L1",
        "DG_STORE_BLOCK_M_L2", "DG_LOCAL_SCATTER_DEDUP", "DG_STORE_BLOCK_M",
        "DG_BLOCK_M", "DG_SCATTER_WARPS",
        "AGY_SM90_CFG", "AGY_FORCE_OFFICIAL",
    };
    return names;
}

void clear_env() {
    for (const auto& name : all_env_names())
        host_unsetenv(name.c_str());
}

void set_env(const char* name, const std::string& value) {
    host_setenv(name, value.c_str(), 1);
}

weave::HardwareArch arch_of(const std::string& name) {
    if (name == "sm103" or name == "b300" or name == "sm100")
        return weave::HardwareArch::B300_SM100;
    if (name == "h20" or name == "sm90")
        return weave::HardwareArch::H20_SM90;
    if (name == "mi308x" or name == "gfx942")
        return weave::HardwareArch::MI308X_GFX942;
    std::fprintf(stderr, "unknown arch: %s\n", name.c_str());
    std::exit(2);
}

weave::PrecisionMode prec_of(const std::string& name) {
    if (name == "fp8xfp4") return weave::PrecisionMode::FP8xFP4;
    if (name == "fp4xfp4" or name == "nvfp4") return weave::PrecisionMode::NVFP4;
    if (name == "fp8xfp8" or name == "fp8") return weave::PrecisionMode::FP8xFP8;
    if (name == "bf16") return weave::PrecisionMode::BF16;
    std::fprintf(stderr, "unknown precision: %s\n", name.c_str());
    std::exit(2);
}

const std::vector<weave::HardwareArch>& all_arches() {
    static const std::vector<weave::HardwareArch> arches = {
        weave::HardwareArch::B300_SM100,
        weave::HardwareArch::H20_SM90,
        weave::HardwareArch::MI308X_GFX942,
    };
    return arches;
}

const char* arch_name(weave::HardwareArch arch) {
    switch (arch) {
        case weave::HardwareArch::B300_SM100: return "sm103";
        case weave::HardwareArch::H20_SM90: return "h20";
        default: return "mi308x";
    }
}

weave::WeaveKnobs resolve(weave::HardwareArch arch,
                          weave::PrecisionMode prec = weave::PrecisionMode::FP8xFP4) {
    return deep_gemm::weave_knobs::from_env(arch, prec);
}

// A request is fail-closed iff from_env() throws the repository DGException.
void expect_reject(const std::string& label, weave::HardwareArch arch) {
    try {
        (void) resolve(arch);
        check(false, label + " [" + arch_name(arch) + "] was accepted, expected rejection");
    } catch (const deep_gemm::DGException&) {
        check(true, label);
    } catch (const std::exception& e) {
        check(false, label + " threw a non-DGException: " + e.what());
    }
}

void expect_reject_all(const std::string& label) {
    for (auto arch : all_arches())
        expect_reject(label, arch);
}

// The 14 public knobs, in documentation order: Spatial 5, Temporal 3,
// Implementation 6 (identical to deep_gemm/mega/knobs.py).
const std::vector<std::string>& public_knobs() {
    static const std::vector<std::string> names = {
        // Spatial
        "WEAVE_DISPATCH_ACTIVE_SMS", "WEAVE_SCATTER_ACTIVE_SMS",
        "WEAVE_DISPATCH_WARPS", "WEAVE_SCATTER_WARPS", "WEAVE_CU_WAVE_SPLIT",
        // Temporal
        "WEAVE_V_HEAD", "WEAVE_READY_GROUP_SIZE", "WEAVE_SCATTER_OWNER",
        // Implementation
        "WEAVE_PRECISION_MODE", "WEAVE_GEMM1_CONFIG",
        "WEAVE_GEMM2_CONFIG", "WEAVE_REG_PULL_BYTES", "WEAVE_DEDUP_MODE",
        "WEAVE_SCATTER_STAGING",
    };
    return names;
}

// Accepted by the host for backward compatibility but NOT public: WEAVE_ADAPT_GEMM_CONFIG
// (0 = fixed canonical tuple, internal legacy/diagnostic override; unset = adaptive, always on).
const std::vector<std::string>& internal_compat_knobs() {
    static const std::vector<std::string> names = {"WEAVE_ADAPT_GEMM_CONFIG"};
    return names;
}

// Six knobs retired from the public surface.  They remain internal compile-time
// macros (default 0) and internal environment diagnostics of the host parsers.
const std::vector<std::string>& retired_internal_knobs() {
    static const std::vector<std::string> names = {
        "WEAVE_REDUCE_WARPS", "WEAVE_COMBINE_SYNC_CHUNK", "WEAVE_REG_BALANCE",
        "WEAVE_LOCAL_SCATTER_DEDUP", "WEAVE_SM0_STREAM_SCHED", "WEAVE_READY_QUEUE",
    };
    return names;
}

// Legacy environment spellings (transparent aliases): WEAVE_ADAPT_BM aliases the internal compat
// name WEAVE_ADAPT_GEMM_CONFIG, the other two alias the public GEMM1/GEMM2 knobs.
const std::vector<std::pair<std::string, std::string>>& legacy_aliases() {
    static const std::vector<std::pair<std::string, std::string>> aliases = {
        {"WEAVE_ADAPT_BM", "WEAVE_ADAPT_GEMM_CONFIG"},
        {"WEAVE_BLOCK_M_L1", "WEAVE_GEMM1_CONFIG"},
        {"WEAVE_BLOCK_M_L2", "WEAVE_GEMM2_CONFIG"},
    };
    return aliases;
}

#define WEAVE_KNOB_FIELDS(X) \
    X(enable_runtime_cost_model) X(override_mask) X(override_r_mask) \
    X(override_scatter_mode) X(override_v_head) X(block_m_fixed) X(precision_mode) \
    X(dispatch_warps) X(scatter_active_sms) X(locality_domains) X(sm0_stream_sched) \
    X(cu_wave_split) X(amd_8wave_min_tok) X(dispatch_active_sms) X(scatter_owner) \
    X(reduce_warps) X(l2_kblock_dep) X(ring_blocks) X(early_stream_reduce) \
    X(tmem_pipeline_mode) X(head_pull_chunk) X(combine_sync_chunk) X(epi_reg_stages) \
    X(sched_l2_ratio) X(adapt_bm) X(block_m_l1) X(block_m_l2) X(store_block_m_l1) \
    X(store_block_m_l2) X(local_scatter_dedup) X(scatter_staging) X(reg_balance)

std::vector<std::pair<std::string, uint32_t>> fields_of(const weave::WeaveKnobs& k) {
    std::vector<std::pair<std::string, uint32_t>> out;
#define WEAVE_EMIT_FIELD(f) out.emplace_back(#f, k.f);
    WEAVE_KNOB_FIELDS(WEAVE_EMIT_FIELD)
#undef WEAVE_EMIT_FIELD
    return out;
}

std::string json_escape(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (const unsigned char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buffer[7];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x", c);
                    out += buffer;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string knobs_json(const weave::WeaveKnobs& k) {
    std::string out = "{";
    bool first = true;
    for (const auto& [name, value] : fields_of(k)) {
        if (not first) out += ", ";
        first = false;
        out += "\"" + name + "\": " + std::to_string(value);
    }
    return out + "}";
}

// ---------------------------------------------------------------------------
// 1. Compiled defaults and the constexpr ABI.
// ---------------------------------------------------------------------------
void test_defaults() {
    section("compiled defaults / ABI");
    clear_env();
    for (auto arch : all_arches()) {
        for (auto prec : {weave::PrecisionMode::FP8xFP4, weave::PrecisionMode::NVFP4,
                          weave::PrecisionMode::FP8xFP8, weave::PrecisionMode::BF16}) {
            const auto knobs = resolve(arch, prec);
            const auto defaults = weave::default_weave_knobs(arch, prec);
            const auto a = fields_of(knobs);
            const auto b = fields_of(defaults);
            check(a.size() == b.size() and a.size() == 32, "WeaveKnobs exposes 32 fields");
            for (size_t i = 0; i < a.size() and i < b.size(); ++i)
                check(a[i] == b[i], std::string("unset env keeps compiled default ") +
                                    arch_name(arch) + "." + a[i].first);
            check(knobs.precision_mode == static_cast<uint32_t>(prec),
                  std::string("precision argument is honoured on ") + arch_name(arch));
        }
    }
    static_assert(WEAVE_DISPATCH_WARPS == 0u, "default dispatch warps is auto");
    static_assert(WEAVE_CU_WAVE_SPLIT <= 3u, "CU wave split stays inside the enum");
    static_assert(WEAVE_COMBINE_SYNC_CHUNK == 0u, "default combine sync is a global barrier");
    check(WEAVE_SCATTER_ACTIVE_SMS == 0u, "WEAVE_SCATTER_ACTIVE_SMS default 0");
    check(WEAVE_DISPATCH_ACTIVE_SMS == 0u, "WEAVE_DISPATCH_ACTIVE_SMS default 0");
    // Retired from the public surface: the internal `#ifndef` fallback is 0 and
    // must keep compiling (WEAVE_READY_QUEUE's fallback lives at the kernel use
    // sites; tests/test_21_weave_knobs_surface.py checks those statically).
    static_assert(WEAVE_SM0_STREAM_SCHED == 0u, "retired SM0 scheduler pin falls back to 0");
    static_assert(WEAVE_REDUCE_WARPS == 0u, "retired Reduce warp pin falls back to 0");
    static_assert(WEAVE_REG_BALANCE == 0u, "retired register-balance pin falls back to 0");
    static_assert(WEAVE_LOCAL_SCATTER_DEDUP == 0u, "retired local-dedup pin falls back to 0");
    check(WEAVE_SM0_STREAM_SCHED == 0u, "WEAVE_SM0_STREAM_SCHED internal fallback 0");
    check(WEAVE_REDUCE_WARPS == 0u, "WEAVE_REDUCE_WARPS internal fallback 0");
    check(WEAVE_CU_WAVE_SPLIT == 0u, "WEAVE_CU_WAVE_SPLIT default auto");
    check(WEAVE_REG_BALANCE == 0u, "WEAVE_REG_BALANCE default 0");
    check(WEAVE_ADAPT_BM == 0u, "WEAVE_ADAPT_BM default 0");
    check(WEAVE_BLOCK_M_L1 == 0u and WEAVE_BLOCK_M_L2 == 0u, "BLOCK_M_L1/L2 default auto");
    check(WEAVE_LOCAL_SCATTER_DEDUP == 0u, "WEAVE_LOCAL_SCATTER_DEDUP default 0");
    check(WEAVE_SCATTER_STAGING == 0u, "WEAVE_SCATTER_STAGING default reg_2stage");
    check(WEAVE_SCATTER_OWNER == 0u, "WEAVE_SCATTER_OWNER default epilogue_only");
    check(WEAVE_COMBINE_SYNC_CHUNK == 0u, "WEAVE_COMBINE_SYNC_CHUNK default global barrier");
    check(WEAVE_PRECISION_MODE == 0u, "WEAVE_PRECISION_MODE default fp8xfp4");
    check(WEAVE_V_HEAD == weave::kAutoU32, "WEAVE_V_HEAD default automatic");
    check(WEAVE_ENABLE_COST_MODEL == 1u, "preset 0 keeps the runtime cost model");
    check(WEAVE_ABLATION_PRESET == 0, "default ablation preset is 0 (Weave)");

    // r_mask packing: [7:0] comm, [15:8] compute, [23:16] groups, [31] hetero.
    constexpr uint32_t mask = weave::encode_r_mask(4u, 8u, 4u, true);
    static_assert(weave::r_comm_units(mask) == 4u, "r_mask comm lane");
    static_assert(weave::r_comp_units(mask) == 8u, "r_mask compute lane");
    static_assert(weave::r_groups(mask) == 4u, "r_mask group lane");
    static_assert(weave::r_heterogeneous(mask), "r_mask heterogeneous bit");
    static_assert(weave::default_r_mask_for_arch(weave::HardwareArch::H20_SM90) ==
                  weave::encode_r_mask(2u, 16u, 5u, false), "H20 default role mask");
    check(weave::default_r_mask_for_arch(weave::HardwareArch::B300_SM100) ==
              weave::encode_r_mask(4u, 8u, 4u, false), "SM103 default role mask");
    check(weave::default_r_mask_for_arch(weave::HardwareArch::MI308X_GFX942) ==
              weave::encode_r_mask(0u, 8u, 8u, false), "MI308X default role mask");
    for (auto mode : {weave::PrecisionMode::FP8xFP4, weave::PrecisionMode::NVFP4,
                      weave::PrecisionMode::FP8xFP8, weave::PrecisionMode::BF16})
        check(std::strlen(weave::precision_mode_name(mode)) > 0,
              "precision_mode_name is non-empty for every mode");
}

// ---------------------------------------------------------------------------
// 2. Every documented integer value of every host-resolved knob.
// ---------------------------------------------------------------------------
struct IntKnobCase {
    const char* env;
    uint32_t weave::WeaveKnobs::*field;
    std::vector<int> legal;
    std::vector<int> illegal;
};

void test_integer_values() {
    section("documented integer values");
    const std::vector<IntKnobCase> cases = {
        {"WEAVE_DISPATCH_ACTIVE_SMS", &weave::WeaveKnobs::dispatch_active_sms,
         {0, 1, 2, 4, 8, 16, 32, 64, 78, 148, 256, 1024}, {1025, 2048, -1, -1024}},
        {"WEAVE_SCATTER_ACTIVE_SMS", &weave::WeaveKnobs::scatter_active_sms,
         {0, 1, 2, 4, 8, 16, 32, 64, 78, 80, 148, 1024}, {1025, 2048, -1}},
        {"WEAVE_SM0_STREAM_SCHED", &weave::WeaveKnobs::sm0_stream_sched,
         {0, 1}, {2, 3, 8, -1}},
        {"WEAVE_DISPATCH_WARPS", &weave::WeaveKnobs::dispatch_warps,
         {0, 4, 8}, {1, 2, 3, 5, 6, 7, 9, 16, -1}},
        {"WEAVE_REDUCE_WARPS", &weave::WeaveKnobs::reduce_warps,
         {0, 1, 2, 4, 8}, {3, 5, 6, 7, 9, 16, -1}},
        {"WEAVE_REG_BALANCE", &weave::WeaveKnobs::reg_balance,
         {0, 1}, {2, 3, -1}},
        {"WEAVE_ADAPT_BM", &weave::WeaveKnobs::adapt_bm,
         {0, 1}, {2, 3, -1}},
        {"WEAVE_ADAPT_GEMM_CONFIG", &weave::WeaveKnobs::adapt_bm,
         {0, 1}, {2, 3, -1}},
        {"WEAVE_LOCAL_SCATTER_DEDUP", &weave::WeaveKnobs::local_scatter_dedup,
         {0, 1}, {2, 3, -1}},
        {"WEAVE_GEMM1_CONFIG", &weave::WeaveKnobs::block_m_l1,
         {0, 16, 32, 64, 80, 96, 128, 192, 240, 256},
         {1, 15, 17, 48, 63, 100, 127, 129, 200, 255, 257, 512, -1}},
        {"WEAVE_GEMM2_CONFIG", &weave::WeaveKnobs::block_m_l2,
         {0, 16, 32, 64, 80, 96, 128, 192, 240, 256},
         {1, 15, 17, 48, 63, 100, 127, 129, 200, 255, 257, 512, -1}},
        {"WEAVE_BLOCK_M_L1", &weave::WeaveKnobs::block_m_l1,
         {0, 16, 32, 64, 80, 96, 128, 192, 240, 256},
         {1, 15, 17, 48, 63, 100, 127, 129, 200, 255, 257, 512, -1}},
        {"WEAVE_BLOCK_M_L2", &weave::WeaveKnobs::block_m_l2,
         {0, 16, 32, 64, 80, 96, 128, 192, 240, 256},
         {1, 15, 17, 48, 63, 100, 127, 129, 200, 255, 257, 512, -1}},
    };
    for (const auto& c : cases) {
        for (int value : c.legal) {
            clear_env();
            set_env(c.env, std::to_string(value));
            for (auto arch : all_arches()) {
                try {
                    const auto knobs = resolve(arch);
                    check(knobs.*(c.field) == static_cast<uint32_t>(value),
                          std::string(c.env) + "=" + std::to_string(value) + " on " +
                              arch_name(arch) + " -> " + std::to_string(knobs.*(c.field)));
                } catch (const deep_gemm::DGException& e) {
                    check(false, std::string(c.env) + "=" + std::to_string(value) +
                                 " on " + arch_name(arch) + " rejected: " + e.what());
                }
            }
        }
        for (int value : c.illegal) {
            clear_env();
            set_env(c.env, std::to_string(value));
            expect_reject_all(std::string(c.env) + "=" + std::to_string(value));
        }
    }
    // Empty values mean "unset", not zero and not an error: for the 14 public
    // names, the internal compat name, the legacy aliases and the internal pins of the retired knobs.
    std::vector<std::string> empty_value_names = public_knobs();
    for (const auto& name : internal_compat_knobs())
        empty_value_names.push_back(name);
    for (const auto& name : retired_internal_knobs())
        empty_value_names.push_back(name);
    for (const auto& alias : legacy_aliases())
        empty_value_names.push_back(alias.first);
    for (const auto& name : empty_value_names) {
        clear_env();
        set_env(name.c_str(), "");
        try {
            const auto knobs = resolve(weave::HardwareArch::B300_SM100);
            const auto defaults = weave::default_weave_knobs(weave::HardwareArch::B300_SM100);
            check(knobs_json(knobs) == knobs_json(defaults),
                  name + "=\"\" falls back to the compiled default");
        } catch (const deep_gemm::DGException& e) {
            check(false, name + "=\"\" rejected: " + e.what());
        }
    }
    // Four public knobs are consumed by the backend JIT define blocks rather
    // than the CTA-0 ABI.  They have no WeaveKnobs field, but from_env() still
    // fail-closes on their documented domains.
    struct RangeCase {
        const char* env;
        std::vector<int> legal;
        std::vector<int> illegal;
    };
    const std::vector<RangeCase> range_cases = {
        {"WEAVE_SCATTER_WARPS", {0, 4, 8}, {1, 2, 3, 5, 6, 7, 9, 16, -1}},
        {"WEAVE_READY_GROUP_SIZE", {0, 1, 2, 4, 8, 16, 64, 256}, {-1}},
        {"WEAVE_REG_PULL_BYTES", {0, 1024, 2048, 4096},
         {1, 512, 1025, 2047, 4097, 8192, -1}},
        {"WEAVE_DEDUP_MODE", {0, 1}, {2, 3, -1}},
    };
    for (const auto& c : range_cases) {
        for (int value : c.legal) {
            clear_env();
            set_env(c.env, std::to_string(value));
            for (auto arch : all_arches()) {
                try {
                    (void) resolve(arch);
                    check(true, std::string(c.env) + "=" + std::to_string(value) +
                                    " accepted on " + arch_name(arch));
                } catch (const deep_gemm::DGException& e) {
                    check(false, std::string(c.env) + "=" + std::to_string(value) +
                                     " on " + arch_name(arch) + " rejected: " + e.what());
                }
            }
        }
        for (int value : c.illegal) {
            clear_env();
            set_env(c.env, std::to_string(value));
            expect_reject_all(std::string(c.env) + "=" + std::to_string(value));
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Named enum strings and their integer codes.
// ---------------------------------------------------------------------------
void test_named_enums() {
    section("named enum strings");
    namespace wk = deep_gemm::weave_knobs;
    struct EnumCase {
        const char* env;
        std::vector<const char*> names;
        int (*parse)(const char*, int);
    };
    // The tables below are the documented public spellings, written out
    // independently of the parser so a silent table edit is caught.
    // WEAVE_PRECISION_MODE has its own parser (precision_mode_env, with the
    // deep_gemm/mega/precision.py aliases) and is covered by
    // test_precision_mode() below.
    const std::vector<EnumCase> cases = {
        {"WEAVE_READY_QUEUE", {"formula", "sm0_fifo"},
         [](const char* e, int fb) { return wk::named_env(e, {"formula", "sm0_fifo"}, fb); }},
        {"WEAVE_CU_WAVE_SPLIT", {"auto", "0-8", "4-4", "8-0"},
         [](const char* e, int fb) { return wk::named_env(e, {"auto", "0-8", "4-4", "8-0"}, fb); }},
        {"WEAVE_SCATTER_OWNER", {"epilogue_only", "comm_only"},
         [](const char* e, int fb) { return wk::named_env(e, {"epilogue_only", "comm_only"}, fb); }},
        {"WEAVE_SCATTER_STAGING", {"reg_2stage", "smem", "hbm_dedup"},
         [](const char* e, int fb) { return wk::named_env(e, {"reg_2stage", "smem", "hbm_dedup"}, fb); }},
    };
    for (const auto& c : cases) {
        for (size_t code = 0; code < c.names.size(); ++code) {
            clear_env();
            set_env(c.env, c.names[code]);
            check(c.parse(c.env, 0) == static_cast<int>(code),
                  std::string(c.env) + "=" + c.names[code] + " -> code " + std::to_string(code));
            clear_env();
            set_env(c.env, std::to_string(code));
            check(c.parse(c.env, 0) == static_cast<int>(code),
                  std::string(c.env) + "=" + std::to_string(code) + " -> code " + std::to_string(code));
        }
        const std::vector<std::string> garbage = {
            std::to_string(c.names.size()), "-1", "99", "auto=1", "FORMULA", "Sm0_Fifo",
            "comm", "epilogue", "reg", "hbm", "fp8", "fp4", "bf16x2", "2-6", "8-8", "1-7",
        };
        for (const auto& bad : garbage) {
            clear_env();
            set_env(c.env, bad);
            bool threw = false;
            try {
                (void) c.parse(c.env, 0);
            } catch (const deep_gemm::DGException&) {
                threw = true;
            }
            check(threw, std::string(c.env) + "=" + bad + " must be rejected");
        }
    }
    // End-to-end: the named string reaches the resolved ABI field.
    struct FieldEnum {
        const char* env;
        uint32_t weave::WeaveKnobs::*field;
        std::vector<std::pair<const char*, uint32_t>> mapping;
    };
    const std::vector<FieldEnum> field_cases = {
        {"WEAVE_CU_WAVE_SPLIT", &weave::WeaveKnobs::cu_wave_split,
         {{"auto", 0}, {"0-8", 1}, {"4-4", 2}, {"8-0", 3}, {"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}}},
        {"WEAVE_SCATTER_OWNER", &weave::WeaveKnobs::scatter_owner,
         {{"epilogue_only", 0}, {"comm_only", 1}, {"0", 0}, {"1", 1}}},
        {"WEAVE_SCATTER_STAGING", &weave::WeaveKnobs::scatter_staging,
         {{"reg_2stage", 0}, {"smem", 1}, {"hbm_dedup", 2}, {"0", 0}, {"1", 1}, {"2", 2}}},
    };
    for (const auto& c : field_cases) {
        for (const auto& [text, code] : c.mapping) {
            clear_env();
            set_env(c.env, text);
            if (std::strcmp(c.env, "WEAVE_SCATTER_STAGING") == 0 and code != 0)
                set_env("WEAVE_SCATTER_OWNER", "comm_only");
            for (auto arch : all_arches()) {
                try {
                    const auto knobs = resolve(arch);
                    check(knobs.*(c.field) == code,
                          std::string(c.env) + "=" + text + " on " + arch_name(arch) +
                              " -> " + std::to_string(knobs.*(c.field)));
                } catch (const deep_gemm::DGException& e) {
                    check(false, std::string(c.env) + "=" + text + " rejected: " + e.what());
                }
            }
        }
    }
    // WEAVE_READY_QUEUE has no ABI field of its own: it selects the CTA-0 lane.
    clear_env();
    set_env("WEAVE_READY_QUEUE", "formula");
    check(resolve(weave::HardwareArch::B300_SM100).sm0_stream_sched == 0u,
          "READY_QUEUE=formula keeps CTA-0 comm warps in the normal pull");
    clear_env();
    set_env("WEAVE_READY_QUEUE", "sm0_fifo");
    check(resolve(weave::HardwareArch::B300_SM100).sm0_stream_sched == 1u,
          "READY_QUEUE=sm0_fifo promotes CTA-0 to the streaming scheduler");
    clear_env();
    set_env("WEAVE_READY_QUEUE", "1");
    check(resolve(weave::HardwareArch::H20_SM90).sm0_stream_sched == 1u,
          "READY_QUEUE=1 is the sm0_fifo code");
    clear_env();
    set_env("WEAVE_READY_QUEUE", "sm0_fifo");
    set_env("WEAVE_SM0_STREAM_SCHED", "0");
    check(resolve(weave::HardwareArch::B300_SM100).sm0_stream_sched == 0u,
          "an explicit WEAVE_SM0_STREAM_SCHED overrides the READY_QUEUE inference");
}

// ---------------------------------------------------------------------------
// 3b. WEAVE_PRECISION_MODE: backend selector, fail-closed on the host.
//
// The env value is the Python dispatcher's backend selector
// (deep_gemm/mega/precision.py).  On the host it is parsed only to reject
// unknown spellings; the resolved precision_mode is always the precision of
// the backend actually being launched (the from_env() argument), so the CTA-0
// cost-model spec can never disagree with the compiled kernel.
// ---------------------------------------------------------------------------
void test_precision_mode() {
    section("WEAVE_PRECISION_MODE backend selector");
    namespace wk = deep_gemm::weave_knobs;
    // Documented spellings (codes, names, aliases; case-insensitive, trimmed),
    // written out independently of the parser table.
    const std::vector<std::pair<std::string, int>> spellings = {
        {"0", 0}, {"fp8xfp4", 0}, {"FP8xFP4", 0},
        {"1", 1}, {"fp4xfp4", 1}, {"nvfp4", 1}, {"NVFP4", 1},
        {"2", 2}, {"fp8xfp8", 2}, {"fp8", 2},
        {"3", 3}, {"bf16", 3}, {"bf16xbf16", 3}, {"BF16", 3},
        {" nvfp4 ", 1}, {"\tbf16\n", 3},
    };
    for (const auto& [text, code] : spellings) {
        clear_env();
        set_env("WEAVE_PRECISION_MODE", text);
        try {
            check(wk::precision_mode_env(-1) == code,
                  "WEAVE_PRECISION_MODE='" + text + "' -> code " + std::to_string(code));
        } catch (const deep_gemm::DGException& e) {
            check(false, "WEAVE_PRECISION_MODE='" + text + "' rejected: " + e.what());
        }
    }
    clear_env();
    check(wk::precision_mode_env(-7) == -7, "unset WEAVE_PRECISION_MODE -> fallback");
    for (const char* text : {"auto", "AUTO", " auto "}) {
        clear_env();
        set_env("WEAVE_PRECISION_MODE", text);
        check(wk::precision_mode_env(-7) == -7,
              std::string("WEAVE_PRECISION_MODE='") + text + "' -> fallback");
    }
    // Legacy alias, read only when the public name is unset.
    clear_env();
    set_env("WEAVE_KNOB_PRECISION_MODE", "bf16");
    check(wk::precision_mode_env(-1) == 3, "legacy WEAVE_KNOB_PRECISION_MODE is honoured");
    set_env("WEAVE_PRECISION_MODE", "nvfp4");
    check(wk::precision_mode_env(-1) == 1, "WEAVE_PRECISION_MODE wins over the legacy alias");

    // Unknown spellings fail closed in the parser and in from_env() on every arch.
    const std::vector<std::string> garbage = {
        "4", "-1", "99", "fp4", "fp16", "nfp4", "bf16x2", "fp8xfp16", "auto=1",
        "0x1", "1.0", "fp8 xfp4", "int8",
    };
    for (const auto& bad : garbage) {
        clear_env();
        set_env("WEAVE_PRECISION_MODE", bad);
        bool threw = false;
        try {
            (void) wk::precision_mode_env(-1);
        } catch (const deep_gemm::DGException&) {
            threw = true;
        }
        check(threw, "WEAVE_PRECISION_MODE=" + bad + " must be rejected");
        expect_reject_all("from_env with WEAVE_PRECISION_MODE=" + bad);
    }

    // End-to-end: a valid selector never overrides the launched backend.
    const std::vector<weave::PrecisionMode> backends = {
        weave::PrecisionMode::FP8xFP4, weave::PrecisionMode::NVFP4,
        weave::PrecisionMode::FP8xFP8, weave::PrecisionMode::BF16,
    };
    for (const auto& [text, code] : spellings) {
        (void) code;
        for (auto arch : all_arches()) {
            for (auto prec : backends) {
                clear_env();
                set_env("WEAVE_PRECISION_MODE", text);
                try {
                    const auto knobs = resolve(arch, prec);
                    check(knobs.precision_mode == static_cast<uint32_t>(prec),
                          "WEAVE_PRECISION_MODE='" + text + "' keeps the launched backend " +
                              weave::precision_mode_name(prec) + " on " + arch_name(arch));
                } catch (const deep_gemm::DGException& e) {
                    check(false, "WEAVE_PRECISION_MODE='" + text + "' rejected on " +
                                     arch_name(arch) + ": " + e.what());
                }
            }
        }
    }
    clear_env();
}

// ---------------------------------------------------------------------------
// 4. WEAVE_V_HEAD: -1 sentinel, 0 auto, positive quota.
// ---------------------------------------------------------------------------
void test_v_head() {
    section("WEAVE_V_HEAD quota");
    for (int value : {1, 2, 16, 64, 128, 256, 1024, 4096, 65536}) {
        clear_env();
        set_env("WEAVE_V_HEAD", std::to_string(value));
        for (auto arch : all_arches()) {
            const auto knobs = resolve(arch);
            check(knobs.override_v_head == static_cast<uint32_t>(value),
                  "V_HEAD=" + std::to_string(value) + " override value");
            check((knobs.override_mask & weave::kOverrideVHead) != 0u,
                  "V_HEAD=" + std::to_string(value) + " sets kOverrideVHead");
        }
    }
    clear_env();
    set_env("WEAVE_V_HEAD", "0");
    for (auto arch : all_arches()) {
        const auto knobs = resolve(arch);
        check(knobs.override_v_head == weave::kAutoU32, "V_HEAD=0 means automatic");
        check((knobs.override_mask & weave::kOverrideVHead) == 0u,
              "V_HEAD=0 clears kOverrideVHead");
    }
    clear_env();
    set_env("WEAVE_V_HEAD", "-1");
    for (auto arch : all_arches()) {
        const auto knobs = resolve(arch);
        check(knobs.override_v_head == weave::kAutoU32, "V_HEAD=-1 keeps the compiled default");
    }
    for (const char* bad : {"-2", "-65536", "65537", "100000"}) {
        clear_env();
        set_env("WEAVE_V_HEAD", bad);
        expect_reject_all(std::string("WEAVE_V_HEAD=") + bad);
    }
}

// ---------------------------------------------------------------------------
// 5. WEAVE_COMBINE_SYNC_CHUNK and its architecture / dedup coupling.
// ---------------------------------------------------------------------------
void test_combine_sync_chunk() {
    section("WEAVE_COMBINE_SYNC_CHUNK");
    for (int value : {0, 16, 32, 64, 128}) {
        clear_env();
        set_env("WEAVE_COMBINE_SYNC_CHUNK", std::to_string(value));
        const auto knobs = resolve(weave::HardwareArch::B300_SM100);
        check(knobs.combine_sync_chunk == static_cast<uint32_t>(value),
              "COMBINE_SYNC_CHUNK=" + std::to_string(value) + " on SM103");
    }
    for (int value : {1, 7, 8, 15, 17, 31, 33, 63, 65, 127, 129, 256, -1}) {
        clear_env();
        set_env("WEAVE_COMBINE_SYNC_CHUNK", std::to_string(value));
        expect_reject_all("COMBINE_SYNC_CHUNK=" + std::to_string(value));
    }
    // Off SM103 the chunked sync needs the local dedup path.
    for (auto arch : {weave::HardwareArch::H20_SM90, weave::HardwareArch::MI308X_GFX942}) {
        for (int value : {16, 32, 64, 128}) {
            clear_env();
            set_env("WEAVE_COMBINE_SYNC_CHUNK", std::to_string(value));
            expect_reject("COMBINE_SYNC_CHUNK=" + std::to_string(value) +
                          " without LOCAL_SCATTER_DEDUP", arch);
            clear_env();
            set_env("WEAVE_COMBINE_SYNC_CHUNK", std::to_string(value));
            set_env("WEAVE_LOCAL_SCATTER_DEDUP", "1");
            try {
                const auto knobs = resolve(arch);
                check(knobs.combine_sync_chunk == static_cast<uint32_t>(value) and
                          knobs.local_scatter_dedup == 1u and knobs.scatter_staging == 2u and
                          knobs.scatter_owner == 1u,
                      "COMBINE_SYNC_CHUNK=" + std::to_string(value) +
                          " + LOCAL_SCATTER_DEDUP=1 on " + arch_name(arch));
            } catch (const deep_gemm::DGException& e) {
                check(false, std::string("chunked dedup combine rejected: ") + e.what());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Scatter ownership / staging / dedup coupling (fail-closed, never mixed).
// ---------------------------------------------------------------------------
void test_scatter_coupling() {
    section("scatter ownership / staging coupling");
    for (const char* staging : {"smem", "hbm_dedup", "1", "2"}) {
        clear_env();
        set_env("WEAVE_SCATTER_OWNER", "epilogue_only");
        set_env("WEAVE_SCATTER_STAGING", staging);
        expect_reject_all(std::string("SCATTER_OWNER=epilogue_only + SCATTER_STAGING=") + staging);
        clear_env();
        set_env("WEAVE_SCATTER_OWNER", "0");
        set_env("WEAVE_SCATTER_STAGING", staging);
        expect_reject_all(std::string("SCATTER_OWNER=0 + SCATTER_STAGING=") + staging);
    }
    for (const char* sms : {"1", "8", "78"}) {
        clear_env();
        set_env("WEAVE_SCATTER_OWNER", "epilogue_only");
        set_env("WEAVE_SCATTER_ACTIVE_SMS", sms);
        expect_reject_all(std::string("SCATTER_OWNER=epilogue_only + SCATTER_ACTIVE_SMS=") + sms);
    }
    // Local bf16x2 dedup requires the HBM dedup staging path: the SM100
    // dispatcher forces scatter mode 3 here, so accepting reg_2stage / smem
    // would make the host sizing disagree with the emitted kernel macro.
    for (const char* staging : {"reg_2stage", "smem", "0", "1"}) {
        clear_env();
        set_env("WEAVE_LOCAL_SCATTER_DEDUP", "1");
        set_env("WEAVE_SCATTER_STAGING", staging);
        expect_reject_all(std::string("LOCAL_SCATTER_DEDUP=1 + SCATTER_STAGING=") + staging);
    }
    clear_env();
    set_env("WEAVE_SCATTER_STAGING", "smem");
    {
        const auto k = resolve(weave::HardwareArch::B300_SM100);
        check(k.scatter_staging == 1u and k.scatter_owner == 1u,
              "staging=smem auto-selects the comm owner");
    }
    clear_env();
    set_env("WEAVE_SCATTER_ACTIVE_SMS", "8");
    {
        const auto k = resolve(weave::HardwareArch::B300_SM100);
        check(k.scatter_active_sms == 8u and k.scatter_owner == 1u,
              "a throttled scatter subset auto-selects the comm owner");
    }
    clear_env();
    set_env("WEAVE_LOCAL_SCATTER_DEDUP", "1");
    {
        const auto k = resolve(weave::HardwareArch::H20_SM90);
        check(k.local_scatter_dedup == 1u and k.scatter_staging == 2u and k.scatter_owner == 1u,
              "LOCAL_SCATTER_DEDUP=1 auto-selects hbm_dedup + comm owner");
    }
    clear_env();
    set_env("WEAVE_SCATTER_STAGING", "hbm_dedup");
    {
        const auto k = resolve(weave::HardwareArch::B300_SM100);
        check(k.scatter_staging == 2u and k.local_scatter_dedup == 1u and k.scatter_owner == 1u,
              "hbm_dedup forces the bf16x2 local dedup and the comm owner");
    }
    clear_env();
    set_env("WEAVE_SCATTER_OWNER", "comm_only");
    {
        const auto k = resolve(weave::HardwareArch::B300_SM100);
        check(k.scatter_owner == 1u and k.scatter_staging == 0u,
              "comm_only alone keeps the register two-stage path");
    }
    clear_env();
    set_env("WEAVE_SCATTER_OWNER", "epilogue_only");
    {
        const auto k = resolve(weave::HardwareArch::B300_SM100);
        check(k.scatter_owner == 0u and k.scatter_staging == 0u,
              "epilogue_only is only legal with reg_2stage");
    }
    // Epilogue-owned local dedup (D7 mode 2): only an explicit
    // WEAVE_SCATTER_OWNER=epilogue_only selects it, only the B300/SM100
    // FP8xFP4 kernel implements it, and it excludes every D6 group.
    for (const char* owner : {"epilogue_only", "0"}) {
        for (const char* staging : {static_cast<const char*>(nullptr), "reg_2stage", "0"}) {
            clear_env();
            set_env("WEAVE_SCATTER_OWNER", owner);
            set_env("WEAVE_LOCAL_SCATTER_DEDUP", "1");
            if (staging != nullptr)
                set_env("WEAVE_SCATTER_STAGING", staging);
            const std::string label = std::string("SCATTER_OWNER=") + owner +
                " + LOCAL_SCATTER_DEDUP=1" +
                (staging != nullptr ? std::string(" + SCATTER_STAGING=") + staging : std::string());
            const auto k = resolve(weave::HardwareArch::B300_SM100);
            check(k.scatter_owner == 0u and k.scatter_staging == 0u and k.local_scatter_dedup == 1u,
                  label + " selects epilogue-owned local dedup (D7 mode 2) on sm103");
            expect_reject(label + " (D7 mode 2 is SM100-only)", weave::HardwareArch::H20_SM90);
            expect_reject(label + " (D7 mode 2 is SM100-only)", weave::HardwareArch::MI308X_GFX942);
        }
    }
    struct Mode2Conflict { const char* env; const char* value; };
    for (const Mode2Conflict& c : {Mode2Conflict{"WEAVE_COMBINE_SYNC_CHUNK", "16"},
                                   Mode2Conflict{"WEAVE_COMBINE_SYNC_CHUNK", "32"},
                                   Mode2Conflict{"WEAVE_EARLY_STREAM_REDUCE", "1"},
                                   Mode2Conflict{"WEAVE_REDUCE_WARPS", "4"},
                                   Mode2Conflict{"WEAVE_SCATTER_STAGING", "hbm_dedup"},
                                   Mode2Conflict{"WEAVE_SCATTER_STAGING", "smem"},
                                   Mode2Conflict{"WEAVE_SCATTER_ACTIVE_SMS", "8"}}) {
        clear_env();
        set_env("WEAVE_SCATTER_OWNER", "epilogue_only");
        set_env("WEAVE_LOCAL_SCATTER_DEDUP", "1");
        set_env(c.env, c.value);
        expect_reject_all(std::string("SCATTER_OWNER=epilogue_only + LOCAL_SCATTER_DEDUP=1 + ") +
                          c.env + "=" + c.value);
    }
    // An explicit comm owner keeps local dedup on the HBM dedup queue (mode 3).
    clear_env();
    set_env("WEAVE_SCATTER_OWNER", "comm_only");
    set_env("WEAVE_LOCAL_SCATTER_DEDUP", "1");
    for (auto arch : all_arches()) {
        const auto k = resolve(arch);
        check(k.scatter_owner == 1u and k.scatter_staging == 2u and k.local_scatter_dedup == 1u,
              std::string("SCATTER_OWNER=comm_only + LOCAL_SCATTER_DEDUP=1 selects hbm_dedup on ") +
              arch_name(arch));
    }
}

// ---------------------------------------------------------------------------
// 6b. The internal compat name WEAVE_ADAPT_GEMM_CONFIG (not public) and the renamed
//     Implementation knobs (WEAVE_GEMM1_CONFIG, WEAVE_GEMM2_CONFIG): whole-tuple grammar, transparent
//     legacy aliases (WEAVE_ADAPT_BM / WEAVE_BLOCK_M_L1 / WEAVE_BLOCK_M_L2) and
//     "the new name wins" precedence.
// ---------------------------------------------------------------------------
void test_gemm_config_surface() {
    section("renamed GEMM config knobs: tuple grammar and legacy aliases");
    namespace wk = deep_gemm::weave_knobs;
    const auto sm103 = weave::HardwareArch::B300_SM100;

    // Whole-tuple grammar BM[:STORE_BM[:BK[:STAGES[:EPI]]]] (':' or ',').
    for (const char* sep : {":", ","}) {
        clear_env();
        const std::string tuple = std::string("64") + sep + "32" + sep + "128" + sep + "6" + sep + "2";
        set_env("WEAVE_GEMM1_CONFIG", tuple);
        const auto cfg = wk::gemm1_config_env();
        check(cfg.set and cfg.bm == 64 and cfg.store_bm == 32 and cfg.bk == 128 and
                  cfg.stages == 6 and cfg.epi_reg_stages == 2,
              "WEAVE_GEMM1_CONFIG=" + tuple + " parses every field");
        const auto knobs = resolve(sm103);
        check(knobs.block_m_l1 == 64u and knobs.store_block_m_l1 == 32u,
              "GEMM1 tuple " + tuple + " drives block_m_l1 / store_block_m_l1");
        check(knobs.block_m_l2 == 0u and knobs.store_block_m_l2 == 0u,
              "GEMM1 tuple " + tuple + " leaves GEMM2 untouched");
    }
    clear_env();
    set_env("WEAVE_GEMM2_CONFIG", "96:16:256:4:1");
    {
        const auto cfg = wk::gemm2_config_env();
        check(cfg.set and cfg.bm == 96 and cfg.store_bm == 16 and cfg.bk == 256 and
                  cfg.stages == 4 and cfg.epi_reg_stages == 1,
              "WEAVE_GEMM2_CONFIG=96:16:256:4:1 parses every field");
        const auto knobs = resolve(sm103);
        check(knobs.block_m_l2 == 96u and knobs.store_block_m_l2 == 16u and knobs.block_m_l1 == 0u,
              "GEMM2 tuple drives only the GEMM2 fields (decoupled from GEMM1)");
    }
    clear_env();
    set_env("WEAVE_GEMM1_CONFIG", "128");
    check(wk::gemm1_config_env().set and wk::gemm1_config_env().bm == 128 and
              wk::gemm1_config_env().store_bm == 0, "a bare BM is a valid GEMM1 config");
    clear_env();
    set_env("WEAVE_GEMM1_CONFIG", "0");
    check(not wk::gemm1_config_env().set, "WEAVE_GEMM1_CONFIG=0 means auto (not pinned)");
    clear_env();
    check(not wk::gemm1_config_env().set and not wk::gemm2_config_env().set,
          "unset GEMM1/GEMM2 config is auto");

    // Malformed or out-of-range tuples are fail-closed for both phases.
    for (const char* env : {"WEAVE_GEMM1_CONFIG", "WEAVE_GEMM2_CONFIG"}) {
        for (const char* bad : {"64:100", "64:72", "64:32:64", "64:32:128:1", "64:32:128:17",
                                "64:32:128:6:3", "64:32:128:6:2:1", "abc", "64:", ":32",
                                "100", "-1", "4097", "1.5", "64;32"}) {
            clear_env();
            set_env(env, bad);
            expect_reject_all(std::string(env) + "=" + bad);
        }
    }

    // BK / STAGES / EPI are launch-wide fields of one kernel: the two phases
    // may pin them independently here, the planner rejects a disagreement.
    clear_env();
    set_env("WEAVE_GEMM1_CONFIG", "64:32:128:6:2");
    set_env("WEAVE_GEMM2_CONFIG", "64:32:256:6:2");
    check(wk::gemm1_config_env().bk != wk::gemm2_config_env().bk,
          "GEMM1 / GEMM2 tuples are parsed independently");

    // Legacy aliases: WEAVE_BLOCK_M_L1/L2 carry a bare BM and resolve like the new names.
    for (int bm : {16, 32, 64, 80, 96, 128, 192, 240, 256}) {
        clear_env();
        set_env("WEAVE_BLOCK_M_L1", std::to_string(bm));
        set_env("WEAVE_BLOCK_M_L2", std::to_string(bm == 16 ? 32 : 16));
        const auto legacy = resolve(sm103);
        check(wk::gemm1_config_env().bm == bm, "WEAVE_BLOCK_M_L1=" + std::to_string(bm) +
              " is read by the GEMM1 config accessor");
        clear_env();
        set_env("WEAVE_GEMM1_CONFIG", std::to_string(bm));
        set_env("WEAVE_GEMM2_CONFIG", std::to_string(bm == 16 ? 32 : 16));
        const auto fresh = resolve(sm103);
        check(knobs_json(legacy) == knobs_json(fresh),
              "WEAVE_BLOCK_M_L1/L2 == WEAVE_GEMM1/2_CONFIG for BM " + std::to_string(bm));
    }
    // The new name outranks the legacy one, per phase.
    clear_env();
    set_env("WEAVE_GEMM1_CONFIG", "64");
    set_env("WEAVE_BLOCK_M_L1", "128");
    set_env("WEAVE_GEMM2_CONFIG", "32");
    set_env("WEAVE_BLOCK_M_L2", "96");
    {
        const auto knobs = resolve(sm103);
        check(knobs.block_m_l1 == 64u and wk::gemm1_config_env().bm == 64,
              "WEAVE_GEMM1_CONFIG outranks WEAVE_BLOCK_M_L1");
        check(knobs.block_m_l2 == 32u and wk::gemm2_config_env().bm == 32,
              "WEAVE_GEMM2_CONFIG outranks WEAVE_BLOCK_M_L2");
    }
    // An empty new name falls through to the legacy one.
    clear_env();
    set_env("WEAVE_GEMM1_CONFIG", "");
    set_env("WEAVE_BLOCK_M_L1", "96");
    check(resolve(sm103).block_m_l1 == 96u and wk::gemm1_config_env().bm == 96,
          "an empty WEAVE_GEMM1_CONFIG falls back to WEAVE_BLOCK_M_L1");

    // WEAVE_ADAPT_GEMM_CONFIG: -1 = unset (adaptive default), 0 fixed, 1 adaptive.
    clear_env();
    check(wk::adapt_gemm_config_env() == -1, "unset WEAVE_ADAPT_GEMM_CONFIG reports -1 (adaptive default)");
    for (int v : {0, 1}) {
        clear_env();
        set_env("WEAVE_ADAPT_GEMM_CONFIG", std::to_string(v));
        check(wk::adapt_gemm_config_env() == v and resolve(sm103).adapt_bm == static_cast<uint32_t>(v),
              "WEAVE_ADAPT_GEMM_CONFIG=" + std::to_string(v));
        clear_env();
        set_env("WEAVE_ADAPT_BM", std::to_string(v));
        check(wk::adapt_gemm_config_env() == v and resolve(sm103).adapt_bm == static_cast<uint32_t>(v),
              "legacy WEAVE_ADAPT_BM=" + std::to_string(v) + " is a transparent alias");
        clear_env();
        set_env("DG_ADAPT_BM", std::to_string(v));
        check(wk::adapt_gemm_config_env() == v,
              "archived DG_ADAPT_BM=" + std::to_string(v) + " is still accepted");
        clear_env();
        set_env("WEAVE_ADAPT_GEMM_CONFIG", std::to_string(v));
        set_env("WEAVE_ADAPT_BM", std::to_string(1 - v));
        check(wk::adapt_gemm_config_env() == v, "WEAVE_ADAPT_GEMM_CONFIG outranks WEAVE_ADAPT_BM");
    }
    clear_env();
    set_env("WEAVE_ADAPT_GEMM_CONFIG", "");
    set_env("WEAVE_ADAPT_BM", "1");
    check(wk::adapt_gemm_config_env() == 1, "an empty WEAVE_ADAPT_GEMM_CONFIG falls back to WEAVE_ADAPT_BM");
    for (const char* bad : {"2", "-1", "abc", "1.5"}) {
        clear_env();
        set_env("WEAVE_ADAPT_GEMM_CONFIG", bad);
        expect_reject_all(std::string("WEAVE_ADAPT_GEMM_CONFIG=") + bad);
        clear_env();
        set_env("WEAVE_ADAPT_BM", bad);
        expect_reject_all(std::string("WEAVE_ADAPT_BM=") + bad);
    }
}

// ---------------------------------------------------------------------------
// 6c. Retired knobs: not public, but still honoured as internal pins so that
//     existing launch scripts and recorded winner presets keep resolving.
// ---------------------------------------------------------------------------
void test_retired_internal_knobs() {
    section("retired knobs: internal macros and env diagnostics only");
    const auto sm103 = weave::HardwareArch::B300_SM100;
    clear_env();
    {
        const auto knobs = resolve(sm103);
        check(knobs.reduce_warps == 0u and knobs.combine_sync_chunk == 0u and
                  knobs.reg_balance == 0u and knobs.local_scatter_dedup == 0u and
                  knobs.sm0_stream_sched == 0u,
              "unset retired knobs keep their internal default 0");
    }
    check(retired_internal_knobs().size() == 6, "exactly six knobs are retired");
    for (const auto& retired : retired_internal_knobs())
        for (const auto& pub : public_knobs())
            check(retired != pub, retired + " is not a public knob");
    for (const auto& alias : legacy_aliases())
        for (const auto& pub : public_knobs())
            check(alias.first != pub, alias.first + " is a legacy alias, not a public knob");
    check(public_knobs().size() == 14, "the public surface has exactly 14 knobs");
    for (const auto& compat : internal_compat_knobs())
        for (const auto& pub : public_knobs())
            check(compat != pub, compat + " is an accepted internal compat name, not a public knob");

    struct Pin { const char* env; const char* value; uint32_t weave::WeaveKnobs::*field; uint32_t expect; };
    const std::vector<Pin> pins = {
        {"WEAVE_REDUCE_WARPS", "4", &weave::WeaveKnobs::reduce_warps, 4u},
        {"WEAVE_COMBINE_SYNC_CHUNK", "0", &weave::WeaveKnobs::combine_sync_chunk, 0u},
        {"WEAVE_REG_BALANCE", "1", &weave::WeaveKnobs::reg_balance, 1u},
        {"WEAVE_SM0_STREAM_SCHED", "1", &weave::WeaveKnobs::sm0_stream_sched, 1u},
        {"WEAVE_READY_QUEUE", "sm0_fifo", &weave::WeaveKnobs::sm0_stream_sched, 1u},
        {"WEAVE_READY_QUEUE", "formula", &weave::WeaveKnobs::sm0_stream_sched, 0u},
    };
    for (const auto& pin : pins) {
        clear_env();
        set_env(pin.env, pin.value);
        check(resolve(sm103).*(pin.field) == pin.expect,
              std::string("internal pin ") + pin.env + "=" + pin.value + " is still honoured");
    }
    clear_env();
    set_env("WEAVE_LOCAL_SCATTER_DEDUP", "1");
    {
        const auto knobs = resolve(sm103);
        check(knobs.local_scatter_dedup == 1u and knobs.scatter_staging == 2u and knobs.scatter_owner == 1u,
              "internal WEAVE_LOCAL_SCATTER_DEDUP=1 still selects the comm-owned hbm_dedup path");
    }
}

// ---------------------------------------------------------------------------
// 7. Environment spelling: empty = unset, malformed = fail-closed, and the
//    archived DG_* aliases never outrank the public WEAVE_* name.
// ---------------------------------------------------------------------------
void test_env_parsing() {
    section("environment spelling / legacy aliases");
    for (const char* bad : {"abc", "4x", "0x10", "1.5", "+", "-",
                            "2147483648", "99999999999999999999"}) {
        clear_env();
        set_env("WEAVE_DISPATCH_WARPS", bad);
        expect_reject_all(std::string("WEAVE_DISPATCH_WARPS=") + bad + " (malformed integer)");
    }
    clear_env();
    set_env("WEAVE_DISPATCH_WARPS", " 8 ");
    check(resolve(weave::HardwareArch::B300_SM100).dispatch_warps == 8u,
          "surrounding whitespace is tolerated");
    struct Alias { const char* primary; const char* legacy; const char* value; };
    const std::vector<Alias> aliases = {
        {"WEAVE_DISPATCH_WARPS", "DG_DISPATCH_WARPS", "8"},
        {"WEAVE_REDUCE_WARPS", "DG_REDUCE_WARPS", "4"},
        {"WEAVE_SCATTER_ACTIVE_SMS", "DG_SCATTER_ACTIVE_SMS", "16"},
        {"WEAVE_ADAPT_BM", "DG_ADAPT_BM", "1"},
        {"WEAVE_BLOCK_M_L1", "DG_BLOCK_M_L1", "128"},
        {"WEAVE_BLOCK_M_L2", "DG_BLOCK_M_L2", "96"},
        {"WEAVE_LOCAL_SCATTER_DEDUP", "DG_LOCAL_SCATTER_DEDUP", "1"},
        {"WEAVE_V_HEAD", "DG_V_HEAD", "64"},
    };
    for (const auto& a : aliases) {
        clear_env();
        set_env(a.legacy, a.value);
        const auto legacy_only = resolve(weave::HardwareArch::B300_SM100);
        clear_env();
        set_env(a.primary, a.value);
        const auto primary_only = resolve(weave::HardwareArch::B300_SM100);
        check(knobs_json(legacy_only) == knobs_json(primary_only),
              std::string(a.legacy) + " is equivalent to " + a.primary);
        clear_env();
        set_env(a.legacy, "0");
        set_env(a.primary, a.value);
        const auto both = resolve(weave::HardwareArch::B300_SM100);
        check(knobs_json(both) == knobs_json(primary_only),
              std::string(a.primary) + " outranks " + a.legacy);
    }
}

// ---------------------------------------------------------------------------
// 8. Constexpr resolvers used by the dispatchers and the cost model.
// ---------------------------------------------------------------------------
void test_resolvers() {
    section("constexpr resolvers");
    clear_env();
    auto k = weave::default_weave_knobs(weave::HardwareArch::B300_SM100);
    check(weave::resolve_dispatch_warps(k, 4u) == 4u, "auto dispatch warps");
    k.dispatch_warps = 8u;
    check(weave::resolve_dispatch_warps(k, 4u) == 8u, "explicit dispatch warps");
    k = weave::default_weave_knobs(weave::HardwareArch::B300_SM100);
    check(weave::resolve_scatter_active_sms(k, 148u) == 148u, "auto scatter SMs = all");
    k.scatter_active_sms = 32u;
    check(weave::resolve_scatter_active_sms(k, 148u) == 32u, "throttled scatter SMs");
    k.scatter_active_sms = 200u;
    check(weave::resolve_scatter_active_sms(k, 148u) == 148u, "over-subscribed scatter SMs clamp");
    k = weave::default_weave_knobs(weave::HardwareArch::B300_SM100);
    check(weave::resolve_dispatch_active_sms(k, 148u) == 148u, "auto dispatch SMs = all");
    k.dispatch_active_sms = 16u;
    check(weave::resolve_dispatch_active_sms(k, 148u) == 16u, "throttled dispatch SMs");
    k = weave::default_weave_knobs(weave::HardwareArch::MI308X_GFX942);
    check(weave::resolve_locality_domains(k, weave::HardwareArch::MI308X_GFX942) == 4u,
          "MI308X auto locality = 4 XCDs");
    check(weave::resolve_locality_domains(k, weave::HardwareArch::B300_SM100) == 2u,
          "SM103 auto locality = 2 dies");
    check(weave::resolve_locality_domains(k, weave::HardwareArch::H20_SM90) == 1u,
          "H20 auto locality = 1");
    k = weave::default_weave_knobs(weave::HardwareArch::B300_SM100);
    check(weave::resolve_block_m(k, 0u, 0u, 64u) == 64u, "auto blockM");
    check(weave::resolve_block_m(k, 32u, 0u, 64u) == 32u, "L1 blockM override");
    check(weave::resolve_block_m(k, 32u, 128u, 64u) == 128u, "L2 blockM wins for GEMM2");
    check(weave::resolve_store_block_m(k, 0u, 0u, 16u, 32u) == 32u, "auto store height");
    check(weave::resolve_store_block_m(k, 24u, 0u, 16u, 32u) == 24u, "L1 store height");
    check(weave::resolve_store_block_m(k, 24u, 40u, 16u, 32u) == 40u, "L2 store height");
    check(weave::resolve_epi_reg_stages(k, 64u, 200u) == 1u, "small tile keeps one reg stage");
    check(weave::resolve_epi_reg_stages(k, 240u, 200u) == 2u, "BM=240 auto two reg stages");
    check(weave::resolve_epi_reg_stages(k, 256u, 175u) == 2u, "BM=256 @1.75 two reg stages");
    check(weave::resolve_epi_reg_stages(k, 256u, 100u) == 1u, "BM=256 @1.0 one reg stage");
    k.epi_reg_stages = 2u;
    check(weave::resolve_epi_reg_stages(k, 16u, 100u) == 2u, "explicit reg stages win");
    check(deep_gemm::weave_knobs::resolve_tmem_pipeline_mode(k, 128u) == 200u,
          "auto TMEM depth below BM=256 is 2.0");
    check(deep_gemm::weave_knobs::resolve_tmem_pipeline_mode(k, 256u) == 175u,
          "auto TMEM depth at BM=256 is 1.75");
    k.tmem_pipeline_mode = 100u;
    check(deep_gemm::weave_knobs::resolve_tmem_pipeline_mode(k, 256u) == 100u,
          "explicit TMEM depth wins");
    check(std::strcmp(deep_gemm::weave_knobs::tmem_pipeline_mode_name(100u), "1.0") == 0 and
              std::strcmp(deep_gemm::weave_knobs::tmem_pipeline_mode_name(175u), "1.75") == 0 and
              std::strcmp(deep_gemm::weave_knobs::tmem_pipeline_mode_name(200u), "2.0") == 0,
          "TMEM depth names");
    check(weave::resolve_ring_blocks(k, 7u) == 7u, "auto ring blocks");
    k.ring_blocks = 3u;
    check(weave::resolve_ring_blocks(k, 7u) == 3u, "explicit ring blocks");
    check(weave::resolve_amd_8wave_min_tok(k, 64u) == 64u, "auto AMD 8-wave floor");
    k.amd_8wave_min_tok = 128u;
    check(weave::resolve_amd_8wave_min_tok(k, 64u) == 128u, "explicit AMD 8-wave floor");
}

// ---------------------------------------------------------------------------
// 9. SM90 spec parser (verbatim slice of sm90_mega_moe.hpp).
// ---------------------------------------------------------------------------
#ifdef WEAVE_SURFACE_SM90_SPEC_SLICE
std::string spec_json(const deep_gemm::AgySm90KnobSpec& spec) {
    std::string out = "{\"enabled\": " + std::string(spec.enabled ? "true" : "false") +
                      ", \"raw\": \"" + json_escape(spec.raw) + "\", \"kv\": {";
    bool first = true;
    for (const auto& [key, value] : spec.kv) {
        if (not first) out += ", ";
        first = false;
        out += "\"" + json_escape(key) + "\": \"" + json_escape(value) + "\"";
    }
    return out + "}}";
}

void test_sm90_spec_parser() {
    section("SM90 spec parser: public knob mapping");
    clear_env();
    check(not deep_gemm::has_any_public_weave_sm90_env(),
          "no public WEAVE_* env -> the unified SM90 path stays off");
    check(not deep_gemm::get_agy_sm90_spec().enabled,
          "no env -> no spec (auto regime is opt-in)");

    struct Mapping { const char* env; const char* value; const char* key; const char* code; };
    const std::vector<Mapping> mappings = {
        {"WEAVE_DISPATCH_ACTIVE_SMS", "16", "dispatch_sms", "16"},
        {"WEAVE_SCATTER_ACTIVE_SMS", "8", "scatter_sms", "8"},
        {"WEAVE_SM0_STREAM_SCHED", "1", "sm0_sched", "1"},
        {"WEAVE_DISPATCH_WARPS", "8", "dispatch_warps", "8"},
        {"WEAVE_SCATTER_WARPS", "4", "scatter_warps", "4"},
        {"WEAVE_REDUCE_WARPS", "4", "comb_warps", "4"},
        {"WEAVE_REG_BALANCE", "1", "reg_balance", "1"},
        {"WEAVE_V_HEAD", "64", "v_head", "64"},
        {"WEAVE_READY_GROUP_SIZE", "2", "meta_g", "2"},
        {"WEAVE_READY_QUEUE", "sm0_fifo", "ready_queue", "1"},
        {"WEAVE_SCATTER_OWNER", "comm_only", "scatter_owner", "1"},
        {"WEAVE_COMBINE_SYNC_CHUNK", "32", "comb_sync_chunk", "32"},
        {"WEAVE_ADAPT_BM", "1", "adapt_bm", "1"},
        {"WEAVE_BLOCK_M_L1", "32", "l1_bm", "32"},
        {"WEAVE_BLOCK_M_L2", "64", "l2_bm", "64"},
        {"WEAVE_ADAPT_GEMM_CONFIG", "1", "adapt_bm", "1"},
        {"WEAVE_GEMM1_CONFIG", "32", "l1_bm", "32"},
        {"WEAVE_GEMM2_CONFIG", "64", "l2_bm", "64"},
        {"WEAVE_DEDUP_MODE", "1", "dedup_disp", "1"},
        {"WEAVE_SCATTER_STAGING", "hbm_dedup", "scatter_staging", "2"},
        {"WEAVE_LOCAL_SCATTER_DEDUP", "1", "local_dedup", "1"},
    };
    for (const auto& m : mappings) {
        clear_env();
        set_env(m.env, m.value);
        check(deep_gemm::has_any_public_weave_sm90_env(),
              std::string(m.env) + " is recognised as a public SM90 knob");
        const auto spec = deep_gemm::get_agy_sm90_spec();
        check(spec.enabled, std::string(m.env) + " enables the unified SM90 spec");
        check(spec.get_int(m.key, -1) == std::atoi(m.code),
              std::string(m.env) + "=" + m.value + " -> " + m.key + "=" + m.code);
        check(spec.raw.find("fast_host=1") != std::string::npos and
                  spec.raw.find("fused=1") != std::string::npos,
              "the unified SM90 skeleton is preserved");
    }
    for (const auto& m : std::vector<Mapping>{
             {"WEAVE_READY_QUEUE", "1", "ready_queue", "1"},
             {"WEAVE_SCATTER_OWNER", "1", "scatter_owner", "1"},
             {"WEAVE_SCATTER_STAGING", "2", "scatter_staging", "2"}}) {
        clear_env();
        set_env(m.env, m.value);
        const auto spec = deep_gemm::get_agy_sm90_spec();
        check(spec.get_int(m.key, -1) == std::atoi(m.code),
              std::string(m.env) + "=" + m.value + " (code form)");
    }
    clear_env();
    set_env("WEAVE_READY_QUEUE", "formula");
    {
        const auto spec = deep_gemm::get_agy_sm90_spec();
        check(spec.enabled and spec.get_int("ready_queue_formula", 0) == 1,
              "READY_QUEUE=formula publishes ready_queue_formula=1");
    }
    clear_env();
    set_env("WEAVE_SCATTER_ACTIVE_SMS", "0");
    {
        const auto spec = deep_gemm::get_agy_sm90_spec();
        check(spec.enabled and not spec.has("scatter_sms"),
              "an explicit zero opts into the unified skeleton without a numeric override");
    }
    for (const char* bad : {"foo", "a=", "=1", "scatter_warps=6", "scatter_warps=2",
                            "dispatch_warps=6", "meta_g=-1", "comb_sync_chunk=7",
                            "comb_warps=3", "l1_bm=100", "l2_bm=257", "adapt_bm=2",
                            "reg_balance=2", "local_dedup=2", "dedup_disp=2",
                            "scatter_owner=2", "scatter_staging=3", "ready_queue=2",
                            "v_head=-1", "dispatch_sms=-1", "scatter_sms=-1"}) {
        clear_env();
        bool threw = false;
        try {
            (void) deep_gemm::parse_agy_sm90_spec_string(bad);
        } catch (const deep_gemm::DGException&) {
            threw = true;
        }
        check(threw, std::string("SM90 spec \"") + bad + "\" must be rejected");
    }
    clear_env();
    set_env("WEAVE_SCATTER_STAGING", "smem");
    set_env("WEAVE_LOCAL_SCATTER_DEDUP", "1");
    {
        bool threw = false;
        try {
            (void) deep_gemm::get_agy_sm90_spec().resolved_scatter_staging();
        } catch (const deep_gemm::DGException&) { threw = true; }
        check(threw, "SM90 rejects smem staging with local dedup");
    }
    clear_env();
    set_env("WEAVE_SCATTER_OWNER", "epilogue_only");
    set_env("WEAVE_SCATTER_STAGING", "smem");
    {
        bool threw = false;
        try {
            (void) deep_gemm::get_agy_sm90_spec().resolved_scatter_owner();
        } catch (const deep_gemm::DGException&) { threw = true; }
        check(threw, "SM90 rejects epilogue_only + smem staging");
    }
    clear_env();
    set_env("WEAVE_SCATTER_ACTIVE_SMS", "8");
    set_env("WEAVE_SCATTER_STAGING", "reg_2stage");
    {
        bool threw = false;
        try {
            (void) deep_gemm::get_agy_sm90_spec().resolved_scatter_staging();
        } catch (const deep_gemm::DGException&) { threw = true; }
        check(threw, "SM90 rejects a throttled scatter subset on reg_2stage");
    }
    clear_env();
    set_env("WEAVE_SCATTER_ACTIVE_SMS", "8");
    {
        const auto spec = deep_gemm::get_agy_sm90_spec();
        check(spec.resolved_scatter_staging() == 2 and spec.resolved_scatter_owner() == 1,
              "SM90 auto-selects hbm_dedup + comm owner for a throttled scatter subset");
    }
    clear_env();
    set_env("WEAVE_LOCAL_SCATTER_DEDUP", "1");
    {
        const auto spec = deep_gemm::get_agy_sm90_spec();
        check(spec.resolved_scatter_staging() == 2 and spec.resolved_scatter_owner() == 1,
              "SM90 auto-selects hbm_dedup for local dedup");
    }
    clear_env();
    set_env("WEAVE_READY_QUEUE", "sm0_fifo");
    set_env("WEAVE_FORCE_OFFICIAL", "1");
    check(not deep_gemm::get_agy_sm90_spec().enabled, "WEAVE_FORCE_OFFICIAL=1 disables the spec");
    clear_env();
    set_env("WEAVE_H20_CONFIG", "bm=64,bn=128,dedup=1");
    {
        const auto spec = deep_gemm::get_agy_sm90_spec();
        check(spec.enabled and spec.get_int("bm", 0) == 64 and
                  spec.get_int("dedup_disp", 0) == 1,
              "WEAVE_H20_CONFIG parses and aliases dedup -> dedup_disp");
    }
    clear_env();
    set_env("WEAVE_BLOCK_M_L1", "32");
    const auto a = deep_gemm::get_agy_sm90_spec();
    clear_env();
    set_env("WEAVE_BLOCK_M_L1", "32");
    const auto b = deep_gemm::get_agy_sm90_spec();
    clear_env();
    set_env("WEAVE_BLOCK_M_L1", "64");
    const auto c = deep_gemm::get_agy_sm90_spec();
    check(a.raw_hash == b.raw_hash and a.raw_hash != c.raw_hash,
          "identical knob requests share a JIT tag hash; different ones do not");
}
#endif  // WEAVE_SURFACE_SM90_SPEC_SLICE

// ---------------------------------------------------------------------------
// CLI helpers used by the Python driver.
// ---------------------------------------------------------------------------
void apply_kv_args(int argc, char** argv, int start) {
    for (int i = start; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto eq = arg.find('=');
        if (eq == std::string::npos or eq == 0) {
            std::fprintf(stderr, "expected KEY=VALUE, got: %s\n", arg.c_str());
            std::exit(2);
        }
        set_env(arg.substr(0, eq).c_str(), arg.substr(eq + 1));
    }
}

int run_resolve(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: --resolve ARCH PREC [KEY=VALUE...]\n");
        return 2;
    }
    clear_env();
    apply_kv_args(argc, argv, 4);
    try {
        const auto knobs = resolve(arch_of(argv[2]), prec_of(argv[3]));
        std::printf("%s\n", knobs_json(knobs).c_str());
        return 0;
    } catch (const deep_gemm::DGException& e) {
        std::printf("{\"error\": \"%s\"}\n", json_escape(e.what()).c_str());
        return 3;
    }
}

#ifdef WEAVE_SURFACE_SM90_SPEC_SLICE
int run_sm90_spec(int argc, char** argv) {
    clear_env();
    apply_kv_args(argc, argv, 2);
    try {
        const auto spec = deep_gemm::get_agy_sm90_spec();
        std::printf("%s\n", spec_json(spec).c_str());
        return 0;
    } catch (const deep_gemm::DGException& e) {
        std::printf("{\"error\": \"%s\"}\n", json_escape(e.what()).c_str());
        return 3;
    }
}

int run_sm90_resolved(int argc, char** argv) {
    clear_env();
    apply_kv_args(argc, argv, 2);
    try {
        const auto spec = deep_gemm::get_agy_sm90_spec();
        const int staging = spec.resolved_scatter_staging();
        const int owner = spec.resolved_scatter_owner();
        std::printf("{\"enabled\": %s, \"staging\": %d, \"owner\": %d}\n",
                    spec.enabled ? "true" : "false", staging, owner);
        return 0;
    } catch (const deep_gemm::DGException& e) {
        std::printf("{\"error\": \"%s\"}\n", json_escape(e.what()).c_str());
        return 3;
    }
}
#endif

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 2 and std::strcmp(argv[1], "--resolve") == 0)
        return run_resolve(argc, argv);
#ifdef WEAVE_SURFACE_SM90_SPEC_SLICE
    if (argc >= 2 and std::strcmp(argv[1], "--sm90-spec") == 0)
        return run_sm90_spec(argc, argv);
    if (argc >= 2 and std::strcmp(argv[1], "--sm90-resolved") == 0)
        return run_sm90_resolved(argc, argv);
#endif
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--quiet") == 0)
            g_quiet = true;

    std::printf("Weave 14-knob surface verification (host C++, no GPU)\n");
    test_defaults();
    test_integer_values();
    test_named_enums();
    test_precision_mode();
    test_v_head();
    test_combine_sync_chunk();
    test_scatter_coupling();
    test_gemm_config_surface();
    test_retired_internal_knobs();
    test_env_parsing();
    test_resolvers();
#ifdef WEAVE_SURFACE_SM90_SPEC_SLICE
    test_sm90_spec_parser();
#else
    std::printf("\n[SKIP] SM90 spec parser slice (build with -DWEAVE_SURFACE_SM90_SPEC_SLICE)\n");
#endif
    std::printf("\nCHECKS=%d FAILURES=%d\n", g_checks, g_failures);
    if (g_failures == 0)
        std::printf("ALL 14-KNOB HOST SURFACE CHECKS PASS\n");
    return g_failures == 0 ? 0 : 1;
}

#if defined(WEAVE_SURFACE_SHARED)
#if defined(_WIN32)
#define WEAVE_SURFACE_EXPORT __declspec(dllexport)
#else
#define WEAVE_SURFACE_EXPORT __attribute__((visibility("default")))
#endif
// In-process entry point for the Python driver on hosts whose application
// control policy blocks freshly built executables but still loads DLLs.
extern "C" WEAVE_SURFACE_EXPORT int weave_surface_entry(int argc, char** argv) {
    const int rc = main(argc, argv);
    std::fflush(stdout);
    std::fflush(stderr);
    return rc;
}
#endif
