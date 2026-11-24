// H20 V9 host-side launch emitter for the sm_90a compile gate.
//
// Runs the LIVE SM90 host heuristics (select_mega_moe_sm90 + get_agy_sm90_spec +
// get_agy_sm90_jit_defines from csrc/jit_kernels/heuristics/sm90_mega_moe.hpp)
// under the requested WEAVE_* environment and writes, for every kernel the
// runtime would launch, the exact JIT source that
// SM90FP8MegaMoERuntime::compile_only (csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp)
// would hand to nvcc: the AGY define block followed by the template
// instantiation.  The instantiation format literal is extracted verbatim from
// that file by tests/check_h20_v9_compile.py into the generated
// h20_v9_instantiate_fmt.hpp (H20_V9_INSTANTIATE_FMT), so BLOCK_M / BLOCK_N / BLOCK_K / stages / thread
// layout / epw / swap-AB / N-major are never hand-written in the gate.
//
// Usage: h20_v9_host_emitter <tokens> <out_prefix> [KEY=VALUE ...]
// stdout: one "KEY=VALUE" summary line per field; files <out_prefix>_<kernel>.cu
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <deep_gemm/layout/sym_buffer.cuh>
#include "sm90_mega_moe_host.hpp"
#include "h20_v9_instantiate_fmt.hpp"

#ifndef H20_V9_INSTANTIATE_FMT
#error "H20_V9_INSTANTIATE_FMT must carry the compile_only format literal"
#endif

using namespace deep_gemm;

namespace {

// Mirrors csrc/jit_kernels/impls/runtime_utils.hpp:to_string(const float&).
std::string float_literal(const float v) {
    return std::format("{}0x{:a}f", std::signbit(v) ? "-" : "", std::abs(v));
}

// H20 EP4 Qwen3.8-Max fixture (prior H20 V8 winner fixture; see results/h20_ep4_prior_verified.json for the EP4 shape).
constexpr int kNumSms = 78, kNumRanks = 4, kNumExperts = 512, kNumTopk = 10;
constexpr int kHidden = 8192, kIntermediate = 2048;
constexpr float kActivationClamp = 10.0f;
constexpr bool kFastMath = true;

enum class Phase { Linear1, Linear2, Fused };

std::string render(const Phase phase, const MegaMoESM90Config& cfg,
                   const bool l1_nmajor_for_fused, const bool bf16_scaled_accum,
                   const int num_max_tokens_per_rank, const std::string& defines) {
    const char* kernel_symbol =
        phase == Phase::Linear1 ? "sm90_fp8_mega_moe_l1_impl" :
        (phase == Phase::Linear2 ? "sm90_fp8_mega_moe_l2_impl" : "sm90_fp8_mega_moe_fused_impl");
    // Same phase tail as SM90FP8MegaMoERuntime::compile_only.
    const auto phase_template_args =
        phase == Phase::Linear1 ?
            std::format(",\n        {}", cfg.nmajor_schedule ? "true" : "false") :
        (phase == Phase::Linear2 ?
            std::format(",\n        {}, {}, {}",
                        cfg.direct_l2_scatter ? "true" : "false",
                        cfg.nmajor_schedule ? "true" : "false",
                        cfg.one_warp_cleanup ? "true" : "false") :
            std::format(",\n        {}, {}, {}, {}",
                        cfg.direct_l2_scatter ? "true" : "false",
                        cfg.nmajor_schedule ? "true" : "false",
                        cfg.one_warp_cleanup ? "true" : "false",
                        l1_nmajor_for_fused ? "true" : "false"));
    return defines + std::format(H20_V9_INSTANTIATE_FMT,
        kernel_symbol,
        num_max_tokens_per_rank,
        kHidden, kIntermediate,
        kNumExperts, kNumTopk,
        cfg.num_experts_per_wave,
        cfg.block_m, cfg.block_n, cfg.block_k,
        cfg.num_max_pool_tokens,
        cfg.num_padded_sf_pool_tokens,
        cfg.sf_pool_stride_tokens,
        cfg.num_stages,
        cfg.num_dispatch_threads, cfg.num_non_epilogue_threads, cfg.num_epilogue_threads,
        cfg.num_sms, kNumRanks,
        float_literal(kActivationClamp),
        kFastMath ? "true" : "false",
        cfg.swap_ab ? "true" : "false",
        bf16_scaled_accum ? "true" : "false",
        phase_template_args);
}

void print_cfg(const char* tag, const MegaMoESM90Config& c) {
    std::printf("%s_BM=%d\n%s_BN=%d\n%s_BK=%d\n%s_STAGES=%d\n%s_SMEM=%d\n"
                "%s_DISP=%d\n%s_NONEPI=%d\n%s_EPI=%d\n%s_SWAP=%d\n%s_NMAJOR=%d\n"
                "%s_EPW=%d\n%s_SF_STRIDE=%d\n%s_POOL=%d\n%s_DIRECT_L2=%d\n",
                tag, c.block_m, tag, c.block_n, tag, c.block_k, tag, c.num_stages,
                tag, c.smem_size, tag, c.num_dispatch_threads, tag, c.num_non_epilogue_threads,
                tag, c.num_epilogue_threads, tag, c.swap_ab ? 1 : 0, tag, c.nmajor_schedule ? 1 : 0,
                tag, c.num_experts_per_wave, tag, c.sf_pool_stride_tokens,
                tag, c.num_max_pool_tokens, tag, c.direct_l2_scatter ? 1 : 0);
}

bool write_file(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::binary);
    f << text;
    return static_cast<bool>(f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <tokens> <out_prefix> [KEY=VALUE ...]\n", argv[0]);
        return 2;
    }
    const int tokens = std::atoi(argv[1]);
    const std::string prefix = argv[2];
    for (int i = 3; i < argc; ++i) {
        char* eq = std::strchr(argv[i], '=');
        if (eq == nullptr) return 2;
        *eq = '\0';
        setenv(argv[i], eq + 1, 1);
    }

    // Same input construction as csrc/apis/sm90_mega.hpp (fp8_mega_moe and
    // fp8_mega_moe_compile_only): the SF pool is allocated for BLOCK_M=64.
    const int epr = kNumExperts / kNumRanks;
    const int pool = layout::nv_moe::get_num_max_pool_tokens(kNumRanks, tokens, kNumTopk, epr);
    const int padded_sf = layout::nv_moe::get_num_sf_ring_tokens(pool, 64);
    const Sm90MoeHeuristicInput input{
        kNumSms, kNumRanks, kNumExperts, epr, tokens, tokens, kNumTopk,
        kHidden, kIntermediate, padded_sf};

    const auto spec = get_agy_sm90_spec(input);
    const auto launch = select_mega_moe_sm90(input);
    const std::string defines = get_agy_sm90_jit_defines(spec);

    std::printf("SPEC_ENABLED=%d\n", spec.enabled ? 1 : 0);
    std::printf("SPEC_RAW=%s\n", spec.raw.c_str());
    std::printf("FUSED=%d\n", launch.fused_single_kernel ? 1 : 0);
    std::printf("BF16_ACC=%d\n", launch.numerical.bf16_scaled_accum ? 1 : 0);
    std::printf("STAGING=%d\nOWNER=%d\nCOMB_WARPS=%d\n",
                spec.resolved_scatter_staging(), spec.resolved_scatter_owner(),
                spec.resolved_comb_warps());
    print_cfg("L1", launch.l1);
    print_cfg("L2", launch.l2);

    const bool bf16 = launch.numerical.bf16_scaled_accum;
    if (launch.fused_single_kernel) {
        // compile_only uses the L2 config for the fused kernel.
        const auto src = render(Phase::Fused, launch.l2, launch.l1.nmajor_schedule, bf16, tokens, defines);
        if (not write_file(prefix + "_fused.cu", src)) return 4;
        std::printf("KERNELS=fused\n");
    } else {
        const auto s1 = render(Phase::Linear1, launch.l1, launch.l1.nmajor_schedule, bf16, tokens, defines);
        const auto s2 = render(Phase::Linear2, launch.l2, launch.l1.nmajor_schedule, bf16, tokens, defines);
        if (not write_file(prefix + "_l1.cu", s1) or not write_file(prefix + "_l2.cu", s2)) return 4;
        std::printf("KERNELS=l1,l2\n");
    }
    // Defines as one escaped line so the driver can diff them across arms.
    std::string flat;
    for (const char c : defines) flat += (c == '\n') ? '|' : c;
    std::printf("DEFINES=%s\n", flat.c_str());
    return 0;
}
