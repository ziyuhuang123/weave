// Compile-only (PTX) instantiation check of the SM100 BF16 MegaMoE kernel body.
// Template argument order mirrors the JIT string in
// csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp; defaults mirror
// get_mega_moe_config (csrc/jit_kernels/heuristics/mega_moe.hpp) for MmaKind::BF16,
// block_m=64 profile: BLOCK_K = 128 bytes / 2 = 64 BF16 elements, 1 epilogue warpgroup.
// Override any knob with -D, e.g. -DT_DISPATCH_THREADS=256 -DWEAVE_DISPATCH_WARPS=8.
// No WEAVE_* macro is defined here; pass -DWEAVE_...=... on the command line if needed.
#ifndef T_BM
#define T_BM 64
#endif
#ifndef T_STORE_M
#define T_STORE_M 32
#endif
#ifndef T_BLOCK_K
#define T_BLOCK_K 64             // in BF16 elements
#endif
#ifndef T_STAGES
#define T_STAGES 6
#endif
#ifndef T_DISPATCH_THREADS
#define T_DISPATCH_THREADS 128
#endif
#ifndef T_NON_EPI_THREADS
#define T_NON_EPI_THREADS 128
#endif
#ifndef T_EPI_THREADS
#define T_EPI_THREADS 128
#endif
#ifndef T_NUM_EXPERTS
#define T_NUM_EXPERTS 256
#endif
#ifndef T_NUM_SMS
#define T_NUM_SMS 148
#endif
#ifndef T_NUM_RANKS
#define T_NUM_RANKS 8
#endif
#ifndef T_NUM_TOPK
#define T_NUM_TOPK 8
#endif
#ifndef T_MAX_TOKENS
#define T_MAX_TOKENS 1024
#endif
#ifndef T_HIDDEN
#define T_HIDDEN 4096
#endif
#ifndef T_INTER
#define T_INTER 2048
#endif
#ifndef T_NUM_SHARED
#define T_NUM_SHARED 0
#endif
#ifndef T_RING_BLOCKS
#define T_RING_BLOCKS 64
#endif
#ifndef T_PULL_BYTES
// heuristic: hidden*2 bytes halved until <= 4096
#define T_PULL_BYTES ((T_HIDDEN * 2) > 4096 ? ((T_HIDDEN) > 4096 ? ((T_HIDDEN / 2) > 4096 ? (T_HIDDEN / 4) : (T_HIDDEN / 2)) : T_HIDDEN) : (T_HIDDEN * 2))
#endif

#include <deep_gemm/impls/sm100_bf16_mega_moe.cuh>

using namespace deep_gemm;

constexpr uint32_t kRingTokens = T_BM * T_RING_BLOCKS;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_impl<
        T_MAX_TOKENS,
        T_HIDDEN, T_INTER,
        T_NUM_EXPERTS, T_NUM_SHARED,
        T_NUM_TOPK,
        T_BM, 128, T_BLOCK_K,
        T_STORE_M,
        kRingTokens,
        T_STAGES,
        T_PULL_BYTES,
        T_DISPATCH_THREADS, T_NON_EPI_THREADS, T_EPI_THREADS,
        T_NUM_SMS, T_NUM_RANKS,
        10.0f,
        true>);
    (void)ptr;
}
