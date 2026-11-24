#pragma once

// ============================================================================
// Subagent 6: 6-Model Architecture & ABI Generalization Header
// Convergence target: ALL 6 MODELS x 5 SEQLENS = 30 CELLS
//
// Models audited:
//  1. minimax_m3:        H=6144, I=3072, E=128, K=4,  E_local=16, n1=24, n2=48
//  2. qwen38max:          H=8192, I=2048, E=512, K=10, E_local=64, n1=16, n2=64
//  3. DeepSeek-V4-Flash:  H=4096, I=2048, E=256, K=6,  E_local=32, n1=16, n2=32
//  4. Qwen3.5-35B-A3B:    H=2048, I=512,  E=256, K=8,  E_local=32, n1=4,  n2=16
//  5. Qwen3.5-122B-A10B:  H=3072, I=1024, E=256, K=8,  E_local=32, n1=8,  n2=24
//  6. Qwen3.5-397B-A17B:  H=4096, I=1024, E=512, K=10, E_local=64, n1=8,  n2=32
//
// Seqlens: S in {16, 128, 1024, 8192, 16384} (EP8 -> capacity in {2, 16, 128, 1024, 2048})
// ============================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stddef.h>

// ----------------------------------------------------------------------------
// 1. Generalized Capacity and Top-K Limits
// ----------------------------------------------------------------------------
#ifdef V809_MAX_TOPK
#undef V809_MAX_TOPK
#endif
#define V809_MAX_TOPK 12u

#define V809_MAX_WORLD 16u
#define V809_MAX_LOCAL_EXPERTS 64u
#define V809_MAX_SCALE_GROUPS 64u
#define V809_MAX_N1_BLOCKS 24u
#define V809_MAX_N2_BLOCKS 64u
#define V809_META_STRIDE 64u
#define V809_META_HEADER 4096u

// ----------------------------------------------------------------------------
// 2. Generalized V809Meta Struct
// Fixed 64-byte layout matching host MaxIpcArena and device IPC strides.
// ----------------------------------------------------------------------------
#ifndef V809_META_STRUCT_DEFINED
struct alignas(64) V809Meta {
    uint32_t home;                  // offset  0: local source token index [0..t_local)
    uint32_t nexp;                  // offset  4: valid routed experts for this dst [0..topk]
    uint16_t lid[V809_MAX_TOPK];    // offset  8: destination-local expert ids (12 * 2 = 24 B)
    uint16_t wbits[V809_MAX_TOPK];  // offset 32: FP16/BF16 wire weight bits (12 * 2 = 24 B)
    uint8_t  _reserved[8];          // offset 56: padding to preserve exact 64 B stride
};

static_assert(sizeof(V809Meta) == 64, "V809Meta stride must stay exactly 64 bytes");
static_assert(sizeof(V809Meta) == V809_META_STRIDE, "V809Meta size mismatch vs V809_META_STRIDE");
static_assert(offsetof(V809Meta, home) == 0,  "V809Meta::home offset must be 0");
static_assert(offsetof(V809Meta, nexp) == 4,  "V809Meta::nexp offset must be 4");
static_assert(offsetof(V809Meta, lid) == 8,   "V809Meta::lid offset must be 8");
static_assert(offsetof(V809Meta, wbits) == 32, "V809Meta::wbits offset must be 32");
#endif

// Type-safe accessor helpers for pointer arithmetic
__device__ __forceinline__ const V809Meta* v809_get_meta_entry(const void* region_base, unsigned slot) {
    return reinterpret_cast<const V809Meta*>(
        reinterpret_cast<const unsigned char*>(region_base) +
        V809_META_HEADER + static_cast<uint64_t>(slot) * V809_META_STRIDE);
}

__device__ __forceinline__ V809Meta* v809_get_meta_entry_mut(void* region_base, unsigned slot) {
    return reinterpret_cast<V809Meta*>(
        reinterpret_cast<unsigned char*>(region_base) +
        V809_META_HEADER + static_cast<uint64_t>(slot) * V809_META_STRIDE);
}

// ----------------------------------------------------------------------------
// 3. Bitpacking Helpers for row_route_id
// Format: (src * (capacity + 1) + slot) | (k << 24)
//
// Proof of bounds:
//   - Lower 24 bits hold g2p = src * (capacity + 1) + slot.
//     For world = 8, src in [0..7], max capacity = 16384 (S = 131,072 on EP8):
//     g2p_max = 7 * 16385 + 16384 = 131,079 < 2^24 = 16,777,216.
//     Headroom is > 128x.
//   - Upper 8 bits hold k in [0..topk-1]. For K = 10, k in [0..9] <= 15 (4 bits).
//     Well within the 8 bits available in [24..31].
//   - Valid packed IDs never collide with inactive sentinel 0xffffffffu.
// ----------------------------------------------------------------------------
__device__ __forceinline__ unsigned v983_pack_route_id(
    unsigned src, unsigned slot, unsigned capacity, unsigned k)
{
    const unsigned g2p = src * (capacity + 1u) + slot;
    return (g2p & 0x00ffffffu) | ((k & 0xffu) << 24);
}

__device__ __forceinline__ unsigned v983_unpack_route_id_g2p(unsigned id)
{
    return id & 0x00ffffffu;
}

__device__ __forceinline__ unsigned v983_unpack_route_id_k(unsigned id)
{
    return id >> 24;
}

__device__ __forceinline__ void v983_unpack_route_id_src_slot(
    unsigned id, unsigned capacity, unsigned& src, unsigned& slot)
{
    const unsigned g2p = id & 0x00ffffffu;
    const unsigned long long divmod = v983_divmod_u32(g2p, capacity + 1u);
    src = static_cast<unsigned>(divmod >> 32);
    slot = static_cast<unsigned>(divmod);
}

// ----------------------------------------------------------------------------
// 4. LDS Layout & Sizing for E_local <= 64
//
// In v983_route_plan:
//   arena + 0:     hist [world * local_experts] (uint32)
//                  For world=8, local_experts=64: 8 * 64 * 4 = 2,048 bytes (0..2047)
//                  Available headroom before 8192: 6,144 bytes!
//   arena + 8192:  curs [world * local_experts] (uint32)
//                  For world=8, local_experts=64: 8 * 64 * 4 = 2,048 bytes (8192..10239)
//                  Available headroom before CK barrier area (32768): 22,528 bytes!
//   arena + 32768: CK barrier structures and hardware epoch arrivals
//   arena + 65440: Sub1 zero-sbarrier 4-wave barrier
//   arena + 65484: route_plan fail indicator
//   arena + 65504: wait context
//
// Zero collision across all 6 models!
// ----------------------------------------------------------------------------
static constexpr unsigned V809_LDS_HIST_OFFSET = 0u;
static constexpr unsigned V809_LDS_CURS_OFFSET = 8192u;
static constexpr unsigned V809_LDS_MAX_HIST_BYTES = 8u * V809_MAX_LOCAL_EXPERTS * sizeof(unsigned); // 2048 B
static_assert(V809_LDS_HIST_OFFSET + V809_LDS_MAX_HIST_BYTES <= V809_LDS_CURS_OFFSET,
              "LDS hist exceeds allocation boundary");
static_assert(V809_LDS_CURS_OFFSET + V809_LDS_MAX_HIST_BYTES <= 32768u,
              "LDS curs collides with CK barrier area (0x8000)");

// ----------------------------------------------------------------------------
// 5. Generalized Entry Guard Validator for v1864_ov1_runtime_roles
// Replaces hardcoded (H=4096, I=2048, N1=16, N2=32) checks with model-general
// invariants verified across all 6 models and 30 cells.
// ----------------------------------------------------------------------------
__device__ __forceinline__ bool v1864_validate_envelope_general(
    const V968DispatchExt& env,
    const V983Ov1Ext& o,
    unsigned tid)
{
    const V964DispatchExt& e = env.base;
    const V968CkExt& ck = env.ck;

    // 1. Role config and world bounds
    if (o.role_config > 2u) return false;
    if (e.world == 0u || e.world > V809_MAX_WORLD || e.rank >= e.world) return false;
    if (e.topk == 0u || e.topk > V809_MAX_TOPK) return false;
    if (e.capacity == 0u) return false;
    if (e.n_experts == 0u || (e.n_experts % e.world) != 0u) return false;

    // 2. Geometry invariants across all 6 models
    // H in {2048, 3072, 4096, 6144, 8192} (all multiples of 128 and 256)
    // I in {512, 1024, 2048, 3072} (all multiples of 128 and 256)
    if (o.hidden == 0u || (o.hidden % 128u) != 0u) return false;
    if (o.intermediate == 0u || (o.intermediate % 128u) != 0u) return false;
    if (o.hidden != ck.ck_hidden || o.intermediate != ck.ck_intermediate) return false;
    if (e.payload_bytes != o.hidden) return false;
    if (e.scale_groups != o.hidden / 128u) return false;

    // 3. Block decomposition invariants
    const unsigned exp_gemm_width = (o.gemm_width == 256u) ? 256u : 128u;
    if (o.gemm_width != exp_gemm_width) return false;
    if (o.n1_blocks != ck.ck_intermediate / 128u) return false;
    if (o.n2_blocks != o.hidden / exp_gemm_width) return false;
    if (o.mode != 0u) return false;
    if (o.comm_chunk_rows == 0u) return false;

    // 4. CK extension parameters
    if (ck.ck_topk != 1u || ck.ck_n_block_cols != 128u) return false;
    if (ck.ck_groups_per_row != ck.ck_hidden / 128u) return false;
#if defined(MI308X_EP16_CK_SHARED_EXPERTS)
    // v59, EP16 TU only (amd/megamoe_amd.hip defines the macro): bits [3:0]
    // carry num_shared_experts (ep16_turbo_build_packet; the shared experts
    // are local rows in the CK space and never ride the network); every other
    // bit must stay 0.  EP8 never defines it and keeps the original predicate.
    if ((ck.ck_flags & ~0xfu) != 0u) return false;
#else
    if (ck.ck_flags != 0u) return false;
#endif
    if (ck.ck_diagnostics == 0u || (ck.ck_diagnostics & 7u) != 0u) return false;
    if (ck.ck_num_tokens != e.world * e.capacity * e.topk) return false;
    if (ck.ck_local_experts != e.n_experts / e.world) return false;
    if (ck.ck_route_rows < ck.ck_num_tokens) return false;

    // 5. Vital pointer validity
    if (e.error == 0u || o.trace == 0u) return false;

    return true;
}

// ----------------------------------------------------------------------------
// 6. Scale and Payload Memory Bounds Helpers
// Supports arbitrary H in [2048..8192] and scale_groups in [16..64]
// ----------------------------------------------------------------------------
__device__ __forceinline__ unsigned v809_compute_scale_groups(unsigned hidden) {
    return hidden / 128u;
}

__device__ __forceinline__ unsigned v809_compute_n1_blocks(unsigned intermediate) {
    return intermediate / 128u;
}

__device__ __forceinline__ unsigned v809_compute_n2_blocks(unsigned hidden, unsigned gemm_width) {
    return hidden / (gemm_width ? gemm_width : 128u);
}
