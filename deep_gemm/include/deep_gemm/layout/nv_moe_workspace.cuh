#pragma once

#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::layout::nv_moe {

using ::deep_gemm::layout::Data;
using ::deep_gemm::layout::Buffer;
using ::deep_gemm::layout::SymBuffer;

static constexpr int kNumCandidateBlockMs = 7;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192};
static constexpr int kMaxCandidateBlockM = 192;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 384;

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks x SF_BLOCK_M
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_sf_ring_tokens(T num_ring_tokens, T block_m) {
    return (num_ring_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Shared L2 input SF capacity: worst-case aligned SF pages over all candidate BLOCK_M.
template <typename T>
CUTLASS_HOST_DEVICE constexpr T get_num_max_shared_sf_tokens(const T& num_max_tokens_per_rank) {
    return math::constexpr_ceil_div<T>(num_max_tokens_per_rank, kMinCandidateBlockM) * 128;
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

// One published row of the global HBM scatter queue.  The descriptor is
// produced by the last FC2 epilogue chunk that finishes a (dst_rank, token)
// staging slot and consumed by a throttled Dispatch/Comm SM.
struct ScatterQueueDesc {
    uint32_t dst_rank;
    uint32_t token;
    uint32_t topk;
    uint32_t reserved;
};

struct Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Ring-buffer capacity used by reusable token/data buffers
    uint32_t num_ring_tokens;
    uint32_t num_ring_blocks;
    uint32_t num_shared_l2_pool_blocks;

    // Full-pool span used by non-ring token metadata
    uint32_t num_max_pool_tokens;

    // Hidden width; non-zero enables the global HBM/L2 scatter staging region.
    // The region is allocated for every SM90 symmetric buffer so the kernel can
    // switch paths at JIT time without reallocating the communication buffer.
    uint32_t hidden;

    // Keep grid/NVLink/schedule counters separated from expert counters.
    // NVIDIA L2 cache lines are 128B, and these counters are hot atomics.
    static constexpr uint64_t kNumBarrierSignalBytes = 128;

    Workspace() = default;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& num_ring_tokens,
              const uint32_t& hidden = 0u):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_ring_tokens(num_ring_tokens), hidden(hidden) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_ring_blocks = num_ring_tokens / kMinCandidateBlockM;
        num_shared_l2_pool_blocks = math::ceil_div<uint32_t>(num_max_tokens_per_rank, kMinCandidateBlockM);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_pre_scatter_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier and in-kernel task scheduling counters
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 full token count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L1 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 full block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // Shared L2 full block count
        num_bytes += num_shared_l2_pool_blocks * sizeof(uint32_t);

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices (full)
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = math::align<uint64_t>(get_pre_scatter_bytes(), 128);
        // Global HBM/L2 dedup scatter staging.  Every logical (dst_rank, token)
        // owns one full BF16 hidden vector.  Chunk counters are sized for the
        // smallest supported WG_BLOCK_N (64) so any build shares one layout.
        num_bytes += get_scatter_stage_bytes();
        num_bytes += get_num_scatter_slots() * sizeof(ScatterQueueDesc);
        num_bytes += get_num_scatter_slots() * sizeof(uint32_t);
        num_bytes += get_num_scatter_slots() * get_max_scatter_chunks() * sizeof(uint32_t);
        num_bytes += get_num_scatter_slots() * sizeof(uint32_t);
        num_bytes = math::align<uint64_t>(num_bytes, 128);
        static_assert((kScatterCtrlWords & 3u) == 0u, "Scatter control must stay 16B aligned");
        num_bytes += kScatterCtrlWords * sizeof(uint32_t);

        // Chunked dedup combine sync region (3 * num_ranks * max_chunks + 2 * max_chunks u32s).
        num_bytes = math::align<uint64_t>(num_bytes, 128);
        num_bytes += get_combine_chunk_sync_bytes();

        // Append the Weave CTA-0 policy header after all existing offsets.
        num_bytes = math::align<uint64_t>(num_bytes, 128);
        num_bytes += kWeaveControlBytes;
        return num_bytes;
    }

    CUTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    static constexpr uint64_t kWeaveControlBytes =
        (static_cast<uint64_t>(sizeof(weave::PolicyControl)) + 127ull) & ~127ull;

    CUTLASS_DEVICE
    weave::PolicyControl* get_weave_policy_control_ptr() const {
        return reinterpret_cast<weave::PolicyControl*>(
            math::advance_ptr<uint8_t>(base, get_num_bytes() - kWeaveControlBytes));
    }

    // ---- Global HBM/L2 dedup scatter region --------------------------------
    static constexpr uint32_t kScatterCtrlWords = 4u;

    CUTLASS_HOST_DEVICE
    uint32_t get_num_scatter_slots() const {
        return num_ranks * num_max_tokens_per_rank;
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_max_scatter_chunks() const {
        // WG_BLOCK_N is at least 64 on every supported SM90 path.
        return hidden == 0u ? 1u : (hidden / 64u);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_scatter_stage_bytes() const {
        return hidden == 0u ? 0ull :
            static_cast<uint64_t>(get_num_scatter_slots()) * hidden * sizeof(uint16_t);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_scatter_region_offset() const {
        return math::align<uint64_t>(get_pre_scatter_bytes(), 128);
    }

    CUTLASS_DEVICE
    void* get_scatter_stage_ptr() const {
        return math::advance_ptr<uint8_t>(base, get_scatter_region_offset());
    }

    CUTLASS_DEVICE
    ScatterQueueDesc* get_scatter_queue_desc_ptr() const {
        return reinterpret_cast<ScatterQueueDesc*>(
            math::advance_ptr<uint8_t>(get_scatter_stage_ptr(), get_scatter_stage_bytes()));
    }

    CUTLASS_DEVICE
    uint32_t* get_scatter_queue_ready_ptr() const {
        return reinterpret_cast<uint32_t*>(
            get_scatter_queue_desc_ptr() + get_num_scatter_slots());
    }

    CUTLASS_DEVICE
    uint32_t* get_scatter_chunk_count_ptr() const {
        return get_scatter_queue_ready_ptr() + get_num_scatter_slots();
    }

    CUTLASS_DEVICE
    uint32_t* get_scatter_chunks_done_ptr() const {
        return get_scatter_chunk_count_ptr() +
            get_num_scatter_slots() * get_max_scatter_chunks();
    }

    CUTLASS_DEVICE
    uint32_t* get_scatter_ctrl_ptr() const {
        return get_scatter_chunks_done_ptr() + get_num_scatter_slots();
    }

    CUTLASS_DEVICE
    uint32_t* get_scatter_queue_tail_ptr() const {
        return get_scatter_ctrl_ptr() + 0;
    }

    CUTLASS_DEVICE
    uint32_t* get_scatter_queue_head_ptr() const {
        return get_scatter_ctrl_ptr() + 1;
    }

    CUTLASS_DEVICE
    uint32_t* get_scatter_queue_completed_ptr() const {
        return get_scatter_ctrl_ptr() + 2;
    }

    CUTLASS_DEVICE
    uint32_t* get_scatter_epilogue_done_ptr() const {
        return get_scatter_ctrl_ptr() + 3;
    }

    // ---- Chunked dedup combine sync region ---------------------------------
    // Supports WEAVE_COMBINE_SYNC_CHUNK = G in {16, 32, 64, 128}.
    // All arrays are sized for G_min = 16 so every build shares one workspace.
    static constexpr uint32_t kMinCombineSyncChunk = 16u;

    CUTLASS_HOST_DEVICE
    uint32_t get_max_combine_chunks() const {
        return math::ceil_div<uint32_t>(num_max_tokens_per_rank, kMinCombineSyncChunk);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_combine_chunk_sync_offset() const {
        uint64_t num_bytes = get_scatter_region_offset();
        num_bytes += get_scatter_stage_bytes();
        num_bytes += get_num_scatter_slots() * sizeof(ScatterQueueDesc);
        num_bytes += get_num_scatter_slots() * sizeof(uint32_t);
        num_bytes += get_num_scatter_slots() * get_max_scatter_chunks() * sizeof(uint32_t);
        num_bytes += get_num_scatter_slots() * sizeof(uint32_t);
        num_bytes = math::align<uint64_t>(num_bytes, 128);
        num_bytes += kScatterCtrlWords * sizeof(uint32_t);
        return math::align<uint64_t>(num_bytes, 128);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_combine_chunk_sync_bytes() const {
        const uint64_t max_chunks = get_max_combine_chunks();
        return (3ull * num_ranks * max_chunks + 2ull * max_chunks) * sizeof(uint32_t);
    }

    CUTLASS_DEVICE
    uint32_t* get_combine_chunk_sync_base_ptr() const {
        return reinterpret_cast<uint32_t*>(
            math::advance_ptr<uint8_t>(base, get_combine_chunk_sync_offset()));
    }

    CUTLASS_DEVICE
    uint32_t* get_combine_chunk_send_cnt_ptr(const uint32_t& rank_idx, const uint32_t& chunk_idx) const {
        return get_combine_chunk_sync_base_ptr() + rank_idx * get_max_combine_chunks() + chunk_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_combine_chunk_expected_ptr(const uint32_t& rank_idx, const uint32_t& chunk_idx) const {
        const uint32_t stride = num_ranks * get_max_combine_chunks();
        return get_combine_chunk_sync_base_ptr() + stride + rank_idx * get_max_combine_chunks() + chunk_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_combine_chunk_sent_ptr(const uint32_t& rank_idx, const uint32_t& chunk_idx) const {
        const uint32_t stride = 2u * num_ranks * get_max_combine_chunks();
        return get_combine_chunk_sync_base_ptr() + stride + rank_idx * get_max_combine_chunks() + chunk_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_combine_chunk_ready_ptr(const uint32_t& chunk_idx) const {
        const uint32_t stride = 3u * num_ranks * get_max_combine_chunks();
        return get_combine_chunk_sync_base_ptr() + stride + chunk_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_combine_chunk_target_ptr(const uint32_t& chunk_idx) const {
        const uint32_t stride = (3u * num_ranks + 1u) * get_max_combine_chunks();
        return get_combine_chunk_sync_base_ptr() + stride + chunk_idx;
    }

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..19]: `uint32_t` NVLink barrier counter
    // [20..27]: 2 x `int` NVLink barrier signals (phase 0 and 1)
    // [28..31]: `uint32_t` L1 schedule task counter
    // [32..35]: `uint32_t` L2 schedule task counter
    // [36..39]: `uint32_t` shared L1 schedule task counter
    // [40..43]: `uint32_t` shared L2 schedule task counter
    // [44..127]: padding to isolate hot expert counters from barrier/schedule counters
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    CUTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    CUTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    CUTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 28u);
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 32u);
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l1_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 36u);
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 40u);
    }

    // [96..99] (header padding): SM90 fused grid-free Notify SM-entry counter.
    // Every CTA bumps it once after reading this launch's policy epoch base;
    // CTA 0 is its sole consumer (waits for kNumSMs, then resets it to 0).
    CUTLASS_DEVICE
    uint32_t* get_weave_entry_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 96u);
    }

    // [100..103] (header padding): SM90 fused L1->L2 interleave, number of L1
    // tiles whose output (and K-block readiness bit) has been published in this
    // launch.  Reset by CTA 0 in the end-of-kernel workspace clean.
    CUTLASS_DEVICE
    uint32_t* get_l1_tiles_done_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 100u);
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l1_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l2_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_full_count_ptr(const uint32_t& block_idx = 0) const {
        const auto base = get_l2_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + block_idx;
    }

    // For dispatch pulling
    CUTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_shared_l2_full_count_ptr(num_shared_l2_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // For combine usages (full)
    CUTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

} // namespace deep_gemm::layout::nv_moe

