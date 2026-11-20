#pragma once

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>
#include <deep_gemm/scheduler/weave_policy_runtime.cuh>

namespace deep_gemm::layout {

static constexpr int kNumCandidateBlockMs = 8;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192, 256};
static constexpr int kMaxCandidateBlockM = 256;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 768;

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

    // Isolated D1 fine-grained group metadata capacities (unconditional layout)
    uint32_t num_max_d1_m_blocks;
    uint32_t num_max_d1_groups_per_expert;
    uint32_t num_d1_bitmap_words_per_expert;

    // Keep grid/NVLink/schedule counters separated from expert counters.
    // NVIDIA L2 cache lines are 128B, and these counters are hot atomics.
    static constexpr uint64_t kNumBarrierSignalBytes = 128;
    static constexpr uint64_t kNumD1SignalBytes = 256;

    Workspace() = default;

    CUTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& num_ring_tokens):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_ring_tokens(num_ring_tokens) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_ring_blocks = num_ring_tokens / kMinCandidateBlockM;
        num_shared_l2_pool_blocks = math::ceil_div<uint32_t>(num_max_tokens_per_rank, kMinCandidateBlockM);
        num_max_d1_m_blocks = num_max_pool_tokens / kMinCandidateBlockM;
        num_max_d1_groups_per_expert = math::ceil_div<uint32_t>(num_max_recv_tokens_per_expert, kMinCandidateBlockM);
        num_d1_bitmap_words_per_expert = math::ceil_div<uint32_t>(num_max_d1_groups_per_expert, 64u);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_d1_base_offset() const {
        uint64_t num_bytes = 0;

        // Barrier and in-kernel task scheduling counters
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += static_cast<uint64_t>(num_experts) * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += static_cast<uint64_t>(num_experts_per_rank) * sizeof(uint64_t);

        // L1/L2 full/empty ring counts
        num_bytes += static_cast<uint64_t>(num_ring_blocks) * sizeof(uint32_t) * 4;

        // Shared L2 full block count
        num_bytes += static_cast<uint64_t>(num_shared_l2_pool_blocks) * sizeof(uint32_t);

        // Dispatch pulling source token-topk
        num_bytes += static_cast<uint64_t>(num_experts_per_rank) * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices (full)
        num_bytes += static_cast<uint64_t>(num_max_pool_tokens) * sizeof(TokenSrcMetadata);

        // Align D1 control region to a dedicated 128B L2 cache line
        return math::align<uint64_t>(num_bytes, 128);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = get_d1_base_offset();

        // Dedicated D1 signal cache lines (2 x 128B)
        num_bytes += kNumD1SignalBytes;

        // Per-local-expert flat slot allocator + per-group metadata/pull counts (8B aligned)
        const uint64_t aligned_experts = math::align<uint64_t>(num_experts_per_rank, 2);
        const uint64_t aligned_group_counts = math::align<uint64_t>(
            static_cast<uint64_t>(num_experts_per_rank) * num_max_d1_groups_per_expert, 2);
        num_bytes += aligned_experts * sizeof(uint32_t);
        num_bytes += aligned_group_counts * sizeof(uint32_t) * 2;

        // Per-local-expert group ready bitmap
        num_bytes += static_cast<uint64_t>(num_experts_per_rank) * num_d1_bitmap_words_per_expert * sizeof(uint64_t);

        // Metadata-ready group queue (+32 zero-padded sentinel tail for warp-wide coalesced scan)
        num_bytes += (static_cast<uint64_t>(num_max_d1_m_blocks) + 32ull) * sizeof(uint64_t);

        // Data-ready (expert, m_block) schedule table
        num_bytes += static_cast<uint64_t>(num_max_d1_m_blocks) * sizeof(uint64_t);
        // Keep DeepGEMM offsets unchanged and append the Weave CTA-0 header.
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

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` NVLink barrier counter
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

    // Domain 0 keeps the historical offsets. Domain 1 lives in otherwise
    // unused padding of the 128B barrier line, so the two per-die queues stay
    // adjacent but never alias the D2/D4 control words at [44..51].
    CUTLASS_DEVICE
    uint32_t* get_l1_task_count_ptr(const uint32_t& locality_domain_idx = 0) const {
        DG_STATIC_ASSERT(sizeof(uint32_t) == 4, "Invalid word size");
        return math::advance_ptr<uint32_t>(base, 28u + 24u * locality_domain_idx);
    }

    CUTLASS_DEVICE
    uint32_t* get_l2_task_count_ptr(const uint32_t& locality_domain_idx = 0) const {
        return math::advance_ptr<uint32_t>(base, 32u + 24u * locality_domain_idx);
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l1_task_count_ptr(const uint32_t& locality_domain_idx = 0) const {
        return math::advance_ptr<uint32_t>(base, 36u + 24u * locality_domain_idx);
    }

    CUTLASS_DEVICE
    uint32_t* get_shared_l2_task_count_ptr(const uint32_t& locality_domain_idx = 0) const {
        return math::advance_ptr<uint32_t>(base, 40u + 24u * locality_domain_idx);
    }

    // Axis 2 (D2): active dispatch pull warp counter at [44..47]
    CUTLASS_DEVICE
    uint32_t* get_d2_pull_warp_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 44u);
    }

    // Axis 4 (D4): ready-queue write counter at [48..51] & upper half of l1_full_count
    CUTLASS_DEVICE
    uint32_t* get_d4_ready_write_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 48u);
    }

    CUTLASS_DEVICE
    uint32_t* get_d4_ready_queue_ptr(const uint32_t& slot_idx = 0) const {
        return get_l1_full_count_ptr((num_ring_blocks / 2u) + slot_idx);
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

    // Isolated D1 fine-grained group metadata pointers
    CUTLASS_DEVICE
    uint8_t* get_d1_signal_base_ptr() const {
        return math::advance_ptr<uint8_t>(base, get_d1_base_offset());
    }

    CUTLASS_DEVICE
    int* get_d1_rank_open_done_ptr() const {
        return reinterpret_cast<int*>(get_d1_signal_base_ptr() + 0);
    }

    CUTLASS_DEVICE
    uint32_t* get_d1_meta_group_write_ptr() const {
        return reinterpret_cast<uint32_t*>(get_d1_signal_base_ptr() + 4);
    }

    CUTLASS_DEVICE
    uint32_t* get_d1_total_meta_groups_ptr() const {
        return reinterpret_cast<uint32_t*>(get_d1_signal_base_ptr() + 8);
    }

    CUTLASS_DEVICE
    uint32_t* get_d1_total_m_blocks_ptr() const {
        return reinterpret_cast<uint32_t*>(get_d1_signal_base_ptr() + 12);
    }

    CUTLASS_DEVICE
    uint32_t* get_d1_ready_sched_write_ptr() const {
        return reinterpret_cast<uint32_t*>(get_d1_signal_base_ptr() + 128);
    }

    CUTLASS_DEVICE
    uint32_t* get_d1_expert_write_cursor_ptr(const uint32_t& expert_idx = 0) const {
        return reinterpret_cast<uint32_t*>(get_d1_signal_base_ptr() + kNumD1SignalBytes) + expert_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_d1_group_meta_count_ptr(const uint32_t& expert_idx = 0, const uint32_t& group_idx = 0) const {
        const uint32_t aligned_experts = math::align<uint32_t>(num_experts_per_rank, 2u);
        return get_d1_expert_write_cursor_ptr(aligned_experts) +
               expert_idx * num_max_d1_groups_per_expert + group_idx;
    }

    CUTLASS_DEVICE
    uint32_t* get_d1_group_pull_count_ptr(const uint32_t& expert_idx = 0, const uint32_t& group_idx = 0) const {
        const uint32_t aligned_group_counts = math::align<uint32_t>(
            num_experts_per_rank * num_max_d1_groups_per_expert, 2u);
        return get_d1_group_meta_count_ptr(0, aligned_group_counts) +
               expert_idx * num_max_d1_groups_per_expert + group_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_d1_ready_bitmap_word_ptr(const uint32_t& expert_idx = 0, const uint32_t& word_idx = 0) const {
        const uint32_t aligned_group_counts = math::align<uint32_t>(
            num_experts_per_rank * num_max_d1_groups_per_expert, 2u);
        auto* bm_base = reinterpret_cast<uint64_t*>(get_d1_group_pull_count_ptr(0, aligned_group_counts));
        return bm_base + expert_idx * num_d1_bitmap_words_per_expert + word_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_d1_meta_group_queue_ptr(const uint32_t& q_idx = 0) const {
        return get_d1_ready_bitmap_word_ptr(num_experts_per_rank, 0) + q_idx;
    }

    CUTLASS_DEVICE
    uint64_t* get_d1_ready_sched_ptr(const uint32_t& ready_seq = 0) const {
        return get_d1_meta_group_queue_ptr(num_max_d1_m_blocks + 32u) + ready_seq;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    Data() = default;

    CUTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    CUTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    CUTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    Buffer() = default;

    CUTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& num_max_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(num_max_tokens_per_rank),
        base(base) {}

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    CUTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
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

    CUTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    CUTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

struct MegaMoEBuffer {
    Workspace workspace;

    // Input buffers (per-rank)
    Buffer input_token_buffer,
           input_sf_buffer,
           input_topk_idx_buffer,
           input_topk_weights_buffer;

    // Routed expert ring buffers
    // NOTE: shared L1 tokens reuse `input_token_buffer`.
    Buffer shared_l1_token_buffer, shared_l1_sf_buffer,
           shared_l2_token_buffer, shared_l2_sf_buffer;

    // Routed expert ring buffers
    Buffer l1_token_buffer,
           l1_sf_buffer,
           l1_topk_weights_buffer,
           l2_token_buffer,
           l2_sf_buffer,
           combine_token_buffer,
           // NVFP4 path: one FP32 top-k weight per routed pool token.  This is
           // larger than the ring-sized L1 weight buffer and is appended after
           // the D7 staging region so existing offsets stay unchanged.
           routed_topk_weights_buffer;

    // Axis 7 (D7 queue mode): global HBM scatter-ready descriptor store.
    // The queue is appended after the routed top-k weights so all pre-existing
    // offsets are unchanged.  Each slot contains an 8-byte descriptor plus a
    // 4-byte publication epoch; the control line is one 128B block.
    uint32_t d7_queue_capacity = 0;
    uint64_t d7_queue_offset = 0;
    uint64_t d7_queue_desc_offset = 0;
    uint64_t d7_queue_ready_offset = 0;
    uint64_t d7_queue_bytes = 0;

    CUTLASS_HOST_DEVICE
    MegaMoEBuffer(void* base,
                  const uint32_t& hidden,
                  const uint32_t& intermediate_hidden,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk,
                  const uint32_t& num_ring_tokens,
                  const uint32_t& num_sf_ring_tokens,
                  const bool& with_sf,
                  const uint32_t& num_shared_experts = 0,
                  const uint32_t& num_mma_elem_bits = 0u,
                  const uint32_t& sf_gran_k = 32u,
                  const uint32_t& shared_num_mma_elem_bits = 0u) {
        // Workspace
        workspace = Workspace(base, num_ranks, num_experts,
                              num_max_tokens_per_rank, num_topk, num_ring_tokens);

        // Shared
        const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        const uint32_t mma_elem_bits = num_mma_elem_bits != 0u
            ? num_mma_elem_bits : (with_sf ? 8u : 16u);
        const uint32_t shared_mma_elem_bits = shared_num_mma_elem_bits != 0u
            ? shared_num_mma_elem_bits : (with_sf ? 8u : 16u);
        const bool shared_with_sf = with_sf && shared_mma_elem_bits != 16u;
        const auto num_max_shared_sf_tokens = shared_with_sf
            ? get_num_max_shared_sf_tokens(num_max_tokens_per_rank) : 0u;

        // Layouts
        const uint32_t num_mma_elem_bytes =
            math::ceil_div<uint32_t>(hidden * mma_elem_bits, 8u);
        const uint32_t intermediate_mma_elem_bytes =
            math::ceil_div<uint32_t>(intermediate_hidden * mma_elem_bits, 8u);
        const uint32_t shared_mma_elem_bytes =
            math::ceil_div<uint32_t>(shared_intermediate_hidden * shared_mma_elem_bits, 8u);
        const uint32_t input_sf_bytes =
            with_sf ? math::ceil_div<uint32_t>(hidden, sf_gran_k) : 0u;
        const uint32_t intermediate_sf_bytes =
            with_sf ? math::ceil_div<uint32_t>(intermediate_hidden, sf_gran_k) : 0u;
        const uint32_t shared_intermediate_sf_bytes =
            shared_with_sf ? math::ceil_div<uint32_t>(shared_intermediate_hidden, sf_gran_k) : 0u;
        const auto input_token_layout = layout::Data(num_mma_elem_bytes);
        const auto bf16_token_layout = layout::Data(hidden * 2);
        const auto intermediate_token_layout = layout::Data(intermediate_mma_elem_bytes);
        const auto shared_intermediate_token_layout = layout::Data(shared_mma_elem_bytes);
        const auto input_sf_layout = layout::Data(input_sf_bytes);
        const auto intermediate_sf_layout = layout::Data(intermediate_sf_bytes);
        const auto shared_intermediate_sf_layout = layout::Data(shared_intermediate_sf_bytes);
        const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
        const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

        // Input buffers
        input_token_buffer = Buffer(
            input_token_layout, 1, num_max_tokens_per_rank,
            workspace.get_end_ptr());
        input_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_tokens_per_rank,
            input_token_buffer.get_end_ptr());
        input_topk_idx_buffer = Buffer(
            input_topk_idx_layout, 1, num_max_tokens_per_rank,
            with_sf ? input_sf_buffer.get_end_ptr() : input_token_buffer.get_end_ptr());
        input_topk_weights_buffer = Buffer(
            input_topk_weights_layout, 1, num_max_tokens_per_rank,
            input_topk_idx_buffer.get_end_ptr());

        // Shared expert buffers.  The L1 input aliases the input token buffer;
        // shared L2 output is BF16 for NVFP4 (shared_num_mma_elem_bits=16).
        shared_l1_token_buffer = input_token_buffer;
        shared_l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_shared_sf_tokens,
            input_topk_weights_buffer.get_end_ptr());
        shared_l2_token_buffer = Buffer(
            shared_intermediate_token_layout, 1, num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            shared_with_sf ? shared_l1_sf_buffer.get_end_ptr() : input_topk_weights_buffer.get_end_ptr());
        shared_l2_sf_buffer = Buffer(
            shared_intermediate_sf_layout, 1, num_max_shared_sf_tokens,
            shared_l2_token_buffer.get_end_ptr());

        // Routed expert ring buffers
        l1_token_buffer = Buffer(
            input_token_layout, 1, num_ring_tokens,
            num_shared_experts > 0 ?
                (shared_with_sf ? shared_l2_sf_buffer.get_end_ptr() : shared_l2_token_buffer.get_end_ptr()) :
                input_topk_weights_buffer.get_end_ptr()
        );
        l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_sf_ring_tokens,
            l1_token_buffer.get_end_ptr());
        l1_topk_weights_buffer = Buffer(
            l1_topk_weights_layout, 1, num_ring_tokens,
            with_sf ? l1_sf_buffer.get_end_ptr() : l1_token_buffer.get_end_ptr());

        l2_token_buffer = Buffer(
            intermediate_token_layout, 1, num_ring_tokens,
            l1_topk_weights_buffer.get_end_ptr());
        l2_sf_buffer = Buffer(
            intermediate_sf_layout, 1, num_sf_ring_tokens,
            l2_token_buffer.get_end_ptr());

        combine_token_buffer = Buffer(
            bf16_token_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), num_max_tokens_per_rank,
            with_sf ? l2_sf_buffer.get_end_ptr() : l2_token_buffer.get_end_ptr());

        routed_topk_weights_buffer = Buffer(
            l1_topk_weights_layout, 1, workspace.num_max_pool_tokens,
            math::advance_ptr<uint8_t>(
                workspace.base,
                get_d7_local_row_offset() + get_d7_local_row_bytes()));

        // D7 queue: one descriptor per (destination rank, token, 128-column
        // hidden subtile) plus a small reservation slack.  Producers reserve
        // slots in batches, so the slack bounds the number of dummy slots that
        // can be left at the end of a CTA's batch.
        const uint64_t d7_queue_slots_per_token = math::ceil_div<uint64_t>(
            static_cast<uint64_t>(bf16_token_layout.num_bytes), 256ull);
        const uint64_t d7_queue_capacity_raw =
            static_cast<uint64_t>(workspace.num_max_pool_tokens) * d7_queue_slots_per_token + 262144ull;
        d7_queue_capacity = static_cast<uint32_t>(d7_queue_capacity_raw);
        const uint64_t routed_topk_weights_end =
            get_d7_local_row_offset() + get_d7_local_row_bytes() +
            static_cast<uint64_t>(routed_topk_weights_buffer.data_layout.num_bytes) *
                static_cast<uint64_t>(workspace.num_max_pool_tokens);
        d7_queue_offset = math::align<uint64_t>(routed_topk_weights_end, 128ull);
        d7_queue_desc_offset = d7_queue_offset + 128ull;
        d7_queue_ready_offset = math::align<uint64_t>(
            d7_queue_desc_offset + static_cast<uint64_t>(d7_queue_capacity) * sizeof(uint64_t), 128ull);
        d7_queue_bytes = math::align<uint64_t>(
            d7_queue_ready_offset + static_cast<uint64_t>(d7_queue_capacity) * sizeof(uint32_t), 128ull) -
            d7_queue_offset;
    }

    static constexpr uint32_t kD6ReadyRowStride = 32u;

    CUTLASS_HOST_DEVICE
    uint64_t get_d6_ctrl_offset() const {
        const uint64_t combine_end_offset = static_cast<uint64_t>(
            static_cast<uint8_t*>(combine_token_buffer.get_end_ptr()) -
            static_cast<uint8_t*>(workspace.base));
        return math::align<uint64_t>(combine_end_offset, 128ull);
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_d6_max_groups() const {
        return combine_token_buffer.data_layout.num_bytes / 256u;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_d6_ready_epoch_offset() const {
        return get_d6_ctrl_offset() + 128ull;
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_d6_ready_epoch_bytes() const {
        const uint64_t raw = static_cast<uint64_t>(workspace.num_max_tokens_per_rank) *
                             static_cast<uint64_t>(get_d6_max_groups()) *
                             static_cast<uint64_t>(kD6ReadyRowStride) * sizeof(uint16_t);
        return math::align<uint64_t>(raw, 128ull);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_d6_local_group_cnt_offset() const {
        return get_d6_ready_epoch_offset() + get_d6_ready_epoch_bytes();
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_d6_routed_max_blocks() const {
        return workspace.num_max_pool_tokens / static_cast<uint32_t>(kMinCandidateBlockM);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_d6_local_group_cnt_bytes() const {
        const uint64_t total_blocks = static_cast<uint64_t>(
            get_d6_routed_max_blocks() + workspace.num_shared_l2_pool_blocks);
        const uint64_t raw = total_blocks * static_cast<uint64_t>(get_d6_max_groups()) * sizeof(uint32_t);
        return math::align<uint64_t>(raw, 128ull);
    }

    CUTLASS_DEVICE
    uint32_t* get_d6_combine_task_counter_ptr() const {
        return math::advance_ptr<uint32_t>(workspace.base, get_d6_ctrl_offset());
    }

    CUTLASS_DEVICE
    uint32_t* get_d6_epoch_ptr() const {
        return math::advance_ptr<uint32_t>(workspace.base, get_d6_ctrl_offset() + 64ull);
    }

    CUTLASS_DEVICE
    uint16_t* get_d6_ready_epoch_ptr(const uint32_t& token_idx, const uint32_t& group_idx) const {
        auto* const base_ptr = math::advance_ptr<uint16_t>(workspace.base, get_d6_ready_epoch_offset());
        return base_ptr + (static_cast<uint64_t>(token_idx) * get_d6_max_groups() + group_idx) * kD6ReadyRowStride;
    }

    CUTLASS_DEVICE
    uint32_t* get_d6_ready_epoch_u32_ptr() const {
        return math::advance_ptr<uint32_t>(workspace.base, get_d6_ready_epoch_offset());
    }

    CUTLASS_DEVICE
    uint32_t get_d6_ready_epoch_num_u32() const {
        return static_cast<uint32_t>(get_d6_ready_epoch_bytes() / sizeof(uint32_t));
    }

    CUTLASS_DEVICE
    uint32_t* get_d6_local_group_cnt_ptr(
        const bool& is_shared, const uint32_t& pool_block_idx, const uint32_t& group_idx) const {
        auto* const base_ptr = math::advance_ptr<uint32_t>(workspace.base, get_d6_local_group_cnt_offset());
        const uint32_t blk = is_shared ? (get_d6_routed_max_blocks() + pool_block_idx) : pool_block_idx;
        return base_ptr + static_cast<uint64_t>(blk) * get_d6_max_groups() + group_idx;
    }

    // Axis 7 (D7 Mode 2): Local HBM/L2 deduplication / pre-reduction staging & subtile state
    CUTLASS_HOST_DEVICE
    uint64_t get_d7_subtile_state_offset() const {
        return math::align<uint64_t>(
            get_d6_local_group_cnt_offset() + get_d6_local_group_cnt_bytes(), 128ull);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_d7_subtile_state_bytes() const {
        const uint64_t raw = static_cast<uint64_t>(workspace.num_ranks) *
                             static_cast<uint64_t>(workspace.num_max_tokens_per_rank) *
                             static_cast<uint64_t>(get_d6_max_groups()) * sizeof(uint32_t);
        return math::align<uint64_t>(raw, 128ull);
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_d7_local_row_offset() const {
        return get_d7_subtile_state_offset() + get_d7_subtile_state_bytes();
    }

    CUTLASS_HOST_DEVICE
    uint64_t get_d7_local_row_bytes() const {
        const uint64_t raw = static_cast<uint64_t>(workspace.num_ranks) *
                             static_cast<uint64_t>(workspace.num_max_tokens_per_rank) *
                             static_cast<uint64_t>(combine_token_buffer.data_layout.num_bytes);
        return math::align<uint64_t>(raw, 128ull);
    }

    CUTLASS_DEVICE
    uint32_t* get_d7_subtile_state_ptr(
        const uint32_t& dst_rank_idx, const uint32_t& dst_token_idx, const uint32_t& sub_n_idx) const {
        auto* const base_ptr = math::advance_ptr<uint32_t>(workspace.base, get_d7_subtile_state_offset());
        const uint64_t tok_linear = static_cast<uint64_t>(dst_rank_idx) * workspace.num_max_tokens_per_rank + dst_token_idx;
        return base_ptr + tok_linear * get_d6_max_groups() + sub_n_idx;
    }

    CUTLASS_DEVICE
    float4* get_d7_local_row_ptr(
        const uint32_t& dst_rank_idx, const uint32_t& dst_token_idx, const uint32_t& n_idx) const {
        auto* const base_ptr = math::advance_ptr<uint8_t>(workspace.base, get_d7_local_row_offset());
        const uint64_t tok_linear = static_cast<uint64_t>(dst_rank_idx) * workspace.num_max_tokens_per_rank + dst_token_idx;
        const uint64_t byte_off = tok_linear * combine_token_buffer.data_layout.num_bytes +
                                  static_cast<uint64_t>(n_idx) * sizeof(nv_bfloat16);
        return reinterpret_cast<float4*>(base_ptr + byte_off);
    }

    // Axis 7 (D7 queue mode): descriptor / publication / control pointers.
    CUTLASS_DEVICE
    uint32_t get_d7_queue_capacity() const {
        return d7_queue_capacity;
    }

    CUTLASS_DEVICE
    uint32_t* get_d7_queue_epoch_ptr() const {
        return math::advance_ptr<uint32_t>(workspace.base, d7_queue_offset + 0ull);
    }

    CUTLASS_DEVICE
    uint32_t* get_d7_queue_tail_ptr() const {
        return math::advance_ptr<uint32_t>(workspace.base, d7_queue_offset + 4ull);
    }

    CUTLASS_DEVICE
    uint32_t* get_d7_queue_head_ptr() const {
        return math::advance_ptr<uint32_t>(workspace.base, d7_queue_offset + 8ull);
    }

    CUTLASS_DEVICE
    uint32_t* get_d7_queue_producer_done_ptr() const {
        return math::advance_ptr<uint32_t>(workspace.base, d7_queue_offset + 12ull);
    }

    CUTLASS_DEVICE
    uint32_t* get_d7_queue_consumer_done_ptr() const {
        return math::advance_ptr<uint32_t>(workspace.base, d7_queue_offset + 16ull);
    }

    CUTLASS_DEVICE
    uint64_t* get_d7_queue_desc_ptr() const {
        return math::advance_ptr<uint64_t>(workspace.base, d7_queue_desc_offset);
    }

    CUTLASS_DEVICE
    uint32_t* get_d7_queue_ready_ptr() const {
        return math::advance_ptr<uint32_t>(workspace.base, d7_queue_ready_offset);
    }

    CUTLASS_HOST_DEVICE
    int64_t get_num_bytes() const {
        return static_cast<int64_t>(d7_queue_offset + d7_queue_bytes);
    }
};

} // namespace deep_gemm::layout

