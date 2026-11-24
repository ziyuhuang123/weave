#pragma once

#include <cutlass/arch/barrier.h>

#include <deep_gemm/comm/barrier_with_timeout_policy.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/layout/sym_buffer.cuh>
#include <deep_gemm/layout/mega_moe.cuh>

namespace deep_gemm::comm {

// 60s timeout, at 2 GHz
constexpr int64_t kNumTimeoutCycles = 60ll * 2000000000ll;

// Spin until `pred()` holds; on timeout, `print_timeout()` runs before the assertion
template <typename pred_t, typename print_timeout_t>
CUTLASS_DEVICE void wait_until(const pred_t& pred, const print_timeout_t& print_timeout) {
    const auto start_clock = clock64();
    while (not pred()) {
        if (clock64() - start_clock >= kNumTimeoutCycles) {
            print_timeout();
            DG_DEVICE_ASSERT(false and "Timeout");
        }
    }
}

CUTLASS_DEVICE void cluster_sync_with_relaxed_arrive() {
    // Perform cluster_sync with `barrier.cluster.arrive.relaxed`
    // This is slightly faster than `cute::cluster_sync` but has weaker memory ordering guarantee
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();
}

template <uint32_t kNumSMs, uint32_t kGridSyncIndex = 0,
          BarrierTimeoutPolicy kTimeoutPolicy = BarrierTimeoutPolicy::Diagnostic,
          typename sync_scope_t, typename WorkspaceT>
CUTLASS_DEVICE void grid_sync(const WorkspaceT& workspace,
                              const uint32_t& sm_idx, const uint32_t& thread_idx,
                              const sync_scope_t& sync_scope) {
    // NOTES: the implementation idea is from `cooperative_groups::this_grid().sync()`
    static constexpr uint32_t kFinishSumTag = 0x80000000u;
    sync_scope();
    if (thread_idx == 0) {
        const auto count_ptr = workspace.template get_grid_sync_count_ptr<kGridSyncIndex>();
        const auto old_value = ptx::atomic_add_rel(
            count_ptr, sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
        uint32_t new_value;
        if constexpr (kTimeoutPolicy == BarrierTimeoutPolicy::Diagnostic) {
            wait_until([&]() { return ((new_value = ptx::ld_acq(count_ptr)) ^ old_value) & kFinishSumTag; }, [&]() {
                printf("DeepGEMM grid sync timeout: sm=%u, thread=%u, grid_sync_idx=%u, old=%u, current=%u, expected_tag=%u\n",
                       sm_idx, thread_idx, kGridSyncIndex, old_value, new_value, old_value ^ kFinishSumTag);
            });
        } else {
            const auto start_clock = clock64();
            do {
                new_value = ptx::ld_acq(count_ptr);
                if (clock64() - start_clock >= kNumTimeoutCycles) {
                    handle_grid_sync_timeout<kTimeoutPolicy>(
                        sm_idx, thread_idx, kGridSyncIndex, old_value, new_value,
                        old_value ^ kFinishSumTag);
                }
            } while (((new_value ^ old_value) & kFinishSumTag) == 0);
        }
    }
    sync_scope();
}

template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kGridSyncIndex, uint32_t kTag,
          BarrierTimeoutPolicy kTimeoutPolicy = BarrierTimeoutPolicy::Diagnostic,
          uint32_t kSpinSleepNs = 0u,
          typename sync_scope_t, typename WorkspaceT>
CUTLASS_DEVICE void nvlink_barrier(const WorkspaceT& workspace,
                                   const layout::SymBuffer<kNumRanks>& sym_buffer,
                                   const uint32_t& sm_idx, const uint32_t& thread_idx,
                                   const sync_scope_t& sync_scope,
                                   const bool& sync_prologue = true,
                                   const bool& sync_epilogue = true,
                                   const bool& defer_wait = false) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");

    // Grid sync before NVLink signaling
    if (sync_prologue)
        grid_sync<kNumSMs, kGridSyncIndex, kTimeoutPolicy>(workspace, sm_idx, thread_idx, sync_scope);

    // NVLink cross-rank barrier, only SM 0 participates
    if (sm_idx == 0) {
        auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
        const auto status = (*counter_ptr) & 3;
        const auto signal_phase = status & 1, signal_sign = status >> 1;
        auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

        // Send signals to remote ranks
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        sync_scope();

        // Update status and wait arrival.  `defer_wait` (only legal with
        // sync_epilogue == false and no later barrier in the launch) leaves
        // the arrival wait to `nvlink_barrier_deferred_wait` at kernel end.
        if (thread_idx == 0 and not defer_wait) {
            ptx::red_add(counter_ptr, 1);
            const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
            if constexpr (kTimeoutPolicy == BarrierTimeoutPolicy::Diagnostic) {
                wait_until([&]() { return ptx::ld_acq_sys(signal_ptr) == target; }, [&]() {
                    printf("DeepGEMM NVLink barrier timeout: rank=%d, counter=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
                           sym_buffer.rank_idx, *counter_ptr, ptx::ld_acq_sys(signal_ptr), target, signal_phase, signal_sign, kTag);
                });
            } else {
                const auto start_clock = clock64();
                while (ptx::ld_acq_sys(signal_ptr) != target) {
                    if constexpr (kSpinSleepNs > 0u)
                        __nanosleep(kSpinSleepNs);
                    if (clock64() - start_clock >= kNumTimeoutCycles) {
                        handle_nvlink_barrier_timeout<kTimeoutPolicy>(
                            sym_buffer.rank_idx, *counter_ptr, ptx::ld_acq_sys(signal_ptr), target,
                            signal_phase, signal_sign, kTag);
                    }
                }
            }
        } else if (thread_idx == 0) {
            ptx::red_add(counter_ptr, 1);
        }
    }

    // Grid sync after NVLink completion
    if (sync_epilogue)
        grid_sync<kNumSMs, kGridSyncIndex, kTimeoutPolicy>(workspace, sm_idx, thread_idx, sync_scope);
}

// Second half of an `nvlink_barrier(..., sync_epilogue = false, defer_wait = true)`
// issued earlier by the same thread (SM 0, thread 0).  That barrier must be the
// last counter bump of the launch: its (phase, sign) is recovered from the
// counter value it bumped from.  Signals of this phase are only re-armed two
// barriers later, i.e. in the next launch, which cannot start before this
// wait returns, so the signal word is still at (or moving towards) `target`.
template <uint32_t kNumRanks, uint32_t kTag,
          BarrierTimeoutPolicy kTimeoutPolicy = BarrierTimeoutPolicy::Diagnostic,
          typename WorkspaceT>
CUTLASS_DEVICE void nvlink_barrier_deferred_wait(const WorkspaceT& workspace,
                                                 const layout::SymBuffer<kNumRanks>& sym_buffer) {
    auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();
    const auto status = (ptx::ld_acq(counter_ptr) - 1u) & 3u;
    const auto signal_phase = status & 1, signal_sign = status >> 1;
    auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);
    const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
    wait_until([&]() { return ptx::ld_acq_sys(signal_ptr) == target; }, [&]() {
        printf("DeepGEMM NVLink deferred barrier timeout: rank=%d, counter=%d, signal=%d, target=%d, phase=%d, sign=%d, tag=%d\n",
               sym_buffer.rank_idx, *counter_ptr, ptx::ld_acq_sys(signal_ptr), target, signal_phase, signal_sign, kTag);
    });
}

// Split-phase cross-rank barrier: the same wire protocol and signal words as
// `nvlink_barrier` (so it interleaves with it on one counter), but without the
// trailing grid sync.  Instead of SM 0 alone spinning on the rank-local signal
// and a second grid sync broadcasting the result, every SM waits on the
// rank-local signal word itself.  This removes one grid-wide round trip from
// the critical path, which is what the upstream combine achieves with its
// per-peer ready words (one local grid sync, then per-token peer polls).
//
// Correctness:
//  * (phase, sign) is derived from the barrier counter, which only SM 0 bumps,
//    and only after the grid sync below.  Every SM reads the counter before it
//    arrives at that grid sync, so all SMs agree on the value it had before
//    the bump.  The previous bump (by an earlier barrier of this launch or the
//    previous launch) is ordered before this read by that barrier's own grid
//    sync / the kernel boundary.
//  * The signal word of this phase cannot move past `target` before every SM
//    of this rank has observed it: its next use is two barriers later, which
//    starts with a grid sync that this SM only joins after its wait returns,
//    and no peer can complete that barrier without this rank's arrival.
//  * Data written by peers before their (release, sys-scope) signal increment
//    is visible to this SM after its acquire load, and to the other threads of
//    the CTA after the trailing `sync_scope()` barrier.
template <uint32_t kNumRanks, uint32_t kNumSMs, uint32_t kNumThreads,
          uint32_t kGridSyncIndex, uint32_t kTag,
          BarrierTimeoutPolicy kTimeoutPolicy = BarrierTimeoutPolicy::Diagnostic,
          typename sync_scope_t, typename WorkspaceT>
CUTLASS_DEVICE void nvlink_barrier_split(const WorkspaceT& workspace,
                                         const layout::SymBuffer<kNumRanks>& sym_buffer,
                                         const uint32_t& sm_idx, const uint32_t& thread_idx,
                                         const sync_scope_t& sync_scope) {
    DG_STATIC_ASSERT(kNumRanks <= kNumThreads, "Insufficient threads");
    auto* counter_ptr = workspace.get_nvl_barrier_counter_ptr();

    // Read the barrier status before the grid sync (SM 0 bumps it afterwards)
    uint32_t status = 0;
    if (thread_idx == 0 or (sm_idx == 0 and thread_idx < kNumRanks))
        status = ptx::ld_acq(counter_ptr) & 3u;

    // Local grid sync: every SM of this rank has finished its prior writes
    grid_sync<kNumSMs, kGridSyncIndex, kTimeoutPolicy>(workspace, sm_idx, thread_idx, sync_scope);

    const auto signal_phase = status & 1u, signal_sign = status >> 1u;
    auto* signal_ptr = workspace.get_nvl_barrier_signal_ptr(signal_phase);

    // SM 0 signals every rank (fire-and-forget), then advances the status
    if (sm_idx == 0) {
        if (thread_idx < kNumRanks)
            ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), signal_sign ? -1 : 1);
        if (thread_idx == 0)
            ptx::red_add(counter_ptr, 1u);
    }

    // Every SM waits for all ranks' arrivals on its rank-local signal word
    if (thread_idx == 0) {
        const int target = signal_sign ? 0 : static_cast<int>(kNumRanks);
        if constexpr (kTimeoutPolicy == BarrierTimeoutPolicy::Diagnostic) {
            wait_until([&]() { return ptx::ld_acq_sys(signal_ptr) == target; }, [&]() {
                printf("DeepGEMM NVLink split barrier timeout: rank=%d, sm=%u, signal=%d, target=%d, phase=%u, sign=%u, tag=%d\n",
                       sym_buffer.rank_idx, sm_idx, ptx::ld_acq_sys(signal_ptr), target, signal_phase, signal_sign, kTag);
            });
        } else {
            const auto start_clock = clock64();
            while (ptx::ld_acq_sys(signal_ptr) != target) {
                if (clock64() - start_clock >= kNumTimeoutCycles) {
                    handle_nvlink_barrier_timeout<kTimeoutPolicy>(
                        sym_buffer.rank_idx, *counter_ptr, ptx::ld_acq_sys(signal_ptr), target,
                        signal_phase, signal_sign, kTag);
                }
            }
        }
    }
    sync_scope();
}

} // namespace deep_gemm::comm
