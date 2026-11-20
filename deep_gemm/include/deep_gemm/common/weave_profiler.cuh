#pragma once

#include <cstdint>
#include <cute/arch/util.hpp>

namespace deep_gemm::prof {

// Canonical WeaveProfiler event IDs (mapped directly to the 7-stage MoE timeline palette)
enum EventId : uint16_t {
    kOpening         = 1,   // Route / Sync (W00-W03)
    kDispatchSync    = 2,   // Route / Sync (W00-W03)
    kDispatchPull    = 3,   // Dispatch / Pull (W00-W03)
    kScheduler       = 4,   // TMA / Tile Load (W07)
    kL1TmaA          = 5,   // TMA / Tile Load (W04)
    kL1TmaB          = 6,   // TMA / Tile Load (W05)
    kL1Umma          = 7,   // L1 GEMM (FC1+SwiGLU) (W06)
    kL1Epilogue      = 8,   // L1 GEMM (FC1+SwiGLU) (W08-W15)
    kL2TmaA          = 9,   // TMA / Tile Load (W04)
    kL2TmaB          = 10,  // TMA / Tile Load (W05)
    kL2Umma          = 11,  // L2 GEMM (FC2) (W06)
    kL2EpiScatter    = 12,  // Scatter [epilogue_only] (W08-W15)
    kL2EpiLocalStage = 13,  // L2 GEMM (FC2) [comm_only local stage, ZERO NVLink scatter] (W08-W15)
    kCommScatter     = 14,  // Scatter [comm_only] (W00-W03)
    kPreCombineSync  = 15,  // Route / Sync
    kCombineReduce   = 16,  // Combine / Reduce (W08-W15)
    kCleanupSync     = 17,  // Route / Sync (W00-W03)
};

// 16-byte record written by lane 0 of each warp with zero atomic contention
struct alignas(16) WeaveProfEvent {
    uint64_t t_begin;
    uint32_t dur_ns;
    uint16_t event_id;
    uint16_t aux;
};

static __device__ __forceinline__ uint64_t read_globaltimer() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

template <bool kEnabled>
struct WeaveProfRecorder {
    __device__ __forceinline__ WeaveProfRecorder(void*, uint32_t*, uint32_t*, uint32_t, uint32_t, uint32_t, uint32_t) {}
    __device__ __forceinline__ uint64_t begin() const { return 0; }
    __device__ __forceinline__ void end(uint64_t, uint16_t, uint16_t = 0) {}
    __device__ __forceinline__ void finish() {}
};

template <>
struct WeaveProfRecorder<true> {
    WeaveProfEvent* base;
    uint32_t* count_ptr;
    uint32_t* overflow_ptr;
    uint32_t cap;
    uint32_t cursor;
    bool is_lane0;

    __device__ __forceinline__ WeaveProfRecorder(
        void* events_base, uint32_t* counts, uint32_t* overflow,
        uint32_t cap_per_warp, uint32_t sm_idx, uint32_t warp_idx, uint32_t lane_idx)
        : cap(cap_per_warp), cursor(0), is_lane0(lane_idx == 0) {
        const uint32_t slot = sm_idx * 16u + warp_idx;
        base = static_cast<WeaveProfEvent*>(events_base) + static_cast<uint64_t>(slot) * cap_per_warp;
        count_ptr = counts + slot;
        overflow_ptr = overflow;
    }

    __device__ __forceinline__ uint64_t begin() const {
        return is_lane0 ? read_globaltimer() : 0ull;
    }

    __device__ __forceinline__ void end(uint64_t t0, uint16_t event_id, uint16_t aux = 0) {
        if (is_lane0) {
            const uint64_t t1 = read_globaltimer();
            if (cursor < cap) {
                const uint32_t dur = t1 >= t0 ? static_cast<uint32_t>(t1 - t0) : 0u;
                WeaveProfEvent ev{t0, dur, event_id, aux};
                *reinterpret_cast<uint4*>(base + cursor) = *reinterpret_cast<const uint4*>(&ev);
                ++cursor;
            } else if (overflow_ptr != nullptr) {
                atomicAdd(overflow_ptr, 1u);
            }
        }
    }

    __device__ __forceinline__ void finish() {
        if (is_lane0 and count_ptr != nullptr) {
            *count_ptr = cursor;
        }
    }
};

} // namespace deep_gemm::prof
