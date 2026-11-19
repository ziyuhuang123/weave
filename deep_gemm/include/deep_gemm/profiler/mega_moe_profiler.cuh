#pragma once

#include <cstdint>

namespace deep_gemm {

enum class MegaMoEProfileMode : uint32_t {
    kOff = 0,
    kEntryOnly = 1,
    kFull = 2,
};

enum class MegaMoEProfileLayer : uint32_t {
    kIgnored = 0,
    kTask = 1,
    kPhase = 2,
};

enum class MegaMoEProfileRole : uint32_t {
    kIgnored = 0,
    kDispatch = 1,
    kLoaderA = 2,
    kLoaderB = 3,
    kScatter = 4,
    kMath = 5,
    kCombine = 6,
};

enum class MegaMoEProfileCtaRole : uint32_t {
    kUnified = 0,
    kCommunication = 1,
    kCompute = 2,
};

enum class MegaMoEProfileTask : uint32_t {
    kDispatchRoutePublish       = 0,
    kDispatchNvlinkPull         = 1,
    kDispatchCleanup            = 2,
    kDispatchSync               = 3,
    kLoaderAWaitPool            = 4,
    kLoaderAPipeline            = 5,
    kLoaderBPipeline            = 6,
    kMathL1GemmPipeline         = 7,
    kMathL1SwiGLUQuantize       = 8,
    kMathL1TmaStore             = 9,
    kMathL2GemmPipeline         = 10,
    kMathL2Bf16Scatter          = 11,
    kMathCombineReduceStore     = 12,
    kMathBeforeCombineBarrier   = 13,
    kMathWaitDispatch           = 14,
    kMathWaitFullBarrier        = 15,
    kLoaderAWaitEmpty           = 16,
    kLoaderBWaitEmpty           = 17,
    kDispatchWaitBeforeCleanup  = 18,
    kEntryRankSync              = 19,
    kMathL1MmaPhase             = 20,
    kMathL2MmaPhase             = 21,
    kMathL2ReadyPublish         = 22,
    kMathL2Bf16Stage            = 23,
    kSupportDeferredScatter     = 24,
    kSupportScatterStore        = 25,
    kSupportScatterRankSync     = 26,
    kSupportWaitForL2           = 27,
    kDispatchWorkerTask         = 28,
    kScatterTileTask            = 29,
    kLoaderAWorkerTask          = 30,
    kLoaderBWorkerTask          = 31,
    kMathL1TileTask             = 32,
    kMathL2TileTask             = 33,
    kCombineWorkerTask          = 34,
    // SM100 production-kernel task layer.  These names are the source-level
    // contract used by the v3 ABI renderer; every pair is recorded at the
    // begin/completion boundary of real work, never around a whole kernel or a
    // grid barrier envelope.
    kOpeningSetup               = 35,  // CTA init / mbarrier / cluster sync
    kOpeningHistogram           = 36,  // token-topk histogram and source publish
    kDispatchGridSyncCompute    = 37,  // expert-count exchange + grid/NVLink barriers
    kDispatchPull               = 38,  // remote token/SF pull into local L1 pool
    kDispatchCleanupTask        = 39,  // workspace cleanup + final NVLink barrier
    kTmaA                       = 40,  // per scheduled tile, A + SFA TMA
    kTmaB                       = 41,  // per scheduled tile, B + SFB TMA
    kUmma                       = 42,  // per scheduled tile, UMMA issue/commit
    kL1EpilogueSwiGLUQuantStore = 43,  // SwiGLU + quant + SF + TMA store
    kL2EpilogueConvertScatter   = 44,  // BF16 convert + remote scatter
    kEpilogueGridSyncCompute    = 45,  // TMEM release + pre-combine grid barriers
    kCombineReduceStore         = 46,  // top-k reduce + output TMA store
    // Eight-dispatch-warp forward-progress probes.  These are point records
    // emitted only by the 256-thread dispatch specialization, so the normal
    // 128-thread production path compiles them out.
    kDispatchMilestoneInit              = 47,
    kDispatchMilestoneInitComplete      = 48,
    kDispatchMilestoneGridSyncEnter     = 49,
    kDispatchMilestoneGridSyncExit      = 50,
    kDispatchMilestoneNvlinkEnter       = 51,
    kDispatchMilestoneNvlinkExit        = 52,
    kDispatchMilestonePullLoopStart     = 53,
    kDispatchMilestonePullIterStart     = 54,
    kDispatchMilestoneRingWaitEnter     = 55,
    kDispatchMilestoneRingWaitExit      = 56,
    kDispatchMilestonePullIterEnd       = 57,
    kDispatchMilestonePullLoopEnd       = 58,
    kDispatchMilestoneCleanupEnter      = 59,
    kDispatchMilestoneCleanupExit       = 60,
    kDispatchMilestoneCleanupNvlinkEnter = 61,
    kDispatchMilestoneCleanupNvlinkExit  = 62,
    kDispatchMilestoneRegDeallocEnter   = 63,
    kDispatchMilestoneRegDeallocExit    = 64,
    kNonEpilogueRegDeallocEnter         = 65,
    kNonEpilogueRegDeallocExit          = 66,
    kEpilogueRegAllocEnter              = 67,
    kEpilogueRegAllocExit               = 68,
    kDispatchEpilogueBarrierEnter       = 69,
    kDispatchEpilogueBarrierExit        = 70,
};

// Device-side recording granularity gate.
//
// 1 = record every event (task + phase); the historical behaviour.
// 0 = record only task-layer and ignored-anchor events; phase-layer events are
//     never written, so they cost nothing on the device.  This is the fast-
//     instrument overhead ladder step 1 ("取消 phase，只测 task") applied at the
//     source instead of only at display time: when the delivered trace is
//     task-granular, phase records were previously written and then discarded.
//
// The gate lives here, in the single record path, so no operation source has
// to change and every one of the ~58 start/end call sites honours it.
#ifndef DG_MEGA_PROFILE_RECORD_PHASE
#define DG_MEGA_PROFILE_RECORD_PHASE 1
#endif
#ifndef DG_MEGA_PROFILE_TS_ONLY
#define DG_MEGA_PROFILE_TS_ONLY 0
#endif
#ifndef DG_MEGA_PROFILE_NO_COUNTER
#define DG_MEGA_PROFILE_NO_COUNTER 0
#endif
#ifndef DG_MEGA_PROFILE_RECORD_WORKER
#define DG_MEGA_PROFILE_RECORD_WORKER 1
#endif

static constexpr uint32_t kMegaMoEProfileAbiRevision = 3;
// Rev3 keeps the record width fixed and makes the per-CTA physical-warp track
// count a launch-topology compile-time parameter.  The dispatch role has four
// or eight warps, the non-epilogue role has four warps, and two or three
// epilogue warpgroups contribute eight or twelve tracks.  The resulting
// 16/20/24-track layouts are all decoded from the same record ABI.
static constexpr uint32_t kMegaMoEProfileDefaultNumTracks = 20;
static constexpr uint32_t kMegaMoEProfileMaxNumTracks = 24;
static constexpr uint32_t kMegaMoEProfileRecordWords = 4;
static constexpr uint32_t kMegaMoEProfileCaptureMapWords = 2;
static constexpr uint32_t kMegaMoEProfileCounterWords = 2;
static constexpr uint32_t kMegaMoEProfileFixedHeaderWords = 8;
static constexpr uint32_t kMegaMoEProfileMaxRecords = (1u << 24) - 1u;
static constexpr uint32_t kMegaMoEProfileDisabledCaptureSlot = 0xffu;
static constexpr uint64_t kMegaMoEProfileMagic = 0x32454c49464f5250ull;

__forceinline__ __device__ MegaMoEProfileLayer mega_moe_profile_layer(
    MegaMoEProfileTask task
) {
    switch (task) {
        case MegaMoEProfileTask::kEntryRankSync:
            return MegaMoEProfileLayer::kIgnored;
        case MegaMoEProfileTask::kDispatchWorkerTask:
        case MegaMoEProfileTask::kScatterTileTask:
        case MegaMoEProfileTask::kLoaderAWorkerTask:
        case MegaMoEProfileTask::kLoaderBWorkerTask:
        case MegaMoEProfileTask::kMathL1TileTask:
        case MegaMoEProfileTask::kMathL2TileTask:
        case MegaMoEProfileTask::kCombineWorkerTask:
        case MegaMoEProfileTask::kOpeningSetup:
        case MegaMoEProfileTask::kOpeningHistogram:
        case MegaMoEProfileTask::kDispatchGridSyncCompute:
        case MegaMoEProfileTask::kDispatchPull:
        case MegaMoEProfileTask::kDispatchCleanupTask:
        case MegaMoEProfileTask::kTmaA:
        case MegaMoEProfileTask::kTmaB:
        case MegaMoEProfileTask::kUmma:
        case MegaMoEProfileTask::kL1EpilogueSwiGLUQuantStore:
        case MegaMoEProfileTask::kL2EpilogueConvertScatter:
        case MegaMoEProfileTask::kEpilogueGridSyncCompute:
        case MegaMoEProfileTask::kCombineReduceStore:
        case MegaMoEProfileTask::kDispatchMilestoneInit:
        case MegaMoEProfileTask::kDispatchMilestoneInitComplete:
        case MegaMoEProfileTask::kDispatchMilestoneGridSyncEnter:
        case MegaMoEProfileTask::kDispatchMilestoneGridSyncExit:
        case MegaMoEProfileTask::kDispatchMilestoneNvlinkEnter:
        case MegaMoEProfileTask::kDispatchMilestoneNvlinkExit:
        case MegaMoEProfileTask::kDispatchMilestonePullLoopStart:
        case MegaMoEProfileTask::kDispatchMilestonePullIterStart:
        case MegaMoEProfileTask::kDispatchMilestoneRingWaitEnter:
        case MegaMoEProfileTask::kDispatchMilestoneRingWaitExit:
        case MegaMoEProfileTask::kDispatchMilestonePullIterEnd:
        case MegaMoEProfileTask::kDispatchMilestonePullLoopEnd:
        case MegaMoEProfileTask::kDispatchMilestoneCleanupEnter:
        case MegaMoEProfileTask::kDispatchMilestoneCleanupExit:
        case MegaMoEProfileTask::kDispatchMilestoneCleanupNvlinkEnter:
        case MegaMoEProfileTask::kDispatchMilestoneCleanupNvlinkExit:
        case MegaMoEProfileTask::kDispatchMilestoneRegDeallocEnter:
        case MegaMoEProfileTask::kDispatchMilestoneRegDeallocExit:
        case MegaMoEProfileTask::kNonEpilogueRegDeallocEnter:
        case MegaMoEProfileTask::kNonEpilogueRegDeallocExit:
        case MegaMoEProfileTask::kEpilogueRegAllocEnter:
        case MegaMoEProfileTask::kEpilogueRegAllocExit:
        case MegaMoEProfileTask::kDispatchEpilogueBarrierEnter:
        case MegaMoEProfileTask::kDispatchEpilogueBarrierExit:
            return MegaMoEProfileLayer::kTask;
        default:
            return MegaMoEProfileLayer::kPhase;
    }
}

__forceinline__ __device__ MegaMoEProfileRole mega_moe_profile_role(
    MegaMoEProfileTask task
) {
    switch (task) {
        case MegaMoEProfileTask::kEntryRankSync:
            return MegaMoEProfileRole::kIgnored;
        case MegaMoEProfileTask::kDispatchRoutePublish:
        case MegaMoEProfileTask::kDispatchNvlinkPull:
        case MegaMoEProfileTask::kDispatchCleanup:
        case MegaMoEProfileTask::kDispatchSync:
        case MegaMoEProfileTask::kDispatchWaitBeforeCleanup:
        case MegaMoEProfileTask::kDispatchWorkerTask:
        case MegaMoEProfileTask::kOpeningSetup:
        case MegaMoEProfileTask::kOpeningHistogram:
        case MegaMoEProfileTask::kDispatchGridSyncCompute:
        case MegaMoEProfileTask::kDispatchPull:
        case MegaMoEProfileTask::kDispatchCleanupTask:
        case MegaMoEProfileTask::kDispatchMilestoneInit:
        case MegaMoEProfileTask::kDispatchMilestoneInitComplete:
        case MegaMoEProfileTask::kDispatchMilestoneGridSyncEnter:
        case MegaMoEProfileTask::kDispatchMilestoneGridSyncExit:
        case MegaMoEProfileTask::kDispatchMilestoneNvlinkEnter:
        case MegaMoEProfileTask::kDispatchMilestoneNvlinkExit:
        case MegaMoEProfileTask::kDispatchMilestonePullLoopStart:
        case MegaMoEProfileTask::kDispatchMilestonePullIterStart:
        case MegaMoEProfileTask::kDispatchMilestoneRingWaitEnter:
        case MegaMoEProfileTask::kDispatchMilestoneRingWaitExit:
        case MegaMoEProfileTask::kDispatchMilestonePullIterEnd:
        case MegaMoEProfileTask::kDispatchMilestonePullLoopEnd:
        case MegaMoEProfileTask::kDispatchMilestoneCleanupEnter:
        case MegaMoEProfileTask::kDispatchMilestoneCleanupExit:
        case MegaMoEProfileTask::kDispatchMilestoneCleanupNvlinkEnter:
        case MegaMoEProfileTask::kDispatchMilestoneCleanupNvlinkExit:
        case MegaMoEProfileTask::kDispatchMilestoneRegDeallocEnter:
        case MegaMoEProfileTask::kDispatchMilestoneRegDeallocExit:
        case MegaMoEProfileTask::kNonEpilogueRegDeallocEnter:
        case MegaMoEProfileTask::kNonEpilogueRegDeallocExit:
        case MegaMoEProfileTask::kEpilogueRegAllocEnter:
        case MegaMoEProfileTask::kEpilogueRegAllocExit:
        case MegaMoEProfileTask::kDispatchEpilogueBarrierEnter:
        case MegaMoEProfileTask::kDispatchEpilogueBarrierExit:
            return MegaMoEProfileRole::kDispatch;
        case MegaMoEProfileTask::kLoaderAWaitPool:
        case MegaMoEProfileTask::kLoaderAPipeline:
        case MegaMoEProfileTask::kLoaderAWaitEmpty:
        case MegaMoEProfileTask::kLoaderAWorkerTask:
        case MegaMoEProfileTask::kTmaA:
            return MegaMoEProfileRole::kLoaderA;
        case MegaMoEProfileTask::kLoaderBPipeline:
        case MegaMoEProfileTask::kLoaderBWaitEmpty:
        case MegaMoEProfileTask::kLoaderBWorkerTask:
        case MegaMoEProfileTask::kTmaB:
            return MegaMoEProfileRole::kLoaderB;
        case MegaMoEProfileTask::kSupportDeferredScatter:
        case MegaMoEProfileTask::kSupportScatterStore:
        case MegaMoEProfileTask::kSupportScatterRankSync:
        case MegaMoEProfileTask::kSupportWaitForL2:
        case MegaMoEProfileTask::kScatterTileTask:
            return MegaMoEProfileRole::kScatter;
        case MegaMoEProfileTask::kMathCombineReduceStore:
        case MegaMoEProfileTask::kCombineWorkerTask:
        case MegaMoEProfileTask::kCombineReduceStore:
            return MegaMoEProfileRole::kCombine;
        default:
            return MegaMoEProfileRole::kMath;
    }
}

template <bool kEnabled, uint32_t kNumSMs, bool kFull = true,
          uint32_t kNumTracks = kMegaMoEProfileDefaultNumTracks>
struct MegaMoEProfiler;

template <uint32_t kNumSMs, bool kFull, uint32_t kNumTracks>
struct MegaMoEProfiler<false, kNumSMs, kFull, kNumTracks> {
    __forceinline__ __device__ void init(int64_t*, uint32_t, uint32_t) {}
    __forceinline__ __device__ void init(
        int64_t*, uint32_t, const int32_t*, uint32_t, uint32_t,
        uint32_t, MegaMoEProfileCtaRole
    ) {}
    __forceinline__ __device__ void init_tail(
        int64_t*, uint32_t, uint32_t
    ) {}
    __forceinline__ __device__ void entry_start(int64_t*, uint32_t) {}
    __forceinline__ __device__ void entry_end(int64_t*, uint32_t) {}
    __forceinline__ __device__ uint64_t timestamp_ns() { return 0; }
    __forceinline__ __device__ void start_at(
        MegaMoEProfileTask, uint64_t, int64_t*, uint32_t,
        uint32_t = 0, uint32_t = 0
    ) {}
    __forceinline__ __device__ void start(
        MegaMoEProfileTask, int64_t*, uint32_t, uint32_t = 0, uint32_t = 0
    ) {}
    __forceinline__ __device__ void end(
        MegaMoEProfileTask, int64_t*, uint32_t, uint32_t = 0, uint32_t = 0
    ) {}
};

template <uint32_t kNumSMs, bool kFull, uint32_t kNumTracks>
struct MegaMoEProfiler<true, kNumSMs, kFull, kNumTracks> {
    static_assert(kNumTracks == 16 or kNumTracks == 20 or kNumTracks == 24,
                  "rev3 profiler supports 16/20/24-track launches");
    uint64_t state = kDisabledState;
    // Register-minimal record path: the profiler keeps ONLY `state` resident.
    // The loop-invariant bases are recomputed per record from one L1-cached
    // header load (buf[2] carries capture_capacity); on the register-starved
    // bm128 dual-math build the previous four cached members (9+ persistent
    // registers per thread) pushed ptxas into spilling the WGMMA accumulator
    // working set around every record site (1790 LDL / 1245 STL vs 262/186
    // uninstrumented), costing +46% kernel time.  Bytes written are unchanged,
    // so the host-side ABI and decoder see exactly the same records.

    static constexpr uint32_t kAttemptedBits = 24;
    static constexpr uint64_t kAttemptedMask = (1ull << kAttemptedBits) - 1ull;
    static constexpr uint32_t kCaptureSlotShift = 24;
    static constexpr uint32_t kPhysicalWarpShift = 32;
    static constexpr uint32_t kCtaRoleShift = 37;
    static constexpr uint32_t kCaptureCapacityShift = 40;
    static constexpr uint64_t kDisabledState = ~0ull;

    __forceinline__ __device__ void init(
        int64_t* buf, uint32_t max_records,
        const int32_t* selected_physical_smids_to_capture_slot,
        uint32_t capture_capacity, uint32_t physical_warp,
        uint32_t observer_warp_mask, MegaMoEProfileCtaRole cta_role
    ) {
        state = kDisabledState;
        if (buf == nullptr || max_records == 0 ||
            max_records > kMegaMoEProfileMaxRecords ||
            selected_physical_smids_to_capture_slot == nullptr ||
            capture_capacity == 0 || capture_capacity > 0xffu ||
            physical_warp >= kNumTracks)
            return;

        uint32_t physical_smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(physical_smid));
        if (physical_smid >= kNumSMs)
            return;
        const int32_t mapped_capture_slot =
            selected_physical_smids_to_capture_slot[physical_smid];
        if (mapped_capture_slot < 0 ||
            static_cast<uint32_t>(mapped_capture_slot) >= capture_capacity)
            return;
        const uint32_t capture_slot = static_cast<uint32_t>(mapped_capture_slot);

        if (physical_warp == 0) {
            buf[0] = static_cast<int64_t>(kMegaMoEProfileMagic);
            buf[1] = static_cast<int64_t>(
                static_cast<uint64_t>(kMegaMoEProfileAbiRevision) |
                (static_cast<uint64_t>(kFull ? MegaMoEProfileMode::kFull
                                             : MegaMoEProfileMode::kEntryOnly) << 32));
            buf[2] = static_cast<int64_t>(
                static_cast<uint64_t>(capture_capacity) |
                (static_cast<uint64_t>(kNumTracks) << 32));
            buf[3] = static_cast<int64_t>(
                static_cast<uint64_t>(max_records) |
                (static_cast<uint64_t>(kNumSMs) << 32));
            buf[4] = static_cast<int64_t>(
                static_cast<uint64_t>(kMegaMoEProfileRecordWords) |
                (static_cast<uint64_t>(kMegaMoEProfileCounterWords) << 32));
            buf[5] = static_cast<int64_t>(kMegaMoEProfileCaptureMapWords);
            buf[6] = 0;
            buf[7] = 0;
            const uint64_t capture_map =
                static_cast<uint64_t>(blockIdx.x) |
                (static_cast<uint64_t>(physical_smid) << 32);
            const uint64_t capture_role =
                static_cast<uint64_t>(cta_role) |
                (static_cast<uint64_t>(kNumTracks) << 32);
            const uint64_t map_base = kMegaMoEProfileFixedHeaderWords +
                                      static_cast<uint64_t>(capture_slot) *
                                          kMegaMoEProfileCaptureMapWords;
            buf[map_base] = static_cast<int64_t>(capture_map);
            buf[map_base + 1] = static_cast<int64_t>(capture_role);
        }

        const uint64_t stream_index =
            static_cast<uint64_t>(capture_slot) * kNumTracks +
            physical_warp;
        const uint64_t counter_base = counters_base(capture_capacity) +
                                      stream_index * kMegaMoEProfileCounterWords;
        buf[counter_base] = 0;
        buf[counter_base + 1] = static_cast<int64_t>(max_records);
        state = static_cast<uint64_t>(capture_slot) << kCaptureSlotShift |
                static_cast<uint64_t>(physical_warp) << kPhysicalWarpShift |
                static_cast<uint64_t>(cta_role) << kCtaRoleShift |
                static_cast<uint64_t>(capture_capacity) << kCaptureCapacityShift;
        if ((observer_warp_mask & (1u << physical_warp)) == 0u)
            state = kDisabledState;
    }

    // A rev3 capture stream is well formed only when every track has a
    // (written=0, max_records, overflow=0) counter.  Warp 0 lane 0 fills the
    // tracks beyond the launch's physical warp count once per captured CTA.
    __forceinline__ __device__ void init_tail(
        int64_t* buf, uint32_t max_records, uint32_t first_track
    ) {
        if (state == kDisabledState)
            return;
        const uint32_t capture_slot =
            static_cast<uint32_t>((state >> kCaptureSlotShift) & 0xffu);
        const uint32_t capture_capacity =
            static_cast<uint32_t>((state >> kCaptureCapacityShift) & 0xffu);
        for (uint32_t track = first_track;
             track < kNumTracks; ++ track) {
            const uint64_t stream_index =
                static_cast<uint64_t>(capture_slot) * kNumTracks +
                track;
            const uint64_t counter_index =
                counters_base(capture_capacity) +
                stream_index * kMegaMoEProfileCounterWords;
            buf[counter_index] = 0;
            buf[counter_index + 1] = static_cast<int64_t>(max_records);
        }
    }

    __forceinline__ __device__ void entry_start(
        int64_t* buf, uint32_t max_records
    ) {
        record(MegaMoEProfileTask::kEntryRankSync, true, buf, max_records, 0, 0);
    }

    __forceinline__ __device__ void entry_end(
        int64_t* buf, uint32_t max_records
    ) {
        record(MegaMoEProfileTask::kEntryRankSync, false, buf, max_records, 0, 0);
    }

    __forceinline__ __device__ uint64_t timestamp_ns() {
        if constexpr (kFull) {
            if (state != kDisabledState) {
                uint64_t value;
                asm volatile("mov.u64 %0, %%globaltimer;"
                             : "=l"(value));
                return value;
            }
        }
        return 0;
    }

    __forceinline__ __device__ void start_at(
        MegaMoEProfileTask task, uint64_t timestamp_ns,
        int64_t* buf, uint32_t max_records,
        uint32_t identity_a = 0, uint32_t identity_b = 0
    ) {
        if constexpr (kFull)
            record_at(task, true, timestamp_ns, buf, max_records,
                      identity_a, identity_b);
    }

    __forceinline__ __device__ void start(
        MegaMoEProfileTask task, int64_t* buf, uint32_t max_records,
        uint32_t identity_a = 0, uint32_t identity_b = 0
    ) {
        if constexpr (kFull)
            record(task, true, buf, max_records, identity_a, identity_b);
    }

    __forceinline__ __device__ void end_at(
        MegaMoEProfileTask task, uint64_t timestamp_ns,
        int64_t* buf, uint32_t max_records,
        uint32_t identity_a = 0, uint32_t identity_b = 0
    ) {
        if constexpr (kFull)
            record_at(task, false, timestamp_ns, buf, max_records,
                      identity_a, identity_b);
    }

    __forceinline__ __device__ void end(
        MegaMoEProfileTask task, int64_t* buf, uint32_t max_records,
        uint32_t identity_a = 0, uint32_t identity_b = 0
    ) {
        if constexpr (kFull)
            record(task, false, buf, max_records, identity_a, identity_b);
    }

private:
    __forceinline__ __device__ static uint64_t counters_base(
        uint32_t capture_capacity
    ) {
        return kMegaMoEProfileFixedHeaderWords +
               static_cast<uint64_t>(capture_capacity) *
                   kMegaMoEProfileCaptureMapWords;
    }

    __forceinline__ __device__ static uint64_t records_base(
        uint32_t capture_capacity
    ) {
        return counters_base(capture_capacity) +
               static_cast<uint64_t>(capture_capacity) *
                   kNumTracks * kMegaMoEProfileCounterWords;
    }

    __forceinline__ __device__ void record(
        MegaMoEProfileTask task, bool is_start, int64_t* buf,
        uint32_t max_records, uint32_t identity_a, uint32_t identity_b
    ) {
        if (state == kDisabledState)
            return;
#if !DG_MEGA_PROFILE_RECORD_PHASE
        if (mega_moe_profile_layer(task) == MegaMoEProfileLayer::kPhase)
            return;
#endif
#if !DG_MEGA_PROFILE_RECORD_WORKER
        if (task == MegaMoEProfileTask::kDispatchWorkerTask ||
            task == MegaMoEProfileTask::kScatterTileTask ||
            task == MegaMoEProfileTask::kLoaderAWorkerTask ||
            task == MegaMoEProfileTask::kLoaderBWorkerTask ||
            task == MegaMoEProfileTask::kCombineWorkerTask)
            return;
#endif
        uint64_t timestamp_ns;
        // No "memory" clobber: a timestamp read does not interact with the
        // surrounding memory state, and the clobber forced ptxas to drain and
        // reload the live working set around every record site.
        asm volatile("mov.u64 %0, %%globaltimer;"
                     : "=l"(timestamp_ns));
        record_at(task, is_start, timestamp_ns, buf, max_records,
                  identity_a, identity_b);
    }

    __forceinline__ __device__ void record_at(
        MegaMoEProfileTask task, bool is_start, uint64_t timestamp_ns,
        int64_t* buf, uint32_t max_records,
        uint32_t identity_a, uint32_t identity_b
    ) {
        if (state == kDisabledState)
            return;
#if !DG_MEGA_PROFILE_RECORD_PHASE
        if (mega_moe_profile_layer(task) == MegaMoEProfileLayer::kPhase)
            return;
#endif

        const uint32_t attempted = static_cast<uint32_t>(state & kAttemptedMask);
        const uint32_t capture_slot =
            static_cast<uint32_t>((state >> kCaptureSlotShift) & 0xffu);
        const uint32_t physical_warp =
            static_cast<uint32_t>((state >> kPhysicalWarpShift) & 0x1fu);
        const uint32_t cta_role =
            static_cast<uint32_t>((state >> kCtaRoleShift) & 0x3u);
        uint32_t physical_smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(physical_smid));
        // capture_capacity is packed in `state`; reading the shared header here
        // races a peer CTA's first init and could turn the record base into 0.
        const uint32_t capture_capacity =
            static_cast<uint32_t>((state >> kCaptureCapacityShift) & 0xffu);
        const uint64_t stream_index =
            static_cast<uint64_t>(capture_slot) * kNumTracks +
            physical_warp;

#if DG_MEGA_PROFILE_TS_ONLY
        // Bisection: timestamp read + state update only, no stores.
        const uint32_t next_attempted0 =
            attempted < kMegaMoEProfileMaxRecords ? attempted + 1u : attempted;
        state = (state & ~kAttemptedMask) | next_attempted0;
        (void)buf; (void)identity_a; (void)identity_b;
        return;
#endif
        const bool can_write = attempted < max_records;
        const uint32_t written = can_write ? attempted + 1u : max_records;
        const uint32_t next_attempted =
            attempted < kMegaMoEProfileMaxRecords ? attempted + 1u : attempted;
        const uint32_t overflow = can_write ? 0u : 1u;

        if (can_write) {
            // Minimal-liveness emission: every word is computed immediately
            // before its store so ptxas never needs the whole payload live at
            // once (the bm128 register-starved build spilled the surrounding
            // WGMMA working set around the previous batched form).
            const uint64_t record_index =
                records_base(capture_capacity) +
                (stream_index * static_cast<uint64_t>(max_records) +
                 attempted) * kMegaMoEProfileRecordWords;
            buf[record_index] = static_cast<int64_t>(timestamp_ns);
            buf[record_index + 1] = static_cast<int64_t>(
                static_cast<uint64_t>(task) |
                (static_cast<uint64_t>(mega_moe_profile_layer(task)) << 16) |
                (static_cast<uint64_t>(is_start ? 1u : 0u) << 18) |
                (static_cast<uint64_t>(mega_moe_profile_role(task)) << 19));
            buf[record_index + 2] = static_cast<int64_t>(
                static_cast<uint64_t>(blockIdx.x) |
                (static_cast<uint64_t>(physical_smid) << 16) |
                (static_cast<uint64_t>(physical_warp) << 24) |
                (static_cast<uint64_t>(capture_slot) << 32) |
                (static_cast<uint64_t>(cta_role) << 40));
            buf[record_index + 3] = static_cast<int64_t>(
                static_cast<uint64_t>(identity_a) |
                (static_cast<uint64_t>(identity_b) << 32));
        }

#if !DG_MEGA_PROFILE_NO_COUNTER
        const uint64_t counter_index =
            counters_base(capture_capacity) +
            stream_index * kMegaMoEProfileCounterWords;
        buf[counter_index] = static_cast<int64_t>(
            static_cast<uint64_t>(next_attempted) |
            (static_cast<uint64_t>(written) << 32));
        buf[counter_index + 1] = static_cast<int64_t>(
            static_cast<uint64_t>(max_records) |
            (static_cast<uint64_t>(overflow) << 32));
#else
        (void)written; (void)overflow;
#endif
        state = (state & ~kAttemptedMask) | next_attempted;
    }
};

}  // namespace deep_gemm
