#pragma once

#include <cstdint>

namespace deep_gemm {

// Canonical WeaveProfiler FILITE02 tags & semantic event IDs (knowledge/semantic_ids.json)
constexpr uint32_t kAgyFiTagEnd    = 0x80000000u;
constexpr uint32_t kAgyFiTagOpen   = 0x40000000u;
constexpr uint32_t kAgyFiTagMark   = 0x20000000u;
constexpr uint32_t kAgyFiTagClk    = 0x10000000u;
constexpr uint32_t kAgyFiTagClose  = 0x08000000u;
constexpr uint32_t kAgyFiTagIdMask = 0x0fffu;

constexpr uint32_t kAgyEvLoad          = 1u;   // "load"          : TMA A / TMA B tile load
constexpr uint32_t kAgyEvWait          = 2u;   // "wait"          : L1 pool ready wait / tile_sync wait
constexpr uint32_t kAgyEvMma           = 3u;   // "mma"           : Math WG WGMMA K-loop
constexpr uint32_t kAgyEvCompute       = 4u;   // "compute"       : SMEM expert cache / prefix scan
constexpr uint32_t kAgyEvEpilogue      = 5u;   // "epilogue"      : SwiGLU / quantize
constexpr uint32_t kAgyEvStore         = 6u;   // "store"         : L1 TMA store / L2 NVLink scatter
constexpr uint32_t kAgyEvBarrier       = 7u;   // "barrier"       : Fused L1->L2 grid_sync / CTA handoff
constexpr uint32_t kAgyEvDispatch      = 8u;   // "dispatch"      : Dispatch notify / cleanup_workspace
constexpr uint32_t kAgyEvCommunication = 9u;   // "communication" : NVLink bar1/bar2/bar3 & token pull
constexpr uint32_t kAgyEvGemm          = 10u;  // "gemm"          : GEMM phase envelope
constexpr uint32_t kAgyEvReduction     = 12u;  // "reduction"     : Combine BF16 top-k reduction
constexpr uint32_t kAgyEvMoe           = 13u;  // "moe"           : MoE phase
constexpr uint32_t kAgyEvMegakernel    = 15u;  // "megakernel"    : Persistent megakernel envelope

// Role tracks per SM (8 tracks per SM, covering all 78 SMs on H20 + up to 136 SMs on SM90)
constexpr uint32_t kAgyProfTrackDispW0    = 0u;
constexpr uint32_t kAgyProfTrackTmaA      = 1u;
constexpr uint32_t kAgyProfTrackTmaB      = 2u;
constexpr uint32_t kAgyProfTrackMathWG0   = 3u;
constexpr uint32_t kAgyProfTrackMathWG1   = 4u;
constexpr uint32_t kAgyProfTrackMathWG2   = 5u;
constexpr uint32_t kAgyProfTrackMathWG3   = 6u;
constexpr uint32_t kAgyProfTrackDispWLast = 7u;

constexpr uint64_t kAgyProfMagic      = 0x314652505947415full;  // "_AGYPRF1"
constexpr uint32_t kAgyProfVersion    = 1u;
constexpr uint32_t kAgyProfMaxSMs     = 136u;
constexpr uint32_t kAgyProfNumTracks  = 8u;
constexpr uint32_t kAgyProfTrackCap   = 512u;

struct alignas(16) AgyFiRec {
    uint64_t t;
    uint32_t tag;
    uint32_t aux;
};
static_assert(sizeof(AgyFiRec) == 16, "AgyFiRec must be 16 bytes (FILITE02 ABI)");

// 512-byte (64 x u64 = 4 x 128B cachelines) per-SM summary record; zero false sharing across SMs.
struct alignas(128) AgyProfSmRecord {
    // [0..3] Metadata & counters
    uint64_t physical_smid;           // [0]  lo32 = %smid, bit32 = 1 valid marker
    uint64_t tile_counts;             // [1]  lo32 = l1_tiles, hi32 = l2_tiles
    uint64_t disp_tokens_and_warps;   // [2]  lo32 = l1_l2_active_flags, hi32 = num_disp_warps
    uint64_t reserved_meta;           // [3]
    // [4..7] Per-track FiRec write cursors (8 x u32 = 32B = 4 x u64)
    uint32_t track_cursors[kAgyProfNumTracks];

    // [8..19] Role 0: Dispatch warps timestamps (ns)
    uint64_t t_cta_entry;             // [8]  kernel entry before __syncthreads()
    uint64_t t_after_cta_init;        // [9]  immediately after __syncthreads()
    uint64_t t_before_nvlink_bar1;    // [10] end of Notify, before nvlink_barrier(tag=1)
    uint64_t t_after_nvlink_bar1;     // [11] after nvlink_barrier(tag=1)
    uint64_t t_before_pull;           // [12] after expert_cache / sync / fetch_recv_count, before pull loop
    uint64_t t_after_pull_w0;         // [13] warp 0 end of pull loop (+ dedup FIFO drain)
    uint64_t t_cleanup_start;         // [14] before cleanup_workspace()
    uint64_t t_cleanup_end;           // [15] after cleanup_workspace()
    uint64_t t_after_nvlink_bar3;     // [16] after nvlink_barrier(tag=3), Role 1 exit
    uint64_t t_l2_disp_prologue_end;  // [17] Split-L2 Role 1 after fetch_recv_count + first sync_unaligned
    uint64_t t_disp_pull_start_min;   // [18] min pull start across dispatch warps on this SM
    uint64_t t_disp_pull_end_max;     // [19] max pull end across ALL dispatch warps on this SM (atomicMax)

    // [20..31] Role 1 & Role 2: TMA A & TMA B timestamps & wait accumulators (ns)
    uint64_t t_tma_a_l1_start;        // [20]
    uint64_t t_tma_a_l1_end;          // [21]
    uint64_t l1_pool_wait_ns;         // [22] cumulative ns TMA A spun on l1_full_count_ptr == valid_m
    uint64_t tile_sync_wait_ns;       // [23] cumulative ns TMA A spun on L2 tile_sync (if enabled)
    uint64_t t_tma_a_sync_start;      // [24] Fused L1->L2 grid_sync start on TMA A
    uint64_t t_tma_a_sync_end;        // [25] Fused L1->L2 grid_sync + proxy fence end on TMA A
    uint64_t t_tma_a_l2_start;        // [26]
    uint64_t t_tma_a_l2_end;          // [27]
    uint64_t t_tma_b_l1_start;        // [28]
    uint64_t t_tma_b_l1_end;          // [29]
    uint64_t t_tma_b_l2_start;        // [30]
    uint64_t t_tma_b_l2_end;          // [31]

    // [32..55] Role 3: Math Warpgroups WG0..WG3 (warp_idx_in_wg == 0, lane == 0)
    uint64_t t_wg_l1_start[4];        // [32..35]
    uint64_t t_wg_l1_end[4];          // [36..39]
    uint64_t t_wg_sync_start[4];      // [40..43]
    uint64_t t_wg_sync_end[4];        // [44..47]
    uint64_t t_wg_l2_start[4];        // [48..51]
    uint64_t t_wg_l2_end[4];          // [52..55]

    // [56..63] Role 3 Tail: NVLink Barrier 2 + Combine
    uint64_t t_before_nvlink_bar2;    // [56] epilogue_thread_idx == 0 before nvlink_barrier(tag=2)
    uint64_t t_after_nvlink_bar2;     // [57] epilogue_thread_idx == 0 after nvlink_barrier(tag=2)
    uint64_t t_combine_start;         // [58] epilogue_thread_idx == 0 after pre-cleanup sync_unaligned
    uint64_t t_wg_combine_end[4];     // [59..62] atomicMax across the 4 warps in each WG
    uint64_t t_disp_wlast_after_pull; // [63] last dispatch warp pull end
};
static_assert(sizeof(AgyProfSmRecord) == 512, "AgyProfSmRecord must be exactly 512 bytes");

struct alignas(128) AgyProfLaunchSlot {
    uint64_t magic;                   // [0] kAgyProfMagic when written by SM 0
    uint32_t version;                 // [1.lo] kAgyProfVersion
    uint32_t phase_kind;              // [1.hi] 1=Linear1, 2=Linear2, 3=Fused
    uint32_t grid_sms;                // [2.lo] kNumSMs
    uint32_t overflow_flags;          // [2.hi] bit0 = event cap overflow, bit1 = sm_idx >= MaxSMs
    uint32_t num_disp_threads;        // [3.lo]
    uint32_t num_epi_threads;         // [3.hi]
    uint32_t block_m;                 // [4.lo]
    uint32_t block_n;                 // [4.hi]
    uint32_t block_k;                 // [5.lo]
    uint32_t num_stages;              // [5.hi]
    uint32_t prof_level;              // [6.lo] AGY_PROF (1 or 2)
    uint32_t num_tokens_arg;          // [6.hi] runtime num_tokens
    uint64_t pad_hdr[9];              // pad header to 128 bytes
    AgyProfSmRecord sm[kAgyProfMaxSMs];
    AgyFiRec events[kAgyProfMaxSMs][kAgyProfNumTracks][kAgyProfTrackCap];
};
static_assert(offsetof(AgyProfLaunchSlot, sm) == 128, "LaunchSlot sm offset must be 128");

struct alignas(128) AgyProfDeviceBuffer {
    // slot[0]: Linear1 (Split Official) OR Fused (Ours Megakernel)
    // slot[1]: Linear2 (Split Official; unused/zero in Fused mode)
    AgyProfLaunchSlot slots[2];
};

#if defined(AGY_PROF) && (AGY_PROF > 0)

__device__ __forceinline__ uint64_t agy_prof_globaltimer_ns() {
    uint64_t ts;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));
    return ts;
}

__device__ __forceinline__ uint32_t agy_prof_smid() {
    uint32_t s;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    return s;
}

__device__ __forceinline__ uint32_t agy_prof_pack_tile_aux(
    const bool is_l1, const uint32_t expert_idx, const uint32_t m_blk, const uint32_t n_blk) {
    return ((is_l1 ? 0u : 1u) << 31) |
           ((expert_idx & 0x7fu) << 24) |
           ((m_blk & 0xfffu) << 12) |
           (n_blk & 0xfffu);
}

__device__ __forceinline__ void agy_prof_emit_rec(
    AgyProfLaunchSlot* __restrict__ slot,
    const uint32_t sm_idx,
    const uint32_t track_id,
    const uint64_t ts,
    const uint32_t tag,
    const uint32_t aux) {
    if (slot == nullptr) return;
    if (sm_idx >= kAgyProfMaxSMs || track_id >= kAgyProfNumTracks) {
        atomicOr(&slot->overflow_flags, 2u);
        return;
    }
    uint32_t idx = slot->sm[sm_idx].track_cursors[track_id];
    if (idx < kAgyProfTrackCap) {
        AgyFiRec* dst = &slot->events[sm_idx][track_id][idx];
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(dst),
                        "r"(static_cast<uint32_t>(ts)),
                        "r"(static_cast<uint32_t>(ts >> 32)),
                        "r"(tag),
                        "r"(aux));
        slot->sm[sm_idx].track_cursors[track_id] = idx + 1u;
    } else if (idx == kAgyProfTrackCap) {
        atomicOr(&slot->overflow_flags, 1u);
        slot->sm[sm_idx].track_cursors[track_id] = idx + 1u;
    }
}

__device__ __forceinline__ void agy_prof_emit_now(
    AgyProfLaunchSlot* __restrict__ slot,
    const uint32_t sm_idx,
    const uint32_t track_id,
    const uint32_t tag,
    const uint32_t aux) {
    if (slot == nullptr) return;
    const uint64_t ts = agy_prof_globaltimer_ns();
    agy_prof_emit_rec(slot, sm_idx, track_id, ts, tag, aux);
}

#define AGY_PROF_KERNEL_ARG_DECL , uint64_t* const agy_prof_raw_ptr
#define AGY_PROF_CORE_ARG_DECL   , uint64_t* const agy_prof_raw_ptr
#define AGY_PROF_KERNEL_ARG_PASS , agy_prof_raw_ptr

#define AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT) \
    ((agy_prof_raw_ptr == nullptr) ? nullptr : \
     (&reinterpret_cast<AgyProfDeviceBuffer*>(agy_prof_raw_ptr)->slots[ \
         (MegaMoEPhaseT::runs_linear2 && !MegaMoEPhaseT::runs_linear1) ? 1 : 0]))

#define AGY_PROF_CTA_ENTRY(MegaMoEPhaseT, sm_idx_, tid_, num_tokens_) \
    do { \
        if ((tid_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            _sl->sm[(sm_idx_)].physical_smid = (1ull << 32) | static_cast<uint64_t>(agy_prof_smid()); \
            _sl->sm[(sm_idx_)].t_cta_entry = _ts; \
            if ((sm_idx_) == 0u) { \
                _sl->magic = kAgyProfMagic; \
                _sl->version = kAgyProfVersion; \
                _sl->phase_kind = MegaMoEPhaseT::is_fused ? 3u : (MegaMoEPhaseT::runs_linear1 ? 1u : 2u); \
                _sl->grid_sms = kNumSMs; \
                _sl->num_disp_threads = kNumDispatchThreads; \
                _sl->num_epi_threads = kNumEpilogueThreads; \
                _sl->block_m = BLOCK_M; \
                _sl->block_n = BLOCK_N; \
                _sl->block_k = BLOCK_K; \
                _sl->num_stages = kNumStages; \
                _sl->prof_level = static_cast<uint32_t>(AGY_PROF); \
                _sl->num_tokens_arg = (num_tokens_); \
            } \
        } \
    } while (0)

#define AGY_PROF_AFTER_CTA_INIT(MegaMoEPhaseT, sm_idx_, tid_) \
    do { \
        if ((tid_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            _sl->sm[(sm_idx_)].t_after_cta_init = agy_prof_globaltimer_ns(); \
        } \
    } while (0)

// Role 1 (Dispatch) macros
#define AGY_PROF_DISP_OPEN(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_, num_disp_warps_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((warp_idx_) == 0u) { \
                const uint64_t _t0 = _sl->sm[(sm_idx_)].t_cta_entry; \
                agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, \
                    _t0 ? _t0 : agy_prof_globaltimer_ns(), \
                    kAgyFiTagOpen | kAgyProfTrackDispW0, agy_prof_smid()); \
            } else if ((num_disp_warps_) > 1u && (warp_idx_) + 1u == (num_disp_warps_)) { \
                agy_prof_emit_now(_sl, (sm_idx_), kAgyProfTrackDispWLast, \
                    kAgyFiTagOpen | kAgyProfTrackDispWLast, agy_prof_smid()); \
            } \
        } \
    } while (0)

#define AGY_PROF_SPLIT_L2_PROLOGUE_BEGIN(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_) \
    do { \
        if ((warp_idx_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            agy_prof_emit_now(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT), (sm_idx_), kAgyProfTrackDispW0, \
                kAgyEvCompute, 2u); \
        } \
    } while (0)

#define AGY_PROF_SPLIT_L2_PROLOGUE_END(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_) \
    do { \
        if ((warp_idx_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            _sl->sm[(sm_idx_)].t_l2_disp_prologue_end = _ts; \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvCompute | kAgyFiTagEnd, 2u); \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvBarrier, 2u); \
        } \
    } while (0)

#define AGY_PROF_DISP_CLEANUP_BEGIN(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_, from_wait_barrier_) \
    do { \
        if ((warp_idx_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            _sl->sm[(sm_idx_)].t_cleanup_start = _ts; \
            if (from_wait_barrier_) { \
                agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvBarrier | kAgyFiTagEnd, 2u); \
            } \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvDispatch, 2u); \
        } \
    } while (0)

#define AGY_PROF_DISP_CLEANUP_END(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_) \
    do { \
        if ((warp_idx_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            _sl->sm[(sm_idx_)].t_cleanup_end = _ts; \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvDispatch | kAgyFiTagEnd, 2u); \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvCommunication, 3u); \
        } \
    } while (0)

#define AGY_PROF_DISP_BAR3_END(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_) \
    do { \
        if ((warp_idx_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            _sl->sm[(sm_idx_)].t_after_nvlink_bar3 = _ts; \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvCommunication | kAgyFiTagEnd, 3u); \
        } \
    } while (0)

#define AGY_PROF_DISP_CLOSE(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_, num_disp_warps_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((warp_idx_) == 0u) { \
                agy_prof_emit_now(_sl, (sm_idx_), kAgyProfTrackDispW0, kAgyFiTagClose, 0u); \
            } else if ((num_disp_warps_) > 1u && (warp_idx_) + 1u == (num_disp_warps_)) { \
                agy_prof_emit_now(_sl, (sm_idx_), kAgyProfTrackDispWLast, kAgyFiTagClose, 0u); \
            } \
        } \
    } while (0)

#define AGY_PROF_NOTIFY_BEGIN(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_) \
    do { \
        if ((warp_idx_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            const uint64_t _t0 = _sl->sm[(sm_idx_)].t_cta_entry; \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, \
                _t0 ? _t0 : agy_prof_globaltimer_ns(), kAgyEvDispatch, 1u); \
        } \
    } while (0)

#define AGY_PROF_BEFORE_BAR1(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_) \
    do { \
        if ((warp_idx_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            _sl->sm[(sm_idx_)].t_before_nvlink_bar1 = _ts; \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvDispatch | kAgyFiTagEnd, 1u); \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvCommunication, 1u); \
        } \
    } while (0)

#define AGY_PROF_AFTER_BAR1(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_) \
    do { \
        if ((warp_idx_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            _sl->sm[(sm_idx_)].t_after_nvlink_bar1 = _ts; \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvCommunication | kAgyFiTagEnd, 1u); \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvCompute, 1u); \
        } \
    } while (0)

#define AGY_PROF_BEFORE_PULL(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_, num_disp_warps_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((warp_idx_) == 0u) { \
                _sl->sm[(sm_idx_)].t_before_pull = _ts; \
                _sl->sm[(sm_idx_)].t_disp_pull_start_min = _ts; \
                agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvCompute | kAgyFiTagEnd, 1u); \
                agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvCommunication, 2u); \
            } else if ((num_disp_warps_) > 1u && (warp_idx_) + 1u == (num_disp_warps_)) { \
                agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispWLast, _ts, kAgyEvCommunication, 2u); \
            } \
        } \
    } while (0)

#define AGY_PROF_AFTER_PULL(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_, num_disp_warps_, is_fused_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            atomicMax(reinterpret_cast<unsigned long long*>(&_sl->sm[(sm_idx_)].t_disp_pull_end_max), \
                      static_cast<unsigned long long>(_ts)); \
            if ((warp_idx_) == 0u) { \
                _sl->sm[(sm_idx_)].t_after_pull_w0 = _ts; \
                agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvCommunication | kAgyFiTagEnd, 2u); \
                if (is_fused_) { \
                    agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvBarrier, 2u); \
                } \
            } else if ((num_disp_warps_) > 1u && (warp_idx_) + 1u == (num_disp_warps_)) { \
                _sl->sm[(sm_idx_)].t_disp_wlast_after_pull = _ts; \
                agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispWLast, _ts, kAgyEvCommunication | kAgyFiTagEnd, 2u); \
            } \
        } \
    } while (0)

// Role-generic phase hooks inside for_each_selected_block_role (role_id: 0=TMA A, 1=TMA B, 2=Math WGs)
#define AGY_PROF_ROLE_PHASE_BEGIN(MegaMoEPhaseT, sm_idx_, role_id_, fused_tid_, is_l1_) \
    do { \
        if (agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((role_id_) == 0 && (fused_tid_) == 0u) { \
                const uint64_t _ts = agy_prof_globaltimer_ns(); \
                if (is_l1_) _sl->sm[(sm_idx_)].t_tma_a_l1_start = _ts; \
                else        _sl->sm[(sm_idx_)].t_tma_a_l2_start = _ts; \
            } else if ((role_id_) == 1 && (fused_tid_) == 32u) { \
                const uint64_t _ts = agy_prof_globaltimer_ns(); \
                if (is_l1_) _sl->sm[(sm_idx_)].t_tma_b_l1_start = _ts; \
                else        _sl->sm[(sm_idx_)].t_tma_b_l2_start = _ts; \
            } else if ((role_id_) == 2 && (((fused_tid_) - 64u) & 127u) == 0u) { \
                const uint32_t _wg = ((fused_tid_) - 64u) >> 7; \
                if (_wg < 4u) { \
                    const uint64_t _ts = agy_prof_globaltimer_ns(); \
                    if (is_l1_) _sl->sm[(sm_idx_)].t_wg_l1_start[_wg] = _ts; \
                    else        _sl->sm[(sm_idx_)].t_wg_l2_start[_wg] = _ts; \
                } \
            } \
        } \
    } while (0)

#define AGY_PROF_ROLE_PHASE_END(MegaMoEPhaseT, sm_idx_, role_id_, fused_tid_, is_l1_) \
    do { \
        if (agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((role_id_) == 0 && (fused_tid_) == 0u) { \
                const uint64_t _ts = agy_prof_globaltimer_ns(); \
                if (is_l1_) _sl->sm[(sm_idx_)].t_tma_a_l1_end = _ts; \
                else        _sl->sm[(sm_idx_)].t_tma_a_l2_end = _ts; \
            } else if ((role_id_) == 1 && (fused_tid_) == 32u) { \
                const uint64_t _ts = agy_prof_globaltimer_ns(); \
                if (is_l1_) _sl->sm[(sm_idx_)].t_tma_b_l1_end = _ts; \
                else        _sl->sm[(sm_idx_)].t_tma_b_l2_end = _ts; \
            } else if ((role_id_) == 2 && (((fused_tid_) - 64u) & 127u) == 0u) { \
                const uint32_t _wg = ((fused_tid_) - 64u) >> 7; \
                if (_wg < 4u) { \
                    const uint64_t _ts = agy_prof_globaltimer_ns(); \
                    if (is_l1_) _sl->sm[(sm_idx_)].t_wg_l1_end[_wg] = _ts; \
                    else        _sl->sm[(sm_idx_)].t_wg_l2_end[_wg] = _ts; \
                } \
            } \
        } \
    } while (0)

#define AGY_PROF_ROLE_SYNC_BEGIN(MegaMoEPhaseT, sm_idx_, role_id_, fused_tid_) \
    do { \
        if (agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((role_id_) == 0 && (fused_tid_) == 0u) { \
                const uint64_t _ts = agy_prof_globaltimer_ns(); \
                _sl->sm[(sm_idx_)].t_tma_a_sync_start = _ts; \
                agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackTmaA, _ts, kAgyEvBarrier, 1u); \
            } else if ((role_id_) == 1 && (fused_tid_) == 32u) { \
                agy_prof_emit_now(_sl, (sm_idx_), kAgyProfTrackTmaB, kAgyEvBarrier, 1u); \
            } else if ((role_id_) == 2 && (((fused_tid_) - 64u) & 127u) == 0u) { \
                const uint32_t _wg = ((fused_tid_) - 64u) >> 7; \
                if (_wg < 4u) { \
                    const uint64_t _ts = agy_prof_globaltimer_ns(); \
                    _sl->sm[(sm_idx_)].t_wg_sync_start[_wg] = _ts; \
                    agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackMathWG0 + _wg, _ts, kAgyEvBarrier, 1u); \
                } \
            } \
        } \
    } while (0)

#define AGY_PROF_INTERLEAVE_TILE_BEGIN(MegaMoEPhaseT, sm_idx_, role_id_, fused_tid_, is_l1_) \
    do { \
        if (agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((role_id_) == 0 && (fused_tid_) == 0u) { \
                const uint64_t _ts = agy_prof_globaltimer_ns(); \
                if (is_l1_) { if (_sl->sm[(sm_idx_)].t_tma_a_l1_start == 0ull) _sl->sm[(sm_idx_)].t_tma_a_l1_start = _ts; } \
                else        { if (_sl->sm[(sm_idx_)].t_tma_a_l2_start == 0ull) _sl->sm[(sm_idx_)].t_tma_a_l2_start = _ts; } \
            } else if ((role_id_) == 1 && (fused_tid_) == 32u) { \
                const uint64_t _ts = agy_prof_globaltimer_ns(); \
                if (is_l1_) { if (_sl->sm[(sm_idx_)].t_tma_b_l1_start == 0ull) _sl->sm[(sm_idx_)].t_tma_b_l1_start = _ts; } \
                else        { if (_sl->sm[(sm_idx_)].t_tma_b_l2_start == 0ull) _sl->sm[(sm_idx_)].t_tma_b_l2_start = _ts; } \
            } else if ((role_id_) == 2 && (((fused_tid_) - 64u) & 127u) == 0u) { \
                const uint32_t _wg = ((fused_tid_) - 64u) >> 7; \
                if (_wg < 4u) { \
                    const uint64_t _ts = agy_prof_globaltimer_ns(); \
                    if (is_l1_) { if (_sl->sm[(sm_idx_)].t_wg_l1_start[_wg] == 0ull) _sl->sm[(sm_idx_)].t_wg_l1_start[_wg] = _ts; } \
                    else        { if (_sl->sm[(sm_idx_)].t_wg_l2_start[_wg] == 0ull) _sl->sm[(sm_idx_)].t_wg_l2_start[_wg] = _ts; } \
                } \
            } \
        } \
    } while (0)

#define AGY_PROF_INTERLEAVE_TILE_END(MegaMoEPhaseT, sm_idx_, role_id_, fused_tid_, is_l1_) \
    AGY_PROF_ROLE_PHASE_END(MegaMoEPhaseT, sm_idx_, role_id_, fused_tid_, is_l1_)

#define AGY_PROF_DISP_SCAT_BEGIN(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_, num_disp_warps_, m_idx_, nb_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            const uint32_t _aux = (1u << 31) | (((m_idx_) & 0xfffu) << 12) | ((nb_) & 0xfffu); \
            if ((warp_idx_) == 0u) { \
                agy_prof_emit_now(_sl, (sm_idx_), kAgyProfTrackDispW0, kAgyEvStore, _aux); \
            } else if ((num_disp_warps_) > 1u && (warp_idx_) + 1u == (num_disp_warps_)) { \
                agy_prof_emit_now(_sl, (sm_idx_), kAgyProfTrackDispWLast, kAgyEvStore, _aux); \
            } \
        } \
    } while (0)

#define AGY_PROF_DISP_SCAT_END(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_, num_disp_warps_, m_idx_, nb_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            const uint32_t _aux = (1u << 31) | (((m_idx_) & 0xfffu) << 12) | ((nb_) & 0xfffu); \
            if ((warp_idx_) == 0u) { \
                agy_prof_emit_now(_sl, (sm_idx_), kAgyProfTrackDispW0, kAgyEvStore | kAgyFiTagEnd, _aux); \
            } else if ((num_disp_warps_) > 1u && (warp_idx_) + 1u == (num_disp_warps_)) { \
                agy_prof_emit_now(_sl, (sm_idx_), kAgyProfTrackDispWLast, kAgyEvStore | kAgyFiTagEnd, _aux); \
            } \
        } \
    } while (0)

#define AGY_PROF_DISP_COMB_BEGIN(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_, num_disp_warps_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((warp_idx_) == 0u) { \
                agy_prof_emit_now(_sl, (sm_idx_), kAgyProfTrackDispW0, kAgyEvReduction, 0u); \
            } else if ((num_disp_warps_) > 1u && (warp_idx_) + 1u == (num_disp_warps_)) { \
                agy_prof_emit_now(_sl, (sm_idx_), kAgyProfTrackDispWLast, kAgyEvReduction, 0u); \
            } \
        } \
    } while (0)

#define AGY_PROF_DISP_COMB_END(MegaMoEPhaseT, sm_idx_, warp_idx_, lane_idx_, num_disp_warps_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            atomicMax(reinterpret_cast<unsigned long long*>(&_sl->sm[(sm_idx_)].t_wg_combine_end[0]), \
                      static_cast<unsigned long long>(_ts)); \
            if ((warp_idx_) == 0u) { \
                agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispW0, _ts, kAgyEvReduction | kAgyFiTagEnd, 0u); \
            } else if ((num_disp_warps_) > 1u && (warp_idx_) + 1u == (num_disp_warps_)) { \
                agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackDispWLast, _ts, kAgyEvReduction | kAgyFiTagEnd, 0u); \
            } \
        } \
    } while (0)

#define AGY_PROF_ROLE_SYNC_END(MegaMoEPhaseT, sm_idx_, role_id_, fused_tid_) \
    do { \
        if (agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((role_id_) == 0 && (fused_tid_) == 0u) { \
                const uint64_t _ts = agy_prof_globaltimer_ns(); \
                _sl->sm[(sm_idx_)].t_tma_a_sync_end = _ts; \
                agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackTmaA, _ts, kAgyEvBarrier | kAgyFiTagEnd, 1u); \
            } else if ((role_id_) == 1 && (fused_tid_) == 32u) { \
                agy_prof_emit_now(_sl, (sm_idx_), kAgyProfTrackTmaB, kAgyEvBarrier | kAgyFiTagEnd, 1u); \
            } else if ((role_id_) == 2 && (((fused_tid_) - 64u) & 127u) == 0u) { \
                const uint32_t _wg = ((fused_tid_) - 64u) >> 7; \
                if (_wg < 4u) { \
                    const uint64_t _ts = agy_prof_globaltimer_ns(); \
                    _sl->sm[(sm_idx_)].t_wg_sync_end[_wg] = _ts; \
                    agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackMathWG0 + _wg, _ts, kAgyEvBarrier | kAgyFiTagEnd, 1u); \
                } \
            } \
        } \
    } while (0)

// Role 2: TMA A & TMA B open/close & per-tile hooks
#define AGY_PROF_TMA_OPEN(MegaMoEPhaseT, sm_idx_, lane_idx_, track_const_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            agy_prof_emit_now(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT), (sm_idx_), (track_const_), \
                kAgyFiTagOpen | (track_const_), agy_prof_smid()); \
        } \
    } while (0)

#define AGY_PROF_TMA_CLOSE(MegaMoEPhaseT, sm_idx_, lane_idx_, track_const_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            agy_prof_emit_now(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT), (sm_idx_), (track_const_), \
                kAgyFiTagClose, 0u); \
        } \
    } while (0)

#define AGY_PROF_TMA_A_COUNT_TILE(MegaMoEPhaseT, sm_idx_, lane_idx_, is_l1_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            _sl->sm[(sm_idx_)].tile_counts += (is_l1_) ? 1ull : (1ull << 32); \
        } \
    } while (0)

#if AGY_PROF == 1
#define AGY_PROF_TMA_TILE_LOAD_BEGIN(MegaMoEPhaseT, sm_idx_, lane_idx_, track_const_, is_l1_, exp_, mb_, nb_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            agy_prof_emit_now(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT), (sm_idx_), (track_const_), \
                kAgyEvLoad, agy_prof_pack_tile_aux((is_l1_), (exp_), (mb_), (nb_))); \
        } \
    } while (0)

#define AGY_PROF_TMA_TILE_LOAD_END(MegaMoEPhaseT, sm_idx_, lane_idx_, track_const_, is_l1_, exp_, mb_, nb_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            agy_prof_emit_now(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT), (sm_idx_), (track_const_), \
                kAgyEvLoad | kAgyFiTagEnd, agy_prof_pack_tile_aux((is_l1_), (exp_), (mb_), (nb_))); \
        } \
    } while (0)

#define AGY_PROF_TILE_MMA_BEGIN(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_, is_l1_, exp_, mb_, nb_) \
    do { \
        if ((warp_in_wg_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            agy_prof_emit_now(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT), (sm_idx_), kAgyProfTrackMathWG0 + (wg_idx_), \
                kAgyEvMma, agy_prof_pack_tile_aux((is_l1_), (exp_), (mb_), (nb_))); \
        } \
    } while (0)

#define AGY_PROF_TILE_MMA_END(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_, is_l1_, exp_, mb_, nb_) \
    do { \
        if ((warp_in_wg_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            agy_prof_emit_now(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT), (sm_idx_), kAgyProfTrackMathWG0 + (wg_idx_), \
                kAgyEvMma | kAgyFiTagEnd, agy_prof_pack_tile_aux((is_l1_), (exp_), (mb_), (nb_))); \
        } \
    } while (0)

#define AGY_PROF_TILE_STORE_BEGIN(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_, is_l1_, exp_, mb_, nb_) \
    do { \
        if ((warp_in_wg_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            agy_prof_emit_now(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT), (sm_idx_), kAgyProfTrackMathWG0 + (wg_idx_), \
                kAgyEvStore, agy_prof_pack_tile_aux((is_l1_), (exp_), (mb_), (nb_))); \
        } \
    } while (0)

#define AGY_PROF_TILE_STORE_END(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_, is_l1_, exp_, mb_, nb_) \
    do { \
        if ((warp_in_wg_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            agy_prof_emit_now(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT), (sm_idx_), kAgyProfTrackMathWG0 + (wg_idx_), \
                kAgyEvStore | kAgyFiTagEnd, agy_prof_pack_tile_aux((is_l1_), (exp_), (mb_), (nb_))); \
        } \
    } while (0)
#else
#define AGY_PROF_TMA_TILE_LOAD_BEGIN(MegaMoEPhaseT, sm_idx_, lane_idx_, track_const_, is_l1_, exp_, mb_, nb_) ((void)0)
#define AGY_PROF_TMA_TILE_LOAD_END(MegaMoEPhaseT, sm_idx_, lane_idx_, track_const_, is_l1_, exp_, mb_, nb_) ((void)0)
#define AGY_PROF_TILE_MMA_BEGIN(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_, is_l1_, exp_, mb_, nb_) ((void)0)
#define AGY_PROF_TILE_MMA_END(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_, is_l1_, exp_, mb_, nb_) ((void)0)
#define AGY_PROF_TILE_STORE_BEGIN(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_, is_l1_, exp_, mb_, nb_) ((void)0)
#define AGY_PROF_TILE_STORE_END(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_, is_l1_, exp_, mb_, nb_) ((void)0)
#endif

// Role 3: Math WG open/close & Bar2 / Combine hooks
#define AGY_PROF_MATH_OPEN(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_) \
    do { \
        if ((warp_in_wg_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint32_t _trk = kAgyProfTrackMathWG0 + (wg_idx_); \
            agy_prof_emit_now(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT), (sm_idx_), _trk, \
                kAgyFiTagOpen | _trk, agy_prof_smid()); \
        } \
    } while (0)

#define AGY_PROF_MATH_CLOSE(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_) \
    do { \
        if ((warp_in_wg_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint32_t _trk = kAgyProfTrackMathWG0 + (wg_idx_); \
            agy_prof_emit_now(AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT), (sm_idx_), _trk, \
                kAgyFiTagClose, 0u); \
        } \
    } while (0)

#define AGY_PROF_BEFORE_BAR2(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_) \
    do { \
        if ((warp_in_wg_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((wg_idx_) == 0u) _sl->sm[(sm_idx_)].t_before_nvlink_bar2 = _ts; \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackMathWG0 + (wg_idx_), _ts, kAgyEvCommunication, 4u); \
        } \
    } while (0)

#define AGY_PROF_AFTER_BAR2(MegaMoEPhaseT, sm_idx_, epi_tid_) \
    do { \
        if ((epi_tid_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            _sl->sm[(sm_idx_)].t_after_nvlink_bar2 = agy_prof_globaltimer_ns(); \
        } \
    } while (0)

#define AGY_PROF_COMBINE_BEGIN(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_) \
    do { \
        if ((warp_in_wg_) == 0u && (lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((wg_idx_) == 0u) _sl->sm[(sm_idx_)].t_combine_start = _ts; \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackMathWG0 + (wg_idx_), _ts, \
                kAgyEvCommunication | kAgyFiTagEnd, 4u); \
            agy_prof_emit_rec(_sl, (sm_idx_), kAgyProfTrackMathWG0 + (wg_idx_), _ts, \
                kAgyEvReduction, (wg_idx_)); \
        } \
    } while (0)

#define AGY_PROF_COMBINE_END_AND_CLOSE(MegaMoEPhaseT, sm_idx_, wg_idx_, warp_in_wg_, lane_idx_) \
    do { \
        if ((lane_idx_) == 0u && agy_prof_raw_ptr != nullptr && (sm_idx_) < kAgyProfMaxSMs) { \
            const uint64_t _ts = agy_prof_globaltimer_ns(); \
            auto* _sl = AGY_PROF_RESOLVE_SLOT_PTR(MegaMoEPhaseT); \
            if ((wg_idx_) < 4u) { \
                atomicMax(reinterpret_cast<unsigned long long*>(&_sl->sm[(sm_idx_)].t_wg_combine_end[(wg_idx_)]), \
                          static_cast<unsigned long long>(_ts)); \
            } \
            if ((warp_in_wg_) == 0u) { \
                const uint32_t _trk = kAgyProfTrackMathWG0 + (wg_idx_); \
                agy_prof_emit_rec(_sl, (sm_idx_), _trk, _ts, kAgyEvReduction | kAgyFiTagEnd, (wg_idx_)); \
                agy_prof_emit_rec(_sl, (sm_idx_), _trk, _ts, kAgyFiTagClose, 0u); \
            } \
        } \
    } while (0)

#else  // AGY_PROF == 0 (100% zero-cost compile-time gate)

#define AGY_PROF_KERNEL_ARG_DECL
#define AGY_PROF_CORE_ARG_DECL
#define AGY_PROF_KERNEL_ARG_PASS

#define AGY_PROF_CTA_ENTRY(...)                  ((void)0)
#define AGY_PROF_AFTER_CTA_INIT(...)             ((void)0)
#define AGY_PROF_DISP_OPEN(...)                  ((void)0)
#define AGY_PROF_SPLIT_L2_PROLOGUE_BEGIN(...)    ((void)0)
#define AGY_PROF_SPLIT_L2_PROLOGUE_END(...)      ((void)0)
#define AGY_PROF_DISP_CLEANUP_BEGIN(...)         ((void)0)
#define AGY_PROF_DISP_CLEANUP_END(...)           ((void)0)
#define AGY_PROF_DISP_BAR3_END(...)              ((void)0)
#define AGY_PROF_DISP_CLOSE(...)                 ((void)0)
#define AGY_PROF_NOTIFY_BEGIN(...)               ((void)0)
#define AGY_PROF_BEFORE_BAR1(...)                ((void)0)
#define AGY_PROF_AFTER_BAR1(...)                 ((void)0)
#define AGY_PROF_BEFORE_PULL(...)                ((void)0)
#define AGY_PROF_AFTER_PULL(...)                 ((void)0)
#define AGY_PROF_ROLE_PHASE_BEGIN(...)           ((void)0)
#define AGY_PROF_ROLE_PHASE_END(...)             ((void)0)
#define AGY_PROF_ROLE_SYNC_BEGIN(...)            ((void)0)
#define AGY_PROF_ROLE_SYNC_END(...)              ((void)0)
#define AGY_PROF_INTERLEAVE_TILE_BEGIN(...)      ((void)0)
#define AGY_PROF_INTERLEAVE_TILE_END(...)        ((void)0)
#define AGY_PROF_DISP_SCAT_BEGIN(...)            ((void)0)
#define AGY_PROF_DISP_SCAT_END(...)              ((void)0)
#define AGY_PROF_DISP_COMB_BEGIN(...)            ((void)0)
#define AGY_PROF_DISP_COMB_END(...)              ((void)0)
#define AGY_PROF_TMA_OPEN(...)                   ((void)0)
#define AGY_PROF_TMA_CLOSE(...)                  ((void)0)
#define AGY_PROF_TMA_A_COUNT_TILE(...)           ((void)0)
#define AGY_PROF_TMA_TILE_LOAD_BEGIN(...)        ((void)0)
#define AGY_PROF_TMA_TILE_LOAD_END(...)          ((void)0)
#define AGY_PROF_TILE_MMA_BEGIN(...)             ((void)0)
#define AGY_PROF_TILE_MMA_END(...)               ((void)0)
#define AGY_PROF_TILE_STORE_BEGIN(...)           ((void)0)
#define AGY_PROF_TILE_STORE_END(...)             ((void)0)
#define AGY_PROF_MATH_OPEN(...)                  ((void)0)
#define AGY_PROF_MATH_CLOSE(...)                 ((void)0)
#define AGY_PROF_BEFORE_BAR2(...)                ((void)0)
#define AGY_PROF_AFTER_BAR2(...)                 ((void)0)
#define AGY_PROF_COMBINE_BEGIN(...)              ((void)0)
#define AGY_PROF_COMBINE_END_AND_CLOSE(...)      ((void)0)

#endif  // AGY_PROF

}  // namespace deep_gemm
