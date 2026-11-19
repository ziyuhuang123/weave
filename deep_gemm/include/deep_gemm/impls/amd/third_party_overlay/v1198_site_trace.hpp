#pragma once

#include <stdint.h>

// Diagnostic-only append-only per-(CTA,wave) site trace.  Every wave owns an
// independent counter and ring, so no cross-tile ordinal is read from shared
// storage.  The logical tile is written into every record.  The formal build
// does not define V1198_DEBUG_SITE_TRACE and every macro below is a no-op.
#ifdef V1198_DEBUG_SITE_TRACE

extern __device__ unsigned long long* v1198d_events;
extern __device__ unsigned long long* v1198d_counts;
extern __device__ unsigned long long* v1198d_overflow;

namespace v1198d {

constexpr unsigned kCtas = 80u;
constexpr unsigned kWaves = 8u;
// 1024 records hold roughly four complete K=2048 M128/CK tiles per wave.
// The ring is deliberately bounded: the liveness snapshot only needs the
// operations leading to the currently blocked barrier, and a smaller tail
// keeps one EP8 snapshot within the local-disk budget.
constexpr unsigned kEventCap = 1024u;
constexpr unsigned kEventWords = 10u;
constexpr unsigned kCountWords = kCtas * kWaves;

enum Site : unsigned {
    kTileBegin = 0u,
    kStageWaitPre = 1u,
    kStageWaitPost = 2u,
    kStageAbort = 3u,
    kRunAtEntry = 4u,
    kExpertReturn = 5u,
    kTokenReturn = 6u,
    kScaleReturn = 7u,
    kScalePass = 8u,
    kPipelineEntry = 9u,
    kPipelineReturn = 10u,
    kBPre = 11u,
    kBPost = 12u,
    kBarrierInit = 13u,
    kBarrierLoopCopy = 14u,
    kBarrierLoopMfma = 15u,
    kBarrierTailCopy = 16u,
    kBarrierTailFinal = 17u,
    kBarrierGeneric = 18u,
    kALdsPre = 19u,
    kALdsPost = 20u,
    kAGlobalPre = 21u,
    kAGlobalPost = 22u,
    kMfmaPre = 23u,
    kMfmaPost = 24u,
    kScaleFmaPre = 25u,
    kScaleFmaPost = 26u,
    kScaleGlobalPre = 27u,
    kScaleGlobalPost = 28u,
    kRunAtExit = 29u,
    kOuterBarrier = 30u,
    kTileDone = 31u,
    kSiteCount = 32u,
};

enum Flag : unsigned long long {
    kFlagBarrierPost = 1ull << 0,
    kFlagStageWaitReturnFalse = 1ull << 1,
    kFlagStageWaitPartialMask = 1ull << 2,
    kFlagLaunchAborted = 1ull << 3,
    kFlagTileBegin = 1ull << 4,
    kFlagTileDone = 1ull << 5,
};

__device__ int v1198d_ctx_tile_m[kCountWords] = {};
__device__ int v1198d_ctx_tile_n[kCountWords] = {};
__device__ unsigned v1198d_ctx_iter[kCountWords] = {};
__device__ unsigned v1198d_ctx_lf[kCountWords] = {};

__device__ __forceinline__ unsigned long long stamp() {
    return static_cast<unsigned long long>(__builtin_amdgcn_s_memrealtime());
}

__device__ __forceinline__ unsigned wave_id() {
    return static_cast<unsigned>(threadIdx.x) >> 6;
}

__device__ __forceinline__ bool wave_leader() {
    const unsigned long long ballot = __ballot(1);
    if(ballot == 0ull)
        return false;
    return (static_cast<unsigned>(threadIdx.x) & 63u) ==
           static_cast<unsigned>(__builtin_ctzll(ballot));
}

// Explicit inline assembly is required here.  __builtin_amdgcn_s_getpc()
// was observed to be merged or hoisted by LLVM in the offline v1198 probe,
// which would erase the distinction between inlined barrier call sites.
__device__ __forceinline__ unsigned long long pc_here() {
    unsigned long long pc;
    asm volatile("s_getpc_b64 %0" : "=s"(pc));
    return pc;
}

__device__ __forceinline__ void init(unsigned long long* events,
                                     unsigned long long* counts,
                                     unsigned long long* overflow) {
    if(threadIdx.x != 0u)
        return;
    v1198d_events = events;
    v1198d_counts = counts;
    v1198d_overflow = overflow;
}

__device__ __forceinline__ void emit(unsigned site,
                                     unsigned long long flags,
                                     unsigned long long observed,
                                     unsigned long long target,
                                     unsigned long long active_mask,
                                     unsigned long long pass_mask,
                                     unsigned long long aux0,
                                     unsigned long long aux1) {
    const unsigned cta = static_cast<unsigned>(blockIdx.x);
    const unsigned wave = wave_id();
    if(cta >= kCtas || site >= kSiteCount || v1198d_events == nullptr ||
       v1198d_counts == nullptr || !wave_leader())
        return;
    const unsigned slot_id = cta * kWaves + wave;
    const unsigned long long seq = atomicAdd(v1198d_counts + slot_id, 1ull);
    if(seq >= 0xffffff00ull) {
        if(v1198d_overflow != nullptr)
            atomicAdd(v1198d_overflow, 1ull);
        return;
    }
    unsigned long long* rec = v1198d_events +
        (static_cast<unsigned long long>(slot_id) * kEventCap +
         (seq & (kEventCap - 1u))) * kEventWords;
    rec[0] = stamp();
    rec[1] = (seq & 0xffffffffull) |
             (static_cast<unsigned long long>(site) << 32) |
             (static_cast<unsigned long long>(wave) << 40) |
             (static_cast<unsigned long long>(threadIdx.x & 63u) << 48);
    rec[2] = static_cast<unsigned>(v1198d_ctx_tile_m[slot_id]) |
             (static_cast<unsigned long long>(
                  static_cast<unsigned>(v1198d_ctx_tile_n[slot_id])) << 32);
    rec[3] = static_cast<unsigned long long>(v1198d_ctx_iter[slot_id]) |
             (static_cast<unsigned long long>(v1198d_ctx_lf[slot_id]) << 32);
    rec[4] = observed | (target << 32);
    rec[5] = active_mask;
    rec[6] = pass_mask;
    rec[7] = flags;
    rec[8] = aux0;
    rec[9] = aux1;
}

__device__ __forceinline__ void begin_tile(int tile_m, int tile_n) {
    const unsigned cta = static_cast<unsigned>(blockIdx.x);
    const unsigned wave = wave_id();
    if(cta >= kCtas || !wave_leader())
        return;
    const unsigned slot_id = cta * kWaves + wave;
    v1198d_ctx_tile_m[slot_id] = tile_m;
    v1198d_ctx_tile_n[slot_id] = tile_n;
    v1198d_ctx_iter[slot_id] = 0xffffffffu;
    v1198d_ctx_lf[slot_id] = 0u;
    emit(kTileBegin, kFlagTileBegin, 0ull, 0ull, 0ull, 0ull, 0ull, 0ull);
}

__device__ __forceinline__ void set_iter(unsigned iter, unsigned lf) {
    const unsigned cta = static_cast<unsigned>(blockIdx.x);
    const unsigned wave = wave_id();
    if(cta >= kCtas || !wave_leader())
        return;
    const unsigned slot_id = cta * kWaves + wave;
    v1198d_ctx_iter[slot_id] = iter;
    v1198d_ctx_lf[slot_id] = lf;
}

__device__ __forceinline__ void site(unsigned site_id,
                                     unsigned long long flags,
                                     unsigned long long observed,
                                     unsigned long long target,
                                     unsigned long long active_mask,
                                     unsigned long long pass_mask,
                                     unsigned long long aux0,
                                     unsigned long long aux1) {
    emit(site_id, flags, observed, target, active_mask, pass_mask, aux0, aux1);
}

__device__ __forceinline__ void barrier_pre(unsigned site_id,
                                            unsigned long long observed,
                                            unsigned long long target) {
    // aux0 carries the pre-barrier PC; aux1 remains zero.
    emit(site_id, 0ull, observed, target, 0ull, 0ull, pc_here(), 0ull);
}

__device__ __forceinline__ void barrier_post(unsigned site_id,
                                             unsigned long long observed,
                                             unsigned long long target) {
    // aux0 remains zero; aux1 carries the post-barrier PC.
    emit(site_id, kFlagBarrierPost, observed, target, 0ull, 0ull, 0ull,
         pc_here());
}

}  // namespace v1198d

#define V1198_INIT(events, counts, overflow) \
    v1198d::init((events), (counts), (overflow))
#define V1198_BEGIN_TILE(tile_m, tile_n) \
    v1198d::begin_tile((tile_m), (tile_n))
#define V1198_SET_ITER(iter, lf) v1198d::set_iter((iter), (lf))
#define V1198_SITE(site_id, flags, observed, target, active, pass, aux0, aux1) \
    v1198d::site((site_id), (flags), (observed), (target), (active), (pass), \
                 (aux0), (aux1))
#define V1198_SITE_SIMPLE(site_id, flags, observed, target) \
    v1198d::site((site_id), (flags), (observed), (target), 0ull, 0ull, 0ull, 0ull)
#define V1198_BARRIER_PRE(site_id, observed, target) \
    v1198d::barrier_pre((site_id), (observed), (target))
#define V1198_BARRIER_POST(site_id, observed, target) \
    v1198d::barrier_post((site_id), (observed), (target))

#else

#define V1198_INIT(events, counts, overflow) ((void)0)
#define V1198_BEGIN_TILE(tile_m, tile_n) ((void)0)
#define V1198_SET_ITER(iter, lf) ((void)0)
#define V1198_SITE(site_id, flags, observed, target, active, pass, aux0, aux1) ((void)0)
#define V1198_SITE_SIMPLE(site_id, flags, observed, target) ((void)0)
#define V1198_BARRIER_PRE(site_id, observed, target) ((void)0)
#define V1198_BARRIER_POST(site_id, observed, target) ((void)0)

#endif
