#pragma once

#include <stdint.h>

// v1195 diagnostic-only per-wave barrier arrival and GEMM2 load-site trace.
// It layers on top of v1194 and is absent from the formal build.
#ifdef V1195_DEBUG_WAVE_TRACE
#ifndef V1194_DEBUG_RUNAT_TRACE
#error "V1195_DEBUG_WAVE_TRACE requires V1194_DEBUG_RUNAT_TRACE"
#endif
#include "v1194_runat_trace.hpp"

extern __device__ unsigned long long* v1195d_barrier_events;
extern __device__ unsigned long long* v1195d_site_events;
extern __device__ unsigned long long* v1195d_overflow;
extern __device__ unsigned v1195d_barrier_count[80u * 8u];

namespace v1195d {

constexpr unsigned kCtas = 80u;
constexpr unsigned kWaves = 8u;
constexpr unsigned kTileSlots = 96u;
constexpr unsigned kBarrierCap = 40u;
constexpr unsigned kBarrierWords = 2u;
constexpr unsigned kSiteCount = 12u;
constexpr unsigned kSiteWords = 2u;

enum Site : unsigned {
    kTargetBPre = 0u,
    kTargetBPost = 1u,
    kTargetALdsWritePre = 2u,
    kTargetALdsWritePost = 3u,
    kTargetAGlobalPre = 4u,
    kTargetAGlobalPost = 5u,
    kTargetMfmaPre = 6u,
    kTargetMfmaPost = 7u,
    kTargetScaleFmaPre = 8u,
    kTargetScaleFmaPost = 9u,
    kTargetScaleGlobalPre = 10u,
    kTargetScaleGlobalPost = 11u,
};

__device__ __forceinline__ unsigned long long stamp() {
    return static_cast<unsigned long long>(__builtin_amdgcn_s_memrealtime());
}

__device__ __forceinline__ bool wave_leader() {
    const unsigned long long ballot = __ballot(1);
    if(ballot == 0ull)
        return false;
    return (threadIdx.x & 63u) == static_cast<unsigned>(__builtin_ctzll(ballot));
}

__device__ __forceinline__ unsigned wave_id() {
    return static_cast<unsigned>(threadIdx.x) >> 6;
}

__device__ __forceinline__ void init(unsigned long long* barrier_events,
                                     unsigned long long* site_events,
                                     unsigned long long* overflow) {
    if(threadIdx.x != 0u)
        return;
    v1195d_barrier_events = barrier_events;
    v1195d_site_events    = site_events;
    v1195d_overflow       = overflow;
}

__device__ __forceinline__ void begin_tile(unsigned cta, unsigned ordinal) {
    if(cta >= kCtas || ordinal >= kTileSlots || !wave_leader())
        return;
    const unsigned wave = wave_id();
    v1195d_barrier_count[cta * kWaves + wave] = 0u;
    if(v1195d_barrier_events != nullptr) {
        for(unsigned call = 0u; call < kBarrierCap; ++call) {
            unsigned long long* rec = v1195d_barrier_events +
                (((static_cast<unsigned long long>(cta) * kTileSlots + ordinal) *
                  kBarrierCap + call) * kWaves + wave) * kBarrierWords;
            rec[0] = 0ull;
            rec[1] = 0ull;
        }
    }
    if(v1195d_site_events != nullptr) {
        for(unsigned site = 0u; site < kSiteCount; ++site) {
            unsigned long long* rec = v1195d_site_events +
                (((static_cast<unsigned long long>(cta) * kTileSlots + ordinal) *
                  kSiteCount + site) * kWaves + wave) * kSiteWords;
            rec[0] = 0ull;
            rec[1] = 0ull;
        }
    }
}

__device__ __forceinline__ void barrier_pre() {
    const unsigned cta = static_cast<unsigned>(blockIdx.x);
    if(cta >= kCtas || v1194d_ctx_active[cta] == 0u ||
       v1195d_barrier_events == nullptr || !wave_leader())
        return;
    const unsigned wave = wave_id();
    const unsigned call = v1195d_barrier_count[cta * kWaves + wave]++;
    if(call >= kBarrierCap) {
        if(v1195d_overflow != nullptr)
            atomicAdd(v1195d_overflow, 1ull);
        return;
    }
    const unsigned ordinal = v1194d_ctx_ordinal[cta];
    if(ordinal >= kTileSlots)
        return;
    unsigned long long* rec = v1195d_barrier_events +
        (((static_cast<unsigned long long>(cta) * kTileSlots + ordinal) *
          kBarrierCap + call) * kWaves + wave) * kBarrierWords;
    rec[0] = stamp();
    rec[1] = 0ull;
}

__device__ __forceinline__ void barrier_post() {
    const unsigned cta = static_cast<unsigned>(blockIdx.x);
    if(cta >= kCtas || v1194d_ctx_active[cta] == 0u ||
       v1195d_barrier_events == nullptr || !wave_leader())
        return;
    const unsigned wave = wave_id();
    const unsigned next = v1195d_barrier_count[cta * kWaves + wave];
    if(next == 0u)
        return;
    const unsigned call = next - 1u;
    if(call >= kBarrierCap)
        return;
    const unsigned ordinal = v1194d_ctx_ordinal[cta];
    if(ordinal >= kTileSlots)
        return;
    unsigned long long* rec = v1195d_barrier_events +
        (((static_cast<unsigned long long>(cta) * kTileSlots + ordinal) *
          kBarrierCap + call) * kWaves + wave) * kBarrierWords;
    rec[1] = stamp();
}

__device__ __forceinline__ void site_mark(unsigned site, unsigned tag) {
    const unsigned cta = static_cast<unsigned>(blockIdx.x);
    if(cta >= kCtas || site >= kSiteCount ||
       v1194d_ctx_active[cta] == 0u || v1195d_site_events == nullptr ||
       !wave_leader())
        return;
    const unsigned ordinal = v1194d_ctx_ordinal[cta];
    if(ordinal >= kTileSlots)
        return;
    const unsigned wave = wave_id();
    unsigned long long* rec = v1195d_site_events +
        (((static_cast<unsigned long long>(cta) * kTileSlots + ordinal) *
          kSiteCount + site) * kWaves + wave) * kSiteWords;
    rec[0] = stamp();
    rec[1] = static_cast<unsigned long long>(tag);
}

}  // namespace v1195d

#define V1195_BARRIER_PRE() v1195d::barrier_pre()
#define V1195_BARRIER_POST() v1195d::barrier_post()
#define V1195_SITE_MARK(site, tag) v1195d::site_mark((site), (tag))
#define V1195_TARGET_MARK(site, iter, lf) \
    do { if((lf) == 1u && (iter) == 4u) v1195d::site_mark((site), ((iter) << 8) | (lf)); } while(0)

#else

#define V1195_BARRIER_PRE() ((void)0)
#define V1195_BARRIER_POST() ((void)0)
#define V1195_SITE_MARK(site, tag) ((void)0)
#define V1195_TARGET_MARK(site, iter, lf) ((void)0)

#endif
