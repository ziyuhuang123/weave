#pragma once

#include <stdint.h>

// Diagnostic-only RunAt tracing.  The formal build leaves V1194_DEBUG_RUNAT_TRACE
// undefined and all macros compile to no-ops.
#ifdef V1194_DEBUG_RUNAT_TRACE

extern __device__ unsigned long long* v1194d_trace;
extern __device__ unsigned long long* v1194d_tile_events;
extern __device__ unsigned long long* v1194d_barrier_events;
extern __device__ unsigned long long* v1194d_barrier_overflow;
extern __device__ unsigned long long* v1194d_wave_masks;
extern __device__ unsigned v1194d_ctx_tile[80];
extern __device__ unsigned v1194d_ctx_ordinal[80];
extern __device__ unsigned v1194d_ctx_active[80];
extern __device__ unsigned v1194d_barrier_seq[80];
extern __device__ unsigned v1194d_barrier_last[80];

namespace v1194d {

constexpr unsigned kCtas       = 80u;
constexpr unsigned kTileSlots  = 96u;
constexpr unsigned kPhases     = 16u;
constexpr unsigned kBarrierCap = 8192u;
constexpr unsigned kBarrierWords = 4u;

__device__ __forceinline__ unsigned long long stamp() {
    return static_cast<unsigned long long>(__builtin_amdgcn_s_memrealtime());
}

__device__ __forceinline__ void init(unsigned long long* trace,
                                     unsigned long long* tile_events,
                                     unsigned long long* barrier_events,
                                     unsigned long long* barrier_overflow,
                                     unsigned long long* wave_masks) {
    if(threadIdx.x != 0u)
        return;
    v1194d_trace            = trace;
    v1194d_tile_events      = tile_events;
    v1194d_barrier_events   = barrier_events;
    v1194d_barrier_overflow = barrier_overflow;
    v1194d_wave_masks       = wave_masks;
}

__device__ __forceinline__ void begin_tile(unsigned cta, unsigned ordinal,
                                           unsigned tile) {
    if(threadIdx.x != 0u || cta >= kCtas)
        return;
    v1194d_ctx_tile[cta]    = tile;
    v1194d_ctx_ordinal[cta] = ordinal;
    v1194d_ctx_active[cta]  = 1u;
    if(v1194d_wave_masks != nullptr)
        v1194d_wave_masks[cta] = 0ull;
}

// Preserve which waves reached each RunAt milestone.  Entry uses the leading
// lane; later marks use the lowest still-active lane so a partially returned
// wave is still represented.  Bits are grouped by phase: phase 8 is bits 0..7,
// phase 9 is bits 8..15, and so on through phase 14.
__device__ __forceinline__ void note_wave(unsigned phase) {
    if(phase < 8u || phase > 14u)
        return;
    const unsigned cta = static_cast<unsigned>(blockIdx.x);
    if(cta >= kCtas || v1194d_wave_masks == nullptr)
        return;
    const unsigned long long ballot = __ballot(1);
    if(ballot == 0ull)
        return;
    const unsigned lane = threadIdx.x & 63u;
    if(lane != static_cast<unsigned>(__builtin_ctzll(ballot)))
        return;
    const unsigned wave = threadIdx.x >> 6;
    atomicOr(v1194d_wave_masks + cta,
             1ull << ((phase - 8u) * 8u + wave));
}

__device__ __forceinline__ void end_tile(unsigned cta) {
    if(threadIdx.x == 0u && cta < kCtas)
        v1194d_ctx_active[cta] = 0u;
}

__device__ __forceinline__ void mark(unsigned phase) {
    note_wave(phase);
    if(threadIdx.x != 0u)
        return;
    const unsigned cta = static_cast<unsigned>(blockIdx.x);
    if(cta >= kCtas || v1194d_tile_events == nullptr)
        return;
    const unsigned ordinal = v1194d_ctx_ordinal[cta];
    if(ordinal >= kTileSlots || phase >= kPhases)
        return;
    unsigned long long* rec = v1194d_tile_events +
        ((static_cast<unsigned long long>(cta) * kTileSlots + ordinal) *
         kPhases + phase) * 2ull;
    rec[0] = stamp();
    rec[1] = static_cast<unsigned long long>(v1194d_ctx_tile[cta]) |
             (static_cast<unsigned long long>(ordinal) << 32);
}

__device__ __forceinline__ void barrier_pre() {
    if(threadIdx.x != 0u)
        return;
    const unsigned cta = static_cast<unsigned>(blockIdx.x);
    if(cta >= kCtas || v1194d_ctx_active[cta] == 0u ||
       v1194d_barrier_events == nullptr)
        return;
    const unsigned seq = v1194d_barrier_seq[cta]++;
    v1194d_barrier_last[cta] = seq;
    if(seq >= kBarrierCap) {
        atomicAdd(v1194d_barrier_overflow, 1ull);
        return;
    }
    unsigned long long* rec = v1194d_barrier_events +
        (static_cast<unsigned long long>(cta) * kBarrierCap + seq) * kBarrierWords;
    rec[0] = stamp();
    rec[1] = 0ull;  // pre-barrier
    rec[2] = static_cast<unsigned long long>(v1194d_ctx_tile[cta]) |
             (static_cast<unsigned long long>(v1194d_ctx_ordinal[cta]) << 32);
    rec[3] = 0ull;
}

__device__ __forceinline__ void barrier_post() {
    if(threadIdx.x != 0u)
        return;
    const unsigned cta = static_cast<unsigned>(blockIdx.x);
    if(cta >= kCtas || v1194d_ctx_active[cta] == 0u ||
       v1194d_barrier_events == nullptr)
        return;
    const unsigned seq = v1194d_barrier_last[cta];
    if(seq >= kBarrierCap)
        return;
    unsigned long long* rec = v1194d_barrier_events +
        (static_cast<unsigned long long>(cta) * kBarrierCap + seq) * kBarrierWords;
    rec[1] = 1ull;  // post-barrier
    rec[3] = stamp();
}

}  // namespace v1194d

#define V1194_RUNAT_MARK(phase) v1194d::mark(phase)
#define V1194_BARRIER_PRE()     v1194d::barrier_pre()
#define V1194_BARRIER_POST()    v1194d::barrier_post()

#else

#define V1194_RUNAT_MARK(phase) ((void)0)
#define V1194_BARRIER_PRE()     ((void)0)
#define V1194_BARRIER_POST()    ((void)0)

#endif
