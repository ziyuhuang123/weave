#pragma once

#include <stdint.h>

// v1196 diagnostic-only DynamicM128::RunAt guard trace.  The formal build
// leaves V1196_DEBUG_GUARD_TRACE undefined and every macro below compiles to
// a no-op.  This header layers on the v1194 per-CTA tile context.
#ifdef V1196_DEBUG_GUARD_TRACE
#ifndef V1194_DEBUG_RUNAT_TRACE
#error "V1196_DEBUG_GUARD_TRACE requires V1194_DEBUG_RUNAT_TRACE"
#endif
#include "v1194_runat_trace.hpp"

extern __device__ unsigned long long* v1196d_guard_events;
extern __device__ unsigned long long* v1196d_overflow;

namespace v1196d {

constexpr unsigned kCtas = 80u;
constexpr unsigned kWaves = 8u;
constexpr unsigned kTileSlots = 96u;
constexpr unsigned kGuardSites = 12u;
constexpr unsigned kGuardWords = 10u;

// The sites bracket each control-flow decision.  A wave that appears at
// *_PRE but not *_PASS has taken the corresponding return edge; the return
// site is emitted from exactly the lanes that took that edge.
enum GuardSite : unsigned {
    kRunAtEntry = 0u,
    kExpertPre = 1u,
    kExpertReturn = 2u,
    kExpertPass = 3u,
    kTokenPre = 4u,
    kTokenReturn = 5u,
    kTokenPass = 6u,
    kScalePre = 7u,
    kScaleReturn = 8u,
    kScalePass = 9u,
    kStageWaitPre = 10u,
    kStageWaitPost = 11u,
};

enum GuardFlag : unsigned {
    kFlagTokenPosFailAny = 1u << 0,
    kFlagToken0Fail = 1u << 1,
    kFlagExpertFail = 1u << 2,
    kFlagScalePosFailAny = 1u << 3,
    kFlagPartialWaveTokenFail = 1u << 4,
    kFlagCurrentDivergedReturn = 1u << 5,
    kFlagStageWaitReturnFalse = 1u << 6,
    kFlagStageWaitPartialMask = 1u << 7,
};

__device__ __forceinline__ unsigned long long stamp() {
    return static_cast<unsigned long long>(__builtin_amdgcn_s_memrealtime());
}

__device__ __forceinline__ unsigned wave_id() {
    return static_cast<unsigned>(threadIdx.x) >> 6;
}

__device__ __forceinline__ void init(unsigned long long* guard_events,
                                     unsigned long long* overflow) {
    if(threadIdx.x != 0u)
        return;
    v1196d_guard_events = guard_events;
    v1196d_overflow = overflow;
}

__device__ __forceinline__ void begin_tile(unsigned cta, unsigned ordinal) {
    const unsigned wave = wave_id();
    if(v1196d_guard_events == nullptr || cta >= kCtas ||
       ordinal >= kTileSlots || !(threadIdx.x == 0u || (threadIdx.x & 63u) == 0u))
        return;
    for(unsigned site = 0u; site < kGuardSites; ++site) {
        unsigned long long* rec = v1196d_guard_events +
            (((static_cast<unsigned long long>(cta) * kTileSlots + ordinal) *
              kGuardSites + site) * kWaves + wave) * kGuardWords;
        for(unsigned word = 0u; word < kGuardWords; ++word)
            rec[word] = 0ull;
    }
}

__device__ __forceinline__ void probe(unsigned site,
                                      unsigned long long active_mask,
                                      unsigned long long pass_mask,
                                      int64_t max_token_id,
                                      int64_t expert_block_id,
                                      int64_t block_m_id,
                                      int64_t token0,
                                      int64_t token_scale_pos,
                                      int64_t token_pos,
                                      int64_t problem_tokens,
                                      int64_t logical_tile_n,
                                      int64_t logical_tile_m,
                                      unsigned flags) {
    const unsigned cta = static_cast<unsigned>(blockIdx.x);
    if(cta >= kCtas || site >= kGuardSites || v1196d_guard_events == nullptr)
        return;
    const unsigned long long ballot = __ballot(1);
    if(ballot == 0ull)
        return;
    const unsigned lane = static_cast<unsigned>(threadIdx.x) & 63u;
    if(lane != static_cast<unsigned>(__builtin_ctzll(ballot)))
        return;
    const unsigned ordinal = v1194d_ctx_ordinal[cta];
    if(ordinal >= kTileSlots) {
        if(v1196d_overflow != nullptr)
            atomicAdd(v1196d_overflow, 1ull);
        return;
    }
    const unsigned wave = wave_id();
    unsigned long long* rec = v1196d_guard_events +
        (((static_cast<unsigned long long>(cta) * kTileSlots + ordinal) *
          kGuardSites + site) * kWaves + wave) * kGuardWords;
    rec[0] = stamp();
    rec[1] = static_cast<unsigned long long>(site) |
             (static_cast<unsigned long long>(wave) << 8) |
             (static_cast<unsigned long long>(threadIdx.x) << 16);
    rec[2] = static_cast<unsigned>(max_token_id) |
             (static_cast<unsigned long long>(static_cast<unsigned>(problem_tokens)) << 32);
    rec[3] = static_cast<unsigned>(expert_block_id) |
             (static_cast<unsigned long long>(static_cast<unsigned>(block_m_id)) << 32);
    rec[4] = static_cast<unsigned>(token0) |
             (static_cast<unsigned long long>(static_cast<unsigned>(token_scale_pos)) << 32);
    rec[5] = static_cast<unsigned>(__builtin_amdgcn_readlane(token_pos, 0)) |
             (static_cast<unsigned long long>(
                  static_cast<unsigned>(__builtin_amdgcn_readlane(token_pos, 63))) << 32);
    rec[6] = active_mask;
    rec[7] = pass_mask;
    rec[8] = static_cast<unsigned long long>(flags);
    rec[9] = static_cast<unsigned>(logical_tile_n) |
             (static_cast<unsigned long long>(static_cast<unsigned>(logical_tile_m)) << 32);
}

}  // namespace v1196d

#define V1196_GUARD_PROBE(site, active, pass, max_token_id, expert_block_id, \
                          block_m_id, token0, token_scale_pos, token_pos, \
                          problem_tokens, tile_n, tile_m, flags) \
    v1196d::probe((site), (active), (pass), (max_token_id), \
                  (expert_block_id), (block_m_id), (token0), \
                  (token_scale_pos), (token_pos), (problem_tokens), \
                  (tile_n), (tile_m), (flags))
#define V1196_GUARD_BEGIN_TILE(cta, ordinal) v1196d::begin_tile((cta), (ordinal))
#define V1196_GUARD_INIT(events, overflow) v1196d::init((events), (overflow))

#else

#define V1196_GUARD_PROBE(site, active, pass, max_token_id, expert_block_id, \
                          block_m_id, token0, token_scale_pos, token_pos, \
                          problem_tokens, tile_n, tile_m, flags) ((void)0)
#define V1196_GUARD_BEGIN_TILE(cta, ordinal) ((void)0)
#define V1196_GUARD_INIT(events, overflow) ((void)0)

#endif
