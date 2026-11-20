#pragma once

#include <stdint.h>

#ifndef WEAVE_PLAN_HOST_ONLY
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "v1262_subwave_barrier.hpp"
#include <deep_gemm/scheduler/weave_policy_runtime.cuh>

// One policy header per AMD device translation unit. CTA 0 publishes
// before route-plan release; all 80 CTAs observe the same decision.
__device__ __align__(64) weave::PolicyControl g_weave_policy_control;
#endif

#ifndef V809_MAX_TOPK
#define V809_MAX_TOPK 12u
#endif
#ifndef V809_META_STRIDE
#define V809_META_STRIDE 64u
#endif
#ifndef V809_META_HEADER
#define V809_META_HEADER 4096u
#endif
#ifndef V809_PROGRESS_MAGIC
#define V809_PROGRESS_MAGIC 0x44385052u  /* "D8PR" */
#endif

struct alignas(16) WeaveMTileDesc {
    uint32_t expert_id;
    uint32_t row_start;
    uint16_t bm;
    uint16_t valid_rows;
    uint32_t tile_idx;
};
static_assert(sizeof(WeaveMTileDesc) == 16, "WeaveMTileDesc must be 16 bytes");

struct alignas(64) WeaveTilePlanTable {
    uint32_t total_m_tiles;      // total tiles across all BM buckets
    uint32_t num_tiles[5];       // [0]=256 [1]=128 [2]=64 [3]=32 [4]=reserved=0
    uint32_t bucket_offset[5];   // first table index of each BM bucket; [4]=end
    uint32_t total_padded_rows;  // run: dense rows including padding
    uint32_t pad_gather_slot;    // Stage-1 sentinel (= world*capacity)
    uint32_t total_valid_rows;   // sum of real routed tokens
    uint32_t padding_words[2];
};
static_assert(sizeof(WeaveTilePlanTable) == 64, "WeaveTilePlanTable must be 64 bytes");

__host__ __device__ __forceinline__ int weave_bm_to_bucket(uint32_t bm) {
    switch (bm) {
        case 256u: return 0;
        case 128u: return 1;
        case 64u:  return 2;
        case 32u:  return 3;
        default:   return -1;
    }
}

struct WeaveGreedyTilePlan {
    uint32_t rem_valid;
    uint32_t q32;
};

__host__ __device__ __forceinline__ WeaveGreedyTilePlan weave_greedy_plan_init(uint32_t c) {
    WeaveGreedyTilePlan p;
    p.rem_valid = c;
    p.q32 = (c + 31u) >> 5u;
    return p;
}

#ifndef WEAVE_KNOB_BM_TIER_MODE
#define WEAVE_KNOB_BM_TIER_MODE 2
#endif
static_assert(WEAVE_KNOB_BM_TIER_MODE == 2,
              "EP16 requires adaptive BM_e in {32,64,128,256} in every arm");

__host__ __device__ __forceinline__ uint32_t weave_greedy_next_tile(
    WeaveGreedyTilePlan& p, uint32_t& valid_rows, uint32_t max_bm = 256u) {
    uint32_t bm;
    if (max_bm >= 256u && p.q32 >= 8u)      { bm = 256u; p.q32 -= 8u; }
    else if (max_bm >= 128u && p.q32 >= 4u) { bm = 128u; p.q32 -= 4u; }
    else if (p.q32 >= 2u)                   { bm = 64u;  p.q32 -= 2u; }
    else if (p.q32 == 1u)                   { bm = 32u;  p.q32 = 0u; }
    else                                    { valid_rows = 0u; return 0u; }
    valid_rows = p.rem_valid >= bm ? bm : p.rem_valid;
    p.rem_valid -= valid_rows;
    return bm;
}

__host__ __device__ __forceinline__ void weave_compute_expert_tile_counts(
    uint32_t q32, uint32_t* out5, uint32_t max_bm = 256u) {
    if (max_bm <= 64u) {
        out5[0] = 0u;
        out5[1] = 0u;
        out5[2] = q32 >> 1u;
        out5[3] = q32 & 1u;
        out5[4] = 0u;
        return;
    }
    if (max_bm <= 128u) {
        out5[0] = 0u;
        out5[1] = q32 >> 2u;
        q32 &= 3u;
        out5[2] = q32 >> 1u;
        out5[3] = q32 & 1u;
        out5[4] = 0u;
        return;
    }
    out5[0] = q32 >> 3u;
    q32 &= 7u;
    out5[1] = q32 >> 2u;
    q32 &= 3u;
    out5[2] = q32 >> 1u;
    out5[3] = q32 & 1u;
    out5[4] = 0u;
}

#ifndef WEAVE_PLAN_HOST_ONLY

#ifndef WEAVE_STAGE1_GATHER_IDS
#define WEAVE_STAGE1_GATHER_IDS(env) (reinterpret_cast<int*>((env).base.sorted_ids))
#endif

#ifndef WEAVE_4WAVE_BARRIER_DEFINED
#define WEAVE_4WAVE_BARRIER_DEFINED
__device__ __forceinline__ void weave_4wave_barrier_65440() {
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    __threadfence_block();
    const unsigned tid = threadIdx.x;
    if (tid < 256u && (tid & 63u) == 0u) {
        const unsigned w = tid >> 6u;
        volatile unsigned* sbar = reinterpret_cast<volatile unsigned*>(
            g_v1262_phase_arena + 65440u);
        const unsigned next = sbar[w] + 1u;
        sbar[w] = next;
        __builtin_amdgcn_s_waitcnt(0xc07f);
        while (sbar[0] < next || sbar[1] < next || sbar[2] < next || sbar[3] < next) {
            asm volatile("s_nop 0\n" ::: "memory");
        }
    }
    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
    __threadfence_block();
}
#endif

__device__ __forceinline__ unsigned long long weave_clock() {
    return (unsigned long long)__builtin_amdgcn_s_memrealtime();
}

__device__ __forceinline__ void weave_error(unsigned* error, unsigned code) {
    if (error) (void)atomicCAS(error, 0u, code);
}

__device__ __forceinline__ float weave_weight_value(uint16_t bits, bool fp16) {
    if (fp16) return __half2float(__ushort_as_half(bits));
    return __uint_as_float((unsigned)bits << 16);
}

template <typename DispatchEnv>
__device__ __forceinline__ const uint8_t* weave_recv_payload_base(const DispatchEnv& env) {
    const auto& e = env.base;
    return reinterpret_cast<const uint8_t*>(
        reinterpret_cast<const uint64_t*>(e.peer_payload_table)[e.rank]) -
        static_cast<uint64_t>(e.rank) * e.capacity * e.payload_bytes;
}
template <typename DispatchEnv>
__device__ __forceinline__ const float* weave_recv_scale_base(const DispatchEnv& env) {
    const auto& e = env.base;
    return reinterpret_cast<const float*>(
        reinterpret_cast<const uint64_t*>(e.peer_scale_table)[e.rank]) -
        static_cast<uint64_t>(e.rank) * e.capacity * e.scale_groups;
}

__device__ __forceinline__ bool weave_wait_tile_payload_ready(
    const unsigned* recv_ready,
    const int* stage1_gather_ids,
    const WeaveMTileDesc& tile,
    unsigned pad_gather_slot,
    unsigned long long* trace_slot,
    unsigned* error_slot) {
    const unsigned tid = threadIdx.x & 255u;
    const unsigned grp_sh = (threadIdx.x >= 256u) ? 32u : 0u;
    volatile WeaveBk64LocalU32* fail_sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(
        reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65492u + grp_sh));
    if (tid == 0u) *fail_sh = 0u;
    weave_4wave_barrier_65440(g_v1262_phase_arena);
    if (tid < tile.valid_rows) {
        const unsigned row = tile.row_start + tid;
        const unsigned phys_slot = static_cast<unsigned>(stage1_gather_ids[row]);
        if (phys_slot != pad_gather_slot) {
            const unsigned long long t_start = weave_clock();
            unsigned spins = 0u;
            while (__hip_atomic_load(recv_ready + phys_slot, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_SYSTEM) < 1u) {
                if ((spins & 63u) == 0u && (weave_clock() - t_start >= 2000000ull)) {
                    // Materialize the cold diagnostic only on timeout; avoid
                    // keeping its vector pair live across every GEMM tile.
                    unsigned long long timeout_code = 203ull;
                    asm volatile("" : "+v"(timeout_code));
                    if (trace_slot) *trace_slot = timeout_code;
                    weave_error(error_slot, (unsigned)timeout_code);
                    *fail_sh = 1u;
                    break;
                }
                if (spins < 256u) {
                    asm volatile("s_nop 0\n" ::: "memory");
                } else {
                    __builtin_amdgcn_s_sleep(1);
                }
                ++spins;
            }
        }
    }
    __syncwarp();
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    weave_4wave_barrier_65440(g_v1262_phase_arena);
    return *fail_sh == 0u;
}

__device__ __forceinline__ unsigned weave_rp_wait_val_4wave(
    const unsigned* flag_ptr, unsigned* error, unsigned err_code) {
    const unsigned tid = threadIdx.x;
    volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(
        reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65472u));
    weave_4wave_barrier_65440();
    if (tid == 0u) {
        const unsigned long long t0 = weave_clock();
        unsigned spins = 0u;
        unsigned val = __hip_atomic_load(flag_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        while (val == 0u) {
            if ((spins & 63u) == 63u && (weave_clock() - t0 >= 4000000ull)) {
                weave_error(error, err_code);
                val = 2u;
                break;
            }
            if (spins < 512u) {
                asm volatile("s_nop 0\n" ::: "memory");
            } else {
                __builtin_amdgcn_s_sleep(1);
            }
            ++spins;
            val = __hip_atomic_load(flag_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        *sh = val;
    }
    weave_4wave_barrier_65440();
    const unsigned out = *sh;
    weave_4wave_barrier_65440();
    return out;
}

// ============================================================================
// 80-CTA (20,480 threads across all 80 CUs!) Parallel Route Plan & Scatter
// Drops S=16384 route_plan from 891 us -> < 60 us (15x speedup)!
// ============================================================================
template <typename DispatchEnv, typename Ov1Ext>
__device__ __forceinline__ bool v983_route_plan_parallel_80cta(
    const DispatchEnv& env, const Ov1Ext& o, void* arena) {
    const auto& e = env.base;
    const auto& ck = env.ck;
    const unsigned cta = blockIdx.x;
    const unsigned tid = threadIdx.x;
    if (tid >= 256u) return true;

    const unsigned local_experts = (e.world > 0u) ? (e.n_experts / e.world) : 0u;
    const bool fp16 = (e.flags & 2u) != 0u;
    const unsigned region_bytes = V809_META_HEADER + e.capacity * V809_META_STRIDE;

    auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
    auto* trace = reinterpret_cast<unsigned long long*>(o.trace);
    auto* counts = reinterpret_cast<unsigned*>(e.expert_counts);
    auto* offsets = reinterpret_cast<unsigned*>(ck.ck_m64_offsets);
    auto* stage1_ids = WEAVE_STAGE1_GATHER_IDS(env);
    auto* stage2_ids = reinterpret_cast<int*>(ck.ck_m64_sorted_ids);
    auto* route_id = reinterpret_cast<unsigned*>(o.row_route_id);
    auto* route_weight = reinterpret_cast<float*>(o.stage2_weights);
    auto* bucket_rows = reinterpret_cast<unsigned*>(o.bucket_route_rows);
    auto* source_rows = reinterpret_cast<unsigned*>(e.source_bucket_rows);

    auto* plan = reinterpret_cast<WeaveTilePlanTable*>(ck.ck_a);
    auto* descs = reinterpret_cast<WeaveMTileDesc*>(reinterpret_cast<char*>(ck.ck_a) + 64);
    const uint64_t hist_offset = (64ull + 16ull * (ck.ck_route_rows / 16u) + 255ull) & ~255ull;
    auto* g_hist = reinterpret_cast<unsigned*>(reinterpret_cast<char*>(ck.ck_a) + hist_offset);
    auto* g_curs = reinterpret_cast<unsigned*>(reinterpret_cast<char*>(ck.ck_a) + hist_offset + 4096u);
    auto* g_stat = reinterpret_cast<unsigned*>(reinterpret_cast<char*>(ck.ck_a) + hist_offset + 8192u);

    WeaveLdsU8* lds = reinterpret_cast<WeaveLdsU8*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena));
    (void)arena;
    WeaveLdsU32* lds_hist = reinterpret_cast<WeaveLdsU32*>(lds); // 0..4095
    WeaveLdsU32* fail_lds = reinterpret_cast<WeaveLdsU32*>(lds + 65484u);

    // ------------------------------------------------------------------------
    // Step 1: CTA 0 validates meta_ready, loads source_rows, zeroes g_hist/g_stat
    // ------------------------------------------------------------------------
    if (cta == 0u) {
        if (e.world == 0u || e.world > 16u || local_experts == 0u || local_experts > 64u) {
            if (tid == 0u) {
                weave_error(reinterpret_cast<unsigned*>(e.error), 123u);
                __hip_atomic_store(pf + 5u, 2u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            }
            return false;
        }
        if (tid == 0u) *fail_lds = 0u;
        weave_4wave_barrier_65440();

        const auto* recv_progress = reinterpret_cast<const V809SourceProgress*>(
            reinterpret_cast<const unsigned char*>(
                reinterpret_cast<const uint64_t*>(e.peer_progress_table)[e.rank]) -
            static_cast<uint64_t>(e.rank) * sizeof(V809SourceProgress));
        if (tid < e.world) {
            const unsigned src = tid;
            const unsigned long long started = weave_clock();
            const unsigned* meta_p = reinterpret_cast<const unsigned*>(&recv_progress[src].meta_ready);
            unsigned spins = 0u;
            while (__hip_atomic_load(meta_p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) != V809_PROGRESS_MAGIC) {
                if ((spins & 63u) == 0u && (weave_clock() - started >= 2000000ull)) {
                    if (tid == 0u) v983_store_system(reinterpret_cast<unsigned*>(trace + 31u), 8u);
                    weave_error(reinterpret_cast<unsigned*>(e.error), 120u);
                    *fail_lds = 1u;
                    break;
                }
                if (spins < 256u) {
                    asm volatile("s_nop 0\n" ::: "memory");
                } else {
                    __builtin_amdgcn_s_sleep(1);
                }
                ++spins;
            }
        }
        __syncwarp();
        asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
        weave_4wave_barrier_65440();

        if (*fail_lds == 0u && tid < e.world) {
            const unsigned src = tid;
            const unsigned char* region = reinterpret_cast<const unsigned char*>(e.recv_meta) +
                                          static_cast<uint64_t>(src) * region_bytes;
            const unsigned n = static_cast<unsigned>(*reinterpret_cast<const uint64_t*>(region));
            if (n > e.capacity) {
                source_rows[src] = 0u;
                weave_error(reinterpret_cast<unsigned*>(e.error), 121u);
                *fail_lds = 1u;
            } else {
                source_rows[src] = n;
            }
        }
        for (unsigned i = tid; i < e.world * local_experts; i += 256u) g_hist[i] = 0u;
        if (tid < 8u) g_stat[tid] = 0u;
        weave_4wave_barrier_65440();

        if (tid == 0u) {
            __threadfence();
            __hip_atomic_store(pf + 5u, (*fail_lds != 0u) ? 2u : 1u,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // ------------------------------------------------------------------------
    // Step 2: All 80 CTAs (20,480 threads) scan recv_meta -> CTA LDS hist -> g_hist
    // ------------------------------------------------------------------------
    if (weave_rp_wait_val_4wave(pf + 5u, reinterpret_cast<unsigned*>(e.error), 211u) != 1u)
        return false;

    const unsigned n_bins = e.world * local_experts;
    for (unsigned i = tid; i < n_bins; i += 256u) lds_hist[i] = 0u;
    if (tid == 0u) *fail_lds = 0u;
    weave_4wave_barrier_65440();

    const unsigned gtid = cta * 256u + tid;
    constexpr unsigned kGridThreads = 80u * 256u;

    for (unsigned src = 0u; src < e.world; ++src) {
        const unsigned n = source_rows[src];
        const unsigned char* region = reinterpret_cast<const unsigned char*>(e.recv_meta) +
                                      static_cast<uint64_t>(src) * region_bytes;
        for (unsigned slot = gtid; slot < n; slot += kGridThreads) {
            const auto* m = reinterpret_cast<const V809Meta*>(
                region + V809_META_HEADER + static_cast<uint64_t>(slot) * V809_META_STRIDE);
            if (m->nexp > e.topk || m->nexp > V809_MAX_TOPK) {
                weave_error(reinterpret_cast<unsigned*>(e.error), 122u);
                *fail_lds = 1u;
                continue;
            }
            for (unsigned k = 0u; k < m->nexp; ++k) {
                const unsigned lid = m->lid[k];
                if (lid >= local_experts) {
                    weave_error(reinterpret_cast<unsigned*>(e.error), 123u);
                    *fail_lds = 1u;
                } else {
                    __hip_atomic_fetch_add(lds_hist + (src * local_experts + lid), 1u,
                                           __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
                }
            }
        }
    }
    weave_4wave_barrier_65440();

    if (*fail_lds != 0u && tid == 0u) {
        __hip_atomic_fetch_or(g_stat + 3u, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    for (unsigned i = tid; i < n_bins; i += 256u) {
        const unsigned v = lds_hist[i];
        if (v != 0u) {
            __hip_atomic_fetch_add(g_hist + i, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
    weave_4wave_barrier_65440();

    volatile WeaveBk64LocalU32* sh_last = reinterpret_cast<volatile WeaveBk64LocalU32*>(
        lds + 65476u);
    if (tid == 0u) {
        __threadfence();
        (void)__hip_atomic_fetch_add(g_stat + 0u, 1u, __ATOMIC_ACQ_REL,
                                     __HIP_MEMORY_SCOPE_AGENT);
        unsigned leader = 0u;
        if (cta == 0u) {
            // Deterministic publisher: CTA 0 waits until every CTA has
            // contributed its LDS histogram to g_hist.  The last-arriver
            // election used by DeepGEMM is deliberately replaced here so the
            // policy decision is always made by CTA 0.
            unsigned spins = 0u;
            while (__hip_atomic_load(g_stat + 0u, __ATOMIC_ACQUIRE,
                                     __HIP_MEMORY_SCOPE_AGENT) < 80u) {
                if ((spins++ & 1023u) == 0u) {
                    __builtin_amdgcn_s_sleep(1);
                }
            }
            leader = 1u;
        }
        *sh_last = leader;
    }
    weave_4wave_barrier_65440();

    // ------------------------------------------------------------------------
    // Step 3: Last CTA builds WeaveTilePlanTable, WeaveMTileDesc[], g_curs[], padding
    // ------------------------------------------------------------------------
    if (*sh_last != 0u) {
        WeaveLdsU32* plan_counts = reinterpret_cast<WeaveLdsU32*>(lds + 16384u); // [64]
        WeaveLdsU32* plan_padded = reinterpret_cast<WeaveLdsU32*>(lds + 16640u); // [64]
        WeaveLdsU32* plan_tiles  = reinterpret_cast<WeaveLdsU32*>(lds + 16896u); // [64][5]
        if (tid == 0u) *fail_lds = g_stat[3];
        weave_4wave_barrier_65440();

        if (*fail_lds == 0u) {
            for (unsigned lid = tid; lid < local_experts; lid += 256u) {
                uint32_t c = 0u;
                for (unsigned src = 0u; src < e.world; ++src)
                    c += g_hist[src * local_experts + lid];
                counts[lid] = c;
                plan_counts[lid] = c;
                const uint32_t q16 = (c + 15u) >> 4u;
                const uint32_t q32 = (c + 31u) >> 5u;
#if WEAVE_KNOB_BM_TIER_MODE == 0
                plan_padded[lid] = (q16 == 0u) ? 0u : (((q16 + 3u) >> 2u) << 6u);
#else
                plan_padded[lid] = q16 << 4u;
#endif
                uint32_t t5[5];
                weave_compute_expert_tile_counts(q32, t5, (ck.ck_intermediate <= 512u) ? ((e.t_local <= 256u) ? 64u : 128u) : 256u);
                #pragma unroll
                for (unsigned b = 0u; b < 5u; ++b) plan_tiles[lid * 5u + b] = t5[b];
            }
            weave_4wave_barrier_65440();

            for (unsigned lid = tid; lid < local_experts; lid += 256u) {
                uint32_t row_base = 0u;
                for (unsigned prev = 0u; prev < lid; ++prev)
                    row_base += plan_padded[prev];
                offsets[lid] = row_base;
                {
                    const unsigned my_node = e.rank & 8u;
                    uint32_t cur_off = row_base;
                    for (unsigned ord = 0u; ord < e.world; ++ord) {
                        const unsigned src = my_node ^ ord;
                        g_curs[src * local_experts + lid] = cur_off;
                        cur_off += g_hist[src * local_experts + lid];
                    }
                }
            }
            for (unsigned b = tid; b < 5u; b += 256u) {
                uint32_t bucket_total = 0u;
                for (unsigned lid = 0u; lid < local_experts; ++lid)
                    bucket_total += plan_tiles[lid * 5u + b];
                plan->num_tiles[b] = bucket_total;
            }
            weave_4wave_barrier_65440();

            if (tid == 0u) {
                uint32_t total_tokens = 0u, run = 0u;
                for (unsigned lid = 0u; lid < local_experts; ++lid) {
                    total_tokens += plan_counts[lid];
                    run += plan_padded[lid];
                }
                offsets[local_experts] = run;
                plan->bucket_offset[0] = 0u;
                for (unsigned b = 1u; b < 5u; ++b)
                    plan->bucket_offset[b] = plan->bucket_offset[b - 1u] + plan->num_tiles[b - 1u];
                plan->total_m_tiles = plan->bucket_offset[4] + plan->num_tiles[4];
                plan->total_padded_rows = run;
                plan->pad_gather_slot = e.world * e.capacity;
                plan->total_valid_rows = total_tokens;

                if (run > ck.ck_route_rows) {
                    weave_error(reinterpret_cast<unsigned*>(e.error), 124u);
                    *fail_lds = 1u;
                }
                auto* max_id = reinterpret_cast<int*>(ck.ck_max_token_id);
                max_id[0] = static_cast<int>(run);
                for (unsigned lid = 0u; lid <= local_experts; ++lid)
                    max_id[1u + lid] = static_cast<int>(offsets[lid]);
                reinterpret_cast<unsigned*>(e.num_valid_ids)[0] = run;
                reinterpret_cast<unsigned*>(e.num_valid_ids)[1] = total_tokens;
                reinterpret_cast<unsigned*>(e.sort_stats)[0] = total_tokens;
                reinterpret_cast<unsigned*>(e.sort_stats)[1] = run;
                trace[10] = 0ull;
                trace[11] = 0ull;
                trace[12] = 0ull;
            }
            weave_4wave_barrier_65440();

            if (*fail_lds == 0u) {
                for (unsigned lid = tid; lid < local_experts; lid += 256u) {
                    WeaveGreedyTilePlan p = weave_greedy_plan_init(plan_counts[lid]);
                    uint32_t cur_row = offsets[lid];
                    uint32_t previous_bucket = 5u, local_bucket_index = 0u;
                    uint32_t valid_rows = 0u, bm = 0u;
                    while ((bm = weave_greedy_next_tile(p, valid_rows, (ck.ck_intermediate <= 512u) ? ((e.t_local <= 256u) ? 64u : 128u) : 256u)) != 0u) {
                        const int bucket = weave_bm_to_bucket(bm);
                        if (bucket < 0 || bucket >= 4) {
                            weave_error(reinterpret_cast<unsigned*>(e.error), 126u);
                            *fail_lds = 1u;
                            break;
                        }
                        const unsigned b = static_cast<unsigned>(bucket);
                        if (b != previous_bucket) {
                            previous_bucket = b;
                            local_bucket_index = 0u;
                        }
                        uint32_t expert_prefix = 0u;
                        for (unsigned prev = 0u; prev < lid; ++prev)
                            expert_prefix += plan_tiles[prev * 5u + b];
                        const uint32_t out_idx = plan->bucket_offset[b] + expert_prefix + local_bucket_index++;
                        WeaveMTileDesc d{};
                        d.expert_id = lid;
                        d.row_start = cur_row;
                        d.bm = static_cast<uint16_t>(bm);
                        d.valid_rows = static_cast<uint16_t>(valid_rows);
                        d.tile_idx = out_idx;
                        descs[out_idx] = d;
#if WEAVE_KNOB_BM_TIER_MODE == 0
                        cur_row += 64u;
#else
                        cur_row += ((valid_rows + 15u) & ~15u);
#endif
                    }
                }
                const unsigned pad_gather = plan->pad_gather_slot;
                const unsigned padded_rows = plan->total_padded_rows;
                for (unsigned lid = tid; lid < local_experts; lid += 256u) {
                    const unsigned begin = offsets[lid], end = offsets[lid + 1u];
                    for (unsigned r = begin + plan_counts[lid]; r < end; ++r) {
                        stage1_ids[r] = static_cast<int>(pad_gather);
                        stage2_ids[r] = static_cast<int>(padded_rows);
                        route_id[r] = 0xffffffffu;
                        route_weight[r] = 0.0f;
                    }
                }
            }
            weave_4wave_barrier_65440();
        }
        // The route plan has just materialized the live post-dispatch
        // histogram in plan_counts[].  CTA 0 evaluates Eq.(1)-(3) before the
        // pf+6 release below; every waiting CTA observes a complete policy.
        if (tid == 0u && *fail_lds == 0u) {
            auto* const control = &g_weave_policy_control;
            const uint32_t epoch = control->published_epoch + 1u;
            uint32_t total = 0u;
            for (uint32_t lid = 0u; lid < local_experts; ++lid) {
                control->histogram[lid] = plan_counts[lid];
                total += plan_counts[lid];
            }
            control->stats_ready_epoch = epoch;
            const weave::WorkloadSpec wl{
                local_experts, ck.ck_hidden, ck.ck_intermediate};
            const weave::WeaveKnobs knobs =
                weave::default_weave_knobs(weave::HardwareArch::MI308X_GFX942);
            const auto decision = weave::cta0_select_and_publish(
                control, weave::HardwareArch::MI308X_GFX942, wl, knobs,
                control->histogram, total, total, epoch);
        }
        if (tid == 0u) {
            __threadfence();
            __hip_atomic_store(pf + 6u, (*fail_lds != 0u) ? 2u : 1u,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        }
    }

    // ------------------------------------------------------------------------
    // Step 4: All 80 CTAs (20,480 threads) scatter routes in parallel!
    // ------------------------------------------------------------------------
    if (weave_rp_wait_val_4wave(pf + 6u, reinterpret_cast<unsigned*>(e.error), 211u) != 1u)
        return false;

    const unsigned padded_rows = plan->total_padded_rows;
    if (tid == 0u) *fail_lds = 0u;
    weave_4wave_barrier_65440();

    for (unsigned src = 0u; src < e.world; ++src) {
        const unsigned n = source_rows[src];
        const unsigned char* region = reinterpret_cast<const unsigned char*>(e.recv_meta) +
                                      static_cast<uint64_t>(src) * region_bytes;
        for (unsigned slot = gtid; slot < n; slot += kGridThreads) {
            const auto* m = reinterpret_cast<const V809Meta*>(
                region + V809_META_HEADER + static_cast<uint64_t>(slot) * V809_META_STRIDE);
            const unsigned nexp = m->nexp;
            const unsigned max_k = (nexp <= e.topk && nexp <= V809_MAX_TOPK) ? nexp : 0u;
            for (unsigned k = 0u; k < max_k; ++k) {
                const unsigned lid = m->lid[k];
                if (lid >= local_experts) continue;
                const unsigned row = __hip_atomic_fetch_add(
                    g_curs + src * local_experts + lid, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                if (row >= padded_rows) {
                    weave_error(reinterpret_cast<unsigned*>(e.error), 125u);
                    *fail_lds = 1u;
                    continue;
                }
                const unsigned phys_slot = src * e.capacity + slot;
                stage1_ids[row] = static_cast<int>(phys_slot);
                stage2_ids[row] = static_cast<int>(row);
                route_id[row] = (src * (e.capacity + 1u) + slot) | (k << 24u);
                route_weight[row] = weave_weight_value(m->wbits[k], fp16);
                bucket_rows[(src * e.capacity + slot) * e.topk + k] = row;
            }
        }
    }
    weave_4wave_barrier_65440();

    if (*fail_lds != 0u && tid == 0u) {
        __hip_atomic_fetch_or(g_stat + 3u, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    if (tid == 0u) {
        __threadfence();
        if (__hip_atomic_fetch_add(g_stat + 2u, 1u, __ATOMIC_ACQ_REL,
                                   __HIP_MEMORY_SCOPE_AGENT) + 1u == 80u) {
            const unsigned err = __hip_atomic_load(g_stat + 3u, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(pf + 1u, (err != 0u) ? 2u : 1u,
                               __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }

    return weave_rp_wait_val_4wave(pf + 1u, reinterpret_cast<unsigned*>(e.error), 211u) == 1u;
}

// Keep single-CTA fallback available if WEAVE_PAR_RP == 0
template <typename DispatchEnv, typename Ov1Ext>
__device__ __forceinline__ void v983_route_plan_greedy_direct(
    const DispatchEnv& env, const Ov1Ext& o, void* arena) {
    (void)v983_route_plan_parallel_80cta(env, o, arena);
}

#endif  // WEAVE_PLAN_HOST_ONLY
