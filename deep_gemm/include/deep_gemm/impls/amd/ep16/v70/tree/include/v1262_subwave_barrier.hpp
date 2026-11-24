#pragma once

// v1262: LDS barrier carriers for a 256-thread official CK body inside a
// persistent 512-thread workgroup.  The four compute waves use two disjoint
// monotonic generation banks:
//   * v1262_sbar_slots()        at 32768..32783 for CK template barriers
//   * v1262_raw_barrier_slots() at 32784..33807 for 64 x 4 epoch arrivals (fixed H4096 uses33 banks)
//   * raw skip[4] / done epoch  at 36864..36883
// Both banks are initialized once at kernel entry while all eight waves are
// still active.  Raw M16 data accesses are confined to 0..8191, so neither
// bank aliases GEMM data.

#include <hip/hip_runtime.h>
#include <stdint.h>

#include "v1198_site_trace.hpp"

#ifndef V70_PATHB
#define V70_PATHB 0
#endif
#if V70_PATHB
// Path B: two independent 256-thread CTAs share one CU.  The R2 work unit needs
// only 31 KiB, so let the host request exactly 32 KiB dynamically and obtain
// 2-CTA occupancy (2 x 32 KiB = the 64 KiB CU LDS budget).
extern __shared__ __align__(16) unsigned char g_v1262_phase_arena[];
#else
__shared__ __align__(16) unsigned char g_v1262_phase_arena[65536];
#endif

__device__ __forceinline__ unsigned* v1262_sbar_slots()
{
    return reinterpret_cast<unsigned*>(g_v1262_phase_arena + 65440u);
}

__device__ __forceinline__ unsigned* v1262_raw_barrier_slots()
{
    return reinterpret_cast<unsigned*>(g_v1262_phase_arena + 32784u);
}

// CK's stage2 template calls the four-wave barrier from headers that have no
// access to the launch envelope.  Keep the timeout/error context in CTA-local
// LDS; one thread publishes it before any pipeline phase can spin, and the
// existing entry __syncthreads() makes it visible.
__device__ __forceinline__ unsigned* v1262_wait_context_slots()
{
    return reinterpret_cast<unsigned*>(g_v1262_phase_arena + 65504u);
}

__device__ __forceinline__ void v1262_publish_wait_context(
    unsigned* error, unsigned rank, unsigned generation)
{
    const uint64_t error_bits = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(error));
    unsigned* slots = v1262_wait_context_slots();
    slots[0] = static_cast<unsigned>(error_bits);
    slots[1] = static_cast<unsigned>(error_bits >> 32);
    slots[2] = rank;
    slots[3] = generation;
    __threadfence_block();
}

__device__ __forceinline__ void v1262_reset_sbar()
{
    unsigned* raw_skip = reinterpret_cast<unsigned*>(
        g_v1262_phase_arena + 36864u);
    unsigned* raw_done = reinterpret_cast<unsigned*>(
        g_v1262_phase_arena + 36880u);
    if(threadIdx.x < 4u)
    {
        v1262_sbar_slots()[threadIdx.x] = 0u;
        raw_skip[threadIdx.x] = 0u;
    }
    if(threadIdx.x < 256u)
        v1262_raw_barrier_slots()[threadIdx.x] = 0u;
    if(threadIdx.x == 0u)
        *raw_done = 0u;
    __threadfence_block();
}

#define V1262_TIMEOUT_SPINS (1u << 24)
#define V1262_ERROR_TIMEOUT 217u
#define V1262_ERROR_RECORD_CLAIM 0x57414954u

__device__ __forceinline__ void v1262_record_timeout(
    unsigned* error, unsigned rank, unsigned generation, unsigned phase,
    unsigned expected, unsigned actual, unsigned spins)
{
    if(error == nullptr)
        return;
    const unsigned old_claim = atomicCAS(
        error + 15u, 0u, V1262_ERROR_RECORD_CLAIM);
    if(old_claim == 0u)
    {
        error[1u] = rank;
        error[2u] = phase;
        error[3u] = blockIdx.x;
        error[4u] = threadIdx.x >> 6u;
        error[5u] = expected;
        error[6u] = actual;
        error[7u] = generation;
        error[8u] = spins;
        error[9u] = blockIdx.x;
        error[10u] = threadIdx.x;
        error[11u] = expected;
        error[12u] = actual;
        error[14u] = 0x4154454du;  // ATOM: shared timeout record format.
        __threadfence_system();
        (void)atomicCAS(error, 0u, V1262_ERROR_TIMEOUT);
        __threadfence_system();
    }
}

__device__ __forceinline__ bool v1262_compute_barrier_4wave(
    unsigned* error, unsigned rank, unsigned generation, unsigned phase)
{
    const unsigned wave = threadIdx.x >> 6;
    if(wave >= 4u)
        return false;

    __builtin_amdgcn_s_waitcnt(0xc07f);
    __builtin_amdgcn_wave_barrier();
#if WEAVE_J_BARRIER_ENTRY_BLOCK
    __threadfence_block();
#else
    __threadfence();
#endif

    unsigned status = 1u;
    volatile unsigned* slots = v1262_sbar_slots();
    if((threadIdx.x & 63u) == 0u)
    {
        const unsigned target = __hip_atomic_fetch_add(
            const_cast<unsigned*>(slots + wave), 1u,
            __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP) + 1u;
        unsigned spins = 0u;
        bool complete = true;
        for(unsigned peer = 0u; peer < 4u; ++peer)
        {
            unsigned observed = slots[peer];
            asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            while(observed < target && spins < V1262_TIMEOUT_SPINS)
            {
                ++spins;
                // The first timed-out peer sets error[0] last.  Check it only
                // at a coarse cadence so the steady-state barrier stays cheap.
                if((spins & 0xffffu) == 0u && error != nullptr &&
                   __hip_atomic_load(error, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_SYSTEM) != 0u)
                {
                    complete = false;
                    break;
                }
                asm volatile("s_nop 0\n" ::: "memory");
                observed = slots[peer];
                asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
            }
            if(observed < target)
            {
                if(spins >= V1262_TIMEOUT_SPINS)
                    v1262_record_timeout(
                        error, rank, generation, phase, target, observed,
                        spins);
                complete = false;
                break;
            }
        }
        status = complete ? 1u : 0u;
    }

    status = __builtin_amdgcn_readfirstlane(status);
    __builtin_amdgcn_wave_barrier();
    __threadfence_block();
    asm volatile("" ::: "memory");
    return status != 0u;
}

__device__ __forceinline__ bool v1262_compute_barrier_4wave()
{
    const unsigned* slots = v1262_wait_context_slots();
    const uint64_t error_bits =
        static_cast<uint64_t>(slots[0]) |
        (static_cast<uint64_t>(slots[1]) << 32);
    return v1262_compute_barrier_4wave(
        reinterpret_cast<unsigned*>(static_cast<uintptr_t>(error_bits)),
        slots[2], slots[3], 40u);
}

// CK_G2_LDS_ONLY_OPTIONS_BEGIN
#ifndef WEAVE_CK_G2_LDS_ONLY
#define WEAVE_CK_G2_LDS_ONLY 0
#endif
#if WEAVE_CK_G2_LDS_ONLY != 0 && WEAVE_CK_G2_LDS_ONLY != 1
#error WEAVE_CK_G2_LDS_ONLY must be0 or1
#endif
// CK_G2_LDS_ONLY_OPTIONS_END
#if WEAVE_CK_G2_LDS_ONLY
// These offsets name the existing fields in the sole65536B phase arena.
// Explicit DS operands keep generic-address-space loads off the normal path.
__device__ __forceinline__ unsigned v1262_ck_lds_load_u32(unsigned byte_offset)
{
    unsigned value;
    asm volatile("ds_read_b32 %0, %1\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=&v"(value) : "v"(byte_offset) : "memory");
    return value;
}

__device__ __forceinline__ unsigned v1262_ck_lds_increment(unsigned byte_offset)
{
    unsigned old;
    const unsigned one = 1u;
    asm volatile("ds_add_rtn_u32 %0, %1, %2\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=&v"(old) : "v"(byte_offset), "v"(one) : "memory");
    return old + 1u;
}

// Called only at the five LDS rendezvous in the G2 v1 blockwise pipeline.
// Every lane completes its preceding LDS accesses before its wave publishes.
// VMEM prefetch reads remain in flight. No global payload is published here.
__device__ __forceinline__ bool v1262_ck_lds_barrier_4wave(
    unsigned* error, unsigned rank, unsigned generation, unsigned phase)
{
    const unsigned wave = threadIdx.x >> 6;
    if(wave >= 4u)
        return false;
    asm volatile("// WEAVE_CK_G2_LDS_ONLY_ENTER\n\t"
                 "s_waitcnt lgkmcnt(0)" ::: "memory");
    __builtin_amdgcn_wave_barrier();

    unsigned status = 1u;
    if((threadIdx.x & 63u) == 0u)
    {
        const unsigned target = v1262_ck_lds_increment(32768u + wave * 4u);
        unsigned spins = 0u;
        bool complete = true;
        for(unsigned peer = 0u; peer < 4u; ++peer)
        {
            unsigned observed = v1262_ck_lds_load_u32(32768u + peer * 4u);
            while(observed < target && spins < V1262_TIMEOUT_SPINS)
            {
                ++spins;
                // Preserve the original rare global error/timeout path.
                if((spins & 0xffffu) == 0u && error != nullptr &&
                   __hip_atomic_load(error, __ATOMIC_RELAXED,
                                     __HIP_MEMORY_SCOPE_SYSTEM) != 0u)
                {
                    complete = false;
                    break;
                }
                asm volatile("s_nop 0\n" ::: "memory");
                observed = v1262_ck_lds_load_u32(32768u + peer * 4u);
            }
            if(observed < target)
            {
                if(spins >= V1262_TIMEOUT_SPINS)
                    v1262_record_timeout(
                        error, rank, generation, phase, target, observed, spins);
                complete = false;
                break;
            }
        }
        status = complete ? 1u : 0u;
    }
    status = __builtin_amdgcn_readfirstlane(status);
    __builtin_amdgcn_wave_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)\n\t"
                 "// WEAVE_CK_G2_LDS_ONLY_EXIT" ::: "memory");
    return status != 0u;
}

__device__ __forceinline__ bool v1262_ck_lds_barrier_4wave()
{
    if(threadIdx.x >= 256u)
        return false;
    using Context = unsigned __attribute__((ext_vector_type(4)));
    Context context;
    const unsigned offset = 65504u;
    asm volatile("ds_read_b128 %0, %1\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=&v"(context) : "v"(offset) : "memory");
    const uint64_t error_bits = static_cast<uint64_t>(context[0]) |
        (static_cast<uint64_t>(context[1]) << 32);
    return v1262_ck_lds_barrier_4wave(
        reinterpret_cast<unsigned*>(static_cast<uintptr_t>(error_bits)),
        context[2], context[3], 40u);
}
#endif
