// GPU-direct transport for every cross-node peer in the NODEDUP production
// path: the rail carries direct dispatch, while return traffic is all-to-all.
// Host code only creates/connects the QPs and prepares the provider-owned rings;
// every forward-data post, doorbell, CQ poll and CQ tail update runs on device.
#pragma once

#include <hip/hip_runtime.h>
#ifndef EP16_GDA_DEVICE_ONLY
#include <infiniband/verbs.h>
extern "C" {
#include <infiniband/fic2dv.h>
}
#endif

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <vector>

#ifndef MN_GDA_MESH_WINDOW
// RACE29 TrackA: 32 in-flight descriptors per QP.  kWindow only bounds
// posted-retired, so the GPU ring (4096 SQE) is never recycled early.
#define MN_GDA_MESH_WINDOW 32
#endif
#ifndef MN_GDA_MESH_SLICE
#define MN_GDA_MESH_SLICE (128u * 1024u)
#endif
#ifndef MN_GDA_MESH_OPCODE
#define MN_GDA_MESH_OPCODE 1u
#endif
#ifndef MN_GDA_POST_NOINLINE
#define MN_GDA_POST_NOINLINE 1
#endif
#if MN_GDA_POST_NOINLINE
#define MN_GDA_POST_ATTR __attribute__((noinline))
#else
#define MN_GDA_POST_ATTR
#endif

namespace mn_gda_mesh {

using U64 = unsigned long long;
constexpr int kMaxRanks = 16;

// Provider AH layout observed by the validated manual-WRITE loopback.
// Keep this a local view so the transport header can compile and be audited
// without pulling in the production kernel's private type declarations.
struct AhView {
    uint8_t pad[0x18];
    uint32_t cid;
    uint8_t pad2[0x1c];
    uint32_t ports[4];
};

// Independent RC queues per peer.  DISPATCH and RETURN reuse the same queues
// in different megakernel phases; the all-CTA barriers between the phases
// guarantee that every WR from the previous phase has retired.
#ifndef EP16_GDA_CHANNELS
#define EP16_GDA_CHANNELS 4
#endif
enum Channel : unsigned {
    CHANNEL0 = 0, CHANNEL1 = 1, CHANNEL2 = 2, CHANNEL3 = 3,
    CHANNEL4 = 4, CHANNEL5 = 5, CHANNEL6 = 6, CHANNEL7 = 7,
};
constexpr unsigned CHANNELS = EP16_GDA_CHANNELS;
static_assert(CHANNELS == 1u || CHANNELS == 2u || CHANNELS == 4u ||
              CHANNELS == 8u, "GDA mesh channels must be 1, 2, 4, or 8");
enum TracePhase : unsigned { DISPATCH = 0, RETURN = 1, TRACE_PHASES = 2 };
constexpr unsigned kWindow = MN_GDA_MESH_WINDOW;
constexpr uint32_t kTags = 65535u;
constexpr unsigned kCopies = 2u;

// Cross-wave report stamps must share a device-wide time base.  clock64()
// counts are wave-local here and cannot be compared across posting CTAs.
// s_memrealtime() advances at 100 MHz; scale it by 18.5 so existing report
// tooling can keep interpreting these fields at 1.85 GHz-equivalent ticks.
__device__ __forceinline__ U64 trace_stamp_1p85ghz() {
    return static_cast<U64>(__builtin_amdgcn_s_memrealtime()) * 37ull / 2ull;
}
static_assert((kWindow & (kWindow - 1u)) == 0 && kWindow > 0,
              "GDA mesh window must be a power of two");
static_assert(kWindow < kTags / 2u, "GDA mesh active tags must be unambiguous");
static_assert(MN_GDA_MESH_SLICE == 128u * 1024u,
              "production GDA transport is validated only for 128 KiB split WRs");

struct alignas(64) Queue {
    volatile uint32_t* sq;
    volatile uint8_t* cq;
    volatile uint64_t* sq_db;
    volatile uint32_t* cq_db;
    uint32_t sq_count;
    uint32_t cq_count;
    uint32_t qpid;
    uint32_t lkey;
    uint32_t rkey;
    uint32_t lkey_chunks[16];
    uint32_t rkey_chunks[16];
    uint32_t cid;
    uint32_t sq_tail;
    uint32_t cq_head;
    uint32_t cq_tail;
    uint32_t cq_phase;
    uint32_t osn;
    uint32_t error;
    uint32_t reserved;
    U64 posted;
    U64 retired;
    U64 unique_cqes;
    U64 duplicate_cqes;
    uint8_t copies[kWindow];
    uint8_t pad[64 - (kWindow & 63u)];
    uint32_t sqe[32];
    uint32_t first_cqe[kWindow][16];
    uint32_t last_cqe[16];
    // Per-direction batch timestamps sampled by the posting lane.  These are
    // report-only: correctness and backpressure use posted/retired/copies.
    U64 trace_drain_start[TRACE_PHASES];
    U64 trace_first_post[TRACE_PHASES];
    U64 trace_last_post[TRACE_PHASES];
    U64 trace_flag_post[TRACE_PHASES];
    // Doorbell timestamps are sampled immediately after the SQ tail store.
    // first_doorbell..all_cqe is the exact per-peer network interval requested
    // by the peertrace diagnostics; last_post is retained for older reports.
    U64 trace_first_doorbell[TRACE_PHASES];
    U64 trace_last_doorbell[TRACE_PHASES];
    U64 trace_flag_doorbell[TRACE_PHASES];
    U64 trace_first_cqe[TRACE_PHASES];
    U64 trace_all_cqe[TRACE_PHASES];
    // Diagnostic-only cumulative clocks.  They separate SQE construction from
    // the SQ doorbell store without changing the posting order.
    U64 trace_sqe_cycles[TRACE_PHASES];
    U64 trace_doorbell_cycles[TRACE_PHASES];
    // Actual WR payload bytes posted by the GPU in each phase.  The flag is a
    // separate 8-byte control WR; begin_trace resets both counters per launch.
    U64 tx_payload_bytes[TRACE_PHASES];
    U64 tx_flag_bytes[TRACE_PHASES];
};

struct alignas(64) Peer {
    Queue q[CHANNELS];
};

struct alignas(64) Runtime {
    uint32_t magic;
    uint32_t enabled;
    int rank;
    int npr;
    int ranks;
    int channels;
    U64 arena_base;
    U64 arena_bytes;
    U64 tx_flags;
    U64 peer_arena[kMaxRanks];
    U64 failed;
    Peer peer[kMaxRanks];
};

struct PeerSpec {
    int peer;
    Channel channel;
    struct ibv_qp* qp;
    struct ibv_cq* cq;
    AhView* ah;
    uint32_t* sq_ring;
    uint8_t* cq_ring;
    void* sq_tail_reg;
    void* cq_tail_reg;
    uint32_t sq_count;
    uint32_t cq_count;
    uint32_t initial_sq_tail;
    U64 peer_arena;
    uint32_t peer_rkey;
    uint32_t lkey_chunks[16];
    uint32_t rkey_chunks[16];
};

struct HostResources {
    Runtime* runtime = nullptr;
    int rank = -1;
    int gpu = -1;
};

// Each owner CTA calls this once at the start of a DISPATCH/RETURN phase.
// Without the reset, first_post/first_cqe/all_cqe retain the first warmup
// launch while drain_start/flag_post move with later graph replays, producing
// a nonsensical mixed-generation cycle interval.
__device__ __forceinline__ void begin_trace(
        const void* opaque_rt, int d, Channel channel, TracePhase phase) {
    if (!opaque_rt || d < 0 || d >= kMaxRanks || channel >= CHANNELS ||
        phase >= TRACE_PHASES)
        return;
    Runtime& rt = *const_cast<Runtime*>(
        reinterpret_cast<const Runtime*>(opaque_rt));
    Queue& q = rt.peer[d].q[channel];
    q.trace_drain_start[phase] = 0ull;
    q.trace_first_post[phase] = 0ull;
    q.trace_last_post[phase] = 0ull;
    q.trace_flag_post[phase] = 0ull;
    q.trace_first_doorbell[phase] = 0ull;
    q.trace_last_doorbell[phase] = 0ull;
    q.trace_flag_doorbell[phase] = 0ull;
    q.trace_first_cqe[phase] = 0ull;
    q.trace_all_cqe[phase] = 0ull;
    q.trace_sqe_cycles[phase] = 0ull;
    q.trace_doorbell_cycles[phase] = 0ull;
    q.tx_payload_bytes[phase] = 0ull;
    q.tx_flag_bytes[phase] = 0ull;
}

__host__ __device__ inline uint32_t tag_for(U64 sequence) {
    return static_cast<uint32_t>((sequence - 1u) % kTags) + 1u;
}

__host__ __device__ inline U64 sequence_for(const Queue& q, uint32_t id) {
    if (!q.posted || !id || id > kTags) return 0;
    const uint32_t back = (tag_for(q.posted) + kTags - id) % kTags;
    return back < q.posted ? q.posted - back : 0;
}

__host__ __device__ inline bool can_post(const Queue& q) {
    return q.posted != UINT64_MAX && q.posted - q.retired < kWindow;
}

__host__ __device__ inline U64 allocate_sequence(Queue& q) {
    if (!can_post(q)) return 0;
    const U64 sequence = ++q.posted;
    q.copies[(sequence - 1u) & (kWindow - 1u)] = 0;
    return sequence;
}

__host__ __device__ inline int complete_tag(Queue& q, uint32_t id) {
    const U64 sequence = sequence_for(q, id);
    if (!sequence || sequence <= q.retired || q.posted - sequence >= kWindow) return -1;
    const unsigned index = static_cast<unsigned>((sequence - 1u) & (kWindow - 1u));
    if (q.copies[index] >= kCopies) return -2;
    const bool first = q.copies[index]++ == 0u;
    while (q.retired < q.posted && q.copies[q.retired & (kWindow - 1u)] == kCopies)
        ++q.retired;
    return first ? 1 : 0;
}

__device__ __forceinline__ uint32_t cqe_word(const volatile uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

__device__ __forceinline__ void publish_single_lane() {
    __threadfence_system();
    __builtin_amdgcn_s_waitcnt(0);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx942__)
    asm volatile("buffer_wbl2 sc0 sc1" ::: "memory");
    __builtin_amdgcn_s_waitcnt(0);
#endif
}

__device__ __forceinline__ void set_failure(
        Runtime& rt, Queue& q, int* d_err, int code) {
    q.error = static_cast<uint32_t>(-code);
    rt.failed = 1;
    if (d_err) atomicCAS(d_err, 0, code);
}

// Consume at most one physical 64-byte CQ record. fic2 emits two identical
// records per logical WR in the validated provider configuration. Return 1
// when a record was consumed, 0 when the current slot is not yet valid, and
// -1 on a transport/protocol failure.
__device__ __forceinline__ int poll_cq_once(
        Runtime& rt, Queue& q, int* d_err) {
    const volatile uint8_t* c = q.cq + static_cast<size_t>(q.cq_head) * 64u;
    if ((c[63] & 1u) == q.cq_phase) return 0;
    __threadfence_system();
    uint32_t words[16];
    #pragma clang loop unroll(disable)
    for (int i = 0; i < 16; ++i) words[i] = cqe_word(c + i * 4);
    const uint32_t status = words[0] & 255u;
    const uint32_t qpid = words[0] >> 8;
    const uint32_t id = words[1] & 65535u;
    const uint32_t opcode = (words[1] >> 16) & 255u;
    if (status || qpid != q.qpid || opcode != MN_GDA_MESH_OPCODE) {
        #pragma clang loop unroll(disable)
        for (int i = 0; i < 16; ++i) q.last_cqe[i] = words[i];
        set_failure(rt, q, d_err, -7101);
        return -1;
    }
    const U64 sequence = sequence_for(q, id);
    if (!sequence || sequence <= q.retired || q.posted - sequence >= kWindow) {
        #pragma clang loop unroll(disable)
        for (int i = 0; i < 16; ++i) q.last_cqe[i] = words[i];
        set_failure(rt, q, d_err, -7102);
        return -1;
    }
    const unsigned index = static_cast<unsigned>((sequence - 1u) & (kWindow - 1u));
    #pragma clang loop unroll(disable)
    for (int i = 0; i < 16; ++i) {
        const uint32_t word = words[i] & (i == 15 ? ~(1u << 24) : ~0u);
        if (!q.copies[index]) q.first_cqe[index][i] = word;
        else if (q.first_cqe[index][i] != word) {
            #pragma clang loop unroll(disable)
            for (int j = 0; j < 16; ++j) q.last_cqe[j] = words[j];
            set_failure(rt, q, d_err, -7103);
            return -1;
        }
    }
    const int accepted = complete_tag(q, id);
    if (accepted < 0) {
        set_failure(rt, q, d_err, -7102);
        return -1;
    }
    if (accepted) ++q.unique_cqes;
    else ++q.duplicate_cqes;
    q.cq_head = (q.cq_head + 1u) & (q.cq_count - 1u);
    q.cq_tail = (q.cq_tail + 1u) & (q.cq_count - 1u);
    if (!q.cq_head) q.cq_phase ^= 1u;
    __threadfence_system();
    *q.cq_db = q.cq_tail;
    __builtin_amdgcn_s_waitcnt(0);
    return 1;
}

__device__ __forceinline__ bool drain(
        Runtime& rt, int d, Channel channel, int* d_err) {
    Queue& q = rt.peer[d].q[channel];
    const long long start = clock64();
    while (q.retired != q.posted) {
        const int rc = poll_cq_once(rt, q, d_err);
        if (rc < 0) return false;
        if (!rc) {
            if (clock64() - start > 120000000000LL) {
                set_failure(rt, q, d_err, -7104);
                return false;
            }
            __builtin_amdgcn_s_sleep(6);
        }
    }
    return true;
}

__device__ __forceinline__ void wave_join() {
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_wave_barrier();
}

// 128-byte FIC2 RC RDMA_WRITE descriptor.  The 2026-11-09 2x1 probe captured
// a successful CPU SQE, replaced only GPU-owned fields, and proved the exact
// GPU WQE with zero-byte expected/actual diff, a successful CQE and 64 KiB
// full-byte verification in both directions.  For a non-inline one-SGE write:
// word0 byte1 is the SGE count (1), word1 has ctrl=0 followed by the 24-bit
// local cid, word3 carries OSN in bits 31:16, and word4 bit24 is SE.  Word 0
// bits 31:16 carry our completion request tag in the provider request-id field.
__device__ __forceinline__ uint32_t sqe_word(
        const Queue& q, uint32_t word, uint32_t id, uint32_t osn,
        U64 src, U64 dst, uint32_t bytes, uint32_t slot) {
    switch (word) {
    case 0:  return (MN_GDA_MESH_OPCODE & 0xffu) | (1u << 8u) |
                    ((id & 0xffffu) << 16u);
    case 1:  return (q.cid & 0x00ffffffu) << 8u;
    case 2:  return static_cast<uint32_t>(dst);
    case 3:  return static_cast<uint32_t>((dst >> 32) & 0xffffu) |
                    ((osn & 0xffffu) << 16u);
    case 4:  return (q.rkey & 0x00ffffffu) | (1u << 24u);
    case 7:  return (slot & 0xffffu) << 16;   // debug_order_id is the SQ slot
    case 8:  return static_cast<uint32_t>(src);
    case 9:  return static_cast<uint32_t>((src >> 32) & 0xffffu) |
                    ((q.lkey & 0xffffu) << 16);
    case 10: return (bytes << 1u) | ((q.lkey >> 16) & 1u);
    default: return 0u;
    }
}

__device__ __forceinline__ void ring_sq_doorbell(Queue& q) {
    *q.sq_db = (static_cast<uint64_t>(q.sq_tail) << 32u) |
               static_cast<uint64_t>(q.qpid & 0x00ffffffu);
}

// A complete 64-lane wave participates in every call. Lane 0 exclusively owns
// all mutable SQ/CQ/credit state; lanes 0..31 each publish four aligned dwords
// of the 128-byte SQE. The wave joins before the owner rings the SQ doorbell.
__device__ __forceinline__ bool post_one(
        Runtime& rt, int d, Channel channel, U64 src, U64 dst,
        uint32_t bytes, bool is_flag, U64 value, int* d_err) {
    Queue& q = rt.peer[d].q[channel];
    const int lane = threadIdx.x & 63;
    bool ok = true;
    uint32_t slot = 0, id = 0, osn = 0;
    if (lane == 0) {
        if (!bytes || bytes > MN_GDA_MESH_SLICE) {
            set_failure(rt, q, d_err, -7105);
            ok = false;
        }
        if (ok && is_flag) {
            if (!drain(rt, d, channel, d_err)) {
                ok = false;
            } else {
                const unsigned source_index = static_cast<unsigned>(q.posted & (kWindow - 1u));
                U64* cell = reinterpret_cast<U64*>(rt.tx_flags +
                    static_cast<size_t>((static_cast<unsigned>(d) * CHANNELS + channel) * kWindow + source_index) * 8u);
                *cell = value;
                src = reinterpret_cast<U64>(cell);
                bytes = 8u;
            }
        }
        if (ok) {
            const U64 begin = rt.arena_base;
            const U64 end = begin + rt.arena_bytes;
            const U64 rbegin = rt.peer_arena[d];
            const U64 rend = rbegin + rt.arena_bytes;
            if (src < begin || src > end - bytes || dst < rbegin || dst > rend - bytes ||
                (src >> 48) || (dst >> 48)) {
                set_failure(rt, q, d_err, -7106);
                ok = false;
            }
        }
        if (ok) {
            const long long start = clock64();
            while (!can_post(q)) {
                const int rc = poll_cq_once(rt, q, d_err);
                if (rc < 0) {
                    ok = false;
                    break;
                }
                if (!rc) {
                    if (clock64() - start > 120000000000LL) {
                        set_failure(rt, q, d_err, -7107);
                        ok = false;
                        break;
                    }
                    __builtin_amdgcn_s_sleep(6);
                }
            }
        }
        if (ok) {
            const U64 sequence = allocate_sequence(q);
            if (!sequence) {
                set_failure(rt, q, d_err, -7108);
                ok = false;
            } else {
                id = tag_for(sequence);
                slot = q.sq_tail;
                osn = q.osn;
                q.osn = (q.osn + 1u) & 0x7fffu;
            }
        }
    }
    ok = __builtin_amdgcn_readfirstlane(ok ? 1 : 0) != 0;
    if (!ok) return false;
    // Lane 0 may redirect a flag request to its private tx_flags cell and
    // shorten it to 8 bytes.  Publish those owner-local values before the
    // other 31 lanes encode their SQE words.
    src = static_cast<U64>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(src))) |
          (static_cast<U64>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(src >> 32))) << 32);
    dst = static_cast<U64>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(dst))) |
          (static_cast<U64>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(dst >> 32))) << 32);
    bytes = static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(bytes));
    slot = static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(slot));
    id = static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(id));
    osn = static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(osn));

    if (lane < 32u) {
        volatile uint32_t* sqe = q.sq + static_cast<size_t>(slot) * 32u + static_cast<size_t>(lane) * 4u;
        #pragma clang loop unroll(disable)
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t word = lane * 4u + i;
            sqe[i] = sqe_word(q, word, id, osn, src, dst, bytes, slot);
        }
    }
    publish_single_lane();   // every participating lane publishes its stores
    wave_join();
    if (lane == 0) {
        q.sq_tail = (slot + 1u) & (q.sq_count - 1u);
        ring_sq_doorbell(q);
    }
    wave_join();
    return true;
}

__device__ __forceinline__ bool peer_present(const Runtime* rt, int d, int ranks) {
    if (!rt || !rt->enabled || d < 0 || d >= ranks || d >= kMaxRanks) return false;
    for (unsigned c = 0; c < CHANNELS; ++c)
        if (rt->peer[d].q[c].sq == nullptr) return false;
    return true;
}

// Called only by the single owner lane for this peer/channel.
__device__ MN_GDA_POST_ATTR bool post(
        const void* opaque_rt, int d, Channel channel, U64 src, U64 dst,
        uint32_t bytes, bool is_flag, U64 value, int* d_err) {
    // Cooperative SQE construction requires every lane of one gfx942 wave.
    // Reject a partial-wave call before any queue state or SQE bytes change.
    if (__ballot(1) != ~0ULL) {
        if (d_err) atomicExch(d_err, -7100);
        return false;
    }
    Runtime& rt = *const_cast<Runtime*>(reinterpret_cast<const Runtime*>(opaque_rt));
    if (!peer_present(&rt, d, rt.ranks) || channel >= CHANNELS) return false;
    if (is_flag) return post_one(rt, d, channel, 0, dst, 8, true, value, d_err);
    for (uint32_t off = 0; off < bytes; off += MN_GDA_MESH_SLICE) {
        const uint32_t len = min(MN_GDA_MESH_SLICE, bytes - off);
        if (!post_one(rt, d, channel, src + off, dst + off, len, false, 0, d_err))
            return false;
    }
    return true;
}


// ---------------------------------------------------------------------------
// EP16 megakernel scalar posting path.
//
// The generic post() above assumes a full 64-lane wave because post_one()
// distributes the 32 SQE dwords over lanes 0..31.  The EP16 kernel calls the
// transport from only the first CTA of each peer/channel while the carrier's
// 80-CTA grid is resident.  Marking that wave-wide helper noinline forced the
// carrier kernel to reserve a scratch frame and spilled four uniform values.
//
// These entry points perform the same queue protocol from lane 0.  The 128-byte
// SQE is written serially by that lane; EP16's 32 posting CTAs are not on
// the GEMM critical path.  Keeping the routine force-inlined and scalar makes
// the complete kernel compile with zero SGPR spill and zero scratch, matching
// the amdser2 carrier resource envelope.  The data and flag WRs remain on the
// same QP, so the trailing flag is ordered after the payload.
__device__ __forceinline__ int poll_cq_scalar(
        Runtime& rt, Queue& q, int* d_err) {
    const volatile uint8_t* c = q.cq + static_cast<size_t>(q.cq_head) * 64u;
    if ((c[63] & 1u) == q.cq_phase) return 0;
    __threadfence_system();
    const uint32_t w0 = cqe_word(c);
    const uint32_t w1 = cqe_word(c + 4u);
    const uint32_t status = w0 & 255u;
    const uint32_t qpid = w0 >> 8;
    const uint32_t id = w1 & 65535u;
    const uint32_t opcode = (w1 >> 16) & 255u;
    if (status || qpid != q.qpid || opcode != MN_GDA_MESH_OPCODE) {
        #pragma clang loop unroll(disable)
        for (int i = 0; i < 16; ++i) q.last_cqe[i] = cqe_word(c + i * 4);
        set_failure(rt, q, d_err, -7101);
        return -1;
    }
    const U64 sequence = sequence_for(q, id);
    if (!sequence || sequence <= q.retired || q.posted - sequence >= kWindow) {
        #pragma clang loop unroll(disable)
        for (int i = 0; i < 16; ++i) q.last_cqe[i] = cqe_word(c + i * 4);
        set_failure(rt, q, d_err, -7102);
        return -1;
    }
    const unsigned index = static_cast<unsigned>((sequence - 1u) & (kWindow - 1u));
    bool mismatch = false;
    if (!q.copies[index]) {
        #pragma clang loop unroll(disable)
        for (int i = 0; i < 16; ++i) {
            const uint32_t word = cqe_word(c + i * 4) &
                                  (i == 15 ? ~(1u << 24) : ~0u);
            q.first_cqe[index][i] = word;
        }
    } else {
        #pragma clang loop unroll(disable)
        for (int i = 0; i < 16; ++i) {
            const uint32_t word = cqe_word(c + i * 4) &
                                  (i == 15 ? ~(1u << 24) : ~0u);
            mismatch |= q.first_cqe[index][i] != word;
        }
        if (mismatch) {
            #pragma clang loop unroll(disable)
            for (int i = 0; i < 16; ++i) q.last_cqe[i] = cqe_word(c + i * 4);
            set_failure(rt, q, d_err, -7103);
            return -1;
        }
    }
    const int accepted = complete_tag(q, id);
    if (accepted < 0) {
        set_failure(rt, q, d_err, -7102);
        return -1;
    }
    if (accepted) ++q.unique_cqes;
    else ++q.duplicate_cqes;
    q.cq_head = (q.cq_head + 1u) & (q.cq_count - 1u);
    q.cq_tail = (q.cq_tail + 1u) & (q.cq_count - 1u);
    if (!q.cq_head) q.cq_phase ^= 1u;
    __threadfence_system();
    *q.cq_db = q.cq_tail;
    __builtin_amdgcn_s_waitcnt(0);
    return 1;
}

__device__ __forceinline__ bool drain_scalar(
        Runtime& rt, int d, Channel channel, int* d_err,
        TracePhase phase = DISPATCH) {
    Queue& q = rt.peer[d].q[channel];
    const long long start = clock64();
    q.trace_drain_start[phase] = trace_stamp_1p85ghz();
    while (q.retired != q.posted) {
        const int rc = poll_cq_scalar(rt, q, d_err);
        if (rc < 0) return false;
        if (rc > 0 && !q.trace_first_cqe[phase])
            q.trace_first_cqe[phase] = trace_stamp_1p85ghz();
        if (!rc) {
            if (clock64() - start > 120000000000LL) {
                set_failure(rt, q, d_err, -7104);
                return false;
            }
            __builtin_amdgcn_s_sleep(6);
        }
    }
    q.trace_all_cqe[phase] = trace_stamp_1p85ghz();
    return true;
}

__device__ __forceinline__ void publish_sq_scalar(
        Queue& q, TracePhase phase = DISPATCH) {
    const U64 publish_begin = trace_stamp_1p85ghz();
    publish_single_lane();
    const U64 doorbell_begin = trace_stamp_1p85ghz();
    ring_sq_doorbell(q);
    const U64 doorbell_end = trace_stamp_1p85ghz();
    q.trace_doorbell_cycles[phase] +=
        (doorbell_begin - publish_begin) + (doorbell_end - doorbell_begin);
    if (!q.trace_first_doorbell[phase])
        q.trace_first_doorbell[phase] = doorbell_end;
    q.trace_last_doorbell[phase] = doorbell_end;
    q.trace_last_post[phase] = doorbell_end;
}

template <bool Publish = true>
__device__ __forceinline__ bool post_one_data_scalar(
        Runtime& rt, int d, Channel channel, U64 src, U64 dst,
        uint32_t bytes, int* d_err, TracePhase phase = DISPATCH,
        bool is_flag = false) {
    Queue& q = rt.peer[d].q[channel];
    if (!bytes || bytes > MN_GDA_MESH_SLICE) {
        set_failure(rt, q, d_err, -7105);
        return false;
    }
    const U64 begin = rt.arena_base;
    const U64 end = begin + rt.arena_bytes;
    const U64 rbegin = rt.peer_arena[d];
    const U64 rend = rbegin + rt.arena_bytes;
    if (src < begin || src > end - bytes || dst < rbegin || dst > rend - bytes ||
        (src >> 48) || (dst >> 48)) {
        set_failure(rt, q, d_err, -7106);
        return false;
    }
    const long long start = clock64();
    while (!can_post(q)) {
        const int rc = poll_cq_scalar(rt, q, d_err);
        if (rc < 0) return false;
        if (!rc) {
            if (clock64() - start > 120000000000LL) {
                set_failure(rt, q, d_err, -7107);
                return false;
            }
            __builtin_amdgcn_s_sleep(6);
        }
    }
    if (!q.trace_first_post[phase])
        q.trace_first_post[phase] = trace_stamp_1p85ghz();
    const U64 sequence = allocate_sequence(q);
    if (!sequence) {
        set_failure(rt, q, d_err, -7108);
        return false;
    }
    if (is_flag) q.tx_flag_bytes[phase] += bytes;
    else q.tx_payload_bytes[phase] += bytes;
    const uint32_t id = tag_for(sequence);
    const uint32_t slot = q.sq_tail;
    const uint32_t osn = q.osn;
    q.osn = (q.osn + 1u) & 0x7fffu;
    volatile uint32_t* sqe = q.sq + static_cast<size_t>(slot) * 32u;
    // Report both CPU-side-equivalent spans on the same 1.85 GHz
    // s_memrealtime axis as every other trace stamp.  clock64() is
    // wave-local and must never be mixed into the cross-CTA timeline.
    const U64 sqe_begin = trace_stamp_1p85ghz();
    #pragma clang loop unroll(disable)
    for (uint32_t word = 0; word < 32u; ++word)
        sqe[word] = sqe_word(q, word, id, osn, src, dst, bytes, slot);
    if constexpr (Publish) {
        publish_single_lane();
        const U64 sqe_end = trace_stamp_1p85ghz();
        q.trace_sqe_cycles[phase] += sqe_end - sqe_begin;
        q.sq_tail = (slot + 1u) & (q.sq_count - 1u);
        const U64 doorbell_begin = trace_stamp_1p85ghz();
        ring_sq_doorbell(q);
        const U64 doorbell_end = trace_stamp_1p85ghz();
        q.trace_doorbell_cycles[phase] += doorbell_end - doorbell_begin;
        const U64 doorbell_done = trace_stamp_1p85ghz();
        if (!q.trace_first_doorbell[phase])
            q.trace_first_doorbell[phase] = doorbell_done;
        q.trace_last_doorbell[phase] = doorbell_done;
        q.trace_last_post[phase] = doorbell_done;
    } else {
        const U64 sqe_end = trace_stamp_1p85ghz();
        q.trace_sqe_cycles[phase] += sqe_end - sqe_begin;
        q.sq_tail = (slot + 1u) & (q.sq_count - 1u);
    }
    return true;
}

__device__ __forceinline__ bool post_data_scalar(
        Runtime& rt, int d, Channel channel, U64 src, U64 dst,
        uint32_t bytes, int* d_err, TracePhase phase = DISPATCH) {
    if (!peer_present(&rt, d, rt.ranks) || channel >= CHANNELS) return false;
    for (uint32_t off = 0; off < bytes; off += MN_GDA_MESH_SLICE) {
        const uint32_t len = min(MN_GDA_MESH_SLICE, bytes - off);
        if (!post_one_data_scalar<true>(rt, d, channel, src + off, dst + off,
                                        len, d_err, phase))
            return false;
    }
    return true;
}

template <bool Publish = true>
__device__ __forceinline__ bool post_flag_scalar(
        Runtime& rt, int d, Channel channel, U64 dst, U64 value, int* d_err,
        TracePhase phase = DISPATCH) {
    if (!peer_present(&rt, d, rt.ranks) || channel >= CHANNELS) return false;
    Queue& q = rt.peer[d].q[channel];
    // RC ordering makes this flag visible only after every earlier payload WR
    // on this QP has completed at the receiver.  Do not insert a host-visible
    // drain here: the caller either drained payload first (preflight) or posts
    // the flag back-to-back and drains payload+flag together (megakernel).
    const unsigned source_index = static_cast<unsigned>(q.posted & (kWindow - 1u));
    U64* cell = reinterpret_cast<U64*>(rt.tx_flags +
        static_cast<size_t>((static_cast<unsigned>(d) * CHANNELS + channel) *
                            kWindow + source_index) * 8u);
    *cell = value;
    // post_one_data_scalar advances all-doorbell timestamps.  Preserve the
    // final payload doorbell separately so the ready publication segment is
    // flag_doorbell - last_post and the post-marker CQ wait is
    // all_cqe - flag_doorbell.
    const U64 final_payload_doorbell = q.trace_last_post[phase];
    const bool ok = post_one_data_scalar<Publish>(
        rt, d, channel, reinterpret_cast<U64>(cell), dst, 8u, d_err, phase,
        true);
    if constexpr (Publish) {
        if (ok) {
            q.trace_last_post[phase] = final_payload_doorbell;
            q.trace_flag_post[phase] = q.trace_last_doorbell[phase];
            q.trace_flag_doorbell[phase] = q.trace_last_doorbell[phase];
        }
    }
    return ok;
}


// Fixed-size global-memory command block used by the EP16 CTA-owned posting
// lane.  Keeping all arguments in one volatile record lets the noinline device
// calls above compile without spilling the carrier's uniform state.  Each CTA
// has a private record in Ep16MegaFront::post_ops.
struct alignas(32) PostOp {
    U64 rt;
    U64 src;
    U64 dst;
    U64 value;
    int* d_err;
    uint32_t bytes;
    uint32_t peer;
    uint32_t channel;
    uint32_t status;
    uint32_t reserved0;
    uint32_t reserved1;
};
static_assert(sizeof(PostOp) == 64, "PostOp ABI");

__device__ __attribute__((noinline)) void post_one_op(volatile PostOp* op) {
    Runtime& rt = *reinterpret_cast<Runtime*>((void*)(uintptr_t)op->rt);
    const bool ok = post_one_data_scalar<false>(
        rt, (int)op->peer, static_cast<Channel>(op->channel),
        op->src, op->dst, op->bytes, op->d_err,
        static_cast<TracePhase>(op->reserved0));
    op->status = ok ? 1u : 0u;
}

__device__ __attribute__((noinline)) void post_publish_op(volatile PostOp* op) {
    Runtime& rt = *reinterpret_cast<Runtime*>((void*)(uintptr_t)op->rt);
    const int peer = (int)op->peer;
    const Channel channel = static_cast<Channel>(op->channel);
    Queue& q = rt.peer[peer].q[channel];
    publish_sq_scalar(q, static_cast<TracePhase>(op->reserved0));
    op->status = 1u;
}

__device__ __attribute__((noinline)) void post_flag_op(volatile PostOp* op) {
    Runtime& rt = *reinterpret_cast<Runtime*>((void*)(uintptr_t)op->rt);
    const int peer = (int)op->peer;
    const Channel channel = static_cast<Channel>(op->channel);
    const TracePhase phase = static_cast<TracePhase>(op->reserved0);
    bool ok = post_flag_scalar<false>(
        rt, peer, channel, op->dst, op->value, op->d_err, phase);
    if (ok) {
        Queue& q = rt.peer[peer].q[channel];
        publish_sq_scalar(q, phase);
        q.trace_flag_post[phase] = q.trace_last_doorbell[phase];
        q.trace_flag_doorbell[phase] = q.trace_last_doorbell[phase];
    }
    op->status = ok ? 1u : 0u;
}

__device__ __attribute__((noinline)) void post_drain_op(volatile PostOp* op) {
    Runtime& rt = *reinterpret_cast<Runtime*>((void*)(uintptr_t)op->rt);
    const bool ok = drain_scalar(
        rt, (int)op->peer, static_cast<Channel>(op->channel), op->d_err,
        static_cast<TracePhase>(op->reserved0));
    op->status = ok ? 1u : 0u;
}

__device__ __forceinline__ bool drain_all(const void* opaque_rt, int ranks, int* d_err) {
    if (!opaque_rt) return true;
    Runtime& rt = *const_cast<Runtime*>(reinterpret_cast<const Runtime*>(opaque_rt));
    for (int d = 0; d < ranks && d < kMaxRanks; ++d) {
        if (!peer_present(&rt, d, ranks)) continue;
        for (unsigned c = 0; c < CHANNELS; ++c)
            if (!drain(rt, d, static_cast<Channel>(c), d_err)) return false;
    }
    return true;
}

#ifndef EP16_GDA_DEVICE_ONLY
namespace host_detail {

struct MappingKey {
    uintptr_t page;
    bool operator<(const MappingKey& rhs) const { return page < rhs.page; }
};
struct Mapping {
    uintptr_t end;
    unsigned flags;
    unsigned references;
    std::map<int, void*> device_pages;
};

inline std::mutex& mapping_mutex() { static std::mutex m; return m; }
inline std::map<MappingKey, Mapping>& mappings() {
    static std::map<MappingKey, Mapping> m;
    return m;
}

[[noreturn]] inline void fail(const char* what, const char* detail = nullptr) {
    std::fprintf(stderr, "[gda-mesh] FATAL rank=%d gpu=%d %s%s%s\n",
                 -1, -1, what, detail ? ": " : "", detail ? detail : "");
    std::fflush(stderr);
    std::fflush(stdout);
    std::_Exit(1);
}

inline void check(hipError_t rc, const char* what) {
    if (rc != hipSuccess) {
        char detail[256];
        std::snprintf(detail, sizeof(detail), "%s (%d)", hipGetErrorString(rc), static_cast<int>(rc));
        fail(what, detail);
    }
}

inline void* map_range(void* host, size_t bytes, unsigned flags, int gpu, const char* what) {
    if (!host || !bytes) fail("empty GDA mapping range", what);
    constexpr uintptr_t kPage = 4096;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(host);
    const uintptr_t first = begin & ~(kPage - 1u);
    const uintptr_t end0 = begin + bytes;
    if (end0 < begin || end0 > UINTPTR_MAX - (kPage - 1u))
        fail("GDA mapping range overflow", what);
    const uintptr_t end = (end0 + kPage - 1u) & ~(kPage - 1u);
    std::lock_guard<std::mutex> lock(mapping_mutex());
    auto& mm = mappings();
    auto it = mm.upper_bound(MappingKey{first});
    if (it != mm.begin()) {
        auto previous = std::prev(it);
        if (previous->second.end > first) it = previous;
    }
    if (it != mm.end() && it->first.page < end) {
        if (it->second.flags != flags || first < it->first.page || end > it->second.end)
            fail("conflicting or partial GDA HIP host registration", what);
        ++it->second.references;
    } else {
        errno = 0;
        const hipError_t rc = hipHostRegister(reinterpret_cast<void*>(first), end - first, flags);
        check(rc, "hipHostRegister(GDA provider page)");
        it = mm.emplace(MappingKey{first}, Mapping{end, flags, 1u, {}}).first;
    }
    auto dit = it->second.device_pages.find(gpu);
    if (dit == it->second.device_pages.end()) {
        void* dev = nullptr;
        check(hipHostGetDevicePointer(&dev, reinterpret_cast<void*>(it->first.page), 0),
              "hipHostGetDevicePointer(GDA provider page)");
        if (!dev) fail("null GDA device mapping", what);
        dit = it->second.device_pages.emplace(gpu, dev).first;
    }
    return static_cast<uint8_t*>(dit->second) + (begin - it->first.page);
}

inline bool fits_48bit(U64 base, U64 bytes) {
    constexpr U64 kLimit = 1ull << 48;
    return bytes && base < kLimit && bytes <= kLimit - base;
}

} // namespace host_detail

inline HostResources* prepare_rank(
        int rank, int gpu, int npr, int ranks,
        char* arena, size_t arena_bytes, struct ibv_mr* arena_mr,
        U64 tx_flags_offset, const PeerSpec* specs, int nspec) {
    if (ranks < 2 || ranks > kMaxRanks || npr < 1 || npr > ranks || ranks % npr)
        host_detail::fail("invalid rank/NPR for GDA mesh");
    if (!arena || !arena_bytes || !arena_mr || !specs || nspec <= 0)
        host_detail::fail("missing GDA mesh host resources");
    if (!host_detail::fits_48bit(reinterpret_cast<U64>(arena), arena_bytes) ||
        arena_mr->lkey > 0x1ffffu)
        host_detail::fail("local arena or lkey exceeds fic2 SQE field width");
    auto* hr = new HostResources{};
    hr->rank = rank;
    hr->gpu = gpu;
    host_detail::check(hipMalloc(&hr->runtime, sizeof(Runtime)), "hipMalloc(GDA mesh runtime)");
    host_detail::check(hipMemset(hr->runtime, 0, sizeof(Runtime)), "hipMemset(GDA mesh runtime)");
    Runtime host{};
    host.magic = 0x4744414eu;
    host.enabled = 1u;
    host.rank = rank;
    host.npr = npr;
    host.ranks = ranks;
    host.channels = CHANNELS;
    host.arena_base = reinterpret_cast<U64>(arena);
    host.arena_bytes = arena_bytes;
    host.tx_flags = reinterpret_cast<U64>(arena + tx_flags_offset);
    for (int i = 0; i < nspec; ++i) {
        const PeerSpec& s = specs[i];
        if (s.peer < 0 || s.peer >= ranks || s.peer == rank || s.channel >= CHANNELS ||
            !s.qp || !s.cq || !s.ah || !s.sq_ring || !s.cq_ring ||
            !s.sq_tail_reg || !s.cq_tail_reg || !s.sq_count || !s.cq_count ||
            s.initial_sq_tail >= s.sq_count || !s.peer_arena ||
            s.peer_rkey > 0xffffffu || s.qp->qp_num > 0xffffffu ||
            s.ah->cid > 0xffffffu)
            host_detail::fail("invalid GDA mesh peer specification");
        if (!host_detail::fits_48bit(s.peer_arena, arena_bytes))
            host_detail::fail("peer arena exceeds fic2 48-bit address field");
        if ((s.sq_count & (s.sq_count - 1u)) ||
            (s.cq_count & (s.cq_count - 1u)) || s.sq_count <= kWindow ||
            s.sq_count > 65536u || s.cq_count <= 2u * kWindow)
            host_detail::fail("caller-owned FIC2 SQ/CQ rings are invalid");
        std::vector<uint8_t> fresh(static_cast<size_t>(s.cq_count) * 64u);
        host_detail::check(
            hipMemcpy(fresh.data(), s.cq_ring,
                      fresh.size(), hipMemcpyDeviceToHost),
            "hipMemcpy(FIC2 CQ takeover check)");
        for (unsigned slot = 0; slot < s.cq_count; ++slot)
            if (fresh[static_cast<size_t>(slot) * 64u + 63u] & 1u)
                host_detail::fail("GDA CQ is not empty at takeover");
        Queue q{};
        q.sq = s.sq_ring;
        q.cq = s.cq_ring;
        q.sq_db = static_cast<volatile uint64_t*>(host_detail::map_range(
            s.sq_tail_reg, sizeof(uint64_t),
            hipHostRegisterMapped | hipHostRegisterIoMemory, gpu, "SQ-doorbell"));
        q.cq_db = static_cast<volatile uint32_t*>(host_detail::map_range(
            s.cq_tail_reg, sizeof(uint32_t),
            hipHostRegisterMapped | hipHostRegisterIoMemory, gpu, "CQ-doorbell"));
        q.sq_count = s.sq_count;
        q.cq_count = s.cq_count;
        q.qpid = s.qp->qp_num;
        q.lkey = arena_mr->lkey;
        q.rkey = s.peer_rkey;
        for (unsigned mc = 0; mc < 16u; ++mc) {
            q.lkey_chunks[mc] = s.lkey_chunks[mc] ? s.lkey_chunks[mc] : q.lkey;
            q.rkey_chunks[mc] = s.rkey_chunks[mc] ? s.rkey_chunks[mc] : q.rkey;
        }
        q.cid = s.ah->cid;
        q.osn = 0u;
        q.sq_tail = s.initial_sq_tail;
        q.cq_head = q.cq_tail = q.cq_phase = 0u;
        host.peer[s.peer].q[s.channel] = q;
        host.peer_arena[s.peer] = s.peer_arena;
        std::printf("[gda-mesh] rank=%d peer=%d channel=%u qpn=%u sq=%u cq=%u "
                    "initial_sq_tail=%u status=GPU-owned-data-plane\n",
                    rank, s.peer, static_cast<unsigned>(s.channel), q.qpid,
                    q.sq_count, q.cq_count, q.sq_tail);
    }
    host_detail::check(hipMemcpy(hr->runtime, &host, sizeof(host), hipMemcpyHostToDevice),
                       "hipMemcpy(GDA mesh runtime)");
    std::fflush(stdout);
    return hr;
}

inline void release_mappings() {
    std::lock_guard<std::mutex> lock(host_detail::mapping_mutex());
    auto& mm = host_detail::mappings();
    std::fprintf(stderr,
                 "[gda-mesh-close] stage=unregister_batch_begin mappings=%zu\n",
                 mm.size());
    std::fflush(stderr);
    size_t index = 0;
    for (const auto& item : mm) {
        std::fprintf(stderr,
                     "[gda-mesh-close] stage=unregister_begin index=%zu page=%p "
                     "end=0x%zx flags=%u\n",
                     index, reinterpret_cast<void*>(item.first.page),
                     static_cast<size_t>(item.second.end), item.second.flags);
        std::fflush(stderr);
        errno = 0;
        const hipError_t rc = hipHostUnregister(reinterpret_cast<void*>(item.first.page));
        std::fprintf(stderr,
                     "[gda-mesh-close] stage=unregister_end index=%zu page=%p "
                     "rc=%d errno=%d\n",
                     index, reinterpret_cast<void*>(item.first.page),
                     static_cast<int>(rc), errno);
        std::fflush(stderr);
        if (rc != hipSuccess) {
            std::fprintf(stderr, "[gda-mesh] WARN hipHostUnregister rc=%d %s\n",
                         static_cast<int>(rc), hipGetErrorString(rc));
            std::fflush(stderr);
        }
        ++index;
    }
    mm.clear();
    std::fprintf(stderr,
                 "[gda-mesh-close] stage=unregister_batch_end mappings=%zu\n",
                 index);
    std::fflush(stderr);
}

inline bool report(HostResources* hr, int* d_err, int rank) {
    if (!hr || !hr->runtime) return true;
    Runtime snapshot{};
    if (hipMemcpy(&snapshot, hr->runtime, sizeof(snapshot), hipMemcpyDeviceToHost) != hipSuccess)
        return false;
    bool ok = snapshot.magic == 0x4744414eu;
    for (int d = 0; d < snapshot.ranks; ++d) {
        for (unsigned c = 0; c < CHANNELS; ++c) {
            const Queue& q = snapshot.peer[d].q[c];
            if (!q.sq) continue;
            const bool good = !q.error && q.posted == q.retired &&
                              q.unique_cqes == q.posted &&
                              q.duplicate_cqes == q.posted;
            ok = ok && good;
            std::printf("[gda-mesh] rank=%d peer=%d channel=%u posted=%llu retired=%llu "
                        "unique=%llu duplicate=%llu error=%u %s\n",
                        rank, d, c, static_cast<unsigned long long>(q.posted),
                        static_cast<unsigned long long>(q.retired),
                        static_cast<unsigned long long>(q.unique_cqes),
                        static_cast<unsigned long long>(q.duplicate_cqes), q.error,
                        good ? "PASS" : "FAIL");
        }
    }
    if (d_err) std::printf("[gda-mesh] rank=%d d_err=%d transport=%s\n",
                           rank, *d_err, ok ? "PASS" : "FAIL");
    return ok && snapshot.failed == 0;
}

#endif  // EP16_GDA_DEVICE_ONLY host-only QP/CQ setup and reporting

} // namespace mn_gda_mesh
