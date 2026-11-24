#if V70_PATHB
#ifndef V70M_LDS
#define V70M_LDS 26624u
#endif
#ifndef R2X_RTAB_LDS
#define R2X_RTAB_LDS 28672u
#endif
#ifndef FI_LDS_CNT
#define FI_LDS_CNT 31744u
#endif
#endif
#include "../common/weave_knobs.hpp"
#include "weave_21knobs.hpp"   // official 21-knob surface (exactly 21 compile-time knobs)
// MEGA_OPT_OPTIONS_BEGIN
#ifndef WEAVE_OPT_RELAXED_SPIN
#define WEAVE_OPT_RELAXED_SPIN 0
#endif
#ifndef WEAVE_OPT_LOCAL_SCOPE
#define WEAVE_OPT_LOCAL_SCOPE 0
#endif
#if (WEAVE_OPT_RELAXED_SPIN != 0 && WEAVE_OPT_RELAXED_SPIN != 1)
#error WEAVE_OPT_RELAXED_SPIN must be 0 or 1
#endif
#if (WEAVE_OPT_LOCAL_SCOPE != 0 && WEAVE_OPT_LOCAL_SCOPE != 1)
#error WEAVE_OPT_LOCAL_SCOPE must be 0 or 1
#endif
#if WEAVE_OPT_LOCAL_SCOPE
#define WEAVE_LOCAL_FENCE()  __threadfence()
#define WEAVE_LOCAL_SCOPE_V  __HIP_MEMORY_SCOPE_AGENT
#else
#define WEAVE_LOCAL_FENCE()  __threadfence_system()
#define WEAVE_LOCAL_SCOPE_V  __HIP_MEMORY_SCOPE_SYSTEM
#endif
// MEGA_OPT_OPTIONS_END
// PARALLEL_ROUTE_PLAN_OPTIONS_BEGIN
#ifndef WEAVE_PARALLEL_ROUTE_PLAN
#define WEAVE_PARALLEL_ROUTE_PLAN 0
#endif
#ifndef WEAVE_TILE_TRACE_ONLY
#define WEAVE_TILE_TRACE_ONLY 0
#endif
#if (WEAVE_PARALLEL_ROUTE_PLAN != 0 && WEAVE_PARALLEL_ROUTE_PLAN != 1) || (WEAVE_TILE_TRACE_ONLY != 0 && WEAVE_TILE_TRACE_ONLY != 1)
#error Route-plan and tile-trace switches must be0 or1
#endif
// PARALLEL_ROUTE_PLAN_OPTIONS_END
// ADAPTIVE_POST_TIMEOUT_OPTIONS_BEGIN
#ifndef WEAVE_ADAPTIVE_POST_TIMEOUT
#define WEAVE_ADAPTIVE_POST_TIMEOUT 0
#endif
#if WEAVE_ADAPTIVE_POST_TIMEOUT
#define WEAVE_POST_QUEUE_BUDGET(e) (((e).t_local > 8u) ? 200000000ull : V1988_POST_WAIT_TICKS)
#else
#define WEAVE_POST_QUEUE_BUDGET(e) V1988_POST_WAIT_TICKS
#endif
// ADAPTIVE_POST_TIMEOUT_OPTIONS_END
// SMALLS16_STAGE_OPTIONS_BEGIN
#ifndef WEAVE_S16_PLAN_CAP
#define WEAVE_S16_PLAN_CAP 0
#endif
#ifndef WEAVE_S16_STATIC_DISPATCH
#define WEAVE_S16_STATIC_DISPATCH 0
#endif
#ifndef WEAVE_S16_MAX_LOCAL_TOKENS
#define WEAVE_S16_MAX_LOCAL_TOKENS 8u
#endif
#if (WEAVE_S16_PLAN_CAP != 0 && WEAVE_S16_PLAN_CAP != 1) || (WEAVE_S16_STATIC_DISPATCH != 0 && WEAVE_S16_STATIC_DISPATCH != 1)
#error S16 scheduling switches must be 0 or 1
#endif
// SMALLS16_STAGE_OPTIONS_END
// Independent CPU-only candidate derived from source_1556/work/post_fix_stage_ov1_current.hip.
// Dynamic global payload-key claims use the existing bucket_cursor scratch word;
// post work uses the existing V1988PostControl::claimed ABI word. Not compiled
// or measured. Merge only after the independent raw-barrier repair lands.
// v1864 runtime-role single-launch OV1 kernel.
// One compiled 80x512 body selects 64/16, 72/8, or 76/4 compute/communication
// CTAs from the 800-byte kernarg. GEMM, dispatch, owner-sum return, and home
// sum bodies are byte-for-byte inherited from v988; only the role bounds and
// loop strides are runtime parameters. No whole-grid barrier exists.
//
// This child version starts from the v964-named v941 integration source, uses
// the v940 route-directory producer for the dispatch->GEMM1 seam, and replaces
// the older 32 B LDS ingress staging with the audited v943 per-source global
// staging.  The route directory is finalized first; a grid barrier publishes
// it to all CTAs; one workgroup then consumes each source, seals every source,
// and one worker finishes the ingress generation.
//
// Inputs are the real router outputs (global expert ids and f32 weights), the
// local FP8/scale (or BF16, with scale_groups=0) token rows, and a packed
// expert map whose value is (destination_rank << 16) | local_expert_id.
//
// The kernel builds one bucket per (token,destination), writes the compact
// metadata locally, then streams payload rows directly into each destination
// rank's IPC receive pool.  A per-source progress region exposes:
//   meta_ready   : destination has the bucket header and all metadata slots
//   payload_rows : number of contiguous bucket rows whose payload and scale
//                  have landed and are visible at system scope
//   chunk_done[] : producer-side chunk acknowledgements used to advance the
//                  contiguous payload_rows value
//
// The object is deliberately single-launch and host-prezeroes plan/progress
// state.  v918 appends a receiver-side, per-local-expert counting sort to the
// accepted v809 dispatch body.  Route IDs use the v67 receiver encoding
// g2p | (k << 24), where g2p = source * (capacity + 1) + source_slot.  It uses the v968 80 workgroups x 512 threads envelope, 65,536 B
// static LDS (one workgroup per CU), SGPR<=112, VGPR+AGPR<=512, no private
// memory, no scratch, no spills.  world<=16 is a compile-time ABI bound; the
// active intra-node transport used here is EP8 IPC.  EP16 will need a separate
// transport.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#include "v1262_subwave_barrier.hpp"
#include "official_m16_m32_configs_v1262.hpp"
#include "weave_ck_512_configs.hpp"
#include "gemm2_configs_v1010_hip.hpp"
#include "v956_stage1_body.hpp"
#include "fi_lite_hip.h"
extern "C" {
__device__ FiBuf fi_lite_global;
}

// fast-instrument post-scheduler timeline.  Group 0 is the low-volume core
// path (claim/wait/task/participation), group 1 adds nested task diagnostics,
// and group 2 covers the communication/GEMM preamble.
enum : uint32_t {
  V1988_EV_ENTER_POST = 1,
  V1988_EV_POST_PLAN_WAIT = 2,
  V1988_EV_POST_SCHED = 3,
  V1988_EV_TICKET_CLAIM = 4,
  V1988_EV_TICKET_BASE = 5,
  V1988_EV_QUEUE_WAIT = 6,
  V1988_EV_SCATTER = 7,
  V1988_EV_SCATTER_READY_WAIT = 8,
  V1988_EV_SCATTER_COPY = 9,
  V1988_EV_HOME_REDUCE = 10,
  V1988_EV_HOME_READY_WAIT = 11,
  V1988_EV_HOME_COMPUTE = 12,
  V1988_EV_LOCAL_PLAN = 13,
  V1988_EV_DISPATCH_WAIT_PLAN = 14,
  V1988_EV_DISPATCH_META = 15,
  V1988_EV_DISPATCH_PAYLOAD = 16,
  V1988_EV_ROUTE_PLAN = 17,
  V1988_EV_ROUTE_WAIT_META = 18,
  V1988_EV_DENSE_A = 19,
  V1988_EV_DENSE_A_WAIT_PLAN = 20,
  V1988_EV_DENSE_A_TILE_WAIT = 21,
  V1988_EV_DENSE_A_TILE_COPY = 22,
  V1988_EV_GEMM1 = 23,
  V1988_EV_GEMM1_WAIT_ROUTE = 24,
  V1988_EV_GEMM1_TILE_WAIT = 25,
  V1988_EV_GEMM1_TILE = 26,
  V1988_EV_GEMM2 = 27,
  V1988_EV_GEMM2_TILE_WAIT = 28,
  V1988_EV_GEMM2_TILE = 29,
  V1988_EV_POST_JOIN = 30,
  V1988_EV_KERNEL_DONE = 31,
  V1988_EV_DISPATCH = 32,
};
#define V1988_FI_CORE 0u
#define V1988_FI_DETAIL 1u
#define V1988_FI_PHASE 2u

#define V809_MAX_WORLD 16u
#define V809_MAX_TOPK 12u
#define V809_META_STRIDE 64u
#define V809_META_HEADER 4096u
#define V809_CHUNK_ROWS 8u
#define V809_MAX_CHUNKS 512u
#define V809_PROGRESS_MAGIC 0x44385052u  /* "D8PR" */
#define V918_SORT_BLOCK 32u
#define V1864_CTAS 80u
#define V1864_ROLE_64_16 0u
#define V1864_ROLE_72_8 1u
#define V1864_ROLE_76_4 2u
#define V1867_TIMEOUT_TICKS 2000000ull

struct __attribute__((packed)) V964DispatchExt {
  uint64_t x;                       // local token rows
  uint64_t x_scale;                 // local scale rows; ignored if scale_groups=0
  uint64_t topk_ids;                // int32 [t_local][topk], global expert ids
  uint64_t topk_weights;            // f32  [t_local][topk]
  uint64_t expert_map;              // uint32 [n_experts], (dst<<16)|lid

  // Every table has [world] uint64 entries.  The peer tables contain the
  // destination-rank VA at which this source rank's receive slice starts.
  uint64_t local_meta_table;        // local send metadata regions
  uint64_t peer_meta_table;         // destination recv metadata regions
  uint64_t peer_payload_table;      // destination recv hidden regions
  uint64_t peer_scale_table;        // destination recv scale regions
  uint64_t peer_progress_table;     // destination SourceProgress regions

  uint64_t bucket_count;            // u32 [world]
  uint64_t bucket_cursor;           // u32 [world]
  uint64_t bucket_prefix;           // u32 [world+1]
  uint64_t chunk_prefix;            // u32 [world+1]
  uint64_t total_chunks;            // u32 [1]
  uint64_t error;                   // u32 [1], sticky first error
  uint64_t sync_state;              // u32 barriers, host-prezeroed
  uint64_t reserved0;               // must be zero

  uint32_t t_local;
  uint32_t topk;
  uint32_t capacity;                // bucket slots available per destination
  uint32_t payload_bytes;           // bytes per hidden row, multiple of 16
  uint32_t scale_groups;            // float scales per row; 0 disables scale
  uint32_t world;
  uint32_t rank;
  uint32_t flags;                   // bit0=include_self, bit1=weight bits are FP16 (else BF16)
  uint32_t n_experts;
  uint32_t max_chunks;              // == V809_MAX_CHUNKS

  // v918 receiver-side route table.  All arrays are on the destination rank.
  // recv_meta points at the local [world][META_HEADER + capacity*META_STRIDE]
  // receive area; source s starts at recv_meta + s*meta_region_bytes.
  uint64_t recv_meta;
  uint64_t expert_counts;           // u32 [local_experts], actual valid routes
  uint64_t expert_offsets;          // u32 [local_experts+1], 32-row padded prefix
  uint64_t expert_cursors;          // u32 [local_experts], host-prezeroed scatter cursors
  uint64_t sorted_ids;              // u32 [sort_capacity], v67 route IDs + raw padding
  uint64_t sorted_weights;          // f32 [sort_capacity], expanded from metadata bits
  uint64_t sorted_expert_ids;       // u32 [ceil(sort_capacity/32)]
  uint64_t num_valid_ids;           // u32 [2] = {padded_rows, physical_route_span}
  uint64_t sort_stats;              // u32 [4] = {total_routes, padded_rows, route_span, local_experts}

  // v964 immutable route-directory publication.  All pointers are on the
  // destination rank.  route_generation points to a host-managed nonzero u32
  // that changes once per logical layer generation; inverse_generation is
  // host-zeroed before the first generation and is never cleared afterwards.
  uint64_t route_header;            // RouteDirectoryHeader[1]
  uint64_t source_bucket_rows;      // u32 [world], terminal recv metadata counts
  uint64_t padded_bases;            // u32 [local_experts]
  uint64_t block_prefix;            // u32 [3][local_experts+1]
  uint64_t inverse_generation;      // u32 [world*(capacity+1)*topk]
  uint64_t inverse_row;             // u32 [world*(capacity+1)*topk]
  uint64_t route_entries;           // RouteDirectoryEntry[padded_rows]
  uint64_t route_generation;        // u32 [1], host nonzero generation
  uint32_t sort_capacity;           // total rows available including 32-row padding
  uint32_t sort_block;              // must be V918_SORT_BLOCK

  // v964 ExpertIngress publication.  block_states contains
  // max_blocks+1 entries; the final entry is the inactive-lane sentinel.
  uint64_t ingress_header;          // IngressHeader[1]
  uint64_t ingress_source_states;   // SourceIngressState[world]
  uint64_t ingress_block_states;    // BlockIngressState[max_blocks+1]
  uint32_t ingress_max_blocks;
  uint32_t ingress_block_m;         // 32, 64, or 128
};
static_assert(sizeof(V964DispatchExt) == 360, "V964DispatchExt ABI changed");

// v968 fail-closed CK stage1 attachment.  The first 360 bytes above are kept
// byte-for-byte identical to v964.  The suffix is 168 bytes and starts on an
// eight-byte boundary, so one 528-byte kernarg carries both the accepted
// ingress contract and the exact v956 GEMM1 body.
struct __attribute__((packed)) V968CkExt {
  uint64_t ck_a;                    // dense FP8 route rows, CK p_a_grid
  union {
    uint64_t ck_a_scale;            // legacy ABI name; unused by direct gather
    uint64_t ck_sorted_token_ids;   // Stage-1 phys_slot gather IDs
  };
  uint64_t ck_b;                    // expert-major pre-shuffled gate/up FP8
  uint64_t ck_b_scale;              // expert-major gate/up F32 scales
  uint64_t ck_dense_a;              // provenance alias of ck_a
  uint64_t ck_dense_a_scale;        // provenance alias of ck_a_scale
  uint64_t ck_stage1_output;        // F32 [dense_row_capacity][intermediate]
  uint64_t ck_quant_output;         // FP8 [dense_row_capacity][intermediate]
  uint64_t ck_quant_scale;          // F32 [dense_row_capacity][groups_per_row]
  uint64_t ck_m64_sorted_ids;       // i32 [route_row_capacity]
  uint64_t ck_m64_expert_ids;       // i32 [ceil(route_row_capacity/64)]
  uint64_t ck_m64_offsets;          // u32 [local_experts+1], 64-row padded
  uint64_t ck_m64_cursors;          // u32 [local_experts+1], exact route prefix
  uint64_t ck_max_token_id;         // i32 [local_experts+2]
  uint64_t ck_diagnostics;          // u64 [8], optional but non-null

  uint32_t ck_route_rows;           // host capacity for the runtime-M route table
  uint32_t ck_num_tokens;           // exact upper bound on real received routes
  uint32_t ck_dense_row_capacity;   // physical rows including one sentinel
  uint32_t ck_hidden;               // K, multiple of 128
  uint32_t ck_intermediate;         // N, multiple of 128
  uint32_t ck_local_experts;
  uint32_t ck_topk;                 // must be 1: route ids carry no high k bits
  uint32_t ck_groups_per_row;       // ck_hidden / 128
  uint32_t ck_m64_block_rows;       // must be 64
  uint32_t ck_n_block_cols;         // must be 128
  uint32_t ck_route_sentinel;       // must equal the actual M64 row count at runtime
  uint32_t ck_flags;                // must be zero
};
static_assert(sizeof(V968CkExt) == 168, "V968CkExt ABI changed");

struct __attribute__((packed)) V968DispatchExt {
  V964DispatchExt base;
  V968CkExt ck;
};
static_assert(sizeof(V968DispatchExt) == 528, "V968DispatchExt ABI changed");

struct alignas(64) V809Meta {
  uint32_t home;                    // local source token index
  uint32_t nexp;                    // valid [0,topk] entries
  uint16_t lid[V809_MAX_TOPK];      // destination-local expert ids
  uint16_t wbits[V809_MAX_TOPK];    // BF16 bits for v67; FP16 bits for official baseline
  uint8_t  _reserved[8];            // pad to exact 64 B stride (6-model generalization)
};
static_assert(sizeof(V809Meta) == 64, "V809Meta stride must stay 64");
#define V809_META_STRUCT_DEFINED 1

struct alignas(64) V809SourceProgress {
  uint64_t meta_ready;
  uint64_t payload_rows;
  uint32_t chunk_done[V809_MAX_CHUNKS];
};
static_assert(sizeof(V809SourceProgress) == 2112, "progress header layout changed");




// v1864 runtime OV1 roles: compute CTAs [0,compute_ctas) and communication
// CTAs [comm_base,80). Every CTA is 512T, so all eight waves in a communication
// CTA execute dispatch, owner-sum return, and home-sum work. No whole-grid
// barrier is used; fine-grained acquire/release flags and s_sleep backoff carry
// every producer-consumer dependency.
struct __attribute__((packed)) V983Ov1Ext {
  uint64_t peer_payload_ready_table;
  uint64_t recv_payload_ready;
  uint64_t peer_return_payload_table;
  uint64_t return_recv_payload;
  uint64_t peer_home_arrival_table;
  uint64_t home_arrival;
  uint64_t home_expected;
  uint64_t home_slots;
  uint64_t plan_flags;
  uint64_t a_tile_ready;
  uint64_t a_tile_rows_done;
  uint64_t stage1_tile_done;
  uint64_t row_n_done;
  uint64_t bucket_done;
  uint64_t bucket_route_rows;
  uint64_t bucket_sent;
  uint64_t row_route_id;
  uint64_t stage2_weights;
  uint64_t final_output;
  uint64_t gemm2_output;
  uint64_t ck_w2;
  uint64_t ck_w2_scale;
  uint64_t trace;
  uint64_t byte_counters;
  uint64_t scatter_time;
  uint64_t dispatch_time;
  uint64_t home_done;
  uint64_t post_control;            // local V1988PostControl, followed by its ready queue
  uint64_t gemm_width;              // 128 selects N128, 256 selects N256
  uint64_t peer_post_state_table;   // uint64 [2][world]: peer controls, peer home_expected
  uint32_t hidden;
  uint32_t intermediate;
  uint32_t n1_blocks;
  uint32_t n2_blocks;
  uint32_t mode;
  uint32_t block_m;
  uint32_t comm_chunk_rows;
  uint32_t role_config;             // 0=64/16, 1=72/8, 2=76/4
};
static_assert(sizeof(V983Ov1Ext) == 272, "V983Ov1Ext ABI changed");
struct __attribute__((packed)) V983Envelope {
  V968DispatchExt base;
  V983Ov1Ext ov1;
};
static_assert(sizeof(V983Envelope) == 800, "V1864 envelope ABI changed");

#include "m16_packet_support.hpp"

// v1988 post-work ready queue. Producers reserve one slot with a system-scope
// atomic, then publish a nonzero encoded task into that slot with release
// semantics. Resident waves dynamically reserve disjoint ticket ranges through
// the existing claimed cursor, then acquire-load every unpublished slot in
// their range until ready. The queue is per destination rank; remote scatter
// producers publish home tokens into the destination rank's IPC-shared
// control block directly.
#define V1988_POST_TOKEN_TAG 0x80000000u
#define V1988_POST_BUCKET_TAG 0x40000000u
#define V1988_POST_INDEX_MASK 0x3fffffffu
#define V1988_POST_BATCH 4u
#define V1988_POST_WORKERS 640u
#define V1988_POST_HEADER_BYTES 128u
#define V1988_POST_WAIT_TICKS 20000000ull
struct alignas(64) V1988PostControl {
  // Remote producers publish with system scope, while local waves claim with
  // agent scope.  Keep the two hot counters on separate cache lines so those
  // traffic classes do not invalidate each other's line.
  uint64_t tail;                    // system-scope reserved queue slots
  uint64_t reserved0[7];
  uint64_t claimed;                 // ABI-stable dynamic next-ticket cursor
  uint64_t reserved1[7];
};
static_assert(sizeof(V1988PostControl) == V1988_POST_HEADER_BYTES,
              "V1988 post control header changed");

// Runtime M selects the official M32/M64/M128 GEMM2 body. M128 also selects
// its native N128 tile; M32/M64 use N256. The full forward remains one CO.

struct V1864Roles {
  unsigned compute_ctas;
  unsigned comm_base;
  unsigned comm_ctas;
};

__device__ __forceinline__ bool v1864_decode_roles(unsigned config, V1864Roles& r) {
  switch(config) {
    case V1864_ROLE_64_16: r = {64u, 64u, 16u}; return true;
    case V1864_ROLE_72_8:  r = {72u, 72u,  8u}; return true;
    case V1864_ROLE_76_4:  r = {76u, 76u,  4u}; return true;
    default: return false;
  }
}

__device__ __forceinline__ unsigned long long v983_clock() {
  return (unsigned long long)__builtin_amdgcn_s_memrealtime();
}
__device__ __forceinline__ void v983_error(unsigned int* error, unsigned code) {
  asm volatile("" : "+v"(code)); // opus55 E2
  (void)atomicCAS(error, 0u, code);
}
__device__ __forceinline__ unsigned v983_load_system(const unsigned* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void v983_store_system(unsigned* p, unsigned value) {
  __hip_atomic_store(p, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void v983_store_system64(unsigned long long* p,
                                                     unsigned long long value) {
  __hip_atomic_store(p, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ void v983_sleep(unsigned backoff) {
  if (backoff <= 512u) {
    asm volatile("s_nop 0" ::: "memory");
  } else {
    __builtin_amdgcn_s_sleep(1);
  }
}
__device__ __forceinline__ unsigned* v1988_post_queue(V1988PostControl* control) {
  return (unsigned*)((unsigned char*)control + V1988_POST_HEADER_BYTES);
}
__device__ __forceinline__ unsigned v1988_post_bucket(unsigned key) {
  return V1988_POST_BUCKET_TAG | key;
}
__device__ __forceinline__ unsigned v1988_post_token(unsigned token) {
  return V1988_POST_TOKEN_TAG | token;
}
// Exactly one producer calls this for each ready bucket/token.  The nonzero
// payload doubles as the slot-ready bit; consumers do not trust `tail` alone.
__device__ __forceinline__ void v1988_post_publish(V1988PostControl* control,
                                                    unsigned item) {
  // EP16 has a dedicated bucket scanner; the EP8 ready queue is not consumed.
  (void)control; (void)item;
}
__device__ __forceinline__ unsigned long long v983_divmod_u32(unsigned value,
                                                               unsigned divisor) {
  divisor = __builtin_amdgcn_readfirstlane(divisor);
  asm volatile("" : "+v"(value), "+s"(divisor));
  const unsigned q = value / divisor;
  const unsigned r = value - q * divisor;
  return ((unsigned long long)q << 32) | (unsigned long long)r;
}

// v1895 diagnostic: every cross-role readiness wait is bounded.  The limit is
// deliberately far above a correct launch's wait time; timeout encodes the
// wait code in the existing per-CTA trace word and publishes a sticky device
// error so the host can return compact state instead of hanging the cell.
#define V1895_WAIT_SPINS 1048576u
#define V1895_TIMEOUT_TAG 0x1895000000000000ull
__device__ __forceinline__ bool v983_wait_ge(const unsigned* p, unsigned target,
                                              unsigned long long* spin,
                                              unsigned* error, unsigned code) {
  unsigned backoff = 1u, spins = 0u;
#if WEAVE_OPT_RELAXED_SPIN
  // relaxed 轮询：不带 acquire 语义，因此不会每圈作废 L1/L2；
  // 条件满足后在函数返回前补一次 acquire 栅栏，保证后续读到的 payload 是新的。
  while(__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < target) {
#else
  while(v983_load_system(p) < target) {
#endif
    v983_sleep(backoff);
    if(spin && ((spins & 255u) == 0u)) atomicAdd(spin, 1ull);
    ++spins;
    if(spins >= V1895_WAIT_SPINS) {
      const unsigned observed = v983_load_system(p);
      if(spin) spin[0] = V1895_TIMEOUT_TAG | ((unsigned long long)code << 32) |
                         (unsigned long long)observed;
      v983_error(error, code);
      return false;
    }
    if (backoff <= 512u) ++backoff;
  }
#if WEAVE_OPT_RELAXED_SPIN
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
#endif
  return true;
}
__device__ __forceinline__ uint16_t v983_weight_bits(float value, bool fp16) {
  if(fp16) return __half_as_ushort(__float2half_rn(value));
  const unsigned u = __float_as_uint(value);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float v983_weight_value(uint16_t bits, bool fp16) {
  if(fp16) return __half2float(__ushort_as_half(bits));
  return __uint_as_float((unsigned)bits << 16);
}
__device__ __forceinline__ uint16_t v983_bf16(float value) {
  const unsigned u = __float_as_uint(value);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float v983_bf16_value(uint16_t bits) {
  return __uint_as_float((unsigned)bits << 16);
}
__device__ __forceinline__ void v983_mark(unsigned long long* trace,
                                           unsigned cta, unsigned slot) {
  if(threadIdx.x == 0u) trace[16u + cta * 16u + slot] = v983_clock();
}

// Wave-local wait.  Lane zero performs the bounded system-scope probe and the
// wave reconverges without involving any other wave in the CTA.
__device__ __forceinline__ bool v1864_wave_wait_ge(
    const unsigned* p, unsigned target, unsigned long long* spin,
    unsigned* error, unsigned code) {
  unsigned ok = 1u;
  if((threadIdx.x & 63u) == 0u)
    ok = v983_wait_ge(p, target, spin, error, code) ? 1u : 0u;
  ok = __builtin_amdgcn_readfirstlane(ok);
  __syncwarp();
  return ok != 0u;
}

// ===========================================================================
// WEAVE v4 MegaMoE integration headers (w1 GEMM, w3 gather/tiles, w4 comm,
// w5 6-model generalization).  The carrier already declares g_v1262_phase_arena
// via v1262_subwave_barrier.hpp, so suppress the winner's redeclaration.
// ===========================================================================
#define WEAVE_BK64_ARENA_DECLARED 1
#include "gemm_bk64_winner.hpp"
#define WEAVE_STAGE1_GATHER_IDS(env) \
  (reinterpret_cast<int*>((env).ck.ck_sorted_token_ids))
#include "direct_gather_and_greedy_tiles.hpp"
#include "weave_comm_4wave_turbo.hpp"
#include "sub6_6models_generalize.hpp"

// Communication waves 4..7 execute this routine concurrently with compute
// waves 0..3.  Each of the 320 (CTA, communication-wave) units independently
// claims destination metadata and payload keys.  There is deliberately no CTA
// barrier in this function: a slow communicator cannot hold a GEMM wave, and
// GEMM retirement never gates an unrelated communicator.

#ifndef WEAVE_CO_PROFILE
#define WEAVE_CO_PROFILE 0
#endif
#ifndef WEAVE_TH_BM64
#define WEAVE_TH_BM64 16u
#endif
#ifndef WEAVE_TH_BM128
#define WEAVE_TH_BM128 96u
#endif

struct __attribute__((packed)) WeaveTrackSlot64B {
  unsigned long long t_open;
  unsigned long long t_close;
  unsigned hw_packed;
  unsigned short cta_id;
  unsigned char role;
  unsigned char flags;
  unsigned acc_ticks[8];
  unsigned short count_a;
  unsigned short count_b;
  unsigned short post_scatter_n;
  unsigned short post_home_n;
};
static_assert(sizeof(WeaveTrackSlot64B) == 64, "WeaveTrackSlot64B size");

struct __attribute__((packed)) WeaveSpanRec32B {
  unsigned long long t_start;
  unsigned dur_ticks;
  unsigned short site_id;
  unsigned short expert_id;
  unsigned tile_id;
  unsigned tile_m;
  unsigned short tile_n;
  unsigned short sub_wait_ticks;
  unsigned aux;
};
static_assert(sizeof(WeaveSpanRec32B) == 32, "WeaveSpanRec32B size");

__device__ __forceinline__ unsigned weave_read_hw_packed() {
#if WEAVE_CO_PROFILE
  const unsigned hw_id = __builtin_amdgcn_s_getreg((4 << 0) | (0 << 6) | ((32 - 1) << 11));
  const unsigned xcc_id = __builtin_amdgcn_s_getreg((20 << 0) | (0 << 6) | ((4 - 1) << 11));
  const unsigned cu = (hw_id >> 8) & 0xfu;
  const unsigned simd = (hw_id >> 4) & 0x3u;
  const unsigned se = (hw_id >> 13) & 0x3u;
  const unsigned xcc = xcc_id & 0xfu;
  return cu | (simd << 4) | (se << 8) | (xcc << 12);
#else
  return 0u;
#endif
}

__device__ __forceinline__ WeaveTrackSlot64B* weave_prof_slot(
    unsigned long long* trace, unsigned cta, unsigned role) {
  return reinterpret_cast<WeaveTrackSlot64B*>(trace + 1312u + (cta * 2u + role) * 8u);
}

__device__ __forceinline__ void weave_prof_record_span(
    unsigned long long* trace, unsigned long long t_start, unsigned dur_ticks,
    unsigned short site_id, unsigned short expert_id, unsigned tile_id,
    unsigned tile_m, unsigned short tile_n, unsigned short sub_wait_ticks,
    unsigned cta) {
#if WEAVE_CO_PROFILE
  const unsigned idx = (unsigned)__hip_atomic_fetch_add(
      trace + 1304u, 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  if (idx < 76u) {
    auto* sp = reinterpret_cast<WeaveSpanRec32B*>(trace + 2592u + idx * 4u);
    sp->t_start = t_start;
    sp->dur_ticks = dur_ticks;
    sp->site_id = site_id;
    sp->expert_id = expert_id;
    sp->tile_id = tile_id;
    sp->tile_m = tile_m;
    sp->tile_n = tile_n;
    sp->sub_wait_ticks = sub_wait_ticks;
    sp->aux = (cta & 0xffu) << 16;
  }
#else
  (void)trace; (void)t_start; (void)dur_ticks; (void)site_id; (void)expert_id;
  (void)tile_id; (void)tile_m; (void)tile_n; (void)sub_wait_ticks; (void)cta;
#endif
}

// ===========================================================================
// WEAVE v4 4-wave (tid < 256) compute-side GEMM drivers.
// Tile claim / wait / completion rendezvous use the 4-wave LDS barrier only;
// no __syncthreads() is ever issued while comm waves 4..7 are running.
// LDS broadcast slots: 65472 tile, 65476 wait, 65480 completion (65440..65455
// are the monotone barrier counters; 65484 route fail; 65504 wait context).
// ===========================================================================
__device__ __forceinline__ unsigned weave_v4_next_tile(
    unsigned long long* cursor, unsigned char* arena) {
  const unsigned grp_sh = (threadIdx.x >= 256u) ? 32u : 0u;
  volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65472u + grp_sh)); (void)arena;
  weave_4wave_barrier_65440(arena);
  if ((threadIdx.x & 255u) == 0u)
    sh[0] = (unsigned)__hip_atomic_fetch_add(
        cursor, 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  weave_4wave_barrier_65440(arena);
  return __builtin_amdgcn_readfirstlane(sh[0]); // opus55 E5
}

__device__ __forceinline__ bool weave_v4_wait_ge_4wave(
    const unsigned* p, unsigned target, unsigned long long* spin,
    unsigned* error, unsigned code, unsigned char* arena) {
  const unsigned grp_sh = (threadIdx.x >= 256u) ? 32u : 0u;
  volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65476u + grp_sh)); (void)arena;
  weave_4wave_barrier_65440(arena);
  if ((threadIdx.x & 255u) == 0u)
    sh[0] = v983_wait_ge(p, target, spin, error, code) ? 1u : 0u;
  weave_4wave_barrier_65440(arena);
  return sh[0] != 0u;
}

__device__ __forceinline__ bool weave_v4_wait_eq1_4wave(
    const unsigned* p, unsigned long long* spin,
    unsigned* error, unsigned code, unsigned char* arena) {
  const unsigned grp_sh = (threadIdx.x >= 256u) ? 32u : 0u;
  volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65476u + grp_sh)); (void)arena;
  weave_4wave_barrier_65440(arena);
  if ((threadIdx.x & 255u) == 0u) {
    const bool ok = v983_wait_ge(p, 1u, spin, error, code) &&
                    (__hip_atomic_load(const_cast<unsigned*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) == 1u);
    sh[0] = ok ? 1u : 0u;
  }
  weave_4wave_barrier_65440(arena);
  return sh[0] != 0u;
}

__device__ __forceinline__ bool weave_v4_group_no_error(
    const unsigned* err_ptr, unsigned char* arena) {
  const unsigned grp_sh = (threadIdx.x >= 256u) ? 32u : 0u;
  volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65476u + grp_sh)); (void)arena;
  weave_4wave_barrier_65440(arena);
  if ((threadIdx.x & 255u) == 0u)
    sh[0] = (__hip_atomic_load(err_ptr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 0u) ? 1u : 0u;
  weave_4wave_barrier_65440(arena);
  return sh[0] != 0u;
}

// Stage-1 gather IDs live in the unused ck_stage1_output buffer; Stage-2 row
// IDs live in ck_m64_sorted_ids.  Small (<64) greedy tiles are padded to a
// 64-row BM64 launch in a per-CTA scratch carved from the ck_a tail; padding
// rows carry the out-of-bounds sentinel so the GEMM gathers zeros and never
// writes their output.
__device__ __forceinline__ int32_t* weave_v4_small_tile_scratch(
    const V968CkExt& ck, unsigned total_m_tiles, unsigned cta) {
  const unsigned desc_bytes = (total_m_tiles * 16u + 255u) & ~255u;
  return reinterpret_cast<int32_t*>(ck.ck_a + 64u + desc_bytes +
                                    static_cast<uint64_t>(cta) * 256u);
}

#include "v44_cka.hpp"  // opus55 v44 G2
__device__ __forceinline__ void weave_v4_gemm1_direct(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned char* arena, bool split_128,
    unsigned long long* claim_ptr, bool skip_rdy_wait FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const V968CkExt& ck = env.ck;
  const unsigned cta = blockIdx.x, tid = threadIdx.x & 255u;
  const unsigned grp_sh = (threadIdx.x >= 256u) ? 32u : 0u;
  auto* trace = (unsigned long long*)o.trace;
  const volatile WeaveTilePlanTable* plan =
      reinterpret_cast<const volatile WeaveTilePlanTable*>(ck.ck_a);
  const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(ck.ck_a + 64u);
  if (__builtin_amdgcn_readfirstlane(plan->total_m_tiles) == 0u ||
      __builtin_amdgcn_readfirstlane(plan->total_padded_rows) == 0u) return;  // opus55 E5
  const unsigned n1_blocks = ck.ck_intermediate / 128u;
#if V44_CKA && V44_CKA_READ
  const unsigned num_tokens = ck.ck_route_rows;  // opus55 v44 G2: identity ids into the RR-row shadow
#else
  const unsigned num_tokens = e.world * e.capacity;
#endif
  const int32_t* stage1_ids = reinterpret_cast<const int32_t*>(ck.ck_sorted_token_ids);
  const unsigned* recv_ready = reinterpret_cast<const unsigned*>(o.recv_payload_ready);
  volatile WeaveBk64LocalU32* sh_tile = reinterpret_cast<volatile WeaveBk64LocalU32*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65472u + grp_sh));
#if defined(WEAVE_KNOB_OVERLAP_PIPELINE) && (WEAVE_KNOB_OVERLAP_PIPELINE == 0)
  {
    // Bulk-synchronous stage serialization (-S-T / -S-T-C): all 256 threads wait in
    // parallel on uncached recv_ready (ZERO buffer_inv contention across 80 vs 48 CTAs).
    volatile WeaveBk64LocalU32* sh_disp = reinterpret_cast<volatile WeaveBk64LocalU32*>(
        reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65476u + grp_sh));
    if (tid == 0u) sh_disp[0] = 1u;
    weave_4wave_barrier_65440(arena);
    const auto* sbr = reinterpret_cast<const unsigned*>(e.source_bucket_rows);
    for (unsigned src = 0u; src < e.world; ++src) {
      const unsigned nb = sbr[src];
      for (unsigned sl = tid; sl < nb; sl += 256u) {
        const unsigned long long t0 = weave_clock();
        unsigned sp = 0u;
        while (__hip_atomic_load(recv_ready + src * e.capacity + sl, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM) < 1u) {
          if ((sp & 63u) == 0u && (weave_clock() - t0 >= 20000000ull)) {
            weave_error(reinterpret_cast<unsigned*>(e.error), 208u);
            sh_disp[0] = 0u;
            break;
          }
          if (sp < 256u) asm volatile("s_nop 0\n" ::: "memory");
          else __builtin_amdgcn_s_sleep(1);
          ++sp;
        }
      }
    }
    asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
    weave_4wave_barrier_65440(arena);
    if (sh_disp[0] == 0u) return;
  }
#endif

  for (;;) {
    const unsigned tile = weave_v4_next_tile(claim_ptr ? claim_ptr : (trace + 11u), arena);
    const unsigned total_m_tiles = __builtin_amdgcn_readfirstlane(plan->total_m_tiles);  // opus55 E5
    if (tile >= total_m_tiles * n1_blocks) break;
    const unsigned mt = __builtin_amdgcn_readfirstlane(tile / n1_blocks);
    const unsigned n1 = __builtin_amdgcn_readfirstlane(tile - mt * n1_blocks);
    const WeaveMTileDesc d_raw = tiles[mt];
    WeaveMTileDesc d;
    d.expert_id  = __builtin_amdgcn_readfirstlane(d_raw.expert_id);
    d.row_start  = __builtin_amdgcn_readfirstlane(d_raw.row_start);
    d.bm         = static_cast<uint16_t>(__builtin_amdgcn_readfirstlane(d_raw.bm));
    d.valid_rows = static_cast<uint16_t>(__builtin_amdgcn_readfirstlane(d_raw.valid_rows));
    d.tile_idx   = __builtin_amdgcn_readfirstlane(d_raw.tile_idx);
    const unsigned run = __builtin_amdgcn_readfirstlane(plan->total_padded_rows);
    const unsigned pad_gather = __builtin_amdgcn_readfirstlane(plan->pad_gather_slot);
#if V44_CKA && V44_CKA_READ
    const unsigned char* recv_payload = v44_cka_a(env);   // opus55 v44 G2: cached shadow
    const float* recv_scale = v44_cka_s(env);
#else
    const unsigned char* recv_payload = reinterpret_cast<const unsigned char*>(
        reinterpret_cast<const uint64_t*>(e.peer_payload_table)[e.rank]) -
        static_cast<uint64_t>(e.rank) * e.capacity * e.payload_bytes;
    const float* recv_scale = reinterpret_cast<const float*>(
        reinterpret_cast<const uint64_t*>(e.peer_scale_table)[e.rank]) -
        static_cast<uint64_t>(e.rank) * e.capacity * e.scale_groups;
#endif
    // Cache one validated readiness bit per M tile; every launch resets this array.
    auto* cached_ready = reinterpret_cast<unsigned*>(o.a_tile_ready) + mt;
    volatile WeaveBk64LocalU32* cached_sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65488u + grp_sh));
    if (!skip_rdy_wait) {
    if (tid == 0u)
      *cached_sh = __hip_atomic_load(cached_ready, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    weave_4wave_barrier_65440(arena);
    if (*cached_sh != 0u) {
      // recv_payload & recv_scale are in hardware-uncached self.arena: ZERO buffer_inv!
    } else {
      FI_BG(V1988_FI_DETAIL, V1988_EV_GEMM1_TILE_WAIT, mt);
      const bool rdy_ok = weave_wait_tile_payload_ready(recv_ready, stage1_ids, d, pad_gather,
                                         trace + 16u + cta * 16u + 14u,
                                         reinterpret_cast<unsigned*>(e.error));
      FI_EG(V1988_FI_DETAIL, V1988_EV_GEMM1_TILE_WAIT, mt);
      if (!rdy_ok) return;
      if (tid == 0u)
        __hip_atomic_store(cached_ready, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    }

    const unsigned g1_aux = ((unsigned)d.bm << 16u) | (unsigned)d.valid_rows;
    FI_BG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
#if V44_CKA && V44_CKA_READ
    const int32_t* ids = v44_cka_ids(env) + d.row_start;
#else
    const int32_t* ids = stage1_ids + d.row_start;
#endif
    if (split_128 && (d.bm == 128u || d.bm == 256u)) {
      #pragma clang loop unroll(disable)
      for (unsigned r_off = 0u; r_off < d.valid_rows; r_off += 64u) {
        const unsigned r_rem = d.valid_rows - r_off;
        const unsigned chunk = (r_rem > 64u) ? 64u : r_rem;
        if (chunk > 32u) {
          weave_dispatch_stage1_bk64<64u>(ids + r_off, (int32_t)d.expert_id, recv_payload,
              reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
              reinterpret_cast<const float*>(ck.ck_b_scale), nullptr,
              reinterpret_cast<uint8_t*>(ck.ck_quant_output),
              reinterpret_cast<float*>(ck.ck_quant_scale), d.row_start + r_off, chunk, n1,
              num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
        } else {
          weave_dispatch_stage1_bk64<32u>(ids + r_off, (int32_t)d.expert_id, recv_payload,
              reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
              reinterpret_cast<const float*>(ck.ck_b_scale), nullptr,
              reinterpret_cast<uint8_t*>(ck.ck_quant_output),
              reinterpret_cast<float*>(ck.ck_quant_scale), d.row_start + r_off, chunk, n1,
              num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
        }
      }
    } else if (d.bm == 256u) {
      const unsigned v0 = (d.valid_rows < 128u) ? d.valid_rows : 128u;
      weave_dispatch_stage1_bk64<128u>(ids, (int32_t)d.expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          nullptr,
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), d.row_start, v0, n1,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
      if (d.valid_rows > 128u) {
        const unsigned rem = d.valid_rows - 128u;
        const int32_t* ids1 = ids + 128u;
        const unsigned rs1 = d.row_start + 128u;
        if (rem > 64u) {
          weave_dispatch_stage1_bk64<128u>(ids1, (int32_t)d.expert_id, recv_payload,
              reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
              reinterpret_cast<const float*>(ck.ck_b_scale), nullptr,
              reinterpret_cast<uint8_t*>(ck.ck_quant_output),
              reinterpret_cast<float*>(ck.ck_quant_scale), rs1, rem, n1,
              num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
        } else if (rem > 32u) {
          weave_dispatch_stage1_bk64<64u>(ids1, (int32_t)d.expert_id, recv_payload,
              reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
              reinterpret_cast<const float*>(ck.ck_b_scale), nullptr,
              reinterpret_cast<uint8_t*>(ck.ck_quant_output),
              reinterpret_cast<float*>(ck.ck_quant_scale), rs1, rem, n1,
              num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
        } else {
          weave_dispatch_stage1_bk64<32u>(ids1, (int32_t)d.expert_id, recv_payload,
              reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
              reinterpret_cast<const float*>(ck.ck_b_scale), nullptr,
              reinterpret_cast<uint8_t*>(ck.ck_quant_output),
              reinterpret_cast<float*>(ck.ck_quant_scale), rs1, rem, n1,
              num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
        }
      }
    } else if (d.bm == 128u) {
      weave_dispatch_stage1_bk64<128u>(ids, (int32_t)d.expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          nullptr,
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), d.row_start, d.valid_rows, n1,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
    } else if (d.bm == 32u) {
      weave_dispatch_stage1_bk64<32u>(ids, (int32_t)d.expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          nullptr,
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), d.row_start, d.valid_rows, n1,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
    } else {
      weave_dispatch_stage1_bk64<64u>(ids, (int32_t)d.expert_id, recv_payload,
          reinterpret_cast<const uint8_t*>(ck.ck_b), recv_scale,
          reinterpret_cast<const float*>(ck.ck_b_scale),
          nullptr,
          reinterpret_cast<uint8_t*>(ck.ck_quant_output),
          reinterpret_cast<float*>(ck.ck_quant_scale), d.row_start, d.valid_rows, n1,
          num_tokens, run, ck.ck_intermediate, ck.ck_hidden, arena);
    }
    __builtin_amdgcn_s_waitcnt(0);
    weave_4wave_barrier_65440(arena);
    FI_EG(V1988_FI_CORE, V1988_EV_GEMM1_TILE, g1_aux);
    if (tid == 0u) {
      // Single AGENT release atomic: LLVM emits buffer_wbl2 sc1 + s_waitcnt + flat_atomic_add (ZERO buffer_inv!)
      const unsigned done_mt = mt;
      __hip_atomic_fetch_add(reinterpret_cast<unsigned*>(o.stage1_tile_done) + done_mt,
                             1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

__device__ __forceinline__ void weave_v4_gemm2_direct(
    const V968DispatchExt& env, const V983Ov1Ext& o, unsigned char* arena, bool split_128, unsigned* compute_done_ptr,
    unsigned long long* claim_ptr, unsigned g2_base_row FI_CTX_PARAM) {
  const V964DispatchExt& e = env.base;
  const V968CkExt& ck = env.ck;
  const unsigned cta = blockIdx.x, tid = threadIdx.x & 255u;
  const unsigned grp_sh = (threadIdx.x >= 256u) ? 32u : 0u;
  auto* trace = (unsigned long long*)o.trace;
  const volatile WeaveTilePlanTable* plan =
      reinterpret_cast<const volatile WeaveTilePlanTable*>(ck.ck_a);
  const auto* tiles = reinterpret_cast<const WeaveMTileDesc*>(ck.ck_a + 64u);
  if (__builtin_amdgcn_readfirstlane(plan->total_m_tiles) == 0u ||
      __builtin_amdgcn_readfirstlane(plan->total_padded_rows) == 0u) return;  // opus55 E5
  const unsigned n1_blocks = ck.ck_intermediate / 128u;
  const unsigned n2_blocks = o.hidden / 128u;
  const int32_t* stage2_ids = reinterpret_cast<const int32_t*>(ck.ck_m64_sorted_ids);
  const float* weights = reinterpret_cast<const float*>(o.stage2_weights);
  const unsigned* row_route_id = reinterpret_cast<const unsigned*>(o.row_route_id);
  const unsigned region_bytes = V809_META_HEADER + e.capacity * V809_META_STRIDE;
  volatile WeaveBk64LocalU32* sh_done = reinterpret_cast<volatile WeaveBk64LocalU32*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65480u + grp_sh));
  volatile WeaveBk64LocalU32* sh_tile = reinterpret_cast<volatile WeaveBk64LocalU32*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65472u + grp_sh));

  const unsigned init_m_tiles = __builtin_amdgcn_readfirstlane(plan->total_m_tiles);
#if WEAVE_KNOB_BN2_MODE == 64
  const bool use_bn64 = true;
#elif WEAVE_KNOB_BN2_MODE == 128
  const bool use_bn64 = false;
#else
  const unsigned n_large_tiles = __builtin_amdgcn_readfirstlane(plan->num_tiles[0] + plan->num_tiles[1]);
  const bool use_bn64 = (n_large_tiles == 0u) && (init_m_tiles >= 16u) && (init_m_tiles * n2_blocks <= 320u);
#endif
#if WEAVE_KNOB_S2_NGROUP_MODE >= 1
  const unsigned n2_group = (!use_bn64 && ((n2_blocks % WEAVE_KNOB_S2_NGROUP_MODE) == 0u)) ? WEAVE_KNOB_S2_NGROUP_MODE : 1u;
#else
  const unsigned ng4_min_work = split_128 ? 256u : 160u;
  const unsigned ng2_min_work = split_128 ? 192u : 128u;
  const unsigned n2_group = (!use_bn64)
      ? ((((n2_blocks & 3u) == 0u) && (init_m_tiles * (n2_blocks >> 2u) >= ng4_min_work)) ? 4u
         : ((((n2_blocks & 1u) == 0u) && (init_m_tiles * (n2_blocks >> 1u) >= ng2_min_work)) ? 2u : 1u))
      : 1u;
#endif
  const unsigned n2_claims_per_mt = use_bn64 ? (n2_blocks * 2u) : (n2_blocks / n2_group);
  const unsigned n2_blocks_eff = use_bn64 ? (n2_blocks * 2u) : n2_blocks;
  unsigned last_acq_mt = 0xffffffffu;

  for (;;) {
    const unsigned tile = weave_v4_next_tile(claim_ptr ? claim_ptr : (trace + 12u), arena);
    const unsigned total_m_tiles = __builtin_amdgcn_readfirstlane(plan->total_m_tiles);
    if (tile >= total_m_tiles * n2_claims_per_mt) break;
    const unsigned mt = __builtin_amdgcn_readfirstlane(tile / n2_claims_per_mt);
    const unsigned claim_idx = __builtin_amdgcn_readfirstlane(tile - mt * n2_claims_per_mt);
    const WeaveMTileDesc d_raw = tiles[mt];
    WeaveMTileDesc d;
    d.expert_id  = __builtin_amdgcn_readfirstlane(d_raw.expert_id);
    d.row_start  = __builtin_amdgcn_readfirstlane(d_raw.row_start);
    d.bm         = static_cast<uint16_t>(__builtin_amdgcn_readfirstlane(d_raw.bm));
    d.valid_rows = static_cast<uint16_t>(__builtin_amdgcn_readfirstlane(d_raw.valid_rows));
    d.tile_idx   = __builtin_amdgcn_readfirstlane(d_raw.tile_idx);
    const unsigned run = __builtin_amdgcn_readfirstlane(plan->total_padded_rows);
    FI_BG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt);
    if (mt != last_acq_mt) {
      volatile WeaveBk64LocalU32* sh_w = reinterpret_cast<volatile WeaveBk64LocalU32*>(
          reinterpret_cast<uintptr_t>(g_v1262_phase_arena + 65476u + grp_sh));
      weave_4wave_barrier_65440(arena);
      if (tid == 0u) {
        const unsigned* s1p = reinterpret_cast<const unsigned*>(o.stage1_tile_done) + mt;
        unsigned backoff = 1u, spins = 0u, ok = 1u;
        while (__hip_atomic_load(s1p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < n1_blocks) {
          if (spins < 128u) asm volatile("s_nop 0\n" ::: "memory");
          else v983_sleep(backoff);
          ++spins;
          if (spins >= V1895_WAIT_SPINS) {
            v983_error(reinterpret_cast<unsigned*>(e.error), 207u);
            ok = 0u;
            break;
          }
          if (backoff <= 64u) ++backoff;
        }
        sh_w[0] = ok;
      }
      weave_4wave_barrier_65440(arena);
      if (__builtin_amdgcn_readfirstlane(sh_w[0]) == 0u) { FI_EG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt); return; }
      // Un-elidable once-per-mt AGENT acquire (L1+L2 inv only; 128MB MALL W2 weights stay 100% hot!)
      asm volatile("buffer_inv sc1\ns_waitcnt vmcnt(0)\n" ::: "memory");
      last_acq_mt = mt;
    }
    FI_EG(V1988_FI_DETAIL, V1988_EV_GEMM2_TILE_WAIT, mt);

    const unsigned g2_aux = ((use_bn64 && d.bm <= 64u) ? 0x80000000u : 0u) |
                            (((n2_group & 0x7u) << 28u)) |
                            (((unsigned)d.bm & 0xfffu) << 16u) |
                            ((unsigned)d.valid_rows & 0xffffu);
    FI_BG(V1988_FI_CORE, V1988_EV_GEMM2_TILE, g2_aux);
    const int32_t* ids = stage2_ids + d.row_start;
    #pragma clang loop unroll(disable)
    for (unsigned g = 0u; g < n2_group; ++g) {
      const unsigned n2_eff = claim_idx * n2_group + g;
      const unsigned n2 = use_bn64 ? (n2_eff >> 1u) : n2_eff;
      if (use_bn64 && d.bm <= 64u) {
        if (d.bm == 32u) {
          weave_s2_small_1x4_bn64<32u>(ids, (int32_t)d.expert_id,
              reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
              reinterpret_cast<const uint8_t*>(o.ck_w2),
              reinterpret_cast<const float*>(ck.ck_quant_scale),
              reinterpret_cast<const float*>(o.ck_w2_scale), weights + d.row_start,
              (reinterpret_cast<uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(g2_base_row) * o.hidden), d.row_start, d.valid_rows, n2_eff, run,
              o.hidden, ck.ck_intermediate, arena);
        } else {
          weave_s2_small_1x4_bn64<64u>(ids, (int32_t)d.expert_id,
              reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
              reinterpret_cast<const uint8_t*>(o.ck_w2),
              reinterpret_cast<const float*>(ck.ck_quant_scale),
              reinterpret_cast<const float*>(o.ck_w2_scale), weights + d.row_start,
              (reinterpret_cast<uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(g2_base_row) * o.hidden), d.row_start, d.valid_rows, n2_eff, run,
              o.hidden, ck.ck_intermediate, arena);
        }
      } else if (use_bn64 && (n2_eff & 1u) != 0u) {
        // Odd half of >=128 row tile already computed by even half (BN=128)
      } else if (split_128 && (d.bm == 128u || d.bm == 256u)) {
        #pragma clang loop unroll(disable)
        for (unsigned r_off = 0u; r_off < d.valid_rows; r_off += 64u) {
          const unsigned r_rem = d.valid_rows - r_off;
          const unsigned chunk = (r_rem > 64u) ? 64u : r_rem;
          const unsigned rs_c = d.row_start + r_off;
          if (chunk > 32u) {
            weave_dispatch_stage2_bk64<64u>(ids + r_off, (int32_t)d.expert_id,
                reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
                reinterpret_cast<const uint8_t*>(o.ck_w2),
                reinterpret_cast<const float*>(ck.ck_quant_scale),
                reinterpret_cast<const float*>(o.ck_w2_scale), weights + rs_c,
                (reinterpret_cast<uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(g2_base_row) * o.hidden), rs_c, chunk, n2, run,
                o.hidden, ck.ck_intermediate, arena);
          } else {
            weave_dispatch_stage2_bk64<32u>(ids + r_off, (int32_t)d.expert_id,
                reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
                reinterpret_cast<const uint8_t*>(o.ck_w2),
                reinterpret_cast<const float*>(ck.ck_quant_scale),
                reinterpret_cast<const float*>(o.ck_w2_scale), weights + rs_c,
                (reinterpret_cast<uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(g2_base_row) * o.hidden), rs_c, chunk, n2, run,
                o.hidden, ck.ck_intermediate, arena);
          }
        }
      } else if (d.bm == 256u) {
        const unsigned v0 = (d.valid_rows > 128u) ? 128u : d.valid_rows;
        weave_dispatch_stage2_bk64<128u>(ids, (int32_t)d.expert_id,
            reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
            reinterpret_cast<const uint8_t*>(o.ck_w2),
            reinterpret_cast<const float*>(ck.ck_quant_scale),
            reinterpret_cast<const float*>(o.ck_w2_scale), weights + d.row_start,
            (reinterpret_cast<uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(g2_base_row) * o.hidden), d.row_start, v0, n2, run,
            o.hidden, ck.ck_intermediate, arena);
        if (d.valid_rows > 128u) {
          const unsigned rem = d.valid_rows - 128u;
          const unsigned rs1 = d.row_start + 128u;
          const int32_t* ids1 = ids + 128u;
          const float* w1 = weights + rs1;
          if (rem > 64u) {
            weave_dispatch_stage2_bk64<128u>(ids1, (int32_t)d.expert_id,
                reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
                reinterpret_cast<const uint8_t*>(o.ck_w2),
                reinterpret_cast<const float*>(ck.ck_quant_scale),
                reinterpret_cast<const float*>(o.ck_w2_scale), w1,
                (reinterpret_cast<uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(g2_base_row) * o.hidden), rs1, rem, n2, run,
                o.hidden, ck.ck_intermediate, arena);
          } else if (rem > 32u) {
            weave_dispatch_stage2_bk64<64u>(ids1, (int32_t)d.expert_id,
                reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
                reinterpret_cast<const uint8_t*>(o.ck_w2),
                reinterpret_cast<const float*>(ck.ck_quant_scale),
                reinterpret_cast<const float*>(o.ck_w2_scale), w1,
                (reinterpret_cast<uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(g2_base_row) * o.hidden), rs1, rem, n2, run,
                o.hidden, ck.ck_intermediate, arena);
          } else {
            weave_dispatch_stage2_bk64<32u>(ids1, (int32_t)d.expert_id,
                reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
                reinterpret_cast<const uint8_t*>(o.ck_w2),
                reinterpret_cast<const float*>(ck.ck_quant_scale),
                reinterpret_cast<const float*>(o.ck_w2_scale), w1,
                (reinterpret_cast<uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(g2_base_row) * o.hidden), rs1, rem, n2, run,
                o.hidden, ck.ck_intermediate, arena);
          }
        }
      } else if (d.bm == 128u) {
        weave_dispatch_stage2_bk64<128u>(ids, (int32_t)d.expert_id,
            reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
            reinterpret_cast<const uint8_t*>(o.ck_w2),
            reinterpret_cast<const float*>(ck.ck_quant_scale),
            reinterpret_cast<const float*>(o.ck_w2_scale), weights + d.row_start,
            (reinterpret_cast<uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(g2_base_row) * o.hidden), d.row_start, d.valid_rows, n2, run,
            o.hidden, ck.ck_intermediate, arena);
      } else if (d.bm == 32u) {
        weave_dispatch_stage2_bk64<32u>(ids, (int32_t)d.expert_id,
            reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
            reinterpret_cast<const uint8_t*>(o.ck_w2),
            reinterpret_cast<const float*>(ck.ck_quant_scale),
            reinterpret_cast<const float*>(o.ck_w2_scale), weights + d.row_start,
            (reinterpret_cast<uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(g2_base_row) * o.hidden), d.row_start, d.valid_rows, n2, run,
            o.hidden, ck.ck_intermediate, arena);
      } else {
        weave_dispatch_stage2_bk64<64u>(ids, (int32_t)d.expert_id,
            reinterpret_cast<const uint8_t*>(ck.ck_quant_output),
            reinterpret_cast<const uint8_t*>(o.ck_w2),
            reinterpret_cast<const float*>(ck.ck_quant_scale),
            reinterpret_cast<const float*>(o.ck_w2_scale), weights + d.row_start,
            (reinterpret_cast<uint16_t*>(o.gemm2_output) + static_cast<uint64_t>(g2_base_row) * o.hidden), d.row_start, d.valid_rows, n2, run,
            o.hidden, ck.ck_intermediate, arena);
      }
    }
    __builtin_amdgcn_s_waitcnt(0);
    weave_4wave_barrier_65440(arena);
    FI_EG(V1988_FI_CORE, V1988_EV_GEMM2_TILE, g2_aux);
    const unsigned done_mt = mt;
    if (tid == 0u) {
      // Single AGENT release atomic: LLVM emits buffer_wbl2 sc1 + s_waitcnt + flat_atomic_add (ZERO buffer_inv!)
      sh_done[0] = __hip_atomic_fetch_add(
          reinterpret_cast<unsigned*>(o.row_n_done) + done_mt, n2_group,
          __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT) + n2_group;
    }
    weave_4wave_barrier_65440(arena);
    const unsigned completed = __builtin_amdgcn_readfirstlane(sh_done[0]);  // opus55 E5
    if (completed == n2_blocks_eff) {
      const WeaveMTileDesc done_desc = tiles[done_mt];
      for (unsigned r = tid; r < done_desc.valid_rows; r += 256u) {
        const unsigned row = done_desc.row_start + r;
        const unsigned id = row_route_id[row];
        if (id == 0xffffffffu) continue;
        const unsigned g2p = id & 0xffffffu;
        const unsigned long long divmod = v983_divmod_u32(g2p, e.capacity + 1u);
        const unsigned src = (unsigned)(divmod >> 32);
        const unsigned slot = (unsigned)divmod;
        const unsigned bucket_key = src * e.capacity + slot;
        __hip_atomic_fetch_add(
            reinterpret_cast<unsigned*>(o.bucket_done) + bucket_key, 1u,
            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
      weave_4wave_barrier_65440(arena);
      if (tid == 0u && compute_done_ptr) {
        __hip_atomic_fetch_add(compute_done_ptr, o.n2_blocks, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
}

#include "ep16_transport.hpp"
#include "ep16_combine.hpp"

// v20 launch ABI: 800-byte by-value envelope + 8-byte context pointer.
#include "v41_dedup_overlap.hpp"
#include "v44_combine.hpp"  // opus55 v44 C1
#include "v45_tail.hpp"  // opus55 v45 M1 (V45_TAIL; alias only when off)
#include "v45_front.hpp"  // opus55 v45 M2 (V45_META/V45_STAGE/V45_PULL; alias only when off)
#include "v46_sp.hpp"  // opus55 v46 SP (3-mode CU scheduler + single-plan 3-pass overlap)

#if V70_PATHB
extern "C" __global__ __launch_bounds__(256, 2)
#else
extern "C" __global__ __launch_bounds__(512, 1)
#endif
void stage_ep16_turbo_overlap(V983Envelope envelope,
                              const Ep16TurboContext* __restrict__ context) {
  const unsigned cta = blockIdx.x, tid = threadIdx.x,
                 wave = __builtin_amdgcn_readfirstlane(tid >> 6u); // opus55 E1
  if (!context) return;
  (void)envelope;
  const V983Envelope* envp = reinterpret_cast<const V983Envelope*>(
      __builtin_amdgcn_kernarg_segment_ptr());
  const auto& env = envp->base; const auto& e = env.base; const auto& o = envp->ov1;
#if V70_PATHB
  if (gridDim.x != 160u || gridDim.y != 1u || gridDim.z != 1u ||
      blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u) {
#else
  if (gridDim.x != 80u || gridDim.y != 1u || gridDim.z != 1u ||
      blockDim.x != 512u || blockDim.y != 1u || blockDim.z != 1u) {
#endif
    if (cta == 0u && tid == 0u && e.error) v983_error((unsigned*)e.error, 99u);
    return;
  }
  const auto& x = *context;
  if (!v1864_validate_envelope_general(env, o, tid) || e.world != 16u ||
      x.magic != EP16_TURBO_MAGIC || !x.generation || !x.runtime ||
      !x.timeout_ticks || !x.compute_done || !x.gda_state) {
    if (tid == 0u && e.error) v983_error((unsigned*)e.error, 900u);
    return;
  }
  auto* arena = g_v1262_phase_arena;
  auto* trace = (unsigned long long*)o.trace;
  if (tid == 0u) trace[16u + cta * 16u] = v983_clock();
#if !V70_PATHB
  weave_4wave_barrier_reset_65440(arena);
  if (tid == 0u) v1262_publish_wait_context((unsigned*)e.error, e.rank, 1u);
  __syncthreads();
#endif
  FI_OPEN(wave);

  const bool dual = (e.t_local >= 16u);
  const unsigned n_comp_groups = dual ? 148u : 80u;
  const unsigned NT = x.node_tokens;
  const unsigned M = v41_num_mb(NT);
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
#if V46_SP
  (void)pf; (void)M; (void)n_comp_groups; (void)dual;
#if V70_PATHB
  v46_main_pathb(env, o, x, arena, cta, wave FI_CTX_ARG);
#else
  v46_main(env, o, x, arena, cta, wave FI_CTX_ARG);
#endif
  FI_CLOSE();
  return;
#else

  // ---------------- ROLE 1: Inter-Node RDMA Engine (CTAs 0..3, w4 = 4 QPs on owner_peer) ----------------
  if (cta < 4u && wave == 4u) {
    ep16_gda_wave4(env, o, x FI_CTX_ARG);
    __syncwarp();
    FI_CLOSE();
    return;
  }

  // ---------------- ROLE 2A: Meta Stage + 33-job Inter-Node RX Watcher + Home (CTAs 0..3, w5) ----------------
  if (cta < 4u && wave == 5u) {
    FI_BG(V1988_FI_CORE, 15u, cta);  // opus55 WP probe 15: local meta + payload stage (FI0: no-op)
#if !V45_META
    v41_stage_local_meta(env, o, x);
#endif
    FI_EG(V1988_FI_CORE, 15u, cta);
    // V42 D1: a PURE RX watcher (no O(t_local) HomeCombine scan in the same loop!)
    FI_BG(V1988_FI_CORE, 45u, cta);  // opus55 WP probe 45: inter-node RX flag watch window (FI0: no-op)
    v41_combine_loop(env, o, x, cta, 4u, true, false, false, false FI_CTX_ARG);
    FI_EG(V1988_FI_CORE, 45u, cta);
#if V44_COMBINE
    V45_COMBINE_LOOP(env, o, x, cta, true, true, true FI_CTX_ARG);  // opus55 v44 C1 / v45 M1
#else
    v41_combine_loop(env, o, x, cta, 4u, false, true, true, true FI_CTX_ARG);
#endif
    if ((tid & 63u) == 0u && trace) trace[16u + cta * 16u + 5u] = v983_clock();
    FI_CLOSE();
    return;
  }

  // ---------------- ROLE 2C: Continuous Return Overlap Engine (CTAs 0..3, w6..w7 = 8 waves) ----------------
  if (cta < 4u && wave >= 6u) {
    const unsigned r_worker = cta * 2u + (wave - 6u);
#if V44_COMBINE
    V45_COMBINE_LOOP(env, o, x, 4u + r_worker, true, true, true FI_CTX_ARG);  // opus55 v44 C1 / v45 M1
#else
    v41_combine_loop(env, o, x, r_worker, 8u, false, true, true, true FI_CTX_ARG);
#endif
    if ((tid & 63u) == 0u && trace) trace[16u + cta * 16u + 5u] = v983_clock();
    FI_CLOSE();
    return;
  }

  // ---------------- ROLE 2B: Local Payload Stage + Intra-Node xGMI Pull -> Return/Combine (CTAs 4..11, w4..w7) ----------------
  if (cta >= 4u && cta < 12u && wave >= 4u) {
    const unsigned p_worker = (cta - 4u) * 4u + (wave - 4u); // 0..31
#if V45_META || V45_STAGE
    v45_front_stage(env, o, x, p_worker, 32u + n_comp_groups * 4u FI_CTX_ARG);  // opus55 v45 M2 F1/F2
#endif
#if !V45_STAGE
    v41_stage_local_payload(env, o, x, p_worker);
#endif
    if (p_worker == 0u && (tid & 63u) == 0u && trace) trace[16u + cta * 16u + 2u] = v983_clock();
    // V42 D5: wait for microbatch b's readiness flags, then join the 624-worker copy
    for (unsigned b = 0u; b < M; ++b) {
#if V45_PULL != 2
      v42_pull_wait_mb(env, o, x, b, p_worker);
#endif
      // V42G G3: compute waves only join MB0's copy; MB1.. are copied by these 32 waves
      V45_PULL_COPY(env, o, x, b, p_worker, (b == 0u) ? (32u + n_comp_groups * 4u) : 32u FI_CTX_ARG);  // v45 F3
    }
    // Immediately help Return/Combine for MB 0..M-1!
#if V44_COMBINE
    V45_COMBINE_LOOP(env, o, x, 12u + p_worker, true, true, true FI_CTX_ARG);  // opus55 v44 C1 / v45 M1
#else
    v41_combine_loop(env, o, x, p_worker, 32u, false, true, true, true FI_CTX_ARG);
#endif
    if ((tid & 63u) == 0u && trace) trace[16u + cta * 16u + 5u] = v983_clock();
    FI_CLOSE();
    return;
  }

  // ---------------- ROLE 3: 148 Compute Groups (592 waves!) -> M-Microbatch Pipelined Compute ----------------
  if (wave >= 4u && !dual) {
    FI_CLOSE();
    return;
  }
  const unsigned grp_id = (wave < 4u) ? cta : (80u + (cta - 12u)); // 0..147 (when dual), 0..79 (when !dual)
  const unsigned ltid = tid & 255u;

#if V45_META || V45_STAGE
  v45_front_stage(env, o, x, 32u + grp_id * 4u + (wave & 3u), 32u + n_comp_groups * 4u FI_CTX_ARG);  // opus55 v45 M2 F1/F2
#endif
  if (!v41_meta_wait_and_scan(env, o, x, arena, grp_id, n_comp_groups)) {
    FI_CLOSE();
    return;
  }

  // V42 D5: the 592 compute waves are idle here -> they join the cooperative
  // intra-node xGMI pull of EVERY microbatch (624 workers instead of 32).
  {
    const unsigned pw = 32u + grp_id * 4u + (wave & 3u);
    const unsigned nw = 32u + n_comp_groups * 4u;
    V45_PULL_COPY(env, o, x, 0u, pw, nw FI_CTX_ARG); // V42G G3: MB0 only / v45 F3
  }
  weave_4wave_barrier_65440(arena);
  if (__builtin_amdgcn_readfirstlane(ep16_load_relaxed((unsigned*)e.error))) { FI_CLOSE(); return; }  // opus55 E5

  const auto start_c = __builtin_amdgcn_s_memrealtime();
  for (unsigned b = 0u; b < M; ++b) {
    unsigned g2_base = 0u;
    if (!v41_mb_route_plan(env, o, x, arena, grp_id, n_comp_groups, b, M, g2_base)) {
      FI_CLOSE();
      return;
    }
    // Wait ONLY for MB b xGMI Pull (MB 1..M-1 are ALREADY pulled in background -> ZERO wait!)
    if (ltid == 0u) {
      unsigned sp = 0u;
      while (v42_ld_agent_rlx(pf + 16u + b) == 0u) {
        if ((sp & 63u) == 0u && (ep16_load_relaxed((unsigned*)e.error) || (__builtin_amdgcn_s_memrealtime() - start_c > x.timeout_ticks))) {
          v983_error((unsigned*)e.error, 916u);
          break;
        }
        if (sp < 64u) asm volatile("s_nop 0\n" ::: "memory");
        else __builtin_amdgcn_s_sleep(1);
        ++sp;
      } v42_acq_agent();
      if (grp_id == 0u && trace) trace[1308u + b] = v983_clock(); // t_g1_start[b]
    }
    weave_4wave_barrier_65440(arena);
    if (__builtin_amdgcn_readfirstlane(ep16_load_relaxed((unsigned*)e.error))) { FI_CLOSE(); return; }  // opus55 E5

#if V44_CKA
    if (!v44_cka_gather(env, o, x, arena, grp_id, n_comp_groups, b)) { FI_CLOSE(); return; }  // opus55 v44 G2
#endif
    // GEMM1(MB b): BK=128 double-buffered + IN-PLACE W1 prefetch + cached A + ZERO wait!
    weave_v4_gemm1_direct(env, o, arena, false, trace + 1340u + b, true FI_CTX_ARG);
    if (ltid == 0u && grp_id == 0u && trace) trace[1312u + b] = v983_clock(); // t_g1_end[b]
    if (__builtin_amdgcn_readfirstlane(ep16_load_relaxed((unsigned*)e.error))) { FI_CLOSE(); return; }  // opus55 E5

    // Ping-pong safety check before GEMM2(MB b) writes gemm2_output + g2_base
    if (b >= 1u) {
      const unsigned half_rr = env.ck.ck_route_rows >> 1u;
      const bool pingpong = (M >= 2u && g2_base == ((b & 1u) * half_rr) && half_rr > 0u); // V42 D4
      const bool need_wait = pingpong ? (b >= 2u) : true;
      const unsigned prev_b = pingpong ? (b - 2u) : (b - 1u);
      if (need_wait && ltid == 0u) {
        while (v42_ld_agent_rlx(pf + 96u + prev_b) == 0u) {
          if (ep16_load_relaxed((unsigned*)e.error) || (__builtin_amdgcn_s_memrealtime() - start_c > x.timeout_ticks)) {
            v983_error((unsigned*)e.error, 917u);
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        } v42_acq_agent();
      }
      weave_4wave_barrier_65440(arena);
      if (__builtin_amdgcn_readfirstlane(ep16_load_relaxed((unsigned*)e.error))) { FI_CLOSE(); return; }  // opus55 E5
    }

    // GEMM2(MB b) -> immediately triggers Return Overlap waves via bucket_done!
    weave_v4_gemm2_direct(env, o, arena, false, (unsigned*)x.compute_done, trace + 1350u + b, g2_base FI_CTX_ARG);
    if (ltid == 0u && grp_id == 0u && trace) trace[1316u + b] = v983_clock(); // t_g2_end[b]

    // Grid barrier across n_comp_groups before starting MB b+1 RoutePlan
    if (ltid == 0u) {
      __threadfence();
      const unsigned done = __hip_atomic_fetch_add(pf + 64u + b, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
      if (done == n_comp_groups) {
        if (b + 1u == M && trace) {
          trace[16u + cta * 16u + 9u] = trace[1312u + b];
          trace[16u + cta * 16u + 10u] = v983_clock();
        }
        v41_st_agent_u32(pf + 48u + b, 1u);
      }
      while (v42_ld_agent_rlx(pf + 48u + b) == 0u) {
        if (ep16_load_relaxed((unsigned*)e.error)) break;
        asm volatile("s_nop 0\n" ::: "memory");
      } v42_acq_agent();
    }
    weave_4wave_barrier_65440(arena);
    if (__builtin_amdgcn_readfirstlane(ep16_load_relaxed((unsigned*)e.error))) { FI_CLOSE(); return; }  // opus55 E5
  }

  if (ltid == 0u && trace) {
    trace[16u + cta * 16u + 9u] = trace[1312u + M - 1u];
    trace[16u + cta * 16u + 10u] = v983_clock();
  }

  // Reclaim ALL 592 Compute waves into final Return/NodeReduce/HomeCombine drain (632 waves total!)
  const unsigned tail_worker = grp_id * 4u + (wave & 3u);
#if V44_COMBINE
  V45_COMBINE_LOOP(env, o, x, 44u + tail_worker, true, true, true FI_CTX_ARG);  // opus55 v44 C1 / v45 M1: all 592 waves
#else
  v41_combine_loop(env, o, x, tail_worker, n_comp_groups * 4u, false, true, true, true FI_CTX_ARG);
#endif
  if (ltid == 0u && trace) trace[16u + cta * 16u + 11u] = v983_clock();
  FI_CLOSE();
#endif  // V46_SP
}
