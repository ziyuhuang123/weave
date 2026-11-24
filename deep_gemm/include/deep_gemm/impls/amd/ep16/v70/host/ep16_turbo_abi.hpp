#pragma once
// ============================================================================
// ep16_turbo_abi.hpp -- HOST-SIDE mirror of the EP16 512-thread 4+4 spatial
// overlap kernel ABI (P_ep16_turbo_overlap.co, kernel
// "stage_ep16_turbo_overlap", grid=80, block=512).
//
// Every struct here is a byte-exact mirror of the device-side declaration in
//   delivery/stage_ep16_turbo_overlap.hip   (V964/V968/V983 envelope chain)
//   delivery/include/ep16_transport.hpp     (Ep16TxJob/Ep16RxJob/Ep16TurboContext)
// The static_asserts below are the host-side half of the ABI proof; the Python
// ctypes mirror is checked against ep16_turbo_abi_probe() at runtime.
//
// PURE C++ -- no HIP, no device code.  Compiles with g++ -std=c++17.
// ============================================================================
#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// Compile-time constants inherited from the device build
// ---------------------------------------------------------------------------
#define EP16T_WORLD              16u
#define EP16T_RANKS_PER_NODE     8u
#define EP16T_PEERS              8u
#define EP16T_CHANNELS           4u
#define EP16T_QPS                (EP16T_PEERS * EP16T_CHANNELS)   /* 32 */
#define EP16T_GRID               80u
#define EP16T_BLOCK              512u
#define EP16T_KERNEL_NAME        "stage_ep16_turbo_overlap"

#define V809_MAX_TOPK            12u
#define V809_META_STRIDE         64u
#define V809_META_HEADER         4096u
#define V809_MAX_CHUNKS          512u
#define V809_PROGRESS_MAGIC      0x44385052u   /* D8PR */
#define V1988_POST_HEADER_BYTES  128u

#define MN_GDA_MESH_WINDOW       32u
#define MN_GDA_MESH_SLICE        (128u * 1024u)
#define MN_GDA_MAX_RANKS         16u

// ---------------------------------------------------------------------------
// RACE29 TrackA (ds1017) transport descriptor encoding.  Mirrors the device
// header include/ep16_transport.hpp byte for byte; kept here because the host
// TU never includes the device headers.
// ---------------------------------------------------------------------------
#ifndef EP16_TXJ_CHUNK_GATE
#define EP16_TXJ_CHUNK_GATE 0x80000000u
#endif
#ifndef EP16_TXJ_STATIC_SLOTS
#define EP16_TXJ_STATIC_SLOTS 0x40000000u
#endif
#ifndef EP16_TXJ_PHASE_MASK
#define EP16_TXJ_PHASE_MASK 0x00000003u
#endif
#ifndef EP16_RXJ_TIER_SHIFT
#define EP16_RXJ_TIER_SHIFT 16u
#define EP16_RXJ_TIER_MASK  0xffff0000u
#define EP16_RXJ_COUNT_MASK 0x0000ffffu
enum : uint32_t { EP16_TIER_LOCAL_READY = 0u, EP16_TIER_REMOTE_PRIMARY = 1u, EP16_TIER_LOCAL_COPY = 2u };
#endif

static const uint64_t EP16_TURBO_MAGIC = 0x4550313654555231ull;

// ---------------------------------------------------------------------------
// V964DispatchExt -- 360 bytes, packed
// ---------------------------------------------------------------------------
struct __attribute__((packed)) V964DispatchExt {
  uint64_t x;
  uint64_t x_scale;
  uint64_t topk_ids;
  uint64_t topk_weights;
  uint64_t expert_map;

  uint64_t local_meta_table;
  uint64_t peer_meta_table;
  uint64_t peer_payload_table;
  uint64_t peer_scale_table;
  uint64_t peer_progress_table;

  uint64_t bucket_count;
  uint64_t bucket_cursor;
  uint64_t bucket_prefix;
  uint64_t chunk_prefix;
  uint64_t total_chunks;
  uint64_t error;
  uint64_t sync_state;
  uint64_t reserved0;

  uint32_t t_local;
  uint32_t topk;
  uint32_t capacity;
  uint32_t payload_bytes;
  uint32_t scale_groups;
  uint32_t world;
  uint32_t rank;
  uint32_t flags;
  uint32_t n_experts;
  uint32_t max_chunks;

  uint64_t recv_meta;
  uint64_t expert_counts;
  uint64_t expert_offsets;
  uint64_t expert_cursors;
  uint64_t sorted_ids;
  uint64_t sorted_weights;
  uint64_t sorted_expert_ids;
  uint64_t num_valid_ids;
  uint64_t sort_stats;

  uint64_t route_header;
  uint64_t source_bucket_rows;
  uint64_t padded_bases;
  uint64_t block_prefix;
  uint64_t inverse_generation;
  uint64_t inverse_row;
  uint64_t route_entries;
  uint64_t route_generation;
  uint32_t sort_capacity;
  uint32_t sort_block;

  uint64_t ingress_header;
  uint64_t ingress_source_states;
  uint64_t ingress_block_states;
  uint32_t ingress_max_blocks;
  uint32_t ingress_block_m;
};
static_assert(sizeof(V964DispatchExt) == 360, "V964DispatchExt ABI changed");

// ---------------------------------------------------------------------------
// V968CkExt -- 168 bytes, packed
// ---------------------------------------------------------------------------
struct __attribute__((packed)) V968CkExt {
  uint64_t ck_a;
  uint64_t ck_a_scale;
  uint64_t ck_b;
  uint64_t ck_b_scale;
  uint64_t ck_dense_a;
  uint64_t ck_dense_a_scale;
  uint64_t ck_stage1_output;
  uint64_t ck_quant_output;
  uint64_t ck_quant_scale;
  uint64_t ck_m64_sorted_ids;
  uint64_t ck_m64_expert_ids;
  uint64_t ck_m64_offsets;
  uint64_t ck_m64_cursors;
  uint64_t ck_max_token_id;
  uint64_t ck_diagnostics;

  uint32_t ck_route_rows;
  uint32_t ck_num_tokens;
  uint32_t ck_dense_row_capacity;
  uint32_t ck_hidden;
  uint32_t ck_intermediate;
  uint32_t ck_local_experts;
  uint32_t ck_topk;
  uint32_t ck_groups_per_row;
  uint32_t ck_m64_block_rows;
  uint32_t ck_n_block_cols;
  uint32_t ck_route_sentinel;
  uint32_t ck_flags;
};
static_assert(sizeof(V968CkExt) == 168, "V968CkExt ABI changed");

struct __attribute__((packed)) V968DispatchExt {
  V964DispatchExt base;
  V968CkExt ck;
};
static_assert(sizeof(V968DispatchExt) == 528, "V968DispatchExt ABI changed");

// ---------------------------------------------------------------------------
// V983Ov1Ext -- 272 bytes, packed
// ---------------------------------------------------------------------------
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
  uint64_t post_control;
  uint64_t gemm_width;
  uint64_t peer_post_state_table;
  uint32_t hidden;
  uint32_t intermediate;
  uint32_t n1_blocks;
  uint32_t n2_blocks;
  uint32_t mode;
  uint32_t block_m;
  uint32_t comm_chunk_rows;
  uint32_t role_config;
};
static_assert(sizeof(V983Ov1Ext) == 272, "V983Ov1Ext ABI changed");

struct __attribute__((packed)) V983Envelope {
  V968DispatchExt base;
  V983Ov1Ext ov1;
};
static_assert(sizeof(V983Envelope) == 800, "V1864 envelope ABI changed");

// ---------------------------------------------------------------------------
// Transport job tables + turbo context (ep16_transport.hpp)
// ---------------------------------------------------------------------------
struct alignas(16) Ep16TxJob {
  uint64_t src, dst, bytes, remote_flag;
  uint64_t ready;             // uint32_t*, local release counter gating this job
  uint32_t target, phase;     // phase: 0 DISPATCH, 1 RETURN
  uint64_t valid_count, count_ready;
  uint32_t index, count_target;
};
struct alignas(16) Ep16RxJob {
  uint64_t flag;              // local 8-byte RDMA-written generation word
  uint64_t publish;           // local uint32_t readiness flag
  uint32_t value, reserved;
  uint64_t valid_count, count_ready;
  uint32_t index, count_target;
};
struct alignas(64) Ep16TurboContext {
  uint64_t magic, generation, runtime;
  uint64_t tx_jobs[32]; uint32_t tx_count[32];
  uint64_t rx_jobs; uint32_t rx_count, node_tokens;
  uint64_t return_ready;
  uint64_t node_partial_table;
  uint64_t node_arrival_table;
  uint64_t node_expected;
  uint64_t node_partial;
  uint64_t node_rank_mask;
  uint64_t node_arrival;
  uint64_t node_output;
  uint64_t node_ready;
  uint64_t node_recv;
  uint64_t node_recv_ready;
  uint64_t compute_done;
  uint64_t gda_state;
  uint64_t timeout_ticks;
};
static_assert(sizeof(Ep16TxJob) == 80 && sizeof(Ep16RxJob) == 48 &&
              sizeof(Ep16TurboContext) == 576,
              "EP16 turbo host ABI size mismatch");

// ---------------------------------------------------------------------------
// Auxiliary device-resident structs the host must size correctly
// ---------------------------------------------------------------------------
struct alignas(64) V809Meta {
  uint32_t home;
  uint32_t nexp;
  uint16_t lid[V809_MAX_TOPK];
  uint16_t wbits[V809_MAX_TOPK];
};
static_assert(sizeof(V809Meta) == 64, "V809Meta stride must stay 64");
static_assert(offsetof(V809Meta, lid) == 8u, "V809Meta.lid offset moved");

struct alignas(64) V809SourceProgress {
  uint64_t meta_ready;
  uint64_t payload_rows;
  uint32_t chunk_done[V809_MAX_CHUNKS];
};
static_assert(sizeof(V809SourceProgress) == 2112, "progress header layout changed");
static_assert(offsetof(V809SourceProgress, meta_ready) == 0,
              "turbo meta_ready gate must sit at offset 0");

struct alignas(64) V1988PostControl {
  uint64_t tail;
  uint64_t reserved0[7];
  uint64_t claimed;
  uint64_t reserved1[7];
};
static_assert(sizeof(V1988PostControl) == 128, "V1988 post control header changed");

struct alignas(16) WeaveMTileDesc {
  uint32_t expert_id;
  uint32_t row_start;
  uint16_t bm;
  uint16_t valid_rows;
  uint32_t tile_idx;
};
static_assert(sizeof(WeaveMTileDesc) == 16, "WeaveMTileDesc must be 16 bytes");

struct alignas(64) WeaveTilePlanTable {
  uint32_t total_m_tiles;
  uint32_t num_tiles[5];
  uint32_t bucket_offset[5];
  uint32_t total_padded_rows;
  uint32_t pad_gather_slot;
  uint32_t padding_words[3];
};
static_assert(sizeof(WeaveTilePlanTable) == 64, "WeaveTilePlanTable must be 64 bytes");
