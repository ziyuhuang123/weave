#pragma once
// ============================================================================
// ep16_turbo_layout.hpp -- symmetric GDA arena layout, rank-local scratch
// layout, peer-table construction and TxJob/RxJob plan generation for the
// EP16 512-thread 4+4 spatial-overlap kernel.
//
// PURE C++ (no HIP).  Everything in this header is deterministic integer
// arithmetic so the identical code path can be exercised by a CPU-only
// g++ unit test and by the hipcc-built shared library.
//
// ---------------------------------------------------------------------------
// WHY A SINGLE SYMMETRIC ARENA
// ---------------------------------------------------------------------------
// mn_gda_mesh::ep16_fic2_post() validates, for every posted WR:
//     src in [rt.arena_base, rt.arena_base + rt.arena_bytes)
//     dst in [rt.peer_arena[peer], rt.peer_arena[peer] + rt.arena_bytes)
// so every RDMA source and destination must live inside the one registered
// arena, and every rank must use the SAME byte offset for the same logical
// buffer.  The same single hipMalloc block is also hipIpcGetMemHandle()-
// exported to the 7 intra-node peers, so xGMI peer tables are simply
// "peer arena base + the same offset".
//
// Address conventions used throughout:
//   arena_va[d] -- the address THIS rank uses to reach rank d's arena.
//                  d == rank            -> own arena VA
//                  d on my node         -> my hipIpcOpenMemHandle() mapping
//                  d on the other node  -> rank d's real VA (RDMA dst only)
//   For a destination on the other node the xGMI-visible peer tables must NOT
//   name the remote VA; they name LOCAL TX STAGING instead (EP16 ABI rule).
// ============================================================================
#include "ep16_turbo_abi.hpp"

#define EP16T_ARENA_ALIGN  4096ull
#define EP16T_SCRATCH_ALIGN 256ull
#define EP16T_DUMMY_BYTES  256ull

// ---------------------------------------------------------------------------
// Shape -- everything the layout and the plan depend on.  96 bytes, POD.
// ---------------------------------------------------------------------------
struct Ep16TurboShape {
  uint32_t rank;                  //  0
  uint32_t world;                 //  4  must be 16
  uint32_t ranks_per_node;        //  8  must be 8
  uint32_t channels;              // 12  must be 4
  uint32_t hidden;                // 16  H, multiple of 128
  uint32_t intermediate;          // 20  I, multiple of 128
  uint32_t topk;                  // 24  1..12
  uint32_t n_experts;             // 28  multiple of 16, n_experts/16 <= 64
  uint32_t capacity;              // 32  bucket slots per destination
  uint32_t t_local;               // 36  this rank's token rows
  uint32_t node_tokens;           // 40  GLOBALLY UNIFORM mode-2 plane capacity
  uint32_t mode;                  // 44  WEAVE_SCATTER_MODE: 0,1,2
  uint32_t route_rows;            // 48  ck_route_rows
  uint32_t slot_chunk;            // 52  must be 1 (see note in builder)
  uint32_t meta_chunk_slots;      // 56  slots per metadata WR descriptor
  uint32_t flags;                 // 60  bit0 include_self, bit1 FP16 weight bits
  uint32_t enable_dispatch_time;  // 64
  uint32_t enable_scatter_time;   // 68
  uint32_t reserved[6];           // 72..96
};
static_assert(sizeof(Ep16TurboShape) == 96, "Ep16TurboShape ABI");

// ---------------------------------------------------------------------------
// Arena regions (RDMA-visible AND xGMI/IPC-visible).  Identical offsets on
// every rank of the job.
// ---------------------------------------------------------------------------
enum Ep16ArenaRegion {
  EP16_AR_TX_FLAGS = 0,            // mn_gda_mesh per-QP 8B flag cells
  EP16_AR_RECV_META,               // [16][META_HEADER + cap*META_STRIDE]
  EP16_AR_RECV_PAYLOAD,            // FP8  [16*cap][H]
  EP16_AR_RECV_SCALE,              // F32  [16*cap][H/128]
  EP16_AR_RECV_PROGRESS,           // V809SourceProgress[16]
  EP16_AR_RECV_PAYLOAD_READY,      // u32  [16*cap]
  EP16_AR_RETURN_RECV_PAYLOAD,     // BF16 [16*cap][H]
  EP16_AR_RETURN_READY,            // u32  [16*cap]
  EP16_AR_NODE_PARTIAL,            // F32  [8][NT][H]   (mode 2)
  EP16_AR_NODE_ARRIVAL,            // u32  [NT]         (mode 2)
  EP16_AR_NODE_OUTPUT,             // BF16 [NT][H]      (mode 2, RDMA source)
  EP16_AR_NODE_RECV,               // BF16 [NT][H]      (mode 2, RDMA dest)
  EP16_AR_NIC_META_FLAG,           // u64  [16]
  EP16_AR_NIC_PAYLOAD_FLAG,        // u64  [16*cap]
  EP16_AR_NIC_RETURN_FLAG,         // u64  [16*cap]     (mode 0/1)
  EP16_AR_NIC_NODE_FLAG,           // u64  [NT]         (mode 2)
  EP16_AR_TX_META_STAGE,           // [8][meta region]
  EP16_AR_TX_PAYLOAD_STAGE,        // FP8  [8*cap][H]
  EP16_AR_TX_SCALE_STAGE,          // F32  [8*cap][H/128]
  EP16_AR_TX_PROGRESS_STAGE,       // V809SourceProgress[8] (turbo meta_ready gate)
  EP16_AR_TX_PAYLOAD_READY_STAGE,  // u32  [8*cap]      (turbo payload gate)
  EP16_AR_TX_RETURN_STAGE,         // BF16 [8*cap][H]   (mode 0/1)
  EP16_AR_TX_HOME_ARRIVAL_STAGE,   // u32  [8*cap]      (never written, must be valid)
  EP16_AR_COUNT
};

// ---------------------------------------------------------------------------
// Rank-local scratch regions (never RDMA, never IPC).
// ---------------------------------------------------------------------------
enum Ep16ScratchRegion {
  EP16_SC_LOCAL_META = 0,          // [16][meta region] producer-side staging
  EP16_SC_PEER_TABLES,             // 10 pointer tables, see EP16_PT_* below
  EP16_SC_BUCKET_COUNT,            // u32 [16]
  EP16_SC_BUCKET_CURSOR,           // u32 [16]
  EP16_SC_BUCKET_PREFIX,           // u32 [17]
  EP16_SC_CHUNK_PREFIX,            // u32 [17]   (unused by turbo)
  EP16_SC_TOTAL_CHUNKS,            // u32 [1]    (unused by turbo)
  EP16_SC_ERROR,                   // u32 [16]   sticky first error
  EP16_SC_SYNC_STATE,              // u32 [64]   (unused by turbo)
  EP16_SC_EXPERT_COUNTS,           // u32 [LE]
  EP16_SC_EXPERT_OFFSETS,          // u32 [LE+1] (unused by turbo)
  EP16_SC_EXPERT_CURSORS,          // u32 [LE]   (unused by turbo)
  EP16_SC_SORTED_IDS,              // u32 [RR]   stage-1 phys_slot gather list
  EP16_SC_SORTED_WEIGHTS,          // f32 [RR]   (unused by turbo)
  EP16_SC_SORTED_EXPERT_IDS,       // u32 [ceil(RR/32)] (unused by turbo)
  EP16_SC_NUM_VALID_IDS,           // u32 [2]
  EP16_SC_SORT_STATS,              // u32 [4]
  EP16_SC_ROUTE_HEADER,            // (unused by turbo)
  EP16_SC_SOURCE_BUCKET_ROWS,      // u32 [16]
  EP16_SC_PADDED_BASES,            // (unused)
  EP16_SC_BLOCK_PREFIX,            // (unused)
  EP16_SC_INVERSE_GENERATION,      // (unused)
  EP16_SC_INVERSE_ROW,             // (unused)
  EP16_SC_ROUTE_ENTRIES,           // (unused)
  EP16_SC_ROUTE_GENERATION,        // u32 [1]
  EP16_SC_INGRESS_HEADER,          // (unused)
  EP16_SC_INGRESS_SOURCE_STATES,   // (unused)
  EP16_SC_INGRESS_BLOCK_STATES,    // (unused)
  EP16_SC_CK_A,                    // WeaveTilePlanTable + WeaveMTileDesc[RR/16]
  EP16_SC_CK_A_SCALE,              // (unused)
  EP16_SC_CK_STAGE1_OUTPUT,        // (unused)
  EP16_SC_CK_QUANT_OUTPUT,         // FP8 [RR][I]
  EP16_SC_CK_QUANT_SCALE,          // f32 [RR][I/128]
  EP16_SC_CK_M64_SORTED_IDS,       // i32 [RR]
  EP16_SC_CK_M64_EXPERT_IDS,       // i32 [ceil(RR/64)] (unused)
  EP16_SC_CK_M64_OFFSETS,          // u32 [LE+1]
  EP16_SC_CK_M64_CURSORS,          // u32 [LE+1] (unused)
  EP16_SC_CK_MAX_TOKEN_ID,         // i32 [LE+2]
  EP16_SC_CK_DIAGNOSTICS,          // u64 [8]
  EP16_SC_PLAN_FLAGS,              // u32 [8]
  EP16_SC_A_TILE_READY,            // (unused)
  EP16_SC_A_TILE_ROWS_DONE,        // (unused)
  EP16_SC_STAGE1_TILE_DONE,        // u32 [RR/16]
  EP16_SC_ROW_N_DONE,              // u32 [RR/16]
  EP16_SC_BUCKET_DONE,             // u32 [16*cap]
  EP16_SC_BUCKET_ROUTE_ROWS,       // u32 [16*cap*topk]
  EP16_SC_BUCKET_SENT,             // u32 [16*cap]
  EP16_SC_ROW_ROUTE_ID,            // u32 [RR]
  EP16_SC_STAGE2_WEIGHTS,          // f32 [RR]
  EP16_SC_GEMM2_OUTPUT,            // BF16 [RR][H]
  EP16_SC_TRACE,                   // u64 [1536]
  EP16_SC_BYTE_COUNTERS,           // u64 [8]
  EP16_SC_SCATTER_TIME,            // u64 [16*cap] or dummy
  EP16_SC_DISPATCH_TIME,           // u64 [16*cap] or dummy
  EP16_SC_HOME_DONE,               // u32 [t_local]
  EP16_SC_HOME_SLOTS,              // u32 [t_local][16]
  EP16_SC_HOME_EXPECTED,           // u32 [t_local]
  EP16_SC_HOME_ARRIVAL,            // u32 [t_local] (unused by turbo)
  EP16_SC_POST_CONTROL,            // V1988PostControl + queue (unused by turbo)
  EP16_SC_PEER_POST_STATE,         // u64 [2][16] (unused by turbo)
  EP16_SC_NODE_EXPECTED,           // u32 [NT]  host-filled
  EP16_SC_NODE_RANK_MASK,          // u32 [NT]  host-filled
  EP16_SC_NODE_READY,              // u32 [NT]
  EP16_SC_NODE_RECV_READY,         // u32 [NT]
  EP16_SC_NODE_SEND_EXPECT,        // u32 [NT]  host-filled: token has cross routes
  EP16_SC_ONES_WORD,               // u32 [16]  host-filled with 1
  EP16_SC_COMPUTE_DONE,            // u32 [16]
  EP16_SC_GDA_STATE,               // u64 [32][4]
  EP16_SC_TX_JOBS,                 // Ep16TxJob[sum tx_count]
  EP16_SC_RX_JOBS,                 // Ep16RxJob[rx_count]
  EP16_SC_ENVELOPE,                // V983Envelope
  EP16_SC_CONTEXT,                 // Ep16TurboContext
  EP16_SC_COUNT
};

// Sub-offsets inside EP16_SC_PEER_TABLES (bytes).
#define EP16_PT_LOCAL_META            0ull
#define EP16_PT_PEER_META             128ull
#define EP16_PT_PEER_PAYLOAD          256ull
#define EP16_PT_PEER_SCALE            384ull
#define EP16_PT_PEER_PROGRESS         512ull
#define EP16_PT_PEER_PAYLOAD_READY    640ull
#define EP16_PT_PEER_RETURN_PAYLOAD   768ull
#define EP16_PT_PEER_HOME_ARRIVAL     896ull
#define EP16_PT_NODE_PARTIAL          1024ull
#define EP16_PT_NODE_ARRIVAL          1088ull
#define EP16_PT_BYTES                 1152ull

struct Ep16ArenaLayout {
  uint64_t off[EP16_AR_COUNT];
  uint64_t size[EP16_AR_COUNT];
  uint32_t zero_on_reset[EP16_AR_COUNT];
  uint64_t total_bytes;
  uint64_t meta_region_bytes;
  uint64_t payload_row_bytes;
  uint64_t scale_row_bytes;
  uint64_t hidden_row_bytes;
};

struct Ep16ScratchLayout {
  uint64_t off[EP16_SC_COUNT];
  uint64_t size[EP16_SC_COUNT];
  uint32_t zero_on_reset[EP16_SC_COUNT];
  uint64_t total_bytes;
  uint32_t tx_count[32];
  uint32_t tx_offset[32];          // job index of QP q inside EP16_SC_TX_JOBS
  uint32_t tx_total;
  uint32_t rx_total;
};

// Buffers owned by the caller (torch tensors / harness allocations).
struct Ep16TurboExternal {
  uint64_t x;                 // FP8  [t_local][H]        (pre-quantized)
  uint64_t x_scale;           // f32  [t_local][H/128]
  uint64_t topk_ids;          // i32  [t_local][topk]
  uint64_t topk_weights;      // f32  [t_local][topk]
  uint64_t expert_map;        // u32  [n_experts] = (dst<<16)|lid
  uint64_t ck_b;              // FP8  [LE][2*I][H]  gate then up
  uint64_t ck_b_scale;        // f32  [LE][2*(I/128)][H/128]
  uint64_t ck_w2;             // FP8  [LE][H][I]
  uint64_t ck_w2_scale;       // f32  [LE][H/128][I/128]
  uint64_t final_output;      // BF16 [t_local][H]
  uint64_t reserved[6];
};
static_assert(sizeof(Ep16TurboExternal) == 128, "Ep16TurboExternal ABI");

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
static inline uint64_t ep16t_align_up(uint64_t v, uint64_t a) {
  return (v + a - 1ull) / a * a;
}
static inline uint64_t ep16t_div_up(uint64_t v, uint64_t d) {
  return (v + d - 1ull) / d;
}
// number of i in [0, n) with (i & 3) == c
static inline uint32_t ep16t_stride4_count(uint32_t n, uint32_t c) {
  return (n > c) ? ((n - c - 1u) / 4u + 1u) : 0u;
}
static inline uint32_t ep16t_local_experts(const Ep16TurboShape& s) {
  return s.n_experts / s.world;
}
static inline uint32_t ep16t_scale_groups(const Ep16TurboShape& s) {
  return s.hidden / 128u;
}
static inline uint32_t ep16t_other_node_base(const Ep16TurboShape& s) {
  return ((s.rank >> 3u) ^ 1u) * 8u;
}
static inline uint32_t ep16t_my_node_base(const Ep16TurboShape& s) {
  return (s.rank >> 3u) * 8u;
}

// ---------------------------------------------------------------------------
// shape validation.  Mirrors exactly the device-side entry guard that raises
// sticky error 900 in stage_ep16_turbo_overlap.
// ---------------------------------------------------------------------------
// returns 0 on success, otherwise a nonzero code; *why receives a static string
static inline int ep16_turbo_validate_shape(const Ep16TurboShape& s, const char** why) {
  const char* msg = nullptr;
  int rc = 0;
  #define EP16T_REQ(cond, code, text) do { if (!rc && !(cond)) { rc = (code); msg = (text); } } while (0)
  EP16T_REQ(s.world == 16u, 1, "world must be 16");
  EP16T_REQ(s.ranks_per_node == 8u, 2, "ranks_per_node must be 8");
  EP16T_REQ(s.channels == 4u, 3, "channels must be 4");
  EP16T_REQ(s.rank < 16u, 4, "rank must be < 16");
  EP16T_REQ(s.capacity != 0u, 5, "capacity must be nonzero");
  EP16T_REQ((uint64_t)s.world * (uint64_t)(s.capacity + 1u) < 0xffffffull, 6,
            "world*(capacity+1) must stay below the 24-bit route payload");
  EP16T_REQ(s.topk != 0u && s.topk <= 12u, 7, "topk must be in 1..12");
  EP16T_REQ(s.n_experts != 0u, 8, "n_experts must be nonzero");
  EP16T_REQ(s.n_experts % s.world == 0u, 9, "n_experts must divide by world");
  EP16T_REQ(s.n_experts / s.world <= 64u, 10, "local experts must be <= 64");
  EP16T_REQ(s.hidden >= 128u && s.hidden % 128u == 0u, 11, "hidden must be a multiple of 128");
  EP16T_REQ(s.intermediate >= 128u && s.intermediate % 128u == 0u, 12,
            "intermediate must be a multiple of 128");
  EP16T_REQ(s.route_rows < 0xffffffu, 13, "ck_route_rows must stay below 0xffffff");
  EP16T_REQ(s.route_rows >= 16u, 14, "ck_route_rows must be at least 16");
  EP16T_REQ(s.mode <= 2u, 15, "mode must be 0, 1 or 2");
  EP16T_REQ(s.slot_chunk >= 1u && s.slot_chunk <= 64u, 16,
            "slot_chunk must be in 1..64");
  EP16T_REQ((s.slot_chunk & (s.slot_chunk - 1u)) == 0u, 20,
            "slot_chunk must be a power of two (1,2,4,8,16,32,64)");
  EP16T_REQ(s.meta_chunk_slots != 0u, 17, "meta_chunk_slots must be nonzero");
  EP16T_REQ(s.t_local != 0u, 18, "t_local must be nonzero");
  EP16T_REQ(s.mode != 2u || s.node_tokens >= s.t_local, 19,
            "node_tokens must be >= every source token count");
  #undef EP16T_REQ
  if (why) *why = msg;
  return rc;
}

static inline uint32_t ep16t_slot_chunk(const Ep16TurboShape& s) {
  uint32_t k = s.slot_chunk ? s.slot_chunk : 1u;
  if (k > s.capacity) k = s.capacity;
  return k ? k : 1u;
}
static inline uint32_t ep16t_nchunk(const Ep16TurboShape& s) {
  const uint32_t k = ep16t_slot_chunk(s);
  return (uint32_t)ep16t_div_up(s.capacity, k);
}
static inline uint32_t ep16t_chunks_on_qp(const Ep16TurboShape& s, uint32_t c) {
  const uint32_t n = ep16t_nchunk(s);
  return (n > c) ? ((n - c + 3u) / 4u) : 0u;
}
// Mode-2 node-return fanout rides the same chunk granularity as the dispatch.
static inline uint32_t ep16t_ret_chunk(const Ep16TurboShape& s) {
  const uint32_t k = ep16t_slot_chunk(s);
  return (k > 16u) ? 16u : k;
}
static inline uint32_t ep16t_nnodechunk(const Ep16TurboShape& s) {
  const uint32_t k = ep16t_ret_chunk(s);
  return (uint32_t)ep16t_div_up(s.node_tokens, k);
}
static inline uint32_t ep16t_nodechunks_on_qp(const Ep16TurboShape& s, uint32_t c) {
  const uint32_t n = ep16t_nnodechunk(s);
  return (n > c) ? ((n - c + 3u) / 4u) : 0u;
}
// Exact number of RDMA_WRITE descriptors for a contiguous run, honouring both
// the 128 KiB slice and the 64 MiB ibv_reg_mr chunk boundary.  Byte-for-byte
// identical to ep16_wr_pieces() on the device posting lane.
static inline uint64_t ep16t_wr_pieces(uint64_t src, uint64_t dst, uint64_t nbytes) {
  uint64_t off = 0, n = 0;
  while (off < nbytes) {
    const uint64_t sr = 0x4000000ull - ((src + off) & 0x3ffffffull);
    const uint64_t dr = 0x4000000ull - ((dst + off) & 0x3ffffffull);
    uint64_t take = 128ull * 1024ull, rem = nbytes - off;
    if (take > rem) take = rem;
    if (take > sr) take = sr;
    if (take > dr) take = dr;
    if (!take) return 0xffffffffull;
    off += take; ++n;
  }
  return n;
}
// ---------------------------------------------------------------------------
// exact per-QP TxJob counts.  Deterministic from the shape alone, so the
// layout and the builder agree without a counting pass.
// ---------------------------------------------------------------------------

#ifdef WEAVE_V41_DEDUP
static inline uint32_t ep16t_v41_kdisp(uint32_t nt) {
  if (nt >= 1024u) return 64u;
  if (nt >= 256u)  return 32u;
  if (nt >= 64u)   return 16u;
  uint32_t k = (nt + 3u) >> 2u;
  return k ? k : 1u;
}
static inline uint32_t ep16t_v41_nchunks(const Ep16TurboShape& s) {
  const uint32_t k = ep16t_v41_kdisp(s.node_tokens);
  return (uint32_t)ep16t_div_up(s.node_tokens, k);
}
static inline uint32_t ep16t_v41_chunks_on_qp(const Ep16TurboShape& s, uint32_t c) {
  const uint32_t n = ep16t_v41_nchunks(s);
  return (n > c) ? ((n - c + 3u) / 4u) : 0u;
}
#endif

static inline void ep16_turbo_tx_counts(const Ep16TurboShape& s, uint32_t out[32]) {
#ifdef WEAVE_V41_DEDUP
  if (s.mode == 2u) {
    const uint32_t owner_peer = s.rank & 7u;
    for (uint32_t p = 0; p < 8u; ++p) {
      for (uint32_t c = 0; c < 4u; ++c) {
        if (p == owner_peer) {
          const uint32_t nc = ep16t_v41_chunks_on_qp(s, c);
          out[p * 4u + c] = (c == 0u ? 1u : 0u) + 3u * nc; // 1 meta (c==0) + 2*nc disp + nc ret
        } else {
          out[p * 4u + c] = 0u;
        }
      }
    }
    return;
  }
#endif
  const uint32_t cap = s.capacity;
  const uint32_t nmeta = (uint32_t)ep16t_div_up(cap, s.meta_chunk_slots);
  const uint32_t owner_peer = s.rank & 7u;          // mode 2: the single return peer
  const bool chunked = ep16t_slot_chunk(s) > 1u;
  for (uint32_t p = 0; p < 8u; ++p) {
    for (uint32_t c = 0; c < 4u; ++c) {
      const uint32_t slots = ep16t_stride4_count(cap, c);
      uint32_t n = 0;
      if (c == 0u) n += 2u + nmeta;                 // header + chunks + trailing flag
      if (chunked) {
        n += 2u * ep16t_chunks_on_qp(s, c);         // payload chunk + scale chunk
      } else {
        n += 2u * slots;                            // payload + scale per slot
      }
      if (s.mode == 2u) {
        if (p == owner_peer) {
          n += chunked ? ep16t_nodechunks_on_qp(s, c)
                       : ep16t_stride4_count(s.node_tokens, c);
        }
      } else {
        n += slots;                                 // per-slot RETURN payload
      }
      out[p * 4u + c] = n;
    }
  }
}

static inline uint32_t ep16_turbo_rx_count(const Ep16TurboShape& s) {
#ifdef WEAVE_V41_DEDUP
  if (s.mode == 2u) {
    const uint32_t nch = ep16t_v41_nchunks(s);
    return 1u + 2u * nch; // 1 partner meta + nch partner disp + nch partner ret
  }
#endif
  const uint32_t cap = s.capacity;
  const uint32_t K = ep16t_slot_chunk(s);
  const uint32_t nch = ep16t_nchunk(s);
  uint32_t n = 8u + ((K > 1u) ? 8u * nch : 8u * cap);   // metadata + dispatch payload
  if (s.mode == 2u) n += (K > 1u) ? ep16t_nnodechunk(s) : s.node_tokens;
  else n += 8u * cap;                                  // per-slot RETURN
  return n;
}

// ---------------------------------------------------------------------------
// arena layout
// ---------------------------------------------------------------------------
static inline int ep16_turbo_plan_arena(const Ep16TurboShape& s, Ep16ArenaLayout* out) {
  const char* why = nullptr;
  int rc = ep16_turbo_validate_shape(s, &why);
  if (rc) return rc;

  const uint64_t cap = s.capacity;
  const uint64_t MR = (uint64_t)V809_META_HEADER + cap * (uint64_t)V809_META_STRIDE;
  const uint64_t PB = s.hidden;                      // FP8 row
  const uint64_t SGB = (uint64_t)ep16t_scale_groups(s) * 4ull;
  const uint64_t HB = (uint64_t)s.hidden * 2ull;     // BF16 row
  const uint64_t NT = s.node_tokens;
  const bool m2 = (s.mode == 2u);
  const uint64_t D = EP16T_DUMMY_BYTES;

  uint64_t sz[EP16_AR_COUNT];
  uint32_t zr[EP16_AR_COUNT];
  for (int i = 0; i < EP16_AR_COUNT; ++i) { sz[i] = D; zr[i] = 0u; }

  // mn_gda_mesh indexes tx_flags with the ABSOLUTE peer rank (0..15), four
  // channels, kWindow cells of 8 bytes each.
  sz[EP16_AR_TX_FLAGS] = (uint64_t)MN_GDA_MAX_RANKS * EP16T_CHANNELS * MN_GDA_MESH_WINDOW * 8ull;
  sz[EP16_AR_RECV_META] = 16ull * MR;
  sz[EP16_AR_RECV_PAYLOAD] = 16ull * cap * PB;
  sz[EP16_AR_RECV_SCALE] = 16ull * cap * SGB;
  sz[EP16_AR_RECV_PROGRESS] = 16ull * sizeof(V809SourceProgress);
  // D1: the readiness words are followed by one u64 group bitmap per 64
  // consecutive slots and source, i.e. world * ceil(cap/64) words.
  sz[EP16_AR_RECV_PAYLOAD_READY] = 16ull * cap * 4ull +
      (uint64_t)s.world * ep16t_div_up(cap, 64ull) * 8ull;
  sz[EP16_AR_RETURN_RECV_PAYLOAD] = 16ull * cap * HB;
  sz[EP16_AR_RETURN_READY] = 16ull * cap * 4ull;
  sz[EP16_AR_NODE_PARTIAL] = m2 ? (8ull * NT * (uint64_t)s.hidden * 4ull) : D;
  sz[EP16_AR_NODE_ARRIVAL] = m2 ? (NT * 4ull) : D;
  sz[EP16_AR_NODE_OUTPUT] = m2 ? (NT * HB) : D;
  sz[EP16_AR_NODE_RECV] = m2 ? (NT * HB) : D;
  sz[EP16_AR_NIC_META_FLAG] = 16ull * 8ull;
  sz[EP16_AR_NIC_PAYLOAD_FLAG] = 16ull * cap * 8ull;
  sz[EP16_AR_NIC_RETURN_FLAG] = m2 ? D : (16ull * cap * 8ull);
  sz[EP16_AR_NIC_NODE_FLAG] = m2 ? (NT * 8ull) : D;
  sz[EP16_AR_TX_META_STAGE] = 8ull * MR;
  sz[EP16_AR_TX_PAYLOAD_STAGE] = 8ull * cap * PB;
  sz[EP16_AR_TX_SCALE_STAGE] = 8ull * cap * SGB;
  sz[EP16_AR_TX_PROGRESS_STAGE] = 8ull * sizeof(V809SourceProgress);
  sz[EP16_AR_TX_PAYLOAD_READY_STAGE] = 8ull * cap * 4ull;
  sz[EP16_AR_TX_RETURN_STAGE] = m2 ? D : (8ull * cap * HB);
  sz[EP16_AR_TX_HOME_ARRIVAL_STAGE] = 8ull * cap * 4ull;

  // Readiness/cursor/flag words must start every generation at zero.  Bulk
  // data regions are fully written before any consumer reads them.
  zr[EP16_AR_RECV_PROGRESS] = 1u;
  zr[EP16_AR_RECV_PAYLOAD_READY] = 1u;
  zr[EP16_AR_RETURN_READY] = 1u;
  zr[EP16_AR_NODE_ARRIVAL] = m2 ? 1u : 0u;
  zr[EP16_AR_NIC_META_FLAG] = 1u;
  zr[EP16_AR_NIC_PAYLOAD_FLAG] = 1u;
  zr[EP16_AR_NIC_RETURN_FLAG] = m2 ? 0u : 1u;
  zr[EP16_AR_NIC_NODE_FLAG] = m2 ? 1u : 0u;
  zr[EP16_AR_TX_PROGRESS_STAGE] = 1u;
  zr[EP16_AR_TX_PAYLOAD_READY_STAGE] = 1u;
  zr[EP16_AR_TX_HOME_ARRIVAL_STAGE] = 1u;

  uint64_t cursor = 0;
  for (int i = 0; i < EP16_AR_COUNT; ++i) {
    out->off[i] = cursor;
    out->size[i] = sz[i];
    out->zero_on_reset[i] = zr[i];
    cursor = ep16t_align_up(cursor + sz[i], EP16T_ARENA_ALIGN);
  }
  out->total_bytes = cursor;
  out->meta_region_bytes = MR;
  out->payload_row_bytes = PB;
  out->scale_row_bytes = SGB;
  out->hidden_row_bytes = HB;
  return 0;
}

// ---------------------------------------------------------------------------
// scratch layout
// ---------------------------------------------------------------------------
static inline int ep16_turbo_plan_scratch(const Ep16TurboShape& s, Ep16ScratchLayout* out) {
  const char* why = nullptr;
  int rc = ep16_turbo_validate_shape(s, &why);
  if (rc) return rc;

  const uint64_t cap = s.capacity;
  const uint64_t MR = (uint64_t)V809_META_HEADER + cap * (uint64_t)V809_META_STRIDE;
  const uint64_t LE = ep16t_local_experts(s);
  const uint64_t RR = s.route_rows;
  const uint64_t T = s.t_local;
  const uint64_t NT = s.node_tokens;
  const uint64_t HB = (uint64_t)s.hidden * 2ull;
  const uint64_t D = EP16T_DUMMY_BYTES;
  const bool m2 = (s.mode == 2u);

  uint32_t txc[32];
  ep16_turbo_tx_counts(s, txc);
  uint32_t tx_total = 0;
  for (int q = 0; q < 32; ++q) { out->tx_offset[q] = tx_total; out->tx_count[q] = txc[q]; tx_total += txc[q]; }
  const uint32_t rx_total = ep16_turbo_rx_count(s);
  out->tx_total = tx_total;
  out->rx_total = rx_total;

  uint64_t sz[EP16_SC_COUNT];
  uint32_t zr[EP16_SC_COUNT];
  for (int i = 0; i < EP16_SC_COUNT; ++i) { sz[i] = D; zr[i] = 0u; }

  sz[EP16_SC_LOCAL_META] = 16ull * MR;
  sz[EP16_SC_PEER_TABLES] = EP16_PT_BYTES;
  sz[EP16_SC_BUCKET_COUNT] = 16ull * 4ull;            zr[EP16_SC_BUCKET_COUNT] = 1u;
  sz[EP16_SC_BUCKET_CURSOR] = 16ull * 4ull;           zr[EP16_SC_BUCKET_CURSOR] = 1u;
  sz[EP16_SC_BUCKET_PREFIX] = 17ull * 4ull;           zr[EP16_SC_BUCKET_PREFIX] = 1u;
  sz[EP16_SC_CHUNK_PREFIX] = 17ull * 4ull;            zr[EP16_SC_CHUNK_PREFIX] = 1u;
  sz[EP16_SC_TOTAL_CHUNKS] = D;                       zr[EP16_SC_TOTAL_CHUNKS] = 1u;
  sz[EP16_SC_ERROR] = D;                              zr[EP16_SC_ERROR] = 1u;
  sz[EP16_SC_SYNC_STATE] = D;                         zr[EP16_SC_SYNC_STATE] = 1u;
  sz[EP16_SC_EXPERT_COUNTS] = LE * 4ull;              zr[EP16_SC_EXPERT_COUNTS] = 1u;
  sz[EP16_SC_EXPERT_OFFSETS] = (LE + 1ull) * 4ull;    zr[EP16_SC_EXPERT_OFFSETS] = 1u;
  sz[EP16_SC_EXPERT_CURSORS] = LE * 4ull;             zr[EP16_SC_EXPERT_CURSORS] = 1u;
  sz[EP16_SC_SORTED_IDS] = (RR + 64ull) * 4ull;
  sz[EP16_SC_SORTED_WEIGHTS] = (RR + 64ull) * 4ull;
  sz[EP16_SC_SORTED_EXPERT_IDS] = ep16t_div_up(RR + 64ull, 32ull) * 4ull;
  sz[EP16_SC_NUM_VALID_IDS] = D;                      zr[EP16_SC_NUM_VALID_IDS] = 1u;
  sz[EP16_SC_SORT_STATS] = D;                         zr[EP16_SC_SORT_STATS] = 1u;
  sz[EP16_SC_SOURCE_BUCKET_ROWS] = 16ull * 4ull;      zr[EP16_SC_SOURCE_BUCKET_ROWS] = 1u;
  sz[EP16_SC_CK_A] = ((64ull + 16ull * (RR / 16ull) + 255ull) & ~255ull) + 2ull * 1024ull * 4ull + 4096ull;
#ifdef V44_CKA_SHADOW
  // opus55 v44 G2: s1 sorted ids (kernel ABI union alias of ck_a_scale, MUST stay intact)
  //   + cached row-sorted A shadow = A rows + A scales + identity ids (r88 layout)
  sz[EP16_SC_CK_A_SCALE] = ep16t_align_up((RR + 64ull) * 4ull, 4096ull) +
                           ep16t_align_up(RR * (uint64_t)s.hidden, 4096ull) +
                           ep16t_align_up(RR * (uint64_t)(s.hidden / 128u) * 4ull, 4096ull) + (RR + 64ull) * 4ull;
#elif defined(WEAVE_V41_DEDUP)
  // V70 private MTYPE=RW cached scratch shadow for pulled RECV_PAYLOAD + RECV_SCALE (+16 zeroed pad rows)
  // Keeps IPC/RDMA arena 100% uncached (zero PCIe/MALL hazard) while GEMM1 hits L1 + 4 MB XCD L2 + 128 MB MALL L3!
  sz[EP16_SC_CK_A_SCALE] = ep16t_align_up((RR + 64ull) * 4ull, 4096ull) +
                           ep16t_align_up((16ull * cap + 16ull) * (uint64_t)s.hidden, 4096ull) +
                           ep16t_align_up((16ull * cap + 16ull) * (uint64_t)(s.hidden / 128u) * 4ull, 4096ull);
#else
  sz[EP16_SC_CK_A_SCALE] = (RR + 64ull) * 4ull;
#endif
  sz[EP16_SC_CK_QUANT_OUTPUT] = (RR + 16ull) * (uint64_t)s.intermediate;
  sz[EP16_SC_CK_QUANT_SCALE] = (RR + 16ull) * (uint64_t)(s.intermediate / 128u) * 4ull;
  sz[EP16_SC_CK_M64_SORTED_IDS] = (RR + 64ull) * 4ull;
  sz[EP16_SC_CK_M64_EXPERT_IDS] = ep16t_div_up(RR + 64ull, 64ull) * 4ull;
  sz[EP16_SC_CK_M64_OFFSETS] = (LE + 1ull) * 4ull;    zr[EP16_SC_CK_M64_OFFSETS] = 1u;
  sz[EP16_SC_CK_M64_CURSORS] = (LE + 1ull) * 4ull;    zr[EP16_SC_CK_M64_CURSORS] = 1u;
  sz[EP16_SC_CK_MAX_TOKEN_ID] = (LE + 2ull) * 4ull;   zr[EP16_SC_CK_MAX_TOKEN_ID] = 1u;
  sz[EP16_SC_CK_DIAGNOSTICS] = D;                     zr[EP16_SC_CK_DIAGNOSTICS] = 1u;
#ifdef WEAVE_V41_DEDUP
  sz[EP16_SC_PLAN_FLAGS] = 4096ull;                   zr[EP16_SC_PLAN_FLAGS] = 1u;
#else
  sz[EP16_SC_PLAN_FLAGS] = D;                         zr[EP16_SC_PLAN_FLAGS] = 1u;
#endif
  sz[EP16_SC_A_TILE_READY] = (RR / 16ull + 64ull) * 4ull;     zr[EP16_SC_A_TILE_READY] = 1u;
  sz[EP16_SC_A_TILE_ROWS_DONE] = (RR / 16ull + 64ull) * 4ull; zr[EP16_SC_A_TILE_ROWS_DONE] = 1u;
  sz[EP16_SC_STAGE1_TILE_DONE] = (RR / 16ull + 64ull) * 4ull; zr[EP16_SC_STAGE1_TILE_DONE] = 1u;
  sz[EP16_SC_ROW_N_DONE] = (RR / 16ull + 64ull) * 4ull;       zr[EP16_SC_ROW_N_DONE] = 1u;
  sz[EP16_SC_BUCKET_DONE] = 16ull * cap * 4ull;       zr[EP16_SC_BUCKET_DONE] = 1u;
  sz[EP16_SC_BUCKET_ROUTE_ROWS] = 16ull * cap * (uint64_t)s.topk * 4ull;
  sz[EP16_SC_BUCKET_SENT] = 16ull * cap * 4ull;       zr[EP16_SC_BUCKET_SENT] = 1u;
  sz[EP16_SC_ROW_ROUTE_ID] = (RR + 64ull) * 4ull;
  sz[EP16_SC_STAGE2_WEIGHTS] = (RR + 64ull) * 4ull;
  sz[EP16_SC_GEMM2_OUTPUT] = (RR + 16ull) * HB;
  sz[EP16_SC_TRACE] = 4096ull * 8ull;                 zr[EP16_SC_TRACE] = 1u;
  sz[EP16_SC_BYTE_COUNTERS] = D;                      zr[EP16_SC_BYTE_COUNTERS] = 1u;
  sz[EP16_SC_SCATTER_TIME] = s.enable_scatter_time ? (16ull * cap * 8ull) : D;
  sz[EP16_SC_DISPATCH_TIME] = s.enable_dispatch_time ? (16ull * cap * 8ull) : D;
  // D6: [0,T) completion words + [T,2T) per-token column-slice bitmasks.
  sz[EP16_SC_HOME_DONE] = 2ull * T * 4ull;            zr[EP16_SC_HOME_DONE] = 1u;
#ifdef WEAVE_V41_DEDUP
  sz[EP16_SC_HOME_SLOTS] = (NT > T ? NT : T) * 16ull * 4ull;
#else
  sz[EP16_SC_HOME_SLOTS] = T * 16ull * 4ull;
#endif
  sz[EP16_SC_HOME_EXPECTED] = T * 4ull;               zr[EP16_SC_HOME_EXPECTED] = 1u;
  sz[EP16_SC_HOME_ARRIVAL] = T * 4ull;                zr[EP16_SC_HOME_ARRIVAL] = 1u;
  sz[EP16_SC_POST_CONTROL] = (uint64_t)V1988_POST_HEADER_BYTES + 4096ull;
  zr[EP16_SC_POST_CONTROL] = 1u;
  sz[EP16_SC_PEER_POST_STATE] = 2ull * 16ull * 8ull;  zr[EP16_SC_PEER_POST_STATE] = 1u;
  sz[EP16_SC_NODE_EXPECTED] = m2 ? (NT * 4ull) : D;   // host-filled, never auto-zeroed
  sz[EP16_SC_NODE_RANK_MASK] = m2 ? (NT * 4ull) : D;  // host-filled
  // D6: [0,NT) completion words + [NT,2NT) per-token column-slice bitmasks.
  sz[EP16_SC_NODE_READY] = m2 ? (NT * 8ull) : D;      zr[EP16_SC_NODE_READY] = m2 ? 1u : 0u;
  sz[EP16_SC_NODE_RECV_READY] = m2 ? (NT * 4ull) : D; zr[EP16_SC_NODE_RECV_READY] = m2 ? 1u : 0u;
  sz[EP16_SC_NODE_SEND_EXPECT] = m2 ? (NT * 4ull) : D;  // host-filled
  sz[EP16_SC_ONES_WORD] = D;                          // host-filled with 1
  sz[EP16_SC_COMPUTE_DONE] = D;                       zr[EP16_SC_COMPUTE_DONE] = 1u;
  sz[EP16_SC_GDA_STATE] = 32ull * 4ull * 8ull;        zr[EP16_SC_GDA_STATE] = 1u;
  sz[EP16_SC_TX_JOBS] = (uint64_t)tx_total * sizeof(Ep16TxJob);
  sz[EP16_SC_RX_JOBS] = (uint64_t)rx_total * sizeof(Ep16RxJob);
  sz[EP16_SC_ENVELOPE] = sizeof(V983Envelope);
  sz[EP16_SC_CONTEXT] = sizeof(Ep16TurboContext);

  uint64_t cursor = 0;
  for (int i = 0; i < EP16_SC_COUNT; ++i) {
    out->off[i] = cursor;
    out->size[i] = sz[i];
    out->zero_on_reset[i] = zr[i];
    cursor = ep16t_align_up(cursor + sz[i], EP16T_SCRATCH_ALIGN);
  }
  out->total_bytes = cursor;
  return 0;
}

// ---------------------------------------------------------------------------
// address helpers
// ---------------------------------------------------------------------------
struct Ep16TurboBases {
  uint64_t arena;                 // my own arena VA
  uint64_t scratch;               // my own scratch VA
  uint64_t arena_va[16];          // address I use to reach rank d's arena
};

static inline uint64_t ep16t_a(const Ep16TurboBases& b, const Ep16ArenaLayout& L, int r) {
  return b.arena + L.off[r];
}
static inline uint64_t ep16t_s(const Ep16TurboBases& b, const Ep16ScratchLayout& L, int r) {
  return b.scratch + L.off[r];
}
static inline uint64_t ep16t_peer_a(const Ep16TurboBases& b, const Ep16ArenaLayout& L,
                                    uint32_t d, int r) {
  return b.arena_va[d] + L.off[r];
}

// ---------------------------------------------------------------------------
// Peer tables.  Writes EP16_PT_BYTES into `blob`, ready for one H2D copy into
// EP16_SC_PEER_TABLES.
//
//   d on my node  -> arena_va[d] + region offset + my-rank slice
//   d remote      -> LOCAL TX staging slice for peer index (d & 7)
// ---------------------------------------------------------------------------
static inline int ep16_turbo_build_peer_tables(const Ep16TurboShape& s,
                                               const Ep16ArenaLayout& L,
                                               const Ep16TurboBases& b,
                                               uint64_t* blob /* EP16_PT_BYTES/8 u64 */) {
  const uint64_t cap = s.capacity;
  const uint64_t MR = L.meta_region_bytes;
  const uint64_t PB = L.payload_row_bytes;
  const uint64_t SGB = L.scale_row_bytes;
  const uint64_t HB = L.hidden_row_bytes;
  const uint64_t PROG = sizeof(V809SourceProgress);
  const uint32_t m = s.rank;
  const uint32_t my_node = m >> 3u;

  uint64_t* peer_meta = blob + EP16_PT_PEER_META / 8;
  uint64_t* peer_payload = blob + EP16_PT_PEER_PAYLOAD / 8;
  uint64_t* peer_scale = blob + EP16_PT_PEER_SCALE / 8;
  uint64_t* peer_progress = blob + EP16_PT_PEER_PROGRESS / 8;
  uint64_t* peer_ready = blob + EP16_PT_PEER_PAYLOAD_READY / 8;
  uint64_t* peer_return = blob + EP16_PT_PEER_RETURN_PAYLOAD / 8;
  uint64_t* peer_home = blob + EP16_PT_PEER_HOME_ARRIVAL / 8;
  uint64_t* node_partial_tbl = blob + EP16_PT_NODE_PARTIAL / 8;
  uint64_t* node_arrival_tbl = blob + EP16_PT_NODE_ARRIVAL / 8;

  // local_meta[] is rank-local staging and needs the scratch layout; it is
  // filled by ep16_turbo_fill_local_meta_table() into the same blob.
#ifdef WEAVE_V41_DEDUP
  if (s.mode == 2u) {
    const uint32_t node_base = my_node * 8u;
    for (uint32_t src = 0; src < 16u; ++src) {
      const uint32_t h = node_base + (src & 7u);
      if (!b.arena_va[h]) return 100 + (int)h;
      peer_meta[src] = ep16t_peer_a(b, L, h, EP16_AR_RECV_META) + (uint64_t)src * MR;
      peer_payload[src] = ep16t_peer_a(b, L, h, EP16_AR_RECV_PAYLOAD) + (uint64_t)src * cap * PB;
      peer_scale[src] = ep16t_peer_a(b, L, h, EP16_AR_RECV_SCALE) + (uint64_t)src * cap * SGB;
      peer_progress[src] = ep16t_peer_a(b, L, h, EP16_AR_RECV_PROGRESS) + (uint64_t)src * PROG;
      peer_ready[src] = ep16t_peer_a(b, L, h, EP16_AR_RECV_PAYLOAD_READY) + (uint64_t)src * cap * 4ull;
      const bool same_node = ((src >> 3u) == my_node);
      if (same_node) {
        peer_return[src] = ep16t_peer_a(b, L, src, EP16_AR_RETURN_RECV_PAYLOAD) + (uint64_t)m * cap * HB;
        peer_home[src] = ep16t_peer_a(b, L, src, EP16_AR_RETURN_READY) + (uint64_t)m * cap * 4ull;
      } else {
        peer_return[src] = ep16t_a(b, L, EP16_AR_TX_RETURN_STAGE);
        peer_home[src] = ep16t_a(b, L, EP16_AR_TX_HOME_ARRIVAL_STAGE) + (uint64_t)(src & 7u) * cap * 4ull;
      }
    }
    for (uint32_t j = 0; j < 8u; ++j) {
      const uint32_t d = node_base + j;
      node_partial_tbl[j] = ep16t_peer_a(b, L, d, EP16_AR_NODE_PARTIAL);
      node_arrival_tbl[j] = ep16t_peer_a(b, L, d, EP16_AR_NODE_ARRIVAL);
    }
    return 0;
  }
#endif
  for (uint32_t d = 0; d < 16u; ++d) {
    const bool same_node = ((d >> 3u) == my_node);
    const uint32_t p = d & 7u;
    if (same_node) {
      if (!b.arena_va[d]) return 100 + (int)d;      // missing IPC mapping
      peer_meta[d] = ep16t_peer_a(b, L, d, EP16_AR_RECV_META) + (uint64_t)m * MR;
      peer_payload[d] = ep16t_peer_a(b, L, d, EP16_AR_RECV_PAYLOAD) + (uint64_t)m * cap * PB;
      peer_scale[d] = ep16t_peer_a(b, L, d, EP16_AR_RECV_SCALE) + (uint64_t)m * cap * SGB;
      peer_progress[d] = ep16t_peer_a(b, L, d, EP16_AR_RECV_PROGRESS) + (uint64_t)m * PROG;
      peer_ready[d] = ep16t_peer_a(b, L, d, EP16_AR_RECV_PAYLOAD_READY) + (uint64_t)m * cap * 4ull;
      peer_return[d] = ep16t_peer_a(b, L, d, EP16_AR_RETURN_RECV_PAYLOAD) + (uint64_t)m * cap * HB;
      peer_home[d] = ep16t_peer_a(b, L, d, EP16_AR_RETURN_READY) + (uint64_t)m * cap * 4ull;
    } else {
      peer_meta[d] = ep16t_a(b, L, EP16_AR_TX_META_STAGE) + (uint64_t)p * MR;
      peer_payload[d] = ep16t_a(b, L, EP16_AR_TX_PAYLOAD_STAGE) + (uint64_t)p * cap * PB;
      peer_scale[d] = ep16t_a(b, L, EP16_AR_TX_SCALE_STAGE) + (uint64_t)p * cap * SGB;
      peer_progress[d] = ep16t_a(b, L, EP16_AR_TX_PROGRESS_STAGE) + (uint64_t)p * PROG;
      peer_ready[d] = ep16t_a(b, L, EP16_AR_TX_PAYLOAD_READY_STAGE) + (uint64_t)p * cap * 4ull;
      peer_return[d] = (s.mode == 2u)
          ? ep16t_a(b, L, EP16_AR_TX_RETURN_STAGE)
          : (ep16t_a(b, L, EP16_AR_TX_RETURN_STAGE) + (uint64_t)p * cap * HB);
      peer_home[d] = ep16t_a(b, L, EP16_AR_TX_HOME_ARRIVAL_STAGE) + (uint64_t)p * cap * 4ull;
    }
  }
  const uint32_t node_base = my_node * 8u;
  for (uint32_t j = 0; j < 8u; ++j) {
    const uint32_t d = node_base + j;
    node_partial_tbl[j] = (s.mode == 2u) ? ep16t_peer_a(b, L, d, EP16_AR_NODE_PARTIAL) : 0ull;
    node_arrival_tbl[j] = (s.mode == 2u) ? ep16t_peer_a(b, L, d, EP16_AR_NODE_ARRIVAL) : 0ull;
  }
  return 0;
}

// local_meta_table is filled separately because it needs the scratch layout.
static inline void ep16_turbo_fill_local_meta_table(const Ep16ArenaLayout& L,
                                                    const Ep16ScratchLayout& S,
                                                    const Ep16TurboBases& b,
                                                    uint64_t* blob) {
  uint64_t* local_meta = blob + EP16_PT_LOCAL_META / 8;
  const uint64_t base = b.scratch + S.off[EP16_SC_LOCAL_META];
  for (uint32_t d = 0; d < 16u; ++d) local_meta[d] = base + (uint64_t)d * L.meta_region_bytes;
}
