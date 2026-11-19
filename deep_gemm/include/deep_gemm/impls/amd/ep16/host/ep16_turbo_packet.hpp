#pragma once
// ============================================================================
// ep16_turbo_packet.hpp -- V983Envelope + Ep16TurboContext construction.
// PURE C++ (no HIP).
//
// IMPORTANT ABI CONSEQUENCE OF THE DEVICE ENTRY GUARD
// ---------------------------------------------------
// stage_ep16_turbo_overlap rejects the launch (sticky error 900) unless
//     e.payload_bytes == o.hidden  AND  e.scale_groups == o.hidden / 128
// so the only reachable dispatch path is the PRE-QUANTIZED one: `x` must be
// FP8 [t_local][hidden] with `x_scale` f32 [t_local][hidden/128].  The BF16
// register-quantization branch inside v983_dispatch_4wave is unreachable in
// this build and the host must quantize before the launch.
// ============================================================================
#include "ep16_turbo_plan.hpp"

struct Ep16TurboLaunchSpec {
  uint64_t generation;      // strictly increasing, nonzero, identical on all ranks
  uint64_t runtime;         // mn_gda_mesh::Runtime* from ep16_gda_runtime_ptr()
  uint64_t timeout_ticks;   // nonzero s_memrealtime budget (100 MHz)
  uint64_t reserved[5];
};
static_assert(sizeof(Ep16TurboLaunchSpec) == 64, "Ep16TurboLaunchSpec ABI");

static inline int ep16_turbo_build_packet(const Ep16TurboShape& s,
                                          const Ep16ArenaLayout& L,
                                          const Ep16ScratchLayout& S,
                                          const Ep16TurboBases& b,
                                          const Ep16TurboExternal& ext,
                                          const Ep16TurboLaunchSpec& spec,
                                          V983Envelope* env_out,
                                          Ep16TurboContext* ctx_out) {
  const char* why = nullptr;
  int rc = ep16_turbo_validate_shape(s, &why);
  if (rc) return rc;
  if (!spec.generation || !spec.runtime || !spec.timeout_ticks) return -30;
  if (!b.arena || !b.scratch) return -31;
  if (!ext.x || !ext.topk_ids || !ext.topk_weights || !ext.expert_map) return -32;
  if (!ext.ck_b || !ext.ck_b_scale || !ext.ck_w2 || !ext.ck_w2_scale) return -33;
  if (!ext.final_output) return -34;
  if (!ext.x_scale) return -35;

  V983Envelope env{};
  V964DispatchExt& e = env.base.base;
  V968CkExt& ck = env.base.ck;
  V983Ov1Ext& o = env.ov1;

  const uint64_t PT = ep16t_s(b, S, EP16_SC_PEER_TABLES);
  const uint32_t LE = ep16t_local_experts(s);
  const uint32_t SG = ep16t_scale_groups(s);

  // ---- V964 base -----------------------------------------------------------
  e.x = ext.x;
  e.x_scale = ext.x_scale;
  e.topk_ids = ext.topk_ids;
  e.topk_weights = ext.topk_weights;
  e.expert_map = ext.expert_map;

  e.local_meta_table = PT + EP16_PT_LOCAL_META;
  e.peer_meta_table = PT + EP16_PT_PEER_META;
  e.peer_payload_table = PT + EP16_PT_PEER_PAYLOAD;
  e.peer_scale_table = PT + EP16_PT_PEER_SCALE;
  e.peer_progress_table = PT + EP16_PT_PEER_PROGRESS;

  e.bucket_count = ep16t_s(b, S, EP16_SC_BUCKET_COUNT);
  e.bucket_cursor = ep16t_s(b, S, EP16_SC_BUCKET_CURSOR);
  e.bucket_prefix = ep16t_s(b, S, EP16_SC_BUCKET_PREFIX);
  e.chunk_prefix = ep16t_s(b, S, EP16_SC_CHUNK_PREFIX);
  e.total_chunks = ep16t_s(b, S, EP16_SC_TOTAL_CHUNKS);
  e.error = ep16t_s(b, S, EP16_SC_ERROR);
  e.sync_state = ep16t_s(b, S, EP16_SC_SYNC_STATE);
  e.reserved0 = 0;

  e.t_local = s.t_local;
  e.topk = s.topk;
  e.capacity = s.capacity;
  e.payload_bytes = s.hidden;         // forced by the device entry guard
  e.scale_groups = SG;                // forced by the device entry guard
  e.world = s.world;
  e.rank = s.rank;
  e.flags = s.flags;
  e.n_experts = s.n_experts;
  e.max_chunks = V809_MAX_CHUNKS;

  e.recv_meta = ep16t_a(b, L, EP16_AR_RECV_META);
  e.expert_counts = ep16t_s(b, S, EP16_SC_EXPERT_COUNTS);
  e.expert_offsets = ep16t_s(b, S, EP16_SC_EXPERT_OFFSETS);
  e.expert_cursors = ep16t_s(b, S, EP16_SC_EXPERT_CURSORS);
  e.sorted_ids = ep16t_s(b, S, EP16_SC_SORTED_IDS);
  e.sorted_weights = ep16t_s(b, S, EP16_SC_SORTED_WEIGHTS);
  e.sorted_expert_ids = ep16t_s(b, S, EP16_SC_SORTED_EXPERT_IDS);
  e.num_valid_ids = ep16t_s(b, S, EP16_SC_NUM_VALID_IDS);
  e.sort_stats = ep16t_s(b, S, EP16_SC_SORT_STATS);

  e.route_header = ep16t_s(b, S, EP16_SC_ROUTE_HEADER);
  e.source_bucket_rows = ep16t_s(b, S, EP16_SC_SOURCE_BUCKET_ROWS);
  e.padded_bases = ep16t_s(b, S, EP16_SC_PADDED_BASES);
  e.block_prefix = ep16t_s(b, S, EP16_SC_BLOCK_PREFIX);
  e.inverse_generation = ep16t_s(b, S, EP16_SC_INVERSE_GENERATION);
  e.inverse_row = ep16t_s(b, S, EP16_SC_INVERSE_ROW);
  e.route_entries = ep16t_s(b, S, EP16_SC_ROUTE_ENTRIES);
  e.route_generation = ep16t_s(b, S, EP16_SC_ROUTE_GENERATION);
  e.sort_capacity = s.route_rows;
  e.sort_block = 32u;

  e.ingress_header = ep16t_s(b, S, EP16_SC_INGRESS_HEADER);
  e.ingress_source_states = ep16t_s(b, S, EP16_SC_INGRESS_SOURCE_STATES);
  e.ingress_block_states = ep16t_s(b, S, EP16_SC_INGRESS_BLOCK_STATES);
  e.ingress_max_blocks = 1u;
  e.ingress_block_m = 32u;

  // ---- V968 CK suffix ------------------------------------------------------
  ck.ck_a = ep16t_s(b, S, EP16_SC_CK_A);
  ck.ck_a_scale = ep16t_s(b, S, EP16_SC_CK_A_SCALE);
  ck.ck_b = ext.ck_b;
  ck.ck_b_scale = ext.ck_b_scale;
  ck.ck_dense_a = ck.ck_a;
  ck.ck_dense_a_scale = ck.ck_a_scale;
  ck.ck_stage1_output = ep16t_s(b, S, EP16_SC_CK_STAGE1_OUTPUT);
  ck.ck_quant_output = ep16t_s(b, S, EP16_SC_CK_QUANT_OUTPUT);
  ck.ck_quant_scale = ep16t_s(b, S, EP16_SC_CK_QUANT_SCALE);
  ck.ck_m64_sorted_ids = ep16t_s(b, S, EP16_SC_CK_M64_SORTED_IDS);
  ck.ck_m64_expert_ids = ep16t_s(b, S, EP16_SC_CK_M64_EXPERT_IDS);
  ck.ck_m64_offsets = ep16t_s(b, S, EP16_SC_CK_M64_OFFSETS);
  ck.ck_m64_cursors = ep16t_s(b, S, EP16_SC_CK_M64_CURSORS);
  ck.ck_max_token_id = ep16t_s(b, S, EP16_SC_CK_MAX_TOKEN_ID);
  ck.ck_diagnostics = ep16t_s(b, S, EP16_SC_CK_DIAGNOSTICS);
  ck.ck_route_rows = s.route_rows;
  ck.ck_num_tokens = s.world * s.capacity * s.topk;
  ck.ck_dense_row_capacity = s.route_rows;
  ck.ck_hidden = s.hidden;
  ck.ck_intermediate = s.intermediate;
  ck.ck_local_experts = LE;
  ck.ck_topk = 1u;
  ck.ck_groups_per_row = SG;
  ck.ck_m64_block_rows = 64u;
  ck.ck_n_block_cols = 128u;
  ck.ck_route_sentinel = 0u;      // published at runtime by the planner
  ck.ck_flags = 0u;

  // ---- V983 OV1 suffix -----------------------------------------------------
  o.peer_payload_ready_table = PT + EP16_PT_PEER_PAYLOAD_READY;
  o.recv_payload_ready = ep16t_a(b, L, EP16_AR_RECV_PAYLOAD_READY);
  o.peer_return_payload_table = PT + EP16_PT_PEER_RETURN_PAYLOAD;
  o.return_recv_payload = ep16t_a(b, L, EP16_AR_RETURN_RECV_PAYLOAD);
  o.peer_home_arrival_table = PT + EP16_PT_PEER_HOME_ARRIVAL;
  o.home_arrival = ep16t_s(b, S, EP16_SC_HOME_ARRIVAL);
  o.home_expected = ep16t_s(b, S, EP16_SC_HOME_EXPECTED);
  o.home_slots = ep16t_s(b, S, EP16_SC_HOME_SLOTS);
  o.plan_flags = ep16t_s(b, S, EP16_SC_PLAN_FLAGS);
  o.a_tile_ready = ep16t_s(b, S, EP16_SC_A_TILE_READY);
  o.a_tile_rows_done = ep16t_s(b, S, EP16_SC_A_TILE_ROWS_DONE);
  o.stage1_tile_done = ep16t_s(b, S, EP16_SC_STAGE1_TILE_DONE);
  o.row_n_done = ep16t_s(b, S, EP16_SC_ROW_N_DONE);
  o.bucket_done = ep16t_s(b, S, EP16_SC_BUCKET_DONE);
  o.bucket_route_rows = ep16t_s(b, S, EP16_SC_BUCKET_ROUTE_ROWS);
  o.bucket_sent = ep16t_s(b, S, EP16_SC_BUCKET_SENT);
  o.row_route_id = ep16t_s(b, S, EP16_SC_ROW_ROUTE_ID);
  o.stage2_weights = ep16t_s(b, S, EP16_SC_STAGE2_WEIGHTS);
  o.final_output = ext.final_output;
  o.gemm2_output = ep16t_s(b, S, EP16_SC_GEMM2_OUTPUT);
  o.ck_w2 = ext.ck_w2;
  o.ck_w2_scale = ext.ck_w2_scale;
  o.trace = ep16t_s(b, S, EP16_SC_TRACE);
  o.byte_counters = ep16t_s(b, S, EP16_SC_BYTE_COUNTERS);
  // Both timers are read through an explicit non-null guard on device, so a
  // null pointer is the documented way to switch them off.
  o.scatter_time = s.enable_scatter_time ? ep16t_s(b, S, EP16_SC_SCATTER_TIME) : 0ull;
  o.dispatch_time = s.enable_dispatch_time ? ep16t_s(b, S, EP16_SC_DISPATCH_TIME) : 0ull;
  o.home_done = ep16t_s(b, S, EP16_SC_HOME_DONE);
  o.post_control = ep16t_s(b, S, EP16_SC_POST_CONTROL);
  o.gemm_width = 128ull;
  o.peer_post_state_table = ep16t_s(b, S, EP16_SC_PEER_POST_STATE);
  o.hidden = s.hidden;
  o.intermediate = s.intermediate;
  o.n1_blocks = s.intermediate / 128u;
  o.n2_blocks = s.hidden / 128u;
  // Ep16TurboShape::mode selects the host-side node transport plan.
  // V983Ov1Ext::mode is an independent device-entry guard tag and must be
  // zero.  Mode-2 transport is represented by node_tokens and the node-plane
  // pointers in Ep16TurboContext.
  o.mode = 0u;
  o.block_m = 32u;
  o.comm_chunk_rows = 8u;
  o.role_config = 0u;          // the turbo entry hardwires roles {80, 0, 80}

  // ---- turbo context -------------------------------------------------------
  Ep16TurboContext x{};
  x.magic = EP16_TURBO_MAGIC;
  x.generation = spec.generation;
  x.runtime = spec.runtime;
  const uint64_t tx_base = ep16t_s(b, S, EP16_SC_TX_JOBS);
  for (uint32_t q = 0; q < 32u; ++q) {
    x.tx_jobs[q] = tx_base + (uint64_t)S.tx_offset[q] * sizeof(Ep16TxJob);
    x.tx_count[q] = S.tx_count[q];
  }
  x.rx_jobs = ep16t_s(b, S, EP16_SC_RX_JOBS);
  x.rx_count = S.rx_total;
  x.node_tokens = s.node_tokens;
  x.return_ready = ep16t_a(b, L, EP16_AR_RETURN_READY);
  x.node_partial_table = PT + EP16_PT_NODE_PARTIAL;
  x.node_arrival_table = PT + EP16_PT_NODE_ARRIVAL;
  x.node_expected = ep16t_s(b, S, EP16_SC_NODE_EXPECTED);
  x.node_partial = ep16t_a(b, L, EP16_AR_NODE_PARTIAL);
  x.node_rank_mask = ep16t_s(b, S, EP16_SC_NODE_RANK_MASK);
  x.node_arrival = ep16t_a(b, L, EP16_AR_NODE_ARRIVAL);
  x.node_output = ep16t_a(b, L, EP16_AR_NODE_OUTPUT);
  x.node_ready = ep16t_s(b, S, EP16_SC_NODE_READY);
  x.node_recv = ep16t_a(b, L, EP16_AR_NODE_RECV);
  x.node_recv_ready = ep16t_s(b, S, EP16_SC_NODE_RECV_READY);
  x.compute_done = ep16t_s(b, S, EP16_SC_COMPUTE_DONE);
  x.gda_state = ep16t_s(b, S, EP16_SC_GDA_STATE);
  x.timeout_ticks = spec.timeout_ticks;

  if (env_out) *env_out = env;
  if (ctx_out) *ctx_out = x;
  return 0;
}

// ---------------------------------------------------------------------------
// Re-implementation of the device entry guard, so a bad packet is rejected on
// the host instead of coming back as sticky error 900 after a 2 ms timeout.
// Returns 0 if the kernel would accept this packet.
// ---------------------------------------------------------------------------
static inline int ep16_turbo_check_entry_guard(const V983Envelope& packet,
                                               const Ep16TurboContext& x) {
  const V964DispatchExt& e = packet.base.base;
  const V968CkExt& ck = packet.base.ck;
  const V983Ov1Ext& o = packet.ov1;
  if (x.magic != EP16_TURBO_MAGIC) return 1;
  if (!x.generation) return 2;
  if (!x.runtime) return 3;
  if (!x.timeout_ticks) return 4;
  if (e.world != 16u) return 5;
  if (o.mode != 0u) return 24;
  if (e.rank >= 16u) return 6;
  if (!e.capacity) return 7;
  if ((uint64_t)e.world * (uint64_t)(e.capacity + 1u) >= 0xffffffull) return 8;
  if (!e.topk || e.topk > 12u) return 9;
  if (!e.n_experts || e.n_experts % e.world) return 10;
  if (e.n_experts / e.world > 64u) return 11;
  if (!e.error) return 12;
  if (e.payload_bytes != o.hidden) return 13;
  if (e.scale_groups != o.hidden / 128u) return 14;
  if (o.hidden < 128u || o.hidden % 128u) return 15;
  if (o.intermediate < 128u || o.intermediate % 128u) return 16;
  if (o.n1_blocks != o.intermediate / 128u) return 17;
  if (o.n2_blocks != o.hidden / 128u) return 18;
  if (ck.ck_hidden != o.hidden) return 19;
  if (ck.ck_intermediate != o.intermediate) return 20;
  if (ck.ck_route_rows >= 0xffffffu) return 21;
  if (ck.ck_num_tokens != e.world * e.capacity * e.topk) return 25;
  if (o.gemm_width != 128ull) return 26;
  if (!x.compute_done) return 22;
  if (!x.gda_state) return 23;
  return 0;
}
