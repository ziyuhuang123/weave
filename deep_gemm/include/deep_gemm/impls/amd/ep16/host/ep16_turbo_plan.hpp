#pragma once
// ============================================================================
// ep16_turbo_plan.hpp -- TxJob / RxJob descriptor-table generation.
//
// PURE C++ (no HIP).  Deterministic from (shape, layouts, base addresses).
//
// ---------------------------------------------------------------------------
// QP OWNERSHIP
// ---------------------------------------------------------------------------
// ep16_gda_wave4() in ep16_transport.hpp binds CTA id -> (peer, channel):
//     id    = blockIdx.x, 0 <= id < 32
//     peer  = ((rank >> 3) ^ 1) * 8 + (id >> 2)     // absolute remote rank
//     chan  = id & 3
// so QP index q = p*4 + c with p = remote peer slot 0..7.  The jobs of QP q
// are consumed strictly in array order by one lane, and the trailing 8-byte
// NIC flag of a descriptor rides the SAME RC QP as its data, so in-order RC
// delivery is what makes "flag after data" safe.
//
// ---------------------------------------------------------------------------
// DESCRIPTOR ORDER PER QP (no queue-ordering cycles)
// ---------------------------------------------------------------------------
//   channel 0 only:  metadata header -> metadata slot chunks -> metadata flag
//   all channels:    per-slot DISPATCH payload -> per-slot DISPATCH scale(+flag)
//   all channels:    RETURN (mode 0/1 per-slot, mode 2 per node token)
// Metadata precedes payload, payload precedes RETURN, and nothing on the
// DISPATCH path is gated on a receive-side plan word.
//
// ---------------------------------------------------------------------------
// WHY slot_chunk MUST BE 1
// ---------------------------------------------------------------------------
// A TxJob carries exactly ONE `ready`/`target` release word.  Dispatch waves
// claim tokens from an atomic cursor, so slot s+1 can be staged before slot s.
// A descriptor spanning slots [s, s+n) therefore has no single word that
// proves all n slots are staged, and gating on the last slot would race.
// The host API accepts slot_chunk only as 1 and validate_shape enforces it.
//
// ---------------------------------------------------------------------------
// GATE SOURCES
// ---------------------------------------------------------------------------
//  TX metadata  ready = tx_progress_stage[p].meta_ready, target = D8PR magic
//               (set by v983_dispatch_4wave step 1 AFTER the staging copy)
//  TX meta chunk skip  valid_count = bucket_count[R], count_ready = plan_flags[0]
//  TX payload   ready = tx_payload_ready_stage[p*cap+s], target = 1
//               (set by v983_dispatch_4wave step 2 after the staged row lands)
//               skip valid_count = bucket_count[R], count_ready = plan_flags[0]
//  TX return    ready = bucket_sent[R*cap+s], target = 1 (mode 0/1)
//               ready = node_ready[t], target = 1        (mode 2)
//               skip  valid_count = received meta header count for source R
//                     count_ready = recv_progress[R].meta_ready, target = magic
//               mode 2 skip: valid_count = node_expected[t], index = 0
//  RX metadata  publish recv_progress[S].meta_ready = D8PR magic
//  RX payload   publish recv_payload_ready[S*cap+s] = 1
//               skip valid_count = header count of source S, count_ready =
//                    recv_progress[S].meta_ready, target = magic
//  RX return    publish return_ready[P*cap+s] = 1 (mode 0/1)
//               skip valid_count = bucket_count[P] (buckets I SENT to P),
//                    count_ready = plan_flags[0]
//               publish node_recv_ready[t] = 1 (mode 2)
//               skip valid_count = node_send_expect[t], index = 0
// ============================================================================
#include "ep16_turbo_layout.hpp"

// ---------------------------------------------------------------------------
// TxJob table
// ---------------------------------------------------------------------------
// Jobs for QP q start at S.tx_offset[q] and run for S.tx_count[q] entries.
// Returns 0 on success, nonzero on a capacity or consistency failure.
// ---------------------------------------------------------------------------
// TxJob table
// ---------------------------------------------------------------------------
// Jobs for QP q start at S.tx_offset[q] and run for S.tx_count[q] entries.
// Returns 0 on success, nonzero on a capacity or consistency failure.
// RACE29 TrackA: slot_chunk == 1 keeps the byte-identical legacy descriptor
// stream.  slot_chunk > 1 replaces the per-slot dispatch descriptors with
// bulk descriptors that each cover K contiguous slots of one peer bucket:
//   * the descriptor's readiness gate is the ALL-OF check over its K per-slot
//     release words, so it can never race a partially staged bucket;
//   * the trailing NIC flag rides the descriptor's own RC QP AFTER its data
//     WRs and is written to the CHUNK BASE slot's flag cell, which is a
//     statically known address on both sides;
//   * the receiver publishes exactly min(K, header_count - base) readiness
//     words, so a partially filled tail chunk is handled without overrun.
// ---------------------------------------------------------------------------
static inline int ep16_turbo_build_tx(const Ep16TurboShape& s,
                                      const Ep16ArenaLayout& L,
                                      const Ep16ScratchLayout& S,
                                      const Ep16TurboBases& b,
                                      Ep16TxJob* out, uint32_t out_cap) {
  if (out_cap < S.tx_total) return -1;
  for (uint32_t i = 0; i < S.tx_total; ++i) {
    Ep16TxJob z{};
    out[i] = z;
  }

  const uint64_t cap = s.capacity;
  const uint64_t MR = L.meta_region_bytes;
  const uint64_t PB = L.payload_row_bytes;
  const uint64_t SGB = L.scale_row_bytes;
  const uint64_t HB = L.hidden_row_bytes;
  const uint64_t PROG = sizeof(V809SourceProgress);
  const uint32_t m = s.rank;
  const uint32_t ob = ep16t_other_node_base(s);
  const uint32_t owner_peer = m & 7u;
  const uint32_t nmeta = (uint32_t)ep16t_div_up(cap, s.meta_chunk_slots);
  const uint32_t K = ep16t_slot_chunk(s);
  const uint32_t nchunk = ep16t_nchunk(s);
  const bool chunked = K > 1u;

  const uint64_t a_tx_meta = ep16t_a(b, L, EP16_AR_TX_META_STAGE);
  const uint64_t a_tx_pay = ep16t_a(b, L, EP16_AR_TX_PAYLOAD_STAGE);
  const uint64_t a_tx_scl = ep16t_a(b, L, EP16_AR_TX_SCALE_STAGE);
  const uint64_t a_tx_prog = ep16t_a(b, L, EP16_AR_TX_PROGRESS_STAGE);
  const uint64_t a_tx_rdy = ep16t_a(b, L, EP16_AR_TX_PAYLOAD_READY_STAGE);
  const uint64_t a_tx_ret = ep16t_a(b, L, EP16_AR_TX_RETURN_STAGE);
  const uint64_t a_node_out = ep16t_a(b, L, EP16_AR_NODE_OUTPUT);
  const uint64_t a_recv_meta = ep16t_a(b, L, EP16_AR_RECV_META);
  const uint64_t a_recv_prog = ep16t_a(b, L, EP16_AR_RECV_PROGRESS);

  const uint64_t s_bucket_count = ep16t_s(b, S, EP16_SC_BUCKET_COUNT);
  const uint64_t s_plan_flags = ep16t_s(b, S, EP16_SC_PLAN_FLAGS);
  const uint64_t s_bucket_sent = ep16t_s(b, S, EP16_SC_BUCKET_SENT);
  const uint64_t s_node_ready = ep16t_s(b, S, EP16_SC_NODE_READY);
  const uint64_t s_node_expected = ep16t_s(b, S, EP16_SC_NODE_EXPECTED);
  const uint64_t s_ones = ep16t_s(b, S, EP16_SC_ONES_WORD);

  for (uint32_t p = 0; p < 8u; ++p) {
    const uint32_t R = ob + p;
    if (!b.arena_va[R]) return -2;
    const uint64_t pa = b.arena_va[R];
    const uint64_t r_meta = pa + L.off[EP16_AR_RECV_META] + (uint64_t)m * MR;
    const uint64_t r_pay = pa + L.off[EP16_AR_RECV_PAYLOAD] + (uint64_t)m * cap * PB;
    const uint64_t r_scl = pa + L.off[EP16_AR_RECV_SCALE] + (uint64_t)m * cap * SGB;
    const uint64_t r_ret = pa + L.off[EP16_AR_RETURN_RECV_PAYLOAD] + (uint64_t)m * cap * HB;
    const uint64_t r_f_meta = pa + L.off[EP16_AR_NIC_META_FLAG] + (uint64_t)m * 8ull;
    const uint64_t r_f_pay = pa + L.off[EP16_AR_NIC_PAYLOAD_FLAG] + (uint64_t)m * cap * 8ull;
    const uint64_t r_f_ret = pa + L.off[EP16_AR_NIC_RETURN_FLAG] + (uint64_t)m * cap * 8ull;
    const uint64_t r_node_recv = pa + L.off[EP16_AR_NODE_RECV];
    const uint64_t r_f_node = pa + L.off[EP16_AR_NIC_NODE_FLAG];

    const uint64_t meta_gate = a_tx_prog + (uint64_t)p * PROG;   // meta_ready at +0
    const uint64_t count_src = s_bucket_count + (uint64_t)R * 4ull;
    const uint64_t recv_hdr = a_recv_meta + (uint64_t)R * MR;    // header word = n
    const uint64_t recv_gate = a_recv_prog + (uint64_t)R * PROG; // meta_ready at +0

    for (uint32_t c = 0; c < 4u; ++c) {
      const uint32_t q = p * 4u + c;
      Ep16TxJob* j = out + S.tx_offset[q];
      uint32_t n = 0;

      // ---------------- DISPATCH metadata (channel 0 only) ----------------
      if (c == 0u) {
        j[n].src = a_tx_meta + (uint64_t)p * MR;
        j[n].dst = r_meta;
        j[n].bytes = 64ull;
        j[n].remote_flag = 0;
        j[n].ready = meta_gate;
        j[n].target = V809_PROGRESS_MAGIC;
        j[n].phase = 0;
        ++n;
        for (uint32_t k = 0; k < nmeta; ++k) {
          const uint64_t s0 = (uint64_t)k * s.meta_chunk_slots;
          const uint64_t ln = (s.meta_chunk_slots < (cap - s0)) ? s.meta_chunk_slots : (cap - s0);
          j[n].src = a_tx_meta + (uint64_t)p * MR + V809_META_HEADER + s0 * V809_META_STRIDE;
          j[n].dst = r_meta + V809_META_HEADER + s0 * V809_META_STRIDE;
          j[n].bytes = ln * V809_META_STRIDE;
          j[n].ready = meta_gate;
          j[n].target = V809_PROGRESS_MAGIC;
          j[n].phase = 0;
          j[n].valid_count = count_src;
          j[n].count_ready = s_plan_flags;
          j[n].index = (uint32_t)s0;
          j[n].count_target = 1u;
          ++n;
        }
        j[n].remote_flag = r_f_meta;
        j[n].ready = meta_gate;
        j[n].target = V809_PROGRESS_MAGIC;
        j[n].phase = 0;
        ++n;
      }

      // ---------------------------- DISPATCH ------------------------------
      if (chunked) {
        for (uint32_t cj = c; cj < nchunk; cj += 4u) {
          const uint64_t s0 = (uint64_t)cj * K;
          const uint64_t gate_base = a_tx_rdy + (p * cap + s0) * 4ull;
          // payload run: K contiguous FP8 rows
          j[n].src = a_tx_pay + (p * cap + s0) * PB;
          j[n].dst = r_pay + s0 * PB;
          j[n].bytes = PB;                       // per-slot bytes
          j[n].remote_flag = 0;
          j[n].ready = gate_base;
          j[n].target = K;
          j[n].phase = EP16_TXJ_CHUNK_GATE;
          j[n].valid_count = count_src;
          j[n].count_ready = s_plan_flags;
          j[n].index = (uint32_t)s0;
          j[n].count_target = 1u;
          ++n;
          // scale run + this descriptor's trailing NIC flag at the CHUNK BASE
          j[n].src = a_tx_scl + (p * cap + s0) * SGB;
          j[n].dst = r_scl + s0 * SGB;
          j[n].bytes = SGB;
          j[n].remote_flag = r_f_pay + s0 * 8ull;
          j[n].ready = gate_base;
          j[n].target = K;
          j[n].phase = EP16_TXJ_CHUNK_GATE;
          j[n].valid_count = count_src;
          j[n].count_ready = s_plan_flags;
          j[n].index = (uint32_t)s0;
          j[n].count_target = 1u;
          ++n;
        }
      } else {
        for (uint64_t slot = c; slot < cap; slot += 4u) {
          const uint64_t gate = a_tx_rdy + (p * cap + slot) * 4ull;
          j[n].src = a_tx_pay + (p * cap + slot) * PB;
          j[n].dst = r_pay + slot * PB;
          j[n].bytes = PB;
          j[n].ready = gate;
          j[n].target = 1u;
          j[n].phase = 0;
          j[n].valid_count = count_src;
          j[n].count_ready = s_plan_flags;
          j[n].index = (uint32_t)slot;
          j[n].count_target = 1u;
          ++n;
          j[n].src = a_tx_scl + (p * cap + slot) * SGB;
          j[n].dst = r_scl + slot * SGB;
          j[n].bytes = SGB;
          j[n].remote_flag = r_f_pay + slot * 8ull;
          j[n].ready = gate;
          j[n].target = 1u;
          j[n].phase = 0;
          j[n].valid_count = count_src;
          j[n].count_ready = s_plan_flags;
          j[n].index = (uint32_t)slot;
          j[n].count_target = 1u;
          ++n;
        }
      }

      // ------------------------------ RETURN ------------------------------
      if (s.mode != 2u) {
        for (uint64_t slot = c; slot < cap; slot += 4u) {
          j[n].src = a_tx_ret + (p * cap + slot) * HB;
          j[n].dst = r_ret + slot * HB;
          j[n].bytes = HB;
          j[n].remote_flag = r_f_ret + slot * 8ull;
          j[n].ready = s_bucket_sent + (R * cap + slot) * 4ull;
          j[n].target = 1u;
          j[n].phase = 1u;
          j[n].valid_count = recv_hdr;
          j[n].count_ready = recv_gate;
          j[n].index = (uint32_t)slot;
          j[n].count_target = V809_PROGRESS_MAGIC;
          ++n;
        }
      } else if (p == owner_peer) {
        if (chunked) {
          // STATIC_SLOTS: `target` is the exact number of tokens in this chunk,
          // baked here, so the posting lane needs no runtime count word.  All
          // `target` node_ready words must be set before the chunk is posted.
          const uint32_t Kr = ep16t_ret_chunk(s);
          const uint32_t nnc = ep16t_nnodechunk(s);
          for (uint32_t cj = c; cj < nnc; cj += 4u) {
            const uint64_t t0 = (uint64_t)cj * Kr;
            const uint64_t ln = (Kr < (s.node_tokens - t0)) ? Kr : (s.node_tokens - t0);
            j[n].src = a_node_out + t0 * HB;
            j[n].dst = r_node_recv + t0 * HB;
            j[n].bytes = HB;
            j[n].remote_flag = r_f_node + t0 * 8ull;
            j[n].ready = s_node_ready + t0 * 4ull;
            j[n].target = (uint32_t)ln;
            j[n].index = (uint32_t)t0;
            j[n].phase = EP16_TXJ_CHUNK_GATE | EP16_TXJ_STATIC_SLOTS | 1u;
            ++n;
          }
        } else {
          for (uint64_t t = c; t < s.node_tokens; t += 4u) {
            j[n].src = a_node_out + t * HB;
            j[n].dst = r_node_recv + t * HB;
            j[n].bytes = HB;
            j[n].remote_flag = r_f_node + t * 8ull;
            j[n].ready = s_node_ready + t * 4ull;
            j[n].target = 1u;
            j[n].phase = 1u;
            j[n].valid_count = s_node_expected + t * 4ull;
            j[n].count_ready = s_ones;
            j[n].index = 0u;
            j[n].count_target = 1u;
            ++n;
          }
        }
      }
      if (n != S.tx_count[q]) return -3;
    }
  }
  return 0;
}

// ---------------------------------------------------------------------------
// RxJob table (rank local).  Table order IS the D5 tier order:
//   [0,8)                      metadata            tier LOCAL_READY
//   next 8*nchunk (or 8*cap)   dispatch payload    tier REMOTE_PRIMARY
//   remainder                  return / node fanout tier LOCAL_COPY
// ---------------------------------------------------------------------------
static inline int ep16_turbo_build_rx(const Ep16TurboShape& s,
                                      const Ep16ArenaLayout& L,
                                      const Ep16ScratchLayout& S,
                                      const Ep16TurboBases& b,
                                      Ep16RxJob* out, uint32_t out_cap) {
  if (out_cap < S.rx_total) return -1;
  const uint64_t cap = s.capacity;
  const uint64_t MR = L.meta_region_bytes;
  const uint64_t PROG = sizeof(V809SourceProgress);
  const uint32_t ob = ep16t_other_node_base(s);
  const uint32_t K = ep16t_slot_chunk(s);
  const uint32_t nchunk = ep16t_nchunk(s);
  const bool chunked = K > 1u;
  const uint64_t a_recv_meta = ep16t_a(b, L, EP16_AR_RECV_META);
  const uint64_t a_recv_prog = ep16t_a(b, L, EP16_AR_RECV_PROGRESS);
  const uint64_t a_recv_rdy = ep16t_a(b, L, EP16_AR_RECV_PAYLOAD_READY);
  const uint64_t a_f_pay = ep16t_a(b, L, EP16_AR_NIC_PAYLOAD_FLAG);
  const uint64_t a_f_ret = ep16t_a(b, L, EP16_AR_NIC_RETURN_FLAG);
  const uint64_t a_f_node = ep16t_a(b, L, EP16_AR_NIC_NODE_FLAG);
  uint32_t n = 0;

  for (uint32_t p = 0; p < 8u; ++p) {
    const uint32_t src = ob + p;
    out[n].flag = ep16t_a(b, L, EP16_AR_NIC_META_FLAG) + (uint64_t)src * 8ull;
    out[n].publish = a_recv_prog + src * PROG;
    out[n].value = V809_PROGRESS_MAGIC;
    out[n].reserved = 1u | ((uint32_t)EP16_TIER_LOCAL_READY << EP16_RXJ_TIER_SHIFT);
    ++n;
  }
  for (uint32_t p = 0; p < 8u; ++p) {
    const uint32_t src = ob + p;
    const uint64_t hdr = a_recv_meta + (uint64_t)src * MR;
    const uint64_t gate = a_recv_prog + (uint64_t)src * PROG;
    if (chunked) {
      for (uint32_t cj = 0; cj < nchunk; ++cj) {
        const uint64_t s0 = (uint64_t)cj * K;
        out[n].flag = a_f_pay + (src * cap + s0) * 8ull;
        out[n].publish = a_recv_rdy + (src * cap + s0) * 4ull;
        out[n].value = 1u;
        out[n].reserved = (K & EP16_RXJ_COUNT_MASK) |
                          ((uint32_t)EP16_TIER_REMOTE_PRIMARY << EP16_RXJ_TIER_SHIFT) |
                          ((src & 0xffu) << 24);
        out[n].valid_count = hdr;
        out[n].count_ready = gate;
        out[n].index = (uint32_t)s0;
        out[n].count_target = V809_PROGRESS_MAGIC;
        ++n;
      }
    } else {
      for (uint64_t slot = 0; slot < cap; ++slot) {
        out[n].flag = a_f_pay + (src * cap + slot) * 8ull;
        out[n].publish = a_recv_rdy + (src * cap + slot) * 4ull;
        out[n].value = 1u;
        out[n].reserved = 1u |
                          ((uint32_t)EP16_TIER_REMOTE_PRIMARY << EP16_RXJ_TIER_SHIFT) |
                          ((src & 0xffu) << 24);
        out[n].valid_count = hdr;
        out[n].count_ready = gate;
        out[n].index = (uint32_t)slot;
        out[n].count_target = V809_PROGRESS_MAGIC;
        ++n;
      }
    }
  }
  if (s.mode != 2u) {
    const uint64_t s_bc = ep16t_s(b, S, EP16_SC_BUCKET_COUNT);
    const uint64_t s_pf = ep16t_s(b, S, EP16_SC_PLAN_FLAGS);
    const uint64_t a_ret_rdy = ep16t_a(b, L, EP16_AR_RETURN_READY);
    for (uint32_t p = 0; p < 8u; ++p) {
      const uint32_t prod = ob + p;
      for (uint64_t slot = 0; slot < cap; ++slot) {
        out[n].flag = a_f_ret + (prod * cap + slot) * 8ull;
        out[n].publish = a_ret_rdy + (prod * cap + slot) * 4ull;
        out[n].value = 1u;
        out[n].reserved = 1u | ((uint32_t)EP16_TIER_LOCAL_COPY << EP16_RXJ_TIER_SHIFT);
        out[n].valid_count = s_bc + prod * 4ull;
        out[n].count_ready = s_pf;
        out[n].index = (uint32_t)slot;
        out[n].count_target = 1u;
        ++n;
      }
    }
  } else {
    const uint64_t s_nrr = ep16t_s(b, S, EP16_SC_NODE_RECV_READY);
    if (chunked) {
      // One RDMA flag per chunk; the poller publishes exactly `reserved` words,
      // which the builder bakes as the true token count of the chunk.
      const uint32_t Kr = ep16t_ret_chunk(s);
      const uint32_t nnc = ep16t_nnodechunk(s);
      for (uint32_t cj = 0; cj < nnc; ++cj) {
        const uint64_t t0 = (uint64_t)cj * Kr;
        const uint64_t ln = (Kr < (s.node_tokens - t0)) ? Kr : (s.node_tokens - t0);
        out[n].flag = a_f_node + t0 * 8ull;
        out[n].publish = s_nrr + t0 * 4ull;
        out[n].value = 1u;
        out[n].reserved = ((uint32_t)ln & EP16_RXJ_COUNT_MASK) |
                          ((uint32_t)EP16_TIER_LOCAL_COPY << EP16_RXJ_TIER_SHIFT);
        out[n].index = (uint32_t)t0;
        ++n;
      }
    } else {
      const uint64_t s_nse = ep16t_s(b, S, EP16_SC_NODE_SEND_EXPECT);
      const uint64_t s_ones = ep16t_s(b, S, EP16_SC_ONES_WORD);
      for (uint64_t t = 0; t < s.node_tokens; ++t) {
        out[n].flag = a_f_node + t * 8ull;
        out[n].publish = s_nrr + t * 4ull;
        out[n].value = 1u;
        out[n].reserved = 1u | ((uint32_t)EP16_TIER_LOCAL_COPY << EP16_RXJ_TIER_SHIFT);
        out[n].valid_count = s_nse + t * 4ull;
        out[n].count_ready = s_ones;
        out[n].index = 0u;
        out[n].count_target = 1u;
        ++n;
      }
    }
  }
  if (n != S.rx_total) return -4;
  return 0;
}
// ---------------------------------------------------------------------------
// Plan validation.  Independent of the builder: re-derives only the address
// ranges and checks every descriptor against the EP16 transport rules.
// ---------------------------------------------------------------------------
struct Ep16PlanAudit {
  uint32_t tx_jobs;
  uint32_t rx_jobs;
  uint32_t tx_flag_jobs;        // descriptors carrying a trailing NIC flag
  uint32_t tx_pure_flag_jobs;   // bytes == 0 descriptors
  uint32_t tx_phase0;
  uint32_t tx_phase1;
  uint64_t tx_max_bytes;
  uint64_t tx_total_bytes;      // worst case: every slot used
  uint32_t worst_wr_per_qp;     // 128 KiB fragments + flags
  uint32_t errors;
  uint32_t first_error_code;
  uint32_t first_error_job;
};

static inline bool ep16t_in(uint64_t a, uint64_t base, uint64_t bytes, uint64_t n) {
  if (!base || n > bytes) return false;
  return a >= base && (a - base) <= (bytes - n);
}

static inline int ep16_turbo_audit_plan(const Ep16TurboShape& s,
                                        const Ep16ArenaLayout& L,
                                        const Ep16ScratchLayout& S,
                                        const Ep16TurboBases& b,
                                        const Ep16TxJob* tx,
                                        const Ep16RxJob* rx,
                                        Ep16PlanAudit* au) {
  Ep16PlanAudit a{};
  a.tx_jobs = S.tx_total;
  a.rx_jobs = S.rx_total;
  const uint64_t AB = L.total_bytes;
  const uint64_t SB = S.total_bytes;
  const uint32_t ob = ep16t_other_node_base(s);

  auto local_word = [&](uint64_t p) -> bool {
    if (!p) return false;
    if (p % 4ull) return false;
    const bool in_arena = ep16t_in(p, b.arena, AB, 4);
    const bool in_scratch = ep16t_in(p, b.scratch, SB, 4);
    return in_arena || in_scratch;
  };

  uint32_t wr_per_qp[32] = {0};

  for (uint32_t q = 0; q < 32u; ++q) {
    const uint32_t p = q >> 2u;
    const uint32_t R = ob + p;
    const uint64_t pa = b.arena_va[R];
    const Ep16TxJob* j = tx + S.tx_offset[q];
    uint32_t seen_phase1 = 0;
    for (uint32_t i = 0; i < S.tx_count[q]; ++i) {
      const Ep16TxJob& t = j[i];
      const uint32_t gid = S.tx_offset[q] + i;
      #define EP16T_BAD(code) do { ++a.errors; if (a.errors == 1u) { a.first_error_code = (code); a.first_error_job = gid; } } while (0)
      const bool chunked = (t.phase & EP16_TXJ_CHUNK_GATE) != 0u;
      const uint32_t ph = t.phase & EP16_TXJ_PHASE_MASK;
      if ((t.phase & ~(EP16_TXJ_CHUNK_GATE | EP16_TXJ_STATIC_SLOTS | EP16_TXJ_PHASE_MASK)) != 0u)
        EP16T_BAD(16);
      if (ph > 1u) EP16T_BAD(1);
      if (ph == 1u) seen_phase1 = 1u;
      else if (seen_phase1) EP16T_BAD(2);          // DISPATCH after RETURN
      if (chunked && (t.target == 0u || t.index >= s.capacity)) EP16T_BAD(17);
      const uint64_t span = chunked
          ? (uint64_t)t.bytes * (uint64_t)t.target : (uint64_t)t.bytes;
      if (t.bytes) {
        if (!ep16t_in(t.src, b.arena, AB, span)) EP16T_BAD(3);
        if (!ep16t_in(t.dst, pa, AB, span)) EP16T_BAD(4);
        if (span > AB) EP16T_BAD(5);
        a.tx_total_bytes += span;
        if (span > a.tx_max_bytes) a.tx_max_bytes = span;
      } else {
        ++a.tx_pure_flag_jobs;
        if (!t.remote_flag) EP16T_BAD(6);          // zero-byte, zero-flag: dead
      }
      if (t.remote_flag) {
        ++a.tx_flag_jobs;
        if (!ep16t_in(t.remote_flag, pa, AB, 8)) EP16T_BAD(7);
        if (t.remote_flag % 8ull) EP16T_BAD(8);
      }
      if (!t.ready) EP16T_BAD(9);                  // every turbo job is gated
      else if (!local_word(t.ready)) EP16T_BAD(10);
      if (t.valid_count) {
        if (!local_word(t.valid_count)) EP16T_BAD(11);
        if (!t.count_ready) EP16T_BAD(12);
        else if (!local_word(t.count_ready)) EP16T_BAD(13);
        if (!t.count_target) EP16T_BAD(14);
      } else if (t.count_ready) {
        EP16T_BAD(15);
      }
      if (ph == 0u) ++a.tx_phase0; else ++a.tx_phase1;
      // Exact 128 KiB / 64 MiB-clipped WR count for this descriptor.
      {
        const uint64_t pieces = span ? ep16t_wr_pieces(t.src - b.arena, t.dst - pa, span) : 0ull;
        wr_per_qp[q] += (uint32_t)pieces + (t.remote_flag ? 1u : 0u);
      }
    }
  }
  for (uint32_t q = 0; q < 32u; ++q)
    if (wr_per_qp[q] > a.worst_wr_per_qp) a.worst_wr_per_qp = wr_per_qp[q];

  for (uint32_t i = 0; i < S.rx_total; ++i) {
    const Ep16RxJob& r = rx[i];
    #define EP16T_BADR(code) do { ++a.errors; if (a.errors == 1u) { a.first_error_code = (code); a.first_error_job = i; } } while (0)
    const uint32_t rcnt = r.reserved & EP16_RXJ_COUNT_MASK;
    const uint32_t rtier = (r.reserved >> EP16_RXJ_TIER_SHIFT) & 0xffu;
    if (rtier > (uint32_t)EP16_TIER_LOCAL_COPY) EP16T_BADR(28);
    if (rcnt == 0u) EP16T_BADR(29);
    if (!ep16t_in(r.flag, b.arena, AB, 8)) EP16T_BADR(20);
    if (r.flag % 8ull) EP16T_BADR(21);
    if (!local_word(r.publish)) EP16T_BADR(22);
    if (!r.value) EP16T_BADR(23);
    if (r.valid_count) {
      if (!local_word(r.valid_count)) EP16T_BADR(24);
      if (!local_word(r.count_ready)) EP16T_BADR(25);
      if (!r.count_target) EP16T_BADR(26);
    } else if (r.count_ready) {
      EP16T_BADR(27);
    }
    #undef EP16T_BADR
  }
  #undef EP16T_BAD
  if (au) *au = a;
  return a.errors ? -1 : 0;
}
