#pragma once
// =============================================================================================================
// v70_scatter_staging_3path.hpp -- Design Point #16 (WEAVE_SCATTER_STAGING 3-path Scatter dataflow)
//                               + Design Point #17 (WEAVE_LOCAL_SCATTER_DEDUP 0/1)
//
// The Scatter stage copies one bf16 row per unique (src_rank, token, dst) out of the local GEMM2 output:
//   * src_rank is in this rank's node  -> peer return payload slot  peer_return_payload_table[src] + t*H
//   * src_rank is in the peer node     -> peer node partial plane   node_partial_table[src&7] + (rank&7)*NT*H + t*H
// The rows of the token's `nexp` local experts are read from gemm2_output (bucket_route_rows[key*topk+k]).
// Every path below publishes exactly ONE arrival per claimed (src,token) -- the caller
// (v70_scatter_owned_pass) keeps the exclusive bucket_done claim, the batch-wide s_waitcnt vmcnt(0) and the
// v46_bucket_flag release exactly as before.  Compute waves W00..W03 never enter this header
// (WEAVE_SCATTER_OWNER=comm_only).
//
// -------------------------------------------------------------------------------------------------------------
// #16 path 0 -- WEAVE_SCATTER_STAGING=0  "direct" / "reg_2stage"
//   512-column tiles per lane, two 128-bit register stages: stage A holds the finished uint4 tile while the
//   VMEM reads of the next tile fill stage B; the wide store of A issues between them.  The merged tile goes
//   straight from the VGPR stage to the peer xGMI / return slot.  No LDS, no HBM pre-reduction.
// #16 path 1 -- WEAVE_SCATTER_STAGING=1  "smem"
//   The same merge, but every finished 128-bit tile is handed through a communication-only LDS window:
//     ds_write_b128 -> s_waitcnt lgkmcnt(0) -> ds_read_b128 -> wide 128-bit store
//   The window is a per-wave ping-pong of 2 x 1024 B carved out of the middle of the single 64 KiB phase
//   arena ([8192, 32768)), which no other phase touches in a communication CTA: compute CTAs own
//   [0,8192) (M16 raw body / CK stage tiles) and every barrier/control carrier lives above 32768.
// #16 path 2 -- WEAVE_SCATTER_STAGING=2  "hbm_dedup"
//   HBM-resident dedup: hit 0 wide-stores the expert row straight into the destination row, every further
//   local-expert hit is folded into that same HBM row with packed bf16x2 hardware atomics
//   (global_atomic_pk_add_bf16).  The token therefore becomes ONE deduplicated row in HBM, and exactly one
//   arrival is published for it, so the xGMI/RDMA peer sees a single deduplicated payload per (token,dst).
// #17 -- WEAVE_LOCAL_SCATTER_DEDUP 0/1
//   1 = local dedup merge before send: the nexp expert rows are merged in FP32 registers and ONE row leaves
//       the wave (a nexp==1 token is a pure 128-bit copy).
//   0 = per-hit scatter: every local-expert hit pushes its own wide store of the running merge, so the
//       destination receives nexp write transactions for the same token (the last one carries the exact
//       FP32 merge).  The two knobs are orthogonal: the per-hit dataflow is only instantiated for the
//       direct/reg_2stage staging path, every other arm runs dedup=1.
//
// Nothing here adds a public environment knob, a kernarg field, or a shared-memory array: the ABI stays
// byte-identical and the static LDS image stays at the single 65536 B arena.
// =============================================================================================================
#ifndef WEAVE_SCATTER_STAGING
#define WEAVE_SCATTER_STAGING 0
#endif
#ifndef WEAVE_LOCAL_SCATTER_DEDUP
#define WEAVE_LOCAL_SCATTER_DEDUP 0
#endif
#if (WEAVE_SCATTER_STAGING) < 0 || (WEAVE_SCATTER_STAGING) > 2
#error "WEAVE_SCATTER_STAGING must be 0 (direct/reg_2stage), 1 (smem) or 2 (hbm_dedup)"
#endif
#if (WEAVE_LOCAL_SCATTER_DEDUP) != 0 && (WEAVE_LOCAL_SCATTER_DEDUP) != 1
#error "WEAVE_LOCAL_SCATTER_DEDUP must be 0 (per-hit scatter) or 1 (local dedup merge before send)"
#endif

// Every arm keeps the production wide-store cache policy (CPol SC0|SC1, write-through).  The staging path is
// the only thing the arm changes, so the measured difference isolates the dataflow.
#define V70_SST_STORE_POLICY 17

// ---- communication-only LDS staging window (path 1) ---------------------------------------------------------
#define V70_SST_LDS_BASE   8192u   // first byte after the compute-only M16/CK tile window
#define V70_SST_LDS_BUF    1024u   // 64 lanes x 16 B = one full coalesced 128-bit wave tile
#define V70_SST_LDS_STRIDE 3072u   // ping-pong pair + 1024 B pad, per wave of the communication CTA
static_assert(V70_SST_LDS_BASE + 8u * V70_SST_LDS_STRIDE <= 32768u,
              "V70 smem Scatter staging must stay inside the communication-only LDS window");
static_assert(V70_SST_LDS_BUF == 64u * 16u, "one LDS stage is one coalesced wave tile");

using V70SstU4 = unsigned __attribute__((ext_vector_type(4)));

// Every arm's wide 128-bit destination store uses one fixed write-through policy, so an arm only changes
// the staging dataflow.  v45_raw_buf_store_i32x4 is the compiler-hazard-tracked raw buffer store already
// used by the v45 tail (same SC0|SC1 policy as the inline "global_store_dwordx4 ... sc0 sc1").
__device__ __forceinline__ void v70_sst_bst(const WeaveI32x4& rs, uint32_t voff, uint4 v) {
  v45_raw_buf_store_i32x4(v45_i4(v), rs, static_cast<int32_t>(voff), 0, V70_SST_STORE_POLICY);
}

__device__ __forceinline__ uint4 v70_sst_unpack(V70SstU4 v) {
  return make_uint4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ V70SstU4 v70_sst_pack(uint4 v) {
  V70SstU4 r;
  r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w;
  return r;
}

// ---- path 1 primitives: the mandated ds_write_b128 -> s_waitcnt lgkmcnt(0) -> ds_read_b128 handoff --------
using V70SstLdsU4 = V70SstU4 __attribute__((address_space(3)));
__device__ __forceinline__ void v70_sst_ds_write_b128(unsigned off, uint4 v) {
  *reinterpret_cast<volatile V70SstLdsU4*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + off)) = v70_sst_pack(v);
}
__device__ __forceinline__ void v70_sst_lds_wait() {
  asm volatile("// V70_SST_LDS_WAIT\n\ts_waitcnt lgkmcnt(0)\n" ::: "memory");
}
__device__ __forceinline__ uint4 v70_sst_ds_read_b128(unsigned off) {
  const V70SstU4 r = *reinterpret_cast<volatile V70SstLdsU4*>(reinterpret_cast<uintptr_t>(g_v1262_phase_arena + off));
  asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
  return v70_sst_unpack(r);
}

// ---- path 2 primitives: packed bf16x2 hardware atomic merge inside the HBM row ------------------------------
__device__ __forceinline__ void v70_sst_pk_add_tile(void* p, uint4 v) {
  unsigned d0, d1, d2, d3;
  asm volatile("// V70_SST_PATH_HBM_DEDUP\n\tglobal_atomic_pk_add_bf16 %0, %1, %2, off sc0 sc1\n"
               : "=&v"(d0) : "v"(p), "v"(v.x) : "memory");
  asm volatile("// V70_SST_PATH_HBM_DEDUP\n\tglobal_atomic_pk_add_bf16 %0, %1, %2, off sc0 sc1\n"
               : "=&v"(d1) : "v"(static_cast<char*>(p) + 4), "v"(v.y) : "memory");
  asm volatile("// V70_SST_PATH_HBM_DEDUP\n\tglobal_atomic_pk_add_bf16 %0, %1, %2, off sc0 sc1\n"
               : "=&v"(d2) : "v"(static_cast<char*>(p) + 8), "v"(v.z) : "memory");
  asm volatile("// V70_SST_PATH_HBM_DEDUP\n\tglobal_atomic_pk_add_bf16 %0, %1, %2, off sc0 sc1\n"
               : "=&v"(d3) : "v"(static_cast<char*>(p) + 12), "v"(v.w) : "memory");
}
// Store/atomic data VGPRs are re-used by the next tile only after this drain (CDNA3 sc0|sc1 operand rule).
__device__ __forceinline__ void v70_sst_drain() {
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
}

// ---- one 128-bit lane tile: 8 bf16 columns of one expert row ------------------------------------------------
__device__ __forceinline__ uint4 v70_sst_copy_tile(const uint16_t* g2o, unsigned row, unsigned H, unsigned col) {
  const WeaveI32x4 rg = weave_make_buf_rsrc(g2o + static_cast<uint64_t>(row) * H, H * 2u);
  return v44_bld(rg, col * 2u);
}
__device__ __forceinline__ uint4 v70_sst_merge_tile(
    const uint16_t* g2o, const unsigned* rows, unsigned nexp, unsigned H, unsigned col) {
  float s[8] = {};
  for (unsigned k = 0u; k < nexp; ++k) {
    const unsigned rk = __builtin_amdgcn_readfirstlane(rows[k]);
    const WeaveI32x4 rg = weave_make_buf_rsrc(g2o + static_cast<uint64_t>(rk) * H, H * 2u);
    v42_acc_bf16x8(s, v44_bld(rg, col * 2u), true);
  }
  return v44_pack_bf16x8(s);
}

// =============================================================================================================
// #16 path 0 / #17 dedup=1 : double-buffered 128-bit register transfer, no staging memory at all
// =============================================================================================================
__device__ __forceinline__ void v70_sst_direct_reg2stage(
    const uint16_t* g2o, const unsigned* rows, unsigned nexp, unsigned H,
    uint16_t* dst, const WeaveI32x4& rd, unsigned lane) {
  if (nexp == 1u) {                                     // dedup==1: nothing to merge, pure wide copy
    const unsigned r0 = __builtin_amdgcn_readfirstlane(rows[0]);
    const WeaveI32x4 rg = weave_make_buf_rsrc(g2o + static_cast<uint64_t>(r0) * H, H * 2u);
    unsigned col = lane * 8u;
    if (col >= H) return;
    uint4 stage_a = v44_bld(rg, col * 2u);              // stage A
    for (col += 512u; col < H; col += 512u) {
      uint4 stage_b = v44_bld(rg, col * 2u);            // stage B: next tile's VMEM reads issue first
      v70_sst_bst(rd, (col - 512u) * 2u, stage_a);          // A leaves the registers for the peer slot
      stage_a = stage_b;                                // rotate (LLVM coalesces the copy)
    }
    v70_sst_bst(rd, (col - 512u) * 2u, stage_a);
    return;
  }
  unsigned col = lane * 8u;
  if (col >= H) return;
  uint4 stage_a = v70_sst_merge_tile(g2o, rows, nexp, H, col);
  for (col += 512u; col < H; col += 512u) {
    uint4 stage_b = v70_sst_merge_tile(g2o, rows, nexp, H, col);
    v70_sst_bst(rd, (col - 512u) * 2u, stage_a);
    stage_a = stage_b;
  }
  v70_sst_bst(rd, (col - 512u) * 2u, stage_a);
}

// =============================================================================================================
// #16 path 1 : communication-wave LDS staging (ds_write_b128 -> lgkmcnt(0) -> ds_read_b128 -> wide store)
// =============================================================================================================
__device__ __forceinline__ void v70_sst_smem(
    const uint16_t* g2o, const unsigned* rows, unsigned nexp, unsigned H,
    uint16_t* dst, const WeaveI32x4& rd, unsigned lane) {
  const unsigned lds = V70_SST_LDS_BASE + (threadIdx.x >> 6u) * V70_SST_LDS_STRIDE + lane * 16u;
  unsigned sel = 0u;                                    // ping-pong between the two 1024 B stages
  unsigned col = lane * 8u;
  if (col >= H) return;
  if (nexp == 1u) {
    const unsigned r0 = __builtin_amdgcn_readfirstlane(rows[0]);
    const WeaveI32x4 rg = weave_make_buf_rsrc(g2o + static_cast<uint64_t>(r0) * H, H * 2u);
    uint4 stage = v44_bld(rg, col * 2u);
    for (; col < H; col += 512u, sel ^= V70_SST_LDS_BUF) {
      uint4 nxt = (col + 512u < H) ? v44_bld(rg, (col + 512u) * 2u) : make_uint4(0u, 0u, 0u, 0u);
      v70_sst_ds_write_b128(lds + sel, stage);          // coalesced 128-bit ds_write_b128
      v70_sst_lds_wait();                               // s_waitcnt lgkmcnt(0)
      v70_sst_bst(rd, col * 2u, v70_sst_ds_read_b128(lds + sel));   // ds_read_b128 -> wide 128-bit store
      stage = nxt;
    }
    return;
  }
  uint4 stage = v70_sst_merge_tile(g2o, rows, nexp, H, col);
  for (; col < H; col += 512u, sel ^= V70_SST_LDS_BUF) {
    uint4 nxt = (col + 512u < H) ? v70_sst_merge_tile(g2o, rows, nexp, H, col + 512u)
                                 : make_uint4(0u, 0u, 0u, 0u);
    v70_sst_ds_write_b128(lds + sel, stage);
    v70_sst_lds_wait();
    v70_sst_bst(rd, col * 2u, v70_sst_ds_read_b128(lds + sel));
    stage = nxt;
  }
}

// =============================================================================================================
// #16 path 2 : HBM-resident dedup -- one deduplicated row per (token,dst), packed bf16x2 atomic merge
// =============================================================================================================
__device__ __forceinline__ void v70_sst_hbm_dedup(
    const uint16_t* g2o, const unsigned* rows, unsigned nexp, unsigned H,
    uint16_t* dst, const WeaveI32x4& rd, unsigned lane) {
  const unsigned col0 = lane * 8u;
  if (col0 >= H) return;
  // hit 0 seeds the HBM row with a wide 128-bit store stream (stage A / stage B register pipeline).
  const unsigned r0 = __builtin_amdgcn_readfirstlane(rows[0]);
  {
    const WeaveI32x4 rg = weave_make_buf_rsrc(g2o + static_cast<uint64_t>(r0) * H, H * 2u);
    uint4 stage_a = v44_bld(rg, col0 * 2u);
    unsigned col = col0;
    for (col += 512u; col < H; col += 512u) {
      uint4 stage_b = v44_bld(rg, col * 2u);
      v70_sst_bst(rd, (col - 512u) * 2u, stage_a);
      stage_a = stage_b;
    }
    v70_sst_bst(rd, (col - 512u) * 2u, stage_a);
  }
  if (nexp == 1u) return;
  // every further local-expert hit is folded into the same HBM row by the packed bf16x2 hardware atomic.
  for (unsigned k = 1u; k < nexp; ++k) {
    const unsigned rk = __builtin_amdgcn_readfirstlane(rows[k]);
    const WeaveI32x4 rg = weave_make_buf_rsrc(g2o + static_cast<uint64_t>(rk) * H, H * 2u);
    for (unsigned c0 = 0u; c0 < H; c0 += 2048u) {
      uint4 stage[4];
      #pragma unroll
      for (unsigned s = 0u; s < 4u; ++s) {
        const unsigned col = c0 + s * 512u + col0;
        stage[s] = make_uint4(0u, 0u, 0u, 0u);
        if (col < H) stage[s] = v44_bld(rg, col * 2u);
      }
      #pragma unroll
      for (unsigned s = 0u; s < 4u; ++s) {
        const unsigned col = c0 + s * 512u + col0;
        if (col < H) v70_sst_pk_add_tile(dst + col, stage[s]);
      }
      v70_sst_drain();                                  // CDNA3 sc0|sc1 operand reuse barrier
    }
  }
}

// =============================================================================================================
// #17 dedup=0 : per-hit scatter -- each local-expert hit pushes its own wide store of the running merge
// =============================================================================================================
__device__ __forceinline__ void v70_sst_perhit(
    const uint16_t* g2o, const unsigned* rows, unsigned nexp, unsigned H,
    uint16_t* dst, const WeaveI32x4& rd, unsigned lane) {
  unsigned col = lane * 8u;
  if (col >= H) return;
  if (nexp == 1u) {                                     // one hit: identical to the merged single push
    const unsigned r0 = __builtin_amdgcn_readfirstlane(rows[0]);
    const WeaveI32x4 rg = weave_make_buf_rsrc(g2o + static_cast<uint64_t>(r0) * H, H * 2u);
    for (; col < H; col += 512u) v70_sst_bst(rd, col * 2u, v44_bld(rg, col * 2u));
    return;
  }
  for (; col < H; col += 512u) {
    float s[8] = {};
    for (unsigned k = 0u; k < nexp; ++k) {
      const unsigned rk = __builtin_amdgcn_readfirstlane(rows[k]);
      const WeaveI32x4 rg = weave_make_buf_rsrc(g2o + static_cast<uint64_t>(rk) * H, H * 2u);
      v42_acc_bf16x8(s, v44_bld(rg, col * 2u), true);
      if (k + 1u < nexp) v70_sst_bst(rd, col * 2u, v44_pack_bf16x8(s));   // per-hit push of the running merge
    }
    v70_sst_bst(rd, col * 2u, v44_pack_bf16x8(s));                       // final exact FP32 merge
  }
}

// =============================================================================================================
// Scatter data entry point (called once per claimed (src,token) inside the communication Scatter owner)
// =============================================================================================================
__device__ __forceinline__ void v70_scatter_stage_rows(
    const uint16_t* g2o, const unsigned* rows, unsigned nexp, unsigned H,
    uint16_t* dst, unsigned lane) {
  const WeaveI32x4 rd = weave_make_buf_rsrc(dst, H * 2u);
#if (WEAVE_SCATTER_STAGING) == 2
  v70_sst_hbm_dedup(g2o, rows, nexp, H, dst, rd, lane);
#elif (WEAVE_SCATTER_STAGING) == 1
  v70_sst_smem(g2o, rows, nexp, H, dst, rd, lane);
#elif (WEAVE_LOCAL_SCATTER_DEDUP) == 0
  v70_sst_perhit(g2o, rows, nexp, H, dst, rd, lane);
#else
  v70_sst_direct_reg2stage(g2o, rows, nexp, H, dst, rd, lane);
#endif
}
