#pragma once
#include "v43_util.hpp"
// ---------------------------------------------------------------------------
// V43C helpers: CDNA3-safe polling + 4-wave-group-uniform decisions.
//   * spins use RELAXED loads only, ONE acquire fence after the condition holds;
//   * every spin escapes on the sticky device error and on timeout;
//   * group exits / branches are decided by tid0 and broadcast through LDS slot
//     65496 (+32 for waves 4..7), which no other v4 path uses.
// ---------------------------------------------------------------------------
#define V43C_SLOT0 65496u
#define V43C_SLOT1 65500u
#define V43C_DIAG_SCAN 960u
#define V43C_DIAG_SPIN 976u
#define V43C_DIAG_WRB 992u
__device__ __forceinline__ bool v43c_spin_ge(const unsigned* p, unsigned target, unsigned* err,
                                             unsigned long long start, unsigned long long tmo,
                                             unsigned code) {
  unsigned sp = 0u;
  while (__hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < target) {
    if ((sp & 31u) == 0u) {
      if (ep16_load_relaxed(err)) return false;
      if (__builtin_amdgcn_s_memrealtime() - start > tmo) { v983_error(err, code); return false; }
    }
    if (sp < 64u) asm volatile("s_nop 0\n" ::: "memory");
    else __builtin_amdgcn_s_sleep(1);
    ++sp;
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  return ep16_load_relaxed(err) == 0u;
}
__device__ __forceinline__ unsigned v43c_bcast_slot(unsigned v, unsigned slot, unsigned char* arena) {
  const unsigned grp_sh = (threadIdx.x >= 256u) ? 32u : 0u;
  volatile WeaveBk64LocalU32* sh = reinterpret_cast<volatile WeaveBk64LocalU32*>(
      reinterpret_cast<uintptr_t>(g_v1262_phase_arena + slot + grp_sh));
  if ((threadIdx.x & 255u) == 0u) sh[0] = v;
  weave_4wave_barrier_65440(arena);
  return __builtin_amdgcn_readfirstlane(sh[0]);
}
// leading barrier: the previous broadcast in SLOT0 is fully consumed before reuse.
__device__ __forceinline__ bool v43c_group_ok(bool ok_tid0, unsigned char* arena) {
  weave_4wave_barrier_65440(arena);
  return v43c_bcast_slot(ok_tid0 ? 1u : 0u, V43C_SLOT0, arena) != 0u;
}
// ============================================================================
// V43 SINGLE-PLAN 3-PASS SANDWICH  (Racer B / deepseek-v4.1-flash-1017)
//
// One WeaveMTileDesc array, one route plan, zero inter-pass grid barriers:
//   logical order : Pass0 local-half0  ->  Pass1 remote  ->  Pass2 local-half1
//   descriptor map: P0 at [0,T0) ; P1 at [T0,T0+T1) ; P2 parked at [P2BASE,..)
//   row map       : P0 rows [0,R0) ; P2 rows [R0,R0+R2) ; P1 rows [R0+R2,..)
// P2 is parked behind a conservative P1 reservation so that BOTH local halves
// can be planned at t~5us (xGMI only) while P1 is filled later by a dedicated
// helper wave the instant inter-node metadata lands.  There is no relocation,
// no second plan and no re-zeroing: the reservation is skipped by the mapping.
// ============================================================================

// ---------------------------------------------------------------------------
// wait for one node's metadata (8 peers) + scan local_nexp for those srcs
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool v43_scan_range(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, unsigned grp_id, unsigned n_comp_groups,
    unsigned src_begin, unsigned src_end, unsigned flag_slot) {
  const auto& e = env.base;
  const unsigned tid = threadIdx.x & 255u;
  const unsigned NT = x.node_tokens;
  const unsigned m = e.rank;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  auto* src_rows = reinterpret_cast<unsigned*>(e.source_bucket_rows);
  const auto* peer_prog_tbl = reinterpret_cast<const uint64_t*>(e.peer_progress_table);
  const auto* peer_meta_tbl = reinterpret_cast<const uint64_t*>(e.peer_meta_table);
  auto* local_nexp = reinterpret_cast<unsigned*>(o.home_slots);
  const auto start = __builtin_amdgcn_s_memrealtime();
  const unsigned nsrc = src_end - src_begin;

  if (grp_id == 0u && tid < nsrc) {
    const unsigned src = src_begin + tid;
    const auto* p_rdy = reinterpret_cast<const unsigned*>(peer_prog_tbl[src]);
    unsigned sp = 0u, last43 = 0u;
    // V43C C1: RELAXED system-scope poll (global_load sc0 sc1), ONE acquire after exit.
    for (;;) {
      last43 = v42_ld_sys_rlx(p_rdy);
      if (last43 == V809_PROGRESS_MAGIC) break;
      if ((sp & 63u) == 0u &&
          (ep16_load_relaxed((unsigned*)e.error) ||
           (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks))) {
        v983_error((unsigned*)e.error, 915u);
        break;
      }
      if (sp < 128u) asm volatile("s_nop 0\n" ::: "memory");
      else __builtin_amdgcn_s_sleep(1);
      ++sp;
    }
    v42_acq_sys();
    pf[V43C_DIAG_SCAN + src] = last43;   // V43C D1
    pf[V43C_DIAG_SPIN + src] = sp;
    if (!ep16_load_relaxed((unsigned*)e.error)) {
      const uint64_t tl = v41_ld_sc1_u64(reinterpret_cast<const void*>(peer_meta_tbl[src]));
      src_rows[src] = static_cast<unsigned>(tl);
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (grp_id == 0u && tid == 0u) {
    __threadfence();
    // V43C C3: the flag VALUE carries the verdict (1 ok / 2 failed) -> uniform exits.
    v41_st_agent_u32(pf + flag_slot, ep16_load_relaxed((unsigned*)e.error) ? 2u : 1u);
  }
  bool ok43 = true;
  if (tid == 0u) {
    unsigned f43 = 0u, sp2 = 0u;
    while ((f43 = v42_ld_agent_rlx(pf + flag_slot)) == 0u) {
      if ((sp2 & 31u) == 0u && ep16_load_relaxed((unsigned*)e.error)) break;
      __builtin_amdgcn_s_sleep(1);
      ++sp2;
    }
    v42_acq_agent();
    ok43 = (f43 == 1u) && !ep16_load_relaxed((unsigned*)e.error);
  }
  if (!v43c_group_ok(ok43, arena)) return false;

  const unsigned span = src_end - src_begin;
  const unsigned total_tok = span * NT;
  const unsigned gtid = grp_id * 256u + tid;
  const unsigned gstr = n_comp_groups * 256u;
  for (unsigned idx = gtid; idx < total_tok; idx += gstr) {
    const unsigned src = src_begin + idx / NT;
    const unsigned t = idx - (idx / NT) * NT;
    unsigned nexp = 0u;
    if (t < src_rows[src]) {
      const auto* tm = reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[src] + 64ull) + t;
      uint4 v0, v1;
      v41_ld2_sc1_u4(reinterpret_cast<const unsigned char*>(tm) + 0u,
                     reinterpret_cast<const unsigned char*>(tm) + 16u, v0, v1);
      const uint4 vz = make_uint4(0u, 0u, 0u, 0u);
      #pragma unroll
      for (unsigned k = 0u; k < 12u; ++k) {
        if (k < e.topk) {
          const uint16_t em = v41_pick_u16(v0, v1, vz, k);
          if (em != 0xffffu && (em >> 12u) == m) ++nexp;
        }
      }
    }
    local_nexp[static_cast<uint64_t>(src) * NT + t] = nexp;
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  return true;
}

// ---------------------------------------------------------------------------
// scatter one token half into ids / weights / row_route_id / bucket rows
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool v43_scatter_half(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, unsigned grp_id, unsigned n_comp_groups,
    unsigned src_begin, unsigned src_end, unsigned t_lo, unsigned t_hi,
    unsigned cnt_base, unsigned off_base, unsigned init_slot, unsigned done_slot) {
  const auto& e = env.base;
  const auto& ck = env.ck;
  const unsigned tid = threadIdx.x & 255u;
  const unsigned NT = x.node_tokens;
  const unsigned m = e.rank;
  const unsigned LE = ck.ck_local_experts;
  const bool fp16 = (e.flags & 2u) != 0u;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const auto* src_rows = reinterpret_cast<const unsigned*>(e.source_bucket_rows);
  const auto* peer_meta_tbl = reinterpret_cast<const uint64_t*>(e.peer_meta_table);
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
  auto* exp_cursors = reinterpret_cast<unsigned*>(e.expert_cursors);
  auto* s1_ids = reinterpret_cast<int32_t*>(ck.ck_sorted_token_ids);
  auto* s2_ids = reinterpret_cast<int32_t*>(ck.ck_m64_sorted_ids);
  auto* r_rid = reinterpret_cast<unsigned*>(o.row_route_id);
  auto* r_wt = reinterpret_cast<float*>(o.stage2_weights);
  auto* b_rows = reinterpret_cast<unsigned*>(o.bucket_route_rows);

  const unsigned long long st43 = __builtin_amdgcn_s_memrealtime();
  if (grp_id == 0u && tid == 0u) {
    const unsigned* offs = (off_base == 0u)
        ? reinterpret_cast<const unsigned*>(e.expert_offsets) : (pf + off_base);
    for (unsigned lid = 0u; lid < LE; ++lid) exp_cursors[lid] = offs[lid];
    __threadfence();
    v41_st_agent_u32(pf + init_slot, 1u);
  }
  bool ok43 = true;
  if (tid == 0u) ok43 = v43c_spin_ge(pf + init_slot, 1u, (unsigned*)e.error, st43, x.timeout_ticks, 955u);
  if (!v43c_group_ok(ok43, arena)) return false;

  (void)cnt_base;
  const unsigned span = src_end - src_begin;
  const unsigned nt_b = t_hi - t_lo;
  const unsigned nkeys = span * nt_b;
  const unsigned gtid = grp_id * 256u + tid;
  const unsigned gstr = n_comp_groups * 256u;
  for (unsigned idx = gtid; idx < nkeys; idx += gstr) {
    const unsigned src = src_begin + idx / nt_b;
    const unsigned t = t_lo + (idx - (idx / nt_b) * nt_b);
    if (t >= src_rows[src]) continue;
    if (local_nexp[static_cast<uint64_t>(src) * NT + t] == 0u) continue;
    const auto* tm = reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[src] + 64ull) + t;
    uint4 v0, v1, v2;
    v41_ld3_sc1_u4(reinterpret_cast<const unsigned char*>(tm) + 0u,
                   reinterpret_cast<const unsigned char*>(tm) + 16u,
                   reinterpret_cast<const unsigned char*>(tm) + 32u, v0, v1, v2);
    const unsigned key = src * e.capacity + t;
    unsigned k_loc = 0u;
    #pragma unroll
    for (unsigned k = 0u; k < 12u; ++k) {
      if (k < e.topk) {
        const uint16_t em = v41_pick_u16(v0, v1, v2, k);
        if (em != 0xffffu && (em >> 12u) == m) {
          const unsigned lid = em & 0xfffu;
          if (lid < LE) {
            const uint16_t wb = v41_pick_u16(v0, v1, v2, 12u + k);
            const unsigned row = __hip_atomic_fetch_add(exp_cursors + lid, 1u,
                                                        __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            s1_ids[row] = static_cast<int32_t>(key);
            s2_ids[row] = static_cast<int32_t>(row);
            r_rid[row] = (src * (e.capacity + 1u) + t) | (k_loc << 24u);
            r_wt[row] = weave_weight_value(wb, fp16);
            b_rows[static_cast<uint64_t>(key) * e.topk + k_loc] = row;  // g2_base == 0
            ++k_loc;
          }
        }
      }
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (tid == 0u) {
    __threadfence();
    __hip_atomic_fetch_add(pf + done_slot, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
  }
  if (grp_id == 0u && tid == 0u) {
    if (v43c_spin_ge(pf + done_slot, n_comp_groups, (unsigned*)e.error, st43, x.timeout_ticks, 956u)) {
      __threadfence();
      v41_st_agent_u32(pf + done_slot, 0x80000000u | n_comp_groups);
    }
  }
  ok43 = true;
  if (tid == 0u) ok43 = v43c_spin_ge(pf + done_slot, 0x80000000u, (unsigned*)e.error, st43, x.timeout_ticks, 957u);
  return v43c_group_ok(ok43, arena);
}

// ---------------------------------------------------------------------------
// V43B pass-scoped payload pull: stage every valid row of [row_lo,row_hi) into
// own RECV_PAYLOAD (UC arena, native pipelined uint4 stores) so the GEMM1
// cached-A shadow copy never reads stale data. ALL reads are node-local xGMI:
// peer_payload_table[src] names the local proxy holder for remote srcs.
// worker = wave id in [0,nworkers); each wave = 64 lanes. Last wave to arrive
// publishes rdy_slot (AGENT release) after one threadfence.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void v43_pull_rows(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned row_lo, unsigned row_hi, unsigned worker, unsigned nworkers,
    unsigned arr_slot, unsigned rdy_slot) {
  const auto& e = env.base;
  const auto& ck = env.ck;
  const unsigned lane = threadIdx.x & 63u;
  const unsigned m = e.rank;
  const unsigned PB = e.payload_bytes;
  const unsigned SG = e.scale_groups;
  // V43B spill-lean: wave-uniform scalars -> SGPR; NO hoisted 64-bit local bases
  // (dx/ds recomputed per row from the kernarg table) to keep VGPR liveness low.
  const unsigned lo43 = __builtin_amdgcn_readfirstlane(row_lo);
  const unsigned hi43 = __builtin_amdgcn_readfirstlane(row_hi);
  const unsigned wk43 = __builtin_amdgcn_readfirstlane(worker);
  const unsigned nw43 = __builtin_amdgcn_readfirstlane(nworkers);
  const unsigned K41 = __builtin_amdgcn_readfirstlane(v41_kdisp(x.node_tokens));
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const auto* s1_ids = reinterpret_cast<const int32_t*>(ck.ck_sorted_token_ids);
  const auto* peer_pay_tbl = reinterpret_cast<const uint64_t*>(e.peer_payload_table);
  const auto* peer_scl_tbl = reinterpret_cast<const uint64_t*>(e.peer_scale_table);
  const auto* peer_rdy_tbl = reinterpret_cast<const uint64_t*>(o.peer_payload_ready_table);
  const unsigned num_tokens = e.world * e.capacity;
  const unsigned partner = m ^ 8u;
  const unsigned long long start = __builtin_amdgcn_s_memrealtime();  // V43C2 F2: u64 origin

  for (unsigned row = lo43 + wk43; row < hi43; row += nw43) {
    const int32_t key = s1_ids[row];
    if ((unsigned)key >= num_tokens) continue;  // padding row
    const unsigned src = (unsigned)key / e.capacity;
    const unsigned t = (unsigned)key - src * e.capacity;
    const unsigned t0 = (t / K41) * K41;
    const auto* rdy_p = reinterpret_cast<const unsigned*>(peer_rdy_tbl[src]) + t0;
    if (lane == 0u) {
      unsigned sp = 0u;
      while (v42_ld_sys_rlx(rdy_p) < 1u) {
        if ((sp & 63u) == 0u && (ep16_load_relaxed((unsigned*)e.error) ||
            (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks))) {
          v983_error((unsigned*)e.error, 949u); break;
        }
        if (sp < 64u) asm volatile("s_nop 0\n" ::: "memory");
        else __builtin_amdgcn_s_sleep(1);
        ++sp;
      } v42_acq_sys();
    }
    __syncwarp();
    if (ep16_load_relaxed((unsigned*)e.error)) break;
    // V43B: copy EVERY valid row (self / partner / proxy) straight into the cached
    // CK_A shadow — GEMM1 reads MALL-hot A, no per-tile UC copy, no CAS.
    const auto* sx = reinterpret_cast<const unsigned char*>(peer_pay_tbl[src]) +
                     static_cast<uint64_t>(t) * PB;
    auto* dx = v43_xshadow(ck) + static_cast<uint64_t>(key) * PB;
    unsigned off = lane * 16u;
    for (; off + 3072u < PB; off += 4096u) {
      uint4 v0, v1, v2, v3;
      v41_ld4_sc1_u4(sx + off, sx + off + 1024u, sx + off + 2048u, sx + off + 3072u, v0, v1, v2, v3);
      *reinterpret_cast<uint4*>(dx + off) = v0;
      *reinterpret_cast<uint4*>(dx + off + 1024u) = v1;
      *reinterpret_cast<uint4*>(dx + off + 2048u) = v2;
      *reinterpret_cast<uint4*>(dx + off + 3072u) = v3;
    }
    for (; off + 1024u < PB; off += 2048u) {
      uint4 v0, v1;
      v41_ld2_sc1_u4(sx + off, sx + off + 1024u, v0, v1);
      *reinterpret_cast<uint4*>(dx + off) = v0;
      *reinterpret_cast<uint4*>(dx + off + 1024u) = v1;
    }
    for (; off < PB; off += 1024u) {
      *reinterpret_cast<uint4*>(dx + off) = v41_ld_sc1_u4(sx + off);
    }
    const auto* ss = reinterpret_cast<const float*>(peer_scl_tbl[src]) +
                     static_cast<uint64_t>(t) * SG;
    auto* ds = reinterpret_cast<float*>(v43_xshadow(ck) +
               static_cast<uint64_t>(num_tokens) * PB) + static_cast<uint64_t>(key) * SG;
    for (unsigned g = lane; g < SG; g += 64u) ds[g] = v41_ld_sc1_f32(ss + g);
  }
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  __syncwarp();
  if (lane == 0u) {
    __threadfence();
    const unsigned arr = __hip_atomic_fetch_add(pf + arr_slot, 1u,
                                                __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (arr == nw43) v41_st_agent_u32(pf + rdy_slot, 1u);
  }
  __syncwarp();
}

// ---------------------------------------------------------------------------
// LOCAL plan: P0 (local half 0) + P2 (local half 1) in one histogram pass
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool v43_plan_local(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, unsigned grp_id, unsigned n_comp_groups) {
  const auto& e = env.base;
  const auto& ck = env.ck;
  const unsigned tid = threadIdx.x & 255u;
  const unsigned NT = x.node_tokens;
  const unsigned m = e.rank;
  const unsigned cap = e.capacity;
  const unsigned half = cap >> 1u;
  const unsigned LE = ck.ck_local_experts;
  const unsigned RR = ck.ck_route_rows;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const auto* src_rows = reinterpret_cast<const unsigned*>(e.source_bucket_rows);
  const auto* peer_meta_tbl = reinterpret_cast<const uint64_t*>(e.peer_meta_table);
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
  auto* counts0 = reinterpret_cast<unsigned*>(e.expert_counts);
  auto* counts2 = pf + V43_CNT2_BASE;
  auto* offs0 = reinterpret_cast<unsigned*>(e.expert_offsets);
  auto* offs2 = pf + V43_OFF2_BASE;
  auto* curs = reinterpret_cast<unsigned*>(e.expert_cursors);
  auto* plan = reinterpret_cast<WeaveTilePlanTable*>(ck.ck_a);
  auto* descs = reinterpret_cast<WeaveMTileDesc*>(ck.ck_a + 64u);
  const unsigned sb = (m >> 3u) * 8u, se = sb + 8u;

  if (LE > 255u) { v983_error(reinterpret_cast<unsigned*>(e.error), 943u); return false; }
  if (!v43_scan_range(env, o, x, arena, grp_id, n_comp_groups, sb, se, V43_PF_M0)) return false;

  // ---- zero both count arrays ----
  const unsigned long long st43 = __builtin_amdgcn_s_memrealtime();
  if (grp_id == 0u) {
    for (unsigned lid = tid; lid < LE; lid += 256u) { counts0[lid] = 0u; counts2[lid] = 0u; }
    __builtin_amdgcn_s_waitcnt(0);
    weave_4wave_barrier_65440(arena);
    if (tid == 0u) { __threadfence(); v41_st_agent_u32(pf + V43_PF_GO0, 1u); }
  }
  {
    bool ok43 = true;
    if (tid == 0u) ok43 = v43c_spin_ge(pf + V43_PF_GO0, 1u, (unsigned*)e.error, st43, x.timeout_ticks, 958u);
    if (!v43c_group_ok(ok43, arena)) return false;
  }

  // ---- one histogram for both local halves ----
  const unsigned nkeys = 8u * NT;
  const unsigned gtid = grp_id * 256u + tid;
  const unsigned gstr = n_comp_groups * 256u;
  for (unsigned idx = gtid; idx < nkeys; idx += gstr) {
    const unsigned src = sb + idx / NT;
    const unsigned t = idx - (idx / NT) * NT;
    if (t >= src_rows[src] || t >= cap) continue;
    if (local_nexp[static_cast<uint64_t>(src) * NT + t] == 0u) continue;
    const auto* tm = reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[src] + 64ull) + t;
    uint4 v0, v1;
    v41_ld2_sc1_u4(reinterpret_cast<const unsigned char*>(tm) + 0u,
                   reinterpret_cast<const unsigned char*>(tm) + 16u, v0, v1);
    const uint4 vz = make_uint4(0u, 0u, 0u, 0u);
    unsigned* dst = (t < half) ? counts0 : counts2;
    #pragma unroll
    for (unsigned k = 0u; k < 12u; ++k) {
      if (k < e.topk) {
        const uint16_t em = v41_pick_u16(v0, v1, vz, k);
        if (em != 0xffffu && (em >> 12u) == m) {
          const unsigned lid = em & 0xfffu;
          if (lid < LE) __hip_atomic_fetch_add(dst + lid, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (tid == 0u) { __threadfence(); __hip_atomic_fetch_add(pf + V43_PF_GOC0, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT); }

  // ---- grp 0 builds descriptors for both halves ----
  if (grp_id == 0u) {
    if (tid == 0u && v43c_spin_ge(pf + V43_PF_GOC0, n_comp_groups, (unsigned*)e.error, st43,
                                   x.timeout_ticks, 953u)) {
      unsigned cur = 0u, tiles = 0u, valid = 0u;
      for (unsigned lid = 0u; lid < LE; ++lid) {
        const unsigned cnt = counts0[lid];
        offs0[lid] = cur; curs[lid] = cur; valid += cnt;
        unsigned rem = cnt;
        while (rem > 0u) {
          const unsigned bm = (rem > 96u) ? 128u : ((rem > 32u) ? 64u : 32u);  // V43B FIX1
          const unsigned vr = (rem > bm) ? bm : rem;
          WeaveMTileDesc d{};
          d.expert_id = lid; d.row_start = cur; d.bm = (uint16_t)bm;
          d.valid_rows = (uint16_t)vr; d.tile_idx = tiles;
          descs[tiles++] = d;
          cur += (vr + 15u) & ~15u;
          rem -= vr;
        }
      }
      offs0[LE] = cur;
      const unsigned R0 = cur, T0 = tiles;
      unsigned cur2 = R0, tiles2 = 0u, valid2 = 0u;
      for (unsigned lid = 0u; lid < LE; ++lid) {
        const unsigned cnt = counts2[lid];
        offs2[lid] = cur2; valid2 += cnt;
        unsigned rem = cnt;
        while (rem > 0u) {
          const unsigned bm = (rem > 96u) ? 128u : ((rem > 32u) ? 64u : 32u);  // V43B FIX1
          const unsigned vr = (rem > bm) ? bm : rem;
          ++tiles2;
          cur2 += (vr + 15u) & ~15u;
          rem -= vr;
        }
      }
      offs2[LE] = cur2;
      const unsigned R2 = cur2 - R0, T2 = tiles2;
      const unsigned t1ub = ((RR > (R0 + R2)) ? ((RR - (R0 + R2)) / 32u) : 0u) + 8u;
      const unsigned pbase = T0 + t1ub;
      const bool fit43 = (pbase + T2 <= (RR / 16u));
      if (!fit43) { v983_error(reinterpret_cast<unsigned*>(e.error), 944u); }
      // write P2 descriptors straight into the parked region: zero relocation.
      unsigned ti = 0u;
      for (unsigned lid = 0u; fit43 && lid < LE; ++lid) {
        unsigned rs = offs2[lid];
        unsigned rem = counts2[lid];
        while (rem > 0u) {
          const unsigned bm = (rem > 96u) ? 128u : ((rem > 32u) ? 64u : 32u);  // V43B FIX1
          const unsigned vr = (rem > bm) ? bm : rem;
          WeaveMTileDesc d{};
          d.expert_id = lid;
          d.row_start = rs;
          d.bm = (uint16_t)bm;
          d.valid_rows = (uint16_t)vr;
          d.tile_idx = T0 + ti;
          descs[pbase + ti] = d;
          ++ti;
          rs += (vr + 15u) & ~15u;
          rem -= vr;
        }
      }
      pf[V43_PF_T0] = T0; pf[V43_PF_T2] = T2; pf[V43_PF_R0] = R0; pf[V43_PF_R2] = R2;
      pf[V43_PF_T1] = 0u; pf[V43_PF_P2BASE] = pbase;
      // CRITICAL: only Pass 0 is visible until the P1 plan lands; P2 must never
      // run before P1 or the logical->physical map would shift under the cursors.
      pf[V43_PF_TOTAL] = T0;
      plan->total_m_tiles = T0;
      plan->total_valid_rows = valid + valid2;
      plan->total_padded_rows = RR;
      plan->pad_gather_slot = e.world * e.capacity;
      for (unsigned bk = 0u; bk < 5u; ++bk) { plan->num_tiles[bk] = 0u; plan->bucket_offset[bk] = 0u; }
      __threadfence();
      v41_st_agent_u32(pf + V43_PF_GO0, 2u);
    }
  }
  {
    bool ok43 = true;
    if (tid == 0u) ok43 = v43c_spin_ge(pf + V43_PF_GO0, 2u, (unsigned*)e.error, st43, x.timeout_ticks, 959u);
    if (!v43c_group_ok(ok43, arena)) return false;
  }

  const unsigned T0 = pf[V43_PF_T0], T2 = pf[V43_PF_T2];
  const unsigned pbase = pf[V43_PF_P2BASE];
  auto* s1_done = reinterpret_cast<unsigned*>(o.stage1_tile_done);
  auto* rn_done = reinterpret_cast<unsigned*>(o.row_n_done);
  for (unsigned mt = tid; mt < T0; mt += 256u) { s1_done[mt] = 0u; rn_done[mt] = 0u; }
  for (unsigned mt = tid; mt < T2; mt += 256u) { s1_done[pbase + mt] = 0u; rn_done[pbase + mt] = 0u; }
  // padding rows (both halves)
  auto* s1_ids = reinterpret_cast<int32_t*>(ck.ck_sorted_token_ids);
  auto* s2_ids = reinterpret_cast<int32_t*>(ck.ck_m64_sorted_ids);
  auto* r_rid = reinterpret_cast<unsigned*>(o.row_route_id);
  auto* r_wt = reinterpret_cast<float*>(o.stage2_weights);
  for (unsigned lid = tid; lid < LE; lid += 256u) {
    for (unsigned pass = 0u; pass < 2u; ++pass) {
      const unsigned beg = pass ? offs2[lid] : offs0[lid];
      const unsigned end = pass ? offs2[lid + 1u] : offs0[lid + 1u];
      const unsigned cnt = pass ? counts2[lid] : counts0[lid];
      for (unsigned r = beg + cnt; r < end; ++r) {
        s1_ids[r] = static_cast<int32_t>(e.world * e.capacity);
        s2_ids[r] = static_cast<int32_t>(RR);
        r_rid[r] = 0xffffffffu;
        r_wt[r] = 0.0f;
      }
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);

  if (!v43_scatter_half(env, o, x, arena, grp_id, n_comp_groups, sb, se, 0u, half,
                        V43_CNT2_BASE, 0u, V43_PF_S0I, V43_PF_S0D)) return false;
  if (!v43_scatter_half(env, o, x, arena, grp_id, n_comp_groups, sb, se, half, cap,
                        V43_CNT2_BASE, V43_OFF2_BASE, V43_PF_S1I, V43_PF_S1D)) return false;

  // V43B FIX4: EVERY group arrives; only the group reaching n_comp_groups fences
  // and publishes LOCAL_DONE. Consumers wait LOCAL_DONE, never the raw counter.
  bool okl43 = true;
  if (tid == 0u) {
    __threadfence();
    const unsigned arr = __hip_atomic_fetch_add(pf + V43_PF_LOCAL_RDY, 1u,
                                                __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) + 1u;
    if (arr == n_comp_groups) {
      __threadfence();
      v41_st_agent_u32(pf + V43_PF_LOCAL_DONE, 1u);
      v41_st_agent_u32(pf + V43_PF_RDY, 1u);
    }
    okl43 = v43c_spin_ge(pf + V43_PF_LOCAL_DONE, 1u, (unsigned*)e.error, st43, x.timeout_ticks, 960u);
  }
  return v43c_group_ok(okl43, arena);
}

// ---------------------------------------------------------------------------
// REMOTE plan: P1 (all remote-node tokens). Runs on one dedicated 256-thread
// helper wave group while Pass 0 compute is already in flight.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool v43_plan_remote(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena) {
  const auto& e = env.base;
  const auto& ck = env.ck;
  const unsigned tid = threadIdx.x & 255u;
  const unsigned NT = x.node_tokens;
  const unsigned m = e.rank;
  const unsigned cap = e.capacity;
  const unsigned LE = ck.ck_local_experts;
  const unsigned RR = ck.ck_route_rows;
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  const auto* src_rows = reinterpret_cast<const unsigned*>(e.source_bucket_rows);
  const auto* peer_meta_tbl = reinterpret_cast<const uint64_t*>(e.peer_meta_table);
  const auto* local_nexp = reinterpret_cast<const unsigned*>(o.home_slots);
  auto* counts = reinterpret_cast<unsigned*>(e.expert_counts);
  auto* offs = reinterpret_cast<unsigned*>(e.expert_offsets);
  auto* curs = reinterpret_cast<unsigned*>(e.expert_cursors);
  auto* plan = reinterpret_cast<WeaveTilePlanTable*>(ck.ck_a);
  auto* descs = reinterpret_cast<WeaveMTileDesc*>(ck.ck_a + 64u);
  const unsigned sb = ((m >> 3u) ^ 1u) * 8u, se = sb + 8u;
  const auto start = __builtin_amdgcn_s_memrealtime();

  if (LE > 255u) { v983_error(reinterpret_cast<unsigned*>(e.error), 943u); return false; }
  // wait for the local plan (row bases / descriptor park position)
  // V43C C1/C3: ONE lane polls relaxed; the 4-wave group follows one LDS verdict.
  {
    bool ok43 = true;
    if (tid == 0u) ok43 = v43c_spin_ge(pf + V43_PF_LOCAL_DONE, 1u, (unsigned*)e.error, start,
                                       x.timeout_ticks, 945u);
    if (!v43c_group_ok(ok43, arena)) return false;
  }
  if (!v43_scan_range(env, o, x, arena, 0u, 1u, sb, se, V43_PF_M1)) return false;

  for (unsigned lid = tid; lid < LE; lid += 256u) counts[lid] = 0u;
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);

  // histogram (256 threads, one group)
  const unsigned nkeys = 8u * NT;
  for (unsigned idx = tid; idx < nkeys; idx += 256u) {
    const unsigned src = sb + idx / NT;
    const unsigned t = idx - (idx / NT) * NT;
    if (t >= src_rows[src] || t >= cap) continue;
    if (local_nexp[static_cast<uint64_t>(src) * NT + t] == 0u) continue;
    const auto* tm = reinterpret_cast<const V41TokenMeta*>(peer_meta_tbl[src] + 64ull) + t;
    uint4 v0, v1;
    v41_ld2_sc1_u4(reinterpret_cast<const unsigned char*>(tm) + 0u,
                   reinterpret_cast<const unsigned char*>(tm) + 16u, v0, v1);
    const uint4 vz = make_uint4(0u, 0u, 0u, 0u);
    #pragma unroll
    for (unsigned k = 0u; k < 12u; ++k) {
      if (k < e.topk) {
        const uint16_t em = v41_pick_u16(v0, v1, vz, k);
        if (em != 0xffffu && (em >> 12u) == m) {
          const unsigned lid = em & 0xfffu;
          if (lid < LE) __hip_atomic_fetch_add(counts + lid, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);

  const unsigned T0 = pf[V43_PF_T0];
  const unsigned R0 = pf[V43_PF_R0], R2 = pf[V43_PF_R2];
  const unsigned pbase = pf[V43_PF_P2BASE];
  const unsigned row_base = R0 + R2;
  unsigned T1 = 0u, R1 = 0u, valid = 0u;
  if (tid == 0u) {
    unsigned cur = row_base, tiles = 0u;
    for (unsigned lid = 0u; lid < LE; ++lid) {
      const unsigned cnt = counts[lid];
      offs[lid] = cur; curs[lid] = cur; valid += cnt;
      unsigned rem = cnt;
      while (rem > 0u) {
        const unsigned bm = (rem > 96u) ? 128u : ((rem > 32u) ? 64u : 32u);  // V43B FIX1
        const unsigned vr = (rem > bm) ? bm : rem;
        WeaveMTileDesc d{};
        d.expert_id = lid; d.row_start = cur; d.bm = (uint16_t)bm;
        d.valid_rows = (uint16_t)vr; d.tile_idx = T0 + tiles;
        if (T0 + tiles < pbase) descs[T0 + tiles] = d;   // V43C C6: never into parked P2
        ++tiles;
        cur += (vr + 15u) & ~15u;
        rem -= vr;
      }
    }
    offs[LE] = cur;
    T1 = tiles; R1 = cur - row_base;
    if (T0 + T1 > pbase) v983_error(reinterpret_cast<unsigned*>(e.error), 954u);
    pf[V43_PF_T1] = T1; pf[V43_PF_R1] = R1;
    // V43C C5: pf[TOTAL] stays T0 here; P1 tiles become claimable only after P1_PULL.
    plan->total_valid_rows += valid;
    plan->total_padded_rows = R0 + R2 + R1;
    __threadfence();
  }
  weave_4wave_barrier_65440(arena);
  T1 = pf[V43_PF_T1];

  auto* s1_done = reinterpret_cast<unsigned*>(o.stage1_tile_done);
  auto* rn_done = reinterpret_cast<unsigned*>(o.row_n_done);
  for (unsigned mt = tid; mt < T1; mt += 256u) { s1_done[T0 + mt] = 0u; rn_done[T0 + mt] = 0u; }
  auto* s1_ids = reinterpret_cast<int32_t*>(ck.ck_sorted_token_ids);
  auto* s2_ids = reinterpret_cast<int32_t*>(ck.ck_m64_sorted_ids);
  auto* r_rid = reinterpret_cast<unsigned*>(o.row_route_id);
  auto* r_wt = reinterpret_cast<float*>(o.stage2_weights);
  for (unsigned lid = tid; lid < LE; lid += 256u) {
    const unsigned beg = offs[lid], end = offs[lid + 1u], cnt = counts[lid];
    for (unsigned r = beg + cnt; r < end; ++r) {
      s1_ids[r] = static_cast<int32_t>(e.world * e.capacity);
      s2_ids[r] = static_cast<int32_t>(RR);
      r_rid[r] = 0xffffffffu;
      r_wt[r] = 0.0f;
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);

  if (!v43_scatter_half(env, o, x, arena, 0u, 1u, sb, se, 0u, cap,
                        0u, 0u, V43_PF_S2I, V43_PF_S2D)) return false;

  // V43B FIX4: publish the full plan + per-MB bucket-drain gates. pf[80+b] is the
  // exact non-empty key count over ALL 16 srcs; ep16_bucket_v41 decrements it and
  // raises pf[96+b] at zero. NEVER pre-set pf[96+b] (that would skip the scatter!).
  if (tid == 0u) {
    const unsigned M = v41_num_mb(NT);
    for (unsigned b = 0u; b < M; ++b) {
      unsigned cb_beg, cb_end, t_beg, t_end;
      v41_mb_range(NT, b, M, cb_beg, cb_end, t_beg, t_end);
      unsigned cnt = 0u;
      for (unsigned src = 0u; src < 16u; ++src)
        for (unsigned t = t_beg; t < t_end; ++t)
          if (local_nexp[static_cast<uint64_t>(src) * NT + t] != 0u) ++cnt;
      pf[80u + b] = cnt;
      if (cnt == 0u) v41_st_agent_u32(pf + 96u + b, 1u);
      __threadfence();
      v41_st_agent_u32(pf + 32u + b, 1u);  // plan ready for bucket drain
    }
    __threadfence();
    v41_st_agent_u32(pf + V43_PF_P1_PLAN, 1u);
  }
  // Wait until the 32 pull waves staged EVERY Pass-1 remote payload row.
  const unsigned long long start_p = __builtin_amdgcn_s_memrealtime();
  {
    bool ok43 = true;
    if (tid == 0u) ok43 = v43c_spin_ge(pf + V43_PF_P1_PULL, 1u, (unsigned*)e.error, start_p,
                                       x.timeout_ticks, 947u);
    if (!v43c_group_ok(ok43, arena)) return false;
  }
  if (tid == 0u) {
    // V43C C5: P1 becomes claimable only now (its xshadow rows are staged + released).
    __threadfence();
    plan->total_m_tiles = T0 + T1;
    v41_st_agent_u32(pf + V43_PF_TOTAL, T0 + T1);
    v41_st_agent_u32(pf + V43_PF_P1_RDY, 1u);
    v41_st_agent_u32(pf + V43_PF_RDY, 1u);
  }
  // Pass-2 local pull runs underneath Pass-1 compute; expose P2 only when staged.
  {
    bool ok43 = true;
    if (tid == 0u) ok43 = v43c_spin_ge(pf + V43_PF_P2_PULL, 1u, (unsigned*)e.error, start_p,
                                       x.timeout_ticks, 948u);
    if (!v43c_group_ok(ok43, arena)) return false;
  }
  if (tid == 0u) {
    const unsigned T0f = pf[V43_PF_T0], T1f = pf[V43_PF_T1], T2f = pf[V43_PF_T2];
    plan->total_m_tiles = T0f + T1f + T2f;
    __threadfence();
    v41_st_agent_u32(pf + V43_PF_TOTAL, T0f + T1f + T2f);
    v41_st_agent_u32(pf + V43_PF_FINAL, 1u);
  }
  return true;
}