#pragma once
#include "v43_util.hpp"
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
    unsigned sp = 0u;
    while (v41_ld_sys_u32(p_rdy) != V809_PROGRESS_MAGIC) {
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
    if (!ep16_load_relaxed((unsigned*)e.error)) {
      const uint64_t tl = v41_ld_sc1_u64(reinterpret_cast<const void*>(peer_meta_tbl[src]));
      src_rows[src] = static_cast<unsigned>(tl);
    }
  }
  __builtin_amdgcn_s_waitcnt(0);
  weave_4wave_barrier_65440(arena);
  if (grp_id == 0u && tid == 0u) {
    __threadfence();
    v41_st_agent_u32(pf + flag_slot, 1u);
  }
  if (tid == 0u) {
    while (v41_ld_agent_u32(pf + flag_slot) == 0u) {
      if (ep16_load_relaxed((unsigned*)e.error)) break;
      __builtin_amdgcn_s_sleep(1);
    }
  }
  weave_4wave_barrier_65440(arena);
  if (ep16_load_relaxed((unsigned*)e.error)) return false;

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
__device__ __forceinline__ void v43_scatter_half(
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

  if (grp_id == 0u && tid == 0u) {
    const unsigned* offs = (off_base == 0u)
        ? reinterpret_cast<const unsigned*>(e.expert_offsets) : (pf + off_base);
    for (unsigned lid = 0u; lid < LE; ++lid) exp_cursors[lid] = offs[lid];
    __threadfence();
    v41_st_agent_u32(pf + init_slot, 1u);
  }
  if (tid == 0u) { while (v41_ld_agent_u32(pf + init_slot) == 0u) asm volatile("s_nop 0\n" ::: "memory"); }
  weave_4wave_barrier_65440(arena);

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
    while (v41_ld_agent_u32(pf + done_slot) < n_comp_groups) asm volatile("s_nop 0\n" ::: "memory");
    __threadfence();
    v41_st_agent_u32(pf + done_slot, 0x80000000u | n_comp_groups);
  }
  if (tid == 0u) {
    while ((v41_ld_agent_u32(pf + done_slot) & 0x80000000u) == 0u) asm volatile("s_nop 0\n" ::: "memory");
  }
  weave_4wave_barrier_65440(arena);
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
  if (grp_id == 0u) {
    for (unsigned lid = tid; lid < LE; lid += 256u) { counts0[lid] = 0u; counts2[lid] = 0u; }
    __builtin_amdgcn_s_waitcnt(0);
    weave_4wave_barrier_65440(arena);
    if (tid == 0u) { __threadfence(); v41_st_agent_u32(pf + V43_PF_GO0, 1u); }
  }
  if (tid == 0u) { while (v41_ld_agent_u32(pf + V43_PF_GO0) == 0u) asm volatile("s_nop 0\n" ::: "memory"); }
  weave_4wave_barrier_65440(arena);

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
    if (tid == 0u) {
      while (v41_ld_agent_u32(pf + V43_PF_GOC0) < n_comp_groups) asm volatile("s_nop 0\n" ::: "memory");
      unsigned cur = 0u, tiles = 0u, valid = 0u;
      for (unsigned lid = 0u; lid < LE; ++lid) {
        const unsigned cnt = counts0[lid];
        offs0[lid] = cur; curs[lid] = cur; valid += cnt;
        unsigned rem = cnt;
        while (rem > 0u) {
          const unsigned bm = (rem > 32u) ? 64u : 32u;
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
          const unsigned bm = (rem > 32u) ? 64u : 32u;
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
      if (pbase + T2 > (RR / 16u)) { v983_error(reinterpret_cast<unsigned*>(e.error), 944u); }
      // write P2 descriptors straight into the parked region: zero relocation.
      unsigned ti = 0u;
      for (unsigned lid = 0u; lid < LE; ++lid) {
        unsigned rs = offs2[lid];
        unsigned rem = counts2[lid];
        while (rem > 0u) {
          const unsigned bm = (rem > 32u) ? 64u : 32u;
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
  if (tid == 0u) { while (v41_ld_agent_u32(pf + V43_PF_GO0) < 2u) asm volatile("s_nop 0\n" ::: "memory"); }
  weave_4wave_barrier_65440(arena);

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

  v43_scatter_half(env, o, x, arena, grp_id, n_comp_groups, sb, se, 0u, half,
                   V43_CNT2_BASE, 0u, V43_PF_S0I, V43_PF_S0D);
  v43_scatter_half(env, o, x, arena, grp_id, n_comp_groups, sb, se, half, cap,
                   V43_CNT2_BASE, V43_OFF2_BASE, V43_PF_S1I, V43_PF_S1D);

  if (grp_id == 0u && tid == 0u) {
    __threadfence();
    v41_st_agent_u32(pf + V43_PF_LOCAL_RDY, 1u);
    v41_st_agent_u32(pf + V43_PF_RDY, 1u);
  }
  if (tid == 0u) { while (v41_ld_agent_u32(pf + V43_PF_LOCAL_RDY) == 0u) __builtin_amdgcn_s_sleep(1); }
  weave_4wave_barrier_65440(arena);
  return !ep16_load_relaxed(reinterpret_cast<unsigned*>(e.error));
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
  while (v41_ld_agent_u32(pf + V43_PF_LOCAL_RDY) == 0u) {
    if (ep16_load_relaxed((unsigned*)e.error) ||
        (__builtin_amdgcn_s_memrealtime() - start > x.timeout_ticks)) {
      v983_error(reinterpret_cast<unsigned*>(e.error), 945u); return false;
    }
    __builtin_amdgcn_s_sleep(1);
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
        const unsigned bm = (rem > 32u) ? 64u : 32u;
        const unsigned vr = (rem > bm) ? bm : rem;
        WeaveMTileDesc d{};
        d.expert_id = lid; d.row_start = cur; d.bm = (uint16_t)bm;
        d.valid_rows = (uint16_t)vr; d.tile_idx = T0 + tiles;
        descs[T0 + tiles] = d;
        ++tiles;
        cur += (vr + 15u) & ~15u;
        rem -= vr;
      }
    }
    offs[LE] = cur;
    T1 = tiles; R1 = cur - row_base;
    pf[V43_PF_T1] = T1; pf[V43_PF_R1] = R1;
    pf[V43_PF_TOTAL] = T0 + T1 + pf[V43_PF_T2];
    plan->total_m_tiles = T0 + T1 + pf[V43_PF_T2];
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

  v43_scatter_half(env, o, x, arena, 0u, 1u, sb, se, 0u, cap,
                   0u, 0u, V43_PF_S2I, V43_PF_S2D);

  if (tid == 0u) {
    const unsigned M = v41_num_mb(NT);
    for (unsigned b = 0u; b < M; ++b) {
      v41_st_agent_u32(pf + 32u + b, 1u);  // plan ready for bucket drain
      v41_st_agent_u32(pf + 96u + b, 1u);  // scatter complete
    }
    __threadfence();
    v41_st_agent_u32(pf + V43_PF_P1_RDY, 1u);
    v41_st_agent_u32(pf + V43_PF_RDY, 1u);
  }
  return !ep16_load_relaxed(reinterpret_cast<unsigned*>(e.error));
}