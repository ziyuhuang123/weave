#pragma once
// =============================================================================================================
// V70 R8 (opus55 r8): Path B scheduler = R6 per-XCD expert-major swizzle (v70c_map + v70_xcd_work) plus
//   (1) hybrid communication CTAs (V70_R8_HYB): a Path B comm CTA with blockIdx in [V70_R8_HYB_LO, NCOMM_CTA)
//       claims GEMM work from the SAME cursors before it enters its Combine-tail role:
//         HYB=1 : GEMM1 only;  HYB=3 : GEMM1 + GEMM2 until the cursors are exhausted;
//         HYB=2 : GEMM1 + GEMM2, but before every GEMM2 claim tid 0 reads the Scatter backlog
//                 B = rows released by GEMM2 (pf[V70R8_RREL]) - rows scattered (pf[V70R8_RSCT]) and hybrid CTA k
//                 hands off for good once B >= V70_R8_BL0 + (k - LO) * V70_R8_BLS.
//   (2) fine GEMM2 tail (V70_R8_FT, permille): the last FT/1000 of the coarse GEMM2 items (n2g/2 units of 256
//       columns each) are split into single-unit items, so the end of GEMM2 is ragged by one 256-column unit
//       instead of one coarse item.  Fine items carry q = 1 in the decision word and run with n2g = 2.
//       Coverage: coarse slot g < Nc - F -> coarse item g; slot Nc - F + f -> unit f % S of coarse item
//       Nc - F + f / S (S = n2g / 2).  Every (tile, 256-column chunk) is executed exactly once and row_n_done
//       still sums to n2_blocks per tile.
//   (3) the launch-error word is polled on every 4th claim (and on every exit decision) instead of every claim.
// Deadlock freedom is the R6 argument: GEMM1 items only wait on the plan gate, a GEMM2 item only waits on GEMM1
// items that are already claimed, and a hybrid that hands off only stops claiming.
// pf[368]/pf[369] are free plan_flags words (in 352..383 only 353..367, 381, 382 are used).
// =============================================================================================================
#ifndef V70_R8_HYB
#define V70_R8_HYB 0
#endif
#ifndef V70R8_RREL
#define V70R8_RREL 368u
#define V70R8_RSCT 369u
#endif
#ifndef V70_R8_CNT
#define V70_R8_CNT ((V70_R8_HYB) == 2)
#endif
#ifndef V70_R8_SCHED
#define V70_R8_SCHED 0
#endif
#ifndef V70_R8_FT
#define V70_R8_FT 0u
#endif
#ifndef V70_R8_ERRP
#define V70_R8_ERRP 4u
#endif
#ifndef V70_R8_HYB_LO
#define V70_R8_HYB_LO 1u
#endif
#ifndef V70_R8_BL0
#define V70_R8_BL0 32u
#endif
#ifndef V70_R8_BLS
#define V70_R8_BLS 32u
#endif
#ifndef V70_E3_QPART
#define V70_E3_QPART 0
#endif
#if V70_R2X && V70_R2X_44 && V70_R6_SCHED && (V70_R8_SCHED || V70_R8_HYB)
#define V70_R8_HYB_EFF (((WEAVE_SM0_STREAM_SCHED) == 0) ? 0 : (V70_R8_HYB))
#define V70_R8_FT_EFF  (((WEAVE_SM0_STREAM_SCHED) == 0) ? 0u : (V70_R8_FT))
#if V70_R8_2WAVE
#define V70P_CUR_BASE 288u  // 6 phases x 4 XCDs = pf[288..311], 100% free & zeroed every launch
__device__ __forceinline__ bool v70_r8_q0_ready(const unsigned* pf) {
  if (v42_ld_agent_rlx(pf + V46_Q0_RDY) == 0u) return false;
  v42_acq_agent();
  if (pf[V46_T + 0u] != 0u && v42_ld_agent_rlx(pf + V46_PULL_RDY + 0u) == 0u) return false;
  return true;
}
__device__ __forceinline__ bool v70_atrex4_r8(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, const unsigned role_in, const unsigned grp_or_cidx_in FI_CTX_PARAM) {
  const unsigned tid = threadIdx.x;
  volatile WeaveBk64LocalU32* sh = v70_8w_sh(V70_8W_SH_WORD);
  volatile WeaveBk64LocalU32* lam_sh = v70_8w_sh(V70_8W_SH_LAM);
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  {
    if (tid == 0u) {
      const unsigned n1b0 = env.ck.ck_intermediate / 128u;
      const unsigned n2b0 = o.hidden / 128u;
      const unsigned n2g0 = v46_n2g(n2b0);
      const unsigned n2c0 = (n2g0 == 8u) ? (n2b0 >> 3u) : ((n2g0 == 4u) ? (n2b0 >> 2u) : ((n2g0 == 2u) ? (n2b0 >> 1u) : n2b0));
      const unsigned hidx0 = (grp_or_cidx_in == 0xffffffffu) ? blockIdx.x : grp_or_cidx_in;
      const unsigned hrel0 = (hidx0 >= (V70_R8_HYB_LO)) ? (hidx0 - (V70_R8_HYB_LO)) : 0u;
      const unsigned thr0 = (V70_R8_BL0) + hrel0 * (V70_R8_BLS);
      v70_8w_sh(30768u)[0] = role_in;
      v70_8w_sh(30772u)[0] = thr0;
      v70_8w_sh(30776u)[0] = n1b0;
      v70_8w_sh(30780u)[0] = n2b0;
      v70_8w_sh(30784u)[0] = n2g0;
      v70_8w_sh(30788u)[0] = n2c0;
      lam_sh[0] = 0xffffffffu;
      unsigned ok = 1u, spins = 0u;
      while (!v70_r8_q0_ready(pf)) {
        __builtin_amdgcn_s_sleep(2);
        if (ep16_load_relaxed((unsigned*)env.base.error) != 0u) { ok = 0u; break; }
        if (++spins > (1u << 24)) { v983_error((unsigned*)env.base.error, 971u); ok = 0u; break; }
      }
      v42_acq_agent();
      const bool is_grp0 = (grp_or_cidx_in == 0xffffffffu) ? (blockIdx.x == (V70_PATHB_NCOMM_CTA)) : (grp_or_cidx_in == 0u);
      if (ok && role_in == 0u && is_grp0) {
        unsigned e0 = 0u;
        __hip_atomic_compare_exchange_strong(pf + V70M_MODE, &e0, 4u, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT);
      }
      sh[0] = ok;
    }
    r2x::bar<V70_R2X_4W_SW>();
    if (__builtin_amdgcn_readfirstlane(sh[0]) == 0u) return false;
    v42_acq_sys();
    if (tid == 0u) {
      v70_8w_sh(30740u)[0] = 0xffffffffu;  // cur_grp = none (first loop iter builds grp 0!)
      v70_8w_sh(30744u)[0] = 0u;           // xmask24 (6 phases x 4 XCDs)
      v70_8w_sh(30748u)[0] = 0u;           // nclaim
      v70_8w_sh(30752u)[0] = 0u;           // rdy_lat (bit0=loc_rdy, bit1=q3_rdy, bit2=p1_plan, bit3=rem_rdy)
      v70_8w_sh(30756u)[0] = 0u;           // idle_spins
    }
  }
  for (;;) {
    r2x::bar<V70_R2X_4W_SW>();
    if (tid == 0u) {
      const unsigned role = v70_8w_sh(30768u)[0];
      const unsigned n1b = v70_8w_sh(30776u)[0];
      const unsigned n2g = v70_8w_sh(30784u)[0];
      const unsigned n2c = v70_8w_sh(30788u)[0];
      const unsigned LE = v70_claim_le(env, pf);
      const unsigned nn1 = n1b / (V55_N1G);
      const unsigned xcd = v55_xcc() & 3u;
      const unsigned S = n2g >> 1u;
      const unsigned cur_grp = v70_8w_sh(30740u)[0];
      unsigned xm = v70_8w_sh(30744u)[0];
      const bool g2ok = (role == 0u) || ((V70_R8_HYB_EFF) >= 2);
      const unsigned all_done = g2ok ? 0x00ffffffu : 0x000000ffu;
      const unsigned nph = g2ok ? 6u : 2u;
      unsigned rl = v70_8w_sh(30752u)[0];
      if ((rl & 1u) == 0u && v42_ld_agent_rlx(pf + V46_LOCAL_RDY) != 0u) {
        v42_acq_agent();
        rl |= 1u;
      }
      if ((rl & 1u) != 0u) {
        if ((rl & 2u) == 0u && ((pf[V46_T + 3u] == 0u) || (v42_ld_agent_rlx(pf + V46_PULL_RDY + 3u) != 0u))) {
          v42_acq_agent();
          rl |= 2u;
        }
        if ((rl & 4u) == 0u && v42_ld_agent_rlx(pf + V46_P1_PLAN) != 0u) {
          v42_acq_agent();
          rl |= 4u;
        }
        if ((rl & 4u) != 0u && (rl & 8u) == 0u) {
          const bool q1ok = (pf[V46_T + 1u] == 0u) || (v42_ld_agent_rlx(pf + V46_PULL_RDY + 1u) != 0u);
          const bool q2ok = (pf[V46_T + 2u] == 0u) || (v42_ld_agent_rlx(pf + V46_PULL_RDY + 2u) != 0u);
          if (q1ok && q2ok) {
            v42_acq_agent();
            rl |= 8u;
          }
        }
      }
      v70_8w_sh(30752u)[0] = rl;
      const bool loc_rdy = (rl & 1u) != 0u;
      const bool q3_rdy  = (rl & 2u) != 0u;
      const bool rem_rdy = (rl & 8u) != 0u;
      unsigned handoff = 0u;
#if (V70_R8_HYB_EFF) == 2
      if (role == 1u) {
        const unsigned thr = v70_8w_sh(30772u)[0];
        const unsigned rel = __hip_atomic_load(pf + V70R8_RREL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const unsigned sct = __hip_atomic_load(pf + V70R8_RSCT, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        handoff = (rel > sct && (rel - sct) >= thr) ? 1u : 0u;
      }
#endif
      unsigned w = V46_WORD(V46_ACT_IDLE, 0u, 0u, 0u);
      if (handoff || (xm & all_done) == all_done) {
        w = V46_W_EXIT;
      } else {
        unsigned win_got = 0xffffffffu, win_g2 = 0u, win_Nc = 0u, win_F = 0u;
        #pragma clang loop unroll(disable)
        for (unsigned ph = 0u; ph < nph; ++ph) {
          const unsigned psh = ph * 4u;
          if (((xm >> psh) & 0xFu) == 0xFu) continue;
          const bool is_g2 = (ph == 2u || ph == 3u || ph == 5u);
          if (is_g2 && !g2ok) continue;
          const unsigned grp = (ph == 0u || ph == 3u) ? 0u : ((ph == 1u || ph == 2u) ? 1u : 2u);
          if (grp == 1u && !rem_rdy) continue;
          if (grp == 2u && !q3_rdy) continue;
          if (ph == 2u && (((xm >> 4u) & 0xFu) != 0xFu)) continue;
          if (ph == 3u && (!loc_rdy || ((xm & 0xFu) != 0xFu))) continue;
          if (ph == 5u && (((xm >> 16u) & 0xFu) != 0xFu)) continue;
          if (cur_grp != grp) {
            w = V46_WORD(V46_ACT_IDLE, 2u, grp, 0u);
            break;
          }
          const unsigned NT = v70m_tab()[LE];
          const unsigned Nc = NT * n2c;
          const unsigned F = (S > 1u) ? (Nc * (V70_R8_FT_EFF)) / 1000u : 0u;
          const unsigned N = is_g2 ? (Nc - F + F * S) : (NT * nn1);
          unsigned* cur = pf + V70P_CUR_BASE + psh;
          unsigned got = 0xffffffffu;
          #pragma clang loop unroll(disable)
          for (unsigned dx = 0u; dx < V70_XCD_N; ++dx) {
            const unsigned xx = (xcd + dx) & 3u;
            if ((xm >> (psh + xx)) & 1u) continue;
            const unsigned bound = v70_xcd_slots(N, xx);
            const unsigned l = __hip_atomic_fetch_add(cur + xx, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            const unsigned g = v70_xcd_work(l, xx);
            if (l < bound && g < N) { got = g; break; }
            xm |= (1u << (psh + xx));
          }
          if (got != 0xffffffffu) {
            win_got = got; win_g2 = is_g2 ? 1u : 0u; win_Nc = Nc; win_F = F;
            break;
          }
        }
        if (win_got != 0xffffffffu) {
          unsigned c, j;
          if (!win_g2) {
            v70c_map(LE, nn1, win_got, c, j);
            w = V46_WORD(V46_ACT_G1, 0u, j * (V55_N1G), V70C_FLAG | c);
          } else if (win_got < win_Nc - win_F) {
            v70c_map(LE, n2c, win_got, c, j);
            w = V46_WORD(V46_ACT_G2, 0u, j, V70C_FLAG | c);
          } else {
            const unsigned f = win_got - (win_Nc - win_F);
            const unsigned s_sh = (n2g == 8u) ? 2u : ((n2g == 4u) ? 1u : 0u);
            const unsigned fq = f >> s_sh;
            const unsigned fr = f & ((1u << s_sh) - 1u);
            v70c_map(LE, n2c, win_Nc - win_F + fq, c, j);
            w = V46_WORD(V46_ACT_G2, 1u, (j << s_sh) + fr, V70C_FLAG | c);
          }
        } else if ((xm & all_done) == all_done && (w >> 30u) == V46_ACT_IDLE && ((w >> 28u) & 3u) == 0u) {
          w = V46_W_EXIT;
        }
      }
      v70_8w_sh(30744u)[0] = xm;
      const unsigned nc = v70_8w_sh(30748u)[0] + 1u;
      v70_8w_sh(30748u)[0] = nc;
      if ((w >> 30u) == V46_ACT_IDLE && ((w >> 28u) & 3u) == 0u) {
        const unsigned isp = v70_8w_sh(30756u)[0] + 1u;
        v70_8w_sh(30756u)[0] = isp;
        if (isp > (1u << 24)) { v983_error((unsigned*)env.base.error, 972u); w = V46_W_ERR; }
        if ((isp & 15u) == 0u && ep16_load_relaxed((unsigned*)env.base.error) != 0u) w = V46_W_ERR;
      } else {
        v70_8w_sh(30756u)[0] = 0u;
        if ((nc % (V70_R8_ERRP)) == 0u || (w >> 30u) == V46_ACT_END)
          if (ep16_load_relaxed((unsigned*)env.base.error) != 0u) w = V46_W_ERR;
      }
      sh[0] = w;
    }
    r2x::bar<V70_R2X_4W_SW>();
    const unsigned w = __builtin_amdgcn_readfirstlane(sh[0]);
    const unsigned act = w >> 30u;
    if (act == V46_ACT_END) {
#if V70_E3_PRIO
      __builtin_amdgcn_s_setprio(0);
#endif
      return ((w >> 28u) & 3u) == 0u;
    }
    if (act == V46_ACT_IDLE) {
      if (((w >> 28u) & 3u) == 2u) {
        const unsigned rg = __builtin_amdgcn_readfirstlane((w >> 20u) & 0xffu);
        const unsigned LE_r = __builtin_amdgcn_readfirstlane(v70_claim_le(env, pf));
        // Full SQC (s_dcache_inv) + L1 (sc0) + XCD L2 (sc1) acquire 3x per launch at rg=0,1,2 group switches!
        v42_acq_sys();
        if (tid < 64u) v70c_build_3g(env, pf, LE_r, rg);
        r2x::bar<V70_R2X_4W_SW>();
        if (tid == 0u) {
          v70_8w_sh(30740u)[0] = rg;
          const unsigned nt_g = v70m_tab()[LE_r];
          if (nt_g == 0u) {
            if (rg == 0u) v70_8w_sh(30744u)[0] |= (0xFu << 0u) | (0xFu << 12u);
            else if (rg == 1u) v70_8w_sh(30744u)[0] |= (0xFu << 4u) | (0xFu << 8u);
            else v70_8w_sh(30744u)[0] |= (0xFu << 16u) | (0xFu << 20u);
          }
        }
        continue;
      }
      __builtin_amdgcn_s_sleep(4);
      continue;
    }
    const unsigned sub = (w >> 20u) & 0xffu, mt = w & 0x1ffffu, s1s = (w >> 17u) & 7u;
    if (act == V46_ACT_G1) {
      #pragma unroll
      for (unsigned s1i = 0u; s1i < (V46_NSUB); ++s1i)
        v70_gemm1_item4(env, o, arena, mt, s1i, sub FI_CTX_ARG);
    } else {
      const unsigned n2gi = (((w >> 28u) & 3u) == 1u) ? 2u : __builtin_amdgcn_readfirstlane(v70_8w_sh(30784u)[0]);
      if (!v70_gemm2_item4(env, o, x, arena, mt, sub, n2gi, (unsigned*)x.compute_done FI_CTX_ARG)) return false;
    }
  }
}
#else
// role 0 = compute CTA (GEMM1 then GEMM2, mode CAS by the first compute CTA), 1 = hybrid comm CTA
__device__ __forceinline__ bool v70_atrex4_r8(
    const V968DispatchExt& env, const V983Ov1Ext& o, const Ep16TurboContext& x,
    unsigned char* arena, const unsigned role, const unsigned grp_or_cidx FI_CTX_PARAM) {
  const unsigned tid = threadIdx.x;
  volatile WeaveBk64LocalU32* sh = v70_8w_sh(V70_8W_SH_WORD);
  volatile WeaveBk64LocalU32* lam_sh = v70_8w_sh(V70_8W_SH_LAM);
  auto* pf = reinterpret_cast<unsigned*>(o.plan_flags);
  unsigned n1b = __builtin_amdgcn_readfirstlane(env.ck.ck_intermediate / 128u);
  unsigned n2b = __builtin_amdgcn_readfirstlane(o.hidden / 128u);
  unsigned n2g = __builtin_amdgcn_readfirstlane(v46_n2g(n2b));
  unsigned n2c = (n2g == 8u) ? (n2b >> 3u) : ((n2g == 4u) ? (n2b >> 2u) : ((n2g == 2u) ? (n2b >> 1u) : n2b));
  asm volatile("" : "+s"(n1b), "+s"(n2b), "+s"(n2g), "+s"(n2c));
  {
    if (tid == 0u) {
      lam_sh[0] = 0xffffffffu;
      unsigned ok = 1u, spins = 0u;
      while (!v70_r6_plan_ready(o, pf)) {
        __builtin_amdgcn_s_sleep(2);
        if (ep16_load_relaxed((unsigned*)env.base.error) != 0u) { ok = 0u; break; }
        if (++spins > (1u << 24)) { v983_error((unsigned*)env.base.error, 971u); ok = 0u; break; }
      }
      v42_acq_agent();
      const bool is_grp0 = (grp_or_cidx == 0xffffffffu) ? (blockIdx.x == (V70_PATHB_NCOMM_CTA)) : (grp_or_cidx == 0u);
      if (ok && role == 0u && is_grp0) {
        unsigned e0 = 0u;
        __hip_atomic_compare_exchange_strong(pf + V70M_MODE, &e0, 4u, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                             __HIP_MEMORY_SCOPE_AGENT);
      }
      sh[0] = ok;
    }
    r2x::bar<V70_R2X_4W_SW>();
    if (__builtin_amdgcn_readfirstlane(sh[0]) == 0u) return false;
    if (tid < 64u) v70c_build(env, pf, v70_claim_le(env, pf));
    r2x::bar<V70_R2X_4W_SW>();
  }
  // Keep tid-0 scheduler state in LDS [30740..30752) so ZERO scheduler VGPRs stay live across GEMM1/GEMM2!
  if (tid == 0u) {
    v70_8w_sh(30740u)[0] = 0u;
    v70_8w_sh(30744u)[0] = 0u;
    v70_8w_sh(30748u)[0] = 0u;
  }
  for (;;) {
#if !V70_E3_3BAR
    r2x::bar<V70_R2X_4W_SW>();
#endif
    if (tid == 0u) {
      const unsigned LE = v70_claim_le(env, pf);
      const unsigned NT = v70m_tab()[LE];
      const unsigned nn1 = n1b / (V55_N1G);
      const unsigned xcd = v55_xcc() & 3u;
      const unsigned S = n2g >> 1u;                       // 256-column units per coarse GEMM2 item
      const unsigned Nc = NT * n2c;
      const unsigned F = (S > 1u) ? (Nc * (V70_R8_FT_EFF)) / 1000u : 0u;
      unsigned stage = v70_8w_sh(30740u)[0];
      unsigned xmask = v70_8w_sh(30744u)[0];
      const bool g2ok = (role == 0u) || ((V70_R8_HYB_EFF) >= 2);
      const unsigned Np = Nc - F + F * S;
      unsigned w = V46_W_EXIT;
      for (;;) {
#if (V70_R8_HYB_EFF) == 2
        if (role == 1u && stage == 1u) {
          const unsigned hidx = (grp_or_cidx == 0xffffffffu) ? blockIdx.x : grp_or_cidx;
          const unsigned hrel = (hidx >= (V70_R8_HYB_LO)) ? (hidx - (V70_R8_HYB_LO)) : 0u;
          const unsigned thr = (V70_R8_BL0) + hrel * (V70_R8_BLS);
          const unsigned rel = __hip_atomic_load(pf + V70R8_RREL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          const unsigned sct = __hip_atomic_load(pf + V70R8_RSCT, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          if (rel > sct && rel - sct >= thr) break;
        }
#endif
        const unsigned N = (stage == 0u) ? NT * nn1 : Np;
        unsigned* cur = pf + ((stage == 0u) ? V70M_G1C : V70M_G2C);
        unsigned got = 0xffffffffu;
        #pragma unroll
        for (unsigned dx = 0u; dx < V70_XCD_N; ++dx) {
          const unsigned xx = (xcd + dx) & 3u;
          if ((xmask >> xx) & 1u) continue;
#if V70_E3_QPART
          const unsigned qlo = (N * xx) >> 2u;
          const unsigned qhi = (N * (xx + 1u)) >> 2u;
          const unsigned bound = qhi - qlo;
          if (bound == 0u) { xmask |= 1u << xx; continue; }
          const unsigned l = __hip_atomic_fetch_add(cur + xx, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          if (l < bound) { got = qlo + l; break; }
#else
          const unsigned bound = v70_xcd_slots(N, xx);
          const unsigned l = __hip_atomic_fetch_add(cur + xx, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
          const unsigned g = v70_xcd_work(l, xx);
          if (l < bound && g < N) { got = g; break; }
#endif
          xmask |= 1u << xx;
        }
        if (got != 0xffffffffu) {
          unsigned c, j;
          if (stage == 0u) {
            v70c_map(LE, nn1, got, c, j);
            w = V46_WORD(V46_ACT_G1, 0u, j * (V55_N1G), V70C_FLAG | c);
          } else if (got < Nc - F) {
            v70c_map(LE, n2c, got, c, j);
            w = V46_WORD(V46_ACT_G2, 0u, j, V70C_FLAG | c);
          } else {
            const unsigned f = got - (Nc - F);
            const unsigned s_sh = (n2g == 8u) ? 2u : ((n2g == 4u) ? 1u : 0u);
            const unsigned fq = f >> s_sh;
            const unsigned fr = f & ((1u << s_sh) - 1u);
            v70c_map(LE, n2c, Nc - F + fq, c, j);
            w = V46_WORD(V46_ACT_G2, 1u, (j << s_sh) + fr, V70C_FLAG | c);
          }
          break;
        }
        if (stage == 0u && g2ok) { stage = 1u; xmask = 0u; continue; }
        break;
      }
#if V70_E3_3BAR && V70_E3_SC1_NOBW
      if ((w >> 30u) == V46_ACT_G2) {
        const unsigned mts = w & 0xffffu;
        unsigned backoff = 1u, spins = 0u;
        while (!v46_s1_ready(reinterpret_cast<const unsigned*>(o.stage1_tile_done), mts, n1b)) {
          if (spins < 128u) asm volatile("s_nop 0\n" ::: "memory");
          else v983_sleep(backoff);
          ++spins;
          if (spins >= V1895_WAIT_SPINS) { v983_error((unsigned*)env.base.error, 207u); w = V46_W_ERR; break; }
          if (backoff <= 64u) ++backoff;
        }
      }
#endif
      v70_8w_sh(30740u)[0] = stage;
      v70_8w_sh(30744u)[0] = xmask;
      const unsigned nc = v70_8w_sh(30748u)[0] + 1u;
      v70_8w_sh(30748u)[0] = nc;
      if ((nc % (V70_R8_ERRP)) == 0u || (w >> 30u) == V46_ACT_END)
        if (ep16_load_relaxed((unsigned*)env.base.error) != 0u) w = V46_W_ERR;
      sh[0] = w;
    }
    r2x::bar<V70_R2X_4W_SW>();
    const unsigned w = __builtin_amdgcn_readfirstlane(sh[0]);
    const unsigned act = w >> 30u;
    if (act == V46_ACT_END) {
#if V70_E3_PRIO
      __builtin_amdgcn_s_setprio(0);
#endif
      return ((w >> 28u) & 3u) == 0u;
    }
    const unsigned sub = (w >> 20u) & 0xffu, mt = w & 0x1ffffu, s1s = (w >> 17u) & 7u;
    if (act == V46_ACT_G1) {
      #pragma unroll
      for (unsigned s1i = 0u; s1i < (V46_NSUB); ++s1i)
        v70_gemm1_item4(env, o, arena, mt, s1i, sub FI_CTX_ARG);
    } else {
      const unsigned n2gi = (((w >> 28u) & 3u) == 1u) ? 2u : n2g;
      if (!v70_gemm2_item4(env, o, x, arena, mt, sub, n2gi, (unsigned*)x.compute_done FI_CTX_ARG)) return false;
    }
  }
}
#endif  // V70_R8_2WAVE
#endif
