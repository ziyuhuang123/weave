# Weave 3-Tier Framework: Hardware Calibration → (Phase, N, K) Offline GEMM Profiles → Online CTA-0 + SM0 Routing-Adaptive Selection

This document describes the scheduling framework implemented on the unified
14-knob public surface (Spatial 5, Temporal 3, Implementation 6). It adds **no
new public environment knobs**: every online behavior activates through existing
auto/default values, and every explicit pin stays 100% deterministic and
bit-identical to the previous behavior.

Naming note: this document predates the 21 -> 14 knob consolidation and keeps the
historical spellings in its body. `WEAVE_BLOCK_M_L1` and
`WEAVE_BLOCK_M_L2` are the legacy names of `WEAVE_GEMM1_CONFIG` and
`WEAVE_GEMM2_CONFIG` (transparent aliases); `WEAVE_ADAPT_BM` / `WEAVE_ADAPT_GEMM_CONFIG`
is no longer a public knob (host shape-adaptive GEMM tuple selection is always on when
GEMM1/GEMM2 config are 0; `=0` stays accepted as an internal legacy/diagnostic override), and
`WEAVE_REDUCE_WARPS`, `WEAVE_COMBINE_SYNC_CHUNK`, `WEAVE_REG_BALANCE`,
`WEAVE_LOCAL_SCATTER_DEDUP`, `WEAVE_SM0_STREAM_SCHED` and `WEAVE_READY_QUEUE` are
internal compile-time pins (default 0) that are no longer public knobs; see
`docs/TUNABLE_PARAMETERS.md` for the current public surface.

The original Two-Level framework (offline hardware calibration + online CTA-0
routing-adaptive selection) is upgraded to a **3-Tier Selection Engine**:

```
Tier 0  (Hardware)                          once per GPU
        results/hardware_calibration_3hw.json
        peak TFLOPS / HBM / NVLink-xGMI / SM-CU count / L2 / SMEM-LDS /
        register budget / sync latencies  (measured + Table I)
            |
Tier 1  (Hardware, Precision, Phase, N, K)  once per launch geometry
        GEMM1 (N1=2I, K1=H, SwiGLU+quant)  and  GEMM2 (N2=H, K2=I,
        bf16 weight-mul + scatter)  solved COMPLETELY independently
        -> PhaseEnvelope (BN*, BK*, max_BM*, stages*, wg_layout*, swizzle,
                          interleave, epw) over BACKEND-LEGAL tile ladders
        -> per-M Inner-Tuple LUT (GemmInnerProfile) over
           M_bucket in {8,16,32,48,64,96,128,192,240}
        results/gemm_profiles_3hw.json  (6 models x 3 hw x 2 phases x 9 M)
            |
Tier 2  (Runtime M_e, mu)                   every kernel step, on device
        CTA-0: Eq.(1)-(3) + OnlineRoutingPolicy + per-expert
               l1_profile_code[e] / l2_profile_code[e]  (bucketed M_e)
        SM0:   build_sm0_expert_order  (dedup clustering / pendulum /
                                        hot-cold pairing / W=1 no-pair)
               lock-free local pre-reduce (red.global.add + fence.acq_rel.gpu)
               Sm0ReadyTokenQueue      (no-CAS MPSC ready-token publication)
            |
Two-Level Scheduling                        inter-cluster + intra-cluster
        INTER: ShardedReadyTokenRing  D in {1,2,4,8}, ~2 KiB/shard, L2-resident
               cluster-exclusive SPSC (default) | no-CAS MPSC | unified-claim
               MPMC work steal  -- three separated protocols, never mixed
        INTRA: ClusterSchedPacket (16 B) pushed leader -> sibling CTAs over
               mapa.shared::cluster + st.async.shared::cluster (DSMEM, no L2)
```

### Scope and wiring status (read first)

* **SM103: the online layer is not compiled by default.** On NVIDIA SM103 the
  CTA-0 online tier (Tier 2: `cta0_select_from_packed_hist`, the SMEM
  `PolicyDecision` copy and the cross-CTA publish/wait handshake) is compiled
  into `sm100_fp8_fp4_mega_moe.cuh` only under the kernel macro
  `WEAVE_RUNTIME_POLICY=1`. The host JIT emits `WEAVE_RUNTIME_POLICY=1` iff the
  resolved `knobs.sm0_stream_sched != 0`, i.e. `WEAVE_SM0_STREAM_SCHED=1` or
  `WEAVE_READY_QUEUE=sm0_fifo`. With the default (`0`) the kernel keeps the
  compiled pins (`active_reduce_warps = R_max`, no `active_v_head` shrink).
  **Every SM103 measurement in this repository was taken with the online layer
  not compiled.** `WEAVE_ADAPT_BM=1` is not a supported way to enable it on
  SM103 (CUDA `misaligned address` at `S >= 4096`; see Known limitations in
  `docs/TUNABLE_PARAMETERS.md`). H20 and MI308X always compile the online tier.
  Every statement below of the form "CTA 0 evaluates / publishes" therefore
  applies to SM103 only for `-DWEAVE_RUNTIME_POLICY=1` builds.
* **Decision-block sizes.** `PolicyDecision` is 84 B (21 x 4-byte fields,
  `weave_cost_model.cuh`); the SM103 kernel-side `SharedStorage` adds 92 B
  (`PolicyDecision` + two `uint32_t`) when `WEAVE_RUNTIME_POLICY=1`, and the host
  SMEM planner reserves 112 B for it.
* **What the production kernels consume.** From the online publication the
  production kernels (`sm100_fp8_fp4_mega_moe.cuh`, `sm90_fp8_mega_moe.cuh`) read
  only `active_reduce_warps` and `active_v_head` (`load_active_reduce_warps` /
  `load_active_v_head`) plus the `PolicyDecision` itself. The other Tier-2
  outputs (ready-group size, dedup verdict, `bm_tier_mean`, the 3-Tier
  envelope and per-expert profile codes) are published as live
  recommendations; no kernel mainloop reads them.
* **Library-level primitives not wired into any production kernel.**
  `build_sm0_expert_order` / `build_sm0_expert_order_warp` (Section 4a),
  `Sm0ReadyTokenQueue` (4b), `lockfree_local_prereduce_*` (4c),
  `ShardedReadyTokenRing` (4d) and `ClusterSchedPacket` /
  `cluster_sched_fanout` (4e) live in `weave_policy_runtime.cuh` and have 0
  call sites in `sm100_fp8_fp4_mega_moe.cuh`, `sm100_bf16_mega_moe.cuh`,
  `sm90_fp8_mega_moe.cuh` and `impls/amd/`. They are exercised only by
  `tests/compile_sm100_dsmem_sched.cu`, `tests/test_cost_model_eq123.cpp` and
  `tests/test_two_level_adaptive_policy.py`. Sections 4a-4e document these
  library primitives and their gates, not behavior of the measured kernels.

## 1. The levels

**Level 1 — offline hardware calibration (once per hardware).**
`deep_gemm/mega/calibrate.py` measures/stores a `HardwareCalibrationProfile`
per hardware from the *real verified measurements* (Section 5). The profile
holds the roofline constants `(F_eff, B_hbm, B_comm, tau_sync, mu_dedup_thresh,
hbm_headroom_thresh, tail_ratio_thresh)` plus the per-expert BM-ladder slack and
the hot-expert head budget, and a ~13-knob compile-time **structural regime**
expressed purely with the existing public `WEAVE_*` knobs.
`select_offline_regime(profile, S_global, num_shared_experts)` returns the exact
structural env dict, the launch-time regime label, and the online thresholds.

**Level 1.5 — offline (Phase, N, K) GEMM profiles (once per launch geometry).**
The same module now also solves the independent GEMM1/GEMM2 launch envelopes
and per-M inner tuples analytically (Section 3). Previously `WEAVE_ADAPT_BM=1`
only shrank `BM` from `M` while every other mainloop/epilogue parameter stayed
fixed across all `M` and shared between GEMM1 and GEMM2; now each phase gets
its own `(BN, BK, max_BM, stages, L2-swizzle, interleave)` envelope and each
`M` bucket its own mainloop/epilogue inner tuple, all derived from Tier-0
physics with zero blind grid search.

**Level 2 — online CTA-0 routing-adaptive selection (every kernel step).**
(On SM103 this level exists only in `-DWEAVE_RUNTIME_POLICY=1` builds; see
"Scope and wiring status".) Inside the *same* kernel compiled with the structural regime, CTA 0 reads the
live post-dispatch packed histogram `n_e[0..E_local)` after the dispatch
barrier, evaluates Eq.(1)-(3) (`weave_cost_model.cuh`), and publishes an
`OnlineRoutingPolicy` in the **global** `PolicyControl` block next to the
existing `PolicyDecision`. The SMEM decision copy is unchanged: **zero SMEM
increase, zero extra spills**. Every other CTA acquire-loads the publication
epoch and reads the online fields with one 4-byte load per consumer. The added
work is one histogram stats pass plus O(1) gate arithmetic on the already
computed winning breakdown, two integer envelope selections (launch-constant
geometry, constexpr-foldable) and one 9-bucket LUT fill per phase.

## 2. Knob binding tiers

| Tier | Bound when | Fields | Mechanism |
|---|---|---|---|
| Structural | offline regime -> JIT compile | `WEAVE_PRECISION_MODE`, `WEAVE_ADAPT_BM`, `WEAVE_BLOCK_M_L1`, `WEAVE_BLOCK_M_L2`, `WEAVE_SCATTER_OWNER`, `WEAVE_SCATTER_STAGING`, `WEAVE_LOCAL_SCATTER_DEDUP`, `WEAVE_COMBINE_SYNC_CHUNK`, `WEAVE_DISPATCH_ACTIVE_SMS`, `WEAVE_SCATTER_ACTIVE_SMS`, `WEAVE_REG_BALANCE`, `WEAVE_REDUCE_WARPS` (R_max), `WEAVE_CU_WAVE_SPLIT` (MI308X) | compile-time template/macro structure; the coupled `(blockM, blockN, blockK, stages)` / epilogue tuples lock synchronously (built-in, no sub-knobs) |
| Launch regime | `select_offline_regime(profile, S_global, num_shared_experts)` | regime label `decode`/`balanced`/`dense` x `verified`/`extrapolated` | selects the structural env + the online thresholds for this launch |
| Online per-step | CTA-0 after the dispatch barrier (SM103: `WEAVE_RUNTIME_POLICY=1` builds only) | `active_reduce_warps`, `active_v_head` (acted on by the kernel); `active_ready_group_size`, `enable_dedup_runtime`, `bm_tier_mean` (live verdicts); `phase_profile_active`, `l1/l2_envelope_code`, `l1/l2_profile_code[e]` (3-Tier publication) | Eq.(1)-(3) on the live histogram; explicit pins always override |

`WEAVE_ADAPT_BM=1` is the existing master switch of the online tier: with
`WEAVE_ADAPT_BM=0` (the default) every Tier-2 output equals the compiled pin
(the profile LUT publishes zeros), so existing unit tests and benchmarks remain
bit-identical. The 3-Tier engine introduces **no new `WEAVE_*` names** — the
envelope/inner-tuple selection is the built-in synchronous coupling that
`WEAVE_BLOCK_M_L1/L2=0` (auto) always promised, now solved per `(Phase, N, K)`.

## 3. Tier 1 — independent (Phase, N, K) envelopes + per-M inner tuples

`calibrate.py select_phase_envelope / select_phase_inner_profile` and the C++
mirror in `weave_cost_model.cuh` are **integer-only** (uint64/uint32, zero
floating point): analytical legal pruning + physics scoring, bit-exact across
languages by construction. GEMM1 (`N1 = 2*I, K1 = H`, SwiGLU + quant epilogue)
and GEMM2 (`N2 = H, K2 = I`, BF16 weight-mul + NVLink/xGMI scatter epilogue)
are solved completely independently — their `(N, K)` aspect ratios, K-loop
lengths, L2 weight footprints and epilogue costs are fundamentally different.

**Legal pruning — the tile ladders are the BACKEND's own contract.** An
envelope the kernel cannot instantiate is not a recommendation, so the
candidate sets are transcribed from the kernels' static asserts:

| Backend | `BLOCK_N` | `BLOCK_K` | `BLOCK_M` | Source |
|---|---|---|---|---|
| SM103 | 64/128/256 | **128/256** | any `% 16` (32…256, incl. 80/96/240) | `sm100_fp8_fp4_mega_moe.cuh:1148-1150` |
| H20 | 64/128/256/**512** | **128/256** (`% 128`) | `% 64` (64/128/192/256); 16/32 only under `kFP8SwapAB` | `sm90_fp8_mega_moe.cuh:623-634` |
| MI308X | 64/128/256 | **64**/128/256 | 16/32/64/128 | `megamoe_amd.hip:54-60`, `impls/amd/include/weave_ck_bm_configs.hpp` |

This per-hardware legality is itself a first-class source of envelope
differentiation: only H20 may pick `BN = 512`, only SM103 may pick the
non-power-of-two heights 80/96/240, only MI308X may pick `BK = 64`.

On top of the ladder:

* `N % BN == 0` (zero tail waste) and `N % (2*BN) == 0` (2-CTA cluster
  N-pairing; for GEMM1 this also keeps the SwiGLU gate/up halves BN-aligned).
* `K % BK == 0`.
* `max_BM * BN <= acc_tile_limit` — the zero-spill accumulator register budget
  (128 regs/thread over 256 acc threads on SM100/SM90; 128 VGPRs/lane over 512
  lanes on MI308X). `max_BM` is the **largest surviving legal height**.
* `num_stages` is searched **downward** from the K-length stage ladder
  (`K<=512 -> 6`, `<=1024 -> 5`, `<=2048 -> 4`, `<=4096 -> 3`, else 2), capped
  by (a) SMEM/LDS capacity `(smem_bytes - reserve) / stage_bytes` with
  `stage_bytes = (max_BM*BK*act_w + BN*BK*w_w)` at the exact precision byte
  widths (1/32-byte units: fp8xfp4 18/16, nvfp4 18/18, fp8 33/33, bf16 65/64),
  (b) the K-loop length (`stages <= K/BK`), and (c) the short-K rule
  (`K <= 1024` never amortizes more than 4 prefetch stages).

**Scoring — integer STRUCTURAL waste, not a roofline time.** The envelope is a
structural choice (tile shape / pipeline depth / layout). A roofline-time
objective is dominated by candidate-independent terms — the runtime M ladder
clamps `BLOCK_M` to 32/64 at small design anchors, so `max_BM` never enters the
score — and collapses distinct geometries onto one constant envelope. The
selector therefore minimizes the summed structural waste (milli-units):

```
clusters_n    = (N/BN) / 2                      # scheduler work unit = N-cluster
cluster_slots = num_sms / 2                     # resident 2-CTA clusters
col_pen = (ceil(clusters_n/cluster_slots)*cluster_slots - clusters_n)*1024/clusters_n
mma_pad = (align(max_BM, MMA_M_atom) - max_BM)*1024/max_BM     # atom 64 NV / 32 MFMA
pad_pen = (M_pad(240) - 240)*1024 / M_pad(240)                 # prefill M-row padding
bk_bias = 96 if (K<=1024 and BK != narrowest legal) or (K>=4096 and BK != 128)
narrow  = 128 if N >= 4096 and BN < 128                        # tile explosion
penalty = max(0, 2*col_pen + 2*mma_pad + pad_pen + bk_bias + narrow - 24*stages)
```

`clusters_n` is the persistent scheduler's actual work unit
(`mega_moe.cuh:1805`, `kNumClusters = (kShapeN / BLOCK_N) / 2`), and the
`N % (2*BN)` prune guarantees it is an integer. The `-24*stages` term is the
per-stage latency-hiding bonus. Strict-`<` tie rule, fixed candidate order
(BN and BK descending, stages descending, largest surviving `max_BM`).

The integer roofline estimate is still computed and reported as `score_ns` for
auditing (it is no longer the objective):

```
t(M) = max(compute, hbm) + fill + iter + epi + scat        at M in {32, 192}
  compute = (2*m_pad*N*K / (TFLOPS*1000)) * ceil(CTAs/num_sms)   [wave quantization]
  hbm     = (W[e]*(1 if L1-protected else tiles_m) + m_pad*K*act_w) / B_hbm
            * (4/stages if stages < 4)                           [>=4-stage saturation]
  fill    = stages * stage_bytes / B_hbm                         [pipeline fill]
  iter    = (K/BK) * 2ns * waves                                 [TMA-pair issue]
  epi     = SwiGLU read + quant write (GEMM1) / bf16 write (GEMM2) at B_hbm
  scat    = M*N*2B / B_comm (GEMM2 only)
```

**L1 intra-expert 2D L2-swizzle protection.** With
`M_blocks = ceil(M/BM) >= 2` and the expert weight inside **half of L2**, the
weight is HBM-read once and L2-hit across the M-blocks: the envelope keeps
`interleave_mode = 1` (expert-major 2D swizzle) with
`l2_swizzle_group = min(M_blocks, 8)` — never the tight wave-level mode that
would let concurrent L2 weight loads evict `W1[e]`. When the whole problem
fits one partial wave the tight interleave is harmless (`mode 0`); otherwise
grouped waves (`mode 2`) with `epw = num_sms / (tiles_n * M_blocks)` derived
from `L2_cap` and the M distribution. The envelope verdict is taken at the
**structural anchor `M = 240`** (the top of the runtime M ladder) through the
**same BM ladder** the inner tuples use, so `max_BM` is genuinely exercised and
offline and online never disagree.

**Per-M inner-tuple LUT (`GemmInnerProfile`, 9 buckets, shares the envelope):**

| Field | Rule |
|---|---|
| `block_m` | `select_expert_tile_bm` ladder inside `max_BM*` (32-row padded fit) |
| `tma_load_m` | `min(block_m, atom)` — SM103 tcgen05 M=128, H20 WGMMA M=64, MI308X MFMA 64 |
| `wg_layout_mode` | H20: split-N (1) for `BM < 64` keeps 4 WGs active, split-M (0) otherwise; SM103: sub-tile (2) for `BM <= 64`; MI308X: split-M |
| `pipe_mode` | SM103 TMEM permille 100/175/200 (fat M + deep stages -> 2.0); H20 WGMMA depth 1..3 from stages; MI308X 1..2 |
| `l2_swizzle_group` | per-M refinement of the L1 protection rule (`min(M_blocks, 8)` or 1) |
| `store_block_m` | largest of {16,32,64} dividing `block_m` (epilogue store height) |
| `epi_reg_stages` | 2 for fat M (`BM >= 128`, or SM103 `BM in {80,96}` with TMEM >= 1.75) to hide TMEM/WGMMA latency; 1 for tiny M |
| `scatter_path` | GEMM2 only: dedup active -> 2 (lock-free local pre-reduce); register-tight SM90 -> 1 (SMEM handoff); else 0 (direct reg scatter). GEMM1: 0 |

`mu` enters only as the **integer dedup verdict** (`mu >= mu_dedup_thresh`, or
the explicit pin) reduced before the call, keeping the selection bit-exact.

**Device-side structs** (`weave_cost_model.cuh`): `PhaseEnvelope`
(`bn, bk, max_bm, num_stages, l2_swizzle_group, interleave_mode, wg_layout_mode`
+ `epw` + the `score_ns` / `penalty` diagnostics) and `GemmInnerProfile` — POD,
trivially copyable, **12 bytes (<= 16)**. Both pack into compact `uint32_t`
codes: a **26-bit envelope code that is lossless for every structural field**
(2-bit BN log-index up to 512, 2-bit BK, 4-bit BLOCK_M height index covering the
non-power-of-two heights, 3-bit stages, 3-bit swizzle group, 2-bit interleave,
2-bit warpgroup layout, 8-bit `epw`) and a 22-bit inner code. `OnlineRoutingPolicy` is extended **append-only** (every legacy
field keeps its offset and semantics) with `phase_profile_active`,
`l1_envelope_code`, `l2_envelope_code` and the independent per-expert
`l1_profile_code[kMaxRuntimeExperts]` / `l2_profile_code[kMaxRuntimeExperts]`.
The arrays live in the **global** control block only (zero SMEM); consumers
read one 4-byte code per expert via `load_expert_profile_code(control, phase,
e)` — no full-LUT copy, zero stage drops. CTA-0 computes each phase's 9-bucket
LUT once per step, then buckets every live `M_e` into it (one ladder walk per
expert). Non-adaptive builds publish zeros and keep their compiled structure.

**Artifact:** `results/gemm_profiles_3hw.json` — all 6 standard models
(`qwen38max`, `dskv4_flash`, `minimax_m3`, `qwen36_35b` = the repo's
Qwen3.5-35B-A3B geometry, `qwen35_122b`, `qwen35_397b`) x 3 hardware x 2 phases
x the 9-bucket M ladder, at each hardware's native launch precision. The
committed report selects **25 distinct envelopes across the 36 phase cells**
(16 distinct `(BN, BK, max_BM, stages)` tile tuples, 13 distinct per-M LUT
signatures); **GEMM1/GEMM2 differ on 17 of the 18 model-hardware cells**, every
per-M LUT varies across the ladder, and all 12 model-phase geometries change
envelope across hardware. No hardware collapses onto a constant envelope
(`per_hw_envelope_variants` in the artifact). The report also carries the
`legal_tiles` contract and the `independent_phase_pair_cells` audit so the
differentiation is checkable without re-running the engine. Regenerate with:

```bash
python deep_gemm/mega/calibrate.py --gemm-profiles \
    --out results/gemm_profiles_3hw.json
```

## 4. Tier 2 — online decision rules (CTA-0)

Wiring status: compiled on H20 and MI308X; on SM103 only with
`-DWEAVE_RUNTIME_POLICY=1` (not the default, and not used by any SM103
measurement). Of the outputs below, only rules 1 and 2 (`active_reduce_warps`,
`active_v_head`) are consumed by a production kernel.

Let `M = sum_e n_e`, `active` = nonzero bins, `mean_n = M/active`,
`skew = n_max/mean_n`, and let `T_comp`, `V_mem`, `T_tail`, `T_total` be the
Eq.(1)-(3) terms of the selected policy.

1. **Dedicated Middle Reduce lanes (KEY gate).** With compiled
   `WEAVE_REDUCE_WARPS=R_max` and `WEAVE_ADAPT_BM=1`:
   `middle_hbm_util = V_mem / (T_comp * 1000 * B_hbm)` and
   `tail_combine_frac = T_tail / T_total`. Keep `active_reduce_warps = R_max`
   iff `middle_hbm_util <= hbm_headroom_thresh` **and**
   `tail_combine_frac >= tail_ratio_thresh`; otherwise `0`. Gated-off warps
   leave the reduce work to the epilogue tail drain (SM100: CTA-private claim
   cursor over the SM's own tile set, zero global atomics; SM90: epi-only work stride) and never contend for Middle HBM.
   With `WEAVE_ADAPT_BM=0`, `active_reduce_warps = R_max` always (deterministic).
2. **Auto head budget.** `WEAVE_V_HEAD=0` (auto): if `skew >= 2` (integer
   cross-multiply `n_max * active >= 2 * M`), shrink to
   `align32(min(window, v_head_hot_budget))` so helping warps return to GEMM
   while the comm warps drain the hot expert; balanced routing keeps the
   existing Eq. window bit-identical. `WEAVE_V_HEAD>0` always overrides.
3. **Auto ready group.** `WEAVE_READY_GROUP_SIZE=0` (auto): `mean_n` ladder
   `{<32: 1, <128: 2, <256: 4, else 8}`; when >= 3/4 of the local experts are
   active (dense prefill), double the group (cap 8) to amortize publication.
   Explicit values always override. The publication granularity itself is
   structural; the online field is the live recommendation.
4. **Dedup verdict.** `mu = M / unique_tokens` (the dedup pull count is the
   live proxy when the unique count is unavailable; `mu=1` when neither is
   known). Auto (`WEAVE_DEDUP_MODE` unset): `enable_dedup_runtime =
   (mu >= mu_dedup_thresh)`. Explicit `0`/`1` pins win. This verdict also
   selects the GEMM2 `scatter_path` in the published profile LUT.
5. **Per-expert BM tier.** `bm_tier_mean` applies the same ladder as
   `select_expert_tile_bm` (`{32,64,128,256}` + non-power-of-two envelope,
   32-row padded size + `bm_pad_slack`) to `mean_n`; with `WEAVE_ADAPT_BM=1`
   the scheduler already applies that ladder per live `n_e`.
6. **Per-expert phase profiles (3-Tier).** With `WEAVE_ADAPT_BM=1`, CTA-0
   publishes the independent GEMM1/GEMM2 envelope codes and the per-expert
   inner-tuple codes bucketed from the live `M_e` (Section 3). The kernels'
   compiled structure is unchanged; the codes are the live, bit-exact
   recommendation of what the offline engine would launch for this expert.

### 4a. SM0 dedup-aware co-scheduling + hot/cold wave pairing

Wiring status: library-level primitive in `weave_policy_runtime.cuh`;
not called by any production kernel (verified by host tests and compile gates
only).

`weave::build_sm0_expert_order` (`weave_policy_runtime.cuh`, Python mirror in
`calibrate.py`) builds a deterministic expert permutation from per-expert
`(token_count, multi_hit_tokens, primary_partner)`:

1. `hot_thresh = max(13, 2*floor(M/active))` — `M <= 12` is always cold.
2. Active experts are counting-sorted count-desc / id-asc (uniform shortcut,
   insertion sort for `n <= 32`, else a 16-bit LSD radix with a skipped
   high-byte pass when all counts share one 256-window and a 4-way split
   prefix — no general comparison sort, no allocation).
3. **Mode verdict — the pairing boundary is EXPLICIT.** A cluster census and a
   multi-hit census run *before* any placement and select one of four modes
   (returned through `mode_out`, mirrored by
   `calibrate.py build_sm0_expert_order_with_mode`):

   | Mode | Condition | Placement |
   |---|---|---|
   | `kSm0ModeDedupCluster` (1) | dedup on **and** real co-occurrence clusters exist | union-find clusters kept adjacent; **no** `[Hot,Cold]` interleave |
   | `kSm0ModePendulum` (2) | dedup on, per-expert `multi_hit_e` only (no partner/anchor edges) | centre-out pendulum |
   | `kSm0ModeHotCold` (0) | dedup off **and** `experts_per_wave >= 2` | capacity-dealt waves with `[Hot, Cold, ...]` |
   | `kSm0ModeNoPair` (3) | `experts_per_wave < 2` (one expert fills the whole card, `W = 1`) | plain load-descending; pairing auto-exits |

   `[Hot, Cold]` mixing is therefore entered **only** when
   `experts_per_wave >= 2` and the dedup verdict is off; high multi-hit always
   prefers co-occurrence clustering, and `W = 1` has no second slot to pair
   with so it exits.
4. **Dedup clustering** (mode 1): union-find over multi-hit partner edges
   (root = min member id). Cluster members stay **adjacent**, so a multi-hit
   token's completion span `max(pos) - min(pos)` collapses and
   `next_cnt == hit_count` fires continuously from wave 1 — Scatter warps never
   wait on an empty queue and the partial-sum scratch rows stay L2-resident.
5. **Pendulum centring** (mode 2): active experts are ranked by
   `(multi_hit desc, token_count desc, id asc)` with four stable 8-bit LSD
   radix passes on a composite key (a pass whose byte is uniform across all
   keys is skipped), then the most-shared expert takes the **centre** slot
   `(active-1)/2` and the rest fan out symmetrically right/left. Every
   multi-hit token's producers land in one compact central window.
6. **Hot/cold wave pairing** (mode 0): clusters are dealt hot-first round-robin
   into the `ceil(active/wave_size)` waves with enough remaining capacity (the
   last wave is partial; an oversized cluster overflows into the emptiest wave,
   keeping adjacency), then cold clusters over the same capacities. Inside each
   wave the units emit `[Hot, Cold, Hot, Cold, ...]`, so no wave is packed purely
   with `M <= 12` cold experts (the H20 15-25% Compute-valley waves), while each
   expert's internal L1 2D L2-swizzle (Section 3) is untouched. Mode 1 uses the
   same capacity dealing but emits whole units hot-first — the dealing already
   spreads hot units over every wave.
7. Zero-token experts are appended last in id order.

**Zero big stack frame.** All state lives in the caller-provided 16 KB
`Sm0OrderScratch` (SMEM or global on device, heap on host); no per-thread array
is ever materialized. `build_sm0_expert_order_warp` is the warp-cooperative
entry point: all 32 lanes run the parallel-safe passes (histogram stats with an
associative `shfl_xor` reduction, union-find parent init, unit init) strided
over the **same helper code** the serial entry point uses, while the
order-dependent passes (stable compaction, radix scatter, capacity dealing,
emission) stay on lane 0 — so the permutation is **bit-identical** to the
serial path. The 256-bucket sweeps stay rolled and the device prefix uses a
single accumulator (a rolled 4-way split would index a `quarter[4]` array
dynamically and ptxas would place it on the device stack); the host keeps the
4-way split, which computes the identical exclusive prefix.
`tests/check_dsmem_sched_compile.py` measures **32 registers, 0 bytes spill,
0 bytes stack frame** for both entry points on `sm_103a` and `sm_90a`.

Measured on the frozen realistic fixtures (Zipf(0.9) proxies + the real H20
pure21 rank aggregates, `tests/test_cost_model_eq123.cpp` /
`test_two_level_adaptive_policy.py`): co-occurrence completion span
**96.9-99.2% shorter** than identity order on the 8 realistic fixtures
(2048 -> 64, 7552 -> 118, 8192 -> 128, 32768 -> 256, ...), and 50% on the
4-expert `tiny_waves_w1` edge case (8 -> 4, the structural floor at that size);
the gate requires >= 30%. Per-wave token-sum variance is strictly lower than
descending packing on **6 of the 10** fixtures where descending packing has
nonzero variance (961122 -> 82289 on Zipf64, 29805445 -> 2525741 on Zipf128,
297025 -> 4900 on ten-hot); the remaining four sit at the structural floor
where wave sizes alone fix the variance. Pure-cold waves are eliminated
wherever the hot population spans the waves (e.g. 7 -> 0 on Zipf128, 1 -> 0 on
the ten-hot fixture). The
**pendulum** path, measured on multi-hit-only fixtures with real multi-expert
hit sets, shortens the completion span by **83.3%** (168 -> 28 at E=32,
432 -> 72 at E=64, 936 -> 156 at E=128). Host latency proxy: <= 5.4 us/call
worst case (E=128, g++ build on a non-target development host, min of
5 x 1000-rep batches); the
device budget is evidenced by the `clock64()` hook in the GPU rail and has not
been measured on a target device.

### 4b. SM0 centralized ready-token queue

Wiring status: library-level primitive in `weave_policy_runtime.cuh`;
not called by any production kernel (verified by host tests and compile gates
only).

`weave::Sm0ReadyTokenQueue` (`weave_policy_runtime.cuh`) is a lock-free MPSC
ring (64-byte header, external `Sm0ReadyCell{token, ready}` array) that
eliminates blind `.sys` polling and pending-bitmask register bloat:

* **Protocol.** Producers reserve a slot with `atomicAdd(tail)`, write the
  token, then release-store the cell's `ready` flag to the current epoch. The
  single consumer acquire-loads the head cell's flag; only a flag equal to the
  epoch guarantees the token write is visible — every pop is a
  guaranteed-ready row. `rtq_begin_epoch` resets the cursors and bumps the
  epoch (monotone, never reused), so stale cells can never be popped; pushes
  beyond capacity set a sticky overflow flag and return false (legacy
  fallback). No CAS, no wrap within an epoch, no ABA.
* **Sender-side local pre-reduce ready queue.** Each `(dst_rank, token)` starts
  with `remaining_hits = hit_count`; every local expert's lock-free pre-reduce
  completion calls `rtq_notify_local_prereduce` (`atomicSub`); the **last** hit
  pushes the encoded row (`rtq_encode_row(dst_rank, token)`), so Scatter warps
  pop only fully pre-reduced deduplicated rows.
* **Receiver-side top-k combine ready queue.** Each local token starts with
  `arrived_count = 0`; every remote arrival calls `rtq_notify_remote_arrival`
  (`atomicAdd`); the expected-th arrival (full top-k or the active remote-rank
  set) pushes the token, so Reduce warps simply acquire-load and pop
  100%-ready tokens — zero blind `.sys` polling, zero pending-bitmask
  registers.

Verified by host multi-threaded stress (4 producers x 2500 pushes with a
concurrent consumer: every token popped exactly once, per-producer FIFO, no
overflow), epoch/overflow semantics, and the device smoke rail (32 concurrent
lane pushes + a 32-hit pre-reduce notification).

### 4c. Lock-free local pre-reduce — GPU-scope memory-order hardening

Wiring status: library-level primitive in `weave_policy_runtime.cuh`;
not called by any production kernel (verified by host tests and compile gates
only).

`weave::lockfree_local_prereduce_accumulate` replaces the
`while (true) atom.cas(...)` spinlock. Every contributor fire-and-forget
reduces its fragment into the shared accumulator and then bumps the completion
counter; the contributor observing `old == expected - 1` is the unique last one
and owns the forward. There is **no loop over a lock**, hence no livelock by
construction, and the fixed-size fragment loop keeps the register footprint
flat.

**Why a fence is mandatory.** `red.global.add` is fire-and-forget and is *not*
ordered by a later atomic on a **different address**. Under Hopper/Blackwell
weak ordering the last contributor can observe `done_cnt == expected` while
another contributor's reductions are still in flight, and read a
half-accumulated row. The rail therefore inserts all three of:

1. an explicit **`fence.acq_rel.gpu`** *after* the reduction loop and *before*
   the counter increment;
2. the counter increment itself is **`atom.acq_rel.gpu.global.add.u32`**;
3. the last claimer issues **`fence.acquire.gpu`** before reading the row
   (`lockfree_local_prereduce_load`).

The reduction uses `red.global.add.v4.f32` (one instruction per 4 floats) when
both buffers are 16-byte aligned, and `red.global.add.f32` for the scalar tail;
HIP falls back to `atomicAdd`. Both reduction loops stay rolled — unrolling
keeps 4 floats per body live and pushes the primitive past its budget.
`tests/check_dsmem_sched_compile.py` measures **15 registers (sm_103a) / 16
(sm_90a), 0 spill, 0 stack** for the accumulate kernel and 8 registers for the
acquire-fenced read-back, and asserts the PTX really contains
`red.global.add.f32`, `fence.acq_rel.gpu`, `fence.acquire.gpu` and
`atom.acq_rel.gpu.global.add.u32`.

Semantics are verified by host tests: vector and scalar paths, invalid-input
guards, and an 8-thread x 2000-round stress with **exactly one last contributor
per round** (`bad_last = 0`).

### 4d. Inter-cluster layer — sharded L2-resident ready-token rings

Wiring status: library-level primitive in `weave_policy_runtime.cuh`;
not called by any production kernel (verified by host tests and compile gates
only).

`weave::ShardedReadyTokenRing`: `D in {1,2,4,8}` shards, each **2112 bytes**
(one 64 B header line + 256 slots x 8 B), so the full 8-shard set is ~16.5 KiB
and stays L2-resident — zero HBM traffic. Shard `d = token % D` (or a locality
domain). Each scheduler SM keeps its per-token readiness counters in its **own
SMEM** (`ShardTokenTracker`), never touched by another SM.

**The duplicate-consumption bug and its fix.** A design where the owning SM
pops with a *plain store* `tail+1` while a neighbour SM *CAS-steals* the same
cursor hands the same slot to two consumers. This rail therefore keeps three
**explicitly separated, never mixed** modes:

| Mode | Producers | Consumers | Claim primitive | Steal |
|---|---|---|---|---|
| `kRingModeClusterSpsc` (0, **default**) | the bound cluster leader only | the **same** leader only | private cursors, `ld.acquire`/`st.release`, **zero CAS** | not offered |
| `kRingModeMpscEpoch` (1) | many (sender last-hit, receiver top-k) | the bound leader only | one `atomicAdd(tail)` + epoch ready flag, **no CAS** | not offered |
| `kRingModeMpmcSteal` (2) | many | any worker SM | Vyukov per-slot sequences + one bounded `atom.acq_rel` CAS | **unified claim** on both shards |

* **Mode 0 is the cleanest path**: one shard is bound to one cluster's leader
  SM (`owner_sm`), which is both the only producer and the only consumer, so
  there is no cross-SM cursor competition at all. A push or pop from a
  non-owner SM is **rejected**. The leader then fans the token out to its
  sibling CTAs over DSMEM (Section 4e) — that is the whole point of binding a
  shard to a cluster.
* **Mode 1** is the no-CAS MPSC enqueue: producers reserve with a single
  `atomicAdd`, publish an epoch flag, and the leader pops in order. No wrap
  within an epoch; capacity is the per-epoch push budget, overflow is sticky
  and returns false so the caller keeps its legacy path. `ready_ring_begin_epoch`
  resets the cursors and clears the overflow.
* **Mode 2** is the optional work-steal interface. `ready_ring_claim_pop` is
  the **one** claim path: it checks the slot's sequence, then CAS-claims the
  cursor, and only a successful claim hands over the token.
  `ready_ring_pop_preferred_or_steal` routes the home shard *and* the neighbour
  shard through that same function, so a slot can never be consumed twice. A
  failed claim means another consumer progressed — lock-free, no retry storm.

Verified by host tests (all three modes: owner-only rejection, FIFO across
three wrap rounds, full-ring rejection, epoch rollover, sticky overflow) and a
**4-producer x 4-consumer threaded stress moving 8000 tokens with
`dup = 0, missing = 0`**, mirrored by an isomorphic Python model in
`tests/test_two_level_adaptive_policy.py` (6000 tokens, same result).

### 4e. Intra-cluster layer — Cluster DSMEM direct push

Wiring status: library-level primitive in `weave_policy_runtime.cuh`;
not called by any production kernel (verified by host tests and compile gates
only).

`weave::ClusterSchedPacket` is **16 bytes, 16-byte aligned** (`epoch`,
`token_id`, `expert_id`, packed `payload`), i.e. exactly one async-store
transaction. The leader pushes it straight into a peer CTA's SMEM using the
repo's existing primitives in `deep_gemm/include/deep_gemm/ptx/ld_st.cuh`:

* `ptx::mapa_shared` -> `mapa.shared::cluster`
* `ptx::st_async_cluster` -> `st.async.shared::cluster.mbarrier::complete_tx::bytes`
* a remote `mbarrier.arrive.expect_tx.shared::cluster.b64` arming the peer's
  barrier with `sizeof(ClusterSchedPacket)` bytes,
* peer-side `mbarrier.try_wait.parity` (non-blocking `cluster_sched_poll` and
  bounded `cluster_sched_wait`).

The transfer bypasses L2 and HBM entirely (~30-40 cycles).
`cluster_sched_fanout` is the bridge from Section 4d: the leader pops one ready
token from its exclusive shard and pushes it to every sibling CTA of the
cluster. Every CTA initializes its **own** channel, which must sit at the same
SMEM offset in every CTA of the cluster.

The real PTX path is compiled for `sm_103a` and `sm_90a` by
`tests/compile_sm100_dsmem_sched.cu` (kernels annotated `__cluster_dims__`), and
the gate asserts the PTX contains `mapa.shared::cluster`,
`st.async.shared::cluster` and `mbarrier.arrive.expect_tx`. The push kernel
measures **14 registers, 0 spill, 0 stack** (budget <= 24). An isomorphic host
simulation (`ClusterDsmemScheduleSim`, plus the channel parity model) verifies
fan-out delivery to every non-leader peer, FIFO order, 5-round wrap-around past
the 32-slot inbox, `expected_tx == completed_tx` accounting at 16 bytes per
packet, all-or-nothing delivery when an inbox is full, and parity flipping once
per packet — all mirrored bit-exactly in the Python gate.

## 5. Calibrated constants and their real sources

Every constant is a transcribed measurement or a documented derivation from
one; `results/hardware_calibration_3hw.json` records each source sha256 and
derivation rule. No performance claim is made for extrapolated regimes.

**Kernel-revision note.** The SM103, H20 and MI308X (EP8) columns were
calibrated from measurements of a **previous kernel revision** and are kept as
calibration anchors; they are not re-measured on the current kernels. For
example the H20 `F_eff` anchor uses the previous-revision best arm 2515.232 us
at `S_global = 1024`, whereas the current H20 default at the same shape is
2230-2236 us (README, measured-performance table). The `mi308x_ep16` column is
the new 2-Node 16xMI308X EP16 profile, calibrated from the V58 ledger
`results/ep16_mi308x_v58_full5arm.json`; the later V62 champion
(`results/ep16_mi308x_v62_16gpu_champion.json`) is not a calibration source.

| Constant | SM103 (148 SM) | H20 (78 SM) | MI308X EP8 (80 CU) | MI308X EP16 `mi308x_ep16` (80 CU, 2 nodes) |
|---|---|---|---|---|
| `F_eff` (TFLOPS) | 1596.0 = 4200 x 38.0 % Tensor-Active median | 99.3333054 = Eq.(1) unpadded F_gemm(rank0) / 2515.232 us (previous revision) | 99.4168015 = 606.2 T x 16.4 % PMC compute max | 89.022789 = 257.698 GFLOP/rank / 4485.619 us, V58 qwen38max (2.4T) P3 arm |
| `B_hbm` (GB/s) | 7680 (Table I) | 3600 (Table I) | 3684 (Table I = PMC normalization) | 3684 (same silicon) |
| `B_comm` (GB/s) | 780 (Table I) | 370 (Table I) | 287.2 (Table I = PMC xGMI) | 287.2 (per-node serial path) |
| `tau_sync` (us) | 14.0 (Table I t_head) | 18.0 | 22.0 | 22.0 |
| `mu_dedup_thresh` | 1.55386949 = (1286+8954)/(1286+5304), recorded routing replay S=1024 | 3.0 = ceil(max measured mean_hits 2.9311); dedup arm regressed at every measured mu | 1.68861687 = (K/ep)/(1-C(E-E/ep,K)/C(E,K)), E=512 K=10 ep=8; dedup=1 in the verified winner | 1.30577445 = same formula at ep=16 |
| `hbm_headroom_thresh` | 0.76 = measured HBM/TMA mean 76.0 % | 0.455443323 = Eq.(2) at the rank0 max-anchored reference (no PMC in the sweep) | 0.7072 = PMC memory_util max 70.72 % | 0.7072 (same silicon) |
| `tail_ratio_thresh` | 0.295702934 = Eq.(3) tail/total at the verified S_rank=1024 EP4 reference | 0.0238960907 = Eq.(3) at the same rank0 reference | 0.0550832227 = Eq.(3) at the uniform S1024 EP8 reference | 0.164592773 = Eq.(3) at the uniform qwen38max EP16 reference (M=5120, E_local=32) |
| `v_head_hot_budget` | 32 (32-row SF quantum, structural) | 64 = measured best temporal arm (3197.776 us, 1.01144x) | 16 = verified winner pin | 16 = V58 opening-pull budget pin |

Sources:

* SM103: `results/sm103_ep4_prior_verified.json` — optimal arm 710.53 us
  (1.2405x vs official), skew-safe arm 716.67 us; WeaveProfiler/nsys counters
  (HBM/TMA mean 76.0 %, Tensor Active median 38.0 %, NVLink mean 47.9 %).
* H20: `pure21_s1024.json` (**not checked in**; 21-arm pure sweep of a
  previous kernel revision, world=4, S_global=1024; the derived constants and
  the source sha256 are recorded in `results/hardware_calibration_3hw.json`) — best
  arm `WEAVE_ADAPT_BM=1` 2515.232 us = 1.28591x vs official 3234.352 us;
  per-rank routing aggregates and rank-hit multiplicities 2.4189-2.9311;
  `comm_only`+`smem` scatter 3171.168/3172.24 us; scatter-SM subset and local
  dedup regressed (0.87573x / 0.87409x) and stay off in the regime.
* MI308X EP8: `S4_MI308X_FINAL.json` (**not checked in**; EP8 final sweep
  summary of a previous kernel revision, constants and sha256 recorded in the
  calibration JSON) — winner `allopt_c` 2316.36/2341.10 us at
  S1024 and 1301.0/1306.13 us at S128 (~1.12x at S128); PMC utilization ranges;
  0 spills in every production config; the 8-0 wave split runtime-FAILs
  (meta_ready handshake timeout) and is never selected.
* MI308X EP16: `results/ep16_mi308x_v58_full5arm.json` — V58 2-Node
  16xMI308X EP16 6-model x 5-arm measured ledger; P3 model-gated arm 1.4495x
  geomean vs RCCL+AITER; launch regime `mi308x_ep16_balanced_verified`.
* Routing replay: `data/routing_stats_3hw.json` (recorded SM100-family replay).

Reference histograms used by the derivations are labeled proxies consistent
with the measured aggregates (max-anchored for H20, Zipf(0.9) for SM103 — the
validator's convention, uniform for MI308X which stores no histogram).

The Tier-1 engine's integer constants (`PHASE_NK_HW` in `calibrate.py`,
`get_pnk_hardware` in C++) are the same Table-I hardware rows (SM/CU count,
L2 MB, HBM/link caps, GPU-wide peak TFLOPS) plus the architectural SMEM/LDS
capacity (227 KB per CTA on SM100/SM90, 64 KB LDS per MI308X CU), a 16 KB
epilogue/scheduler reserve, and the zero-spill accumulator budget — all
documented at their use sites, no fitted numbers.

## 6. Offline calibration commands

```bash
# Regenerate the committed profile file (pure Python, no torch/CUDA). The
# `regimes` block (4 entries incl. mi308x_ep16_balanced_verified) is written
# only when --s-global is given:
python deep_gemm/mega/calibrate.py --hardware all --s-global 4096 \
    --out results/hardware_calibration_3hw.json

# Regenerate the committed (Phase,N,K) GEMM profile report:
python deep_gemm/mega/calibrate.py --gemm-profiles \
    --out results/gemm_profiles_3hw.json

# Inspect the launch regime for a concrete global sequence length:
python deep_gemm/mega/calibrate.py --hardware sm103 --s-global 4096

# Guided calibration of a NEW GPU from a pure-21-style sweep artifact:
python deep_gemm/mega/calibrate.py --guided --hardware h20 \
    --sweep-json my_pure21_sweep.json --out my_profile.json
```

The guided path expects a pure-21-style sweep JSON (arms with env + median
latency, per-rank routing aggregates, optional PMC block; the required keys are
listed in `calibrate.py`, the original `pure21_s1024.json` snapshot is not
checked in) and applies
the identical derivation rules. After calibrating a new GPU, regenerate the
C++ constants in `weave_cost_model.cuh` to match; the parity gate enforces it.

## 7. Verification

* `tests/test_two_level_adaptive_policy.py` — builds
  `tests/test_cost_model_eq123.cpp` with g++ (`-pthread`) and compares
  **108 OnlineRoutingPolicy instances** (12 routing distributions x 3 hardware
  x 3 knob modes) field-by-field against the Python mirror, floats by binary32
  bit pattern; the calibration constants C++ == calibrate.py == cost_model.py
  == JSON; provenance hashes; pure-21-knob regimes; Eq.(1)-(3) golden
  bit-identity; explicit-pin semantics; **plus the 3-Tier parity grid: 192
  (HW, Precision, Phase, N, K) envelopes + 3456 per-M inner tuples (integer
  codes, 4 precisions x 16 geometries incl. non-standard N/K), 13 SM0 order
  permutations bit-exact (incl. the capacity-overflow path), 36 online
  profile-LUT publications / 10368 per-expert codes, 4 pendulum / no-pair mode
  fixtures, and the pre-reduce / sharded-ring / cluster-DSMEM simulations,
  plus the `mi308x_ep16` profile provenance and its 6 EP16 fail-closed knob
  checks** — **50307 checks, 0 failures**.
* `tests/test_phase_nk_profiles.py` — offline engine gate: 36-cell coverage,
  bit-identical JSON regeneration, every pruning rule **plus the per-backend
  tile legality contract**, L1 swizzle preservation (108 multi-block tuples),
  diversity (**25 envelopes / 17-of-18 independent phase pairs** / 12 cross-hw
  differences / no constant per-hardware envelope), SM0 edge cases + metrics,
  the explicit pairing-mode boundary, no-new-knob invariant — 3515 checks,
  0 failures.
* `tests/check_dsmem_sched_compile.py` — real `nvcc -cubin -Xptxas -v` gate on
  `tests/compile_sm100_dsmem_sched.cu` for **`sm_103a` and `sm_90a`**: 14
  kernels, **0 errors, 0 bytes spill stores/loads, 0 bytes stack frame** on
  every kernel; DSMEM push 14 regs (<= 24), lock-free pre-reduce 15/16 regs
  (<= 28), every other primitive <= 64. A separate `-ptx` pass asserts the real
  instructions are present (`mapa.shared::cluster`, `st.async.shared::cluster`,
  `mbarrier.arrive.expect_tx`, `red.global.add.f32`, `fence.acq_rel.gpu`,
  `fence.acquire.gpu`, `atom.acq_rel.gpu.global.add.u32`). Verbatim report:
  `results/dsmem_sched_compile.json`.
* `tests/check_two_level_compile.py` — the **full production kernel**
  `tests/compile_sm100_two_level.cu` on **`sm_100a` and `sm_103a`**:
  **128 registers, 0 bytes spill stores, 0 bytes spill loads, 80 B stack
  frame** on the default build (`WEAVE_RUNTIME_POLICY=0`); the
  `-DWEAVE_RUNTIME_POLICY=1` arm (online layer compiled) gives 128 registers,
  0 spill, 104 B stack, and the fused shared-expert INT7 arm 128 registers,
  0 spill, 112 B stack; plus `tests/compile_sm90_two_level.cu` on `sm_90a` with
  0 errors. Verbatim report: `results/phase_nk_compile_gates.json`. (Earlier
  revisions of this document quoted a 248 B stack frame; that number belongs to
  a previous kernel revision.)
* `tests/test_cost_model_eq123.cpp` C++ suites — SM0 order span >= 30% /
  variance / no-pure-cold-wave / adjacency / determinism; ready-queue FIFO +
  epoch + overflow + pre-reduce/top-k notify + 4-producer MPSC stress (10000
  tokens); device rail (when a CUDA device is present): E_local=256 accepted /
  257 rejected, device/host bit-identity, SM0 order permutation + `clock64()`
  latency, 32-lane concurrent queue pushes + 32-hit pre-reduce, per-expert
  profile-code loads.
* The existing gates stay green: `check_weave_knob_schema.py` (21 entries),
  `check_kernel_policy_wiring.py` (10 rails), `validate_cost_model_3hw.py`
  (OVERALL=PASS), `test_21_weave_knobs_surface.py` (C++ 1744 + Python 473
  checks), `verify_tunable_params_doc.py` (63 rows / 126 citations),
  `validate_weaveprofiler_trace.py --self-test` (39/39),
  `check_precision_mode_wiring.py`.

## 8. Reality boundary

No new GPU measurements were produced by this change; the framework is
verified by cross-language bit-exact parity, host/device model tests,
multi-threaded host stress and real PTX/cubin compilation. Specifically:

* The Tier-1 selector is **analytical integer physics** (backend-legal pruning
  plus a structural-waste objective), not a GPU autotune. The emitted tuples
  are structural recommendations; `score_ns` is an analytical roofline
  estimate, not a measured latency. Regimes beyond the verified anchor range
  are flagged `extrapolated`.
* The SM0 span/variance numbers are computed on **frozen deterministic
  fixtures** (labeled Zipf proxies + the real H20 rank aggregates), not GPU
  traces. The pendulum fixtures are synthetic multi-hit distributions whose
  hit sets are real multi-expert sets; the 83.3% span reduction is a property
  of those fixtures, not a measured speedup.
* The **cluster DSMEM and sharded-ring device paths were not executed on a
  GPU**. Their semantics are proven by the isomorphic host simulations and the
  threaded exactly-once stress; their lowering is proven by the
  `sm_103a`/`sm_90a` cubin + PTX gates. No target GPU (SM103 / H20 / MI308X)
  was available to these gates, so all C++ host tests were built and run with
  g++ on a Linux userland, as the existing gates do.
* None of these library primitives (Sections 4a-4e) is called by a production
  kernel, and on SM103 the whole online tier is compiled only with
  `-DWEAVE_RUNTIME_POLICY=1`, which no SM103 measurement in this repository
  used (see "Scope and wiring status").
* The SM0 ordering host latency proxy (<= 5.4 us worst case on a non-target
  development host) is reported as measured. The **< 2 us device budget is not
  device-verified**: it is supported by the op-count structure and the
  `clock64()` hook, which runs automatically when a target device is present.
* The three ring modes are **separate protocols**, not one adaptive queue.
  Work stealing is only available in `kRingModeMpmcSteal`; the default
  cluster-exclusive mode intentionally offers none, because a shard bound to
  one leader has no competing consumer to steal from.
* On H20 the measured-best `comm_only`+`smem` scatter structure does not enable
  the chunked streaming-reduce path, so the offline regime keeps
  `WEAVE_REDUCE_WARPS=0` there; the runtime gate is wired and activates on
  `hbm_dedup`+chunk builds.
* `WEAVE_ADAPT_BM=0` (the default) keeps every legacy output bit-identical: the
  profile LUT publishes zeros and the kernels keep their compiled structure.
  The per-expert codes are live recommendations consumed via
  `load_expert_profile_code`; no kernel mainloop was restructured (zero stage
  drops, zero SMEM increase). The two 256-entry code arrays grow the **global**
  `PolicyControl` by ~2 KiB and are filled **in place** — never as a device
  stack temporary.
