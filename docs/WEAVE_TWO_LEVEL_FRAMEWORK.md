# Weave Two-Level Scheduling Framework: Offline Hardware Calibration and Online Routing-Adaptive Selection

This document describes the architecture of Weave's scheduling engine built over the unified **14-knob surface** (Spatial 5, Temporal 3, Implementation 6; see [TUNABLE_PARAMETERS.md](TUNABLE_PARAMETERS.md)). Every online decision operates through the `0` / `auto` defaults, while any explicit `WEAVE_*` override remains deterministic.

```
Tier 0  (Hardware Profile)                   once per GPU
        results/hardware_calibration_3hw.json
        peak TFLOPS / HBM BW / NVLink-xGMI-RDMA BW / SM-CU count /
        L2 & SMEM-LDS capacity / register budget
            |
Tier 1  (Independent Operator Profiles)      once per (Hardware, Precision, Phase, N, K)
        GEMM1 (N1=2I, K1=H, SwiGLU+quant) and GEMM2 (N2=H, K2=I, FC2+scatter)
        solved independently over backend-legal tile ladders
        -> PhaseEnvelope (BN, BK, max_BM, mainloop_stages, L2 swizzle, interleave)
        -> Per-M Inner-Tuple LUT (GemmInnerProfile, multi-stage epilogue EPI=1..4)
        results/gemm_profiles_3hw.json
            |
Tier 2  (Per-GPU Live Routing Adaptation)    every MoE step
        Reads per-GPU routing histogram (m_e, D_in, M_out)
        Selects multi-granularity spatial ratio (cross-SM/CU + intra-SM/CU),
        temporal role handoffs (V_head, N-tile ready group, scatter owner),
        and Dispatch / Scatter data paths (dedup, reg / smem / hbm staging)
```

## 1. Multi-granularity spatial & temporal model

1. **Cross-GPU independence**: under skewed MoE routing, communication-heavy GPUs and compute-heavy GPUs choose distinct $(r_g, s_g)$ schedules locally rather than sharing one cluster-wide static split, interacting solely via peer-to-peer arrival counters and token-ready flags.
2. **Cross-SM/CU & intra-SM/CU spatial partitioning**: rather than forcing every SM/CU on a GPU to run the same warp split, Weave supports mixing `k` communication-rich SMs/CUs with `N - k` compute-rich SMs/CUs alongside independent active SM subsets for Dispatch (`WEAVE_DISPATCH_ACTIVE_SMS`) and Scatter (`WEAVE_SCATTER_ACTIVE_SMS`), keeping Tensor/Matrix Cores active on all `N` SMs/CUs.
3. **Dynamic temporal role transitions**:
   - **Head**: helper warps assist opening remote pulls up to `WEAVE_V_HEAD`, while `WEAVE_READY_GROUP_SIZE` starts GEMM1 at `N`-tile granularity;
   - **Middle**: freed communication warps transition to async FC2 scatter (`WEAVE_SCATTER_OWNER=comm_only` with `smem` or `hbm` staging), or epilogue warps stream multi-stage register-direct writes (`epilogue_only` with `reg`);
   - **Tail**: per-token (`k/k`) readiness wakes all warps/wavefronts to perform cooperative Combine reduction.

## 2. Independent GEMM1 & GEMM2 envelope selection

`deep_gemm/mega/calibrate.py` (`select_phase_envelope` / `select_phase_inner_profile`) and its C++ counterpart in `weave_cost_model.cuh` use integer-only arithmetic so host and device selections match bit-for-bit. GEMM1 (`N1 = 2I, K1 = H`) and GEMM2 (`N2 = H, K2 = I`) are solved independently over each backend's legal tile set:

| Backend | `BLOCK_N` | `BLOCK_K` | `BLOCK_M` |
|---|---|---|---|
| NVIDIA SM103 | `64 / 128 / 256` | `128 / 256` | `16, 32, 64, 80, 96, 128, 192, 240, 256` |
| NVIDIA H20 | `64 / 128 / 256 / 512` | `128 / 256` | `16, 32, 64, 128, 192, 256` |
| AMD MI308X | `64 / 128 / 256` | `64 / 128 / 256` | `16, 32, 64, 128` |

For each candidate `(BN, BK, max_BM)`, `num_stages` is maximized within the remaining SMEM/LDS budget after accounting for Dispatch/Scatter staging (`WEAVE_REG_PULL_BYTES` register-direct pulls preserve SMEM/LDS for deeper GEMM pipelines), and `GemmInnerProfile` selects the per-`M` bucket tile height, store height, and multi-stage epilogue register depth (`EPI = 1..4`).

Regenerate the multi-hardware GEMM profile ledger with:

```bash
python deep_gemm/mega/calibrate.py --gemm-profiles --out results/gemm_profiles_3hw.json
```
