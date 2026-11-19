# Weave: Unified C++ MegaMoE for B300 / H20 / MI308X

Weave is a unified C++/CUDA/HIP implementation of the MoE megakernel described in
**Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for
Compute-Communication Overlap** ([arXiv:2609.21483](https://arxiv.org/abs/2609.21483)).
The repository keeps one scheduling contract across three production backends:

1. **Spatial scheduling** `r = (N_c, N_e)`: communication/compute resource split per SM/CU.
2. **Temporal scheduling** `s in {epi, comm}`, a head-pull budget `V_head`, and a fine-grained tail reduce.
3. **CTA-0 runtime cost model**: reads the live post-dispatch histogram inside the kernel and evaluates Eq.(1)-(3).
4. **Explicit knobs and ablations**: `WeaveKnobs` overrides the automatic choice and reproduces `-S`, `-S-T`, and `-S-T-C`.

## Architecture

| Hardware | Language | Precision | Entry point |
|---|---|---|---|
| NVIDIA B300 (SM100/SM103) | CUDA C++ | `mma_type in {fp8xfp4, fp4xfp4 (NVFP4)}` | `sm100_fp8_fp4_mega_moe.cuh` / `sm100_fp4_fp4_mega_moe.cuh` via unified `weave_mega_moe(..., mma_type=...)` |
| NVIDIA H20 (SM90) | CUDA C++ | `fp8` | `sm90_fp8_mega_moe.cuh` |
| AMD MI308X (gfx942) | HIP C++ | `fp8` | `gfx942_fp8_mega_moe.cuh` (`amd/megamoe_amd.hip`) |

## CTA-0 runtime selection

H20 and MI308X use the same publication rail:

```text
dispatch completes
  -> expert_recv_count_sum / plan_counts become the live post-dispatch histogram m_e
  -> CTA 0 reads m_e, computes M = sum_e m_e, reads D_in and M_out
  -> CTA 0 evaluates 2 spatial candidates x 2 temporal modes with Eq.(1)-(3)
  -> CTA 0 writes PolicyControl::decision and release-stores published_epoch
  -> every other CTA acquire-waits on the same epoch and copies the decision
  -> all CTAs execute selected_r_mask / selected_scatter_mode / selected_v_head
```

The control object lives in device global workspace; there is no host round trip and
no dynamic parallel launch. H20 keeps the decision out of the pre-barrier SMEM region
that scatter/combine recycles, and all readers use acquire semantics. The B300
SM100 kernels dispatch through the precision-selectable MegaMoE scheduler; they do
not embed this publication rail. That architectural split is audited explicitly by
`tests/check_kernel_policy_wiring.py`.

### Paper equations (ASCII)

```text
Eq.(1)
F_gemm(r) = sum_e K_e(r) * B_M(r) * 6 * H * I
V_mem(r)  = sum_e K_e(r) * 3 * H * I * b_w + M * b_a * (H + I)

Eq.(2)
T_mma(r) = max(V_mem(r) / B_mem, F_gemm(r) / P_tc(r))
V_epi    = M * (b_a * I + b_o * H)
T_epi    = V_epi / B_epi(N_e)
B300:           T_comp(r) = max(T_mma(r), T_epi(N_e))
H20 / MI308X:   T_comp(r) = T_mma(r) + T_epi(N_e)

Eq.(3)
T_pull(N_c) = D_in  * H * b_a / min(B_link, N_c * b_pull)
T_scat(N)   = M_out * H * b_o / min(B_link, N * b_scat)
T(r, epi)   = max(T_pull(N_c), T_comp(r) + T_scat(N_e))
T(r, comm)  = max(T_pull(N_c) + T_scat(N_c), T_comp(r) + T_hoff)
```

with `K_e(r) = ceil(m_e / B_M(r))`. `b_a`, `b_w`, and `P_tc(r)` are selected by the
first-class precision knob, while the default `FP8xFP4` row is bit-identical to
Paper Table I.

### Table I spatial candidates

```text
B300:   (N_c, N_e) = (4, 8) or (8, 4) per SM
H20:    (N_c, N_e) = (2, 16) or (10, 8) per SM
MI308X: (N_c, N_e) = (0, 8) or (4, 4) per CU
```

## Repository layout

```text
weave/
├── README.md
├── setup.py / CMakeLists.txt
├── deep_gemm/
│   ├── mega/
│   │   ├── __init__.py                  # weave_mega_moe(..., mma_type=...) and existing DeepGEMM wrappers
│   │   └── cost_model.py                # pure-Python mirror of Eq.(1)-(3)
│   └── include/deep_gemm/
│       ├── scheduler/
│       │   ├── mega_moe.cuh             # persistent MegaMoE scheduler
│       │   ├── weave_knobs.cuh          # unified spatial/temporal/dataflow/precision knobs
│       │   ├── weave_cost_model.cuh     # __host__ __device__ Eq.(1)-(3) + CTA-0 selection
│       │   └── weave_policy_runtime.cuh # publication / acquire / all-CTA consumption
│       ├── layout/mega_moe.cuh          # symmetric buffer and workspace layout
│       └── impls/
│           ├── sm100_fp8_fp4_mega_moe.cuh
│           ├── sm100_fp4_fp4_mega_moe.cuh
│           ├── sm90_fp8_mega_moe.cuh
│           ├── gfx942_fp8_mega_moe.cuh
│           └── amd/
│               ├── megamoe_amd.hip      # unified policy-wired EP16 kernel
│               ├── include/             # shared AMD helpers
│               ├── ep8/                 # standalone zero-spill EP8 kernel + build_one.sh + bench_ep8.py
│               ├── ep16/                # EP16 v28 headers + build_port.py + host plan headers
│               ├── tools/               # relax_amdgpu_inline_branches_v1.py + audit_isa.py
│               └── third_party_overlay/ # small vendored overlay headers (no 1100-file CK tree)
├── csrc/
│   ├── apis/mega.hpp                    # unified C++ precision dispatch table
│   └── jit_kernels/
│       ├── heuristics/{mega_moe.hpp, sm90_mega_moe.hpp, sm100.hpp}
│       └── impls/{sm100_fp8_fp4_mega_moe.hpp, sm90_fp8_mega_moe.hpp, gfx942_fp8_mega_moe.hpp}
├── docs/
│   ├── TUNABLE_PARAMETERS.md
│   └── scaling-factor-format.md
└── tests/
    ├── test_cost_model_eq123.cpp
    ├── test_ablation_presets.cpp
    ├── validate_cost_model_3hw.py
    ├── check_kernel_policy_wiring.py
    └── verify_tunable_params_doc.py
```

﻿
## Build

### Dependencies

```bash
git submodule update --init --recursive
# third-party/cutlass  f3fde58372d33e9a5650ba7b80fc48b3b49d40c8
# third-party/deep_jit   e5bdee2bc4ca519eba00cfc5f0c6e950e6a96a16
```

- NVIDIA: CUDA 13.x for SM100/SM103, CUDA 12.8+ for SM90, Python 3.10+, PyTorch 2.x.
- AMD: ROCm 7.2 (`clang`/`ld.lld`/`hipcc`) and gfx942.
- The pure cost-model tests need only `g++`/`clang++`; no GPU is required.

### Install

```bash
pip install -e .
# or a source-only develop install
python setup.py develop
```

### Unified Python entry point

`deep_gemm.mega.weave_mega_moe` is the single public entry point. It forwards all
keyword arguments to the selected backend unchanged, so the existing DeepGEMM
functions remain available and behavior-compatible.

```python
from deep_gemm.mega import get_symm_buffer_for_mega_moe, weave_mega_moe

buf = get_symm_buffer_for_mega_moe(
    group, num_experts=512, num_max_tokens_per_rank=1024, num_topk=10,
    hidden=8192, intermediate_hidden=2048,
    mma_type="fp8xfp4",          # "fp4xfp4" / "nvfp4" selects the NVFP4 backend
)

weave_mega_moe(
    y, l1_weights, l2_weights, buf,
    mma_type="fp8xfp4",          # or "fp4xfp4", "fp8" (SM90), "bf16"
    recipe=(1, 1, 32),
    activation="swiglu",
)
```

`mma_type` accepts `fp8xfp4`, `fp4xfp4`, `nvfp4`, `fp8`, `fp8xfp8`, and `bf16`.
The C++ dispatch table is `deep_gemm::weave_api::weave_mega_moe_dispatch` in
`csrc/apis/mega.hpp`.

### Per-backend JIT use

The original DeepGEMM wrappers are unchanged:

```python
from deep_gemm.mega import fp8_fp4_mega_moe, fp4_fp4_mega_moe, bf16_mega_moe
from deep_gemm.mega import fp8_mega_moe, get_symm_buffer_for_sm90_mega_moe
```

### AMD EP8 standalone kernel

```bash
cd deep_gemm/include/deep_gemm/impls/amd/ep8
bash ./build_one.sh bin_v20_full 0 16 96 "-DWEAVE_ABLATION_PRESET=0"
# The script checks spill=0, scratch=0 from audit_isa.py.
# Run on 8x MI308X:
torchrun --standalone --nproc_per_node=8 ./bench_ep8.py \
  --co ./bin_v20_full/P.co --out-dir /path/to/results --warmup 5 --rounds 3
```

`build_one.sh` uses repository-relative include paths. The only external include root
is the aiter_meta CK tree, overridable with `WEAVE_AITER_SITE`.

### AMD EP16 kernel

```bash
cd deep_gemm/include/deep_gemm/impls/amd/ep16
WEAVE_ABLATION_PRESET=0 python build_port.py
# Output: build/fi0_mode2_p0/P.co
# The build gate requires vgpr_spill_count_max=0, private_segment_fixed_size=0,
# and exactly one s_barrier in the production (FI_LEVEL=0) artifact.
```

`build_port.py` compiles `../megamoe_amd.hip`, the unified policy-wired EP16 kernel,
and uses `amd/tools/audit_isa.py` for the post-link ELF gate. The single-block
`asm volatile` `ep16_ld2_agent_u4` path lives in `amd/ep16/ep16_combine.hpp`.

﻿
## Verification gates

The repository ships seven executable gates. They are meant to be run from the
repository root.

```bash
# 1. Host Eq.(1)-(3), CTA-0 argmin, FP8xFP4/NVFP4/H20/MI308X
g++ -std=c++17 -O2 -Wall -Wextra -I deep_gemm/include \
    tests/test_cost_model_eq123.cpp -o /path/to/test_cost_model
/path/to/test_cost_model

# 2. Ablation presets 0/1/2/3
for p in 0 1 2 3; do
  g++ -std=c++17 -O2 -Wall -Wextra -I deep_gemm/include \
      -DWEAVE_ABLATION_PRESET=$p tests/test_ablation_presets.cpp -o /path/to/test_ablation_$p
  /path/to/test_ablation_$p
done

# 3. CUDA device smoke test
nvcc -std=c++17 -x cu -arch=sm_90a \
  -I deep_gemm/include -I third-party/cutlass/include -I third-party/deep_jit/include \
  tests/test_cost_model_eq123.cpp -o /path/to/test_cost_model_device

# 4. Kernel front-end parse (PTX)
nvcc -std=c++20 -ptx -arch=sm_90a \
  -I deep_gemm/include -I third-party/cutlass/include -I third-party/deep_jit/include \
  -x cu deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh -o /path/to/h20.ptx
nvcc -std=c++20 -ptx -arch=sm_100a \
  -I deep_gemm/include -I third-party/cutlass/include -I third-party/deep_jit/include \
  -x cu deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh -o /path/to/b300_fp8.ptx
nvcc -std=c++20 -ptx -arch=sm_100a \
  -I deep_gemm/include -I third-party/cutlass/include -I third-party/deep_jit/include \
  -x cu deep_gemm/include/deep_gemm/impls/sm100_fp4_fp4_mega_moe.cuh -o /path/to/b300_fp4.ptx

# 5. Three-hardware routing replay
python tests/validate_cost_model_3hw.py

# 6. Source-level policy wiring audit
python tests/check_kernel_policy_wiring.py

# 7. Exact file:line tunable-parameter audit
python tests/verify_tunable_params_doc.py
```

Phase-2 actual results on the build host:

- Host cost-model test: PASS. Default predictions remain the Paper Table I values
  `B300 14.022 us`, `H20 18.107 us`, `MI308X 22.103 us`; the new precision test
  covers `FP8xFP4`, `NVFP4`, `FP8xFP8`, and `BF16`.
- Ablation presets 0/1/2/3: PASS.
- `nvcc` device smoke (`sm_90a`, CUDA 13.3): compilation PASS.
- `nvcc -ptx` front-end parse: H20 (`sm_90a`), B300 FP8xFP4 (`sm_100a`), and
  B300 FP4xFP4 (`sm_100a`) all PASS.
- `validate_cost_model_3hw.py`: PASS on B300, B300 NVFP4, H20, MI308X EP8, and
  MI308X EP16 replay cells.
- `check_kernel_policy_wiring.py`: PASS for all seven audited files.
- `verify_tunable_params_doc.py`: PASS, every citation resolves to the exact
  parameter or its documented kernel-side alias.
- AMD `P.co`: the EP8/EP16 build scripts and ISA gates are checked in. The local
  Phase-2 host has no ROCm/gfx942 toolchain; the AMD production compile and
  `spill=0, scratch=0, s_barrier=1` audit are executed on gfx942 hosts by
  `amd/ep8/build_one.sh` and `amd/ep16/build_port.py`.

## Results and provenance

`results/e2e_3hw_verified.json` is the machine-readable source of truth for the
Phase 4 real-hardware verification. All latency values are measured on the named
devices; every TFLOPS value is a single-GPU value, never a cluster aggregate.

The Architecture table has exactly three hardware rows and presents B300
`mma_type in {fp8xfp4, fp4xfp4 (NVFP4)}` as a tunable parameter, with H20 and
MI308X fixed to `fp8`.

### Measured end-to-end results

| Hardware / scale | Precision (`mma_type`) | Full-output correctness | Measured latency (us) | Compile / ISA audit | Verdict |
|---|---|---|---|---|---|
| 8x NVIDIA L20D (B300/SM103 class); max admissible window 4 GPUs | `fp4xfp4 (NVFP4)` | EP4, S=16, H=6144, I=2048, E=128, K=8; per-rank `calc_diff` 4.672e-7..6.645e-7, `rel_l2` 9.666e-4..1.153e-3, `max_abs_err` 2..4; all 4 ranks PASS | median-of-rank-medians 143.552; worst rank 143.584; 65.757 TFLOPS/GPU nominal (range 62.571..72.148) | ptxas: 0 spill stores, 0 spill loads, 128 registers, 80 B stack | PASS |
| same 8x B300 installation; EP8 not admissible | `fp8xfp4` | EP1 and EP2 built the reference tensors, then the first `run_direct` call did not return; no kernel output or latency is claimed | not measured | ptxas: 0 spill stores, 0 spill loads, 128 registers, 56 B stack; `compute-sanitizer memcheck` found no invalid access before the hang | BLOCKED_RUNTIME_HANG |
| 8x NVIDIA H20 (SM90) | `fp8` | EP8, Qwen3.8-Max, E=512, K=10, H=8192, I=2048; worst-rank `calc_diff` 5.924e-4..5.948e-4, worst `rel_l2` 0.034483, `cos_sim` 0.999405; all ranks PASS | S=128 AUTO=1 1452.512 (AUTO=0 1942.784; 25.24% gain); S=1024 AUTO=1 4977.584 (AUTO=0 4981.344) | `nvcc -ptx -arch=sm_90a` PASS; Phase-3 run did not emit ptxas spill verbose | PASS |
| 8x AMD Instinct MI308X (gfx942) | `fp8` | EP8: 4/4 full-output cells PASS across all 8 ranks; max `rel_l2` 0.001545..0.001908, max row `rel_l2` 0.001952..0.005731, `device_error=0` | qwen38max S=16 687.422 / 713.942; S=128 1539.625 / 1554.185; DeepSeek-V4-Flash S=16 378.761 / 395.421; S=128 699.642 / 709.743; single-GPU TFLOPS 1.595..9.219 | EP8 P.co `d5f977...`; EP16 P.co `2d2bbd...`; both: `spill=0`, `scratch_load=0`, `scratch_store=0`, `s_barrier=1`, `co_passed=true` | EP8 PASS; EP16 compile + ISA PASS |

For the AMD latency cells, the two numbers are cross-rank median and slowest-rank
median, respectively. The exact per-rank measurements, TFLOPS, hashes, and gates
are in `results/e2e_3hw_verified.json`.

**B300 boundary.** GPU0/1 on the B300 host were held by a long-running sglang
service, and the remaining GPUs were intermittently occupied by other sessions;
there was never an admissible 8-GPU window. The delivery therefore reports the
real EP4 NVFP4 result above and explicitly records EP8 as not admitted. The
installed `sgl_deep_gemm 0.2.3.post2` reference control (238.14 us median of rank
medians, 596.42 us copy+kernel) is preserved only under `reference_control` in the
JSON; it is not attributed to this repository.

**AMD EP16 boundary.** EP16 was compiled and ISA-audited with zero spills and
zero scratch, but the two-node 16-GPU runtime was not executed in this phase;
the artifact hash and gate values are recorded separately from the EP8 results.

The previously archived rebuttal artifacts remain byte-for-byte under `results/`:
`results/h20_table4_48cells_ablation_and_costmodel.json`,
`results/ep8_all16_synced_summary.json`,
`results/ep16_baselines_and_roofline.json`, and
`results/cta0_costmodel_latency_summary.json`.

## Tunable Parameters & Code Locations

The tables below are generated from the working tree and verified by `tests/verify_tunable_params_doc.py`. The full version, including host-only legacy knobs and the external CK include contract, is in `docs/TUNABLE_PARAMETERS.md`.

### Unified `WeaveKnobs` and cost model

| Knob | Valid values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|
| `WEAVE_ABLATION_PRESET` | 0/1/2/3 | 0 | Preset 0 = full Weave; 1 = -S; 2 = -S-T; 3 = -S-T-C. Presets 1..3 disable the runtime search and force a fixed spatial/scatter candidate. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:40` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:40` |
| `WEAVE_KNOB_ENABLE_RUNTIME_COST_MODEL` / `enable_runtime_cost_model` | 0/1 | 1 when preset=0, else 0 | Master switch for the CTA-0 Eq.(1)-(3) search. An explicit -D...=1 overrides the preset default. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:47` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:436` |
| `WEAVE_KNOB_COMM_UNITS` | uint32 | 4 | Spatial default N_c per SM/CU when the policy search is disabled. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:51` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:258` |
| `WEAVE_KNOB_COMP_UNITS` | uint32 | 4 | Spatial default N_e per SM/CU when the policy search is disabled. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:54` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:259` |
| `WEAVE_KNOB_DUALGROUP_MIN_TOKENS` | uint32 | 64 | AMD dual-group threshold: t_local at or above this value rebinds communication waves into a second compute group. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:57` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1211` |
| `WEAVE_KNOB_BM_TIER_MODE` | 0/1/2 | 2 | AMD BM tier selector. Mode 2 enables the five-tier {16,32,64,128,256} planner. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:60` | `deep_gemm/include/deep_gemm/impls/amd/include/direct_gather_and_greedy_tiles.hpp:439` |
| `WEAVE_KNOB_BN1` | 64/128/256 | 128 | First-layer N tile width default. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:63` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:263` |
| `WEAVE_KNOB_BN2_MODE` | 0/64/128 | 0 | Second-layer N tile mode; 64 enables the BN64 small-M path. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:66` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:953` |
| `WEAVE_KNOB_BK` | 64/128/256 | 128 | K tile width default. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:69` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:265` |
| `WEAVE_KNOB_S2_NGROUP_MODE` | 0/1/2/4 | 0 | AMD stage-2 N grouping width; 0 = auto. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:72` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:961` |
| `WEAVE_KNOB_HETERO_SM_MASK` | bitmask | 0 | Per-SM heterogeneous role mask seed. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:75` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:267` |
| `WEAVE_KNOB_OVERLAP_PIPELINE` | 0/1 | 1 (0 at preset >= 2) | Temporal overlap switch. 0 falls back to bulk-synchronous stage serialization. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:78` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:706` |
| `WEAVE_KNOB_SCATTER_MODE` / `WEAVE_SCATTER_MODE` | 0/1/2 | 1 | Scatter path selector: 0 = inline epilogue, 1 = comm-queue async, 2 = local dedup first. The AMD EP16 header consumes the WEAVE_SCATTER_MODE alias. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:126` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:52` |
| `WEAVE_KNOB_GEMM_STAGES` | 1..4 | 2 | Mainloop software pipeline depth. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:84` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:270` |
| `WEAVE_KNOB_ALL_WAVE_COMBINE` | 0/1 | 1 (0 at preset >= 2) | Use all waves for the cooperative tail combine; 0 restricts to four waves. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:87` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:194` |
| `WEAVE_KNOB_SPIN_BEFORE_SLEEP` | uint32 | 256 | Polling spin count before issuing the first sleep/backoff. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:90` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:272` |
| `WEAVE_KNOB_SELECTIVE_DEDUP` | 0/1 | 1 (0 at preset >= 3) | Cross-rank selective token dedup. EP16 requires 1. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:93` | `deep_gemm/include/deep_gemm/impls/amd/include/weave_comm_4wave_turbo.hpp:506` |
| `WEAVE_KNOB_REGISTER_DIRECT_COMM` | 0/1 | 1 | Register-direct zero-copy gather (no dense communication staging). | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:96` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:276` |
| `WEAVE_KNOB_PARALLEL_ROUTE_PLAN` / `WEAVE_PARALLEL_ROUTE_PLAN` | 0/1 | 1 | All-CTA parallel route histogram/prefix/scatter. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:99` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:26` |
| `WEAVE_KNOB_LOCAL_FIRST_ORDER` | 0/1 | 1 | LOCAL_READY -> REMOTE_PRIMARY -> LOCAL_COPY worklist priority. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:102` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:278` |
| `WEAVE_KNOB_FUSED_SWIGLU_REQUANT` | 0/1 | 1 | Fuse SwiGLU with per-128 blockscale requantization. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:105` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:280` |
| `WEAVE_KNOB_TAIL_FINE_GRAINED` | 0/1 | 1 (0 at preset >= 2) | Fine-grained tail reduce; 0 uses the constant t_tail_us. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:108` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:359` |
| `WEAVE_KNOB_V_HEAD` | uint32 | 0xffffffff (auto) | Manual V_head override; effective_v_head applies it after Eq.(3). | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:111` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:274` |
| `WEAVE_KNOB_PRECISION_MODE` / `precision_mode` | 0/1/2/3 | 0 (FP8xFP4) | First-class B300 precision selector: 0=FP8xFP4, 1=NVFP4, 2=FP8xFP8, 3=BF16. Mirrored by Python mma_type. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:117` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:368` |
| `precision_mode` struct field | uint32 | 0 | Runtime field in WeaveKnobs; the CTA-0 selector reads it to pick byte widths and P_tc(r). | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:229` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:368` |
| `weave_mega_moe(..., mma_type=...)` (Python) | fp8xfp4/fp4xfp4/nvfp4/fp8/bf16 | sym_buffer.mma_type | Unified `weave_mega_moe` entry point; dispatches to the four existing DeepGEMM wrappers without altering them. | `deep_gemm/mega/__init__.py:535` | `deep_gemm/mega/__init__.py:571` |
| `mma_type` (C++) | fp8xfp4/fp4xfp4/nvfp4/fp8/bf16 | fp8xfp4 | Host dispatch table mapping precision to backend, PrecisionMode, and kernel header. | `csrc/apis/mega.hpp:68` | `csrc/apis/mega.hpp:73` |
| `override_mask` | bit0..bit3 | 0 | Explicit override bits for spatial / scatter / V_head / temporal. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:195` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:287` |
| `override_r_mask` | packed uint32 | auto | Packed r=(N_c,N_e,groups,hetero); applied when bit0 is set. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:196` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:300` |
| `override_scatter_mode` | 0/1/2 | auto | Forced scatter path; applied when bit1 is set. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:197` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:293` |
| `override_v_head` | uint32 | auto | Forced V_head; applied when bit2 is set. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:198` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:307` |
| `selected_r_mask` | packed uint32 | argmin | Published spatial decision consumed by every CTA. | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:98` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:457` |
| `selected_scatter_mode` | 0/1/2 | argmin | Published temporal decision consumed by the scatter path. | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:99` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:458` |
| `selected_v_head` | uint32 | argmin | Published head-pull budget consumed by the pull loops. | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:100` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:459` |

### B300 SM100/SM103 (D1-D7 + precision + 2-CTA multicast + shared L2)

| Knob | Valid values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|
| `DG_TUNE_D1_G` / `DG_ISOLATED_D1_G` | 0/1/2/4/8/16 | 0 | D1 ready-group granularity G; 0 disables grouped publication. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:129` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:520` |
| `DG_TUNE_D2_MODE` / `DG_ISOLATED_D2_MODE` | 0..15 | 0 | D2 dispatch/register-pull mode; >=1 enables warp reconfiguration. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:131` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:52` |
| `DG_TUNE_D2_REG_PULL_BYTES` / `DG_ISOLATED_D2_REG_PULL_BYTES` | 0/1024/2048/4096 | 0 | D2 register-direct pull width in bytes; 0 uses SMEM staging. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:132` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:62` |
| `DG_ISOLATED_D2_D8_SM_LIMIT` | 0..148 | 74 | Maximum SM id participating in the D8 role split. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:133` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:645` |
| `DG_ISOLATED_D2_DISP_REGS` | register count | 0 | Explicit register budget for dispatch warps under D2. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:134` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:851` |
| `DG_ISOLATED_D2_EPI_REGS` | register count | 0 | Explicit register budget for epilogue warps under D2. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:135` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:860` |
| `DG_TUNE_D3_BM` / `DG_D3_BM` | 32/64/96/128/192 | 0 (heuristic) | D3 block-M override. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:137` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:80` |
| `DG_TUNE_D3_BN` / `DG_D3_BN` | 128/256 | 0 (heuristic) | D3 block-N override. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:138` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:87` |
| `DG_TUNE_D3_BK` / `DG_D3_BK` | 64/128/256 | 0 (heuristic) | D3 block-K override. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:139` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:94` |
| `DG_TUNE_D3_STORE_M` / `DG_D3_STORE_M` | 4/8/16/32 | 0 (heuristic) | TMA-store M granularity. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:140` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:102` |
| `DG_TUNE_D3_DYN_EPI` / `DG_D3_DYN_EPI` | 0/1/2 | 0 | Dynamic epilogue pass count; 2 expects STORE_BLOCK_M=16. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:141` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:111` |
| `DG_TUNE_D3_DYN_TMA_A` / `DG_D3_DYN_TMA_A` | 0/1/2 | 0 | Dynamic TMA-A descriptor mode for small M. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:142` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:121` |
| `DG_TUNE_D3_OVERLAP_SMEM` / `DG_D3_OVERLAP_EXPERT_SMEM` | 0/1 | 0 | Overlap expert metadata SMEM with the next expert tile. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:143` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:131` |
| `DG_TUNE_D3_EXPERT_STEP_MODE` / `DG_D3_EXPERT_STEP_MODE` | 0/1/2 | 0 | Expert-step scheduling mode; non-zero raises scheduler register demand. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:144` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:852` |
| `DG_D3_ACCUM_STAGES` | 2/3/4 | 2 | TMEM accumulator stage count. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:145` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:519` |
| `DG_D3_TMA_STORE_STAGES` | 2/3/4/5/6 | 2 | TMA-store ring depth. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:146` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:521` |
| `DG_TUNE_D3_SCHED_STAGES` / `DG_D3_SCHED_STAGES` | 2/3 | 2 | Scheduler look-ahead depth. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:147` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:25` |
| `DG_TUNE_D4_EPW` / `DG_ISOLATED_D4_EPW` | 1/2/4/8/16/32/64 | 0 (heuristic) | Experts-per-wave override for the D4 dynamic queue. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:149` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:73` |
| `DG_TUNE_D4_STATIC` / `DG_ISOLATED_D4_STATIC` | 0/1 | 0 | Use a static expert-order table instead of the ready queue. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:150` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:73` |
| `DG_TUNE_D4_READYQ` / `DG_ISOLATED_D4_READYQ` | 0/1 | 0 | D4 ready-queue scheduling path. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:151` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:2473` |
| `DG_TUNE_D4_MODE` / `DG_ISOLATED_D4_MODE` | 0..3 | 0 | D4 scheduling mode selector. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:152` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:664` |
| `DG_TUNE_D5_DEDUP` / `DG_ISOLATED_D5_DEDUP` | 0/1/2 | 0 | D5 dedup mode: 0 off, 1 dedup, 2 local-dedup pre-reduce. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:155` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:413` |
| `DG_TUNE_D5_SCHED` / `DG_ISOLATED_D5_SCHED` | 0/1 | 1 when dedup=2 | D5 dedup scheduler toggle. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:156` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:88` |
| `DG_TUNE_D6_G` / `DG_ISOLATED_D6_G` | 1/2/4/8/16 | 0 | D6 combine group tile count. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:158` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:155` |
| `DG_TUNE_D6_REDUCE_WARPS` / `DG_ISOLATED_D6_REDUCE_WARPS` | 0/1/2/4/8/16 | 0 (all epilogue warps) | Static cap on combine warps; the kernel takes min(policy N_e, this cap). | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:159` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:162` |
| `DG_TUNE_D7_SCATTER_MODE` / `DG_ISOLATED_D7_SCATTER` | 0/1/2 | 0 | D7 scatter mode: 0 inline epilogue, 1 handoff, 2 local dedup. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:161` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:253` |
| `DG_TUNE_D7_SMEM_ALIAS` | 0/1 | 1 | Alias the D7 staging SMEM with reusable epilogue storage. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:163` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:574` |
| `DG_TUNE_D7_WARPS` / `DG_TUNE_D7_SCATTER_WARPS` | 0/2/4/8 | 0 (auto) | D7 scatter executor warp count. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:164` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:217` |
| `DG_TUNE_D7_HANDOFF_WAIT` | 0/1 | 1 | Wait for the D6 ready epoch before the D7 handoff. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:165` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:229` |
| `DG_SHARED_L2_EARLY_BLOCKS` | 0..148 | 10 | Number of early blocks that seed the shared-L2 cache. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:264` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:170` |
| `kEnableMulticast` / `cluster_size` | true / 2 | true | 2-CTA multicast invariant: cluster_size=2 on the host; kEnableMulticast is the compiled kernel-side assertion. | `csrc/jit_kernels/heuristics/mega_moe.hpp:109` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:66` |

### H20 SM90

Production delivery configuration (DELIVERY_20261116): `DG_MOE_SWAPAB_AUTO=1`, `DG_MOE_SWAPAB_EPI_WG=4`, `DG_MOE_SWAPAB_MAX_LOAD=100000`, `DG_MOE_SWAPAB_PAD_PCT=25`, `DG_MOE_SWAPAB_MIN_LOAD=8`; source defaults below stay backward-compatible with the official rule.

| Knob | Valid values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|
| `DG_MOE_SWAPAB_AUTO` / `kSwapABActive` | 0/1 | 0 | Two-layer AUTO gate. 1 makes the swapAB decision a superset of the official n<=24 decision and lets pad_frac widen the tile. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:616` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:367` |
| `DG_MOE_SWAPAB_EPI_WG` / `kNumEpilogueWarpgroups` | 2/4 | 2 | Requested epilogue warpgroups. 4 enables BN=256 + 4WG only when pad_ok; otherwise returns 2. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:612` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:370` |
| `DG_MOE_SWAPAB_MAX_LOAD` / `kSwapABActive` | int | 24 compact / 16 other | Explicit swapAB tokens-per-expert ceiling when AUTO=0. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:631` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:367` |
| `DG_MOE_SWAPAB_PAD_PCT` / `kSwapABActive` | percent | 25 | Minimum padding waste that justifies the wide tile. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:601` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:367` |
| `DG_MOE_SWAPAB_MIN_LOAD` / `kSwapABActive` | tokens/expert | 8 | Token floor below which the wide tile is rejected (n<=5 is a measured regression). | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:602` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:367` |
| `kSwapSFSubBlocks` | 1/2 | derived | One activation SF per 64 output columns. BLOCK_N=128 -> 1; BLOCK_N=256 -> 2. Eliminates the 0.366 -> 5.94e-04 drift. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:1095` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:5199` |
| `SWAPAB_FINE_LADDER` / `n_swap` | 8/16/24/32/40/48/56/64 | 8-token ladder | Both L1 and L2 dispatch points select the WGMMA N_SWAP template in 8-token steps. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:1096` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:3599` |
| `AGY_MCAST` | 1/2 | 1 | Adaptive multicast: 2 = pair M blocks; 1 = single-tile stream. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:442` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:529` |
| `AGY_SWIZZLE_TILES` / `kAgySwizzleTiles` | 0/1 | 1 in AGY presets | Tile-order swizzle for L1/L2 expert caches. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:429` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:597` |
| `AGY_HOT_COLD_L1` / `kAgyHotColdMode` | 0/1/2 | 0 | Hot/cold expert pairing inside each experts-per-wave window. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:481` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:485` |
| `AGY_DEDUP_DISP` | 0/1 | 0 | Dedup at dispatch time; duplicate memberships reuse the primary payload. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:411` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:547` |
| `AGY_DEDUP_SCAT` | 0/1 | 0 | Dedup-aware scatter path. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:413` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:559` |
| `AGY_REG_COMB` / `kAgyRegComb` | 0/1/2 | 0 | Flattened direct-register combine; avoids SMEM round trips. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:418` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:580` |
| `AGY_ALL_IDLE_COMB` | 0/1 | 0 | Let all idle epilogue warps join the tail reduce. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:515` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:489` |
| `AGY_DYN_SCHED` / `kAgyDynSched` | 0/1 | 0 | Dynamic queue scheduler instead of the static task cursor. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:403` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:494` |
| `AGY_INTERLEAVE_L1_L2` / `kAgyInterleaveMode` | 0..6 | 0 | Interleave L1/L2 tile execution; 4 = epw L1/L2, 5 = B300-style, 6 = wave-lagged. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:521` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:375` |
| `AGY_SCATTER_ROLE_DISP` | 0/1 | 0 | Dispatch warps execute scatter instead of dedicated epilogue warps. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:528` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:381` |
| `AGY_D_MASK_MODE` / `kAgyHeteroD8D4` | 0..5 | 0 | Heterogeneous D8/D4 SM-role mask mode. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:464` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:576` |
| `AGY_PIPE_WGMMA` | 0/1 | 0 | Pipelined WGMMA issue across K blocks. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:405` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:537` |
| `AGY_SWAP_PIPE` | 0/1 | 0 | Pipelined swapAB WGMMA path. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:407` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:542` |
| `AGY_FAST_PULL` | 0/1 | 0 | Fast dispatch pull path. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:409` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:554` |
| `AGY_ADAPT_BM` | 0/1 | 0 | Per-expert adaptive BLOCK_M tiers. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:394` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:512` |

### MI308X gfx942

| Knob | Valid values | Default | Mechanism | Host parse | Kernel consumption |
|---|---|---|---|---|---|
| `WEAVE_ABLATION_PRESET` | 0/1/2/3 | 0 | EP16 build-time ablation selector; preset 3 also switches WEAVE_SCATTER_MODE to 0. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:82` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1206` |
| `WEAVE_SCATTER_MODE` | 0/1/2 | 2 (0 at preset 3) | EP16 scatter and return mode; 2 enables the node-partial path. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:106` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:52` |
| `WEAVE_KNOB_BM_TIER_MODE` | 0/2 | 2 | Five-tier BM planner ({16,32,64,128,256}) when non-zero. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:109` | `deep_gemm/include/deep_gemm/impls/amd/include/direct_gather_and_greedy_tiles.hpp:439` |
| `WEAVE_KNOB_SELECTIVE_DEDUP` | 1 | 1 | EP16 requires selective input dedup in every ablation arm. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:109` | `deep_gemm/include/deep_gemm/impls/amd/include/weave_comm_4wave_turbo.hpp:506` |
| `WEAVE_KNOB_DUALGROUP_MIN_TOKENS` | uint32 | 64 | Local-token threshold for rebinding comm waves into a second compute group. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:57` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1211` |
| `WEAVE_KNOB_OVERLAP_PIPELINE` | 0/1 | 1 | 0 disables the EP16 overlap pipeline and runs the bulk-synchronous path. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:78` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1264` |
| `WEAVE_KNOB_ALL_WAVE_COMBINE` | 0/1 | 1 | 0 limits combine to 64 workers instead of 96. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:87` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:194` |
| `WEAVE_KNOB_BN2_MODE` | 0/64/128 | 0 | Stage-2 BN mode; 64 selects the small-M 1x4 BN64 kernel. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:66` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:953` |
| `WEAVE_KNOB_S2_NGROUP_MODE` | 0/1/2/4 | 0 | Stage-2 N-group width. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:72` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:961` |
| `WEAVE_TH_BM64` | uint32 | 16 | Small-M threshold for BM64 tiles. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:107` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:542` |
| `WEAVE_TH_BM128` | uint32 | 96 | Small-M threshold for BM128 tiles. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:107` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:545` |
| `WEAVE_PARALLEL_ROUTE_PLAN` | 0/1 | 1 | All-CTA parallel route histogram/prefix/scatter. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:101` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:26` |
| `V1864Roles` / `WEAVE_PARALLEL_ROUTE_PLAN` | struct | roles={80,0,80} | Wave-role descriptor: {non_epilogue_waves, comm_waves, mma_waves}. CTA-0 remaps it from selected_r_mask. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:101` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1200` |
| `EP16_SLOT_CHUNK` / `ep16t_slot_chunk` | 1/2/4/8/16/32/64 | adaptive | Bulk DISPATCH descriptor granularity; host clamps to 1..64 and the plan validates power-of-two. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/run_turbo512_vs_rccl.py:159` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:289` |
| `ep16t_ret_chunk` | derived 1/2/4 | derived | RETURN descriptor granularity, derived from slot_chunk and capped at 4. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:303` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:308` |
| `meta_chunk_slots` | >=1 | max(256, t_local) | Metadata slots per RDMA WR descriptor. | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:55` | `deep_gemm/include/deep_gemm/impls/amd/ep16/host/ep16_turbo_layout.hpp:338` |
| `ep16_ld2_agent_u4` | single-block asm volatile | enabled | Single-block 2x-MLP global_load_dwordx4 sc1 helper used by the EP16 bucket reduce (v28). | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:27` | `deep_gemm/include/deep_gemm/impls/amd/ep16/ep16_combine.hpp:62` |
| `FI_LEVEL` | 0/1/2 | 0 | WeaveProfiler instrumentation level: 0 production, 1 open/close, 2 full events. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:100` | `deep_gemm/include/deep_gemm/impls/amd/third_party_overlay/fi_lite_hip.h:32` |

﻿
## Citation

If you use this repository, please cite the Weave paper:

```bibtex
@misc{huang2026weavefinegraineddynamicsm,
      title={Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap}, 
      author={Ziyu Huang and Yangjie Zhou and Chenhao Zhu and Peng Yu and Zihan Liu and Jinyu Liu and Shulai Zhang and Xingxun Tang and Hongzhe Yan and Xinhao Luo and Minyi Guo and Xiu Lin and Yinghao Yu and Guodong Yang and Liping Zhang and Shixuan Sun and Jingwen Leng},
      year={2026},
      eprint={2609.21483},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2609.21483}, 
}
```

Paper: <https://arxiv.org/abs/2609.21483>

DeepGEMM, CUTLASS, and the AMD aiter/CK components retain their original licenses;
see `LICENSE` and `third-party/`.
