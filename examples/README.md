# Reproducible WeaveProfiler Presets for Qwen3.8-Max

This directory contains executable presets for the three matched
`Qwen3.8-Max, EP8, S=1024` WeaveProfiler timelines plus the two-node
`EP16, S=128` MI308X path. Every preset is expressed with unified `WEAVE_*`
names. No legacy tuning prefixes are used.

The files are:

```
examples/
  README.md                                  this file
  README_zh.md                               Simplified Chinese version
  sm103_qwen38max_ep8_s1024.sh               148-SM SM103, fp8xfp4 (+ optional fp4xfp4)
  h20_qwen38max_ep8_s1024.sh                 8x H20, exact tuned spec string
  mi308x_qwen38max_ep8_s1024.sh              8x MI308X EP8 build + run
  mi308x_qwen38max_ep16_s128.sh              2-node 16x MI308X EP16 build + run
  profile_and_render_timeline.py             3-axis table printer + render driver
```

The annotated previews live one level above this directory:

```
annotated_timelines/
  weaveprofiler_sm103_timeline.png
  weaveprofiler_h20_timeline.png
  weaveprofiler_mi308x_timeline.png
  render_stats.json
```

---

## 1. The 3-axis taxonomy

Every knob below belongs to exactly one axis. The axis names are descriptive
labels introduced by these examples; they are not kernel enum values.

| Axis | Question it answers | Examples |
|---|---|---|
| **Spatial** | How is work partitioned across SMs/CUs and tiles? | `WEAVE_BLOCK_M/N/K`, `WEAVE_STORE_BLOCK_M`, `WEAVE_BM_TIER_MODE`, wave roles |
| **Temporal** | When do stages overlap, and how is readiness ordered? | `WEAVE_SCATTER_MODE`, `WEAVE_OVERLAP_PIPELINE`, `WEAVE_INTERLEAVE_L1_L2`, `WEAVE_READY_GROUP_SIZE`, `WEAVE_COMBINE_*` |
| **Implementation** | Which code path or numeric mode is selected? | `WEAVE_PRECISION_MODE`, `WEAVE_DEDUP_*`, `WEAVE_REG_PULL_BYTES`, `WEAVE_SCATTER_SMEM_ALIAS`, `WEAVE_PROFILER_LEVEL` |

This split makes the presets auditable: a change in tile shape cannot hide
inside a scheduler knob, and a capture-only switch cannot masquerade as a
production tuning value.

---

## 2. Evidence classes

The tables use six evidence classes. They are deliberately strict.

| Class | Meaning |
|---|---|
| `trace-metadata` | A literal field in the archived trace, including the H20 `selected_cfg` string and the MI308X grid/wave metadata. |
| `archived-script` | A literal value in the archived capture or build script that produced the trace. |
| `current-build-default` | The active default of the current unified build helper. |
| `heuristic-derived` | A value produced by the documented block-shape heuristic, not a literal trace field. |
| `capture-only` | Set only while capturing a timeline; not a production tuning value. |
| `documented-behavior` | The documented semantics of a reserved or auto value. |

A value is never presented as `trace-metadata` unless the trace actually
contains it.

---

## 3. Preset matrix

| Preset | Hardware | Parallelism | Workload | Script | Preview |
|---|---|---|---|---|---|
| `sm103` | NVIDIA SM103, 148 SMs | 8 ranks, EP8 | Qwen3.8-Max, S=1024 | `sm103_qwen38max_ep8_s1024.sh` | `weaveprofiler_sm103_timeline.png` |
| `h20` | NVIDIA H20, 78 SMs | 8 ranks, EP8 | Qwen3.8-Max, S=1024 | `h20_qwen38max_ep8_s1024.sh` | `weaveprofiler_h20_timeline.png` |
| `mi308x_ep8` | AMD MI308X, 80 CUs | 8 ranks, EP8 | Qwen3.8-Max, S=1024 | `mi308x_qwen38max_ep8_s1024.sh` | `weaveprofiler_mi308x_timeline.png` |
| `mi308x_ep16` | AMD MI308X, 80 CUs | 16 ranks, EP16, 2 nodes | Qwen3.8-Max, S=128 | `mi308x_qwen38max_ep16_s128.sh` | none (no matched 3-panel preview) |

All three previews use the same `Qwen3.8-Max, EP8, S=1024` workload and the
same 3-panel layout, so they are directly comparable.

---

## 4. SM103 (148-SM): tuned `fp8xfp4` preset

### 4.1 Parameter table

| Axis | Parameter | Value | Evidence | Notes |
|---|---|---:|---|---|
| Spatial | `WEAVE_BLOCK_M` | 64 | heuristic-derived | expected tokens/expert = 1024 * 8 * 10 / 512 = 20.0, in the <= 32.5 branch |
| Spatial | `WEAVE_BLOCK_N` | 128 | heuristic-derived | fixed BN of the same block-shape branch |
| Spatial | `WEAVE_BLOCK_K` | 128 | heuristic-derived | K tile for the fp8 activation element width |
| Spatial | `WEAVE_STORE_BLOCK_M` | 32 | heuristic-derived | TMA-store M granularity |
| Spatial | `WEAVE_ENABLE_MULTICAST` | 1 | current-build-default | required by the 148-SM dataflow |
| Temporal | `WEAVE_SCATTER_MODE` | 1 | archived-script | dispatch-warp scatter, communication-offloaded ready queue |
| Temporal | `WEAVE_READY_GROUP_SIZE` | 2 | archived-script | ready-group granularity |
| Temporal | `WEAVE_COMBINE_GROUP_TILES` | 2 | archived-script | combine tile grouping |
| Temporal | `WEAVE_COMBINE_REDUCE_WARPS` | 16 | archived-script | 16 reduce warps, 128 tiles per warp |
| Temporal | `WEAVE_SHARED_L2_EARLY_BLOCKS` | 77 | capture-only | production default is 10 |
| Implementation | `WEAVE_PRECISION_MODE` | 0 | archived-script | fp8xfp4; optional 1 = fp4xfp4 |
| Implementation | `WEAVE_DEDUP_MODE` | 1 | archived-script | unique (token, destination) buckets |
| Implementation | `WEAVE_EXPERTS_PER_WAVE` | 2 | archived-script | two experts per wave |
| Implementation | `WEAVE_REG_PULL_BYTES` | 4096 | archived-script | 4 KiB register pull |
| Implementation | `WEAVE_SCATTER_SMEM_ALIAS` | 1 | archived-script | alias scatter staging with the epilogue buffer |
| Implementation | `WEAVE_SCHED_MODE` | 3 | documented-behavior | reserved; currently identical to mode 0 |
| Implementation | `WEAVE_ABLATION_PRESET` | 0 | archived-script | full Weave preset |

### 4.2 Evidence boundary

- The archived trace has no `selected_cfg` field. The scalar knobs are taken
  from the archived capture worker script.
- The tile `64 x 128 x 128` plus `store_m = 32` is not a literal trace field.
  It is the documented block-shape branch for expected tokens per expert equal
  to 20.0. The archived run enabled that tuned block-shape heuristic; in the
  unified code the explicit `WEAVE_BLOCK_*` / `WEAVE_STORE_BLOCK_M` overrides
  select the same branch, so no extra master switch is required.
- `WEAVE_SCHED_MODE=3` is reserved. It is kept only because the archived run
  set it; mode 3 currently behaves like mode 0.
- `WEAVE_SHARED_L2_EARLY_BLOCKS=77` was used only for the timeline capture.
  The clean latency measurement used the default value 10. Set
  `WEAVE_CAPTURE=1` to reproduce the capture-time value.
- The archived clean measurement reports a median of 359.55 us, an exact
  SHA-256 match on 8 of 8 ranks, 16 reduce warps x 128 events covering tiles
  0..2047, a speedup of 1.0168, and a minimum cosine of about 0.99999988.
  These numbers belong to the archived run; they are not re-measured here.
- The current unified benchmark uses random routing. The archived trace used a
  captured routing table. The preset is reproducible; a byte-identical trace
  is not claimed.

### 4.3 Run

```bash
export WEAVE_UNIFIED_ROOT=/path/to/weave_unified_cpp
bash examples/sm103_qwen38max_ep8_s1024.sh
```

Optional `fp4xfp4` mode:

```bash
PRECISION=fp4xfp4 bash examples/sm103_qwen38max_ep8_s1024.sh
```

`fp4xfp4` is an optional precision switch, not a separately re-tuned preset.
The archived tile values were tuned for `fp8xfp4`.

---

## 5. H20 (SM90): exact tuned preset

### 5.1 The exact ground-truth string

The archived trace contains this literal `otherData.meta.selected_cfg`:

```text
fast_host=1,fused=1,interleave=5,epw=8,compact=1,dyn=1,alias=1,scat_disp=1,d_mask_mode=1,bm=32,bn=256,bk=128,wgs=4,swap=1,stages=0,swap_pipe=1,fast_swap_epi=1,rank_cache=1,fast_clean=1,fast_pull=1,dedup_disp=1,dedup_tier=1,reg_comb=1,fast_comb=1,all_idle_comb=1
```

It is passed through the unified public entry `WEAVE_H20_CONFIG`. The keys
inside the string are H20 spec fields, not legacy environment-variable names.

### 5.2 Mapping of every active key

| Axis | Unified name / field | Value | Evidence | Notes |
|---|---|---:|---|---|
| Spatial | `WEAVE_H20_CONFIG.bm` | 32 | trace-metadata | spec field; no standalone public env binding |
| Spatial | `WEAVE_H20_CONFIG.bn` | 256 | trace-metadata | spec field; no standalone public env binding |
| Spatial | `WEAVE_BLOCK_K` | 128 | trace-metadata | `bk=128` |
| Spatial | `WEAVE_H20_CONFIG.wgs` | 4 | trace-metadata | 4 epilogue warp groups |
| Spatial | `WEAVE_H20_CONFIG.swap` | 1 | trace-metadata | swap-AB layout |
| Temporal | `WEAVE_INTERLEAVE_L1_L2` | 5 | trace-metadata | `interleave=5`: expert-per-wave alternation + L2 K-block dependency |
| Temporal | `WEAVE_DYN_SCHED` | 1 | trace-metadata | `dyn=1`: dynamic expert cursor |
| Temporal | `WEAVE_H20_CONFIG.stages` | 0 | documented-behavior | runtime auto-selects pipeline depth; NOT a zero-stage pipeline |
| Temporal | `WEAVE_SWAP_PIPE` | 1 | trace-metadata | `swap_pipe=1` |
| Implementation | `WEAVE_H20_CONFIG.epw` | 8 | trace-metadata | 8 experts per wave; spec field only |
| Implementation | `WEAVE_SCATTER_ROLE_DISP` | 1 | trace-metadata | `scat_disp=1` |
| Implementation | `WEAVE_HETERO_MASK_MODE` | 1 | trace-metadata | `d_mask_mode=1` |
| Implementation | `WEAVE_DEDUP_DISPATCH` | 1 | trace-metadata | `dedup_disp=1` |
| Implementation | `WEAVE_DEDUP_SCATTER` | 0 | trace-metadata | `dedup_scat` is absent, so scatter dedup is OFF |
| Implementation | `WEAVE_REG_COMBINE` | 1 | trace-metadata | `reg_comb=1` |
| Implementation | `WEAVE_ALL_IDLE_COMBINE` | 1 | trace-metadata | `all_idle_comb=1` |
| Implementation | `WEAVE_FAST_PULL` | 1 | trace-metadata | `fast_pull=1` |
| Implementation | `WEAVE_H20_CONFIG.dedup_tier` | 1 | trace-metadata | spec field; no standalone public env binding |
| Implementation | `WEAVE_H20_CONFIG.*` | exact string | trace-metadata | `fast_host`, `fused`, `compact`, `alias`, `fast_swap_epi`, `rank_cache`, `fast_clean`, `fast_comb` |

Important details:

- The trace has `dedup_disp=1` and `dedup_tier=1`, but **no** `dedup_scat=1`.
  Scatter dedup is off.
- `stages=0` means the runtime chooses the pipeline depth. It does not mean a
  zero-stage pipeline and it is not `WEAVE_GEMM_STAGES=0`.
- Several spec keys (`bm`, `bn`, `wgs`, `swap`, `epw`, `fast_host`, `fused`,
  `compact`, `alias`, `fast_swap_epi`, `rank_cache`, `fast_clean`,
  `dedup_tier`, `fast_comb`) have no independent public `WEAVE_*` environment
  binding. The only exact reproduction entry is the full
  `WEAVE_H20_CONFIG` string.

### 5.3 Run

```bash
export WEAVE_UNIFIED_ROOT=/path/to/weave_unified_cpp
bash examples/h20_qwen38max_ep8_s1024.sh
```

The script runs the unified SM90 correctness suite on a production SM90 shape
with the exact `WEAVE_H20_CONFIG` exported. The unified suite does not contain
the exact Qwen3.8-Max S=1024 cell; the exact timeline capture is archived. To
replay that timeline, use `profile_and_render_timeline.py --preset h20 --trace`
or set `WEAVE_H20_TRACE`.

---

## 6. MI308X EP8: tuned build + run preset

### 6.1 Parameter table

| Axis | Parameter / field | Value | Evidence | Notes |
|---|---|---:|---|---|
| Spatial | gridX / waves_per_cu | 80 / 8 | trace-metadata | 80 unique physical CUs |
| Spatial | BN / BK | 128 / 128 | current-build-default | fixed tile constants |
| Spatial | `WEAVE_BM_TIER_MODE` | 2 | current-build-default | M tiers 16/32/64/128/256 |
| Temporal | `WEAVE_SCATTER_MODE` | 2 | current-build-default | local-dedup / node-partial pre-reduce |
| Temporal | `WEAVE_OVERLAP_PIPELINE` | 1 | current-build-default | compute-communication overlap |
| Temporal | `WEAVE_DUALGROUP_MIN_TOKENS` | 64 | current-build-default | local tokens 1024 / 8 = 128 >= 64 -> 8-wave dual compute |
| Implementation | `WEAVE_ABLATION_PRESET` | 0 | current-build-default | full Weave preset |
| Implementation | `WEAVE_SELECTIVE_DEDUP` | 1 | current-build-default | required by the EP8 path |
| Implementation | `WEAVE_ALL_WAVE_COMBINE` | 1 | current-build-default | all waves join the tail combine |
| Implementation | `WEAVE_BN2_MODE` | 0 | current-build-default | auto second-stage BN |
| Implementation | `WEAVE_S2_NGROUP_MODE` | 0 | current-build-default | auto N-group width |
| Implementation | `WEAVE_PARALLEL_ROUTE_PLAN` | 1 | current-build-default | 80-CTA parallel route plan |
| Implementation | `WEAVE_PROFILER_LEVEL` | 2 | capture-only | production is 0 |

Fixed build macros in the current helper: `WEAVE_PAR_RP=1`,
`WEAVE_CK_G2_LDS_ONLY=0`, `WEAVE_TILE_TRACE_ONLY=1`,
`WEAVE_ADAPTIVE_POST_TIMEOUT=1`, `WEAVE_S16_PLAN_CAP=0`,
`WEAVE_S16_STATIC_DISPATCH=0`, `WEAVE_OPT_RELAXED_SPIN=1`,
`WEAVE_OPT_LOCAL_SCOPE=1`.

### 6.2 Evidence boundary

- The archived trace metadata records `gridX=80`, `waves_per_cu=8`,
  `unique_physical_cus=80`, `S=1024`, and `rank=2`. It does not contain the
  full compile-time knob string.
- The archived EP8 timeline was produced by the archived EP8 source tree. This
  script rebuilds the **current unified EP8 kernel** with the same tuned
  preset. It does not claim a byte-identical binary or timeline.
- `WEAVE_DUALGROUP_MIN_TOKENS=64` is reached at S=1024 because the local token
  count is 128. The 8-wave dual-compute (4+4) path is therefore active.
- `WEAVE_PROFILER_LEVEL=2` is capture-only. Production uses 0.

### 6.3 External artifacts

The benchmark driver is not self-contained. It needs three archived
host/runtime artifacts that are not vendored in the public repository:

| Variable | Meaning |
|---|---|
| `WEAVE_EP8_LAUNCHER` | frozen host launcher shared object |
| `WEAVE_EP8_IPC_LIB` | IPC driver shared object |
| `WEAVE_EP8_ROUTES` | directory containing the Qwen3.8-Max route table |

The script checks all three and fails fast with a clear message if one is
missing. The Python reference, ABI, and IPC wrapper modules are present in the
source tree under `deep_gemm/include/deep_gemm/impls/amd/ep8/support/`.
The archived host launcher must also match the ABI of the freshly built
`P.co`; if it does not, rebuild the host adapter from the same source revision
before running the benchmark.

### 6.4 Build isolation

The unified build helper writes `compiler_work/` and its output below its own
source directory. To leave the checkout untouched, the script:

1. creates a fresh `RUN_ROOT/overlay-<timestamp>/deep_gemm/include` copy;
2. runs the unmodified helper from that copy;
3. keeps compiler caches, temporary files, `P.co`, and results under
   `RUN_ROOT`.

### 6.5 Run

```bash
export WEAVE_UNIFIED_ROOT=/path/to/weave_unified_cpp
export WEAVE_EP8_LAUNCHER=/abs/path/to/frozen_host_launcher.so
export WEAVE_EP8_IPC_LIB=/abs/path/to/libc406_ipc_driver.so
export WEAVE_EP8_ROUTES=/abs/path/to/routes
bash examples/mi308x_qwen38max_ep8_s1024.sh
```

Instrumented build:

```bash
PROFILE_LEVEL=2 bash examples/mi308x_qwen38max_ep8_s1024.sh
```

`--save-trace` in the benchmark driver writes the harness probe trace, not a
paired SW+HW Perfetto timeline. The annotated preview uses the archived trace.

---

## 7. MI308X EP16: two-node tuned build + run preset

### 7.1 Parameter table

| Axis | Parameter | Value | Evidence | Notes |
|---|---|---:|---|---|
| Spatial | BN / BK | 128 / 128 | current-build-default | fixed tile constants |
| Spatial | `WEAVE_BM_TIER_MODE` | 2 | current-build-default | M tiers 16/32/64/128/256 |
| Temporal | `WEAVE_SCATTER_MODE` | 2 | current-build-default | node-partial pre-reduced scatter |
| Temporal | `WEAVE_OVERLAP_PIPELINE` | 1 | current-build-default | mode-2 overlap pipeline |
| Temporal | `WEAVE_DUALGROUP_MIN_TOKENS` | 64 | documented-behavior | local tokens 128 / 16 = 8 < 64 -> threshold NOT reached |
| Implementation | `WEAVE_ABLATION_PRESET` | 0 | current-build-default | full Weave preset |
| Implementation | `WEAVE_SELECTIVE_DEDUP` | 1 | current-build-default | required by the EP16 path |
| Implementation | `WEAVE_ALL_WAVE_COMBINE` | 1 | current-build-default | all waves join the tail combine |
| Implementation | `WEAVE_BN2_MODE` | 0 | current-build-default | auto second-stage BN |
| Implementation | `WEAVE_S2_NGROUP_MODE` | 0 | current-build-default | auto N-group width |
| Implementation | `WEAVE_PARALLEL_ROUTE_PLAN` | 1 | current-build-default | 80-CTA parallel route plan |
| Implementation | `WEAVE_PROFILER_LEVEL` | 0 | current-build-default | 2 for a profiler build |
| Implementation | `WEAVE_TURBO_MODES` | 2 | current-build-default | runtime mode-2 overlap arm |

### 7.2 Why EP16 differs from EP8

For Qwen3.8-Max EP16 at S=128:

```text
global S        = 128
world size      = 16
local tokens    = 128 / 16 = 8
local experts   = 512 / 16 = 32
```

`8 < 64`, so `WEAVE_DUALGROUP_MIN_TOKENS=64` does **not** trigger the second
compute group. EP16 stays on the 4+4 static overlap path with
`WEAVE_SCATTER_MODE=2` and `WEAVE_OVERLAP_PIPELINE=1`.

This is materially different from EP8 S=1024, where the local token count is
128 and the 8-wave dual-compute path is active.

### 7.3 External artifacts

| Variable | Meaning |
|---|---|
| `WEAVE_EP16_PKG` | package root containing `baseline/ep_common.py`, `baseline/ep_harness.py`, `baseline/ep_reference.py`, `baseline/paper_step0.py`, `ep16/`, and `ep16_turbo_runner.py` |
| `WEAVE_EP16_HOST_LIBS` | directory containing `libep16_turbo_host_mode2.so` and `gda_turbo_mode2.so` |
| `WEAVE_EP16_ROUTE_MANIFEST` | Qwen3.8-Max route manifest |
| `WEAVE_EP16_WEIGHT_CACHE` | Qwen3.8-Max weight cache |
| `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` | two-node rendezvous |
| `WEAVE_EP16_USE_ARCHIVED_CO` | optional: `1` = use the archived `P_ep16_turbo_overlap_mode2.co` instead of the freshly built one |

The public tree contains the kernel and the Python reference/ABI modules, but
not the full host `.so` / GDA `.so` build chain. The script fails fast if any
artifact is missing.

### 7.4 Verified result from the archived two-node run

For Qwen3.8-Max EP16 S=128:

```text
t_local                    = 8
local_experts              = 32
elements checked globally  = 1,048,576
FP64 reference             = on every rank
rank set                   = complete (16 ranks)
rel_l2 max over ranks      = 0.01253   (gate 0.02)
max row rel_l2 over ranks  = 0.01961   (gate 0.05)
status                     = PASS
```

These are archived measurements. The script builds and runs the current
unified path; it does not re-measure them here.

### 7.5 Run

```bash
export WEAVE_UNIFIED_ROOT=/path/to/weave_unified_cpp
export WEAVE_EP16_PKG=/abs/path/to/pkg
export WEAVE_EP16_HOST_LIBS=/abs/path/to/live_arms
export WEAVE_EP16_ROUTE_MANIFEST=/abs/path/route_manifest.json
export WEAVE_EP16_WEIGHT_CACHE=/abs/path/weights.pt
export NODE_RANK=0
export MASTER_ADDR=10.0.0.1
export MASTER_PORT=29500
bash examples/mi308x_qwen38max_ep16_s128.sh
```

Run the same command on the second node with `NODE_RANK=1`.

The default path uses the freshly built `.co`. The archived host `.so` must
match the current kernel ABI. If it does not, set
`WEAVE_EP16_USE_ARCHIVED_CO=1` to use the archived `.co` that is known to
match the archived host `.so`.

---

## 8. Timeline rendering and parameter annotation

`profile_and_render_timeline.py` does three things:

1. prints the Spatial / Temporal / Implementation table with an evidence class
   for every row;
2. optionally launches the preset script (`--run`);
3. optionally invokes the v2 renderer with `--config-banner` and
   `--repro-script`.

Examples:

```bash
# print the table, then render the archived SM103 timeline
python examples/profile_and_render_timeline.py --preset sm103 --render

# print the table and run the preset, then render the archived timeline
python examples/profile_and_render_timeline.py --preset sm103 --run --render

# use an explicit trace
python examples/profile_and_render_timeline.py --preset h20 \
  --trace /abs/path/to/h20_trace.sw_hw.perfetto.json.gz --render

# EP16: print the table only; there is no matched 3-panel preview
python examples/profile_and_render_timeline.py --preset mi308x_ep16 --no-render
```

Trace resolution:

| Preset | Environment variables |
|---|---|
| `sm103` | `WEAVE_SM103_TRACE` or `WEAVE_SM103_RESULT_ROOT` |
| `h20` | `WEAVE_H20_TRACE` or `WEAVE_H20_RESULT_ROOT` |
| `mi308x_ep8` | `WEAVE_MI308X_TRACE` or `WEAVE_MI308X_TRACE_ROOT` |
| `mi308x_ep16` | no 3-panel renderer |

The figure places a four-line parameter pill between the main title and
Panel A: three configuration lines (Spatial / Temporal / Implementation) plus
one `Reproduce:` line. The renderer asserts that the pill does not overlap the
main title or the Panel A title.

The renderer keeps the strict gates of the original figure:

- exactly 3 panels;
- no expanded zoom panel;
- no SMEM-occupancy panel;
- Panel A and Panel C legends outside and above their axes;
- 4 hardware-utilization curves from the same run;
- 7-stage MoE palette with 100% of plotted categories in the legend;
- zero forbidden display strings;
- zero text overlap and zero clipped text;
- fixed 3520 x 2244 px output.

---

## 9. Correctness protocol

All presets use random-normal inputs and full-output checks. No preset claims
correctness from a handful of sampled values.

| Path | Inputs | Check |
|---|---|---|
| SM103 `fp8xfp4` | `torch.randn` activations, weights, routing scores | fused path vs the non-fused baseline path; exact bitwise full-output equality when `num_shared_experts=0`, plus exact cumulative stats |
| SM103 `fp4xfp4` | `torch.randn` activations, weights, routing scores | independent FP32 reference; `calc_diff < 1e-3`, relative L2 `< 2e-2`, full tensor |
| H20 | `torch.randn` activations, weights, routing scores | independent PyTorch reference; full-output `calc_diff` gate (default 0.01) |
| MI308X EP8 | `torch.randn` activations, weights, routing scores | independent FP64/FP32 reference; all elements; `rel_l2 <= 0.02`; max-row hard gate 0.10 (0.05 is a soft diagnostic for official-code arms) |
| MI308X EP16 | `torch.randn` activations, weights, routing scores | independent FP64 reference on every rank; all 1,048,576 elements; `rel_l2 <= 0.02`; `max_row_rel_l2 <= 0.05` |

If a GPU is already busy with another process, stop and wait for it to become
idle. The presets never kill or preempt another user's process.

---

## 10. Known limitations and honest notes

1. **No GPU re-run in this delivery.** The three preview PNGs are rendered from
   archived traces. The preset scripts are provided for reproduction on the
   corresponding GPU hosts; they were not executed during this documentation
   build.
2. **SM103 tile parameters are heuristic-derived.** The archived trace does not
   carry `selected_cfg`. The tile and store granularity come from the
   documented block-shape branch for 20.0 expected tokens per expert.
3. **H20 compact spec keys are not all standalone environment variables.** The
   exact reproduction entry is the full `WEAVE_H20_CONFIG` string. Mapping
   every key to a fabricated standalone variable would be inaccurate.
4. **MI308X EP8 trace metadata has no full compile string.** The preset uses
   the current unified build defaults plus the archived build command. The
   archived timeline came from the archived EP8 source tree; byte-identical
   binary or trace is not claimed.
5. **MI308X EP16 is not a matched preview.** The three annotated previews are
   the EP8 S=1024 set. EP16 is a different workload and a different two-node
   path.
6. **Some hardware-utilization series are volume-based reconstructions.** The
   H20 and MI308X trace metadata record that direct PMU sampling was
   unavailable. Their four series are synchronized reconstructions from the
   same run using the kernel window plus exact per-tile FLOP / HBM / xGMI
   volumes. They are not raw PMU counter dumps. The SM103 trace metadata was
   stripped before archiving; the same caveat applies unless a PMU-backed
   trace is supplied.
7. **The preset run does not automatically produce a paired SW+HW trace.**
   `--render` uses the archived trace resolved by the environment variables or
   `--trace`.

---

## 11. Quick reference

```bash
# 1. Print the table for every preset
for p in sm103 h20 mi308x_ep8 mi308x_ep16; do
  python examples/profile_and_render_timeline.py --preset "$p" --no-render
done

# 2. Re-render all three annotated previews from the archived traces
python examples/profile_and_render_timeline.py --preset sm103 --render
python examples/profile_and_render_timeline.py --preset h20 --render
python examples/profile_and_render_timeline.py --preset mi308x_ep8 --render

# 3. Optional: launch the SM103 preset first, then render
python examples/profile_and_render_timeline.py --preset sm103 --run --render
```
