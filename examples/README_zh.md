# Qwen3.8-Max 可复现 WeaveProfiler 预设

本目录提供三组匹配的 `Qwen3.8-Max, EP8, S=1024` WeaveProfiler 时间线预设，
以及 MI308X 两节点 `EP16, S=128` 路径。所有预设只使用统一的 `WEAVE_*`
名称，不包含任何旧版调参前缀。

文件列表：

```
examples/
  README.md                                  英文说明
  README_zh.md                               本文件
  sm103_qwen38max_ep8_s1024.sh               148-SM SM103，fp8xfp4（可选 fp4xfp4）
  h20_qwen38max_ep8_s1024.sh                 8x H20，精确 tuned 配置串
  mi308x_qwen38max_ep8_s1024.sh              8x MI308X EP8 构建 + 运行
  mi308x_qwen38max_ep16_s128.sh              2 节点 16x MI308X EP16 构建 + 运行
  profile_and_render_timeline.py             三轴参数表打印 + 渲染驱动
```

标注后的预览图位于上一级目录：

```
annotated_timelines/
  weaveprofiler_sm103_timeline.png
  weaveprofiler_h20_timeline.png
  weaveprofiler_mi308x_timeline.png
  render_stats.json
```

---

## 1. 三轴分类

下面三个轴名是为这套示例引入的描述性术语，不是 kernel 内部的枚举值。
每个可调参数只属于一个轴。

| 轴 | 回答的问题 | 典型参数 |
|---|---|---|
| **Spatial（空间）** | 工作如何分到 SM/CU 和 tile 上？ | `WEAVE_BLOCK_M/N/K`、`WEAVE_STORE_BLOCK_M`、`WEAVE_BM_TIER_MODE`、warp 角色 |
| **Temporal（时间）** | 各阶段何时重叠、就绪顺序如何管理？ | `WEAVE_SCATTER_MODE`、`WEAVE_OVERLAP_PIPELINE`、`WEAVE_INTERLEAVE_L1_L2`、`WEAVE_READY_GROUP_SIZE`、`WEAVE_COMBINE_*` |
| **Implementation（实现）** | 选择哪条代码路径或数值模式？ | `WEAVE_PRECISION_MODE`、`WEAVE_DEDUP_*`、`WEAVE_REG_PULL_BYTES`、`WEAVE_SCATTER_SMEM_ALIAS`、`WEAVE_PROFILER_LEVEL` |

这样拆分后，tile 形状的变化不会被藏在调度参数里，抓取专用的开关也不会
被误当成生产调参值。

---

## 2. 证据分级

所有参数表都标注证据等级：

| 等级 | 含义 |
|---|---|
| `trace-metadata` | 归档 trace 中的字面字段，包括 H20 的 `selected_cfg` 和 MI308X 的 grid/wave 元数据。 |
| `archived-script` | 生成该 trace 的归档抓取或构建脚本中的字面值。 |
| `current-build-default` | 当前统一构建脚本中的生效默认值。 |
| `heuristic-derived` | 由有文档的块形状启发式推导出的值，不是 trace 的字面字段。 |
| `capture-only` | 只在抓取时间线时设置，不是生产调参值。 |
| `documented-behavior` | 保留值或自动值的文档语义。 |

只有 trace 中真实存在的字段才会标成 `trace-metadata`。

---

## 3. 预设矩阵

| 预设 | 硬件 | 并行 | 负载 | 脚本 | 预览图 |
|---|---|---|---|---|---|
| `sm103` | NVIDIA SM103，148 SM | 8 rank，EP8 | Qwen3.8-Max，S=1024 | `sm103_qwen38max_ep8_s1024.sh` | `weaveprofiler_sm103_timeline.png` |
| `h20` | NVIDIA H20，78 SM | 8 rank，EP8 | Qwen3.8-Max，S=1024 | `h20_qwen38max_ep8_s1024.sh` | `weaveprofiler_h20_timeline.png` |
| `mi308x_ep8` | AMD MI308X，80 CU | 8 rank，EP8 | Qwen3.8-Max，S=1024 | `mi308x_qwen38max_ep8_s1024.sh` | `weaveprofiler_mi308x_timeline.png` |
| `mi308x_ep16` | AMD MI308X，80 CU | 16 rank，EP16，2 节点 | Qwen3.8-Max，S=128 | `mi308x_qwen38max_ep16_s128.sh` | 无匹配三面板预览 |

三张预览图使用完全相同的 `Qwen3.8-Max, EP8, S=1024` 负载和三面板布局，
可以直接横向比较。

---

## 4. SM103（148-SM）：tuned `fp8xfp4` 预设

### 4.1 参数表

| 轴 | 参数 | 值 | 证据 | 说明 |
|---|---|---:|---|---|
| Spatial | `WEAVE_BLOCK_M` | 64 | heuristic-derived | 每专家期望 token 数 = 1024 * 8 * 10 / 512 = 20.0，落在 <= 32.5 分支 |
| Spatial | `WEAVE_BLOCK_N` | 128 | heuristic-derived | 同一块形状分支的固定 BN |
| Spatial | `WEAVE_BLOCK_K` | 128 | heuristic-derived | fp8 激活元素宽度下的 K tile |
| Spatial | `WEAVE_STORE_BLOCK_M` | 32 | heuristic-derived | TMA store 的 M 粒度 |
| Spatial | `WEAVE_ENABLE_MULTICAST` | 1 | current-build-default | 148-SM 数据流要求 |
| Temporal | `WEAVE_SCATTER_MODE` | 1 | archived-script | dispatch warp 执行 scatter，通信分区异步排空 ready queue |
| Temporal | `WEAVE_READY_GROUP_SIZE` | 2 | archived-script | ready group 粒度 |
| Temporal | `WEAVE_COMBINE_GROUP_TILES` | 2 | archived-script | combine tile 分组 |
| Temporal | `WEAVE_COMBINE_REDUCE_WARPS` | 16 | archived-script | 16 个 reduce warp，每个 128 个 tile |
| Temporal | `WEAVE_SHARED_L2_EARLY_BLOCKS` | 77 | capture-only | 生产默认值是 10 |
| Implementation | `WEAVE_PRECISION_MODE` | 0 | archived-script | fp8xfp4；可选 1 = fp4xfp4 |
| Implementation | `WEAVE_DEDUP_MODE` | 1 | archived-script | 唯一 (token, destination) 桶 |
| Implementation | `WEAVE_EXPERTS_PER_WAVE` | 2 | archived-script | 每 wave 两个 expert |
| Implementation | `WEAVE_REG_PULL_BYTES` | 4096 | archived-script | 4 KiB 寄存器 pull |
| Implementation | `WEAVE_SCATTER_SMEM_ALIAS` | 1 | archived-script | scatter 暂存区与 epilogue buffer 别名 |
| Implementation | `WEAVE_SCHED_MODE` | 3 | documented-behavior | 保留值，当前与 mode 0 相同 |
| Implementation | `WEAVE_ABLATION_PRESET` | 0 | archived-script | 完整 Weave 预设 |

### 4.2 证据边界

- 归档 trace 没有 `selected_cfg` 字段。标量参数来自归档抓取脚本。
- tile `64 x 128 x 128` 加 `store_m = 32` 不是 trace 的字面字段，而是
  每专家期望 token 数为 20.0 时文档化块形状分支的结果。归档运行启用了该
  tuned 块形状启发式；在统一代码里，显式的 `WEAVE_BLOCK_*` /
  `WEAVE_STORE_BLOCK_M` 覆盖会选择同一分支，不需要额外总开关。
- `WEAVE_SCHED_MODE=3` 是保留值，当前行为与 mode 0 相同。保留它只是为了
  忠实复现归档运行。
- `WEAVE_SHARED_L2_EARLY_BLOCKS=77` 只用于抓取时间线。干净延迟测量使用
  默认值 10。设置 `WEAVE_CAPTURE=1` 可以复现抓取时的值。
- 归档的干净测量记录：中位数 359.55 us，8/8 rank 的 SHA-256 完全一致，
  16 个 reduce warp x 128 个事件覆盖 tile 0..2047，speedup 1.0168，
  最小 cosine 约 0.99999988。这些是归档数字，本次没有重新测量。
- 当前统一 benchmark 使用随机路由；归档 trace 使用抓取的路由表。脚本
  复现的是调参预设，不承诺逐字节相同的时间线。

### 4.3 运行

```bash
export WEAVE_UNIFIED_ROOT=/path/to/weave_unified_cpp
bash examples/sm103_qwen38max_ep8_s1024.sh
```

可选 `fp4xfp4`：

```bash
PRECISION=fp4xfp4 bash examples/sm103_qwen38max_ep8_s1024.sh
```

`fp4xfp4` 是可选的精度模式，不是单独重新调参的预设。归档 tile 是针对
`fp8xfp4` 调的。

---

## 5. H20（SM90）：精确 tuned 预设

### 5.1 精确的 ground-truth 串

归档 trace 的 `otherData.meta.selected_cfg` 字面值为：

```text
fast_host=1,fused=1,interleave=5,epw=8,hot_cold=2,compact=1,dyn=1,alias=1,scat_disp=1,d_mask_mode=1,bm=32,bn=256,bk=128,wgs=4,swap=1,stages=0,swap_pipe=1,fast_swap_epi=1,rank_cache=1,fast_clean=1,fast_pull=1,dedup_disp=1,dedup_tier=1,reg_comb=1,fast_comb=1,all_idle_comb=1
```

它通过统一公开入口 `WEAVE_H20_CONFIG` 传入。字符串内部的 key 是 H20 的
spec 字段，不是旧版环境变量名。

### 5.2 每个生效 key 的映射

| 轴 | 统一名称 / 字段 | 值 | 证据 | 说明 |
|---|---|---:|---|---|
| Spatial | `WEAVE_H20_CONFIG.bm` | 32 | trace-metadata | spec 字段；没有独立的公开环境变量绑定 |
| Spatial | `WEAVE_H20_CONFIG.bn` | 256 | trace-metadata | spec 字段；没有独立绑定 |
| Spatial | `WEAVE_BLOCK_K` | 128 | trace-metadata | `bk=128` |
| Spatial | `WEAVE_H20_CONFIG.wgs` | 4 | trace-metadata | 4 个 epilogue warp group |
| Spatial | `WEAVE_H20_CONFIG.swap` | 1 | trace-metadata | swap-AB 布局 |
| Temporal | `WEAVE_INTERLEAVE_L1_L2` | 5 | trace-metadata | `interleave=5`：EPW 交替 + L2 K-block 依赖 |
| Temporal | `WEAVE_HOT_COLD_MODE` | 2 | trace-metadata | `hot_cold=2`：LPT + 冷热配对 |
| Temporal | `WEAVE_DYN_SCHED` | 1 | trace-metadata | `dyn=1`：动态 expert 游标 |
| Temporal | `WEAVE_H20_CONFIG.stages` | 0 | documented-behavior | 运行时自动选择流水线深度；不是零 stage 流水线 |
| Temporal | `WEAVE_SWAP_PIPE` | 1 | trace-metadata | `swap_pipe=1` |
| Implementation | `WEAVE_H20_CONFIG.epw` | 8 | trace-metadata | 每 wave 8 个 expert；仅 spec 字段 |
| Implementation | `WEAVE_SCATTER_ROLE_DISP` | 1 | trace-metadata | `scat_disp=1` |
| Implementation | `WEAVE_HETERO_MASK_MODE` | 1 | trace-metadata | `d_mask_mode=1` |
| Implementation | `WEAVE_DEDUP_DISPATCH` | 1 | trace-metadata | `dedup_disp=1` |
| Implementation | `WEAVE_DEDUP_SCATTER` | 0 | trace-metadata | 没有 `dedup_scat`，scatter dedup 关闭 |
| Implementation | `WEAVE_REG_COMBINE` | 1 | trace-metadata | `reg_comb=1` |
| Implementation | `WEAVE_ALL_IDLE_COMBINE` | 1 | trace-metadata | `all_idle_comb=1` |
| Implementation | `WEAVE_FAST_PULL` | 1 | trace-metadata | `fast_pull=1` |
| Implementation | `WEAVE_H20_CONFIG.dedup_tier` | 1 | trace-metadata | spec 字段；没有独立绑定 |
| Implementation | `WEAVE_H20_CONFIG.*` | 精确串 | trace-metadata | `fast_host`、`fused`、`compact`、`alias`、`fast_swap_epi`、`rank_cache`、`fast_clean`、`fast_comb` |

关键细节：

- trace 有 `dedup_disp=1` 和 `dedup_tier=1`，但**没有** `dedup_scat=1`。
  scatter dedup 是关闭的。
- `stages=0` 表示运行时自动选择流水线深度，不表示零 stage，也不是
  `WEAVE_GEMM_STAGES=0`。
- `bm`、`bn`、`wgs`、`swap`、`epw`、`fast_host`、`fused`、`compact`、
  `alias`、`fast_swap_epi`、`rank_cache`、`fast_clean`、`dedup_tier`、
  `fast_comb` 没有独立的公开 `WEAVE_*` 环境变量绑定。精确复现入口只有完整
  的 `WEAVE_H20_CONFIG` 串。

### 5.3 运行

```bash
export WEAVE_UNIFIED_ROOT=/path/to/weave_unified_cpp
bash examples/h20_qwen38max_ep8_s1024.sh
```

脚本在导出精确 `WEAVE_H20_CONFIG` 的前提下，运行统一 SM90 正确性套件中的
一个生产形状。统一套件没有 Qwen3.8-Max S=1024 的精确 cell；精确时间线抓取
是归档的。要回放该时间线，使用
`profile_and_render_timeline.py --preset h20 --trace` 或设置
`WEAVE_H20_TRACE`。

---

## 6. MI308X EP8：tuned 构建 + 运行预设

### 6.1 参数表

| 轴 | 参数 / 字段 | 值 | 证据 | 说明 |
|---|---|---:|---|---|
| Spatial | gridX / waves_per_cu | 80 / 8 | trace-metadata | 80 个唯一物理 CU |
| Spatial | BN / BK | 128 / 128 | current-build-default | 固定 tile 常量 |
| Spatial | `WEAVE_BM_TIER_MODE` | 2 | current-build-default | M tier 16/32/64/128/256 |
| Spatial | `WEAVE_TH_BM64` | 16 | current-build-default | 小 M tier 阈值 |
| Spatial | `WEAVE_TH_BM128` | 96 | current-build-default | 大 M tier 阈值 |
| Temporal | `WEAVE_SCATTER_MODE` | 2 | current-build-default | local-dedup / node-partial 预归约 |
| Temporal | `WEAVE_OVERLAP_PIPELINE` | 1 | current-build-default | 计算通信重叠 |
| Temporal | `WEAVE_DUALGROUP_MIN_TOKENS` | 64 | current-build-default | 本地 token = 1024 / 8 = 128 >= 64，8-wave 双计算生效 |
| Implementation | `WEAVE_ABLATION_PRESET` | 0 | current-build-default | 完整 Weave 预设 |
| Implementation | `WEAVE_SELECTIVE_DEDUP` | 1 | current-build-default | EP8 路径要求 |
| Implementation | `WEAVE_ALL_WAVE_COMBINE` | 1 | current-build-default | 所有 wave 参与尾部 combine |
| Implementation | `WEAVE_BN2_MODE` | 0 | current-build-default | 第二阶段 BN 自动 |
| Implementation | `WEAVE_S2_NGROUP_MODE` | 0 | current-build-default | N-group 宽度自动 |
| Implementation | `WEAVE_PARALLEL_ROUTE_PLAN` | 1 | current-build-default | 80-CTA 并行 route plan |
| Implementation | `WEAVE_PROFILER_LEVEL` | 2 | capture-only | 生产为 0 |

当前构建辅助脚本中的固定宏：`WEAVE_PAR_RP=1`、
`WEAVE_CK_G2_LDS_ONLY=0`、`WEAVE_TILE_TRACE_ONLY=1`、
`WEAVE_ADAPTIVE_POST_TIMEOUT=1`、`WEAVE_S16_PLAN_CAP=0`、
`WEAVE_S16_STATIC_DISPATCH=0`、`WEAVE_OPT_RELAXED_SPIN=1`、
`WEAVE_OPT_LOCAL_SCOPE=1`。

### 6.2 证据边界

- 归档 trace 元数据记录 `gridX=80`、`waves_per_cu=8`、
  `unique_physical_cus=80`、`S=1024`、`rank=2`。它不包含完整的编译期
  参数串。
- 归档 EP8 时间线由归档 EP8 源码树生成。本脚本用相同的 tuned 预设重建
  **当前统一 EP8 kernel**，不承诺二进制或时间线逐字节相同。
- S=1024 时本地 token 数为 128，达到 `WEAVE_DUALGROUP_MIN_TOKENS=64`，
  因此 8-wave 双计算（4+4）路径生效。
- `WEAVE_PROFILER_LEVEL=2` 只用于抓取；生产使用 0。

### 6.3 外部产物

benchmark 驱动不是自包含的，需要三个未随公开仓库分发的归档宿主/运行时
产物：

| 变量 | 含义 |
|---|---|
| `WEAVE_EP8_LAUNCHER` | frozen host launcher 共享库 |
| `WEAVE_EP8_IPC_LIB` | IPC driver 共享库 |
| `WEAVE_EP8_ROUTES` | 包含 Qwen3.8-Max 路由表的目录 |

脚本会检查三项，缺失时立即报清晰错误。Python 参考实现、ABI 和 IPC
封装模块已经在源码树
`deep_gemm/include/deep_gemm/impls/amd/ep8/support/` 中。归档 host launcher
还必须与新构建 `P.co` 的 ABI 匹配；如果不匹配，请用同一源码版本重建 host
adapter 后再运行 benchmark。

### 6.4 构建隔离

统一构建辅助脚本会把 `compiler_work/` 和输出写到它自己所在的源码目录。
为了不修改 checkout，脚本会：

1. 新建 `RUN_ROOT/overlay-<timestamp>/deep_gemm/include` 拷贝；
2. 从该拷贝运行未修改的构建辅助脚本；
3. 把编译缓存、临时文件、`P.co` 和结果都放在 `RUN_ROOT` 下。

### 6.5 运行

```bash
export WEAVE_UNIFIED_ROOT=/path/to/weave_unified_cpp
export WEAVE_EP8_LAUNCHER=/abs/path/to/frozen_host_launcher.so
export WEAVE_EP8_IPC_LIB=/abs/path/to/libc406_ipc_driver.so
export WEAVE_EP8_ROUTES=/abs/path/to/routes
bash examples/mi308x_qwen38max_ep8_s1024.sh
```

带插桩构建：

```bash
PROFILE_LEVEL=2 bash examples/mi308x_qwen38max_ep8_s1024.sh
```

benchmark 驱动的 `--save-trace` 写的是 harness probe trace，不是配对的
SW+HW Perfetto 时间线。标注预览使用归档 trace。

---

## 7. MI308X EP16：两节点 tuned 构建 + 运行预设

### 7.1 参数表

| 轴 | 参数 | 值 | 证据 | 说明 |
|---|---|---:|---|---|
| Spatial | BN / BK | 128 / 128 | current-build-default | 固定 tile 常量 |
| Spatial | `WEAVE_BM_TIER_MODE` | 2 | current-build-default | M tier 16/32/64/128/256 |
| Spatial | `WEAVE_TH_BM64` | 16 | current-build-default | 小 M tier 阈值 |
| Spatial | `WEAVE_TH_BM128` | 96 | current-build-default | 大 M tier 阈值 |
| Temporal | `WEAVE_SCATTER_MODE` | 2 | current-build-default | node-partial 预归约 scatter |
| Temporal | `WEAVE_OVERLAP_PIPELINE` | 1 | current-build-default | mode-2 overlap 流水线 |
| Temporal | `WEAVE_DUALGROUP_MIN_TOKENS` | 64 | documented-behavior | 本地 token = 128 / 16 = 8 < 64，阈值未触发 |
| Implementation | `WEAVE_ABLATION_PRESET` | 0 | current-build-default | 完整 Weave 预设 |
| Implementation | `WEAVE_SELECTIVE_DEDUP` | 1 | current-build-default | EP16 路径要求 |
| Implementation | `WEAVE_ALL_WAVE_COMBINE` | 1 | current-build-default | 所有 wave 参与尾部 combine |
| Implementation | `WEAVE_BN2_MODE` | 0 | current-build-default | 第二阶段 BN 自动 |
| Implementation | `WEAVE_S2_NGROUP_MODE` | 0 | current-build-default | N-group 宽度自动 |
| Implementation | `WEAVE_PARALLEL_ROUTE_PLAN` | 1 | current-build-default | 80-CTA 并行 route plan |
| Implementation | `WEAVE_PROFILER_LEVEL` | 0 | current-build-default | 插桩构建用 2 |
| Implementation | `WEAVE_TURBO_MODES` | 2 | current-build-default | 运行时 mode-2 overlap 臂 |

### 7.2 EP16 与 EP8 的差异

Qwen3.8-Max EP16、S=128：

```text
global S        = 128
world size      = 16
local tokens    = 128 / 16 = 8
local experts   = 512 / 16 = 32
```

`8 < 64`，所以 `WEAVE_DUALGROUP_MIN_TOKENS=64` **不会**触发第二计算组。
EP16 保持 4+4 静态 overlap 路径，配合 `WEAVE_SCATTER_MODE=2` 和
`WEAVE_OVERLAP_PIPELINE=1`。

这与 EP8 S=1024 有本质区别：后者本地 token 数为 128，8-wave 双计算路径
生效。

### 7.3 外部产物

| 变量 | 含义 |
|---|---|
| `WEAVE_EP16_PKG` | package 根目录，含 `baseline/ep_common.py`、`baseline/ep_harness.py`、`baseline/ep_reference.py`、`baseline/paper_step0.py`、`ep16/`、`ep16_turbo_runner.py` |
| `WEAVE_EP16_HOST_LIBS` | 含 `libep16_turbo_host_mode2.so` 和 `gda_turbo_mode2.so` 的目录 |
| `WEAVE_EP16_ROUTE_MANIFEST` | Qwen3.8-Max 路由 manifest |
| `WEAVE_EP16_WEIGHT_CACHE` | Qwen3.8-Max 权重缓存 |
| `NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT` | 两节点 rendezvous |
| `WEAVE_EP16_USE_ARCHIVED_CO` | 可选：设为 `1` 时使用归档的 `P_ep16_turbo_overlap_mode2.co`，而不是新构建的 `.co` |

公开源码树包含 kernel 和 Python 参考/ABI 模块，但不包含完整的 host `.so` /
GDA `.so` 构建链。缺失任一产物时脚本立即失败。

### 7.4 归档两节点运行的验证结果

Qwen3.8-Max EP16 S=128：

```text
t_local                    = 8
local_experts              = 32
全局检查元素数             = 1,048,576
FP64 参考                  = 每个 rank 独立
rank 集合                  = 完整（16 rank）
rel_l2 跨 rank 最大值      = 0.01253   (门槛 0.02)
max row rel_l2 最大值      = 0.01961   (门槛 0.05)
状态                       = PASS
```

这些是归档测量值。脚本构建并运行当前统一路径，本次没有重新测量。

### 7.5 运行

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

第二个节点使用相同命令，仅把 `NODE_RANK` 改为 1。

默认使用新构建的 `.co`。归档 host `.so` 必须与当前 kernel ABI 匹配。如果不
匹配，设置 `WEAVE_EP16_USE_ARCHIVED_CO=1`，改用与归档 host `.so` 配套的
归档 `.co`。

---

## 8. 时间线渲染与参数标注

`profile_and_render_timeline.py` 做三件事：

1. 打印 Spatial / Temporal / Implementation 三轴表，并为每一行标注证据等级；
2. 可选启动预设脚本（`--run`）；
3. 可选调用 v2 渲染器，并传入 `--config-banner` 与 `--repro-script`。

示例：

```bash
# 打印参数表，然后渲染归档 SM103 时间线
python examples/profile_and_render_timeline.py --preset sm103 --render

# 先运行预设，再渲染归档时间线
python examples/profile_and_render_timeline.py --preset sm103 --run --render

# 显式指定 trace
python examples/profile_and_render_timeline.py --preset h20 \
  --trace /abs/path/to/h20_trace.sw_hw.perfetto.json.gz --render

# EP16：只打印参数表，没有匹配的三面板预览
python examples/profile_and_render_timeline.py --preset mi308x_ep16 --no-render
```

trace 解析：

| 预设 | 环境变量 |
|---|---|
| `sm103` | `WEAVE_SM103_TRACE` 或 `WEAVE_SM103_RESULT_ROOT` |
| `h20` | `WEAVE_H20_TRACE` 或 `WEAVE_H20_RESULT_ROOT` |
| `mi308x_ep8` | `WEAVE_MI308X_TRACE` 或 `WEAVE_MI308X_TRACE_ROOT` |
| `mi308x_ep16` | 无三面板渲染器 |

图在主标题和 Panel A 之间放置四行参数条：三行配置（Spatial / Temporal /
Implementation）加一行 `Reproduce:`。渲染器会断言参数条不与主标题或
Panel A 标题重叠。

渲染器保留原图的严格门禁：

- 恰好 3 个 panel；
- 没有 expanded zoom；
- 没有 SMEM 占用 panel；
- Panel A 和 Panel C 图例位于坐标轴外上方；
- 同一次运行的 4 条硬件利用率曲线；
- 7 阶段 MoE 调色板，所有绘制类别都在图例中；
- 零禁止显示字符串；
- 零文本重叠、零越界文本；
- 固定 3520 x 2244 像素输出。

---

## 9. 正确性协议

所有预设都使用随机正态输入和完整输出检查，不用少量采样值宣称正确性。

| 路径 | 输入 | 检查 |
|---|---|---|
| SM103 `fp8xfp4` | `torch.randn` 激活、权重、路由分数 | fused 路径对比非 fused baseline 路径；`num_shared_experts=0` 时完整输出逐位相等，并检查完整累积统计 |
| SM103 `fp4xfp4` | `torch.randn` 激活、权重、路由分数 | 独立 FP32 参考；`calc_diff < 1e-3`、relative L2 `< 2e-2`、完整 tensor |
| H20 | `torch.randn` 激活、权重、路由分数 | 独立 PyTorch 参考；完整输出 `calc_diff` 门槛（默认 0.01） |
| MI308X EP8 | `torch.randn` 激活、权重、路由分数 | 独立 FP64/FP32 参考；全元素；`rel_l2 <= 0.02`；max-row 硬门槛 0.10（0.05 是官方代码臂的软诊断） |
| MI308X EP16 | `torch.randn` 激活、权重、路由分数 | 每个 rank 独立 FP64 参考；全部 1,048,576 元素；`rel_l2 <= 0.02`；`max_row_rel_l2 <= 0.05` |

如果 GPU 已被其他进程占用，暂停并等待空闲。预设脚本不会 kill 或抢占
其他用户的进程。

---

## 10. 已知限制与诚实说明

1. **本次没有重新跑 GPU。** 三张预览 PNG 由归档 trace 渲染。预设脚本用于
   在对应 GPU 主机上复现，本次文档构建没有执行它们。
2. **SM103 tile 参数是 heuristic-derived。** 归档 trace 没有 `selected_cfg`。
   tile 和 store 粒度来自每专家 20.0 个期望 token 的文档化块形状分支。
3. **H20 的 compact spec key 并非都有独立环境变量。** 精确复现入口是完整
   `WEAVE_H20_CONFIG` 串。为每个 key 编造一个独立变量是不准确的。
4. **MI308X EP8 trace 元数据没有完整编译串。** 预设使用当前统一构建默认值
   加归档构建命令。归档时间线来自归档 EP8 源码树，不承诺二进制或时间线
   逐字节相同。
5. **MI308X EP16 不是匹配预览。** 三张标注预览是 EP8 S=1024 集合。EP16 是
   不同负载和不同两节点路径。
6. **部分硬件利用率曲线是体积重建值。** H20 和 MI308X 的 trace 元数据记录
   直接 PMU 采样不可用；四条曲线是同一次运行中基于 kernel window 和逐 tile
   FLOP / HBM / xGMI 体积的同步重建，不是原始 PMU 计数器转储。SM103 归档
   前元数据被剥离；除非提供 PMU 支撑的 trace，同样按此理解。
7. **预设运行不会自动生成配对 SW+HW trace。** `--render` 使用环境变量或
   `--trace` 指定的归档 trace。

---

## 11. 快速参考

```bash
# 1. 打印四个预设的三轴表
for p in sm103 h20 mi308x_ep8 mi308x_ep16; do
  python examples/profile_and_render_timeline.py --preset "$p" --no-render
done

# 2. 从归档 trace 重新渲染三张标注预览
python examples/profile_and_render_timeline.py --preset sm103 --render
python examples/profile_and_render_timeline.py --preset h20 --render
python examples/profile_and_render_timeline.py --preset mi308x_ep8 --render

# 3. 可选：先运行 SM103 预设，再渲染
python examples/profile_and_render_timeline.py --preset sm103 --run --render
```