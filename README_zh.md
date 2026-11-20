[English](README.md) | [简体中文](README_zh.md)

# Weave：面向 MoE Megakernel 的路由自适应运行时动态调度系统

Weave 是一个面向 MoE 推理的**路由自适应运行时动态 Megakernel**（C++/CUDA/HIP
统一实现，论文：[Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel
for Compute-Communication Overlap](https://arxiv.org/abs/2609.21483)），支持
**NVIDIA SM103 (148-SM)**、**NVIDIA H20** 与 **AMD MI308X**。内核在每次路由
（dispatch）完成后直接读取各专家的实时 token 分布，在 GPU 上实时完成**空间切分与
时间调度**。

## 架构与运行时调度

| 硬件 | 语言 | 支持的精度取值 | 入口 |
|---|---|---|---|
| NVIDIA SM103 (148-SM) | CUDA C++ | `0 = fp8xfp4`（默认）、`1 = fp4xfp4`（NVFP4）、`3 = bf16`；不支持 `2 = fp8` | sm100_fp8_fp4_mega_moe.cuh / sm100_fp4_fp4_mega_moe.cuh，通过 weave_mega_moe(..., mma_type=...) 分派 |
| NVIDIA H20（SM90） | CUDA C++ | 仅 `2 = fp8`（fp8xfp8） | sm90_fp8_mega_moe.cuh |
| AMD MI308X（gfx942） | HIP C++ | 仅 `2 = fp8`（fp8xfp8）；支持 EP8 与 EP16 | gfx942_fp8_mega_moe.cuh 与 amd/megamoe_amd.hip |

![Weave 系统设计总览](docs/figures/weave_design_overview.png)

![Weave 空间通算配比与时间角色调度对比 MegaMoE](docs/figures/weave_design_rolegraph.png)

每次路由完成后，CTA 0 根据当前卡收到的专家 token 分布与跨卡收发量，用片上代价模型
即时选出本层的**空间切分**与**时间调度**方案，写入全局显存后其余 CTA 直接读取执行：

- **空间切分（Spatial）**：决定每个 SM/CU 内通信 warp 与计算/epilogue warp 的配比，
  并按每个专家实际收到的 token 数动态选择 tile 大小，减少稀疏专家的 padding 浪费。
- **时间调度（Temporal，消除首、中、尾三段气泡）**：
  - **首部（Head）**：层刚启动、尚无 GEMM tile 到齐时，原本等待的计算/TMA/epilogue
    warp 按各自寄存器上限一起参与远端 token 拉取，先把跨卡带宽打满，首批 tile 到齐
    后立即切回计算；
  - **中部（Middle）**：按当次通信与计算耗时比，决定 FC2 输出的 Scatter 由
    epilogue warp 就地发出，还是交给通信 warp 与后续 GEMM tile 并行发出；
  - **尾部（Tail）**：去掉传统跨全卡的全局同步等待，改为**逐 token 就绪即归约**
    （单个 token 收齐全部 `K` 个专家输出后立即可做加权求和），并让已完成前序任务的
    **所有空闲 warp（通信、TMA/MMA、epilogue）一起参与**尾部归约。

## 硬件分区与全局就绪队列

每个 SM/CU 内的 warp 划分为**通信分区（`N_c`）**与**计算/GEMM 分区（`N_e`）**，
两个分区各自维护独立的全局任务队列：

- **GEMM Ready Queue（计算分区 `N_e` 消费）**：存放待计算的 `(expert_id, m_tile_idx)`
  tile 任务。远端 token 陆续写入本卡缓冲时，通信 warp 只对相应专家（或专家内分组）的
  到达计数器做原子累加；无需在运行时额外构造并排序整张 tile 列表，计算 CTA 直接按
  预设专家顺序（如按负载降序或 L2 tile-swizzle）和专家前缀和定位已到齐的 tile，或在
  `WEAVE_READY_QUEUE=1` 时从全局 FIFO 队列弹出刚到齐的 tile。若
  `WEAVE_SCATTER_MODE=0`，epilogue warp 算完 FC2 后直接在计算分区内把输出发往目标卡。
- **Communication Ready Queue（通信分区 `N_c` 消费）**：存放跨卡拉取（Pull）与异步
  发送（Scatter）任务。当 `WEAVE_SCATTER_MODE=1` 时，epilogue warp 算完 FC2 tile 后
  把发送任务交接入该队列；当 `WEAVE_SCATTER_MODE=2` 时，先把落在本卡（或本节点）多个
  专家上的同一源 token 部分和在 HBM 中合并成一份，再入队单次发送任务，由 `N_c` 在
  后续 GEMM 计算期间异步发走。

## 快速开始

    git clone https://github.com/ziyuhuang123/weave.git
    cd weave
    python -m pip install -e .
    python -m py_compile baselines/*.py ablation/*.py
    python tests/verify_tunable_params_doc.py

Python 入口示意：

    import torch
    from deep_gemm.mega import SymmBuffer, weave_mega_moe

    # 先为进程组和模型形状构造对称缓冲区，然后调用：
    # weave_mega_moe(y, l1_weights, l2_weights, sym_buffer,
    #                mma_type="fp4xfp4")
    # NVIDIA SM103 (148-SM) 支持 fp8xfp4、fp4xfp4 与 bf16；
    # NVIDIA H20 与 AMD MI308X 使用 fp8。

也可以通过环境变量指定精度：

    export WEAVE_PRECISION_MODE=fp4xfp4

## 实测性能

![6 模型 x 5 batch size 的三硬件端到端加速比](docs/figures/weave_eval_3hw_speedup.png)

论文 Fig. 7：6 模型 x 5 batch size 的三硬件 MoE 层实测加速比。

![6 模型 x 5 batch size x 3 硬件平台的绑定硬件资源利用率 %](docs/figures/weave_bound_utilization_3hw.png)

各模型与 batch 下的瓶颈硬件资源利用率（分别以当前受限的 HBM 带宽、跨卡通信带宽或
算力峰值为分母）。对于 `Qwen3.5-35B`（`H=2048, I=512`）与 `Qwen3.5-122B` 等较小
模型，在 8 卡 EP8 + 小 batch（`S=16..1024`）下属于典型的 **8 卡固定延迟下限主导**
（例如 148-SM SM103 上 `Qwen3.5-35B` 全量理论 HBM/跨卡耗时仅 `2-10 us`，远低于
8 卡跨卡同步与常驻 kernel 启动的 `~35 us` 物理下限）；当 batch 增至 `S=16K` 摊薄
固定开销后，SM103 利用率回升至 `79.4%`，H20 回升至 `60.8%`。在适配 8 卡的大模型
（`Qwen3.8-2.4T`、`MiniMax-M3`）上，Weave 在 148-SM SM103 最高达 `97.6%`、H20
最高达 `85.5%`；其中 H20 在 `Qwen3.8-2.4T` `S=1024` 处受 HBM3 带宽受限
（`T_hbm=755 us` 对比 `T_comp=437 us`），物理上将 FP8 Tensor Core 活跃度封顶在
`58%` 以下。

## WeaveProfiler 软硬件时间线

![NVIDIA SM103 (148-SM) WeaveProfiler SW+HW 时间线](docs/figures/weaveprofiler_sm103_timeline.png)

![NVIDIA H20 WeaveProfiler SW+HW 时间线](docs/figures/weaveprofiler_h20_timeline.png)

![AMD MI308X WeaveProfiler SW+HW 时间线](docs/figures/weaveprofiler_mi308x_timeline.png)

三张图均为 `Qwen3.8-Max, EP8, S=1024` 负载下的软硬件对齐时间线（顶部硬件利用率曲
线、中部全卡物理 SM/CU 泳道、底部代表性 CTA 内各角色流水线）。完整调优
`WEAVE_*` 参数（按 **空间维度 / 时间维度 / 工程实现** 分类）及一键复现脚本位于
[examples/](examples/README_zh.md)：

- NVIDIA SM103 (148-SM)：[examples/sm103_qwen38max_ep8_s1024.sh](examples/sm103_qwen38max_ep8_s1024.sh)
- NVIDIA H20：[examples/h20_qwen38max_ep8_s1024.sh](examples/h20_qwen38max_ep8_s1024.sh)
- AMD MI308X：[examples/mi308x_qwen38max_ep8_s1024.sh](examples/mi308x_qwen38max_ep8_s1024.sh)（EP8）与 [examples/mi308x_qwen38max_ep16_s128.sh](examples/mi308x_qwen38max_ep16_s128.sh)（2 节点 EP16）
- 一键运行与重绘时间线：`python examples/profile_and_render_timeline.py --preset sm103`（支持 `sm103`、`h20`、`mi308x_ep8`、`mi308x_ep16`，并支持 `--run` / `--render`）

## 参数分类：空间维度、时间维度与工程实现

Weave 的调度与内核参数按职责分为三类：

- **空间维度（Spatial）**：控制计算、通信与 tile 资源在 SM/CU、
  warp、cluster 及数值精度上的空间切分，包括通信/计算 warp 比例、异构 SM 角色
  掩码、双 CTA 多播、精度模式（`fp8xfp4` / `fp4xfp4` / `fp8` / `bf16`），以及
  依据实时逐专家 token 数的**运行时逐专家 tile 自适应**（MI308X 五档动态规划
  `BM_e in {16, 32, 64, 128, 256}`，EP16 下限 32；H20 八档 swapAB `{8..64}`
  由 padding 比例门控；148-SM SM103 动态 TMA-A `m8/m16/BM`、动态 epilogue
  `store_m8/m16` 及逐专家 tile 阶梯）。
- **时间维度（Temporal）**：控制执行顺序、通算重叠与阶段衔接，
  包括首部全 warp 预拉取 token 数、Scatter 执行归属（Epilogue 内联 / 通信队列
  异步 / 本地预归约去重后异步）、就绪队列与 L2 tile-swizzle 顺序、L1/L2 交错、
  尾部逐 token 就绪与全 warp 协作归约，以及**运行时自适应流水深度**（GEMM 流水
  级数 `num_stages` 按当前 tile 形状与剩余 SMEM 预算在启动时动态推导，而非编译
  期写死）。
- **工程实现与已有工作（Implementation / Prior Art）**：来自已有 GPU/MoE 系统
  工作的通用实现技术与实验开关，包括寄存器直拉/直归约、选择性去重、RDMA 描述符
  分片、SMEM 复用、Profiler 插桩级别与消融预设。

运行时 CTA 0 用轻量片上代价模型对候选调度排序：共享固定开销在候选比较中相互
抵消，在 48 个实测 148-SM SM103 测试点上与穷举 Oracle 匹配 38/48，取得
1.301x 几何平均加速（Oracle 为 1.332x，选择损失仅 2.4%；单点绝对延迟平均误差
34.0%）。

下表每个 `file:line` 引用均由 `tests/verify_tunable_params_doc.py` 自动校验。
完整参数文档见 [docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md)。

## 统一核心 WEAVE_* 参数

本参数面严格按 **空间 / 时间 / 实现** 三轴组织。每行同时给出 host 解析位置与 kernel 消费位置；`tests/verify_tunable_params_doc.py` 会逐条解析 `file:line` 引用。完整取值语义见 [docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md)。

### 空间（Spatial）

| 参数 | 分类 | 取值（每个接受值均已命名） | 默认值 | 机制 | Host 解析 | Kernel 消费 |
|---|---|---|---|---|---|---|
| `WEAVE_PRECISION_MODE` | 空间 | `0 = fp8xfp4`; `1 = fp4xfp4`; `2 = fp8xfp8`; `3 = bf16` | 0 | Numeric precision and tensor-core mode; the source of every byte width in the cost model. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:273` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:25` |
| `WEAVE_DISPATCH_WARPS` | 空间 | `0 = auto` (4 warps); `4`; `8` | 0 | Per-CTA communication warp budget; 8 warps double the dispatch pull width and raise the CTA to 256 dispatch threads. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:49` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:940` |
| `WEAVE_SCATTER_ACTIVE_SMS` | 空间 | `0 = auto` (all SMs/CUs participate); `K = 1..num_sms`: only K SMs drain the scatter queue | 0 | Decoupled throttled active scatter SMs: the epilogue only publishes descriptors and a bounded SM subset owns NVLink/xGMI scatter. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:173` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:499` |
| `WEAVE_LOCALITY_DOMAINS` | 空间 | `0 = auto`; `1 = single domain`; `2 = dual-die SM103`; `4 = XCD MI308X` | 0 | Splits every L1/L2 N-cluster queue across locality domains; each domain claims locally and steals from the peer at the tail. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:184` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:832` |
| `WEAVE_SM0_STREAM_SCHED` | 空间 | `0 = SM 0 communication warps pull`; `1 = SM 0 communication warps become the streaming scheduler lane` | 0 | Repurposes SM 0/CU 0 communication warps as a dedicated scheduler lane; the remaining SMs evenly own the full dispatch pull range. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:58` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:2773` |
| `WEAVE_AMD_8WAVE_MIN_TOK` | 空间 | `N = minimum local tokens per CU-expert`; `0 = follow WEAVE_DUALGROUP_MIN_TOKENS` | 64 | MI308X 8-wave schedule admission floor: below N tokens the CU keeps the 4-wave schedule. | `deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh:67` | `deep_gemm/include/deep_gemm/impls/amd/ep8/megamoe_amd_ep8.hip:1215` |
| `WEAVE_DUALGROUP_MIN_TOKENS` | 空间 | `N = second compute-group threshold in local tokens` | 64 | Historical MI308X dual-group threshold; the 8-wave knob defaults to this value. | `deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh:14` | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip:1159` |
| `WEAVE_BM_TIER_MODE` | 空间 | `0 = five-tier`; `1 = four-tier`; `2 = EP16 four-tier {32,64,128,256}` | 2 | Per-expert block-M ladder used by the MI308X greedy tile planner. | `deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh:12` | `deep_gemm/include/deep_gemm/impls/amd/include/direct_gather_and_greedy_tiles.hpp:71` |

### 时间（Temporal）

| 参数 | 分类 | 取值（每个接受值均已命名） | 默认值 | 机制 | Host 解析 | Kernel 消费 |
|---|---|---|---|---|---|---|
| `WEAVE_L2_KBLOCK_DEP` | 时间 | `0 = default L1-first order`; `1 = 64-bit XOR L2KBlockDependency bitmask with a tight 1 L2 <-> 1 L1 alternating wave schedule` | 0 | Starts the L2 K block as soon as its matching L1 N subtile completes instead of waiting for the full L1 wave. | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:247` | `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh:833` |
| `WEAVE_RING_BLOCKS` | 时间 | `0 = auto capacity from get_num_max_live_pool_blocks`; `N >= 1 = forced ring block count` | 0 | Bounded ring-buffer reuse for the intermediate SwiGLU activations; the ring wraps after N live pool blocks instead of allocating the whole pool. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:209` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1057` |
| `WEAVE_EARLY_STREAM_REDUCE` | 时间 | `0 = global barrier before combine`; `1 = barrier-free per-token cnt==topk tail-interleaved early reduce` | 0 | Epilogue warps enter the per-token reduce as soon as their last L2 tile is handed off, overlapping the tail with the remaining scatter. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:210` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1368` |
| `WEAVE_TMEM_PIPELINE_MODE` | 时间 | `0 = auto` (2.0 at BM<=240 with one prefetch stage; 1.75 at BM=256); `100 = 1.0`; `175 = 1.75`; `200 = 2.0` | 0 | TMEM epilogue pipeline depth; the 1.75 mode overlaps the tail epilogue with the next accumulator column range at BM=256. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:211` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:3674` |
| `WEAVE_HEAD_PULL_CHUNK` | 时间 | `0 = pull the opening head budget as one prefix`; `N = tokens per head-pull chunk` | 0 | Bounds the all-warp opening head-pull prefix; after N tokens only the selected communication warps keep pulling. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:78` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:2813` |
| `WEAVE_V_HEAD` | 时间 | `0 = automatic`; `N = opening head-pull budget in tokens` | automatic | Opening head-pull budget: all warps pull the first N remote tokens at peak link bandwidth before the first GEMM tile is ready. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:263` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:30` |
| `WEAVE_SCATTER_MODE` | 时间 | `0 = epilogue inline`; `1 = async communication queue`; `2 = local pre-reduce dedup first`; `3 = local bf16x2 dedup + HBM scatter queue` | 1 | Scatter executor ownership: which partition issues the FC2 scatter and whether a local dedup pass runs first. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:192` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:163` |
| `WEAVE_OVERLAP_PIPELINE` | 时间 | `0 = serialized`; `1 = compute-communication overlap` | 1 (forced 0 at preset 2 and above) | Turns compute-communication overlap on or off; the ablation presets derive it automatically. | `deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh:23` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:151` |
| `WEAVE_ALL_WAVE_COMBINE` | 时间 | `0 = compute-group waves only`; `1 = all waves join the cooperative tail combine` | 1 (forced 0 at preset 2 and above) | Cooperative all-wave tail combine; the ablation presets derive it automatically. | `deep_gemm/include/deep_gemm/impls/amd/ep8/build_one.sh:24` | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:181` |
| `WEAVE_READY_QUEUE` | 时间 | `0 = preset expert order`; `1 = global FIFO ready queue` | 0 | Selects the ready-tile source: prefix-summed expert offsets or a global FIFO. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:243` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:354` |
| `WEAVE_READY_GROUP_SIZE` | 时间 | `0 = per-expert readiness`; `N = group-granularity readiness publication` | 0 | Publishes readiness per expert or per group of N experts. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:222` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:58` |
| `WEAVE_SCHED_MODE` | 时间 | `0 = descending load`; `1 = L2 tile swizzle`; `2 = expert-major` | 0 | Tile ordering key used when the ready queue is not a FIFO. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:244` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:370` |
| `WEAVE_EXPERTS_PER_WAVE` | 时间 | `0 = auto`; `N = LPT scheduling window`; `100 + N = inline static order` | 0 | Longest-processing-time scheduling window across experts. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:241` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:322` |

### 实现（Implementation）

| 参数 | 分类 | 取值（每个接受值均已命名） | 默认值 | 机制 | Host 解析 | Kernel 消费 |
|---|---|---|---|---|---|---|
| `WEAVE_ABLATION_PRESET` | 实现 | `0 = full Weave`; `1 = -S`; `2 = -S-T`; `3 = -S-T-C` | 0 | Compile-time progressive ablation selector. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:41` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:26` |
| `WEAVE_ENABLE_COST_MODEL` | 实现 | `0 = static policy`; `1 = CTA-0 runtime cost model` | 1 at preset 0; 0 at presets 1-3 | Master switch for the on-device CTA-0 policy search. | `deep_gemm/include/deep_gemm/scheduler/weave_knobs.cuh:51` | `deep_gemm/include/deep_gemm/scheduler/weave_cost_model.cuh:27` |
| `WEAVE_ADAPT_BM` | 实现 | `0 = fixed per-expert BM`; `1 = per-expert dynamic BM ladder` | 0 | Enables the per-expert adaptive block-M ladder (H20 swapAB tiers, SM103 expert step mode). | `csrc/jit_kernels/heuristics/sm90_mega_moe.hpp:242` | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh:513` |
| `WEAVE_BLOCK_M_L1` | 实现 | `0 = auto`; `16, 32, 64, 96, 128, 192, 240, 256` | 0 | GEMM1 (FC1) per-expert block-M envelope; the adaptive ladder picks BM_e inside it. SM103 and H20 have one static BLOCK_M template, so a non-zero value must equal it. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:86` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1059` |
| `WEAVE_BLOCK_M_L2` | 实现 | `0 = auto`; `16, 32, 64, 96, 128, 192, 240, 256` | 0 | GEMM2 (FC2) per-expert block-M envelope; independent from GEMM1 on the dynamic-scheduler backends. SM103 and H20 require a value equal to the compiled BLOCK_M. | `csrc/jit_kernels/heuristics/weave_knob_config.hpp:89` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1061` |
| `WEAVE_STORE_BLOCK_M_L1` | 实现 | `0 = auto`; positive multiple of 8 (for example `24`) | 0 (auto 24 at BM=240) | GEMM1 epilogue store height; the asymmetric 24/8 split unlocks num_stages = 6. H20 accepts only the full compiled BLOCK_M. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:217` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1063` |
| `WEAVE_STORE_BLOCK_M_L2` | 实现 | `0 = auto`; positive multiple of 8 (for example `8`) | 0 (auto 8 at BM=240) | GEMM2 epilogue store height; the asymmetric 24/8 split unlocks num_stages = 6. H20 accepts only the full compiled BLOCK_M. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:218` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:204` |
| `WEAVE_LOCAL_SCATTER_DEDUP` | 实现 | `0 = full-precision local accumulation`; `1 = bf16x2 hardware L2 atomic dedup before send` | 0 | Fire-and-forget red.add.noftz.bf16x2 merge of duplicate expert contributions into a pre-zeroed L2-resident buffer; one descriptor is published per deduplicated row. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:213` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1371` |
| `WEAVE_REG_PULL_BYTES` | 实现 | `0 = SMEM staging`; `1024, 2048, 4096 = register-direct pull width in bytes` | 0 | Pulls remote token payloads directly into registers instead of staging them in SMEM. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:224` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:90` |
| `WEAVE_SCATTER_SMEM_ALIAS` | 实现 | `0 = dedicated scatter SMEM`; `1 = alias the scatter buffer into the dispatch dead zone` | 1 | Reuses the dispatch buffer's idle window for scatter staging. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:193` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:461` |
| `WEAVE_SCATTER_WARPS` | 实现 | `0 = automatic`; `4`; `8` | 0 | Scatter warp allocation inside the dispatch partition. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:194` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:472` |
| `WEAVE_DEDUP_MODE` | 实现 | `0 = off`; `1 = unique (token, destination) buckets`; `2 = local pre-reduce plus 32-lane fanout` | 0 | Selective input deduplication of duplicate routed reads. | `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:246` | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:386` |
| `WEAVE_PROFILER_LEVEL` | 实现 | `0 = production`; `1 = phase open/close events`; `2 = full per-stage events` | 0 | WeaveProfiler instrumentation level for the AMD port. | `deep_gemm/include/deep_gemm/impls/amd/ep16/build_port.py:97` | `deep_gemm/include/deep_gemm/impls/amd/third_party_overlay/fi_lite_hip.h:8` |

`num_shared_experts` 是 Python/C++ API 参数而非环境变量 knob。所有入口默认取 `0`，只有显式传入非零值时才分配 shared-expert 缓冲区。

## 基线对比

官方 DeepGEMM、DeepEP + DeepGEMM、FlashInfer、SGLang、vLLM、RCCL + AITER
（一阶段与两阶段 CK）等基线的安装与评测脚本见
[baselines/README.md](baselines/README.md) 与
[baselines/README_zh.md](baselines/README_zh.md)。

    python baselines/run_b300_baselines.py --baseline all --dry-run
    python baselines/run_h20_baselines.py --baseline all --dry-run
    python baselines/run_mi308x_baselines.py --baseline all --ep 16 --dry-run

## 消融实验与复现

逐步消融、代价模型相对穷举 Oracle 的评测、EPLB 叠加实验及 WeaveProfiler 时间线
采集脚本见 [ablation/README.md](ablation/README.md) 与
[ablation/README_zh.md](ablation/README_zh.md)。

    python ablation/run_progressive_ablation.py --backend b300 --presets 0,1,2,3 --dry-run
    python ablation/eval_cost_model_vs_oracle.py
    python ablation/capture_weaveprofiler_timeline.py --trace result.sw_hw.perfetto.json.gz --out-png result.png

48 组测试配置下相对官方基线的几何平均加速比（论文 Table IV）：

| 平台 | Weave | -S | -S-T | -S-T-C |
|---|---:|---:|---:|---:|
| NVIDIA SM103 (148-SM) | 1.301 | 1.192 | 1.010 | 0.996 |
| NVIDIA H20 | 1.162 | 1.137 | 1.053 | 0.947 |
| AMD MI308X EP16 | 2.147 | 1.808 | 1.537 | 1.362 |

与专家负载均衡（EPLB）叠加效果（论文 Table V）：不开启 EPLB 时相对基线加速
1.274x；开启 EPLB 后基线本身提升 1.128x，在此基础上叠加 Weave 仍可进一步加速
1.144x。

## 编译与安装

依赖：CUDA 13.x 或 ROCm 7.x、CMake、C++20 编译器、PyTorch 及仓库内置的 CUTLASS；
AMD 构建另需通过 `WEAVE_AITER_SITE` 指定外部 AITER 头文件目录。

    python -m pip install -e .

NVIDIA 内核单文件编译检查：

    nvcc -std=c++20 -ptx -arch=sm_90a -I deep_gemm/include         -I third-party/cutlass/include -I third-party/deep_jit/include         -x cu deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh -o h20.ptx

    nvcc -std=c++20 -ptx -arch=sm_100a -I deep_gemm/include         -I third-party/cutlass/include -I third-party/deep_jit/include         -x cu deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh -o sm103_fp8.ptx

    nvcc -std=c++20 -ptx -arch=sm_100a -I deep_gemm/include         -I third-party/cutlass/include -I third-party/deep_jit/include         -x cu deep_gemm/include/deep_gemm/impls/sm100_fp4_fp4_mega_moe.cuh -o sm103_fp4.ptx

AMD EP16 构建：

    cd deep_gemm/include/deep_gemm/impls/amd/ep16
    WEAVE_PROFILER_LEVEL=0 python build_port.py

## 验证门禁

在仓库根目录运行：

    g++ -std=c++17 -O2 -Wall -Wextra -I deep_gemm/include         tests/test_cost_model_eq123.cpp -o test_cost_model
    ./test_cost_model

    for p in 0 1 2 3; do
      g++ -std=c++17 -O2 -Wall -Wextra -I deep_gemm/include           -DWEAVE_ABLATION_PRESET=$p tests/test_ablation_presets.cpp           -o test_ablation_$p
      ./test_ablation_$p
    done

    python tests/validate_cost_model_3hw.py
    python tests/check_kernel_policy_wiring.py
    python tests/verify_tunable_params_doc.py
    python -m py_compile baselines/*.py ablation/*.py
    python ablation/eval_cost_model_vs_oracle.py

## 引用

如果本项目对您的研究有帮助，请引用我们的论文：

Ziyu Huang, Yangjie Zhou, Chenhao Zhu, Peng Yu, Zihan Liu, Jinyu Liu,
Shulai Zhang, Xingxun Tang, Hongzhe Yan, Xinhao Luo, Minyi Guo, Xiu Lin,
Yinghao Yu, Guodong Yang, Liping Zhang, Shixuan Sun, and Jingwen Leng.
Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for
Compute-Communication Overlap. arXiv:2609.21483, 2026.
<https://arxiv.org/abs/2609.21483>

    @misc{huang2026weavefinegraineddynamicsm,
          title={Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap},
          author={Ziyu Huang and Yangjie Zhou and Chenhao Zhu and Peng Yu and Zihan Liu and Jinyu Liu and Shulai Zhang and Xingxun Tang and Hongzhe Yan and Xinhao Luo and Minyi Guo and Xiu Lin and Yinghao Yu and Guodong Yang and Liping Zhang and Shixuan Sun and Jingwen Leng},
          year={2026},
          eprint={2609.21483},
          archivePrefix={arXiv},
          primaryClass={cs.DC},
          url={https://arxiv.org/abs/2609.21483},
    }
