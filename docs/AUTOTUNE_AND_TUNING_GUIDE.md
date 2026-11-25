# Weave 参数调优与自动选参指南

本文档说明 Weave 在 **NVIDIA SM100 / SM103**、**NVIDIA Hopper（SM90，如 H20 / H100 / H200）** 与 **AMD CDNA3（gfx942，如 MI308X / MI300X / MI325X）** 三类架构上的自动校准与手动调优方法。整套机制基于统一的 **14 个参数（Spatial 5 + Temporal 3 + Implementation 6）**，硬件拓扑与片上资源容量均在运行时动态查询。

---

## 1. 两步使用流程（离线校准 + 在线自动选参）

1. **离线校准（`deep_gemm.mega.autotune`）**：
   自动探测当前 GPU 的架构、物理 SM/CU 数、每 SM/CU 片上内存、L2 容量与 EP 并行度，针对各全局 Token 分桶 `S_bucket in {16, 128, 512, 1024, 4096, 8192, 16384}` 测量候选分块元组与通算配比，生成本机 JSON 配置文件。
2. **在线自动选参（`weave_mega_moe`）**：
   正常调用 `weave_mega_moe(...)` 时无需手工指定环境变量；运行时自动缓存本机配置文件，按当前全局 Token 数与当步路由负载选择空间配比、时间切换策略和四算子配置（CUDA / HIP Graph 重放阶段无额外 Host 开销）。

### 快速命令

```bash
# 第一步：按预设模型或自定义 MoE 维度运行一次离线校准
python -m deep_gemm.mega.autotune --model qwen3.8-2.4t --num-processes 4
# 或直接指定自定义 MoE 形状：
python -m deep_gemm.mega.autotune --num-experts 512 --num-topk 10 \
    --hidden 8192 --intermediate-hidden 2048 --num-processes 4
```

```python
# 第二步：在线推理直接调用
from deep_gemm.mega import weave_mega_moe
weave_mega_moe(y, l1_weights, l2_weights, sym_buffer)
```

运行时配置优先级从高到低依次为：
1. **显式环境变量覆盖**：若设置了任意 `WEAVE_*` 参数，优先按显式指定值执行；
2. **禁用自动选参（`WEAVE_AUTOTUNE=0`）**：使用编译默认参数；
3. **本机校准配置**：匹配硬件指纹与模型维度后按实时路由统计选参；
4. **架构默认策略**：未生成校准文件时使用各架构的默认安全配置。

---

## 2. 14 个参数的职责与调优准则

完整代码定位见 [TUNABLE_PARAMETERS.md](TUNABLE_PARAMETERS.md)。

### 2.1 空间维度（Spatial，5 个 —— 跨 SM/CU 与单 SM/CU 内资源配比）

| 参数 | 取值 | 物理含义与调优准则 |
|---|---|---|
| `WEAVE_DISPATCH_ACTIVE_SMS` | `0=auto`，`D=1..N` | 跨 SM/CU 限制参与远端拉取的 SM/CU 数 `D`，让其余 `N-D` 个单元立即预取权重并启动 GEMM1；与 `WEAVE_SCATTER_ACTIVE_SMS` 相互独立。 |
| `WEAVE_SCATTER_ACTIVE_SMS` | `0=auto`，`K=1..N` | 跨 SM/CU 独立控制参与异步 Scatter 排空的 SM/CU 数 `K`；尾部 Combine 仍使用全卡全部单元。 |
| `WEAVE_DISPATCH_WARPS` | `0=auto`，`{4, 8}` | 单 SM 内分配给远端拉取的通信 warp 数；跨卡流量密集的大批量场景可增至 `8`。 |
| `WEAVE_SCATTER_WARPS` | `0=auto`，`{4, 8}` | 单 SM/CU 内分配给异步 FC2 Scatter 写出的通信 warp/wave 数。 |
| `WEAVE_CU_WAVE_SPLIT` | MI308X：`{auto, 0-8, 4-4, 8-0}`；H20：`auto` 或 `E` | MI308X 上控制单 CU 8 个 Wave64 的通算切分（小批量权重受限倾向 `0-8`，大批量跨卡通信密集倾向 `4-4` 混合同驻）；H20 上控制 GEMM1/GEMM2 专家任务流的交错步幅。 |

### 2.2 时间维度（Temporal，3 个 —— 阶段内角色切换与就绪粒度）

| 参数 | 取值 | 物理含义与调优准则 |
|---|---|---|
| `WEAVE_V_HEAD` | `0=auto`，`N > 0` | 层启动阶段空闲计算/尾声 warp 临时协助拉取的任务配额，用于缩短首块 GEMM1 的启动等待。 |
| `WEAVE_READY_GROUP_SIZE` | `0=整专家`，`{1, 2, 4, 8, 16}` | 按 `N` 个 `BLOCK_M` tile 为粒度发布到达信号，热点专家无需等全部 Token 拉齐即可启动 GEMM1。 |
| `WEAVE_SCATTER_OWNER` | `0=epilogue_only`，`1=comm_only` | 中部阶段 FC2 跨卡发送的角色归属：小批量直接由尾声写出（`epilogue_only`），中大批量交由释放后的通信 warp 后台异步发出（`comm_only`）以重叠后续矩阵计算。 |

### 2.3 单算子实现维度（Implementation，6 个 —— 四算子独立配置）

| 参数 | 取值 | 物理含义与调优准则 |
|---|---|---|
| `WEAVE_PRECISION_MODE` | `fp8xfp4`、`fp4xfp4`、`fp8xfp8`、`bf16` | 选择 GEMM1/GEMM2 的张量核数值精度与对应后端。 |
| `WEAVE_GEMM1_CONFIG` | `0=auto` 或 `BM[:STORE_BM[:BK[:STAGES[:EPI]]]]` | 独立配置 GEMM1（FC1）：`0` 按每专家期望行数 `M_exp = S_global * topk / E` 自动选择完整分块与流水元组；在 SM103 上支持显式指定 `EPI in 1..4` 多级尾声寄存器流水以提前释放 TMEM。 |
| `WEAVE_GEMM2_CONFIG` | `0=auto` 或显式元组 | 与 GEMM1 独立解耦的 GEMM2（FC2）配置，可根据 `(N2, K2) = (H, I)` 选择不同的 `BM` 与尾声写出高度。 |
| `WEAVE_REG_PULL_BYTES` | `0`、NVIDIA `{1024, 2048, 4096}`、MI308X `{8192}` | Dispatch 算子单轮寄存器直传拉取宽度，绕过片上共享内存以保留更多容量给 GEMM 主循环流水。 |
| `WEAVE_DEDUP_MODE` | `0=off`，`1=on` | Dispatch 算子跨卡拉取去重：当同一远端 Token 命中本卡多个专家时仅跨卡拉取一次并在片内扇出。 |
| `WEAVE_SCATTER_STAGING` | `0=reg`，`1=smem`，`2=hbm` | Scatter 算子暂存通路：`reg` 为多级寄存器直传，`smem` 为片上共享内存暂存并由通信 warp 异步发出，`hbm` 为片外显存缓冲本地预合并同专家重复命中后再发送。 |

---

## 3. 在线路由自适应规则概要

在线阶段读取当步路由统计 `(m_rem, d_in, d_out, k_tiles, m_max, mean_m)`，在已验证的候选集合内动态调整：
1. **跨卡去重判定**：当去重节省的跨卡行数 `m_rem - d_in` 超过校准阈值时启用 `WEAVE_DEDUP_MODE=1`；
2. **异步 Scatter 切换**：当跨卡写出规模 `d_out` 与计算块数 `k_tiles` 足以掩盖交接开销时切换为 `comm_only + smem`，否则采用 `epilogue_only + reg`；
3. **跨 SM/CU 通算配比调整**：根据实时计算块数与远端拉取量之比在通信密集与计算密集配比间切换；
4. **热点专家细粒度起算**：当热点专家偏斜度 `m_max / mean_m > 3` 时缩小 `WEAVE_READY_GROUP_SIZE`，提前启动热点专家的 GEMM1。
