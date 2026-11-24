# Weave 参数调优与两阶段自动选参指南（Autotune + Runtime Selection）

本文档说明 Weave 在 **NVIDIA Blackwell（SM100 / SM103）**、**NVIDIA Hopper（SM90，如 H20 / H100 / H200）** 与 **AMD CDNA3（gfx942，如 MI308X / MI300X / MI325X）** 三类架构上的完整参数确定与调优方法。整套机制严格建立在 **14 个公开参数（Spatial 5 + Temporal 3 + Implementation 6）** 之上，代码中不包含任何卡名字符串分支或固定 SM/CU 数常量。

---

## 1. 整体流程概览（先定标，再零环境变量运行）

当用户拿到一台同架构家族的新机器时，只需执行两步：

1. **编译期一次性定标（Stage 1，`deep_gemm.mega.autotune`）**：
   动态读取当前卡硬件指纹（架构号、物理 SM/CU 数、每 SM/CU 共享内存、L2 容量、EP 并行度），对不同全局 token 桶 `S_bucket in {16, 128, 512, 1024, 4096, 8192, 16384}` 自动完成编译筛选（含 0-spill 与同精度全张量 `cos > 0.99` 校验）、分步回文 ABBA 实测优选、以及 4 条在线路由规则的盈亏阈值标定，写出本机专属 JSON 配置文件。
2. **运行时极简自动选参（Stage 2，`weave_mega_moe`）**：
   线上正常调用 `weave_mega_moe(...)`，**无需设置任何 `WEAVE_*` 环境变量**。运行时首次加载一次 JSON 进内存缓存，按 `abs(log2(S_global) - log2(S_bucket))` 命中最近分桶，并根据当步路由统计执行 4 条轻量级阈值判定（仅允许切向定标阶段实测验证过的 `verified_variants`）。CUDA / HIP Graph 重放阶段 host 侧选参开销为 `0`。

### 快速命令

```bash
# 第一步：在新硬件上按模型形状跑一次编译期实测定标（每个 GPU 型号 x EP 规模 x 模型形状跑一次）
python -m deep_gemm.mega.autotune --model qwen3.8-2.4t --num-processes 4
# 或直接传入任意自定义 MoE 维度：
python -m deep_gemm.mega.autotune --num-experts 512 --num-topk 10 \
    --hidden 8192 --intermediate-hidden 2048 --num-processes 4
# 产物默认保存在：
#   ~/.deep_gemm/weave_autotune/{hw_id}__E512_K10_H8192_I2048_S0_{mma}.json
# 可通过 WEAVE_AUTOTUNE_DIR 或 WEAVE_AUTOTUNE_FILE 指定目录或文件路径
```

```python
# 第二步：线上推理直接调用，不设任何 WEAVE_* 调参环境变量
from deep_gemm.mega import weave_mega_moe
weave_mega_moe(y, l1_weights, l2_weights, sym_buffer)
```

运行时优先级从高到低依次为：
1. **手动覆盖模式（Manual）**：若环境变量中显式设置了任意公开 `WEAVE_*` 调参变量，自动选参立即退出，100% 原样执行用户指定配置；
2. **关闭自动选参（Disabled）**：`WEAVE_AUTOTUNE=0`，直接使用内核编译默认配置；
3. **定标配置模式（Profile）**：严格匹配 `hw_id` 与模型维度，查桶并执行 4 条在线阈值规则；
4. **架构族冷启动（Cold Start）**：未找到定标文件时自动使用架构族安全默认（Blackwell 用自适应编译默认，Hopper 用 `comm_only + smem`，CDNA3 小 `S <= 128` 用 `0-8`、大 `S` 用 `4-4 + REG_PULL_BYTES=8192`）。

---

## 2. 14 个公开参数的分类与调优准则

完整逐项代码行号引用见 [TUNABLE_PARAMETERS.md](TUNABLE_PARAMETERS.md)。当需要手动精调或理解定标器行为时，按以下三类职责调整：

### 2.1 空间维度（Spatial，5 个 —— 跨 SM/CU 与单 SM/CU 内通算资源配比）

| 参数 | 取值 | 何时调整与物理准则 |
|---|---|---|
| `WEAVE_DISPATCH_ACTIVE_SMS` | `0=全部`，`D=1..N` | 当小中 `S` 下全卡同时拉取导致 NVLink/xGMI 争用或延误首轮权重预取时，设为 `round(0.5 * N_SM)` 或 `round(0.85 * N_SM)`（SM10x 需对齐 cluster=2，CDNA3 对齐 XCD=4）。与 `WEAVE_SCATTER_ACTIVE_SMS` 相互独立。 |
| `WEAVE_SCATTER_ACTIVE_SMS` | `0=全部`，`K=1..N` | 仅在使用 HBM 队列异步排空或限制跨卡写出并发度时收紧为 `K < N_SM`；`smem` 环模式下由各 SM 自身通信 warp 排空，保持 `0`。 |
| `WEAVE_DISPATCH_WARPS` | `0=默认`，`{4, 8}` | 通信受限的大 `S` 场景（如 H20 `S >= 8192`）切 `8` 可翻倍拉取与发送并发；算力或寄存器紧缺场景用 `4` 或 `0`。 |
| `WEAVE_SCATTER_WARPS` | `0=默认`，`{4, 8}` | 控制 CTA 内专职异步 FC2 Scatter 的通信 warp/wave 数；双机 EP16 大流量回传时可开 `8`。 |
| `WEAVE_CU_WAVE_SPLIT` | MI308X：`{auto, 0-8, 4-4, 8-0}`；H20：`7-W` / `5-E` / `6-E` | **MI308X**：`S <= 128` 权重受限选 `0-8`（8 个计算 wave 跑满 HBM），`S >= 1024` 跨卡通信受限选 `4-4`（4 通信 wave 与 4 计算 wave 同驻并发）。**H20**：默认 `0/auto` 启用无全局栅栏的 `7-W` Lag-W 专家任务流（`GEMM1` 领先 `W` 个专家，按 K-block 位图细粒度交错，消除算子间 `grid_sync`）。 |

### 2.2 时间维度（Temporal，3 个 —— 首、中、尾流水重叠与同步粒度）

| 参数 | 取值 | 何时调整与物理准则 |
|---|---|---|
| `WEAVE_V_HEAD` | `0=auto`，`N > 0` | 层启动阶段空闲计算/尾声 warp 下场协助拉取的首批配额。中小 `S`（如 SM103 `S <= 2048` 设 `1`，H20 设 `10 * N_SM`）可缩短首块 GEMM1 等待时间；超大 `S` 或严重偏斜路由下减小配额避免拖慢计算起跑。 |
| `WEAVE_READY_GROUP_SIZE` | `0=整专家`，`{1, 2, 4, 8, 16}` | 热点专家负载严重偏斜（`m_max / mean_m > 3`）时设为 `2..16`，无需等整专家 token 到齐即可按块组起算 GEMM1。 |
| `WEAVE_SCATTER_OWNER` | `0=epilogue_only`，`1=comm_only` | 小 `S` 且 GEMM2 块数不足以掩盖交接开销时用 `epilogue_only`；中大 `S` 切 `comm_only`（配合 `WEAVE_SCATTER_STAGING=smem`）让通信 warp 在后台与下一块 WGMMA/UMMA 完全重叠。 |

### 2.3 单算子实现维度（Implementation，6 个 —— 四算子独立配置）

| 参数 | 取值 | 何时调整与物理准则 |
|---|---|---|
| `WEAVE_PRECISION_MODE` | `fp8xfp4`、`fp4xfp4`、`fp8xfp8`、`bf16` | 选择张量核数值精度（SM10x 默认 `fp8xfp4`，H20 与 MI308X 使用 `fp8`）。 |
| `WEAVE_GEMM1_CONFIG` | `0=auto` 或显式 `BM[:STORE_BM[:BK[:STAGES[:EPI]]]]` | **保持 `0` 时默认常开形状自适应元组选择**（不再需要旧的 `WEAVE_ADAPT_GEMM_CONFIG`）。手动指定或定标分桶时按每专家期望行数 `M_exp = S_global * topk / E` 选邻近 `BM`：选定 `BM` 即同步锁定匹配的 `BN/BK/stages/epi_stages`（SM103 上 `BM=128` 自动钳位 `epi=1` 保证 0 spill）。 |
| `WEAVE_GEMM2_CONFIG` | `0=跟随/自适应` 或显式元组 | 与 GEMM1 独立解耦（例如 H20 小中 `S` 大中间维模型上 `GEMM1=32, GEMM2=64` 省去 FC1 padding 同时保住 FC2 带宽）。 |
| `WEAVE_REG_PULL_BYTES` | `0`、NVIDIA `{1024, 2048, 4096}`、AMD `{8192}` | 按通信 warp 空闲寄存器预算计算：`B_reg_pull <= floor((R_comm - R_ctrl) / 4) * 16 * Lanes_per_warp`。SM10x 因 TMEM 解放累加器寄存器，选 `1024` 或 `2048` 省出 32 KB SMEM 给更多 TMA stage；AMD 64-lane wavefront 选 `8192`；H20 寄存器紧缺保持默认轻量档。 |
| `WEAVE_DEDUP_MODE` | `0=off`，`1=on` | 当多专家同卡命中率 `mu = m_rem / d_in` 较高、节省行数 `m_rem - d_in >= dedup_min_saved_rows` 时开启 `1`，跨卡只拉取一次唯一 token 并在片内扇出（双机 EP16 同时启用节点内预归约减少跨机 RDMA 包数）。 |
| `WEAVE_SCATTER_STAGING` | `0=reg_2stage`，`1=smem`，`2=hbm_dedup` | `epilogue_only` 配 `0 (reg_2stage)`；`comm_only` 在 NVIDIA 上首选 `1 (smem)`（每 SM 独立双缓冲环，零 HBM 往返），在 CDNA3 寄存器直传通路下配 `0`。 |

---

## 3. 编译期定标与运行时选参算法细节

### 3.1 两遍预编译与 0-Spill 门禁（`DG_JIT_PTXAS_CHECK=2`）
- 上游 `DG_JIT_PTXAS_CHECK=1` 只要出现非零栈帧即报错，而 SM10x 内核普遍带有 `56 B` 非溢出栈帧。
- 本仓库在 `csrc/jit/compiler.hpp` 中新增 `DG_JIT_PTXAS_CHECK=2`：打开 `ptxas --warn-on-spills` 并精确解析 `spill stores / spill loads` 字节数。
- **第一遍预编译**在独立临时缓存目录开启 `DG_JIT_PTXAS_CHECK=2`，在 Blackwell 上对任何 `spill > 0` 的候选直接剔除并记录进 `spill_verdicts`；**第二遍预编译**使用生产编译参数将通过门禁的候选写入正式 `DG_JIT_CACHE_DIR`，确保校准期与线上推理命中逐字节一致的 cubin。

### 3.2 分桶搜索与保底复测
对每个 `S_bucket`：
1. 按 `M_exp = S_bucket * topk / E` 从架构合法阶梯中取出 `3~5` 组 `GEMM1/GEMM2` 元组做配对回文 ABBA；
2. 固定当前最优 GEMM 元组，逐组单步扫描 `pull`、`scatter`、`SM/CU 比例`、`warps/waves`、`v_head`；
3. 将各组正收益项合并复测，并尝试邻近次优 GEMM 元组二次确认；
4. **保底复测（Never Worse Than Default）**：将当前最优与冷启动、编译默认 `{}` 同场回文 ABBA 复测，若相对 `{}` 净收益不足 `0.5%`，直接采用 `{}`，杜绝过拟合噪声。

### 3.3 运行时 4 条轻量级阈值规则
当线上拿到当步路由统计 `(m_rem, d_in, d_out, k_tiles, m_max, mean_m)` 时，按优先级顺序计算增量覆盖：
1. `dedup`：`m_rem - d_in >= dedup_min_saved_rows` -> `WEAVE_DEDUP_MODE=1` 否则 `0`；
2. `comm_scatter`：`d_out >= comm_scatter_min_rows and k_tiles >= comm_scatter_min_tiles` -> `comm_only + smem` 否则 `epilogue_only + reg_2stage`；
3. `spatial_tier`：相对均匀路由期望块数，`k_tiles < 0.6 * exp_tiles and d_in > exp_pull` 切 `skew_comm_heavy`，`k_tiles > 1.5 * exp_tiles` 切 `skew_comp_heavy`；
4. `ready_group`：`m_max > 3.0 * mean_m` -> `WEAVE_READY_GROUP_SIZE = skew_rgs`。

若组合后的配置不在该桶定标阶段实测通过的 `verified_variants` 集合内，自动按低优先级依次回退直至落入已验证集合（桶基准 `base_env` 始终在集合内）。

---

## 4. 跨架构实测验证汇总

- **SM100 / SM103 跨 SKU 编译证明**（`results/blackwell_sm100a_sm103a_compile_proof.json`，1440 个编译单元覆盖 `sm_100a` 与 `sm_103a` 及 `num_sms in {144, 148, 160}`）：`sm_100a` 与 `sm_103a` 结果 100% 一致，`compile_failed = 0`。
- **H20 零环境变量端到端实测**（`results/autotune_e2e_b300_h20.json`，3 模型 x 10 格 uniform/skew）：30/30 格 `auto` 全部落在手调 Oracle 的 `2%` 以内（几何平均 `auto/Oracle` 为 `0.954 ~ 0.997`），对官方 DeepGEMM 几何平均加速 **`1.13x ~ 1.25x`**（单点最高 **`1.69x`**）。
- **SM103 (148-SM) 零环境变量端到端实测**（`results/autotune_e2e_b300_h20.json`，2 模型 x 10 格）：19/20 格落在手调 Oracle 的 `2%` 以内（几何平均 `auto/Oracle` 为 `0.973` 与 `0.992`）。
- **AMD MI308X 双机 16x EP16 V70 实测**（`results/ep16_mi308x_v70_30cells_bound_util.json`、`results/autotune_e2e_mi308x_v70.json`，6 模型 x 5 S = 30 格，16-rank FP64 全元素校验 100% 通过）：30/30 格超越 `AITER-1stage` 与 `vLLM`，几何平均加速 **`1.875x`**（`S=16384` 六模型瓶颈峰值利用率达 **`58.0% ~ 92.0%`**）。
