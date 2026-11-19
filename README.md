# Weave: Unified C++ MegaMoE for B300 / H20 / MI308X

本仓库是论文 **Weave** 的统一 C++ 实现，直接基于 DeepGEMM 的 MegaMoE 代码组织方式，用同一套 C++17 / CUDA C++ / HIP C++ 接口覆盖三种硬件：

| 后端 | 硬件 | 语言 / 编译单元 | 入口 |
|---|---|---|---|
| B300 FP8xFP4 | NVIDIA Blackwell SM100/SM103 | CUDA C++ | `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` |
| B300 NVFP4 | NVIDIA Blackwell SM100/SM103 | CUDA C++ | `deep_gemm/include/deep_gemm/impls/sm100_fp4_fp4_mega_moe.cuh` |
| H20 FP8 | NVIDIA Hopper SM90 | CUDA C++ | `deep_gemm/include/deep_gemm/impls/sm90_fp8_mega_moe.cuh` |
| MI308X FP8 | AMD CDNA3 gfx942 | HIP C++ | `deep_gemm/include/deep_gemm/impls/amd/megamoe_amd.hip` |

核心目标不是把三份 kernel 放进一个目录，而是让三者共享同一套：

1. **空间调度 r = (N_c, N_e)**：通信 / 计算资源划分；
2. **时间调度 s in {epi, comm}**、head pull 预算 V_head、tail fine-grained reduce；
3. **CTA-0 运行时 cost model**：在 kernel 内读取真实 post-dispatch 直方图，按论文 Eq.(1)-(3) 选策略；
4. **显式 knob / 消融**：`WeaveKnobs` 可覆盖自动选择，支持 `-S`、`-S-T`、`-S-T-C`。

---

## 1. 目录结构

```text
weave/
├── README.md
├── setup.py / CMakeLists.txt
├── deep_gemm/
│   ├── __init__.py
│   ├── mega/
│   │   ├── __init__.py                 # 统一 Python 绑定
│   │   └── cost_model.py               # 与 C++ 同参数的纯 Python 镜像
│   └── include/deep_gemm/
│       ├── scheduler/
│       │   ├── mega_moe.cuh            # MegaMoE persistent scheduler
│       │   ├── weave_knobs.cuh         # 统一 spatial / temporal / dataflow knobs
│       │   ├── weave_cost_model.cuh    # __host__ __device__ Eq.(1)-(3) + CTA-0 选择
│       │   └── weave_policy_runtime.cuh# 发布 / acquire / 全 CTA 消费
│       ├── layout/
│       │   └── mega_moe.cuh            # symmetric buffer & workspace layout
│       ├── profiler/
│       │   └── mega_moe_profiler.cuh   # 16B timeline recorder
│       └── impls/
│           ├── sm100_fp8_fp4_mega_moe.cuh
│           ├── sm100_fp4_fp4_mega_moe.cuh
│           ├── sm90_fp8_mega_moe.cuh
│           ├── gfx942_fp8_mega_moe.cuh
│           └── amd/                    # MI308X HIP 实现与 include/*.hpp
├── csrc/
│   ├── apis/mega.hpp
│   └── jit_kernels/
│       ├── heuristics/{mega_moe.hpp, sm90_mega_moe.hpp, sm100.hpp}
│       └── impls/{sm100_fp8_fp4_mega_moe.hpp, sm90_fp8_mega_moe.hpp, gfx942_fp8_mega_moe.hpp}
├── tests/
│   ├── test_cost_model_eq123.cpp
│   ├── test_ablation_presets.cpp
│   ├── validate_cost_model_3hw.py
│   ├── check_kernel_policy_wiring.py
│   ├── bench_b300_ep8.py
│   ├── bench_h20_ep8.py
│   └── bench_mi308x_ep8_ep16.py
├── data/
│   ├── routing_stats_3hw.json
│   └── routing_duplicate_stats_b300.json
└── results/
    ├── cost_model_validation_3hw.json
    ├── kernel_policy_wiring.json
    ├── local_validation_summary.json
    ├── cta0_costmodel_latency_summary.json
    ├── h20_table4_48cells_ablation_and_costmodel.json
    ├── ep8_all16_synced_summary.json
    ├── ep16_baselines_and_roofline.json
    └── provenance.json
```

---

## 2. CTA-0 运行时选择：真正在 kernel 内完成

所有 CUDA/HIP 后端都走同一条 rail：

```text
dispatch 完成
  -> expert_recv_count_sum / plan_counts 成为 live post-dispatch histogram m_e
  -> CTA 0 读取 m_e，计算 M = sum_e m_e，读取 D_in、M_out
  -> CTA 0 在 2 个 spatial candidate x 2 个 temporal mode 上评估 Eq.(1)-(3)
  -> CTA 0 写 PolicyControl::decision，release-store published_epoch
  -> 其他 CTA acquire-wait 同一个 epoch，读取同一份 decision
  -> 所有 CTA 执行 selected_r_mask / selected_scatter_mode / selected_v_head
```

发布对象 `weave::PolicyControl` 位于 device global workspace 的末尾（workspace 内部对齐 128B），不需要 host round trip，也不需要 dynamic parallel launch。H20 不把 decision 放进会被 scatter/combine 复用的 pre-barrier SMEM 区域；读侧使用 acquire load。

### 2.1 论文公式（ASCII）

硬件参数与候选集在 `weave_cost_model.cuh` 和 `deep_gemm/mega/cost_model.py` 中保持同一份数值。

```text
Eq.(1)
F_gemm(r) = sum_e K_e(r) * B_M(r) * 6 * H * I
V_mem(r)  = sum_e K_e(r) * 3 * H * I * b_w + M * b_a * (H + I)

Eq.(2)
T_mma(r) = max(V_mem(r) / B_mem, F_gemm(r) / P_tc(r))
V_epi    = M * (b_a * I + b_o * H)
T_epi    = V_epi / B_epi(N_e)
B300:        T_comp(r) = max(T_mma(r), T_epi(N_e))
H20 / MI308X: T_comp(r) = T_mma(r) + T_epi(N_e)

Eq.(3)
T_pull(N_c) = D_in * H * b_a / min(B_link, N_c * b_pull)
T_scat(N)   = M_out * H * b_o / min(B_link, N * b_scat)
T(r, epi)   = max(T_pull(N_c), T_comp(r) + T_scat(N_e))
T(r, comm)  = max(T_pull(N_c) + T_scat(N_c), T_comp(r) + T_hoff)
```

其中 `K_e(r) = ceil(m_e / B_M(r))`，`N_c` / `N_e` 在 Eq.(3) 中按 GPU-wide 资源数展开：

```text
N_c_gpu = num_SMs_or_CUs * N_c_per_SM
N_e_gpu = num_SMs_or_CUs * N_e_per_SM
```

Table I 候选：

```text
B300:   (N_c, N_e) = (4, 8) or (8, 4) per SM
H20:    (N_c, N_e) = (2, 16) or (10, 8) per SM
MI308X: (N_c, N_e) = (0, 8) or (4, 4) per CU
```

### 2.2 显式 knob 与渐进消融

`weave_knobs.cuh` 提供：

```text
WeaveKnobs {
  enable_runtime_cost_model
  ablation_preset
  override_mask        # bit0 spatial, bit1 scatter, bit2 V_head, bit3 temporal
  override_r_mask      # bits[7:0]=N_c, bits[15:8]=N_e, bits[23:16]=groups, bit31=hetero
  override_scatter_mode
  override_v_head
  comm_units / comp_units / block_m_fixed / ...
}
```

编译期 preset：

```text
WEAVE_ABLATION_PRESET=0  full Weave, CTA-0 auto selection
WEAVE_ABLATION_PRESET=1  -S      fixed spatial candidate, no spatial search
WEAVE_ABLATION_PRESET=2  -S-T    additionally disables overlap pipeline / all-wave combine / tail fine-grained
WEAVE_ABLATION_PRESET=3  -S-T-C  additionally disables selective dedup
```

Preset 1/2/3 默认把 `enable_runtime_cost_model` 置 0，并设置 spatial + scatter override；因此消融不会“偷偷”继续跑自动搜索。显式定义 `WEAVE_KNOB_ENABLE_RUNTIME_COST_MODEL=1` 可以覆盖这一默认。

### 2.3 各后端实际消费的字段

| 后端 | selected_r_mask | selected_scatter_mode | selected_v_head |
|---|---|---|---|
| B300 FP8xFP4 | 限制 head 之后 active pull warps、tail combine warps | 切换 `CommQueueAsync` handoff / inline epilogue | 限制 head pull 预算 |
| B300 NVFP4 | 限制 head 之后 active pull warps | 仅发布；该 kernel 无独立 comm-scatter 路径 | 限制 head pull 预算 |
| H20 FP8 | 限制 active pull warps | `s=comm` 走 dispatch L2 scatter loop；`s=epi` 跳过 | 限制 head pull 预算 |
| MI308X FP8 | 映射到 V1864 roles，但当前 v4 wave 映射尚未用 roles 改变 dispatch/GEMM 分区 | `s=epi` 时关闭 scatter reclaim 的 dual 模式 | 调整 comm batch size |

这张表刻意区分“已发布”和“已改变执行路径”。B300 NVFP4 与 MI308X 的空间角色消费仍有边界，不能把“policy 里有字段”说成“调度已完全生效”。

---

## 3. 构建

### 3.1 依赖

```bash
git submodule update --init --recursive
# third-party/cutlass  f3fde58372d33e9a5650ba7b80fc48b3b49d40c8
# third-party/deep_jit   e5bdee2bc4ca519eba00cfc5f0c6e950e6a96a16
```

- NVIDIA 路径：CUDA 12.8+（SM100/SM103 需要对应 toolkit），Python 3.10+，PyTorch 2.x。
- AMD 路径：ROCm 6.x + `hipcc`，gfx942。
- 纯 cost model 测试只需要 g++ / clang++，不需要 GPU。

### 3.2 安装

```bash
pip install -e .
# 或仅做 JIT 源码安装
python setup.py develop
```

### 3.3 最小 usage

Python（纯 cost model，不需要编译扩展；根目录的 `weave_cost_model.py` 会直接加载镜像）：

```python
from weave_cost_model import WorkloadSpec, select_policy

br = select_policy(
    "b300", WorkloadSpec(local_experts=64, hidden=8192, intermediate=2048),
    m_e=[...], dedup_pulls=..., scatter_count=...)
print(br.r_mask, br.scatter_mode, br.v_head, br.t_total_us)
```

C++：

```cpp
#include <deep_gemm/scheduler/weave_cost_model.cuh>

weave::WorkloadSpec wl{64, 8192, 2048};
auto knobs = weave::default_weave_knobs(weave::HardwareArch::B300_SM100);
auto decision = weave::cta0_select_policy(
    weave::HardwareArch::B300_SM100, wl, knobs, m_e, d_in, m_out);
```

---

## 4. 测试与当前验证状态

### 4.1 已真实通过（本机）

```bash
# 1. 纯 C++17 host 版 Eq.(1)-(3) + CTA-0 argmin
mkdir -p build
g++ -std=c++17 -O2 -I deep_gemm/include \
    tests/test_cost_model_eq123.cpp -o build/test_cost_model
./build/test_cost_model

# 2. 消融 preset 0/1/2/3
for p in 0 1 2 3; do
  g++ -std=c++17 -O2 -I deep_gemm/include -DWEAVE_ABLATION_PRESET=$p \
      tests/test_ablation_presets.cpp -o build/test_ablation_$p
  ./build/test_ablation_$p
done

# 3. 三硬件真实 routing 聚合回放（m_e 为 proxy，见下）
python tests/validate_cost_model_3hw.py

# 4. 源码级 wiring gate
python tests/check_kernel_policy_wiring.py
```

本机实测结果：

```text
[PASS] B300_SM100: candidates=2 auto=(r=0,s=0,v_head=32,total=14.022us)
[PASS] H20_SM90:   candidates=2 auto=(r=1,s=0,v_head=32,total=18.107us)
[PASS] MI308X_GFX942: candidates=2 auto=(r=0,s=0,v_head=32,total=22.103us)
ALL EQ(1)-(3) AND CTA-0 SELECTION TESTS PASSED

ABLATION PRESET 0/1/2/3 PASSED

[PASS] b300_ep8: cells=6 selection_counts={'0': 1, '1': 5}
[PASS] h20_ep8: cells=6 selection_counts={'0': 0, '1': 6}
[PASS] mi308x_ep8: cells=6 selection_counts={'0': 0, '1': 6}
[PASS] mi308x_ep16: cells=6 selection_counts={'0': 0, '1': 6}
OVERALL=PASS

PASS b300
PASS b300_nvfp4
PASS h20
PASS mi308x_body
PASS mi308x_route
PASS mi308x_comm
```

其中 `b300_ep8` 在真实 B300 routing 聚合上复现了论文趋势：小 S 选 `s=epi`，大 S 选 `s=comm`。但 `m_e` 是由 `local_contributions` 用 Zipf(0.9) 确定性拆分得到的 proxy，因为原始 artifact 只有聚合值，没有 per-local-expert 直方图。该边界在 `data/routing_stats_3hw.json` 和 `results/cost_model_validation_3hw.json` 中都有显式标注。

### 4.2 编译验证

- `tests/test_cost_model_eq123.cpp`：
  - WSL Ubuntu-24.04 + g++ 13.3：真实编译并运行通过；
  - Windows + nvcc 13.3 + `-arch=sm_90a`：device smoke kernel 真实编译通过（生成 exe）。
- 三个 kernel translation unit：
  - nvcc 13.3 `-ptx -arch=sm_90a`（H20）/`-arch=sm_100a`（B300 FP8、B300 NVFP4）：front-end 全量解析通过。
  - 这是 **syntax / semantic parse**，不是 template instantiation，也不是 GPU 运行。
- AMD：
  - 本机没有 `hipcc`/ROCm，且归档的 AMD v4 源缺少 `official_m16_m32_configs_v1262.hpp`、`gemm2_configs_v1010_hip.hpp`、`v956_stage1_body.hpp`、`m16_packet_support.hpp`、`fi_lite_hip.h`，因此 AMD 端到端未编译、未运行。

### 4.3 尚未完成 / 不能声称的部分

1. **没有 B300 / H20 / MI308X 实机**：本机只有 RTX 3050 Laptop 4 GB，不能跑三硬件端到端。
2. **没有重新测量性能**：`results/` 中的 CSV/JSON 是从旧 rebuttal 归档按字节复制的 measured 结果，`results/provenance.json` 记录了源路径和 SHA256；本次构建没有重新跑这些 benchmark。
3. **CTA-0 latency 边界**：旧归档 `cta0_costmodel_latency_summary.json` 记录 B300 median 3.648 us、max 4.288 us；H20 median 1.856 us、max 1.920 us；MI308X median 3.000 us、max 3.360 us。B300 max 超过 3.65 us，本仓库新的 selection loop 尚未在 B300 上重新测量，不能宣称“新实现 < 3.65 us”。
4. **D_in 边界**：B300/H20 的 policy decision 点在 dedup pull 之前，真实的 dedup key count 尚未物化；因此 `D_in` 暂用 `M`（无 dedup 上界），`M_out` 使用 live send counters。AMD 同样暂用 `M`。cost model API 本身接受真实 `D_in`，Python 验证使用的是真实 routing 聚合中的 `local_contributions + dedup_remote_scatter_rows`。
5. **B300 NVFP4 的 s 字段**：只发布，不切换路径；见 2.3 表。
6. **MI308X spatial roles**：V1864Roles 已按 `selected_r_mask` 赋值，但当前 v4 kernel 的 wave 映射仍是原实现；不能声称 spatial schedule 已完全改变 MI308X 的 dispatch/GEMM 分区。

---

## 5. 结果文件

| 文件 | 内容 | 状态 |
|---|---|---|
| `results/cost_model_validation_3hw.json` | 三硬件真实 routing 聚合回放 | 本次生成 |
| `results/kernel_policy_wiring.json` | 源码 wiring gate | 本次生成 |
| `results/local_validation_summary.json` | 本机编译/运行/PTX parse 证据汇总 | 本次生成 |
| `results/cta0_costmodel_latency_summary.json` | 旧 CTA-0 latency | 归档 measured，未重跑 |
| `results/h20_table4_48cells_ablation_and_costmodel.json` | H20 48-cell ablation | 归档 measured，未重跑 |
| `results/ep8_all16_synced_summary.json` | B300 EP8 trace summary | 归档 measured，未重跑 |
| `results/ep16_baselines_and_roofline.json` | MI308X EP16 baseline/roofline | 归档 measured，未重跑 |

所有实测结果都保留原始 provenance；没有把 simulated / static-check 结果写成 measured。

---

## 6. 引用

如果本仓库对你有帮助，请引用 Weave 论文。DeepGEMM 原始项目与 CUTLASS 的许可证与引用信息见 `LICENSE` 和 `third-party/`。
