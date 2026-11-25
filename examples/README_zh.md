# 可复现 Weave 配置预设（空间 / 时间 / 实现）

本目录提供 NVIDIA SM103、NVIDIA H20 与 AMD Instinct MI308X（单机 `EP8` 与双机 `EP16`）的可执行预设脚本与 WeaveProfiler 时间线渲染驱动，统一使用 [../docs/TUNABLE_PARAMETERS.md](../docs/TUNABLE_PARAMETERS.md) 中定义的 **14 个 `WEAVE_*` 参数**。

## 文件列表

- `sm103_qwen38max_ep8_s1024.sh` — NVIDIA SM103（148-SM），`fp8xfp4` / `fp4xfp4`
- `h20_qwen38max_ep8_s1024.sh` — NVIDIA H20（78-SM），`fp8xfp8`
- `mi308x_qwen38max_ep8_s1024.sh` — AMD MI308X（80-CU），单机 EP8
- `mi308x_qwen38max_ep16_s128.sh` — AMD MI308X（80-CU），双机 16 卡 EP16
- `profile_and_render_timeline.py` — 三维 `WEAVE_*` 参数表打印与 WeaveProfiler 时间线渲染驱动

## 三维职责划分

| 维度 | 职责边界 | 统一 `WEAVE_*` 参数 |
|---|---|---|
| **空间维度（Spatial，5 个）** | 跨 SM/CU 异构配比、独立 Dispatch/Scatter 活跃单元子集与单 SM/CU 内 Warp/Wave 配比 | `WEAVE_DISPATCH_ACTIVE_SMS`、`WEAVE_SCATTER_ACTIVE_SMS`、`WEAVE_DISPATCH_WARPS`、`WEAVE_SCATTER_WARPS`、`WEAVE_CU_WAVE_SPLIT` |
| **时间维度（Temporal，3 个）** | 首部协助拉取、细粒度就绪起算与中部异步 Scatter 角色切换 | `WEAVE_V_HEAD`、`WEAVE_READY_GROUP_SIZE`、`WEAVE_SCATTER_OWNER` |
| **单算子实现（Implementation，6 个）** | GEMM1、GEMM2、Dispatch、Scatter 四算子各自独立的分块、多级流水与数据通路配置 | `WEAVE_PRECISION_MODE`、`WEAVE_GEMM1_CONFIG`、`WEAVE_GEMM2_CONFIG`、`WEAVE_REG_PULL_BYTES`、`WEAVE_DEDUP_MODE`、`WEAVE_SCATTER_STAGING` |

## 快速查看与运行

```bash
python examples/profile_and_render_timeline.py --preset sm103 --no-render
python examples/profile_and_render_timeline.py --preset h20 --no-render
python examples/profile_and_render_timeline.py --preset mi308x_ep8 --no-render
```
