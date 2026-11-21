# 可复现 Weave 调优预设（空间 / 时间 / 实现）

本目录提供 NVIDIA SM103、NVIDIA H20 与 AMD Instinct MI308X（单机 `EP8` 与双机 `EP16`）的可执行预设脚本。所有脚本**严格只使用** [../docs/TUNABLE_PARAMETERS.md](../docs/TUNABLE_PARAMETERS.md) 中定义的 21 项统一 `WEAVE_*` 参数，不包含任何旧版内部配置串或别名。

## 文件列表

- `sm103_qwen38max_ep8_s1024.sh` — NVIDIA SM103（148-SM），`fp8xfp4` / `fp4xfp4`
- `h20_qwen38max_ep8_s1024.sh` — NVIDIA H20（78-SM），`fp8xfp8`
- `mi308x_qwen38max_ep8_s1024.sh` — AMD MI308X（80-CU），单机 EP8
- `mi308x_qwen38max_ep16_s128.sh` — AMD MI308X（80-CU），双机 16 卡 EP16
- `profile_and_render_timeline.py` — 三轴 `WEAVE_*` 参数表打印与 WeaveProfiler 时间线渲染驱动

## 三轴职责划分

| 维度 | 职责边界 | 统一 `WEAVE_*` 参数 |
|---|---|---|
| **空间维度（Spatial）** | 物理 SM/CU 级与单 SM/CU 内 Warp/Wave 级的硬件资源配比 | `WEAVE_DISPATCH_ACTIVE_SMS`、`WEAVE_SCATTER_ACTIVE_SMS`、`WEAVE_SM0_STREAM_SCHED`、`WEAVE_DISPATCH_WARPS`、`WEAVE_SCATTER_WARPS`、`WEAVE_REDUCE_WARPS`、`WEAVE_CU_WAVE_SPLIT`、`WEAVE_REG_BALANCE` |
| **时间维度（Temporal）** | 跨阶段角色切换、全局任务派发、通算重叠与同步粒度 | `WEAVE_V_HEAD`、`WEAVE_READY_GROUP_SIZE`、`WEAVE_READY_QUEUE`、`WEAVE_SCATTER_OWNER`、`WEAVE_COMBINE_SYNC_CHUNK` |
| **单算子与实现（Implementation）** | GEMM1、GEMM2、Dispatch、Scatter 四算子各自独立的单算子参数选择（选定 `blockM` 同步锁定 `blockN/K`、`mainloop_stages`、`epi_M/N/K` 与 `epi_stages`） | `WEAVE_PRECISION_MODE`、`WEAVE_ADAPT_BM`、`WEAVE_BLOCK_M_L1`、`WEAVE_BLOCK_M_L2`、`WEAVE_REG_PULL_BYTES`、`WEAVE_DEDUP_MODE`、`WEAVE_SCATTER_STAGING`、`WEAVE_LOCAL_SCATTER_DEDUP` |

## 快速查看与运行

```bash
python examples/profile_and_render_timeline.py --preset sm103 --no-render
python examples/profile_and_render_timeline.py --preset h20 --no-render
python examples/profile_and_render_timeline.py --preset mi308x_ep8 --no-render
```
