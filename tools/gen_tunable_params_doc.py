#!/usr/bin/env python3
"""Generate the machine-verified tunable-parameter reference.

The schema lives in deep_gemm/mega/knobs.py.  Each entry names a host parse
needle and a kernel consumption needle; this generator resolves both to the
current line numbers and writes:

  * docs/TUNABLE_PARAMETERS.md        (full exhaustive specification + measured gains)
  * the compact parameter section of README.md and README_zh.md (with per-knob measured gains)

Run from the repository root:

    python tools/gen_tunable_params_doc.py           # regenerate the three files
    python tools/gen_tunable_params_doc.py --check   # exit 1 if any file is stale
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "deep_gemm" / "mega" / "knobs.py"

CATEGORY_ZH = {"Spatial": "空间", "Temporal": "时间", "Implementation": "实现"}

# Full 8-column headers for docs/TUNABLE_PARAMETERS.md
HEADERS_FULL_EN = [
    "Parameter", "Category", "Accepted values (per-hardware where it differs)",
    "Default", "Mechanism", "Measured isolated / paired gain", "Host parse", "Kernel consumption",
]

# Compact 6-column headers for README.md and README_zh.md (saves >70% space while proving every knob's real gain)
HEADERS_COMPACT_EN = [
    "Parameter", "Category", "Values (`0` = default / auto)", "Mechanism",
    "Measured gain (same-batch ABBA)", "Code (`host` / `kernel`)",
]
HEADERS_COMPACT_ZH = [
    "参数", "分类", "核心取值（默认 `0` / `auto`）", "作用机制",
    "实测单参收益（同批对比）", "代码定位（`Host` / `Kernel`）",
]


KNOWN_LIMITATIONS_EN = [
    ("NVIDIA SM103",
     [
         "The internal `WEAVE_LOCAL_SCATTER_DEDUP=1` pin (retired) near-livelocks (`81-363 ms` at `S=1024..8192`); do not enable on SM103.",
         "Manual `BM=128` + `EPI=2` is clamped to `EPI=1` by the host (`BM128 epi2` spills `208 B/304 B`, whereas `BM128 epi1` and `BM80/96 epi2` compile with `0` spill). Legacy `WEAVE_ADAPT_GEMM_CONFIG=1` (`WEAVE_ADAPT_BM=1`, compat-only) previously triggered `misaligned address` at `S >= 4096` when driving device-side per-expert sub-tiling; the current always-on host adaptive selector binds launch-level tuples only.",
         "Internal `WEAVE_REDUCE_WARPS > 0` is slower than `0` (`315-317 us` vs. `295 us` at `S=1024`); `WEAVE_REG_BALANCE=1` only activates at `E/rank > 64`; `WEAVE_CU_WAVE_SPLIT` is parsed as a no-op on SM103; online CTA-0 policy compiles in only with internal `WEAVE_SM0_STREAM_SCHED=1` or `WEAVE_READY_QUEUE=sm0_fifo`.",
     ]),
    ("NVIDIA H20",
     [
         "4-warpgroup Swap-AB launches show `~1e-8` BF16 last-bit repeat variance (`calc_diff ~ 3.55e-4 < 1e-3`, `cos > 0.9996` vs. FP32).",
         "Decoupled split with `WEAVE_GEMM1_CONFIG=64` (`WEAVE_BLOCK_M_L1=64`) is rejected (split L1 spills); use `16`/`32` or fused equal `BM`. Pinned GEMM1/GEMM2 tuples override host adaptive selection.",
         "Internal `WEAVE_REDUCE_WARPS=1/2` is `1.0-2.5%` slower than `0`; internal `WEAVE_COMBINE_SYNC_CHUNK > 0` is not part of any winner. On fused H20, `WEAVE_CU_WAVE_SPLIT` defaults to barrier-free `7-W` (legacy coarse-barrier modes `0..4` are rejected).",
     ]),
]

KNOWN_LIMITATIONS_ZH = [
    ("NVIDIA SM103",
     [
         "内部开关 `WEAVE_LOCAL_SCATTER_DEDUP=1`（已退出公开面）在 `S=1024..8192` 下近似活锁（`81-363 ms`），请勿开启。",
         "手动指定 `BM=128` + `EPI=2` 会被 host 自动钳位为 `EPI=1` 以保证 0 spill（`BM80/96 epi2` 与 `BM128 epi1` 均为 128 寄存器 0 spill）；当前默认常开的 host 自适应仅选择 launch 级元组。",
         "内部 `WEAVE_REDUCE_WARPS > 0` 实测慢于 `0`（`S=1024` 下 `315-317 us` vs `295 us`）；内部 `WEAVE_REG_BALANCE=1` 仅在 `E/rank > 64` 生效；`WEAVE_CU_WAVE_SPLIT` 在 SM103 上无设备端效果；CTA-0 在线层仅在设置内部开关 `WEAVE_SM0_STREAM_SCHED=1` 或 `WEAVE_READY_QUEUE=sm0_fifo` 时编入。",
     ]),
    ("NVIDIA H20",
     [
         "4-warpgroup Swap-AB 在相同输入重复 launch 间存在约 `1e-8` 的 BF16 末位差异（相对 FP32 `calc_diff ~ 3.55e-4 < 1e-3`，`cos > 0.9996`）。",
         "`WEAVE_GEMM1_CONFIG=64`（旧名 `WEAVE_BLOCK_M_L1=64`）与不等 `GEMM2` 的拆分组合因寄存器溢出被 host 拒绝，拆分请用 `16`/`32` 或使用等 `BM` 融合发射；显式固定元组优先于自适应。",
         "内部 `WEAVE_REDUCE_WARPS=1/2` 实测慢 `1.0-2.5%`，内部 `WEAVE_COMBINE_SYNC_CHUNK > 0` 不属于任何胜者；H20 融合路径下 `WEAVE_CU_WAVE_SPLIT` 默认采用无算子间栅栏的 `7-W`（旧粗粒度栅栏模式 `0..4` 已拒绝）。",
     ]),
]


def known_limitations(lang: str, heading_prefix: str) -> str:
    if lang == "zh":
        title = f"{heading_prefix} 已知限制"
        intro = "以下边界均来自实测或代码审计（完整细节见 [docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md)）：\n"
        items = KNOWN_LIMITATIONS_ZH
    else:
        title = f"{heading_prefix} Known limitations"
        intro = "Summary of measured and audited boundaries (full details in [docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md)):\n"
        items = KNOWN_LIMITATIONS_EN
    out = [title + "\n", intro]
    for hw, bullets in items:
        out.append(f"**{hw}**\n")
        out.extend(f"- {b}" for b in bullets)
        out.append("")
    return "\n".join(out)


def load_schema():
    spec = importlib.util.spec_from_file_location("weave_knob_schema", SCHEMA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve(rel: str, needle: str) -> str:
    path = ROOT / rel
    if not path.exists():
        raise SystemExit(f"missing file: {rel}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines, 1):
        if needle in line:
            return f"{rel}:{idx}"
    raise SystemExit(f"needle not found in {rel}: {needle!r}")


def row_full(knob) -> str:
    host = resolve(knob.host_file, knob.host_needle)
    kernel = resolve(knob.kernel_file, knob.kernel_needle)
    cells = [
        f"`{knob.name}`", knob.category, knob.values, knob.default,
        knob.mechanism, knob.gain_en, f"`{host}`", f"`{kernel}`",
    ]
    return "| " + " | ".join(cells) + " |"


def row_compact(knob, lang: str) -> str:
    host = resolve(knob.host_file, knob.host_needle)
    kernel = resolve(knob.kernel_file, knob.kernel_needle)
    if lang == "zh":
        cells = [
            f"`{knob.name}`", CATEGORY_ZH[knob.category], knob.compact_values_zh,
            knob.compact_mech_zh, knob.gain_zh, f"`{host}` / `{kernel}`",
        ]
    else:
        cells = [
            f"`{knob.name}`", knob.category, knob.compact_values_en,
            knob.compact_mech_en, knob.gain_en, f"`{host}` / `{kernel}`",
        ]
    return "| " + " | ".join(cells) + " |"


def table_full(knobs) -> str:
    out = [
        "| " + " | ".join(HEADERS_FULL_EN) + " |",
        "|" + "|".join(["---"] * len(HEADERS_FULL_EN)) + "|",
    ]
    for knob in knobs:
        out.append(row_full(knob))
    return "\n".join(out)


def table_compact(knobs, lang: str) -> str:
    headers = HEADERS_COMPACT_ZH if lang == "zh" else HEADERS_COMPACT_EN
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for knob in knobs:
        out.append(row_compact(knob, lang))
    return "\n".join(out)


def retired_and_alias_note(schema, lang: str, heading: str) -> str:
    """Prose (never a table row) listing the retired internal knobs and legacy aliases."""
    retired = ", ".join(f"`{n}`" for n in schema.RETIRED_INTERNAL)
    aliases = [(old, new) for old, new in schema.LEGACY_ALIASES.items()]
    counts = schema.EXPECTED_COUNTS
    if lang == "zh":
        lines = [
            f"{heading} 已退出公开参数面的内部开关与旧名别名\n",
            f"公开参数面严格固定为上述 **14 个 knob**（空间 {counts['Spatial']} / 时间 {counts['Temporal']} / 实现 {counts['Implementation']}），且**全部 14 个参数均在对应负载形态上取得经同批 ABBA 验证的实测正收益**。"
            f"原 21-knob 中的 6 个开关（{retired}）已退为内部编译期宏（默认 `0`）与诊断开关；"
            "`WEAVE_ADAPT_GEMM_CONFIG`（旧别名 `WEAVE_ADAPT_BM`）亦不再公开——当 `WEAVE_GEMM1_CONFIG=0` 且 `WEAVE_GEMM2_CONFIG=0` 时 host 形状自适应元组选择默认常开，`=0` 仅作内部固定元组诊断覆盖。透明兼容旧名别名：\n",
        ]
        lines += [f"- 旧名 `{old}` -> `{new}`" for old, new in aliases]
        lines.append("")
    else:
        lines = [
            f"{heading} Retired internal knobs and legacy aliases\n",
            f"The public surface is strictly the **14 knobs** above (Spatial {counts['Spatial']} / Temporal {counts['Temporal']} / Implementation {counts['Implementation']}), and **every one of the 14 knobs delivers a verified positive speedup** in same-batch ABBA sweeps on its target regime. "
            f"Six former knobs ({retired}) are retired to internal compile-time macros (default `0`) and host diagnostics; "
            "`WEAVE_ADAPT_GEMM_CONFIG` (legacy alias `WEAVE_ADAPT_BM`) is likewise non-public—host shape-adaptive GEMM tuple selection is ALWAYS ON when `WEAVE_GEMM1_CONFIG=0` and `WEAVE_GEMM2_CONFIG=0` (`=0` stays accepted only as an internal canonical-tuple diagnostic). Transparent legacy aliases:\n",
        ]
        lines += [f"- legacy `{old}` -> `{new}`" for old, new in aliases]
        lines.append("")
    return "\n".join(lines)


def parameter_section(schema, lang: str) -> str:
    groups = {c: [k for k in schema.KNOBS if k.category == c] for c in schema.CATEGORIES}
    if lang == "zh":
        intro = (
            "公开参数面严格为 **14 个 knob、3 个类别**（空间 5 / 时间 3 / 实现 6）。下表同时给出每个参数的**同批实测单参收益**（台账：`results/ep16_mi308x_v70_16gpu_champion.json`、`results/h20_v12_comm_bound_sweep.json`、`results/sm103_v3_clean_config_sweep.json`、`results/sm103_qwen38max_full_sweep.json`）与 `tests/verify_tunable_params_doc.py` 自动校验的代码行号；完整逐硬件语法见 [docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md)，调参方法见 [docs/AUTOTUNE_AND_TUNING_GUIDE.md](docs/AUTOTUNE_AND_TUNING_GUIDE.md)。\n"
        )
        headings = {
            "Spatial": "### 空间（Spatial）",
            "Temporal": "### 时间（Temporal）",
            "Implementation": "### 实现（Implementation）",
        }
    else:
        intro = (
            "The public surface is strictly **14 knobs in 3 categories** (Spatial 5 / Temporal 3 / Implementation 6). Every row reports **verified same-batch measured gains** (ledgers: `results/ep16_mi308x_v70_16gpu_champion.json`, `results/h20_v12_comm_bound_sweep.json`, `results/sm103_v3_clean_config_sweep.json`, `results/sm103_qwen38max_full_sweep.json`) alongside machine-verified `file:line` citations (`tests/verify_tunable_params_doc.py`); full per-hardware value grammar is in [docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md) and tuning workflow in [docs/AUTOTUNE_AND_TUNING_GUIDE.md](docs/AUTOTUNE_AND_TUNING_GUIDE.md).\n"
        )
        headings = {
            "Spatial": "### Spatial",
            "Temporal": "### Temporal",
            "Implementation": "### Implementation",
        }
    parts = [intro]
    for category in schema.CATEGORIES:
        parts.append(headings[category] + "\n")
        parts.append(table_compact(groups[category], lang))
        parts.append("")
    parts.append(retired_and_alias_note(schema, lang, "###"))
    if lang == "zh":
        parts.append(
            "`num_shared_experts` 为 Python/C++ API 参数（默认 `0`，显式非零时才分配共享专家缓冲区）。\n"
        )
    else:
        parts.append(
            "`num_shared_experts` is a Python/C++ API argument (default `0`; shared-expert buffers are allocated only when non-zero).\n"
        )
    parts.append(known_limitations(lang, "###"))
    return "\n".join(parts)


def render_docs(schema) -> str:
    groups = {c: [k for k in schema.KNOBS if k.category == c] for c in schema.CATEGORIES}
    parts = [
        "# Weave Tunable Parameters, Measured Gains, and Exact Code Locations\n",
        "Weave exposes one unified knob space across SM103, H20 and MI308X: "
        "**14 public knobs in 3 categories** (Spatial 5, Temporal 3, Implementation 6). "
        "Every parameter below changes real host dispatch or kernel execution and delivers a verified positive speedup on its target workload regime; "
        "pipeline depth is derived at launch time from the tile shape and remaining SMEM/LDS budget.\n",
        "Each row carries its measured gain, host parse location, and kernel consumption location. "
        "`tests/verify_tunable_params_doc.py` resolves every citation against the working tree. "
        "For turnkey compile-time calibration and zero-env runtime selection, see [AUTOTUNE_AND_TUNING_GUIDE.md](AUTOTUNE_AND_TUNING_GUIDE.md).\n",
    ]
    for title, category in (
        ("1. Spatial dimension", "Spatial"),
        ("2. Temporal dimension", "Temporal"),
        ("3. Implementation and prior art", "Implementation"),
    ):
        parts.append(f"## {title}\n")
        parts.append(table_full(groups[category]))
        parts.append("")
        if category == "Implementation":
            parts.append(retired_and_alias_note(schema, "en", "###"))
    parts.append("## 4. API argument: shared experts\n")
    parts.append(
        "`num_shared_experts` is a Python/C++ API argument, not an environment "
        "knob. Every MegaMoE entry point defaults it to `0`; the shared-expert buffers are "
        "allocated only when it is explicitly non-zero (the gate API `bf16_mega_gate` "
        "takes it explicitly together with `use_shared_as_routed`).\n"
    )
    parts.append("## 5. Built-in mechanisms & offline-coupled profile tuples (not standalone knobs)\n")
    parts.append(
        "Once the GEMM tuple is selected for GEMM1 or GEMM2 (either by the host-launch "
        "always-on shape-adaptive pick (`WEAVE_GEMM1_CONFIG=0` / `WEAVE_GEMM2_CONFIG=0`), or by pinning "
        "`WEAVE_GEMM1_CONFIG` / `WEAVE_GEMM2_CONFIG`), its entire offline-calibrated "
        "single-operator tuple—`(blockM, blockN, blockK, mainloop_stages)` and `(epi_M, epi_N, epi_K, "
        "epi_stages)` (including `1.75/2.0`-stage TMEM accumulation and 2-stage "
        "register epilogue prefetch)—is locked synchronously with zero user sub-knobs. "
        "On SM103 a manual `BM=128` + `EPI=2` request is clamped to `EPI=1` by the host to guarantee 0 spill. "
        "Barrier-free fine-grained atomic cross-stage handoffs, cooperative all-idle-warp tail "
        "Combine reduction, and modulo intermediate-activation ring-buffer reuse "
        "(8-9x HBM compression) are always-on built-in mechanisms.\n"
    )
    parts.append("## 6. Architecture coverage\n")
    parts.append(
        "| Parameter group | SM103 fp8xfp4 | SM103 fp4xfp4 | H20 (SM90) | MI308X EP8 | MI308X EP16 (V70) |\n"
        "|---|---|---|---|---|---|\n"
        "| Spatial: SM/CU subsets (`DISPATCH_ACTIVE_SMS`, `SCATTER_ACTIVE_SMS`) | yes | n/a | yes | yes | yes |\n"
        "| Spatial: per-SM/CU warp ratios (`DISPATCH_WARPS`, `SCATTER_WARPS`) | yes | n/a | yes | n/a | yes |\n"
        "| Spatial: `CU_WAVE_SPLIT` | accepted, no-op | n/a | yes: `7-W` / `5-E` / `6-E` (0 `grid_sync`) | yes: `0-8` / `4-4` / `8-0` | yes: `0-8` / `4-4` / `8-0` |\n"
        "| Temporal: `V_HEAD`, `READY_GROUP_SIZE`, `SCATTER_OWNER` | yes | overlap | yes | yes (`READY_GROUP_SIZE` n/a) | yes |\n"
        "| Implementation: `PRECISION_MODE`, `GEMM1_CONFIG`, `GEMM2_CONFIG` | yes (whole-tuple; `BM=128+EPI=2` clamped to `EPI=1`) | fixed | yes (bare `BM`; decoupled `GEMM1=64` rejected) | yes | yes |\n"
        "| Implementation: `REG_PULL_BYTES`, `DEDUP_MODE`, `SCATTER_STAGING` | yes | n/a | yes | `DEDUP_MODE` | yes (`REG_PULL_BYTES=8192`) |\n"
    )
    parts.append("## 7. Two-Stage Autotune + Online CTA-0 Routing-Adaptive Selection\n")
    parts.append(
        "Weave's turnkey `python -m deep_gemm.mega.autotune` calibrates a per-SKU JSON profile once "
        "(0-spill check + per-`S` bucket ABBA + 4-rule break-even thresholds), and `weave_mega_moe(...)` "
        "selects optimal parameters at runtime with zero `WEAVE_*` environment variables required "
        "(see [AUTOTUNE_AND_TUNING_GUIDE.md](AUTOTUNE_AND_TUNING_GUIDE.md) and "
        "[WEAVE_TWO_LEVEL_FRAMEWORK.md](WEAVE_TWO_LEVEL_FRAMEWORK.md)).\n"
    )
    parts.append(known_limitations("en", "## 8."))
    return "\n".join(parts)


def render_readme(path: pathlib.Path, lang: str, body: str) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if (stripped.startswith("## ") and "WEAVE_*" in stripped
                and "SM103" not in stripped and "H20" not in stripped
                and "AMD" not in stripped):
            start = i
            break
    if start is None:
        raise SystemExit(f"unified parameter heading not found in {path}")
    end = None
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped.startswith("## ") and "WEAVE_*" not in stripped:
            end = j
            break
    if end is None:
        raise SystemExit(f"next section not found in {path}")
    heading_line = lines[start].rstrip("\n")
    return "".join(lines[:start]) + heading_line + "\n\n" + body.strip() + "\n\n" + "".join(lines[end:])


def render_all(schema) -> dict:
    """Return {path: text} for every generated file (nothing is written here)."""
    return {
        ROOT / "docs" / "TUNABLE_PARAMETERS.md": render_docs(schema),
        ROOT / "README.md": render_readme(
            ROOT / "README.md", "en", parameter_section(schema, "en")),
        ROOT / "README_zh.md": render_readme(
            ROOT / "README_zh.md", "zh", parameter_section(schema, "zh")),
    }


def _normalized(text: str) -> str:
    return text.replace("\r\n", "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="do not write; exit 1 when a generated file is stale")
    args = parser.parse_args(argv)
    schema = load_schema()
    outputs = render_all(schema)
    if args.check:
        stale = []
        for path, text in outputs.items():
            on_disk = _normalized(path.read_bytes().decode("utf-8")) if path.exists() else None
            if on_disk != _normalized(text):
                stale.append(path.relative_to(ROOT).as_posix())
        if stale:
            print("STALE generated file(s), run `python tools/gen_tunable_params_doc.py`: "
                  + ", ".join(stale))
            return 1
        print(f"generated files are up to date: {len(schema.KNOBS)} parameter rows "
              f"in {len(schema.CATEGORIES)} categories")
        return 0
    for path, text in outputs.items():
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print(f"generated {len(schema.KNOBS)} parameter rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
