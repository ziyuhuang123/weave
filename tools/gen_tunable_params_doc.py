#!/usr/bin/env python3
"""Generate the machine-verified tunable-parameter reference.

The schema lives in deep_gemm/mega/knobs.py. Each entry names a host parse
needle and a kernel consumption needle; this generator resolves both to the
current line numbers and writes:

  * docs/TUNABLE_PARAMETERS.md
  * the compact parameter section of README.md and README_zh.md

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

HEADERS_FULL_EN = [
    "Parameter", "Category", "Accepted values",
    "Default", "Mechanism", "Measured single-knob gain", "Host parse", "Kernel consumption",
]

HEADERS_COMPACT_EN = [
    "Parameter", "Category", "Values (`0` = auto)", "Mechanism",
    "Measured gain", "Code (`host` / `kernel`)",
]
HEADERS_COMPACT_ZH = [
    "参数", "分类", "取值（`0` = 自动）", "作用机制",
    "实测单参收益", "代码位置（`Host` / `Kernel`）",
]


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


def parameter_section(schema, lang: str) -> str:
    groups = {c: [k for k in schema.KNOBS if k.category == c] for c in schema.CATEGORIES}
    if lang == "zh":
        intro = (
            "Weave 通过统一的 **14 个参数（空间 5 / 时间 3 / 实现 6）** 控制跨卡、跨 SM/CU、单 SM/CU 内配比，以及时间角色切换与四算子独立实现。未显式设置时均由离线校准与在线代价模型自动选择；完整说明见 [docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md) 与 [docs/AUTOTUNE_AND_TUNING_GUIDE.md](docs/AUTOTUNE_AND_TUNING_GUIDE.md)。\n"
        )
        headings = {
            "Spatial": "### 空间维度（Spatial）",
            "Temporal": "### 时间维度（Temporal）",
            "Implementation": "### 单算子实现维度（Implementation）",
        }
    else:
        intro = (
            "Weave exposes **14 unified parameters across 3 axes** (Spatial 5 / Temporal 3 / Implementation 6) covering cross-SM/CU and intra-SM/CU partitioning, dynamic temporal role transitions, and independent per-operator configurations. Unset parameters (`0` / `auto`) are selected automatically from the calibrated hardware profile and live routing statistics; see [docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md) and [docs/AUTOTUNE_AND_TUNING_GUIDE.md](docs/AUTOTUNE_AND_TUNING_GUIDE.md).\n"
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
    return "\n".join(parts)


def render_docs(schema) -> str:
    groups = {c: [k for k in schema.KNOBS if k.category == c] for c in schema.CATEGORIES}
    parts = [
        "# Weave Tunable Parameters Reference\n",
        "Weave exposes one unified **14-knob surface across 3 orthogonal axes** "
        "(Spatial 5, Temporal 3, Implementation 6) shared across NVIDIA SM100/SM103, NVIDIA H20, and AMD MI308X. "
        "When left at `0` (`auto`), parameters are chosen automatically by the offline hardware profile and online routing-adaptive cost model "
        "(see [AUTOTUNE_AND_TUNING_GUIDE.md](AUTOTUNE_AND_TUNING_GUIDE.md)).\n",
    ]
    for title, category in (
        ("1. Spatial scheduling (5 knobs)", "Spatial"),
        ("2. Temporal scheduling (3 knobs)", "Temporal"),
        ("3. Independent operator implementation (6 knobs)", "Implementation"),
    ):
        parts.append(f"## {title}\n")
        parts.append(table_full(groups[category]))
        parts.append("")
    parts.append("## 4. Coupled per-operator pipelines and built-in handoffs\n")
    parts.append(
        "Selecting `WEAVE_GEMM1_CONFIG` or `WEAVE_GEMM2_CONFIG` (either automatically from `(M_expected, N, K)` at `0`, "
        "or via an explicit tuple `BM[:STORE_BM[:BK[:STAGES[:EPI]]]]`) locks the paired `(blockM, blockN, blockK, mainloop_stages)` "
        "and multi-stage epilogue register pipeline (`EPI = 1..4`) for each GEMM independently. "
        "Cross-stage tile arrival counters (`Dispatch -> GEMM1`), intermediate tile-ready bitmaps/rings (`GEMM1 -> GEMM2`), "
        "multi-stage `reg` / `smem` / `hbm` scatter handoffs, and per-token (`k/k`) acquire-ready all-warp Combine reduction "
        "are always enabled.\n"
    )
    parts.append("## 5. Hardware support summary\n")
    parts.append(
        "| Parameter group | SM103 (`fp8xfp4` / `fp4xfp4`) | H20 (`fp8`) | MI308X (`fp8`, EP8 & EP16) |\n"
        "|---|---|---|---|\n"
        "| Spatial: cross-SM/CU subsets (`DISPATCH_ACTIVE_SMS`, `SCATTER_ACTIVE_SMS`) | yes | yes | yes |\n"
        "| Spatial: intra-SM/CU warp/wave ratios (`DISPATCH_WARPS`, `SCATTER_WARPS`) | yes | yes | yes |\n"
        "| Spatial: `CU_WAVE_SPLIT` | accepted (no-op) | yes (expert-stream lead) | yes (`0-8` / `4-4` / `8-0`) |\n"
        "| Temporal: `V_HEAD`, `READY_GROUP_SIZE`, `SCATTER_OWNER` | yes | yes | yes |\n"
        "| Implementation: `PRECISION_MODE`, `GEMM1_CONFIG`, `GEMM2_CONFIG` | yes (full tuple, `EPI=1..4`) | yes (independent `BM`) | yes (independent `BM`) |\n"
        "| Implementation: `REG_PULL_BYTES`, `DEDUP_MODE`, `SCATTER_STAGING` | yes (`reg` / `smem` / `hbm`) | yes (`reg` / `smem` / `hbm`) | yes (`REG_PULL_BYTES=8192`) |\n"
    )
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
