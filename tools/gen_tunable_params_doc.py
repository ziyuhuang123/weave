#!/usr/bin/env python3
"""Generate the machine-verified tunable-parameter reference.

The schema lives in deep_gemm/mega/knobs.py.  Each entry names a host parse
needle and a kernel consumption needle; this generator resolves both to the
current line numbers and writes:

  * docs/TUNABLE_PARAMETERS.md
  * the parameter section of README.md and README_zh.md

Run from the repository root:

    python tools/gen_tunable_params_doc.py
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "deep_gemm" / "mega" / "knobs.py"

CATEGORY_ZH = {"Spatial": "空间", "Temporal": "时间", "Implementation": "实现"}
HEADERS_EN = ["Parameter", "Category", "Values (every accepted value is named)",
              "Default", "Mechanism", "Host parse", "Kernel consumption"]
HEADERS_ZH = ["参数", "分类", "取值（每个接受值均已命名）", "默认值",
              "机制", "Host 解析", "Kernel 消费"]


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


def row(knob, lang: str) -> str:
    host = resolve(knob.host_file, knob.host_needle)
    kernel = resolve(knob.kernel_file, knob.kernel_needle)
    category = CATEGORY_ZH[knob.category] if lang == "zh" else knob.category
    cells = [f"`{knob.name}`", category, knob.values, knob.default,
             knob.mechanism, f"`{host}`", f"`{kernel}`"]
    return "| " + " | ".join(cells) + " |"


def table(knobs, lang: str) -> str:
    headers = HEADERS_ZH if lang == "zh" else HEADERS_EN
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for knob in knobs:
        out.append(row(knob, lang))
    return "\n".join(out)


def parameter_section(schema, lang: str) -> str:
    groups = {c: [k for k in schema.KNOBS if k.category == c]
              for c in ("Spatial", "Temporal", "Implementation")}
    if lang == "zh":
        intro = (
            "本参数面严格按 **空间 / 时间 / 实现** 三轴组织。每行同时给出 host 解析位置与 "
            "kernel 消费位置；`tests/verify_tunable_params_doc.py` 会逐条解析 "
            "`file:line` 引用。完整取值语义见 "
            "[docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md)。\n"
        )
        headings = {"Spatial": "### 空间（Spatial）",
                    "Temporal": "### 时间（Temporal）",
                    "Implementation": "### 实现（Implementation）"}
    else:
        intro = (
            "The parameter surface is organized strictly as **Spatial / Temporal / "
            "Implementation**. Every row names both its host parse and its kernel "
            "consumption; `tests/verify_tunable_params_doc.py` resolves each "
            "`file:line` citation. The full value semantics are in "
            "[docs/TUNABLE_PARAMETERS.md](docs/TUNABLE_PARAMETERS.md).\n"
        )
        headings = {"Spatial": "### Spatial",
                    "Temporal": "### Temporal",
                    "Implementation": "### Implementation"}
    parts = [intro]
    for category in ("Spatial", "Temporal", "Implementation"):
        parts.append(headings[category] + "\n")
        parts.append(table(groups[category], lang))
        parts.append("")
    if lang == "zh":
        parts.append(
            "`num_shared_experts` 是 Python/C++ API 参数而非环境变量 knob。所有入口默认取 "
            "`0`，只有显式传入非零值时才分配 shared-expert 缓冲区。\n"
        )
    else:
        parts.append(
            "`num_shared_experts` is a Python/C++ API argument rather than an "
            "environment knob. Its default is `0` on every entry point; shared-expert "
            "buffers are allocated only when it is explicitly non-zero.\n"
        )
    return "\n".join(parts)


def write_docs(schema) -> None:
    groups = {c: [k for k in schema.KNOBS if k.category == c]
              for c in ("Spatial", "Temporal", "Implementation")}
    parts = [
        "# Weave Tunable Parameters and Exact Code Locations\n",
        "Weave exposes one unified knob space across SM103, H20 and MI308X. "
        "Every parameter below changes real host dispatch or kernel execution; "
        "the 24 historical fields that had no reader in the runtime policy ABI "
        "are removed, and the historical H20 ordering switch is no longer part of the public "
        "surface. Pipeline depth is derived at launch time from the tile shape "
        "and the remaining SMEM budget, not frozen by a public knob.\n",
        "The document has three sections: **Spatial**, **Temporal** and "
        "**Implementation**. Each row carries the host parse location and the "
        "kernel consumption location. `tests/verify_tunable_params_doc.py` "
        "resolves every citation against the working tree.\n",
    ]
    for title, category in (("1. Spatial dimension", "Spatial"),
                            ("2. Temporal dimension", "Temporal"),
                            ("3. Implementation and prior art", "Implementation")):
        parts.append(f"## {title}\n")
        parts.append(table(groups[category], "en"))
        parts.append("")
    parts.append("## 4. API argument: shared experts\n")
    parts.append(
        "`num_shared_experts` is a Python/C++ API argument, not an environment "
        "knob. Every entry point defaults it to `0`; the shared-expert buffers are "
        "allocated only when it is explicitly non-zero. The unified surface keeps "
        "`0` as the default so routed-only workloads do not pay the shared-expert "
        "workspace or scheduling cost.\n"
    )
    parts.append("## 5. Values derived at launch time (not knobs)\n")
    parts.append(
        "The GEMM stage count, accumulator ring depth, TMA-store ring depth and "
        "scheduler look-ahead are computed from the live tile shape and the SMEM "
        "budget in `csrc/jit_kernels/heuristics/mega_moe.hpp`. The per-expert "
        "BM ladder is selected inside the scheduler when `WEAVE_ADAPT_BM=1`. "
        "These are derivations, not configuration inputs.\n"
    )
    parts.append("## 6. Architecture coverage\n")
    parts.append(
        "| Parameter group | SM103 fp8xfp4 | SM103 fp4xfp4 | H20 | MI308X EP8 | MI308X EP16 |\n"
        "|---|---|---|---|---|---|\n"
        "| Spatial: dispatch warps, scatter SMs, locality domains, SM 0 scheduler | yes | n/a | yes | yes | yes |\n"
        "| Temporal: L2 K-block dependency, ring blocks, early reduce, TMEM pipeline, head pull | yes | ring only | yes | yes | yes |\n"
        "| Implementation: per-expert BM, asymmetric store, local dedup, precision | yes | fixed | yes | yes | yes |\n"
    )
    (ROOT / "docs" / "TUNABLE_PARAMETERS.md").write_text("\n".join(parts), encoding="utf-8")


def replace_readme_section(path: pathlib.Path, lang: str, body: str) -> None:
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
    new_text = "".join(lines[:start]) + heading_line + "\n\n" + body.strip() + "\n\n" + "".join(lines[end:])
    path.write_text(new_text, encoding="utf-8")


def main() -> None:
    schema = load_schema()
    write_docs(schema)
    replace_readme_section(
        ROOT / "README.md",
        "en",
        parameter_section(schema, "en"),
    )
    replace_readme_section(
        ROOT / "README_zh.md",
        "zh",
        parameter_section(schema, "zh"),
    )
    print(f"generated {len(schema.KNOBS)} parameter rows")


if __name__ == "__main__":
    main()
