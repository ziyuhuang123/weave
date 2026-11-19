#!/usr/bin/env python3
"""Hardened AMDGPU gfx942 ISA & Post-Link ELF Code-Object Gate (GPT-6.2-Astra Audited)."""
import argparse
import json
import pathlib
import re
import subprocess
import sys


def audit_asm_text(text: str) -> dict:
    instructions = []
    for line in text.splitlines():
        code = line.split(";", 1)[0].split("//", 1)[0].strip()
        if code:
            instructions.append(code)
    scratch_load = sum(1 for line in instructions if re.match(r"^scratch_load\b", line))
    scratch_store = sum(1 for line in instructions if re.match(r"^scratch_store\b", line))
    s_barrier = sum(1 for line in instructions if re.match(r"^s_barrier\b", line))
    s_sleep = sum(1 for line in instructions if re.match(r"^s_sleep\b", line))
    vgpr_spills = [int(v) for v in re.findall(r"\.vgpr_spill_count:\s*(\d+)", text)]
    sgpr_spills = [int(v) for v in re.findall(r"\.sgpr_spill_count:\s*(\d+)", text)]
    private = [int(v) for v in re.findall(r"\.private_segment_fixed_size:\s*(\d+)", text)]
    vgpr_cnt = [int(v) for v in re.findall(r"\.vgpr_count:\s*(\d+)", text)]
    agpr_cnt = [int(v) for v in re.findall(r"\.agpr_count:\s*(\d+)", text)]
    sgpr_cnt = [int(v) for v in re.findall(r"\.sgpr_count:\s*(\d+)", text)]
    lds_bytes = [int(v) for v in re.findall(r"\.group_segment_fixed_size:\s*(\d+)", text)]
    has_meta = bool(vgpr_spills) and bool(private)
    return {
        "has_metadata": has_meta,
        "scratch_load": scratch_load,
        "scratch_store": scratch_store,
        "s_barrier": s_barrier,
        "s_sleep": s_sleep,
        "vgpr_count_max": max(vgpr_cnt, default=0),
        "agpr_count_max": max(agpr_cnt, default=0),
        "sgpr_count_max": max(sgpr_cnt, default=0),
        "group_segment_fixed_size_max": max(lds_bytes, default=0),
        "vgpr_spill_count_max": max(vgpr_spills, default=-1),
        "vgpr_spill_kernels": sum(1 for v in vgpr_spills if v != 0),
        # Note: on gfx942, when private_segment_fixed_size == 0 and scratch_load/store == 0,
        # nonzero sgpr_spill_count means scalar registers spilled into unused lanes of a VGPR
        # via v_writelane_b32 / v_readlane_b32 (0 B memory/scratch traffic).
        "sgpr_to_vgpr_lane_spill_max": max(sgpr_spills, default=0),
        "private_segment_fixed_size_max": max(private, default=-1),
    }


def audit_co_elf(co_path: str, llvm_bin: str = "/opt/rocm-7.2.0/lib/llvm/bin",
                 symbol: str = "v1864_ov1_runtime_roles") -> dict:
    readobj = subprocess.run(
        [f"{llvm_bin}/llvm-readobj", "--notes", co_path],
        capture_output=True, text=True, check=True,
    ).stdout
    objdump = subprocess.run(
        [f"{llvm_bin}/llvm-objdump", "-d", "--mcpu=gfx942", co_path],
        capture_output=True, text=True, check=True,
    ).stdout
    dis_lines = [ln.strip() for ln in objdump.splitlines() if "\t" in ln or "s_" in ln or "scratch_" in ln]
    co_scratch_ld = sum(1 for ln in dis_lines if re.search(r"\bscratch_load", ln))
    co_scratch_st = sum(1 for ln in dis_lines if re.search(r"\bscratch_store", ln))
    co_s_barrier  = sum(1 for ln in dis_lines if re.search(r"\bs_barrier\b", ln))
    co_priv = [int(v) for v in re.findall(r"\.private_segment_fixed_size:\s*(\d+)", readobj)]
    co_vgpr_spill = [int(v) for v in re.findall(r"\.vgpr_spill_count:\s*(\d+)", readobj)]
    co_lds = [int(v) for v in re.findall(r"\.group_segment_fixed_size:\s*(\d+)", readobj)]
    # The Weave entry symbol differs per backend: EP8 uses
    # v1864_ov1_runtime_roles, EP16 uses stage_ep16_turbo_overlap.  The caller
    # selects the expected symbol via --symbol (default keeps EP8 behavior).
    has_sym = symbol in objdump
    return {
        "co_path": co_path,
        "entry_symbol_present": has_sym,
        "co_scratch_load": co_scratch_ld,
        "co_scratch_store": co_scratch_st,
        "co_s_barrier": co_s_barrier,
        "co_vgpr_spill_count_max": max(co_vgpr_spill, default=-1),
        "co_private_segment_fixed_size_max": max(co_priv, default=-1),
        "co_lds_bytes_max": max(co_lds, default=0),
        "co_passed": (
            has_sym
            and bool(co_priv)
            and max(co_priv) == 0
            and max(co_vgpr_spill, default=-1) == 0
            and co_scratch_ld == 0
            and co_scratch_st == 0
            and max(co_lds, default=0) <= 65536
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("asm_path")
    ap.add_argument("--co", default="", help="Optional post-link P.co to verify")
    ap.add_argument("--max-barriers", type=int, default=1)
    ap.add_argument("--symbol", default="v1864_ov1_runtime_roles",
                    help="Expected kernel entry symbol (EP8=v1864_ov1_runtime_roles, "
                         "EP16=stage_ep16_turbo_overlap)")
    ap.add_argument("--advisory", action="store_true")
    args = ap.parse_args()

    rep = audit_asm_text(pathlib.Path(args.asm_path).read_text(encoding="utf-8", errors="replace"))
    rep["path"] = args.asm_path
    asm_ok = (
        rep["has_metadata"]
        and rep["scratch_load"] == 0
        and rep["scratch_store"] == 0
        and rep["vgpr_spill_count_max"] == 0
        and rep["private_segment_fixed_size_max"] == 0
        and rep["group_segment_fixed_size_max"] <= 65536
        and rep["s_barrier"] <= args.max_barriers
    )
    if args.co:
        co_rep = audit_co_elf(args.co, symbol=args.symbol)
        rep["post_link_elf"] = co_rep
        asm_ok = asm_ok and co_rep["co_passed"] and (co_rep["co_s_barrier"] <= args.max_barriers)
    rep["passed"] = bool(asm_ok)
    print(json.dumps(rep, indent=2))
    if not rep["passed"] and not args.advisory:
        sys.exit(1)


if __name__ == "__main__":
    main()
