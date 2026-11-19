#!/usr/bin/env python3
"""CPU-only source/ownership checks. This is NOT CUDA correctness evidence."""
import argparse
import ast
import collections
import hashlib
import json
import pathlib
import random
import re
import subprocess


BASE = "b64107f2b9599ca76445b7f62eedf66bae1d095b"
DEVICE = "deep_gemm/include/deep_gemm/impls/"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def check_sources(root):
    hashes = {}
    for name in ("sm90_fp8_mega_moe.cuh", "sm90_mega_moe_math.cuh"):
        relative = DEVICE + name
        baseline = git(root, "show", BASE + ":" + relative).replace(b"\r\n", b"\n")
        actual = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        assert actual == baseline, relative + " changed"
        hashes[relative] = hashlib.sha256(actual).hexdigest()
    official = (root / (DEVICE + "sm90_fp8_mega_moe.cuh")).read_text(encoding="utf-8")
    weave = (root / (DEVICE + "weave_sm90_fp8_mega_moe.cuh")).read_text(encoding="utf-8")
    # The complete GEMM + SwiGLU computation must be the official computation.
    for start, end in [
        ("            // GEMM\n", "            // Skip epilogue when block is past valid M"),
        ("            if (is_linear1_phase) {\n                if constexpr (kSwapABActive)",
         "                // L2 epilogue: BF16 cast"),
    ]:
        original_math = official[official.index(start):official.index(end)]
        migrated_math = weave[weave.index(start):weave.index(end)]
        assert migrated_math.replace("weave_sm90_fp8_mega_moe_", "sm90_fp8_mega_moe_") == original_math
    # The generated disabled JIT source and actual launch statements are unchanged.
    host_path = "csrc/jit_kernels/impls/sm90_fp8_mega_moe.hpp"
    old_host = git(root, "show", BASE + ":" + host_path).decode()
    host = (root / host_path).read_text(encoding="utf-8")
    a = old_host.index("        const char* kernel_symbol")
    b = old_host.index("\n    }\n};", a)
    assert old_host[a:b] in host, "Disabled JIT compile/launch block changed"
    assert "655360" not in weave, "Legacy oversized pool leaked into official migration"
    assert weave.count("__syncthreads();") == 1, "Only isolated initialization may use hardware barrier 0"
    assert 'barrier.sync 15, %0;' in weave, "All-CTA joins must not alias dispatch barrier 0"
    assert 'ready = __shfl_sync(0xffffffff, ready, 0)' in weave, "E1 must break warp-uniformly"
    for path in (root / "csrc/jit_kernels").glob("weave_*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # Lightweight delimiter validation, deliberately not described as compilation.
    text = re.sub(r"/\*.*?\*/|//[^\n]*|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", "", weave, flags=re.S)
    stack = []
    for char in text:
        if char in "({[":
            stack.append(char)
        elif char in ")}]":
            assert stack and stack.pop() == {")": "(", "}": "{", "]": "["}[char], "Unbalanced device delimiters"
    assert not stack
    return hashes


def check_ownership():
    rng = random.Random(641)
    cases = 0
    # Exhaustion, idle CTAs, partial M blocks, uneven experts, and head handoff.
    for total in [0, 1, 63, 64, 65, 77, 78, 79, 511, 2048, 20000]:
        claimed = []
        for cta in range(78):
            stream = list(range(cta, total, 78))
            cutoff = rng.randrange(len(stream) + 1)
            # An arbitrary interleaving of helper warps consumes the prefix;
            # dedicated communication warps continue the same ticket sequence.
            head, remainder = stream[:cutoff], stream[cutoff:]
            rng.shuffle(head)
            rng.shuffle(remainder)
            claimed.extend(head + remainder)
        assert sorted(claimed) == list(range(total))
        cases += 1
    # The new communication scatter must cover exactly the official output tile.
    for bm, bn, split_n in [(64, 128, 2), (64, 256, 2), (128, 128, 1), (128, 256, 1)]:
        wm, wn = 64, bn // split_n
        for warps in (2, 4, 6):
            for valid in (0, 1, min(63, bm), 64, bm):
                owned = collections.Counter()
                for warp in range(warps):
                    for lane in range(32):
                        for flat in range(warp * 2 + lane // 16, 2 * wm, warps * 2):
                            wg, row = divmod(flat, wm)
                            m = (wg // split_n) * wm + row
                            if m < valid:
                                n = (wg % split_n) * wn + (lane % 16) * (wn // 16)
                                for col in range(n, n + wn // 16):
                                    owned[m, col] += 1
                expected = {(m, n) for m in range(valid) for n in range(bn)}
                assert set(owned) == expected and all(v == 1 for v in owned.values())
                cases += 1
    # All-warp tail owns every (token, column) once, including empty ranks.
    for warps in (12, 16):
        for tokens in (0, 1, 8, 79, 513):
            for hidden in (256, 3072, 4096, 7168):
                chunks = hidden // 256
                owned = []
                for cta in range(78):
                    for warp in range(warps):
                        owned.extend(range(cta * warps + warp, tokens * chunks, 78 * warps))
                assert sorted(owned) == list(range(tokens * chunks))
                cases += 1
    return cases


def check_buffers():
    # Random producer/consumer timing: a slot cannot be reused before every
    # assigned communication warp finished reading its generation.
    rng = random.Random(452)
    for _ in range(100):
        ready, consumed, payload = [0, 0], [0, 0], [None, None]
        produced = taken = 0
        while taken < 100:
            sequence = produced + 1
            slot = (sequence - 1) & 1
            can_produce = produced < 100 and (sequence <= 2 or consumed[slot] >= sequence - 2)
            next_take = taken + 1
            read_slot = (next_take - 1) & 1
            can_take = ready[read_slot] == next_take
            if can_produce and (not can_take or rng.randrange(2)):
                payload[slot] = sequence
                ready[slot] = sequence
                produced += 1
            elif can_take:
                assert payload[read_slot] == next_take
                consumed[read_slot] = next_take
                taken += 1
            else:
                raise AssertionError("Queue protocol deadlock in abstract ownership model")
    return 100


def check_resources():
    align = lambda x: (x + 1023) // 1024 * 1024
    resources = []
    for comm, producer, math in ((2, 2, 208), (4, 4, 208), (6, 2, 200)):
        prod_regs = 40 if comm == 4 else 48
        registers = comm * 32 * 48 + producer * 32 * prod_regs + 256 * math
        assert registers <= 64512
        for delegated in (False, True):
            # Representative compact H4096/I2048/E256 BM64 BN256 BK128.
            bm, bn, bk, hidden, experts = 64, 256, 128, 4096, 256
            extra = 1024 + (bm * bn * 2 if delegated else 0)
            fixed = align(experts * 4) + align(hidden * comm) + align(bm * bn * 2) + (comm + 16) * 8
            stage = bm * bk + bn * bk + (bk // 64) * ((bm * 4 + 127) // 128 * 128) + 16
            stages = min(4, (232448 - fixed - extra) // stage)
            assert stages >= 2
            allocation = fixed + extra + stages * stage
            assert allocation <= 232448
            # Device appends just five control words after its actual barriers;
            # host allocates the full 1024-byte reserve in both phases.
            assert allocation - 1024 + 5 * 4 <= allocation
            resources.append(dict(comm_warps=comm, producer_warps=producer, math_warps=8,
                                  scatter_comm=delegated, registers=registers,
                                  stages=stages, smem_bytes=allocation))
    return resources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.root.resolve()
    result = dict(status="static_checks_passed", cuda_compiled=False, gpu_measured=False,
                  source_hashes=check_sources(root), ownership_cases=check_ownership(),
                  queue_interleavings=check_buffers(), representative_resources=check_resources())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
