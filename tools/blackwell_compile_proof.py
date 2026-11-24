#!/usr/bin/env python3
"""Compile-only proof that the Blackwell MegaMoE kernels build with 0 local-memory spill
for BOTH sm_100a (B200) and sm_103a (B300), on one GPU, without launching any kernel.

How it works
  * Every worker process sets DG_JIT_COMPILE_ONLY=1, DG_JIT_TARGET_ARCH, DG_JIT_TARGET_NUM_SMS,
    DG_JIT_PTXAS_VERBOSE=1 and a private DG_JIT_CACHE_DIR before importing deep_gemm.
    The JIT then builds the cubin for the requested target and returns before loading/launching.
  * A worker handles one (arch, num_sms, shape, precision) group: it allocates the symmetric
    buffer (EP4 sizes, one local allocation, buffer_ptrs=[ptr]*4, rank=0) and the weight / SF
    tensors once (torch.empty: contents are irrelevant, nothing launches), then compiles its
    cells one by one, switching the WEAVE_* environment per cell.
  * The driver fans groups out into chunks over a subprocess pool and aggregates the ptxas logs.

Cell status
  ok                 compiled, 0 B spill stores and 0 B spill loads (a non-zero 'stack frame' from local
                     arrays is NOT a spill; it is reported separately via stack_frame_bytes /
                     local_memory_warning and counted as stack_frame_nonzero in the summary)
  spill              compiled but ptxas reports spill stores/loads > 0
  compile_failed     nvcc / ptxas / static_assert failure
  rejected_by_host   a host-side DG_HOST_ASSERT / ValueError refused the env combination
                     (before any nvcc run) -> not counted as a compile failure

Usage
  python tools/blackwell_compile_proof.py --out results/blackwell_sm100a_sm103a_compile_proof.json \
      --workdir <dir> --jobs 48 [--coverage full|smoke]
"""
import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

ARCHS = ("100a", "103a")
SHAPES = {
    "qwen3.8":      dict(E=512, topk=10, H=8192, I=2048),
    "qwen3.5-35b":  dict(E=256, topk=8, H=2048, I=512),
}
PRECISIONS = ("fp8xfp4", "fp4xfp4")
GEMM_TUPLES = [
    ("16:8:256", "16:8:256"), ("32:16:128", "32:16:128"), ("64:16:128", "64:16:128"),
    ("80:8:128:0:2", "80:8:128:0:2"), ("80:8", "80:8"), ("96:16", "96:8"),
    ("128:32:128:0:1", "128:32:128:0:1"), ("192:32", "192:32"), ("240:24", "240:8"),
    ("0", "0"),
]
REG_PULL = (0, 1024, 2048)
SCATTER = (("epilogue_only", "reg_2stage"), ("comm_only", "smem"))
DEDUP = (0, 1)
NUM_RANKS = 4
NUM_MAX_TOKENS = 4096
NUM_TOKENS = 1024
DEFAULT_CO = dict(reg_pull=0, scatter=0, dedup=0)  # indices into the tables above


# ----------------------------------------------------------------------------- cell matrix
def make_cell(arch, sms, shape, prec, gemm, reg_pull, scatter, dedup):
    return dict(arch=arch, num_sms=sms, shape=shape, precision=prec,
                gemm1=GEMM_TUPLES[gemm][0], gemm2=GEMM_TUPLES[gemm][1],
                reg_pull_bytes=REG_PULL[reg_pull],
                scatter_owner=SCATTER[scatter][0], scatter_staging=SCATTER[scatter][1],
                dedup_mode=DEDUP[dedup])


def build_matrix(coverage):
    cells = []
    full_co = list(itertools.product(range(len(REG_PULL)), range(len(SCATTER)), range(len(DEDUP))))
    one_at_a_time = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (0, 0, 1)]
    if coverage == "smoke":
        for arch in ARCHS:
            for prec in PRECISIONS:
                for g in (9, 5):
                    cells.append(make_cell(arch, 148, "qwen3.5-35b", prec, g, 0, 0, 0))
        return cells
    for arch in ARCHS:
        for prec in PRECISIONS:
            for sms in (148, 144):
                for g in range(len(GEMM_TUPLES)):
                    for co in full_co:                                       # full product, Qwen3.8
                        cells.append(make_cell(arch, sms, "qwen3.8", prec, g, *co))
                    for co in one_at_a_time:                                 # reduced, Qwen3.5-35B
                        cells.append(make_cell(arch, sms, "qwen3.5-35b", prec, g, *co))
            for shape in SHAPES:                                             # num_sms=160: GEMM tuples only
                for g in range(len(GEMM_TUPLES)):
                    cells.append(make_cell(arch, 160, shape, prec, g, 0, 0, 0))
    return cells


# ----------------------------------------------------------------------------- ptxas parsing
SPILL_RE = re.compile(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
REGS_RE = re.compile(r"Used (\d+) registers")
SMEM_RE = re.compile(r"(\d+) bytes smem")
TARGET_RE = re.compile(r"Compiling entry function '[^']*' for '(sm_\w+)'")
LOCAL_RE = re.compile(r"local memory used|Local memory used|spilled to local", re.I)
PATH_RE = re.compile(r"(?:[A-Za-z]:)?(?:/[\w.+\-]+){2,}")


def sanitize(text, limit=600):
    text = PATH_RE.sub("<path>", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def parse_ptxas(log):
    spills = SPILL_RE.findall(log)
    regs = [int(x) for x in REGS_RE.findall(log)]
    smem = [int(x) for x in SMEM_RE.findall(log)]
    targets = sorted(set(TARGET_RE.findall(log)))
    return dict(
        ptxas_target=targets,
        registers=max(regs) if regs else None,
        smem_bytes=max(smem) if smem else None,
        stack_frame_bytes=max((int(a) for a, _, _ in spills), default=None),
        spill_store_bytes=sum(int(b) for _, b, _ in spills) if spills else None,
        spill_load_bytes=sum(int(c) for _, _, c in spills) if spills else None,
        local_memory_warning=bool(LOCAL_RE.search(log)),
    )


def classify_exception(exc_text, phase_compiled_started):
    t = exc_text
    nvcc_fail = re.search(r"NVCC compilation failed|NVCC did not produce|nvcc|ptxas error|error: static assertion|"
                          r"Command failed|compilation terminated|PTXAS reported", t, re.I)
    return "compile_failed" if nvcc_fail else "rejected_by_host"


# ----------------------------------------------------------------------------- worker
def worker_main(spec_path):
    spec = json.load(open(spec_path))
    g = spec["group"]
    os.environ.update({
        "DG_JIT_COMPILE_ONLY": "1",
        "DG_JIT_TARGET_ARCH": g["arch"],
        "DG_JIT_TARGET_NUM_SMS": str(g["num_sms"]),
        "DG_JIT_PTXAS_VERBOSE": "1",
        "DG_JIT_CACHE_DIR": spec["cache_dir"],
    })
    os.makedirs(spec["cache_dir"], exist_ok=True)
    for k in [k for k in os.environ if k.startswith("WEAVE_")]:
        del os.environ[k]

    import ctypes
    import torch
    import deep_gemm
    from deep_gemm import _C

    libc = ctypes.CDLL(None)
    shp = SHAPES[g["shape"]]
    E, topk, H, I = shp["E"], shp["topk"], shp["H"], shp["I"]
    prec = g["precision"]
    el = E // NUM_RANKS
    dev = torch.device("cuda")

    # Symmetric buffer: one local allocation reused as all 4 ranks' buffers
    align = _C.get_token_alignment_for_mega_moe(prec)
    max_tok = ((NUM_MAX_TOKENS + align - 1) // align) * align
    nbytes, slicer = _C.get_symm_buffer_size_for_mega_moe(NUM_RANKS, E, max_tok, topk, H, I, prec, "swiglu", 0)
    buf = torch.empty(nbytes, dtype=torch.int8, device=dev)
    print('[worker] symm buffer bytes', nbytes, flush=True)
    ptrs = [buf.data_ptr()] * NUM_RANKS
    slicer(buf)  # exercises the slicing path (tensor views are not needed afterwards)

    def mn_major_sf(n, k, gran):
        # int32-packed scale factors (4 per int32), MN-major: shape (el, n, k/gran/4), strides (n*kk, 1, n)
        kk = k // gran // 4
        return torch.empty((el, kk, n), dtype=torch.int32, device=dev).transpose(1, 2)

    gran = 32 if prec == "fp8xfp4" else 16
    l1 = (torch.empty((el, 2 * I, H // 2), dtype=torch.int8, device=dev), mn_major_sf(2 * I, H, gran))
    l2 = (torch.empty((el, H, I // 2), dtype=torch.int8, device=dev), mn_major_sf(H, I, gran))
    y = torch.empty((NUM_TOKENS, H), dtype=torch.bfloat16, device=dev)

    out_path = spec["out_jsonl"]
    log_dir = spec["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    for cell in spec["cells"]:
        row = dict(cell)
        for k in [k for k in os.environ if k.startswith("WEAVE_")]:
            del os.environ[k]
        os.environ.update({
            "WEAVE_GEMM1_CONFIG": cell["gemm1"], "WEAVE_GEMM2_CONFIG": cell["gemm2"],
            "WEAVE_REG_PULL_BYTES": str(cell["reg_pull_bytes"]),
            "WEAVE_SCATTER_OWNER": cell["scatter_owner"], "WEAVE_SCATTER_STAGING": cell["scatter_staging"],
            "WEAVE_DEDUP_MODE": str(cell["dedup_mode"]),
        })
        # fresh JIT disk cache so that ptxas always runs (and logs) for this cell
        shutil.rmtree(spec["cache_dir"], ignore_errors=True)
        os.makedirs(spec["cache_dir"], exist_ok=True)
        log_file = os.path.join(log_dir, "cell_%d.log" % cell["id"])
        sys.stdout.flush()
        saved = os.dup(1)
        fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        os.dup2(fd, 1)
        exc_text, t0 = None, time.time()
        try:
            if prec == "fp8xfp4":
                _C.fp8_fp4_mega_moe(y, l1, l2, None, None, None, buf, ptrs, 0, max_tok, E, topk,
                                    (1, 1, 32), "swiglu", None, True, None, None)
            else:
                _C.fp4_fp4_mega_moe(y, l1, l2, None, None, None, None, buf, ptrs, 0, max_tok, E, topk,
                                    (1, 1, 16), "swiglu", None, True, None, None, None, 1.0)
        except BaseException as e:  # noqa: BLE001
            exc_text = "%s: %s" % (type(e).__name__, e)
        finally:
            sys.stdout.flush()
            libc.fflush(None)
            os.dup2(saved, 1)
            os.close(saved)
            os.close(fd)
        row["seconds"] = round(time.time() - t0, 1)
        log = open(log_file, errors="replace").read()
        info = parse_ptxas(log)
        row.update(info)
        if exc_text is not None:
            row["status"] = classify_exception(exc_text + "\n" + log, True)
            row["error"] = sanitize(exc_text if row["status"] == "rejected_by_host" else (exc_text + " | " + log[-400:]))
        elif info["spill_store_bytes"] is None:
            row["status"] = "compile_failed"
            row["error"] = "no ptxas log captured (no ptxas 'Used N registers' line)"
        elif info["spill_store_bytes"] > 0 or info["spill_load_bytes"] > 0:
            row["status"] = "spill"
        else:
            row["status"] = "ok"
        if row["status"] == "ok":
            os.remove(log_file)  # keep only logs of interesting cells
        with open(out_path, "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()


# ----------------------------------------------------------------------------- driver
def summarize(rows):
    def key(r):
        return "%s/sms%d" % (r["arch"], r["num_sms"])
    summary = {}
    for r in rows:
        s = summary.setdefault(key(r), dict(total=0, ok=0, spill=0, compile_failed=0, rejected_by_host=0,
                                            stack_frame_nonzero=0))
        s["total"] += 1
        s[r["status"]] += 1
        if (r.get("stack_frame_bytes") or 0) > 0:
            s["stack_frame_nonzero"] += 1
    per_arch = {}
    for a in ARCHS:
        per_arch[a] = {k: sum(v[k] for kk, v in summary.items() if kk.startswith(a + "/"))
                       for k in ("total", "ok", "spill", "compile_failed", "rejected_by_host", "stack_frame_nonzero")}
    by_prec = {}
    for r in rows:
        s = by_prec.setdefault("%s/%s" % (r["arch"], r["precision"]),
                               dict(total=0, ok=0, spill=0, compile_failed=0, rejected_by_host=0))
        s["total"] += 1
        s[r["status"]] += 1
    return dict(per_arch_and_num_sms=summary, per_arch=per_arch, per_arch_and_precision=by_prec)


def driver_main(args):
    cells = build_matrix(args.coverage)
    for i, c in enumerate(cells):
        c["id"] = i
    workdir = os.path.abspath(args.workdir)
    shutil.rmtree(workdir, ignore_errors=True)
    os.makedirs(workdir)
    groups = {}
    for c in cells:
        groups.setdefault((c["arch"], c["num_sms"], c["shape"], c["precision"]), []).append(c)
    # chunk every group so that all workers stay busy
    chunks = []
    per_chunk = max(1, args.chunk)
    for gk, gc in groups.items():
        for j in range(0, len(gc), per_chunk):
            chunks.append((gk, gc[j:j + per_chunk]))
    print("[proof] %d cells, %d groups, %d worker chunks, %d jobs" % (len(cells), len(groups), len(chunks), args.jobs), flush=True)
    jobs, running, done = list(enumerate(chunks)), [], 0
    t0 = time.time()
    me = os.path.abspath(__file__)
    while jobs or running:
        while jobs and len(running) < args.jobs:
            idx, (gk, gc) = jobs.pop(0)
            spec = dict(group=dict(arch=gk[0], num_sms=gk[1], shape=gk[2], precision=gk[3]), cells=gc,
                        cache_dir=os.path.join(workdir, "jit_%d" % idx), log_dir=os.path.join(workdir, "logs"),
                        out_jsonl=os.path.join(workdir, "rows_%d.jsonl" % idx))
            sp = os.path.join(workdir, "spec_%d.json" % idx)
            json.dump(spec, open(sp, "w"))
            lf = open(os.path.join(workdir, "worker_%d.out" % idx), "w")
            p = subprocess.Popen([sys.executable, "-u", me, "--worker", sp], stdout=lf, stderr=subprocess.STDOUT)
            running.append((idx, p, lf))
        time.sleep(2)
        for item in list(running):
            if item[1].poll() is not None:
                running.remove(item)
                item[2].close()
                done += 1
                shutil.rmtree(os.path.join(workdir, "jit_%d" % item[0]), ignore_errors=True)
                print("[proof] chunk %d rc=%s (%d/%d) %.0fs" % (item[0], item[1].returncode, done, len(chunks),
                                                                 time.time() - t0), flush=True)
    rows = []
    for fn in sorted(os.listdir(workdir)):
        if fn.startswith("rows_"):
            rows += [json.loads(l) for l in open(os.path.join(workdir, fn)) if l.strip()]
    seen = {r["id"] for r in rows}
    for c in cells:  # chunks that died before writing a row (worker crash)
        if c["id"] not in seen:
            r = dict(c)
            r.update(status="compile_failed", error="worker crashed before producing a result")
            rows.append(r)
    rows.sort(key=lambda r: r["id"])
    for r in rows:
        r.pop("id", None)
    result = dict(
        description="Compile-only proof (nvcc + ptxas, nothing launched); spill = ptxas spill stores/loads > 0",
        toolchain=args.toolchain or None,
        shapes=SHAPES, ep_ranks=NUM_RANKS, num_max_tokens_per_rank=NUM_MAX_TOKENS, num_tokens=NUM_TOKENS,
        coverage=dict(
            mode=args.coverage,
            archs=list(ARCHS), precisions=list(PRECISIONS), gemm_tuples=GEMM_TUPLES,
            qwen3_8=("full product: arch x num_sms{148,144} x precision x 10 GEMM tuples x reg_pull{0,1024,2048}"
                     " x scatter{(epilogue_only,reg_2stage),(comm_only,smem)} x dedup{0,1}"),
            qwen3_5_35b=("reduced: arch x num_sms{148,144} x precision x 10 GEMM tuples x {default co-knobs,"
                         " reg_pull 1024, reg_pull 2048, scatter comm_only/smem, dedup 1} (one varied at a time)"),
            num_sms_160=("reduced: arch x precision x both shapes x 10 GEMM tuples x default co-knobs"),
        ),
        summary=summarize(rows), cells=rows)
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps(result["summary"]["per_arch"], indent=1))
    print("[proof] wrote", args.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--out", default="results/blackwell_sm100a_sm103a_compile_proof.json")
    ap.add_argument("--workdir", default="blackwell_compile_proof_work")
    ap.add_argument("--jobs", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=16, help="cells per worker process")
    ap.add_argument("--coverage", choices=("full", "smoke"), default="full")
    ap.add_argument("--toolchain", default="", help="free-form toolchain label recorded in the JSON (e.g. 'CUDA 13.2')")
    args = ap.parse_args()
    if args.worker:
        try:
            worker_main(args.worker)
        except BaseException:  # noqa: BLE001
            traceback.print_exc()
            sys.exit(1)
    else:
        driver_main(args)


if __name__ == "__main__":
    main()
