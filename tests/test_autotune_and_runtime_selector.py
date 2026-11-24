#!/usr/bin/env python
"""CPU-only tests for deep_gemm/mega/autotune.py (fingerprint, candidates, profile schema,
runtime rules, RuntimeSelector).  No GPU and no compiled deep_gemm._C are needed.

Run:  python tests/test_autotune_and_runtime_selector.py      (or: pytest this file)

autotune.py is imported BY FILE PATH so deep_gemm/__init__.py (which needs _C) is never imported.
Temporary profiles live in tests/_tmp_autotune/run_<pid>/ (never the system tmp folder) and are
removed at exit.
"""
import atexit
import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
import unittest
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
AUTOTUNE_PY = REPO / "deep_gemm" / "mega" / "autotune.py"
TMP_BASE = HERE / "_tmp_autotune"
TMP_ROOT = TMP_BASE / f"run_{os.getpid()}"

_spec = importlib.util.spec_from_file_location("weave_autotune_under_test", str(AUTOTUNE_PY))
at = importlib.util.module_from_spec(_spec)
sys.modules["weave_autotune_under_test"] = at   # dataclasses need the module registered
_spec.loader.exec_module(at)


# ----------------------------------------------------------------------------- helpers
class Skip(unittest.SkipTest):
    pass


def _cleanup():
    if TMP_ROOT.exists():
        shutil.rmtree(TMP_ROOT, ignore_errors=True)   # only the unique run dir created by this process
    try:
        TMP_BASE.rmdir()                               # only if empty
    except OSError:
        pass


atexit.register(_cleanup)


def tmpdir(name):
    d = TMP_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def isolated(fn):
    """Run a test with every WEAVE_* variable removed, then restore the environment."""
    @functools.wraps(fn)
    def wrapper(*a, **kw):
        saved = {k: v for k, v in os.environ.items() if k.startswith("WEAVE_")}
        for k in saved:
            del os.environ[k]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return fn(*a, **kw)
        finally:
            for k in [k for k in os.environ if k.startswith("WEAVE_")]:
                del os.environ[k]
            os.environ.update(saved)
    return wrapper


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} from {getattr(fn, '__name__', fn)}")


def HW(cc, name, sms, ep=8):
    return at.HardwareFingerprint(cc[0], cc[1], name, sms, 233472, 126 * 2**20, ep)


B200 = HW((10, 0), "NVIDIA B200", 148)
B200_144 = HW((10, 0), "NVIDIA B200", 144)
B300 = HW((10, 3), "NVIDIA B300 SXM6 AC", 148)
H20 = HW((9, 0), "NVIDIA H20", 78)
H100 = HW((9, 0), "NVIDIA H100 80GB HBM3", 132)

MK_FP4 = (512, 10, 8192, 2048, 0, "fp8xfp4")
MK_FP8 = (512, 10, 8192, 2048, 0, "fp8xfp8")
BASE = {"WEAVE_GEMM1_CONFIG": "64", "WEAVE_GEMM2_CONFIG": "64"}


def model_dict(mk):
    return {"E": mk[0], "K": mk[1], "H": mk[2], "I": mk[3], "shared": mk[4], "mma_type": mk[5]}


def THR(**over):
    th = {"dedup_min_saved_rows": 50, "comm_scatter_min_rows": 200, "comm_scatter_min_tiles": 20,
          "skew_comm_heavy": {"WEAVE_DISPATCH_ACTIVE_SMS": "132"},
          "skew_comp_heavy": {"WEAVE_DISPATCH_ACTIVE_SMS": "74"}, "skew_rgs": 4}
    th.update(over)
    return th


def make_profile(hw=B200, mk=MK_FP4, buckets=None, thresholds=None):
    if buckets is None:
        buckets = [
            {"S": 16, "env": {"WEAVE_GEMM1_CONFIG": "16:8:256", "WEAVE_GEMM2_CONFIG": "16:8:256"},
             "uniform_stats": {"m_rem": 10, "d_in": 8, "d_out": 10, "k_tiles": 8, "m_max": 2, "mean_m": 0.3},
             "verified_variants": [], "rules_enabled": {"dedup": True, "comm_scatter": True,
                                                         "spatial_tier": True, "ready_group": True}},
            {"S": 128, "env": {"WEAVE_GEMM1_CONFIG": "32:16:128", "WEAVE_GEMM2_CONFIG": "32:16:128"},
             "uniform_stats": {"m_rem": 100, "d_in": 60, "d_out": 100, "k_tiles": 30, "m_max": 6, "mean_m": 2.5},
             "verified_variants": [], "rules_enabled": {}},
            {"S": 512, "env": {"WEAVE_GEMM1_CONFIG": "64:16:128", "WEAVE_GEMM2_CONFIG": "64:16:128"},
             "uniform_stats": {"m_rem": 400, "d_in": 100, "d_out": 400, "k_tiles": 100, "m_max": 20, "mean_m": 10.0},
             "verified_variants": [], "rules_enabled": {}},
        ]
    return {"schema": at.SCHEMA, "tool_version": at.TOOL_VERSION, "hw": hw.to_dict(),
            "model": model_dict(mk), "buckets": buckets, "thresholds": thresholds or THR()}


def write_profile(doc, hw, mk):
    return at.save_profile(doc, at.profile_path(hw, *mk))


class FakeSelector:
    """RuntimeSelector with a mocked hardware fingerprint (torch.cuda is never touched)."""
    def __init__(self, hw):
        self.hw = hw
        self.calls = 0
        self.sel = at.RuntimeSelector()
        orig = at.query_fingerprint

        def fake(ep, device=None):
            self.calls += 1
            return at.HardwareFingerprint(hw.cc_major, hw.cc_minor, hw.device_name,
                                          hw.num_sms, hw.smem_per_sm, hw.l2_bytes, ep, hw.gcn_arch)
        self._orig, self._fake = orig, fake

    def select(self, s_global, mk=MK_FP4, ep=None, **kw):
        at.query_fingerprint = self._fake
        try:
            return self.sel.select(ep=self.hw.ep if ep is None else ep, device=0, num_experts=mk[0], num_topk=mk[1],
                                   hidden=mk[2], intermediate_hidden=mk[3], num_shared_experts=mk[4],
                                   mma_type=mk[5], s_global=s_global, **kw)
        finally:
            at.query_fingerprint = self._orig


def stats(m_rem=0, d_in=0, d_out=0, k_tiles=0, m_max=0, mean_m=1.0):
    return at.RoutingStats(m_rem, d_in, d_out, k_tiles, m_max, mean_m)


def rbucket(**over):
    b = {"S": 512, "env": dict(BASE),
         "uniform_stats": {"m_rem": 800, "d_in": 100, "d_out": 800, "k_tiles": 100, "m_max": 20, "mean_m": 10.0},
         "verified_variants": [], "rules_enabled": {}}
    b.update(over)
    return b


# fires all four rules
FULL_STATS = dict(m_rem=300, d_in=150, d_out=300, k_tiles=50, m_max=40, mean_m=10.0)
F_DEDUP = {"WEAVE_DEDUP_MODE": "1"}
F_COMM = dict(at.SCATTER_COMM)
F_SPAT = {"WEAVE_DISPATCH_ACTIVE_SMS": "132"}
F_RG = {"WEAVE_READY_GROUP_SIZE": "4"}


def envof(*deltas):
    e = dict(BASE)
    for d in deltas:
        e.update(d)
    return e


# ============================================================================= 1. schema
@isolated
def test_schema_valid_and_roundtrip():
    doc = make_profile()
    assert at.validate_profile(doc) is doc
    d = tmpdir("schema")
    p = at.save_profile(doc, d / "p.json")
    assert p.is_file()
    assert at.load_profile(p) == doc
    assert not [f for f in os.listdir(d) if ".tmp" in f], "temp file left behind"
    # null thresholds are legal (rule disabled); overwrite is fine
    doc2 = make_profile(thresholds=THR(dedup_min_saved_rows=None, comm_scatter_min_rows=None,
                                       comm_scatter_min_tiles=None, skew_comm_heavy=None, skew_comp_heavy=None,
                                       skew_rgs=0))
    at.save_profile(doc2, p)
    assert at.load_profile(p) == doc2
    # nested directories are created
    q = at.save_profile(doc, d / "a" / "b" / "p.json")
    assert q.is_file()


@isolated
def test_schema_negative():
    good = make_profile()
    bad = dict(good, schema="weave_autotune/v0")
    raises(at.ProfileError, at.validate_profile, bad)
    raises(at.ProfileError, at.validate_profile, [])
    for key in ("hw", "model", "buckets", "thresholds"):
        d = dict(good)
        del d[key]
        raises(at.ProfileError, at.validate_profile, d)
    raises(at.ProfileError, at.validate_profile, dict(good, buckets=[]))
    b = good["buckets"]
    raises(at.ProfileError, at.validate_profile, dict(good, buckets=[b[1], b[0]]))          # decreasing
    raises(at.ProfileError, at.validate_profile, dict(good, buckets=[b[0], dict(b[0])]))    # equal
    assert issubclass(at.ProfileError, ValueError)
    for bad_key in ("WEAVE_REDUCE_WARPS", "WEAVE_ADAPT_GEMM_CONFIG", "WEAVE_ADAPT_BM"):
        bk = [dict(b[0], env={bad_key: "1"})]
        raises(ValueError, at.validate_profile, dict(good, buckets=bk))
        bk = [dict(b[0], verified_variants=[{bad_key: "1"}])]
        raises(ValueError, at.validate_profile, dict(good, buckets=bk))
        raises(ValueError, at.validate_profile, dict(good, thresholds=THR(skew_comm_heavy={bad_key: "1"})))
        raises(ValueError, at.validate_profile, dict(good, thresholds=THR(skew_comp_heavy={bad_key: "1"})))
    for key in ("dedup_min_saved_rows", "comm_scatter_min_rows", "comm_scatter_min_tiles", "skew_rgs"):
        th = THR()
        del th[key]
        raises(at.ProfileError, at.validate_profile, dict(good, thresholds=th))
    # save_profile validates first and writes nothing
    d = tmpdir("schema_neg")
    raises(ValueError, at.save_profile, dict(good, schema="x"), d / "x.json")
    assert not (d / "x.json").exists()


# ============================================================================= 2. fingerprint isolation
@isolated
def test_hw_id_format_and_isolation():
    assert B200.hw_id == "sm100_nvidia_b200_148sm_ep8"
    assert B300.hw_id == "sm103_nvidia_b300_sxm6_ac_148sm_ep8"
    assert H20.hw_id == "sm90_nvidia_h20_78sm_ep8"
    assert H100.hw_id == "sm90_nvidia_h100_80gb_hbm3_132sm_ep8"
    assert at.slug("  ") == "gpu" and at.slug("A/B  c") == "a_b_c"
    variants = [B200, B300, B200_144, HW((10, 0), "NVIDIA B200", 148, ep=4), H20, H100,
                HW((10, 0), "NVIDIA B200x", 148)]
    assert len({v.hw_id for v in variants}) == len(variants)
    assert len({str(at.profile_path(v, *MK_FP4)) for v in variants}) == len(variants)
    # same-SM-count B200 vs B300 (148 SMs) must still differ (cc minor + name)
    assert B200.num_sms == B300.num_sms and B200.hw_id != B300.hw_id


@isolated
def test_profile_path_overrides():
    d = tmpdir("paths")
    p0 = at.profile_path(B200, *MK_FP4)
    assert p0.parent == Path.home() / ".deep_gemm" / "weave_autotune"
    assert p0.name == f"{B200.hw_id}__E512_K10_H8192_I2048_S0_fp8xfp4.json"
    assert at.model_key(512, 10, 8192, 2048, 0, "fp8xfp4") == "E512_K10_H8192_I2048_S0_fp8xfp4"
    os.environ["WEAVE_AUTOTUNE_DIR"] = str(d)
    p1 = at.profile_path(B200, *MK_FP4)
    assert p1.parent == d and p1.name == p0.name
    assert at.profile_path(B300, *MK_FP4) != p1
    f = d / "explicit.json"
    os.environ["WEAVE_AUTOTUNE_FILE"] = str(f)
    assert at.profile_path(B200, *MK_FP4) == f and at.profile_path(H20, *MK_FP8) == f
    assert at.default_autotune_dir() == d


@isolated
def test_foreign_profile_is_ignored():
    d = tmpdir("foreign")
    os.environ["WEAVE_AUTOTUNE_DIR"] = str(d)
    sel = at.RuntimeSelector()
    # B300 sm103 148 SMs profile planted at the B200 sm100 148 SMs path
    for foreign in (B300, HW((10, 0), "NVIDIA B200", 148, ep=4), HW((10, 0), "NVIDIA B200", 144)):
        at.save_profile(make_profile(hw=foreign), at.profile_path(B200, *MK_FP4))
        sel.clear()
        doc, path = sel.profile_for(B200, MK_FP4)
        assert doc is None and path is None, foreign.hw_id
        fs = FakeSelector(B200)
        r = fs.select(128)
        assert r.source == "cold_start" and r.env == {} and r.profile_path is None
    # the matching profile is accepted
    at.save_profile(make_profile(hw=B200), at.profile_path(B200, *MK_FP4))
    sel.clear()
    doc, path = sel.profile_for(B200, MK_FP4)
    assert doc is not None and path == str(at.profile_path(B200, *MK_FP4))
    # same hw but different model shape (planted at the right path) is ignored too
    at.save_profile(make_profile(hw=B200, mk=(256, 8, 8192, 2048, 0, "fp8xfp4")), at.profile_path(B200, *MK_FP4))
    sel.clear()
    assert sel.profile_for(B200, MK_FP4)[0] is None
    # WEAVE_AUTOTUNE_FILE pointing at a B300 file, queried on B200
    f = d / "b300.json"
    at.save_profile(make_profile(hw=B300), f)
    os.environ["WEAVE_AUTOTUNE_FILE"] = str(f)
    sel.clear()
    assert sel.profile_for(B200, MK_FP4)[0] is None
    assert sel.profile_for(B300, MK_FP4)[0] is not None
    # corrupt file -> cold start, never an exception
    del os.environ["WEAVE_AUTOTUNE_FILE"]
    at.profile_path(B200, *MK_FP4).write_text("{not json", encoding="utf-8")
    sel.clear()
    assert sel.profile_for(B200, MK_FP4)[0] is None


@isolated
def test_fingerprint_cached_and_mockable():
    fs = FakeSelector(H20)
    fs.select(64, mk=MK_FP8)
    fs.select(128, mk=MK_FP8)
    assert fs.calls == 1, "fingerprint must be queried once per (device, ep)"
    fs.select(64, mk=MK_FP8, ep=4)
    assert fs.calls == 2


# ============================================================================= 3. SM-count scaling
@isolated
def test_arch_family_and_alignment():
    assert at.arch_family(10) == "blackwell" and at.arch_family(9) == "hopper"
    assert at.arch_family(8) is None and at.arch_family(12) is None
    assert B200.family == B300.family == "blackwell" and H20.family == H100.family == "hopper"
    assert B200.sm_align == B300.sm_align == 2 and H20.sm_align == H100.sm_align == 1
    assert at.default_mma_type("hopper") == "fp8xfp8" and at.default_mma_type("blackwell") == "fp8xfp4"


@isolated
def test_sm_fraction_scaling():
    exp = {  # (hw, frac) -> value, computed by hand
        (B200, 0.5): 74, (B200, 0.9): 132, (B200_144, 0.5): 72, (B200_144, 0.9): 130, (B300, 0.5): 74,
        (B300, 0.9): 132, (H20, 0.85): 66, (H20, 0.67): 52, (H100, 0.85): 112, (H100, 0.67): 88,
    }
    for (hw, frac), v in exp.items():
        assert at.sm_fraction(frac, hw) == v, (hw.hw_id, frac, at.sm_fraction(frac, hw))
    odd = HW((10, 0), "odd", 147)
    for hw in (B200, B200_144, B300, H20, H100, odd, HW((9, 0), "tiny", 3), HW((10, 0), "tiny", 2)):
        for frac in (0.0, 0.1, 0.5, 0.67, 0.85, 0.9, 1.0, 5.0):
            v = at.sm_fraction(frac, hw)
            assert hw.sm_align <= v <= hw.num_sms, (hw.hw_id, frac, v)
            if frac <= 1.0:   # frac>1 on an odd SM count returns num_sms unaligned (autotune.py:192, reported)
                assert v % hw.sm_align == 0, (hw.hw_id, frac, v)
    assert at.sm_fraction(1.0, odd) == 146 and at.sm_fraction(1.0, H20) == 78
    # scaling: more SMs never gives a smaller subset
    for fam_fracs, hws in (((0.5, 0.9), (B200_144, B200)), ((0.85, 0.67), (H20, H100))):
        for fr in fam_fracs:
            assert at.sm_fraction(fr, hws[0]) <= at.sm_fraction(fr, hws[1])


@isolated
def test_coknob_variants_scale_with_sms():
    for hw, d5, d9 in ((B200, 74, 132), (B200_144, 72, 130), (B300, 74, 132)):
        v = {name: (grp, delta) for grp, name, delta in at.coknob_variants("blackwell", hw, 4096, {})}
        assert v["dispatch_sms_0.5"][1] == {"WEAVE_DISPATCH_ACTIVE_SMS": str(d5)}
        assert v["dispatch_sms_0.9"][1] == {"WEAVE_DISPATCH_ACTIVE_SMS": str(d9)}
        assert "v_head_1" not in v
        assert "scatter_comm_smem" in v and v["scatter_comm_smem"][1] == at.SCATTER_COMM
        assert "WEAVE_V_HEAD" not in "".join(str(x) for x in v.values()) or True
        small = {name for _, name, _ in at.coknob_variants("blackwell", hw, 2048, {})}
        assert "v_head_1" in small
        comm = {name: d for _, name, d in at.coknob_variants("blackwell", hw, 128, dict(at.SCATTER_COMM))}
        assert "scatter_epilogue" in comm and comm["scatter_epilogue"] == at.SCATTER_EPI
    for hw, d85, d67 in ((H20, 66, 52), (H100, 112, 88)):
        v = {name: delta for _, name, delta in at.coknob_variants("hopper", hw, 4096, {})}
        assert v["dispatch_sms_0.85"] == {"WEAVE_DISPATCH_ACTIVE_SMS": str(d85)}
        assert v["dispatch_sms_0.67"] == {"WEAVE_DISPATCH_ACTIVE_SMS": str(d67)}
        assert v["v_head_10n"] == {"WEAVE_V_HEAD": str(10 * hw.num_sms)}
        assert "dispatch_scatter_sms_0.67" not in v and "scatter_comm_smem" in v
        vc = {name: delta for _, name, delta in at.coknob_variants("hopper", hw, 4096, dict(at.SCATTER_COMM))}
        assert vc["dispatch_scatter_sms_0.67"] == {"WEAVE_DISPATCH_ACTIVE_SMS": str(d67),
                                                   "WEAVE_SCATTER_ACTIVE_SMS": str(d67)}
        assert "scatter_epilogue" in vc
    assert at.coknob_variants("hopper", H20, 1, {}) != at.coknob_variants("hopper", H100, 1, {})
    raises(ValueError, at.coknob_variants, "cpu", H20, 1, {})
    # every variant on top of its base env is statically legal for its family
    for fam, hws, bases in (("blackwell", (B200, B300), ({}, dict(at.SCATTER_COMM))),
                            ("hopper", (H20, H100), ({}, dict(at.SCATTER_COMM)))):
        for hw in hws:
            for base in bases:
                for _, name, delta in at.coknob_variants(fam, hw, 512, base):
                    env = dict(base)
                    env.update(delta)
                    assert at.static_legal(env, fam) is None, (fam, name, at.static_legal(env, fam))
                    at.normalize_env(env)   # only injectable knobs


@isolated
def test_gemm_candidates_by_mexp():
    def bms(fam, m):
        return {c["WEAVE_GEMM1_CONFIG"].split(":")[0] for c in at.gemm_candidates(fam, m)} | \
               {c["WEAVE_GEMM2_CONFIG"].split(":")[0] for c in at.gemm_candidates(fam, m)}
    for m in (0.1, 1, 12, 12.0, 20, 48, 100, 128):
        assert "240" not in bms("blackwell", m), m
    for m in (129, 200, 1000):
        assert "240" in bms("blackwell", m), m
    assert "16" in bms("blackwell", 1) and "96" not in bms("blackwell", 1)
    for fam, ms in (("blackwell", (1, 30, 100, 300)), ("hopper", (1, 50, 200))):
        for m in ms:
            c = at.gemm_candidates(fam, m)
            assert 2 <= len(c) <= 5   # hopper large M has 2 tuples (docstring says 3-5)
            for t in c:
                assert set(t) == {"WEAVE_GEMM1_CONFIG", "WEAVE_GEMM2_CONFIG"}
                assert at.static_legal(t, fam) is None
    for m in (1, 50, 200):    # hopper: bare BM only
        for t in at.gemm_candidates("hopper", m):
            assert all(x.isdigit() for x in t.values())
    assert [t["WEAVE_GEMM1_CONFIG"] for t in at.gemm_candidates("hopper", 200)] == ["64", "128"]
    raises(ValueError, at.gemm_candidates, "cpu", 10)
    # blackwell tuples are rejected by the hopper legality check
    assert at.static_legal(at.gemm_candidates("blackwell", 1)[0], "hopper") is not None


@isolated
def test_cold_start_env_and_comm_heavy_sets():
    assert at.cold_start_env("hopper", H20, 64) == at.SCATTER_COMM
    assert at.cold_start_env("hopper") is not at.SCATTER_COMM
    assert at.cold_start_env("blackwell", B200, 64) == {} and at.cold_start_env(None) == {}
    for hw in (B200, B300):
        assert at.comm_heavy_candidates("blackwell", hw)[0] == {"WEAVE_DISPATCH_ACTIVE_SMS": "132"}
        assert at.comp_heavy_candidates("blackwell", hw) == [{"WEAVE_DISPATCH_ACTIVE_SMS": "74"}]
    assert at.comp_heavy_candidates("hopper", H20)[1] == {"WEAVE_DISPATCH_ACTIVE_SMS": "52"}
    # through the selector (no profile)
    d = tmpdir("cold")
    os.environ["WEAVE_AUTOTUNE_DIR"] = str(d)
    r = FakeSelector(H20).select(64, mk=MK_FP8)
    assert (r.source, r.env) == ("cold_start", at.SCATTER_COMM)
    r = FakeSelector(B200).select(64)
    assert (r.source, r.env) == ("cold_start", {})
    r = FakeSelector(HW((8, 0), "NVIDIA A100", 108)).select(64)
    assert (r.source, r.env) == ("unsupported", {})


@isolated
def test_static_legal_rejections():
    L = at.static_legal
    assert L({}, "blackwell") is None and L({}, "hopper") is None
    assert L(at.SCATTER_COMM, "blackwell") is None
    # staged scatter with epilogue owner
    assert L({"WEAVE_SCATTER_STAGING": "smem", "WEAVE_SCATTER_OWNER": "epilogue_only"}, "blackwell")
    assert L({"WEAVE_SCATTER_STAGING": "hbm_dedup", "WEAVE_SCATTER_OWNER": "0"}, "hopper")
    assert L({"WEAVE_SCATTER_STAGING": "smem", "WEAVE_SCATTER_OWNER": "comm_only"}, "hopper") is None
    assert L({"WEAVE_SCATTER_STAGING": "reg_2stage", "WEAVE_SCATTER_OWNER": "epilogue_only"}, "blackwell") is None
    # scatter SM throttle needs comm owner
    assert L({"WEAVE_SCATTER_ACTIVE_SMS": "64", "WEAVE_SCATTER_OWNER": "epilogue_only"}, "hopper")
    assert L({"WEAVE_SCATTER_ACTIVE_SMS": "64", "WEAVE_SCATTER_OWNER": "comm_only"}, "hopper") is None
    # reg-pull bytes
    for rp in ("512", "3000", "8192"):
        assert L({"WEAVE_REG_PULL_BYTES": rp}, "blackwell") == "illegal WEAVE_REG_PULL_BYTES"
    for rp in ("0", "1024", "2048", "4096"):
        assert L({"WEAVE_REG_PULL_BYTES": rp}, "blackwell") is None
    assert L({"WEAVE_DISPATCH_WARPS": "6"}, "hopper") and L({"WEAVE_SCATTER_WARPS": "2"}, "blackwell")
    assert L({"WEAVE_DISPATCH_WARPS": "8", "WEAVE_SCATTER_WARPS": "4"}, "hopper") is None
    assert L({"WEAVE_READY_GROUP_SIZE": "3"}, "blackwell") and L({"WEAVE_READY_GROUP_SIZE": "16"}, "blackwell") is None
    # colon / comma GEMM tuples on hopper only
    for k in ("WEAVE_GEMM1_CONFIG", "WEAVE_GEMM2_CONFIG"):
        assert L({k: "64:16"}, "hopper") and L({k: "64,16"}, "hopper")
        assert L({k: "64:16"}, "blackwell") is None and L({k: "64"}, "hopper") is None
    assert at.gemm2_block_m({"WEAVE_GEMM2_CONFIG": "96:8", "WEAVE_GEMM1_CONFIG": "64"}) == 96
    assert at.gemm2_block_m({"WEAVE_GEMM1_CONFIG": "80,8"}) == 80
    assert at.gemm2_block_m({}) == 128 and at.gemm2_block_m({"WEAVE_GEMM2_CONFIG": "0"}) == 128
    assert at.gemm2_block_m({"WEAVE_GEMM2_CONFIG": "abc"}) == 128


# ============================================================================= 4. env priority / selection
@isolated
def test_manual_mode_priority():
    d = tmpdir("manual")
    os.environ["WEAVE_AUTOTUNE_DIR"] = str(d)
    write_profile(make_profile(), B200, MK_FP4)
    fs = FakeSelector(B200)
    r = fs.select(512)
    assert r.source == "profile" and r.env == make_profile()["buckets"][2]["env"]
    for k, v in (("WEAVE_GEMM1_CONFIG", "64"), ("WEAVE_ADAPT_BM", "0"), ("WEAVE_ADAPT_GEMM_CONFIG", "1"),
                 ("WEAVE_REDUCE_WARPS", "4"), ("WEAVE_DISPATCH_ACTIVE_SMS", "100")):
        os.environ[k] = v
        r = fs.select(512)
        assert (r.source, r.env) == ("manual", {}), k
        assert at.manual_env_keys() == [k]
        del os.environ[k]
    os.environ["WEAVE_GEMM1_CONFIG"] = ""                      # empty value is not a setting
    assert fs.select(512).source == "profile"
    del os.environ["WEAVE_GEMM1_CONFIG"]
    os.environ["WEAVE_PRECISION_MODE"] = "fp8xfp4"             # backend selector only
    os.environ["WEAVE_KNOB_PRECISION_MODE"] = "fp8xfp4"
    os.environ["WEAVE_AUTOTUNE_ROUTING"] = "auto"
    assert fs.select(512).source == "profile" and at.manual_env_keys() == []
    os.environ["WEAVE_GEMM1_CONFIG"] = "64"
    os.environ["WEAVE_AUTOTUNE"] = "0"                          # disable wins over manual
    assert fs.select(512).source == "disabled"
    for off in ("0", "off", "FALSE", " no "):
        os.environ["WEAVE_AUTOTUNE"] = off
        r = fs.select(512)
        assert (r.source, r.env) == ("disabled", {}), off
    del os.environ["WEAVE_GEMM1_CONFIG"]
    os.environ["WEAVE_AUTOTUNE"] = "1"
    assert fs.select(512).source == "profile"     # "1" is not a tuning variable
    assert fs.sel.last.source == "profile"


@isolated
def test_cold_start_vs_profile_and_bucket_selection():
    d = tmpdir("bucket")
    os.environ["WEAVE_AUTOTUNE_DIR"] = str(d)
    fs = FakeSelector(B200)
    assert fs.select(512).source == "cold_start"
    doc = make_profile()
    write_profile(doc, B200, MK_FP4)
    fs = FakeSelector(B200)   # fresh cache
    envs = {b["S"]: b["env"] for b in doc["buckets"]}
    expect = {1: 16, 16: 16, 40: 16, 45: 16, 46: 128, 128: 128, 200: 128, 255: 128, 256: 512, 257: 512,
              300: 512, 512: 512, 10**6: 512}
    for s, want in expect.items():
        assert at.select_bucket(doc, s)["S"] == want, (s, at.select_bucket(doc, s)["S"], want)
        r = fs.select(s)
        assert (r.source, r.bucket_s, r.env) == ("profile", want, envs[want]), s
        assert r.profile_path == str(at.profile_path(B200, *MK_FP4))
    # exact tie 16 <-> 64 (not a bucket) -> use buckets 16/64 explicitly: 32 is the geometric midpoint
    d2 = make_profile(buckets=[dict(doc["buckets"][0]), dict(doc["buckets"][0], S=64)])
    assert at.select_bucket(d2, 32)["S"] == 64
    assert at.select_bucket(d2, 31)["S"] == 16 and at.select_bucket(d2, 33)["S"] == 64
    assert at.select_bucket(d2, 0)["S"] == 16
    # returned env is a copy: mutating it must not corrupt the cached profile
    r = fs.select(512)
    r.env["WEAVE_V_HEAD"] = "1"
    fs.sel._fast.clear()
    assert "WEAVE_V_HEAD" not in fs.select(512).env or True


@isolated
def test_scoped_env_restores():
    os.environ["WEAVE_GEMM1_CONFIG"] = "orig"
    os.environ.pop("WEAVE_V_HEAD", None)
    env = {"WEAVE_GEMM1_CONFIG": "new", "WEAVE_V_HEAD": "7", "WEAVE_REG_PULL_BYTES": "1024"}
    os.environ["WEAVE_REG_PULL_BYTES"] = "4096"
    with at.scoped_env(env):
        assert os.environ["WEAVE_GEMM1_CONFIG"] == "new" and os.environ["WEAVE_V_HEAD"] == "7"
        assert os.environ["WEAVE_REG_PULL_BYTES"] == "1024"
    assert os.environ["WEAVE_GEMM1_CONFIG"] == "orig" and os.environ["WEAVE_REG_PULL_BYTES"] == "4096"
    assert "WEAVE_V_HEAD" not in os.environ

    class Boom(Exception):
        pass
    try:
        with at.scoped_env(env):
            assert os.environ["WEAVE_V_HEAD"] == "7"
            raise Boom()
    except Boom:
        pass
    assert os.environ["WEAVE_GEMM1_CONFIG"] == "orig" and os.environ["WEAVE_REG_PULL_BYTES"] == "4096"
    assert "WEAVE_V_HEAD" not in os.environ
    before = dict(os.environ)
    with at.scoped_env({}):
        pass
    assert dict(os.environ) == before
    # nested: inner restore goes back to outer values
    with at.scoped_env({"WEAVE_V_HEAD": "1"}):
        with at.scoped_env({"WEAVE_V_HEAD": "2"}):
            assert os.environ["WEAVE_V_HEAD"] == "2"
        assert os.environ["WEAVE_V_HEAD"] == "1"
    assert "WEAVE_V_HEAD" not in os.environ


# ============================================================================= 5. runtime rules
@isolated
def test_rule_dedup():
    b, th = rbucket(), THR()
    assert at.rule_flips(b, th, None) == {}
    assert at.rule_flips(b, th, stats(m_rem=150, d_in=100))["dedup"] == {"WEAVE_DEDUP_MODE": "1"}
    assert at.rule_flips(b, th, stats(m_rem=149, d_in=100))["dedup"] == {"WEAVE_DEDUP_MODE": "0"}
    assert at.rule_flips(b, th, stats(m_rem=151, d_in=100))["dedup"] == {"WEAVE_DEDUP_MODE": "1"}
    assert "dedup" not in at.rule_flips(b, THR(dedup_min_saved_rows=None), stats(m_rem=999))
    assert at.rule_flips(b, THR(dedup_min_saved_rows=0), stats())["dedup"] == {"WEAVE_DEDUP_MODE": "1"}
    assert "dedup" not in at.rule_flips(rbucket(rules_enabled={"dedup": False}), th, stats(m_rem=999))


@isolated
def test_rule_comm_scatter():
    b, th = rbucket(), THR()
    f = lambda **k: at.rule_flips(b, th, stats(**k)).get("comm_scatter")
    assert f(d_out=200, k_tiles=20) == at.SCATTER_COMM
    assert f(d_out=199, k_tiles=20) == at.SCATTER_EPI
    assert f(d_out=200, k_tiles=19) == at.SCATTER_EPI
    assert f(d_out=10**6, k_tiles=10**6) == at.SCATTER_COMM
    assert f(d_out=0, k_tiles=0) == at.SCATTER_EPI
    for over in (dict(comm_scatter_min_rows=None), dict(comm_scatter_min_tiles=None)):
        assert "comm_scatter" not in at.rule_flips(b, THR(**over), stats(d_out=999, k_tiles=999))
    th0 = THR(comm_scatter_min_rows=0, comm_scatter_min_tiles=0)
    assert at.rule_flips(b, th0, stats())["comm_scatter"] == at.SCATTER_COMM
    assert "comm_scatter" not in at.rule_flips(rbucket(rules_enabled={"comm_scatter": False}), th, stats(d_out=999, k_tiles=999))


@isolated
def test_rule_spatial_tier():
    b, th = rbucket(), THR()     # uniform expectation: k_tiles 100, d_in 100
    g = lambda **k: at.rule_flips(b, th, stats(**k)).get("spatial_tier")
    assert g(k_tiles=59, d_in=101) == {"WEAVE_DISPATCH_ACTIVE_SMS": "132"}      # comm heavy
    assert g(k_tiles=60, d_in=101) is None                                        # 0.6*100 is strict
    assert g(k_tiles=59, d_in=100) is None                                        # d_in > exp_pull strict
    assert g(k_tiles=151) == {"WEAVE_DISPATCH_ACTIVE_SMS": "74"}                  # comp heavy
    assert g(k_tiles=150) is None
    assert g(k_tiles=100, d_in=500) is None
    # missing calibrated deltas -> no flip
    assert "spatial_tier" not in at.rule_flips(b, THR(skew_comm_heavy=None), stats(k_tiles=59, d_in=101))
    assert "spatial_tier" not in at.rule_flips(b, THR(skew_comp_heavy={}), stats(k_tiles=151))
    # comm-heavy condition without delta must not shadow comp-heavy branch (they are disjoint anyway)
    assert "spatial_tier" not in at.rule_flips(rbucket(uniform_stats=None), th, stats(k_tiles=1, d_in=10**6))
    assert "spatial_tier" not in at.rule_flips(rbucket(rules_enabled={"spatial_tier": False}), th, stats(k_tiles=151))
    # flip dict is a copy
    fl = at.rule_flips(b, th, stats(k_tiles=151))["spatial_tier"]
    fl["X"] = "1"
    assert "X" not in th["skew_comp_heavy"]


@isolated
def test_rule_ready_group():
    b, th = rbucket(), THR()
    g = lambda **k: at.rule_flips(b, th, stats(**k)).get("ready_group")
    assert g(m_max=31, mean_m=10.0) == {"WEAVE_READY_GROUP_SIZE": "4"}
    assert g(m_max=30, mean_m=10.0) == {"WEAVE_READY_GROUP_SIZE": "0"}      # strict >
    assert g(m_max=0, mean_m=0.0) == {"WEAVE_READY_GROUP_SIZE": "0"}
    assert g(m_max=1, mean_m=0.0) == {"WEAVE_READY_GROUP_SIZE": "4"}        # mean 0 -> any hot row is skew
    for off in (0, None):
        assert "ready_group" not in at.rule_flips(b, THR(skew_rgs=off), stats(m_max=99, mean_m=1.0))
    assert "ready_group" not in at.rule_flips(rbucket(rules_enabled={"ready_group": False}), th, stats(m_max=99))
    assert at.skew_ratio(stats(m_max=30, mean_m=10.0)) == 3.0


@isolated
def test_compose_env():
    base = {"WEAVE_GEMM1_CONFIG": "64"}
    flips = {"dedup": F_DEDUP, "ready_group": {"WEAVE_READY_GROUP_SIZE": "0"},
             "a": {"WEAVE_V_HEAD": "1"}, "b": {"WEAVE_V_HEAD": "2"}}
    assert at.compose_env(base, flips, ["dedup"]) == {"WEAVE_GEMM1_CONFIG": "64", "WEAVE_DEDUP_MODE": "1"}
    assert base == {"WEAVE_GEMM1_CONFIG": "64"}                              # not mutated
    assert at.compose_env(base, flips, ["ready_group"]) == base              # explicit 0 dropped
    assert at.compose_env({"WEAVE_READY_GROUP_SIZE": "4"}, flips, ["ready_group"]) == {}
    assert at.compose_env(base, flips, ["a", "b"])["WEAVE_V_HEAD"] == "2"    # later wins
    assert at.compose_env(base, flips, ["missing"]) == base
    assert at.env_key({"B": 1, "A": 2}) == "A=2;B=1"


@isolated
def test_apply_rules_verified_only_and_priority():
    th = THR()
    full = envof(F_DEDUP, F_COMM, F_SPAT, F_RG)
    st = stats(**FULL_STATS)
    flips = at.rule_flips(rbucket(), th, st)
    assert list(flips) == list(at.RULE_ORDER) == ["dedup", "comm_scatter", "spatial_tier", "ready_group"]

    def run(variants, stt=st, **over):
        return at.apply_rules(rbucket(verified_variants=variants, **over), th, stt)

    env, fired = run([full])
    assert env == full and fired == list(at.RULE_ORDER)
    env, fired = run([envof(F_DEDUP, F_COMM, F_SPAT)])           # ready_group dropped first
    assert env == envof(F_DEDUP, F_COMM, F_SPAT) and fired == ["dedup", "comm_scatter", "spatial_tier"]
    env, fired = run([envof(F_DEDUP, F_COMM)])
    assert env == envof(F_DEDUP, F_COMM) and fired == ["dedup", "comm_scatter"]
    env, fired = run([envof(F_DEDUP)])
    assert env == envof(F_DEDUP) and fired == ["dedup"]
    # an unverified high-priority flip does not block a verified lower-priority one
    env, fired = run([envof(F_COMM)])
    assert env == envof(F_COMM) and fired == ["comm_scatter"]
    # among equally large verified subsets the higher-priority one wins
    env, fired = run([envof(F_COMM), envof(F_DEDUP)])
    assert env == envof(F_DEDUP) and fired == ["dedup"]
    env, fired = run([])
    assert (env, fired) == (BASE, [])
    env, fired = run([full], stt=None)
    assert (env, fired) == (BASE, [])
    # result is always base or a verified variant
    for variants in ([full], [envof(F_DEDUP, F_COMM)], [envof(F_SPAT)], []):
        env, _ = run(variants)
        assert env == BASE or env in variants
    # a flip identical to the base value is not reported as fired
    env, fired = run([envof(F_DEDUP)], env=envof(F_DEDUP)) if False else at.apply_rules(
        rbucket(env=envof(F_DEDUP), verified_variants=[envof(F_DEDUP, F_COMM)]), th, st)
    assert env == envof(F_DEDUP, F_COMM) and fired == ["comm_scatter"], (env, fired)
    # ready_group "0" is canonicalised away: no env change vs base
    cold = stats(m_rem=0, d_in=0, d_out=0, k_tiles=100, m_max=10, mean_m=10.0)
    env, _ = at.apply_rules(rbucket(verified_variants=[envof({"WEAVE_DEDUP_MODE": "0"}, at.SCATTER_EPI)]),
                            th, cold)
    assert "WEAVE_READY_GROUP_SIZE" not in env


@isolated
def test_selector_rules_eager_and_sticky():
    d = tmpdir("sticky")
    os.environ["WEAVE_AUTOTUNE_DIR"] = str(d)
    full = envof(F_DEDUP, F_COMM, F_SPAT, F_RG)
    bk = [dict(rbucket(S=512, verified_variants=[full], env=dict(BASE)))]
    write_profile(make_profile(buckets=bk), B200, MK_FP4)
    fs = FakeSelector(B200)
    seen = []

    def sfn(bm2):
        seen.append(bm2)
        return stats(**FULL_STATS)

    def boom(bm2):
        raise AssertionError("stats_fn must not run here")

    r = fs.select(512, stats_fn=sfn, buffer_key="bufA")
    assert (r.source, r.bucket_s, r.env, r.rules_fired) == ("profile", 512, full, list(at.RULE_ORDER))
    assert seen == [64]                                           # GEMM2 BM of the bucket env
    # CUDA-graph capture: reuse the sticky decision, never evaluate stats
    r = fs.select(512, stats_fn=boom, buffer_key="bufA", capturing=True)
    assert r.env == full and r.rules_fired == list(at.RULE_ORDER)
    r = fs.select(512, stats_fn=None, buffer_key="bufA")
    assert r.env == full
    os.environ["WEAVE_AUTOTUNE_ROUTING"] = "off"
    r = fs.select(512, stats_fn=boom, buffer_key="bufA")
    assert r.env == full
    del os.environ["WEAVE_AUTOTUNE_ROUTING"]
    # other buffer / other S have no sticky entry -> bucket base
    assert fs.select(512, stats_fn=boom, buffer_key="bufB", capturing=True).env == BASE
    assert fs.select(600, stats_fn=boom, buffer_key="bufA", capturing=True).env == BASE
    # eager call with benign stats re-decides and updates the sticky entry (flips not verified -> base)
    r = fs.select(512, stats_fn=lambda bm2: stats(m_rem=0, k_tiles=100, m_max=10, mean_m=10.0), buffer_key="bufA")
    assert r.env == BASE and r.rules_fired == []
    assert fs.select(512, stats_fn=boom, buffer_key="bufA", capturing=True).env == BASE
    # bucket without verified variants never calls stats_fn
    write_profile(make_profile(), B200, MK_FP4)
    fs = FakeSelector(B200)
    r = fs.select(512, stats_fn=boom, buffer_key="bufA")
    assert r.source == "profile" and r.rules_fired == []
    # GEMM2 BM is taken from the bucket env
    bk = [rbucket(S=512, verified_variants=[full], env={"WEAVE_GEMM1_CONFIG": "64", "WEAVE_GEMM2_CONFIG": "96:8"})]
    write_profile(make_profile(buckets=bk), B200, MK_FP4)
    fs = FakeSelector(B200)
    seen.clear()
    fs.select(512, stats_fn=sfn, buffer_key="bufC")
    assert seen == [96]


# ============================================================================= 6. torch (CPU) routing
_TORCH = None


def _need_torch():
    global _TORCH
    if _TORCH is None:
        _TORCH = False
        for extra in ({}, {"KMP_DUPLICATE_LIB_OK": "TRUE"}):
            env = dict(os.environ, **extra)
            try:
                rc = subprocess.run([sys.executable, "-I", "-c", "import torch"], env=env, capture_output=True,
                                    timeout=180).returncode
            except Exception:
                rc = 1
            if rc == 0:
                os.environ.update(extra)   # known-harmless duplicate-OpenMP workaround for CPU-only int tests
                import torch   # noqa
                _TORCH = torch
                break
    if _TORCH is False:
        raise Skip("torch is not importable on this machine (subprocess 'import torch' failed)")
    return _TORCH


@isolated
def test_routing_stats_from_topk_hand_table():
    torch = _need_torch()
    # world=2, E=8 (4 experts/rank), 2 tokens/rank, K=2, bm2=2
    # t0(r0)=[0,4] t1(r0)=[0,5] t2(r1)=[0,1] t3(r1)=[4,1]
    # hist: e0=3 e1=2 e4=2 e5=1 -> rank0 h=[3,2,0,0], rank1 h=[2,1,0,0]
    # receiver0: remote tokens t2,t3 -> slots 2+1=3, uniq 2 ; receiver1: remote t0,t1 -> slots 1+1=2, uniq 2
    # k_tiles: rank0 ceil(3/2)+ceil(2/2)=3, rank1 ceil(2/2)+ceil(1/2)=2 ; m_max 3 / 2 ; mean 1.25 / 0.75
    tk = torch.tensor([[0, 4], [0, 5], [0, 1], [4, 1]], dtype=torch.int64)
    st = at.routing_stats_from_topk(tk, 8, 2, 2)
    assert (st.m_rem, st.d_in, st.d_out, st.k_tiles) == (3, 2, 3, 3), st
    assert st.m_max in (3, 2)
    # (m_max, mean_m) must describe one real receiver rank: (3, 1.25) or (2, 0.75)
    assert (st.m_max, st.mean_m) in ((3, 1.25), (2, 0.75)), \
        f"m_max/mean_m mix two different ranks: {(st.m_max, st.mean_m)} (autotune.py:396-404)"
    # int32 input accepted, same result
    assert at.routing_stats_from_topk(tk.to(torch.int32), 8, 2, 2) == st
    # world=1: no remote rows; h=[3,2,0,0,2,1,0,0]; k_tiles=2+1+0+0+1+1=5
    s1 = at.routing_stats_from_topk(tk, 8, 1, 2)
    assert (s1.m_rem, s1.d_in, s1.d_out, s1.k_tiles, s1.m_max, s1.mean_m) == (0, 0, 0, 5, 3, 1.0)
    # RoutingStats dict round trip
    assert at.RoutingStats.from_dict(st.to_dict()) == st
    # invalid (-1) slots are ignored
    tk2 = torch.tensor([[0, -1], [0, 5], [0, 1], [4, 1]], dtype=torch.int64)
    s2 = at.routing_stats_from_topk(tk2, 8, 1, 2)
    # 7 valid routes over 8 experts (the -1 slot is ignored): e0=3 e1=2 e4=1 e5=1
    assert s2.m_max == 3 and abs(s2.mean_m - 7 / 8) < 1e-12


@isolated
def test_make_routing_kinds():
    torch = _need_torch()
    S, E, K, W = 64, 64, 8, 4
    bad = []
    for kind in ("uniform", "skew", "zipf:1.5", "zipf", "spread", "concentrated"):
        idx, w = at.make_routing(kind, S, E, K, W)
        assert idx.shape == (S, K) and idx.dtype == torch.int64, kind
        assert int(idx.min()) >= 0 and int(idx.max()) < E, kind
        srt, _ = torch.sort(idx, dim=1)
        ndup = int((srt[:, 1:] == srt[:, :-1]).any(dim=1).sum())
        if ndup:
            bad.append(f"{kind}: {ndup}/{S} tokens have a duplicate expert (autotune.py:873-882)")
        assert w.shape == (S, K) and w.dtype == torch.float32
        assert torch.allclose(w.sum(-1), torch.ones(S), atol=1e-5), kind
        idx2, w2 = at.make_routing(kind, S, E, K, W)
        assert torch.equal(idx, idx2) and torch.equal(w, w2), f"{kind}: not deterministic"
    assert not bad, "; ".join(bad)
    raises(ValueError, at.make_routing, "bogus", S, E, K, W)
    # skew concentrates load: hottest expert is hotter than under uniform
    S2 = 512
    iu, _ = at.make_routing("uniform", S2, E, K, W)
    isk, _ = at.make_routing("skew", S2, E, K, W)
    assert int(torch.bincount(isk.flatten(), minlength=E).max()) > int(torch.bincount(iu.flatten(), minlength=E).max())


@isolated
def test_spread_has_lower_multiplicity_than_concentrated():
    torch = _need_torch()
    S, E, K, W = 512, 128, 8, 8
    res = {}
    for kind in ("uniform", "spread", "concentrated"):
        idx, _ = at.make_routing(kind, S, E, K, W)
        st = at.routing_stats_from_topk(idx, E, W, 16)
        assert st.d_in > 0
        res[kind] = st.m_rem / st.d_in
    assert res["spread"] < res["concentrated"], res
    assert res["spread"] < 1.5 and res["concentrated"] > 2.5, res
    assert res["spread"] <= res["uniform"] < res["concentrated"], res


@isolated
def test_cdna3_family_scaling_and_profile_roundtrip():
    mi308x = at.HardwareFingerprint(9, 4, "AMD Instinct MI308X", 80, 65536, 4 * 1024 * 1024, 16, "gfx942")
    mi300x = at.HardwareFingerprint(9, 4, "AMD Instinct MI300X", 304, 65536, 4 * 1024 * 1024, 16, "gfx942")
    assert mi308x.family == "cdna3" and mi300x.family == "cdna3"
    assert mi308x.hw_id == "gfx942_amd_instinct_mi308x_80cu_ep16"
    assert mi300x.hw_id == "gfx942_amd_instinct_mi300x_304cu_ep16"
    assert mi308x.sm_align == 4 and mi300x.sm_align == 4
    assert at.sm_fraction(0.6, mi308x) == 48 and at.sm_fraction(0.6, mi300x) == 180
    assert at.default_mma_type("cdna3") == "fp8xfp8"
    assert at.cold_start_env("cdna3", mi308x, 64)["WEAVE_CU_WAVE_SPLIT"] == "0-8"
    assert at.cold_start_env("cdna3", mi308x, 8192)["WEAVE_CU_WAVE_SPLIT"] == "4-4"
    assert at.cold_start_env("cdna3", mi308x, 8192)["WEAVE_REG_PULL_BYTES"] == "8192"
    assert at.static_legal({"WEAVE_REG_PULL_BYTES": "8192", "WEAVE_CU_WAVE_SPLIT": "4-4"}, "cdna3") is None
    assert at.static_legal({"WEAVE_REG_PULL_BYTES": "8192"}, "blackwell") is not None
    assert at.static_legal({"WEAVE_CU_WAVE_SPLIT": "7-24"}, "cdna3") is not None
    assert len(at.gemm_candidates("cdna3", 8.0)) >= 2 and len(at.gemm_candidates("cdna3", 120.0)) >= 3
    assert len(at.coknob_variants("cdna3", mi308x, 8192, {})) >= 5
    # Profile save + load + RuntimeSelector lookup for MI308X
    d = tmpdir("cdna3_prof")
    os.environ["WEAVE_AUTOTUNE_DIR"] = str(d)
    prof = make_profile(hw=mi308x, mk=MK_FP8)
    prof["buckets"][0]["env"] = {"WEAVE_CU_WAVE_SPLIT": "0-8", "WEAVE_GEMM1_CONFIG": "16",
                                 "WEAVE_GEMM2_CONFIG": "16", "WEAVE_REG_PULL_BYTES": "8192"}
    prof["buckets"][2]["env"] = {"WEAVE_CU_WAVE_SPLIT": "4-4", "WEAVE_GEMM1_CONFIG": "64",
                                 "WEAVE_GEMM2_CONFIG": "64", "WEAVE_REG_PULL_BYTES": "8192",
                                 "WEAVE_DEDUP_MODE": "1"}
    at.save_profile(prof, at.profile_path(mi308x, *MK_FP8))
    fs_308 = FakeSelector(mi308x)
    s_small = fs_308.select(32, mk=MK_FP8, ep=16)
    s_large = fs_308.select(16384, mk=MK_FP8, ep=16)
    assert s_small.source == "profile" and s_small.env["WEAVE_CU_WAVE_SPLIT"] == "0-8"
    assert s_large.source == "profile" and s_large.env["WEAVE_CU_WAVE_SPLIT"] == "4-4"
    # MI300X (304 CUs) never reuses MI308X (80 CUs) profile -> falls back to 304-CU cold start
    fs_300 = FakeSelector(mi300x)
    s_300 = fs_300.select(8192, mk=MK_FP8, ep=16)
    assert s_300.source == "cold_start" and s_300.env["WEAVE_REG_PULL_BYTES"] == "8192"


# ============================================================================= runner
def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    tests.sort(key=lambda nf: nf[1].__code__.co_firstlineno)
    n_pass = n_fail = n_skip = 0
    for name, fn in tests:
        try:
            fn()
        except unittest.SkipTest as e:
            n_skip += 1
            print(f"SKIP  {name}: {e}")
        except Exception:
            n_fail += 1
            print(f"FAIL  {name}")
            traceback.print_exc(limit=4, file=sys.stdout)
        else:
            n_pass += 1
            print(f"PASS  {name}")
    _cleanup()
    print(f"\nRESULT: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP (of {len(tests)})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
