"""`profile=True` on every estimator: a `Profile` instead of a fit, P from the
model and the outcome's levels, and an upper bound that holds."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import polars as pl
import pytest
import stream_limits
from hdfe_stream import Profile

import logit_stream as ls

# These tests compare the profile with a measured peak, so they need a peak-memory reading (Linux,
# macOS and, from stream-limits 0.1.1, Windows). The model was calibrated on Linux and checked on
# macOS; on Windows a failure here is information about the model, not a broken test.
needs_peak_rss = pytest.mark.skipif(
    stream_limits.peak_rss_gb() is None, reason="peak memory cannot be read on this platform")


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(3)
    n = 20_000
    d = pl.DataFrame({f"x{i}": rng.normal(size=n) for i in range(1, 5)})
    return d.with_columns(
        g=pl.Series(rng.integers(0, 400, n)), y5=pl.Series(rng.integers(0, 5, n)),
        d=pl.Series((rng.random(n) < 0.4).astype("int8")))


@pytest.fixture(scope="module")
def choices():
    rng = np.random.default_rng(4)
    cases, alts = 3000, 4
    n = cases * alts
    case = np.repeat(np.arange(cases), alts)
    alt = np.tile(np.arange(alts), cases)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    u = 0.8 * x1 - 0.5 * x2 + rng.gumbel(size=n)
    best = u.reshape(cases, alts).argmax(axis=1)
    return pl.DataFrame({"case": case, "alt": alt, "x1": x1, "x2": x2,
                         "chosen": (alt == best[case]).astype(np.int8),
                         "nest": np.where(alt < 2, "a", "b")})


def test_parameter_counts_follow_the_model(df):
    m = ls.mlogit_stream("y5 ~ x1 + x2", df, profile=True, verbose=False)
    assert isinstance(m, Profile) and m.notes["P"] == 3 * 4 and m.notes["J"] == 5   # (K + 1) x (J - 1)
    o = ls.ologit_stream("y5 ~ x1 + x2", df, profile=True, verbose=False)
    assert o.notes["P"] == 2 + 4                                                    # K + J - 1
    assert o.memory_gb[0] <= o.memory_gb[1]


def test_levels_option_skips_the_outcome_scan(df):
    p = ls.mlogit_stream("y5 ~ x1", df, levels=[0, 1, 2], profile=True, verbose=False)
    assert p.notes["J"] == 3


def test_no_run_directory_and_no_fit(df, tmp_path):
    ls.mlogit_stream("y5 ~ x1 + x2", df, workdir=tmp_path, profile=True, verbose=False)
    assert list(tmp_path.iterdir()) == []


def test_fixed_effect_and_cluster_arrays(df):
    p = ls.mlogit_stream("y5 ~ x1 + x2 | g", df, vcov={"CRV1": "g"}, profile=True, verbose=False)
    names = [c.name for c in p.components]
    assert any("cross blocks" in n for n in names) and any(n.startswith("cluster scores") for n in names)


def test_crv3_is_reported(df):
    p = ls.mlogit_stream("y5 ~ x1 + x2", df, vcov={"CRV3": "g"}, profile=True, verbose=False)
    assert any(c.name.startswith("CRV3") for c in p.components)


def test_binary_logit_and_probit_use_hdfe_streams_profile(df):
    for f in (ls.logit_stream, ls.probit_stream):
        p = f("d ~ x1 | g", df, profile=True, verbose=False)
        assert isinstance(p, Profile) and "StreamingGLM" in p.model


def test_conditional_nested_and_mixed(choices):
    c = ls.clogit_stream("chosen ~ x1 + x2", choices, case="case", profile=True, verbose=False)
    assert c.notes["P"] == 2 and c.notes["cases"] > 2000
    n = ls.nlogit_stream("chosen ~ x1 + x2", choices, case="case", alt="alt",
                         nests={"a": [0, 1], "b": [2, 3]}, profile=True, verbose=False)
    assert n.notes["P"] == 2 + 2
    m = ls.mixlogit_stream("chosen ~ x1 + x2", choices, case="case", random={"x1": "normal"},
                           draws=50, profile=True, verbose=False)
    assert m.notes["P"] == 3
    u = ls.cfelogit_stream("chosen ~ x1 + x2 | case", choices, profile=True, verbose=False)
    assert isinstance(u, Profile)


def test_several_fixed_effects_in_mlogit_profile_the_expanded_poisson_fit(df):
    p = ls.mlogit_stream("y5 ~ x1 + x2 | g + y5b", df.with_columns(y5b=pl.col("g") % 7),
                         profile=True, verbose=False)
    assert isinstance(p, Profile) and p.notes["route"] == "poisson"
    assert p.notes["J"] == 5 and p.notes["K"] == 2
    assert p.notes["expanded_rows"] == 5 * len(df) and p.scan.n_rows == 5 * len(df)
    assert any("expanded" in w for w in p.warnings)
    # K x (J - 1) covariates in the expanded regression, a group of J rows per observation
    assert p.notes["m"] == 1 + 2 * 4
    clustered = ls.mlogit_stream("y5 ~ x1 + x2 | g + y5b", df.with_columns(y5b=pl.col("g") % 7),
                                 vcov={"CRV1": "g"}, profile=True, verbose=False)
    assert any(c.name.startswith("cluster scores") or c.name.startswith("CRV")
               for c in clustered.components) or clustered.memory_gb[1] >= p.memory_gb[1] - 0.05


def test_threads_scale_the_per_thread_terms(df):
    a = ls.mlogit_stream("y5 ~ x1 + x2", df, profile=True, verbose=False, n_threads=1)
    b = ls.mlogit_stream("y5 ~ x1 + x2", df, profile=True, verbose=False, n_threads=4)
    # the request is capped at the machine's cores (a 3-core CI runner gives 3)
    assert a.threads == 1 and 1 < b.threads <= 4
    pt = lambda p: sum(c.expected for c in p.components if c.name.startswith("per-thread"))
    assert pt(b) == pytest.approx(b.threads * pt(a))


def test_validation_still_happens(df):
    with pytest.raises(ValueError, match="storage"):
        ls.mlogit_stream("y5 ~ x1", df, storage="tape", profile=True, verbose=False)


MEASURE = r'''
import json, sys, numpy as np, polars as pl
import logit_stream as ls
fn, fml, kw = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
rng = np.random.default_rng(0); n = 300_000
df = pl.DataFrame({f"x{i}": rng.normal(size=n) for i in range(1, 5)}).with_columns(
    g=pl.Series(rng.integers(0, 3000, n)), y5=pl.Series(rng.integers(0, 5, n)),
    h=pl.Series(rng.integers(0, 12, n)))
f = getattr(ls, fn)
p = f(fml, df, profile=True, verbose=False, **kw)
r = f(fml, df, verbose=False, **kw)
print(json.dumps({"mem": p.memory_gb, "rss": r.diagnostics["peak_rss_gb"],
                  "disk": p.disk_gb, "disk_peak": r.diagnostics.get("disk_peak_gb")}))
'''


@needs_peak_rss
@pytest.mark.parametrize("fn,fml,kw", [
    ("mlogit_stream", "y5 ~ x1 + x2 + x3", {}),
    ("mlogit_stream", "y5 ~ x1 + x2 | g", {}),
    ("mlogit_stream", "y5 ~ x1 + x2", {"vcov": {"CRV1": "g"}}),
    ("ologit_stream", "y5 ~ x1 + x2 + x3", {}),
    ("mlogit_stream", "y5 ~ x1 + x2 | g + h", {}),
])
def test_the_upper_bound_holds(fn, fml, kw):
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", MEASURE, fn, fml, json.dumps(kw)],
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-500:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["rss"] <= r["mem"][1]
    # a loose check that the estimate is not wildly pessimistic. A bare process holds less resident
    # memory on Windows than the Linux-calibrated baseline allowance (0.395 of expected was measured)
    assert r["rss"] >= (0.3 if sys.platform == "win32" else 0.4) * r["mem"][0]
    if r["disk_peak"] is not None:          # the several-fixed-effects route does not record it
        assert r["disk_peak"] <= r["disk"][1] + 0.01
