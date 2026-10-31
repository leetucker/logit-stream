"""Memory guards follow the headroom (limit, less a reserve, less what the
process holds); `max_memory_gb` sets the limit. What cannot shrink is refused
up front, with the numbers."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import logit_stream as ls


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(5)
    n = 30_000
    return pl.DataFrame({
        "x1": rng.normal(size=n), "x2": rng.normal(size=n),
        "g": rng.integers(0, 300, n), "y3": rng.integers(0, 3, n)})


def test_fixed_effect_blocks_that_do_not_fit_are_refused(df):
    ok = ls.mlogit_stream("y3 ~ x1 + x2 | g", df, verbose=False, max_memory_gb=500)
    assert len(ok.coefnames) > 0
    with pytest.raises(MemoryError, match="max_memory_gb"):
        ls.mlogit_stream("y3 ~ x1 + x2 | g", df, verbose=False, max_memory_gb=0.01)


def test_crv3_is_refused_when_it_does_not_fit(df):
    ok = ls.mlogit_stream("y3 ~ x1 + x2", df, vcov={"CRV3": "g"}, verbose=False,
                          max_memory_gb=500)
    assert len(ok.coefnames) > 0
    with pytest.raises(MemoryError, match="max_memory_gb"):
        ls.mlogit_stream("y3 ~ x1 + x2", df, vcov={"CRV3": "g"}, verbose=False,
                         max_memory_gb=0.01)


def test_the_profile_warns_where_the_fit_would_refuse(df):
    p = ls.mlogit_stream("y3 ~ x1 + x2 | g", df, profile=True, verbose=False, max_memory_gb=0.01)
    assert any("MemoryError" in w for w in p.warnings)
    assert p.memory_limit_gb == 0.01
    q = ls.mlogit_stream("y3 ~ x1 + x2 | g", df, profile=True, verbose=False, max_memory_gb=500)
    assert not any("MemoryError" in w for w in q.warnings)


def test_every_estimator_accepts_the_limit(df):
    rng = np.random.default_rng(6)
    cases, alts = 600, 3
    n = cases * alts
    ch = pl.DataFrame({"case": np.repeat(np.arange(cases), alts), "alt": np.tile(np.arange(alts), cases),
                       "x1": rng.normal(size=n), "chosen": rng.integers(0, 2, n).astype("int8")})
    for p in (ls.ologit_stream("y3 ~ x1", df, profile=True, verbose=False, max_memory_gb=8),
              ls.clogit_stream("chosen ~ x1", ch, case="case", profile=True, verbose=False,
                               max_memory_gb=8),
              ls.nlogit_stream("chosen ~ x1", ch, case="case", alt="alt",
                               nests={"a": [0, 1], "b": [2]}, profile=True, verbose=False,
                               max_memory_gb=8),
              ls.mixlogit_stream("chosen ~ x1", ch, case="case", random={"x1": "normal"},
                                 draws=20, profile=True, verbose=False, max_memory_gb=8)):
        assert p.memory_limit_gb == 8
