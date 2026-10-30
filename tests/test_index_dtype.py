"""Polars' 64-bit runtime (hdfe-stream issue #5): row indices are UInt64.

`with_row_index` returns UInt32 on the default runtime and UInt64 with
`polars-runtime-64`, whatever the size of the frame. hdfe-stream coded levels
with it, which broke its fits (and so the binary and multi-fixed-effect
routes here). The runtime is chosen at install time, so the test makes
`with_row_index` return UInt64 on any install; with that runtime installed
the patch changes nothing and the test passes all the same.
"""

import numpy as np
import polars as pl
import pytest

from logit_stream import logit_stream, mlogit_stream


def _widen_row_index(monkeypatch):
    for cls in (pl.DataFrame, pl.LazyFrame):
        original = cls.with_row_index

        def with_row_index(self, name="index", offset=None, *, _original=original):
            out = _original(self, name, offset)
            return out.with_columns(pl.col(name).cast(pl.UInt64))

        monkeypatch.setattr(cls, "with_row_index", with_row_index)
    assert pl.DataFrame({"a": [1]}).with_row_index().schema["index"] == pl.UInt64


def _fits(df):
    binary = df.with_columns((pl.col("y") > 1).cast(pl.Int64).alias("b"))
    return {
        "logit_fe": logit_stream("b ~ x1 + x2 | g", binary, verbose=False),
        "mlogit_fe": mlogit_stream("y ~ x1 + x2 | g", df, verbose=False),
        # two fixed effects take the Poisson route through hdfe-stream
        "mlogit_two_fe": mlogit_stream("y ~ x1 + x2 | g + cl", df, verbose=False),
    }


@pytest.mark.parametrize("name", ["logit_fe", "mlogit_fe", "mlogit_two_fe"])
def test_fixed_effect_fits_with_uint64_row_index(fe_data, monkeypatch, name):
    native = _fits(fe_data)[name]
    with monkeypatch.context() as m:
        _widen_row_index(m)
        wide = _fits(fe_data)[name]
        assert wide.coefnames == native.coefnames
        np.testing.assert_allclose(wide.beta, native.beta, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(wide.se, native.se, rtol=1e-7)
        assert wide.loglik == pytest.approx(native.loglik, rel=1e-10)
        got = wide.predict(fe_data).collect()
        assert got.height == fe_data.height
