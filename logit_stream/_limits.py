"""Resource limits for a fit: the `n_threads=` option, and the peak memory the
fit used. The code is hdfe-stream's (`hdfe_stream.core.resources`); see
"Resource limits and profiling" in its documentation."""

from __future__ import annotations

import functools

from hdfe_stream.core.profile import Profile
from hdfe_stream.core.resources import peak_rss_gb, reset_peak_rss, thread_scope


def limited(fit):
    """Give an estimator's fit function an ``n_threads`` keyword that holds
    numba and BLAS to that many threads for the fit (restored afterwards; Polars
    cannot be limited per fit), and record ``diagnostics["peak_rss_gb"]``."""

    @functools.wraps(fit)
    def wrapper(*args, n_threads=None, **kwargs):
        reset_peak_rss()
        with thread_scope(n_threads):
            out = fit(*args, **kwargs)
        rss = peak_rss_gb()
        rss = None if rss is None else round(rss, 3)
        if isinstance(out, Profile):
            return out
        for r in ([out] if hasattr(out, "diagnostics") else out):
            r.diagnostics["peak_rss_gb"] = rss
        return out

    return wrapper
