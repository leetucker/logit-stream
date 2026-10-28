"""logit-stream: logit models fitted out of core by streamed Newton.

Binary (`logit_stream`, `probit_stream`: hdfe-stream's `feglm_stream`, with
any number of absorbed fixed effects), multinomial (`mlogit_stream`),
ordered (`ologit_stream`, `oprobit_stream`), conditional (`clogit_stream`),
conditional fixed-effects (`cfelogit_stream`), mixed (`mixlogit_stream`) and
nested (`nlogit_stream`) logit, from pyfixest-style formulas, on data that
need not fit in memory: Parquet files, Polars LazyFrames, or DataFrames.

Layout
    api.py           the estimators
    _frontend.py     formulas -> models (hdfe-stream's formula compiler)
    store.py         pass 0, into hdfe-stream's block store (hdfe_stream.core.store)
    models.py        the likelihoods: per chunk, log-likelihood, gradient,
                     information, scores
    newton.py        Newton's method, collinearity, the variance's middle matrix
    estimator.py     formula + source -> result, and the inference
    fe.py            one absorbed fixed effect (multinomial, ordered)
    multi_fe.py      several, for the multinomial logit (a Poisson regression)
    random_effects/  random intercepts (Laplace's approximation)
    optimize.py      Newton and quasi-Newton steps from an analytic gradient,
                     for the likelihoods that are not concave
    _choice.py       what the mixed and nested logit's fits share
    mixed.py         the mixed logit (simulated likelihood)
    nested.py        the nested logit
    device/          the passes on a GPU (CuPy)
    results.py       the result classes
    simulate.py      simulated data for examples and tests
"""

from __future__ import annotations

from hdfe_stream.core.multcomp import bonferroni, holm
from hdfe_stream.core.reporting import etable

from .api import (
    cfelogit_stream,
    clogit_stream,
    logit_stream,
    mixlogit_stream,
    mlogit_stream,
    nlogit_stream,
    ologit_stream,
    oprobit_stream,
    probit_stream,
)
from .results import (
    ConditionalFELogitResult,
    ConditionalLogitResult,
    LogitMulti,
    LogitResult,
    MixedLogitResult,
    MultinomialLogitResult,
    NestedLogitResult,
    OrderedLogitResult,
    OrderedProbitResult,
)


def _version():
    try:
        from importlib.metadata import version
        return version("logit-stream")
    except Exception:                   # noqa: BLE001 -- not installed
        return "0.1.0"


__version__ = _version()

__all__ = [
    "logit_stream", "probit_stream", "mlogit_stream", "ologit_stream", "oprobit_stream",
    "clogit_stream", "cfelogit_stream", "mixlogit_stream", "nlogit_stream",
    "LogitResult", "MultinomialLogitResult", "OrderedLogitResult", "OrderedProbitResult",
    "ConditionalLogitResult", "ConditionalFELogitResult",
    "MixedLogitResult", "NestedLogitResult",
    "LogitMulti", "etable", "bonferroni", "holm", "__version__",
]
