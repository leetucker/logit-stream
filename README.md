# logit-stream

Logit models for data that do not fit in memory: binary, multinomial,
ordered, conditional, nested and mixed logit, fitted by Newton's method with
the data streamed off disk, from
[pyfixest](https://github.com/py-econometrics/pyfixest)-style formulas, with
absorbed fixed effects or random intercepts.

```python
from logit_stream import mlogit_stream

fit = mlogit_stream(
    "race ~ C(agecat) + female + C(inccat) + C(cit) | puma",
    "acs/*.parquet",               # a Parquet path or glob, never fully loaded
    vcov={"CRV1": "puma"},
)
fit.summary()
fit.predict("people/*.parquet").sink_parquet("imputed.parquet")   # lazily
```

| | |
|---|---|
| `logit_stream`, `probit_stream` | binary logit and probit with any number of fixed effects ([hdfe-stream](https://github.com/leetucker/hdfe-stream)'s `feglm_stream`) |
| `mlogit_stream` | multinomial logit (any number of fixed effects; several through a Poisson regression) |
| `ologit_stream`, `oprobit_stream` | ordered logit and probit |
| `clogit_stream` | conditional (McFadden) logit on long-format choice data |
| `cfelogit_stream` | Chamberlain's conditional fixed-effects logit for panels |
| `mixlogit_stream`, `nlogit_stream` | mixed and nested logit (prototypes) |

Newton's method for a logit needs the data only through sums over rows -- the
log-likelihood, its gradient and the information matrix -- so each step is one
read of the data, block by block, and one dense solve; the likelihoods are
concave, so a handful of steps suffice. Memory grows with the number of
coefficients, not of rows. The multinomial and ordered logit absorb one fixed
effect exactly, by eliminating its per-level blocks of the information matrix
in every step.

Results share [hdfe-stream](https://github.com/leetucker/hdfe-stream)'s
reporting: `summary()`, `tidy()`, `summary_dict()`, `summary_json()`,
`confint()`, `wald_test()`, `with_vcov()`, `etable()`, and lazy `predict()`,
`resid()`, `sample()` and `fixef()` frames. Estimates and standard errors
match statsmodels and scikit-learn (see `tests/`).

Documentation: <https://leetucker.github.io/logit-stream/> (build locally with
`pip install -e ".[docs]"` and `sphinx-build -b html docs docs/_build/html`).
`examples/` has a runnable script for each page of its guide, on simulated
data.

## Install

```bash
pip install logit-stream
```

Python 3.10 or later.

## Benchmarks

`benchmarks/` compares logit-stream with scikit-learn, statsmodels and mord on
American Community Survey microdata (10,000 to 12.9 million adults), and with
xlogit, pylogit, statsmodels and jaxlogit on the Swissmetro choice survey,
resampled to 10 million choice sets. See `benchmarks/README.md` and the
documentation's benchmark pages.

## License

MIT.
