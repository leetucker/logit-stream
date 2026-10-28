# logit-stream

Logit models for data that do not fit in memory -- binary
(`logit_stream`, `probit_stream`), multinomial (`mlogit_stream`), ordered
(`ologit_stream`, `oprobit_stream`), conditional (`clogit_stream`),
Chamberlain's fixed-effects (`cfelogit_stream`), nested (`nlogit_stream`)
and mixed (`mixlogit_stream`) logit -- fitted by Newton's method with the
data streamed off disk, from
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

# impute: the probability of each level, for data of any size, lazily
fit.predict("people/*.parquet").sink_parquet("imputed.parquet")
```

The data can be a Parquet path or glob, a Polars LazyFrame, or a Polars
DataFrame. It is a companion to [hdfe-stream](https://leetucker.github.io/hdfe-stream/),
whose formula parser, result classes and reporting it shares, and whose
streamed logit with any number of fixed effects is this package's binary
logit.

## Why

Logit models are the workhorse of imputation and prediction with survey and
administrative data: race and ethnicity from names and places, education from
age, income and occupation, the choice of a mode of transport or a product.
The data such models are fitted to have grown to tens or hundreds of millions
of records, and the models have grown with them: many categorical covariates,
interactions, and a geography (a county, a census tract, a PUMA) with its own
effect for every outcome.

The standard tools need the design matrix in memory, often in several working
copies: scikit-learn's solvers, statsmodels' maximum-likelihood models, and the
discrete-choice packages (xlogit, pylogit, Biogeme). At ten million rows and a
few dozen covariates, the design alone is several gigabytes, and the solvers'
working arrays are a multiple of that.

logit-stream holds a bounded number of rows in memory at a time. It rests on
one observation: Newton's method for a logit needs the data only through sums
over rows -- the log-likelihood, its gradient and the information matrix --
and each is accumulated block by block. The likelihoods are globally concave,
so Newton's method with step-halving converges from the null model in a
handful of steps (four to eight on the benchmarks), each one read of the data
and one dense solve.

## Features

- **One interface for the logit family.** Binary logit and probit,
  multinomial logit, ordered logit and probit on ordinary data; McFadden's
  conditional logit on long-format choice data with alternative-specific
  attributes and unbalanced choice sets, and the nested and mixed logit
  (experimental) on the same data; Chamberlain's conditional fixed-effects
  logit for short panels. See the [examples](examples/index.md).
- **Absorbed fixed effects** for the multinomial and ordered logit
  (`"y ~ x | puma"`): each level gets its own intercept for every outcome
  level, solved exactly within each Newton step by eliminating the per-level
  blocks of the information matrix, with standard errors identical to the fit
  with indicators. The binary logit absorbs any number of fixed effects
  (hdfe-stream's), with an optional correction of the incidental-parameter
  bias, and so does the multinomial logit, through the equivalent Poisson
  regression.
- **Random intercepts** (experimental) for the multinomial and ordered
  logit, by Laplace's approximation, with nested or crossed groupings, as
  lme4's `glmer` and ordinal's `clmm` fit them.
- **pyfixest formulas**: `C()`, `i()`, interactions, `I()` and transforms,
  multiple estimation (`csw()`, `sw()`, several outcomes), with pyfixest's own
  coefficient names and categorical coding.
- **Standard errors**: model-based, heteroskedasticity-robust and clustered
  (one-way, by an interaction, or multi-way), and the cluster jackknife,
  with Stata's conventions for maximum likelihood, checked against
  statsmodels; frequency and analytic weights.
- **Post-estimation like hdfe-stream's**: `summary()`, `tidy()`,
  `summary_dict()`/`summary_json()`, `confint()` (pointwise or simultaneous),
  `wald_test()`, `with_vcov()`, `etable()`, and lazy `predict()`, `resid()`,
  `sample()` and `fixef()` frames.
- **Out of core**: one pass over the source writes the evaluated design as
  blocks; every Newton step reads them, a block ahead in a background thread.
  `storage="memory"` keeps them in memory instead; `dtype="float32"` halves
  the disk they take.
- **Convergence safeguards**: Jacobi equilibration of every solve (what
  standardizing the covariates would do, without touching the data),
  step-halving and a Levenberg–Marquardt fallback, collinearity detection,
  warnings for separated coefficients, an optional ridge penalty (`l2=`),
  and Firth's penalized likelihood (`firth=True`).
- **GPUs** (experimental): `device="cuda"` forms each Newton step's sums on
  an NVIDIA GPU through CuPy, and `polars_engine="gpu"` runs the passes over
  the source on Polars' GPU engine, with fallback to the CPU query by query.

## When to use this, and when not to

**When the data do not fit in memory, or only just do.** That is what it is
for: memory grows with the number of covariates, not the number of rows.

**For the multinomial and ordered logit, it is also the faster choice once
the data reach about 100,000 rows, even when they fit in memory.** On American Community Survey microdata
(see the [benchmarks](benchmarks/index.md)), all 12.9 million adults:

| model | logit-stream | fastest other library |
|---|---|---|
| multinomial, 5 levels, 164 coefficients | 70 s, 1.0 GB | 351 s, 13.5 GB (scikit-learn, Newton–Cholesky) |
| ordered, 5 levels | 45 s, 1.0 GB | statsmodels' `OrderedModel` timed out (20 minutes) from 3 million rows |
| binary | 64 s, 4.3 GB | 46 s, 12.2 GB (scikit-learn, Newton–Cholesky) |

The binary logit is hdfe-stream's, and on a design with only 41 columns
scikit-learn's Newton–Cholesky solver is faster at every size. Its
advantage there is memory, and absorbing fixed effects that would otherwise
be thousands of indicator columns.

**On small data it gains you little.** A fit costs about a second and a half
of fixed overhead (imports, formula planning, pass 0), and a process with
logit-stream loaded starts at about half a gigabyte. Below about 30,000 rows,
statsmodels and scikit-learn's Newton–Cholesky solver are faster.

**It is not a machine-learning toolkit.** It fits one parametric model by
maximum likelihood (or ridge-penalized), with inference; for regularization
paths, cross-validation and nonlinear models, use scikit-learn.

## Install

```bash
pip install logit-stream
```

Python 3.10 or later. It installs hdfe-stream (with its formula extra, which
brings pyfixest), Polars, NumPy, SciPy, numba and PyArrow.

```{toctree}
:hidden:
:caption: Guide

examples/index
how_it_works
```

```{toctree}
:hidden:
:caption: Benchmarks

benchmarks/index
benchmarks/acs
benchmarks/choice
benchmarks/mixed
```

```{toctree}
:hidden:
:caption: Reference

reference/index
changelog
```
