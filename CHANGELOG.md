# Changelog

## 0.1.0

The first release: logit models by streamed Newton's method, for data that
do not fit in memory. Formulas are pyfixest-style (hdfe-stream's front end);
data are Parquet files, Polars LazyFrames or DataFrames. Pass 0 writes the
evaluated design as blocks of arrays, and each Newton step is one read of
them.

### Models

- **Binary:** `logit_stream` and `probit_stream` (hdfe-stream's `feglm_stream`),
  with any number of absorbed fixed effects, fit statistics (`loglik`,
  `aic`, `bic`, `pseudo_r2_within`) and the incidental-parameter bias
  correction (`bias_correction=`).
- **Multinomial and ordered:** `mlogit_stream`, `ologit_stream` and
  `oprobit_stream`.
- **Conditional (McFadden):** `clogit_stream`; the conditional logit drops a
  choice set whole when an alternative has missing values (Stata's
  `cmclogit`), or keeps the complete alternatives with `altwise=True`.
- **Panels:** `cfelogit_stream`, Chamberlain's conditional fixed-effects logit
  (Stata's `xtlogit, fe`); matches statsmodels' `ConditionalLogit`.
- **Nested logit** (experimental): `nlogit_stream`, a tree of nests of any
  depth, in the utility-consistent normalization (Stata's `nlogit`
  default). Matches R's mlogit.
- **Mixed logit** (experimental prototype): `mixlogit_stream`, by streamed
  maximum simulated likelihood, with normal and lognormal coefficients,
  panels, and Halton or pseudo-random draws regenerated in every read.
- **Firth's penalized likelihood** (`firth=True`) for the multinomial and
  binary logit, matching brglm2 and logistf; an optional ridge penalty
  (`l2=`).

### Fixed and random effects

- One absorbed fixed effect for the multinomial and ordered logit, solved
  exactly in each Newton step (`"y ~ x | puma"`); several for the
  multinomial logit (`"y ~ x | a + b"`), fitted as the equivalent Poisson
  regression on the data expanded by outcome level (`fepois_stream`).
- Random intercepts for the multinomial and ordered logit (`random=`;
  experimental): Laplace's approximation (matching `glmer` and ordinal's
  `clmm`) or the joint mode, with an unstructured or diagonal covariance
  over the outcome levels. Nested groupings at any depth, optionally within
  a fixed effect, or two crossed. `varcorr()` and `ranef()` report the
  covariance and the predicted effects.
- With a fixed effect, `loglik_fe_null` and `pseudo_r2_within`.

### Inference and results

- Variances: iid, heteroskedasticity-robust and clustered (one-way,
  multi-way, or by an interaction), with Stata's small-sample factors for
  maximum likelihood; the cluster jackknife (`{"CRV3": var}`). Frequency
  weights, and analytic weights rescaled to average 1 as Stata does.
- Results subclass hdfe-stream's `RegressionResult`: `summary`, `tidy`,
  `confint` (pointwise or simultaneous), `wald_test`, `with_vcov`,
  `to_pyfixest` and `etable`, and lazy `predict`, `resid` (including the
  ordered models' probability-scale and generalized residuals), `sample` and
  `fixef`.
- `offset=` and warm starts (`start=`).
- Convergence by the Newton decrement, with step-halving and a
  Levenberg–Marquardt fallback, collinearity detection and warnings for
  separated coefficients.

### Scale and speed

- **GPU** (`device="cuda"`, experimental): the Newton passes and the
  variances' scores pass of every model, on an NVIDIA GPU through CuPy, with
  the blocks kept on the device when they fit; `polars_engine="gpu"` runs
  pass 0 on Polars' GPU engine. Tested on a numpy stand-in; the GPU
  benchmark covers the hardware.
- 0/1 design columns are stored as one byte each, and the multinomial
  information is computed from each row's nonzeros when the design is mostly
  indicators.
- Numba kernels that accumulate per thread (the nested, mixed and
  conditional fixed-effects logit).
- `storage=`, `dtype=`, `block_mb=` and `chunk_mb=` control the block store.

### Tools

- `logit_stream.simulate`: `simulate_people`, `simulate_choices` (with
  nests, or tasks and a random coefficient) and `simulate_panel`.
- `examples/`: a standalone script for each page of the guide.
- `benchmarks/`: ACS microdata, the Swissmetro survey, simulated panels and a
  GPU suite, against scikit-learn, statsmodels, mord, pyfixest, xlogit,
  jaxlogit, Biogeme, and R's glm, fixest, nnet, MASS, ordinal, survival,
  mlogit, logitr and alpaca. Resumable.
