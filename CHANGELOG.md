# Changelog

## Unreleased

New
- **`random_method="adaptive"`: adaptive Gauss-Hermite quadrature for one scalar random
  intercept** (`logit_stream` and `ologit_stream`, optionally within an absorbed fixed effect),
  with `random_nodes` points per group (default 9, at least 5), centred at each group's
  conditional mode, as lme4's `nAGQ > 1`. It removes the Laplace approximation's bias with small
  groups. In a simulation with about 3 rows per group and variance 0.64, Laplace's estimate was
  0.494 and adaptive quadrature's 0.644. It agrees with brute-force maximum likelihood to 1e-6
  standard errors in the tests, and in the simulations was faster than the Laplace fit. Only
  the iid variance is available with it, on the CPU.
- **`random_method="adaptive"` for the multinomial logit**: adaptive Gauss-Hermite quadrature for
  the (J - 1)-vector of random intercepts of one grouping (`mlogit_stream`, optionally within an
  absorbed fixed effect), with the product of `random_nodes` points per dimension (default 5, at
  least 5; at most 20,000 points per group; groups of 30 rows or more take 3), centred at each group's conditional mode and scaled
  by its conditional covariance. Laplace's approximation is biased with small groups. With J = 5
  and three rows per group in a simulation, it put the common levels' variances about a fifth
  too low, the rarest level's twice too high, and the correlations 40% too low. The adaptive
  fit agrees with brute-force maximum likelihood (estimates to 3e-3, standard errors of the
  coefficients and of Sigma to 3%). It takes about 1.6 times the Laplace fit's time at J = 5 (a numba kernel per group).
  `random_nodes` now defaults to the method's own value.
- **Absorbed alternative constants in `clogit_stream`** (`"chosen ~ x | estab"`, one or more
  dimensions): one constant per level of a characteristic of the alternatives. The first pass
  finds the groups of levels that the choice sets link (union-find; with establishments chosen
  within firms, the firms) and drops levels never chosen or always chosen. `fe_solver="auto"`
  (default) picks between `"blocks"`, exact Newton steps with each group's information dense,
  and `"contraction"`, the constants by count matching and the coefficients' information net of
  them by conjugate gradients. Both agree with the dummy-coded fit, for one and two dimensions,
  to 1e-6 in the estimates and standard errors. With one dimension and the blocks, `fixef()` has
  the constants' standard errors and `draw_parameters` draws them jointly with the
  coefficients. Robust and clustered (CRV1) variances, from each set's score net of the
  constants, work with either solver and any number of dimensions.
- **Constrained assignment (experimental): `logit_stream.assign` and
  `ConditionalLogitResult.sample_assignments(newdata, {"estab": totals}, sampler=...)`.** Draws
  each choice set's alternative so that the alternatives' counts match known totals: exactly
  (`SwapMCMC()`, Metropolis-Hastings by swaps from an assignment meeting the targets;
  `ExactSmall()`, exact recursion for small groups), approximately (`penalty=GaussianPenalty(sd,
  rel)`, or any function of the counts), or with the calibrated probabilities kept (`Balanced()`,
  the cube method). Streamed by group; draws keyed by (seed, group), so they do not depend on
  the batches. Hooks for a project's own targets, utilities and draws. Warns with
  `logit_stream.ExperimentalWarning`, a subclass of hdfe-stream's.
- **`predict(..., calibrate={"estab": totals})`** for the conditional logit: one dimension's
  constants solved from known totals, so that each level's predicted choices equal its total,
  given the coefficients (a draw's, under `draw_parameters`). It is for imputation where the
  alternatives are new but their totals are known. `newdata` is streamed through disk a batch
  of whole groups (a firm's establishments, or whatever levels its sets link) at a time, so
  memory is bounded by the batch or the largest group, not by `newdata`; the result is a lazy
  scan in `newdata`'s order. Contraction rounds, then Newton's method when they slow.
- **`logit_stream(random=...)`**: random intercepts for the binary logit, fitted by this package's
  engine (as the two-level ordered logit), Laplace by default, with lme4's conventions for the
  variance (no small-sample factor). Before, `random=` reached `feglm_stream` and failed.
- **`draw_parameters(n, seed)`: draws of the parameters from their estimated sampling
  distribution**, for multiple imputation and simulated predictions. Each draw is a copy of the
  result whose `beta`, `fixef()`, `random_effects` and `predict` are the draw's. It covers the
  coefficients (from `vcov`), one absorbed fixed effect's levels jointly with them (each level
  given the coefficients, from the blocks the fit already computed; no data are read), and, with
  `random_method="adaptive"`, the random intercept's sd. The groups' own intercepts are not
  drawn: a drawn result's `predict` leaves them out. Not available for the Laplace or joint-mode
  random intercepts, several fixed effects, or the mixed and nested logit.
- **`draw_ranef()` and `draw_parameters(..., ranef=True)`: draws of each group's random
  intercept from its conditional distribution** given the group's rows and the parameters (at the
  estimates or at a parameter draw), against each row's own linear predictor. Each draw is exact,
  from the density on a grid around the group's mode, sampled by its inverse cdf. The drawn
  result's `ranef()` holds the draws and `predict` adds them. `newdata=` gives groups the fit did
  not see a draw from N(0, sd^2). With `draw_parameters(n, ranef=True)`, all n draws come from one
  read of the estimation sample (sorted by group, in batches of whole groups). Available for the
  binary, ordered and multinomial logit and probit with one random grouping. For the multinomial
  logit, each group's J - 1 intercepts are drawn jointly by importance resampling around its
  mode, and `draw_parameters` (adaptive fits) draws Sigma on its log-Cholesky scale.
- **`fixef()` has standard errors** (`se_<name>`, or `se_<name>:<level>`) where
  `draw_parameters` is available: each level's variance given the coefficients, plus what the
  coefficients' uncertainty adds.
- **`ranef()` has each group's conditional sd** (`sd_<grouping>`, or `sd_<grouping>:<level>` for
  the multinomial logit): lme4's `condVar`, at the estimates.

Documentation
- The random-intercepts page has a section on Laplace's small-group bias and the adaptive
  quadrature.
- New example, "Multiple imputation": parameter and firm-intercept draws, and the imputations in
  Polars (`examples/multiple_imputation.py`).

## 0.4.0

New
- **`mlogit_stream(own_covariates=...)`: covariates that enter only their own level's equation.**
  For example, each group's baseline share can predict that group alone, leaving out the other
  groups' shares. This drops the off-diagonal coefficients, which are often poorly determined
  where the groups are segregated, and fits fewer parameters.
- **`logit_stream` and `probit_stream` have their own engine** for no fixed effect or one, about
  2 times faster than hdfe-stream's. They fit the two-level ordered logit or probit with this
  package's Newton iterations and return a `BinaryLogitResult` or `BinaryProbitResult` (new
  classes; `predict(type="class")` is new). Estimates are at the maximum to machine precision
  (Newton's decrement judges convergence; `iwls_tol` and `iwls_maxiter` still work). Coefficients,
  variances, degrees of freedom, fit statistics, `fixef`, `predict`, `resid` and `sample` agree
  with `feglm_stream`'s; the probit's variances use the expected information, as its iteratively
  reweighted fit does. Several fixed effects, `bias_correction=True`, `bootcluster=`, CRV3,
  `separation_check=False` and `profile=True` still run `feglm_stream` and return its
  `HDFEResult`.

Faster (estimates unchanged)
- **Multinomial logit** without fixed effects: about 5 times on dense designs, 1.3-1.8 times on
  designs of indicators. **Ordered logit and probit** without fixed effects: about 4 times. Each
  pass over the data is now one fused parallel kernel.
- **Multinomial and ordered logit with an absorbed fixed effect:** 2-4 times. The first pass
  writes the rows grouped by the fixed effect, so each thread owns whole groups. Fixed effects
  with many small groups are a further 1.5-2 times faster, and the ordered logit and probit no
  longer slow down with many groups (100,000 groups: 47 s, now 2 s).
- **Robust and clustered variances:** 3-18 times faster for the multinomial and ordered logit, and
  2.5-7 times for the conditional logit and `cfelogit_stream`, from a fused kernel that forms each
  score in a thread.
- `fit.diagnostics["information"]` says how the information matrix was formed.

Fixed
- Cluster counts `G` and degrees of freedom count only clusters with rows used, and analytic
  weights are rescaled over the rows used after a fixed effect's separation drops some, in the
  binary, multinomial and ordered logit, as hdfe-stream does. Before, the multinomial and ordered
  logit counted clusters whose rows were all dropped (variances slightly small) and scaled the
  weights over all rows read (iid variance and log-likelihoods).
- Sorting rows by a fixed effect respects `max_memory_gb`: rows not already grouped are
  partitioned in memory to sort them, and when that would not fit they stay in their order and the
  passes are slower.
- `resid()` of a result fitted with an offset works.
- The within pseudo R-squared is nan for an ordered or multinomial fit with an offset and an
  absorbed fixed effect (the fixed-effect-only log-likelihood it came from ignored the offset).

## 0.3.1

- **Works with pyfixest 0.60 and 0.70.** Formulas are read through hdfe-stream's own formula
  parser, so either pyfixest version can be installed.
- Requires hdfe-stream 0.10.0 or later.

## 0.3.0

- **The conditional logit is 2.3-2.8 times faster.** Its log-likelihood, gradient and information
  are computed in one parallel numba pass over each block's choice sets, giving the same
  estimates (4 million sets: 13 s, now 5 s; 8 million: 26 s, now 9-12 s). Designs of more than
  256 parameters keep the previous BLAS-based pass, which is faster there.
- **The fixed-effects logit (`cfelogit_stream`) balances work across threads better.** Panels
  are dealt out in small pieces to more parts than threads, so a run of long panels no longer
  leaves threads idle: 0.55 of the time on panels sorted by length, 0.8-1.0 on shuffled
  ones, with the same results.
- **Lower memory for conditional, mixed and fixed-effects logit.** Choice sets are now checked
  and counted a block at a time instead of from a table of every set, so memory no longer grows
  with the number of sets (8 million sets: peak 1.71 GB down to 1.33 GB). Same errors, same
  results.
- **Memory tests.** `pytest -m resource` fits the multinomial, ordered and conditional logit
  on data larger than a fixed memory limit; CI runs them on Linux, macOS and Windows. The
  default `pytest` run leaves them out.
- Requires hdfe-stream 0.9.0 or later, which reads Parquet in less memory.

## 0.2.0

- **Fit resources (CPUs and memory) are constrained to your job's limits automatically.** On a
  cluster (PBS, Slurm, or a container with cgroup limits), importing `logit_stream` caps the threads of
  Polars, numba, BLAS and pyarrow at the job's CPU allocation, and with Polars 2.x gives Polars a
  memory budget so its streaming operations spill to disk instead of growing. Caveats: the Polars
  budget is a target for its buffers, not a hard cap on the process, and Polars 1.x has no such
  option, so there only the threads are capped. The caps must be in place before Polars is
  imported: import `logit_stream` first, or call `stream_limits.set_limits()` at the top of the script.
  With no limit found nothing changes. See "Resource limits and profiling" in the documentation.
- **`profile=True`: estimate a fit's memory and disk before running it.** On
  every estimator, profiling returns a `hdfe_stream.Profile` -- the estimated
  memory and disk, from one scan of the data -- instead of fitting. P comes from
  the model and the outcome's levels; the information matrix, its per-thread
  copies, the fixed effect's cross blocks and the clusters' score sums are sized
  from it. The multinomial logit with several fixed effects (fitted by
  `fepois_stream` on data expanded J-fold) is profiled as that regression, from
  counts worked out from one scan of the original data. Not modeled: random
  intercepts, and the mixed and nested logit's workspaces beyond the per-thread
  accumulators.
- **`max_memory_gb` and `memory_reserve`**: the fixed effect's cross-block guard and
  the random intercepts' guard (both a fixed 8 GB) and CRV3's (25% of physical
  memory) now ask whether their exact allocation fits in the memory left (the limit
  less a reserve, less what the process holds). The limit defaults to the job's
  allocation (cgroup, `STREAM_LIMITS_MEMORY_GB`, or on PBS `qstat`).
- **`n_threads=`** on every estimator: numba and BLAS threads for one fit,
  restored afterwards (Polars' pool is fixed at import; see hdfe-stream's
  "Resource limits and profiling").
- **`diagnostics["peak_rss_gb"]`** in every fit.
- Depends on `stream-limits`, and needs hdfe-stream 0.8.0 or later (the `<0.7` upper bound on
  hdfe-stream is gone).

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
