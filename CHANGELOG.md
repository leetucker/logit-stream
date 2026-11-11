# Changelog

## Unreleased

- **`logit_stream` is about 2 times faster with no fixed effect or one, and its estimates are at
  the maximum to machine precision.** It now fits the binary logit as the two-level ordered logit
  (the cutpoint is minus the intercept) with this package's Newton iterations over the stored blocks
  and fused kernels, and returns a `BinaryLogitResult`. The coefficients, variances (every
  `vcov`, with hdfe-stream's and pyfixest's small-sample factors, a fixed effect nested in the
  clusters left out of $K$), degrees of freedom, `loglik`, `aic`, `bic`, `fixef`, `predict`, `resid`
  and `sample` agree with `feglm_stream`'s to 1e-12 or better. Newton's decrement judges
  convergence (hdfe-stream's relative change in deviance leaves the coefficients about 1e-7 from the
  maximum); `iwls_tol` and `iwls_maxiter` are still accepted and mean the Newton tolerance and
  maximum. Several fixed effects, `bias_correction=True`, `bootcluster=`, CRV3 and
  `separation_check=False` still run hdfe-stream's engine and return its `HDFEResult`;
  `probit_stream` always does. `predict(type="class")` is new.
- **Fixed effects with many small groups are 1.5-2 times faster** for the binary and the two-level
  multinomial and ordered logit (10 million rows in 2 million units of 5: 7.1 s, now 4.2 s). The
  within pseudo R-squared's null model of two levels is now its closed form (each group's share),
  which took 2 s of Newton's method on 2 million groups, and the Schur step inverts a one-parameter
  group's block as a reciprocal (1 s of batched matrix inverses). `logit_stream(..., bias_correction=False)`
  and `bootcluster=None` are the defaults, not options only hdfe-stream's engine has.
- **A cluster whose rows were all dropped is not counted** in a binary logit's cluster-robust
  variance (`G` and the degrees of freedom count the clusters of the rows used), and analytic
  weights are rescaled over the rows used after a fixed effect's separation drops some, as
  hdfe-stream does.
- **A fixed effect's sort of the rows respects `max_memory_gb`.** Rows not already grouped by the
  fixed effect are partitioned in memory to sort them; when that would not fit, they stay in their
  order and the passes are slower.
- **`resid()` of a result fitted with an offset works** (the offset's column was not read).
- **The within pseudo R-squared is not reported (nan) for an ordered or multinomial fit with an offset and an
  absorbed fixed effect**: the fixed-effect-only log-likelihood it was computed from ignored the offset.
- **Robust and clustered variances of the multinomial and ordered logit are 3-18 times faster.** The
  extra pass that forms the variance's middle matrices is now a fused parallel kernel: each row's
  score is formed in registers and added to the outer-product sums for `hetero` and to each
  clustering's cluster sums, with no array of scores and no sorting by cluster. Without a fixed
  effect (1 million rows, 10 covariates): `hetero` 0.45 s, now 0.13 s; 2,000 clusters 0.9 s, now
  0.12 s; 200,000 clusters 1.4 s, now 0.3 s. With an absorbed fixed effect, each row's score net of
  the fixed effect is formed while its group's block is in cache (the rows are grouped): 500
  groups, `hetero` 0.8 s, now 0.16 s; 2,000 clusters 0.95 s, now 0.09 s, which takes the variance
  from 70-80% of such a fit to 4-16%. The variances are the same.
- **Robust and clustered variances of the conditional logit and the fixed-effects logit
  (`cfelogit_stream`) are 2.5-7 times faster.** The same kernel treats a choice set or a unit as
  the row of the sums, forming its score in a thread (for the fixed-effects logit, by the same
  recursion as the likelihood's). 1 million choice sets of 4 alternatives and 6 covariates:
  `hetero` 0.33 s, now 0.08 s; 30 covariates 0.97 s for half the sets, now 0.15 s. The variances are
  the same.
- **The multinomial and ordered logit with an absorbed fixed effect are 2-4 times faster.** Each
  pass over the data is now a fused parallel kernel, 4-11 times faster per pass (1 million rows,
  500 groups: 0.8 s per pass for the multinomial logit, now 0.2 s; 0.6 s for the ordered logit,
  now 0.05 s). The first pass writes the rows grouped by the fixed effect, so that each thread
  owns whole groups (about 0.3 s more for 1 million rows; free if they are grouped already).
  The estimates are the same.
- **The ordered logit and probit with an absorbed fixed effect no longer slow down with many
  groups.** The within pseudo R-squared's null model, a fit of one location per group, is now
  solved by Newton's method instead of L-BFGS: 100,000 groups took 47 s, now 2 s in all. The
  Newton step for any fixed-effects fit also uses all threads for its largest product
  (100,000 groups: 12.7 s, now 5.3 s for the multinomial logit).
- **The ordered logit and probit (without fixed effects) are about 4 times faster.** Their passes
  over the data are now one parallel numba kernel too, 12-14 times faster per pass with 5-10
  covariates (1 million rows, 5 levels: 0.26 s per pass, now 0.02 s), which leaves reading the data
  and setup as most of a fit. A design of mostly indicators (76 covariates) is 3.3 times faster
  and a wide dense one (more than 96 covariates) keeps the previous pass. The estimates are the
  same.
- **`fit.diagnostics["information"]` says how the multinomial, ordered logit and probit passes
  formed the information matrix** (`"outer-product"`, `"nonzero"` or `"array"`), with the
  design's density.
- **The multinomial logit (without fixed effects) is about 5 times faster on dense designs and
  1.3-1.8 times faster on designs of indicators.** Each pass over the data is now one parallel
  numba kernel that computes the probabilities, log-likelihood, gradient and information row
  by row, as the conditional logit's does, so all threads stay busy and no row-sized
  temporaries are allocated (1 million rows, 5 equations of 7 covariates: 1.0 s per pass, now
  0.15 s). The estimates are the same. The information comes from each row's outer product, or,
  for a design with 20% or fewer nonzero entries, from its nonzeros. Dense designs of more than
  96 columns keep the previous BLAS-based pass.
- **`mlogit_stream(own_covariates=...)`: covariates that enter only their own level's equation.**
  For example, each group's baseline share can predict that group alone, leaving out the other
  groups' shares. This drops the off-diagonal coefficients, which are often poorly determined
  where the groups are segregated, and fits fewer parameters.

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
