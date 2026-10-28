# Changelog

## Unreleased: 0.1.0

The first version.

- `mlogit_stream`, `ologit_stream` and `clogit_stream`: multinomial, ordered
  and conditional (McFadden) logit by streamed Newton's method, from
  pyfixest-style formulas (hdfe-stream's formula front end), on Parquet files,
  Polars LazyFrames or DataFrames. Pass 0 writes the evaluated design as blocks
  of arrays; each Newton step is one read of them.
- `cfelogit_stream`: Chamberlain's conditional fixed-effects logit (Stata's
  `xtlogit, fe`), the unit effects conditioned out -- consistent with short
  panels; one recursion per unit for the likelihood, gradient and
  information; matches statsmodels' `ConditionalLogit`.
- `oprobit_stream` (ordered probit: the ordered model with the normal
  distribution, fixed effects included; `OrderedProbitResult`) and
  `probit_stream` (hdfe-stream's `feglm_stream(..., "probit")`).
- `logit_stream`: binary logit, hdfe-stream's `feglm_stream(..., "logit")`,
  with any number of absorbed fixed effects, its fit statistics (`loglik`,
  `loglik_null`, `aic`, `bic`, `pseudo_r2_within`) and its
  incidental-parameter bias correction (`bias_correction=`).
- One absorbed fixed effect for the multinomial and ordered logit
  (`"y ~ x | puma"`), solved exactly in each Newton step through the Schur
  complement of the per-level blocks of the information matrix; levels whose
  outcome never varies are dropped, and outcome levels absent from a group get
  probability zero there.
- Variances: iid (inverse information), heteroskedasticity-robust and
  clustered (one-way, or by an interaction), with Stata's small-sample factors
  for maximum likelihood; frequency and analytic weights; `cluster=` for
  further clusterings, switched to with `with_vcov`.
- Results subclass hdfe-stream's `RegressionResult`: `summary`, `tidy`,
  `summary_dict`/`summary_json`, `confint` (pointwise or simultaneous),
  `wald_test`, `with_vcov`, `to_pyfixest` and `etable`, plus lazy `predict`
  (`"response"`, `"link"`, `"class"`), `resid`, `sample` and `fixef`.
- Convergence: the Newton decrement (`tol=1e-8`), after which the step it
  computed is taken (no read; R's glm convention: the log-likelihood and
  information reported are those of the point before), equilibrated solves,
  step-halving with a Levenberg–Marquardt fallback, collinearity detection
  (also net of the fixed effect), warnings for separated coefficients, and an
  optional ridge penalty (`l2=`, scikit-learn's `C = 1 / l2`).
- `firth=True` (multinomial logit, and so the binary logit as two levels):
  Firth's penalized likelihood, matching brglm2's `brmultinom` and, for
  binary data separated by a covariate, `glm(..., method = "brglmFit")` and
  logistf's estimates; one read per Newton step (two when the iteration
  slows).
- Multi-way clustering (`{"CRV1": "a+b"}`, as hdfe-stream), and the cluster
  jackknife (`{"CRV3": var}`, one Newton step per cluster; multinomial and
  ordered logit without fixed or random effects). The conditional logit
  checks that clusters are constant within choice sets.
- `device="cuda"` (experimental): the Newton passes of the multinomial,
  ordered (logit and probit), conditional and conditional fixed-effects
  logit (its recursion vectorized over the units of each length), of the
  multinomial and ordered with an absorbed fixed effect or random
  intercepts, Firth's adjustment, and the mixed and nested logit (their
  likelihoods vectorized: over rows and draws, the draws generated on the
  device; over choice sets and nests), on an NVIDIA GPU through CuPy, the
  blocks kept on the device when they fit (the design's 0/1 columns as
  bytes there, put back together each pass); tested on a numpy stand-in,
  `benchmarks/gpu_benchmark.py` for the hardware (every model, with the
  passes' time apart from the rest). The robust and clustered variances'
  scores pass runs there too, for every model (the cluster jackknife,
  CRV3, stays on the CPU). On the CPU the mixed and nested logit
  keep their numba kernels, about ten and twenty times faster there than
  the vectorized forms.
- The numba kernels of the nested, mixed and conditional fixed-effects
  logit accumulate per thread and write out once (adding into neighbouring
  slices of shared arrays contended): the nested logit's CPU passes 5-10x
  faster.
- Several absorbed fixed effects for the multinomial logit
  (`"y ~ x | a + b"`): fitted as the equivalent Poisson regression on the
  data expanded by outcome level (`fepois_stream`, with an effect per
  observation and each fixed effect interacted with the level), the same
  estimates, log-likelihood and variances as the multinomial logit (the
  robust one clustered by observation; not with frequency weights), and a
  `MultinomialLogitResult` with `fixef()` per dimension and `predict()`.
  The data grow J-fold. `device="cuda"` is then fepois_stream's (its fixed
  effects' solve on the GPU), and `poisson_options=` passes it any of its
  other options. Its separation warning names the empty (group, outcome
  level) cells.
- `OrderedLogitResult.resid()` (and the probit's): `resid_prob`, Li and
  Shepherd's probability-scale residual (statsmodels' `resid_prob`), and
  `resid_gen`, the generalized residual (the score in the linear index),
  which for the logit are the same.
- Analytic weights are rescaled to average 1 (over choice sets for the
  conditional logit), as Stata does for maximum likelihood, so that the iid
  vcov, the log-likelihood, Firth's fit and the random intercepts do not
  depend on their scale.
- The conditional logit drops a choice set whole when an alternative has
  missing values (Stata's `cmclogit`), with a warning; `altwise=True` keeps
  its complete alternatives. `sample()` labels those rows "incomplete set".
- `offset=` (ordered and conditional logit, with fixed effects and random
  intercepts) and `start=` (warm starts: by name, a vector, or a result).
- With an absorbed fixed effect, `loglik_fe_null` and `pseudo_r2_within`:
  the null with only the fixed effect (each level's outcome shares, or for
  the ordered logit a location per level and common cutpoints), computed
  from the outcome counts per level.
- `storage=`, `dtype=`, `block_mb=`, `chunk_mb=`, `polars_engine="gpu"`.
- The design's 0/1 columns are stored as one byte each (hdfe-stream's
  `BlockStore(binary=)`) and the design rebuilt as each block is read: the
  ACS store a sixth the size, reads from disk faster, a few percent slower
  in memory.
- The multinomial information is computed from each row's nonzeros (a numba
  kernel) when the design is mostly indicators, with BLAS held to one thread
  during those passes (new dependency: threadpoolctl).
- `mixlogit_stream` (experimental prototype): mixed logit by streamed
  maximum simulated likelihood, with normal and lognormal coefficients,
  panels (`panel=`), Halton or pseudo-random draws regenerated in every read
  rather than stored, and Newton then quasi-Newton steps with a
  finite-difference Hessian of the analytic gradient. The standard
  deviations are reported as |sd| (their sign is not identified; the signed
  estimates in `diagnostics["sd_signed"]`), and the robust variance is
  clustered by chooser, `{"CRV1": panel}` (G / (G - 1), df G - 1), or
  `"hetero"` without `panel=`. No prediction, weights or correlated
  coefficients yet.
- `nlogit_stream` (experimental): the nested logit with a tree of nests of
  any depth (a dict over an `alt=` column; a nest of nests as a nested
  dict), in the utility-consistent normalization (RU2, Stata's `nlogit`
  default); started from the conditional logit and fitted as the mixed
  logit is (analytic gradient, its central differences, quasi-Newton
  steps). Weights constant within choice sets, robust and clustered
  (multi-way) variances, `offset=`, `predict()` and `resid()` as for the
  conditional logit; nest-level covariates are covariates constant within
  the nest (the same model under RU2). Matches R's mlogit with two levels. A warning for dissimilarity
  parameters outside (0, 1] or above their parent nest's.
- `random=` on `mlogit_stream` and `ologit_stream` (experimental): random
  intercepts by group, with an unstructured or diagonal covariance over the
  outcome levels, by Laplace's approximation (lme4's nAGQ = 1, matching
  `glmer` and ordinal's `clmm`; `random_method="joint-mode"` for nAGQ = 0).
  Several groupings (`random=["state", "puma"]`): nested at any depth,
  optionally within an absorbed fixed effect they nest in
  (`"y ~ x | region"`), or two crossed (the smaller one's effects solved
  densely, up to 6,000). The coefficients' standard errors include the
  covariance's uncertainty; the covariance's standard errors, and the
  standard deviations' and correlations' with intervals on the log and
  Fisher-z scales, are in `random_effects[name]` and `varcorr()`; `ranef(name)`,
  and `predict()` with the predicted effects (0 for unseen groups). A
  variance estimated at zero is held at a floor (sd 0.001), with a warning.
  When the covariance's fixed-point updates slow (tiny groups), the exact
  Laplace rounds switch to quasi-Newton steps in it with b re-solved for
  each (the ACS multinomial at 10,000 rows in 2,382 PUMAs: 52 s, from 130).
- `logit_stream.simulate`: `simulate_people` (a population with binary,
  multinomial and ordered demographic outcomes and PUMA effects),
  `simulate_choices` (long-format mode-choice data; with `nests=` drawn from
  the nested logit, a tree of any depth, and with `tasks=` and `sd_time=`
  from the mixed logit with panels of repeated choices) and
  `simulate_panel` (a binary panel with unit effects correlated with a
  covariate).
- `examples/`: a script for each page of the guide (quickstart, binary,
  multinomial, ordered, conditional, nested, mixed, fixed effects, random
  intercepts, incidental parameters, imputation), on simulated data, each
  standing alone; the pages show their output and link the script for
  download.
- Benchmarks on ACS PUMS microdata, the Swissmetro survey and simulated
  panels, against scikit-learn, statsmodels, mord, pyfixest, xlogit,
  jaxlogit and Biogeme, and R's glm, fixest, nnet (multinom), MASS (polr),
  ordinal (clm), survival (clogit), mlogit, logitr and alpaca (R and Biogeme
  run from `benchmarks/external/`, in environments of their own); the nested
  logit and the panel logits (conditional, fixed-effects, bias-corrected)
  have their own series; and a GPU suite, every model on one machine's CPU
  and GPU, with xlogit and jaxlogit on both. Resumable (a ledger of finished
  measurements), and a configuration that fails at one size is skipped at
  the larger ones. pylogit is no longer run.
