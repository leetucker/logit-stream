"""The public estimators: `logit_stream`, `mlogit_stream`, `ologit_stream`,
`clogit_stream`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from hdfe_stream.core.typing import PathLike, Source, Vcov

from .estimator import _fit
from .mixed import _fit_mixed

_COMMON = """
    data : str, Path, polars.LazyFrame or polars.DataFrame
        A Parquet path or glob, or a Polars LazyFrame (never collected as a
        whole). A DataFrame works too.
    workdir : str or Path, optional
        Where the run directory for the fit's blocks goes. Default: the
        ``LOGIT_STREAM_WORKDIR`` environment variable, else
        ``HDFE_STREAM_WORKDIR``, else the system temporary directory. The
        run directory is deleted when the fit ends.
    vcov : str or dict, default "iid"
        ``"iid"`` (the inverse of the information matrix), ``"hetero"`` (the
        sandwich, with N / (N - 1)), ``{"CRV1": var}`` / ``"CRV1:var"``
        (clustered by `var`, or by an interaction ``"a^b"``, with
        G / (G - 1)) -- Stata's conventions for maximum likelihood --
        multi-way ``{"CRV1": "a+b"}`` (inclusion-exclusion over a, b and
        a^b, each scaled with the smaller G, df G - 1, as hdfe-stream and
        pyfixest), or ``{"CRV3": var}``, the cluster jackknife (one-way;
        the multinomial and ordered logit without fixed effects or random
        intercepts; one more read and G x P^2 memory).
    cluster : str or sequence of str, optional
        Further clusterings to compute, for `with_vcov`.
    weights : str, optional
        A column of positive weights, each multiplying its row's
        log-likelihood.
    weights_type : {"aweights", "fweights"}, default "aweights"
        As in hdfe-stream: frequency weights count as that many
        observations (`n_obs` is their sum, and the robust variances treat
        each as that many independent rows); analytic weights do not, and
        are rescaled to average 1 over the sample (over the choice sets for
        the conditional logit), as Stata does for maximum likelihood, so
        that nothing depends on their scale. For survey (probability)
        weights use "aweights" with a robust or clustered vcov: that is
        Stata's ``[pweight]`` fit.
    tol : float, default 1e-8
        Convergence tolerance on the Newton decrement g' H^-1 g: stop when
        the next step would move the estimates by less than about
        sqrt(tol) standard errors.
    maxiter : int, default 50
        The most Newton steps.
    l2 : float, default 0
        An L2 (ridge) penalty, ``l2 / 2`` times the sum of squared slopes
        (not intercepts or cutpoints), subtracted from the log-likelihood:
        scikit-learn's ``C`` is ``1 / l2`` for the sum of the losses. Use it
        when the outcome is separated by a covariate. The standard errors
        are then those of the penalized fit.
    offset : str or polars.Expr, optional
        A column or expression added to the linear predictor with a
        coefficient of one, for the ordered logit (a known location shift)
        and the conditional logit (e.g. McFadden's correction for sampled
        alternatives, the log of each alternative's sampling probability);
        `predict` and `resid` include it, so new data need it too. Not for
        the multinomial logit, which would need one per outcome level.
    start : dict, sequence or result, optional
        Starting values for the coefficients (warm start): {name: value} for
        some of them, all of them in the order of `coefnames`, or a previous
        result's. The rest, and any fixed or random effects, start at the
        null model.
    device : {"cpu", "cuda"}, default "cpu"
        ``"cuda"`` (experimental) forms each Newton step's log-likelihood,
        gradient and information on an NVIDIA GPU through CuPy, keeping the
        blocks there when the sample fits in half its free memory; the
        robust and clustered variances' scores pass runs there too (the
        cluster jackknife's stays on the CPU), and the step itself on the
        CPU. The estimates agree with the CPU's to the tolerance. With an
        absorbed fixed effect and with random intercepts too (their
        per-level and per-unit sums, and the exact Laplace terms, by
        scatter-adds on the device), Firth's adjustment, and
        `cfelogit_stream`. Needs ``pip install cupy-cuda12x`` (or
        cupy-cuda13x).
    firth : bool, default False
        Maximize Firth's penalized likelihood, the log-likelihood plus half
        the log-determinant of the information (a Jeffreys prior), instead
        of the likelihood: estimates with the O(1/n) bias removed, and
        finite when a covariate separates the outcome. Standard errors are
        from the information at the estimates; `loglik` is the unpenalized
        log-likelihood there; they agree with brglm2's ``brmultinom`` (type
        ``"AS_mean"``). Multinomial logit only (a two-level outcome gives the
        binary logit's Firth fit, as R's logistf and brglm2), without fixed
        effects, random intercepts or `l2`. A step is one read, two when the
        iteration slows (near separation).
    storage : {"disk", "memory"}, default "disk"
        Where the blocks live between Newton steps: files in the run
        directory, read one block at a time, or memory.
    dtype : {"float64", "float32"}, default "float64"
        The blocks' storage precision. Computation is always in float64;
        float32 halves the disk the design takes, rounding it to 7 digits
        (exact for indicators and small integers).
    block_mb : float, default 64
        The size of a block of the design, in MB.
    chunk_mb : float, default 64
        The most memory a chunk's working arrays take: blocks are worked on
        in chunks of rows sized by this.
    polars_engine : {"cpu", "gpu"}, default "cpu"
        The engine for the fit's Polars queries (pass 0); "gpu" uses
        cudf-polars and falls back to the CPU, query by query.
    keep_intermediates : bool, default False
        Keep the run directory.
    verbose : bool, default True
        Progress messages (to `logger` if given).
    logger : logging.Logger, optional
    log_level : int, optional"""

_RANDOM = """
    random : str or list of str, optional
        Random intercepts by this grouping column (or interaction ``a^b``):
        each group's intercepts are drawn from a normal distribution with
        mean 0 and a covariance Sigma that is estimated -- one intercept
        per non-base outcome level (multinomial), one location shift
        (ordered). A list gives several groupings, each with its own Sigma:
        nested in one another at any depth (``["state", "puma"]``, PUMAs
        within states), or two that cross (``["puma", "occupation"]``; the
        smaller one's at most 6,000 effects are solved densely). The fixed
        intercepts or cutpoints stay in the model, unless the formula
        absorbs a fixed effect (``"y ~ x | region"``), within which nested
        groupings must nest in turn (crossed ones allow none). Not with
        `l2`. `vcov` may be ``"iid"`` or clustered by a variable that
        contains whole groups of the coarsest random grouping (crossed
        groupings allow none; not ``"hetero"``). The coefficients' standard
        errors include Sigma's uncertainty. The predicted effects are in
        `ranef`, and `predict` uses them (0 for a group the fit did not
        see); Sigma, its standard errors and the sds' and correlations' (with
        intervals on the log and Fisher-z scales) are in `random_effects`
        and `varcorr`. A variance estimated at zero is held at a floor (an
        sd of 0.001), with a warning. See `logit_stream.random_effects` for
        the method.
    random_method : {"laplace", "joint-mode"}, default "laplace"
        The approximation to the likelihood: Laplace's (lme4's ``nAGQ = 1``,
        ordinal's ``clmm`` default), or Laplace's with the coefficients and
        the effects at their joint mode (lme4's ``nAGQ = 0``), which is
        faster and close to it when groups have many rows each (with ~100
        rows per group, the coefficients can be a few tenths of a standard
        error away).
    random_cov : {"unstructured", "diagonal"}, default "unstructured"
        The form of Sigma (multinomial; the ordered logit's is 1 x 1).
    random_tol : float, default 1e-6
        Convergence when an update changes Sigma by less than this,
        relative to its largest element."""


def logit_stream(fml: str, data: Source, workdir: PathLike | None = None,
                 vcov: Vcov = "iid", cluster: Sequence[str] | str = (),
                 **options: Any):
    """Out-of-core binary logit, with any number of high-dimensional fixed
    effects: hdfe-stream's `hdfe_stream.feglm_stream` with
    ``family="logit"``.

    The fit is iteratively reweighted least squares with the fixed effects
    absorbed (hdfe-stream's `StreamingGLM`), which for the logit is Newton's
    method; its options and its result (`hdfe_stream.HDFEResult`, with
    `fixef`, `resid`, `predict` and `sample`, and `loglik`, `loglik_null`,
    `pseudo_r2`, `aic` and `bic`, and with fixed effects
    `pseudo_r2_within`) are hdfe-stream's.

    Parameters
    ----------
    fml : str
        A pyfixest-style formula with a 0/1 outcome, e.g.
        ``"hispanic ~ age + C(sex) | puma"``.
    data, workdir, vcov, cluster
        As in `mlogit_stream` (and `hdfe_stream.feglm_stream`).
    **options
        `hdfe_stream.feglm_stream`'s: ``weights``, ``weights_type``,
        ``iwls_tol``, ``iwls_maxiter``, ``separation_check``,
        ``within_pseudo_r2``, ``bias_correction`` (the incidental-parameter
        bias of the fixed effects, as R's alpaca), ``fe_dof``, the memory
        options, ``verbose``, ...

    Returns
    -------
    hdfe_stream.HDFEResult or hdfe_stream.HDFEMulti

    Notes
    -----
    Two conventions are hdfe-stream's (and pyfixest's) rather than this
    package's: convergence is judged on the relative change in deviance
    (``iwls_tol``), and the iid vcov has the factor (N - 1) / (N - K).
    A two-level `mlogit_stream` fits the same model, with this package's
    conventions, but absorbs at most one fixed effect; it is also the way
    to Firth's penalized likelihood for a binary outcome (``firth=True``).
    """
    from hdfe_stream import feglm_stream
    return feglm_stream(fml, data, "logit", workdir=workdir, vcov=vcov, cluster=cluster,
                        **options)


def mlogit_stream(fml: str, data: Source, workdir: PathLike | None = None,
                  vcov: Vcov = "iid", cluster: Sequence[str] | str = (),
                  levels: Sequence | None = None, **options: Any):
    return _fit("mlogit", fml, data, workdir=workdir, vcov=vcov, cluster=cluster,
                levels=levels, **options)


def ologit_stream(fml: str, data: Source, workdir: PathLike | None = None,
                  vcov: Vcov = "iid", cluster: Sequence[str] | str = (),
                  levels: Sequence | None = None, **options: Any):
    return _fit("ologit", fml, data, workdir=workdir, vcov=vcov, cluster=cluster,
                levels=levels, **options)


def oprobit_stream(fml: str, data: Source, workdir: PathLike | None = None,
                   vcov: Vcov = "iid", cluster: Sequence[str] | str = (),
                   levels: Sequence | None = None, **options: Any):
    return _fit("ologit", fml, data, workdir=workdir, vcov=vcov, cluster=cluster,
                levels=levels, link="probit", **options)


def probit_stream(fml: str, data: Source, workdir: PathLike | None = None,
                  vcov: Vcov = "iid", cluster: Sequence[str] | str = (),
                  **options: Any):
    """Out-of-core binary probit, with any number of high-dimensional fixed
    effects: hdfe-stream's `hdfe_stream.feglm_stream` with
    ``family="probit"``, as `logit_stream` is with ``family="logit"`` (see
    there: the options, the result, the bias correction)."""
    from hdfe_stream import feglm_stream
    return feglm_stream(fml, data, "probit", workdir=workdir, vcov=vcov, cluster=cluster,
                        **options)


def cfelogit_stream(fml: str, data: Source, workdir: PathLike | None = None,
                    vcov: Vcov = "iid", cluster: Sequence[str] | str = (), **options: Any):
    return _fit("cfelogit", fml, data, workdir=workdir, vcov=vcov, cluster=cluster,
                **options)


def clogit_stream(fml: str, data: Source, case: str | Sequence[str],
                  workdir: PathLike | None = None, vcov: Vcov = "iid",
                  cluster: Sequence[str] | str = (), altwise: bool = False, **options: Any):
    return _fit("clogit", fml, data, case=case, workdir=workdir, vcov=vcov, cluster=cluster,
                altwise=altwise, **options)



def nlogit_stream(fml: str, data: Source, case: str | Sequence[str], alt: str,
                  nests: dict, workdir: PathLike | None = None, vcov: Vcov = "iid",
                  cluster: Sequence[str] | str = (), weights: str | None = None,
                  weights_type: str = "aweights", offset: str | Any | None = None,
                  **options: Any):
    """Out-of-core nested logit (RU2, any depth of nests), from long-format
    choice data as `clogit_stream` takes it (prototype).

    Parameters
    ----------
    fml : str
        ``"chosen ~ x1 + x2 + C(alt)"``: the chosen alternative (0/1, one
        per choice set) on the utilities' covariates, as for `clogit_stream`.
    data : str, path, LazyFrame or DataFrame
        One row per alternative of each choice set.
    case : str or list of str
        The column(s) identifying the choice sets.
    alt : str
        The column naming each row's alternative.
    nests : dict
        The tree: ``{nest name: [alternatives]}`` for two levels, and a nest
        of nests as a dict, ``{"public": {"rail": ["train", "metro"],
        "road": ["bus"]}, "private": ["car", "bike"]}``. Every alternative
        of `alt` in one nest, at least two nests at the top; a nest holds
        alternatives or nests, not both. A nest with one child has no
        dissimilarity parameter (it is 1).
    vcov : {"iid", "hetero"}, or a clustered one
        The inverse information; the sandwich with the choice sets' scores
        (``"hetero"``); or clustered, ``{"CRV1": var}`` (multi-way
        ``"a+b"``), the clusters constant within choice sets.
    cluster : str or list of str, optional
        Further clusterings to compute (`with_vcov` switches to them).
    weights : str, optional
        A column of weights, constant within each choice set.
    weights_type : {"aweights", "fweights"}
        Analytic weights are rescaled to average 1 over the choice sets.
    offset : str or Polars expression, optional
        A known term of each alternative's utility (coefficient 1).
    **options
        ``tol`` (the Newton decrement, 1e-8), ``maxiter`` (200),
        ``device`` ("cpu" or "cuda": the passes on an NVIDIA GPU through
        CuPy, experimental), and the storage options of `clogit_stream`
        (``storage``, ``dtype``, ``block_mb``, ``workdir``,
        ``keep_intermediates``, ``verbose``, ``logger``, ``log_level``,
        ``polars_engine``).

    Returns
    -------
    NestedLogitResult
        The coefficients, then ``lambda.<nest>`` for each nest of two or
        more children. A child c of nest n has the value U_c = V_c (an
        alternative, V its utility) or lambda_c I_c (a nest), with I_n =
        log sum_{c in n} exp(U_c / lambda_n), and the conditional
        probability exp(U_c / lambda_n - I_n); an alternative's probability
        is their product along its path (Stata's ``nlogit`` default
        normalization). All lambdas 1 is the conditional logit, the starting
        point; a warning if an estimate is outside (0, 1] or above its
        parent nest's, where the model is not consistent with utility
        maximization everywhere. `predict` and `resid` as for the
        conditional logit.
    """
    from .nested import _fit_nested
    return _fit_nested(fml, data, case=case, alt=alt, nests=nests, workdir=workdir, vcov=vcov,
                       cluster=cluster, weights=weights, weights_type=weights_type,
                       offset=offset, **options)


def mixlogit_stream(fml: str, data: Source, case: str | Sequence[str],
                    random: dict, panel: str | Sequence[str] | None = None,
                    workdir: PathLike | None = None, vcov: Vcov = "iid", **options: Any):
    return _fit_mixed(fml, data, case=case, panel=panel, random=random, workdir=workdir,
                      vcov=vcov, **options)

mlogit_stream.__doc__ = """Out-of-core multinomial logit by streamed Newton.

    P(y = j | x) = exp(x'b_j) / sum_l exp(x'b_l), with b = 0 for the base
    level (the first of `levels`). Each Newton step is one read of the
    sample's blocks (see `logit_stream.newton`).

    Parameters
    ----------
    fml : str
        A pyfixest-style formula, e.g. ``"race ~ age + C(sex) + C(state)"``.
        The outcome is any column (strings, categoricals, integers); its
        levels are its distinct values. An intercept is included unless the
        formula removes it (``- 1``) or there is a fixed effect. A fixed
        effect can be absorbed, ``"race ~ age | puma"``; then each of its
        levels has its own intercept for each outcome level. With several
        (``"race ~ age | puma + year"``) the model is fitted as the
        equivalent Poisson regression on the data expanded by outcome level,
        by hdfe-stream's `fepois_stream` (see the fixed-effects guide); then
        ``device="cuda"`` is fepois_stream's (its fixed effects' solve on the
        GPU), and `poisson_options` passes it any of its other options.
        Several outcomes and ``sw()``/``csw()`` give several models.""" + _COMMON + """
    levels : sequence, optional
        The outcome's levels in the order wanted, base first. Default: the
        distinct values, sorted.
    poisson_options : dict, optional
        With several fixed effects: further options of hdfe-stream's
        `fepois_stream` (``solver``, ``precond``, ``max_memory_gb``,
        ``fixef_rm``, ...), except those the multinomial logit sets itself
        (vcov, cluster, weights, keep, workdir).""" + _RANDOM + """

    Returns
    -------
    MultinomialLogitResult or LogitMulti

    Examples
    --------
    >>> fit = mlogit_stream("race ~ age + C(sex) + log_income | puma",
    ...                     "acs/*.parquet", vcov={"CRV1": "puma"})
    >>> fit.summary()
    >>> probs = fit.predict("people/*.parquet")    # lazy: prob_<level> columns
    """

ologit_stream.__doc__ = """Out-of-core ordered logit by streamed Newton.

    P(y <= j | x) = F(cut_j - x'b), F the logistic distribution, with
    cutpoints cut_1 < ... < cut_{J-1} estimated with the slopes (there is no
    intercept). Each Newton step is one read of the sample's blocks.

    Parameters
    ----------
    fml : str
        A pyfixest-style formula, e.g. ``"educ ~ age + C(sex) | puma"``. The
        outcome's levels are ordered as `levels`, or else sorted (numbers by
        value, strings alphabetically, an Enum by its categories). One fixed
        effect can be absorbed; ``cut1`` is then normalized to 0.""" + _COMMON + """
    levels : sequence, optional
        The outcome's levels from lowest to highest.""" + _RANDOM + """

    Returns
    -------
    OrderedLogitResult or LogitMulti

    Notes
    -----
    The cutpoints are estimated directly (not as increments), so the
    reported standard errors are theirs; a step that would put them out of
    order is halved, which the concavity of the log-likelihood makes safe.
    """

oprobit_stream.__doc__ = ologit_stream.__doc__.replace(
    "Out-of-core ordered logit by streamed Newton.",
    "Out-of-core ordered probit by streamed Newton.").replace(
    "F the logistic distribution", "F the standard normal distribution").replace(
    "OrderedLogitResult or LogitMulti", "OrderedProbitResult or LogitMulti") + """
    Random intercepts (``random=``) and ``firth=`` are not available for the
    ordered probit.
    """

cfelogit_stream.__doc__ = """Out-of-core conditional fixed-effects logit
    (Chamberlain 1980; Stata's ``xtlogit, fe``, R's ``survival::clogit``),
    by streamed Newton.

    For a binary outcome observed several times per unit (a panel), the
    likelihood of each unit's outcomes given their number of 1s does not
    depend on the unit's effect, which is thus conditioned out rather than
    estimated: the estimates are consistent however few the periods, where
    the logit with the effects estimated (`logit_stream`) is biased with
    short panels (see its ``bias_correction=``, for longer ones). Units whose
    outcome never varies carry no information and are dropped. There are no
    unit effects, so no predicted probabilities or marginal effects.

    Parameters
    ----------
    fml : str
        A pyfixest-style formula whose one fixed effect is the unit, e.g.
        ``"employed ~ age + C(year) | person_id"``. Covariates constant
        within every unit are dropped (their effect is the unit's).""" + _COMMON + """

    Returns
    -------
    ConditionalFELogitResult or LogitMulti

    Notes
    -----
    Each unit's contribution is log P(y | sum y = k) = sum y eta -
    log e_k(exp eta), e_k the elementary symmetric polynomial of degree k,
    computed with its gradient and information in one recursion over the
    unit's rows: O(T k P^2) work for a unit of T rows (k <= T / 2), so long
    panels with many 1s are costly. Each unit's rows are written together in
    pass 0 (sorted by unit if they are not already). The robust variance
    treats units as the observations (clustered by unit); weights and
    clusters must be constant within units.
    """

clogit_stream.__doc__ = """Out-of-core conditional (McFadden) logit by streamed Newton.

    Long-format data: one row per alternative of each choice set, the
    outcome 1 on the chosen alternative and 0 on the others. P(alternative
    a chosen in set i) = exp(x_ia'b) / sum over the set's alternatives.
    Alternatives that were not available are simply absent. Each Newton
    step is one read of the sample's blocks, each of which holds whole
    choice sets.

    Parameters
    ----------
    fml : str
        A pyfixest-style formula, e.g.
        ``"choice ~ cost + time + C(alt)"`` (``C(alt)``: alternative-specific
        constants). There is no intercept, and a variable constant within a
        choice set cancels: interact it with the alternatives
        (``income:C(alt)``).
    case : str or sequence of str
        The column(s) identifying a choice set. Rows of a set need not be
        consecutive (if they are not, pass 0 sorts them, out of core).""" + _COMMON + """
    altwise : bool, default False
        A choice set with an alternative whose values are missing is dropped
        whole, with a warning, as Stata's ``cmclogit`` does: fitted on its
        other alternatives it would be a different choice set. With
        ``altwise=True`` only the incomplete alternatives are dropped (a set
        that loses its chosen alternative is then an error).

    Returns
    -------
    ConditionalLogitResult or LogitMulti

    Notes
    -----
    Every choice set must have exactly one chosen alternative after rows with
    missing values are dropped; weights and clusters must be constant within
    a set. The robust and clustered variances treat a choice set as the
    unit.
    """


mixlogit_stream.__doc__ = """Out-of-core mixed logit by streamed maximum simulated likelihood
    (prototype: normal and lognormal coefficients, uncorrelated).

    The conditional logit of `clogit_stream`, with the coefficients named in
    `random` varying across choosers: normal, mu + s z, or lognormal,
    exp(mu + s z), z standard normal. A chooser's probability of their
    choices is simulated with `draws` draws, which are regenerated in each
    read of the data rather than stored. The optimizer takes BHHH steps,
    then Newton steps with a finite-difference Hessian of the analytic
    gradient; see `logit_stream.mixed`.

    Parameters
    ----------
    fml : str
        ``"chosen ~ x1 + x2 + ..."``, as for `clogit_stream`; the outcome is
        1 for the chosen alternative, 0 otherwise.
    data : str, Path, LazyFrame or DataFrame
        Long-format data: one row per available alternative.
    case : str or list of str
        The column(s) identifying each choice set.
    random : dict
        Covariate name -> ``"normal"`` or ``"lognormal"``. A lognormal
        coefficient is positive: for a negative effect (a cost), use the
        negated covariate.
    panel : str or list of str, optional
        The column(s) identifying each chooser, whose choice sets share one
        draw of the random coefficients (a panel of repeated choices). By
        default each choice set is its own chooser.
    workdir : str or Path, optional
        Where the blocks are written (as for `clogit_stream`).
    vcov : {"iid", "hetero"} or {"CRV1": panel}
        The variance reported: the inverse Hessian, or the sandwich with the
        choosers' scores -- clustered by chooser, so ``{"CRV1": panel}``
        (G / (G - 1), df G - 1) with `panel`, and ``"hetero"`` without one
        (each choice set its own chooser). Both are computed. The random
        coefficients' standard deviations are reported as absolute values (their sign
        is not identified; the standard errors are the same), the signed
        estimates in ``diagnostics["sd_signed"]``.
    draws : int
        Draws per chooser (default 500).
    halton : bool
        Halton draws (default) or pseudo-random ones.
    seed : int
        The pseudo-random draws' seed.
    drop : int
        Halton points discarded at the start of each sequence (default 100).
    tol, maxiter, switch : float, int, float
        Convergence when the Newton decrement is below `tol` (1e-8); at
        most `maxiter` steps (200); BHHH steps until their decrement is
        below `switch` (default: none, Newton throughout, with BHHH steps
        where the Hessian is not negative definite).
    draws_cache_mb : float
        Keep all choosers' draws in memory if they fit (default 512 MB);
        otherwise they are regenerated in every pass. With ``device="cuda"``
        the budget is the device's memory, where the draws are generated.
    device : {"cpu", "cuda"}
        ``"cuda"`` (experimental): the passes on an NVIDIA GPU through CuPy,
        the likelihood vectorized over rows and draws (on the CPU a numba
        kernel, which is about ten times faster there than the vectorized
        form).
    **options
        `storage`, `dtype`, `block_mb`, `polars_engine`, `keep_intermediates`,
        `verbose`, `logger`, `log_level`, as for `clogit_stream`.

    Returns
    -------
    MixedLogitResult
        `summary`, `tidy`, `confint`, `wald_test` and the other reporting
        methods; `predict` and `resid` are not in the prototype.
    """
