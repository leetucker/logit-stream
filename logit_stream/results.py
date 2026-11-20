"""Fit results: `MultinomialLogitResult`, `OrderedLogitResult`,
`ConditionalLogitResult`, and `LogitMulti` for a formula that expands to
several models.

They are `hdfe_stream.core.results_base.RegressionResult` subclasses, so the
coefficient table, confidence intervals, Wald tests, `with_vcov`, the
``summary*`` reports, `etable` and the multiple-testing corrections are
hdfe-stream's own, and the conventions (names, JSON fields, normal inference
for maximum likelihood) are the same as for its GLMs.

Nothing row-sized is kept: `predict`, `resid` and `sample` are lazy Polars
frames over the source (or new data), evaluated when they are collected or
sunk, with the fit's coefficients -- and its fixed effects, which are small
tables held in memory -- applied by a numpy function on each batch. So a
result holds no files, needs no cleanup, and can predict for data far larger
than memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from hdfe_stream.core.results_base import RegressionMulti, RegressionResult
from scipy import special

from .mi import categorical
from .models import LINKS
from .store import check_source


def as_lazy(source):
    from hdfe_stream.core.utils import as_lazy as f
    return f(source)


# --------------------------------------------------------------------------
# shared
# --------------------------------------------------------------------------

@dataclass(kw_only=True)
class LogitResult(RegressionResult):
    """What every logit result has; see the subclasses.

    Attributes
    ----------
    fml, depvar, coefnames, beta, vcov, vcov_type, df_t, n_obs, all_vcovs, n_clusters, \
        collin_vars, diagnostics, weights, weights_type
        As in `hdfe_stream.HDFEResult`. `df_t` is used only by `wald_test`
        with ``distribution="F"``; p-values and intervals are normal.
    model : str
        ``"multinomial logit"``, ``"ordered logit"`` or ``"conditional logit"``.
    loglik : float
        The log-likelihood at the estimates (without the L2 or Firth
        penalty).
    loglik_null : float
        The null model's: the outcome's sample shares (multinomial, ordered),
        or equal probabilities within each choice set (conditional).
    pseudo_r2 : float
        McFadden's, 1 - loglik / loglik_null: with a fixed effect, it credits
        the covariates and the fixed effect together.
    loglik_fe_null : float
        With an absorbed fixed effect, the log-likelihood of the model with
        it and no covariates: each level's outcome shares (multinomial), or
        a location per level and common cutpoints (ordered); computed from
        the outcome counts per level, without reading the data. Else nan.
    pseudo_r2_within : float
        1 - loglik / loglik_fe_null: what the covariates add beyond the fixed
        effect (fixest's "within pseudo R2"); nan without one.
    aic, bic : float
        -2 loglik + 2 k and -2 loglik + k log N, with k the estimated
        parameters, fixed effects included.
    n_params : int
        k.
    fe_names : list of str
        The absorbed fixed-effect dimension, if any.
    random_effects : dict
        With random intercepts (``random=``): the grouping (`group`), its
        number of groups, the covariance of the intercepts (`Sigma`, over the
        non-base outcome levels; 1 x 1 for the ordered logit), its form
        (`cov`), the outcome levels it is over, the method, and Sigma's
        standard errors: `Sigma_se` (each element's), `sd` and `sd_se`,
        `corr` and `corr_se` (delta method), and `vcov_Sigma` with its
        `Sigma_params` (the (j, l) elements it is over). Empty otherwise.
    n_levels : dict
        Its number of levels.
    l2 : float
        The L2 penalty (0: maximum likelihood).
    firth : bool
        Whether the estimates maximize Firth's penalized likelihood
        (``firth=True``).
    converged : bool
        Whether Newton's method met its tolerance.
    """

    model: str
    loglik: float
    loglik_null: float
    pseudo_r2: float
    aic: float
    bic: float
    n_params: int
    converged: bool
    fe_names: list = field(default_factory=list)
    n_levels: dict = field(default_factory=dict)
    random_effects: dict = field(default_factory=dict)
    l2: float = 0.0
    loglik_fe_null: float = np.nan
    pseudo_r2_within: float = np.nan
    firth: bool = False
    _spec: dict | None = field(default=None, repr=False, compare=False)

    _normal_inference = True

    # ------------------------------------------------------------ helpers
    def _frame(self, newdata):
        spec = self._spec
        if spec is None:
            raise ValueError("this result has no reference to its data or design")
        if newdata is None:
            frame = spec["source"]
        else:
            frame = as_lazy(newdata)
        check_source(frame.collect_schema().names())
        return frame

    def _xexprs(self):
        """The design's columns the estimates use, as Polars expressions."""
        return [(nm, self._spec["xexprs"][nm]) for nm in self._spec["xcols"]]

    def _in_sample(self, frame):
        """The source with `row_id`, restricted to the estimation sample."""
        spec = self._spec
        frame = frame.with_row_index("row_id").with_columns(pl.col("row_id").cast(pl.Int64))
        frame = frame.filter(spec["usable"])
        if spec.get("incomplete") is not None:
            table, cols = spec["incomplete"]
            frame = frame.join(table.lazy(), on=cols, how="anti", maintain_order="left")
        for d, table in spec.get("dropped", {}).items():
            frame = frame.join(table.lazy(), on=spec["fe_cols"][d], how="anti",
                               maintain_order="left")
        return frame

    def sample(self, check: bool = True) -> pl.LazyFrame:
        """The source, lazily, with `row_id` (the row's position) first and
        two columns added: ``in_sample``, and ``dropped_because`` (null for
        the rows used; ``"missing"`` for a null or non-finite value in a
        column the model uses, ``"incomplete set"`` for the other
        alternatives of a choice set with such a row (conditional logit,
        without ``altwise=True``), ``"separation"`` for a fixed-effect level
        whose outcome never varies)."""
        spec = self._spec
        source = spec["source"]
        names = source.collect_schema().names()
        if check:
            n = source.select(pl.len()).collect(engine="streaming").item()
            if n != spec["n_raw"]:
                raise ValueError(f"the source has {n:,} rows, and had {spec['n_raw']:,} when "
                                 "the model was fitted")
        frame = source.with_row_index("row_id").with_columns(pl.col("row_id").cast(pl.Int64))
        reason = pl.when(~spec["usable"]).then(pl.lit("missing"))
        if spec.get("incomplete") is not None:
            table, cols = spec["incomplete"]
            frame = frame.join(table.lazy().with_columns(pl.lit(True).alias("__ls_inc")),
                               on=cols, how="left", maintain_order="left")
            reason = reason.when(pl.col("__ls_inc").is_not_null()).then(pl.lit("incomplete set"))
        for d, table in spec.get("dropped", {}).items():
            flag = f"__ls_sep_{d}"
            frame = frame.join(table.lazy().with_columns(pl.lit(True).alias(flag)),
                               on=spec["fe_cols"][d], how="left", maintain_order="left")
            reason = reason.when(pl.col(flag).is_not_null()).then(pl.lit("separation"))
        out = frame.with_columns(reason.otherwise(None).cast(
            pl.Enum(["missing", "incomplete set", "separation"])).alias("dropped_because"))
        out = out.with_columns(pl.col("dropped_because").is_null().alias("in_sample"))
        return out.select("row_id", *names, "in_sample", "dropped_because")

    def fixef(self, name: str | None = None) -> pl.LazyFrame:
        """The absorbed fixed effects: one row per level of the dimension,
        with its effects (see the subclass), their standard errors
        (``se_<name>``, or ``se_<name>:<level>``, where `draw_parameters` is
        available: the iid variance given the coefficients plus what the
        coefficients' uncertainty, from `vcov`, adds) and its number of
        observations.
        The table is held in memory; this is a lazy view of it."""
        tables = self._spec.get("fixef", {}) if self._spec else {}
        if not tables and self.random_effects:
            raise KeyError("this model has random intercepts, not fixed effects; see ranef()")
        if not tables:
            raise KeyError("this model has no absorbed fixed effects")
        if name is None:
            if len(tables) > 1:
                raise ValueError(f"name one of {list(tables)}")
            name = next(iter(tables))
        if name not in tables:
            raise KeyError(f"{name!r} is not a fixed effect of this model; it has "
                           + ", ".join(map(repr, tables)))
        table = tables[name]
        if self._spec.get("draw_index") is None:
            from .draws import level_se
            se = level_se(self)
            if se is not None:
                cols = [c for d, values, kind in self._spec["effects"] if d == name
                        for c in values]
                ses = [pl.Series("se_" + c[3:], np.where(np.isfinite(se[:, j]), se[:, j], np.nan))
                       .fill_nan(None) for j, c in enumerate(cols)]
                rest = [c for c in table.columns if c != "n_obs"]
                table = table.with_columns(ses).select(*rest, *[x.name for x in ses],
                                                       *(["n_obs"] if "n_obs" in table.columns
                                                         else []))
        return table.lazy()

    def ranef(self, name: str | None = None) -> pl.LazyFrame:
        """A random grouping's predicted intercepts (their modes given the
        data, the conditional means' Laplace approximation): one row per
        group, with one column per outcome level (the base level's 0) for
        the multinomial logit, one for the ordered logit, then each
        effect's conditional standard deviation (``sd_<grouping>``, or
        ``sd_<grouping>:<level>``, 0 for the base level; lme4's ``condVar``:
        given the coefficients, any fixed effect and Sigma), and the group's
        number of observations. `predict` adds every grouping's to the
        linear predictor, using 0 for a group the fit did not see. `name`:
        the grouping, needed when there are several."""
        if not self.random_effects:
            raise KeyError("this model has no random intercepts")
        if self._spec.get("draw_index") is not None and not self._spec.get("ranef_drawn"):
            raise ValueError("a parameter draw does not carry the group intercepts: draw them "
                             "given the draw (draw_ranef, or draw_parameters(ranef=True))")
        tables = self._spec["ranef"]
        if name is None:
            if len(tables) > 1:
                raise ValueError(f"name one of {list(tables)}")
            name = next(iter(tables))
        if name not in tables:
            raise KeyError(f"{name!r} is not a random grouping of this model; it has "
                           + ", ".join(map(repr, tables)))
        return tables[name].lazy()

    def draw_parameters(self, n: int, seed: Any = None, ranef: bool = False,
                        newdata: Any = None) -> list:
        """`n` independent draws of the parameters from their estimated
        sampling distribution (the normal approximation: the estimates as
        the mean, their variance as the variance), each returned as a copy
        of this result at the drawn values, for multiple imputation or for
        simulating the uncertainty of a prediction: its `beta`, `fixef()`,
        `random_effects` (``sd`` and ``Sigma``) and `predict` are the draw's.

        The coefficients are drawn from N(beta, vcov), with this result's
        `vcov` (iid or robust). An absorbed fixed effect's levels are drawn
        jointly with them: each level from its distribution given the drawn
        coefficients, from the model's information (exact for the normal
        approximation, since a row is in one level; nothing is read). With a
        random intercept fitted by ``random_method="adaptive"``, the
        coefficients and log sd are drawn jointly (from the inverse of their
        information net of the fixed effect, as `vcov` is), then the fixed
        effect's levels given them. The groups' own intercepts are not part
        of a draw: a drawn result's `predict` leaves them out (as for a group
        the fit did not see), and its `ranef()` raises; draw each group's
        intercept from its distribution given its rows and the draw.

        Available without a fixed effect, with one absorbed fixed effect,
        and with an adaptive random intercept; not with the Laplace or
        joint-mode random intercepts, several fixed effects, the mixed or
        nested logit. `seed`: an int, None or a `numpy.random.Generator`.

        For the multinomial logit's random intercepts (adaptive fit), Sigma
        is drawn on the scale of its log-Cholesky factor, carried over from
        the information by the delta method, so every draw is a valid
        covariance matrix.

        With ``ranef=True`` (a model with one random grouping), each draw
        also carries a draw of every group's intercept given the group's
        rows and that draw's parameters, as `draw_ranef` makes them, all
        `n` from one read of the estimation sample; then
        each draw's `predict` includes them and `ranef()` returns them.
        `newdata`: as for `draw_ranef`."""
        from .draws import draw, draw_ranef_many
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        if ((self._spec or {}).get("draw") or {}).get("kind") == "alt_slopes":
            # random slopes by region: every region's slopes are part of each draw
            from .draws import draw_slopes
            return draw_slopes(self, n, rng, newdata)
        out = draw(self, n, rng)
        if ranef:
            out = draw_ranef_many(out, rng, newdata)
        elif newdata is not None:
            raise ValueError("newdata= is for ranef=True")
        return out

    def draw_ranef(self, seed: Any = None, newdata: Any = None):
        """A copy of this result with one draw of every group's random
        intercept from its conditional distribution: given the group's rows
        in the estimation sample and this result's parameters (the
        estimates, or a draw from `draw_parameters`),

            p(d_g | rows) proportional to prod_i P(y_i | eta_i + d_g)^w_i N(d_g; 0, sd^2),

        with eta_i each row's own linear predictor without the group's
        intercept. Each group's draw is exact (on a grid of the density
        around its mode, sampled by its inverse cdf), whatever its size or
        how unbalanced its outcomes. The copy's `ranef()` holds the draws
        (``re_<grouping>``) and each group's conditional sd at these
        parameters (``sd_<grouping>``); its `predict` adds them, so its
        predictions for a group's other members (those to impute, say) use
        one shared draw per group. `newdata`: rows (or just the grouping's
        columns) whose groups the fit did not see; each such group gets a
        draw from N(0, sd^2), so that `predict` on them shares it too.

        Reads the estimation sample once (from the fit's source), sorted by
        group. For several imputations, ``draw_parameters(n, ranef=True)``
        draws all of them in that one read. Available for the binary,
        ordered and multinomial logit (and probit) with one random grouping.

        For the multinomial logit each group's J - 1 intercepts are drawn
        jointly from N(0, Sigma) times its rows' probabilities, by
        importance resampling: from the mode (Newton's method) and the
        inverse curvature there, 1,000 candidates from a multivariate t
        with 5 degrees of freedom, one picked in proportion to its weight
        (exact as the candidates grow; a group whose weights' effective
        number is under 200 is redone with 16 times as many). The copy's
        ``diagnostics["ranef_ess"]`` has each group's effective number."""
        from .draws import draw_ranef_many
        return draw_ranef_many([self], seed, newdata)[0]

    def varcorr(self) -> pl.DataFrame:
        """The random intercepts' standard deviations and correlations, one
        row each per grouping, with their standard errors and 95% Wald
        intervals on the scales that respect their bounds (log sd, Fisher's
        z of a correlation)."""
        if not self.random_effects:
            raise KeyError("this model has no random intercepts")
        rows = []
        for name, re in self.random_effects.items():
            lv = re["levels"]
            for i, l in enumerate(lv):
                rows.append((name, f"sd({l})", float(re["sd"][i]), float(re["sd_se"][i]),
                             float(re["sd_ci"][i][0]), float(re["sd_ci"][i][1])))
            if re["cov"] != "diagonal":
                for i in range(len(lv)):
                    for j in range(i + 1, len(lv)):
                        rows.append((name, f"corr({lv[i]},{lv[j]})", float(re["corr"][i][j]),
                                     float(re["corr_se"][i][j]), float(re["corr_ci"][i][j][0]),
                                     float(re["corr_ci"][i][j][1])))
        return pl.DataFrame(rows, schema=["group", "term", "estimate", "std_error", "ci_lower",
                                          "ci_upper"], orient="row")

    def _with_fe(self, frame):
        """Join the fixed and random effects onto `frame` (columns
        `__ls_fe<k>`, table after table, in `_spec["effects"]`'s order)."""
        spec = self._spec
        tmp = []
        for d, values, kind in spec.get("effects", []):
            table = spec["fixef" if kind == "fixed" else "ranef"][d]
            cols = spec["fe_cols"][d]
            names = [f"__ls_fe{len(tmp) + k}" for k in range(len(values))]
            tmp += names
            tschema = table.collect_schema()
            right = table.lazy().select(*cols, *[pl.col(e).alias(t)
                                                 for e, t in zip(values, names)])
            frame = frame.with_columns(pl.col(c).cast(tschema[c], strict=False) for c in cols)
            frame = frame.join(right, on=cols, how="left", maintain_order="left")
        return frame, tmp

    def _matrix(self, df: pl.DataFrame, k, n_fe):
        """The batch's covariates (its first k columns) and fixed effects
        (the next n_fe), as float arrays; A is None without fixed effects."""
        X = df.select(df.columns[:k]).to_numpy().astype(np.float64) if k else \
            np.zeros((df.height, 0))
        if not n_fe:
            return X, None
        # each table's effects, summed: a random effect's group the fit did
        # not see gets its mean, 0; a fixed effect's keeps its nulls (levels
        # that never occur in the group, and unseen groups)
        Araw = df.select(df.columns[k:k + n_fe]).to_numpy().astype(np.float64)
        A, at = None, 0
        for _, values, kind in self._spec["effects"]:
            block = Araw[:, at:at + len(values)]
            at += len(values)
            if kind == "random":
                block = np.nan_to_num(block)
            A = block if A is None else A + block
        return X, A

    # ------------------------------------------------------------ reporting hooks
    def _summary_lines(self) -> list[str]:
        lines = [f"{self.model.capitalize()} (logit_stream, Newton)   vcov: {self.vcov_type}"]
        lines += self._model_lines()
        lines.append(f"log-likelihood: {self.loglik:.8g}   null: {self.loglik_null:.8g}   "
                     f"pseudo R2: {self.pseudo_r2:.6f}   AIC: {self.aic:.8g}   "
                     f"BIC: {self.bic:.8g}")
        if not np.isnan(self.pseudo_r2_within):
            lines.append(f"within pseudo R2 (beyond {self.fe_names[0]}): "
                         f"{self.pseudo_r2_within:.6f}   fixed-effect-only log-likelihood: "
                         f"{self.loglik_fe_null:.8g}")
        nt = self.diagnostics.get("newton", {})
        if nt:
            lines.append(f"Newton steps: {nt['iterations']} ("
                         + ("converged" if nt["converged"] else "NOT converged")
                         + f", decrement {nt['decrement']:.2e}"
                         + (f", {nt['step_halvings']} step-halvings" if nt["step_halvings"]
                            else "") + ")")
        if self.fe_names:
            lines.append("fixed effects: " + ", ".join(
                f"{d} ({self.n_levels.get(d, 0):,} levels)" for d in self.fe_names))
            sep = self.diagnostics.get("separation", {})
            if sep.get("observations"):
                lines.append(f"dropped for separation: {sep['observations']:,} observations "
                             f"in {sep['levels']:,} levels whose outcome never varies")
        if self.random_effects:
            first = next(iter(self.random_effects.values()))
            how = {"laplace": "Laplace approximation",
                   "joint-mode": "Laplace approximation at the joint mode",
                   "adaptive": f"adaptive Gauss-Hermite quadrature, {first.get('nodes')} "
                               "nodes"}.get(first.get("method"), "Laplace approximation")
            names = list(self.random_effects)
            struct = ("" if len(names) == 1 else
                      " (nested: " + " in ".join(names) + ")" if first["structure"] == "tree"
                      else " (crossed: " + " x ".join(names) + ")")
            lines.append(f"random intercepts{struct}; {how}")

            def se(x):
                return f" ({x:.4f})" if x is not None and np.isfinite(x) else ""
            for re in self.random_effects.values():
                sd = np.sqrt(np.diag(re["Sigma"]))
                sd_se = re.get("sd_se")
                lines.append(f"  by {re['group']} ({re['n_groups']:,} groups, {re['cov']} "
                             "covariance), sd (se): " + "   ".join(
                                 f"{lv}: {v:.4f}{se(sd_se[i] if sd_se is not None else None)}"
                                 for i, (lv, v) in enumerate(zip(re["levels"], sd))))
                if len(sd) > 1 and re["cov"] != "diagonal":
                    R = np.asarray(re["Sigma"]) / np.outer(sd, sd)
                    cse = re.get("corr_se")
                    pairs = [f"{re['levels'][i]}~{re['levels'][j]}: {R[i, j]:+.3f}"
                             + se(cse[i][j] if cse is not None else None)
                             for i in range(len(sd)) for j in range(i + 1, len(sd))]
                    lines.append("    correlations: " + "   ".join(pairs))
        if self.weights:
            lines.append(f"weights: {self.weights} ({self.weights_type})")
        if self.l2:
            lines.append(f"L2 penalty: {self.l2:g} (standard errors are those of the "
                         "penalized fit)")
        if self.firth:
            lines.append("Firth's penalized likelihood (bias-reduced estimates; standard "
                         "errors from the information at them)")
        if self.collin_vars:
            lines.append(f"dropped as collinear: {self.collin_vars}")
        return lines

    def _model_lines(self):
        return [f"obs: {self.n_obs:,}"]

    def _summary_fields(self) -> dict:
        d = {
            "formula": self.fml,
            "depvar": self.depvar,
            "model": self.model,
            **self._model_fields(),
            "vcov": self.vcov_type,
            "df_t": self.df_t,
            "n_obs": self.n_obs,
            "n_clusters": dict(self.n_clusters),
            "fixed_effects": list(self.fe_names),
            "n_levels": dict(self.n_levels),
            "loglik": self.loglik,
            "loglik_null": self.loglik_null,
            "pseudo_r2": self.pseudo_r2,
            "loglik_fe_null": self.loglik_fe_null,
            "pseudo_r2_within": self.pseudo_r2_within,
            "aic": self.aic,
            "bic": self.bic,
            "n_params": self.n_params,
            "converged": self.converged,
            "newton": self.diagnostics.get("newton"),
            "l2": self.l2,
            "firth": self.firth,
            "weights": self.weights,
            "weights_type": self.weights_type if self.weights else None,
            "collinear_dropped": list(self.collin_vars),
        }
        if self.fe_names:
            d["separation"] = self.diagnostics.get("separation")
        if self.random_effects:
            d["random_effects"] = {
                name: {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in info.items()}
                for name, info in self.random_effects.items()}
        return d

    def _model_fields(self):
        return {}

    def _pyfixest_attrs(self) -> dict:
        return {
            "_fixef": "+".join(self.fe_names) if self.fe_names else None,
            "_has_fixef": bool(self.fe_names),
            "_r2": np.nan, "_adj_r2": np.nan, "_r2_adj": np.nan,
            "_r2_within": np.nan, "_adj_r2_within": np.nan, "_rmse": np.nan,
            "deviance": -2 * self.loglik, "_method": "feglm",
            "_loglik": self.loglik, "_pseudo_r2": self.pseudo_r2,
            "_is_iv": False, "_f_stat_1st_stage": None,
        }


def _level_name(v):
    return str(v)


def _prob_dtype(levels, prefix):
    return pl.Struct({f"{prefix}_{_level_name(v)}": pl.Float64 for v in levels})


# --------------------------------------------------------------------------
# multinomial
# --------------------------------------------------------------------------

@dataclass(kw_only=True)
class MultinomialLogitResult(LogitResult):
    """The result of `mlogit_stream`.

    The coefficients are those of each non-base level against the base
    level (the first of `levels`), named ``"<level>:<covariate>"``; the
    base level's are zero. With an absorbed fixed effect, each of its levels
    has an effect for every outcome level but its reference, see `fixef`.

    Attributes
    ----------
    levels : list
        The outcome's levels, base first.
    base : object
        The base level.
    (and those of `LogitResult`)
    """

    levels: list
    base: Any = None

    def _model_lines(self):
        return [f"obs: {self.n_obs:,}   outcome {self.depvar}: {len(self.levels)} levels, "
                f"base {self.base!r}"]

    def _model_fields(self):
        return {"levels": list(self.levels), "base": self.base}

    def coef_matrix(self) -> pl.DataFrame:
        """The coefficients as a table: a row per covariate, a column per
        non-base level."""
        xs = self._spec["xcols"]
        B = self._B()
        free = self._spec.get("free")
        if free is not None:                # an own covariate's other equations
            B = np.where(free.T, B, np.nan)
        return pl.DataFrame({"Coefficient": xs,
                             **{_level_name(lv): B[:, j] for j, lv in enumerate(self.levels[1:])}})

    def _B(self):
        J1, K = len(self.levels) - 1, len(self._spec["xcols"])
        free = self._spec.get("free")
        if free is None:
            return np.asarray(self.beta).reshape(J1, K).T
        B = np.zeros((J1, K))
        B[free] = self.beta
        return B.T

    def predict(self, newdata: Any = None, type: str = "response",
                prefix: str | None = None, seed: int = 0,
                id: str | list | None = None, effects: str = "mode") -> pl.LazyFrame:
        """Predictions as a lazy frame: `newdata` (default: the fit's source,
        every row) with columns added.

        type="response"
            The probability of each level, ``prob_<level>``.
        type="link"
            The linear predictor of each non-base level, ``xb_<level>``
            (with its fixed effect).
        type="class"
            The most probable level, ``predicted``.
        type="draw"
            A level drawn from the row's probabilities, ``drawn``: for
            imputation (with the parameters of a `draw_parameters` draw,
            for multiple imputation). The draw's uniform depends only on
            `seed` and the row's identifier, the column(s) `id` (default:
            the row's position in `newdata`), not on the batches; see
            `logit_stream.mi`.

        A row with a null or non-finite covariate, or a fixed-effect level
        the fit did not estimate, gets nulls. `prefix` renames the columns'
        prefix (or, for "class" and "draw", the column).

        `effects` says how a random intercept enters a probability:
        ``"mode"`` (the default) at the group's predicted effect, its
        conditional mode (0, the mean, for a group the fit did not see);
        ``"average"`` averaged over the effect's distribution: the group's
        conditional one given its rows (normal, with the mode as mean and
        the conditional covariance, whose diagonal `ranef` reports as
        ``sd_<grouping>``), or for a group the fit did not see the prior,
        N(0, Sigma). The averaged probability is the expected share of
        such rows across groups like the row's, which the probability at
        the mean effect is not: it is too extreme, by more the larger
        Sigma. For "response" and "class" (the most probable level of the
        averaged probabilities), by Gauss-Hermite quadrature.
        """
        if type not in ("response", "link", "class", "draw"):
            raise ValueError(f"type must be 'response', 'link', 'class' or 'draw', got {type!r}")
        _check_effects(effects, type)
        frame = self._frame(newdata)
        if effects == "average":
            return self._predict(frame, type, prefix, average=True)
        if type != "draw":
            return self._predict(frame, type, prefix)
        from .mi import uniform_expr, with_ids
        frame, ids, drop = with_ids(frame, id)
        return self._predict(frame, type, prefix, u=uniform_expr(ids, seed)).drop(drop)

    def _predict(self, frame, type, prefix, u=None, average=False):
        xs = self._xexprs()
        k = len(xs)
        frame, fe = self._with_fe(frame)
        gnames, covs = [], []
        if average:
            from ._average import spread_inputs
            frame, gnames, covs = spread_inputs(self, frame)
        B = self._B()
        levels = self.levels
        if type == "response":
            pre = prefix or "prob"
            dtype = _prob_dtype(levels, pre)
        elif type == "link":
            pre = prefix or "xb"
            dtype = _prob_dtype(levels[1:], pre)
        else:
            pre = prefix or ("drawn" if type == "draw" else "predicted")
            lv_series = pl.Series(levels)
            dtype = pl.Struct({pre: lv_series.dtype})

        def fn(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            X, A = self._matrix(df, k, len(fe))
            full = np.column_stack([np.zeros(len(X)), X @ B])
            if A is not None:
                # fixed effects, one column per level: a level that never
                # occurs in the row's group has a null effect, and so
                # probability 0; a group the fit did not see, all nulls
                unseen = np.all(np.isnan(A), axis=1)
                full = np.where(np.isnan(A), -np.inf, full + np.nan_to_num(A))
                full[unseen] = np.nan
            if type == "link":
                with np.errstate(invalid="ignore"):
                    E = full[:, 1:] - full[:, :1]
                E[~np.isfinite(E)] = np.nan
                out = {f"{pre}_{_level_name(lv)}": E[:, j] for j, lv in enumerate(levels[1:])}
                return pl.DataFrame(out).fill_nan(None).to_struct(s.name)
            if average:
                from ._average import average_softmax, row_covariance
                P = average_softmax(full, row_covariance(df, gnames, covs, X, len(levels)))
            else:
                with np.errstate(invalid="ignore"):
                    m = np.max(full, axis=1, keepdims=True)
                    e = np.exp(full - m)
                    P = e / e.sum(axis=1, keepdims=True)
            if type == "response":
                out = {f"{pre}_{_level_name(lv)}": P[:, j] for j, lv in enumerate(levels)}
                return pl.DataFrame(out).fill_nan(None).to_struct(s.name)
            if type == "draw":
                idx = categorical(P, df["__ls_u"].to_numpy())
                ok = idx >= 0
                idx = np.maximum(idx, 0)
            else:
                ok = ~np.isnan(P).any(axis=1)
                idx = np.where(ok, np.nanargmax(np.where(ok[:, None], P, 0), axis=1), 0)
            col = pl.Series(pre, [levels[i] for i in idx], dtype=lv_series.dtype)
            col = pl.select(pl.when(pl.Series(ok)).then(col).otherwise(None)).to_series()
            return pl.DataFrame({pre: col}).to_struct(s.name)

        inputs = [ex.alias(f"__ls_p{j}") for j, (_, ex) in enumerate(xs)] + \
                 [pl.col(c) for c in fe] + ([u.alias("__ls_u")] if u is not None else []) \
            + [pl.col(g) for g in gnames]
        pred = (pl.struct(inputs) if inputs else pl.struct(pl.lit(0.0).alias("__ls_none")))
        tmp = "__ls_pred"
        return (frame.with_columns(pred.map_batches(fn, return_dtype=dtype, is_elementwise=True)
                                   .alias(tmp))
                     .drop(*fe, *gnames).unnest(tmp))

    def resid(self) -> pl.LazyFrame:
        """The estimation sample, lazily: `row_id` (the row's position in the
        source), the outcome, each level's probability ``prob_<level>``, and
        each level's response residual ``resid_<level>`` = 1[y = level] -
        prob. Computed from the source when collected (nothing is stored)."""
        spec = self._spec
        frame = self._in_sample(spec["source"])
        y = spec["yexpr"]
        frame = frame.select("row_id", y.alias(self.depvar), *self._input_columns(frame))
        out = self._predict(frame, "response", "prob")
        return out.select("row_id", self.depvar,
                          *[f"prob_{_level_name(lv)}" for lv in self.levels],
                          *[((pl.col(self.depvar) == pl.lit(lv)).cast(pl.Float64)
                             - pl.col(f"prob_{_level_name(lv)}")).alias(f"resid_{_level_name(lv)}")
                            for lv in self.levels])

    def _input_columns(self, frame):
        names = set(frame.collect_schema().names())
        need = set()
        for _, ex in self._xexprs():
            need |= set(ex.meta.root_names())
        for d in self.fe_names:
            need |= set(self._spec["fe_cols"][d])
        off = self._spec.get("offset")
        if off is not None:
            need |= set(off.meta.root_names())
        return [pl.col(c) for c in sorted(need & names) if c not in ("row_id", self.depvar)]


def _check_effects(effects, type):
    if effects not in ("mode", "average"):
        raise ValueError(f"effects must be 'mode' or 'average', got {effects!r}")
    if effects == "average" and type not in ("response", "class"):
        raise ValueError('effects="average" averages probabilities: it is for type="response" '
                         'or "class" (for draws that keep each group\'s effect shared by its '
                         'rows, use draw_parameters(ranef=True, newdata=...))')


# --------------------------------------------------------------------------
# ordered
# --------------------------------------------------------------------------

@dataclass(kw_only=True)
class OrderedLogitResult(LogitResult):
    """The result of `ologit_stream`.

    P(y <= level j) = F(cut_j - x'b), F the logistic distribution: positive
    coefficients move probability toward the higher levels. The cutpoints
    are the last coefficients, ``cut1`` ... ``cut{J-1}``, ``cut_j`` between
    levels j and j + 1. With an absorbed fixed effect, ``cut1`` is
    normalized to zero (the effects carry the location) and is not reported.

    Attributes
    ----------
    levels : list
        The outcome's levels, lowest first.
    (and those of `LogitResult`)
    """

    levels: list

    link = "logit"          # the error distribution (OrderedProbitResult: "probit")

    def _model_lines(self):
        lv = ", ".join(map(repr, self.levels[:8])) + (", ..." if len(self.levels) > 8 else "")
        return [f"obs: {self.n_obs:,}   outcome {self.depvar}: {len(self.levels)} ordered levels "
                f"({lv})"]

    def _model_fields(self):
        return {"levels": list(self.levels)}

    @property
    def cutpoints(self) -> np.ndarray:
        """All J - 1 cutpoints (with a fixed effect, the first is 0)."""
        c = np.asarray(self.beta)[len(self._spec["xcols"]):]
        return np.r_[0.0, c] if self.fe_names else c

    def predict(self, newdata: Any = None, type: str = "response",
                prefix: str | None = None, seed: int = 0,
                id: str | list | None = None, effects: str = "mode") -> pl.LazyFrame:
        """Predictions as a lazy frame, as `MultinomialLogitResult.predict`:
        ``type="response"`` the probability of each level (``prob_<level>``),
        ``"link"`` the linear predictor x'b (with the fixed effect),
        ``"class"`` the most probable level, ``"draw"`` a level drawn from
        the row's probabilities (``drawn``; `seed` and `id` as there).
        `effects` as there: ``"average"`` averages the probabilities over
        the random intercept's distribution."""
        if type not in ("response", "link", "class", "draw"):
            raise ValueError(f"type must be 'response', 'link', 'class' or 'draw', got {type!r}")
        _check_effects(effects, type)
        frame = self._frame(newdata)
        if effects == "average":
            return self._predict(frame, type, prefix, average=True)
        if type != "draw":
            return self._predict(frame, type, prefix)
        from .mi import uniform_expr, with_ids
        frame, ids, drop = with_ids(frame, id)
        return self._predict(frame, type, prefix, u=uniform_expr(ids, seed)).drop(drop)

    def _predict(self, frame, type, prefix, u=None, average=False):
        xs = self._xexprs()
        k = len(xs)
        frame, fe = self._with_fe(frame)
        gnames, covs = [], []
        if average:
            from ._average import spread_inputs
            frame, gnames, covs = spread_inputs(self, frame)
        b = np.asarray(self.beta)[:k]
        cuts = self.cutpoints
        levels = self.levels
        if type == "response":
            pre = prefix or "prob"
            dtype = _prob_dtype(levels, pre)
        elif type == "link":
            pre = prefix or "xb"
            dtype = pl.Struct({pre: pl.Float64})
        else:
            pre = prefix or ("drawn" if type == "draw" else "predicted")
            lv_series = pl.Series(levels)
            dtype = pl.Struct({pre: lv_series.dtype})

        off = self._spec.get("offset")

        cdf_fn = LINKS[self.link].cdf

        def fn(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            X, A = self._matrix(df, k, len(fe))
            eta = X @ b
            if A is not None:
                eta = eta + A[:, 0]
            if off is not None:
                eta = eta + df["__ls_off"].to_numpy().astype(float)
            if type == "link":
                return pl.DataFrame({pre: eta}).fill_nan(None).to_struct(s.name)
            def probs(e):
                c = cdf_fn(cuts - e[..., None])
                c = np.concatenate([np.zeros(c.shape[:-1] + (1,)), c,
                                    np.ones(c.shape[:-1] + (1,))], axis=-1)
                return np.diff(c, axis=-1)

            if average:
                from ._average import average_scalar, row_covariance
                sd = np.sqrt(row_covariance(df, gnames, covs, X, 1)[:, 0, 0])
                with np.errstate(invalid="ignore"):
                    P = average_scalar(eta, sd, probs)
            else:
                P = probs(eta)
            P[np.isnan(eta)] = np.nan
            if type == "response":
                return pl.DataFrame({f"{pre}_{_level_name(lv)}": P[:, j]
                                     for j, lv in enumerate(levels)}).fill_nan(None) \
                         .to_struct(s.name)
            if type == "draw":
                idx = categorical(P, df["__ls_u"].to_numpy())
                ok = idx >= 0
                idx = np.maximum(idx, 0)
            else:
                ok = ~np.isnan(eta)
                idx = np.where(ok, np.argmax(np.where(ok[:, None], P, 0), axis=1), 0)
            col = pl.Series(pre, [levels[i] for i in idx], dtype=lv_series.dtype)
            col = pl.select(pl.when(pl.Series(ok)).then(col).otherwise(None)).to_series()
            return pl.DataFrame({pre: col}).to_struct(s.name)

        inputs = [ex.alias(f"__ls_p{j}") for j, (_, ex) in enumerate(xs)] + \
                 [pl.col(c) for c in fe] + ([off.alias("__ls_off")] if off is not None else []) \
            + ([u.alias("__ls_u")] if u is not None else []) + [pl.col(g) for g in gnames]
        pred = pl.struct(inputs) if inputs else pl.struct(pl.lit(0.0).alias("__ls_none"))
        if not xs and not fe:
            # no covariates: x'b = 0 for every row; the function still runs
            k = 0
        tmp = "__ls_pred"
        return (frame.with_columns(pred.map_batches(fn, return_dtype=dtype, is_elementwise=True)
                                   .alias(tmp))
                     .drop(*fe, *gnames).unnest(tmp))

    def resid(self) -> pl.LazyFrame:
        """The estimation sample, lazily: `row_id`, the outcome, ``xb``, each
        level's probability and response residual (1[y = level] - prob),
        and two residuals for the ordered outcome as a whole:
        ``resid_prob``, the probability-scale residual P(Y < y) - P(Y > y)
        (Li and Shepherd 2012; statsmodels' ``resid_prob``), in [-1, 1] and
        mean zero; and ``resid_gen``, the generalized residual d log P(y) /
        d xb (Stata's ``predict, scores``), positive when the outcome is
        higher than the fit expects. For the logit the two are equal (both
        are F(cut_{y-1} - xb) + F(cut_y - xb) - 1, since the logistic
        density is F (1 - F)); for the probit they differ."""
        spec = self._spec
        frame = self._in_sample(spec["source"])
        frame = frame.select("row_id", spec["yexpr"].alias(self.depvar),
                             *MultinomialLogitResult._input_columns(self, frame))
        out = self._predict(self._predict(frame, "link", "xb"), "response", "prob")
        link, cuts, J = LINKS[self.link], self.cutpoints, len(self.levels)
        full = np.r_[-np.inf, cuts, np.inf]

        def overall(st: pl.Series) -> pl.Series:
            df = st.struct.unnest()
            j = df["__ls_j"].to_numpy()
            eta = df["xb"].to_numpy()
            ok = ~np.isnan(eta) & (j >= 0)
            jj = np.where(ok, j, 0)
            up, lo = full[jj + 1] - eta, full[jj] - eta
            Fu, Fl = link.cdf(up), link.cdf(lo)
            fu, fl = link.pdf(up, Fu, link.sf(up)), link.pdf(lo, Fl, link.sf(lo))
            with np.errstate(divide="ignore", invalid="ignore"):
                gen = (fl - fu) / (Fu - Fl)
            prob = Fl - (1.0 - Fu)
            return pl.DataFrame({"resid_prob": np.where(ok, prob, np.nan),
                                 "resid_gen": np.where(ok, gen, np.nan)}).fill_nan(None) \
                     .to_struct(st.name)

        code = pl.col(self.depvar).replace_strict(self.levels, list(range(J)), default=-1,
                                                  return_dtype=pl.Int64)
        out = out.with_columns(pl.struct(code.alias("__ls_j"), pl.col("xb")).map_batches(
            overall, return_dtype=pl.Struct({"resid_prob": pl.Float64, "resid_gen": pl.Float64}),
            is_elementwise=True).alias("__ls_res")).unnest("__ls_res")
        return out.select("row_id", self.depvar, "xb",
                          *[f"prob_{_level_name(lv)}" for lv in self.levels],
                          *[((pl.col(self.depvar) == pl.lit(lv)).cast(pl.Float64)
                             - pl.col(f"prob_{_level_name(lv)}")).alias(f"resid_{_level_name(lv)}")
                            for lv in self.levels],
                          "resid_prob", "resid_gen")


# --------------------------------------------------------------------------
# conditional
# --------------------------------------------------------------------------

@dataclass(kw_only=True)
class OrderedProbitResult(OrderedLogitResult):
    """The result of `oprobit_stream`: as `OrderedLogitResult`, with F the
    standard normal distribution."""

    link = "probit"


@dataclass(kw_only=True)
class BinaryLogitResult(OrderedLogitResult):
    """The result of `logit_stream`: a binary logit, fitted as the two-level
    ordered logit (its cutpoint is minus the intercept).

    The coefficients are ``Intercept`` first (without an absorbed fixed
    effect, which carries the location) and then the formula's. The
    conventions are hdfe-stream's and pyfixest's: the iid variance has the
    factor (N - 1) / (N - K), the heteroskedasticity-robust one N / (N - K)
    and a cluster-robust one G / (G - 1) (N - 1) / (N - K), with K counting
    the fixed effect's parameters (unless it is nested in the cluster
    variable); the degrees of freedom of a t statistic are N - K, or G - 1.

    Attributes
    ----------
    family : str
        ``"logit"``.
    deviance : float
        -2 loglik.
    k_fe : int
        The fixed effect's parameters (its levels kept), 0 without one.
    (and those of `LogitResult`; `levels` is [0, 1])
    """

    family: str = "logit"
    deviance: float = np.nan
    k_fe: int = 0

    @staticmethod
    def _mu(eta: pl.Expr) -> pl.Expr:
        """The probability that the outcome is 1, given the linear predictor."""
        return 1.0 / (1.0 + (-eta).exp())

    @staticmethod
    def _mu_eta(eta: pl.Expr) -> pl.Expr:
        """Its derivative in the linear predictor."""
        mu = 1.0 / (1.0 + (-eta).exp())
        return mu * (1.0 - mu)

    def _model_lines(self):
        return [f"obs: {self.n_obs:,}   outcome {self.depvar}: binary"]

    def _model_fields(self):
        return {}

    @property
    def cutpoints(self) -> np.ndarray:
        """The one cutpoint, minus the intercept (0 with a fixed effect)."""
        return np.array([0.0 if self.fe_names else -float(self.beta[0])])

    def _linear(self, frame, extra=(), average=False):
        """`frame` with the linear predictor's parts as columns: ``xb``
        (covariates and intercept), ``eta`` (xb, the fixed effect and the
        offset), and the fixed effect ``fe_<name>``, null where it is not
        estimated; with `average`, also ``__ls_sd``, the sd of the row's
        random effect (its group's conditional one, or the prior's)."""
        xs = self._xexprs()
        k = len(xs)
        frame, fe = self._with_fe(frame)
        gnames, covs = [], []
        if average:
            from ._average import spread_inputs
            frame, gnames, covs = spread_inputs(self, frame)
        beta = np.asarray(self.beta, dtype=float)
        icpt = float(beta[0]) if self.coefnames and self.coefnames[0] == "Intercept" else 0.0
        b = beta[1:] if self.coefnames and self.coefnames[0] == "Intercept" else beta
        off = self._spec.get("offset")
        fname = "fe_" + self.fe_names[0] if self.fe_names else None
        fields = {"xb": pl.Float64, "eta": pl.Float64}
        if fname:
            fields[fname] = pl.Float64
        if average:
            fields["__ls_sd"] = pl.Float64

        def fn(s: pl.Series) -> pl.Series:
            df = s.struct.unnest()
            X, A = self._matrix(df, k, len(fe))
            xb = X @ b + icpt
            eta = xb if A is None else xb + A[:, 0]
            if off is not None:
                eta = eta + df["__ls_off"].to_numpy().astype(float)
            cols = {"xb": xb, "eta": eta}
            if fname:
                # the fixed effect's own column (A also holds any random effects)
                at = 0
                for _, values, kind in self._spec["effects"]:
                    if kind == "fixed":
                        cols[fname] = df[df.columns[k + at]].to_numpy().astype(np.float64)
                    at += len(values)
            if average:
                from ._average import row_covariance
                cols["__ls_sd"] = np.sqrt(row_covariance(df, gnames, covs, X, 1)[:, 0, 0])
            return pl.DataFrame(cols).fill_nan(None).to_struct(s.name)

        inputs = [ex.alias(f"__ls_p{j}") for j, (_, ex) in enumerate(xs)] + \
                 [pl.col(c) for c in fe] + ([off.alias("__ls_off")] if off is not None else []) \
            + [pl.col(g) for g in gnames]
        pred = pl.struct(inputs) if inputs else pl.struct(pl.lit(0.0).alias("__ls_none"))
        if not xs and not fe:
            k = 0
        tmp = "__ls_lin"
        return (frame.with_columns(pred.map_batches(fn, return_dtype=pl.Struct(fields),
                                                    is_elementwise=True).alias(tmp))
                     .drop(*fe, *gnames).unnest(tmp))

    def predict(self, newdata: Any = None, type: str = "link",
                name: str = "prediction", seed: int = 0,
                id: str | list | None = None, effects: str = "mode") -> pl.LazyFrame:
        """Predictions on new data, or on the source, as a lazy frame:
        `newdata` with one column added, ``name``: the linear predictor
        (``type="link"``, with the fixed effect and the offset), the
        probability that the outcome is 1 (``"response"``), the more
        probable outcome, 0 or 1 (``"class"``), or an outcome drawn from
        the probability (``"draw"``: 1 where a uniform falls below it, for
        imputation; the uniform depends only on `seed` and the row's
        identifier, the column(s) `id`, default the row's position, not on
        the batches; see `logit_stream.mi`). A row whose fixed-effect level
        was not estimated, or with a null covariate, gets null. `effects`:
        with random intercepts, ``"mode"`` (the default) puts the group's
        predicted effect in the probability, ``"average"`` averages the
        probability over the effect's distribution (as
        `MultinomialLogitResult.predict`), for "response" and "class"."""
        if type not in ("link", "response", "class", "draw"):
            raise ValueError(f"type must be 'link', 'response', 'class' or 'draw', got {type!r}")
        _check_effects(effects, type)
        frame = self._frame(newdata)
        drop = []
        if type == "draw":
            from .mi import uniform_expr, with_ids
            frame, ids, drop = with_ids(frame, id)
        average = effects == "average"
        out = self._linear(frame, average=average)
        eta = pl.col("eta")
        if average:
            from ._average import average_scalar
            cdf = LINKS["probit" if self.family == "probit" else "logit"].cdf

            def avg(s: pl.Series) -> pl.Series:
                d = s.struct.unnest()
                e = d["eta"].fill_null(np.nan).to_numpy()
                with np.errstate(invalid="ignore"):
                    p = average_scalar(e, d["__ls_sd"].fill_null(0.0).to_numpy(), cdf)
                return pl.Series(s.name, p).fill_nan(None)

            mu = pl.struct("eta", "__ls_sd").map_batches(avg, return_dtype=pl.Float64,
                                                          is_elementwise=True)
            drop = [*drop, "__ls_sd"]
        else:
            mu = self._mu(eta)
        if type == "link":
            val = eta
        elif type == "response":
            val = mu
        elif type == "draw":
            val = pl.when(eta.is_null()).then(None).otherwise(
                (uniform_expr(ids, seed) < self._mu(eta)).cast(pl.Int64))
        elif average:
            val = pl.when(eta.is_null()).then(None).otherwise((mu > 0.5).cast(pl.Int64))
        else:
            val = pl.when(eta.is_null()).then(None).otherwise((eta > 0).cast(pl.Int64))
        keep = [c for c in out.collect_schema().names() if c not in ("xb", "eta", *drop)
                and not c.startswith("fe_")]
        return out.with_columns(val.alias(name)).select(*keep, name)

    def resid(self) -> pl.LazyFrame:
        """The estimation sample, lazily: `row_id`, the outcome, ``xb``,
        ``eta``, ``fitted`` (the probability), ``resid`` (outcome - fitted),
        ``resid_working`` (resid / (fitted (1 - fitted)), the working
        residual of the iteratively reweighted fit) and, with a fixed
        effect, its estimate ``fe_<name>``. Computed from the source when
        collected (nothing is stored)."""
        spec = self._spec
        frame = self._in_sample(spec["source"])
        frame = frame.select("row_id", spec["yexpr"].alias(self.depvar),
                             *MultinomialLogitResult._input_columns(self, frame))
        out = self._linear(frame)
        eta, y = pl.col("eta"), pl.col(self.depvar).cast(pl.Float64)
        out = out.with_columns(self._mu(eta).alias("fitted"))
        out = out.with_columns((y - pl.col("fitted")).alias("resid"))
        out = out.with_columns((pl.col("resid") / self._mu_eta(eta)).alias("resid_working"))
        extra = ["fe_" + self.fe_names[0]] if self.fe_names else []
        return out.select("row_id", self.depvar, "xb", "eta", "fitted", "resid",
                          "resid_working", *extra)


@dataclass(kw_only=True)
class BinaryProbitResult(BinaryLogitResult):
    """The result of `probit_stream`: a binary probit, fitted as the two-level
    ordered probit. As `BinaryLogitResult`, with the standard normal in place
    of the logistic distribution: ``family`` is ``"probit"``, predicted
    probabilities are Phi(eta), and ``resid_working`` is resid / phi(eta).
    The variances use the expected information, as an iteratively
    reweighted fit's do (the observed one differs from it for the probit).
    """

    family: str = "probit"

    @staticmethod
    def _mu(eta: pl.Expr) -> pl.Expr:
        return eta.map_batches(lambda s: pl.Series(s.name, special.ndtr(s.to_numpy()))
                               .fill_nan(None), return_dtype=pl.Float64, is_elementwise=True)

    @staticmethod
    def _mu_eta(eta: pl.Expr) -> pl.Expr:
        return (-0.5 * eta * eta).exp() / math.sqrt(2.0 * math.pi)


@dataclass(kw_only=True)
class ConditionalLogitResult(LogitResult):
    """The result of `clogit_stream`: McFadden's conditional logit on
    long-format data, one row per alternative in each choice set.

    Attributes
    ----------
    case : list of str
        The columns that identify a choice set.
    n_cases : int
        Choice sets used. `n_obs` is the same number (the sum of the weights
        with frequency weights); ``diagnostics["n_rows"]`` counts the rows
        (alternatives).
    (and those of `LogitResult`)
    """

    case: list
    n_cases: int = 0

    def _model_lines(self):
        return [f"choice sets: {self.n_cases:,}   alternatives (rows): "
                f"{self.diagnostics.get('n_rows', 0):,}   case: {'+'.join(self.case)}"]

    def _model_fields(self):
        return {"case": list(self.case), "n_cases": self.n_cases,
                "n_rows": self.diagnostics.get("n_rows")}

    def predict(self, newdata: Any = None, type: str = "response",
                prefix: str | None = None, case: str | list | None = None,
                calibrate: dict | None = None, seed: int = 0,
                alt: str | list | None = None, effects: str = "mode") -> pl.LazyFrame:
        """Predictions as a lazy frame, one per row (alternative):
        ``type="response"`` the choice probability within the row's choice
        set (``prob``), ``"link"`` the utility x'b (``xb``), ``"draw"`` one
        alternative drawn per set from the probabilities (``drawn``: True at
        the drawn row; null for a set with no probabilities), for imputation.
        The draw's uniform depends only on `seed` and the set's `case`
        columns, not on the batches (see `logit_stream.mi`). The set's
        alternatives are ordered by the column(s) `alt` (default: the
        calibrated dimension's, with `calibrate`; else the fit's
        alternative column if it has one, or the rows' order), so that the
        draw does not depend on the rows' order either; with
        `calibrate`, the draws are independent across sets, so the totals
        hold only on average (`sample_assignments` keeps them). The choice sets
        are identified by `case` (default: the fit's). The probabilities
        are a window over each set, which Polars evaluates set by set.

        With absorbed alternative constants, their values join by level; a
        level the fit did not see (or dropped) gets a null, which leaves
        the row out of its set. `calibrate` instead solves one dimension's
        constants from known totals: ``{"estab": totals}``, `totals` a
        frame of the dimension's column(s) and one more, the level's total
        of choices among `newdata`'s sets. Each level's constant is then
        the one at which its predicted choices, summed over the sets, equal
        its total (the contraction delta += log(total / predicted)),
        given the coefficients (this result's: a draw's, for multiple
        imputation) and any other terms. Within each group of levels that
        the sets link (a firm's establishments), the totals are first
        scaled to the group's number of sets, with a warning if they
        differ by more than 1%. A level with a total of 0 has probability
        0; a level in `newdata` without a total is an error. The solve
        needs a group's rows at once, so `newdata` is streamed through
        disk a batch of whole groups at a time (about 4 million rows, or
        the largest group); the result is a lazy scan of the batches'
        files (in the run directory, removed when Python exits) in
        `newdata`'s order, or in memory when `newdata` is one batch."""
        if type not in ("response", "link", "draw"):
            raise ValueError(f"type must be 'response', 'link' or 'draw', got {type!r}")
        case = self.case if case is None else ([case] if isinstance(case, str) else list(case))
        if effects not in ("mode", "average"):
            raise ValueError(f"effects must be 'mode' or 'average', got {effects!r}")
        if effects == "average":
            if type != "response" or calibrate is not None:
                raise ValueError('effects="average" is for type="response", without calibrate=')
            return self._average_slopes(self._frame(newdata), prefix or "prob", case)
        if type == "draw":
            from .mi import draw_choice
            if alt is None and calibrate is not None and len(calibrate) == 1:
                alt = list(self._spec["fe_cols"][next(iter(calibrate))]) \
                    if next(iter(calibrate)) in self._spec.get("fe_cols", {}) else None
            if alt is None and getattr(self, "alt", None):
                alt = self.alt
            alt = [alt] if isinstance(alt, str) else alt
            lf = self.predict(newdata, "response", "__ls_prob", case, calibrate)
            return draw_choice(lf, "__ls_prob", case, seed, prefix or "drawn", alt) \
                .drop("__ls_prob")
        if calibrate is not None:
            from .calibrate import calibrated
            return calibrated(self, self._frame(newdata), type, prefix, case, calibrate)
        return self._predict(self._frame(newdata), type, prefix, case)

    def _average_slopes(self, frame, name, case):
        """Choice probabilities averaged over the random slopes'
        distribution (a seen region's conditional one, normal at its mode;
        an unseen region's N(0, Sigma)), by Gauss-Hermite quadrature: 15
        nodes for one slope, 7 x 7 for two. Needs the choice set's region
        (each set's alternatives share it)."""
        from numpy.polynomial.hermite_e import hermegauss
        spec = self._spec
        sl = spec.get("slopes")
        if not sl:
            raise ValueError('effects="average" needs random slopes (random=)')
        if spec.get("draw_index") is not None:
            raise ValueError('effects="average" is for the fitted result: a draw\'s slopes are '
                             "draws")
        (region, zs), = sl.items()
        info = self.random_effects[region]
        if info.get("region_of") != "choice set":
            raise ValueError('effects="average" needs each choice set\'s alternatives to share '
                             "a region (the slopes are the alternatives' here: use "
                             "draw_parameters for imputation)")
        q = len(zs)
        if q > 2:
            raise ValueError('effects="average" takes at most two random slopes')
        z, w = hermegauss(15 if q == 1 else 7)
        w = w / w.sum()
        Z = np.array(np.meshgrid(*([z] * q), indexing="ij")).reshape(q, -1).T
        W = np.prod(np.array(np.meshgrid(*([w] * q), indexing="ij")).reshape(q, -1), axis=0)
        # each region's mean and the Cholesky factor of its covariance; the prior's
        table = spec["ranef"][region]
        cols = spec["fe_cols"][region]
        C = spec["ranef_cov"][region]
        Ls = np.linalg.cholesky(C + 1e-300 * np.eye(q))
        S0 = np.linalg.cholesky(np.asarray(info["Sigma"], float))
        lnames = [(i, j) for i in range(q) for j in range(i + 1)]
        lt = table.select(*cols, *[pl.col(f"re_{region}:{zn}").alias(f"__ls_mu{k}")
                                   for k, zn in enumerate(zs)]).with_columns(
            *[pl.Series(f"__ls_L{i}{j}", Ls[:, i, j]) for i, j in lnames])
        bare = dict(spec)
        bare["effects"] = [e for e in spec.get("effects", []) if e[2] != "slope"]
        from dataclasses import replace as _replace
        base = _replace(self, _spec=bare)._predict(frame, "link", "__ls_u0", case)
        tsch = lt.collect_schema()
        base = base.with_columns(pl.col(c).cast(tsch[c], strict=False) for c in cols)
        lf = base.join(lt.lazy(), on=cols, how="left", maintain_order="left")
        seen = pl.col("__ls_mu0").is_not_null()
        mu = [pl.col(f"__ls_mu{k}").fill_null(0.0) for k in range(q)]
        L = {(i, j): pl.when(seen).then(pl.col(f"__ls_L{i}{j}")).otherwise(float(S0[i, j]))
             for i, j in lnames}
        xe = spec["xexprs"]
        terms = []
        for zeta, wk in zip(Z, W):
            u = pl.col("__ls_u0")
            for i, zn in enumerate(zs):
                bi = mu[i] + sum(L[(i, j)] * float(zeta[j]) for j in range(i + 1))
                u = u + xe[zn] * bi
            u = u - u.max().over(case)
            terms.append(float(wk) * u.exp() / u.exp().sum().over(case))
        prob = pl.when(pl.col("__ls_u0").is_null()).then(None).otherwise(pl.sum_horizontal(terms))
        drop = ["__ls_u0", *[f"__ls_mu{k}" for k in range(q)],
                *[f"__ls_L{i}{j}" for i, j in lnames]]
        return lf.with_columns(prob.alias(name)).drop(drop)

    def sample_assignments(self, newdata: Any, totals: dict, sampler=None, seed: int = 0,
                           case: str | list | None = None,
                           prefix: str = "assigned") -> pl.LazyFrame:
        """Draw each choice set's alternative so that one dimension's
        levels' counts match known totals, exactly or approximately
        (experimental: see `logit_stream.assign`).

        `totals` is ``{"estab": frame}`` as for ``predict(calibrate=)``.
        `sampler` is a `logit_stream.assign.ConstrainedSampler`: by default
        ``SwapMCMC()``, exact counts (the totals scaled to each group's
        number of sets and rounded at random); ``SwapMCMC(penalty=
        GaussianPenalty(...))`` for soft ones, ``ExactSmall`` for exact
        draws in small groups, ``Balanced`` for the cube method. The
        utilities are this result's (a draw's, for multiple imputation),
        without the dimension's constants; `newdata` is streamed a batch of
        whole groups at a time. Returns `newdata`'s rows, lazily and in
        order, with `prefix`: whether the row is its set's draw (null for a
        set with nothing that can be chosen). Each `seed` gives the same
        draw whatever the batches."""
        from .assign import _experimental, _run
        from .calibrate import _prepare
        _experimental()
        case = self.case if case is None else ([case] if isinstance(case, str) else list(case))
        lf, L, keep, _ = _prepare(self, self._frame(newdata), case, totals, "assign")
        return _run(lf, L, keep, case, sampler, seed, prefix, None, None)

    def _predict(self, frame, type, prefix, case):
        xs = self._xexprs()
        b = np.asarray(self.beta)
        xb = (pl.sum_horizontal([float(c) * ex for c, (_, ex) in zip(b, xs)]) if xs
              else pl.lit(0.0))
        # a null covariate makes the utility null, not a partial sum
        if xs:
            xb = pl.when(pl.any_horizontal([ex.is_null() | ~ex.is_finite() for _, ex in xs])) \
                   .then(None).otherwise(xb)
        if self._spec.get("offset") is not None:
            xb = xb + self._spec["offset"]
        tmp = []
        if self._spec.get("effects"):
            # the absorbed alternative constants: null for a level the fit did not
            # see, or dropped (never chosen: its probability is 0), which leaves the
            # row out of its set's normalization; random slopes: z'b of the row's
            # region, 0 for a region the fit did not see
            frame, tmp = self._with_fe(frame)
            fixed, slope = [], []
            at = 0
            for d, values, kind in self._spec["effects"]:
                names = tmp[at:at + len(values)]
                at += len(values)
                if kind == "slope":
                    zs = self._spec["slopes"][d]
                    xe = self._spec["xexprs"]
                    slope += [pl.col(t).fill_null(0.0) * xe[z] for t, z in zip(names, zs)]
                else:
                    fixed += names
            if slope:
                xb = xb + pl.sum_horizontal(slope)
            if fixed:
                xb = pl.when(pl.any_horizontal([pl.col(t).is_null() for t in fixed])) \
                    .then(None).otherwise(xb + pl.sum_horizontal([pl.col(t) for t in fixed]))
        if type == "link":
            return frame.with_columns(xb.alias(prefix or "xb")).drop(tmp)
        u = xb - xb.max().over(case)
        p = u.exp() / u.exp().sum().over(case)
        return frame.with_columns(p.alias(prefix or "prob")).drop(tmp)

    def resid(self) -> pl.LazyFrame:
        """The estimation sample, lazily, a row per alternative: `row_id`,
        the case columns, the outcome, ``xb``, ``prob`` and the response
        residual ``resid`` = y - prob."""
        spec = self._spec
        frame = self._in_sample(spec["source"])
        frame = self._predict(self._predict(frame, "link", "xb", self.case), "response", "prob",
                              self.case)
        return frame.select("row_id", *self.case, spec["yexpr"].cast(pl.Float64).alias(self.depvar),
                            "xb", "prob",
                            (spec["yexpr"].cast(pl.Float64) - pl.col("prob")).alias("resid"))


@dataclass(kw_only=True)
class MixedLogitResult(LogitResult):
    """The result of `mixlogit_stream` (prototype).

    The coefficients are the fixed coefficients and the means of the random
    ones (named as the covariates; for a lognormal coefficient, the mean of
    its log), then the standard deviations of the random ones, ``"sd.<name>"``
    (of the log, for a lognormal one). The sign of a standard deviation is
    not identified, so its absolute value is reported (the signed estimates are in
    ``diagnostics["sd_signed"]``; the standard errors are the same). `n_obs`
    is the number of choosers (panels), the unit of the likelihood; the
    robust variance is clustered by them, ``"CRV1:<panel>"`` (``"hetero"``
    without `panel`, each choice set its own chooser).

    Attributes
    ----------
    case, panel : list of str
        The columns identifying the choice sets and the choosers.
    n_cases, n_panels : int
        Their numbers.
    random : dict
        Each random coefficient's distribution.
    draws, halton, seed, drop
        The simulation: draws per chooser, Halton or pseudo-random, the
        pseudo-random seed, the Halton points discarded at the start.
    loglik_clogit : float
        The conditional logit's log-likelihood (the starting point).
    (and those of `LogitResult`)
    """

    case: list
    panel: list
    n_cases: int = 0
    n_panels: int = 0
    random: dict = field(default_factory=dict)
    draws: int = 0
    halton: bool = True
    seed: int = 0
    drop: int = 100
    loglik_clogit: float = math.nan

    def _model_lines(self):
        sim = (f"{self.draws} Halton draws" if self.halton
               else f"{self.draws} pseudo-random draws (seed {self.seed})")
        rand = ", ".join(f"{k} {v}" for k, v in self.random.items())
        return [f"choosers: {self.n_panels:,}   choice sets: {self.n_cases:,}   "
                f"alternatives (rows): {self.diagnostics.get('n_rows', 0):,}",
                f"random: {rand}   simulation: {sim} per chooser",
                f"log-likelihood of the conditional logit: {self.loglik_clogit:,.3f}"]

    def _model_fields(self):
        return {"case": list(self.case), "panel": list(self.panel), "n_cases": self.n_cases,
                "n_panels": self.n_panels, "random": dict(self.random), "draws": self.draws,
                "halton": self.halton, "seed": self.seed, "loglik_clogit": self.loglik_clogit}

    def predict(self, newdata: Any = None, type: str = "response"):
        raise NotImplementedError("predictions from the mixed logit are not in the prototype")

    def resid(self, newdata: Any = None):
        raise NotImplementedError("residuals of the mixed logit are not in the prototype")


@dataclass(kw_only=True)
class NestedLogitResult(ConditionalLogitResult):
    """The result of `nlogit_stream` (prototype).

    The coefficients are those of the utilities, then each nest's
    dissimilarity parameter, ``"lambda.<nest>"``, in the order of `nests=`
    (RU2: the model is the conditional logit when they are all 1; nests of
    one child have none). `n_obs` is the number of choice sets (their
    weight, with frequency weights).

    Attributes
    ----------
    case : list of str
        The columns identifying the choice sets.
    alt : str
        The column naming the alternatives.
    nests : dict
        Each nest's alternatives (all of them, for a nest of nests).
    n_cases : int
        The number of choice sets.
    loglik_clogit : float
        The conditional logit's log-likelihood (the starting point).
    (and those of `LogitResult`)
    """

    alt: str
    nests: dict = field(default_factory=dict)
    loglik_clogit: float = math.nan

    def _model_lines(self):
        tree = self._spec["tree"] if self._spec else None
        nests = ("; ".join(f"{k}: {', '.join(map(str, v))}"
                           for k, v in self.nests.items() if tree is None
                           or tree.kids[k][0] == "alts"))
        return [f"choice sets: {self.n_cases:,}   alternatives (rows): "
                f"{self.diagnostics.get('n_rows', 0):,}",
                f"nests ({self.alt}): {nests}",
                f"log-likelihood of the conditional logit: {self.loglik_clogit:,.3f}"]

    def _model_fields(self):
        return {"case": list(self.case), "alt": self.alt, "nests": dict(self.nests),
                "n_cases": self.n_cases, "loglik_clogit": self.loglik_clogit}

    def predict(self, newdata: Any = None, type: str = "response",
                prefix: str | None = None, case: str | list | None = None,
                seed: int = 0) -> pl.LazyFrame:
        """Predictions as a lazy frame, one per row (alternative):
        ``type="response"`` the choice probability within the row's choice
        set (``prob``), ``"link"`` the utility x'b (``xb``). The choice sets
        are identified by `case` (default: the fit's), the alternatives by
        the fit's `alt` column; an alternative in no nest of the fit gets a
        null, and so does every alternative of a set with a null utility.
        The probabilities need each whole set, so Polars evaluates the frame
        in memory, as for the conditional logit's window. ``type="draw"``
        draws one alternative per set, as the conditional logit's."""
        return super().predict(newdata, type, prefix, case, seed=seed)

    def _predict(self, frame, type, prefix, case):
        if type == "link":
            return super()._predict(frame, type, prefix, case)
        from .nested import _nl_probs
        tree = self._spec["tree"]
        K = len(self._spec["xcols"])
        lam = tree.lam(np.asarray(self.beta)[K:])
        code = {n: i for i, n in enumerate(tree.bottom)}
        alts = list(tree.of)
        nest = pl.col(self.alt).replace_strict(alts, [code[tree.of[a]] for a in alts],
                                               default=-1, return_dtype=pl.Int64)
        tmp = "__ls_xb"
        frame = super()._predict(frame, "link", tmp, case)
        gid = pl.struct(case).hash()

        def fn(st: pl.Series) -> pl.Series:
            df = st.struct.unnest()
            V = df["v"].cast(pl.Float64).fill_null(np.nan).to_numpy().copy()
            nc = df["n"].to_numpy().astype(np.int64)
            gids = df["g"].to_numpy()
            # an alternative in no nest: its set's probabilities are undefined
            V[nc < 0] = np.nan
            nc = np.where(nc < 0, 0, nc)
            out = _nl_probs(V, nc, np.unique(gids, return_inverse=True)[1].astype(np.int64),
                            lam, tree.parent, tree.order)
            return pl.Series(st.name, out).fill_nan(None)

        p = pl.struct(pl.col(tmp).alias("v"), nest.alias("n"), gid.alias("g")).map_batches(
            fn, return_dtype=pl.Float64)
        return frame.with_columns(p.alias(prefix or "prob")).drop(tmp)


class LogitMulti(RegressionMulti):
    """Results of a formula that expands to several models (several outcomes,
    ``sw()``, ``csw()``, ...): iterate over it, or take one with
    `fetch_model`; ``len()`` counts them."""


def _ic(ll, k, N):
    return -2 * ll + 2 * k, -2 * ll + k * math.log(max(N, 1))


@dataclass(kw_only=True)
class ConditionalFELogitResult(LogitResult):
    """The result of `cfelogit_stream`: Chamberlain's conditional
    fixed-effects logit. The unit effects are conditioned out, not
    estimated, so there are none to report or predict with: `predict` gives
    the linear predictor x'b only (the probabilities need the unit's
    effect), and `loglik` is the conditional log-likelihood (its null: every
    arrangement of a unit's 1s equally likely).

    Attributes
    ----------
    unit : list of str
        The columns that identify a unit (the formula's fixed effect).
    n_units : int
        Units whose outcome varies, the ones the likelihood uses; `n_obs` is
        the same number (the sum of their weights with frequency weights).
    n_units_dropped : int
        Units whose outcome is all 0 or all 1, which carry no information.
    n_rows : int
        The rows of the units used.
    (and those of `LogitResult`)
    """

    unit: list
    n_units: int = 0
    n_units_dropped: int = 0
    n_rows: int = 0

    def _model_lines(self):
        return [f"units: {self.n_units:,} with variation ({self.n_units_dropped:,} all 0 or "
                f"all 1, dropped)   rows: {self.n_rows:,}   unit: {'^'.join(self.unit)}"]

    def _model_fields(self):
        return {"unit": list(self.unit), "n_units": self.n_units,
                "n_units_dropped": self.n_units_dropped, "n_rows": self.n_rows}

    def predict(self, newdata: Any = None, type: str = "link",
                prefix: str | None = None) -> pl.LazyFrame:
        """The linear predictor x'b (``xb``), lazily; only ``type="link"``:
        the unit effects were conditioned out, so probabilities cannot be
        formed."""
        if type != "link":
            raise ValueError("the conditional fixed-effects logit does not estimate the unit "
                             "effects, so only type='link' (x'b) can be predicted")
        xs = self._xexprs()
        b = np.asarray(self.beta)
        xb = pl.sum_horizontal([float(c) * ex for c, (_, ex) in zip(b, xs)])
        if self._spec.get("offset") is not None:
            xb = xb + self._spec["offset"]
        return self._frame(newdata).with_columns(xb.alias(prefix or "xb"))

    def resid(self):
        raise ValueError("the conditional fixed-effects logit has no fitted probabilities "
                         "(the unit effects are conditioned out), so no residuals")

    def fixef(self):
        raise ValueError("the conditional fixed-effects logit does not estimate the unit "
                         "effects")
