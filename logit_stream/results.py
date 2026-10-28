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
        with its effects (see the subclass) and its number of observations.
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
        return tables[name].lazy()

    def ranef(self, name: str | None = None) -> pl.LazyFrame:
        """A random grouping's predicted intercepts (their modes given the
        data, the conditional means' Laplace approximation): one row per
        group, with one column per outcome level (the base level's 0) for
        the multinomial logit, one for the ordered logit, and the group's
        number of observations. `predict` adds every grouping's to the
        linear predictor, using 0 for a group the fit did not see. `name`:
        the grouping, needed when there are several."""
        if not self.random_effects:
            raise KeyError("this model has no random intercepts")
        tables = self._spec["ranef"]
        if name is None:
            if len(tables) > 1:
                raise ValueError(f"name one of {list(tables)}")
            name = next(iter(tables))
        if name not in tables:
            raise KeyError(f"{name!r} is not a random grouping of this model; it has "
                           + ", ".join(map(repr, tables)))
        return tables[name].lazy()

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
                   "joint-mode": "Laplace approximation at the joint mode"}.get(
                       first.get("method"), "Laplace approximation")
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
        return pl.DataFrame({"Coefficient": xs,
                             **{_level_name(lv): B[:, j] for j, lv in enumerate(self.levels[1:])}})

    def _B(self):
        J1, K = len(self.levels) - 1, len(self._spec["xcols"])
        return np.asarray(self.beta).reshape(J1, K).T

    def predict(self, newdata: Any = None, type: str = "response",
                prefix: str | None = None) -> pl.LazyFrame:
        """Predictions as a lazy frame: `newdata` (default: the fit's source,
        every row) with columns added.

        type="response"
            The probability of each level, ``prob_<level>``.
        type="link"
            The linear predictor of each non-base level, ``xb_<level>``
            (with its fixed effect).
        type="class"
            The most probable level, ``predicted``.

        A row with a null or non-finite covariate, or a fixed-effect level
        the fit did not estimate, gets nulls. `prefix` renames the columns'
        prefix (or, for "class", the column).
        """
        if type not in ("response", "link", "class"):
            raise ValueError(f"type must be 'response', 'link' or 'class', got {type!r}")
        frame = self._frame(newdata)
        return self._predict(frame, type, prefix)

    def _predict(self, frame, type, prefix):
        xs = self._xexprs()
        k = len(xs)
        frame, fe = self._with_fe(frame)
        B = self._B()
        levels = self.levels
        if type == "response":
            pre = prefix or "prob"
            dtype = _prob_dtype(levels, pre)
        elif type == "link":
            pre = prefix or "xb"
            dtype = _prob_dtype(levels[1:], pre)
        else:
            pre = prefix or "predicted"
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
            with np.errstate(invalid="ignore"):
                m = np.max(full, axis=1, keepdims=True)
                e = np.exp(full - m)
                P = e / e.sum(axis=1, keepdims=True)
            if type == "response":
                out = {f"{pre}_{_level_name(lv)}": P[:, j] for j, lv in enumerate(levels)}
                return pl.DataFrame(out).fill_nan(None).to_struct(s.name)
            ok = ~np.isnan(P).any(axis=1)
            idx = np.where(ok, np.nanargmax(np.where(ok[:, None], P, 0), axis=1), 0)
            col = pl.Series(pre, [levels[i] for i in idx], dtype=lv_series.dtype)
            col = pl.select(pl.when(pl.Series(ok)).then(col).otherwise(None)).to_series()
            return pl.DataFrame({pre: col}).to_struct(s.name)

        inputs = [ex.alias(f"__ls_p{j}") for j, (_, ex) in enumerate(xs)] + \
                 [pl.col(c) for c in fe]
        pred = (pl.struct(inputs) if inputs else pl.struct(pl.lit(0.0).alias("__ls_none")))
        tmp = "__ls_pred"
        return (frame.with_columns(pred.map_batches(fn, return_dtype=dtype, is_elementwise=True)
                                   .alias(tmp))
                     .drop(fe).unnest(tmp))

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
        return [pl.col(c) for c in sorted(need & names) if c not in ("row_id", self.depvar)]


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
                prefix: str | None = None) -> pl.LazyFrame:
        """Predictions as a lazy frame, as `MultinomialLogitResult.predict`:
        ``type="response"`` the probability of each level (``prob_<level>``),
        ``"link"`` the linear predictor x'b (with the fixed effect),
        ``"class"`` the most probable level."""
        if type not in ("response", "link", "class"):
            raise ValueError(f"type must be 'response', 'link' or 'class', got {type!r}")
        return self._predict(self._frame(newdata), type, prefix)

    def _predict(self, frame, type, prefix):
        xs = self._xexprs()
        k = len(xs)
        frame, fe = self._with_fe(frame)
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
            pre = prefix or "predicted"
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
            cdf = cdf_fn(cuts[None, :] - eta[:, None])
            cdf = np.column_stack([np.zeros(len(eta)), cdf, np.ones(len(eta))])
            P = np.diff(cdf, axis=1)
            P[np.isnan(eta)] = np.nan
            if type == "response":
                return pl.DataFrame({f"{pre}_{_level_name(lv)}": P[:, j]
                                     for j, lv in enumerate(levels)}).fill_nan(None) \
                         .to_struct(s.name)
            ok = ~np.isnan(eta)
            idx = np.where(ok, np.argmax(np.where(ok[:, None], P, 0), axis=1), 0)
            col = pl.Series(pre, [levels[i] for i in idx], dtype=lv_series.dtype)
            col = pl.select(pl.when(pl.Series(ok)).then(col).otherwise(None)).to_series()
            return pl.DataFrame({pre: col}).to_struct(s.name)

        inputs = [ex.alias(f"__ls_p{j}") for j, (_, ex) in enumerate(xs)] + \
                 [pl.col(c) for c in fe] + ([off.alias("__ls_off")] if off is not None else [])
        pred = pl.struct(inputs) if inputs else pl.struct(pl.lit(0.0).alias("__ls_none"))
        if not xs and not fe:
            # no covariates: x'b = 0 for every row; the function still runs
            k = 0
        tmp = "__ls_pred"
        return (frame.with_columns(pred.map_batches(fn, return_dtype=dtype, is_elementwise=True)
                                   .alias(tmp))
                     .drop(fe).unnest(tmp))

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
                prefix: str | None = None, case: str | list | None = None) -> pl.LazyFrame:
        """Predictions as a lazy frame, one per row (alternative):
        ``type="response"`` the choice probability within the row's choice
        set (``prob``), ``"link"`` the utility x'b (``xb``). The choice sets
        are identified by `case` (default: the fit's). The probabilities
        are a window over each set, which Polars evaluates set by set."""
        if type not in ("response", "link"):
            raise ValueError(f"type must be 'response' or 'link', got {type!r}")
        case = self.case if case is None else ([case] if isinstance(case, str) else list(case))
        return self._predict(self._frame(newdata), type, prefix, case)

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
        if type == "link":
            return frame.with_columns(xb.alias(prefix or "xb"))
        u = xb - xb.max().over(case)
        p = u.exp() / u.exp().sum().over(case)
        return frame.with_columns(p.alias(prefix or "prob"))

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
                prefix: str | None = None, case: str | list | None = None) -> pl.LazyFrame:
        """Predictions as a lazy frame, one per row (alternative):
        ``type="response"`` the choice probability within the row's choice
        set (``prob``), ``"link"`` the utility x'b (``xb``). The choice sets
        are identified by `case` (default: the fit's), the alternatives by
        the fit's `alt` column; an alternative in no nest of the fit gets a
        null, and so does every alternative of a set with a null utility.
        The probabilities need each whole set, so Polars evaluates the frame
        in memory, as for the conditional logit's window."""
        return super().predict(newdata, type, prefix, case)

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
