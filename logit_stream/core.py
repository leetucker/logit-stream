"""Building blocks for estimators built on this package's streamed passes
(penalized-stream's penalized logit is one): the binary logit's stored
design and its Newton pass, for an estimator that runs its own iteration.

Stable names
    The names below are the supported API for packages built on this one.
    They change only in a minor release (0.x -> 0.y), with the change in the
    changelog; tests/test_core_api.py pins them. Everything else in the
    package below the estimators and result classes is internal.

    logit_design      formula + data -> LogitDesign (pass 0)
    LogitDesign       the stored rows of a binary logit, with no absorbed
                      fixed effect or one; `newton_pass`, `rows`, `cleanup`
    NewtonMoments     what one pass returns

The model is the binary logit with linear predictor

    eta_i = x_i'b + a_g(i)

where a_g is the intercept of the row's group: one group (the intercept)
without a fixed effect, or one per level of the absorbed fixed effect. The
passes return the log-likelihood, its gradient and the information (minus
its Hessian) with respect to b and a; the information has an arrow shape (a
diagonal block for a, which each row touches one entry of), and
`NewtonMoments.profiled` eliminates a from it exactly (the Schur
complement), so that a Newton step, penalized or not, is a k x k problem in
b whatever the number of groups; `NewtonMoments.effects_step` then gives
a's part of the step.

One pass can evaluate several models (coefficient vectors) at once, each on
its own set of the rows: with `split_by=` the rows carry an integer split
(a cross-validation fold, say), and each model in a pass names the splits
its sums cover. The rows are read once; the arithmetic is per model. The
log-likelihood is returned for every model on every split, so a model's
held-out fit comes from the same pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from hdfe_stream.core.report import warn
from hdfe_stream.core.resources import MemoryBudget
from hdfe_stream.core.store import PolarsEngine
from hdfe_stream.core.utils import as_lazy
from hdfe_stream.core.workspace import Run
from scipy.special import expit

from ._frontend import _plan
from .estimator import _base_dir, _log
from .fe import _segments
from .newton import _chunks
from .store import CODE, SORT_FE, _build

__all__ = ["logit_design", "LogitDesign", "NewtonMoments"]

_SPLIT = "__ls_split"


@dataclass
class NewtonMoments:
    """One pass's sums for M models (the first axis of every array), with
    k covariates (`columns` of them in the information) and G groups.

    Attributes
    ----------
    loglik : (M, S) array
        Each model's log-likelihood on each split's rows (all splits, those
        its sums cover and those they do not).
    gradient : (M, k) array
        d loglik / d b over the model's splits, every covariate.
    effects_gradient : (M, G) array
        d loglik / d a.
    information : (M, c, c) array or None
        -d2 loglik / d b d b' for the `columns` (c of them).
    effects_information : (M, G) array or None
        -d2 loglik / d a_g^2 (the diagonal block).
    cross : (M, G, c) array or None
        -d2 loglik / d a_g d b.
    skipped : (M, S) array
        Rows left out of each model because their group's effect is not
        finite (see `LogitDesign.newton_pass`).
    columns : array of int
    """

    loglik: np.ndarray
    gradient: np.ndarray
    effects_gradient: np.ndarray
    information: np.ndarray | None
    effects_information: np.ndarray | None
    cross: np.ndarray | None
    skipped: np.ndarray
    columns: np.ndarray

    def _dinv(self, m):
        D = self.effects_information[m]
        with np.errstate(divide="ignore"):
            return np.where(D > 0, 1.0 / D, 0.0)

    def profiled(self, m: int = 0):
        """(information, gradient) of model m for b (its `columns`) with the
        effects profiled out: I_bb - I_ba I_aa^-1 I_ab and g_b - I_ba
        I_aa^-1 g_a. A group with no information (no rows in the model) is
        left out."""
        if self.information is None:
            raise ValueError("the pass was made without the information")
        Di = self._dinv(m)
        A = self.cross[m]
        AD = A * Di[:, None]
        C = self.information[m] - A.T @ AD
        g = self.gradient[m, self.columns] - AD.T @ self.effects_gradient[m]
        return (C + C.T) / 2, g

    def effects_step(self, dbeta, m: int = 0):
        """The effects' part of the Newton step, given b's part (over
        `columns`): I_aa^-1 (g_a - I_ab db); zero for a group with no
        information."""
        return self._dinv(m) * (self.effects_gradient[m] - self.cross[m] @ np.asarray(dbeta))


@dataclass
class LogitDesign:
    """The rows of a binary logit, stored by pass 0 (`logit_design`).

    Attributes
    ----------
    formula : str
    depvar : str
    names : list of str
        The covariates (the columns of b; no intercept, which is the one
        group's effect without a fixed effect).
    fe : str or None
        The absorbed fixed effect.
    n_groups : int
        G: the groups of `fe` kept (1 without a fixed effect).
    fe_levels : polars.DataFrame or None
        The kept groups' values of the fixed effect's columns, in the order
        of the effects (row g is group g).
    n_splits : int
        S (1 without `split_by`).
    n_obs : int
        Rows used (after the fixed effect's separation).
    weight_sum : float
        Their summed weight (analytic weights are scaled to a mean of 1 over
        the rows read, as the package's estimators do).
    outcome_counts : (S, G, 2) array
        The weighted counts of y = 0 and y = 1 in each split and group.
    x_sums, x_squares : (S, k) arrays
        Sum w x and sum w x^2 of each covariate in each split (for
        standardizing).
    separation : dict
        The groups dropped because their outcome never varies, and their rows.
    diagnostics : dict
    """

    formula: str
    depvar: str
    names: list
    fe: str | None
    n_groups: int
    fe_levels: Any
    n_splits: int
    n_obs: int
    weight_sum: float
    outcome_counts: np.ndarray
    x_sums: np.ndarray
    x_squares: np.ndarray
    separation: dict
    diagnostics: dict = field(default_factory=dict)
    _store: Any = None
    _run: Any = None
    _keep: Any = None
    _split_of: Any = None
    _remap: Any = None
    _chunk_bytes: float = 64e6

    @property
    def passes(self) -> int:
        return int(self.diagnostics.get("passes", 0))

    def _blocks(self):
        k = len(self.names)
        rows = max(256, int(self._chunk_bytes // (8 * max(k + 8, 1))))
        for blk in self._store.iterate(["X", "y", "w", "fe", "cl"]):
            for ch in _chunks(blk, rows):
                n = len(ch["y"])
                f = ch.get("fe")
                f = np.zeros(n, np.int64) if f is None else f
                keep = self._keep[f]
                if not keep.all():
                    ch = {key: (v[keep] if v is not None else None) for key, v in ch.items()}
                    f = f[keep]
                    n = len(f)
                if not n:
                    continue
                if self.fe is not None:
                    f = self._remap[f]
                s = (self._split_of[ch["cl"][:, 0]] if self._split_of is not None
                     else np.zeros(n, np.int64))
                w = ch.get("w")
                yield ch["X"], ch["y"].astype(np.float64), w, f, s

    def rows(self):
        """The stored rows, in chunks: ``(X, y, w, group, split)`` arrays,
        for sums a pass does not give (X the covariates, y 0/1 as floats, w
        the weights or None, group each row's effect index 0..G-1, split its
        split 0..S-1). Rows of dropped groups are left out. Counts as a pass."""
        self.diagnostics["passes"] = self.passes + 1
        yield from self._blocks()

    def newton_pass(self, beta, effects, *, models=None, columns=None,
                    information: bool = True) -> NewtonMoments:
        """One read of the rows: each model's log-likelihood on every split,
        and its gradient and information over its splits.

        Parameters
        ----------
        beta : (M, k) or (k,) array
            Each model's coefficients.
        effects : (M, G) or (G,) array
            Each model's group effects (without a fixed effect, G = 1: the
            intercept). A row whose effect is not finite (+-inf, nan) is left
            out of that model -- its log-likelihood, gradient and information
            -- and counted in `skipped`: the way to hold a group out of a
            model (one whose outcome does not vary within the model's rows,
            say, whose effect would be infinite).
        models : (M, S) boolean array, optional
            The splits each model's gradient and information cover; default
            all.
        columns : sequence of int, optional
            The covariates the information is formed for (an active set);
            default all. The gradient is always for every covariate.
        information : bool
            Whether to form the information (False: the log-likelihood and
            gradients only, which cost about a tenth as much).
        """
        beta = np.atleast_2d(np.asarray(beta, dtype=np.float64))
        effects = np.atleast_2d(np.asarray(effects, dtype=np.float64))
        M, k = beta.shape
        G, S = self.n_groups, self.n_splits
        if k != len(self.names):
            raise ValueError(f"beta has {k} columns; the design has {len(self.names)}")
        if effects.shape != (M, G):
            raise ValueError(f"effects must be ({M}, {G})")
        models = (np.ones((M, S), bool) if models is None
                  else np.asarray(models, dtype=bool).reshape(M, S))
        cols = np.arange(k) if columns is None else np.asarray(columns, dtype=np.int64)
        c = len(cols)
        ll = np.zeros((M, S))
        skipped = np.zeros((M, S), np.int64)
        gb = np.zeros((M, k))
        ga = np.zeros((M, G))
        H = np.zeros((M, c, c)) if information else None
        D = np.zeros((M, G)) if information else None
        A = np.zeros((M, G, c)) if information else None
        full = c == k
        for X, y, w, f, s in self._blocks():
            n = len(y)
            wt = np.ones(n) if w is None else w
            eta = X @ beta.T + effects[:, f].T                     # (n, M)
            fin = np.isfinite(eta)
            eta = np.where(fin, eta, 0.0)
            lterm = np.where(fin, wt[:, None] * (y[:, None] * eta - np.logaddexp(0.0, eta)),
                             0.0)
            inc = models[:, s].T & fin
            p = expit(eta)
            r = np.where(inc, wt[:, None] * (y[:, None] - p), 0.0)
            gb += (X.T @ r).T
            for m in range(M):
                ll[m] += np.bincount(s, lterm[:, m], S)
                skipped[m] += np.bincount(s[~fin[:, m]], minlength=S)
                ga[m] += np.bincount(f, r[:, m], G)
            if not information:
                continue
            h = np.where(inc, wt[:, None] * p * (1.0 - p), 0.0)
            Xc = X if full else X[:, cols]
            seg = _segments(f) if G > 1 else None
            for m in range(M):
                Xh = Xc * h[:, m:m + 1]
                H[m] += Xc.T @ Xh
                D[m] += np.bincount(f, h[:, m], G)
                if seg is None:
                    A[m, 0] += Xh.sum(axis=0)
                else:
                    order, starts, groups = seg
                    A[m, groups] += np.add.reduceat(Xh[order], starts[:-1], axis=0)
        if H is not None:
            H = (H + H.transpose(0, 2, 1)) / 2
        self.diagnostics["passes"] = self.passes + 1
        return NewtonMoments(loglik=ll, gradient=gb, effects_gradient=ga, information=H,
                             effects_information=D, cross=A, skipped=skipped, columns=cols)

    def cleanup(self) -> None:
        """Delete the stored rows."""
        if self._run is not None:
            self._run.cleanup()
            self._run = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()


def logit_design(fml: str, data: Any, *, weights: str | None = None,
                 weights_type: str = "aweights", split_by: str | None = None,
                 n_splits: int | None = None, workdir=None, storage: str = "disk",
                 dtype: str = "float64", block_mb: float = 64.0, chunk_mb: float = 64.0,
                 polars_engine: str = "cpu", max_memory_gb: float | None = None,
                 memory_reserve: float | None = None, verbose: bool = False,
                 logger=None) -> LogitDesign:
    """Pass 0 of a binary logit: the rows stored for `LogitDesign.newton_pass`.

    Parameters
    ----------
    fml : str
        One model, ``"y ~ x1 + x2"`` or ``"y ~ x1 + x2 | fe"`` (at most one
        absorbed fixed effect; pyfixest's syntax for the covariates). The
        outcome is 0 or 1. The intercept is the effects' (no ``Intercept``
        column).
    data : str, Path, polars.LazyFrame or polars.DataFrame
    weights, weights_type
        As for `logit_stream`.
    split_by : str, optional
        An integer column with values 0, ..., n_splits - 1: each row's split.
    n_splits : int, optional
        With `split_by`.
    workdir, storage, dtype, block_mb, chunk_mb, polars_engine, max_memory_gb,
    memory_reserve, verbose, logger
        As for `logit_stream`.

    Returns
    -------
    LogitDesign
        Call `cleanup()` (or use it as a context manager) to delete the rows.

    Groups of the fixed effect whose outcome never varies are dropped (their
    effects would be infinite), with a warning, as `logit_stream` does.
    """
    if weights_type not in ("aweights", "fweights"):
        raise ValueError("weights_type must be 'aweights' or 'fweights'")
    if storage not in ("disk", "memory"):
        raise ValueError("storage must be 'disk' or 'memory'")
    if (split_by is None) != (n_splits is None):
        raise ValueError("give split_by and n_splits together")
    if n_splits is not None and (isinstance(n_splits, bool) or not isinstance(n_splits, int)
                                 or n_splits < 1):
        raise ValueError("n_splits must be a positive integer")

    def log(msg):
        _log(verbose, msg, logger)

    lf = as_lazy(data)
    engine = PolarsEngine(polars_engine)
    specs = _plan(fml, lf, intercept=False, collect=lambda q, what: engine.collect(q, what))
    if len(specs) != 1:
        raise ValueError(f"{fml}: one model (no multiple-estimation syntax)")
    spec = specs[0]
    if len(spec.fe) > 1:
        raise ValueError(f"{fml}: at most one absorbed fixed effect (got {list(spec.fe)})")
    if split_by is not None and split_by not in lf.collect_schema():
        raise ValueError(f"split_by: column {split_by!r} is not in the data")
    budget = MemoryBudget(max_memory_gb, memory_reserve)
    run = Run(_base_dir(workdir), auto_cleanup=True)
    try:
        p0 = _build(lf, spec, "ologit", run.path, engine, weights=weights,
                    clusters={_SPLIT: [split_by]} if split_by else {}, case=None, levels=None,
                    in_memory=storage == "memory", sort_fe=SORT_FE and len(spec.fe) == 1,
                    dtype=np.float32 if dtype == "float32" else np.float64,
                    block_mb=block_mb, log=log, weights_type=weights_type, budget=budget)
        try:
            ok = [float(v) for v in p0.levels] == [0.0, 1.0]
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"{fml}: the outcome must be 0 or 1; its values are "
                             f"{list(p0.levels)[:6]}")
        split_of = None
        S = 1
        if split_by:
            t = p0.cluster_tables[_SPLIT]
            vals = t[split_by].to_numpy()
            codes = t[CODE].to_numpy()
            if (not np.issubdtype(vals.dtype, np.integer) or vals.min() < 0
                    or vals.max() >= n_splits):
                raise ValueError(f"split_by: {split_by!r} must hold integers 0 to "
                                 f"{n_splits - 1}")
            split_of = np.zeros(int(codes.max()) + 1, np.int64)
            split_of[codes] = vals
            S = n_splits
        fe = spec.fe[0] if spec.fe else None
        G0 = p0.fe_tables[fe].height if fe else 1
        K = len(spec.x)
        counts = np.zeros((S, G0, 2))
        xs, xq = np.zeros((S, K)), np.zeros((S, K))
        for blk in p0.store.iterate(["X", "y", "w", "fe", "cl"]):
            n = len(blk["y"])
            f = blk["fe"] if fe else np.zeros(n, np.int64)
            s = split_of[blk["cl"][:, 0]] if split_of is not None else np.zeros(n, np.int64)
            w = blk.get("w")
            wt = np.ones(n) if w is None else w
            np.add.at(counts, (s, f, blk["y"].astype(np.int64)), wt)
            X = blk["X"]
            for j in range(S):
                rows = s == j
                if rows.any():
                    xs[j] += wt[rows] @ X[rows]
                    xq[j] += wt[rows] @ (X[rows] ** 2)
        tot = counts.sum(axis=0)
        keep = (tot > 0).all(axis=1) if fe else np.ones(1, bool)
        separation = {"checked": bool(fe), "levels": 0, "observations": 0}
        if fe and not keep.all():
            drop = ~keep
            n_drop = 0
            for blk in p0.store.iterate(["fe"]):
                n_drop += int(drop[blk["fe"]].sum())
            separation.update(levels=int(drop.sum()), observations=n_drop)
            warn(f"{n_drop:,} observations in {int(drop.sum()):,} levels of {fe} dropped "
                 "because their outcome never varies (their effects would be infinite)",
                 logger)
            # the dropped rows' x sums
            for blk in p0.store.iterate(["X", "w", "fe", "cl"]):
                gone = drop[blk["fe"]]
                if not gone.any():
                    continue
                n = len(gone)
                s = (split_of[blk["cl"][:, 0]] if split_of is not None
                     else np.zeros(n, np.int64))
                w = blk.get("w")
                wt = (np.ones(n) if w is None else w)[gone]
                X = blk["X"][gone]
                for j in range(S):
                    rows = s[gone] == j
                    if rows.any():
                        xs[j] -= wt[rows] @ X[rows]
                        xq[j] -= wt[rows] @ (X[rows] ** 2)
            if not keep.any():
                raise ValueError(f"{fml}: no observations left after dropping the levels of "
                                 f"{fe} whose outcome never varies")
        if not fe and (tot[0] <= 0).any():
            raise ValueError(f"{fml}: the outcome takes only one value")
        counts = counts[:, keep]
        G = int(keep.sum())
        remap = np.cumsum(keep) - 1
        levels = None
        if fe:
            t = p0.fe_tables[fe].sort(CODE)
            levels = t.filter(pl.Series(keep)).drop(CODE)
        return LogitDesign(
            formula=fml, depvar=spec.depvar,
            names=[nm for nm, _ in spec.x], fe=fe, n_groups=G, fe_levels=levels, n_splits=S,
            n_obs=int(p0.n_obs - separation["observations"]),
            weight_sum=float(counts.sum()), outcome_counts=counts, x_sums=xs, x_squares=xq,
            separation=separation,
            diagnostics={"passes": 0, "pass0_seconds": p0.seconds, "n_raw": p0.n_raw,
                         "fe_sorted": p0.fe_sorted},
            _store=p0.store, _run=run, _keep=keep, _split_of=split_of,
            _remap=remap, _chunk_bytes=float(chunk_mb) * 1e6)
    except BaseException:
        run.cleanup()
        raise
