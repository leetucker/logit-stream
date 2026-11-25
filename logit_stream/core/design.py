"""Building blocks for estimators that run their own Newton iteration on a
multinomial logit, or an ordered logit or probit, with no absorbed fixed
effect or one: the stored design (pass 0) and the streamed pass that gives
each model's log-likelihood, gradient and information.

Stable names
    The names below are the supported API for packages built on this one
    (penalized-stream's penalized multinomial and ordered logit). They change
    only in a minor release (0.x -> 0.y), with the change in the changelog;
    tests/test_core_design.py pins them. The rest of `logit_stream.core` is
    the package's engine, internal::

        logit_design      formula + data -> LogitDesign (pass 0)
        LogitDesign       the stored rows; `newton_pass`, `start`, `feasible`,
                          `rows`, `cleanup`
        NewtonMoments     what one pass returns; `profiled`, `effects_step`

The passes are the ones `mlogit_stream`, `ologit_stream` and `oprobit_stream`
fit by, with the same parameters (`LogitDesign.names`):

multinomial
    equation-major: the k covariates of the first non-base outcome level,
    then of the second, ...; named ``"<level>:<covariate>"``. Without a fixed
    effect the covariates include the ``Intercept``.
ordered
    the k covariates, then the J - 1 cutpoints, with P(y <= j) =
    F(cut_j - x'b). With a fixed effect the first cutpoint is 0 (the effects
    carry the location), and the parameters are the k covariates and cut2,
    ..., cut{J-1}.

With a fixed effect, each of its G groups has effects of its own: one per
outcome level for the multinomial logit (``(G, J)``, added to each level's
linear predictor) and one location for the ordered model (``(G, 1)``, added
to x'b). The information then has an arrow shape (a small block per group,
which each row touches one of), and `NewtonMoments.profiled` eliminates the
effects from it exactly (the Schur complement), so that a Newton step,
penalized or not, is a problem in the parameters alone whatever the number
of groups; `NewtonMoments.effects_step` then gives the effects' part of the
step.

One pass can evaluate several models (parameter vectors) at once, each on
its own set of the rows: with `split_by=` the rows carry an integer split
(a cross-validation fold, say), and each model in a pass names the splits
its sums cover. The rows are read once; the arithmetic is per model. The
log-likelihood is returned for every model on every split, so a model's
held-out fit comes from the same pass.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np
from hdfe_stream.core.resources import MemoryBudget
from hdfe_stream.core.store import PolarsEngine
from hdfe_stream.core.utils import as_lazy
from hdfe_stream.core.utils import single_blas as _single_blas
from hdfe_stream.core.workspace import Run

from .fe import _AbsorbedFE, _subset
from .models import MNL_KERNEL_MIN_ROWS, _Multinomial, _Ordered
from .newton import _chunks, _design
from .store import CODE, PREFIX, SORT_FE, _build, _rescale_weights

__all__ = ["logit_design", "LogitDesign", "NewtonMoments"]

SPLIT = f"{PREFIX}split"            # the split's clustering
# The rows from which a block's sums are the fused kernels' (where the model
# has them). The estimators wait for MNL_KERNEL_MIN_ROWS, as a pass over a
# few rows does not repay the kernels' threads there; a design's pass cuts
# each block by split and evaluates several models on each piece, where the
# kernels win from far fewer rows.
KERNEL_MIN_ROWS = 256
MODELS = {"multinomial": "mlogit", "ordered": "ologit"}


# --------------------------------------------------------------------------
# results of a pass
# --------------------------------------------------------------------------

@dataclass
class NewtonMoments:
    """One pass's sums for M models (the first axis of every array), with P
    parameters (`parameters`, c of them, in the information), and, with a
    fixed effect, G groups of m effects each (m = J for the multinomial
    logit, 1 for the ordered model).

    Attributes
    ----------
    loglik : (M, S) array
        Each model's log-likelihood on each split's rows (all splits, those
        its sums cover and those they do not); -inf where a model gives some
        row probability zero.
    gradient : (M, P) array
        d loglik / d theta over the model's splits, every parameter.
    information : (M, c, c) array or None
        Minus the Hessian for the `parameters`.
    effects_gradient : (M, G, m) array or None
        d loglik / d effects (zero for an effect held, see
        `LogitDesign.newton_pass`); None without a fixed effect.
    effects_information : (M, G, m, m) array or None
        Each group's block of the information (the identity for the effects
        held).
    cross : (M, G, m, c) array or None
        The information between each group's effects and the `parameters`.
    skipped : (M, S) array
        Rows left out of each model because their group is (see
        `LogitDesign.newton_pass`).
    parameters : array of int
        The parameters (indices into theta) the information is formed for.
    """

    loglik: np.ndarray
    gradient: np.ndarray
    information: np.ndarray | None
    effects_gradient: np.ndarray | None
    effects_information: np.ndarray | None
    cross: np.ndarray | None
    skipped: np.ndarray
    parameters: np.ndarray
    _fe: Any = None
    _free: Any = None
    _reduced: dict = field(default_factory=dict)

    def _schur(self, m):
        if m not in self._reduced:
            H = SimpleNamespace(tt=self.information[m], aa=self.effects_information[m],
                                ta=self.cross[m])
            self._reduced[m] = self._fe._schur(H)
        return self._reduced[m]

    def _gradients(self, m, gradient):
        if gradient is None:
            return self.gradient[m], self.effects_gradient[m] if self._fe is not None else None
        other, j = gradient
        return other.gradient[j], (other.effects_gradient[j] if self._fe is not None else None)

    def profiled(self, m: int = 0, gradient=None):
        """(information, gradient) of model m for the `parameters`, with the
        effects profiled out: I_tt - I_ta I_aa^-1 I_at and g_t - I_ta I_aa^-1
        g_a. Without a fixed effect, the model's information and gradient
        over the `parameters`.

        `gradient`, a pair ``(moments, index)`` of another pass (made with or
        without the information), takes the gradients from that pass's model
        `index` in place of this one's: a Newton step with this pass's
        information at another point's gradient (an iteration that keeps the
        information across steps, as it changes little, and makes the
        passes between without it, which cost much less)."""
        if self.information is None:
            raise ValueError("the pass was made without the information")
        gt, ga = self._gradients(m, gradient)
        g = gt[self.parameters]
        if self._fe is None:
            return self.information[m], g
        H = self._schur(m)
        G, J = self.effects_gradient.shape[1:]
        Bf = H.ta.reshape(G * J, -1)
        return H.red, g - Bf.T @ (H.Dinv @ ga[..., None]).reshape(G * J)

    def effects_step(self, dtheta, m: int = 0, gradient=None):
        """The effects' part of the Newton step, given the parameters' part
        (over `parameters`): I_aa^-1 (g_a - I_at dtheta), zero for an effect
        held. None without a fixed effect. `gradient` as for `profiled`."""
        if self._fe is None:
            return None
        if self.information is None:
            raise ValueError("the pass was made without the information")
        H = self._schur(m)
        G, J = self.effects_gradient.shape[1:]
        Bf = H.ta.reshape(G * J, -1)
        ga = self._gradients(m, gradient)[1]
        d = ga - (Bf @ np.asarray(dtheta, np.float64)).reshape(G, J)
        return (H.Dinv @ d[..., None])[..., 0] * self._free[m]


# --------------------------------------------------------------------------
# one model's running sums in a pass
# --------------------------------------------------------------------------

class _Sums:
    """A model's sums over the blocks it is given: log-likelihood, gradient
    and (with `hessian`) information, by the passes `mlogit_stream` and
    `ologit_stream` fit by (`_Driver._plain_pass`, `_AbsorbedFE.pass_`),
    block by block."""

    def __init__(self, model, fe, theta, A, hessian, rows, K0):
        self.model, self.fe, self.theta, self.hessian = model, fe, theta, hessian
        self.rows, self.K0 = rows, K0
        P = model.P
        self.ok = True
        self.g = np.zeros(P)
        if fe is None:
            self.H = np.zeros((P, P)) if hessian else None
            return
        G, m = fe.G, fe.m
        self.A = A
        self.ga = np.zeros((G, m))
        self.Htt = np.zeros((P, P)) if hessian else None
        self.Haa = np.zeros((G, m, m)) if hessian else None
        self.B = np.zeros((G, m, P)) if hessian else None
        self.kp = fe._kernel_pass(theta, A, hessian, self.ga, self.Haa, self.B)

    def add(self, blk) -> float:
        """Add a block's sums; returns its log-likelihood (-inf: some row
        has probability zero, and the gradient and information are no
        longer meaningful, `ok` false)."""
        ll = self._add_plain(blk) if self.fe is None else self._add_fe(blk)
        if not np.isfinite(ll):
            self.ok = False
            return -np.inf
        return ll

    def _add_plain(self, blk):
        m, th, hessian = self.model, self.theta, self.hessian
        n = len(blk["y"])
        if n >= KERNEL_MIN_ROWS and m.kernel_ok(max(n, MNL_KERNEL_MIN_ROWS)):
            l, g, H = m.accumulate_full(th, blk["X"], m.cols, blk["y"], blk.get("w"), None,
                                        hessian)
            if not np.isfinite(l):
                return -np.inf
            self.g += g
            if hessian:
                self.H += H
            return l
        ll = 0.0
        for ch in _chunks(blk, self.rows):
            X = _design(ch, m.cols, self.K0)
            l, g, H = m.accumulate(th, X, ch["y"], ch.get("w"), None, hessian)
            if not np.isfinite(l):
                return -np.inf
            ll += l
            self.g += g
            if hessian:
                self.H += H
        return ll

    def _add_fe(self, blk):
        fe, m, th = self.fe, self.model, self.theta
        if self.kp is not None and len(blk["y"]) >= KERNEL_MIN_ROWS:
            return self.kp.add(blk)
        ll = 0.0
        for ch in _chunks(blk, self.rows):
            keep = fe.keep_g[ch["fe"]]
            if not keep.all():
                # (an unweighted design's chunks have no weights to subset)
                ch = _subset({k: v for k, v in ch.items() if v is not None}, keep)
            if not len(ch["y"]):
                continue
            X = _design(ch, m.cols, self.K0)
            y, w, f = ch["y"], ch.get("w"), ch["fe"]
            if m.kind == "mlogit":
                l = fe._mnl_chunk(th, self.A, X, y, w, f, self.g, self.ga, self.Htt, self.Haa,
                                  self.B)
            else:
                l = fe._ord_chunk(th, self.A, X, y, w, f, self.g, self.ga, self.Htt, self.Haa,
                                  self.B)
            if not np.isfinite(l):
                return -np.inf
            ll += l
        return ll

    def finish(self):
        """(gradient, information) -- with a fixed effect, the gradient over
        the parameters and the effects, and the information as the
        namespace `_AbsorbedFE` uses (tt, aa, ta), the effects held fixed
        given zero gradient and unit information."""
        if self.fe is None:
            return self.g, self.H
        if self.kp is not None:
            if self.model.kind == "mlogit":
                self.kp.finish(self.g, self.Htt)
            else:
                self.kp.finish(self.g, self.ga, self.Htt, self.Haa, self.B)
        _, g, H = self.fe._finish(0.0, self.g, self.ga, self.Htt, self.Haa, self.B,
                                  self.hessian)
        return g, H


# --------------------------------------------------------------------------
# the stored design
# --------------------------------------------------------------------------

@dataclass
class LogitDesign:
    """The rows of a multinomial or ordered model, stored by pass 0
    (`logit_design`).

    Attributes
    ----------
    formula : str
    model : str
        ``"multinomial"`` or ``"ordered"``.
    link : str
        The ordered model's, ``"logit"`` or ``"probit"`` (``"logit"`` for the
        multinomial).
    depvar : str
    levels : list
        The outcome's levels, in order (the first is the multinomial logit's
        base level); a row's outcome is its level's position.
    covariates : list of str
        The design's columns (k of them).
    names : list of str
        The parameters (P of them; see the module docstring).
    penalized : (P,) boolean array
        The slopes: False for the multinomial logit's intercepts and the
        ordered model's cutpoints.
    parameter_columns : (P,) int array
        Each parameter's column of the design (-1 for a cutpoint).
    fe : str or None
        The absorbed fixed effect.
    n_groups : int
        G: its groups (the levels of `fe`; 0 without one), including those
        dropped (`group_kept`).
    fe_levels : polars.DataFrame or None
        The groups' values of the fixed effect's columns, row g for group g.
    group_kept : (G,) boolean array or None
        The groups kept: not those whose outcome never varies (see
        `separation`), whose effects would be infinite.
    n_splits : int
        S (1 without `split_by`).
    n_obs : int
        Rows used (after dropping the groups whose outcome never varies).
    weight_sum : float
        Their summed weight (analytic weights are scaled to a mean of 1 over
        the rows kept, as the package's estimators report them).
    outcome_counts : (S, G, J) array
        The summed weight of each split's rows at each outcome level in each
        group (G = 1 without a fixed effect).
    row_counts : (S, G) array
        Their numbers of rows.
    x_sums, x_squares : (S, k) arrays
        Sum w x and sum w x^2 of each column of the design in each split.
    x_nonzero : (k,) array
        Each column's share of nonzero rows.
    separation : dict
        The groups dropped because their outcome never varies, and their
        rows.
    diagnostics : dict
    """

    formula: str
    model: str
    link: str
    depvar: str
    levels: list
    covariates: list
    names: list
    penalized: np.ndarray
    parameter_columns: np.ndarray
    fe: str | None
    n_groups: int
    fe_levels: Any
    group_kept: Any
    n_splits: int
    n_obs: int
    weight_sum: float
    outcome_counts: np.ndarray
    row_counts: np.ndarray
    x_sums: np.ndarray
    x_squares: np.ndarray
    x_nonzero: np.ndarray
    separation: dict
    diagnostics: dict = field(default_factory=dict)
    _model: Any = None
    _fe: Any = None
    _store: Any = None
    _run: Any = None
    _split_of: Any = None
    _budget: Any = None
    _chunk_bytes: float = 64e6

    @property
    def passes(self) -> int:
        return int(self.diagnostics.get("passes", 0))

    @property
    def n_params(self) -> int:
        return len(self.names)

    # ------------------------------------------------------------ parameters
    def _effects_width(self):
        return len(self.levels) if self.model == "multinomial" else 1

    def start(self, splits=None):
        """The null model's parameters on the rows of `splits` (default
        all): ``(theta, effects)``, effects None without a fixed effect.

        The slopes are zero, and the intercepts or cutpoints give each
        outcome level its share of those rows -- with a fixed effect, each
        group's effects its own shares (multinomial: -inf for a level the
        group's rows never have, the first level the group has at 0) or each
        group at minus the first cutpoint (ordered). A group whose effects
        would be infinite on these rows is nan (so that `newton_pass` leaves
        it out): for the multinomial logit one with a single outcome level,
        for the ordered model one whose rows are all at the lowest or all at
        the highest level."""
        S = self.n_splits
        sel = np.ones(S, bool) if splits is None else np.isin(np.arange(S), splits)
        n = self.outcome_counts[sel].sum(axis=0)            # (G, J)
        m = self._model
        th = m.start(n.sum(axis=0), fe=self._fe is not None)
        if self._fe is None:
            return th, None
        has = n > 0
        nlev = has.sum(axis=1)
        if m.kind == "mlogit":
            first = np.argmax(has, axis=1)
            with np.errstate(divide="ignore"):
                ln = np.log(n)
            A = ln - ln[np.arange(len(n)), first][:, None]
            A[nlev < 2] = np.nan
        else:
            A = np.full((len(n), 1), -m.marginal_cuts(n.sum(axis=0))[0])
            A[(nlev == 1) & (has[:, 0] | has[:, -1])] = np.nan
        A[~self.group_kept] = np.nan
        return th, A

    def feasible(self, theta) -> bool:
        """Whether `theta` is a valid point: finite, and for the ordered
        model with its cutpoints increasing."""
        return self._model.feasible(np.asarray(theta, np.float64))

    def _columns(self, columns):
        """The design columns the information is formed for: `columns`
        (default all), with the multinomial logit's Intercept."""
        k = len(self.covariates)
        if columns is None:
            return np.arange(k)
        cols = np.unique(np.asarray(columns, dtype=np.int64))
        if len(cols) and (cols[0] < 0 or cols[-1] >= k):
            raise ValueError(f"columns must be in 0..{k - 1}")
        ic = self._model.intercept if self.model == "multinomial" else None
        if ic is not None:
            cols = np.union1d(cols, [ic])
        return cols

    def _params(self, cols):
        """The parameters of the design columns `cols` (and, for the ordered
        model, the cutpoints), in their order in theta."""
        pc = self.parameter_columns
        return np.flatnonzero(np.isin(pc, cols) | (pc < 0))

    def _restricted(self, cols):
        """A copy of the model on the design columns `cols` (the passes then
        form only their parameters' sums)."""
        m = copy.copy(self._model)
        m.cols = np.asarray(cols, dtype=np.int64)
        if m.kind == "mlogit":
            m._free_cache = None
        if len(cols):
            m.density = float(np.mean(self.x_nonzero[cols]))
        return m

    def _model_fe(self, model, A):
        """The fixed effect for one model of a pass, with its effects `A`
        (G x m): the groups with a nan effect (any, for the ordered model a
        non-finite one) left out; for the multinomial logit, the levels at
        -inf held there and each group's first finite effect held at its
        value (the effects are identified up to a common shift). Returns
        (fe, effects as the passes take them)."""
        base = self._fe
        fe = copy.copy(base)
        fe.model = model
        A = np.array(A, dtype=np.float64)
        if model.kind == "mlogit":
            fin = np.isfinite(A)
            skip = np.isnan(A).any(axis=1) | (A == np.inf).any(axis=1) | ~fin.any(axis=1)
            first = np.argmax(fin, axis=1)
            free = fin.copy()
            free[np.arange(len(A)), first] = False
            keep = base.keep_g & ~skip
            fe.keep_g = keep
            fe.free = free & keep[:, None]
            A = np.where(skip[:, None], 0.0, A)
            fe.fixed = A
            A = np.where(fe.free, A, fe.fixed)
        else:
            skip = ~np.isfinite(A[:, 0])
            keep = base.keep_g & ~skip
            fe.keep_g = keep
            fe.free = base.free & keep[:, None]
            A = np.where(skip[:, None], 0.0, A)
        return fe, np.ascontiguousarray(A), ~keep & base.keep_g

    # ------------------------------------------------------------ passes
    def _rows_per_chunk(self, model):
        return max(256, int(self._chunk_bytes // (8 * max(model.width(), 1))))

    def _blocks(self):
        """The stored blocks, each cut by split: ``(split, block)`` pairs."""
        S = self.n_splits
        for blk in self._store.iterate(["X", "y", "w", "fe", "cl"]):
            sub = {"X": np.ascontiguousarray(blk["X"]), "y": blk["y"], "w": blk.get("w"),
                   "fe": blk.get("fe")}
            if S == 1:
                yield 0, sub
                continue
            s = self._split_of[blk["cl"][:, 0]]
            order = np.argsort(s, kind="stable")
            bounds = np.searchsorted(s[order], np.arange(S + 1))
            for sp in range(S):
                idx = order[bounds[sp]:bounds[sp + 1]]
                if len(idx):
                    yield sp, {k: (None if v is None else v[idx]) for k, v in sub.items()}

    def _sorted_blocks(self):
        """The stored blocks with their rows sorted by split: ``(block,
        bounds)``, split s at rows bounds[s]:bounds[s + 1]."""
        S = self.n_splits
        for blk in self._store.iterate(["X", "y", "w", "fe", "cl"]):
            s = self._split_of[blk["cl"][:, 0]]
            order = np.argsort(s, kind="stable")
            bounds = np.searchsorted(s[order], np.arange(S + 1))
            sub = {"X": np.ascontiguousarray(blk["X"][order]), "y": blk["y"][order],
                   "w": None if blk.get("w") is None else blk["w"][order]}
            yield sub, bounds

    @staticmethod
    def _runs(cover):
        """The runs of consecutive splits in `cover` (booleans): (first, end) pairs."""
        out, start = [], None
        for sp, c in enumerate(list(cover) + [False]):
            if c and start is None:
                start = sp
            elif not c and start is not None:
                out.append((start, sp))
                start = None
        return out

    def newton_pass(self, theta, effects=None, *, models=None, columns=None,
                    information: bool = True) -> NewtonMoments:
        """One read of the rows: each model's log-likelihood on every split,
        and its gradient and information over its splits.

        Parameters
        ----------
        theta : (M, P) or (P,) array
            Each model's parameters (`names`).
        effects : (M, G, m) or (G, m) array, optional
            With a fixed effect, each model's group effects (m = J for the
            multinomial logit, 1 for the ordered model); `start` gives the
            null model's. A group with a nan effect (for the ordered model, a
            non-finite one) is left out of that model -- its rows' log-
            likelihood, gradient and information -- and its rows counted in
            `skipped`: the way to hold a group out of a model (one whose
            outcome does not vary within the model's rows, whose effects
            would be infinite). For the multinomial logit an effect of -inf
            holds that outcome level at probability zero in the group (a
            level the model's rows of the group never have), and each
            group's first finite effect is held at its value: the effects
            are identified only up to a shift common to the group, and the
            pass gives that one zero gradient and step.
        models : (M, S) boolean array, optional
            The splits each model's gradient and information cover; default
            all.
        columns : sequence of int, optional
            The design's columns (`covariates`) the information is formed
            for (an active set): their parameters in every equation of the
            multinomial logit, with the Intercept's, and for the ordered
            model with the cutpoints. Default all. The gradient is always
            for every parameter.
        information : bool
            Whether to form the information (False: the log-likelihood and
            gradients only, which cost much less).
        """
        P = self.n_params
        theta = np.asarray(theta, dtype=np.float64)
        theta = theta[None, :] if theta.ndim == 1 else theta
        if theta.ndim != 2 or theta.shape[1] != P:
            raise ValueError(f"theta must be (M, {P}) or ({P},)")
        M, S = len(theta), self.n_splits
        if models is None:
            models = np.ones((M, S), bool)
        models = np.asarray(models, dtype=bool)
        if models.shape != (M, S):
            raise ValueError(f"models must be ({M}, {S})")
        fe0 = self._fe
        if fe0 is None:
            if effects is not None:
                raise ValueError("effects= is for a design with a fixed effect")
        else:
            shape = (fe0.G, fe0.m)
            if effects is None:
                raise ValueError("a design with a fixed effect needs effects=")
            effects = np.asarray(effects, dtype=np.float64)
            effects = effects[None] if effects.ndim == 2 else effects
            if effects.shape != (M, *shape):
                raise ValueError(f"effects must be ({M}, {shape[0]}, {shape[1]}) or {shape}")
        cols = self._columns(columns)
        params = self._params(cols)
        k = len(self.covariates)
        full_model = self._model
        pc = self.parameter_columns
        rows = self._rows_per_chunk(full_model)
        sums, held, restr, subs, skipped = [], [], [], [], np.zeros((M, S))
        free = np.zeros((M, *shape), bool) if fe0 is not None else None
        for i in range(M):
            th = theta[i]
            # the columns the information is formed on: those asked for, and
            # any with a nonzero parameter (whose linear predictor needs them)
            nz = np.unique(pc[(th != 0) & (pc >= 0)])
            ev = np.union1d(cols, nz).astype(np.int64)
            narrow = information and len(ev) < k
            fe_i = A_i = fe_r = None
            if fe0 is not None:
                fe_i, A_i, out = self._model_fe(full_model, effects[i])
                free[i] = fe_i.free
                skipped[i] = self.row_counts[:, out].sum(axis=1)
            sums.append(_Sums(full_model, fe_i, th, A_i, information and not narrow, rows, k))
            held.append(_Sums(full_model, fe_i, th, A_i, False, rows, k)
                        if not models[i].all() else None)
            if narrow:
                mr = self._restricted(ev)
                if fe0 is not None:
                    fe_r = copy.copy(fe_i)
                    fe_r.model = mr
                pr = self._params(ev)
                restr.append(_Sums(mr, fe_r, th[pr], A_i, True, self._rows_per_chunk(mr), k))
                subs.append(np.searchsorted(pr, params))
            else:
                restr.append(None)
                subs.append(params)
        if fe0 is not None and information:
            gb = sum(fe0.G * fe0.m * len(params) * 8 for _ in range(M)) / 1e9
            if not self._budget.fits(gb):
                raise MemoryError(
                    f"the pass's per-group blocks for {M} models need about {gb:.1f} GB and "
                    f"{self._budget.headroom_gb():.1f} GB is free ({self._budget}); pass fewer "
                    "models or columns, or a larger max_memory_gb")
        ll = np.zeros((M, S))
        numba = information and (fe0.numba_pass() if fe0 is not None
                                 else full_model.sparse(None))
        with _single_blas(numba):
            if fe0 is None and S > 1 and information:
                self._merged_pass(models, sums, held, restr, ll, rows, k)
            else:
                for sp, blk in self._blocks():
                    for i in range(M):
                        if models[i, sp]:
                            ll[i, sp] += sums[i].add(blk)
                            if restr[i] is not None:
                                restr[i].add(blk)
                        else:
                            ll[i, sp] += held[i].add(blk)
        self.diagnostics["passes"] = self.passes + 1
        return self._moments(sums, restr, subs, params, ll, skipped, free, information)

    def _merged_pass(self, models, sums, held, restr, ll, rows, k):
        """The pass without a fixed effect, over splits: each block's rows
        sorted by split once, so that a model's information comes from one
        call per run of its consecutive splits (at most two), not one per
        split -- a call's fixed cost outweighs its rows' on a split's piece
        of a block. Each split's log-likelihood (and, when the information
        is restricted, the gradient) still comes split by split, from the
        passes without the information, which cost little."""
        M, S = models.shape
        runs = [self._runs(models[i]) for i in range(M)]
        # the log-likelihood of each split when the information's sums are
        # over runs of splits (theirs is only the runs' total)
        llo = [(_Sums(sums[i].model, None, sums[i].theta, None, False, rows, k)
                if restr[i] is None else None) for i in range(M)]
        for blk, bounds in self._sorted_blocks():
            def piece(a, b, blk=blk, bounds=bounds):
                lo, hi = bounds[a], bounds[b]
                return {key: (None if v is None else v[lo:hi]) for key, v in blk.items()}
            pieces = [piece(sp, sp + 1) for sp in range(S)]
            for i in range(M):
                for sp in range(S):
                    if bounds[sp + 1] == bounds[sp]:
                        continue
                    if models[i, sp]:
                        one = sums[i] if restr[i] is not None else llo[i]
                        ll[i, sp] += one.add(pieces[sp])
                    else:
                        ll[i, sp] += held[i].add(pieces[sp])
                info = restr[i] if restr[i] is not None else sums[i]
                for a, b in runs[i]:
                    if bounds[b] > bounds[a]:
                        info.add(piece(a, b) if b - a > 1 else pieces[a])

    def _moments(self, sums, restr, subs, params, ll, skipped, free, information):
        M, P, c = len(sums), self.n_params, len(params)
        fe = self._fe
        grad = np.zeros((M, P))
        info = np.zeros((M, c, c)) if information else None
        eg = ei = cross = None
        if fe is not None:
            G, m = fe.G, fe.m
            eg = np.zeros((M, G, m))
            if information:
                ei = np.zeros((M, G, m, m))
                cross = np.zeros((M, G, m, c))
        for i in range(M):
            ok = sums[i].ok and (restr[i] is None or restr[i].ok)
            g, H = sums[i].finish()
            if restr[i] is not None:
                _, H = restr[i].finish()
            if not ok:
                ll[i, ~np.isfinite(ll[i])] = -np.inf
                grad[i] = np.nan
                if eg is not None:
                    eg[i] = np.nan
                if information:
                    info[i] = np.nan
                    if fe is not None:
                        ei[i] = np.nan
                        cross[i] = np.nan
                continue
            sub = subs[i]
            if fe is None:
                grad[i] = g
                if information:
                    info[i] = H[np.ix_(sub, sub)]
                continue
            grad[i] = g[:P]
            eg[i] = g[P:].reshape(G, m)
            if information:
                info[i] = H.tt[np.ix_(sub, sub)]
                ei[i] = H.aa
                cross[i] = H.ta[:, :, sub]
        return NewtonMoments(loglik=ll, gradient=grad, information=info, effects_gradient=eg,
                             effects_information=ei, cross=cross, skipped=skipped,
                             parameters=np.asarray(params, dtype=np.int64), _fe=fe, _free=free)

    def rows(self):
        """The stored rows of the kept groups, in chunks: ``(X, y, w, group,
        split)`` arrays, for sums a pass does not give (X the design, y the
        outcome's level 0..J-1, w the weights or None, group the row's group
        0..G-1 (0 without a fixed effect), split its split 0..S-1). Counts as
        a pass."""
        self.diagnostics["passes"] = self.passes + 1
        keep = self.group_kept
        for blk in self._store.iterate(["X", "y", "w", "fe", "cl"]):
            n = len(blk["y"])
            f = blk.get("fe")
            f = np.zeros(n, np.int64) if f is None else np.asarray(f, np.int64)
            s = (self._split_of[blk["cl"][:, 0]] if self.n_splits > 1
                 else np.zeros(n, np.int64))
            X, y, w = blk["X"], blk["y"], blk.get("w")
            if keep is not None and not keep[f].all():
                r = keep[f]
                X, y, f, s = X[r], y[r], f[r], s[r]
                w = None if w is None else w[r]
            if len(y):
                yield X, y, w, f, s

    def cleanup(self) -> None:
        """Delete the stored rows."""
        if self._run is not None:
            self._run.cleanup()
            self._run = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()


# --------------------------------------------------------------------------
# pass 0
# --------------------------------------------------------------------------

def logit_design(fml: str, data: Any, *, model: str, link: str = "logit",
                 levels=None, weights: str | None = None, weights_type: str = "aweights",
                 split_by: str | None = None, n_splits: int | None = None, workdir=None,
                 storage: str = "disk", dtype: str = "float64", block_mb: float = 64.0,
                 chunk_mb: float = 64.0, polars_engine: str = "cpu",
                 max_memory_gb: float | None = None, memory_reserve: float | None = None,
                 verbose: bool = False, logger=None) -> LogitDesign:
    """Pass 0 of a multinomial logit or an ordered logit or probit: the rows
    stored for `LogitDesign.newton_pass`.

    Parameters
    ----------
    fml : str
        One model, ``"y ~ x1 + x2"`` or ``"y ~ x1 + x2 | fe"`` (at most one
        absorbed fixed effect; pyfixest's syntax for the covariates), as for
        `mlogit_stream` and `ologit_stream`.
    data : str, Path, polars.LazyFrame or polars.DataFrame
    model : {"multinomial", "ordered"}
    link : {"logit", "probit"}
        The ordered model's (the multinomial is a logit).
    levels : sequence, optional
        The outcome's levels in order, as for `mlogit_stream` and
        `ologit_stream` (default: its values, sorted).
    weights, weights_type
        Analytic weights (scaled to a mean of 1) or frequency weights.
    split_by : str, optional
        An integer column with values 0, ..., n_splits - 1: each row's split.
    n_splits : int, optional
        With `split_by`.
    workdir : path, optional
        Where the rows are stored (the system's temporary directory by
        default, or ``LOGIT_STREAM_WORKDIR``).
    storage : {"disk", "memory"}
    dtype : {"float64", "float32"}
        The stored design's precision (the 0/1 columns take one byte either
        way).
    block_mb, chunk_mb : float
        The stored blocks' size, and the chunks a pass works on.
    polars_engine, max_memory_gb, memory_reserve, verbose, logger
        As for the estimators.

    Returns
    -------
    LogitDesign
        Call `cleanup()` (or use it as a context manager) to delete the rows.

    With a fixed effect, its groups whose outcome never varies (multinomial:
    a single level; ordered: all at the lowest or all at the highest level)
    are dropped, with a warning, as the estimators drop them.
    """
    from .._frontend import _plan
    from ..estimator import _base_dir, _log

    if model not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}, not {model!r}")
    kind = MODELS[model]
    if link not in ("logit", "probit"):
        raise ValueError("link must be 'logit' or 'probit'")
    if kind == "mlogit" and link != "logit":
        raise ValueError("the multinomial model is a logit (link='logit')")
    if weights_type not in ("aweights", "fweights"):
        raise ValueError("weights_type must be 'aweights' or 'fweights'")
    if storage not in ("disk", "memory"):
        raise ValueError("storage must be 'disk' or 'memory'")
    if dtype not in ("float64", "float32"):
        raise ValueError("dtype must be 'float64' or 'float32'")
    if (split_by is None) != (n_splits is None):
        raise ValueError("give split_by and n_splits together")
    if n_splits is not None and (isinstance(n_splits, bool) or not isinstance(n_splits, int)
                                 or n_splits < 1):
        raise ValueError("n_splits must be a positive integer")

    def log(msg):
        _log(verbose, msg, logger)

    t0 = time.time()
    lf = as_lazy(data)
    schema = lf.collect_schema()
    if split_by is not None and split_by not in schema:
        raise ValueError(f"split_by: column {split_by!r} is not in the data")
    engine = PolarsEngine(polars_engine)
    specs = _plan(fml, lf, intercept=kind == "mlogit",
                  collect=lambda q, what: engine.collect(q, what))
    if len(specs) != 1:
        raise ValueError(f"{fml}: one model (not a multiple-estimation formula)")
    spec = specs[0]
    if len(spec.fe) > 1:
        raise ValueError(f"{fml}: at most one absorbed fixed effect (got {list(spec.fe)})")
    budget = MemoryBudget(max_memory_gb, memory_reserve)
    clusters = {SPLIT: [split_by]} if split_by is not None else {}
    run = Run(_base_dir(workdir), auto_cleanup=True)
    try:
        p0 = _build(lf, spec, kind, run.path, engine, weights=weights, clusters=clusters,
                    case=None, levels=levels, in_memory=storage == "memory",
                    dtype=np.float32 if dtype == "float32" else np.float64,
                    block_mb=block_mb, log=log, weights_type=weights_type,
                    sort_fe=SORT_FE and len(spec.fe) == 1, budget=budget)
        xnames = [nm for nm, _ in spec.x]
        K = len(xnames)
        if kind == "mlogit":
            ic = xnames.index("Intercept") if "Intercept" in xnames else None
            mdl = _Multinomial(xnames, K, p0.levels, ic)
        else:
            mdl = _Ordered(xnames, K, p0.levels, link=link)
        if K:
            mdl.density = float(np.mean(p0.xnonzero))
        fe = None
        if spec.fe:
            fe = _AbsorbedFE(mdl, p0, spec.fe[0], log=log, logger=logger, budget=budget)
        if weights and weights_type != "fweights" and p0.wsum != p0.n_obs:
            # analytic weights to a mean of 1 over the rows kept (after the
            # fixed effect's dropped groups), as the estimators report
            _rescale_weights(p0.store, p0.n_obs / p0.wsum)
            p0.wsum = float(p0.n_obs)
        S = n_splits or 1
        split_of = None
        if split_by is not None:
            t = p0.cluster_tables[SPLIT].sort(CODE)
            vals = t[split_by]
            if not vals.dtype.is_integer():
                raise ValueError(f"split_by: column {split_by!r} must be integers")
            split_of = vals.to_numpy().astype(np.int64)
            if split_of.min() < 0 or split_of.max() >= S:
                raise ValueError(f"split_by: values must be in 0..{S - 1}")
        J = len(p0.levels)
        G = fe.G if fe is not None else 1
        keep = fe.keep_g if fe is not None else None
        counts = np.zeros((S, G, J))
        nrows = np.zeros((S, G))
        xs = np.zeros((S, K))
        xq = np.zeros((S, K))
        log("counting the outcome and the design's moments by split")
        for blk in p0.store.iterate(["X", "y", "w", "fe", "cl"]):
            n = len(blk["y"])
            f = np.zeros(n, np.int64) if blk.get("fe") is None else blk["fe"]
            s = split_of[blk["cl"][:, 0]] if split_of is not None else np.zeros(n, np.int64)
            X, y, w = blk["X"], blk["y"], blk.get("w")
            if keep is not None and not keep[f].all():
                r = keep[f]
                X, y, f, s = X[r], y[r], f[r], s[r]
                w = None if w is None else w[r]
            wt = np.ones(len(y)) if w is None else w
            counts += np.bincount((s * G + f) * J + y, weights=wt,
                                  minlength=S * G * J).reshape(S, G, J)
            nrows += np.bincount(s * G + f, minlength=S * G).reshape(S, G)
            for sp in range(S):
                r = s == sp if S > 1 else slice(None)
                Xs, ws = X[r], wt[r]
                xs[sp] += ws @ Xs
                xq[sp] += ws @ (Xs * Xs)
        if kind == "mlogit":
            pcols = np.tile(np.arange(K), J - 1)
        else:
            pcols = np.r_[np.arange(K), np.full(mdl.n_cuts, -1)].astype(np.int64)
        ld = LogitDesign(
            formula=spec.fml, model=model, link=link if kind == "ologit" else "logit",
            depvar=spec.depvar, levels=list(p0.levels), covariates=xnames, names=mdl.names(),
            penalized=np.asarray(mdl.penalized(), bool), parameter_columns=pcols,
            fe=spec.fe[0] if spec.fe else None, n_groups=fe.G if fe is not None else 0,
            fe_levels=(p0.fe_tables[spec.fe[0]].sort(CODE).drop(CODE) if fe is not None
                       else None),
            group_kept=keep, n_splits=S, n_obs=int(p0.n_obs), weight_sum=float(p0.wsum),
            outcome_counts=counts, row_counts=nrows, x_sums=xs, x_squares=xq,
            x_nonzero=np.asarray(p0.xnonzero, float),
            separation=(fe.separation if fe is not None
                        else {"checked": False, "levels": 0, "observations": 0}),
            diagnostics={"passes": 0, "seconds": round(time.time() - t0, 3)},
            _model=mdl, _fe=fe, _store=p0.store, _run=run, _split_of=split_of,
            _budget=budget, _chunk_bytes=float(chunk_mb) * 1e6)
        log(f"pass 0 done: {ld.n_obs:,} rows, {len(ld.names)} parameters "
            f"({time.time() - t0:.1f} s)")
        return ld
    except BaseException:
        run.cleanup()
        raise
