"""Constrained assignment (experimental): draws of each choice set's
alternative such that the alternatives' counts match known totals,
exactly or approximately -- for imputation, workers to the establishments
of their firm with each establishment's employment known.

The model's utilities u (a conditional logit's x'b, with any other
absorbed constants) give each assignment a of sets to alternatives the
weight prod_i exp(u[i, a(i)]). The samplers draw from

* exact: that distribution given the alternatives' counts n_j equal
  integer targets N_j. The alternatives' own constants then cancel (each
  is in every allowed assignment N_j times), so the utilities leave them
  out;
* soft: that distribution with the calibrated constants (those whose
  expected counts are the totals: see `calibrate.py`) times a penalty on
  the counts, prod_j g_j(n_j), e.g. a Gaussian around the reported total
  (`GaussianPenalty`). Between the two limits: a weak penalty gives the
  calibrated model's independent draws; a sharp one, the exact draw.

`newdata` is streamed a batch of whole groups of linked alternatives (a
firm's establishments) at a time, as in the calibration. An alternative
with a total of 0 is never chosen. Random numbers are keyed by (seed,
group), so a draw does not depend on the batches.

The samplers
------------
``SwapMCMC``
    Metropolis-Hastings from an assignment that meets the counts: swaps of
    two sets' alternatives (which keep the counts) and, when soft, moves of
    one set (which change two counts by one).
``ExactSmall``
    Exact, by recursion over the sets with the counts so far as the state:
    for small groups (the states multiply with the alternatives), with a
    fallback sampler for the others.
``Balanced``
    The cube method's balanced sampling (Deville and Tille 2004), on the
    sets x alternatives table: each set's probability of each alternative
    is the calibrated model's exactly, and the counts are the targets
    exactly. The two-way margins make the polytope's vertices integral, so
    the flight phase -- here dependent rounding along cycles of fractional
    cells (Gandhi, Khuller, Parthasarathy and Srinivasan 2006) -- needs no
    landing phase. It targets those margins, not the conditional
    distribution the others draw from.

Customizing
-----------
A sampler is a `ConstrainedSampler`, with hooks a subclass overrides:
``targets`` (each alternative's target: by default, for the exact
samplers, the totals scaled to each group's number of sets and rounded at
random keeping the group's sum), ``utilities``, and ``sample`` (a batch
at a time; by default a loop over ``sample_group``, for samplers written
in Python). The soft constraint is a penalty: any function of (counts,
totals, alternatives' ids) returning log g, vectorized.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numba as nb
import numpy as np
import polars as pl
from hdfe_stream.core.report import ExperimentalWarning as _Experimental

from .calibrate import Batch, _stream, _with_totals, calibrate_batch
from .draws import _mix, _uniform

EPS = 1e-9                  # fractional cells of the balanced sampler
MAX_CELLS = 20_000_000      # of ExactSmall's recursion, per group


class ExperimentalWarning(_Experimental):
    """A logit-stream feature its authors have not yet validated.

    It subclasses ``hdfe_stream.ExperimentalWarning``: filter this class to
    silence logit-stream's experimental warnings, or that one for every
    package built on hdfe-stream.
    """


def _experimental():
    warnings.warn("constrained assignment is experimental: its samplers and their interface "
                  "may change", ExperimentalWarning, stacklevel=3)


# ------------------------------------------------------------------- penalties
class GaussianPenalty:
    """The soft constraint log g_j(n) = -(n - N_j)^2 / (2 s_j^2), the
    count n near the reported total N_j with s_j^2 = sd^2 + (rel N_j)^2:
    an absolute and a relative discrepancy between the totals and the
    assigned counts."""

    def __init__(self, sd: float = 1.0, rel: float = 0.05):
        if sd < 0 or rel < 0 or sd + rel == 0:
            raise ValueError("GaussianPenalty: sd and rel are >= 0, not both 0")
        self.sd, self.rel = float(sd), float(rel)

    def __call__(self, n, total, level_id):
        s2 = self.sd ** 2 + (self.rel * total) ** 2
        return -0.5 * (n - total) ** 2 / s2

    def __repr__(self):
        return f"GaussianPenalty(sd={self.sd}, rel={self.rel})"


# ---------------------------------------------------------------- the samplers
class ConstrainedSampler:
    """The base class: the hooks, and the arrays they work on.

    Parameters
    ----------
    penalty : callable, optional
        None for exact counts; for soft ones, log g as a function of
        (counts, totals, the alternatives' ids in the totals frame), each
        an array, returning an array (e.g. `GaussianPenalty`).

    After a run, ``diagnostics`` holds the sampler's counts over the
    batches.
    """

    def __init__(self, penalty=None):
        self.penalty = penalty
        self.diagnostics = {}

    @property
    def exact(self):
        return self.penalty is None

    # hooks ----------------------------------------------------------------
    def targets(self, batch, seed):
        """Each alternative's target (per ``batch.level_id``): the totals
        scaled to each group's number of sets (every assignment's counts
        sum to it), and when exact rounded at random keeping each group's
        sum (``round_targets``). (``batch.total`` has them as given.)"""
        if self.exact:
            return round_targets(batch, batch.scaled(), seed)
        return batch.scaled()

    def utilities(self, batch, targets):
        """Each row's utility. Exact: the model's, without the dimension's
        constants (they cancel). Soft: with the constants calibrated to the
        totals scaled to each group's number of sets."""
        if self.exact:
            return batch.xb
        delta, _, _ = calibrate_batch(batch, batch.scaled())
        return batch.xb + delta[batch.lev]

    def sample(self, batch, view, seed):
        """Each set's chosen row of `view` (a ``View``: the live rows), -1
        for none; by default a loop over the groups' ``sample_group``."""
        out = np.full(view.n_sets, -1, np.int64)
        for g in range(batch.n_groups):
            sets = view.gsets[view.gstart[g]:view.gstart[g + 1]]
            if len(sets):
                out[sets] = self.sample_group(view.group(g), seed)
        return out

    def sample_group(self, group, seed):
        """One group's draw: each of its sets' chosen row of the view."""
        raise NotImplementedError(f"{type(self).__name__} defines neither sample nor "
                                  "sample_group")

    # bookkeeping -----------------------------------------------------------
    def _count(self, **kw):
        for k, v in kw.items():
            self.diagnostics[k] = self.diagnostics.get(k, 0) + v

    def __repr__(self):
        return f"{type(self).__name__}(penalty={self.penalty!r})"


class SwapMCMC(ConstrainedSampler):
    """Metropolis-Hastings over assignments, from one that meets the
    targets (greedy, then augmenting paths).

    Exact: swaps -- two sets at alternatives a and b exchange them, which
    keeps every count. Soft: also moves of one set from a to b, with the
    penalty's change in the acceptance ratio (a share `p_swap` of the
    proposals are swaps). `sweeps` proposals per set; each draw starts from
    the same assignment, so they are independent given enough sweeps.
    ``diagnostics`` counts the proposals and acceptances of each kind, and
    the sets with a choice (``sets_movable``) and those whose alternative
    ended other than at the start (``sets_moved``): with good mixing about
    as many as the draw's own spread implies. For a check, compare the
    results with twice the sweeps.
    """

    def __init__(self, penalty=None, sweeps: int = 200, p_swap: float = 0.5):
        super().__init__(penalty)
        if sweeps < 1:
            raise ValueError("SwapMCMC: sweeps >= 1")
        if not 0 < p_swap <= 1:
            raise ValueError("SwapMCMC: p_swap in (0, 1]")
        self.sweeps, self.p_swap = int(sweeps), float(p_swap)

    def sample(self, batch, view, seed):
        a = view.start(batch, self.start_targets(batch, view, seed), strict=self.exact)
        a0 = a.copy()
        n_moves = (view.group_sizes() * self.sweeps).astype(np.int64)
        acc = np.zeros((batch.n_groups, 4), np.int64)
        pen, poff = view.penalty_table(self.penalty) if not self.exact else \
            (np.zeros(1), np.zeros(1, np.int64))
        _mcmc(view.gstart, view.gsets, view.ls, view.llev, view.lu, a, view.members,
              view.mstart, view.count, view.pos, not self.exact, pen, poff, n_moves,
              1.0 if self.exact else self.p_swap, _key(seed, 1), batch.group_key.astype(np.uint64),
              acc)
        t = acc.sum(0)
        movable = (view.ls[1:] - view.ls[:-1]) > 1
        self._count(swaps_proposed=int(t[0]), swaps_accepted=int(t[1]),
                    moves_proposed=int(t[2]), moves_accepted=int(t[3]),
                    sets_movable=int(movable.sum()),
                    sets_moved=int((view.llev[a[movable]] != view.llev[a0[movable]]).sum()))
        return a

    def start_targets(self, batch, view, seed):
        """The counts the start meets: the targets when exact; when soft,
        the totals scaled and rounded (the start then need not meet them)."""
        return view.targets if self.exact else round_targets(batch, batch.scaled(), seed)

    def __repr__(self):
        return f"SwapMCMC(penalty={self.penalty!r}, sweeps={self.sweeps}, p_swap={self.p_swap})"


class ExactSmall(ConstrainedSampler):
    """Exact draws by recursion over a group's sets, the state being the
    counts so far: W_i(c) = log sum over set i's alternatives j of
    exp(u_ij) W_{i+1}(c + e_j), ending in the constraint (exact: c = N) or
    the penalty (soft: sum_j log g_j(c_j)); then forward, each set's
    alternative with probability exp(u_ij) W_{i+1}(c + e_j) / W_i(c). Its
    cost is the sets times the product over the alternatives of their
    counts' ranges; a group above `max_cells` goes to `fallback` (an error
    without one)."""

    def __init__(self, penalty=None, max_cells: int = MAX_CELLS,
                 fallback: ConstrainedSampler | None = None):
        super().__init__(penalty)
        self.max_cells = int(max_cells)
        if fallback is not None and (fallback.penalty is None) != (penalty is None):
            raise ValueError("ExactSmall: the fallback must be exact (or soft) as this sampler is")
        self.fallback = fallback

    def sample(self, batch, view, seed):
        big = [g for g in range(batch.n_groups) if view.gstart[g + 1] > view.gstart[g]
               and self._cells(view.group(g)) > self.max_cells]
        if big and self.fallback is None:
            raise ValueError(f"ExactSmall: {len(big):,} groups need more than max_cells="
                             f"{self.max_cells:,} states; pass a fallback= sampler")
        out = self.fallback.sample(batch, view, seed) if big else \
            np.full(view.n_sets, -1, np.int64)
        big = set(big)
        for g in range(batch.n_groups):
            sets = view.gsets[view.gstart[g]:view.gstart[g + 1]]
            if len(sets) and g not in big:
                out[sets] = self.sample_group(view.group(g), seed)
        self._count(groups_exact=batch.n_groups - len(big), groups_fallback=len(big))
        return out

    def _dims(self, G):
        return (G.targets.astype(np.int64) + 1) if self.exact else (G.avail + 1)

    def _cells(self, G):
        return float(np.prod(self._dims(G).astype(float))) * (G.n + 1)

    def sample_group(self, G, seed):
        dims = tuple(int(d) for d in self._dims(G))
        J = len(dims)
        if self.exact:
            W = np.full(dims, -np.inf)
            W[tuple(G.targets.astype(np.int64))] = 0.0
        else:
            W = np.zeros(dims)
            for j in range(J):
                n = np.arange(dims[j], dtype=float)
                pj = np.asarray(self.penalty(n, np.full(dims[j], G.total[j]),
                                             np.full(dims[j], G.level_id[j])), float)
                W = W + pj.reshape([dims[j] if k == j else 1 for k in range(J)])
        Ws = [None] * (G.n + 1)
        Ws[G.n] = W
        for i in range(G.n - 1, -1, -1):
            acc = np.full(dims, -np.inf)
            for r in range(G.ls[i], G.ls[i + 1]):
                acc = np.logaddexp(acc, G.lu[r] + _shift(Ws[i + 1], G.llev[r]))
            Ws[i] = acc
        if not np.isfinite(Ws[0][(0,) * J]):
            raise ValueError(f"ExactSmall: the targets of group {G.key} cannot be met by its "
                             "sets' alternatives")
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, int(G.key)])
        c = [0] * J
        out = np.empty(G.n, np.int64)
        for i in range(G.n):
            rows = np.arange(G.ls[i], G.ls[i + 1])
            lp = np.array([G.lu[r] + (Ws[i + 1][tuple(c[k] + (k == G.llev[r]) for k in range(J))]
                                      if c[G.llev[r]] + 1 < dims[G.llev[r]] else -np.inf)
                           for r in rows])
            p = np.exp(lp - lp.max())
            r = rows[rng.choice(len(rows), p=p / p.sum())]
            out[i] = G.rows[r]
            c[G.llev[r]] += 1
        return out

    def __repr__(self):
        return (f"ExactSmall(penalty={self.penalty!r}, max_cells={self.max_cells}, "
                f"fallback={self.fallback!r})")


class Balanced(ConstrainedSampler):
    """Balanced sampling by the cube method: each set's probability of each
    alternative is the calibrated model's (the constants solved for the
    integer targets), and every count is its target, by dependent rounding
    of the sets x alternatives table of probabilities along cycles of its
    fractional cells, each step a martingale (so each cell's expectation is
    its probability) that makes at least one cell 0 or 1. Exact counts only."""

    def __init__(self):
        super().__init__(None)

    def utilities(self, batch, targets):
        return batch.xb

    def sample(self, batch, view, seed):
        tgt = np.zeros(len(batch.level_id))
        tgt[view.levels] = view.targets[view.levels]
        live = np.zeros(len(batch.lev), bool)
        live[view.crow] = True
        _, p, _ = calibrate_batch(batch, tgt, live=live)
        x = p[view.crow].copy()
        a = np.full(view.n_sets, -1, np.int64)
        order = np.argsort(view.llev, kind="stable").astype(np.int64)
        astart = np.r_[0, np.cumsum(np.bincount(view.llev, minlength=view.n_levels))] \
            .astype(np.int64)
        snaps = np.zeros(batch.n_groups, np.int64)
        sets = view.gsets
        nrows = np.add.reduceat(np.r_[view.ls[sets + 1] - view.ls[sets], 0],
                                view.gstart[:-1]) if len(sets) else \
            np.zeros(batch.n_groups, np.int64)
        _round(view.gstart, view.gsets, view.ls, view.llev, order, astart, x, a,
               _key(seed, 2), batch.group_key.astype(np.uint64), snaps,
               np.asarray(nrows, np.int64))
        self._count(snapped=int(snaps.sum()))
        return a

    def __repr__(self):
        return "Balanced()"


def round_targets(batch, x, seed):
    """`x` (per level) rounded to integers keeping each group's sum (its
    number of sets): systematic sampling of the fractional parts within
    each group, from a uniform keyed by (seed, group), so each level rounds
    up with probability its fractional part."""
    g = batch.level_group
    ok = g >= 0
    f = np.floor(x + 1e-9)
    frac = np.where(ok, np.clip(x - f, 0.0, 1.0), 0.0)
    order = np.lexsort((np.arange(len(x)), g))
    order = order[ok[order]]
    out = np.where(ok, f, 0.0)
    if len(order) == 0:
        return out
    go = g[order]
    cum = np.cumsum(frac[order])
    first = np.r_[True, go[1:] != go[:-1]]
    base = np.maximum.accumulate(np.where(first, cum - frac[order], -np.inf))
    cum = np.round(cum - base, 9)
    prev = np.round(cum - frac[order], 9)
    u = _group_uniforms(_key(seed, 3), batch.group_key)[go]
    up = np.floor(cum - u + 1.0) > np.floor(prev - u + 1.0)
    out[order] += up
    return out


def _group_uniforms(key, group_key):
    return np.array([_uniform(key, int(k), 0) for k in group_key])


def _key(seed, stream):
    # (numba returns a Python int: kept unsigned, as the kernels' hashing needs)
    return np.uint64(_mix(np.uint64((int(seed) * 0x9E3779B97F4A7C15 + stream) % 2 ** 64)))


def _shift(W, j):
    """V[c] = W[c + e_j] (-inf past the end)."""
    V = np.full(W.shape, -np.inf)
    src = [slice(None)] * W.ndim
    dst = [slice(None)] * W.ndim
    src[j] = slice(1, None)
    dst[j] = slice(0, -1)
    V[tuple(dst)] = W[tuple(src)]
    return V


# ------------------------------------------------------------------- the view
class View:
    """A batch's live rows (those whose alternative's target is positive,
    when exact), compactly: the arrays the samplers' kernels take.

    ls (n_sets + 1), llev, lu, crow: each set's first live row; each live
    row's level, utility and row in the batch. gstart, gsets: each group's
    sets. avail: each level's number of live rows. members, mstart, count,
    pos: each level's sets (in slots mstart[j] .. mstart[j] + count[j]) and
    each set's slot, for the samplers' bookkeeping. targets, total, levels:
    per level of the batch."""

    def __init__(self, batch, u, targets, exact):
        self.batch = batch
        live = batch.live & np.isfinite(u)
        if exact:
            live &= targets[batch.lev] > 0
        st0 = batch.starts[:-1]
        cnt = np.add.reduceat(live.astype(np.int64), st0)
        self.n_sets = len(st0)
        self.ls = np.r_[0, np.cumsum(cnt)].astype(np.int64)
        self.crow = np.flatnonzero(live).astype(np.int64)
        self.llev = batch.lev[self.crow]
        same = np.r_[False, self.llev[1:] == self.llev[:-1]]
        if np.any(same & (np.arange(len(self.llev)) != np.repeat(self.ls[:-1], cnt))):
            raise ValueError("constrained assignment: a choice set lists an alternative twice")
        self.lu = np.asarray(u, float)[self.crow]
        self.n_levels = len(batch.level_id)
        sg = np.where(cnt > 0, batch.set_group, -1)
        self.set_group = sg
        idx = np.flatnonzero(sg >= 0)
        self.gsets = idx[np.argsort(sg[idx], kind="stable")].astype(np.int64)
        self.gstart = np.r_[0, np.cumsum(np.bincount(sg[idx], minlength=batch.n_groups))] \
            .astype(np.int64)
        self.avail = np.bincount(self.llev, minlength=self.n_levels).astype(np.int64)
        self.mstart = np.r_[0, np.cumsum(self.avail)].astype(np.int64)
        self.members = np.full(self.mstart[-1], -1, np.int64)
        self.count = np.zeros(self.n_levels, np.int64)
        self.pos = np.full(self.n_sets, -1, np.int64)
        self.targets = np.asarray(targets, float)
        self.total = batch.total
        self.levels = np.flatnonzero(self.avail > 0)

    def group_sizes(self):
        return np.diff(self.gstart)

    def start(self, batch, cap, strict):
        """An assignment meeting the integer counts `cap` (greedy by
        utility, then augmenting paths); strict: an error where none
        exists, else the sets left over go to their best alternative."""
        cap = np.asarray(np.round(cap), np.int64)
        a = np.full(self.n_sets, -1, np.int64)
        left = _start(self.ls, self.llev, self.lu, cap, a, self.members, self.mstart,
                      self.count, self.pos, strict)
        if strict and left:
            bad = np.unique(self.set_group[a < 0])
            bad = bad[bad >= 0]
            raise ValueError(f"constrained assignment: the targets of {len(bad):,} groups "
                             f"cannot be met by their sets' alternatives (e.g. group "
                             f"{int(batch.group_key[bad[0]])}, by its first alternative's "
                             "row in the totals frame)")
        return a

    def penalty_table(self, penalty):
        """log g_j(n) for n = 0 .. avail_j, each level's in pen[poff[j]:]."""
        poff = np.r_[0, np.cumsum(self.avail + 1)].astype(np.int64)
        lv = np.repeat(np.arange(self.n_levels), self.avail + 1)
        n = (np.arange(poff[-1]) - poff[:-1][lv]).astype(float)
        pen = np.asarray(penalty(n, self.total[lv], self.batch.level_id[lv]), float)
        if pen.shape != n.shape:
            raise ValueError("the penalty must return an array of its arguments' shape")
        return np.where(np.isnan(pen), -np.inf, pen), poff

    def group(self, g):
        """Group g's arrays, with its levels coded 0 .. J - 1: n (sets),
        sets (theirs in the view), ls, llev, lu, rows (its rows in the
        view: a draw of row r is ``rows[r]``), targets, total, avail,
        level_id, key."""
        sets = self.gsets[self.gstart[g]:self.gstart[g + 1]]
        lens = self.ls[sets + 1] - self.ls[sets]
        ls = np.r_[0, np.cumsum(lens)].astype(np.int64)
        rows = np.repeat(self.ls[sets] - ls[:-1], lens) + np.arange(ls[-1])
        levels, llev = np.unique(self.llev[rows], return_inverse=True)
        return SimpleNamespace(n=len(sets), sets=sets, ls=ls, llev=llev.astype(np.int64),
                               lu=self.lu[rows], rows=rows.astype(np.int64),
                               targets=self.targets[levels], total=self.total[levels],
                               avail=self.avail[levels], level_id=self.batch.level_id[levels],
                               key=int(self.batch.group_key[g]))


# ------------------------------------------------------------------ the kernels
@nb.njit(cache=True)
def _find(llev, lo, hi, j):
    """The row of level j among rows lo .. hi - 1 (sorted by level), or -1."""
    while lo < hi:
        m = (lo + hi) >> 1
        if llev[m] < j:
            lo = m + 1
        else:
            hi = m
    return lo if lo < len(llev) and llev[lo] == j else -1


@nb.njit(cache=True, inline="always")
def _place(i, r, llev, a, members, mstart, count, pos):
    j = llev[r]
    a[i] = r
    s = mstart[j] + count[j]
    members[s] = i
    pos[i] = s
    count[j] += 1


@nb.njit(cache=True, inline="always")
def _remove(i, llev, a, members, mstart, count, pos):
    j = llev[a[i]]
    last = mstart[j] + count[j] - 1
    s = pos[i]
    k = members[last]
    members[s] = k
    pos[k] = s
    members[last] = -1
    count[j] -= 1
    a[i] = -1
    pos[i] = -1


@nb.njit(cache=True)
def _start(ls, llev, lu, cap, a, members, mstart, count, pos, strict):
    """Greedy (each set its best alternative with room), then an augmenting
    path for each set left (breadth first over the alternatives: a full
    one passes a set of its own on to another alternative of that set).
    Returns the number of sets left without a place; not strict, they then
    go to their best alternative regardless of room."""
    n = len(ls) - 1
    L = len(cap)
    for i in range(n):
        best = -1
        for r in range(ls[i], ls[i + 1]):
            if count[llev[r]] < cap[llev[r]] and (best < 0 or lu[r] > lu[best]):
                best = r
        if best >= 0:
            _place(i, best, llev, a, members, mstart, count, pos)
    seen = np.zeros(L, np.bool_)
    par_set = np.full(L, -1, np.int64)
    par_row = np.full(L, -1, np.int64)
    queue = np.empty(L, np.int64)
    left = 0
    for i in range(n):
        if a[i] >= 0 or ls[i + 1] == ls[i]:
            continue
        qh = 0
        qt = 0
        for r in range(ls[i], ls[i + 1]):
            j = llev[r]
            if not seen[j]:
                seen[j] = True
                par_set[j] = i
                par_row[j] = r
                queue[qt] = j
                qt += 1
        found = -1
        while qh < qt:
            j = queue[qh]
            qh += 1
            if count[j] < cap[j]:
                found = j
                break
            for s in range(mstart[j], mstart[j] + count[j]):
                k = members[s]
                for r in range(ls[k], ls[k + 1]):
                    j2 = llev[r]
                    if not seen[j2]:
                        seen[j2] = True
                        par_set[j2] = k
                        par_row[j2] = r
                        queue[qt] = j2
                        qt += 1
        if found >= 0:
            j = found
            while True:
                k = par_set[j]
                r = par_row[j]
                old = llev[a[k]] if a[k] >= 0 else -1
                if a[k] >= 0:
                    _remove(k, llev, a, members, mstart, count, pos)
                _place(k, r, llev, a, members, mstart, count, pos)
                if old < 0:
                    break
                j = old
        else:
            left += 1
        for t in range(qt):
            seen[queue[t]] = False
    if not strict and left:
        for i in range(n):
            if a[i] < 0 and ls[i + 1] > ls[i]:
                best = ls[i]
                for r in range(ls[i], ls[i + 1]):
                    if lu[r] > lu[best]:
                        best = r
                _place(i, best, llev, a, members, mstart, count, pos)
    return left


@nb.njit(cache=True, parallel=True)
def _mcmc(gstart, gsets, ls, llev, lu, a, members, mstart, count, pos, soft, pen, poff,
          n_moves, p_swap, key, gkey, acc):
    """Each group's chain, in parallel: `n_moves[g]` proposals. A swap:
    set i (uniform) proposes another of its alternatives b (uniform), and
    a set k at b (uniform) takes i's a, if it has a. The same swap is also
    proposed from k (proposing a, then i among a's sets), so its
    probability is [1/((d_i - 1) n_b) + 1/((d_k - 1) n_a)] / n, and back
    [1/((d_i - 1) n_a) + 1/((d_k - 1) n_b)] / n (the counts unchanged):
    the Hastings ratio [(d_k - 1) n_b + (d_i - 1) n_a] / [(d_k - 1) n_a +
    (d_i - 1) n_b], 1 when the two sets have as many alternatives. A move
    (soft): i to b (symmetric), the counts' penalty in the ratio. acc[g]
    = swaps proposed, accepted, moves proposed, accepted."""
    for g in nb.prange(len(gstart) - 1):
        n = gstart[g + 1] - gstart[g]
        if n == 0:
            continue
        c = 0
        for _ in range(n_moves[g]):
            i = gsets[gstart[g] + min(int(_uniform(key, gkey[g], c) * n), n - 1)]
            c += 1
            d = ls[i + 1] - ls[i]
            if d < 2:
                c += 3
                continue
            ra = a[i]
            ja = llev[ra]
            t = min(int(_uniform(key, gkey[g], c) * (d - 1)), d - 2)
            c += 1
            rb = ls[i] + t
            if rb >= ra:
                rb += 1
            jb = llev[rb]
            swap = (not soft) or _uniform(key, gkey[g], c) < p_swap
            c += 1
            if swap:
                acc[g, 0] += 1
                nb_ = count[jb]
                if nb_ == 0:
                    c += 1
                    continue
                k = members[mstart[jb] + min(int(_uniform(key, gkey[g], c) * nb_), nb_ - 1)]
                c += 1
                rka = _find(llev, ls[k], ls[k + 1], ja)
                if rka < 0:
                    c += 1
                    continue
                dk = ls[k + 1] - ls[k]
                na = count[ja]
                lr = (lu[rb] + lu[rka] - lu[ra] - lu[a[k]]
                      + np.log((dk - 1) * nb_ + (d - 1) * na)
                      - np.log((dk - 1) * na + (d - 1) * nb_))
                if lr >= 0 or _uniform(key, gkey[g], c) < np.exp(lr):
                    acc[g, 1] += 1
                    # i takes k's slot at b, k takes i's slot at a
                    si = pos[i]
                    sk = pos[k]
                    members[si] = k
                    members[sk] = i
                    pos[i] = sk
                    pos[k] = si
                    a[i] = rb
                    a[k] = rka
                c += 1
            else:
                acc[g, 2] += 1
                na = count[ja]
                nb_ = count[jb]
                lr = (lu[rb] - lu[ra] + pen[poff[jb] + nb_ + 1] - pen[poff[jb] + nb_]
                      + pen[poff[ja] + na - 1] - pen[poff[ja] + na])
                if lr >= 0 or _uniform(key, gkey[g], c) < np.exp(lr):
                    acc[g, 3] += 1
                    _remove(i, llev, a, members, mstart, count, pos)
                    _place(i, rb, llev, a, members, mstart, count, pos)
                c += 1


@nb.njit(cache=True)
def _next_frac(node_is_set, node, exclude, ls, order, astart, x, ptr_s, ptr_l):
    """A fractional cell of `node` other than `exclude`, or -1; the node's
    pointer skips the integral cells at the front of its list."""
    if node_is_set:
        lo, hi = ls[node], ls[node + 1]
        p = ptr_s[node]
        while p < hi - lo and not (EPS < x[lo + p] < 1 - EPS):
            p += 1
        ptr_s[node] = p
        for q in range(lo + p, hi):
            if q != exclude and EPS < x[q] < 1 - EPS:
                return q
    else:
        lo, hi = astart[node], astart[node + 1]
        p = ptr_l[node]
        while p < hi - lo and not (EPS < x[order[lo + p]] < 1 - EPS):
            p += 1
        ptr_l[node] = p
        for q in range(lo + p, hi):
            r = order[q]
            if r != exclude and EPS < x[r] < 1 - EPS:
                return r
    return -1


@nb.njit(cache=True, parallel=True)
def _round(gstart, gsets, ls, llev, order, astart, x, a, key, gkey, snaps, nrows):
    """Dependent rounding of each group's table, in parallel: from a set
    with a fractional cell, a walk set -> alternative -> set ... over
    fractional cells until it meets itself; around that (even) cycle, +-
    alternately, the largest steps l1 (+) and l2 (-) that keep the cells
    in [0, 1]; + l1 with probability l2 / (l1 + l2), else - l2. Each set's
    and alternative's sum is unchanged and each cell's expectation too.
    A cell left alone at a node by rounding error is snapped (counted)."""
    n_all = len(ls) - 1
    L = len(astart) - 1
    ptr_s = np.zeros(n_all, np.int64)
    ptr_l = np.zeros(L, np.int64)
    on_s = np.full(n_all, -1, np.int64)
    on_l = np.full(L, -1, np.int64)
    for g in nb.prange(len(gstart) - 1):
        n = gstart[g + 1] - gstart[g]
        if n == 0:
            continue
        cap = 2 * (n + min(L, nrows[g]) + 2)
        nodes = np.empty(cap, np.int64)          # >= 0 a set, < 0 an alternative (-1 - j)
        edges = np.empty(cap, np.int64)
        c = 0
        for t in range(gstart[g], gstart[g + 1]):
            s0 = gsets[t]
            while True:
                if _next_frac(True, s0, -1, ls, order, astart, x, ptr_s, ptr_l) < 0:
                    break
                depth = 0
                node = s0
                inc = -1
                cyc0 = -1
                while True:
                    is_set = node >= 0
                    r = _next_frac(is_set, node if is_set else -1 - node, inc, ls, order,
                                   astart, x, ptr_s, ptr_l)
                    if r < 0:                       # rounding error: snap the cell in
                        if inc >= 0:
                            x[inc] = 1.0 if x[inc] > 0.5 else 0.0
                            snaps[g] += 1
                        break
                    if is_set:
                        on_s[node] = depth
                    else:
                        on_l[-1 - node] = depth
                    nodes[depth] = node
                    edges[depth] = r
                    depth += 1
                    if is_set:
                        other = -1 - llev[r]
                        k = on_l[llev[r]]
                    else:
                        # the set of row r: binary search on ls
                        lo = 0
                        hi = n_all
                        while hi - lo > 1:
                            m = (lo + hi) >> 1
                            if ls[m] <= r:
                                lo = m
                            else:
                                hi = m
                        other = lo
                        k = on_s[lo]
                    if k >= 0:
                        cyc0 = k
                        break
                    node = other
                    inc = r
                if cyc0 >= 0:
                    l1 = np.inf
                    l2 = np.inf
                    for q in range(cyc0, depth):
                        v = x[edges[q]]
                        if (q - cyc0) % 2 == 0:
                            l1 = min(l1, 1 - v)
                            l2 = min(l2, v)
                        else:
                            l1 = min(l1, v)
                            l2 = min(l2, 1 - v)
                    up = _uniform(key, gkey[g], c) < l2 / (l1 + l2)
                    c += 1
                    step = l1 if up else -l2
                    for q in range(cyc0, depth):
                        e = edges[q]
                        x[e] += step if (q - cyc0) % 2 == 0 else -step
                        if x[e] < EPS:
                            x[e] = 0.0
                        elif x[e] > 1 - EPS:
                            x[e] = 1.0
                for q in range(depth):
                    if nodes[q] >= 0:
                        on_s[nodes[q]] = -1
                    else:
                        on_l[-1 - nodes[q]] = -1
        for t in range(gstart[g], gstart[g + 1]):
            i = gsets[t]
            best = ls[i]
            for r in range(ls[i], ls[i + 1]):
                if x[r] > x[best]:
                    best = r
            a[i] = best


# ------------------------------------------------------------------ the runs
def _run(lf, L, keep, case, sampler, seed, prefix, batch_rows, workdir):
    sampler = SwapMCMC() if sampler is None else sampler
    if not isinstance(sampler, ConstrainedSampler):
        raise TypeError("sampler= is a ConstrainedSampler (SwapMCMC, ExactSmall, Balanced, or "
                        "a subclass)")
    sampler.diagnostics = {}

    def per_batch(df):
        B = Batch(df, case, positive=True)
        if B.empty:
            return B.df.with_columns(pl.lit(None, pl.Boolean).alias("__a"))
        tgt = np.asarray(sampler.targets(B, seed), float)
        u = np.asarray(sampler.utilities(B, tgt), float)
        V = View(B, u, tgt, sampler.exact)
        a = np.asarray(sampler.sample(B, V, seed), np.int64)
        sampler._count(groups=B.n_groups, sets=int((a >= 0).sum()))
        if sampler.exact:
            got = np.bincount(V.llev[a[a >= 0]], minlength=V.n_levels)
            bad = int(np.abs(got - np.round(tgt))[V.levels].max(initial=0))
            if bad:
                raise RuntimeError(f"{type(sampler).__name__}: the counts missed their "
                                   f"targets (by up to {bad})")
        chosen = np.zeros(len(B.lev), bool)
        chosen[V.crow[a[a >= 0]]] = True
        has = np.repeat(a >= 0, np.diff(B.starts))
        col = pl.Series("__a", chosen).zip_with(
            pl.Series(has), pl.repeat(None, len(has), dtype=pl.Boolean, eager=True))
        return B.df.with_columns(col.alias("__a"))

    out = _stream(lf, L, case, per_batch, ["__a"], "assign", positive=True,
                  batch_rows=batch_rows, workdir=workdir)
    return out.select(*keep, pl.col("__a").alias(prefix))


def sample_assignments(data, totals, *, case, alt, utility, sampler=None, seed=0,
                       prefix="assigned", batch_rows=None, workdir=None) -> pl.LazyFrame:
    """Draw each choice set's alternative under constraints on the
    alternatives' counts, from utilities (experimental).

    Parameters
    ----------
    data : DataFrame, LazyFrame or path
        A row per (set, alternative): the `case` column(s), the `alt`
        column(s) and the `utility`.
    totals : DataFrame or LazyFrame
        The `alt` column(s) and one more, each alternative's total.
    sampler : ConstrainedSampler, optional
        Default ``SwapMCMC()`` (exact).
    seed : int
        Draws with the same seed are the same, whatever the batches.

    Returns
    -------
    polars.LazyFrame
        `data`'s rows, in order, with `prefix`: whether the row's
        alternative is its set's draw (null for a set with no alternative
        that can be chosen).
    """
    from hdfe_stream.core.utils import as_lazy
    _experimental()
    case = [case] if isinstance(case, str) else list(case)
    alt = [alt] if isinstance(alt, str) else list(alt)
    frame = as_lazy(data)
    keep = frame.collect_schema().names()
    lf = frame.with_row_index("__row").with_columns(pl.col(utility).cast(pl.Float64)
                                                    .alias("__xb"))
    lf, L = _with_totals(lf, alt, as_lazy(totals).collect(), "assign")
    return _run(lf, L, keep, case, sampler, seed, prefix, batch_rows, workdir)
