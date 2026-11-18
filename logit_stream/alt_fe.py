"""Absorbed alternative-specific constants for the conditional logit
(`clogit_stream("chosen ~ x | estab", ...)`): one constant per level of each
of D >= 1 fixed-effect dimensions, which vary across a choice set's
alternatives (an establishment, a product, a destination), in

    P(i chooses r) = exp(x_r'b + sum_d alpha^d[lev_d(r)]) / sum_s exp(...),

with s over choice set i. A characteristic of the chooser cancels from the
conditional logit; these do not.

**Structure.** The constants' information couples two levels exactly when
some choice set contains both,

    H_ab = sum_i w_i (sum_{r in i: a in r} p_r 1[b in r]
                      - (sum_{r: a in r} p_r)(sum_{s: b in s} p_s)),

so it is block-diagonal over the connected groups of levels that choice
sets link (union-find over each set's alternatives' levels, in one pass).
With establishments nested in firms and each worker choosing among their
firm's establishments, the groups are the firms. Nothing about firms is
given: the groups come from the data.

**Separation.** A level never chosen has a constant at -infinity: its rows
are dropped (the alternatives leave their choice sets). A level chosen in
every set it appears in has one at +infinity: its sets are dropped. Sets
left with fewer than two alternatives carry no information and are
dropped. The rules repeat until nothing changes. With one dimension they
are exact; with several they remove the common cases, and a remaining
direction of separation shows as a constant that diverges, with a warning.

**Two solvers** (`fe_solver`):

* ``"blocks"``: exact Newton in (b, all constants), each group's block of
  the information dense, the constants eliminated block by block (batched
  over groups of equal size). A block's null space -- adding a constant to
  every level of a group cancels, and with several dimensions so can other
  combinations -- is handled by its pseudo-inverse, so no reference level
  is needed during the fit. Standard errors from the same blocks.
* ``"contraction"``: no blocks. Given b, the constants solve each level's
  count-matching condition sum_i w_i p_r = sum_i w_i y_r (its score), by
  the contraction alpha_l += log(observed_l / predicted_l), dimension by
  dimension; b then takes a Newton step with the information net of the
  constants, H_bb - H_ba H_aa^-1 H_ab, H_aa^-1 H_ab by preconditioned
  conjugate gradients whose products H_aa v are passes over the data.
* ``"auto"`` (default): the blocks when the largest group has at most
  BLOCK_MAX levels and all the blocks fit the memory budget.

Both find the same maximum; they differ in passes and memory. The
reported constants of one dimension are normalized to 0 at each group's
reference level (its most-chosen); with several dimensions the constants
are reported as fitted (identified only up to the combinations that
cancel), and only their sums matter.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numba as nb
import numpy as np
import polars as pl
from hdfe_stream.core.report import warn
from hdfe_stream.core.resources import MemoryBudget

from .newton import _Driver, _Fit
from .store import CODE

BLOCK_MAX = 5_000           # "auto": the largest group of levels the blocks take
PINV_TOL = 1e-10            # a block's eigenvalues below this times its largest are 0
CONTRACT_TOL = 1e-10        # the contraction: max |log(observed / predicted)|
CG_TOL = 1e-10              # the conjugate gradients' relative residual
SEPARATION_ROUNDS = 50


# ------------------------------------------------------------------ setup kernels
@nb.njit(cache=True)
def _scan(lev, y, starts, level_alive, level_always, case_alive, seen, chosen):
    """One block's choice sets: drop a set (case_alive False) with fewer
    than two alternatives whose levels are all alive, without exactly one
    of them chosen, or containing a level chosen everywhere it appears;
    count each alive level's appearances and choices in the others. Returns
    the number of sets dropped."""
    n_sets = len(starts) - 1
    D = lev.shape[1]
    dropped = 0
    for c in range(n_sets):
        if not case_alive[c]:
            continue
        a, b = starts[c], starts[c + 1]
        nal, nch, kill = 0, 0, False
        for r in range(a, b):
            ok = True
            for d in range(D):
                if not level_alive[lev[r, d]]:
                    ok = False
            if ok:
                nal += 1
                if y[r] == 1:
                    nch += 1
                for d in range(D):
                    if level_always[lev[r, d]]:
                        kill = True
        if nal < 2 or nch != 1 or kill:
            case_alive[c] = False
            dropped += 1
            continue
        for r in range(a, b):
            ok = True
            for d in range(D):
                if not level_alive[lev[r, d]]:
                    ok = False
            if ok:
                for d in range(D):
                    seen[lev[r, d]] += 1
                    if y[r] == 1:
                        chosen[lev[r, d]] += 1
    return dropped


@nb.njit(cache=True)
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@nb.njit(cache=True)
def _link(lev, starts, row_alive, case_alive, parent):
    """Union-find: link the levels of each alive set's alive rows."""
    n_sets = len(starts) - 1
    D = lev.shape[1]
    for c in range(n_sets):
        if not case_alive[c]:
            continue
        root = -1
        for r in range(starts[c], starts[c + 1]):
            if not row_alive[r]:
                continue
            for d in range(D):
                x = _find(parent, lev[r, d])
                if root < 0:
                    root = x
                elif x != root:
                    parent[x] = root


# ------------------------------------------------------------------ the pass
@nb.njit(parallel=True, cache=True, error_model="numpy")
def _alt_pass(X, cols, beta, alpha, lev, y, w, off, starts, row_alive, set_comp, loc,
              comp_pos, comp_size, mode, ll_t, gb_t, Htt_t, ga, Hta, Haa, Hdiag):
    """The conditional logit's log-likelihood, gradient and information
    over a block's alive choice sets, with the constants. The sets are
    handed to threads by their group of levels (set_comp % threads), so
    that the levels' accumulators (ga, Hta, Haa, Hdiag), which belong to
    one group each, are written by one thread. mode 0: no information; 1:
    with the groups' dense blocks (Haa, packed: group c's at comp_pos[c],
    comp_size[c] square, levels at their local index `loc`); 2: with only
    the diagonal of the constants' block (Hdiag)."""
    T = len(ll_t)
    n_sets = len(starts) - 1
    P = len(cols)
    D = lev.shape[1]
    has_w, has_off = len(w) > 0, len(off) > 0
    max_size = 1
    for c in range(n_sets):
        if starts[c + 1] - starts[c] > max_size:
            max_size = starts[c + 1] - starts[c]
    for t in nb.prange(T):
        eta = np.empty(max_size)
        xs = np.empty((max_size, P))
        xbar = np.empty(P)
        dx = np.empty(P)
        g = np.zeros(P)
        H = np.zeros((P, P))
        llt = 0.0
        for c in range(n_sets):
            cc = set_comp[c]
            if cc < 0 or cc % T != t:
                continue
            a, b = starts[c], starts[c + 1]
            ws = w[a] if has_w else 1.0
            m = -np.inf
            chosen = 0.0
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                e = off[r] if has_off else 0.0
                for k in range(P):
                    x = X[r, cols[k]]
                    xs[r - a, k] = x
                    e += x * beta[k]
                for d in range(D):
                    e += alpha[lev[r, d]]
                eta[r - a] = e
                if y[r] == 1:
                    chosen += e
                if e > m:
                    m = e
            S = 0.0
            for r in range(a, b):
                if row_alive[r]:
                    eta[r - a] = np.exp(eta[r - a] - m)
                    S += eta[r - a]
            llt += ws * (chosen - m - np.log(S))
            for k in range(P):
                xbar[k] = 0.0
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                p = eta[r - a] / S
                eta[r - a] = p
                res = ws * ((1.0 if y[r] == 1 else 0.0) - p)
                for k in range(P):
                    g[k] += res * xs[r - a, k]
                    xbar[k] += p * xs[r - a, k]
                for d in range(D):
                    ga[lev[r, d]] += res
            if mode == 0:
                continue
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                pw = ws * eta[r - a]
                for k in range(P):
                    dx[k] = xs[r - a, k] - xbar[k]
                for k in range(P):
                    v = pw * dx[k]
                    for l2 in range(P):
                        H[k, l2] += v * dx[l2]
                for d in range(D):
                    lv = lev[r, d]
                    for k in range(P):
                        Hta[lv, k] += pw * dx[k]
            if mode == 2:
                for r in range(a, b):
                    if not row_alive[r]:
                        continue
                    p = eta[r - a]
                    for d in range(D):
                        Hdiag[lev[r, d]] += ws * p * (1.0 - p)
                continue
            pos, s = comp_pos[cc], comp_size[cc]
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                pr = ws * eta[r - a]
                for d in range(D):
                    ia = loc[lev[r, d]]
                    for e2 in range(D):
                        Haa[pos + ia * s + loc[lev[r, e2]]] += pr
                    for q in range(a, b):
                        if not row_alive[q]:
                            continue
                        v = pr * eta[q - a]
                        for e2 in range(D):
                            Haa[pos + ia * s + loc[lev[q, e2]]] -= v
        ll_t[t] = llt
        for k in range(P):
            gb_t[t, k] = g[k]
            for l2 in range(P):
                Htt_t[t, k, l2] = H[k, l2]


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _alt_matvec(X, cols, beta, alpha, lev, w, off, starts, row_alive, set_comp, V, out, T):
    """out = H_aa V for the constants' information (L x k columns V),
    without forming it: per set, with u_r = sum_d V[lev_d(r)] and
    ubar = sum_r p_r u_r, out[lev_d(r)] += w p_r (u_r - ubar). The sets are
    handed to the T threads by their group, as in `_alt_pass`."""
    n_sets = len(starts) - 1
    P = len(cols)
    D = lev.shape[1]
    kk = V.shape[1]
    has_w, has_off = len(w) > 0, len(off) > 0
    max_size = 1
    for c in range(n_sets):
        if starts[c + 1] - starts[c] > max_size:
            max_size = starts[c + 1] - starts[c]
    for t in nb.prange(T):
        eta = np.empty(max_size)
        u = np.empty((max_size, kk))
        ub = np.empty(kk)
        for c in range(n_sets):
            cc = set_comp[c]
            if cc < 0 or cc % T != t:
                continue
            a, b = starts[c], starts[c + 1]
            ws = w[a] if has_w else 1.0
            m = -np.inf
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                e = off[r] if has_off else 0.0
                for k in range(P):
                    e += X[r, cols[k]] * beta[k]
                for d in range(D):
                    e += alpha[lev[r, d]]
                eta[r - a] = e
                if e > m:
                    m = e
            S = 0.0
            for r in range(a, b):
                if row_alive[r]:
                    eta[r - a] = np.exp(eta[r - a] - m)
                    S += eta[r - a]
            for j in range(kk):
                ub[j] = 0.0
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                p = eta[r - a] / S
                eta[r - a] = p
                for j in range(kk):
                    x = 0.0
                    for d in range(D):
                        x += V[lev[r, d], j]
                    u[r - a, j] = x
                    ub[j] += p * x
            for r in range(a, b):
                if not row_alive[r]:
                    continue
                pw = ws * eta[r - a]
                for d in range(D):
                    lv = lev[r, d]
                    for j in range(kk):
                        out[lv, j] += pw * (u[r - a, j] - ub[j])


# ------------------------------------------------------------------ the class
class _AlternativeFE:
    """Absorbed alternative-specific constants of D >= 1 dimensions for the
    conditional logit; plays `_AbsorbedFE`'s role for the Newton driver
    (the "blocks" solver) or fits by its own loop ("contraction"). See the
    module docstring."""

    def __init__(self, model, p0, names, log, logger=None, budget=None, solver="auto"):
        if solver not in ("auto", "blocks", "contraction"):
            raise ValueError("fe_solver must be 'auto', 'blocks' or 'contraction'")
        self.model, self.p0, self.names = model, p0, list(names)
        self.store = p0.store
        self.log, self.logger = log, logger
        self.budget = budget if budget is not None else MemoryBudget()
        self.D = len(self.names)
        self.keys = ["fe"] + [f"fe{k + 1}" for k in range(1, self.D)]
        self.G = [p0.fe_tables[nm].height for nm in self.names]
        self.off = np.r_[0, np.cumsum(self.G)[:-1]].astype(np.int64)
        L = int(sum(self.G))
        self.L = L
        log("alternative constants: "
            + ", ".join(f"{nm} ({g:,})" for nm, g in zip(self.names, self.G)))
        # ---- separation, by repeated scans
        self.level_alive = np.ones(L, bool)
        always = np.zeros(L, bool)
        self.case_alive = np.ones(int(p0.n_cases), bool)
        n_drop = 0
        for _ in range(SEPARATION_ROUNDS):
            seen, chosen = np.zeros(L, np.int64), np.zeros(L, np.int64)
            for blk, c0 in self._blocks(["y"]):
                st = blk["starts"]
                n_drop += _scan(self._lev(blk), blk["y"].astype(np.int64), st,
                                self.level_alive, always, self.case_alive[c0:c0 + len(st) - 1],
                                seen, chosen)
            dead = self.level_alive & (chosen == 0)
            alw = self.level_alive & (seen > 0) & (chosen == seen) & ~always
            if not dead.any() and not alw.any():
                break
            self.level_alive &= ~dead
            always |= alw
        self.seen, self.chosen = seen, chosen
        self.level_alive &= seen > 0
        n_lev_dropped = int((~self.level_alive).sum())
        n_sets_dropped = int((~self.case_alive).sum())
        self.separation = {"checked": True, "levels": n_lev_dropped, "sets": n_sets_dropped}
        if n_sets_dropped or n_lev_dropped:
            warn(f"{n_lev_dropped:,} levels of the alternative constants dropped (never chosen, "
                 "or chosen wherever they appear: their constants are infinite), with their "
                 f"alternatives; and {n_sets_dropped:,} of {len(self.case_alive):,} choice sets "
                 "(left with fewer than two alternatives, or with one of those)", logger)
        if not self.case_alive.any():
            raise ValueError("no choice sets left after dropping the alternatives whose "
                             "constants are infinite")
        p0.n_cases = int(self.case_alive.sum())
        # ---- the groups of linked levels
        parent = np.arange(L, dtype=np.int64)
        for blk, c0 in self._blocks([]):
            st = blk["starts"]
            lev = self._lev(blk)
            _link(lev, st, self._row_alive(lev), self.case_alive[c0:c0 + len(st) - 1], parent)
        roots = _roots(parent)
        alive = np.flatnonzero(self.level_alive)
        _, comp_alive = np.unique(roots[alive], return_inverse=True)
        self.comp = np.full(L, -1, np.int64)
        self.comp[alive] = comp_alive
        C = int(comp_alive.max()) + 1 if len(alive) else 0
        self.C = C
        sizes = np.bincount(comp_alive, minlength=C).astype(np.int64)
        self.comp_size = sizes
        # each level's local index in its group, and the groups' levels in order
        order = alive[np.argsort(comp_alive, kind="stable")]
        self.level_order = order
        self.comp_start = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
        self.loc = np.full(L, -1, np.int64)
        self.loc[order] = np.arange(len(order)) - np.repeat(self.comp_start, sizes)
        self.comp_pos = np.r_[0, np.cumsum(sizes * sizes)[:-1]].astype(np.int64)
        block_gb = float((sizes.astype(float) ** 2).sum()) * 8 * 2 / 1e9
        biggest = int(sizes.max()) if C else 0
        log(f"  {C:,} groups of linked levels (largest {biggest:,}); the blocks would take "
            f"{block_gb:.2f} GB")
        if solver == "auto":
            solver = ("blocks" if biggest <= BLOCK_MAX and self.budget.fits(block_gb)
                      else "contraction")
            log(f"  fe_solver: {solver}")
        elif solver == "blocks" and not self.budget.fits(block_gb):
            raise MemoryError(f"fe_solver='blocks' needs about {block_gb:.1f} GB for the groups' "
                              f"blocks and {self.budget.headroom_gb():.1f} GB is free "
                              f"({self.budget}); use fe_solver='contraction'")
        self.solver = solver
        # the free constants: per group its rank, known after the first factorization;
        # until then the levels less one per group
        self.n_params = len(alive) - C
        self.method = solver

    # ------------------------------------------------------------ layout
    def _blocks(self, fields):
        """The store's blocks with the fixed-effect codes, and each block's
        first set's global index."""
        c0 = 0
        for blk in self.store.iterate(["starts", *self.keys, *fields]):
            n = len(blk["starts"]) - 1
            yield blk, c0
            c0 += n

    def _lev(self, blk):
        return np.column_stack([blk[k].astype(np.int64) + o
                                for k, o in zip(self.keys, self.off)])

    def _row_alive(self, lev):
        return self.level_alive[lev].all(axis=1)

    def _set_comp(self, lev, starts, row_alive, case_alive):
        """Each set's group (-1 for a dropped set)."""
        c = np.where(row_alive, self.comp[lev[:, 0]], -1)
        sc = np.maximum.reduceat(c, starts[:-1]) if len(starts) > 1 else np.zeros(0, np.int64)
        return np.where(case_alive, sc, -1).astype(np.int64)

    # ------------------------------------------------------------ parameters
    def start(self):
        """The coefficients at 0 and each level's constant at the log of its
        choices over its appearances (the null model's shares, roughly)."""
        a = np.zeros(self.L)
        ok = self.level_alive
        a[ok] = np.log(self.chosen[ok] / self.seen[ok])
        if self.D > 1:
            a[ok] /= self.D
        return np.r_[self.model.start(), a]

    def numba_pass(self):
        return True

    # ------------------------------------------------------------ passes
    def pass_(self, driver, theta, hessian, mode=None):
        """One read: the log-likelihood, gradient (b, then the L constants)
        and, with `hessian`, the information as a namespace (tt, ta, and
        the groups' blocks `aa` packed, or `diag` with mode 2)."""
        driver.passes += 1
        m = self.model
        P, L = m.P, self.L
        mode = (1 if self.solver == "blocks" else 2) if hessian and mode is None else \
            (0 if mode is None else mode)
        beta = np.ascontiguousarray(theta[:P], np.float64)
        alpha = np.ascontiguousarray(theta[P:], np.float64)
        T = nb.get_num_threads()
        ll, gb = 0.0, np.zeros(P)
        Htt = np.zeros((P, P))
        ga, Hta = np.zeros(L), np.zeros((L, P) if mode else (0, P))
        Haa = np.zeros(int((self.comp_size.astype(np.int64) ** 2).sum()) if mode == 1 else 0)
        Hdiag = np.zeros(L if mode == 2 else 0)
        empty = np.empty(0)
        cols = np.asarray(m.cols, np.int64)
        for blk, c0 in self._blocks(["X", "y", "w", "off"]):
            st = np.asarray(blk["starts"], np.int64)
            lev = self._lev(blk)
            ra = self._row_alive(lev)
            sc = self._set_comp(lev, st, ra, self.case_alive[c0:c0 + len(st) - 1])
            X = np.ascontiguousarray(blk["X"], np.float64)      # (cols picks the model's)
            ll_t, gb_t, Htt_t = np.zeros(T), np.zeros((T, P)), np.zeros((T, P, P))
            w = blk.get("w")
            off = blk.get("off")
            _alt_pass(X, cols, beta, alpha, lev, blk["y"].astype(np.int64),
                      empty if w is None else np.ascontiguousarray(w, np.float64),
                      empty if off is None else np.ascontiguousarray(off, np.float64),
                      st, ra, sc, self.loc, self.comp_pos, self.comp_size, mode,
                      ll_t, gb_t, Htt_t, ga, Hta, Haa, Hdiag)
            ll += ll_t.sum()
            gb += gb_t.sum(axis=0)
            Htt += Htt_t.sum(axis=0)
        if not np.isfinite(ll):
            return -np.inf, None, None
        g = np.r_[gb, ga]
        if not hessian:
            return float(ll), g, None
        H = SimpleNamespace(tt=(Htt + Htt.T) / 2, ta=Hta, aa=Haa if mode == 1 else None,
                            diag=Hdiag if mode == 2 else None, red=None)
        return float(ll), g, H

    def matvec(self, driver, theta, V):
        """H_aa V for the constants' information, by one read."""
        driver.passes += 1
        m = self.model
        P = m.P
        beta = np.ascontiguousarray(theta[:P], np.float64)
        alpha = np.ascontiguousarray(theta[P:], np.float64)
        out = np.zeros((self.L, V.shape[1]))
        V = np.ascontiguousarray(V, np.float64)
        empty = np.empty(0)
        cols = np.asarray(m.cols, np.int64)
        for blk, c0 in self._blocks(["X", "w", "off"]):
            st = np.asarray(blk["starts"], np.int64)
            lev = self._lev(blk)
            ra = self._row_alive(lev)
            sc = self._set_comp(lev, st, ra, self.case_alive[c0:c0 + len(st) - 1])
            w, off = blk.get("w"), blk.get("off")
            _alt_matvec(np.ascontiguousarray(blk["X"], np.float64), cols, beta, alpha, lev,
                        empty if w is None else np.ascontiguousarray(w, np.float64),
                        empty if off is None else np.ascontiguousarray(off, np.float64),
                        st, ra, sc, V, out, nb.get_num_threads())
        return out

    # ------------------------------------------------------------ the blocks' algebra
    def _groups_by_size(self):
        """(size, the groups of that size) for each size."""
        if not hasattr(self, "_by_size"):
            sizes = self.comp_size
            o = np.argsort(sizes, kind="stable")
            ss = sizes[o]
            cut = np.flatnonzero(np.diff(ss)) + 1
            self._by_size = [(int(ss[a]), o[a:b]) for a, b in
                             zip(np.r_[0, cut], np.r_[cut, len(o)])]
        return self._by_size

    def _schur(self, H):
        """Each group's block's pseudo-inverse (packed like H.aa), its
        product with the cross terms (BD, L x P, by level), and the
        coefficients' information net of the constants."""
        if H.red is not None:
            return H
        P = H.tt.shape[0]
        Dinv = np.zeros_like(H.aa)
        BD = np.zeros((self.L, P))
        red = H.tt.copy()
        rank = 0
        order, start = self.level_order, self.comp_start
        for s, comps in self._groups_by_size():
            k = len(comps)
            idx = self.comp_pos[comps][:, None] + np.arange(s * s)[None, :]
            A = H.aa[idx].reshape(k, s, s)
            A = (A + np.swapaxes(A, 1, 2)) / 2
            ev, Q = np.linalg.eigh(A)
            cut = PINV_TOL * np.maximum(ev[:, -1:], 1e-300)
            inv = np.where(ev > cut, 1.0 / np.where(ev > cut, ev, 1.0), 0.0)
            rank += int((ev > cut).sum())
            Ai = (Q * inv[:, None, :]) @ np.swapaxes(Q, 1, 2)
            Dinv[idx] = Ai.reshape(k, -1)
            lv = order[start[comps][:, None] + np.arange(s)[None, :]]           # k x s
            B = H.ta[lv]                                                        # k x s x P
            BDk = Ai @ B
            BD[lv] = BDk
            red -= np.einsum("ksp,ksq->pq", B, BDk)
        H.Dinv, H.BD, H.red, H.rank = Dinv, BD, (red + red.T) / 2, rank
        self.n_params = rank
        return H

    def _apply_dinv(self, H, v):
        """Dinv v, group by group (v and the result by level)."""
        out = np.zeros_like(v)
        order, start = self.level_order, self.comp_start
        for s, comps in self._groups_by_size():
            k = len(comps)
            idx = self.comp_pos[comps][:, None] + np.arange(s * s)[None, :]
            Ai = H.Dinv[idx].reshape(k, s, s)
            lv = order[start[comps][:, None] + np.arange(s)[None, :]]
            out[lv] = (Ai @ v[lv][..., None])[..., 0]
        return out

    def reduced(self, H):
        if self.solver != "blocks":
            return H.red
        return self._schur(H).red

    def solve(self, H, g):
        H = self._schur(H)
        P = H.tt.shape[0]
        gt, ga = g[:P], g[P:]
        gred = gt - H.BD.T @ ga
        dt, pd = _Driver.solve(H.red, gred)
        da = self._apply_dinv(H, ga - H.ta @ dt)
        return np.r_[dt, da], pd

    def vcov_beta(self, H):
        return _Driver.inverse(self.reduced(H))

    def sandwich(self, H, M):
        V = self.vcov_beta(H)
        out = V @ M @ V
        return (out + out.T) / 2

    def penalize(self, H, add):
        return SimpleNamespace(tt=H.tt + np.diag(add), ta=H.ta, aa=H.aa, diag=H.diag, red=None)

    def damp(self, H, lam):
        from .newton import _floored
        dt = _floored(np.abs(np.diag(H.tt)))
        aa = H.aa.copy()
        for s, comps in self._groups_by_size():
            di = self.comp_pos[comps][:, None] + (np.arange(s) * (s + 1))[None, :]
            aa[di] += lam * _floored(np.abs(aa[di]))
        return SimpleNamespace(tt=H.tt + lam * np.diag(dt), ta=H.ta, aa=aa, diag=None,
                               red=None)

    def subset(self, theta, g, H, idx):
        P_old = len(theta) - self.L
        keep = lambda v: np.r_[v[idx], v[P_old:]]           # noqa: E731
        H2 = SimpleNamespace(tt=H.tt[np.ix_(idx, idx)], ta=H.ta[:, idx], aa=H.aa,
                             diag=H.diag, red=None)
        return keep(theta), keep(g), H2

    def meat(self, driver, theta, H, weights_type, clusters, hetero):
        """The robust variances' middle matrices from each choice set's
        score for b net of the constants (its influence on b-hat): with
        B = H_aa^-1 H_ab (the blocks' BD, or the contraction's conjugate
        gradients' solution), sum_r (y_r - p_r) (x_r - sum_d B[lev_d(r)])."""
        if self.solver == "blocks":
            H = self._schur(H)
        BD = H.BD
        m = self.model
        P = m.P
        beta, alpha = theta[:P], theta[P:]
        Mt = np.zeros((P, P)) if hetero else None
        U = {name: np.zeros((Gc, P)) for name, (_, Gc) in clusters.items()}
        for blk, c0 in self._blocks(["X", "y", "w", "off", "cl"]):
            st = np.asarray(blk["starts"], np.int64)
            n_sets = len(st) - 1
            if not n_sets:
                continue
            lev = self._lev(blk)
            ca = self.case_alive[c0:c0 + n_sets]
            ra = self._row_alive(lev) & np.repeat(ca, np.diff(st))
            X = np.asarray(blk["X"], np.float64)[:, m.cols]
            eta = X @ beta + alpha[lev].sum(axis=1)
            if blk.get("off") is not None:
                eta = eta + blk["off"]
            eta = np.where(ra, eta, -np.inf)
            mx = np.maximum.reduceat(eta, st[:-1])
            mx = np.where(np.isfinite(mx), mx, 0.0)
            e = np.where(ra, np.exp(eta - np.repeat(mx, np.diff(st))), 0.0)
            S = np.add.reduceat(e, st[:-1])
            p = e / np.repeat(np.where(S > 0, S, 1.0), np.diff(st))
            r = np.where(ra, blk["y"] - p, 0.0)
            Xt = X - BD[lev].sum(axis=1)
            Sc = np.add.reduceat(r[:, None] * Xt, st[:-1], axis=0)[ca]
            w, cl = blk.get("w"), blk.get("cl")
            w = None if w is None else np.asarray(w)[st[:-1]][ca]
            cl = None if cl is None else np.asarray(cl)[st[:-1]][ca]
            driver._add_meat(Sc, w, cl, weights_type, Mt, U, clusters)
        return Mt, U

    # ------------------------------------------------------------ the contraction
    def _contract(self, driver, theta, max_rounds=10_000):
        """The constants given b, by count matching, dimension by dimension:
        alpha_l += log(chosen_l / predicted_l) on the weighted counts."""
        P = self.model.P
        theta = theta.copy()
        ok = self.level_alive
        target = None
        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            worst = 0.0
            for d in range(self.D):
                ll, g, _ = self.pass_(driver, theta, False)
                if target is None:
                    target = self._chosen_w(driver, theta, g)
                ga = g[P:]
                pred = target - ga
                sl = np.zeros(self.L, bool)
                sl[self.off[d]:self.off[d] + self.G[d]] = True
                sl &= ok
                step = np.log(target[sl] / np.maximum(pred[sl], 1e-300))
                theta[P:][sl] += step
                worst = max(worst, float(np.max(np.abs(step), initial=0.0)))
            if worst < CONTRACT_TOL:
                break
        return theta, rounds

    def _chosen_w(self, driver, theta, g):
        """Each level's weighted choices: its score plus its predicted
        choices (from one more read, at alpha shifted by nothing: the
        predicted are the score's complement at theta)."""
        # predicted = sum_i w p: computed by a pass at theta with y ignored is the
        # score's difference; the weighted chosen counts are fixed, so read them once
        if not hasattr(self, "_wchosen"):
            wc = np.zeros(self.L)
            for blk, c0 in self._blocks(["y", "w"]):
                st = blk["starts"]
                lev = self._lev(blk)
                ra = self._row_alive(lev)
                ca = np.repeat(self.case_alive[c0:c0 + len(st) - 1], np.diff(st))
                sel = ra & ca & (blk["y"] == 1)
                w = blk.get("w")
                ww = np.ones(len(sel)) if w is None else w
                for d in range(self.D):
                    wc += np.bincount(lev[sel, d], weights=ww[sel], minlength=self.L)
            self._wchosen = wc
        return self._wchosen

    def _cg(self, driver, theta, Hdiag, Bt, tol=CG_TOL, maxiter=500):
        """Z = H_aa^+ B (B: L x k), by preconditioned conjugate gradients
        (block, each column its own), the products by passes. Levels not
        alive stay 0."""
        ok = self.level_alive
        Mi = np.where(ok & (Hdiag > 0), 1.0 / np.where(Hdiag > 0, Hdiag, 1.0), 0.0)[:, None]
        Z = np.zeros_like(Bt)
        R = Bt * ok[:, None]
        Zp = Mi * R
        Pd = Zp.copy()
        rz = (R * Zp).sum(axis=0)
        b0 = np.sqrt((R * R).sum(axis=0))
        steps = 0
        while steps < maxiter:
            steps += 1
            AP = self.matvec(driver, theta, Pd) * ok[:, None]
            pap = (Pd * AP).sum(axis=0)
            alpha = np.where(pap > 0, rz / np.where(pap > 0, pap, 1.0), 0.0)
            Z += Pd * alpha
            R -= AP * alpha
            res = np.sqrt((R * R).sum(axis=0))
            if np.all(res <= tol * np.maximum(b0, 1e-300)):
                break
            Zp = Mi * R
            rz_new = (R * Zp).sum(axis=0)
            beta = np.where(rz > 0, rz_new / np.where(rz > 0, rz, 1.0), 0.0)
            Pd = Zp + Pd * beta
            rz = rz_new
        return Z, steps

    def run_contraction(self, driver, theta, fml, halvings=30):
        """The fit by the contraction (see the module docstring): returns a
        `_Fit` whose H carries the coefficients' information net of the
        constants (`H.red`)."""
        t0 = time.time()
        P = self.model.P
        theta, rounds = self._contract(driver, theta)
        ll, g, H = self.pass_(driver, theta, True, mode=2)
        ll_start = ll
        path, converged, dec, it = [ll], False, np.inf, 0
        for it in range(1, driver.maxiter + 1):
            Z, n_cg = self._cg(driver, theta, H.diag, H.ta)
            red = H.tt - H.ta.T @ Z
            red = (red + red.T) / 2
            gb = g[:P]
            db, pd = _Driver.solve(red, gb)
            dec = float(gb @ db)
            driver.log(f"  contraction round {it}: log-likelihood {ll:.12g}, decrement "
                       f"{dec:.3e} ({rounds} contraction rounds, {n_cg} CG steps)")
            if dec < driver.tol:
                converged = True
                H.red, H.BD = red, Z
                break
            t = 1.0
            for _ in range(halvings):
                th = theta.copy()
                th[:P] += t * db
                # the constants follow b to first order, then the contraction
                th[P:] -= t * (Z @ db)
                th, rounds = self._contract(driver, th)
                ll_c, g_c, _ = self.pass_(driver, th, False)
                if np.isfinite(ll_c) and ll_c >= ll - 1e-10 * abs(ll):
                    break
                t /= 2
            else:
                warn(f"{fml}: the contraction's line search failed", self.logger)
                H.red, H.BD = red, Z
                break
            theta = th
            ll, g, H = self.pass_(driver, theta, True, mode=2)
            path.append(ll)
        if H.red is None:
            Z, _ = self._cg(driver, theta, H.diag, H.ta)
            H.red = (H.tt - H.ta.T @ Z + (H.tt - H.ta.T @ Z).T) / 2
            H.BD = Z
        return _Fit(theta=theta, ll=ll, g=g, H=H, ll_start=ll_start, iterations=it,
                    converged=converged, decrement=dec, halvings=0, path=path,
                    seconds=time.time() - t0)

    # ------------------------------------------------------------ results
    def references(self):
        """Each alive level's group's reference level (the group's most
        chosen); -1 for a level dropped."""
        ref = np.full(self.L, -1, np.int64)
        alive = np.flatnonzero(self.level_alive)
        if not len(alive):
            return ref
        comp = self.comp[alive]
        o = np.lexsort((-self.chosen[alive], comp))
        first = np.r_[True, comp[o][1:] != comp[o][:-1]]
        r = np.empty(self.C, np.int64)
        r[comp[o][first]] = alive[o][first]
        ref[alive] = r[comp]
        return ref

    def normalized(self, theta):
        """The constants as reported: with one dimension, 0 at each group's
        reference level (its most chosen); NaN for a level dropped."""
        P = self.model.P
        a = theta[P:].copy()
        ok = self.level_alive
        if self.D == 1 and self.C:
            ref = self.references()
            a[ok] -= a[ref[ok]]
        return np.where(ok, a, np.nan)

    def draw_info(self, theta, H):
        """What `draw_parameters` and the constants' standard errors need
        (one dimension, the blocks): the constants, their cross terms with
        b (BD), each group's pseudo-inverse and a square root of it, and
        the layout of the groups."""
        from .draws import DRAW_INFO_GB
        if self.D != 1 or self.solver != "blocks":
            return {"kind": None, "why": "with absorbed alternative constants it needs one "
                    "dimension and fe_solver='blocks'"}
        gb = (2 * H.aa.size + self.L * self.model.P) * 8 / 1e9
        if gb > DRAW_INFO_GB:
            return {"kind": None, "why": f"the constants' blocks would take {gb:.1f} GB (more "
                    f"than {DRAW_INFO_GB:g}); they were not kept"}
        H = self._schur(H)
        S = np.zeros_like(H.aa)
        for s, comps in self._groups_by_size():
            k = len(comps)
            idx = self.comp_pos[comps][:, None] + np.arange(s * s)[None, :]
            A = H.aa[idx].reshape(k, s, s)
            ev, Q = np.linalg.eigh((A + np.swapaxes(A, 1, 2)) / 2)
            cut = PINV_TOL * np.maximum(ev[:, -1:], 1e-300)
            r = np.where(ev > cut, 1.0 / np.sqrt(np.where(ev > cut, ev, 1.0)), 0.0)
            S[idx] = (Q * r[:, None, :]).reshape(k, -1)
        P = self.model.P
        return {"kind": "alt_fe", "alpha": np.asarray(theta[P:], float).copy(),
                "BD": H.BD.copy(), "Dinv": H.Dinv.copy(), "S": S, "ref": self.references(),
                "alive": self.level_alive.copy(), "loc": self.loc, "comp": self.comp,
                "comp_pos": self.comp_pos, "comp_size": self.comp_size,
                "level_order": self.level_order, "comp_start": self.comp_start,
                "by_size": self._groups_by_size(), "P": P}

    def result_spec(self, theta):
        a = self.normalized(theta)
        out = {"fixef": {}, "effects": []}
        for nm, g, o in zip(self.names, self.G, self.off):
            table = self.p0.fe_tables[nm]
            vals = pl.Series(f"fe_{nm}", a[o:o + g]).fill_nan(None)
            t = table.with_columns(vals, pl.Series("n_obs", self.seen[o:o + g]),
                                   pl.Series("n_chosen", self.chosen[o:o + g]))
            out["fixef"][nm] = t.drop(CODE)
            out["effects"].append((nm, [f"fe_{nm}"], "fixed"))
        return out


@nb.njit(cache=True)
def _roots(parent):
    out = np.empty(len(parent), np.int64)
    for a in range(len(parent)):
        out[a] = _find(parent, a)
    return out
