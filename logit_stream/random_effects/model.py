"""The random intercepts' model (`_RandomEffects`): the groupings and their
units, the data passes, the Laplace approximation's terms, the Newton
step through the effects' block structure, and inference; see the
package docstring."""

from __future__ import annotations

from types import SimpleNamespace

import numba as nb
import numpy as np
import polars as pl
from hdfe_stream.core.report import warn

from ..fe import _MAX_GROUP_GB, _group_sum, _mnl_group_info, _ordered_info_derivs, _segments
from ..models import _gram, _mnl_info_sparse
from ..newton import _chunks, _design, _Driver
from ..store import CODE
from .linalg import (
    _crossed_apply,
    _crossed_cov,
    _crossed_factor,
    _group_w,
    _sum_by,
    _tree_apply,
    _tree_cov,
    _tree_factor,
)
from .sigma import _SigmaMixin

DENSE_MAX = 6000            # effects of a crossed grouping's dense system, at most


class _Factor(SimpleNamespace):
    """A grouping: its name, store key, kind ("random" or "fixed"), number of
    groups, each unit's group (`of_unit`), the groups' free levels and
    fixed values, and, if random, Sigma."""


class _RandomEffects(_SigmaMixin):
    """Random intercepts at one or more groupings; see the module
    docstring. Plays the role of `_AbsorbedFE` for the Newton driver."""

    def __init__(self, model, p0, randoms, fixed, keys, log, logger=None, cov="unstructured"):
        self.model, self.p0, self.store, self.logger = model, p0, p0.store, logger
        self.cov = cov
        J = model.J
        self.J = J
        mlogit = model.kind == "mlogit"
        self.m = J if mlogit else 1
        self.lv = (np.arange(J) > 0) if mlogit else np.ones(1, bool)
        self.d = int(self.lv.sum())
        names = list(randoms) + ([fixed] if fixed else [])
        Gs = {nm: p0.fe_tables[nm].height for nm in names}
        # pass 1: how the groupings relate, and the crossed cells
        maps = {(a, b): np.full(Gs[a], -1, np.int64) for a in names for b in names if a != b}
        nested = {k: True for k in maps}
        cells = []
        for blk in self.store.iterate([keys[nm] for nm in names]):
            codes = {nm: blk[keys[nm]] for nm in names}
            for (a, b), mp in maps.items():
                if not nested[(a, b)]:
                    continue
                ca, cb = codes[a], codes[b]
                prev = mp[ca]
                o = np.argsort(ca, kind="stable")
                if (np.any((prev >= 0) & (prev != cb))
                        or np.any((ca[o][1:] == ca[o][:-1]) & (cb[o][1:] != cb[o][:-1]))):
                    nested[(a, b)] = False
                else:
                    mp[ca] = cb
            if len(randoms) == 2 and not fixed:
                cells.append(np.unique(codes[randoms[0]] * Gs[randoms[1]] + codes[randoms[1]]))
        # the structure
        chain = sorted(randoms, key=lambda nm: -Gs[nm])
        ok_chain = all(nested[(chain[i], chain[i + 1])] for i in range(len(chain) - 1))
        if fixed:
            ok_chain = ok_chain and nested[(chain[-1], fixed)]
        if ok_chain:
            self.kind = "tree"
            order = chain + ([fixed] if fixed else [])
            finest = order[0]
            U = Gs[finest]
            of_unit = {finest: np.arange(U, dtype=np.int64)}
            for nm in order[1:]:
                of_unit[nm] = np.maximum(maps[(finest, nm)], 0)
            self.unit_key = keys[finest]
        elif len(randoms) == 2 and not fixed:
            self.kind = "crossed"
            a, b = sorted(randoms, key=lambda nm: -Gs[nm])
            if Gs[b] * self.m > DENSE_MAX:
                raise ValueError(
                    f"random intercepts by {a} and {b}, which are crossed (neither nests in the "
                    f"other), need a dense system of {Gs[b] * self.m:,} effects for {b}; at most "
                    f"{DENSE_MAX:,} are supported (lme4 or glmmTMB fit larger crossed designs)")
            order = [a, b]
            keys_ab = np.unique(np.concatenate(cells)) if cells else np.zeros(0, np.int64)
            of_unit = {randoms[0]: keys_ab // Gs[randoms[1]], randoms[1]: keys_ab % Gs[randoms[1]]}
            self.cell_keys = keys_ab
            self.cell_mult = Gs[randoms[1]]
            self.cell_cols = (keys[randoms[0]], keys[randoms[1]])
            U = len(keys_ab)
        else:
            raise ValueError(
                "random intercepts need groupings that nest in one another (each group of a "
                "finer one within one group of the next), all within the fixed effect if any, "
                f"or exactly two crossed groupings without a fixed effect; got {list(randoms)}"
                + (f" with the fixed effect {fixed}" if fixed else ""))
        self.U = U
        # pass 2: outcome counts per unit
        ncnt, wcnt = np.zeros(U * J), np.zeros(U * J)
        for blk in self.store.iterate(["y", "w", *[keys[nm] for nm in names]]):
            u = self._units(blk)
            ncnt += np.bincount(u * J + blk["y"], minlength=U * J)
            wcnt += np.bincount(u * J + blk["y"], weights=blk.get("w"), minlength=U * J)
        self.ncnt_u, wcnt = ncnt.reshape(U, J), wcnt.reshape(U, J)
        self.keep_u = np.ones(U, bool)
        self.separation = {"checked": False}
        # the factors, finest first (tree) or larger first (crossed)
        self.factors = []
        for nm in order:
            f = _Factor(name=nm, key=keys[nm], kind="fixed" if nm == fixed else "random",
                        G=Gs[nm], of_unit=of_unit[nm])
            f.ncnt = _sum_by(f.of_unit, self.ncnt_u, f.G)
            self.factors.append(f)
        if fixed:
            fe = self.factors[-1]
            obs = fe.ncnt > 0
            nlev = obs.sum(axis=1)
            drop = nlev < 2 if mlogit else (nlev == 1) & (obs[:, 0] | obs[:, -1])
            fe.keep = ~drop
            self.keep_u = fe.keep[fe.of_unit]
            if mlogit:
                fe.ref = np.argmax(obs, axis=1)
                fe.free = obs & (np.arange(J)[None, :] != fe.ref[:, None]) & fe.keep[:, None]
                fe.fixed = np.where(obs, 0.0, -np.inf)
            else:
                fe.free = fe.keep[:, None].copy()
                fe.fixed = np.zeros((fe.G, 1))
                model.fix_first = True
            fw = _sum_by(fe.of_unit, wcnt, fe.G)
            n_drop = int(fe.ncnt[drop].sum())
            self.separation = {"checked": True, "levels": int(drop.sum()),
                               "observations": n_drop}
            if n_drop:
                warn(f"{n_drop:,} observations in {int(drop.sum()):,} levels of {fixed} dropped "
                     "because their outcome never varies (their effects would be infinite)",
                     logger)
                p0.n_obs -= n_drop
                p0.wsum -= float(fw[drop].sum())
                p0.counts = fw[fe.keep].sum(axis=0)
                if p0.n_obs <= 0 or np.any(p0.counts <= 0):
                    raise ValueError(f"too few observations left after dropping the levels of "
                                     f"{fixed} whose outcome never varies")
        else:
            model.fix_first = False
        for f in self.factors:
            if f.kind == "random":
                f.keep = _sum_by(f.of_unit, self.keep_u.astype(float), f.G) > 0
                f.free = self.lv[None, :] & f.keep[:, None]
                f.fixed = np.zeros((f.G, self.m))
                f.n_eff = int(f.keep.sum())
        self.random = [f for f in self.factors if f.kind == "random"]
        self.fixed_factor = self.factors[-1] if fixed else None
        n_cov = self.d if cov == "diagonal" else self.d * (self.d + 1) // 2
        self.n_cov = n_cov
        self.n_params = n_cov * len(self.random) + (int(self.fixed_factor.free.sum())
                                                    if fixed else 0)
        G_all = sum(f.G for f in self.factors)
        gb = (G_all * self.m * max(model.P, 1) * 16
              + (U * (self.m ** 2 + self.d ** 3) * 8 if self.kind == "crossed" else 0)) / 1e9
        if gb > _MAX_GROUP_GB:
            raise MemoryError(f"random intercepts need about {gb:.1f} GB for their blocks; fit "
                              "with fewer covariates, or coarser groupings")
        if self.kind == "tree":
            self._tree_setup()
        for f in self.random:
            log(f"random intercepts by {f.name}: {f.n_eff:,} groups")
        log("structure: " + ("nested: " + " in ".join(f.name for f in self.factors)
                             if self.kind == "tree" else
                             f"crossed: {self.factors[0].name} x {self.factors[1].name}"))
        # (relative to the coefficients' end: collinear columns can be dropped
        # after the first step, which shortens them)
        self.offsets = []
        o = 0
        for f in self.factors:
            self.offsets.append(o)
            o += f.G * self.m
        self.method = "joint-mode"
        self._cache = None
        self._Vu = None
        self.set_covs([self._start_cov(f) for f in self.random])

    # ------------------------------------------------------------ structure
    def _units(self, blk):
        if self.kind == "tree":
            return blk[self.unit_key]
        a, b = self.cell_cols
        return np.searchsorted(self.cell_keys, blk[a] * self.cell_mult + blk[b])

    def _tree_setup(self):
        """Each level's ancestors at the levels above (from any of its units)."""
        for k, f in enumerate(self.factors):
            f.anc = {}
            rep = np.zeros(f.G, np.int64)
            rep[f.of_unit] = np.arange(self.U)
            for j in range(k + 1, len(self.factors)):
                f.anc[j] = self.factors[j].of_unit[rep]

    # ------------------------------------------------------------ Sigma
    def _parent_logodds(self, f):
        """The group's smoothed log-odds (multinomial: against the base;
        ordered: the logit of its mean score's share), less those of its
        parent (the next coarser grouping, or the sample) for a start."""
        J = self.J

        def lo(cnt):
            if self.model.kind == "mlogit":
                return np.log(cnt[:, 1:] + 0.5) - np.log(cnt[:, :1] + 0.5)
            sh = (cnt @ np.arange(J) + 0.5) / (cnt.sum(axis=1) * (J - 1) + 1.0)
            return np.log(sh / (1 - sh))[:, None]

        own = lo(f.ncnt)
        k = self.factors.index(f)
        if self.kind == "tree" and k + 1 < len(self.factors):
            par = self.factors[k + 1]
            return own - lo(par.ncnt)[f.anc[k + 1]]
        return own - lo(self.p0.counts[None, :])

    def _start_cov(self, f):
        lo = self._parent_logodds(f)[f.keep]
        S = np.atleast_2d(np.cov(lo, rowvar=False)) if len(lo) > 1 else np.eye(self.d)
        S = S + np.diag(np.maximum(0.05 - np.diag(S), 0.0))
        return np.diag(np.diag(S)) if self.cov == "diagonal" else S

    def _embed(self, S):
        Lam = np.zeros((self.m, self.m))
        Lam[np.ix_(self.lv, self.lv)] = np.linalg.inv(S)
        return Lam

    def set_covs(self, Ss):
        for f, S in zip(self.random, Ss):
            f.S = np.atleast_2d(np.asarray(S, dtype=float))
            f.Lam = self._embed(f.S)

    @property
    def covs(self):
        return [f.S for f in self.random]

    @staticmethod
    def cov_change(new, old):
        return max(float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))
                   for a, b in zip(new, old))

    # ------------------------------------------------------------ parameters
    def _theta_of(self, theta, f):
        o = self.model.P + self.offsets[self.factors.index(f)]
        A = theta[o:o + f.G * self.m].reshape(f.G, self.m)
        return np.where(f.free, A, f.fixed)

    def _unit_effects(self, theta):
        A = np.zeros((self.U, self.m))
        for f in self.factors:
            A = A + self._theta_of(theta, f)[f.of_unit]
        return A

    def start(self):
        m = self.model
        th = m.start(self.p0.counts, fe=self.fixed_factor is not None)
        parts = [th]
        for f in self.factors:
            if f.kind == "random":
                n = f.ncnt.sum(axis=1)
                A = np.zeros((f.G, self.m))
                A[:, self.lv] = self._parent_logodds(f) * (n / (n + 10.0))[:, None]
                parts.append(np.where(f.free, A, 0.0).ravel())
            else:
                if m.kind == "mlogit":
                    with np.errstate(divide="ignore", invalid="ignore"):
                        lc = (np.log(f.ncnt)
                              - np.log(f.ncnt[np.arange(f.G), f.ref])[:, None])
                    A = np.where(f.free, lc, 0.0)
                else:
                    A = np.full((f.G, 1), -m.marginal_cuts(self.p0.counts)[0])
                parts.append(A.ravel())
        return np.concatenate(parts)

    # ------------------------------------------------------------ passes
    def _data_pass(self, driver, theta, extra):
        """One read: the log-likelihood, b's gradient and information, and
        each unit's gradient and information for its linear predictor (all
        m levels) and their information with b (per unit when the groupings
        nest; per group of each grouping when they cross). With `extra`,
        the exact Laplace approximation's terms, from the units' total
        effects' covariances `self._Vu`."""
        driver.passes += 1
        if driver.device is not None:
            from .. import device
            return device.re_pass(driver.device.xp, self, driver, theta, extra)
        m = self.model
        P, U, M = m.P, self.U, self.m
        th = theta[:P]
        Au = self._unit_effects(theta)
        D = SimpleNamespace(theta=theta.copy(), ll=0.0, gt=np.zeros(P), Htt=np.zeros((P, P)),
                            ga=np.zeros((U, M)), Haa=np.zeros((U, M, M)), extra=extra,
                            Dt=None, T=None, Vu=None)
        if self.kind == "tree":
            D.B = np.zeros((U, M, P))
        else:
            D.Bf = [np.zeros((f.G, M, P)) for f in self.factors]
            scratch = [np.zeros((f.G, M, M)) for f in self.factors]
        if extra:
            d = self.d
            D.Dt, D.T, D.Vu = np.zeros(P), np.zeros((U, d, d, d)), self._Vu
        rows = driver._rows()
        for blk in self.store:
            for ch in _chunks(blk, rows):
                u = self._units(ch)
                keep = self.keep_u[u]
                if not keep.all():
                    ch = {k: (v[keep] if k != "starts" else v) for k, v in ch.items()}
                    u = u[keep]
                if not len(ch["y"]):
                    continue
                X = _design(ch, m.cols, driver.K0)
                y, w = ch["y"], ch.get("w")
                ww = np.ones(len(y)) if w is None else w
                Ar = Au[u]
                if ch.get("off") is not None:       # the offset (ordered logit) joins the effects
                    Ar = Ar + ch["off"][:, None]
                if m.kind == "mlogit":
                    l, Pr = self._mnl_rows(th, Ar, X, y, w, D)
                    if not np.isfinite(l):
                        D.ll = -np.inf
                        return D
                    order, seg, segg = _segments(u)
                    Xc = np.ascontiguousarray(X)
                    if self.kind == "tree":
                        _mnl_group_info(order, seg, segg, Xc, Pr, ww, D.Haa, D.B)
                    else:
                        _group_w(order, seg, segg, Pr, ww, D.Haa)
                        for f, Bf, Sc in zip(self.factors, D.Bf, scratch):
                            o2, s2, g2 = _segments(f.of_unit[u])
                            _mnl_group_info(o2, s2, g2, Xc, Pr, ww, Sc, Bf)
                    R = -Pr
                    R[np.arange(len(y)), y] += 1.0
                    D.ga += _group_sum(u, R * ww[:, None], U)
                else:
                    r = m.rows(th, X, y, w, offset=Ar[:, 0], hessian=True)
                    l, g, H = m._assemble(r, X, y, w, True)
                    if not np.isfinite(l):
                        D.ll = -np.inf
                        return D
                    D.gt += g
                    D.Htt += H
                    D.ga[:, 0] += np.bincount(u, weights=r["s"], minlength=U)
                    D.Haa[:, 0, 0] += np.bincount(u, weights=r["h"], minlength=U)
                    Q = r["Q"][:, 1:] if m.fix_first else r["Q"]
                    Bv = np.column_stack([X * r["h"][:, None], Q])
                    if self.kind == "tree":
                        D.B[:, 0, :] += _group_sum(u, Bv, U)
                    else:
                        for f, Bf in zip(self.factors, D.Bf):
                            Bf[:, 0, :] += _group_sum(f.of_unit[u], Bv, f.G)
                D.ll += l
                if extra:
                    self._laplace_rows(D, th, Ar, X, y, ww, u)
        return D

    def _mnl_rows(self, th, Arow, X, y, w, D):
        """The multinomial rows' log-likelihood, b's gradient and information
        (as `_AbsorbedFE._mnl_chunk`), returning the probabilities."""
        n, K, J = len(y), X.shape[1], self.J
        E = np.empty((n, J))
        E[:, 0] = 0.0
        E[:, 1:] = X @ th.reshape(J - 1, K).T
        E += Arow
        mx = E.max(axis=1)
        ex = np.exp(E - mx[:, None])
        den = ex.sum(axis=1)
        Pr = ex / den[:, None]
        r = np.arange(n)
        lp = E[r, y] - mx - np.log(den)
        ll = float(lp.sum() if w is None else lp @ w)
        if not np.isfinite(ll):
            return ll, Pr
        R = -Pr
        R[r, y] += 1.0
        if w is not None:
            R = R * w[:, None]
        D.gt += (X.T @ R[:, 1:]).T.ravel()
        model = self.model
        if model.sparse(X):
            H = np.zeros_like(D.Htt)
            _mnl_info_sparse(np.ascontiguousarray(X), np.ascontiguousarray(Pr[:, 1:]),
                             np.ones(n) if w is None else w, nb.get_num_threads(), H)
            for j in range(J - 1):
                for k in range(j + 1, J - 1):
                    H[k * K:(k + 1) * K, j * K:(j + 1) * K] = \
                        H[j * K:(j + 1) * K, k * K:(k + 1) * K].T
            D.Htt += H
        else:
            Pw = Pr if w is None else Pr * w[:, None]
            for j in range(1, J):
                sj = slice((j - 1) * K, j * K)
                D.Htt[sj, sj] += _gram(X, Pw[:, j] * ((den - ex[:, j]) / den))
                for k in range(j + 1, J):
                    sk = slice((k - 1) * K, k * K)
                    Hjk = _gram(X, Pw[:, j] * Pr[:, k])
                    D.Htt[sj, sk] -= Hjk
                    D.Htt[sk, sj] -= Hjk
        return ll, Pr

    def _laplace_rows(self, D, th, Arow, X, y, ww, u, sub=20_000):
        """c_i = d log|M| / d eta_i, from the row's unit's total effect's
        covariance V_u, into sum_i c_i (x) x_i; and each unit's derivatives
        of its information in its linear predictor (d x d x d)."""
        m, d = self.model, self.d
        lv = self.lv
        Vall = D.Vu[:, lv][:, :, lv]
        if m.kind == "mlogit":
            J = self.J
            n = len(y)
            E = np.empty((n, J))
            E[:, 0] = 0.0
            E[:, 1:] = X @ th.reshape(J - 1, -1).T
            E += Arow
            ex = np.exp(E - E.max(axis=1)[:, None])
            p_all = (ex / ex.sum(axis=1)[:, None])[:, 1:]
            eye, idx = np.eye(d), np.arange(d)
            for a in range(0, n, sub):
                sl = slice(a, a + sub)
                p, us, ws = p_all[sl], u[sl], ww[sl]
                Dm = -p[:, :, None] * p[:, None, :]
                Dm[:, idx, idx] += p
                V = Vall[us]
                v = np.diagonal(V, axis1=1, axis2=2) - 2 * np.einsum("nij,nj->ni", V, p)
                c = np.einsum("nij,nj->ni", Dm, v) * ws[:, None]
                D.Dt += (X[sl].T @ c).T.ravel()
                Tr = (eye[None, :, :, None] * Dm[:, :, None, :]
                      - Dm[:, :, None, :] * p[:, None, :, None]
                      - p[:, :, None, None] * Dm[:, None, :, :]) * ws[:, None, None, None]
                D.T += _group_sum(us, Tr.reshape(len(us), -1), self.U).reshape(self.U, d, d, d)
        else:
            cuts = m.cuts(th)
            eta = X @ th[:m.K] + Arow[:, 0]
            _, dwe, dwu, dwl = _ordered_info_derivs(cuts, eta, y)
            vi = Vall[u, 0, 0]
            c = vi * dwe * ww
            C = len(cuts)
            dcut = np.zeros(C)
            up, lo = y < C, y >= 1
            np.add.at(dcut, y[up], (vi * dwu * ww)[up])
            np.add.at(dcut, y[lo] - 1, (vi * dwl * ww)[lo])
            if m.fix_first:
                dcut = dcut[1:]
            D.Dt += np.r_[X.T @ c, dcut]
            D.T[:, 0, 0, 0] += np.bincount(u, weights=dwe * ww, minlength=self.U)

    # ------------------------------------------------------------ blocks
    def _blocks(self, Haa, ga, D, Lams, with_beta=True, random_only=False):
        """Each factor's blocks from the units': D (penalized if random; the
        identity on levels that are not free), g, B, and the information
        with the other groupings (the tree's ancestors, or the crossed
        units')."""
        facs = self.random if random_only else self.factors
        levels = []
        for f in facs:
            fr = f.free
            both = fr[:, :, None] & fr[:, None, :]
            Dd = _sum_by(f.of_unit, Haa, f.G)
            Df = Dd * both
            if f.kind == "random":
                Df = Df + Lams[self.random.index(f)][None] * both
            gi, li = np.nonzero(~fr)
            Df[gi, li, li] = 1.0
            g = _sum_by(f.of_unit, ga, f.G) * fr
            lev = SimpleNamespace(f=f, D=Df, Dd=Dd, g=g, B=None, X={}, anc={})
            if with_beta:
                Bsrc = (_sum_by(f.of_unit, D.B, f.G) if self.kind == "tree"
                        else D.Bf[self.factors.index(f)])
                lev.B = Bsrc * fr[:, :, None]
            levels.append(lev)
        if self.kind == "tree":
            for k, lev in enumerate(levels):
                for j in range(k + 1, len(levels)):
                    a = lev.f.anc[self.factors.index(levels[j].f)]
                    lev.anc[j] = a
                    lev.X[j] = lev.Dd * (lev.f.free[:, :, None] & levels[j].f.free[a][:, None, :])
        return levels

    def _crossed_X(self, Haa):
        fa, fb = self.factors
        return Haa * (fa.free[fa.of_unit][:, :, None] & fb.free[fb.of_unit][:, None, :])

    def _factorize(self, levels, Htt=None, Haa=None):
        if self.kind == "tree":
            return _tree_factor(levels, Htt)
        A, Bl = levels
        F = _crossed_factor(A.D, Bl.D, self._crossed_X(Haa), self.factors[0].of_unit,
                            self.factors[1].of_unit, A.B, Bl.B, Htt)
        return F

    def _apply(self, F, r_levels, r_beta=None):
        if F.kind == "tree":
            return _tree_apply(F, r_levels, r_beta)
        return _crossed_apply(F, r_levels[0], r_levels[1], r_beta)

    def _cov(self, F):
        return _tree_cov(F) if F.kind == "tree" else _crossed_cov(F)

    def _random_factor(self, Haa, ga, D, Lams):
        """The random effects' own factorization (b and the fixed effect
        held), its covariance blocks and log-determinant."""
        levels = self._blocks(Haa, ga, D, Lams, with_beta=False, random_only=True)
        F = self._factorize(levels, Haa=Haa)
        return levels, F

    def prepare_laplace(self):
        """The units' total effects' covariances at the cached estimates and
        the current Sigma, for the next read's exact Laplace terms."""
        D = self._cache
        _, F = self._random_factor(D.Haa, D.ga, D, [f.Lam for f in self.random])
        self._Vu = self._cov(F)[1]

    def _assemble(self, D, Lams, corrections):
        """The penalized log-likelihood, its gradient and the blocks, from
        a data pass D, for the penalties Lams; with `corrections`, the exact
        Laplace approximation's terms in the gradient for b and the fixed
        effect."""
        ll = D.ll
        for f, Lam in zip(self.random, Lams):
            A = self._theta_of(D.theta, f)
            ll -= 0.5 * float(np.sum((A @ Lam) * A))
        levels = self._blocks(D.Haa, D.ga, D, Lams)
        gt = D.gt.copy()
        g_lev = []
        for lev in levels:
            g = lev.g
            if lev.f.kind == "random":
                g = g - self._theta_of(D.theta, lev.f) @ Lams[self.random.index(lev.f)]
            g_lev.append(g)
        if corrections and D.extra:
            gt, g_lev = self._corrections(D, Lams, levels, gt, g_lev)
        for lev, g in zip(levels, g_lev):
            lev.g = g
        H = SimpleNamespace(tt=D.Htt, levels=levels, Haa=D.Haa)
        return ll, np.concatenate([gt] + [g.ravel() for g in g_lev]), H

    def _corrections(self, D, Lams, levels, gt, g_lev):
        lv = self.lv
        Vu = D.Vu[:, lv][:, :, lv]
        Cu = np.einsum("ulj,ujlk->uk", Vu, D.T)
        Cfull = np.zeros((self.U, self.m))
        Cfull[:, lv] = Cu
        Cfull *= self.keep_u[:, None]
        _, F = self._random_factor(D.Haa, D.ga, D, Lams)
        C_lev = [_sum_by(f.of_unit, Cfull, f.G) * f.free for f in self.random]
        u_lev, _, _ = self._apply(F, C_lev)
        corr_t = D.Dt.copy()
        for lev in levels:
            if lev.f.kind == "random":
                corr_t -= np.einsum("gmp,gm->p", lev.B, u_lev[self.random.index(lev.f)])
        gt = gt - 0.5 * corr_t
        if self.fixed_factor is not None:
            fe = self.fixed_factor
            Cf = Cfull.copy()
            if self.model.kind == "mlogit":
                Cf[:, 0] = -Cu.sum(axis=1) * self.keep_u
            cf = _sum_by(fe.of_unit, Cf, fe.G)
            k_fe = len(levels) - 1
            for lev in levels[:-1]:
                cross = lev.X[k_fe]                         # (G_k, m_k rows, m_fe cols)
                cf -= _sum_by(lev.anc[k_fe], np.einsum("gij,gi->gj", cross,
                                                       u_lev[self.random.index(lev.f)]), fe.G)
            g_lev = list(g_lev)
            g_lev[-1] = g_lev[-1] - 0.5 * cf * fe.free
        return gt, g_lev

    def pass_(self, driver, theta, hessian):
        extra = self.method == "laplace"
        c = self._cache
        if c is None or not np.array_equal(c.theta, theta) or (extra and not c.extra):
            D = self._data_pass(driver, theta, extra)
            if not np.isfinite(D.ll):
                return -np.inf, None, None
            self._cache = D
        ll, g, H = self._assemble(self._cache, [f.Lam for f in self.random], extra)
        return ll, g, (H if hessian else None)

    def numba_pass(self):
        """Whether a pass does its heavy work in numba (the multinomial's
        group blocks), so that BLAS should keep to one thread."""
        return self.model.kind == "mlogit"

    # ------------------------------------------------------------ the Newton step
    def _full(self, H):
        if getattr(H, "F", None) is None:
            H.F = self._factorize(H.levels, H.tt, H.Haa)
        return H.F

    def solve(self, H, g):
        F = self._full(H)
        P = H.tt.shape[0]
        r_lev, o = [], P
        for lev in H.levels:
            n = lev.f.G * self.m
            r_lev.append(g[o:o + n].reshape(lev.f.G, self.m))
            o += n
        d_lev, db, pd = self._apply(F, r_lev, g[:P])
        return np.concatenate([db] + [(d * lev.f.free).ravel()
                                      for d, lev in zip(d_lev, H.levels)]), pd

    def reduced(self, H):
        return self._full(H).R

    def vcov_beta(self, H):
        return _Driver.inverse(self.reduced(H))

    def sandwich(self, H, M):
        V = self.vcov_beta(H)
        out = V @ M @ V
        return (out + out.T) / 2

    def damp(self, H, lam):
        from .newton import _floored
        idx = np.arange(self.m)
        levels = []
        for lev in H.levels:
            D = lev.D.copy()
            D[:, idx, idx] += lam * _floored(np.abs(np.diagonal(lev.D, axis1=1, axis2=2))) \
                * lev.f.free
            levels.append(SimpleNamespace(**{**vars(lev), "D": D}))
        return SimpleNamespace(tt=H.tt + lam * np.diag(_floored(np.abs(np.diag(H.tt)))),
                               levels=levels, Haa=H.Haa)

    def subset(self, theta, g, H, idx):
        n_eff = sum(f.G for f in self.factors) * self.m
        P_old = len(theta) - n_eff
        keep = lambda v: np.r_[v[idx], v[P_old:]]           # noqa: E731
        # (the next pass's theta differs from the cached one, so it reads afresh)
        levels = [SimpleNamespace(**{**vars(lev), "B": lev.B[:, :, idx]}) for lev in H.levels]
        return keep(theta), keep(g), SimpleNamespace(tt=H.tt[np.ix_(idx, idx)], levels=levels,
                                                     Haa=H.Haa)

    # ------------------------------------------------------------ inference
    def meat(self, driver, theta, H, weights_type, clusters, hetero):
        """Cluster sums of the rows' scores for b (the profile scores),
        concentrated with respect to the fixed effect if there is one."""
        if hetero:
            raise ValueError("heteroskedasticity-robust (by row) variances are not available "
                             "with random intercepts, whose rows within a group are dependent; "
                             "cluster by a grouping that contains whole groups")
        if driver.device is not None:
            from .. import device
            return device.re_meat(driver.device.xp, self, driver, theta, H, weights_type,
                                  clusters)
        m = self.model
        P = m.P
        th = theta[:P]
        Au = self._unit_effects(theta)
        EF = None
        fe = self.fixed_factor
        if fe is not None:
            F = self._full(H)
            k = len(self.factors) - 1
            EF = F.Dinv[k] @ F.B[k]                             # the fixed effect's F^-1 E
        U_ = {name: np.zeros((Gc, P)) for name, (_, Gc) in clusters.items()}
        rows = max(256, int(driver.chunk_bytes // (8 * max(P * (self.m + 1), 1))))
        for blk in self.store:
            for ch in _chunks(blk, rows):
                u = self._units(ch)
                keep = self.keep_u[u]
                if not keep.all():
                    ch = {k: (v[keep] if k != "starts" else v) for k, v in ch.items()}
                    u = u[keep]
                if not len(ch["y"]):
                    continue
                X = _design(ch, m.cols, driver.K0)
                y = ch["y"]
                Ar = Au[u]
                if ch.get("off") is not None:
                    Ar = Ar + ch["off"][:, None]
                if m.kind == "mlogit":
                    n = len(y)
                    E = np.empty((n, self.J))
                    E[:, 0] = 0.0
                    E[:, 1:] = X @ th.reshape(self.J - 1, -1).T
                    E += Ar
                    ex = np.exp(E - E.max(axis=1)[:, None])
                    R = -ex / ex.sum(axis=1)[:, None]
                    R[np.arange(n), y] += 1.0
                    S = (R[:, 1:, None] * X[:, None, :]).reshape(n, -1)
                    if EF is not None:
                        sg = fe.of_unit[u]
                        S = S - np.einsum("nj,njp->np", R * fe.free[sg], EF[sg])
                else:
                    S = m.scores(th, X, y, None, offset=Ar[:, 0])
                    if EF is not None:
                        sc = m.rows(th, X, y, None, offset=Ar[:, 0], hessian=False)["s"]
                        S = S - sc[:, None] * EF[fe.of_unit[u], 0, :]
                driver._add_meat(S, ch.get("w"), ch.get("cl"), weights_type, None, U_, clusters)
        return None, U_

    def nested(self, clusters):
        """Whether each clustering contains whole groups of the coarsest
        random grouping (the independent blocks); never for crossed
        groupings, whose groups are all linked."""
        top = self.random[-1]
        out = {}
        for k, (i, _) in clusters.items():
            if self.kind == "crossed":
                out[k] = False
                continue
            seen = np.full(top.G, -1, np.int64)
            ok = True
            for blk in self.store.iterate(["cl", "y", *self._unit_keys()]):
                f = top.of_unit[self._units(blk)]
                c = blk["cl"][:, i]
                prev = seen[f]
                o = np.argsort(f, kind="stable")
                if (np.any((prev >= 0) & (prev != c))
                        or np.any((f[o][1:] == f[o][:-1]) & (c[o][1:] != c[o][:-1]))):
                    ok = False
                    break
                seen[f] = c
            out[k] = ok
        return out

    def _unit_keys(self):
        return [self.unit_key] if self.kind == "tree" else list(self.cell_cols)

    # ------------------------------------------------------------ results
    def result_spec(self, theta):
        """Each random grouping's predicted effects (`ranef`), the fixed
        effect's (`fixef`), and how `predict` combines them."""
        levels = self.p0.levels
        mlogit = self.model.kind == "mlogit"
        out = {"ranef": {}, "effects": []}
        for f in self.random:
            A = self._theta_of(theta, f)
            vals = ({f"re_{f.name}:{lv}": pl.Series(A[:, j]) for j, lv in enumerate(levels)}
                    if mlogit else {f"re_{f.name}": pl.Series(A[:, 0])})
            n = pl.Series(f.ncnt.sum(axis=1).astype(np.int64))
            table = self.p0.fe_tables[f.name].with_columns(**vals, n_obs=n)
            out["ranef"][f.name] = table.filter(pl.Series(f.keep)).drop(CODE)
            out["effects"].append((f.name, list(vals), "random"))
        fe = self.fixed_factor
        if fe is not None:
            Af = self._theta_of(theta, fe)
            fvals = ({f"fe_{fe.name}:{lv}": pl.Series(np.where(np.isfinite(Af[:, j]), Af[:, j],
                                                               np.nan)).fill_nan(None)
                      for j, lv in enumerate(levels)}
                     if mlogit else {f"fe_{fe.name}": pl.Series(Af[:, 0])})
            ft = self.p0.fe_tables[fe.name].with_columns(
                **fvals, n_obs=pl.Series(fe.ncnt.sum(axis=1).astype(np.int64)))
            keep = pl.Series(fe.keep)
            out["fixef"] = {fe.name: ft.filter(keep).drop(CODE)}
            out["effects"].append((fe.name, list(fvals), "fixed"))
            dropped = ft.filter(~keep).drop(CODE)
            if dropped.height:
                out["dropped"] = {fe.name: dropped}
        return out
