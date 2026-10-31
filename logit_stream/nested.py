"""The nested logit (`nlogit_stream`): McFadden's (1978) generalized
extreme value model with the alternatives grouped into a tree of nests, in
the random-utility-consistent parameterization ("RU2", Stata's `nlogit`
default; Train 2009, ch. 4; Hensher, Rose and Greene 2015, ch. 14).

Each nest n has a dissimilarity parameter lambda_n (the root's is 1). With
V_r = x_r'b the utility of alternative r, a child c of nest n has the value
U_c = V_c (an alternative) or U_c = lambda_c I_c (a nest), and

    I_n            = log sum_{c in n} exp(U_c / lambda_n)   (inclusive value)
    P(c | n)       = exp(U_c / lambda_n - I_n)

so that an alternative's probability is the product of the conditional
probabilities along its path to the root. With two levels:
P(r) = exp(V_r / lambda_m - I_m) exp(lambda_m I_m - I_root). All lambdas 1
is the conditional logit, which is the starting point. The model is
consistent with utility maximization for 0 < lambda_child <= lambda_parent
<= 1 (beyond that for some range of the data only); a nest with one child
has no lambda (its value does not depend on it).

The fit is optimize.py's, as the mixed logit's: each read evaluates the
log-likelihood and its analytic gradient at several parameter vectors (a
numba kernel over the choice sets in parallel: one pass over a set's rows
up the tree, the gradient of each node's value carried with it), the
information is the central differences of the gradient, and between them
the steps are quasi-Newton (BFGS) with a line search.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numba as nb
import numpy as np

from ._limits import limited
from .optimize import _GradientDriver

# --------------------------------------------------------------------------
# the tree's likelihood
# --------------------------------------------------------------------------


@nb.njit(cache=True, inline="always")
def _lse_add(n, x, u, du, mx, s, su, sd, P):
    """Add a child with exponent x (its value over the parent's lambda),
    value u and value gradient du to node n's running log-sum-exp, the
    weighted sums of the values and gradients rescaled as the maximum
    moves."""
    if s[n] == 0.0:
        mx[n] = x
        s[n] = 1.0
        su[n] = u
        for p in range(P):
            sd[n, p] = du[p]
    elif x > mx[n]:
        f = math.exp(mx[n] - x)
        mx[n] = x
        s[n] = s[n] * f + 1.0
        su[n] = su[n] * f + u
        for p in range(P):
            sd[n, p] = sd[n, p] * f + du[p]
    else:
        e = math.exp(x - mx[n])
        s[n] += e
        su[n] += e * u
        for p in range(P):
            sd[n, p] += e * du[p]


@nb.njit(parallel=True, cache=True)
def _nl_kernel(X, y, w, off, starts, nest, TH, K, parent, lam_of, order, n_bhhh, n_threads,
               Gs):
    """The nested logit's log-likelihood and gradient at each row of TH (b,
    then the free lambdas) over the choice sets starts[c]:starts[c+1] (one
    chosen alternative each, y > 0), each set weighted by w (its first
    row's), each row's utility plus its offset `off`, and the sum of the
    sets' weighted score outer products at the
    first n_bhhh rows. The tree: nodes 0..M-1 the nests, M the root;
    parent[n] each nest's parent; lam_of[n] the index among the free
    lambdas of nest n's, -1 for one fixed at 1; `order` the nests children
    first; nest[r] the nest of row r's alternative. With Gs of C rows, each
    set's unweighted score at TH[0] is written to it. A row of TH with a
    lambda not above 0 gets -inf."""
    C, T, P = len(starts) - 1, TH.shape[0], TH.shape[1]
    M = len(parent)
    ll_t = np.zeros((n_threads, T))
    G_t = np.zeros((n_threads, T, P))
    B_t = np.zeros((n_threads, max(n_bhhh, 1), P, P))
    step = (C + n_threads - 1) // n_threads
    want = Gs.shape[0] == C
    for th in nb.prange(n_threads):
        lam = np.ones(M + 1)
        mx = np.empty(M + 1)
        s = np.empty(M + 1)
        su = np.empty(M + 1)
        sd = np.empty((M + 1, P))
        U = np.empty(M + 1)
        IV = np.empty(M + 1)
        dU = np.empty((M + 1, P))
        dIV = np.empty((M + 1, P))
        du = np.empty(P)
        g = np.empty(P)
        # (accumulated here and written out once per row of TH: threads adding
        # into neighbouring slots of the shared arrays set by set contend)
        gl = np.empty(P)
        Bl = np.empty((P, P))
        for t in range(T):
            ll_l = 0.0
            for p in range(P):
                gl[p] = 0.0
            if t < n_bhhh:
                for p in range(P):
                    for q in range(P):
                        Bl[p, q] = 0.0
            bad = False
            for m in range(M):
                lam[m] = 1.0 if lam_of[m] < 0 else TH[t, K + lam_of[m]]
                if lam[m] <= 0.0:
                    bad = True
            if bad:
                ll_t[th, t] = -np.inf
                continue
            for c in range(th * step, min(C, (th + 1) * step)):
                r0, r1 = starts[c], starts[c + 1]
                for m in range(M + 1):
                    s[m] = 0.0
                rc, Vc = r0, 0.0
                for r in range(r0, r1):
                    v = off[r]
                    for k in range(K):
                        v += X[r, k] * TH[t, k]
                        du[k] = X[r, k]
                    for p in range(K, P):
                        du[p] = 0.0
                    m = nest[r]
                    _lse_add(m, v / lam[m], v, du, mx, s, su, sd, P)
                    if y[r] > 0:
                        rc, Vc = r, v
                # up the tree: each nest's inclusive value and value, with their
                # gradients, then added to its parent
                for i in range(M + 1):
                    n = order[i] if i < M else M
                    if s[n] == 0.0:
                        continue
                    ln = lam[n]
                    IV[n] = mx[n] + math.log(s[n])
                    ub = su[n] / s[n]
                    j = lam_of[n] if n < M else -1
                    for p in range(P):
                        dIV[n, p] = sd[n, p] / s[n] / ln
                    if j >= 0:
                        dIV[n, K + j] -= ub / (ln * ln)
                    if n == M:
                        break
                    U[n] = ln * IV[n]
                    for p in range(P):
                        dU[n, p] = ln * dIV[n, p]
                    if j >= 0:
                        dU[n, K + j] += IV[n]
                    q = parent[n]
                    _lse_add(q, U[n] / lam[q], U[n], dU[n], mx, s, su, sd, P)
                # the chosen alternative's log-probability: along its path,
                # log P(child | n) = U_child / lambda_n - I_n
                n = nest[rc]
                uc = Vc
                for k in range(K):
                    g[k] = X[rc, k]
                for p in range(K, P):
                    g[p] = 0.0
                lp = 0.0
                # g holds d U_child; the score accumulates in du
                for p in range(P):
                    du[p] = 0.0
                while True:
                    ln = lam[n]
                    lp += uc / ln - IV[n]
                    j = lam_of[n] if n < M else -1
                    for p in range(P):
                        du[p] += g[p] / ln - dIV[n, p]
                    if j >= 0:
                        du[K + j] -= uc / (ln * ln)
                    if n == M:
                        break
                    uc = U[n]
                    for p in range(P):
                        g[p] = dU[n, p]
                    n = parent[n]
                wc = w[r0]
                ll_l += wc * lp
                for p in range(P):
                    gl[p] += wc * du[p]
                if want and t == 0:
                    for p in range(P):
                        Gs[c, p] = du[p]
                if t < n_bhhh:
                    for p in range(P):
                        for q in range(P):
                            Bl[p, q] += wc * du[p] * du[q]
            ll_t[th, t] = ll_l
            for p in range(P):
                G_t[th, t, p] = gl[p]
            if t < n_bhhh:
                for p in range(P):
                    for q in range(P):
                        B_t[th, t, p, q] = Bl[p, q]
    ll = np.zeros(T)
    G = np.zeros((T, P))
    B = np.zeros((max(n_bhhh, 1), P, P))
    for th in range(n_threads):
        ll += ll_t[th]
        G += G_t[th]
        B += B_t[th]
    return ll, G, B


@nb.njit(cache=True)
def _nl_probs(V, nest, gid, lam, parent, order):
    """Each row's probability within its choice set (rows with the same gid;
    a null utility, NaN, makes its whole set NaN), for the tree as in
    `_nl_kernel` with the lambdas given."""
    n = len(V)
    M = len(parent)
    out = np.full(n, np.nan)
    idx = np.argsort(gid, kind="mergesort")
    mx = np.empty(M + 1)
    s = np.empty(M + 1)
    U = np.empty(M + 1)
    IV = np.empty(M + 1)
    a = 0
    while a < n:
        b = a
        while b < n and gid[idx[b]] == gid[idx[a]]:
            b += 1
        ok = True
        for i in range(a, b):
            if np.isnan(V[idx[i]]):
                ok = False
        if ok:
            for m in range(M + 1):
                s[m] = 0.0
            for i in range(a, b):
                r = idx[i]
                m = nest[r]
                x = V[r] / lam[m]
                if s[m] == 0.0:
                    mx[m], s[m] = x, 1.0
                elif x > mx[m]:
                    s[m] = s[m] * math.exp(mx[m] - x) + 1.0
                    mx[m] = x
                else:
                    s[m] += math.exp(x - mx[m])
            for i in range(M + 1):
                m = order[i] if i < M else M
                if s[m] == 0.0:
                    continue
                IV[m] = mx[m] + math.log(s[m])
                if m == M:
                    break
                U[m] = lam[m] * IV[m]
                q = parent[m]
                x = U[m] / lam[q]
                if s[q] == 0.0:
                    mx[q], s[q] = x, 1.0
                elif x > mx[q]:
                    s[q] = s[q] * math.exp(mx[q] - x) + 1.0
                    mx[q] = x
                else:
                    s[q] += math.exp(x - mx[q])
            for i in range(a, b):
                r = idx[i]
                m = nest[r]
                lp = V[r] / lam[m] - IV[m]
                while m != M:
                    q = parent[m]
                    lp += U[m] / lam[q] - IV[q]
                    m = q
                out[r] = math.exp(lp)
        a = b
    return out


class _NestedDriver(_GradientDriver):
    """`optimize._GradientDriver`'s Newton and quasi-Newton steps on the nested logit's
    likelihood (`_nl_kernel`)."""

    def __init__(self, store, K, tree, *, tol, maxiter, log, device=None):
        self.store, self.K, self.tree = store, K, tree
        self.device = device            # the GPU's array namespace (device/), or None
        self._dstore = None
        self.P = K + tree.n_lambda
        self.tol, self.maxiter, self.switch = float(tol), int(maxiter), np.inf
        self.log = log
        self.passes = 0

    def _blocks(self):
        for blk in self.store.iterate(["X", "y", "w", "off", "starts", "fe", "cl"]):
            n = len(blk["y"])
            w, off = blk.get("w"), blk.get("off")
            blk["off"] = (np.zeros(n) if off is None
                          else np.ascontiguousarray(off, dtype=np.float64))
            yield blk, (np.ones(n) if w is None else np.ascontiguousarray(w, dtype=np.float64))

    def _run(self, blk, w, TH, n_bhhh, Gs):
        t = self.tree
        return _nl_kernel(blk["X"], blk["y"], w, blk["off"], blk["starts"],
                          np.ascontiguousarray(blk["fe"], dtype=np.int64), TH, self.K,
                          t.parent, t.lam_of, t.order, n_bhhh, nb.get_num_threads(), Gs)

    def _device_evaluate(self, TH, n_bhhh):
        """`evaluate` on the GPU (device/'s `nl_pass`)."""
        from . import device
        xp = self.device.xp
        if self._dstore is None:
            self._dstore = device.DeviceStore(self.store, self.device)
        M = TH.shape[0]
        ll, G = np.zeros(M), np.zeros((M, self.P))
        B = np.zeros((max(n_bhhh, 1), self.P, self.P))
        for blk in self._dstore:
            l, g, b = device.nl_pass(xp, blk["X"], blk["y"], blk.get("w"), blk.get("off"),
                                     blk["starts"].astype(xp.int64),
                                     blk["fe"].astype(xp.int64), TH, self.K, self.tree, n_bhhh)
            ll += l
            G += g
            B += b
        self.passes += 1
        return ll, G, B

    def evaluate(self, thetas, n_bhhh=0):
        TH = np.ascontiguousarray(np.atleast_2d(thetas), dtype=np.float64)
        if self.device is not None:
            return self._device_evaluate(TH, n_bhhh)
        M = TH.shape[0]
        ll, G = np.zeros(M), np.zeros((M, self.P))
        B = np.zeros((max(n_bhhh, 1), self.P, self.P))
        none = np.zeros((0, self.P))
        for blk, w in self._blocks():
            l, g, b = self._run(blk, w, TH, n_bhhh, none)
            ll += l
            G += g
            B += b
        self.passes += 1
        return ll, G, B

    def meat(self, theta, weighted, weights_type, clusters, hetero):
        """The robust variances' middle matrices from the choice sets'
        scores, as `_Driver.meat` forms them: one read."""
        from .newton import _Driver
        P = self.P
        TH = np.ascontiguousarray(theta[None, :], dtype=np.float64)
        if self.device is not None:
            from . import device
            xp = self.device.xp
            Md = xp.zeros((P, P)) if hetero else None
            Ud = {name: xp.zeros((G, P)) for name, (_, G) in clusters.items()}
            for blk in self._dstore:
                st = blk["starts"].astype(xp.int64)
                S = device.nl_pass(xp, blk["X"], blk["y"], blk.get("w"), blk.get("off"), st,
                                   blk["fe"].astype(xp.int64), TH, self.K, self.tree, 0,
                                   scores=True)
                w, cl = blk.get("w"), blk.get("cl")
                device.meat_sums(xp, S, None if w is None or not weighted else w[st[:-1]],
                                 None if cl is None else cl[st[:-1]], weights_type, Md, Ud,
                                 clusters)
            self.passes += 1
            return device.to_host(xp, Md, Ud)
        M = np.zeros((P, P)) if hetero else None
        U = {name: np.zeros((G, P)) for name, (_, G) in clusters.items()}
        for blk, w in self._blocks():
            st = blk["starts"]
            S = np.zeros((len(st) - 1, P))
            self._run(blk, w, TH, 0, S)
            cl = blk.get("cl")
            _Driver._add_meat(S, w[st[:-1]] if weighted else None,
                              None if cl is None else cl[st[:-1]], weights_type, M, U, clusters)
        self.passes += 1
        return M, U


# --------------------------------------------------------------------------
# the tree of nests
# --------------------------------------------------------------------------

class _Tree:
    """The nests as arrays for the kernels: `bottom` the nests that hold
    alternatives, coded as pass 0 codes them (their sorted names), then the
    nests above them; parent, lam_of and order as `_nl_kernel` takes them;
    `of[alt]` each alternative's nest; `names` the nests' names by code;
    `lambdas` the names of the nests with a lambda, in the order given."""

    def __init__(self, nests, seen):
        self.of, kids, upper, given = {}, {}, {}, []

        def walk(name, spec, up):
            name = str(name)
            if name in kids:
                raise ValueError(f"nest {name!r} appears twice in nests=")
            given.append(name)
            upper[name] = up
            if isinstance(spec, dict):
                if not spec:
                    raise ValueError(f"nest {name!r} has no nests in it")
                kids[name] = ("nests", [str(k) for k in spec])
                for k, v in spec.items():
                    walk(k, v, name)
                return
            alts = [spec] if isinstance(spec, (str, int, float)) else list(spec)
            if not alts:
                raise ValueError(f"nest {name!r} has no alternatives")
            if any(isinstance(a, (dict, list, tuple)) for a in alts):
                raise ValueError(f"nest {name!r}: a nest holds alternatives (a list) or "
                                 "nests (a dict), not both")
            for a in alts:
                if a in self.of:
                    raise ValueError(f"alternative {a!r} is in nests {self.of[a]!r} and "
                                     f"{name!r}")
                self.of[a] = name
            kids[name] = ("alts", alts)

        if not isinstance(nests, dict) or len(nests) < 2:
            raise ValueError("nests= is a dict {nest name: [alternatives] or {nest: ...}} with "
                             "at least two nests at the top")
        for k, v in nests.items():
            walk(k, v, None)
        missing = [a for a in seen if a is not None and a not in self.of]
        if missing:
            raise ValueError(f"alternatives {missing[:10]} are in no nest of nests=")
        absent = [a for a in self.of if a not in seen]
        if absent:
            raise ValueError(f"alternatives {absent[:10]} of nests= never occur in the data")
        self.spec = nests
        self.kids, self.given = kids, given
        self.bottom = sorted(n for n in kids if kids[n][0] == "alts")
        self.names = self.bottom + sorted(n for n in kids if kids[n][0] == "nests")
        code = {n: i for i, n in enumerate(self.names)}
        M = len(self.names)
        self.parent = np.array([M if upper[n] is None else code[upper[n]] for n in self.names],
                               dtype=np.int64)
        depth = {}

        def d(n):
            if n not in depth:
                depth[n] = 0 if upper[n] is None else d(upper[n]) + 1
            return depth[n]

        self.order = np.array(sorted(range(M), key=lambda i: -d(self.names[i])),
                              dtype=np.int64)
        self.lambdas = [n for n in given if len(kids[n][1]) > 1]
        self.lam_of = np.array([self.lambdas.index(n) if n in self.lambdas else -1
                                for n in self.names], dtype=np.int64)
        self.n_lambda = len(self.lambdas)
        self.upper = upper

    def lam(self, values):
        """The lambdas by node (root last, 1), from the free ones' values."""
        out = np.ones(len(self.names) + 1)
        for i, j in enumerate(self.lam_of):
            if j >= 0:
                out[i] = values[j]
        return out

    def leaves(self, name):
        kind, ks = self.kids[name]
        return list(ks) if kind == "alts" else [a for k in ks for a in self.leaves(k)]


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

@limited
def _fit_nested(fml, data, *, case, alt, nests, vcov="iid", cluster=(), weights=None,
                weights_type="aweights", offset=None, workdir=None, tol=1e-8, maxiter=200,
                storage="disk", dtype="float64", block_mb=64.0, polars_engine="cpu",
                keep_intermediates=False, verbose=True, logger=None, log_level=None,
                device="cpu", profile=False, chunk_mb=64.0, max_memory_gb=None,
               memory_reserve=None):
    import polars as pl
    from hdfe_stream.core.report import warn
    from hdfe_stream.core.resources import MemoryBudget
    from hdfe_stream.core.store import PolarsEngine
    from hdfe_stream.core.utils import as_lazy
    from hdfe_stream.core.workspace import Run

    from ._choice import clogit_start, fit_diagnostics, plan_one
    from .device import resolve
    from .estimator import _base_dir, _clogit_null, _clustered, _clusterings, _log
    from .newton import _Driver
    from .results import NestedLogitResult, _ic
    from .store import CODE, _build

    if weights_type not in ("aweights", "fweights"):
        raise ValueError("weights_type must be 'aweights' or 'fweights'")
    vkey, clusters, hetero, requests = _clusterings(vcov, cluster)
    if any(r.startswith("CRV3:") for r in requests):
        raise ValueError("CRV3 is not available for the nested logit; use CRV1")
    case = [case] if isinstance(case, str) else list(case)
    if not isinstance(alt, str):
        raise ValueError("alt= is the column naming each row's alternative")
    if offset is not None:
        offset = (pl.col(offset) if isinstance(offset, str) else offset).cast(pl.Float64)
    device = resolve(device)

    def log(msg):
        _log(verbose, msg, logger, log_level)

    lf = as_lazy(data)
    engine = PolarsEngine(polars_engine)
    seen = engine.collect(lf.select(pl.col(alt)).unique(), "alternatives")[alt].to_list()
    tree = _Tree(nests, seen)
    if not tree.n_lambda:
        raise ValueError("every nest has one child: the model is the conditional logit (use "
                         "clogit_stream)")
    # each row's bottom nest travels as pass 0's fixed-effect codes (0, 1, ...
    # in the nests' sorted names, as `tree.bottom`)
    NEST = f"{alt}_nest"
    if NEST in lf.collect_schema().names():
        raise ValueError(f"the data has a column {NEST!r}, the name the nested logit gives "
                         "the nests; rename it")
    lf = lf.with_columns(pl.col(alt).replace_strict(
        list(tree.of), [tree.of[a] for a in tree.of], return_dtype=pl.String).alias(NEST))
    spec = replace(plan_one(fml, lf, engine, log, "nested logit"), fe=(NEST,))
    xnames = [nm for nm, _ in spec.x]
    K = len(xnames)
    if K == 0:
        raise ValueError(f"{spec.fml}: the nested logit needs covariates")

    if profile:
        from .profile import profile_model
        return profile_model("nlogit", spec, lf, data, weights=weights, clusters=clusters,
                             requests=requests, case=case, storage=storage, dtype=dtype,
                             block_mb=block_mb, chunk_mb=chunk_mb, offset=offset,
                             n_lambda=tree.n_lambda, n_alt=len(seen),
                             budget=MemoryBudget(max_memory_gb, memory_reserve))
    t0 = time.time()
    log(f"nested logit: {spec.fml}")
    run = Run(_base_dir(workdir), auto_cleanup=not keep_intermediates)
    try:
        p0 = _build(lf, spec, "clogit", run.path, engine, weights=weights, clusters=clusters,
                    case=case, levels=None, in_memory=(storage == "memory"),
                    dtype=np.float32 if dtype == "float32" else np.float64,
                    block_mb=block_mb, log=log, weights_type=weights_type, offset=offset)
        coded = p0.fe_tables[NEST]
        if coded.sort(CODE)[NEST].to_list() != tree.bottom:
            raise RuntimeError("the nests' codes are not their sorted names")
        cfit = clogit_start(p0, xnames, spec, logger, log)
        drv = _NestedDriver(p0.store, K, tree, tol=tol, maxiter=maxiter, log=log,
                            device=device)
        fit = drv.run(np.r_[cfit.theta[:K], np.ones(tree.n_lambda)])
        if not fit.converged:
            warn(f"{spec.fml}: the nested logit did not converge (decrement "
                 f"{fit.decrement:.3g} after {fit.iterations} steps)", logger)
        lam = dict(zip(tree.lambdas, fit.theta[K:]))
        out = [n for n, v in lam.items() if not 0 < v <= 1]
        if out:
            warn(f"{spec.fml}: the dissimilarity parameters of nests {out} are outside "
                 "(0, 1], where the nested logit is consistent with random utility "
                 "maximization for all values of the covariates", logger)
        above = [n for n, v in lam.items()
                 if tree.upper[n] in lam and v > lam[tree.upper[n]] * (1 + 1e-9)]
        if above:
            warn(f"{spec.fml}: the dissimilarity parameters of nests {above} exceed their "
                 "parent's, which is consistent with random utility maximization for some "
                 "values of the covariates only", logger)
        coefnames = xnames + [f"lambda.{n}" for n in tree.lambdas]
        P = len(coefnames)
        N = p0.wcases if weights and weights_type == "fweights" else int(p0.n_cases)
        Hinv = _Driver.inverse(fit.H)
        all_vcovs, n_clusters = {"iid": (Hinv, int(N - P))}, {}
        if hetero or clusters:
            log("final pass: scores for the robust variance")
            cl_index = {k: (i, int(p0.cluster_tables[k].height))
                        for i, k in enumerate(clusters)}
            M, U = drv.meat(fit.theta, bool(weights), weights_type, cl_index, hetero)

            def sandwich(Mid):
                V = Hinv @ Mid @ Hinv
                return (V + V.T) / 2

            if hetero:
                all_vcovs["hetero"] = (sandwich(M) * N / max(N - 1, 1), int(N - P))
            raw = {k: sandwich(U[k].T @ U[k]) for k in cl_index}
            for req, parts in requests.items():
                V, df, Gs = _clustered(raw, parts, cl_index, req, spec.fml, logger)
                all_vcovs[req] = (V, df)
                n_clusters[req.split(":", 1)[1]] = Gs[0] if len(Gs) == 1 else Gs
        V, df_t = all_vcovs[vkey]
        ll0 = _clogit_null(p0)
        aic, bic = _ic(fit.ll, P, N)
        diagnostics = fit_diagnostics(fit, cfit, p0, storage=storage, dtype=dtype, tol=tol,
                                      t0=t0, device=device, drv=drv)
        rspec = {"source": lf, "xexprs": dict(spec.x), "xcols": xnames,
                 "usable": p0.usable, "yexpr": spec.y, "n_raw": p0.n_raw, "fe_cols": {},
                 "offset": offset, "incomplete": p0.incomplete, "tree": tree}
        res = NestedLogitResult(
            fml=spec.fml, depvar=spec.depvar, coefnames=coefnames, beta=fit.theta, vcov=V,
            vcov_type=vkey, df_t=df_t, n_obs=int(N) if float(N).is_integer() else N,
            all_vcovs=all_vcovs, n_clusters=n_clusters, diagnostics=diagnostics,
            weights=weights, weights_type=weights_type, model="nested logit",
            loglik=float(fit.ll), loglik_null=ll0,
            pseudo_r2=float(1 - fit.ll / ll0) if ll0 else np.nan, aic=aic, bic=bic,
            n_params=P, converged=fit.converged, case=case, alt=alt,
            nests={n: tree.leaves(n) for n in tree.given}, n_cases=int(p0.n_cases),
            loglik_clogit=float(cfit.ll), _spec=rspec)
        p0.store.delete()
        return res
    finally:
        if not keep_intermediates:
            run.cleanup()
