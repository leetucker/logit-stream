"""Mixed logit by maximum simulated likelihood, streamed (prototype).

The model is McFadden's conditional logit with some coefficients random
across choosers: for chooser (panel) n, with choice situations t and
alternatives j,

    U_ntj = x_ntj' beta_n + e_ntj,    beta_nk = mu_k                  (fixed)
                                      beta_nk = mu_k + s_k z_nk        (normal)
                                      beta_nk = exp(mu_k + s_k z_nk)   (lognormal)

with z_nk independent standard normals. The likelihood of a chooser is the
expectation over z of the product of their choice probabilities, which is
simulated with R draws per chooser:

    SL_n = (1/R) sum_r prod_t P_nt(beta_n^r),     ll = sum_n log SL_n.

Why it streams. The simulated log-likelihood and its gradient are sums over
choosers, so they are accumulated block by block like the other models'; a
chooser's rows must be in one block, which pass 0 arranges (``panel=``).
The draws are not stored: a chooser's draws are a function of its position
among all choosers (Halton points at consecutive indices, or a counter-based
hash for pseudo-random draws), regenerated in each pass, so they are the
same in every pass and do not depend on the blocks.

The optimizer is `optimize._GradientDriver`'s (see there).

The draws of all choosers are kept in memory when they fit
``draws_cache_mb``; otherwise each pass regenerates them (cheaply: the
Halton points of a chooser are consecutive, so each is its predecessor
with its digits incremented).
"""

from __future__ import annotations

import math
import time

import numba as nb
import numpy as np

from ._limits import limited
from .newton import _Driver
from .optimize import _GradientDriver

PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                   73, 79, 83, 89, 97], dtype=np.int64)
DISTRIBUTIONS = {"normal": 1, "n": 1, "lognormal": 2, "ln": 2}

# --------------------------------------------------------------------------
# draws
# --------------------------------------------------------------------------

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


@nb.njit(cache=True)
def _radical_inverse(i, base):
    """The i-th point of the base-`base` van der Corput sequence."""
    f, r = 1.0, 0.0
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r


@nb.njit(cache=True)
def _splitmix(x):
    z = x + _GOLDEN
    z = (z ^ (z >> np.uint64(30))) * _MIX1
    z = (z ^ (z >> np.uint64(27))) * _MIX2
    return z ^ (z >> np.uint64(31))


@nb.njit(cache=True)
def _ndtri(p):
    """The standard normal quantile by Acklam's rational approximation
    (relative error below 1.2e-9: far below the simulation's own error, and
    without a refinement step's erfc and exp, which would double the cost
    of a draw)."""
    if p < 0.02425:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((((-7.784894002430293e-03 * q - 3.223964580411365e-01) * q
                 - 2.400758277161838e+00) * q - 2.549732539343734e+00) * q
               + 4.374664141464968e+00) * q + 2.938163982698783e+00)
             / ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q
                  + 2.445134137142996e+00) * q + 3.754408661907416e+00) * q + 1.0))
    elif p <= 1.0 - 0.02425:
        q = p - 0.5
        r = q * q
        x = (((((-3.969683028665376e+01 * r + 2.209460984245205e+02) * r
                - 2.759285104469687e+02) * r + 1.383577518672690e+02) * r
              - 3.066479806614716e+01) * r + 2.506628277459239e+00) * q \
            / (((((-5.447609879822406e+01 * r + 1.615858368580409e+02) * r
                  - 1.556989798598866e+02) * r + 6.680131188771972e+01) * r
                - 1.328068155288572e+01) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -((((((-7.784894002430293e-03 * q - 3.223964580411365e-01) * q
                  - 2.400758277161838e+00) * q - 2.549732539343734e+00) * q
                + 4.374664141464968e+00) * q + 2.938163982698783e+00)
              / ((((7.784695709041462e-03 * q + 3.224671290700398e-01) * q
                   + 2.445134137142996e+00) * q + 3.754408661907416e+00) * q + 1.0))
    return x


@nb.njit(cache=True)
def _draw(g, r, k, R, halton, drop, prime, seed):
    """The r-th standard normal draw of random coefficient k for chooser g
    (its position among all choosers)."""
    if halton:
        u = _radical_inverse(drop + g * R + r, prime)
    else:
        idx = np.uint64((g * R + r) * 64 + k) + np.uint64(seed) * _GOLDEN
        u = ((_splitmix(idx) >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0)
    return _ndtri(u)


@nb.njit(cache=True)
def _fill_draws(Z, g, R, Kr, halton, drop, primes, seed, digits):
    """Z[r, k] = the r-th draw of random coefficient k for chooser g. The
    Halton points of a chooser are consecutive, so each is the previous one
    with its base-b digits incremented (about two operations per point,
    rather than a radical inverse's division per digit); `digits` is
    scratch space."""
    for k in range(Kr):
        if not halton:
            for r in range(R):
                Z[r, k] = _draw(g, r, k, R, halton, drop, primes[k], seed)
            continue
        b = primes[k]
        i = drop + g * R
        nd = 0
        u = 0.0
        f = 1.0
        while i > 0:
            f /= b
            digits[nd] = i % b
            u += f * digits[nd]
            i //= b
            nd += 1
        for r in range(R):
            Z[r, k] = _ndtri(u)
            # next index: add one in base b, carrying
            j = 0
            f = 1.0 / b
            while True:
                if j == nd:
                    digits[nd] = 0
                    nd += 1
                if digits[j] < b - 1:
                    digits[j] += 1
                    u += f
                    break
                u -= f * (b - 1)
                digits[j] = 0
                j += 1
                f /= b


@nb.njit(parallel=True, cache=True)
def _block_draws(pidx, R, Kr, halton, drop, primes, seed):
    """The draws of a block's choosers, n_choosers x R x Kr."""
    n = pidx.shape[0]
    out = np.empty((n, R, max(Kr, 1)))
    for p in nb.prange(n):
        digits = np.zeros(64, np.int64)
        _fill_draws(out[p], pidx[p], R, Kr, halton, drop, primes, seed, digits)
    return out


def draws(n_choosers, R, n_random, halton=True, drop=100, seed=0):
    """The standard normal draws, n_choosers x R x n_random, as the
    estimator generates them (for checks and post-estimation)."""
    return _block_draws(np.arange(n_choosers, dtype=np.int64), np.int64(R), np.int64(n_random),
                        bool(halton), np.int64(drop), PRIMES[:n_random], np.int64(seed))


# --------------------------------------------------------------------------
# the kernel
# --------------------------------------------------------------------------

@nb.njit(parallel=True, cache=True)
def _mxl_kernel(X, y, cst, pst, pidx, TH, kind, ridx, R, halton, drop, primes, seed,
                n_bhhh, n_threads, Zc):
    """Simulated log-likelihood and gradient at each row of TH (M x P), and
    the score outer products at the first `n_bhhh` rows, over the panels of
    one block. X: rows x K; y: 0/1; cst: choice-set starts (rows, with the
    end); pst: panel starts (rows, with the end); pidx: each panel's
    position among all panels; Zc: the block's draws (`_block_draws`), or
    an empty array to generate them here."""
    n, K = X.shape
    M, P = TH.shape
    Kr = primes.shape[0]
    npan = pst.shape[0] - 1
    maxsize = 1
    for c in range(cst.shape[0] - 1):
        maxsize = max(maxsize, cst[c + 1] - cst[c])
    nb_ = max(n_bhhh, 1)
    ll_t = np.zeros((n_threads, M))
    g_t = np.zeros((n_threads, M, P))
    B_t = np.zeros((n_threads, nb_, P, P))
    step = (npan + n_threads - 1) // n_threads
    logR = math.log(R)
    for t in nb.prange(n_threads):
        Z = np.empty((R, max(Kr, 1)))
        BT = np.empty((R, K))
        D = np.empty((R, K))
        logL = np.empty(R)
        V = np.empty(maxsize)
        s = np.empty(P)
        digits = np.zeros(64, np.int64)
        # (the thread's own sums, copied out at the end: adding into the
        # threads' neighbouring slices chooser by chooser contends)
        ll_l = np.zeros(M)
        g_l = np.zeros((M, P))
        B_l = np.zeros((nb_, P, P))
        for p in range(t * step, min(npan, (t + 1) * step)):
            g = pidx[p]
            r0, r1 = pst[p], pst[p + 1]
            c0 = np.searchsorted(cst, r0)
            c1 = np.searchsorted(cst, r1)
            if Zc.shape[0]:
                Z[:, :] = Zc[p]
            else:
                _fill_draws(Z, g, R, Kr, halton, drop, primes, seed, digits)
            for m in range(M):
                for r in range(R):
                    for k in range(K):
                        if kind[k] == 0:
                            BT[r, k] = TH[m, k]
                        else:
                            e = TH[m, k] + TH[m, K + ridx[k]] * Z[r, ridx[k]]
                            BT[r, k] = e if kind[k] == 1 else math.exp(e)
                        D[r, k] = 0.0
                    lr = 0.0
                    for c in range(c0, c1):
                        a, b = cst[c], cst[c + 1]
                        vmax = -np.inf
                        vch = 0.0
                        for i in range(a, b):
                            v = 0.0
                            for k in range(K):
                                v += X[i, k] * BT[r, k]
                            V[i - a] = v
                            if v > vmax:
                                vmax = v
                            if y[i] == 1:
                                vch = v
                        S = 0.0
                        for i in range(a, b):
                            V[i - a] = math.exp(V[i - a] - vmax)
                            S += V[i - a]
                        lr += vch - vmax - math.log(S)
                        for i in range(a, b):
                            res = (1.0 if y[i] == 1 else 0.0) - V[i - a] / S
                            for k in range(K):
                                D[r, k] += res * X[i, k]
                    logL[r] = lr
                mx = -np.inf
                for r in range(R):
                    if logL[r] > mx:
                        mx = logL[r]
                tot = 0.0
                for r in range(R):
                    tot += math.exp(logL[r] - mx)
                ll_l[m] += mx + math.log(tot) - logR
                for q in range(P):
                    s[q] = 0.0
                for r in range(R):
                    w = math.exp(logL[r] - mx) / tot
                    for k in range(K):
                        d = D[r, k] * w
                        if kind[k] == 0:
                            s[k] += d
                        elif kind[k] == 1:
                            s[k] += d
                            s[K + ridx[k]] += d * Z[r, ridx[k]]
                        else:
                            s[k] += d * BT[r, k]
                            s[K + ridx[k]] += d * BT[r, k] * Z[r, ridx[k]]
                for q in range(P):
                    g_l[m, q] += s[q]
                if m < n_bhhh:
                    for q in range(P):
                        for q2 in range(P):
                            B_l[m, q, q2] += s[q] * s[q2]
        ll_t[t] = ll_l
        g_t[t] = g_l
        B_t[t] = B_l
    ll = np.zeros(M)
    G = np.zeros((M, P))
    B = np.zeros((nb_, P, P))
    for t in range(n_threads):
        ll += ll_t[t]
        G += g_t[t]
        B += B_t[t]
    return ll, G, B


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------

class _MixedDriver(_GradientDriver):
    """Streamed maximum simulated likelihood; see the module docstring."""

    def __init__(self, store, K, kind, ridx, *, R, halton, drop, seed, tol, maxiter,
                 switch, log, n_panels=0, cache_mb=512.0, device=None):
        self.store, self.K = store, K
        self.kind = np.asarray(kind, dtype=np.int64)
        self.ridx = np.asarray(ridx, dtype=np.int64)
        self.Kr = int((self.kind > 0).sum())
        if self.Kr > len(PRIMES):
            raise ValueError(f"at most {len(PRIMES)} random coefficients")
        self.P = K + self.Kr
        self.R, self.halton, self.drop, self.seed = int(R), bool(halton), int(drop), int(seed)
        self.tol, self.maxiter, self.switch = float(tol), int(maxiter), float(switch)
        self.log = log
        self.passes = 0
        # the draws are kept in memory if they fit cache_mb, otherwise each
        # pass regenerates them
        self.cache = {} if n_panels * self.R * max(self.Kr, 1) * 8 <= cache_mb * 1e6 else None
        self.device = device            # the GPU's array namespace (device/), or None
        self._dstore = None

    def _draws(self, i, blk):
        if self.cache is None:
            return np.empty((0, 1, 1))
        if i not in self.cache:
            self.cache[i] = _block_draws(blk["pidx"], self.R, self.Kr, self.halton, self.drop,
                                         PRIMES[:self.Kr], self.seed)
        return self.cache[i]

    def _device_evaluate(self, TH, n_bhhh):
        """`evaluate` on the GPU (device/'s `mxl_pass`): the blocks and,
        when they fit the cache, the draws kept there; otherwise the draws
        are generated there each pass."""
        from . import device
        xp = self.device.xp
        if self._dstore is None:
            self._dstore = device.DeviceStore(
                self.store, self.device,
                extra=lambda blk: {"pstarts": blk["pstarts"], "pidx": blk["pidx"]})
        M = TH.shape[0]
        ll, G = np.zeros(M), np.zeros((M, self.P))
        B = np.zeros((max(n_bhhh, 1), self.P, self.P))
        primes = PRIMES[:self.Kr]
        for i, blk in enumerate(self._dstore):
            if self.cache is not None and i in self.cache:
                Z = self.cache[i]
            else:
                Z = device.mxl_draws(xp, blk["pidx"], self.R, self.Kr, self.halton, self.drop,
                                     primes, self.seed)
                if self.cache is not None:
                    self.cache[i] = Z
            l, g, b = device.mxl_pass(xp, blk["X"], blk["y"], blk["starts"], blk["pstarts"], Z,
                                      TH, self.kind, self.ridx, n_bhhh)
            ll += l
            G += g
            B += b
        self.passes += 1
        return ll, G, B

    def evaluate(self, thetas, n_bhhh=0):
        """One read: the simulated log-likelihood and gradient at each of
        `thetas`, and the score outer products at the first `n_bhhh`."""
        TH = np.ascontiguousarray(np.atleast_2d(thetas), dtype=np.float64)
        if self.device is not None:
            return self._device_evaluate(TH, n_bhhh)
        M = TH.shape[0]
        ll, G = np.zeros(M), np.zeros((M, self.P))
        B = np.zeros((max(n_bhhh, 1), self.P, self.P))
        primes = PRIMES[:self.Kr]
        for i, blk in enumerate(self.store.iterate(["X", "y", "starts", "pstarts", "pidx"])):
            l, g, b = _mxl_kernel(blk["X"], blk["y"], blk["starts"], blk["pstarts"],
                                  blk["pidx"], TH, self.kind, self.ridx, self.R, self.halton,
                                  self.drop, primes, self.seed, n_bhhh, nb.get_num_threads(),
                                  self._draws(i, blk))
            ll += l
            G += g
            B += b
        self.passes += 1
        return ll, G, B

# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------

@limited
def _fit_mixed(fml, data, *, case, panel=None, random, draws=500, halton=True, seed=0,
               drop=100, vcov="iid", workdir=None, tol=1e-8, maxiter=200, switch=np.inf,
               draws_cache_mb=512.0,
               storage="disk", dtype="float64", block_mb=64.0, polars_engine="cpu",
               keep_intermediates=False, verbose=True, logger=None, log_level=None,
               device="cpu", profile=False, chunk_mb=64.0, max_memory_gb=None,
               memory_reserve=None):
    from hdfe_stream.core.report import warn
    from hdfe_stream.core.resources import MemoryBudget
    from hdfe_stream.core.results_base import vcov_key
    from hdfe_stream.core.store import PolarsEngine
    from hdfe_stream.core.utils import as_lazy
    from hdfe_stream.core.workspace import Run

    from ._choice import clogit_start, fit_diagnostics, plan_one
    from .device import resolve
    from .estimator import _base_dir, _clogit_null, _log
    from .results import MixedLogitResult, _ic
    from .store import _build
    vcov = vcov_key(vcov)
    if not random:
        raise ValueError("random= is empty: with no random coefficients, use clogit_stream")
    if int(draws) < 1:
        raise ValueError("draws must be at least 1")
    device = resolve(device)
    case = [case] if isinstance(case, str) else list(case)
    own_panel = panel is not None
    panel = case if panel is None else ([panel] if isinstance(panel, str) else list(panel))
    # the robust variance is the sandwich of the choosers' scores: clustered by
    # the panel when there is one, "hetero" when each choice set is its own
    robust = f"CRV1:{'^'.join(panel)}" if own_panel else "hetero"
    if vcov not in ("iid", robust):
        raise ValueError(f"vcov must be 'iid' or {robust!r} for this mixed logit (the robust "
                         "variance is clustered by chooser: " + (
                             f"{{'CRV1': {'^'.join(panel)!r}}} with panel=" if own_panel
                             else "'hetero' without panel=, each choice set its own chooser")
                         + ")")

    def log(msg):
        _log(verbose, msg, logger, log_level)

    lf = as_lazy(data)
    engine = PolarsEngine(polars_engine)
    spec = plan_one(fml, lf, engine, log, "mixed logit")
    xnames = [nm for nm, _ in spec.x]
    K = len(xnames)
    unknown = [nm for nm in random if nm not in xnames]
    if unknown:
        raise ValueError(f"random coefficients {unknown} are not covariates of the formula; "
                         f"its covariates are {xnames}")
    kind, ridx, rnames = [0] * K, [-1] * K, []
    for k, nm in enumerate(xnames):
        if nm in random:
            dist = str(random[nm]).lower()
            if dist not in DISTRIBUTIONS:
                raise ValueError(f"random coefficient {nm!r}: distribution {random[nm]!r} is not "
                                 "available; use 'normal' or 'lognormal'")
            kind[k], ridx[k] = DISTRIBUTIONS[dist], len(rnames)
            rnames.append(nm)

    if profile:
        from .profile import profile_model
        return profile_model("mixlogit", spec, lf, data, case=case, panel=panel if own_panel else None,
                             storage=storage, dtype=dtype, block_mb=block_mb, chunk_mb=chunk_mb,
                             n_random=len(rnames), draws=int(draws),
                             draws_cache_mb=draws_cache_mb,
                             budget=MemoryBudget(max_memory_gb, memory_reserve))
    t0 = time.time()
    log(f"mixed logit: {spec.fml}")
    run = Run(_base_dir(workdir), auto_cleanup=not keep_intermediates)
    try:
        p0 = _build(lf, spec, "mixlogit", run.path, engine, weights=None, clusters={},
                    case=case, levels=None, in_memory=(storage == "memory"),
                    dtype=np.float32 if dtype == "float32" else np.float64,
                    block_mb=block_mb, log=log, panel=panel)
        cfit = clogit_start(p0, xnames, spec, logger, log)
        b = cfit.theta[:K]
        # standard deviations start at half the conditional logit's
        # coefficient (at least 0.1), 0.5 for a lognormal's log: far from
        # zero, where the likelihood is flat in them (a saddle at exactly 0)
        theta0 = np.r_[b, np.zeros(len(rnames))]
        for k in range(K):
            if kind[k]:
                theta0[K + ridx[k]] = max(0.5 * abs(b[k]), 0.1) if kind[k] == 1 else 0.5
            if kind[k] == 2:
                if b[k] <= 0:
                    warn(f"{xnames[k]!r}: a lognormal coefficient is positive, but the "
                         f"conditional logit's estimate is {b[k]:.3g}; for a negative "
                         "effect, use the negated covariate", logger)
                theta0[k] = np.log(max(abs(b[k]), 1e-3))
        drv = _MixedDriver(p0.store, K, kind, ridx, R=draws, halton=halton, drop=drop,
                           seed=seed, tol=tol, maxiter=maxiter, switch=switch, log=log,
                           n_panels=p0.n_panels, cache_mb=draws_cache_mb, device=device)
        fit = drv.run(theta0)
        if not fit.converged:
            warn(f"{spec.fml}: the mixed logit did not converge (decrement "
                 f"{fit.decrement:.3g} after {fit.iterations} steps)", logger)
        names = xnames + [f"sd.{nm}" for nm in rnames]
        P, N = len(names), int(p0.n_panels)
        Hinv = _Driver.inverse(fit.H)
        rob = Hinv @ fit.B @ Hinv * N / max(N - 1, 1)
        # A random coefficient's sd enters only through sd * draw, with draws
        # symmetric about 0, so its sign is not identified: report |sd|, its
        # rows and columns of the vcov flipped with it (the same standard
        # errors), and keep the signed estimates in the diagnostics.
        theta = np.array(fit.theta, dtype=float)
        sign = np.ones(P)
        sign[K:] = np.where(theta[K:] < 0, -1.0, 1.0)
        theta = theta * sign
        flip = np.outer(sign, sign)
        all_vcovs = {"iid": (Hinv * flip, N - P), robust: ((rob + rob.T) / 2 * flip, N - 1)}
        V, df_t = all_vcovs[vcov]
        ll0 = _clogit_null(p0)
        aic, bic = _ic(fit.ll, P, N)
        diagnostics = fit_diagnostics(
            fit, cfit, p0, storage=storage, dtype=dtype, tol=tol, t0=t0, device=device, drv=drv,
            sd_signed=dict(zip(names[K:], map(float, fit.theta[K:]))))
        if device is not None:
            diagnostics["device"]["draws_cached"] = drv.cache is not None
        res = MixedLogitResult(
            fml=spec.fml, depvar=spec.depvar, coefnames=names, beta=theta, vcov=V,
            vcov_type=vcov, df_t=df_t, n_obs=N, all_vcovs=all_vcovs,
            n_clusters={"^".join(panel): N} if own_panel else {},
            collin_vars=[], diagnostics=diagnostics, weights=None, weights_type="aweights",
            model="mixed logit", loglik=fit.ll, loglik_null=ll0,
            pseudo_r2=float(1 - fit.ll / ll0) if ll0 else np.nan, aic=aic, bic=bic,
            n_params=P, converged=fit.converged, case=case, panel=panel,
            n_cases=int(p0.n_cases), n_panels=N,
            random={nm: ("normal" if kind[xnames.index(nm)] == 1 else "lognormal")
                    for nm in rnames},
            draws=int(draws), halton=bool(halton), seed=int(seed), drop=int(drop),
            loglik_clogit=float(cfit.ll))
        p0.store.delete()
        return res
    finally:
        if not keep_intermediates:
            run.cleanup()
