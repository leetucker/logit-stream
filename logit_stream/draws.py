"""Draws of the parameters from their estimated sampling distribution
(`LogitResult.draw_parameters`), for multiple imputation and for simulating
the uncertainty of predictions.

The draws are from the normal approximation to the estimates' distribution:
the estimates as the mean and the inverse of the information as the
variance, with the fixed effect's levels included. That covariance matrix is
never formed. The fixed effect's block of the information is block-diagonal,
one block per level (a row is in one level), so the joint normal factorizes
exactly into the slopes' marginal and each level's conditional:

    theta* ~ N(theta_hat, V),        V the covariance of the slopes (and,
                                     for a random intercept, of log tau)
    a_g*  | theta* ~ N(a_hat_g - D_g^-1 J_g' (theta* - theta_hat), D_g^-1),

D_g the level's own block of the information and J_g its cross terms with
theta. The pieces are what the fit's Newton steps already eliminate the
fixed effect with, kept at the estimates; a draw reads no data.

Three cases:

* no fixed effect, no random intercepts: the coefficients from N(beta_hat,
  vcov), with the result's chosen `vcov` (iid or robust);
* one absorbed fixed effect: the coefficients from the chosen `vcov`, each
  level given them from the model-based blocks;
* a random intercept fitted by adaptive quadrature (optionally within a
  fixed effect): (coefficients, log tau) jointly from the inverse of their
  information net of the fixed effect, then each level given them. The
  group intercepts are not drawn: given a draw, each group's intercept has a
  conditional distribution given its rows, which the caller draws from (the
  drawn result's `predict` leaves them out).
* the multinomial logit's random intercepts fitted by adaptive quadrature:
  (coefficients, Sigma's log-Cholesky elements) jointly, from the inverse
  of the information in (coefficients, Sigma^-1) carried over by its
  Jacobian, then each fixed-effect level given them (linear in the drawn
  values). A draw of Sigma^-1 itself can be near singular, which gives
  Sigma a heavy tail; the log-Cholesky scale cannot.
"""

from __future__ import annotations

from dataclasses import replace

import numba as nb
import numpy as np
import polars as pl

DRAW_INFO_GB = 1.0          # the most memory kept on a result for its draws


def _factor(V):
    """A matrix L with L L' = V, for a symmetric positive semi-definite V
    (eigenvalues below zero, from rounding, count as zero)."""
    V = (np.asarray(V, float) + np.asarray(V, float).T) / 2
    w, Q = np.linalg.eigh(V)
    return Q * np.sqrt(np.clip(w, 0.0, None))[None, :]


def _batched_factor(D):
    """`_factor` for each of a stack of k x k matrices (G x k x k)."""
    if D.shape[1] == 1:
        return np.sqrt(np.clip(D, 0.0, None))
    D = (D + np.swapaxes(D, 1, 2)) / 2
    w, Q = np.linalg.eigh(D)
    return Q * np.sqrt(np.clip(w, 0.0, None))[:, None, :]


def _too_big(gb):
    return (f"the fixed effect's blocks for draw_parameters would take {gb:.1f} GB (more than "
            f"{DRAW_INFO_GB:g}); they were not kept")


def fe_draw_info(fe, theta, H):
    """What a one-way absorbed fixed effect (`_AbsorbedFE`) keeps for the
    draws: its estimated levels, and D^-1 and D^-1 J' from the information
    at the estimates."""
    G, m, P = fe.G, fe.m, fe.model.P
    gb = G * m * (m + P + 2) * 8 / 1e9
    if gb > DRAW_INFO_GB:
        return {"kind": None, "why": _too_big(gb)}
    H = fe._schur(H)
    free = np.asarray(fe.free, bool)
    Dinv = np.where(free[:, :, None] & free[:, None, :], H.Dinv, 0.0)
    from .core.newton import _Driver
    Vred = _Driver.inverse(H.red) if P else np.zeros((0, 0))
    return {"kind": "fe", "A": fe._effects(theta).copy(), "free": free, "Vred": Vred,
            "fixed": (fe.fixed if fe.model.kind == "mlogit" else None),
            "keep": np.asarray(fe.keep_g, bool), "S": _batched_factor(Dinv),
            "BD": H.BD * free[:, :, None], "P": P}


def adaptive_draw_info(ad, E, b, alpha, nu, nu_free):
    """What the adaptive random-intercept fit keeps for the draws: the
    covariance of (b, nu) net of the fixed effect, and each free level's
    information and cross terms."""
    P = ad.m.P
    sysm = ad._system(E, nu_free)
    if sysm is None:
        return {"kind": None, "why": "the fixed effect's information is not positive"}
    A, _, Jea, Daa, idx = sysm
    from .core.newton import _Driver
    V = _Driver.inverse(A)
    th = np.r_[b, nu] if nu_free else np.asarray(b, float).copy()
    out = {"kind": "adaptive", "theta": th, "V": V, "P": P, "nu": float(nu),
           "nu_free": bool(nu_free), "alpha": None}
    if alpha is not None and len(idx):
        out.update(alpha=np.asarray(alpha, float).copy(), idx=np.asarray(idx),
                   Jea=np.asarray(Jea, float), Daa=np.asarray(Daa, float),
                   keep=np.asarray(ad.fe.keep, bool))
    elif alpha is not None:
        out.update(alpha=np.asarray(alpha, float).copy(), idx=np.zeros(0, int),
                   keep=np.asarray(ad.fe.keep, bool))
    return out


def adaptive_mnl_draw_info(fit, E, b, alpha, lam):
    """What the multinomial adaptive fit (`_AdaptiveMNL`) keeps for the
    draws: the covariance of (b, lambda) net of the fixed effect (lambda
    Sigma^-1's elements), and each fixed-effect level's conditional
    covariance and its cross terms with them."""
    sysm = fit._system(E)
    if sysm is None:
        return {"kind": None, "why": "the fixed effect's information is not positive"}
    A, _, JaE, Dinv, free = sysm
    from .core.newton import _Driver
    # the draws are of (b, psi), psi Sigma's log-Cholesky elements (always a valid
    # Sigma, without the heavy tail that Sigma = Lambda^-1 has when a draw of Lambda
    # is near singular); their covariance from (b, lambda)'s by the Jacobian
    own, d, P = list(fit.own), fit.d, fit.m.P
    lam = np.asarray(lam, float)
    psi = _psi_of_lam(lam, own, d)
    Jac = np.empty((len(lam), len(lam)))
    for i in range(len(lam)):
        h = 1e-6 * max(abs(lam[i]), 1.0)
        e = np.zeros(len(lam))
        e[i] = h
        Jac[:, i] = (_psi_of_lam(lam + e, own, d) - _psi_of_lam(lam - e, own, d)) / (2 * h)
    T = np.eye(P + len(lam))
    T[P:, P:] = Jac
    V = _Driver.inverse(A)
    out = {"kind": "adaptive_mnl", "theta": np.r_[b, lam], "psi": np.r_[b, psi],
           "V": T @ V @ T.T, "P": P, "own": own, "d": d, "alpha": None}
    if alpha is not None:
        G, J = free.shape
        gb = G * J * (J + A.shape[0] + 2) * 8 / 1e9
        if gb > DRAW_INFO_GB:
            return {"kind": None, "why": _too_big(gb)}
        Dm = np.where(free[:, :, None] & free[:, None, :], Dinv, 0.0)
        # the levels' conditional mean, linear in (b, psi): the same variance as in
        # (b, lambda) (the map is invertible), and it is what the draws use
        M = np.einsum("gab,gbe->gae", Dm, JaE) @ np.linalg.inv(T)
        out.update(alpha=np.asarray(alpha, float).copy(), free=free,
                   M=M, S=_batched_factor(Dm),
                   keep=np.asarray(fit.fe.keep, bool))
    return out


def _lam_matrix(lam, own, d):
    L = np.zeros((d, d))
    for x, (j, k) in zip(lam, own):
        L[j, k] = L[k, j] = x
    return L


def _psi_of_lam(lam, own, d):
    """Sigma's log-Cholesky elements, in the order of `own` ((j, j): log
    L_jj; (j, k), j < k: L_kj), from Lambda = Sigma^-1's."""
    C = np.linalg.cholesky(np.linalg.inv(_lam_matrix(lam, own, d)))
    return np.array([np.log(C[j, j]) if j == k else C[k, j] for j, k in own])


def _sigma_of_psi(psi, own, d):
    C = np.zeros((d, d))
    for x, (j, k) in zip(psi, own):
        if j == k:
            C[j, j] = np.exp(x)
        else:
            C[k, j] = x
    return C @ C.T


# ------------------------------------------------------------------ drawing
def _fe_columns(spec):
    for name, values, kind in spec.get("effects", []):
        if kind == "fixed":
            return name, values
    return None, None


def _iid_factor(result, info):
    """The square root of the iid variance's small-sample factor (the binary
    logit's (N - 1) / (N - K); 1 for the other models), which the levels'
    conditional variance carries too, so that the joint covariance of a
    draw is the factor times the inverse information."""
    if info.get("kind") != "fe" or not info["P"] or not info["Vred"][0, 0] > 0:
        return 1.0
    Viid = result.all_vcovs["iid"]
    Viid = Viid[0] if isinstance(Viid, tuple) else Viid
    return float(np.sqrt(np.asarray(Viid)[0, 0] / info["Vred"][0, 0]))


def level_se(result):
    """The fixed effect's standard errors (G_kept x m, nan where a level is
    not estimated), from the same pieces as the draws: the level's variance
    given the coefficients plus what the coefficients' uncertainty adds,
    var(a_g) = D_g^-1 + (D_g^-1 J_g') V (J_g D_g^-1); None when not kept."""
    info = (result._spec or {}).get("draw")
    if not info or info.get("kind") not in ("fe", "adaptive", "adaptive_mnl", "alt_fe",
                                            "alt_slopes"):
        return None
    if info["kind"] == "alt_slopes" and not info["D"]:
        return None
    if info["kind"] in ("alt_fe", "alt_slopes"):
        # each constant less its group's reference: (e_j - e_r)' (Dinv + BD V BD') (e_j - e_r)
        ok, ref = info["alive"], info["ref"]
        j = np.flatnonzero(ok)
        r = ref[j]
        c = info["comp"][j]
        pos, s = info["comp_pos"][c], info["comp_size"][c]
        lj, lr = info["loc"][j], info["loc"][r]
        Di = info["Dinv"]
        own = Di[pos + lj * s + lj] + Di[pos + lr * s + lr] - 2 * Di[pos + lj * s + lr]
        dB = info["BD"][j] - info["BD"][r]
        Vd = info["V"] if info["kind"] == "alt_slopes" else np.asarray(result.vcov, float)
        add = np.einsum("np,pq,nq->n", dB, Vd, dB)
        var = np.full(len(ok), np.nan)
        var[j] = np.where(j == r, np.nan, own + add)
        return np.sqrt(np.clip(var, 0, None))[:, None]
    if info["kind"] == "adaptive_mnl":
        if info["alpha"] is None:
            return None
        S, M = info["S"], info["M"]
        var = (np.einsum("gjk,gjk->gj", S, S)
               + np.einsum("gjp,pq,gjq->gj", M, info["V"], M))
        return np.sqrt(np.where(info["free"], var, np.nan))[info["keep"]]
    if info["kind"] == "fe":
        c = _iid_factor(result, info)
        S = info["S"]
        own = np.einsum("gjk,gjk->gj", S, S) * c * c
        V = np.asarray(result.vcov, float)
        BD = info["BD"]
        add = np.einsum("gjp,pq,gjq->gj", BD, V, BD) if BD.shape[2] else 0.0
        var = np.where(info["free"], own + add, np.nan)
        return np.sqrt(var)[info["keep"]]
    if info["alpha"] is None:
        return None
    var = np.full(len(info["alpha"]), np.nan)
    idx = info["idx"]
    if len(idx):
        M = info["Jea"].T / info["Daa"][:, None]                    # n_free x dim theta
        var[idx] = 1.0 / info["Daa"] + np.einsum("ip,pq,iq->i", M, info["V"], M)
    return np.sqrt(var)[info["keep"]][:, None]


def _result_order(result, b):
    """A draw of the model's coefficients in the result's order: the binary
    logit reports its cutpoint (the model's last) as the intercept, minus it,
    first."""
    names = list(result.coefnames)
    if names and names[0] == "Intercept" and getattr(result, "family", None) in ("logit",
                                                                              "probit"):
        return np.r_[-b[-1], b[:-1]]
    return b


def draw(result, n, seed=None):
    """`n` copies of `result` at independent draws of its parameters (see
    the module docstring)."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    spec = result._spec or {}
    if spec.get("draw_index") is not None:
        raise ValueError("this result is already a draw; draw from the fitted result")
    info = spec.get("draw")
    if info is None:
        raise ValueError(f"draw_parameters is not available for this model ({result.model})")
    if info.get("kind") is None:
        raise ValueError(f"draw_parameters: {info['why']}")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    beta_hat = np.asarray(result.beta, float)
    P = len(beta_hat)
    kind = info["kind"]
    if kind == "adaptive_mnl":
        from ._experimental import experimental
        experimental("parameter draws of an adaptive fit with a vector of group effects (the "
                     "multinomial logit, or random slopes)", "checked against brute-force "
                     "integration only")
    fname, fcols = _fe_columns(spec)
    out = []
    c = _iid_factor(result, info)
    L = _factor(result.vcov if kind in ("beta", "beta_fixed", "fe", "alt_fe") else info["V"])
    for i in range(n):
        tau = None
        Sigma = None
        levels = None
        if kind == "adaptive_mnl":
            d, own = info["d"], info["own"]
            ps = info["psi"] + L @ rng.standard_normal(len(info["psi"]))
            beta = _result_order(result, ps[:P])
            Sigma = _sigma_of_psi(ps[P:], own, d)
            if info["alpha"] is not None:
                A = info["alpha"] - info["M"] @ (ps - info["psi"])
                z = rng.standard_normal(A.shape)
                A = A + (info["S"] @ z[..., None])[..., 0]
                A = np.where(info["free"], A, info["alpha"])
                levels = A[info["keep"]]
            out.append(_drawn(result, spec, i, beta, levels, None, fname, fcols, Sigma))
            continue
        if kind == "alt_fe":
            beta = beta_hat + L @ rng.standard_normal(P)
            A = info["alpha"] - info["BD"] @ (beta - beta_hat)
            z = rng.standard_normal(len(A))
            S, order, start = info["S"], info["level_order"], info["comp_start"]
            for s_, comps in info["by_size"]:
                k = len(comps)
                idx = info["comp_pos"][comps][:, None] + np.arange(s_ * s_)[None, :]
                lv = order[start[comps][:, None] + np.arange(s_)[None, :]]
                A[lv] += (S[idx].reshape(k, s_, s_) @ z[lv][..., None])[..., 0]
            ok, ref = info["alive"], info["ref"]
            A = np.where(ok, A - np.where(ok, A[np.maximum(ref, 0)], 0.0), np.nan)
            out.append(_drawn(result, spec, i, beta, A[:, None], None, fname, fcols))
            continue
        if kind in ("beta", "beta_fixed", "fe"):
            # beta_fixed (several absorbed fixed effects): the coefficients
            # drawn, the fixed effects held at their estimates
            beta = beta_hat + L @ rng.standard_normal(P)
            if kind == "fe":
                d = beta - beta_hat
                A = info["A"] - info["BD"] @ d                       # G x m
                z = rng.standard_normal(A.shape) * c
                S = info["S"]
                A = A + (S[:, :, 0] * z if S.shape[1] == 1 else (S @ z[..., None])[..., 0])
                if info["fixed"] is not None:
                    A = np.where(info["free"], A, info["fixed"])
                levels = A[info["keep"]]
        else:
            th = info["theta"] + L @ rng.standard_normal(len(info["theta"]))
            beta = _result_order(result, th[:P])
            if info["nu_free"]:
                tau = float(np.exp(th[P]))
            else:
                tau = float(np.exp(info["nu"]))
            if info["alpha"] is not None:
                alpha = info["alpha"].copy()
                idx = info["idx"]
                if len(idx):
                    d = th - info["theta"]
                    alpha[idx] = (alpha[idx] - (info["Jea"].T @ d) / info["Daa"]
                                  + rng.standard_normal(len(idx)) / np.sqrt(info["Daa"]))
                levels = alpha[info["keep"]][:, None]
        out.append(_drawn(result, spec, i, beta, levels, tau, fname, fcols))
    return out


def _drawn(result, spec, i, beta, levels, tau, fname, fcols, Sigma=None):
    s = dict(spec)
    s["draw_index"] = i
    s.pop("draw", None)
    if levels is not None and fname is not None:
        table = spec["fixef"][fname]
        cols = {}
        for j, c in enumerate(fcols):
            v = levels[:, j]
            cols[c] = pl.Series(c, np.where(np.isfinite(v), v, np.nan)).fill_nan(None)
        s["fixef"] = {fname: table.with_columns(**cols)}
    re = result.random_effects
    if re:
        # the group intercepts are not part of a draw
        s["effects"] = [e for e in spec.get("effects", [])
                        if e[2] not in ("random", "random_slopes")]
        s["ranef_base"] = spec.get("ranef_base", spec.get("ranef", {}))   # its groups
        s["ranef"] = {}
        re = {k: dict(v) for k, v in re.items()}
        if tau is not None:
            (name,) = re
            re[name]["sd"] = np.array([tau])
            re[name]["Sigma"] = np.array([[tau * tau]])
        if Sigma is not None:
            (name,) = re
            re[name]["Sigma"] = Sigma
            re[name]["sd"] = np.sqrt(np.diag(Sigma))
    diag = dict(result.diagnostics or {})
    diag["draw"] = i
    return replace(result, beta=np.asarray(beta, float), random_effects=re, diagnostics=diag,
                   _spec=s)


# ------------------------------------------------------------------ random slopes
def draw_slopes(result, n, rng, newdata=None):
    """`n` draws for the conditional logit with random slopes by region:
    beta and every region's slopes jointly from their normal approximation
    (given Sigma: N(estimates, the inverse of their information net of the
    constants)), the constants given them (as with absorbed constants
    alone), and Sigma from its own (log-Cholesky) normal approximation;
    with `newdata`, its regions the fit did not see get draws from N(0,
    that draw's Sigma)."""
    from .alt_slopes import _sigma_of_psi
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    spec = result._spec
    if spec.get("draw_index") is not None:
        raise ValueError("this result is already a draw; draw from the fitted result")
    info = spec["draw"]
    P, Pd, R, q = info["P"], info["Pd"], info["R"], info["q"]
    (region, zs), = spec["slopes"].items()
    table = spec["ranef"][region]
    cols = spec["fe_cols"][region]
    Lf = _factor(info["V"])
    Lp = _factor(info["Vpsi"]) if np.all(np.isfinite(info["Vpsi"])) else \
        np.zeros_like(info["Vpsi"])
    fname, fcols = _fe_columns(spec)
    new = None
    if newdata is not None:
        from .results import as_lazy as _lazy
        new = (_lazy(newdata).select(cols).unique()
               .join(table.select(cols).lazy(), on=cols, how="anti").collect().sort(cols))
    keep = info["keep"]
    out = []
    for i in range(n):
        coef = info["coef"] + Lf @ rng.standard_normal(Pd)
        beta = coef[:P]
        B = coef[P:].reshape(R, q)[keep]
        Sigma = _sigma_of_psi(info["psi"] + Lp @ rng.standard_normal(len(info["psi"])), q)
        levels = None
        if info["D"]:
            A = info["alpha"] - info["BD"] @ (coef - info["coef"])
            z = rng.standard_normal(len(A))
            S, order, start = info["S"], info["level_order"], info["comp_start"]
            for s_, comps in info["by_size"]:
                k = len(comps)
                idx = info["comp_pos"][comps][:, None] + np.arange(s_ * s_)[None, :]
                lv = order[start[comps][:, None] + np.arange(s_)[None, :]]
                A[lv] += (S[idx].reshape(k, s_, s_) @ z[lv][..., None])[..., 0]
            ok, ref = info["alive"], info["ref"]
            levels = np.where(ok, A - np.where(ok, A[np.maximum(ref, 0)], 0.0), np.nan)[:, None]
        s = dict(spec)
        s["draw_index"] = i
        s.pop("draw", None)
        if levels is not None and fname is not None:
            ft = spec["fixef"][fname]
            v = levels[:, 0]
            s["fixef"] = {fname: ft.with_columns(
                pl.Series(fcols[0], np.where(np.isfinite(v), v, np.nan)).fill_nan(None))}
        t = table.with_columns(*[pl.Series(f"re_{region}:{zn}", B[:, j])
                                 for j, zn in enumerate(zs)])
        if new is not None and new.height:
            Lc = np.linalg.cholesky(Sigma)
            Bn = rng.standard_normal((new.height, q)) @ Lc.T
            add = new.with_columns(
                *[pl.Series(f"re_{region}:{zn}", Bn[:, j]) for j, zn in enumerate(zs)],
                *[pl.Series(f"sd_{region}:{zn}", np.full(new.height, np.sqrt(Sigma[j, j])))
                  for j, zn in enumerate(zs)],
                pl.lit(0, dtype=t.schema["n_obs"]).alias("n_obs"),
                pl.lit(0, dtype=t.schema["n_chosen"]).alias("n_chosen"))
            add = add.with_columns(pl.col(c).cast(t.schema[c]) for c in cols)
            t = pl.concat([t, add.select(t.columns)])
        s["ranef"] = {region: t}
        s["ranef_drawn"] = True
        re = {k: dict(v) for k, v in result.random_effects.items()}
        re[region]["Sigma"] = Sigma
        re[region]["sd"] = np.sqrt(np.diag(Sigma))
        diag = dict(result.diagnostics or {})
        diag["draw"] = i
        out.append(replace(result, beta=np.asarray(beta, float), random_effects=re,
                           diagnostics=diag, _spec=s))
    return out


# ------------------------------------------------------------------ group intercepts
RANEF_GRID = 101            # points of each group's grid
RANEF_SPAN = 8.0            # the grid's half-width, in conditional sds at the mode
RANEF_CHUNK_CELLS = 8_000_000   # rows x grid points held at once


def _ranef_setup(result):
    """Check that `draw_ranef` applies, and return (grouping name, its key
    columns, the link)."""
    from .results import MultinomialLogitResult, OrderedLogitResult
    re = result.random_effects
    if not re:
        raise ValueError("draw_ranef needs a model with random intercepts (random=)")
    mnl = isinstance(result, MultinomialLogitResult)
    if not (mnl or isinstance(result, OrderedLogitResult)):
        raise ValueError("draw_ranef is available for the binary, ordered and multinomial logit "
                         "and probit")
    if mnl and (result._spec or {}).get("free") is not None:
        raise ValueError("draw_ranef is not available with own_covariates=")
    if len(re) != 1:
        raise ValueError(f"draw_ranef needs one random grouping, not {len(re)} ({list(re)})")
    (name,) = re
    if re[name].get("structure") not in (None, "tree"):
        raise ValueError("draw_ranef needs nested (not crossed) groupings")
    if mnl:
        return name, list(result._spec["fe_cols"][name]), None
    from .core.models import LINKS
    return name, list(result._spec["fe_cols"][name]), LINKS[result.link]


def _ordered_row_terms(link, y, z, cuts):
    """log P(y | z) of each row and its first two derivatives in z, for the
    ordered model P(y <= j) = F(c_j - z); y the level's code."""
    c = np.r_[-np.inf, cuts, np.inf]
    up, lo = c[y + 1] - z, c[y] - z
    Fu, Fl = link.cdf(up), link.cdf(lo)
    Su, Sl = link.sf(up), link.sf(lo)
    # the probability from the nearer tail, for precision
    p = np.where(lo > 0, Sl - Su, Fu - Fl)
    p = np.maximum(p, 1e-300)
    fu = np.where(np.isfinite(up), link.pdf(up, Fu, Su), 0.0)
    fl = np.where(np.isfinite(lo), link.pdf(lo, Fl, Sl), 0.0)
    du = np.where(np.isfinite(up), link.dpdf(up, fu, Fu), 0.0)
    dl = np.where(np.isfinite(lo), link.dpdf(lo, fl, Fl), 0.0)
    d1 = -(fu - fl) / p                       # d log p / dz
    d2 = (du - dl) / p - d1 * d1              # d2 log p / dz2
    return np.log(p), d1, d2


def _ordered_logp(link, y, z, cuts):
    c = np.r_[-np.inf, cuts, np.inf]
    up, lo = c[y + 1] - z, c[y] - z
    p = np.where(lo > 0, link.sf(lo) - link.sf(up), link.cdf(up) - link.cdf(lo))
    return np.log(np.maximum(p, 1e-300))


def _draw_chunk(link, g, starts, y, w, eta, cuts, tau, u, grid_pts=RANEF_GRID,
                span=RANEF_SPAN):
    for sp in (span, 2 * span, 4 * span):
        out = _draw_chunk1(link, g, starts, y, w, eta, cuts, tau, u, grid_pts, sp)
        if out is not None:
            return out
    raise RuntimeError("draw_ranef: a group's conditional distribution reaches past its grid")


def _draw_chunk1(link, g, starts, y, w, eta, cuts, tau, u, grid_pts, span):
    """Each group's draw (and its conditional mean and sd) from
    p(d | its rows) proportional to prod_i p(y_i | eta_i + d)^w_i N(d; 0, tau^2),
    for the rows of whole groups sorted by group: g the rows' local group
    index (0..n-1), `starts` each group's first row, u one uniform per
    group. Newton for the mode, then the exact density on a grid around it,
    sampled by its inverse cdf (each point's mass spread evenly over its
    cell)."""
    n = len(starts)
    t2 = tau * tau
    d = np.zeros(n)
    for _ in range(100):
        _, d1, d2 = _ordered_row_terms(link, y, eta + d[g], cuts)
        s = np.add.reduceat(w * d1, starts) - d / t2
        h = -np.add.reduceat(w * d2, starts) + 1 / t2
        step = np.clip(s / h, -2.0, 2.0)
        d += step
        if np.max(np.abs(step), initial=0.0) < 1e-9:
            break
    _, _, d2 = _ordered_row_terms(link, y, eta + d[g], cuts)
    sd0 = 1 / np.sqrt(np.maximum(-np.add.reduceat(w * d2, starts) + 1 / t2, 1e-300))
    zs = np.linspace(-span, span, grid_pts)
    grid = d[:, None] + sd0[:, None] * zs[None, :]                      # n x K
    lp = np.add.reduceat(w[:, None] * _ordered_logp(link, y[:, None], eta[:, None] + grid[g],
                                                    cuts), starts, axis=0)
    lp -= 0.5 * grid * grid / t2
    lp -= lp.max(axis=1, keepdims=True)
    pmf = np.exp(lp)
    pmf /= pmf.sum(axis=1, keepdims=True)
    edge = pmf[:, 0] + pmf[:, -1]
    if edge.max(initial=0.0) > 1e-9:
        return None                             # (wider, then)
    mean = (pmf * grid).sum(axis=1)
    sd = np.sqrt(np.maximum((pmf * (grid - mean[:, None]) ** 2).sum(axis=1), 0.0))
    cdf = np.cumsum(pmf, axis=1)
    k = np.minimum((cdf < u[:, None]).sum(axis=1), grid_pts - 1)
    r = np.arange(n)
    lo_c = np.where(k > 0, cdf[r, np.maximum(k - 1, 0)], 0.0)
    frac = np.clip((u - lo_c) / np.maximum(cdf[r, k] - lo_c, 1e-300), 0.0, 1.0)
    h = grid[:, 1] - grid[:, 0]
    return grid[r, k] + (frac - 0.5) * h, mean, sd


def _ranef_rows(result, name, keys):
    """The estimation sample as a lazy frame of what the draws need: the
    group's row in the ranef table (`__g`), the outcome's code, the weight,
    the covariates, the fixed effect's row in the fixef table, the offset;
    sorted by group."""
    spec = result._spec
    frame = result._in_sample(spec["source"])
    gtab = (spec.get("ranef_base") or spec["ranef"])[name].select(keys).with_row_index("__g")
    frame = frame.join(gtab.lazy(), on=keys, how="inner")
    levels = list(result.levels)
    cols = [pl.col("__g"),
            spec["yexpr"].replace_strict(levels, list(range(len(levels))),
                                         return_dtype=pl.Int64).alias("__y")]
    if result.weights:
        cols.append(pl.col(result.weights).cast(pl.Float64).alias("__w"))
    xs = result._xexprs()
    cols += [ex.cast(pl.Float64).alias(f"__x{j}") for j, (_, ex) in enumerate(xs)]
    fname = result.fe_names[0] if result.fe_names else None
    if fname:
        fcols = spec["fe_cols"][fname]
        ftab = spec["fixef"][fname].select(fcols).with_row_index("__f")
        frame = frame.join(ftab.lazy(), on=fcols, how="inner")
        cols.append(pl.col("__f"))
    if spec.get("offset") is not None:
        cols.append(spec["offset"].cast(pl.Float64).alias("__off"))
    return frame.select(cols).sort("__g"), len(xs), fname


def _location(res, X, f, off, fname):
    """The rows' linear predictor without the group intercept, and the
    cutpoints, at `res`'s parameters (the binary logit's intercept as minus
    its cutpoint)."""
    beta = np.asarray(res.beta, float)
    names = list(res.coefnames)
    icpt = bool(names) and names[0] == "Intercept"
    k = X.shape[1]
    b = beta[1:1 + k] if icpt else beta[:k]
    eta = X @ b
    if fname:
        a = res._spec["fixef"][fname][f"fe_{fname}"].to_numpy().astype(float)
        eta = eta + a[f]
    if off is not None:
        eta = eta + off
    return eta, np.asarray(res.cutpoints, float)


def draw_ranef_many(results, seed=None, newdata=None, chunk_cells=RANEF_CHUNK_CELLS):
    """For each result in `results` (copies of one fit, at the estimates or
    at draws of the parameters), one draw of every group's intercept from
    its distribution given the group's rows and that result's parameters,
    in one read of the estimation sample; returns copies whose `ranef()` is
    the draw and whose `predict` adds it. Groups in `newdata` that the fit
    did not see draw from N(0, sd^2)."""
    first = results[0]
    name, keys, link = _ranef_setup(first)
    from ._experimental import experimental
    experimental("draws of the group effects given their rows (draw_ranef, "
                 "draw_parameters(ranef=True))", "checked against brute-force posteriors only")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    if (first._spec or {}).get("slope_layout"):
        return _draw_ranef_many_slopes(results, rng, newdata, name, keys, chunk_cells)
    if link is None:
        return _draw_ranef_many_mnl(results, rng, newdata, name, keys, chunk_cells)
    spec = first._spec
    base = spec.get("ranef_base") or spec["ranef"]
    table = base[name]
    G = table.height
    M = len(results)
    taus = np.array([float(r.random_effects[name]["sd"][0]) for r in results])
    U = rng.random((M, G))                     # one uniform per group and draw
    draws, means, sds = np.zeros((M, G)), np.zeros((M, G)), np.zeros((M, G))
    seen = np.zeros(G, bool)
    lf, k, fname = _ranef_rows(first, name, keys)
    wscale = _weight_scale(first)
    rows_per_chunk = max(int(chunk_cells // RANEF_GRID), 1000)

    def process(df):
        gi = df["__g"].to_numpy()
        starts = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
        gid = gi[starts]
        g = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(gi)]))
        y = df["__y"].to_numpy()
        w = df["__w"].to_numpy() * wscale if "__w" in df.columns else np.ones(len(gi))
        X = (df.select([f"__x{j}" for j in range(k)]).to_numpy().astype(float) if k
             else np.zeros((len(gi), 0)))
        f = df["__f"].to_numpy() if fname else None
        off = df["__off"].to_numpy() if "__off" in df.columns else None
        for m, res in enumerate(results):
            eta, cuts = _location(res, X, f, off, fname)
            dm, mu, sd = _draw_chunk(link, g, starts, y, w, eta, cuts, taus[m], U[m, gid])
            draws[m, gid], means[m, gid], sds[m, gid] = dm, mu, sd
        seen[gid] = True

    _by_whole_groups(lf, rows_per_chunk, process)
    # a group without rows (all its weights zero, say): its prior
    for m in range(M):
        miss = ~seen
        draws[m, miss] = taus[m] * _probit(U[m, miss])
        sds[m, miss] = taus[m]
    new = None
    if newdata is not None:
        from .results import as_lazy as _lazy
        # sorted, so that each new group's draw does not depend on the join's
        # order (or on newdata's)
        new = (_lazy(newdata).select(keys).unique()
               .join(table.select(keys).lazy(), on=keys, how="anti").collect().sort(keys))
    out = []
    for m, res in enumerate(results):
        t = table.with_columns(pl.Series(f"re_{name}", draws[m]),
                               pl.Series(f"sd_{name}", sds[m]))
        if new is not None and new.height:
            z = rng.standard_normal(new.height) * taus[m]
            add = new.with_columns(pl.Series(f"re_{name}", z),
                                   pl.Series(f"sd_{name}", np.full(new.height, taus[m])),
                                   pl.lit(0, dtype=t.schema["n_obs"]).alias("n_obs"))
            add = add.with_columns(pl.col(c).cast(t.schema[c]) for c in keys)
            t = pl.concat([t, add.select(t.columns)])
        s = dict(res._spec)
        s["ranef_base"] = base
        s["ranef"] = {name: t}
        s["effects"] = ([e for e in spec["effects"] if e[2] != "random"]
                        + [(name, [f"re_{name}"], "random")])
        s["ranef_drawn"] = True
        diag = dict(res.diagnostics or {})
        diag["ranef_mean"] = means[m]
        out.append(replace(res, diagnostics=diag, _spec=s))
    return out


def _by_whole_groups(lf, rows_per_chunk, process):
    """Stream `lf` (sorted by `__g`) in batches, calling `process` on whole
    groups only (a batch's last group waits for the next batch)."""
    carry = None
    for batch in lf.collect_batches(chunk_size=rows_per_chunk):
        if carry is not None:
            batch = pl.concat([carry, batch])
        if batch.height == 0:
            continue
        last = batch["__g"][-1]
        tail = batch["__g"] == last
        carry = batch.filter(tail)
        body = batch.filter(~tail)
        if body.height:
            process(body)
    if carry is not None and carry.height:
        process(carry)


def _weight_scale(first):
    """Analytic weights rescaled to a mean of one over the usable rows."""
    if first.weights and first.weights_type != "fweights":
        st = first._spec["source"].filter(first._spec["usable"]).select(
            pl.len().alias("n"), pl.col(first.weights).sum().alias("w")).collect()
        return float(st["n"][0]) / float(st["w"][0])
    return 1.0


def _draw_ranef_many_mnl(results, rng, newdata, name, keys, chunk_cells):
    """`draw_ranef_many` for the multinomial logit: each group's (J - 1)
    intercepts by importance resampling (`_mnl_sir_kernel`)."""
    first = results[0]
    spec = first._spec
    base = spec.get("ranef_base") or spec["ranef"]
    table = base[name]
    G = table.height
    M = len(results)
    levels = list(first.levels)
    d = len(levels) - 1
    Sigmas = [np.asarray(r.random_effects[name]["Sigma"], float) for r in results]
    Lams = [np.linalg.inv(S) for S in Sigmas]
    keys64 = rng.integers(0, 2**63, M)
    draws, means, sds = np.zeros((M, G, d)), np.zeros((M, G, d)), np.zeros((M, G, d))
    ess = np.full((M, G), np.nan)
    seen = np.zeros(G, bool)
    lf, k, fname = _ranef_rows(first, name, keys)
    wscale = _weight_scale(first)
    rows_per_chunk = max(int(chunk_cells // MNL_CANDIDATES), 1000)

    def process(df):
        gi = df["__g"].to_numpy()
        starts = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
        gid = gi[starts]
        y = df["__y"].to_numpy()
        w = df["__w"].to_numpy() * wscale if "__w" in df.columns else np.ones(len(gi))
        X = (df.select([f"__x{j}" for j in range(k)]).to_numpy().astype(float) if k
             else np.zeros((len(gi), 0)))
        f = df["__f"].to_numpy() if fname else None
        st = np.r_[starts, len(gi)]
        for m, res in enumerate(results):
            E = _mnl_location(res, X, f, fname)
            dm, mu, sd, es = _mnl_draw(E, y, w, st, gid, Lams[m], int(keys64[m]))
            draws[m, gid], means[m, gid], sds[m, gid], ess[m, gid] = dm, mu, sd, es
        seen[gid] = True

    _by_whole_groups(lf, rows_per_chunk, process)
    for m in range(M):            # a group without rows: its prior
        miss = np.flatnonzero(~seen)
        if len(miss):
            C = np.linalg.cholesky(Sigmas[m])
            draws[m, miss] = rng.standard_normal((len(miss), d)) @ C.T
            sds[m, miss] = np.sqrt(np.diag(Sigmas[m]))
    new = None
    if newdata is not None:
        from .results import as_lazy as _lazy
        # sorted, so that each new group's draw does not depend on the join's
        # order (or on newdata's)
        new = (_lazy(newdata).select(keys).unique()
               .join(table.select(keys).lazy(), on=keys, how="anti").collect().sort(keys))
    rcols = [f"re_{name}:{lv}" for lv in levels]
    scols = [f"sd_{name}:{lv}" for lv in levels]
    out = []
    for m, res in enumerate(results):
        def cols(Dr, Sd):
            n = Dr.shape[0]
            return ([pl.Series(rcols[0], np.zeros(n))]
                    + [pl.Series(rcols[j + 1], Dr[:, j]) for j in range(d)]
                    + [pl.Series(scols[0], np.zeros(n))]
                    + [pl.Series(scols[j + 1], Sd[:, j]) for j in range(d)])
        t = table.with_columns(cols(draws[m], sds[m]))
        if new is not None and new.height:
            C = np.linalg.cholesky(Sigmas[m])
            z = rng.standard_normal((new.height, d)) @ C.T
            sdp = np.tile(np.sqrt(np.diag(Sigmas[m])), (new.height, 1))
            add = new.with_columns(cols(z, sdp)).with_columns(
                pl.lit(0, dtype=t.schema["n_obs"]).alias("n_obs"))
            add = add.with_columns(pl.col(c).cast(t.schema[c]) for c in keys)
            t = pl.concat([t, add.select(t.columns)])
        s = dict(res._spec)
        s["ranef_base"] = base
        s["ranef"] = {name: t}
        s["effects"] = ([e for e in spec["effects"] if e[2] != "random"]
                        + [(name, rcols, "random")])
        s["ranef_drawn"] = True
        diag = dict(res.diagnostics or {})
        diag["ranef_mean"] = means[m]
        diag["ranef_ess"] = ess[m]
        out.append(replace(res, diagnostics=diag, _spec=s))
    return out


def _draw_ranef_many_slopes(results, rng, newdata, name, keys, chunk_cells):
    """`draw_ranef_many` with random slopes: each group's effects (its
    intercepts and slopes, `slope_layout`) by importance resampling given its
    rows, the multinomial or the ordered logit (`_slopes_sir_kernel`)."""
    from .results import MultinomialLogitResult
    first = results[0]
    spec = first._spec
    base = spec.get("ranef_base") or spec["ranef"]
    table = base[name]
    G = table.height
    M = len(results)
    mnl = isinstance(first, MultinomialLogitResult)
    layout = spec["slope_layout"][name]
    values = spec["slope_values"][name]
    terms = list(first.random_effects[name]["terms"])
    xcols = list(spec["xcols"])
    zidx = np.array([-1 if t == "1" else xcols.index(t) for t in terms], np.int64)
    r, q = len(terms), len(layout)
    Sigmas = [np.asarray(r_.random_effects[name]["Sigma"], float) for r_ in results]
    Lams = [np.linalg.inv(S) for S in Sigmas]
    keys64 = rng.integers(0, 2**63, M)
    draws, means, sds = np.zeros((M, G, q)), np.zeros((M, G, q)), np.zeros((M, G, q))
    ess = np.full((M, G), np.nan)
    seen = np.zeros(G, bool)
    lf, k, fname = _ranef_rows(first, name, keys)
    wscale = _weight_scale(first)
    rows_per_chunk = max(int(chunk_cells // MNL_CANDIDATES), 1000)

    def process(df):
        gi = df["__g"].to_numpy()
        starts = np.flatnonzero(np.r_[True, gi[1:] != gi[:-1]])
        gid = gi[starts]
        y = df["__y"].to_numpy()
        w = df["__w"].to_numpy() * wscale if "__w" in df.columns else np.ones(len(gi))
        X = (df.select([f"__x{j}" for j in range(k)]).to_numpy().astype(float) if k
             else np.zeros((len(gi), 0)))
        f = df["__f"].to_numpy() if fname else None
        off = df["__off"].to_numpy() if "__off" in df.columns else None
        Z = np.column_stack([np.ones(len(gi)) if c < 0 else X[:, c] for c in zidx])
        st = np.r_[starts, len(gi)]
        for m, res in enumerate(results):
            if mnl:
                E, cuts = _mnl_location(res, X, f, fname), np.zeros(0)
            else:
                eta, cuts = _location(res, X, f, off, fname)
                E = eta[:, None]
            dm, mu, sd, es = _slopes_draw(0 if mnl else 1, E, cuts, Z, r, y, w, st, gid,
                                          Lams[m], int(keys64[m]))
            draws[m, gid], means[m, gid], sds[m, gid], ess[m, gid] = dm, mu, sd, es
        seen[gid] = True

    _by_whole_groups(lf, rows_per_chunk, process)
    for m in range(M):            # a group without rows: its prior
        miss = np.flatnonzero(~seen)
        if len(miss):
            C = np.linalg.cholesky(Sigmas[m])
            draws[m, miss] = rng.standard_normal((len(miss), q)) @ C.T
            sds[m, miss] = np.sqrt(np.diag(Sigmas[m]))
    new = None
    if newdata is not None:
        from .results import as_lazy as _lazy
        new = (_lazy(newdata).select(keys).unique()
               .join(table.select(keys).lazy(), on=keys, how="anti").collect().sort(keys))
    scols = ["sd_" + c[3:] for c in values]
    out = []
    for m, res in enumerate(results):
        def cols(Dr, Sd):
            return ([pl.Series(c, Dr[:, j]) for j, c in enumerate(values)]
                    + [pl.Series(c, Sd[:, j]) for j, c in enumerate(scols)])
        t = table.with_columns(cols(draws[m], sds[m]))
        if new is not None and new.height:
            C = np.linalg.cholesky(Sigmas[m])
            z = rng.standard_normal((new.height, q)) @ C.T
            sdp = np.tile(np.sqrt(np.diag(Sigmas[m])), (new.height, 1))
            add = new.with_columns(cols(z, sdp)).with_columns(
                pl.lit(0, dtype=t.schema["n_obs"]).alias("n_obs"))
            add = add.with_columns(pl.col(c).cast(t.schema[c]) for c in keys)
            t = pl.concat([t, add.select(t.columns)])
        s = dict(res._spec)
        s["ranef_base"] = base
        s["ranef"] = {name: t}
        s["effects"] = ([e for e in spec["effects"] if e[2] not in ("random", "random_slopes")]
                        + [(name, values, "random_slopes")])
        s["ranef_drawn"] = True
        diag = dict(res.diagnostics or {})
        diag["ranef_mean"] = means[m]
        diag["ranef_ess"] = ess[m]
        out.append(replace(res, diagnostics=diag, _spec=s))
    return out


@nb.njit(cache=True)
def _sl_rows(kind, E0, cuts, Z, r, y, w, r0, r1, a, Lam, s, H, need):
    """A group's log posterior at a (up to a constant), and with `need` its
    gradient s and information H (prior included): the multinomial (kind 0;
    level j's predictor E0[i, j] + sum_t Z[i, t] a[(j - 1) r + t]) or the
    ordered logit (kind 1; E0[i, 0] + Z[i] . a, the cutpoints `cuts`)."""
    q = len(a)
    lp = 0.0
    if need:
        for j in range(q):
            s[j] = 0.0
            for k in range(q):
                s[j] -= Lam[j, k] * a[k]
                H[j, k] = Lam[j, k]
    if kind == 0:
        J = E0.shape[1]
        p = np.empty(J)
        e = np.empty(J)
        for i in range(r0, r1):
            mx = -np.inf
            for j in range(J):
                v = E0[i, j]
                if j > 0:
                    for t in range(r):
                        v += Z[i, t] * a[(j - 1) * r + t]
                e[j] = v
                if v > mx:
                    mx = v
            den = 0.0
            for j in range(J):
                p[j] = np.exp(e[j] - mx)
                den += p[j]
            for j in range(J):
                p[j] /= den
            lp += w[i] * (e[y[i]] - mx - np.log(den))
            if need:
                for la in range(J - 1):
                    ra = (1.0 if y[i] == la + 1 else 0.0) - p[la + 1]
                    for ta in range(r):
                        ia = la * r + ta
                        s[ia] += w[i] * ra * Z[i, ta]
                        for lc in range(J - 1):
                            wac = (p[la + 1] if la == lc else 0.0) - p[la + 1] * p[lc + 1]
                            for tc in range(r):
                                H[ia, lc * r + tc] += w[i] * wac * Z[i, ta] * Z[i, tc]
    else:
        C = len(cuts)
        for i in range(r0, r1):
            eta = E0[i, 0]
            for t in range(r):
                eta += Z[i, t] * a[t]
            yi = y[i]
            Fh, fh, gh = 1.0, 0.0, 0.0
            if yi < C:
                Fh = 1.0 / (1.0 + np.exp(-(cuts[yi] - eta)))
                fh = Fh * (1.0 - Fh)
                gh = fh * (1.0 - 2.0 * Fh)
            Fl, fl, gl = 0.0, 0.0, 0.0
            if yi > 0:
                Fl = 1.0 / (1.0 + np.exp(-(cuts[yi - 1] - eta)))
                fl = Fl * (1.0 - Fl)
                gl = fl * (1.0 - 2.0 * Fl)
            pr = max(Fh - Fl, 1e-300)
            lp += w[i] * np.log(pr)
            if need:
                d1 = (fl - fh) / pr                       # d log p / d eta
                hh = d1 * d1 - (gh - gl) / pr             # -d2 log p / d eta2
                for ta in range(r):
                    s[ta] += w[i] * d1 * Z[i, ta]
                    for tc in range(r):
                        H[ta, tc] += w[i] * hh * Z[i, ta] * Z[i, tc]
    quad = 0.0
    for j in range(q):
        for k in range(q):
            quad += a[j] * Lam[j, k] * a[k]
    return lp - 0.5 * quad


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _slopes_sir_kernel(kind, E0, cuts, Z, r, y, w, st, gid, Lam, key, n_cand, nu, draw, mean,
                       sd, ess):
    """`_mnl_sir_kernel` for a group's effects with a row design (see
    `_sl_rows`)."""
    G = len(gid)
    q = Lam.shape[0]
    for g in nb.prange(G):
        r0, r1 = st[g], st[g + 1]
        a = np.zeros(q)
        s = np.empty(q)
        H = np.empty((q, q))
        for _ in range(100):
            _sl_rows(kind, E0, cuts, Z, r, y, w, r0, r1, a, Lam, s, H, True)
            step = np.linalg.solve(H, s)
            big = 0.0
            for j in range(q):
                big = max(big, abs(step[j]))
            if big > 2.0:
                step *= 2.0 / big
            a += step
            if big < 1e-10:
                break
        _sl_rows(kind, E0, cuts, Z, r, y, w, r0, r1, a, Lam, s, H, True)
        V = np.linalg.inv(H)
        R = np.linalg.cholesky((V + V.T) / 2)
        lw = np.empty(n_cand)
        cand = np.empty((n_cand, q))
        z = np.empty(q)
        cnt = 0
        for k in range(n_cand):
            nz = q + nu
            zz = np.empty(nz + 1)
            for t in range(0, nz, 2):
                u1 = _uniform(key, gid[g], cnt)
                u2 = _uniform(key, gid[g], cnt + 1)
                cnt += 2
                rr = np.sqrt(-2.0 * np.log(u1))
                zz[t] = rr * np.cos(2 * np.pi * u2)
                zz[t + 1] = rr * np.sin(2 * np.pi * u2)
            c2 = 0.0
            for t in range(q, nz):
                c2 += zz[t] * zz[t]
            sc = np.sqrt(nu / c2)
            qq = 0.0
            for j in range(q):
                z[j] = zz[j] * sc
                qq += z[j] * z[j]
            for j in range(q):
                x = a[j]
                for l in range(j + 1):
                    x += R[j, l] * z[l]
                cand[k, j] = x
            lw[k] = (_sl_rows(kind, E0, cuts, Z, r, y, w, r0, r1, cand[k], Lam, s, H, False)
                     + 0.5 * (nu + q) * np.log1p(qq / nu))
        mxw = lw.max()
        tot, tot2 = 0.0, 0.0
        for k in range(n_cand):
            lw[k] = np.exp(lw[k] - mxw)
            tot += lw[k]
            tot2 += lw[k] * lw[k]
        ess[g] = tot * tot / tot2
        for j in range(q):
            m1, m2 = 0.0, 0.0
            for k in range(n_cand):
                m1 += lw[k] * cand[k, j]
                m2 += lw[k] * cand[k, j] * cand[k, j]
            m1 /= tot
            mean[g, j] = m1
            sd[g, j] = np.sqrt(max(m2 / tot - m1 * m1, 0.0))
        u = _uniform(key, gid[g], cnt) * tot
        acc = 0.0
        pick = n_cand - 1
        for k in range(n_cand):
            acc += lw[k]
            if acc >= u:
                pick = k
                break
        for j in range(q):
            draw[g, j] = cand[pick, j]


def _slopes_draw(kind, E, cuts, Z, r, y, w, st, gid, Lam, key):
    n = len(gid)
    q = Lam.shape[0]
    out = [np.empty((n, q)), np.empty((n, q)), np.empty((n, q)), np.empty(n)]
    args = [np.ascontiguousarray(E, float), np.ascontiguousarray(cuts, float),
            np.ascontiguousarray(Z, float), int(r), y.astype(np.int64),
            np.ascontiguousarray(w, float)]
    _slopes_sir_kernel(kind, *args, st.astype(np.int64), gid.astype(np.int64),
                       np.ascontiguousarray(Lam), np.uint64(key), MNL_CANDIDATES, MNL_T_DF,
                       *out)
    draw, mean, sd, ess = out
    low = np.flatnonzero(ess < MNL_ESS_MIN)
    if len(low):
        rows = np.concatenate([np.arange(st[g], st[g + 1]) for g in low])
        st2 = np.r_[0, np.cumsum(st[low + 1] - st[low])]
        o2 = [np.empty((len(low), q)), np.empty((len(low), q)), np.empty((len(low), q)),
              np.empty(len(low))]
        _slopes_sir_kernel(kind, args[0][rows], args[1], args[2][rows], args[3],
                           args[4][rows], args[5][rows], st2.astype(np.int64),
                           gid[low].astype(np.int64), np.ascontiguousarray(Lam),
                           np.uint64(key ^ 0x5BD1E995), 16 * MNL_CANDIDATES, MNL_T_DF, *o2)
        draw[low], mean[low], sd[low], ess[low] = o2
    return draw, mean, sd, ess


def _probit(u):
    from scipy.special import ndtri
    return ndtri(np.clip(u, 1e-16, 1 - 1e-16))


# ------------------------------------------------------------------ multinomial intercepts
MNL_CANDIDATES = 1000       # the importance-resampling candidates per group and draw
MNL_ESS_MIN = 200           # fewer effective candidates: the group is redone with 16x
MNL_T_DF = 5                # the proposal's degrees of freedom
_M64 = (1 << 64) - 1


@nb.njit(cache=True, inline="always")
def _mix(x):
    """splitmix64's finalizer: a counter to 64 well-mixed bits."""
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@nb.njit(cache=True, inline="always")
def _uniform(key, g, c):
    """A uniform on (0, 1) from (key, group, counter)."""
    x = _mix(key ^ _mix(np.uint64(g) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(c)))
    return (np.float64(x >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0)


@nb.njit(cache=True)
def _mnl_logpost(E0, y, w, r0, r1, a, Lam):
    """A group's log posterior density at a (up to a constant)."""
    J = E0.shape[1]
    d = J - 1
    lp = 0.0
    for i in range(r0, r1):
        mx = E0[i, 0]
        for j in range(1, J):
            if E0[i, j] + a[j - 1] > mx:
                mx = E0[i, j] + a[j - 1]
        den = 0.0
        for j in range(J):
            den += np.exp(E0[i, j] + (a[j - 1] if j > 0 else 0.0) - mx)
        ey = E0[i, y[i]] + (a[y[i] - 1] if y[i] > 0 else 0.0)
        lp += w[i] * (ey - mx - np.log(den))
    q = 0.0
    for j in range(d):
        for k in range(d):
            q += a[j] * Lam[j, k] * a[k]
    return lp - 0.5 * q


@nb.njit(parallel=True, cache=True, error_model="numpy")
def _mnl_sir_kernel(E0, y, w, st, gid, Lam, key, n_cand, nu, draw, mean, sd, ess):
    """Each group's draw of its intercepts (levels 1..J-1) from
    p(a | rows) proportional to prod_i P(y_i | eta_i + a)^w_i N(a; 0, Lam^-1),
    by importance resampling: Newton for the mode, `n_cand` candidates from
    a multivariate t (`nu` degrees of freedom) at the mode with the inverse
    curvature as its scale, one picked by its weight. The random numbers
    are a hash of (key, the group's id, a counter), so they do not depend on
    how the rows are read. Also each group's weighted mean, sd and the
    weights' effective number."""
    G = len(gid)
    J = E0.shape[1]
    d = J - 1
    for g in nb.prange(G):
        r0, r1 = st[g], st[g + 1]
        a = np.zeros(d)
        H = np.zeros((d, d))
        p = np.empty(J)
        for _ in range(100):
            s = -(Lam @ a)
            H[:, :] = Lam
            for i in range(r0, r1):
                mx = E0[i, 0]
                for j in range(1, J):
                    if E0[i, j] + a[j - 1] > mx:
                        mx = E0[i, j] + a[j - 1]
                den = 0.0
                for j in range(J):
                    p[j] = np.exp(E0[i, j] + (a[j - 1] if j > 0 else 0.0) - mx)
                    den += p[j]
                for j in range(J):
                    p[j] /= den
                for j in range(d):
                    s[j] += w[i] * ((1.0 if y[i] == j + 1 else 0.0) - p[j + 1])
                    H[j, j] += w[i] * p[j + 1]
                    for k in range(d):
                        H[j, k] -= w[i] * p[j + 1] * p[k + 1]
            step = np.linalg.solve(H, s)
            big = 0.0
            for j in range(d):
                big = max(big, abs(step[j]))
            if big > 2.0:
                step *= 2.0 / big
            a += step
            if big < 1e-10:
                break
        V = np.linalg.inv(H)
        R = np.linalg.cholesky((V + V.T) / 2)
        # the candidates
        lw = np.empty(n_cand)
        cand = np.empty((n_cand, d))
        z = np.empty(d)
        cnt = 0
        for k in range(n_cand):
            # d + nu normals by Box-Muller, the last nu for the chi-square
            nz = d + nu
            zz = np.empty(nz + 1)
            for t in range(0, nz, 2):
                u1 = _uniform(key, gid[g], cnt)
                u2 = _uniform(key, gid[g], cnt + 1)
                cnt += 2
                rr = np.sqrt(-2.0 * np.log(u1))
                zz[t] = rr * np.cos(2 * np.pi * u2)
                zz[t + 1] = rr * np.sin(2 * np.pi * u2)
            c2 = 0.0
            for t in range(d, nz):
                c2 += zz[t] * zz[t]
            sc = np.sqrt(nu / c2)
            q = 0.0
            for j in range(d):
                z[j] = zz[j] * sc
                q += z[j] * z[j]
            for j in range(d):
                x = a[j]
                for l in range(j + 1):
                    x += R[j, l] * z[l]
                cand[k, j] = x
            lw[k] = (_mnl_logpost(E0, y, w, r0, r1, cand[k], Lam)
                     + 0.5 * (nu + d) * np.log1p(q / nu))
        mxw = lw.max()
        tot, tot2 = 0.0, 0.0
        for k in range(n_cand):
            lw[k] = np.exp(lw[k] - mxw)
            tot += lw[k]
            tot2 += lw[k] * lw[k]
        ess[g] = tot * tot / tot2
        for j in range(d):
            m1, m2 = 0.0, 0.0
            for k in range(n_cand):
                m1 += lw[k] * cand[k, j]
                m2 += lw[k] * cand[k, j] * cand[k, j]
            m1 /= tot
            mean[g, j] = m1
            sd[g, j] = np.sqrt(max(m2 / tot - m1 * m1, 0.0))
        u = _uniform(key, gid[g], cnt) * tot
        acc = 0.0
        pick = n_cand - 1
        for k in range(n_cand):
            acc += lw[k]
            if acc >= u:
                pick = k
                break
        for j in range(d):
            draw[g, j] = cand[pick, j]


def _mnl_location(res, X, f, fname):
    """The rows' linear predictors (n x J, the base level's 0) without the
    group intercepts at `res`'s parameters; a level that a fixed-effect
    level never has is at -inf."""
    levels = list(res.levels)
    J = len(levels)
    names = list(res.coefnames)
    pos = {nm: i for i, nm in enumerate(names)}
    xn = list(res._spec["xcols"])
    beta = np.asarray(res.beta, float)
    E = np.zeros((X.shape[0], J))
    # (X has every design column, the intercept's ones among them)
    for j, lv in enumerate(levels[1:], start=1):
        for k, nm in enumerate(xn):
            E[:, j] += X[:, k] * beta[pos[f"{lv}:{nm}"]]
    if fname:
        t = res._spec["fixef"][fname]
        Af = np.column_stack([t[f"fe_{fname}:{lv}"].fill_null(float("-inf")).to_numpy()
                              .astype(float) for lv in levels])
        E += Af[f]
    return E


def _mnl_draw(E, y, w, st, gid, Lam, key):
    n = len(gid)
    d = Lam.shape[0]
    draw, mean, sd, ess = (np.empty((n, d)), np.empty((n, d)), np.empty((n, d)),
                           np.empty(n))
    args = (np.ascontiguousarray(E), y.astype(np.int64), np.ascontiguousarray(w, float),
            st.astype(np.int64), gid.astype(np.int64), np.ascontiguousarray(Lam))
    _mnl_sir_kernel(*args, np.uint64(key), MNL_CANDIDATES, MNL_T_DF, draw, mean, sd, ess)
    low = np.flatnonzero(ess < MNL_ESS_MIN)
    if len(low):
        # these groups again, with 16x the candidates (and new ones)
        rows = np.concatenate([np.arange(st[g], st[g + 1]) for g in low])
        st2 = np.r_[0, np.cumsum(st[low + 1] - st[low])]
        d2, m2, s2, e2 = (np.empty((len(low), d)), np.empty((len(low), d)),
                          np.empty((len(low), d)), np.empty(len(low)))
        _mnl_sir_kernel(np.ascontiguousarray(E[rows]), args[1][rows], args[2][rows],
                        st2.astype(np.int64), args[4][low], args[5],
                        np.uint64(key ^ 0x5BD1E995), 16 * MNL_CANDIDATES, MNL_T_DF, d2, m2,
                        s2, e2)
        draw[low], mean[low], sd[low], ess[low] = d2, m2, s2, e2
    return draw, mean, sd, ess
