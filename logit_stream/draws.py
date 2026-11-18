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
"""

from __future__ import annotations

from dataclasses import replace

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
    from .newton import _Driver
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
    from .newton import _Driver
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
    if not info or info.get("kind") not in ("fe", "adaptive"):
        return None
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
    fname, fcols = _fe_columns(spec)
    out = []
    c = _iid_factor(result, info)
    L = _factor(result.vcov if kind in ("beta", "fe") else info["V"])
    for i in range(n):
        tau = None
        levels = None
        if kind in ("beta", "fe"):
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
            beta = th[:P]
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


def _drawn(result, spec, i, beta, levels, tau, fname, fcols):
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
        s["effects"] = [e for e in spec.get("effects", []) if e[2] != "random"]
        s["ranef"] = {}
        re = {k: dict(v) for k, v in re.items()}
        if tau is not None:
            (name,) = re
            re[name]["sd"] = np.array([tau])
            re[name]["Sigma"] = np.array([[tau * tau]])
    diag = dict(result.diagnostics or {})
    diag["draw"] = i
    return replace(result, beta=np.asarray(beta, float), random_effects=re, diagnostics=diag,
                   _spec=s)
