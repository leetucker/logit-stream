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
        s["ranef_base"] = spec.get("ranef_base", spec.get("ranef", {}))   # its groups
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
    if isinstance(result, MultinomialLogitResult) or not isinstance(result,
                                                                    OrderedLogitResult):
        raise ValueError("draw_ranef is available for the binary and ordered logit and probit "
                         "(one scalar random intercept); not for the multinomial logit yet")
    if len(re) != 1:
        raise ValueError(f"draw_ranef needs one random grouping, not {len(re)} ({list(re)})")
    (name,) = re
    if re[name].get("structure") not in (None, "tree"):
        raise ValueError("draw_ranef needs nested (not crossed) groupings")
    from .models import LINKS
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
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
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
    wscale = 1.0
    if first.weights and first.weights_type != "fweights":
        st = first._spec["source"].filter(spec["usable"]).select(
            pl.len().alias("n"), pl.col(first.weights).sum().alias("w")).collect()
        wscale = float(st["n"][0]) / float(st["w"][0])
    rows_per_chunk = max(int(chunk_cells // RANEF_GRID), 1000)
    carry = None

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
    # a group without rows (all its weights zero, say): its prior
    for m in range(M):
        miss = ~seen
        draws[m, miss] = taus[m] * _probit(U[m, miss])
        sds[m, miss] = taus[m]
    new = None
    if newdata is not None:
        from .results import as_lazy as _lazy
        new = (_lazy(newdata).select(keys).unique(maintain_order=True)
               .join(table.select(keys).lazy(), on=keys, how="anti").collect())
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


def _probit(u):
    from scipy.special import ndtri
    return ndtri(np.clip(u, 1e-16, 1 - 1e-16))
