"""Draws for the multinomial logit with random intercepts fitted by
adaptive quadrature: `draw_parameters` (the coefficients, Sigma and the
fixed effect's levels jointly) against the fit's covariances, and
`draw_ranef` (each group's J - 1 intercepts given its rows) against each
group's conditional distribution by brute-force integration on a grid:
its mean and sd, and the draws' probability integral transforms;
invariance to the read's chunks; predict and new groups."""

import numpy as np
import polars as pl
import pytest
from scipy.special import logsumexp
from test_random_adaptive_mnl import _data

from logit_stream import draws as D
from logit_stream import mlogit_stream


@pytest.fixture(scope="module")
def fitted():
    df = _data(seed=2, G=250, cells=3)
    fit = mlogit_stream("y ~ x | cell", df, random="g", random_method="adaptive", verbose=False)
    return df, fit


def _brute(df, fit, n=161):
    """Each group's conditional density on a grid: (grid a1, a2, weights)."""
    S = fit.random_effects["g"]["Sigma"]
    Lam = np.linalg.inv(S)
    lev = fit.fixef().collect().sort("cell").select("fe_cell:1", "fe_cell:2").to_numpy()
    b = dict(zip(fit.coefnames, fit.beta))
    x, c, y, g = (df[k].to_numpy() for k in ("x", "cell", "y", "g"))
    E = np.column_stack([np.zeros(len(x)), b["1:x"] * x + lev[c, 0], b["2:x"] * x + lev[c, 1]])
    s1, s2 = np.sqrt(np.diag(S))
    a1, a2 = np.meshgrid(np.linspace(-6 * s1, 6 * s1, n), np.linspace(-6 * s2, 6 * s2, n),
                         indexing="ij")
    A = np.column_stack([a1.ravel(), a2.ravel()])
    prior = -0.5 * np.einsum("ka,ab,kb->k", A, Lam, A)
    keys = fit.ranef().collect()["g"].to_numpy()
    out = []
    for key in keys:
        r = g == key
        tot = E[r][:, None, :] + np.concatenate([np.zeros((len(A), 1)), A], 1)[None]
        lp = (np.take_along_axis(tot, y[r][:, None, None].repeat(len(A), 1), 2)[..., 0]
              - logsumexp(tot, 2)).sum(0) + prior
        wts = np.exp(lp - lp.max())
        out.append((A, wts / wts.sum()))
    return out


def test_group_draws_are_the_conditional_distribution(fitted):
    df, fit = fitted
    ref = _brute(df, fit)
    one = fit.draw_ranef(seed=1)
    t = one.ranef().collect()
    mean_b = np.array([(w[:, None] * A).sum(0) for A, w in ref])
    sd_b = np.array([np.sqrt((w[:, None] * (A - m) ** 2).sum(0)) for (A, w), m in
                     zip(ref, mean_b)])
    # the weighted candidates' moments (1,000 candidates: Monte Carlo error)
    mc = sd_b / np.sqrt(one.diagnostics["ranef_ess"])[:, None]       # their Monte Carlo error
    assert np.all(np.abs(one.diagnostics["ranef_mean"] - mean_b) < 4 * mc)
    assert np.all(np.abs(t.select("sd_g:1", "sd_g:2").to_numpy() / sd_b - 1)
                  < 4 * 0.7 / np.sqrt(one.diagnostics["ranef_ess"])[:, None])
    assert (t["sd_g:0"] == 0).all() and (t["re_g:0"] == 0).all()
    # K draws at the estimates: the PIT of each intercept, and of their sum, under the
    # brute-force marginals is uniform
    K = 200
    many = D.draw_ranef_many([fit] * K, seed=3)
    R = np.stack([r.ranef().collect().select("re_g:1", "re_g:2").to_numpy() for r in many])
    pits = []
    for i, (A, w) in enumerate(ref[:120]):
        for proj, v in ((A[:, 0], R[:, i, 0]), (A[:, 1], R[:, i, 1]),
                        (A.sum(1), R[:, i].sum(1))):
            o = np.argsort(proj, kind="stable")
            pits.append(np.interp(v, proj[o], np.cumsum(w[o])))
    pit = np.concatenate(pits)
    assert abs(pit.mean() - 0.5) < 0.01
    hist = np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit)
    np.testing.assert_allclose(hist, 0.1, atol=0.01)
    assert len(set(R[:, 0, 0])) == K


def test_chunks_and_predict(fitted):
    df, fit = fitted
    a = D.draw_ranef_many([fit], seed=7)[0].ranef().collect()
    b = D.draw_ranef_many([fit], seed=7, chunk_cells=1)[0].ranef().collect()
    np.testing.assert_allclose(a["re_g:1"].to_numpy(), b["re_g:1"].to_numpy(), rtol=1e-12)
    new = pl.DataFrame({"x": [0.0, 1.0], "cell": [0, 1], "g": [5, 10**6]})
    d = fit.draw_parameters(2, seed=4, ranef=True, newdata=new)
    many = pl.DataFrame({"x": 0.0, "cell": 0, "g": np.arange(10**6, 10**6 + 500)})
    t = [r[0].ranef().collect().sort("g") for r in
         (fit.draw_parameters(1, seed=4, ranef=True, newdata=many),
          fit.draw_parameters(1, seed=4, ranef=True, newdata=many.reverse()))]
    assert t[0]["g"].equals(t[1]["g"])
    np.testing.assert_allclose(t[0]["re_g:1"].to_numpy(), t[1]["re_g:1"].to_numpy(), rtol=1e-12)
    for r in d:
        t = r.ranef().collect()
        assert t.filter(pl.col("g") == 10**6)["n_obs"][0] == 0
        re = {k: t.filter(pl.col("g") == k).select("re_g:1", "re_g:2").to_numpy()[0]
              for k in (5, 10**6)}
        lev = r.fixef().collect().sort("cell").select("fe_cell:1", "fe_cell:2").to_numpy()
        bb = dict(zip(r.coefnames, r.beta))
        eta = np.array([[bb["1:x"] * x + lev[c, 0], bb["2:x"] * x + lev[c, 1]]
                        for x, c in zip(new["x"], new["cell"])]) + np.array([re[5], re[10**6]])
        e = np.column_stack([np.zeros(2), eta])
        p = np.exp(e - logsumexp(e, 1, keepdims=True))
        got = r.predict(new, type="response").collect().select("prob_0", "prob_1",
                                                               "prob_2").to_numpy()
        np.testing.assert_allclose(got, p, rtol=1e-10)


def test_parameter_draws_follow_the_fit():
    # (enough groups that Sigma is well determined: its draws, on the log-Cholesky
    # scale, then have about the delta method's standard errors)
    fit = mlogit_stream("y ~ x | cell", _data(seed=8, G=2000, cells=3), random="g",
                        random_method="adaptive", verbose=False)
    n = 3000
    ds = fit.draw_parameters(n, seed=5)
    B = np.array([d.beta for d in ds])
    np.testing.assert_allclose(np.cov(B.T), fit.vcov, rtol=0.1, atol=1e-5)
    # Sigma's elements: centred near the estimates, with about the reported se
    re = fit.random_effects["g"]
    Sg = np.array([d.random_effects["g"]["Sigma"][np.triu_indices(2)] for d in ds])
    se = re["Sigma_se"][np.triu_indices(2)]
    assert np.all(np.abs(np.median(Sg, 0) - re["Sigma"][np.triu_indices(2)]) < 0.25 * se)
    np.testing.assert_allclose(np.std(Sg, 0), se, rtol=0.25)
    # the levels: the draws' sd is the fixef standard error
    fx = fit.fixef().collect().sort("cell")
    L = np.array([d.fixef().collect().sort("cell").select("fe_cell:1", "fe_cell:2").to_numpy()
                  for d in ds])
    np.testing.assert_allclose(L.std(0), fx.select("se_cell:1", "se_cell:2").to_numpy(),
                               rtol=0.06)
    np.testing.assert_allclose(L.mean(0), fx.select("fe_cell:1", "fe_cell:2").to_numpy(),
                               atol=0.06 * L.std(0).max())


def test_laplace_draws_refused_but_group_draws_allowed():
    df = _data(seed=6, G=80)
    lap = mlogit_stream("y ~ x", df, random="g", verbose=False)
    with pytest.raises(ValueError, match="adaptive"):
        lap.draw_parameters(2)
    t = lap.draw_ranef(seed=1).ranef().collect()
    assert t.height == 80 and t["sd_g:1"].min() > 0
