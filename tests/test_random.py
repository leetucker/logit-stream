"""Random intercepts (``random=``) for the multinomial and ordered logit:
agreement with lme4 (glmer, nAGQ = 0 and 1) and ordinal (clmm, Laplace);
the exact Laplace approximation's derivatives and stationary point, checked
by brute force; Sigma's standard errors; recovery of simulated parameters;
the large-group limit (the fixed-effect fit); random intercepts within an
absorbed fixed effect (the same fit as with the fixed effect's indicators);
inference and post-estimation."""

import numpy as np
import polars as pl
import pytest

from logit_stream import mlogit_stream, ologit_stream
from logit_stream.simulate import RACES, simulate_people


def _re(fit):
    """The random-effects summary of a fit with one random grouping."""
    (info,) = fit.random_effects.values()
    return info


def _binary_data():
    rng = np.random.default_rng(11)
    G = 60
    sizes = rng.poisson(80, G) + 3
    g = np.repeat(np.arange(G), sizes)
    n = len(g)
    x1, x2 = rng.normal(size=n), rng.binomial(1, 0.4, n).astype(float)
    a = rng.normal(0, 0.7, G)
    eta = -0.4 + 0.8 * x1 - 0.5 * x2 + a[g]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(np.int64)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})


def test_matches_lme4_nagq0():
    """lme4 2.0.6: glmer(y ~ x1 + x2 + (1 | g), family = binomial, nAGQ = 0)
    on this data gives the coefficients and standard errors below, sd
    0.6813543906 and log-likelihood -2815.40170289. The same approximation
    (b and the effects at their joint mode); Sigma by EM here, by direct
    optimization of the Laplace deviance there, which differ slightly."""
    df = _binary_data()
    assert df.height == 4893
    f = mlogit_stream("y ~ x1 + x2", df, random="g", verbose=False, tol=1e-12,
                      random_tol=1e-8, random_method="joint-mode")
    b = np.array([-0.474195551373, 0.790921745665, -0.391590383714])
    se = np.array([0.09744770435, 0.03688368747, 0.06774238204])
    assert np.all(np.abs(f.beta - b) < 0.03 * se), (f.beta - b) / se
    np.testing.assert_allclose(f.se, se, rtol=0.02)
    assert np.sqrt(_re(f)["Sigma"][0][0]) == pytest.approx(0.6813543906, rel=0.02)
    assert f.loglik == pytest.approx(-2815.40170289, abs=0.05)


def test_matches_lme4_nagq1():
    """lme4 2.0.6: glmer(y ~ x1 + x2 + (1 | g), family = binomial, nAGQ = 1)
    on the same data: the exact Laplace approximation, the default."""
    f = mlogit_stream("y ~ x1 + x2", _binary_data(), random="g", verbose=False, tol=1e-12,
                      random_tol=1e-8)
    assert _re(f)["method"] == "laplace"
    b = np.array([-0.480425667054, 0.799880277051, -0.395924654280])
    se = np.array([0.09757510799, 0.03724080413, 0.06809067919])
    assert np.all(np.abs(f.beta - b) < 0.002 * se), (f.beta - b) / se
    # lme4's standard errors differentiate the whole approximation; these
    # are from the joint information
    np.testing.assert_allclose(f.se, se, rtol=0.01)
    assert _re(f)["sd"][0] == pytest.approx(0.6814538185, rel=1e-4)
    assert f.loglik == pytest.approx(-2815.36912312, abs=1e-3)


def _ordered_data():
    rng = np.random.default_rng(12)
    G = 60
    sizes = rng.poisson(60, G) + 3
    g = np.repeat(np.arange(G), sizes)
    n = len(g)
    x1, x2 = rng.normal(size=n), rng.binomial(1, 0.4, n).astype(float)
    a = rng.normal(0, 0.6, G)
    lat = 0.7 * x1 - 0.4 * x2 + a[g] + rng.logistic(size=n)
    y = np.searchsorted([-1.0, 0.2, 1.4], lat).astype(np.int64)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})


def test_matches_clmm_laplace():
    """ordinal 2026.7.26: clmm(factor(y, ordered) ~ x1 + x2 + (1 | g),
    link = "logit", nAGQ = 1): slopes, cutpoints, their standard errors,
    the sd of the effects (its standard error 0.11622 on the log scale, so
    0.4766 x 0.11622 = 0.05539 on the sd's) and the log-likelihood."""
    df = _ordered_data()
    assert df.height == 3726
    f = ologit_stream("y ~ x1 + x2", df, random="g", verbose=False, tol=1e-12, random_tol=1e-8)
    b = np.array([0.748063562552, -0.539505358487, -0.983044214582, 0.264930119729,
                  1.501062380355])
    se = np.array([0.03328335537, 0.06300399771, 0.07697463199, 0.07543760103, 0.07995320088])
    assert f.coefnames == ["x1", "x2", "cut1", "cut2", "cut3"]
    assert np.all(np.abs(f.beta - b) < 0.002 * se), (f.beta - b) / se
    np.testing.assert_allclose(f.se, se, rtol=0.01)
    assert _re(f)["sd"][0] == pytest.approx(0.4765789316, rel=1e-4)
    assert _re(f)["sd_se"][0] == pytest.approx(0.4765789316 * 0.11622432987, rel=0.01)
    assert f.loglik == pytest.approx(-4708.33374158, abs=1e-3)


def test_third_derivatives():
    """The exact Laplace approximation's terms are derivatives of the
    information: check them by differencing it."""
    from logit_stream.fe import _ordered_info_derivs
    rng = np.random.default_rng(0)
    # multinomial: d (diag p - p p') / d eta_k, and c = its trace with M^-1
    d = 3
    eta = rng.normal(size=d)

    def W(e):
        ex = np.exp(np.r_[0.0, e])
        p = (ex / ex.sum())[1:]
        return np.diag(p) - np.outer(p, p), p

    W0, p = W(eta)
    Dm = W0                                           # d p / d eta = diag p - p p'
    T = (np.eye(d)[:, :, None] * Dm[:, None, :] - Dm[:, None, :] * p[None, :, None]
         - p[:, None, None] * Dm[None, :, :])
    for k in range(d):
        e = np.zeros(d)
        e[k] = 1e-6
        np.testing.assert_allclose((W(eta + e)[0] - W(eta - e)[0]) / 2e-6, T[:, :, k], atol=1e-8)
    Mi = np.linalg.inv(W0 * 50 + np.eye(d))
    c = Dm @ (np.diag(Mi) - 2 * Mi @ p)
    np.testing.assert_allclose(c, np.einsum("lj,jlk->k", Mi, T), atol=1e-12)
    # ordered: w = -d2 log P / d eta2 and its derivatives in eta and the cutpoints
    cuts = np.array([-1.0, 0.3, 1.2])
    etas = rng.normal(size=8) * 2
    ys = np.array([0, 1, 2, 3, 0, 3, 1, 2])
    w, dwe, dwu, dwl = _ordered_info_derivs(cuts, etas, ys)

    def logP(c, e):
        from scipy.special import expit
        full = np.r_[-np.inf, c, np.inf]
        return np.log(expit(full[ys + 1] - e) - expit(full[ys] - e))

    h = 1e-4
    w_num = -(logP(cuts, etas + h) - 2 * logP(cuts, etas) + logP(cuts, etas - h)) / h ** 2
    np.testing.assert_allclose(w, w_num, rtol=1e-5)
    np.testing.assert_allclose(dwe, (_ordered_info_derivs(cuts, etas + 1e-6, ys)[0]
                                     - _ordered_info_derivs(cuts, etas - 1e-6, ys)[0]) / 2e-6,
                               rtol=1e-5, atol=1e-8)
    for j in range(3):
        e = np.zeros(3)
        e[j] = 1e-6
        num = (_ordered_info_derivs(cuts + e, etas, ys)[0]
               - _ordered_info_derivs(cuts - e, etas, ys)[0]) / 2e-6
        ana = np.where(ys == j, dwu, 0.0) + np.where(ys == j + 1, dwl, 0.0)
        np.testing.assert_allclose(ana, num, rtol=1e-5, atol=1e-8)


def _laplace_brute(beta, S, X, y, g, G):
    """The Laplace approximation for a 3-level multinomial with random
    intercepts, independently: each group's mode by Newton's method, then
    l_g(mode) - mode' S^-1 mode / 2 - log|S| / 2 - log|I_g + S^-1| / 2."""
    B = beta.reshape(2, -1)
    Si = np.linalg.inv(S)
    total = -0.5 * G * np.linalg.slogdet(S)[1]
    for k in range(G):
        Xi, yi = X[g == k], y[g == k]
        a = np.zeros(2)
        for _ in range(50):
            E = np.column_stack([np.zeros(len(yi)), Xi @ B.T + a])
            Pm = np.exp(E - E.max(1, keepdims=True))
            Pm /= Pm.sum(1, keepdims=True)
            R = -Pm[:, 1:]
            R[np.arange(len(yi)), yi - 1] += yi > 0
            M = np.diag(Pm[:, 1:].sum(0)) - Pm[:, 1:].T @ Pm[:, 1:] + Si
            step = np.linalg.solve(M, R.sum(0) - Si @ a)
            a = a + step
            if np.abs(step).max() < 1e-13:
                break
        E = np.column_stack([np.zeros(len(yi)), Xi @ B.T + a])
        lse = np.log(np.exp(E).sum(1))
        total += (E[np.arange(len(yi)), yi] - lse).sum() - 0.5 * a @ Si @ a \
            - 0.5 * np.linalg.slogdet(M)[1]
    return total


def test_multinomial_laplace_stationary_point():
    """At the estimates, the gradient of an independent implementation of
    the Laplace approximation is zero, in the coefficients and in Sigma."""
    rng = np.random.default_rng(4)
    G = 40
    sizes = rng.poisson(40, G) + 2
    g = np.repeat(np.arange(G), sizes)
    n = len(g)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    A = rng.multivariate_normal([0, 0], [[0.3, 0.1], [0.1, 0.5]], G)
    E = np.column_stack([np.zeros(n), X @ np.array([[-0.2, 0.7], [0.3, -0.5]]).T + A[g]])
    Pm = np.exp(E)
    Pm /= Pm.sum(1, keepdims=True)
    y = (Pm.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    df = pl.DataFrame({"y": y, "x": X[:, 1], "g": g})
    f = mlogit_stream("y ~ x", df, random="g", verbose=False, tol=1e-14, random_tol=1e-9)
    beta, S = f.beta.copy(), np.asarray(_re(f)["Sigma"])
    h = 1e-5
    gb = [(_laplace_brute(beta + h * e, S, X, y, g, G)
           - _laplace_brute(beta - h * e, S, X, y, g, G)) / (2 * h) for e in np.eye(4)]
    assert np.max(np.abs(gb)) < 1e-4, gb
    gs = []
    for j, l in [(0, 0), (1, 1), (0, 1)]:
        E2 = np.zeros((2, 2))
        E2[j, l] = E2[l, j] = h
        gs.append((_laplace_brute(beta, S + E2, X, y, g, G)
                   - _laplace_brute(beta, S - E2, X, y, g, G)) / (2 * h))
    assert np.max(np.abs(gs)) < 1e-3, gs
    assert f.loglik == pytest.approx(_laplace_brute(beta, S, X, y, g, G), abs=1e-6)


def test_sigma_standard_error_matches_the_profile_curvature():
    """Sigma's standard error against the curvature of the brute-force
    Laplace approximation profiled over the coefficients (3-level
    multinomial, diagonal Sigma, so that the profile is over few
    parameters)."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(6)
    G = 50
    sizes = rng.poisson(50, G) + 2
    g = np.repeat(np.arange(G), sizes)
    n = len(g)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    A = rng.normal(0, [0.5, 0.7], (G, 2))
    E = np.column_stack([np.zeros(n), X @ np.array([[-0.2, 0.7], [0.3, -0.5]]).T + A[g]])
    Pm = np.exp(E)
    Pm /= Pm.sum(1, keepdims=True)
    y = (Pm.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    df = pl.DataFrame({"y": y, "x": X[:, 1], "g": g})
    f = mlogit_stream("y ~ x", df, random="g", random_cov="diagonal", verbose=False,
                      tol=1e-14, random_tol=1e-9)
    s0 = np.diag(np.asarray(_re(f)["Sigma"]))

    def profile(v):
        res = minimize(lambda b: -_laplace_brute(b, np.diag(v), X, y, g, G), f.beta,
                       method="BFGS", options={"gtol": 1e-8})
        return -res.fun

    h = 1e-3 * s0
    Hm = np.empty((2, 2))
    L0 = profile(s0)
    for i in range(2):
        e = np.zeros(2)
        e[i] = h[i]
        Hm[i, i] = (profile(s0 + e) - 2 * L0 + profile(s0 - e)) / h[i] ** 2
    ej, ek = np.array([h[0], 0]), np.array([0, h[1]])
    Hm[0, 1] = Hm[1, 0] = (profile(s0 + ej + ek) - profile(s0 + ej - ek) - profile(s0 - ej + ek)
                           + profile(s0 - ej - ek)) / (4 * h[0] * h[1])
    se_brute = np.sqrt(np.diag(np.linalg.inv(-Hm)))
    se = np.diag(np.asarray(_re(f)["Sigma_se"]))
    np.testing.assert_allclose(se, se_brute, rtol=0.05)


@pytest.fixture(scope="module")
def people():
    return simulate_people(60_000, n_states=10, pumas_per_state=12, seed=3)


@pytest.fixture(scope="module")
def race_fit(people):
    return mlogit_stream("race ~ age10 + female + log_income + owner + urban", people,
                         levels=RACES, random="puma", vcov={"CRV1": "puma"},
                         cluster=["state"], verbose=False)


def test_em_fixed_point(race_fit):
    """At the estimates the EM update returns Sigma (to the tolerance), and
    Sigma is a proper covariance matrix."""
    f = race_fit
    re = f.diagnostics["random_effects"]
    assert re["converged"] and f.converged
    assert re["sigma_change"] < 1e-6
    S = np.asarray(_re(f)["Sigma"])
    assert S.shape == (5, 5) and np.all(np.linalg.eigvalsh(S) > 0)


def test_recovers_the_simulated_parameters(race_fit):
    # simulate_people: PUMA effects independent with sd 0.5 per race equation
    sd = np.sqrt(np.diag(_re(race_fit)["Sigma"]))
    assert np.all(np.abs(sd - 0.5) < 0.2), sd
    from logit_stream.simulate import truth_people
    tr = truth_people()["race"]
    c = race_fit.coef()
    z = [(c[k] - v) / race_fit.se[race_fit.coefnames.index(k)] for k, v in tr.items()
         if not k.endswith("Intercept") and k in c]
    assert np.max(np.abs(z)) < 4, z


def test_large_groups_approach_the_fixed_effects():
    """With many rows per group, the effects are estimated almost without
    shrinkage, and the slopes approach the absorbed fixed effect's."""
    df = simulate_people(80_000, n_states=4, pumas_per_state=5, seed=4)
    fml = "race ~ age10 + female + log_income + owner + urban"
    r = mlogit_stream(fml, df, levels=RACES, random="puma", verbose=False)
    fe = mlogit_stream(fml + " | puma", df, levels=RACES, verbose=False)
    slopes = [k for k in fe.coefnames]
    d = np.array([r.coef()[k] - fe.coef()[k] for k in slopes])
    se = np.array([fe.se[fe.coefnames.index(k)] for k in slopes])
    assert np.all(np.abs(d) < 0.25 * se), d / se


def test_ordered(people):
    f = ologit_stream("education ~ age10 + female + log_income + owner + urban", people,
                      random="puma", verbose=False)
    assert f.converged
    assert f.coefnames[-4:] == ["cut1", "cut2", "cut3", "cut4"]
    sd = np.sqrt(_re(f)["Sigma"][0][0])
    assert abs(sd - 0.5) < 0.2
    assert _re(f)["levels"] == ["location"]
    tbl = f.ranef().collect()
    assert tbl.columns[-2:] == ["re_puma", "n_obs"]


def test_inference_and_postestimation(people, race_fit):
    f = race_fit
    assert set(f.all_vcovs) == {"iid", "CRV1:puma", "CRV1:state"}
    assert f.vcov_type == "CRV1:puma"
    assert f.n_clusters == {"puma": 120, "state": 10}
    assert f.n_params == len(f.coefnames) + 15        # 5 sds and 10 covariances
    assert "by puma (120 groups" in f.summary_text()
    assert f.summary_dict()["random_effects"]["puma"]["n_groups"] == 120
    tbl = f.ranef().collect()
    assert tbl.height == 120 and tbl["re_puma:White"].abs().max() == 0
    with pytest.raises(KeyError, match="ranef"):
        f.fixef()
    # a group the fit did not see gets its mean effect, 0
    new = people.head(5).with_columns(pl.lit(10_000).cast(people["puma"].dtype).alias("puma"))
    p_new = f.predict(new).collect().select(pl.col("^prob_.*$")).to_numpy()
    assert np.all(np.isfinite(p_new))
    np.testing.assert_allclose(p_new.sum(axis=1), 1.0)
    p_in = f.predict(people.head(5)).collect().select(pl.col("^prob_.*$")).to_numpy()
    assert not np.allclose(p_new, p_in)


def test_errors(people):
    fml = "race ~ age10 + female"
    with pytest.raises(ValueError, match="dependent"):
        mlogit_stream(fml, people, random="puma", vcov="hetero", verbose=False)
    with pytest.raises(ValueError, match="whole groups"):
        mlogit_stream(fml, people, random="state", vcov={"CRV1": "puma"}, verbose=False)
    with pytest.raises(ValueError, match="nest in one another"):
        mlogit_stream(fml + " | puma", people, random="state", verbose=False)
    with pytest.raises(ValueError, match="both a fixed and a random"):
        mlogit_stream(fml + " | puma", people, random="puma", verbose=False)
    with pytest.raises(ValueError, match="random_cov"):
        mlogit_stream(fml, people, random="puma", random_cov="banded", verbose=False)
    with pytest.raises(ValueError, match="random_method"):
        mlogit_stream(fml, people, random="puma", random_method="agq", verbose=False)


@pytest.mark.parametrize("method", ["laplace", "joint-mode"])
def test_within_a_fixed_effect_equals_its_indicators(people, method):
    """Random PUMA intercepts within absorbed state effects are the model
    with state indicators: the same coefficients, standard errors
    (clustered by state), Sigma and its standard errors, log-likelihood."""
    rhs = "age10 + female + log_income + owner + urban"
    kw = dict(levels=RACES, random="puma", random_method=method, vcov={"CRV1": "state"},
              verbose=False, tol=1e-12, random_tol=1e-8)
    a = mlogit_stream(f"race ~ {rhs} | state", people, **kw)
    b = mlogit_stream(f"race ~ {rhs} + C(state)", people, **kw)
    assert a.fe_names == ["state"] and a.n_levels == {"state": 10}
    assert _re(a)["within"] == "state"
    for k in a.coefnames:
        assert a.coef()[k] == pytest.approx(b.coef()[k], abs=1e-8)
        assert a.se[a.coefnames.index(k)] == pytest.approx(b.se[b.coefnames.index(k)], rel=1e-6)
    np.testing.assert_allclose(_re(a)["Sigma"], _re(b)["Sigma"], atol=1e-8)
    np.testing.assert_allclose(_re(a)["sd_se"], _re(b)["sd_se"], rtol=1e-4)
    assert a.loglik == pytest.approx(b.loglik, abs=1e-6)
    # the covariance's 15 parameters and the states' effects (each state's
    # levels but its reference)
    assert a.n_params == len(a.coefnames) + 15 + 10 * 5
    p = a.predict(people.head(20)).collect().select(pl.col("^prob_.*$")).to_numpy()
    q = b.predict(people.head(20)).collect().select(pl.col("^prob_.*$")).to_numpy()
    np.testing.assert_allclose(p, q, atol=1e-8)


def test_ordered_within_a_fixed_effect(people):
    rhs = "age10 + female + log_income + owner + urban"
    a = ologit_stream(f"education ~ {rhs} | state", people, random="puma", verbose=False,
                      tol=1e-12, random_tol=1e-8)
    b = ologit_stream(f"education ~ {rhs} + C(state)", people, random="puma", verbose=False,
                      tol=1e-12, random_tol=1e-8)
    for k in rhs.split(" + "):
        assert a.coef()[k] == pytest.approx(b.coef()[k], abs=1e-8)
    assert _re(a)["sd"][0] == pytest.approx(_re(b)["sd"][0], rel=1e-7)
    assert a.loglik == pytest.approx(b.loglik, abs=1e-6)
    assert a.fixef().collect().height == 10 and a.ranef().collect().height == 120


def test_collinear_columns_are_dropped(people):
    """A column dropped as collinear after the first step leaves the same fit
    as leaving it out (the effects' place in the parameters moves with it)."""
    df = _nested_fe_data(5).with_columns(x2=2 * pl.col("x"))
    kw = dict(random=["state", "puma"], verbose=False, tol=1e-12, random_tol=1e-8)
    with pytest.warns(UserWarning, match="collinear"):
        a = mlogit_stream("y ~ x + x2 | region", df, **kw)
    b = mlogit_stream("y ~ x | region", df, **kw)
    np.testing.assert_allclose(a.beta, b.beta, atol=1e-9)
    assert a.loglik == pytest.approx(b.loglik, abs=1e-8)
    p = people.head(20_000).with_columns(age10b=-pl.col("age10"))
    with pytest.warns(UserWarning, match="collinear"):
        c = ologit_stream("education ~ age10 + age10b + female", p, random="puma",
                          verbose=False)
    d = ologit_stream("education ~ age10 + female", p, random="puma", verbose=False)
    np.testing.assert_allclose(c.beta, d.beta, atol=1e-7)


def test_diagonal_covariance(people):
    f = mlogit_stream("race ~ age10 + female + log_income + owner + urban", people,
                      levels=RACES, random="puma", random_cov="diagonal", verbose=False)
    S = np.asarray(_re(f)["Sigma"])
    assert np.count_nonzero(S - np.diag(np.diag(S))) == 0
    assert f.n_params == len(f.coefnames) + 5


# --------------------------------------------------------------------------
# several groupings: nested, crossed
# --------------------------------------------------------------------------

def _laplace_dense(beta, Ss, X, y, groupings):
    """The Laplace approximation for a 3-level multinomial with random
    intercepts at several groupings (any structure), by brute force: the
    joint mode of all the effects by Newton's method on the dense
    information, then l(mode) - sum_k mode_k' S_k^-1 mode_k / 2
    - sum_k G_k log|S_k| / 2 - log|I + Lam| / 2."""
    B = beta.reshape(2, -1)
    Gs = [g.max() + 1 for g in groupings]
    offs = np.r_[0, np.cumsum([2 * G for G in Gs])]
    N = offs[-1]
    Lam = np.zeros((N, N))
    for k, (G, S) in enumerate(zip(Gs, Ss)):
        Si = np.linalg.inv(S)
        for g in range(G):
            o = offs[k] + 2 * g
            Lam[o:o + 2, o:o + 2] = Si
    n = len(y)
    idx = np.stack([offs[k] + 2 * g for k, g in enumerate(groupings)], axis=1)   # n x K
    a = np.zeros(N)

    def at(a):
        E = np.column_stack([np.zeros(n), X @ B.T])
        for k in range(len(groupings)):
            E[:, 1:] += np.column_stack([a[idx[:, k]], a[idx[:, k] + 1]])
        Pm = np.exp(E - E.max(1, keepdims=True))
        Pm /= Pm.sum(1, keepdims=True)
        return E, Pm

    for _ in range(60):
        E, Pm = at(a)
        R = -Pm[:, 1:]
        R[np.arange(n), y - 1] += y > 0
        gr = -Lam @ a
        H = Lam.copy()
        W = np.einsum("ni,ij->nij", Pm[:, 1:], np.eye(2)) - Pm[:, 1:, None] * Pm[:, None, 1:]
        for k in range(len(groupings)):
            for c in range(2):
                np.add.at(gr, idx[:, k] + c, R[:, c])
            for k2 in range(len(groupings)):
                for c in range(2):
                    for c2 in range(2):
                        np.add.at(H, (idx[:, k] + c, idx[:, k2] + c2), W[:, c, c2])
        step = np.linalg.solve(H, gr)
        a = a + step
        if np.abs(step).max() < 1e-13:
            break
    E, _ = at(a)
    ll = (E[np.arange(n), y] - np.log(np.exp(E).sum(1))).sum()
    return (ll - 0.5 * a @ Lam @ a - 0.5 * sum(G * np.linalg.slogdet(S)[1] for G, S in zip(Gs, Ss))
            - 0.5 * np.linalg.slogdet(H)[1])


def _two_grouping_data(kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "nested":
        s = np.repeat(np.arange(8), 4)
        sizes = rng.poisson(30, 32) + 3
        g2 = np.repeat(np.arange(32), sizes)
        g1 = s[g2]
    else:
        n = 2500
        g1, g2 = rng.integers(0, 25, n), rng.integers(0, 7, n)
    n = len(g1)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    A1 = rng.multivariate_normal([0, 0], [[0.3, 0.1], [0.1, 0.4]], g1.max() + 1)
    A2 = rng.multivariate_normal([0, 0], [[0.2, -0.05], [-0.05, 0.3]], g2.max() + 1)
    E = np.column_stack([np.zeros(n), X @ np.array([[-0.2, 0.7], [0.3, -0.5]]).T
                         + A1[g1] + A2[g2]])
    Pm = np.exp(E)
    Pm /= Pm.sum(1, keepdims=True)
    y = (Pm.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    return X, y, g1, g2


@pytest.mark.parametrize("kind", ["nested", "crossed"])
def test_two_groupings_laplace_stationary_point(kind):
    """With two random groupings, nested or crossed, the estimates are a
    stationary point of a dense, independent implementation of the Laplace
    approximation, in the coefficients and in both Sigmas."""
    X, y, g1, g2 = _two_grouping_data(kind, 7)
    df = pl.DataFrame({"y": y, "x": X[:, 1], "g1": g1, "g2": g2})
    f = mlogit_stream("y ~ x", df, random=["g1", "g2"], verbose=False, tol=1e-14,
                      random_tol=1e-9)
    assert f.random_effects["g1"]["structure"] == ("tree" if kind == "nested" else "crossed")
    beta = f.beta.copy()
    Ss = [np.asarray(f.random_effects[k]["Sigma"]) for k in ("g1", "g2")]
    groupings = [g1, g2]
    h = 1e-5
    gb = [(_laplace_dense(beta + h * e, Ss, X, y, groupings)
           - _laplace_dense(beta - h * e, Ss, X, y, groupings)) / (2 * h) for e in np.eye(4)]
    assert np.max(np.abs(gb)) < 1e-4, gb
    for i in range(2):
        for j, l in [(0, 0), (1, 1), (0, 1)]:
            E2 = np.zeros((2, 2))
            E2[j, l] = E2[l, j] = h
            up = [S + E2 if k == i else S for k, S in enumerate(Ss)]
            dn = [S - E2 if k == i else S for k, S in enumerate(Ss)]
            gs = (_laplace_dense(beta, up, X, y, groupings)
                  - _laplace_dense(beta, dn, X, y, groupings)) / (2 * h)
            assert abs(gs) < 1e-3, (i, j, l, gs)
    assert f.loglik == pytest.approx(_laplace_dense(beta, Ss, X, y, groupings), abs=1e-6)


def _nested_fe_data(seed):
    """Three outcomes; 4 regions (fixed) of 6 states of 5 PUMAs, with
    effects at both random levels."""
    rng = np.random.default_rng(seed)
    sizes = rng.poisson(40, 120) + 5
    puma = np.repeat(np.arange(120), sizes)
    state = puma // 5
    region = state // 6
    n = len(puma)
    x = rng.normal(size=n)
    A1 = rng.multivariate_normal([0, 0], [[0.3, 0.1], [0.1, 0.4]], 24)
    A2 = rng.multivariate_normal([0, 0], [[0.2, -0.05], [-0.05, 0.3]], 120)
    R = rng.normal(0, 0.5, (4, 2))
    E = np.column_stack([np.zeros(n), np.outer(x, [0.7, -0.5]) + R[region] + A1[state]
                         + A2[puma]])
    Pm = np.exp(E)
    Pm /= Pm.sum(1, keepdims=True)
    y = (Pm.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    return pl.DataFrame({"y": y, "x": x, "state": state, "puma": puma, "region": region})


def test_nested_within_a_fixed_effect_equals_its_indicators():
    """Random state and PUMA intercepts within an absorbed region effect
    are the model with region indicators."""
    df = _nested_fe_data(5)
    kw = dict(random=["state", "puma"], verbose=False, tol=1e-12, random_tol=1e-8,
              vcov={"CRV1": "region"})
    a = mlogit_stream("y ~ x | region", df, **kw)
    b = mlogit_stream("y ~ x + C(region)", df, **kw)
    assert a.fe_names == ["region"] and not a.diagnostics["random_effects"]["singular"]
    for k in a.coefnames:
        assert a.coef()[k] == pytest.approx(b.coef()[k], abs=1e-7)
        assert a.se[a.coefnames.index(k)] == pytest.approx(b.se[b.coefnames.index(k)], rel=1e-5)
    for g in ("state", "puma"):
        np.testing.assert_allclose(a.random_effects[g]["Sigma"], b.random_effects[g]["Sigma"],
                                   atol=1e-7)
    assert a.loglik == pytest.approx(b.loglik, abs=1e-5)
    p = a.predict(df.head(20)).collect().select(pl.col("^prob_.*$")).to_numpy()
    q = b.predict(df.head(20)).collect().select(pl.col("^prob_.*$")).to_numpy()
    np.testing.assert_allclose(p, q, atol=1e-7)
    vc = a.varcorr()
    assert set(vc["group"]) == {"state", "puma"} and vc.height == 2 * (2 + 1)
    assert (vc["ci_lower"] <= vc["estimate"]).all() and (vc["estimate"] <= vc["ci_upper"]).all()
    assert a.ranef("state").collect().height == 24 and a.ranef("puma").collect().height == 120
    with pytest.raises(ValueError, match="name one of"):
        a.ranef()


def test_zero_variance_is_held_at_the_floor():
    """Groups that differ no more than chance allows: the variance is held at
    its floor (an sd of 0.001), the fit converges and says so."""
    rng = np.random.default_rng(12)
    n = 4000
    x = rng.normal(size=n)
    df = pl.DataFrame({"y": (rng.random(n) < 1 / (1 + np.exp(0.3 - 0.8 * x))).astype(int),
                       "x": x, "g": rng.integers(0, 40, n)})
    with pytest.warns(UserWarning, match="variance of zero"):
        f = mlogit_stream("y ~ x", df, random="g", verbose=False)
    assert f.converged and f.diagnostics["random_effects"]["singular"] == ["g"]
    assert np.sqrt(_re(f)["Sigma"][0][0]) == pytest.approx(0.001, rel=0.01)


def test_crossed_restrictions(people):
    """Clustering and an absorbed effect are refused for crossed groupings
    (before any fitting)."""
    df = people.with_columns(occ=(pl.col("person_id") % 9))
    fml = "race ~ age10 + female"
    with pytest.raises(ValueError, match="crossed"):
        mlogit_stream(fml, df, random=["puma", "occ"], vcov={"CRV1": "state"}, verbose=False)
    with pytest.raises(ValueError, match="nest in one another"):
        mlogit_stream(fml + " | state", df, random=["puma", "occ"], verbose=False)
