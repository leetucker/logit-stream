"""The nested logit (`nlogit_stream`)."""

import numpy as np
import polars as pl
import pytest

from logit_stream import NestedLogitResult, clogit_stream, nlogit_stream
from logit_stream.nested import _nl_kernel, _nl_probs

ALTS = ["car", "bike", "bus", "train", "walk"]
NESTS = {"private": ["car", "bike"], "public": ["bus", "train"], "foot": ["walk"]}
LAM = {"private": 0.7, "public": 0.45, "foot": 1.0}


def _loglik(beta, lam, X, y, starts, nest):
    """The RU2 log-likelihood, written out directly."""
    ll = 0.0
    for c in range(len(starts) - 1):
        r = slice(starts[c], starts[c + 1])
        V, m = X[r] @ beta, nest[r]
        I = {k: np.log(np.sum(np.exp(V[m == k] / lam[k]))) for k in set(m)}
        D = np.log(sum(np.exp(lam[k] * I[k]) for k in I))
        i = np.flatnonzero(y[r])[0]
        k = m[i]
        ll += V[i] / lam[k] - I[k] + lam[k] * I[k] - D
    return ll


def _simulate(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    J = len(ALTS)
    cost = rng.uniform(0.5, 4, (n, J))
    time = rng.uniform(5, 60, (n, J))
    asc = np.array([0.0, -1.0, -0.3, -0.2, -0.8])
    V = asc - 0.6 * cost - 0.04 * time
    of = {a: k for k, alts in NESTS.items() for a in alts}
    P = np.zeros((n, J))
    nests = list(NESTS)
    I = np.column_stack([np.log(np.exp(V[:, [ALTS.index(a) for a in NESTS[k]]] / LAM[k])
                                .sum(axis=1)) for k in nests])
    Pk = np.exp(I * np.array([LAM[k] for k in nests]))
    Pk /= Pk.sum(axis=1, keepdims=True)
    for j, a in enumerate(ALTS):
        k = nests.index(of[a])
        P[:, j] = Pk[:, k] * np.exp(V[:, j] / LAM[of[a]] - I[:, k])
    choice = (P.cumsum(axis=1) > rng.random(n)[:, None]).argmax(axis=1)
    return pl.DataFrame({"case": np.repeat(np.arange(n), J), "alt": np.tile(ALTS, n),
                         "cost": cost.ravel(), "time": time.ravel(),
                         "chosen": (np.arange(J)[None, :] == choice[:, None]).astype(int)
                         .ravel()})


@pytest.fixture(scope="module")
def data():
    return _simulate()


@pytest.fixture(scope="module")
def fit(data):
    return nlogit_stream("chosen ~ cost + time + C(alt)", data, case="case", alt="alt",
                         nests=NESTS, vcov="hetero", verbose=False, tol=1e-12)


def _tree_loglik(beta, lam, X, y, starts, leaf, parent, M):
    """The log-likelihood of a tree of nests written out recursively:
    nodes 0..M-1, root M, lam[n] (lam[M] = 1), leaf[r] row r's nest."""
    ll = 0.0
    for c in range(len(starts) - 1):
        r = np.arange(starts[c], starts[c + 1])
        V = X[r] @ beta

        def iv(n, V=V, r=r):
            terms = [V[i] / lam[n] for i in range(len(r)) if leaf[r[i]] == n]
            terms += [lam[k] * iv(k) / lam[n] for k in range(M)
                      if parent[k] == n and _has(k, leaf[r], parent)]
            return np.log(np.sum(np.exp(terms)))

        i = int(np.flatnonzero(y[r])[0])
        n = leaf[r[i]]
        lp, u = 0.0, V[i]
        while True:
            lp += u / lam[n] - iv(n)
            if n == M:
                break
            u, n = lam[n] * iv(n), parent[n]
        ll += lp
    return ll


def _has(k, leaves, parent):
    """Whether nest k has any of the set's alternatives below it."""
    for m in set(leaves.tolist()):
        while m < len(parent):
            if m == k:
                return True
            m = parent[m]
    return False


def test_kernel_matches_the_formula():
    """Three levels (nests 0, 1 under 3; 2 under the root, as is 3), sets
    missing some nests, and the score by finite differences."""
    rng = np.random.default_rng(1)
    C = 60
    sizes = rng.integers(2, 6, C)
    starts = np.r_[0, np.cumsum(sizes)]
    n = starts[-1]
    X = rng.normal(size=(n, 3))
    leaf = rng.integers(0, 3, n)
    y = np.zeros(n, np.int8)
    for c in range(C):
        y[starts[c] + rng.integers(0, sizes[c])] = 1
    w = rng.uniform(0.5, 2, C).repeat(sizes)
    parent = np.array([3, 3, 4, 4])
    lam_of = np.array([0, 1, -1, 2])
    order = np.array([0, 1, 2, 3])
    beta, th = np.array([0.5, -0.3, 0.2]), np.array([0.5, -0.3, 0.2, 0.4, 0.6, 0.8])
    lam = np.array([0.4, 0.6, 1.0, 0.8, 1.0])
    none, z0 = np.zeros((0, 6)), np.zeros(n)

    def kern(ww, t, Gs=none, threads=2):
        return _nl_kernel(X, y, ww, z0, starts, leaf, t[None, :], 3, parent, lam_of, order, 0,
                          threads, Gs)

    ll, G, _ = kern(np.ones(n), th)
    assert ll[0] == pytest.approx(_tree_loglik(beta, lam, X, y, starts, leaf, parent, 4),
                                  rel=1e-12)
    lw = kern(w, th)
    Gs = np.zeros((C, 6))
    kern(np.ones(n), th, Gs, threads=1)
    np.testing.assert_allclose(Gs.sum(axis=0), G[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(w[starts[:-1]] @ Gs, lw[1][0], rtol=1e-12, atol=1e-12)
    h, num = 1e-6, np.zeros(6)
    for j in range(6):
        e = np.zeros(6)
        e[j] = h
        f = [kern(np.ones(n), th + sg * e, threads=1)[0][0] for sg in (1, -1)]
        num[j] = (f[0] - f[1]) / (2 * h)
    np.testing.assert_allclose(G[0], num, rtol=1e-6, atol=1e-7)


def test_recovers_the_parameters(fit):
    assert isinstance(fit, NestedLogitResult) and fit.converged
    assert fit.coefnames[-2:] == ["lambda.private", "lambda.public"]
    c = fit.coef()
    truth = {"cost": -0.6, "time": -0.04, "lambda.private": 0.7, "lambda.public": 0.45}
    se = dict(zip(fit.coefnames, fit.se))
    for k, v in truth.items():
        assert abs(c[k] - v) < 3.5 * se[k], (k, c[k], v, se[k])
    assert fit.loglik > fit.loglik_clogit
    assert set(fit.all_vcovs) == {"iid", "hetero"} and fit.n_obs == 4000
    # the gradient at the stopping point is 1e-6 on Linux and Windows, and
    # 1.6e-4 on macOS (CI), where floating-point rounding differs: the
    # estimates move by about gradient / information, far below their errors
    assert np.max(np.abs(fit.diagnostics["gradient"])) < 1e-3


def test_matches_r_mlogit(fit):
    """R's mlogit (3.0) on the same data, `nests=` as here, unscaled = FALSE
    (RU2), the walk nest's parameter held at 1: its estimates (to its
    convergence tolerance); its standard errors are the BHHH ones, which
    the scores here reproduce to six digits (checked when written)."""
    r = {"cost": -0.590172, "time": -0.040302, "C(alt)[T.bus]": 0.671819,
         "C(alt)[T.car]": 0.977775, "C(alt)[T.train]": 0.776576, "C(alt)[T.walk]": 0.216576,
         "lambda.private": 0.648473, "lambda.public": 0.438938}
    c = fit.coef()
    for k, v in r.items():
        assert c[k] == pytest.approx(v, abs=1e-5), k
    assert fit.loglik == pytest.approx(-4750.245, abs=1e-3)


def test_all_lambdas_one_is_the_conditional_logit(data):
    """Nests of single alternatives: refused (the model is the conditional
    logit); and the likelihood at lambda = 1 is the conditional logit's."""
    c = clogit_stream("chosen ~ cost + time + C(alt)", data, case="case", verbose=False,
                      tol=1e-12)
    with pytest.warns(UserWarning, match="did not converge"):
        f = nlogit_stream("chosen ~ cost + time + C(alt)", data, case="case", alt="alt",
                          nests=NESTS, verbose=False, maxiter=1)
    assert f.loglik_clogit == pytest.approx(c.loglik, abs=1e-8)
    with pytest.raises(ValueError, match="conditional logit"):
        nlogit_stream("chosen ~ cost", data, case="case", alt="alt",
                      nests={a: [a] for a in ALTS}, verbose=False)


def test_invariant_to_blocks_and_row_order(data, fit):
    shuffled = data.sample(fraction=1.0, shuffle=True, seed=3)
    f = nlogit_stream("chosen ~ cost + time + C(alt)", shuffled, case="case", alt="alt",
                      nests=NESTS, vcov="hetero", verbose=False, tol=1e-12, block_mb=0.05)
    assert f.diagnostics["store"]["blocks"] > 3
    np.testing.assert_allclose(f.beta, fit.beta, rtol=1e-6, atol=1e-8)


def test_nests_validated(data):
    with pytest.raises(ValueError, match="in no nest"):
        nlogit_stream("chosen ~ cost", data, case="case", alt="alt",
                      nests={"a": ["car", "bike"], "b": ["bus"]}, verbose=False)
    with pytest.raises(ValueError, match="in nests"):
        nlogit_stream("chosen ~ cost", data, case="case", alt="alt",
                      nests={"a": ALTS, "b": ["bus"]}, verbose=False)
    with pytest.raises(ValueError, match="not both"):
        nlogit_stream("chosen ~ cost", data, case="case", alt="alt",
                      nests={"a": ["car", ["bike"]], "b": ["bus", "train", "walk"]},
                      verbose=False)
    with pytest.raises(ValueError, match="twice"):
        nlogit_stream("chosen ~ cost", data, case="case", alt="alt",
                      nests={"a": {"a": ["car", "bike"]}, "b": ["bus", "train", "walk"]},
                      verbose=False)
    with pytest.raises(ValueError, match="never occur"):
        nlogit_stream("chosen ~ cost", data, case="case", alt="alt",
                      nests={"a": ALTS, "b": ["plane"]}, verbose=False)


def test_predictions(data, fit):
    """The probabilities sum to 1 in each set and match the formula; the
    residuals; an alternative in no nest nulls its set."""
    p = fit.predict(data.head(50)).collect()
    assert p.group_by("case").agg(pl.col("prob").sum())["prob"].to_numpy() == \
        pytest.approx(1.0)
    th = np.asarray(fit.beta)
    tree = fit._spec["tree"]
    X = fit.predict(data.head(50), type="link").collect()["xb"].to_numpy()
    code = {n: i for i, n in enumerate(tree.bottom)}
    nest = np.array([code[tree.of[a]] for a in data.head(50)["alt"]])
    q = _nl_probs(X, nest, data.head(50)["case"].to_numpy().astype(np.int64),
                  tree.lam(th[-2:]), tree.parent, tree.order)
    np.testing.assert_allclose(p["prob"].to_numpy(), q)
    # against the written-out tree likelihood: the chosen ones' log-probabilities
    h = data.head(50)
    ll = _tree_loglik(np.ones(1), tree.lam(th[-2:]), X[:, None], h["chosen"].to_numpy(),
                      np.arange(0, 51, 5), nest, tree.parent, len(tree.names))
    assert ll == pytest.approx(float(np.log(p["prob"].to_numpy()[h["chosen"].to_numpy() == 1])
                                     .sum()))
    r = fit.resid().collect()
    assert r.height == data.height and abs(r["resid"].sum()) < 1e-6
    bad = data.head(10).with_columns(pl.when(pl.col("alt") == "bus").then(pl.lit("plane"))
                                     .otherwise(pl.col("alt")).alias("alt"))
    assert fit.predict(bad).collect()["prob"].null_count() == 10


def test_weights_and_clustering(data):
    """Frequency weights of 2 double the information; analytic weights do
    not depend on their scale; clustering by a column constant within sets,
    and with every set its own cluster equal to the robust variance."""
    fml = "chosen ~ cost + time + C(alt)"
    kw = dict(case="case", alt="alt", nests=NESTS, verbose=False, tol=1e-12)
    d = data.with_columns(two=pl.lit(2.0), aw=(pl.col("case") % 5 + 1).cast(pl.Float64),
                          region=pl.col("case") % 37)
    a = nlogit_stream(fml, d, vcov="hetero", cluster=["region", "case"], **kw)
    f = nlogit_stream(fml, d, weights="two", weights_type="fweights", **kw)
    np.testing.assert_allclose(f.beta, a.beta, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(f.se, np.sqrt(np.diag(a.all_vcovs["iid"][0]) / 2), rtol=1e-4)
    assert f.n_obs == 8000 and f.loglik == pytest.approx(2 * a.loglik, rel=1e-9)
    w1 = nlogit_stream(fml, d, weights="aw", **kw)
    w10 = nlogit_stream(fml, d.with_columns(pl.col("aw") * 10), weights="aw", **kw)
    np.testing.assert_allclose(w10.beta, w1.beta, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(w10.se, w1.se, rtol=1e-6)
    assert not np.allclose(w1.beta, a.beta, rtol=1e-3)
    assert a.n_clusters["region"] == 37 and a.all_vcovs["CRV1:region"][1] == 36
    np.testing.assert_allclose(a.all_vcovs["CRV1:case"][0], a.all_vcovs["hetero"][0],
                               rtol=1e-8)
    with pytest.raises(ValueError, match="constant within"):
        nlogit_stream(fml, d.with_columns(r=pl.col("cost") > 2), vcov={"CRV1": "r"}, **kw)


THREE = {"motor": {"private": ["car", "bike"], "public": ["bus", "train"]}, "foot": ["walk"]}


def test_three_levels():
    """A nest of nests: data simulated from it, the parameters recovered,
    the fit consistent with the written-out likelihood, and the two-level
    model nested in it (its upper lambda at 1 is the two-level fit)."""
    rng = np.random.default_rng(7)
    n, J = 6000, len(ALTS)
    cost, time = rng.uniform(0.5, 4, (n, J)), rng.uniform(5, 60, (n, J))
    V = np.array([0.0, -1.0, -0.3, -0.2, -0.8]) - 0.6 * cost - 0.04 * time
    lp, lu = {"private": 0.5, "public": 0.4}, 0.8
    ix = {a: ALTS.index(a) for a in ALTS}
    Ib = {k: np.log(np.exp(V[:, [ix[a] for a in v]] / lp[k]).sum(1)) for k, v in
          THREE["motor"].items()}
    Im = np.log(sum(np.exp(lp[k] * Ib[k] / lu) for k in Ib))
    Ir = np.logaddexp(lu * Im, V[:, ix["walk"]])
    P = np.zeros((n, J))
    for k, v in THREE["motor"].items():
        for a in v:
            P[:, ix[a]] = np.exp(V[:, ix[a]] / lp[k] - Ib[k] + lp[k] * Ib[k] / lu - Im
                                 + lu * Im - Ir)
    P[:, ix["walk"]] = np.exp(V[:, ix["walk"]] - Ir)
    np.testing.assert_allclose(P.sum(1), 1.0)
    ch = (P.cumsum(1) > rng.random(n)[:, None]).argmax(1)
    d = pl.DataFrame({"case": np.repeat(np.arange(n), J), "alt": np.tile(ALTS, n),
                      "cost": cost.ravel(), "time": time.ravel(),
                      "chosen": (np.arange(J)[None] == ch[:, None]).astype(int).ravel()})
    f = nlogit_stream("chosen ~ cost + time + C(alt)", d, case="case", alt="alt",
                      nests=THREE, verbose=False, tol=1e-12)
    assert f.converged
    assert f.coefnames[-3:] == ["lambda.motor", "lambda.private", "lambda.public"]
    c, se = f.coef(), dict(zip(f.coefnames, f.se))
    for k, v in {"cost": -0.6, "time": -0.04, "lambda.motor": lu, "lambda.private": 0.5,
                 "lambda.public": 0.4}.items():
        assert abs(c[k] - v) < 3.5 * se[k], (k, c[k], v, se[k])
    p = f.predict(d).collect()
    chosen = p.filter(pl.col("chosen") == 1)["prob"].to_numpy()
    assert float(np.log(chosen).sum()) == pytest.approx(f.loglik, abs=1e-6)
    assert f.nests["motor"] == ["car", "bike", "bus", "train"]


def test_offset(data):
    """An offset of 0.3 cost is a coefficient of 0.3 on cost held fixed: the
    estimate moves by exactly that, and predictions include it."""
    kw = dict(case="case", alt="alt", nests=NESTS, verbose=False, tol=1e-12)
    a = nlogit_stream("chosen ~ cost + time + C(alt)", data, **kw)
    b = nlogit_stream("chosen ~ cost + time + C(alt)", data, offset=pl.col("cost") * 0.3, **kw)
    assert b.coef()["cost"] == pytest.approx(a.coef()["cost"] - 0.3, abs=1e-6)
    assert b.loglik == pytest.approx(a.loglik, abs=1e-6)
    np.testing.assert_allclose(b.predict(data.head(20)).collect()["prob"].to_numpy(),
                               a.predict(data.head(20)).collect()["prob"].to_numpy(), atol=1e-6)

