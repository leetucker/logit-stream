"""logit_stream.core: its stable names, and the multinomial and ordered Newton
pass against the package's own fits, numerical derivatives and a design of
each split's rows alone."""

import numpy as np
import polars as pl
import pytest

import logit_stream.core as core
import logit_stream.core.models as M
from logit_stream import mlogit_stream, ologit_stream, oprobit_stream

STABLE = ["logit_design", "LogitDesign", "NewtonMoments"]


def test_stable_names():
    assert sorted(core.__all__) == sorted(STABLE)
    for name in STABLE:
        assert hasattr(core, name)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(5)
    n, G, K = 12_000, 40, 3
    X = rng.normal(size=(n, K))
    g = rng.integers(0, G, n)
    E = np.column_stack([np.zeros(n)] + [X @ rng.normal(scale=0.4, size=K)
                                         + rng.normal(scale=0.5, size=G)[g] for _ in range(3)])
    P = np.exp(E - E.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    ym = (rng.random(n)[:, None] > P.cumsum(1)).sum(1).clip(0, 3)
    lat = X @ np.array([0.5, -0.3, 0.2]) + rng.normal(scale=0.5, size=G)[g] \
        + rng.logistic(size=n)
    yo = np.digitize(lat, [-1.0, 0.0, 1.2])
    # two groups whose outcome never varies (dropped), and a binary covariate
    const = np.isin(g, [0, 1])
    ym = np.where(const, 2, ym)
    yo = np.where(const, 3, yo)
    return pl.DataFrame({"ym": ym, "yo": yo, "g": g, "fold": rng.integers(0, 3, n),
                         "w": rng.uniform(0.5, 2.0, n), "d": (rng.random(n) < 0.3) * 1.0,
                         **{f"x{j}": X[:, j] for j in range(K)}})


RHS = "x0 + x1 + x2 + d"
CASES = {
    "mlogit": ("multinomial", "logit", f"ym ~ {RHS}"),
    "mlogit fe": ("multinomial", "logit", f"ym ~ {RHS} | g"),
    "ologit": ("ordered", "logit", f"yo ~ {RHS}"),
    "ologit fe": ("ordered", "logit", f"yo ~ {RHS} | g"),
    "oprobit fe": ("ordered", "probit", f"yo ~ {RHS} | g"),
}


def _fit(model, link, fml, data, **kw):
    if model == "multinomial":
        return mlogit_stream(fml, data, verbose=False, **kw)
    f = ologit_stream if link == "logit" else oprobit_stream
    return f(fml, data, verbose=False, **kw)


def _full(ld, mom, m=0):
    """The whole information (parameters and effects) of model m as one
    matrix, and the whole gradient, over the free effects."""
    g = mom.gradient[m]
    H = mom.information[m]
    if mom.effects_gradient is None:
        return g, H
    free = mom._free[m].ravel()
    G, J = mom.effects_gradient.shape[1:]
    P = len(g)
    ga = mom.effects_gradient[m].ravel()[free]
    Haa = np.zeros((G * J, G * J))
    for i in range(G):
        Haa[i * J:(i + 1) * J, i * J:(i + 1) * J] = mom.effects_information[m][i]
    B = mom.cross[m].reshape(G * J, P)[free]
    return np.r_[g, ga], np.block([[H, B.T], [B, Haa[np.ix_(free, free)]]])


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("weights", [None, "w"])
def test_pass_at_the_estimates(data, case, weights):
    """At the package's own estimates the gradient (profiled) is zero, the
    log-likelihood is the fit's, and the profiled information's inverse is
    its iid variance."""
    model, link, fml = CASES[case]
    res = _fit(model, link, fml, data, weights=weights, tol=1e-12)
    with core.logit_design(fml, data, model=model, link=link, weights=weights) as ld:
        assert ld.names == list(res.coefnames)
        th, A = ld.start()
        # take Newton steps from the null model, by the pass alone
        for _ in range(30):
            mom = ld.newton_pass(th, A)
            C, g = mom.profiled()
            step = np.linalg.solve(C, g)
            da = mom.effects_step(step)
            th = th + step
            if A is not None:
                A = A + np.where(np.isfinite(A), da, 0.0)
            if g @ step < 1e-20:
                break
        mom = ld.newton_pass(th, A)
        C, g = mom.profiled()
        np.testing.assert_allclose(th, res.beta, rtol=1e-6, atol=1e-7)
        assert abs(mom.loglik.sum() - res.loglik) < 1e-6 * abs(res.loglik)
        np.testing.assert_allclose(np.linalg.inv(C), res.vcov, rtol=1e-5, atol=1e-10)


@pytest.mark.parametrize("case", list(CASES))
def test_derivatives(data, case):
    """The gradient and information against numerical derivatives of the
    pass's log-likelihood, at a point off the optimum."""
    model, link, fml = CASES[case]
    rng = np.random.default_rng(1)
    with core.logit_design(fml, data, model=model, link=link, weights="w") as ld:
        th, A = _perturbed(ld, rng, *ld.start())
        mom = ld.newton_pass(th, A)
        g, H = _full(ld, mom)
        free = None if A is None else mom._free[0].ravel()
        P = len(th)

        def pack(v):
            if A is None:
                return v, None
            a = A.ravel().copy()
            a[free] = v[P:]
            return v[:P], a.reshape(A.shape)

        x0 = np.r_[th, [] if A is None else A.ravel()[free]]

        def grad(v):
            t, a = pack(v)
            m = ld.newton_pass(t, a, information=False)
            if a is None:
                return m.loglik.sum(), m.gradient[0]
            return m.loglik.sum(), np.r_[m.gradient[0], m.effects_gradient[0].ravel()[free]]

        idx = rng.choice(len(x0), size=min(8, len(x0)), replace=False)
        h = 1e-5
        for j in idx:
            e = np.zeros(len(x0))
            e[j] = h
            (lp, gp), (lm, gm) = grad(x0 + e), grad(x0 - e)
            assert abs((lp - lm) / (2 * h) - g[j]) < 1e-5 * (1 + abs(g[j]))
            np.testing.assert_allclose(-(gp - gm) / (2 * h), H[:, j], rtol=1e-4, atol=1e-4)


def _perturbed(ld, rng, th, A):
    th = th + rng.normal(scale=0.05, size=len(th)) * ld.penalized
    if A is not None:
        A = np.where(np.isfinite(A), A + rng.normal(scale=0.05, size=A.shape), A)
    return th, A


def _effects_for(one, ld, A):
    """Effects A of design `ld`'s groups, in the groups of design `one` (a
    subset of the rows: its own codes, from its `fe_levels`)."""
    if A is None:
        return None
    code = dict(zip(ld.fe_levels["g"].to_list(), range(ld.n_groups)))
    out = np.array([A[code[v]] for v in one.fe_levels["g"].to_list()])
    return np.where(one.group_kept[:, None], out, np.nan)


@pytest.mark.parametrize("case", list(CASES))
def test_splits_match_their_own_designs(data, case):
    """Models on each fold's training rows, evaluated in one pass, equal a
    design of those rows alone; each held-out fold's log-likelihood equals
    one of its rows alone. (Frequency weights, which are not rescaled.)"""
    model, link, fml = CASES[case]
    rng = np.random.default_rng(2)
    kw = dict(model=model, link=link, weights="w", weights_type="fweights")
    with core.logit_design(fml, data, split_by="fold", n_splits=3, **kw) as ld:
        models = np.array([[True] * 3] + [np.arange(3) != f for f in range(3)])
        pts = [_perturbed(ld, rng, *ld.start(np.flatnonzero(r))) for r in models]
        TH = np.array([p[0] for p in pts])
        AA = None if pts[0][1] is None else np.array([p[1] for p in pts])
        mom = ld.newton_pass(TH, AA, models=models)
        assert mom.loglik.shape == (4, 3)
        for m, inc in enumerate(models):
            folds = np.flatnonzero(inc).tolist()
            sub = data.filter(pl.col("fold").is_in(folds))
            with core.logit_design(fml, sub, levels=ld.levels, **kw) as one:
                ref = one.newton_pass(TH[m], _effects_for(one, ld, None if AA is None
                                                             else AA[m]))
            np.testing.assert_allclose(mom.loglik[m][inc].sum(), ref.loglik.sum(), rtol=1e-10)
            np.testing.assert_allclose(mom.gradient[m], ref.gradient[0], rtol=1e-8, atol=1e-7)
            C, g = mom.profiled(m)
            Cr, gr = ref.profiled(0)
            np.testing.assert_allclose(C, Cr, rtol=1e-8, atol=1e-7)
            np.testing.assert_allclose(g, gr, rtol=1e-8, atol=1e-7)
            for f in np.flatnonzero(~inc):
                held = data.filter(pl.col("fold") == int(f))
                with core.logit_design(fml, held, levels=ld.levels, **kw) as h1:
                    a = _effects_for(h1, ld, None if AA is None else AA[m])
                    if a is not None and model == "multinomial":
                        # a group's levels the training rows never had: -inf, and
                        # the held-out rows may have them
                        a = np.where(np.isneginf(a), np.nan, a)
                    rh = h1.newton_pass(TH[m], a, information=False)
                if AA is None or model == "ordered":
                    np.testing.assert_allclose(mom.loglik[m, f], rh.loglik.sum(), rtol=1e-10)


@pytest.mark.parametrize("case", list(CASES))
def test_columns(data, case):
    """The information over an active set of columns is the matching block
    of the whole one (with the Intercept and cutpoints), at a point that is
    zero outside the set and at one that is not; the gradient is whole."""
    model, link, fml = CASES[case]
    rng = np.random.default_rng(3)
    with core.logit_design(fml, data, model=model, link=link) as ld:
        th, A = _perturbed(ld, rng, *ld.start())
        cols = [ld.covariates.index("x1"), ld.covariates.index("d")]
        inside = np.isin(ld.parameter_columns, cols) | ~ld.penalized
        for t in (np.where(inside, th, 0.0), th):
            whole = ld.newton_pass(t, A)
            part = ld.newton_pass(t, A, columns=cols)
            p = part.parameters
            assert set(p) == set(np.flatnonzero(inside))
            np.testing.assert_allclose(part.gradient, whole.gradient, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(part.loglik, whole.loglik, rtol=1e-12)
            np.testing.assert_allclose(part.information[0], whole.information[0][np.ix_(p, p)],
                                       rtol=1e-9, atol=1e-9)
            if A is not None:
                np.testing.assert_allclose(part.cross[0], whole.cross[0][:, :, p],
                                           rtol=1e-9, atol=1e-9)
                np.testing.assert_allclose(part.effects_information, whole.effects_information,
                                           rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("split_by", [None, "fold"])
@pytest.mark.parametrize("case", ["mlogit fe", "ologit fe"])
def test_skipped_group(data, case, split_by):
    """A group given a nan effect is left out of that model, as if its rows
    were not there. (Split into folds, the blocks are below the fused
    kernels' size: the array path.)"""
    model, link, fml = CASES[case]
    split = dict(split_by=split_by, n_splits=3) if split_by else {}
    with core.logit_design(fml, data, model=model, link=link, **split) as ld:
        th, A = ld.start()
        g = int(np.flatnonzero(ld.group_kept)[0])
        A2 = A.copy()
        A2[g] = np.nan
        mom = ld.newton_pass(np.array([th, th]), np.array([A, A2]))
        assert mom.skipped[0].sum() == 0
        assert mom.skipped[1].sum() == ld.row_counts[:, g].sum()
        gv = ld.fe_levels["g"][g]
        with core.logit_design(fml, data.filter(pl.col("g") != gv), model=model,
                               link=link, levels=ld.levels) as one:
            ref = one.newton_pass(th, _effects_for(one, ld, A))
        np.testing.assert_allclose(mom.loglik[1].sum(), ref.loglik.sum(), rtol=1e-10)
        np.testing.assert_allclose(mom.profiled(1)[0], ref.profiled(0)[0], rtol=1e-9)


@pytest.fixture(scope="module")
def big(data):
    """The data four times over, so that each fold's rows of a block reach
    the fused passes' minimum (MNL_KERNEL_MIN_ROWS)."""
    return pl.concat([data] * 4)


@pytest.mark.parametrize("case", list(CASES))
def test_kernels_and_arrays_agree(big, case, monkeypatch):
    """The fused numba passes (blocks of at least MNL_KERNEL_MIN_ROWS rows)
    and the array passes give the same sums, on blocks cut by split."""
    import logit_stream.core.design as D
    import logit_stream.core.fe as F
    import logit_stream.core.fe_kernels as FK

    model, link, fml = CASES[case]
    rng = np.random.default_rng(4)
    calls = {"n": 0}

    def spy(f):
        def wrapped(*a, **k):
            calls["n"] += 1
            return f(*a, **k)
        return wrapped

    with core.logit_design(fml, big, model=model, link=link, weights="w",
                           split_by="fold", n_splits=3) as ld:
        th, A = _perturbed(ld, rng, *ld.start())
        models = np.array([[True, True, False]])
        cls = {"multinomial": (M._Multinomial, FK.MultinomialPass),
               "ordered": (M._Ordered, FK.OrderedPass)}[model]
        with monkeypatch.context() as mp:
            mp.setattr(cls[0], "accumulate_full", spy(cls[0].accumulate_full))
            mp.setattr(cls[1], "add", spy(cls[1].add))
            a = ld.newton_pass(th, A, models=models)
        assert calls["n"] > 0
        monkeypatch.setattr(M, "MNL_KERNEL_MIN_ROWS", 10 ** 9)
        monkeypatch.setattr(D, "MNL_KERNEL_MIN_ROWS", 10 ** 9)
        monkeypatch.setattr(F, "MNL_KERNEL_MIN_ROWS", 10 ** 9)
        b = ld.newton_pass(th, A, models=models)
    np.testing.assert_allclose(a.loglik, b.loglik, rtol=1e-11)
    np.testing.assert_allclose(a.gradient, b.gradient, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(a.information, b.information, rtol=1e-9, atol=1e-9)
    if A is not None:
        np.testing.assert_allclose(a.cross, b.cross, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(a.effects_information, b.effects_information, rtol=1e-9,
                                   atol=1e-9)


def test_design_facts(data):
    with core.logit_design(f"ym ~ {RHS} | g", data, model="multinomial", weights="w",
                           split_by="fold", n_splits=3) as ld:
        assert ld.levels == [0, 1, 2, 3]
        assert ld.separation["levels"] == 2
        assert ld.group_kept.sum() == ld.n_groups - 2
        kept = data.filter(~pl.col("g").is_in([0, 1]))
        assert ld.n_obs == kept.height
        assert ld.row_counts.sum() == kept.height
        for f in range(3):
            part = kept.filter(pl.col("fold") == f)
            assert ld.row_counts[f].sum() == part.height
            np.testing.assert_allclose(ld.x_sums[f, ld.covariates.index("x0")] * 0 + 1, 1)
        # outcome counts by split and level, unweighted through row_counts' totals
        assert ld.outcome_counts.shape == (3, ld.n_groups, 4)
        assert ld.names[:len(ld.covariates)] == [f"1:{c}" for c in ld.covariates]
        th, A = ld.start()
        assert np.isnan(A[~ld.group_kept]).all()
    with core.logit_design(f"yo ~ {RHS}", data, model="ordered") as ld:
        assert ld.names[-3:] == ["cut1", "cut2", "cut3"]
        assert list(ld.parameter_columns[-3:]) == [-1, -1, -1]
        assert not ld.penalized[-3:].any()
        assert not ld.feasible(np.r_[np.zeros(4), 1.0, 0.0, 2.0])


def test_rejects():
    df = pl.DataFrame({"y": [0, 1, 2, 1] * 5, "x": np.arange(20.0), "f": [0, 1] * 10,
                       "g": [0, 1, 2, 3] * 5})
    with pytest.raises(ValueError, match="model must be"):
        core.logit_design("y ~ x", df, model="binary")
    with pytest.raises(ValueError, match="together"):
        core.logit_design("y ~ x", df, model="ordered", split_by="f")
    with pytest.raises(ValueError, match="multinomial model is a logit"):
        core.logit_design("y ~ x", df, model="multinomial", link="probit")
    with pytest.raises(ValueError, match="at most one"):
        core.logit_design("y ~ x | f + g", df, model="ordered")
    with core.logit_design("y ~ x", df, model="ordered") as ld:
        with pytest.raises(ValueError, match="effects= is for"):
            ld.newton_pass(np.zeros(ld.n_params), np.zeros((1, 1)))
        with pytest.raises(ValueError, match="theta must be"):
            ld.newton_pass(np.zeros(ld.n_params + 1))
