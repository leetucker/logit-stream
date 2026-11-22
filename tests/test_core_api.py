"""logit_stream.core: its stable names, and the Newton pass against the
package's own fit and against numerical derivatives."""

import numpy as np
import polars as pl
import pytest

import logit_stream.core as core
from logit_stream import logit_stream

STABLE = ["logit_design", "LogitDesign", "NewtonMoments"]


def test_stable_names():
    assert sorted(core.__all__) == sorted(STABLE)
    for name in STABLE:
        assert hasattr(core, name)


@pytest.fixture(scope="module")
def data():
    r = np.random.default_rng(11)
    n = 5000
    df = pl.DataFrame({"x1": r.normal(size=n), "x2": r.normal(size=n),
                       "x3": r.uniform(0, 3, n), "g": r.integers(0, 30, n),
                       "w": r.uniform(.5, 2, n), "fold": r.integers(0, 4, n)})
    eta = (0.7 * df["x1"].to_numpy() - 0.4 * df["x2"].to_numpy()
           + (df["g"].to_numpy() % 5 - 2) * 0.3 - 0.2)
    y = (r.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    df = df.with_columns(pl.Series("y", y))
    # a level whose outcome never varies
    return df.with_columns(pl.when(pl.col("g") == 0).then(1).otherwise(pl.col("y")).alias("y"))


def _newton(design, models=None, m=0):
    """An unpenalized Newton fit from the passes (model m)."""
    k, G = len(design.names), design.n_groups
    M = 1 if models is None else len(models)
    b, a = np.zeros((M, k)), np.zeros((M, G))
    for _ in range(30):
        mom = design.newton_pass(b, a, models=models)
        C, g = mom.profiled(m)
        db = np.linalg.solve(C, g)
        da = mom.effects_step(db, m)
        b[m] += db
        a[m] += da
        if np.max(np.abs(db)) < 1e-12:
            break
    return b[m], a[m], mom


@pytest.mark.parametrize("fe", [False, True])
def test_newton_matches_logit_stream(data, tmp_path, fe):
    fml = "y ~ x1 + x2 + x3" + (" | g" if fe else "")
    with pytest.warns(UserWarning) if fe else _null():
        design = core.logit_design(fml, data, weights="w", workdir=tmp_path)
    with design:
        b, a, mom = _newton(design)
        with pytest.warns(UserWarning) if fe else _null():
            ref = logit_stream(fml, data, weights="w", workdir=tmp_path, verbose=False)
        coef = ref.coef()
        np.testing.assert_allclose(b, [coef[n] for n in design.names], rtol=1e-9, atol=1e-11)
        if not fe:
            assert a[0] == pytest.approx(coef["Intercept"], rel=1e-9)
        else:
            assert design.n_groups == 29 and design.separation["levels"] == 1
            assert design.fe_levels.height == 29
        assert design.n_obs == ref.n_obs
        assert np.abs(mom.gradient).max() < 1e-8
    assert design._run is None


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_derivatives_and_splits(data, tmp_path):
    with pytest.warns(UserWarning):
        design = core.logit_design("y ~ x1 + x2 + x3 | g", data, split_by="fold",
                                   n_splits=4, workdir=tmp_path)
    with design:
        k, G = 3, design.n_groups
        rng = np.random.default_rng(0)
        b, a = rng.normal(0, .2, k), rng.normal(0, .2, G)
        models = np.array([[True, True, True, True], [False, True, True, True],
                           [True, False, True, False]])
        B, A = np.tile(b, (3, 1)), np.tile(a, (3, 1))
        mom = design.newton_pass(B, A, models=models)
        # the log-likelihood on each split is the same for every model (same
        # coefficients); the sums over a model's splits are those splits'
        np.testing.assert_allclose(mom.loglik[0], mom.loglik[1])
        one = [design.newton_pass(b, a, models=np.eye(4, dtype=bool)[s][None])
               for s in range(4)]
        for m in range(3):
            sel = np.flatnonzero(models[m])
            np.testing.assert_allclose(mom.gradient[m], sum(one[s].gradient[0] for s in sel))
            np.testing.assert_allclose(mom.information[m],
                                       sum(one[s].information[0] for s in sel))
            np.testing.assert_allclose(mom.cross[m], sum(one[s].cross[0] for s in sel))
        # numerical derivatives of the log-likelihood over all splits
        def ll(bb, aa):
            return design.newton_pass(bb, aa, information=False).loglik.sum()
        eps = 1e-6
        num = [(ll(b + eps * e, a) - ll(b - eps * e, a)) / (2 * eps) for e in np.eye(k)]
        np.testing.assert_allclose(mom.gradient[0], num, rtol=1e-6)
        e0 = np.zeros(G)
        e0[3] = eps
        assert mom.effects_gradient[0, 3] == pytest.approx(
            (ll(b, a + e0) - ll(b, a - e0)) / (2 * eps), rel=1e-6)
        # an active set: the information's rows and columns of those covariates
        sub = design.newton_pass(b, a, columns=[0, 2])
        np.testing.assert_allclose(sub.information[0], mom.information[0][np.ix_([0, 2], [0, 2])])
        np.testing.assert_allclose(sub.gradient[0], mom.gradient[0])
        # a group held out by an infinite effect
        a2 = a.copy()
        a2[5] = np.inf
        held = design.newton_pass(b, a2)
        np.testing.assert_array_equal(held.skipped[0], design.outcome_counts[:, 5].sum(axis=1))
        assert held.effects_information[0, 5] == 0
        # the rows, as stored
        n = 0
        for X, y, w, f, s in design.rows():
            assert X.shape[1] == k and w is None and f.max() < G and s.max() < 4
            n += len(y)
        assert n == design.n_obs
        # the counts and moments per split
        assert design.outcome_counts.shape == (4, G, 2)
        assert design.outcome_counts.sum() == pytest.approx(design.n_obs)
        assert design.passes >= 12


def test_errors(data, tmp_path):
    with pytest.raises(ValueError, match="at most one"):
        core.logit_design("y ~ x1 | g + fold", data, workdir=tmp_path)
    with pytest.raises(ValueError, match="0 or 1"):
        core.logit_design("x3 ~ x1", data, workdir=tmp_path)
    with pytest.raises(ValueError, match="integers 0 to 2"):
        core.logit_design("y ~ x1", data, split_by="fold", n_splits=3, workdir=tmp_path)
    with pytest.raises(ValueError, match="together"):
        core.logit_design("y ~ x1", data, split_by="fold", workdir=tmp_path)
