"""Constrained assignment (experimental): each sampler's draws against
the distribution it targets, by enumerating a tiny firm's assignments
(the firm copied as many independent groups, so one call gives many
draws): SwapMCMC and ExactSmall, exact and soft, against the conditional
distribution; Balanced against the calibrated probabilities. And: the
targets' rounding, invariance to the batches, the hooks of a subclass,
the warning, the errors, and the result's method."""

import itertools
import warnings

import hdfe_stream
import numpy as np
import polars as pl
import pytest
from scipy.special import logsumexp

import logit_stream
from logit_stream import assign as A
from logit_stream import clogit_stream

pytestmark = pytest.mark.filterwarnings("ignore::logit_stream.ExperimentalWarning")

# the firm: 6 workers, 3 establishments; workers 4 and 5 cannot reach the third
AVAIL = [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1], [0, 1]]
U = np.array([[0.3, -0.2, 0.5], [-0.4, 0.1, 0.0], [0.8, 0.2, -0.3], [0.0, 0.6, 0.4],
              [0.2, -0.5, np.nan], [-0.1, 0.3, np.nan]])
N = np.array([3.0, 2.0, 1.0])


def _firms(G):
    """G copies of the firm, each its own group (its own establishments)."""
    rows = [(f"{g}_{i}", 3 * g + j, U[i, j]) for g in range(G) for i in range(6)
            for j in AVAIL[i]]
    data = pl.DataFrame(rows, schema=["case", "alt", "u"], orient="row")
    totals = pl.DataFrame({"alt": np.arange(3 * G), "n": np.tile(N, G)})
    return data, totals


def _assignments():
    for a in itertools.product(*AVAIL):
        yield np.array(a)


def _marginals(logw):
    """P(worker i at j) under the weights exp(logw(a)) over assignments."""
    A_ = list(_assignments())
    lw = np.array([logw(a) for a in A_])
    p = np.exp(lw - logsumexp(lw))
    P = np.zeros((6, 3))
    for a, pa in zip(A_, p):
        P[np.arange(6), a] += pa
    return P


def _calibrated_delta(target, tol=1e-13):
    d = np.log(target)
    for _ in range(100_000):
        e = np.exp(U + d)
        P = np.where(np.isnan(e), 0, e)
        P /= P.sum(1, keepdims=True)
        step = np.log(target / P.sum(0))
        d += step
        if np.abs(step).max() < tol:
            return d, P
    raise RuntimeError


def _exact_logw(a):
    if not np.array_equal(np.bincount(a, minlength=3), N):
        return -np.inf
    return U[np.arange(6), a].sum()


def _draws(sampler, G=10_000, seed=0, **kw):
    data, totals = _firms(G)
    r = A.sample_assignments(data, totals, case="case", alt="alt", utility="u",
                             sampler=sampler, seed=seed, **kw).collect()
    r = r.with_columns(pl.col("case").str.split("_").list.get(1).cast(pl.Int64).alias("i"),
                       (pl.col("alt") % 3).alias("j"), (pl.col("alt") // 3).alias("g"))
    return r


def _freq(r):
    F = r.filter("assigned").group_by("i", "j").len()
    out = np.zeros((6, 3))
    out[F["i"].to_numpy(), F["j"].to_numpy()] = F["len"].to_numpy()
    return out / r["g"].n_unique()


def _check(r, P, G=10_000):
    """The draws' frequencies against P, within 4.5 binomial sds."""
    F = _freq(r)
    sd = np.sqrt(P * (1 - P) / G) + 1e-12
    assert np.all(np.abs(F - P) < 4.5 * sd + 1e-9), np.abs(F - P) / sd


@pytest.mark.parametrize("sampler", [A.SwapMCMC(sweeps=50), A.ExactSmall()],
                         ids=["swap", "exact_small"])
def test_exact_samplers_draw_the_conditional_distribution(sampler):
    r = _draws(sampler)
    counts = r.filter("assigned").group_by("g", "j").len()
    assert (counts.join(pl.DataFrame({"j": [0, 1, 2], "N": N}), on="j")
            .select((pl.col("len") == pl.col("N")).all()).item())
    assert (r.group_by("case").agg(pl.col("assigned").sum())["assigned"] == 1).all()
    _check(r, _marginals(_exact_logw))


@pytest.mark.parametrize("make", [lambda p: A.SwapMCMC(penalty=p, sweeps=100),
                                  lambda p: A.ExactSmall(penalty=p)],
                         ids=["swap", "exact_small"])
def test_soft_samplers_draw_the_penalized_distribution(make):
    pen = A.GaussianPenalty(sd=0.8, rel=0.0)
    delta, _ = _calibrated_delta(N)
    Ud = U + delta

    def logw(a):
        n = np.bincount(a, minlength=3)
        return Ud[np.arange(6), a].sum() + pen(n.astype(float), N, np.arange(3)).sum()

    P = _marginals(logw)
    r = _draws(make(pen))
    _check(r, P)
    # the counts vary (it is soft), around the totals
    counts = r.filter("assigned").group_by("g", "j").len()
    assert counts["len"].n_unique() > 1


def test_balanced_keeps_the_calibrated_probabilities_and_the_counts():
    sampler = A.Balanced()
    r = _draws(sampler)
    counts = r.filter("assigned").group_by("g", "j").len()
    assert counts.join(pl.DataFrame({"j": [0, 1, 2], "N": N}), on="j") \
        .select((pl.col("len") == pl.col("N")).all()).item()
    _, P = _calibrated_delta(N)
    _check(r, P)
    assert sampler.diagnostics["snapped"] == 0


def test_round_targets_keeps_the_sums_and_the_expectations():
    data, totals = _firms(4000)
    totals = totals.with_columns(pl.Series("n", np.tile([2.6, 2.1, 1.3], 4000)))
    # the start's rounding: in each group a 3 or a 2 at the first, summing to 6
    sampler = A.SwapMCMC(sweeps=1)
    seen = []

    class Spy(A.SwapMCMC):
        def targets(self, batch, seed):
            t = super().targets(batch, seed)
            seen.append(t.reshape(-1, 3))
            return t

    A.sample_assignments(data, totals, case="case", alt="alt", utility="u", sampler=Spy(sweeps=1),
                         seed=3).collect()
    T = np.concatenate(seen)
    assert np.all(T.sum(1) == 6) and np.all(T == np.round(T))
    np.testing.assert_allclose(T.mean(0), [2.6, 2.1, 1.3], atol=0.03)
    assert sampler.sweeps == 1


def test_draws_do_not_depend_on_the_batches():
    a = _draws(A.SwapMCMC(sweeps=20), G=200, seed=5)
    b = _draws(A.SwapMCMC(sweeps=20), G=200, seed=5, batch_rows=100)
    assert a["assigned"].to_list() == b["assigned"].to_list()
    c = _draws(A.SwapMCMC(sweeps=20), G=200, seed=6)
    assert a["assigned"].to_list() != c["assigned"].to_list()
    for s in (A.Balanced(), A.ExactSmall(), A.SwapMCMC(penalty=A.GaussianPenalty(), sweeps=20)):
        x = _draws(s, G=200, seed=5)
        y = _draws(s, G=200, seed=5, batch_rows=100)
        assert x["assigned"].to_list() == y["assigned"].to_list()


def test_a_subclass_customizes_the_targets_and_the_draw():
    """Targets given outright (a project's own reconciliation), and a draw
    written in Python a group at a time."""
    class Mine(A.ConstrainedSampler):
        def targets(self, batch, seed):
            return np.where(batch.level_group >= 0, np.array([4.0, 1.0, 1.0])[
                batch.level_id % 3], 0.0)

        def sample_group(self, group, seed):
            # the last workers fill the first establishments
            out = np.empty(group.n, np.int64)
            left = group.targets.astype(int).copy()
            for i in range(group.n - 1, -1, -1):
                for r in range(group.ls[i], group.ls[i + 1]):
                    if left[group.llev[r]] > 0:
                        left[group.llev[r]] -= 1
                        out[i] = group.rows[r]
                        break
            return out

    r = _draws(Mine(), G=10)
    counts = r.filter("assigned").group_by("g", "j").len()
    assert counts.sort("g", "j")["len"].to_list() == [4, 1, 1] * 10


def test_the_warning_and_the_errors():
    assert issubclass(logit_stream.ExperimentalWarning, hdfe_stream.ExperimentalWarning)
    data, totals = _firms(2)
    with pytest.warns(logit_stream.ExperimentalWarning, match="experimental"):
        A.sample_assignments(data, totals, case="case", alt="alt", utility="u")
    # targets the alternatives cannot meet: 5 at the third, which 4 workers reach
    bad = totals.with_columns(pl.Series("n", [0.5, 0.5, 5.0, 3.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="cannot be met"):
        A.sample_assignments(data, bad, case="case", alt="alt", utility="u").collect()
    with pytest.raises(ValueError, match="cannot be met"):
        A.sample_assignments(data, bad, case="case", alt="alt", utility="u",
                             sampler=A.ExactSmall()).collect()
    with pytest.raises(ValueError, match="max_cells"):
        A.sample_assignments(data, totals, case="case", alt="alt", utility="u",
                             sampler=A.ExactSmall(max_cells=10)).collect()
    with pytest.raises(ValueError, match="fallback"):
        A.ExactSmall(fallback=A.SwapMCMC(penalty=A.GaussianPenalty()))
    with pytest.raises(TypeError, match="ConstrainedSampler"):
        A.sample_assignments(data, totals, case="case", alt="alt", utility="u", sampler="swap")
    with pytest.raises(ValueError, match="twice"):
        A.sample_assignments(pl.concat([data, data.head(1)]), totals, case="case", alt="alt",
                             utility="u").collect()
    with pytest.raises(ValueError, match="without a total"):
        A.sample_assignments(data, totals.head(4), case="case", alt="alt",
                             utility="u").collect()


def test_the_results_method():
    """From a fit: the utilities without the establishments' constants; a
    draw's coefficients; the totals scaled and rounded; rows in order."""
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from test_clogit_fe import _data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = clogit_stream("chosen ~ dist | estab", _data(seed=8), case="case", verbose=False)
    new = _data(seed=9, F=20).with_columns(pl.col("estab") + 10_000)
    tot = new.group_by("estab").agg(pl.col("chosen").sum().cast(pl.Float64).alias("n"))
    with pytest.warns(logit_stream.ExperimentalWarning):
        lf = fit.sample_assignments(new.drop("chosen"), {"estab": tot}, seed=2)
    r = lf.collect()
    assert r["case"].to_list() == new["case"].to_list()
    got = r.group_by("estab").agg(pl.col("assigned").sum()).join(tot, on="estab")
    assert (got["assigned"] == got["n"]).all()
    d = fit.draw_parameters(1, seed=1)[0]
    r2 = d.sample_assignments(new.drop("chosen"), {"estab": tot}, seed=2).collect()
    assert r2["assigned"].to_list() != r["assigned"].to_list()
