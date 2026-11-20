"""Multiple imputation: drawing outcomes, and combining the analyses.

Drawing. ``predict(type="draw", seed=m)`` draws each row's outcome from its
predicted distribution: a 0 or 1 for the binary logit and probit, a level
for the ordered and multinomial logit, one alternative per choice set for
the conditional logit. Each draw comes from a uniform that depends only on
the seed and the row's identifier (`id=`, or the choice set's columns), not
on how the frame is split into batches. With the parameters from
`draw_parameters`, imputation m is ``draws[m].predict(newdata,
type="draw", seed=m)``.

Pooling (Rubin 1987). Each completed data set m is analyzed as if it were
complete, giving an estimate Q_m and its variance U_m. Over the M analyses,

    Q = mean(Q_m)                      the pooled estimate
    W = mean(U_m)                      the within-imputation variance
    B = var(Q_m)                       the between-imputation variance
    T = W + (1 + 1/M) B                the total variance, se = sqrt(T)

with a t reference distribution on nu = (M - 1) (1 + 1/r)^2 degrees of
freedom, r = (1 + 1/M) B / W the relative increase in variance, or Barnard
and Rubin's (1999) small-sample degrees of freedom when the complete-data
degrees of freedom `df_com` are given. `pool` does this statistic by
statistic, for a frame of many statistics; `pool_matrix` for a vector of
estimates with covariance matrices, such as a regression's coefficients,
with the joint Wald test of Li, Raghunathan and Rubin (1991).

When the data are a whole population (an administrative file, say), a
statistic has no sampling variance: pass no standard errors, so that W = 0
and all of T is the imputation's, (1 + 1/M) B.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numba as nb
import numpy as np
import polars as pl
from scipy import stats

from .draws import _mix, _uniform

__all__ = ["pool", "pool_matrix", "PooledEstimate"]

_STREAM = 0x4F7574636F6D65        # the outcome draws' stream ("Outcome")


def _key(seed: int) -> np.uint64:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    return np.uint64(_mix(np.uint64((int(seed) * 0x9E3779B97F4A7C15 + _STREAM) % 2 ** 64)))


@nb.njit(cache=True)
def _uniforms(key, ids):
    out = np.empty(ids.shape[0])
    for i in range(ids.shape[0]):
        out[i] = _uniform(key, ids[i], 0)
    return out


def _as_u64(s: pl.Series) -> np.ndarray:
    """A row identifier as 64 bits: an integer column's own value; anything
    else (strings, several columns as a struct) by Polars' hash, which is
    stable within a Polars version."""
    if s.dtype.is_integer():
        return s.cast(pl.Int64).to_numpy().view(np.uint64)
    return s.hash(seed=0).to_numpy()


def uniform_expr(id_cols: list[str], seed: int) -> pl.Expr:
    """A uniform on (0, 1) per row, from the seed and the row's identifier
    (`id_cols`): the same for the same identifier and seed, whatever the
    frame's batches."""
    key = _key(seed)
    src = pl.col(id_cols[0]) if len(id_cols) == 1 else pl.struct(id_cols)

    def fn(s: pl.Series) -> pl.Series:
        if s.null_count():
            raise ValueError("the rows' identifier (id=) has nulls")
        return pl.Series(s.name, _uniforms(key, np.ascontiguousarray(_as_u64(s))))

    return src.map_batches(fn, return_dtype=pl.Float64, is_elementwise=True)


def with_ids(frame: pl.LazyFrame, id) -> tuple[pl.LazyFrame, list[str], list[str]]:
    """`frame` with the draws' identifier: the `id` column(s), or the row's
    position. Returns (frame, the identifier's columns, columns to drop)."""
    if id is None:
        return frame.with_row_index("__ls_row"), ["__ls_row"], ["__ls_row"]
    cols = [id] if isinstance(id, str) else list(id)
    names = frame.collect_schema().names()
    missing = [c for c in cols if c not in names]
    if missing:
        raise ValueError(f"id column(s) {missing} not in newdata")
    return frame, cols, []


def categorical(P: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Each row's draw from its probabilities P (rows x levels) by the
    inverse cdf at u: the level's index, or -1 for a row without
    probabilities."""
    ok = ~np.isnan(P).any(axis=1)
    C = np.cumsum(np.where(ok[:, None], P, 0.0), axis=1)
    C[:, -1] = np.inf                       # rounding: the last level takes the rest
    idx = (C < u[:, None]).sum(axis=1)
    return np.where(ok, idx, -1)


def draw_choice(lf: pl.LazyFrame, prob: str, case: list[str], seed: int,
                name: str, alt: list[str] | None = None) -> pl.LazyFrame:
    """One alternative per choice set, from the probabilities in `prob`:
    `name` is True at the drawn row, False at the set's others, null where
    the set has no probabilities. The uniform is the set's (from `case`);
    the alternatives are taken in the order of `alt` (default: the rows')."""
    u = uniform_expr(case, seed)
    p = pl.col(prob).fill_null(0.0)
    tot = p.sum().over(case)
    cum = p.cum_sum().over(case, order_by=alt) if alt else p.cum_sum().over(case)
    target = u * tot
    drawn = (cum - p < target) & (cum >= target) & (p > 0)
    return lf.with_columns(pl.when(tot > 0).then(drawn).otherwise(None).alias(name))


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------

def _df(M, W, B, df_com):
    """Rubin's degrees of freedom, or Barnard and Rubin's with `df_com`;
    and the fraction of missing information (as mice reports it)."""
    W = np.asarray(W, float)
    B = np.asarray(B, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(W + (1 + 1 / M) * B > 0,
                       (1 + 1 / M) * B / (W + (1 + 1 / M) * B), 0.0)
        nu_old = np.where(lam > 0, (M - 1) / lam ** 2, np.inf)
        if df_com is not None:
            nu_obs = (df_com + 1) / (df_com + 3) * df_com * (1 - lam)
            nu = np.where(np.isinf(nu_old), nu_obs, nu_old * nu_obs / (nu_old + nu_obs))
        else:
            nu = nu_old
        r = np.where(W > 0, (1 + 1 / M) * B / W, np.where(B > 0, np.inf, 0.0))
        fmi = np.where(np.isinf(r), 1.0,
                       np.where(np.isinf(nu), lam, (r + 2 / (nu + 3)) / (r + 1)))
    return nu, r, fmi


def _table(Q, W, B, M, df_com, level):
    T = W + (1 + 1 / M) * B
    se = np.sqrt(T)
    nu, r, fmi = _df(M, W, B, df_com)
    crit = np.where(np.isinf(nu), stats.norm.ppf(0.5 + level / 2),
                    stats.t.ppf(0.5 + level / 2, np.where(np.isinf(nu), 1.0, nu)))
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = np.where(se > 0, Q / se, np.nan)
        p = np.where(np.isinf(nu), 2 * stats.norm.sf(np.abs(tval)),
                     2 * stats.t.sf(np.abs(tval), np.where(np.isinf(nu), 1.0, nu)))
    return {"estimate": Q, "se": se, "df": nu, "ci_low": Q - crit * se,
            "ci_high": Q + crit * se, "t": tval, "p": p, "within": W, "between": B,
            "riv": r, "fmi": fmi, "m": np.full(len(Q), M, dtype=np.int64)}


def pool(data: Any, estimate: str = "estimate", se: str | None = "se", *,
         by: str | list | None = None, df_com: float | None = None,
         level: float = 0.95) -> pl.DataFrame:
    """Combine scalar estimates from M analyses of completed data sets by
    Rubin's rules, statistic by statistic.

    `data` has one row per imputation and statistic: the statistic's
    identifying columns `by` (none for a single statistic), its estimate
    (column `estimate`) and the estimate's standard error as if the data
    were complete (column `se`). ``se=None`` means no sampling variance,
    for statistics of a whole population, so that only the imputation's
    variance remains. `df_com` gives the complete-data degrees of freedom
    for Barnard and Rubin's (1999) small-sample degrees of freedom.

    Returns a row per statistic: `by`, ``estimate`` (the mean over
    imputations), ``se`` (sqrt of the total variance), ``df``, the
    `level` interval ``ci_low``/``ci_high`` (t on ``df``), ``t`` and ``p``
    (for estimate = 0), ``within`` (W, the mean squared standard error),
    ``between`` (B, the estimates' variance over imputations), ``riv``
    (the relative increase in variance, (1 + 1/M) B / W), ``fmi`` (the
    fraction of missing information) and ``m`` (the imputations).
    Every statistic needs the same M, at least 2.
    """
    df = data.collect() if isinstance(data, pl.LazyFrame) else pl.DataFrame(data)
    by = [] if by is None else ([by] if isinstance(by, str) else list(by))
    for c in [*by, estimate, *([se] if se else [])]:
        if c not in df.columns:
            raise ValueError(f"column {c!r} not in the data")
    q = pl.col(estimate).cast(pl.Float64)
    aggs = [q.mean().alias("__Q"), q.var(ddof=1).alias("__B"), pl.len().alias("__M")]
    aggs.append((pl.col(se).cast(pl.Float64) ** 2).mean().alias("__W") if se
                else pl.lit(0.0).alias("__W"))
    g = (df.group_by(by, maintain_order=True).agg(aggs) if by
         else df.select(aggs))
    Ms = g["__M"].unique().to_list()
    if len(Ms) != 1:
        raise ValueError(f"every statistic needs the same number of imputations, got {sorted(Ms)}")
    M = int(Ms[0])
    if M < 2:
        raise ValueError("pooling needs at least 2 imputations")
    out = _table(g["__Q"].to_numpy(), g["__W"].to_numpy(),
                 g["__B"].fill_null(0.0).to_numpy(), M, df_com, level)
    return pl.concat([g.select(by), pl.DataFrame(out)], how="horizontal") if by \
        else pl.DataFrame(out)


@dataclass
class PooledEstimate:
    """Estimates of a vector pooled over imputations (`pool_matrix`).

    Attributes
    ----------
    names : list of str
    estimate : array (k,)
        The mean of the imputations' estimates.
    vcov : array (k, k)
        The total covariance, W + (1 + 1/M) B.
    within, between : arrays (k, k)
        W, the mean of the imputations' covariances; B, the covariance of
        their estimates.
    m : int
        The imputations.
    df_com : float or None
    """

    names: list
    estimate: np.ndarray
    vcov: np.ndarray
    within: np.ndarray
    between: np.ndarray
    m: int
    df_com: float | None = None
    _level: float = field(default=0.95, repr=False)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.vcov))

    def tidy(self, level: float = 0.95) -> pl.DataFrame:
        """A row per element, with the columns of `pool`."""
        out = _table(self.estimate, np.diag(self.within), np.diag(self.between),
                     self.m, self.df_com, level)
        return pl.concat([pl.DataFrame({"term": self.names}), pl.DataFrame(out)],
                         how="horizontal")

    def wald_test(self, R: Any = None, q: Any = None) -> dict:
        """The joint test of R @ estimate = q (default: every element 0),
        by Li, Raghunathan and Rubin's (1991) D1: the Wald statistic with
        the within covariance inflated by the average relative increase in
        variance, on an F(k, nu) reference. `R` is a matrix (or a list of
        the names to test, each against its q). Returns ``statistic``,
        ``df1``, ``df2``, ``p`` and ``riv``. The restricted within
        covariance must be invertible: with shares that sum to one, leave
        one out."""
        Q, W, B, M = self.estimate, self.within, self.between, self.m
        k_all = len(Q)
        if R is None:
            R = np.eye(k_all)
        elif isinstance(R, (list, tuple)) and R and isinstance(R[0], str):
            idx = [self.names.index(n) for n in R]
            R = np.eye(k_all)[idx]
        R = np.atleast_2d(np.asarray(R, float))
        k = R.shape[0]
        q = np.zeros(k) if q is None else np.broadcast_to(np.asarray(q, float), (k,))
        d = R @ Q - q
        WR, BR = R @ W @ R.T, R @ B @ R.T
        try:
            Wi = np.linalg.inv(WR)
            if np.linalg.cond(WR) > 1e14:
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            raise ValueError("the within-imputation covariance of the tested elements is "
                             "singular (no sampling variance, or elements that sum to a "
                             "constant): D1 needs it invertible") from None
        r = (1 + 1 / M) * np.trace(BR @ Wi) / k
        stat = float(d @ Wi @ d / (k * (1 + r)))
        t = k * (M - 1)
        if t > 4:
            df2 = 4 + (t - 4) * (1 + (1 - 2 / t) / r) ** 2 if r > 0 else np.inf
        else:
            df2 = t * (1 + 1 / k) * (1 + 1 / r) ** 2 / 2 if r > 0 else np.inf
        p = float(stats.chi2.sf(stat * k, k) if np.isinf(df2) else stats.f.sf(stat, k, df2))
        return {"statistic": stat, "df1": k, "df2": float(df2), "p": p, "riv": float(r)}


def pool_matrix(estimates: Any, vcovs: Any = None, *, names: list | None = None,
                df_com: float | None = None) -> PooledEstimate:
    """Combine vector estimates from M analyses of completed data sets by
    Rubin's rules, with their covariances.

    `estimates` is a list of fitted results (anything with ``beta``,
    ``vcov`` and ``coefnames``, such as this package's or hdfe-stream's),
    or an M x k array with `vcovs` a list of M k x k covariance matrices
    (None: no sampling variance). Returns a `PooledEstimate`: the
    estimate, the total covariance W + (1 + 1/M) B, `tidy()` for a table
    as `pool`'s, and `wald_test()` for joint tests.
    """
    items = list(estimates)
    if items and hasattr(items[0], "beta") and hasattr(items[0], "vcov"):
        if vcovs is not None:
            raise ValueError("vcovs is taken from the results; pass it only with arrays")
        names = list(items[0].coefnames) if names is None else names
        for it in items[1:]:
            if list(it.coefnames) != list(items[0].coefnames):
                raise ValueError("the results have different coefficients")
        Qs = np.array([np.asarray(it.beta, float) for it in items])
        Us = np.array([np.asarray(it.vcov, float) for it in items])
    else:
        Qs = np.atleast_2d(np.asarray(items, float))
        M0, k0 = Qs.shape
        Us = (np.zeros((M0, k0, k0)) if vcovs is None
              else np.array([np.atleast_2d(np.asarray(v, float)) for v in vcovs]))
        if Us.shape != (M0, k0, k0):
            raise ValueError(f"vcovs must be {M0} matrices of {k0} x {k0}, got {Us.shape}")
    M, k = Qs.shape
    if M < 2:
        raise ValueError("pooling needs at least 2 imputations")
    names = [f"x{j}" for j in range(k)] if names is None else list(names)
    if len(names) != k:
        raise ValueError(f"{len(names)} names for {k} estimates")
    Q = Qs.mean(axis=0)
    W = Us.mean(axis=0)
    B = np.cov(Qs, rowvar=False, ddof=1).reshape(k, k)
    return PooledEstimate(names=names, estimate=Q, vcov=W + (1 + 1 / M) * B, within=W,
                          between=B, m=M, df_com=df_com)
