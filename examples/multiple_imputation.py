"""Multiple imputation with group effects: workers in firms in industry x
area cells, a binary characteristic known for some workers and imputed for
the rest, with a random firm intercept within cell fixed effects.

For each imputation m:
  1. draw the parameters (coefficients, cell effects, the firm sd) from
     their estimated sampling distribution: `draw_parameters`;
  2. draw each firm's intercept delta* from its distribution given the
     firm's known workers, at the drawn parameters (below: a grid, ~30 lines);
  3. impute the unknown workers from the drawn log-odds plus their firm's
     delta*, one delta* per firm shared by its unknown workers.

    python examples/multiple_imputation.py
"""

import numpy as np
import polars as pl
from scipy.special import expit, log_expit

from logit_stream import logit_stream

# ---- simulated data: 20,000 firms in 50 cells; firm sizes 1 + floor(exp N(1, 1.1));
# age sorts across firms; firm effects with sd 0.8; 25% of workers unknown
rng = np.random.default_rng(0)
G, n_cells = 20_000, 50
cell = rng.integers(0, n_cells, G)
size = 1 + np.floor(np.exp(rng.normal(1.0, 1.1, G))).astype(int)
firm = np.repeat(np.arange(G), size)
n = len(firm)
age10 = 4.0 + 0.8 * rng.normal(size=G)[firm] + 0.9 * rng.normal(size=n)   # age / 10
x = rng.normal(size=n)
p = expit(rng.uniform(-2.5, 2.5, n_cells)[cell][firm] - 0.3 * (age10 - 4) + 0.5 * x
          + rng.normal(0, 0.8, G)[firm])
truth = (rng.random(n) < p).astype(int)
known = rng.random(n) >= 0.25
workers = pl.DataFrame({"worker": np.arange(n), "firm": firm, "cell": cell[firm],
                        "age10": age10, "x": x,
                        "female": np.where(known, truth, -1)})      # -1: unknown
obs = workers.filter(pl.col("female") >= 0)

# ---- the model, on the known workers
fit = logit_stream("female ~ age10 + x | cell", obs, random="firm",
                   random_method="adaptive", verbose=False)
fit.summary()
print(fit.fixef().head(3).collect())             # cell effects with their standard errors


# ---- step 2: each firm's intercept given its known workers, at a draw
def draw_firm_effects(group, y, eta, n_groups, tau, rng, n_grid=61, span=6.0):
    """One draw of each group's intercept d from its distribution given the
    group's rows (outcomes y, log-odds eta without the group intercept):
    proportional to prod_i Lambda(eta_i + d)^y_i (1 - Lambda(eta_i + d))^(1 - y_i)
    times N(d; 0, tau^2). On a grid around each group's mode; a group without
    rows draws from N(0, tau^2)."""
    d = np.zeros(n_groups)
    for _ in range(50):                                   # Newton for the modes
        pr = expit(eta + d[group])
        step = ((np.bincount(group, y - pr, n_groups) - d / tau**2)
                / (np.bincount(group, pr * (1 - pr), n_groups) + 1 / tau**2))
        d += np.clip(step, -2, 2)
        if np.abs(step).max() < 1e-10:
            break
    pr = expit(eta + d[group])
    sd = 1 / np.sqrt(np.bincount(group, pr * (1 - pr), n_groups) + 1 / tau**2)
    grid = d[:, None] + sd[:, None] * np.linspace(-span, span, n_grid)[None, :]
    z = eta[:, None] + grid[group]
    logpost = -0.5 * grid**2 / tau**2
    np.add.at(logpost, group, y[:, None] * log_expit(z) + (1 - y[:, None]) * log_expit(-z))
    w = np.exp(logpost - logpost.max(axis=1, keepdims=True))
    cdf = np.cumsum(w, axis=1)
    cdf /= cdf[:, -1:]
    u = rng.random(n_groups)
    k = (cdf < u[:, None]).sum(axis=1)                    # the grid cell of each draw
    lo = np.where(k > 0, cdf[np.arange(n_groups), np.maximum(k - 1, 0)], 0.0)
    frac = (u - lo) / (cdf[np.arange(n_groups), k] - lo)
    h = grid[:, 1] - grid[:, 0]
    return grid[np.arange(n_groups), k] + (frac - 0.5) * h


# ---- the imputations
M = 10
f_obs = obs["firm"].to_numpy()
y_obs = obs["female"].to_numpy().astype(float)
miss = workers["female"].to_numpy() < 0
imputed = []
for m, draw in enumerate(fit.draw_parameters(M, seed=1)):
    tau = draw.random_effects["firm"]["sd"][0]
    # step 1's log-odds for every worker: cell effect + covariates, without the firm's
    eta = draw.predict(workers, type="link").collect()["prediction"].to_numpy()
    delta = draw_firm_effects(f_obs, y_obs, eta[known], G, tau, rng)       # step 2
    y = workers["female"].to_numpy().copy()
    y[miss] = rng.random(miss.sum()) < expit(eta[miss] + delta[firm[miss]])   # step 3
    imputed.append(y)
    if m < 2:
        print(f"imputation {m}: sd {tau:.4f}, age10 {draw.beta[0]:.4f}, "
              f"first cell {draw.fixef().collect()['fe_cell'][0]:.4f}")

# ---- checks against the truth (which a real application does not have)
Y = np.stack(imputed)                                     # M x workers
share = Y[:, miss].mean(axis=1)
print(f"share female among the unknown: truth {truth[miss].mean():.4f}, "
      f"imputed {share.mean():.4f} (between-imputation sd {share.std(ddof=1):.4f})")
# firm by firm, among the unknown workers of firms with 10 or more known workers: the
# imputed share follows the true one, which the cell effects alone cannot
nk = np.bincount(firm[known], minlength=G)
sel = miss & (nk[firm] >= 10)
fm = firm[sel]
nm = np.bincount(fm, minlength=G)
ids = np.flatnonzero(nm)
imp_share = np.bincount(fm, Y[:, sel].mean(axis=0), G)[ids] / nm[ids]
true_share = np.bincount(fm, truth[sel], G)[ids] / nm[ids]
print(f"{len(ids):,} firms with 10+ known workers: correlation of the imputed and true shares "
      f"of their unknown workers {np.corrcoef(imp_share, true_share)[0, 1]:.3f}")
