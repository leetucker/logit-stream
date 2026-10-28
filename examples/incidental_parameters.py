"""The incidental-parameter problem: the fixed-effects logit's bias in short
panels, the analytic bias correction, and Chamberlain's conditional logit,
which has none.

    python examples/incidental_parameters.py
"""

import warnings

import numpy as np
import polars as pl

from logit_stream import cfelogit_stream, logit_stream, mlogit_stream, ologit_stream
from logit_stream.simulate import simulate_panel

# Balanced panels with a binary outcome: P(y = 1) = logistic(a_unit + 1.0 x1
# - 0.5 x2), the unit effect a correlated with x1. About 20,000 rows each.
TRUTH = {"x1": 1.0, "x2": -0.5}
rows = []
for T in (2, 3, 4, 6, 10, 20):
    panel = simulate_panel(n_units=20_000 // T, periods=T, seed=T)
    with warnings.catch_warnings():
        # units whose outcome never varies are dropped (separation), and the
        # correction warns that it is unreliable for very short panels
        warnings.simplefilter("ignore")
        fe = logit_stream("y ~ x1 + x2 | unit", panel, verbose=False)
        bc = logit_stream("y ~ x1 + x2 | unit", panel, bias_correction=True, verbose=False)
    cfe = cfelogit_stream("y ~ x1 + x2 | unit", panel, verbose=False)
    rows.append({"T": T, "units": cfe.n_units, "FE logit": fe.coef()["x1"],
                 "bias-corrected": bc.coef()["x1"], "conditional": cfe.coef()["x1"],
                 "conditional se": cfe.se[0]})
print("x1 (truth 1.0):")
print(pl.DataFrame(rows))

# ---- one panel in detail: T = 4
panel = simulate_panel(n_units=5_000, periods=4, seed=4)
cfe = cfelogit_stream("y ~ x1 + x2 | unit", panel, vcov="hetero", verbose=False)
cfe.summary()
print(f"units used {cfe.n_units:,}, dropped (all 0 or all 1) {cfe.n_units_dropped:,}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    bc = logit_stream("y ~ x1 + x2 | unit", panel, bias_correction=True, verbose=False)
d = bc.diagnostics["bias_correction"]
print("uncorrected:", {k: round(v, 3) for k, v in d["beta_uncorrected"].items()})
print("bias:       ", {k: round(v, 3) for k, v in d["bias"].items()})
print("corrected:  ", {k: round(float(v), 3) for k, v in bc.coef().items()})

# ---- ignoring the effects, or treating them as random (independent of
# the covariates), is biased too, for another reason: x1 is correlated with
# the effects, so it picks up some of theirs
pooled = logit_stream("y ~ x1 + x2", panel, verbose=False)
random = mlogit_stream("y ~ x1 + x2", panel, levels=[0, 1], random="unit", verbose=False)
print(pl.DataFrame({
    "coefficient": ["x1", "x2"], "truth": [TRUTH["x1"], TRUTH["x2"]],
    "pooled": [pooled.coef()[c] for c in ("x1", "x2")],
    "random effects": [random.coef()[f"1:{c}"] for c in ("x1", "x2")],
    "fixed effects": [bc.diagnostics["bias_correction"]["beta_uncorrected"][c]
                      for c in ("x1", "x2")],
    "conditional": [cfe.coef()[c] for c in ("x1", "x2")]}))

# ---- correlated random effects (Mundlak 1978): the random-effects model
# with each unit's means of the covariates, which absorb the part of the
# effect correlated with them
means = panel.with_columns(x1_bar=pl.col("x1").mean().over("unit"),
                           x2_bar=pl.col("x2").mean().over("unit"))
mundlak = mlogit_stream("y ~ x1 + x2 + x1_bar + x2_bar", means, levels=[0, 1],
                        random="unit", verbose=False)
print("correlated random effects:", {c: round(float(mundlak.coef()[f"1:{c}"]), 3)
                                     for c in ("x1", "x2")})

# ---- the ordered logit: its fixed-effects fit is biased too. The "blow-up
# and cluster" estimator (Baetschmann, Staub and Winkelmann 2015) fits the
# conditional logit to every dichotomization 1[y >= k] of the outcome,
# stacked, each unit and cut its own group, clustered by the unit.
rng = np.random.default_rng(5)
N, T = 3_000, 4
unit = np.repeat(np.arange(N), T)
a = rng.normal(0, 1, N)[unit]
x = 0.5 * a + rng.normal(size=N * T)
y = np.digitize(a + 1.0 * x + rng.logistic(size=N * T), [-1.0, 0.5, 2.0])  # 0, 1, 2, 3
ordered = pl.DataFrame({"unit": unit, "x": x, "y": y})
stacked = pl.concat([ordered.with_columns(yk=(pl.col("y") >= k).cast(pl.Int8),
                                          group=pl.col("unit") * 10 + k)
                     for k in (1, 2, 3)])
buc = cfelogit_stream("yk ~ x | group", stacked, vcov={"CRV1": "unit"}, verbose=False)
ofe = ologit_stream("y ~ x | unit", ordered, verbose=False)
print(f"ordered logit, x (truth 1.0): fixed effects {ofe.coef()['x']:.3f}, "
      f"blow-up and cluster {buc.coef()['x']:.3f} (se {buc.se[0]:.3f})")

# ---- what the conditional logit cannot do: the unit effects are not
# estimated, so it predicts only x'b, and has no fixed effects to report
print(cfe.predict(panel).select("unit", "period", "xb").head(3).collect())
