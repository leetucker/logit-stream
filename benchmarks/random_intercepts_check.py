"""logit-stream's random intercepts against lme4 (binary: glmer with
nAGQ = 0, 1 and 25), ordinal::clmm (ordered: Laplace and adaptive
quadrature) and mclogit::mblogit (3-level multinomial, PQL), on simulated
data.

    RSCRIPT=/path/to/Rscript python benchmarks/random_intercepts_check.py SEED GROUPS MEAN_SIZE
    e.g. python benchmarks/random_intercepts_check.py 0 200 100

Needs R with lme4, ordinal, mclogit and arrow (e.g. conda-forge's r-base,
r-lme4, r-ordinal, r-mclogit, r-arrow). Writes its data and R's results to
benchmarks/results/random_check/.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from logit_stream import mlogit_stream, ologit_stream

R_SCRIPT = Path(__file__).with_suffix(".R")
HERE = Path(__file__).parent / "results" / "random_check"
HERE.mkdir(parents=True, exist_ok=True)
RS = os.environ.get("RSCRIPT", "Rscript")
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
G, nbar = int(sys.argv[2]), int(sys.argv[3])
sizes = rng.poisson(nbar, G) + 3
g = np.repeat(np.arange(G), sizes)
n = len(g)
x1, x2 = rng.normal(size=n), rng.binomial(1, 0.4, n).astype(float)

def row(label, values, fmt="9.5f"):
    return f"  {label:14s} " + " ".join(f"{v:{fmt}}" for v in values)


def r(what, df):
    path = HERE / f"{what}.parquet"
    df.write_parquet(path)
    subprocess.run([RS, str(R_SCRIPT), str(path), str(HERE), what], check=True)
    out = {}
    for f in HERE.glob(f"{what}_*.csv"):
        t = pl.read_csv(f)
        out[f.stem.split("_", 1)[1]] = dict(zip(t["name"], t["value"]))
        f.unlink()
    return out

print(f"G={G} groups, mean size {sizes.mean():.0f}, n={n:,}")
# ---- binary
a = rng.normal(0, 0.7, G)
eta = -0.4 + 0.8 * x1 - 0.5 * x2 + a[g]
y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
df = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})
t = time.perf_counter()
f = mlogit_stream("y ~ x1 + x2", df, random="g", verbose=False)
tl = time.perf_counter() - t
R = r("binary", df)
print("\nbinary (truth: -0.4, 0.8, -0.5; sd 0.7)")
ours = np.r_[f.beta, np.sqrt(f.random_effects["g"]["Sigma"][0][0])]
print(row("logit-stream", ours) + f"   ll {f.loglik:.3f}  {tl:.1f} s")
for k in ("nAGQ0", "nAGQ1", "nAGQ25"):
    v = R[k]
    rv = np.array([v["(Intercept)"], v["x1"], v["x2"], v["sd.(Intercept)"]])
    se = np.array([v["se.(Intercept)"], v["se.x1"], v["se.x2"]])
    print(row("glmer " + k, rv) + f"   ll {v['loglik']:.3f}  {v['seconds']:.1f} s"
          f"   (ours - glmer)/se: {np.round((ours[:3] - rv[:3]) / se, 3)}")
# ---- ordered
a = rng.normal(0, 0.6, G)
lat = 0.7 * x1 - 0.4 * x2 + a[g] + rng.logistic(size=n)
yo = np.searchsorted([-1.0, 0.2, 1.4], lat)
df = pl.DataFrame({"y": yo, "x1": x1, "x2": x2, "g": g})
t = time.perf_counter()
fo = ologit_stream("y ~ x1 + x2", df, random="g", verbose=False)
tl = time.perf_counter() - t
R = r("ordered", df)
print("\nordered (truth: 0.7, -0.4; cuts -1, 0.2, 1.4; sd 0.6)")
ours = np.r_[fo.beta, np.sqrt(fo.random_effects["g"]["Sigma"][0][0])]
print(row("logit-stream", ours) + f"   ll {fo.loglik:.3f}  {tl:.1f} s")
for k in ("nAGQ1", "nAGQ10"):
    v = R[k]
    rv = np.array([v["x1"], v["x2"], v["0|1"], v["1|2"], v["2|3"], v["sd"]])
    se = np.array([v["se.x1"], v["se.x2"]])
    print(row("clmm " + k, rv) + f"   ll {v['loglik']:.3f}  {v['seconds']:.1f} s"
          f"   (ours - clmm)/se slopes: {np.round((ours[:2] - rv[:2]) / se, 3)}")
# ---- multinomial, 3 levels
S = np.array([[0.25, 0.10], [0.10, 0.36]])
A = rng.multivariate_normal(np.zeros(2), S, G)
E = np.column_stack([np.zeros(n), (-0.3 + 0.8 * x1 - 0.5 * x2) + A[g, 0],
                     (0.4 - 0.6 * x1 + 0.3 * x2) + A[g, 1]])
P = np.exp(E - E.max(1, keepdims=True))
P /= P.sum(1, keepdims=True)
ym = (P.cumsum(1) > rng.random(n)[:, None]).argmax(1)
df = pl.DataFrame({"y": ym, "x1": x1, "x2": x2, "g": g})
t = time.perf_counter()
fm = mlogit_stream("y ~ x1 + x2", df, random="g", verbose=False)
tl = time.perf_counter() - t
R = r("multinomial", df)
v = R["PQL"]
print("\nmultinomial, 3 levels (truth: -0.3 0.8 -0.5 | 0.4 -0.6 0.3; Sigma 0.25 0.10 0.10 0.36)")
print(row("logit-stream", fm.beta, "8.4f") + "   Sigma "
      + " ".join(f"{x:.4f}" for x in np.ravel(fm.random_effects["g"]["Sigma"])) + f"   {tl:.1f} s")
# mblogit's coefficients alternate between the equations; put them in ours
rb = [v[f"{j}~{c}"] for j in (1, 2) for c in ("(Intercept)", "x1", "x2")]
rs = [v[c] for c in v if c.startswith("Sigma")]
print(row("mblogit PQL", rb, "8.4f") + "   Sigma "
      + " ".join(f"{x:.4f}" for x in rs) + f"   {v['seconds']:.1f} s")
