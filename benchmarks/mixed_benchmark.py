"""Time and peak memory of a mixed logit on the Swissmetro panel, against
xlogit, jaxlogit, Biogeme and R's logitr, as the number of respondents
grows; and the accuracy of each against the exact maximum-likelihood
estimates.

    python benchmarks/mixed_benchmark.py check            # accuracy, the survey itself
    python benchmarks/mixed_benchmark.py                  # scaling, the default sizes
    python benchmarks/mixed_benchmark.py 752 10000        # pick your own (respondents)

The model is the conditional logit of choice_benchmark.py (Biogeme's first
Swissmetro example) with a normally distributed coefficient of travel time,
and the panel structure of the survey: each respondent's nine choices share
one draw of the coefficient (xlogit's Swissmetro mixed-logit example):

    U_alt = ASC_alt + b_TT,n TT_alt + B_COST CO_alt,   b_TT,n ~ N(mu, s^2)

With a single random coefficient the likelihood of a respondent is a
one-dimensional integral, which quadrature computes exactly; `check`
maximizes that exact likelihood (scipy) and compares each library's
simulated estimates with it, at several numbers of draws. (Gauss-Hermite
quadrature does not converge here: with a standard deviation of travel
time's coefficient near 3.6 and trips of several hundred minutes, a
respondent's likelihood is nearly a step function of the draw. Gauss-
Legendre on [-9, 9] with 2,000 nodes is converged to 1e-6.)

xlogit runs with its L-BFGS-B optimizer: its default BFGS stops after two
iterations on this model ("local search could not find a higher log
likelihood value", log-likelihood -5,058 against -4,360). jaxlogit
estimates a standard deviation through a softplus (its default
force_positive_chol_diag=True) and reports the parameter before it; the
benchmark reports softplus(parameter), the standard deviation, with the
delta-method standard error. logit-stream, xlogit and jaxlogit use the same
Halton draws (consecutive points of the base-2 sequence per respondent, the
first 100 discarded), so at the same number of draws their simulated
log-likelihoods agree; Biogeme (NORMAL_HALTON2, its base-2 Halton draws,
benchmarks/external/biogeme_fit.py) and logitr (drawType "halton") draw
their own, so theirs differ by the simulation error. Larger samples are
drawn from the survey's 752 respondents with replacement, whole respondents
at a time. Results go to
benchmarks/results/mixed.csv and mixed_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from _harness import (
    DATA_DIR,
    RESULTS_DIR,
    external_command,
    external_missing,
    measure,
    run_child,
    run_configs,
    warm_up,
    write_machine_info,
)
from choice_benchmark import long_from_wide, wide

SCRIPT = Path(__file__).resolve()
VARS = ["ASC_TRAIN", "ASC_CAR", "TT", "CO"]
FML = "chosen ~ ASC_TRAIN + ASC_CAR + TT + CO - 1"
RANDOM = {"TT": "normal"}
NAMES = VARS + ["sd.TT"]
CONFIGS = {
    "logit-stream": "ls",
    "xlogit (L-BFGS-B)": "xlogit",
    "jaxlogit": "jax",
    "Biogeme": "biogeme",
    "logitr (R)": "r-logitr",
}
DEFAULT_SIZES = [752, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
DRAWS = 500
TIMEOUT_S = 1800


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def data_path(n):
    """Swissmetro with n respondents (the survey's own 752 for n = 752,
    otherwise drawn from them with replacement), in long format with all
    three alternatives, `avail`, and the respondent `ID`."""
    import polars as pl
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"swissmetro_panel_{n}.parquet"
    if path.exists():
        return path
    w = wide()
    ids = w["ID"].unique().sort()
    if n != len(ids):
        pick = ids.gather(np.random.default_rng(20261028).integers(0, len(ids), n))
        w = (pl.DataFrame({"ID": pick, "panel": np.arange(n)})
               .join(w, on="ID", how="left").drop("ID").rename({"panel": "ID"}))
    w = w.sort("ID", maintain_order=True).with_columns(pl.int_range(pl.len()).alias("case"))
    long = long_from_wide(w)
    long = long.join(w.select("case", "ID"), on="case", how="left").sort("case", "alt_id")
    long.write_parquet(path, row_group_size=500_000)
    return path


def _arrays(path):
    import polars as pl
    df = pl.read_parquet(path)
    X = df.select(VARS).to_numpy()
    y, alts, ids, av, pan = (df[c].to_numpy() for c in ("chosen", "alt", "case", "avail", "ID"))
    return X, y, alts, ids, av, pan


# --------------------------------------------------------------------------
# the child
# --------------------------------------------------------------------------

def run_logit_stream(path, workdir, draws):
    import polars as pl

    import logit_stream as ls
    t0 = time.perf_counter()
    lf = pl.scan_parquet(path).filter(pl.col("avail") == 1)
    fit = ls.mixlogit_stream(FML, lf, case="case", panel="ID", random=RANDOM, draws=draws,
                             workdir=str(workdir), verbose=False)
    t_fit = time.perf_counter() - t0
    opt = fit.diagnostics["optimizer"]
    return {"t_read": 0.0, "t_fit": t_fit, "n_obs": int(fit.n_panels),
            "iterations": int(opt["iterations"]), "converged": bool(fit.converged),
            "loglik": float(fit.loglik), "reported_disk_mb": fit.diagnostics["disk_peak_gb"] * 1e3,
            "coefs": dict(zip(fit.coefnames, map(float, fit.beta))),
            "se": dict(zip(fit.coefnames, map(float, fit.se)))}


def run_xlogit(path, draws):
    from xlogit import MixedLogit
    t0 = time.perf_counter()
    X, y, alts, ids, av, pan = _arrays(path)
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    m = MixedLogit()
    m.fit(X, y, varnames=VARS, alts=alts, ids=ids, randvars={"TT": "n"}, avail=av, panels=pan,
          n_draws=draws, optim_method="L-BFGS-B", verbose=0)
    t_fit = time.perf_counter() - t0
    coef = dict(zip(map(str, m.coeff_names), map(float, m.coeff_)))
    se = dict(zip(map(str, m.coeff_names), map(float, m.stderr)))
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(np.unique(pan))),
            "iterations": int(getattr(m, "total_iter", 0) or 0),
            "converged": bool(getattr(m, "convergence", True)), "loglik": float(m.loglikelihood),
            "coefs": coef, "se": se}


def run_jaxlogit(path, draws):
    import jax
    jax.config.update("jax_enable_x64", True)
    from jaxlogit.mixed_logit import ConfigData, MixedLogit
    t0 = time.perf_counter()
    X, y, alts, ids, av, pan = _arrays(path)
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    m = MixedLogit()
    m.fit(X=X, y=y, varnames=VARS, alts=alts, ids=ids, randvars={"TT": "n"},
          config=ConfigData(avail=av, panels=pan, n_draws=draws), verbose=0)
    t_fit = time.perf_counter() - t0
    coef = dict(zip(map(str, m.coeff_names), map(float, m.coeff_)))
    se = dict(zip(map(str, m.coeff_names), map(float, m.stderr)))
    # the standard deviation is softplus(parameter) (force_positive_chol_diag)
    raw = coef["sd.TT"]
    coef["sd.TT"] = float(np.logaddexp(0.0, raw))
    se["sd.TT"] = se["sd.TT"] / (1.0 + np.exp(-raw))
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(np.unique(pan))),
            "iterations": int(getattr(m, "total_iter", 0) or 0),
            "converged": bool(getattr(m, "convergence", True)), "loglik": float(m.loglikelihood),
            "coefs": coef, "se": se}


def commands(path, draws):
    """R's and Biogeme's commands, or a reason to skip; None for the script's
    own child."""
    def command(label, kind):
        lib = kind.split(":")[0]
        if lib == "biogeme":
            return (external_command("biogeme", fit="mixed", path=path, draws=draws)
                    or external_missing("biogeme"))
        if lib == "r-logitr":
            return (external_command("R", fit="logitr", path=path, filter_avail=True,
                                     columns=["case", "alt", "chosen", "avail", "ID", *VARS],
                                     panel="ID", pars=VARS, random=RANDOM, draws=draws)
                    or external_missing("R"))
        return None
    return command


def runner(config, n, path, workdir):
    lib, draws = config.split(":")
    draws = int(draws)
    if lib == "ls":
        return run_logit_stream(path, workdir, draws)
    return {"xlogit": run_xlogit, "jax": run_jaxlogit}[lib](path, draws)


# --------------------------------------------------------------------------
# the exact likelihood (one random coefficient: Gauss-Hermite quadrature)
# --------------------------------------------------------------------------

def exact_fit(path, nodes=2000):
    """Maximum likelihood with the integral over the random coefficient by
    Gauss-Legendre quadrature on [-9, 9] (converged to 1e-6 in the
    log-likelihood by 2,000 nodes; see the module docstring), maximized by
    scipy's BFGS and polished with Nelder-Mead."""
    import polars as pl
    from scipy.optimize import minimize
    df = pl.read_parquet(path).filter(pl.col("avail") == 1).sort("ID", "case", "alt_id")
    X = df.select(VARS).to_numpy()
    y = df["chosen"].to_numpy().astype(bool)
    case = df["case"].to_numpy()
    pan = df["ID"].to_numpy()
    cs = np.r_[np.flatnonzero(np.diff(case, prepend=-1) != 0), len(df)]
    case_of_row = np.repeat(np.arange(len(cs) - 1), np.diff(cs))
    pan_of_case = pan[cs[:-1]]
    pstarts = np.r_[np.flatnonzero(np.diff(pan_of_case, prepend=-1) != 0), len(cs) - 1]
    xl, wl = np.polynomial.legendre.leggauss(nodes)
    z = 9.0 * xl
    w = 9.0 * wl * np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi)
    jt = VARS.index("TT")

    def negll(theta):
        b, s = theta[:4], theta[4]
        V = (X @ b)[:, None] + s * X[:, [jt]] * z[None, :]           # rows x nodes
        V -= np.maximum.reduceat(V, cs[:-1])[case_of_row]
        lse = np.log(np.add.reduceat(np.exp(V), cs[:-1]))
        lp = V[y] - lse                                               # cases x nodes
        lpan = np.add.reduceat(lp, pstarts[:-1])                      # panels x nodes
        m = lpan.max(axis=1, keepdims=True)
        return -float(np.sum(m[:, 0] + np.log(np.exp(lpan - m) @ w)))

    theta = np.array([-0.57, 0.28, -3.2, -1.65, 3.6])
    res = minimize(negll, theta, method="BFGS", options={"gtol": 1e-10, "maxiter": 2000})
    res = minimize(negll, res.x, method="Nelder-Mead",
                   options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    return dict(zip(NAMES, map(float, res.x))), -float(res.fun)


def check(draws_list=(100, 500, 2000)):
    path = data_path(752)
    exact, ll = exact_fit(path)
    print(f"exact (quadrature): log-likelihood {ll:.4f}")
    print("  " + "  ".join(f"{k} {v:.5f}" for k, v in exact.items()))
    out = {"exact": {"coefs": exact, "loglik": ll}, "fits": []}
    for draws in draws_list:
        command = commands(path, draws)
        for label, lib in CONFIGS.items():
            cmd = command(label, f"{lib}:{draws}")
            if isinstance(cmd, str):
                print(f"{label:10s} R={draws:5d}  {cmd}")
                continue
            if cmd is None and label.startswith("logit-stream"):
                warm_up(SCRIPT, f"{lib}:{draws}", lambda: (752, path),
                        RESULTS_DIR / "work" / lib)
            rep = measure(SCRIPT, f"{lib}:{draws}", 752, path, RESULTS_DIR / "work" / lib,
                          timeout=TIMEOUT_S, command=cmd)
            if rep.get("status") != "ok":
                print(f"{label:10s} R={draws:5d}  {rep.get('status')}")
                continue
            c = rep["coefs"]
            diff = {k: c[k] - exact[k] for k in NAMES if k in c}
            # the sign of the standard deviation is not identified
            if "sd.TT" in c:
                diff["sd.TT"] = abs(c["sd.TT"]) - abs(exact["sd.TT"])
            print(f"{label:10s} R={draws:5d}  log-lik {rep['loglik']:.4f}  max |diff| "
                  f"{max(abs(v) for v in diff.values()):.4f}  "
                  + "  ".join(f"{k} {v:+.4f}" for k, v in diff.items())
                  + f"   {rep['t_read'] + rep['t_fit']:.1f} s")
            out["fits"].append({"library": label, "draws": draws, "loglik": rep["loglik"],
                                "coefs": c, "se": rep.get("se"), "diff": diff,
                                "seconds": rep["t_read"] + rep["t_fit"],
                                "iterations": rep.get("iterations")})
    (RESULTS_DIR / "mixed_check.json").write_text(json.dumps(out, indent=1))


# --------------------------------------------------------------------------
# the parent
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args(argv)
    if args.sizes[:1] == ["check"]:
        check()
        return
    sizes = [int(s) for s in args.sizes] or DEFAULT_SIZES
    out = RESULTS_DIR / "mixed.csv"
    write_machine_info(RESULTS_DIR / "machine.json")
    for n in sizes:
        path = data_path(n)
        configs = {label: f"{lib}:{args.draws}" for label, lib in CONFIGS.items()}
        run_configs(SCRIPT, out, "mixed", n, path, configs, command=commands(path, args.draws),
                    timeout=args.timeout, only=args.only, absolute=("sd.TT",),
                    extra=lambda _label: {"draws": args.draws},
                    warmup=lambda _kind: (752, data_path(752)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, config, n, path, workdir = sys.argv
        run_child(runner, config, n, path, workdir)
    else:
        main()
