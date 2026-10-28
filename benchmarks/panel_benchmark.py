"""Time and peak memory of the logit for short panels, as the panel grows from
100,000 to 10 million rows (20,000 to 2 million units of five periods):

    conditional     Chamberlain's conditional logit: logit-stream's
                    cfelogit_stream, statsmodels' ConditionalLogit and R's
                    survival::clogit (method "exact", its default, which is
                    the conditional likelihood)
    fixed effects   the logit with the unit effects estimated: logit-stream's
                    logit_stream (hdfe-stream's feglm_stream), R's fixest
                    feglm and pyfixest's feglm
    bias-corrected  the same with the analytic incidental-parameter bias
                    correction: logit-stream's bias_correction=True and R's
                    alpaca (feglm, then biasCorr), which it reproduces

    python benchmarks/panel_benchmark.py                 # the default sizes
    python benchmarks/panel_benchmark.py 100000 1000000  # rows

The data are logit_stream.simulate.simulate_panel: P(y = 1) = logistic(a_unit
+ x1 - 0.5 x2), the unit effects correlated with x1, written once per size.
Every library is given the same model with its default standard errors
(fixest asked for iid ones, its default with fixed effects being clustered).
max_abs_diff compares each fit's coefficients with logit-stream's of the same
kind, and max_abs_err with the truth (1, -0.5): with five periods the
fixed-effects estimates are biased (about 30% here), and the other two kinds
are not. Results go to benchmarks/results/panel.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from _harness import (
    DATA_DIR,
    RESULTS_DIR,
    external_command,
    external_missing,
    run_child,
    run_configs,
    upsert_csv,
    write_machine_info,
)

SCRIPT = Path(__file__).resolve()
PERIODS = 5
TRUTH = {"x1": 1.0, "x2": -0.5}
CONFIGS = {
    "conditional": {
        "logit-stream": "ls-cfe",
        "statsmodels": "sm-cfe",
        "clogit (R survival)": "r-clogit",
    },
    "fixed effects": {
        "logit-stream": "ls-fe",
        "fixest (R)": "r-feglm",
        "pyfixest": "pyfixest",
    },
    "bias-corrected": {
        "logit-stream": "ls-bc",
        "alpaca (R)": "r-alpaca",
    },
}
DEFAULT_SIZES = [100_000, 1_000_000, 3_000_000, 10_000_000]
TIMEOUT_S = 1800


def data_path(n):
    """simulate_panel with n rows (n / 5 units of five periods)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"panel_{n}.parquet"
    if not path.exists():
        from logit_stream.simulate import simulate_panel
        simulate_panel(n_units=n // PERIODS, periods=PERIODS, seed=20261028).write_parquet(
            path, row_group_size=500_000)
    return path


# --------------------------------------------------------------------------
# the child
# --------------------------------------------------------------------------

def run_logit_stream(lib, path, workdir):
    import logit_stream as ls
    t0 = time.perf_counter()
    if lib == "ls-cfe":
        fit = ls.cfelogit_stream("y ~ x1 + x2 | unit", str(path), workdir=str(workdir),
                                 verbose=False)
        steps, conv = fit.diagnostics["newton"]["iterations"], fit.converged
        disk = fit.diagnostics.get("disk_peak_gb", 0) * 1000
    else:
        fit = ls.logit_stream("y ~ x1 + x2 | unit", str(path), workdir=str(workdir),
                              bias_correction=(lib == "ls-bc"), save_resid=False,
                              verbose=False)
        steps, conv = fit.diagnostics["irls"]["iterations"], fit.diagnostics["irls"]["converged"]
        disk = fit.diagnostics.get("disk_peak_gb", 0) * 1000
    t_fit = time.perf_counter() - t0
    out = {"t_read": 0.0, "t_fit": t_fit, "n_obs": int(fit.n_obs), "iterations": int(steps),
           "converged": bool(conv), "reported_disk_mb": disk,
           "coefs": {c: float(fit.coef()[c]) for c in TRUTH}}
    if hasattr(fit, "cleanup"):
        fit.cleanup()
    return out


def run_statsmodels(path):
    import polars as pl
    from statsmodels.discrete.conditional_models import ConditionalLogit
    t0 = time.perf_counter()
    df = pl.read_parquet(path)
    X, y, g = df.select("x1", "x2").to_numpy(), df["y"].to_numpy(), df["unit"].to_numpy()
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    r = ConditionalLogit(y, X, groups=g).fit(disp=0)
    np.asarray(r.bse)
    t_fit = time.perf_counter() - t0
    rv = getattr(r, "mle_retvals", {}) or {}
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(y)),
            "iterations": int(rv.get("iterations", 0) or 0),
            "converged": bool(rv.get("converged", True)),
            "coefs": dict(zip(TRUTH, map(float, r.params)))}


def run_pyfixest(path):
    import polars as pl
    import pyfixest as pf
    t0 = time.perf_counter()
    df = pl.read_parquet(path).to_pandas()
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    fit = pf.feglm("y ~ x1 + x2 | unit", df, family="logit")
    np.asarray(fit.se())
    t_fit = time.perf_counter() - t0
    coef = fit.coef().to_dict()
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(fit._N), "iterations": 0,
            "converged": True, "coefs": {c: float(coef[c]) for c in TRUTH}}


def runner(config, n, path, workdir):
    lib = config.split(":", 1)[1]
    if lib.startswith("ls-"):
        return run_logit_stream(lib, path, workdir)
    return {"sm-cfe": run_statsmodels, "pyfixest": run_pyfixest}[lib](path)


def commands(path):
    fits = {"r-clogit": ("clogit", "y ~ x1 + x2 + strata(unit)"),
            "r-feglm": ("feglm", "y ~ x1 + x2 | unit"),
            "r-alpaca": ("alpaca", "y ~ x1 + x2 | unit")}

    def command(label, kind):
        lib = kind.split(":", 1)[1]
        if lib not in fits:
            return None
        fit, fml = fits[lib]
        return (external_command("R", fit=fit, path=path, fml=fml,
                                 columns=["unit", "y", "x1", "x2"])
                or external_missing("R"))
    return command


# --------------------------------------------------------------------------
# the parent
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int)
    ap.add_argument("--model", choices=list(CONFIGS), action="append")
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args(argv)
    out = RESULTS_DIR / "panel.csv"
    write_machine_info(RESULTS_DIR / "machine.json")
    for n in args.sizes or DEFAULT_SIZES:
        path = data_path(n)
        for model in args.model or list(CONFIGS):
            configs = {label: f"{model}:{lib}" for label, lib in CONFIGS[model].items()}
            run_configs(SCRIPT, out, model, n, path, configs, command=commands(path),
                        timeout=args.timeout, only=args.only,
                        extra=lambda _label: {"periods": PERIODS})
            add_errors(out, model, n)


def add_errors(out, model, n):
    """max_abs_err: each fit's largest error against the truth."""
    import csv
    import json
    with open(out, newline="") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["model"] == model and int(r["n"]) == n and r.get("coefs")]
    for r in rows:
        coefs = json.loads(r["coefs"])
        err = max(abs(coefs[c] - v) for c, v in TRUTH.items() if c in coefs)
        if str(r.get("max_abs_err")) != str(err):
            r["max_abs_err"] = err
            upsert_csv(out, [r])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, config, n, path, workdir = sys.argv
        run_child(runner, config, n, path, workdir)
    else:
        main()
