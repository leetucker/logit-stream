"""Time, peak memory and peak disk of the conditional and nested logit on the
Swissmetro mode-choice data, as the number of choice sets grows from the
survey's 6,768 to 10 million: the conditional logit against xlogit,
statsmodels, jaxlogit, Biogeme, and R's survival::clogit and mlogit; the
nested logit (train and car, the existing modes, in a nest: Biogeme's
nested-logit example) against Biogeme and mlogit.

    python benchmarks/choice_benchmark.py                # the default sizes
    python benchmarks/choice_benchmark.py 6768 100000    # pick your own
    python benchmarks/plot_benchmarks.py                 # the figures

Swissmetro (Bierlaire, Axhausen and Abay, 2001) is a stated-preference survey
of 1,192 respondents in Switzerland, each choosing among train, car and the
proposed Swissmetro in nine scenarios; it is the example data of Biogeme,
xlogit and jaxlogit. As there, the sample is the commuting and business
trips with a known choice (6,768 choice sets), the travel times and costs
are in hundreds of minutes and francs, and the cost of train and Swissmetro
is zero for holders of an annual pass (GA). The model is Biogeme's first
example, a conditional logit with alternative-specific constants for train
and car:

    U_alt = ASC_alt + B_TIME TT_alt + B_COST CO_alt

on the alternatives available to each respondent (car only to car owners).
Larger samples are drawn from the 6,768 choice sets with replacement, so the
estimates stay those of the survey (up to sampling) while the data grow.

Each library gets the data in the form it takes: logit-stream, statsmodels
and the R packages the long format with the available alternatives only;
xlogit and jaxlogit, which need the same alternatives in every set, all
three with an availability indicator; Biogeme its wide format, one row per
set (built from the long file, timed as reading; benchmarks/external/
biogeme_fit.py, in an environment of its own). survival::clogit uses its
default, the exact conditional likelihood (with one choice per set, the
conditional logit's). In the nested logit, the dissimilarity parameter is
lambda of the utility-consistent normalization (RU2) in every library:
Biogeme's mu is reported as 1 / mu. pylogit, in the first runs, ran out of
memory at 30,000 sets and is no longer run. Results go to
benchmarks/results/choice.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from _harness import (
    CACHE_DIR,
    DATA_DIR,
    RESULTS_DIR,
    external_command,
    external_missing,
    run_child,
    run_configs,
    write_machine_info,
)

SCRIPT = Path(__file__).resolve()
RAW = CACHE_DIR / "swissmetro.dat"
URL = "http://transp-or.epfl.ch/data/swissmetro.dat"
VARS = ["ASC_TRAIN", "ASC_CAR", "TT", "CO"]
FML = "chosen ~ ASC_TRAIN + ASC_CAR + TT + CO"
NESTS = {"existing": ["TRAIN", "CAR"], "future": ["SM"]}
CONFIGS = {
    "conditional": {
        "logit-stream": "ls",
        "xlogit": "xlogit",
        "statsmodels": "sm",
        "jaxlogit (no random parameters)": "jax",
        "Biogeme": "biogeme",
        "clogit (R survival)": "r-clogit",
        "mlogit (R)": "r-mlogit",
    },
    "nested": {
        "logit-stream": "ls-nested",
        "Biogeme": "biogeme-nested",
        "mlogit (R)": "r-mlogit-nested",
    },
}
DEFAULT_SIZES = [6_768, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]
TIMEOUT_S = 1200


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def wide():
    import polars as pl
    if not RAW.exists():
        import subprocess
        RAW.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", str(RAW), URL], check=True)
    df = pl.read_csv(RAW, separator="\t")
    return (df.filter(pl.col("PURPOSE").is_in([1, 3]) & (pl.col("CHOICE") != 0))
              .with_row_index("respondent_row"))


def long_from_wide(w):
    """All three alternatives of every set, with availability."""
    import polars as pl
    parts = []
    for code, alt, pre in ((1, "TRAIN", "TRAIN"), (2, "SM", "SM"), (3, "CAR", "CAR")):
        cost = pl.col(f"{pre}_CO")
        if alt in ("TRAIN", "SM"):
            cost = pl.when(pl.col("GA") == 1).then(0).otherwise(cost)
        parts.append(w.select(
            pl.col("case"), pl.lit(code).alias("alt_id"), pl.lit(alt).alias("alt"),
            (pl.col("CHOICE") == code).cast(pl.Int8).alias("chosen"),
            pl.col(f"{pre}_AV").cast(pl.Int8).alias("avail"),
            pl.lit(1.0 if alt == "TRAIN" else 0.0).alias("ASC_TRAIN"),
            pl.lit(1.0 if alt == "CAR" else 0.0).alias("ASC_CAR"),
            (pl.col(f"{pre}_TT") / 100).cast(pl.Float64).alias("TT"),
            (cost / 100).cast(pl.Float64).alias("CO")))
    return pl.concat(parts).sort("case", "alt_id")


def data_path(n):
    """Swissmetro with n choice sets (the survey's own for n = 6,768,
    otherwise drawn from them with replacement), in long format with all
    three alternatives and `avail`."""
    import polars as pl
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"swissmetro_{n}.parquet"
    if path.exists():
        return path
    w = wide()
    if n != w.height:
        idx = np.random.default_rng(20261027).integers(0, w.height, n)
        w = w[idx]
    w = w.with_columns(pl.int_range(pl.len()).alias("case"))
    long_from_wide(w).write_parquet(path, row_group_size=500_000)
    return path


# --------------------------------------------------------------------------
# the child
# --------------------------------------------------------------------------

def _read(path, available_only):
    import polars as pl
    df = pl.read_parquet(path)
    if available_only:
        df = df.filter(pl.col("avail") == 1)
    return df


def run_logit_stream(path, workdir):
    import polars as pl

    import logit_stream as ls
    t0 = time.perf_counter()
    lf = pl.scan_parquet(path).filter(pl.col("avail") == 1)
    fit = ls.clogit_stream(FML, lf, case="case", workdir=str(workdir), verbose=False)
    t_fit = time.perf_counter() - t0
    return {"t_read": 0.0, "t_fit": t_fit, "n_obs": int(fit.n_cases),
            "iterations": int(fit.diagnostics["newton"]["iterations"]),
            "converged": bool(fit.converged),
            "reported_disk_mb": fit.diagnostics["disk_peak_gb"] * 1000,
            "coefs": dict(zip(fit.coefnames, map(float, fit.beta)))}


def run_logit_stream_nested(path, workdir):
    import polars as pl

    import logit_stream as ls
    t0 = time.perf_counter()
    lf = pl.scan_parquet(path).filter(pl.col("avail") == 1)
    fit = ls.nlogit_stream(FML, lf, case="case", alt="alt", nests=NESTS, workdir=str(workdir),
                           verbose=False)
    t_fit = time.perf_counter() - t0
    return {"t_read": 0.0, "t_fit": t_fit, "n_obs": int(fit.n_obs),
            "iterations": int(fit.diagnostics["optimizer"]["iterations"]),
            "converged": bool(fit.converged), "loglik": float(fit.loglik),
            "reported_disk_mb": fit.diagnostics["disk_peak_gb"] * 1000,
            "coefs": dict(zip(fit.coefnames, map(float, fit.beta)))}


def run_xlogit(path):
    from xlogit import MultinomialLogit
    t0 = time.perf_counter()
    df = _read(path, available_only=False)
    X = df.select(VARS).to_numpy()
    y, alts, ids, av = (df[c].to_numpy() for c in ("chosen", "alt", "case", "avail"))
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    m = MultinomialLogit()
    m.fit(X, y, varnames=VARS, alts=alts, ids=ids, avail=av, verbose=0)
    t_fit = time.perf_counter() - t0
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(np.unique(ids))),
            "iterations": int(getattr(m, "total_iter", 0) or 0),
            "converged": bool(getattr(m, "convergence", True)),
            "coefs": dict(zip(m.coeff_names, map(float, m.coeff_)))}


def run_statsmodels(path):
    from statsmodels.discrete.conditional_models import ConditionalLogit
    t0 = time.perf_counter()
    df = _read(path, available_only=True)
    X = df.select(VARS).to_numpy()
    y, g = df["chosen"].to_numpy(), df["case"].to_numpy()
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    r = ConditionalLogit(y, X, groups=g).fit(disp=0)
    np.asarray(r.bse)
    t_fit = time.perf_counter() - t0
    rv = getattr(r, "mle_retvals", {}) or {}
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(np.unique(g))),
            "iterations": int(rv.get("iterations", 0) or 0),
            "converged": bool(rv.get("converged", True)),
            "coefs": dict(zip(VARS, map(float, r.params)))}


def run_jaxlogit(path):
    import jax
    jax.config.update("jax_enable_x64", True)
    from jaxlogit.mixed_logit import ConfigData, MixedLogit
    t0 = time.perf_counter()
    df = _read(path, available_only=False)
    X = df.select(VARS).to_numpy()
    y, alts, ids, av = (df[c].to_numpy() for c in ("chosen", "alt", "case", "avail"))
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    m = MixedLogit()
    m.fit(X=X, y=y, varnames=VARS, alts=alts, ids=ids, randvars={},
          config=ConfigData(avail=av, n_draws=1), verbose=0)
    t_fit = time.perf_counter() - t0
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(np.unique(ids))),
            "iterations": int(getattr(m, "total_iter", 0) or 0),
            "converged": bool(getattr(m, "convergence", True)),
            "coefs": dict(zip(m.coeff_names, map(float, m.coeff_)))}


def runner(config, n, path, workdir):
    lib = config.split(":", 1)[1]
    if lib == "ls":
        return run_logit_stream(path, workdir)
    if lib == "ls-nested":
        return run_logit_stream_nested(path, workdir)
    return {"xlogit": run_xlogit, "sm": run_statsmodels, "jax": run_jaxlogit}[lib](path)


COLUMNS = ["case", "alt", "chosen", "avail", *VARS]


def commands(path):
    def command(label, kind):
        lib = kind.split(":", 1)[1]
        if lib.startswith("biogeme"):
            cmd = external_command("biogeme", path=path,
                                   fit="nested" if lib.endswith("nested") else "logit")
            return cmd or external_missing("biogeme")
        if lib == "r-clogit":
            cmd = external_command("R", fit="clogit", path=path, columns=COLUMNS,
                                   filter_avail=True,
                                   fml=f"chosen ~ {' + '.join(VARS)} + strata(case)")
        elif lib.startswith("r-mlogit"):
            cmd = external_command("R", fit="mlogit", path=path, columns=COLUMNS,
                                   filter_avail=True, fml=f"chosen ~ {' + '.join(VARS)} | 0",
                                   nests=NESTS if lib.endswith("nested") else None)
        else:
            return None
        return cmd or external_missing("R")
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
    out = RESULTS_DIR / "choice.csv"
    write_machine_info(RESULTS_DIR / "machine.json")
    for n in args.sizes or DEFAULT_SIZES:
        path = data_path(n)
        for model in args.model or list(CONFIGS):
            configs = {label: f"{model}:{lib}" for label, lib in CONFIGS[model].items()}
            run_configs(SCRIPT, out, model, n, path, configs, command=commands(path),
                        timeout=args.timeout, only=args.only,
                        warmup=lambda _kind: (6_768, data_path(6_768)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, config, n, path, workdir = sys.argv
        run_child(runner, config, n, path, workdir)
    else:
        main()
