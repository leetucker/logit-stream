"""Time, peak memory and peak disk of binary, multinomial and ordered logits
on American Community Survey microdata, against scikit-learn, statsmodels,
mord and pyfixest (and, for the multinomial logit, the discrete-choice
libraries xlogit and jaxlogit), and R's glm, fixest, nnet's multinom,
MASS's polr and ordinal's clm, as the sample grows from 10,000 to 12.9
million adults.

    python benchmarks/prepare_acs.py                       # the data, once
    python benchmarks/acs_benchmark.py                     # every model, default sizes
    python benchmarks/acs_benchmark.py --model multinomial 10000 100000
    python benchmarks/plot_benchmarks.py                   # the figures

The data are the adults of the 2019-2023 ACS 5-year PUMS in a random order
(see prepare_acs.py), so the first n rows are a random sample of n. The
models are those of a demographic imputation, with the same covariates in
each -- age group, sex, their interaction, income group, citizenship and
nativity (one variable), English ability, marital status, disability,
Census division (41 columns with the intercept):

    binary       hispanic ~ RHS                    (13.5% Hispanic)
    multinomial  race5    ~ RHS, base White        (5 levels; 164 coefficients)
    ordered      educ     ~ RHS, without intercept (5 levels, adults 25+)

plus, for logit-stream only, the same models with the 2,462 PUMAs absorbed
as a fixed effect in place of the divisions (a richer model, drawn dashed),
and with random PUMA intercepts in their place (multinomial and ordered;
drawn dash-dotted), alone and within state fixed effects (tables only).

Every library is asked for an unpenalized maximum-likelihood fit and the
default (model-based) standard errors where it computes them, starting from
a Parquet file of the n rows: a library that needs the data in memory reads
the file and builds its design matrix (timed as "read"), logit-stream is
given the path. The R packages read the file with arrow (timed as "read")
and are given the same model as a formula with factor() terms (their
references are the lowest levels, as here); fixest is asked for iid
standard errors (its default with fixed effects is clustered), multinom for
its Hessian (what summary() computes) and more iterations than its default
100, which stop BFGS short of convergence with 164 coefficients.
R's glm, polr, clm and multinom use one thread (their own default); fixest
every logical CPU. Each configuration runs in its own process; see _harness.py.
A configuration that fails, runs out of memory or exceeds the timeout is not
run at larger sizes. The coefficients on a few covariates are compared with
logit-stream's at the same size (column max_abs_diff).

Results go to benchmarks/results/acs.csv, one row per (model, size,
configuration).
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
    write_machine_info,
)

SCRIPT = Path(__file__).resolve()
SOURCE = DATA_DIR / "acs_2023_5yr.parquet"
CATS = ["agecat", "inccat", "cit", "eng", "mar", "division"]
NUMS = ["female", "disabled"]
RHS = ("C(agecat) + female + C(agecat):female + C(inccat) + C(cit) + C(eng) + C(mar)"
       " + disabled + C(division)")
RHS_FE = ("C(agecat) + female + C(agecat):female + C(inccat) + C(cit) + C(eng) + C(mar)"
          " + disabled | puma_id")
RHS_RE = RHS_FE.split("|")[0].strip()       # with random="puma_id"
# the same, for R: factor() terms (whose names logit.R maps to C(x)[T.v])
RHS_R = RHS.replace("C(", "factor(")
RHS_R_FE = RHS_FE.replace("C(", "factor(")
OUTCOME = {"binary": "hispanic", "multinomial": "race5", "ordered": "educ"}
RACES = ["White", "Black", "Hispanic", "API", "Other"]
# the coefficients compared across libraries
CHECK = {"binary": ["female", "C(cit)[T.5]", "disabled"],
         "multinomial": ["Black:female", "Hispanic:C(cit)[T.5]", "API:C(cit)[T.4]",
                         "Other:disabled", "Black:C(inccat)[T.6]"],
         # not "female": in the 25+ sample, the formula's age reference
         # (the 18-24 level, absent) is dropped as collinear, and the other
         # libraries' reference is the lowest age group present, which
         # changes the main effect of sex (it is the reference group's)
         "ordered": ["C(cit)[T.5]", "disabled", "C(inccat)[T.6]"]}

# model -> label -> library key
CONFIGS = {
    "binary": {
        "logit-stream": "ls",
        "logit-stream (PUMA fixed effects)": "ls-fe",
        "scikit-learn (lbfgs)": "sk-lbfgs",
        "scikit-learn (newton-cholesky)": "sk-newton",
        "statsmodels": "sm",
        "pyfixest": "pyfixest",
        "glm (R)": "r-glm",
        "fixest (R)": "r-feglm",
        "fixest (R, PUMA fixed effects)": "r-feglm-fe",
    },
    "multinomial": {
        "logit-stream": "ls",
        "logit-stream (low memory)": "ls-lowmem",
        "logit-stream (PUMA fixed effects)": "ls-fe",
        "logit-stream (PUMA random intercepts)": "ls-re",
        "logit-stream (state fixed + PUMA random)": "ls-fere",
        "scikit-learn (lbfgs)": "sk-lbfgs",
        "scikit-learn (newton-cholesky)": "sk-newton",
        "statsmodels": "sm",
        "xlogit (OPG standard errors)": "xlogit",
        "jaxlogit (no random parameters)": "jax",
        "multinom (R)": "r-multinom",
    },
    "ordered": {
        "logit-stream": "ls",
        "logit-stream (PUMA fixed effects)": "ls-fe",
        "logit-stream (PUMA random intercepts)": "ls-re",
        "logit-stream (state fixed + PUMA random)": "ls-fere",
        "statsmodels": "sm",
        "mord (all-threshold loss)": "mord",
        "polr (R)": "r-polr",
        "clm (R)": "r-clm",
    },
}
# the R configurations: logit.R's fit and the formula's right-hand side
R_FITS = {"r-glm": ("glm", RHS_R), "r-feglm": ("feglm", RHS_R),
          "r-feglm-fe": ("feglm", RHS_R_FE), "r-multinom": ("multinom", RHS_R),
          "r-polr": ("polr", RHS_R), "r-clm": ("clm", RHS_R)}
# configurations of a different model (run after the others), and the fit
# each is compared with (None: none)
OTHER_MODEL = {"ls-fe", "ls-re", "ls-fere", "r-feglm-fe"}
REFERENCE = {"*": "logit-stream", "logit-stream (PUMA fixed effects)": None,
             "logit-stream (PUMA random intercepts)": None,
             "logit-stream (state fixed + PUMA random)": None,
             "fixest (R, PUMA fixed effects)": "logit-stream (PUMA fixed effects)"}
DEFAULT_SIZES = [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 6_000_000, 12_853_377]
TIMEOUT_S = 1800
LOW_MEMORY = {"block_mb": 8, "chunk_mb": 8}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def warmup_data(kind):
    """The data a logit-stream configuration warms up on: 10,000 people, or
    100,000 for the random intercepts (at 10,000 a PUMA has about four)."""
    n = 100_000 if kind.endswith(("-re", "-fere")) else 10_000
    return n, sample_path(n)


def sample_path(n):
    import polars as pl
    if not SOURCE.exists():
        # built from the Census download, kept in CACHE_DIR (prepare_acs.py)
        import prepare_acs
        prepare_acs.main([])
    total = pl.scan_parquet(SOURCE).select(pl.len()).collect().item()
    if n >= total:
        return SOURCE
    path = DATA_DIR / f"acs_{n}.parquet"
    if not path.exists():
        pl.scan_parquet(SOURCE).head(n).sink_parquet(path, row_group_size=250_000)
    return path


def dense_design(df, intercept=True):
    """The design of RHS as a dense matrix, as formulaic codes it (the
    lowest level of each categorical is the reference)."""
    import polars as pl
    cols, names = [], []
    if intercept:
        cols.append(np.ones(df.height))
        names.append("Intercept")
    lv_age = sorted(df["agecat"].unique().to_list())
    for c in CATS:
        lv = sorted(df[c].unique().to_list())
        for v in lv[1:]:
            cols.append((df[c] == v).cast(pl.Float64).to_numpy())
            names.append(f"C({c})[T.{v}]")
        if c == "agecat":
            cols.append(df["female"].cast(pl.Float64).to_numpy())
            names.append("female")
            for v in lv_age[1:]:
                cols.append(((df["agecat"] == v) & (df["female"] == 1)).cast(pl.Float64)
                            .to_numpy())
                names.append(f"C(agecat)[T.{v}]:female")
    for c in NUMS[1:]:
        cols.append(df[c].cast(pl.Float64).to_numpy())
        names.append(c)
    return np.ascontiguousarray(np.column_stack(cols)), names


def load(model, path):
    import polars as pl
    y = OUTCOME[model]
    df = pl.read_parquet(path, columns=[y, *CATS, *NUMS]).drop_nulls()
    X, names = dense_design(df, intercept=(model != "ordered"))
    if model == "multinomial":
        yv = df[y].replace_strict(RACES, list(range(len(RACES)))).to_numpy()
    else:
        yv = df[y].to_numpy().astype(np.int64)
    return X, yv, names


# --------------------------------------------------------------------------
# the child process
# --------------------------------------------------------------------------

def run_logit_stream(model, lib, path, workdir):
    import logit_stream as ls
    rhs = {"ls-fe": RHS_FE, "ls-re": RHS_RE, "ls-fere": RHS_RE + " | state"}.get(lib, RHS)
    fml = f"{OUTCOME[model]} ~ {rhs}"
    opts = dict(workdir=str(workdir), verbose=False)
    if lib in ("ls-re", "ls-fere"):
        opts["random"] = "puma_id"
    if lib == "ls-lowmem":
        opts.update(LOW_MEMORY)
    t0 = time.perf_counter()
    if model == "binary":
        fit = ls.logit_stream(fml, str(path), save_resid=False, **opts)
        steps = fit.diagnostics["irls"]["iterations"]
        disk = fit.diagnostics.get("disk_peak_gb", 0) * 1000
    elif model == "multinomial":
        fit = ls.mlogit_stream(fml, str(path), levels=RACES, **opts)
        steps = fit.diagnostics["newton"]["iterations"]
        disk = fit.diagnostics["disk_peak_gb"] * 1000
    else:
        fit = ls.ologit_stream(fml, str(path), **opts)
        steps = fit.diagnostics["newton"]["iterations"]
        disk = fit.diagnostics["disk_peak_gb"] * 1000
    t_fit = time.perf_counter() - t0
    coefs = fit.coef()
    conv = (fit.diagnostics["irls"]["converged"] if model == "binary" else fit.converged)
    out = {"t_read": 0.0, "t_fit": t_fit, "n_obs": int(fit.n_obs), "iterations": int(steps),
           "converged": bool(conv),
           "k": len(fit.coefnames), "reported_disk_mb": disk,
           "coefs": {c: float(coefs[c]) for c in CHECK[model] if c in coefs}}
    if hasattr(fit, "cleanup"):
        fit.cleanup()
    return out


def run_sklearn(model, solver, path):
    from sklearn.linear_model import LogisticRegression
    t0 = time.perf_counter()
    X, y, names = load(model, path)
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    m = LogisticRegression(C=np.inf, solver=solver, max_iter=10_000, tol=1e-8)
    m.fit(X[:, 1:], y)          # sklearn fits its own intercept
    t_fit = time.perf_counter() - t0
    if model == "binary":
        coef = dict(zip(names[1:], m.coef_[0]))
    else:
        W = m.coef_ - m.coef_[0]                     # relative to White (class 0)
        coef = {f"{RACES[j]}:{nm}": W[j, i] for j in range(1, len(RACES))
                for i, nm in enumerate(names[1:])}
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(y)),
            "iterations": int(np.max(m.n_iter_)), "converged": bool(np.max(m.n_iter_) < 10_000),
            "k": int(m.coef_.size + m.intercept_.size),
            "coefs": {c: float(coef[c]) for c in CHECK[model] if c in coef}}


def run_statsmodels(model, path):
    import statsmodels.api as sm
    t0 = time.perf_counter()
    X, y, names = load(model, path)
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    if model == "binary":
        r = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=100)
        coef = dict(zip(names, r.params))
        bse = r.bse
    elif model == "multinomial":
        r = sm.MNLogit(y, X).fit(disp=0, method="newton", maxiter=100)
        coef = {f"{RACES[j + 1]}:{nm}": r.params[i, j] for j in range(len(RACES) - 1)
                for i, nm in enumerate(names)}
        bse = r.bse
    else:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        r = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=0, maxiter=5000)
        coef = dict(zip(names, r.params[:X.shape[1]]))
        bse = r.bse
    np.asarray(bse)
    t_fit = time.perf_counter() - t0
    rv = getattr(r, "mle_retvals", {}) or {}
    its = rv.get("iterations", 0)
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(y)), "iterations": int(its or 0),
            "converged": bool(rv.get("converged", True)),
            "k": int(np.size(r.params)),
            "coefs": {c: float(coef[c]) for c in CHECK[model] if c in coef}}


def run_pyfixest(path):
    """pyfixest's feglm, the binary logit, from its formula (the same as
    logit-stream's), with its default (iid) standard errors."""
    import polars as pl
    import pyfixest as pf
    t0 = time.perf_counter()
    df = (pl.read_parquet(path, columns=[OUTCOME["binary"], *CATS, *NUMS]).drop_nulls()
          .to_pandas())
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    fit = pf.feglm(f"{OUTCOME['binary']} ~ {RHS}", df, family="logit")
    np.asarray(fit.se())
    t_fit = time.perf_counter() - t0
    coef = fit.coef().to_dict()
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(fit._N),
            "iterations": int(getattr(fit, "_iter", 0) or 0), "converged": True,
            "k": len(coef), "coefs": {c: float(coef[c]) for c in CHECK["binary"] if c in coef}}


def run_mord(path):
    import mord
    t0 = time.perf_counter()
    X, y, names = load("ordered", path)
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    m = mord.LogisticAT(alpha=0.0, max_iter=10_000)
    m.fit(X, y)
    t_fit = time.perf_counter() - t0
    coef = dict(zip(names, m.coef_))
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(y)), "iterations": 0,
            "converged": True, "k": int(m.coef_.size + m.theta_.size),
            "coefs": {c: float(coef[c]) for c in CHECK["ordered"] if c in coef}}


def long_format(y):
    """The long-format bookkeeping of the discrete-choice libraries: every
    person once per race, with alternatives 0..J-1 (White, the base, first),
    the person's id, and whether that race is the person's."""
    J, N = len(RACES), len(y)
    alts = np.tile(np.arange(J), N)
    return alts, np.repeat(np.arange(N), J), (alts == np.repeat(y, J)).astype(np.int8)


def _choice_coefs(names, values):
    """'<covariate>.<alternative index>' (xlogit's naming) to logit-stream's
    '<level>:<covariate>'."""
    out = {}
    for nm, v in zip(names, values):
        var, j = str(nm).rsplit(".", 1)
        var = "Intercept" if var == "_intercept" else var
        out[f"{RACES[int(j)]}:{var}"] = float(v)
    return out


def run_xlogit(path):
    """xlogit's MultinomialLogit with the covariates as individual-specific
    variables (isvars): it expands them, per alternative, itself. BFGS (its
    default), with standard errors from the outer product of the gradients
    (num_hess=False). Its default numerical Hessian differentiates the
    gradient one coefficient at a time with a full finite-difference pass
    each, K (K + 1) evaluations of the log-likelihood and gradient -- 27,060
    for these 164 coefficients, 239 s at 10,000 rows on the benchmark
    machine -- so it would only time the finite differences."""
    from xlogit import MultinomialLogit
    t0 = time.perf_counter()
    X, y, names = load("multinomial", path)
    alts, ids, chosen = long_format(y)
    Xl = np.repeat(X[:, 1:], len(RACES), axis=0)       # xlogit adds the intercept
    del X
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    m = MultinomialLogit()
    m.fit(Xl, chosen, varnames=names[1:], alts=alts, ids=ids, isvars=names[1:],
          fit_intercept=True, num_hess=False, verbose=0)
    t_fit = time.perf_counter() - t0
    coef = _choice_coefs(m.coeff_names, m.coeff_)
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(y)),
            "iterations": int(getattr(m, "total_iter", 0) or 0),
            "converged": bool(getattr(m, "convergence", True)), "k": int(len(m.coeff_)),
            "coefs": {c: coef[c] for c in CHECK["multinomial"] if c in coef}}


def run_jaxlogit(path):
    """jaxlogit's MixedLogit with no random parameters, which is the
    multinomial logit. It has no individual-specific variables, so the
    design is expanded here, as xlogit does internally: in the row of
    alternative l, the covariates fill the columns of equation l and the
    others are zero (timed with the reading). Its defaults otherwise:
    L-BFGS, then the Hessian by automatic differentiation."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from jaxlogit.mixed_logit import ConfigData, MixedLogit
    t0 = time.perf_counter()
    X, y, names = load("multinomial", path)
    J, (N, K) = len(RACES), X.shape
    alts, ids, chosen = long_format(y)
    Xe = np.zeros((N, J, J - 1, K))
    for l in range(1, J):
        Xe[:, l, l - 1, :] = X
    Xe = Xe.reshape(N * J, (J - 1) * K)
    del X
    varnames = [f"{RACES[l]}:{nm}" for l in range(1, J) for nm in names]
    t_read = time.perf_counter() - t0
    t0 = time.perf_counter()
    m = MixedLogit()
    m.fit(X=Xe, y=chosen, varnames=varnames, alts=alts, ids=ids, randvars={},
          config=ConfigData(n_draws=1), verbose=0)
    t_fit = time.perf_counter() - t0
    coef = dict(zip(map(str, m.coeff_names), map(float, m.coeff_)))
    return {"t_read": t_read, "t_fit": t_fit, "n_obs": int(len(y)),
            "iterations": int(getattr(m, "total_iter", 0) or 0),
            "converged": bool(getattr(m, "convergence", True)), "k": int(len(m.coeff_)),
            "coefs": {c: coef[c] for c in CHECK["multinomial"] if c in coef}}


def runner(config, n, path, workdir):
    model, lib = config.split(":", 1)
    if lib.startswith("ls"):
        return run_logit_stream(model, lib, path, workdir)
    if lib == "sk-lbfgs":
        return run_sklearn(model, "lbfgs", path)
    if lib == "sk-newton":
        return run_sklearn(model, "newton-cholesky", path)
    if lib == "sm":
        return run_statsmodels(model, path)
    if lib == "mord":
        return run_mord(path)
    if lib == "pyfixest":
        return run_pyfixest(path)
    if lib == "xlogit":
        return run_xlogit(path)
    if lib == "jax":
        return run_jaxlogit(path)
    raise ValueError(config)


# --------------------------------------------------------------------------
# the parent
# --------------------------------------------------------------------------

def r_command(model, path):
    def command(label, kind):
        lib = kind.split(":", 1)[1]
        if lib not in R_FITS:
            return None
        fit, rhs = R_FITS[lib]
        columns = [OUTCOME[model], *CATS, *NUMS] + (["puma_id"] if "puma_id" in rhs else [])
        cmd = external_command("R", fit=fit, path=path, columns=columns,
                               fml=f"{OUTCOME[model]} ~ {rhs}", outcome=OUTCOME[model],
                               levels=RACES)
        return cmd or external_missing("R")
    return command


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("sizes", nargs="*", type=int)
    ap.add_argument("--model", choices=list(CONFIGS), action="append")
    ap.add_argument("--only", nargs="+", help="configuration labels to run")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    args = ap.parse_args(argv)
    sizes = args.sizes or DEFAULT_SIZES
    models = args.model or list(CONFIGS)
    out = RESULTS_DIR / "acs.csv"
    write_machine_info(RESULTS_DIR / "machine.json")
    for n in sizes:
        path = sample_path(n)
        for model in models:
            configs = {label: f"{model}:{lib}" for label, lib in CONFIGS[model].items()
                       if lib not in OTHER_MODEL}
            configs.update({label: f"{model}:{lib}" for label, lib in CONFIGS[model].items()
                            if lib in OTHER_MODEL})
            run_configs(SCRIPT, out, model, n, path, configs, command=r_command(model, path),
                        timeout=args.timeout, only=args.only, reference=REFERENCE,
                        warmup=warmup_data)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, config, n, path, workdir = sys.argv
        run_child(runner, config, n, path, workdir)
    else:
        main()
