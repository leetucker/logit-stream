"""One fit with Biogeme, for benchmarks/_harness.py, in Biogeme's own
environment (benchmarks/external/.biogeme: Biogeme needs pandas < 3, which the
benchmarks' environment does not have). Reads the long-format Swissmetro file
of choice_benchmark.py or mixed_benchmark.py, makes Biogeme's wide database
from it (timed as "read"), fits, and prints one line,

    RESULT {"t_read": ..., "t_fit": ..., "peak_memory_mb": ..., "coefs": {...}, ...}

    python biogeme_fit.py '<spec as JSON>'

spec["fit"]: "logit" (Biogeme's first Swissmetro example, the benchmark's
conditional logit), "nested" (train and car in a nest; its parameter mu is
reported as lambda = 1 / mu, logit-stream's normalization) or "mixed" (a
normal coefficient of travel time, panel by respondent, spec["draws"] Halton
draws). Biogeme's defaults otherwise: its own optimizer and its standard
errors (the inverse of the analytic Hessian; for the mixed logit, of the
simulated one).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

ALTS = {1: "TRAIN", 2: "SM", 3: "CAR"}


def hwm_mb():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    return 0.0


def wide(path, panel):
    """One row per choice set: CHOICE (1-3), and per alternative its
    availability, time and cost (the costs already zero for GA holders)."""
    import polars as pl
    long = pl.read_parquet(path)
    keys = ["case", *(["ID"] if panel else [])]
    w = long.pivot(on="alt_id", index=keys, values=["avail", "TT", "CO"],
                   aggregate_function="first")
    choice = (long.filter(pl.col("chosen") == 1).select("case", pl.col("alt_id").alias("CHOICE")))
    w = w.join(choice, on="case").sort(keys[::-1])
    rename = {f"{v}_{a}": f"{v}{a}" for v in ("avail", "TT", "CO") for a in ALTS}
    w = w.rename({k: v for k, v in rename.items() if k in w.columns})
    for a in ALTS:
        for v in ("avail", "TT", "CO"):
            if f"{v}{a}" not in w.columns:
                w = w.with_columns(pl.lit(0.0).alias(f"{v}{a}"))
    return w.fill_null(0).to_pandas()


def main():
    spec = json.loads(sys.argv[1])
    baseline = hwm_mb()
    import biogeme.biogeme as bio
    import biogeme.database as db
    from biogeme import models
    from biogeme.expressions import Beta, Variable

    t0 = time.perf_counter()
    panel = spec["fit"] == "mixed"
    df = wide(spec["path"], panel)
    t_read = time.perf_counter() - t0

    t0 = time.perf_counter()
    database = db.Database("swissmetro", df)
    asc_train = Beta("ASC_TRAIN", 0, None, None, 0)
    asc_car = Beta("ASC_CAR", 0, None, None, 0)
    b_cost = Beta("CO", 0, None, None, 0)
    b_time = Beta("TT", 0, None, None, 0)
    av = {a: Variable(f"avail{a}") for a in ALTS}
    choice = Variable("CHOICE")
    extra = {}
    if spec["fit"] == "mixed":
        from biogeme.expressions import MonteCarlo, PanelLikelihoodTrajectory, log
        try:                                    # Biogeme 3.3
            from biogeme.expressions import Draws
        except ImportError:                     # 3.2
            from biogeme.expressions import bioDraws as Draws
        s_time = Beta("sd.TT", 1, None, None, 0)
        b_time_rnd = b_time + s_time * Draws("b_time_rnd", "NORMAL_HALTON2")
        V = {a: (asc_train if a == 1 else asc_car if a == 3 else 0)
             + b_time_rnd * Variable(f"TT{a}") + b_cost * Variable(f"CO{a}") for a in ALTS}
        database.panel("ID")
        prob = models.logit(V, av, choice)
        formula = log(MonteCarlo(PanelLikelihoodTrajectory(prob)))
    else:
        V = {a: (asc_train if a == 1 else asc_car if a == 3 else 0)
             + b_time * Variable(f"TT{a}") + b_cost * Variable(f"CO{a}") for a in ALTS}
        if spec["fit"] == "nested":
            from biogeme.nests import NestsForNestedLogit, OneNestForNestedLogit
            mu = Beta("mu.existing", 1, 1, 10, 0)
            nests = NestsForNestedLogit(choice_set=list(ALTS), tuple_of_nests=(
                OneNestForNestedLogit(nest_param=mu, list_of_alternatives=[1, 3],
                                      name="existing"),
                OneNestForNestedLogit(nest_param=1.0, list_of_alternatives=[2],
                                      name="future")))
            formula = models.lognested(V, av, nests, choice)
        else:
            formula = models.loglogit(V, av, choice)
    kwargs = {"number_of_draws": int(spec["draws"])} if panel else {}
    work = tempfile.mkdtemp(prefix="biogeme_")
    os.chdir(work)
    try:
        model = bio.BIOGEME(database, formula, generate_html=False, generate_yaml=False,
                            save_iterations=False, **kwargs)
    except TypeError:                           # 3.2: the parameters are attributes
        model = bio.BIOGEME(database, formula, **kwargs)
        model.generate_html = model.generate_pickle = model.save_iterations = False
    model.model_name = "benchmark"
    results = model.estimate()
    try:
        betas = results.get_beta_values()
        table = results.get_estimated_parameters()
    except AttributeError:
        betas = results.getBetaValues()
        table = results.getEstimatedParameters()
    _ = table
    t_fit = time.perf_counter() - t0
    coefs = {k: float(v) for k, v in betas.items()}
    if "mu.existing" in coefs:
        coefs["lambda.existing"] = 1.0 / coefs.pop("mu.existing")
    for name in ("final_loglikelihood", "final_log_likelihood"):
        value = getattr(results, name, None)
        if value is not None:
            extra["loglik"] = float(value)
            break
    else:
        try:
            extra["loglik"] = float(results.data.logLike)
        except AttributeError:
            pass
    os.chdir(tempfile.gettempdir())
    shutil.rmtree(work, ignore_errors=True)
    print("RESULT " + json.dumps({
        "t_read": t_read, "t_fit": t_fit, "n_obs": int(df["ID"].nunique() if panel else len(df)),
        "iterations": 0, "converged": True, "peak_memory_mb": hwm_mb(),
        "baseline_memory_mb": baseline, "peak_disk_mb": 0.0, "coefs": coefs, **extra}))


if __name__ == "__main__":
    main()
