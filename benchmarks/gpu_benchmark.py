"""logit-stream on the CPU and on a GPU (``device="cuda"``), and xlogit and
jaxlogit on both, on one machine: wall time, and how far apart the
estimates are.

    python benchmarks/gpu_benchmark.py                         # every model, its sizes
    python benchmarks/gpu_benchmark.py --model mixed --sizes 10000

Needs an NVIDIA GPU, CuPy (``pip install cupy-cuda13x``, or cupy-cuda12x) and,
for jaxlogit on the GPU, JAX's CUDA plugin (``pip install "jax[cuda13]"``).
The CPU configurations run on the same machine, so the comparison is like for
like; the comparison across packages on a CPU machine is the CPU suite's
(acs_, choice_, mixed_ and panel_benchmark.py).

Models (``--model``, default all), at the sizes of SIZES:

    multinomial, ordered          ACS (acs_benchmark.py), no fixed effects
    multinomial-fe, ordered-fe    ACS, PUMA fixed effects
    multinomial-re, ordered-re    ACS, PUMA random intercepts (exact Laplace)
    conditional                   Swissmetro choice sets, resampled (choice_benchmark.py);
                                  also xlogit (CPU only: its conditional logit has
                                  no GPU code) and jaxlogit
    nested                        the same, train and car in a nest (Biogeme's example)
    mixed                         Swissmetro respondents' panels (mixed_benchmark.py),
                                  a normal coefficient on travel time, 500 Halton
                                  draws; also xlogit and jaxlogit
    conditional-fe                Chamberlain's logit on a simulated panel

Every fit is in a process of its own, which first fits the same model on a
small sample (so that numba's and CuPy's compilation, which they cache, is
not timed; JAX compiles again for the new shapes, which is then timed, as a
user would see it), then times the fit asked for. The variance is the robust
one ("hetero"; clustered by chooser for the mixed logit and by PUMA for the
random intercepts), whose scores pass runs on the device too. The time of
logit-stream's passes (``passes_s``) is reported apart from the total: what
is left is pass 0 and the scores pass (``after_s``). The estimates are
compared with logit-stream's on the CPU (``max_abs_diff``). Timeouts: an
hour for logit-stream, ten minutes for the others. Writes
benchmarks/results/gpu.csv and gpu_machine.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import acs_benchmark as A  # noqa: E402
import choice_benchmark as CH  # noqa: E402
import mixed_benchmark as MX  # noqa: E402
from _harness import DATA_DIR, RESULTS_DIR, run_child, run_configs, write_machine_info  # noqa: E402

SCRIPT = Path(__file__).resolve()
# the sizes of each model (read by benchmarks/ec2/bench_ec2.py, one step each)
SIZES = {
    "multinomial": [1_000_000, 12_853_377],
    "ordered": [1_000_000, 12_853_377],
    "multinomial-fe": [1_000_000, 12_853_377],
    "ordered-fe": [1_000_000, 12_853_377],
    "multinomial-re": [1_000_000, 3_000_000],
    "ordered-re": [1_000_000, 3_000_000],
    "conditional": [1_000_000, 10_000_000],
    "nested": [1_000_000, 10_000_000],
    "mixed": [10_000, 100_000, 1_000_000],
    "conditional-fe": [1_000_000, 10_000_000],
}
WARMUP = {"acs": 10_000, "sets": 6_768, "panels": 752, "cfe": 10_000}
# random intercepts warm up on more rows: at 10,000 the PUMAs have about four
# rows each, the slow case for the covariance's updates (a minute)
WARMUP_RE = 100_000
TIMEOUT_S = {"logit-stream": 3600, "other": 600}
LS = {"logit-stream (CPU)": "ls:cpu", "logit-stream (GPU)": "ls:cuda"}
OTHERS = {
    "conditional": {"xlogit (CPU)": "xlogit:cpu", "jaxlogit (CPU)": "jax:cpu",
                    "jaxlogit (GPU)": "jax:cuda"},
    "mixed": {"xlogit (CPU)": "xlogit:cpu", "xlogit (GPU)": "xlogit:cuda",
              "jaxlogit (CPU)": "jax:cpu", "jaxlogit (GPU)": "jax:cuda"},
}
# Biogeme's nests of the Swissmetro modes: the existing ones, and the new one
NESTS = CH.NESTS


def _acs(n):
    return A.sample_path(n)


def _sets(n):
    return CH.data_path(n)


def _panels(n):
    return MX.data_path(n)


def _cfe(n):
    """A simulated panel of about n rows: units of 2 to 9 periods, a unit
    effect correlated with a regressor, two more regressors."""
    import polars as pl
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"cfe_panel_{n}.parquet"
    if not path.exists():
        rng = np.random.default_rng(20261028)
        N = max(1, n // 5)
        T = rng.integers(2, 10, N)
        i = np.repeat(np.arange(N), T)
        a = rng.normal(0, 1.5, N)[i]
        m = len(i)
        x1, x2, x3 = rng.normal(size=m) + 0.5 * a, rng.normal(size=m), rng.normal(size=m)
        eta = 0.8 * x1 - 0.5 * x2 + 0.3 * x3 + a
        y = (rng.random(m) < 1 / (1 + np.exp(-eta))).astype(np.int8)
        pl.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3, "unit": i}).write_parquet(path)
    return path


DATA = {"multinomial": ("acs", _acs), "ordered": ("acs", _acs),
        "multinomial-fe": ("acs", _acs), "ordered-fe": ("acs", _acs),
        "multinomial-re": ("acs", _acs), "ordered-re": ("acs", _acs),
        "conditional": ("sets", _sets), "nested": ("sets", _sets),
        "mixed": ("panels", _panels), "conditional-fe": ("cfe", _cfe)}


# --------------------------------------------------------------------------
# the child
# --------------------------------------------------------------------------

def fit_logit_stream(model, path, device):
    """logit-stream's fit of `model`, robust variance, on `device`."""
    import polars as pl

    import logit_stream as ls
    kw = dict(verbose=False, device=device, vcov="hetero")
    if model in ("multinomial", "multinomial-fe", "multinomial-re"):
        rhs = {"multinomial": A.RHS, "multinomial-fe": A.RHS_FE}.get(model, A.RHS_RE)
        if model.endswith("-re"):
            kw.update(random="puma_id", vcov={"CRV1": "puma_id"})
        return ls.mlogit_stream(f"{A.OUTCOME['multinomial']} ~ {rhs}", str(path),
                                levels=A.RACES, **kw)
    if model in ("ordered", "ordered-fe", "ordered-re"):
        rhs = {"ordered": A.RHS, "ordered-fe": A.RHS_FE}.get(model, A.RHS_RE)
        if model.endswith("-re"):
            kw.update(random="puma_id", vcov={"CRV1": "puma_id"})
        return ls.ologit_stream(f"{A.OUTCOME['ordered']} ~ {rhs}", str(path), **kw)
    if model == "conditional":
        lf = pl.scan_parquet(path).filter(pl.col("avail") == 1)
        return ls.clogit_stream(CH.FML, lf, case="case", **kw)
    if model == "nested":
        lf = pl.scan_parquet(path).filter(pl.col("avail") == 1)
        return ls.nlogit_stream(CH.FML, lf, case="case", alt="alt", nests=NESTS, **kw)
    if model == "mixed":
        lf = pl.scan_parquet(path).filter(pl.col("avail") == 1)
        kw["vcov"] = {"CRV1": "ID"}
        return ls.mixlogit_stream(MX.FML, lf, case="case", panel="ID", random=MX.RANDOM,
                                  draws=MX.DRAWS, **kw)
    if model == "conditional-fe":
        return ls.cfelogit_stream("y ~ x1 + x2 + x3 | unit", str(path), **kw)
    raise ValueError(model)


def run_logit_stream(model, path, device, warmup):
    fit_logit_stream(model, warmup, device)
    t0 = time.perf_counter()
    fit = fit_logit_stream(model, path, device)
    t_fit = time.perf_counter() - t0
    d = fit.diagnostics
    passes = float((d.get("newton") or d.get("optimizer"))["seconds"])
    dev = d.get("device") or {}
    out = {"t_read": 0.0, "t_fit": t_fit, "n_obs": int(fit.n_obs),
           "passes_s": passes, "after_s": t_fit - d.get("pass0_seconds", 0.0) - passes,
           "pass0_s": d.get("pass0_seconds", 0.0), "converged": bool(fit.converged),
           "device_name": dev.get("name", "cpu"), "resident": dev.get("resident", ""),
           "device_bytes": dev.get("bytes", 0),
           "coefs": dict(zip(fit.coefnames, map(float, fit.beta)))}
    if hasattr(fit, "cleanup"):
        fit.cleanup()
    return out


def run_other(model, lib, path, device, warmup):
    """xlogit or jaxlogit, as the CPU suite runs them, on `device` (jaxlogit's
    platform is set by the parent, JAX_PLATFORMS)."""
    if lib == "xlogit":
        import xlogit
        if device == "cpu":
            xlogit.device.disable_gpu_acceleration()
        else:
            xlogit.device.enable_gpu_acceleration()
    module = MX if model == "mixed" else CH
    fit = {"xlogit": module.run_xlogit, "jax": module.run_jaxlogit}[lib]
    args = (MX.DRAWS,) if model == "mixed" else ()
    fit(warmup, *args)
    out = fit(path, *args)
    out["device_name"] = device
    return out


def runner(config, n, path, workdir):
    model, lib, device = config.split(":")
    kind, data = DATA[model]
    warmup = data(WARMUP_RE if model.endswith("-re") else WARMUP[kind])
    if lib == "ls":
        return run_logit_stream(model, path, device, warmup)
    return run_other(model, lib, path, device, warmup)


# --------------------------------------------------------------------------
# the parent
# --------------------------------------------------------------------------

def gpu_info():
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        return {"gpu": props["name"].decode(), "gpu_memory_gb":
                round(props["totalGlobalMem"] / 1024 ** 3, 1),
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_driver": cp.cuda.runtime.driverGetVersion()}
    except Exception as exc:                          # noqa: BLE001
        return {"gpu": f"unavailable: {exc}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", choices=list(SIZES), default=list(SIZES))
    ap.add_argument("--sizes", nargs="+", type=int)
    ap.add_argument("--only", nargs="+")
    args = ap.parse_args(argv)
    out = RESULTS_DIR / "gpu.csv"
    write_machine_info(RESULTS_DIR / "gpu_machine.json")
    info = json.loads((RESULTS_DIR / "gpu_machine.json").read_text())
    info.update(gpu_info())
    (RESULTS_DIR / "gpu_machine.json").write_text(json.dumps(info, indent=1) + "\n")
    for model in args.model:
        _kind, data = DATA[model]
        configs = {label: f"{model}:{kind}" for label, kind in
                   {**LS, **OTHERS.get(model, {})}.items()}
        for n in args.sizes or SIZES[model]:
            path = data(n)
            run_configs(
                SCRIPT, out, model, n, path, configs, reference="logit-stream (CPU)",
                only=args.only, absolute=("sd.TT",),
                timeout=lambda label: TIMEOUT_S["logit-stream" if label.startswith(
                    "logit-stream") else "other"],
                env=lambda label: {"JAX_PLATFORMS": "cpu" if "(CPU)" in label else "cuda"},
                extra=lambda label: {"device": "cpu" if "(CPU)" in label else "cuda"})


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, config, n, path, workdir = sys.argv
        run_child(runner, config, n, path, workdir)
    else:
        main()
