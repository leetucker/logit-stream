"""Shared machinery for the benchmarks: one configuration per subprocess, peak
memory and disk measured the same way for every library, results to CSV.

Adapted from hdfe-stream's benchmarks/_harness.py. Each configuration runs in
its own subprocess because peak memory is a per-process high-water mark. The
child measures

  * wall-clock seconds, split into reading the data and the estimation itself
    (logit-stream reads the data inside its fit, so its read time is zero and
    its fit time includes the reading);
  * peak resident set size (``VmHWM``) of the whole process, and the baseline
    after imports, before the data are read;
  * peak disk, by polling the allocated size of the configuration's own work
    directory (zero for libraries that do not write to disk).

The parent writes one CSV row per (model, size, configuration) as soon as it
has it, replacing an earlier row with the same key, so an interrupted run keeps
what it measured.

**R and Biogeme.** The R packages run from benchmarks/external/logit.R, and
Biogeme from benchmarks/external/biogeme_fit.py in an environment of its own
(it needs an older pandas than the benchmarks' environment has): each is given
what to fit as JSON, measures itself as the Python child does (VmHWM for peak
memory) and prints the same RESULT line. Neither is a dependency: where one is
not installed, its configurations are skipped, and nothing is written for them
(benchmarks/README.md says how to install them).

**Resuming.** When the environment variable LOGIT_BENCH_LEDGER names a file
(the EC2 runner sets it), every row written is also recorded there with a
digest, and `is_done` reports a configuration as finished only if its row is
still in the CSV exactly as the ledger recorded it. Without the variable,
nothing is skipped and nothing extra is written.

**Failures carry over.** A configuration that timed out or ran out of memory
at one size is not run at larger sizes (`failed_smaller`), whether the smaller
size was measured in this process or an earlier one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
# where the CSVs go: benchmarks/results, or LOGIT_BENCH_RESULTS (for a trial
# run that should not touch the published numbers)
RESULTS_DIR = Path(os.environ.get("LOGIT_BENCH_RESULTS", HERE / "results"))
EXTERNAL_DIR = HERE / "external"
# large downloads that survive a spot stop (the EC2 runner points this at the
# root volume; the data directory may be on an instance store that does not)
CACHE_DIR = Path(os.environ.get("LOGIT_BENCH_CACHE", DATA_DIR / "raw"))
KEY = ("model", "n", "configuration")


def peak_rss_mb():
    """Peak resident set size of this process in MB, from /proc (VmHWM, which
    unlike ru_maxrss is not inherited across fork+exec)."""
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def allocated_bytes(root):
    total = 0
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_blocks * 512
            except OSError:
                pass
    return total


class DiskPoller:
    """The maximum allocated size of a directory, sampled in a thread."""

    def __init__(self, root, interval=0.2):
        self.root = Path(root)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, allocated_bytes(self.root))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, allocated_bytes(self.root))


def run_child(runner, config, n, path, workdir):
    """In the child: run one configuration and print its report as JSON."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    baseline = peak_rss_mb()
    with DiskPoller(workdir) as disk:
        result = runner(config, int(n), Path(path), workdir)
    result["peak_memory_mb"] = peak_rss_mb()
    result["baseline_memory_mb"] = baseline
    result["peak_disk_mb"] = max(disk.peak / 1e6, result.pop("reported_disk_mb", 0.0))
    print("RESULT " + json.dumps(result, default=float))


# how R and Python say that an allocation was refused
OUT_OF_MEMORY = ("MemoryError", "Unable to allocate", "OutOfMemory", "cannot allocate",
                 "out of memory", "RESOURCE_EXHAUSTED",
                 "exceed the memory budget")  # Biogeme refusing a Hessian that is too large


def measure(script, config, n, path, workdir, timeout=None, command=None, env=None):
    """Run one configuration in a subprocess; its report, with ``status``.
    The script's own child, or `command` (from `external_command`)."""
    workdir = Path(workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = command or [sys.executable, str(script), "--child", config, str(n), str(path),
                      str(workdir)]
    env = dict(os.environ, PYTHONWARNINGS="ignore", **(env or {}))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": f"timeout after {timeout:.0f} s"}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode in (-9, 137):
        return {"status": "killed (out of memory)"}
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        oom = next((ln.strip() for ln in lines if any(s in ln for s in OUT_OF_MEMORY)), None)
        if oom:
            return {"status": "out of memory: " + oom[:200]}
        return {"status": "failed: " + " | ".join(lines[-3:])[:400]}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            rep = json.loads(line[len("RESULT "):])
            rep["status"] = "ok"
            rep["process_s"] = elapsed
            return rep
    return {"status": "failed: no result line"}


# --------------------------------------------------------------------------
# R and Biogeme
# --------------------------------------------------------------------------

EXTERNAL = {
    # kind -> (what, executable, its environment variable, its environment, script)
    "R": ("R", "Rscript", "LOGIT_BENCH_RSCRIPT", ".r", "logit.R"),
    "biogeme": ("Biogeme", "python", "LOGIT_BENCH_BIOGEME_PYTHON", ".biogeme",
                "biogeme_fit.py"),
}


def external_executable(kind):
    """Rscript, or the Python of Biogeme's environment: from its environment
    variable if set, else benchmarks/external/.r or .biogeme (benchmarks/
    README.md), else (Rscript only) PATH. None if not installed."""
    _what, name, variable, local, _script = EXTERNAL[kind]
    if os.environ.get(variable):
        return os.environ[variable]
    path = EXTERNAL_DIR / local / "bin" / name
    if path.exists():
        return str(path)
    return shutil.which(name) if kind == "R" else None


def external_command(kind, **spec):
    """The command that runs one fit in R (``kind="R"``, `spec["fit"]` one of
    logit.R's) or Biogeme, or None if that is not installed. Every spec gets
    `threads`, every logical CPU (for the packages that take a number of
    threads: fixest; the Python libraries use them all by default)."""
    executable = external_executable(kind)
    if executable is None:
        return None
    spec = {"threads": os.cpu_count(), **spec}
    for k, v in spec.items():
        if isinstance(v, Path):
            spec[k] = str(v)
    return [executable, str(EXTERNAL_DIR / EXTERNAL[kind][4]), json.dumps(spec)]


def external_missing(kind):
    what, name, variable, local, _script = EXTERNAL[kind]
    return (f"skipped: {what} is not installed (no benchmarks/external/{local}, "
            f"or set {variable})")


def external_versions():
    """R and its packages, and Biogeme, where installed."""
    out = {}
    rscript = external_executable("R")
    if rscript:
        code = ('pk <- c("fixest", "alpaca", "nnet", "MASS", "ordinal", "survival", '
                '"mlogit", "logitr"); v <- sapply(pk, function(p) tryCatch('
                'as.character(packageVersion(p)), error = function(e) NA)); '
                'cat(jsonlite::toJSON(c(R = paste(R.version$major, R.version$minor, '
                'sep = "."), as.list(v)), auto_unbox = TRUE))')
        try:
            text = subprocess.run([rscript, "-e", code], capture_output=True, text=True,
                                  timeout=300).stdout
            out.update({f"r:{k}": v for k, v in json.loads(text).items()})
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    python = external_executable("biogeme")
    if python:
        try:
            out["biogeme"] = subprocess.run(
                [python, "-c", "import importlib.metadata as m; print(m.version('biogeme'))"],
                capture_output=True, text=True, timeout=300).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            pass
    return out


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

LEDGER_ENV = "LOGIT_BENCH_LEDGER"

# a row with one of these statuses is a result, to be kept rather than
# measured again; a failure may be the process being stopped, so it is not
_FINAL = ("ok", "timeout", "killed (out of memory)", "out of memory")
# and with one of these, the configuration is not run at larger sizes
_FATAL = ("timeout", "killed (out of memory)", "out of memory")


def _digest(row):
    """Identify a row by its non-empty values, as text, so that the row as
    written and as read back from the CSV give the same digest."""
    items = sorted((k, str(v)) for k, v in row.items() if v not in (None, ""))
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()


def _ledger():
    name = os.environ.get(LEDGER_ENV)
    if not name:
        return None, {}
    path = Path(name)
    try:
        return path, json.loads(path.read_text())
    except (OSError, ValueError):
        return path, {}


def _entry(path, row, key):
    return Path(path).name, "|".join(str(row[k]) for k in key)


def _atomic_write(path, write):
    """Write through `write(handle)` to a temporary file, then rename it, so
    that a process stopped part-way never leaves half a file."""
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    with open(temporary, "w", newline="") as handle:
        write(handle)
    os.replace(temporary, path)


def _rows(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def is_done(path, values, key=KEY):
    """Whether the configuration with these key `values` (a dict) was already
    measured by this run: its row is in the CSV at `path`, unchanged since it
    was written, with a final status. Always False unless the ledger is on."""
    ledger_path, ledger = _ledger()
    if ledger_path is None:
        return False
    name, entry = _entry(path, values, key)
    for row in _rows(path):
        if "|".join(str(row[k]) for k in key) == entry:
            return (row["status"].startswith(_FINAL)
                    and ledger.get(name, {}).get(entry) == _digest(row))
    return False


def failed_smaller(path, model, configuration, n):
    """The status of this configuration at a smaller size where it timed out
    or ran out of memory, if any (then it is not run at `n`)."""
    for row in _rows(path):
        if (row.get("model") == model and row["configuration"] == configuration
                and int(row["n"]) < int(n) and row["status"].startswith(_FATAL)):
            return f"{row['status']} at {int(row['n']):,}"
    return None


def upsert_csv(path, rows, key=KEY):
    """Write `rows` into the CSV at `path`, replacing rows with the same key."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _rows(path)
    new = {tuple(str(r[k]) for k in key) for r in rows}
    merged = [r for r in existing if tuple(str(r[k]) for k in key) not in new] + list(rows)
    fields = list(dict.fromkeys(f for r in merged for f in r))
    merged.sort(key=lambda r: (str(r.get("model", "")), int(r["n"]), int(r.get("order") or 0)))

    def write(handle):
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)

    _atomic_write(path, write)
    ledger_path, ledger = _ledger()
    if ledger_path is not None:
        for r in rows:
            name, entry = _entry(path, r, key)
            ledger.setdefault(name, {})[entry] = _digest({k: v for k, v in r.items()})
        _atomic_write(ledger_path, lambda handle: handle.write(json.dumps(ledger)))


def coef_diff(coefs, ref, absolute=()):
    """The largest absolute difference between two {name: value} dicts on
    their common names (`absolute`: compared as absolute values, for standard
    deviations whose sign is not identified); "" if none are common."""
    common = [c for c in coefs if c in ref]
    if not common:
        return ""
    return max(abs(abs(coefs[c]) - abs(ref[c])) if c in absolute else abs(coefs[c] - ref[c])
               for c in common)


# what this process has warmed up (warm_up): a kind once per process, so each
# step of an EC2 run, and a step restarted after a spot interruption, warms
# up again before it times anything
_WARMED = set()


def warm_up(script, kind, warmup, workdir):
    """Fit `kind` once, untimed and discarded, on `warmup()`'s small data
    ((n, path)), so that compiling the numba kernels (cached on disk) and
    reading them back after a reboot is not charged to the first
    configuration that happens to run. logit-stream's own configurations
    only, as in hdfe-stream's benchmarks: the other packages' compilation is
    per process, which every one of their fits pays."""
    if kind in _WARMED:
        return
    _WARMED.add(kind)
    n, path = warmup()
    rep = measure(script, kind, n, path, workdir, timeout=600)
    took = (f"{rep['t_read'] + rep['t_fit']:.1f} s" if rep.get("status") == "ok"
            else rep.get("status"))
    print(f"(warm-up: {kind} on {n:,} rows, {took})", flush=True)


def run_configs(script, out, model, n, path, configs, *, command=None, timeout=None,
                reference="logit-stream", absolute=(), only=None, extra=None, env=None,
                warmup=None):
    """Measure each of `configs` ({label: kind}) at size `n` and write its row:
    skip what this run measured already (`is_done`) and what failed at a
    smaller size (`failed_smaller`). `command(label, kind)` gives an external
    command, None for the script's own child, or a string, a reason to skip
    (R or Biogeme not installed). `reference` is the label whose
    coefficients the others are compared with (max_abs_diff), or a dict
    {label: its reference} for the labels compared with another (a fit of the
    same model, measured before them; None: not compared); `extra(label)`
    adds columns to the row; `env(label)` sets environment variables for the
    child; `timeout` is seconds, or a function of the label; `warmup(kind)`
    gives the small data ((n, path)) a logit-stream configuration is warmed
    up on first (warm_up)."""
    # the fits measured before at this size (a reference for a configuration
    # run alone, with --only), then this call's
    measured = {r["configuration"]: json.loads(r["coefs"]) for r in _rows(out)
                if r.get("model") == model and str(r["n"]) == str(n)
                and r["status"] == "ok" and r.get("coefs")}

    def ref_of(label):
        if isinstance(reference, dict) and label in reference:
            return reference[label]
        return reference if isinstance(reference, str) else reference.get("*")

    for order, (label, kind) in enumerate(configs.items()):
        if only and label not in only:
            continue
        row = {"model": model, "n": n, "configuration": label, "order": order,
               **((extra or (lambda _l: {}))(label))}
        name = f"{model:14s} n={n:>11,}  {label:42s}"
        if is_done(out, row):
            print(f"{name} measured before, kept", flush=True)
            kept = next((json.loads(r["coefs"]) for r in _rows(out)
                         if all(str(r[k]) == str(row[k]) for k in KEY) and r.get("coefs")),
                        None)
            if kept is not None:
                measured[label] = kept
            continue
        failed = failed_smaller(out, model, label, n)
        if failed:
            print(f"{name} not run ({failed})", flush=True)
            continue
        cmd = command(label, kind) if command else None
        if isinstance(cmd, str):
            print(f"{name} {cmd}", flush=True)
            continue
        workdir = RESULTS_DIR / "work" / kind.replace(":", "_")
        if warmup and cmd is None and label.startswith("logit-stream"):
            warm_up(script, kind, lambda k=kind: warmup(k), workdir)
        print(name, end=" ", flush=True)
        limit = timeout(label) if callable(timeout) else timeout
        rep = measure(script, kind, n, path, workdir,
                      timeout=limit, command=cmd, env=env(label) if env else None)
        row["status"] = rep.pop("status")
        if row["status"] != "ok":
            print(row["status"], flush=True)
            upsert_csv(out, [row])
            continue
        coefs = rep.pop("coefs", {})
        rep.pop("se", None)
        row.update(rep)
        row["wall_s"] = rep["t_read"] + rep["t_fit"]
        row["coefs"] = json.dumps(coefs)
        measured[label] = coefs
        ref = measured.get(ref_of(label))
        if ref is not None and coefs and ref_of(label) != label:
            row["max_abs_diff"] = coef_diff(coefs, ref, absolute)
        conv = "" if str(row.get("converged", True)).lower() in ("true", "1") \
            else " (NOT converged)"
        print(f"{row['wall_s']:9.2f} s  {row['peak_memory_mb'] / 1e3:6.2f} GB  "
              f"disk {row['peak_disk_mb'] / 1e3:6.2f} GB  steps {row.get('iterations', '')}"
              f"{conv}  diff {row.get('max_abs_diff', '')}", flush=True)
        upsert_csv(out, [row])


def machine_info():
    import importlib.metadata as md

    def version(name):
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None

    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_gb = None
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    mem_gb = round(int(line.split()[1]) / 1024 ** 2, 1)
    except OSError:
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "memory_gb": mem_gb,
            "platform": platform.platform(), "python": platform.python_version(),
            **{name: version(name) for name in
               ("logit-stream", "hdfe-stream", "scikit-learn", "statsmodels", "xlogit",
                "jaxlogit", "jax", "mord", "pyfixest", "polars", "numpy", "numba", "scipy",
                "pandas", "cupy-cuda13x")},
            **external_versions(),
            "measured": time.strftime("%Y-%m-%d")}


def write_machine_info(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(machine_info(), indent=1) + "\n")
