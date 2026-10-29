"""Write the benchmark tables of the documentation from the CSVs.

    python benchmarks/make_tables.py

Each table sits in a page of docs/benchmarks between a pair of markers,

    <!-- table: acs binary python -->
    ...
    <!-- /table -->

and this script replaces what is between them with wall-time and peak-memory
tables of that block (BLOCKS below: the CSV, the model, and the
configurations as columns, in order; a column with no rows is left out).
A dash means the configuration was not run at that size, because it failed or
timed out at a smaller one. A fit that reports no convergence and whose
coefficients differ from logit-stream's by more than 0.01 is "did not
converge" in place of its time and memory.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("LOGIT_BENCH_RESULTS", HERE / "results"))
DOCS = HERE.parent / "docs" / "benchmarks"

LS = ["logit-stream", "logit-stream (low memory)", "logit-stream (PUMA fixed effects)",
      "logit-stream (PUMA random intercepts)", "logit-stream (state fixed + PUMA random)"]
ACS_PY = {
    "binary": LS + ["scikit-learn (lbfgs)", "scikit-learn (newton-cholesky)", "statsmodels"],
    "multinomial": LS + ["scikit-learn (lbfgs)", "scikit-learn (newton-cholesky)", "statsmodels",
                         "xlogit (OPG standard errors)", "jaxlogit (no random parameters)"],
    "ordered": LS + ["statsmodels", "mord (all-threshold loss)"],
}
ACS_R = {
    "binary": ["logit-stream", "logit-stream (PUMA fixed effects)", "glm (R)", "fixest (R)",
               "fixest (R, PUMA fixed effects)", "pyfixest"],
    "multinomial": ["logit-stream", "multinom (R)"],
    "ordered": ["logit-stream", "polr (R)", "clm (R)"],
}
BLOCKS = {
    **{f"acs {m} python": ("acs", m, cols, "n") for m, cols in ACS_PY.items()},
    **{f"acs {m} r": ("acs", m, cols, "n") for m, cols in ACS_R.items()},
    "choice conditional": ("choice", "conditional",
                           ["logit-stream", "xlogit", "pylogit", "statsmodels",
                            "jaxlogit (no random parameters)", "Biogeme",
                            "clogit (R survival)", "mlogit (R)"], "choice sets"),
    "choice nested": ("choice", "nested", ["logit-stream", "Biogeme", "mlogit (R)"],
                      "choice sets"),
    "mixed": ("mixed", "mixed", ["logit-stream", "xlogit (L-BFGS-B)", "jaxlogit", "Biogeme",
                                 "logitr (R)"], "respondents"),
    "panel conditional": ("panel", "conditional",
                          ["logit-stream", "statsmodels", "clogit (R survival)"], "rows"),
    "panel fixed effects": ("panel", "fixed effects",
                            ["logit-stream", "fixest (R)", "pyfixest"], "rows"),
    "panel bias-corrected": ("panel", "bias-corrected", ["logit-stream", "alpaca (R)"], "rows"),
}
GPU_MODELS = [("conditional", "conditional logit (Swissmetro, choice sets)"),
              ("conditional-fe", "Chamberlain's conditional logit (simulated panel, rows)"),
              ("nested", "nested logit"),
              ("multinomial", "multinomial logit"), ("multinomial-fe", "multinomial, PUMA fixed effects"),
              ("multinomial-re", "multinomial, PUMA random intercepts"),
              ("ordered", "ordered logit"), ("ordered-fe", "ordered, PUMA fixed effects"),
              ("ordered-re", "ordered, PUMA random intercepts"),
              ("mixed", "mixed logit (respondents)")]


def read(name):
    with open(RESULTS / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))


def cell(row, field, scale, digits):
    if row is None:
        return "–"
    status = row["status"]
    if status != "ok":
        if "memory" in status:
            return "out of memory"
        if status.startswith("timeout"):
            return "timeout"
        return "failed"
    if row.get("converged") == "False" and float(row.get("max_abs_diff") or 0) > 1e-2:
        return "did not converge"  # not clm's flag at 3-6M adults: its estimates agree
    v = float(row[field]) * scale
    if digits == 1:
        return f"{v:,.0f}" if v >= 100 else f"{v:,.1f}" if v >= 1 else f"{v:.2f}"
    return f"{v:,.{digits}f}"


def table(rows, model, cols, nname):
    rows = [r for r in rows if r["model"] == model]
    cols = [c for c in cols if any(r["configuration"] == c for r in rows)]
    ns = sorted({int(r["n"]) for r in rows})
    by = {(r["configuration"], int(r["n"])): r for r in rows}
    out = []
    for title, field, scale, digits in (("Wall time (seconds)", "wall_s", 1.0, 1),
                                        ("Peak memory (GB)", "peak_memory_mb", 1e-3, 2)):
        out += [f"**{title}**", "", f"| {nname} | " + " | ".join(cols) + " |",
                "|---:|" + "---:|" * len(cols)]
        for n in ns:
            out.append(f"| {n:,} | " + " | ".join(
                cell(by.get((c, n)), field, scale, digits) for c in cols) + " |")
        out.append("")
    return "\n".join(out)


def gpu_table():
    rows = read("gpu")
    by = {(r["model"], r["configuration"], int(r["n"])): r for r in rows}
    libs = []
    for r in rows:
        lib = r["configuration"].rsplit(" (", 1)[0]
        if lib not in libs:
            libs.append(lib)
    out = ["| model | rows | " + " | ".join(
        f"{lib} {dev}" for lib in libs for dev in ("CPU", "GPU")
        if any(r["configuration"] == f"{lib} ({dev})" for r in rows)) + " |"]
    cols = [(lib, dev) for lib in libs for dev in ("CPU", "GPU")
            if any(r["configuration"] == f"{lib} ({dev})" for r in rows)]
    out.append("|---|---:|" + "---:|" * len(cols))
    for m, label in GPU_MODELS:
        for n in sorted({int(r["n"]) for r in rows if r["model"] == m}):
            out.append(f"| {label} | {n:,} | " + " | ".join(
                cell(by.get((m, f"{lib} ({dev})", n)), "wall_s", 1.0, 1)
                for lib, dev in cols) + " |")
    return "\n".join(out) + "\n"


def mixed_accuracy():
    """logit-stream's, xlogit's, jaxlogit's, Biogeme's and logitr's fits of the
    survey at 100, 500 and 2,000 draws (mixed_check.json), against the exact
    likelihood's estimates."""
    import json
    check = json.loads((RESULTS / "mixed_check.json").read_text())
    libs = list(dict.fromkeys(f["library"] for f in check["fits"]))
    draws = sorted({f["draws"] for f in check["fits"]})
    by = {(f["library"], f["draws"]): f for f in check["fits"]}
    out = ["| draws | " + " | ".join(libs) + " |", "|---:|" + "---:|" * len(libs)]
    for d in draws:
        out.append(f"| {d:,} | " + " | ".join(
            f"{by[lib, d]['loglik']:,.3f} ({max(abs(x) for x in by[lib, d]['diff'].values()):.3f})"
            if (lib, d) in by else "–" for lib in libs) + " |")
    return "\n".join(out) + "\n"


def main():
    cache = {}
    for page in sorted(DOCS.glob("*.md")):
        text = page.read_text()

        def fill(m):
            key = m.group(1).strip()
            if key == "mixed accuracy":
                return f"<!-- table: mixed accuracy -->\n\n{mixed_accuracy()}\n<!-- /table -->"
            if key == "gpu":
                return f"<!-- table: gpu -->\n\n{gpu_table()}\n<!-- /table -->"
            name, model, cols, nname = BLOCKS[key]
            rows = cache.setdefault(name, read(name))
            return f"<!-- table: {key} -->\n\n{table(rows, model, cols, nname)}\n<!-- /table -->"

        new = re.sub(r"<!-- table: (.*?) -->.*?<!-- /table -->", fill, text, flags=re.S)
        if new != text:
            page.write_text(new)
            print(f"wrote {page}")


if __name__ == "__main__":
    main()
