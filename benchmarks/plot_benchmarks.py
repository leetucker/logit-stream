"""Draw the benchmark figures from the CSVs the benchmarks write.

    python benchmarks/plot_benchmarks.py

Reads benchmarks/results/acs.csv, choice.csv, mixed.csv and panel.csv and writes
figures of wall time, peak memory and peak disk against the sample size,
one line per configuration, in a light and a dark version:

    docs/figures/{acs,acs_r,choice,choice_other,mixed,panel}_{time,memory,disk}.{light,dark}.svg
    docs/figures/acs_choice_{time,memory}.{light,dark}.svg

The ACS figures have three panels (binary, multinomial, ordered) with one
legend; acs_choice compares logit-stream's multinomial logit with the
discrete-choice libraries. A library has the same color and marker in every
panel of a figure. logit-stream is magenta: solid, dotted for its
low-memory setting, dashed for its fixed-effect model and dash-dotted for
its random-intercept model (different, richer models). Colors are the
reference categorical palette's, chosen per panel so that every pair
passes the palette validator's color-vision checks in both themes (see
FIGS); every series also has its own marker, so no line is identified by
color alone. Points that failed, ran out of memory or timed
out are left out and listed under the figure.

Only this file needs changing to restyle them; the numbers stay in the CSVs.
"""

from __future__ import annotations

import csv
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
# LOGIT_BENCH_RESULTS and LOGIT_BENCH_FIGURES: elsewhere, for a trial run
RESULTS = Path(os.environ.get("LOGIT_BENCH_RESULTS", HERE / "results"))
FIGURES = Path(os.environ.get("LOGIT_BENCH_FIGURES", HERE.parent / "docs" / "figures"))

SLOTS = {
    "blue": ("#2a78d6", "#3987e5"), "orange": ("#eb6834", "#d95926"),
    "aqua": ("#1baf7a", "#199e70"), "yellow": ("#eda100", "#c98500"),
    "magenta": ("#e87ba4", "#d55181"), "green": ("#008300", "#008300"),
    "violet": ("#6250d6", "#9085e9"), "red": ("#e34948", "#e66767"),
}
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e",
              "grid": "#e4e3de", "axis": "#b9b8b1", "slot": 0},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7",
             "grid": "#34342f", "axis": "#5b5a55", "slot": 1},
}

# Each figure: its CSV, its series -- configuration -> (color slot, marker,
# line style) -- its panels (title, model) and x label. Lines of one panel
# cross each other, so every pair of colors in a panel must be told apart,
# not only neighbors in the palette's order: the hues of each panel were
# chosen among the sets that pass the palette validator with --pairs all in
# both themes. Only two sets of four do ({magenta, blue, yellow, green} and
# {magenta, yellow, green, violet}), so no panel has more than three other
# libraries: the discrete-choice libraries on the ACS data, and the R
# libraries, get figures of their own. A figure whose panels give one color
# to different libraries has a legend in each panel ("legend": "panel").
# logit-stream's settings share its magenta, told apart by line style.
LS = {
    "logit-stream": ("magenta", "v", "-"),
    "logit-stream (low memory)": ("magenta", "X", ":"),
    "logit-stream (PUMA fixed effects)": ("magenta", "^", "--"),
    "logit-stream (PUMA random intercepts)": ("magenta", "P", "-."),
}
FE = ("magenta", "^", "--")
FIGS = {
    "acs": {
        "csv": "acs", "xlabel": "Adults in the sample",
        "panels": [("Binary (Hispanic origin)", "binary"),
                   ("Multinomial (race, 5 levels)", "multinomial"),
                   ("Ordered (education, 5 levels)", "ordered")],
        "series": {**LS,
                   "scikit-learn (lbfgs)": ("violet", "o", "-"),
                   "scikit-learn (newton-cholesky)": ("yellow", "s", "-"),
                   "statsmodels": ("green", "D", "-"),
                   "mord (all-threshold loss)": ("blue", "p", "-")},
    },
    "acs_choice": {
        "csv": "acs", "xlabel": "Adults in the sample", "metrics": ("time", "memory"),
        "panels": [("Multinomial (race): discrete-choice libraries", "multinomial")],
        "series": {"logit-stream": LS["logit-stream"],
                   "xlogit (OPG standard errors)": ("blue", "h", "-"),
                   "jaxlogit (no random parameters)": ("yellow", "*", "-")},
    },
    "acs_r": {
        "csv": "acs", "xlabel": "Adults in the sample",
        "panels": [("Binary: R and pyfixest", "binary"),
                   ("Multinomial: R", "multinomial"),
                   ("Ordered: R", "ordered")],
        "series": {"logit-stream": LS["logit-stream"],
                   "logit-stream (PUMA fixed effects)": FE,
                   "glm (R)": ("blue", "o", "-"),
                   "fixest (R)": ("yellow", "s", "-"),
                   "fixest (R, PUMA fixed effects)": ("yellow", "s", "--"),
                   "pyfixest": ("green", "D", "-"),
                   "multinom (R)": ("blue", "p", "-"),
                   "polr (R)": ("blue", "h", "-"),
                   "clm (R)": ("yellow", "*", "-")},
        "legend": "panel",
    },
    "mixed": {
        "csv": "mixed", "xlabel": "Respondents (9 choices each)",
        "panels": [("Mixed logit (Swissmetro panel, 500 draws): Python", "mixed"),
                   ("R", "mixed")],
        "panel_series": [["logit-stream", "xlogit (L-BFGS-B)", "jaxlogit", "Biogeme"],
                         ["logit-stream", "logitr (R)"]],
        "sharey": True,
        "series": {"logit-stream": LS["logit-stream"],
                   "xlogit (L-BFGS-B)": ("blue", "h", "-"),
                   "jaxlogit": ("yellow", "*", "-"),
                   "Biogeme": ("green", "D", "-"),
                   "logitr (R)": ("violet", "o", "-")},
    },
    "choice": {
        "csv": "choice", "xlabel": "Choice sets",
        "panels": [("Conditional logit (Swissmetro): Python", "conditional")],
        "series": {"logit-stream": LS["logit-stream"],
                   "xlogit": ("blue", "h", "-"),
                   "statsmodels": ("green", "D", "-"),
                   "jaxlogit (no random parameters)": ("yellow", "*", "-")},
    },
    "choice_other": {
        "csv": "choice", "xlabel": "Choice sets",
        "panels": [("Conditional logit: Biogeme and R", "conditional"),
                   ("Nested logit (Swissmetro)", "nested")],
        "panel_series": [["logit-stream", "Biogeme", "clogit (R survival)", "mlogit (R)"],
                         ["logit-stream", "Biogeme", "mlogit (R)"]],
        "series": {"logit-stream": LS["logit-stream"],
                   "Biogeme": ("violet", "D", "-"),
                   "clogit (R survival)": ("yellow", "s", "-"),
                   "mlogit (R)": ("green", "o", "-")},
    },
    "panel": {
        "csv": "panel", "xlabel": "Rows (units of five periods)",
        "panels": [("Conditional (Chamberlain) logit", "conditional"),
                   ("Fixed-effects logit", "fixed effects"),
                   ("Bias-corrected", "bias-corrected")],
        "series": {"logit-stream": LS["logit-stream"],
                   "statsmodels": ("green", "D", "-"),
                   "clogit (R survival)": ("yellow", "s", "-"),
                   "fixest (R)": ("blue", "o", "-"),
                   "pyfixest": ("green", "p", "-"),
                   "alpaca (R)": ("yellow", "h", "-")},
        "legend": "panel",
    },
}
METRICS = {
    "time": ("wall_s", "Wall time (seconds)", 1.0),
    "memory": ("peak_memory_mb", "Peak memory (GB)", 1e-3),
    "disk": ("peak_disk_mb", "Peak disk (GB)", 1e-3),
}


def read(name):
    path = RESULTS / f"{name}.csv"
    if not path.exists():
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _count(v, _pos=None):
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def _value(v, _pos=None):
    if v >= 100:
        return f"{v:,.0f}"
    if v >= 1:
        return f"{v:g}"
    return f"{v:.2g}"


def figure(name, metric, theme):
    spec = FIGS[name]
    rows = read(spec["csv"])
    if not rows:
        return None
    col, ylabel, scale = METRICS[metric]
    t = THEMES[theme]
    panels, series = spec["panels"], spec["series"]
    width = max(4.2 * len(panels) + 0.6, 7.0)
    # panels of one model on one scale (the mixed logit's two)
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 4.0), squeeze=False,
                             sharey=bool(spec.get("sharey")))
    fig.patch.set_facecolor(t["surface"])
    failed, handles = [], {}
    per_panel = spec.get("panel_series") or [list(series)] * len(panels)
    for ax, (title, model), labels in zip(axes[0], panels, per_panel):
        ax.set_facecolor(t["surface"])
        mrows = [r for r in rows if r.get("model", model) == model]
        local = {}
        for label in labels:
            slot, marker, style = series[label]
            pts = []
            for r in mrows:
                if r["configuration"] != label:
                    continue
                if r["status"] != "ok":
                    where = f" ({model})" if len(panels) > 1 else ""
                    failed.append(f"{label}{where} at {_count(float(r['n']))}: "
                                  f"{r['status'].split(':')[0]}")
                    continue
                v = float(r.get(col) or 0) * scale
                if metric == "disk" and v <= 0:
                    continue
                pts.append((float(r["n"]), max(v, 1e-3)))
            if not pts:
                continue
            pts.sort()
            color = SLOTS[slot][t["slot"]]
            (h,) = ax.plot(*zip(*pts), linestyle=style, marker=marker, color=color,
                           linewidth=2, markersize=7, markeredgecolor=t["surface"],
                           markeredgewidth=1.2, label=label)
            handles.setdefault(label, h)
            local[label] = h
        if spec.get("legend") == "panel" and local:
            ax.legend(list(local.values()), list(local), loc="upper left", frameon=False,
                      fontsize=7.5, labelcolor=t["text"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, color=t["text"], fontsize=11, loc="left")
        ax.set_xlabel(spec["xlabel"], color=t["muted"], fontsize=9)
        ax.xaxis.set_major_formatter(FuncFormatter(_count))
        ax.yaxis.set_major_formatter(FuncFormatter(_value))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color=t["grid"], linewidth=0.8)
        ax.tick_params(colors=t["muted"], labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(t["axis"])
    axes[0][0].set_ylabel(ylabel, color=t["muted"], fontsize=9)
    order = [lb for lb in series if lb in handles]
    # the note of what is not shown at the bottom, the legend above it
    note = ""
    if failed and metric != "disk":          # only logit-stream writes to disk
        note = "Not shown: " + "; ".join(dict.fromkeys(failed)) + "."
    if spec.get("note") and metric != "disk":
        note = (note + " " + spec["note"]).strip()
    if note:
        note = "\n".join(textwrap.wrap(note, int(width * 19)))
        fig.text(0.01, 0.01, note, color=t["muted"], fontsize=7, va="bottom")
    note_h = (0.03 * (note.count("\n") + 1) + 0.01) if note else 0.0
    if spec.get("legend") == "panel":
        order = []
    ncol = min(len(order), 4 if len(panels) > 1 else 3)
    nrow = -(-len(order) // max(ncol, 1)) if order else 0
    if order:
        fig.legend([handles[lb] for lb in order], order, loc="lower center",
                   ncol=ncol, frameon=False, fontsize=8.5,
                   labelcolor=t["text"], bbox_to_anchor=(0.5, note_h))
    fig.tight_layout(rect=(0, note_h + 0.06 + 0.05 * nrow, 1, 1))
    return fig


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, spec in FIGS.items():
        for metric in spec.get("metrics", METRICS):
            for theme in THEMES:
                fig = figure(name, metric, theme)
                if fig is None:
                    continue
                path = FIGURES / f"{name}_{metric}.{theme}.svg"
                fig.savefig(path, facecolor=fig.get_facecolor())
                plt.close(fig)
                print(f"wrote {path}")


if __name__ == "__main__":
    main()
