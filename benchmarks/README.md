# Benchmarks

```bash
pip install -e ".[benchmark]"          # scikit-learn, statsmodels, mord, xlogit, pylogit, matplotlib
pip install "jax[cpu]" jaxlogit         # jaxlogit

python benchmarks/prepare_acs.py        # downloads the ACS 2019-2023 PUMS (2.2 GB), builds the sample
python benchmarks/acs_benchmark.py      # binary, multinomial, ordered: 10k to 12.9M adults
python benchmarks/choice_benchmark.py   # conditional and nested logit: Swissmetro, 6,768 to 10M choice sets
python benchmarks/mixed_benchmark.py check   # mixed logit: accuracy against the exact likelihood
python benchmarks/mixed_benchmark.py    # mixed logit: 752 to 1M respondents
python benchmarks/panel_benchmark.py    # conditional, fixed-effects and bias-corrected logit for panels
python benchmarks/gpu_benchmark.py      # CPU against GPU (needs an NVIDIA GPU and CuPy)
python benchmarks/plot_benchmarks.py    # docs/figures/*.svg from results/*.csv
```

| suite | models | compared with |
|---|---|---|
| `acs_benchmark.py` | binary, multinomial, ordered logit (41 covariates), and logit-stream's PUMA fixed effects and random intercepts | scikit-learn (L-BFGS, Newton–Cholesky), statsmodels, pyfixest (binary), mord (ordered), xlogit and jaxlogit (multinomial); R: glm and fixest (binary, fixest also with PUMA fixed effects), nnet's multinom, MASS's polr and ordinal's clm |
| `choice_benchmark.py` | conditional logit; nested logit (train and car in a nest) | xlogit, statsmodels, jaxlogit, Biogeme; R: survival's clogit, mlogit (both models) |
| `mixed_benchmark.py` | mixed logit, a normal coefficient, panels of nine choices, 500 draws | xlogit, jaxlogit, Biogeme; R: logitr; and the exact likelihood (`check`) |
| `panel_benchmark.py` | Chamberlain's conditional logit; the fixed-effects logit; its bias correction | statsmodels' ConditionalLogit, R's survival clogit; R's fixest and pyfixest; R's alpaca |
| `gpu_benchmark.py` | every model, on one machine's CPU and GPU | xlogit and jaxlogit on both, for the conditional and mixed logit |

Each configuration runs in its own subprocess (`_harness.py`), which reports
wall time (reading the data plus fitting), peak resident memory (`VmHWM`) and
the peak size of its work directory. Results are upserted into
`results/<suite>.csv` as they come, so a run can be interrupted and resumed for
some sizes (`acs_benchmark.py 3000000 --model ordered --only "polr (R)"`);
`results/machine.json` records the machine and library versions. A
configuration that fails, runs out of memory or exceeds `--timeout` (30
minutes; on the GPU suite, an hour for logit-stream and ten minutes for the
others) is not run at larger sizes, including in a later process: the CSV
says it failed. `LOGIT_BENCH_RESULTS` (and, for the figures,
`LOGIT_BENCH_FIGURES`) send a trial run elsewhere, leaving the published
numbers alone.

## R and Biogeme

The R packages run from `external/logit.R` and Biogeme from
`external/biogeme_fit.py`, each given what to fit as JSON and reporting
its own time and memory as the Python children do. Neither is a dependency:
where one is not installed, its configurations print "skipped" and write
nothing. Install them where `_harness.py` looks:

```bash
# R and its packages, from conda-forge, at the versions measured
micromamba create -p benchmarks/external/.r -f benchmarks/external/environment.yml
# Biogeme, in an environment of its own (it needs pandas < 3)
uv venv --python 3.12 benchmarks/external/.biogeme
uv pip install --python benchmarks/external/.biogeme/bin/python biogeme polars pyarrow
```

or point `LOGIT_BENCH_RSCRIPT` at an `Rscript` and `LOGIT_BENCH_BIOGEME_PYTHON`
at a Python with Biogeme. The R packages use their defaults (one thread),
except fixest, which is given every logical CPU, as hdfe-stream's benchmarks
give it.

## Notes on the libraries

pylogit 1.0.1 imports `collections.Iterable` and friends, which moved to
`collections.abc` in Python 3.10, and calls `np.in1d`, which NumPy 2.4
removed. It ran out of memory at 30,000 choice sets in the first runs, and is
no longer run. jaxlogit fits only the mixed logit; it is run with no random
parameters, which is the conditional (and, in long format, the multinomial)
logit. xlogit's conditional logit has no GPU code (only its mixed logit
uses CuPy), so the GPU suite runs it on the CPU only. nnet's multinom is
given more iterations than its default (100, short of convergence with 164
coefficients) and asked for its Hessian, as summary() would; its BFGS stops
at a relative tolerance of 1e-8, about 5e-4 from the maximum on these data.

The data descriptions and the reasons for the specifications are at the top
of each script.

## On EC2

`ec2/` (not in the public repository: its settings name the AWS account's
bucket) runs these suites on spot instances, the CPU suites on a
c8id.4xlarge and the GPU suite on a p5.4xlarge, as hdfe-stream's benchmarks
do, with the results written to S3 as they come and a resume after an
interruption. See `ec2/README.md`.
