# Benchmarks

Five benchmarks compare logit-stream with the other Python and R libraries,
on real data or data resampled from it, as the sample grows:

| benchmark | models | data | compared with |
|---|---|---|---|
| [ACS microdata](acs.md) | binary, multinomial and ordered logit of three demographic characteristics on 41 covariates | the adults of the 2019–2023 ACS 5-year PUMS, 10,000 to 12.9 million | scikit-learn (L-BFGS and Newton–Cholesky), statsmodels, mord; for the multinomial logit also xlogit and jaxlogit; R's glm, fixest, nnet, MASS and ordinal; pyfixest |
| [Swissmetro](choice.md) | conditional and nested logit of mode choice (Biogeme's first examples) | the Swissmetro survey's 6,768 choice sets, resampled to 10 million | xlogit, pylogit, statsmodels, jaxlogit, Biogeme; R's survival and mlogit |
| [Panels](panel.md) | Chamberlain's conditional logit, the fixed-effects logit and its bias correction | simulated panels of five periods, 100,000 to 10 million rows | statsmodels; R's survival, fixest and alpaca; pyfixest |
| [Mixed logit](mixed.md) (experimental) | mixed logit with a normal coefficient of travel time, panel by respondent | the Swissmetro survey's 752 respondents, resampled to 1 million (27 million rows) | xlogit, jaxlogit, Biogeme, R's logitr; and the exact likelihood |
| [GPU](gpu.md) (experimental) | every model above, on the CPU and on an NVIDIA H100 | the same data, at the largest sizes | xlogit and jaxlogit on both |

The CPU benchmarks ran on one AWS EC2 spot instance each (a c8id.4xlarge:
an Intel Xeon 6975P-C with 16 vCPUs and 30.8 GB of memory, local NVMe disk,
Linux; the benchmark process capped at 96% of memory) and the GPU benchmark
on a p5.4xlarge (an AMD EPYC 7R13 with 16 vCPUs, an H100 with 80 GB, and the
process capped at 29.6 GB of host memory).
`benchmarks/results/machine.json` and `gpu_machine.json` have the details and
the versions, and the commits of the code. Each point is a single run. A spot
instance can be interrupted and restarted during a benchmark; the harness
fits a small model first in every process so that compilation is never
counted, and fits that were the first after a restart were run again. We have
not measured the run-to-run spread on EC2; on the 8-CPU WSL2 machine that the
first runs used, the same fit could take up to about 40% longer from one run
to the next. Read the differences between libraries, which are several-fold,
rather than the kinks in a line.

## How it is measured

Every configuration runs in its own subprocess, since peak memory is a
per-process high-water mark (see `benchmarks/_harness.py`, adapted from
hdfe-stream's). The measurements are:

- **Wall time:** end to end from a Parquet file of the sample. The in-memory
  libraries need the data as arrays first: the file is read and the design
  matrix built with Polars, and that time is included (it is a few percent of
  theirs). logit-stream is given the path.
- **Peak memory:** the process's `VmHWM`, including what importing the
  libraries costs (about 0.5 GB for logit-stream, which imports hdfe-stream,
  pyfixest and pandas for the formula front end).
- **Peak disk:** the allocated size of the configuration's own work
  directory, polled; logit-stream also reports its own. The other libraries
  write nothing.

Every library is asked for the same unpenalized maximum-likelihood fit, at its
defaults otherwise (scikit-learn with `C=np.inf` and `tol=1e-8`, statsmodels'
Newton for the binary and multinomial logit and BFGS for `OrderedModel`, its
default). A configuration that fails, runs out of memory, or exceeds its time limit (30 minutes, 20
for Swissmetro) is not run at larger
sizes. The script compares a few coefficients
of each fit with logit-stream's at the same size (`max_abs_diff` in the CSVs).

```bash
python benchmarks/prepare_acs.py
python benchmarks/acs_benchmark.py
python benchmarks/choice_benchmark.py
python benchmarks/panel_benchmark.py
python benchmarks/mixed_benchmark.py check
python benchmarks/mixed_benchmark.py
python benchmarks/gpu_benchmark.py       # needs an NVIDIA GPU and CuPy
python benchmarks/plot_benchmarks.py     # the figures
python benchmarks/make_tables.py         # the tables of these pages
```

`benchmarks/ec2/` (not published) ran them on EC2.
