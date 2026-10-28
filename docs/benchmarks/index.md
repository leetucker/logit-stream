# Benchmarks

Three benchmarks compare logit-stream with the in-memory Python libraries, on
real data, as the sample grows:

| benchmark | models | data | compared with |
|---|---|---|---|
| [ACS microdata](acs.md) | binary, multinomial and ordered logit of three demographic characteristics on 41 covariates | the adults of the 2019–2023 ACS 5-year PUMS, 10,000 to 12.9 million | scikit-learn (L-BFGS and Newton–Cholesky), statsmodels, mord; for the multinomial logit also xlogit and jaxlogit |
| [Swissmetro](choice.md) | conditional logit of mode choice (Biogeme's first example) | the Swissmetro survey's 6,768 choice sets, resampled to 10 million | xlogit, pylogit, statsmodels, jaxlogit |
| [Mixed logit](mixed.md) (experimental) | mixed logit with a normal coefficient of travel time, panel by respondent | the Swissmetro survey's 752 respondents, resampled to 1 million (27 million rows) | xlogit, jaxlogit; and the exact likelihood |

All were run on one machine: 8 logical CPUs, 26 GB of memory, Linux under
WSL2 (`benchmarks/results/machine.json` has the details and versions). They
have not yet been run on a larger machine. Each point is a single run, and on
this machine the same fit can take up to about 40% longer from one run to
the next (logit-stream's multinomial logit at 6 million rows took 17 s in the
benchmark and 29 s when repeated). Read the differences between libraries,
which are several-fold, rather than the kinks in a line.

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
default). A configuration that fails, runs out of memory, or takes longer than
20 minutes is not run at larger sizes. The script compares a few coefficients
of each fit with logit-stream's at the same size (`max_abs_diff` in the CSVs).

```bash
python benchmarks/prepare_acs.py
python benchmarks/acs_benchmark.py
python benchmarks/choice_benchmark.py
python benchmarks/mixed_benchmark.py check
python benchmarks/mixed_benchmark.py
python benchmarks/plot_benchmarks.py
```
