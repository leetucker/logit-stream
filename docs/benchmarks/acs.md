# ACS microdata

Three models of a demographic imputation, fitted to the adults of the
[American Community Survey](https://www.census.gov/programs-surveys/acs/microdata.html)
2019–2023 5-year Public Use Microdata Sample, from a random 10,000 of them to
all 12,853,377:

| model | outcome | coefficients |
|---|---|---|
| binary | Hispanic or Latino origin (13.5%) | 41 |
| multinomial | race and ethnicity in five groups: White (base), Black, Hispanic, Asian or Pacific Islander, other (incl. American Indian, multiracial) | 164 |
| ordered | educational attainment in five levels, adults 25 and over | 40 + 4 cutpoints |

on the same covariates: age group (7) and its interaction with sex,
personal income group (7, in 2023 dollars), citizenship and nativity (4),
English ability (4), marital status (5), disability, and Census division (9).
`benchmarks/prepare_acs.py` builds the data from the Census Bureau's files;
the rows are in a random order, so the first n are a random sample of n.

Two notes on the specification. The division, not the state, is the
geography, and two rare categories are merged (the territory-born with the
US-born, "not well" with "not at all" for English): at 10,000 rows, states and
rare categories leave some cells without anyone of some race, which separates
their coefficients -- their maximum-likelihood estimates are infinite -- and
turns the comparison into one of how each library fails. And logit-stream is
also run with the 2,462 PUMAs absorbed as a fixed effect in place of the
divisions (dashed in the figures): a richer model, which the in-memory
libraries could fit only with 2,461 indicator columns per equation. And it is
run with random PUMA intercepts in their place (dash-dotted;
*experimental*, see [random intercepts](../examples/fixed_effects.md)): the
same richer model with the PUMA effects drawn from a normal distribution
whose covariance across the outcome levels is estimated, by Laplace's
approximation (the default; the tables also have it within fixed state
effects). On all 12.9 million adults it took 334 s for the multinomial and
133 s for the ordered logit, 5.5 and 2.2 times the fixed-effect fits, with
about the same memory: each read also gathers the approximation's
third-derivative terms, 64 per PUMA for the multinomial's four equations.
The joint-mode approximation (`random_method="joint-mode"`) took 134 s and
107 s there, but even with ~5,000 adults per PUMA its multinomial
coefficients were up to 0.41 standard errors from Laplace's (the ordered
logit's, 0.06; the standard deviations agreed to 0.3%). The small samples
are the slow ones for the multinomial: at 10,000 adults a PUMA has about
four, so the effects' covariance is weakly determined and its updates
converge slowly (106 s, 504 reads; see the design notes).

The multinomial logit is also fitted with the discrete-choice libraries
xlogit and jaxlogit, which take long-format data: one row per person and
outcome level. xlogit is given the 40 covariates as individual-specific
variables (`isvars`) and expands them per level itself. jaxlogit has no
individual-specific variables, so the benchmark expands the design for it,
164 columns per row, the same array xlogit builds internally; the
expansion is timed with the reading. jaxlogit fits mixed logit only, and
runs here with no random parameters, which is this model. xlogit runs with
standard errors from the outer product of the gradients (`num_hess=False`):
its default numerical Hessian makes K(K+1) = 27,060 passes over the data
for these 164 coefficients, and took 239 s at 10,000 rows in all.

Where the libraries finished, their coefficients agree with logit-stream's to
within 1e-5 (scikit-learn's L-BFGS, which stops on its own gradient
tolerance), 1e-7 (its Newton–Cholesky) and 1e-8 (statsmodels' Newton). For
the ordered logit, statsmodels' `OrderedModel` agrees to within 4e-4: its
default optimizer, BFGS, stops before Newton's method would. mord's
`LogisticAT` differs by 0.02 to 0.06 at every size. That isn't a convergence
failure: mord minimizes a different loss, the "all-threshold" surrogate of
Rennie and Srebro (2005), not the ordered-logit likelihood, so it estimates a
different model with the same parameters. It's included because it's the
scikit-learn-style ordinal regression people use. The ordered logit's main
effect of sex is not compared: in the 25-and-over sample the formula's
reference age group (18–24) is absent, so logit-stream's reference is a
different age group from the other libraries'; the fits are the same. The
numbers are in
[benchmarks/results/acs.csv](https://github.com/leetucker/logit-stream/blob/main/benchmarks/results/acs.csv).

```{image} ../figures/acs_time.light.svg
:class: only-light benchmark
:alt: ACS microdata: wall time against the number of adults, one panel per model (binary, multinomial, ordered), one line per configuration, log-log
```
```{image} ../figures/acs_time.dark.svg
:class: only-dark benchmark
:alt: ACS microdata: wall time against the number of adults, one panel per model (binary, multinomial, ordered), one line per configuration, log-log
```

```{image} ../figures/acs_memory.light.svg
:class: only-light benchmark
:alt: ACS microdata: peak memory against the number of adults, one panel per model, one line per configuration, log-log
```
```{image} ../figures/acs_memory.dark.svg
:class: only-dark benchmark
:alt: ACS microdata: peak memory against the number of adults, one panel per model, one line per configuration, log-log
```

```{image} ../figures/acs_disk.light.svg
:class: only-light benchmark
:alt: ACS microdata: peak disk use of logit-stream against the number of adults, one panel per model, log-log
```
```{image} ../figures/acs_disk.dark.svg
:class: only-dark benchmark
:alt: ACS microdata: peak disk use of logit-stream against the number of adults, one panel per model, log-log
```

The discrete-choice libraries on the multinomial logit are in their own
figure, so that every color in a figure can be told from every other.
Long-format data cost them one row per person and level, and the
expanded design 164 columns per row: at a million adults xlogit took 77 s
and 18.5 GB and jaxlogit 417 s and 25.3 GB (nearly all of this machine's
memory), against logit-stream's 5.6 s and 1.1 GB, and both ran out of
memory at 3 million. At 10,000 adults xlogit is faster (1.0 s against
1.8 s). xlogit agrees with logit-stream to within 1e-5; jaxlogit to within
7e-4, its L-BFGS stopping on its own gradient tolerance after 130 to 165
iterations.

```{image} ../figures/acs_choice_time.light.svg
:class: only-light benchmark
:alt: ACS microdata, multinomial logit: wall time of logit-stream, xlogit and jaxlogit against the number of adults, log-log
```
```{image} ../figures/acs_choice_time.dark.svg
:class: only-dark benchmark
:alt: ACS microdata, multinomial logit: wall time of logit-stream, xlogit and jaxlogit against the number of adults, log-log
```

```{image} ../figures/acs_choice_memory.light.svg
:class: only-light benchmark
:alt: ACS microdata, multinomial logit: peak memory of logit-stream, xlogit and jaxlogit against the number of adults, log-log
```
```{image} ../figures/acs_choice_memory.dark.svg
:class: only-dark benchmark
:alt: ACS microdata, multinomial logit: peak memory of logit-stream, xlogit and jaxlogit against the number of adults, log-log
```

## Tables

The numbers behind the figures. A dash means the configuration was not run at that size, because it failed or timed out at a smaller one.

### Binary logit

**Wall time (seconds)**

| n | logit-stream | logit-stream (PUMA fixed effects) | scikit-learn (lbfgs) | scikit-learn (newton-cholesky) | statsmodels |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 1.4 | 1.4 | 0.4 | 0.1 | 0.3 |
| 30,000 | 1.7 | 1.5 | 6.2 | 0.2 | 0.3 |
| 100,000 | 2.1 | 1.6 | 6.5 | 0.4 | 0.6 |
| 300,000 | 3.2 | 2.4 | 6.0 | 0.8 | 1.1 |
| 1,000,000 | 7.0 | 5.4 | 9.2 | 2.2 | 5.5 |
| 3,000,000 | 17.1 | 13.9 | 19.1 | 9.8 | 14.3 |
| 6,000,000 | 32.4 | 24.3 | 36.9 | 21.9 | 26.7 |
| 12,853,377 | 63.9 | 50.6 | 74.5 | 45.5 | 56.0 |

**Peak memory (GB)**

| n | logit-stream | logit-stream (PUMA fixed effects) | scikit-learn (lbfgs) | scikit-learn (newton-cholesky) | statsmodels |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.58 | 0.56 | 0.36 | 0.36 | 0.38 |
| 30,000 | 0.64 | 0.65 | 0.41 | 0.39 | 0.41 |
| 100,000 | 0.82 | 0.84 | 0.48 | 0.49 | 0.48 |
| 300,000 | 1.20 | 1.21 | 0.67 | 0.75 | 0.70 |
| 1,000,000 | 2.07 | 2.15 | 1.33 | 1.62 | 1.33 |
| 3,000,000 | 3.79 | 3.75 | 3.01 | 3.55 | 3.03 |
| 6,000,000 | 4.20 | 4.99 | 5.77 | 6.02 | 5.71 |
| 12,853,377 | 4.33 | 7.16 | 10.67 | 12.19 | 8.35 |

### Multinomial logit

**Wall time (seconds)**

| n | logit-stream | logit-stream (low memory) | logit-stream (PUMA fixed effects) | logit-stream (PUMA random intercepts) | logit-stream (state fixed + PUMA random) | scikit-learn (lbfgs) | scikit-learn (newton-cholesky) | statsmodels | xlogit (OPG standard errors) | jaxlogit (no random parameters) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 1.8 | 1.8 | 1.9 | 106 | 98.4 | 15.9 | 0.8 | 0.5 | 1.0 | 7.7 |
| 30,000 | 1.8 | 2.1 | 1.9 | 30.8 | 30.1 | 17.8 | 1.3 | 1.0 | 2.4 | 11.8 |
| 100,000 | 1.8 | 2.9 | 2.0 | 20.1 | 20.1 | 23.6 | 3.1 | 5.0 | 6.7 | 37.4 |
| 300,000 | 2.6 | 5.2 | 2.8 | 27.3 | 31.4 | 32.4 | 7.4 | 13.3 | 21.8 | 103 |
| 1,000,000 | 5.6 | 5.5 | 6.5 | 48.3 | 56.0 | 61.6 | 25.0 | 37.3 | 77.3 | 417 |
| 3,000,000 | 13.0 | 16.7 | 16.4 | 102 | 94.6 | 134 | 106 | 152 | out of memory | out of memory |
| 6,000,000 | 17.2 | 36.3 | 26.5 | 191 | 184 | 249 | 173 | 214 | – | – |
| 12,853,377 | 70.1 | 94.1 | 61.0 | 334 | 367 | 463 | 351 | 502 | – | – |

**Peak memory (GB)**

| n | logit-stream | logit-stream (low memory) | logit-stream (PUMA fixed effects) | logit-stream (PUMA random intercepts) | logit-stream (state fixed + PUMA random) | scikit-learn (lbfgs) | scikit-learn (newton-cholesky) | statsmodels | xlogit (OPG standard errors) | jaxlogit (no random parameters) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.57 | 0.57 | 0.62 | 0.71 | 0.69 | 0.38 | 0.39 | 0.41 | 0.44 | 0.97 |
| 30,000 | 0.62 | 0.61 | 0.66 | 0.76 | 0.76 | 0.42 | 0.41 | 0.45 | 0.82 | 1.66 |
| 100,000 | 0.71 | 0.66 | 0.81 | 0.87 | 0.87 | 0.54 | 0.55 | 0.56 | 2.15 | 3.61 |
| 300,000 | 0.96 | 0.75 | 0.99 | 1.05 | 1.07 | 0.76 | 0.86 | 0.83 | 5.76 | 9.46 |
| 1,000,000 | 1.12 | 0.76 | 1.25 | 1.47 | 1.47 | 1.50 | 1.76 | 1.41 | 18.51 | 25.26 |
| 3,000,000 | 1.06 | 0.74 | 1.13 | 1.23 | 1.25 | 3.47 | 3.56 | 3.04 | out of memory | out of memory |
| 6,000,000 | 1.06 | 0.78 | 1.10 | 1.28 | 1.26 | 5.86 | 6.57 | 5.86 | – | – |
| 12,853,377 | 1.02 | 0.79 | 1.10 | 1.24 | 1.23 | 9.72 | 13.51 | 9.46 | – | – |
### Ordered logit

**Wall time (seconds)**

| n | logit-stream | logit-stream (PUMA fixed effects) | logit-stream (PUMA random intercepts) | logit-stream (state fixed + PUMA random) | statsmodels | mord (all-threshold loss) |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 1.0 | 1.1 | 1.7 | 1.6 | 5.2 | 0.6 |
| 30,000 | 1.1 | 1.3 | 2.1 | 2.1 | 37.3 | 7.1 |
| 100,000 | 1.5 | 1.5 | 3.3 | 3.5 | 93.3 | 9.8 |
| 300,000 | 2.2 | 2.6 | 6.3 | 6.1 | 366 | 15.7 |
| 1,000,000 | 5.1 | 5.5 | 15.7 | 15.0 | 1,008 | 28.8 |
| 3,000,000 | 12.8 | 15.0 | 31.1 | 29.8 | timeout | 82.1 |
| 6,000,000 | 22.6 | 29.4 | 64.5 | 59.2 | – | 157 |
| 12,853,377 | 45.5 | 60.4 | 133 | 132 | – | 316 |

**Peak memory (GB)**

| n | logit-stream | logit-stream (PUMA fixed effects) | logit-stream (PUMA random intercepts) | logit-stream (state fixed + PUMA random) | statsmodels | mord (all-threshold loss) |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.54 | 0.54 | 0.54 | 0.54 | 0.36 | 0.36 |
| 30,000 | 0.58 | 0.59 | 0.59 | 0.60 | 0.39 | 0.40 |
| 100,000 | 0.73 | 0.74 | 0.76 | 0.77 | 0.49 | 0.45 |
| 300,000 | 0.99 | 1.02 | 1.05 | 1.07 | 0.72 | 0.63 |
| 1,000,000 | 1.16 | 1.22 | 1.26 | 1.18 | 1.43 | 1.16 |
| 3,000,000 | 1.06 | 1.15 | 1.19 | 1.22 | timeout | 2.41 |
| 6,000,000 | 1.04 | 1.12 | 1.12 | 1.16 | – | 4.35 |
| 12,853,377 | 1.02 | 1.04 | 1.19 | 1.17 | – | 7.97 |