# Swissmetro

The conditional logit of [Swissmetro](https://transp-or.epfl.ch/documents/technicalReports/CS_SwissmetroDescription.pdf)
(Bierlaire, Axhausen and Abay, 2001), a stated-preference survey in which
1,192 Swiss travellers each chose among the train, their car and a proposed
maglev, the Swissmetro, in nine scenarios. It is the example data of Biogeme,
xlogit and jaxlogit. The model is Biogeme's first example:

$$
U_{\text{alt}} = \text{ASC}_{\text{alt}} + \beta_{\text{time}}\,\text{TT}_{\text{alt}}
+ \beta_{\text{cost}}\,\text{CO}_{\text{alt}},
$$

with constants for the train and the car (the Swissmetro the reference), on
the commuting and business trips with a known choice (6,768 choice sets), the
times and costs in hundreds of minutes and francs, the cost zero for holders
of an annual pass, and the car available only to car owners. logit-stream's
estimates on the survey are Biogeme's published ones: ASC_TRAIN −0.701,
ASC_CAR −0.155, B_TIME −1.278, B_COST −1.084, log-likelihood −5,331.252.

To see how the libraries scale, larger samples are drawn from the 6,768
choice sets with replacement, up to 10 million (28 million rows of
alternatives): the estimates stay those of the survey, up to sampling, while
the data grow. Each library gets the data as it takes them: logit-stream,
pylogit and statsmodels the long format with the available alternatives
only; xlogit and jaxlogit, which need every set to have the same
alternatives, all three with an availability indicator. jaxlogit fits mixed
logit only, and runs here with no random parameters, which is this model.

## Results

- **On the survey itself, xlogit is the fastest** (0.09 s against
  logit-stream's 1.1 s, most of which is fixed overhead); R's `clogit`,
  `mlogit` and Biogeme take 0.6 to 0.7 s. xlogit and logit-stream draw level
  at about 300,000 choice sets.
- **From a million choice sets, logit-stream is the smallest, and nearly the
  fastest.** At 10 million sets (28 million rows of alternatives) it took
  17 s and 1.7 GB, against 49 s and 10.2 GB for jaxlogit, 66 s and 7.6 GB for
  xlogit, 148 s and 20 GB for `clogit` and 388 s and 25 GB for `mlogit`.
  **Biogeme was faster, at 14 s**, on 9.7 GB. logit-stream's memory
  rose by 0.5 GB from 1 to 10 million sets; the others' grow in proportion
  to the data, by 0.7 to 2.5 GB per million sets.
- **jaxlogit** costs about 2 s of compilation up front. After that it
  scales a little better than xlogit, which it passes at 3 million sets.
  Both pad every choice set to all three alternatives, about 7% more rows
  here.
- **statsmodels' `ConditionalLogit`** is two orders of magnitude slower
  (8 minutes at a million sets) and timed out at 3 million.
- **pylogit** reproduces the survey's estimates, but needed 2.3 GB for it
  and ran out of memory at 30,000 sets. It runs on current Python and NumPy
  only with compatibility aliases, which the benchmark sets.
- **The nested logit** (the train and the car in a nest; Biogeme's example,
  the nesting parameter estimated) took logit-stream 33 s and 3.8 GB at 10
  million sets, against 119 s and 28 GB for Biogeme; `mlogit` took 223 s at
  3 million sets and ran out of memory at 10 million. On the survey itself
  `mlogit` and Biogeme are a little faster (0.9 s and 1.0 s against 1.2 s).

xlogit, jaxlogit, pylogit, Biogeme, `clogit` and `mlogit` agree with
logit-stream to within 1e-5 (`clogit`, 6e-13) at every size where they
finished; statsmodels to within 5e-4 (it stops on its own, looser criterion).
The nested logit's estimates agree to within 1e-4.

```{image} ../figures/choice_time.light.svg
:class: only-light benchmark
:alt: Swissmetro: wall time against the number of choice sets, one line per library, log-log
```
```{image} ../figures/choice_time.dark.svg
:class: only-dark benchmark
:alt: Swissmetro: wall time against the number of choice sets, one line per library, log-log
```

```{image} ../figures/choice_memory.light.svg
:class: only-light benchmark
:alt: Swissmetro: peak memory against the number of choice sets, one line per library, log-log
```
```{image} ../figures/choice_memory.dark.svg
:class: only-dark benchmark
:alt: Swissmetro: peak memory against the number of choice sets, one line per library, log-log
```

```{image} ../figures/choice_disk.light.svg
:class: only-light benchmark
:alt: Swissmetro: peak disk use of logit-stream against the number of choice sets, log-log
```
```{image} ../figures/choice_disk.dark.svg
:class: only-dark benchmark
:alt: Swissmetro: peak disk use of logit-stream against the number of choice sets, log-log
```

## Tables

The numbers behind the figures. A dash means the library was not run at that size, because it failed or timed out at a smaller one.

### Conditional logit

<!-- table: choice conditional -->

**Wall time (seconds)**

| choice sets | logit-stream | xlogit | pylogit | statsmodels | jaxlogit (no random parameters) | Biogeme | clogit (R survival) | mlogit (R) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6,768 | 1.1 | 0.09 | 0.71 | 3.1 | 1.8 | 0.70 | 0.57 | 0.71 |
| 30,000 | 1.1 | 0.21 | out of memory | 13.3 | 1.9 | 0.74 | 0.98 | 1.6 |
| 100,000 | 1.4 | 0.54 | – | 45.5 | 2.1 | 0.84 | 2.1 | 3.8 |
| 300,000 | 1.7 | 1.5 | – | 135 | 2.8 | 1.1 | 4.5 | 10.0 |
| 1,000,000 | 3.0 | 5.4 | – | 471 | 6.6 | 2.3 | 14.2 | 32.6 |
| 3,000,000 | 6.5 | 19.7 | – | timeout | 15.1 | 4.8 | 44.8 | 110 |
| 10,000,000 | 17.2 | 66.2 | – | out of memory | 49.1 | 14.2 | 148 | 388 |

**Peak memory (GB)**

| choice sets | logit-stream | xlogit | pylogit | statsmodels | jaxlogit (no random parameters) | Biogeme | clogit (R survival) | mlogit (R) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6,768 | 0.55 | 0.18 | 2.30 | 0.30 | 0.57 | 0.66 | 0.36 | 0.38 |
| 30,000 | 0.57 | 0.20 | out of memory | 0.39 | 0.61 | 0.74 | 0.40 | 0.45 |
| 100,000 | 0.69 | 0.25 | – | 0.64 | 0.73 | 0.83 | 0.55 | 0.68 |
| 300,000 | 0.91 | 0.41 | – | 1.38 | 1.08 | 1.07 | 0.99 | 1.19 |
| 1,000,000 | 1.18 | 0.94 | – | 3.94 | 1.67 | 1.71 | 2.18 | 3.10 |
| 3,000,000 | 1.54 | 2.42 | – | timeout | 3.95 | 3.95 | 5.65 | 7.58 |
| 10,000,000 | 1.71 | 7.64 | – | out of memory | 10.17 | 9.70 | 20.32 | 25.01 |

<!-- /table -->

### Nested logit

<!-- table: choice nested -->

**Wall time (seconds)**

| choice sets | logit-stream | Biogeme | mlogit (R) |
|---:|---:|---:|---:|
| 6,768 | 1.2 | 1.0 | 0.88 |
| 30,000 | 1.2 | 1.2 | 2.6 |
| 100,000 | 1.6 | 1.9 | 6.5 |
| 300,000 | 2.2 | 3.8 | 18.9 |
| 1,000,000 | 4.1 | 9.8 | 67.4 |
| 3,000,000 | 8.7 | 34.5 | 223 |
| 10,000,000 | 33.4 | 119 | out of memory |

**Peak memory (GB)**

| choice sets | logit-stream | Biogeme | mlogit (R) |
|---:|---:|---:|---:|
| 6,768 | 0.56 | 0.68 | 0.39 |
| 30,000 | 0.61 | 0.79 | 0.48 |
| 100,000 | 0.75 | 0.98 | 0.78 |
| 300,000 | 1.00 | 1.50 | 1.35 |
| 1,000,000 | 1.44 | 3.33 | 3.85 |
| 3,000,000 | 1.61 | 8.87 | 10.54 |
| 10,000,000 | 3.78 | 27.92 | out of memory |

<!-- /table -->
