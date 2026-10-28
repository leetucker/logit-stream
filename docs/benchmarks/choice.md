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

- **On the survey itself, xlogit is the fastest** (0.1 s against
  logit-stream's 1 s, most of which is fixed overhead). The two draw level at
  300,000 choice sets.
- **From a million choice sets, logit-stream is the fastest and the
  smallest.** At 10 million sets (28 million rows of alternatives) it took
  24 s and 1.8 GB, against 54 s and 10.3 GB for jaxlogit and 66 s and 7.8 GB
  for xlogit. Its memory rose by 0.6 GB from 1 to 10 million sets; xlogit's
  and jaxlogit's grow in proportion to the data, by about 0.75 and 1 GB per
  million sets.
- **jaxlogit** costs about 2 s of compilation up front. After that it
  scales a little better than xlogit, which it passes at 3 million sets.
  Both pad every choice set to all three alternatives, about 7% more rows
  here.
- **statsmodels' `ConditionalLogit`** is two orders of magnitude slower
  (18 minutes at 3 million sets) and ran out of memory at 10 million.
- **pylogit** reproduces the survey's estimates, but needed 2.3 GB for it
  and ran out of memory at 30,000 sets. It runs on current Python and NumPy
  only with compatibility aliases, which the benchmark sets.

xlogit, jaxlogit and pylogit agree with logit-stream to within 5e-6 at every
size; statsmodels to within 5e-4 (it stops on its own, looser criterion).


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

A dash means the library was not run at that size, because it failed at a smaller one.

**Wall time (seconds)**

| choice sets | logit-stream | xlogit | pylogit | statsmodels | jaxlogit (no random parameters) |
|---:|---:|---:|---:|---:|---:|
| 6,768 | 1.0 | 0.1 | 0.7 | 2.5 | 2.3 |
| 30,000 | 1.0 | 0.2 | out of memory | 11.0 | 2.0 |
| 100,000 | 1.2 | 0.5 | – | 36.3 | 2.4 |
| 300,000 | 1.7 | 1.6 | – | 107 | 3.0 |
| 1,000,000 | 3.0 | 5.5 | – | 354 | 5.7 |
| 3,000,000 | 7.3 | 18.3 | – | 1,069 | 13.6 |
| 10,000,000 | 24.3 | 65.9 | – | out of memory | 53.6 |

**Peak memory (GB)**

| choice sets | logit-stream | xlogit | pylogit | statsmodels | jaxlogit (no random parameters) |
|---:|---:|---:|---:|---:|---:|
| 6,768 | 0.53 | 0.25 | 2.30 | 0.35 | 0.60 |
| 30,000 | 0.55 | 0.27 | out of memory | 0.43 | 0.64 |
| 100,000 | 0.70 | 0.33 | – | 0.70 | 0.77 |
| 300,000 | 0.92 | 0.50 | – | 1.47 | 1.06 |
| 1,000,000 | 1.22 | 1.04 | – | 4.09 | 2.06 |
| 3,000,000 | 1.42 | 2.55 | – | 11.55 | 4.19 |
| 10,000,000 | 1.83 | 7.79 | – | out of memory | 10.32 |

