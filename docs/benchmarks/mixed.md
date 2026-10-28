# Mixed logit (experimental)

`mixlogit_stream` is a prototype (see the
[reference](../reference/estimation.md)); this benchmark checks it against
the exact estimates and against xlogit and jaxlogit, which are built for
the mixed logit.

The model is the conditional logit of the [Swissmetro benchmark](choice.md)
with a normally distributed coefficient of travel time, and the survey's
panel structure: each respondent's nine choices share one draw of the
coefficient (xlogit's Swissmetro example).

$$
U_{n,\text{alt}} = \text{ASC}_{\text{alt}} + \beta_{\text{time},n}\,\text{TT}_{\text{alt}}
+ \beta_{\text{cost}}\,\text{CO}_{\text{alt}}, \qquad
\beta_{\text{time},n} \sim N(\mu, \sigma^2)
$$

## Accuracy

With one random coefficient, a respondent's likelihood is a one-dimensional
integral, which quadrature computes exactly. Gauss–Hermite quadrature does
not converge here: with σ near 3.6 and trips of several hundred minutes,
the likelihood is nearly a step function of the draw. Gauss–Legendre
quadrature on [−9, 9] with 2,000 nodes is converged to 1e-6. Maximizing that
exact likelihood gives ASC_TRAIN −0.575, ASC_CAR 0.282, TT −3.222, CO −1.660,
σ 3.652, log-likelihood −4,359.413.

The three libraries use the same Halton draws (consecutive points of the
base-2 sequence for each respondent, the first 100 discarded), so at the
same number of draws they maximize the same simulated likelihood:

| draws | logit-stream | xlogit | jaxlogit | largest difference from the exact estimates |
|---:|---:|---:|---:|---:|
| 100 | −4,359.763 | −4,360.227 | −4,359.096 | 0.12, 0.12, 0.02 |
| 500 | −4,360.183 | −4,360.183 | −4,360.183 | 0.008 |
| 2,000 | −4,359.894 | −4,359.894 | −4,359.894 | 0.012 |

The differences from the exact estimates are simulation error, far below
the standard errors (0.1 to 0.2). With 100 draws the simulated likelihood
has several local maxima, and the libraries stop at different ones.

Two settings differ from the libraries' defaults. xlogit runs with its
L-BFGS-B optimizer: its default BFGS stops after two iterations on this
model ("local search could not find a higher log likelihood value"), at a
log-likelihood of −5,058. jaxlogit estimates a standard deviation through a
softplus and reports the parameter before the transformation; the benchmark
reports the standard deviation itself.

## Scaling

Larger samples are drawn from the 752 respondents with replacement, whole
respondents at a time, up to a million respondents (27 million rows of
alternatives), with 500 draws per respondent.

- **logit-stream was the fastest at every size**, the survey itself
  included: 2.9 s at 752 respondents, against 5.7 s for jaxlogit and 7.6 s
  for xlogit; at 10,000, 14 s against 42 s and 108 s.
- **Its memory stayed between 0.6 and 2.2 GB** up to a million respondents.
  jaxlogit's grew by about 2 GB per thousand respondents and ran out of
  memory at 30,000; xlogit's by about 0.55 GB per thousand, and it ran out
  at 100,000.
- **The time is linear in the data**, about 1.5 ms per respondent at 500
  draws: 25 minutes for a million respondents. Unlike the other models,
  the mixed logit is limited by arithmetic, not by reading the data, so
  the time also grows in proportion to the number of draws.
- The libraries agree with logit-stream to within 2e-5 at every size where
  they finished.

logit-stream's optimizer takes Newton steps with a finite-difference
Hessian, then quasi-Newton (BFGS) steps, confirming convergence with the
Hessian itself: 15 to 18 steps here, of which one or two are Newton steps.


```{image} ../figures/mixed_time.light.svg
:class: only-light benchmark
:alt: Mixed logit on the Swissmetro panel: wall time against the number of respondents, one line per library, log-log
```
```{image} ../figures/mixed_time.dark.svg
:class: only-dark benchmark
:alt: Mixed logit on the Swissmetro panel: wall time against the number of respondents, one line per library, log-log
```

```{image} ../figures/mixed_memory.light.svg
:class: only-light benchmark
:alt: Mixed logit on the Swissmetro panel: peak memory against the number of respondents, one line per library, log-log
```
```{image} ../figures/mixed_memory.dark.svg
:class: only-dark benchmark
:alt: Mixed logit on the Swissmetro panel: peak memory against the number of respondents, one line per library, log-log
```

## Tables

A dash means the library was not run at that size, because it ran out of memory at a smaller one.

**Wall time (seconds)**

| respondents | logit-stream | xlogit (L-BFGS-B) | jaxlogit |
|---:|---:|---:|---:|
| 752 | 2.9 | 7.6 | 5.7 |
| 3,000 | 6.1 | 31.5 | 9.8 |
| 10,000 | 14.0 | 108 | 42.5 |
| 30,000 | 60.6 | 449 | out of memory |
| 100,000 | 142 | out of memory | – |
| 300,000 | 439 | – | – |
| 1,000,000 | 1,476 | – | – |

**Peak memory (GB)**

| respondents | logit-stream | xlogit (L-BFGS-B) | jaxlogit |
|---:|---:|---:|---:|
| 752 | 0.58 | 0.68 | 2.90 |
| 3,000 | 0.62 | 1.85 | 6.48 |
| 10,000 | 0.74 | 5.56 | 19.62 |
| 30,000 | 0.97 | 16.13 | out of memory |
| 100,000 | 1.32 | out of memory | – |
| 300,000 | 1.46 | – | – |
| 1,000,000 | 2.16 | – | – |

