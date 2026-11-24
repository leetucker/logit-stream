# Panels

Three ways to fit the logit of a short panel, as the panel grows from 100,000
to 10 million rows (20,000 to 2 million units of five periods):

- **Chamberlain's conditional logit** (`cfelogit_stream`), which conditions
  the unit effects away; compared with statsmodels' `ConditionalLogit` and R's
  `survival::clogit`.
- **The fixed-effects logit**, the unit effects estimated
  (`logit_stream("y ~ x1 + x2 | unit", ...)`); compared with R's `fixest` and pyfixest.
- **The same with the analytic incidental-parameter bias correction**
  (`bias_correction=True`); compared with R's `alpaca`, which it reproduces.

The data are `logit_stream.simulate.simulate_panel`: P(y = 1) is the logistic
of a unit effect plus x1 − 0.5·x2, the unit effects correlated with x1. Each
library gets the same model with its default standard errors (fixest asked
for iid ones, its default with fixed effects being clustered). With five
periods the fixed-effects estimates are biased, by about 30% here; the
conditional and the bias-corrected ones are not.

## Results

At 10 million rows (2 million units):

| model | logit-stream | others |
|---|---|---|
| conditional | 3.1 s, 1.2 GB | `clogit` 55 s, 5.4 GB; statsmodels 590 s, 6.2 GB |
| fixed effects | 4.0 s, 1.4 GB | fixest 8.1 s, 3.0 GB; pyfixest 9.2 s, 3.9 GB |
| bias-corrected | 7.4 s, 1.4 GB | alpaca 67 s, 4.0 GB |

- **The conditional logit** is faster than `clogit` at every size (0.59 s
  against 1.1 s at 100,000 rows, 0.84 s against 5.8 s at a million), and
  statsmodels', which takes ten minutes at 10 million rows, is 190 times
  slower.
- **The fixed-effects logit:** fixest is faster at 100,000 rows (0.43 s
  against 0.62 s), logit-stream from about a million (0.94 s against 1.1 s at
  1 million, 1.6 s against 2.5 s at 3 million), with less than half of the
  memory at 10 million. The
  fixed effects are absorbed, so memory does not grow with the number of
  units. pyfixest takes the same estimates as fixest, to within 6e-8 of
  logit-stream's, and is slower than fixest and than logit-stream at every size.
- **The bias correction** costs logit-stream about 85% more than the
  uncorrected fit at 10 million rows. alpaca is about as fast at 100,000 rows
  (0.70 s against 0.68 s) and 9 times slower at 10 million. The
  estimates agree to within 3e-4.
- Memory grows slowly in logit-stream (0.6 to 1.4 GB over the range) and
  its disk use, 3 to 550 MB, is what it writes of the rows it re-reads.
- The other libraries' coefficients agree with logit-stream's to within 6e-8
  (fixest, pyfixest), 3e-4 (alpaca), 5e-15 (`clogit`) and 2e-4 (statsmodels).

```{image} ../figures/panel_time.light.svg
:class: only-light benchmark
:alt: Panels: wall time against the number of rows, one panel per model (conditional, fixed-effects and bias-corrected logit), one line per library, log-log
```
```{image} ../figures/panel_time.dark.svg
:class: only-dark benchmark
:alt: Panels: wall time against the number of rows, one panel per model (conditional, fixed-effects and bias-corrected logit), one line per library, log-log
```

```{image} ../figures/panel_memory.light.svg
:class: only-light benchmark
:alt: Panels: peak memory against the number of rows, one panel per model, one line per library, log-log
```
```{image} ../figures/panel_memory.dark.svg
:class: only-dark benchmark
:alt: Panels: peak memory against the number of rows, one panel per model, one line per library, log-log
```

```{image} ../figures/panel_disk.light.svg
:class: only-light benchmark
:alt: Panels: peak disk use of logit-stream against the number of rows, one panel per model, log-log
```
```{image} ../figures/panel_disk.dark.svg
:class: only-dark benchmark
:alt: Panels: peak disk use of logit-stream against the number of rows, one panel per model, log-log
```

## Tables

The numbers behind the figures.

### Conditional logit

<!-- table: panel conditional -->

**Wall time (seconds)**

| rows | logit-stream | statsmodels | clogit (R survival) |
|---:|---:|---:|---:|
| 100,000 | 0.59 | 5.9 | 1.1 |
| 1,000,000 | 0.84 | 63.3 | 5.8 |
| 3,000,000 | 1.3 | 176 | 14.7 |
| 10,000,000 | 3.1 | 590 | 54.7 |

**Peak memory (GB)**

| rows | logit-stream | statsmodels | clogit (R survival) |
|---:|---:|---:|---:|
| 100,000 | 0.54 | 0.27 | 0.36 |
| 1,000,000 | 0.84 | 0.82 | 0.83 |
| 3,000,000 | 1.19 | 2.02 | 2.11 |
| 10,000,000 | 1.23 | 6.25 | 5.41 |

<!-- /table -->

### Fixed-effects logit

<!-- table: panel fixed effects -->

**Wall time (seconds)**

| rows | logit-stream | fixest (R) | pyfixest |
|---:|---:|---:|---:|
| 100,000 | 0.62 | 0.43 | 1.0 |
| 1,000,000 | 0.94 | 1.1 | 1.6 |
| 3,000,000 | 1.6 | 2.5 | 3.1 |
| 10,000,000 | 4.0 | 8.1 | 9.2 |

**Peak memory (GB)**

| rows | logit-stream | fixest (R) | pyfixest |
|---:|---:|---:|---:|
| 100,000 | 0.55 | 0.27 | 0.36 |
| 1,000,000 | 0.89 | 0.57 | 0.80 |
| 3,000,000 | 1.21 | 1.20 | 1.47 |
| 10,000,000 | 1.41 | 3.03 | 3.89 |

<!-- /table -->

### Bias-corrected

<!-- table: panel bias-corrected -->

**Wall time (seconds)**

| rows | logit-stream | alpaca (R) |
|---:|---:|---:|
| 100,000 | 0.68 | 0.70 |
| 1,000,000 | 1.3 | 5.6 |
| 3,000,000 | 2.6 | 16.1 |
| 10,000,000 | 7.4 | 66.7 |

**Peak memory (GB)**

| rows | logit-stream | alpaca (R) |
|---:|---:|---:|
| 100,000 | 0.61 | 0.27 |
| 1,000,000 | 1.01 | 0.68 |
| 3,000,000 | 1.19 | 1.35 |
| 10,000,000 | 1.43 | 3.98 |

<!-- /table -->
