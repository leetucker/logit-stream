# GPU (experimental)

`device="cuda"` forms each Newton step's sums on an NVIDIA GPU through CuPy
(see [on a GPU](../how_it_works.md#on-a-gpu)). This page compares it with the
CPU on the models of the other benchmarks, at their largest sizes, and with
xlogit and jaxlogit, which have GPU code of their own for the mixed logit
(jaxlogit also for the others; xlogit's conditional logit has none, so it is
run on the CPU only).

Everything ran on one machine: an AWS p5.4xlarge with an NVIDIA H100 (80 GB),
an AMD EPYC 7R13 with 16 vCPUs, the process capped at 29.6 GB of host memory,
CUDA 13.2, CuPy 14.2. The CPU fits are on the same machine, so each pair is
like for like; they are not the CPU benchmarks' (an Intel Xeon with a
different clock), whose times are somewhat shorter (the multinomial logit of
all 12.9 million adults: 54 s here, 37 s there). Every fit was preceded by a
small fit of the same model in its process, so that numba's and CuPy's
compilation is not timed (JAX compiles again for the new shapes, which is
timed, as a user would see it). The variance is the robust one (clustered by
chooser for the mixed logit and by PUMA for the random intercepts), whose
scores pass runs on the device too.

## Results

- **The multinomial and ordered logit** took 6.5 s and 5.4 s for all 12.9
  million adults on the GPU, against 54 s and 44 s on the CPU, about 8
  times faster; with 2,462 PUMA fixed effects, 10 times (6.8 s against 69 s,
  5.5 s against 57 s). The random-intercept fits gain most: 3 million adults
  took 6.9 s against 91 s (multinomial, 13 times) and 2.3 s against 28 s
  (ordered, 12 times).
- **The conditional logit** of 10 million Swissmetro choice sets took 4.1 s
  against 27 s (6.5 times); jaxlogit's fits took 74 s on the CPU and 71 s on
  the GPU, and xlogit's 90 s on the CPU. Chamberlain's logit on a simulated
  panel of 10 million rows gained less, 3.6 s against 8.9 s.
- **The nested logit** gains little (32 s against 38 s at 10 million sets):
  21 s of either is outside the Newton steps, in pass 0 and the scores pass,
  which the device does not speed up.
- **The mixed logit** (500 draws) is 2.3 times faster on the GPU: 434 s
  against 987 s for a million respondents, 41 s against 97 s for 100,000.
  xlogit on the GPU took 20 s at 10,000 respondents and 282 s at 100,000, and
  timed out (10 minutes) at a million; jaxlogit ran out of memory on the GPU
  at 100,000 respondents. The CPU versions of both ran out of host memory at
  100,000.
- **Memory:** the blocks are copied to the device once, in the compressed form
  (0.5 GB for all 12.9 million adults, 0.8 GB for 10 million nested choice
  sets); host memory is 0.5 to 0.6 GB higher than on the CPU.
- **The estimates agree** with the CPU's to within 4e-13 (the nested logit,
  9e-8), and xlogit's and jaxlogit's to within 2e-5.

## Table

Wall time (seconds). A dash means the library was not run for that model
(or, for xlogit's conditional logit on a GPU, that it has no GPU code).

<!-- table: gpu -->

| model | rows | logit-stream CPU | logit-stream GPU | xlogit CPU | xlogit GPU | jaxlogit CPU | jaxlogit GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| conditional logit (Swissmetro, choice sets) | 1,000,000 | 3.1 | 0.51 | 8.0 | – | 7.6 | 9.3 |
| conditional logit (Swissmetro, choice sets) | 10,000,000 | 26.6 | 4.1 | 90.0 | – | 73.8 | 70.9 |
| Chamberlain's conditional logit (simulated panel, rows) | 1,000,000 | 0.90 | 0.38 | – | – | – | – |
| Chamberlain's conditional logit (simulated panel, rows) | 10,000,000 | 8.9 | 3.6 | – | – | – | – |
| nested logit | 1,000,000 | 4.0 | 3.7 | – | – | – | – |
| nested logit | 10,000,000 | 38.3 | 32.0 | – | – | – | – |
| multinomial logit | 1,000,000 | 4.6 | 0.81 | – | – | – | – |
| multinomial logit | 12,853,377 | 54.1 | 6.5 | – | – | – | – |
| multinomial, PUMA fixed effects | 1,000,000 | 5.8 | 0.93 | – | – | – | – |
| multinomial, PUMA fixed effects | 12,853,377 | 68.6 | 6.8 | – | – | – | – |
| multinomial, PUMA random intercepts | 1,000,000 | 45.4 | 6.0 | – | – | – | – |
| multinomial, PUMA random intercepts | 3,000,000 | 90.9 | 6.9 | – | – | – | – |
| ordered logit | 1,000,000 | 3.7 | 0.61 | – | – | – | – |
| ordered logit | 12,853,377 | 44.3 | 5.4 | – | – | – | – |
| ordered, PUMA fixed effects | 1,000,000 | 5.1 | 0.96 | – | – | – | – |
| ordered, PUMA fixed effects | 12,853,377 | 57.3 | 5.5 | – | – | – | – |
| ordered, PUMA random intercepts | 1,000,000 | 13.5 | 1.2 | – | – | – | – |
| ordered, PUMA random intercepts | 3,000,000 | 28.1 | 2.3 | – | – | – | – |
| mixed logit (respondents) | 10,000 | 9.2 | 4.2 | 93.4 | 19.7 | 42.2 | 11.9 |
| mixed logit (respondents) | 100,000 | 96.9 | 40.6 | out of memory | 282 | out of memory | out of memory |
| mixed logit (respondents) | 1,000,000 | 987 | 434 | – | timeout | – | – |

<!-- /table -->
