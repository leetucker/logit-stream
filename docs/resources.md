# Resource limits and profiling

(quickstart)=
## Quickstart for cluster jobs

Two habits cover most of it.

**1. Profile your fits before you run them.** `profile=True` reads the data once and returns
the estimated memory and disk instead of fitting, so you can request the right resources:

```python
from logit_stream import mlogit_stream

p = mlogit_stream("occupation ~ age + C(sex)", "data/*.parquet", profile=True)
print(p.recommend())      # ncpus=8  mem=14gb  disk=40gb
print(p.recommend().pbs)  # select=1:ncpus=8:mem=14gb
```

**2. If your script imports Polars, run `stream_limits.set_limits()` *before* the import.**

```python
import stream_limits
stream_limits.set_limits()     # caps threads and Polars' memory at the job's allocation

import polars as pl            # only now
from logit_stream import mlogit_stream
```

Polars fixes its thread pool and its memory budget when it is imported, and nothing can change
them afterwards, so a limit set after the import cannot reach it (you get a warning that says
so). On a cluster, importing logit-stream applies the limits for you; the line above matters when
something imports Polars *before* logit-stream does, such as your own `import polars` at the top of the
script.

The rest of this page is the short version of
[hdfe-stream's Resource limits and profiling](https://leetucker.github.io/hdfe-stream/resources.html), the full reference.

## What the profile models

The profile depends on the model: a multinomial logit has K x (J - 1) parameters, an ordered one
K + J - 1, a conditional one K, and the information matrix, its per-thread copies, the absorbed
fixed effect's G x J x P cross blocks and the clusters' score sums are sized from them.
`logit_stream` and `probit_stream` are hdfe-stream's `feglm_stream` and profile as it does.

### The multinomial logit with several absorbed fixed effects

`mlogit_stream` with two or more absorbed fixed effects is fitted as a Poisson
regression on the data expanded J-fold (one row per observation and outcome level, a
fixed effect per observation, each fixed effect interacted with the level, each covariate
with the non-base levels). Its profile is hdfe-stream's profile of that regression (see [Reading a profile](https://leetucker.github.io/hdfe-stream/resources.html#reading-a-profile)), from
counts worked out from one scan of the *original* data: N x J rows, N groups of J rows,
G x J levels per fixed effect, K x (J - 1) covariates. The profile expands nothing, and
says in its warnings that it describes the expanded regression.

This case has had one real-fit check (300,000 rows, four outcome levels, two fixed
effects: expected 1.3 times the measured peak, upper bound 2.2 times), not a check at
scale. J is the scan's approximate count of the outcome's values unless you give
`levels=`, and the memory of the lazy expansion itself (a cross join in Polars' streaming
pass 0) is not modeled: pass 0 is treated as the ordinary sort of N x J rows.

That route does not record `diagnostics["disk_peak_gb"]`; `diagnostics["peak_rss_gb"]` is recorded
as for every fit.

## Thread and memory limits

Polars, numba, BLAS and pyarrow each default to every core of the node. logit-stream caps all of
them at a scheduler's allocation automatically when it finds one, and gives Polars 2.x a memory
budget past which it spills to disk. This is done by the standalone
[`stream-limits`](https://leetucker.github.io/stream-limits/) package; its documentation covers
[thread limits](https://leetucker.github.io/stream-limits/threads.html),
[memory limits](https://leetucker.github.io/stream-limits/memory.html) and
[schedulers and containers](https://leetucker.github.io/stream-limits/schedulers.html) (PBS, Slurm,
cgroups, and a `qstat` fallback for sites that set no memory variable). `python -m stream_limits`
shows what was found, from inside a job. The `n_threads=` option holds numba and BLAS to that many
threads for one fit.

## Choices that follow the memory left

Choices that trade speed for memory follow what is *left* of the job's memory: the `max_memory_gb`
option sets the limit and `memory_reserve` the share held back (default 0.15), and what cannot
shrink is refused up front with the numbers. Every fit records `diagnostics["peak_rss_gb"]`. The
full explanation, with the profile's accuracy and what each package does with the headroom, is in
[hdfe-stream's documentation](https://leetucker.github.io/hdfe-stream/resources.html).
