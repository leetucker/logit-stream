# How it works

## Pass 0: the sample as blocks

A fit starts by reading the source once. Polars' streaming engine evaluates
the formula's design -- every column of the model matrix is a Polars
expression, compiled from pyfixest's own model matrix by hdfe-stream's
formula front end -- drops incomplete rows, and writes the sample to the run
directory as Parquet. That copy is then read back in batches and rewritten as
fixed-size blocks of plain arrays (`.npy` files: the design as a row-major
float64 matrix, the coded outcome, weights, and integer codes for the
clusters, the fixed effect and the choice sets), and deleted. The outcome's
levels, the cluster and fixed-effect levels, and the design's moments come
from the same local copy.

Every later step reads the blocks in order, each read in a background thread
while the previous one is worked on. Nothing row-sized is held beyond a block
(64 MB by default) and its working arrays, which are bounded separately
(`chunk_mb`). With `storage="memory"` the blocks stay in memory instead; with
`dtype="float32"` they take half the disk, rounded to seven digits (exact for
indicators). The design's 0/1 columns -- indicators of factor levels, binary
covariates -- are stored apart as one byte each and the design put back
together as each block is read, in one parallel pass: on the ACS design of
40 columns, all but two of them indicators, the store is a sixth the size,
and a read from disk is faster (34 ms from 58 for a million rows); in
memory it costs about 20 ms a read, a few percent of a Newton step.

## Newton's method, one read per step

All three likelihoods are sums over rows (over choice sets for the
conditional logit), and so are their gradients and information matrices.
A Newton step therefore needs one read of the blocks, accumulating, block by
block,

- the log-likelihood $\ell(\theta)$,
- its gradient $g = \nabla \ell$,
- the information $H = -\nabla^2 \ell$,

and one dense solve, $\theta \leftarrow \theta + H^{-1} g$. Per chunk, the work
is one product $XB$ for the linear predictors, elementwise functions of them,
and weighted Gram matrices $X'\operatorname{diag}(v)X$, each computed as the
symmetric rank-$k$ update of $\sqrt{v} \odot X$. For the multinomial logit
with $J$ levels and $K$ covariates the information has $J(J-1)/2$ blocks of
$K \times K$; for the ordered logit, one $K \times K$ block and the cutpoints;
for the conditional logit, one $K \times K$ block less the within-set means'.

The iteration starts from the null model (zero slopes, intercepts or
cutpoints at the sample shares), so the first log-likelihood is the null one.

### Convergence

Every model here has a globally concave log-likelihood, so the Newton
direction always points uphill, and Newton's method converges quadratically
near the optimum: four to eight steps on the benchmarks. The criterion is the
Newton decrement $\lambda^2 = g'H^{-1}g$: twice the increase in the
log-likelihood the next step would bring if the model were quadratic, and the
squared length of that step in standard-error units (as $H^{-1}$ is the
variance of the estimates). The default tolerance, `tol=1e-8`, stops when the
next step would move the estimates by about $10^{-4}$ standard errors -- the
quantity Stata's `nrtolerance()` measures, whose default is $10^{-5}$.

The reported estimates are those at which the decrement was computed, so
the log-likelihood, gradient and information reported are all exactly theirs
(the final step is not taken).

### Safeguards

- **Equilibration.** Every solve scales $H$ to unit diagonal first, which
  makes the Cholesky factorization indifferent to the scale of the covariates
  -- what standardizing them would do, without touching the data. Newton's
  method is invariant to such rescaling in exact arithmetic, so this changes
  rounding only.
- **Step-halving.** A step that lowers the log-likelihood (possible far from
  the optimum), or would put the ordered logit's cutpoints out of order, is
  halved, up to twelve times, each trial costing a read of the rows for the
  log-likelihood alone.
- **Levenberg–Marquardt.** If halving fails -- which happens when the data
  separate some coefficient, whose estimate is heading for infinity and whose
  information is vanishing, so that the Newton step is astronomically long --
  the step is damped, $(H + \lambda \operatorname{diag} H)^{-1} g$, with
  $\lambda$ growing tenfold, bending it toward scaled gradient ascent.
- **Collinearity.** A covariate whose residual variance, given the covariates
  before it, is below $10^{-10}$ of its variance is dropped from the first
  step's information (with a fixed effect, from the information net of it),
  with a warning, as pyfixest does.
- **Separation.** The fit warns when a coefficient grows implausibly large
  with an exploding standard error, or when Newton's method needed more than
  15 steps (its convergence on separated data is linear, not quadratic).
- **Ridge.** `l2=` adds $\tfrac{\lambda}{2}\lVert\beta\rVert^2$ (not on
  intercepts or cutpoints), which makes every problem strictly concave.

## Inference

The iid variance is $H^{-1}$ at the estimates. The robust and clustered ones
are sandwiches $H^{-1} M H^{-1}$, with $M = \sum_i s_i s_i'$ (scores of rows,
or of choice sets) or $\sum_c (\sum_{i \in c} s_i)(\cdots)'$, collected in one
more read of the data, with Stata's small-sample factors for
maximum-likelihood estimators: $N/(N-1)$ and $G/(G-1)$. p-values and
intervals use the normal distribution, and Wald tests the chi-squared, as for
hdfe-stream's GLMs.

## An absorbed fixed effect

See [fixed effects](examples/fixed_effects.md): the information matrix has an
arrow shape, and each Newton step solves it exactly through the Schur
complement of its per-group blocks.

## Predictions

`predict`, `resid` and `sample` are lazy Polars frames over the source or new
data: the design's expressions are evaluated by Polars, batch by batch, and a
numpy function applies the coefficients (and the fixed effects, joined from
their table) to each batch. So results hold no files.

## On a GPU

The heavy work of a step is dense linear algebra on a block -- one product
for the linear predictors, elementwise functions, and Gram matrices -- which
maps directly onto a GPU array library. With `device="cuda"` (experimental;
CuPy and an NVIDIA GPU) the blocks are copied to the device once and kept
there when they fit in half its free memory (else copied each step), and
only the $P$-vector gradient and the $P \times P$ information come back.
That covers the multinomial, ordered (logit and probit), conditional and
conditional fixed-effects logit, the multinomial and ordered logit with an
absorbed fixed effect (its per-group blocks by scatter-adds) or random
intercepts (with the exact Laplace terms), Firth's adjustment, and the
mixed and nested logit. A design's 0/1 columns stay bytes on the device and
are expanded each pass, so about six times as much of an indicator-heavy
sample stays resident. The mixed logit's utilities are a rows x draws array
per batch of choosers, each choice set's probabilities a softmax down its
rows for every draw; every term of its gradient is a covariate times a
per-row sum over the draws, so nothing chooser x draws x covariates is
formed, and the draws (Halton or the counter-based pseudo-random ones) are
generated on the device. The nested logit's node values are dense choice
sets x nests arrays, filled from the rows by scatter-max and scatter-add
and carried up the tree a nest at a time. On the CPU both keep their numba
kernels, which are an order of magnitude faster there. The robust and
clustered variances' scores pass runs on the device too: each model's
scores (concentrated for an absorbed fixed effect, profiled for random
intercepts) there, then S'S and the scatter-added cluster sums, P x P and
G x P, brought back; only the cluster jackknife (CRV3) reads on the CPU.
The
conditional logit's segment sums over choice sets are cumulative sums; the
conditional fixed-effects logit's recursion over each unit's periods runs
for all the units of one length at once (their rows gathered into a units x
periods x covariates array), the recursion over the number of 1s
vectorized too. The code is written against the array namespace, so the
test suite runs it on numpy in place of CuPy; `benchmarks/gpu_benchmark.py`
compares the two on the hardware. Pass 0's Polars queries can run on the
GPU too (`polars_engine="gpu"`).
