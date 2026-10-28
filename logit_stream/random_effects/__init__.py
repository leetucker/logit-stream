"""Random intercepts for the multinomial and ordered logit (experimental): one
grouping, several nested ones (random state intercepts with random PUMA
intercepts within), optionally inside an absorbed fixed effect they all nest
in, or two crossed ones (random PUMA and random occupation intercepts).

**The model.** Each grouping f has, for each of its groups, intercepts a_fg
~ N(0, Sigma_f) -- one per non-base outcome level (multinomial, Sigma_f
(J-1) x (J-1), unstructured or diagonal) or one location shift (ordered) --
independent across groupings and groups, added to the linear predictor of
the group's rows.

**The likelihood** integrates the effects out, and is approximated by
Laplace's method:

    L(b, Sigma) = l(b, a) - sum_f a_f' Lam_f a_f / 2 - sum_f G_f log|Sigma_f| / 2
                  - log|M| / 2,                               Lam_f = Sigma_f^-1,

with a the mode of h(a) = l(b, a) - sum_f a_f' Lam_f a_f / 2 given b, and M
the information for all the effects at the mode plus the Lam_f -- one big
matrix, whose structure the groupings set.

**Units.** A unit is a set of rows that share all their groups: a group of
the finest grouping when the groupings nest, a cell (PUMA x occupation)
when they cross. A read accumulates each unit's blocks (the gradient and
information of its rows' linear predictor, and its information with b), and
every group's blocks are sums of its units' (the effects enter the linear
predictor alike), as is the information between a group and another
grouping's group: the sum over their common units. The information matrix
of the effects is then

- **nested**: a tree -- a group is coupled only to its ancestors -- which is
  eliminated from the finest level up, each group updating only its own
  ancestors (no fill-in), the fixed effect last, then b;
- **crossed**: the larger grouping's blocks are diagonal and eliminated
  first; the smaller grouping's effects then form one dense system (each
  pair of its groups coupled through the units they share), of at most
  `DENSE_MAX` effects, solved directly, then b.

**For fixed Sigma**, b and the effects at their joint mode maximize h:
Newton's method with that elimination, one read per step (lme4's
``nAGQ = 0``, ``random_method="joint-mode"``). The exact Laplace
approximation (``"laplace"``, lme4's ``nAGQ = 1``, the default) adds the
log-determinant's gradient in b, directly and through the modes,

    -(1/2) [ sum_i c_i (x) x_i - B' M^-1 C ],

where c_i = d log|M| / d eta_i depends on the row's unit through the
covariance of its total effect, V_u = sum of the blocks of M^-1 along the
unit's path (its group in every grouping): for the multinomial,
c = (diag p - p p') (diag V_u - 2 V_u p); C stacks the groups' sums of c,
and B the groups' information with b. Each Sigma_f is updated at the fixed
point of L's gradient in it,

    Sigma_f = mean_g [ a_g a_g' + Cov(a_g) - sym(a_g u_g') ],  u = M^-1 C,

with Cov(a_g) the group's block of M^-1 (the last term is the exact
approximation's; without it this is the Laplace-EM update), iterated
between reads on the data's quadratic model with b held and the units'
information updated to first order through its derivatives, and
accelerated by SQUAREM.

**Variances.** b's: the inverse of its information net of all the
effects (and the fixed effect), plus J V(Sigma) J' for Sigma's uncertainty
(J = d b / d Sigma). Sigma's: minus the inverse curvature of L in Sigma's
elements, by differencing its gradient between reads. Robust variances
cluster the profile scores (the rows' scores for b, concentrated with
respect to the fixed effect) and need the clusters to contain whole groups
of the coarsest random grouping (nested groupings only).

Layout
    linalg.py   the effects' information matrix, factorized (tree or crossed)
    model.py    `_RandomEffects`: groupings, data passes, Laplace terms, the
                Newton step, inference
    sigma.py    Sigma's updates and variance (a mixin of `_RandomEffects`)
    fit.py      the rounds (joint mode, then the exact Laplace refinement),
                and Sigma's standard errors
"""

from .fit import GAIN_TOL, SIGMA_STEP, _fit_random, _refine_laplace, _sigma_inference
from .model import DENSE_MAX, _RandomEffects
from .sigma import VAR_FLOOR

__all__ = ["DENSE_MAX", "GAIN_TOL", "SIGMA_STEP", "VAR_FLOOR", "_RandomEffects", "_fit_random",
           "_refine_laplace", "_sigma_inference"]
