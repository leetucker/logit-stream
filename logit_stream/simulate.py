"""Simulated data for examples, tests and benchmarks.

`simulate_people` is a population shaped like the demographic imputation
problem: people in PUMAs nested in states, with age, sex, income and a few
household characteristics, and three outcomes drawn from logit models of
those -- a binary one (Hispanic origin), a multinomial one (race/ethnicity,
six categories) and an ordered one (educational attainment, five levels) --
each with PUMA effects. `simulate_choices` is long-format discrete-choice
data in the shape of a mode-choice survey (Swissmetro), for the conditional
logit, and with options for the nested and mixed logit. `simulate_panel`
is a panel with a binary outcome and unit effects, for the fixed-effects
logit.

The true coefficients are returned in ``frame.attrs``-like dicts by the
``truth_*`` functions, so tests can check recovery.
"""

from __future__ import annotations

import numpy as np
import polars as pl

RACES = ["White", "Black", "Hispanic", "Asian", "AIAN", "Multiracial"]
EDUCATION = ["Less than HS", "High school", "Some college", "Bachelor's", "Graduate"]

# race equations (vs White): intercept, age/10, female, log income, owner, urban
_RACE_B = np.array([
    [-1.6, -0.10, 0.05, -0.35, -0.50, 0.9],     # Black
    [-1.2, -0.25, 0.00, -0.30, -0.30, 0.6],     # Hispanic
    [-2.6, -0.05, 0.05, 0.25, -0.10, 1.0],      # Asian
    [-3.8, -0.05, 0.00, -0.40, 0.10, -0.8],     # AIAN
    [-3.0, -0.30, 0.05, -0.05, -0.20, 0.3],     # Multiracial
])
_EDUC_B = np.array([0.05, 0.25, 0.85, 0.30, 0.25])      # age/10, female, log inc, owner, urban
_EDUC_CUTS = np.array([7.0, 8.5, 9.6, 10.9])
_HISP_B = np.array([-1.0, -0.20, 0.0, -0.30, -0.30, 0.5])


def truth_people() -> dict:
    """The coefficients `simulate_people` draws from (before the PUMA effects)."""
    xs = ["Intercept", "age10", "female", "log_income", "owner", "urban"]
    return {"race": {f"{r}:{x}": float(b) for r, row in zip(RACES[1:], _RACE_B)
                     for x, b in zip(xs, row)},
            "education": {**{x: float(b) for x, b in zip(xs[1:], _EDUC_B)},
                          **{f"cut{j + 1}": float(c) for j, c in enumerate(_EDUC_CUTS)}},
            "hispanic": {x: float(b) for x, b in zip(xs, _HISP_B)}}


def simulate_people(n: int = 100_000, n_states: int = 51, pumas_per_state: int = 20,
                    puma_sd: float = 0.5, seed: int = 0) -> pl.DataFrame:
    """A simulated population of `n` people.

    Columns: ``person_id``, ``state``, ``puma`` (nested in state;
    ``n_states * pumas_per_state`` of them, with Zipf-like sizes),
    ``age`` (18-90), ``age10`` (age / 10), ``female`` (0/1),
    ``log_income``, ``owner`` (0/1), ``urban`` (0/1), ``weight`` (a
    survey-style weight), and the outcomes ``hispanic`` (0/1), ``race``
    (one of `RACES`) and ``education`` (an Enum of `EDUCATION`, ordered).

    Each outcome follows a logit model of the covariates (`truth_people`)
    plus PUMA effects with standard deviation `puma_sd` (one per race
    equation, one each for education and Hispanic origin).
    """
    rng = np.random.default_rng(seed)
    G = n_states * pumas_per_state
    size = 1.0 / np.arange(1, G + 1) ** 0.6
    rng.shuffle(size)
    puma = rng.choice(G, size=n, p=size / size.sum())
    state = puma // pumas_per_state
    urban_rate = rng.uniform(0.2, 0.95, G)
    age = np.clip(rng.gamma(9.0, 5.0, n) + 5, 18, 90).round()
    female = (rng.random(n) < 0.51).astype(np.int8)
    urban = (rng.random(n) < urban_rate[puma]).astype(np.int8)
    puma_inc = rng.normal(0, 0.25, G)
    log_income = (10.4 + puma_inc[puma] + 0.012 * (age - 45) - 0.0004 * (age - 45) ** 2
                  - 0.1 * female + 0.15 * urban + rng.normal(0, 0.8, n))
    owner = (rng.random(n) < 1 / (1 + np.exp(-(-4 + 0.05 * age + 0.3 * (log_income - 10.4)
                                                 - 0.5 * urban)))).astype(np.int8)
    X = np.column_stack([np.ones(n), age / 10, female, log_income - 10.4, owner, urban])

    # race: multinomial with PUMA effects per equation
    A = rng.normal(0, puma_sd, (G, len(RACES) - 1))
    E = np.column_stack([np.zeros(n), X @ _RACE_B.T + A[puma]])
    P = np.exp(E - E.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    race = (P.cumsum(axis=1) > rng.random(n)[:, None]).argmax(axis=1)
    # education: ordered with a PUMA shift
    shift = rng.normal(0, puma_sd, G)
    lat = X[:, 1:] @ _EDUC_B + 10.4 * _EDUC_B[2] + shift[puma] + rng.logistic(size=n)
    educ = np.searchsorted(_EDUC_CUTS, lat)
    # Hispanic origin: binary with a PUMA effect
    h = X @ _HISP_B + rng.normal(0, puma_sd, G)[puma]
    hisp = (rng.random(n) < 1 / (1 + np.exp(-h))).astype(np.int8)
    weight = rng.gamma(4.0, 25.0, n).round(1) + 1.0
    return pl.DataFrame({
        "person_id": np.arange(n),
        "state": state.astype(np.int32),
        "puma": puma.astype(np.int32),
        "age": age.astype(np.int16),
        "age10": age / 10,
        "female": female,
        "log_income": log_income,
        "owner": owner,
        "urban": urban,
        "weight": weight,
        "hispanic": hisp,
        "race": np.array(RACES, dtype=object)[race],
        "education": pl.Series(np.array(EDUCATION, dtype=object)[educ]).cast(pl.Enum(EDUCATION)),
    })


def simulate_choices(n_cases: int = 10_000, alternatives=("train", "metro", "car"),
                     beta_cost: float = -1.1, beta_time: float = -1.3, p_unavailable=0.15,
                     seed: int = 0, *, nests: dict | None = None,
                     lambdas: float | dict = 0.5, tasks: int = 1,
                     sd_time: float = 0.0) -> pl.DataFrame:
    """Long-format mode-choice data for the conditional, nested and mixed
    logit.

    One row per available alternative of each choice set: ``case``,
    ``person`` (the chooser), ``alt``, ``cost`` and ``time`` (in 100s, as
    Swissmetro's are usually scaled), ``income`` (of the chooser), and
    ``chosen`` (1 for the alternative chosen). Each alternative but the
    first is unavailable with probability `p_unavailable`. Utilities:
    beta_cost cost + beta_time time + an alternative-specific constant
    (0 for the first alternative, then evenly from 0.4 to -0.3)
    + 0.3 income for the last alternative, plus Gumbel noise: the
    conditional logit.

    Parameters
    ----------
    nests : dict, optional
        ``{nest: [alternatives]}``, or with nests of nests as dicts, as
        `nlogit_stream` takes it, every alternative in one nest: the choices
        are drawn from that nested logit (RU2) instead.
    lambdas : float or dict, default 0.5
        The nests' dissimilarity parameters, one for all or ``{nest:
        lambda}`` (a nest of one child has none).
    tasks : int, default 1
        Choice sets per chooser: ``person`` is ``case // tasks``, and
        ``income`` is the chooser's.
    sd_time : float, default 0
        The standard deviation of the time coefficient across choosers:
        each draws beta_time + sd_time z, z standard normal, for all their
        choice sets (the mixed logit with a panel). Not with `nests`.
    """
    if nests is not None and sd_time:
        raise ValueError("simulate_choices draws the nested or the mixed logit, not both")
    rng = np.random.default_rng(seed)
    J = len(alternatives)
    person = np.arange(n_cases) // tasks
    asc = np.r_[0.0, np.linspace(0.4, -0.3, J - 1)]
    cost = rng.gamma(4, 0.25, (n_cases, J)) * np.linspace(0.8, 1.2, J)
    time = rng.gamma(5, 0.25, (n_cases, J)) * np.linspace(1.2, 0.8, J)
    income = rng.normal(0, 1, person[-1] + 1)[person]
    avail = np.ones((n_cases, J), bool)
    avail[:, 1:] = rng.random((n_cases, J - 1)) > p_unavailable
    V = beta_cost * cost + asc + np.outer(income, np.r_[np.zeros(J - 1), 0.3])
    if nests is None:
        U = V + beta_time * time + rng.gumbel(size=(n_cases, J))
        if sd_time:
            U += sd_time * rng.standard_normal(person[-1] + 1)[person, None] * time
        U[~avail] = -np.inf
        choice = U.argmax(axis=1)
    else:
        V = np.where(avail, V + beta_time * time, -np.inf)
        P = _nested_probs(V, list(alternatives), nests, lambdas)
        choice = (P.cumsum(axis=1) > rng.random(n_cases)[:, None]).argmax(axis=1)
    case, alt = np.nonzero(avail)
    return pl.DataFrame({
        "case": case.astype(np.int64),
        "person": person[case].astype(np.int64),
        "alt": np.array(alternatives, dtype=object)[alt],
        "cost": cost[case, alt],
        "time": time[case, alt],
        "income": income[case],
        "chosen": (choice[case] == alt).astype(np.int8),
    })


def _nested_probs(V, alternatives, nests, lambdas):
    """The nested logit's probabilities (RU2) of utilities V, a row per
    choice set (-inf for an unavailable alternative), for a tree of any
    depth: a nest's value in its parent is lambda times the log-sum of its
    children's values over lambda, and an alternative's probability the
    product of the conditional probabilities along its path."""

    def leaves(node):
        return [a for v in node.values() for a in (leaves(v) if isinstance(v, dict) else v)]

    if sorted(leaves(nests)) != sorted(alternatives):
        raise ValueError("nests must hold every alternative exactly once")

    def lam(name, children):
        if len(children) == 1:
            return 1.0
        return lambdas[name] if isinstance(lambdas, dict) else float(lambdas)

    def lse(W):
        m = W.max(axis=1, keepdims=True)
        m[~np.isfinite(m)] = 0.0
        return m[:, 0] + np.log(np.exp(W - m).sum(axis=1))      # -inf if none available

    def node(children, la):
        """The children's values, the node's log-sum I, and each
        alternative's log-probability within the node."""
        vals, paths = [], []
        for name, sub in children:
            if isinstance(sub, dict):           # a nest of nests
                u, path = nest(name, list(sub.items()))
            elif isinstance(sub, list):         # a nest of alternatives
                u, path = nest(name, [(a, None) for a in sub])
            else:                               # an alternative
                j = alternatives.index(name)
                u, path = V[:, j], {j: np.zeros(len(V))}
            vals.append(u)
            paths.append(path)
        W = np.column_stack(vals) / la
        I = lse(W)
        out = {j: lp + W[:, c] - I for c, path in enumerate(paths) for j, lp in path.items()}
        return I, out

    def nest(name, children):
        la = lam(name, children)
        I, path = node(children, la)
        return la * I, path

    with np.errstate(divide="ignore", invalid="ignore"):
        _, path = node(list(nests.items()), 1.0)
        P = np.zeros_like(V)
        for j, lp in path.items():
            P[:, j] = np.nan_to_num(np.exp(lp))
    return P


def simulate_panel(n_units: int = 10_000, periods: int = 5, beta=(1.0, -0.5),
                   unit_sd: float = 1.0, corr: float = 0.5, seed: int = 0) -> pl.DataFrame:
    """A balanced panel with a binary outcome and unit effects, for the
    fixed-effects logit.

    Columns: ``unit``, ``period`` (1 to `periods`), ``x1``, ``x2`` and
    ``y``, with P(y = 1) = logistic(a_unit + beta[0] x1 + beta[1] x2). The
    unit effects a are normal with sd `unit_sd`, and ``x1`` is
    `corr` times the unit's effect plus standard normal noise, so a model
    that ignores the effects, or treats them as independent of the
    covariates (random effects), is biased; ``x2`` is independent of them.
    """
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), periods)
    a = rng.normal(0, unit_sd, n_units)[unit]
    x1 = corr * a + rng.standard_normal(len(unit))
    x2 = rng.standard_normal(len(unit))
    eta = a + beta[0] * x1 + beta[1] * x2
    y = (rng.random(len(unit)) < 1 / (1 + np.exp(-eta))).astype(np.int8)
    return pl.DataFrame({"unit": unit.astype(np.int64),
                         "period": np.tile(np.arange(1, periods + 1), n_units).astype(np.int16),
                         "x1": x1, "x2": x2, "y": y})
