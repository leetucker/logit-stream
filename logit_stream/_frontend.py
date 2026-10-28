"""Formula front end: a pyfixest-style formula -> the models to fit.

The right-hand side is compiled by hdfe-stream's formula compiler
(`hdfe_stream.core.formula`), which runs pyfixest's own parser and model
matrix on a tiny synthetic frame and turns each column back into a Polars
expression. Coefficient names and the coding of categorical variables are
therefore exactly pyfixest's. The outcome is handled here: for the
multinomial and ordered models it is a categorical variable, not a number,
so it is kept as a raw column and coded once its levels are known.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
from hdfe_stream.core.feterms import parse_fe
from hdfe_stream.core.formula import (
    categorical_vars,
    compile_design,
    numeric_expr,
    pyfixest_formula_api,
)


@dataclass
class _Spec:
    """One model: its formula, outcome, covariates and fixed effects."""

    fml: str
    depvar: str
    y: pl.Expr                      # the outcome, uncast
    x: list = field(default_factory=list)       # [(coefficient name, Float64 expr)]
    fe: tuple = ()


def _outcome_expr(dep, schema):
    """A column is taken as it is (strings, categoricals, integers);
    anything else is a numeric expression such as ``I(educ > 12)``."""
    if dep in schema:
        return pl.col(dep)
    return numeric_expr(dep)


def _plan(fml: str, lf: pl.LazyFrame, intercept: bool, collect) -> list[_Spec]:
    """Expand `fml` into its models (pyfixest's multiple-estimation syntax:
    several outcomes, ``sw()``, ``csw()``, ...).

    `intercept`: whether a model without fixed effects gets the model
    matrix's constant (multinomial logit does; ordered and conditional logit
    do not, their cutpoints and choice sets absorbing it). Without it the
    categorical coding is still reduced rank, as with it.
    `collect(lf, what)` runs a query (so that the GPU engine applies).
    """
    Formula, _ = pyfixest_formula_api()
    schema = lf.collect_schema()
    specs = Formula.parse(fml)
    cats = set()
    for s in specs:
        if s.first_stage is not None:
            raise ValueError(f"{s.formula}: IV formulas are not available for logit models")
        cats |= categorical_vars(s.second_stage.split("~", 1)[1], schema)
    levels = {}
    for v in sorted(cats):
        u = collect(lf.select(pl.col(v).drop_nulls().unique().sort()), f"levels of {v}")
        levels[v] = u[v].to_list()
    out = []
    for s in specs:
        dep, rhs = (t.strip() for t in s.second_stage.split("~", 1))
        fe = tuple(parse_fe(s.fixed_effects))
        for d in fe:
            if "[" in d:
                raise ValueError(f"{s.formula}: varying slopes are not available for logit models")
        _, cols = compile_design(dep, rhs, schema, levels, intercept=intercept and not fe)
        if not intercept:
            cols = [(nm, ex) for nm, ex in cols if nm != "Intercept"]
        out.append(_Spec(fml=s.formula, depvar=dep, y=_outcome_expr(dep, schema), x=cols, fe=fe))
    return out
