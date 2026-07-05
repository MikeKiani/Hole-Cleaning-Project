"""
Drilling-parameter optimisation.

Given a trained model and a current set of operating conditions, search a bounded
neighbourhood of the *controllable* parameters for the combination that minimises
predicted cuttings concentration. This is the "what should I change, and by how
much, to clean this hole?" tool.

This is a clean, vectorised rewrite of the original ``optimizer.py``. The old
version relied on injecting variables into ``globals()`` and a fragile chain of
merges; here the parameter grid is built explicitly and evaluated in one batched
prediction.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from . import config as C


def optimize_row(
    model,
    row: pd.Series,
    pct_range: float = 0.2,
    flex_params: list[str] | None = None,
    n_levels: int = 8,
    return_full: bool = True,
):
    """Optimise the controllable parameters of a single operating point.

    Parameters
    ----------
    model : fitted regressor
    row : pd.Series
        One row of features (no target) describing current conditions.
    pct_range : float
        Fractional half-width of the search window around each flex parameter
        (0.2 -> ±20 %).
    flex_params : list[str] | None
        Parameters allowed to vary. Defaults to the operationally controllable
        set (Flow rate, Pipe rotation, ROP). Everything else is held fixed.
    n_levels : int
        Grid resolution per flex parameter.
    return_full : bool
        If True return ``(best_settings_series, best_concentration)``;
        otherwise just the best concentration (handy for vectorised .apply).
    """
    features = list(row.index)
    if flex_params is None:
        flex_params = ["Flow rate", "Pipe rotation", "ROP"]
    flex_params = [p for p in flex_params if p in features]

    # Build the 1-D level set for each parameter.
    axes = {}
    for p in features:
        if p in flex_params:
            lo, hi = row[p] * (1 - pct_range), row[p] * (1 + pct_range)
            axes[p] = np.linspace(lo, hi, n_levels)
        else:
            axes[p] = np.array([row[p]])

    grid = pd.DataFrame(
        list(itertools.product(*[axes[p] for p in features])), columns=features
    )
    grid["Concentration"] = model.predict(grid[features])

    best = grid.loc[grid["Concentration"].idxmin()]
    best_conc = float(best["Concentration"])
    if not return_full:
        return best_conc
    return best[features], round(best_conc, 4)


def optimize_dataset(model, df: pd.DataFrame, pct_range: float = 0.2,
                     flex_params: list[str] | None = None) -> pd.Series:
    """Apply :func:`optimize_row` across every row, returning the achievable
    minimum concentration for each — the basis of the before/after comparison."""
    X = df.drop(columns=[C.TARGET]) if C.TARGET in df.columns else df
    return X.apply(
        lambda r: optimize_row(model, r, pct_range=pct_range,
                               flex_params=flex_params, return_full=False),
        axis=1,
    )


def grid_search_optimum(model, param_levels: dict) -> pd.DataFrame:
    """Exhaustively evaluate a user-specified Cartesian grid and return the
    single minimum-concentration setting. ``param_levels`` maps each feature to
    a list of candidate values."""
    tic = time.time()
    features = [c for c in C.FEATURE_ORDER if c in param_levels]
    grid = pd.DataFrame(
        list(itertools.product(*[param_levels[f] for f in features])),
        columns=features,
    )
    grid["Concentration"] = np.round(model.predict(grid[features]), 3)
    toc = time.time()
    name = type(model).__name__
    best = grid.loc[grid["Concentration"].idxmin()]
    print(f"{name}: searched {len(grid):,} combinations in {toc - tic:.2f}s")
    return best.to_frame(name=name)
