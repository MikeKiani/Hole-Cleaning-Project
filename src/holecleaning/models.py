"""
Model zoo and training utilities.

The original study compared three tree ensembles (Random Forest, Gradient
Boosting, AdaBoost) plus a Ridge meta-learner. This module keeps those but
widens the field to a comparative benchmark of eight regressors spanning three
model families — linear, distance/kernel, and tree ensembles — so the choice of
the final model is evidence-based rather than assumed.

All estimators are wrapped in pipelines where scaling matters (linear, kNN, SVR)
so that leakage-free cross-validation is automatic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from . import config as C


def _scaled(estimator) -> Pipeline:
    """Wrap a scale-sensitive estimator with a StandardScaler."""
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def model_zoo() -> dict:
    """Return the full benchmark of estimators, keyed by short name.

    Hyper-parameters here are sensible, lightly-tuned defaults; the search
    routines below refine the top performers.
    """
    rs = C.RANDOM_STATE
    return {
        # --- Linear family ------------------------------------------------- #
        "Ridge": _scaled(Ridge(alpha=1.0, random_state=rs)),
        "ElasticNet": _scaled(
            ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=rs, max_iter=10000)
        ),
        # --- Distance / kernel family ------------------------------------- #
        "kNN": _scaled(KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "SVR": _scaled(SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale")),
        # --- Tree-ensemble family ----------------------------------------- #
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=rs, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300, random_state=rs, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=rs),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=rs, max_iter=400, learning_rate=0.05
        ),
        "AdaBoost": AdaBoostRegressor(
            estimator=GradientBoostingRegressor(random_state=rs),
            n_estimators=60,
            learning_rate=0.05,
            loss="exponential",
            random_state=rs,
        ),
    }


def build_stack(base_models: dict | None = None) -> StackingRegressor:
    """Assemble a stacked ensemble.

    Base learners are the three best tree ensembles; the meta-learner is a Ridge
    regressor (matching the original study's design) that learns how to blend
    their predictions. ``passthrough=False`` keeps the meta-features limited to
    the base predictions, which is the intent of a classic stack.
    """
    rs = C.RANDOM_STATE
    if base_models is None:
        base_models = {
            "rf": RandomForestRegressor(
                n_estimators=200, random_state=rs, n_jobs=-1
            ),
            "gb": GradientBoostingRegressor(random_state=rs),
            "et": ExtraTreesRegressor(
                n_estimators=200, random_state=rs, n_jobs=-1
            ),
        }
    return StackingRegressor(
        estimators=list(base_models.items()),
        final_estimator=Ridge(alpha=0.25, random_state=rs),
        cv=3,
        n_jobs=-1,
    )


# --------------------------------------------------------------------------- #
# Hyper-parameter search
# --------------------------------------------------------------------------- #
@dataclass
class SearchSpaces:
    """Curated search grids for the tree ensembles, kept small enough to run in
    seconds on 116 rows yet wide enough to matter."""

    random_forest = {
        "n_estimators": np.arange(100, 500, 50),
        "max_depth": [None, 5, 10, 20, 40],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 1.0],
    }
    gradient_boosting = {
        "n_estimators": np.arange(100, 400, 50),
        "learning_rate": np.logspace(-2, -0.3, 8),
        "max_depth": [2, 3, 4],
        "subsample": [0.7, 0.85, 1.0],
    }
    extra_trees = {
        "n_estimators": np.arange(100, 500, 50),
        "max_depth": [None, 10, 20, 40],
        "min_samples_leaf": [1, 2, 4],
    }


def random_search(estimator, params, X, y, n_iter=200, cv=C.CV_FOLDS):
    """RandomizedSearchCV wrapper scored on RMSE (negated)."""
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=C.RANDOM_STATE,
        verbose=0,
    )
    search.fit(X, y)
    return search


def grid_search(estimator, params, X, y, cv=C.CV_FOLDS):
    """GridSearchCV wrapper scored on RMSE (negated)."""
    search = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    return search
