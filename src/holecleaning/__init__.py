"""
holecleaning
============

A machine-learning workflow for predicting downhole cuttings concentration and
determining hole-cleaning conditions from surface drilling parameters.

Public sub-modules
------------------
config        Paths, constants, geometry and styling.
data          Loading, cleaning, splitting.
features      Physics-informed feature engineering (annular velocity, CCI).
eda           Correlation, distributions, feature ranking.
models        Model zoo, hyper-parameter search, stacked ensemble.
evaluation    Metrics, cross-validated benchmark, plots.
optimization  Drilling-parameter optimisation.
zones         Hole-cleaning zone (feed-zone) mapping.
"""

from . import (  # noqa: F401
    config,
    data,
    eda,
    evaluation,
    features,
    models,
    optimization,
    zones,
)

__version__ = "1.0.0"
__all__ = [
    "config",
    "data",
    "eda",
    "evaluation",
    "features",
    "models",
    "optimization",
    "zones",
]
