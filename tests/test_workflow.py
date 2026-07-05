"""
Lightweight test suite. Runs in a few seconds and guards the core contracts:
data cleaning, physics features, zone classification, optimisation and the
model zoo. Run with:  pytest
"""

import numpy as np
import pandas as pd
import pytest

from holecleaning import config as C
from holecleaning import data, features, models, optimization, zones


@pytest.fixture(scope="module")
def clean_df():
    return data.load_clean(save_to=None)


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #
def test_clean_drops_redundant_columns(clean_df):
    assert "PV" not in clean_df.columns          # collinear with YP
    assert "Test No." not in clean_df.columns
    assert clean_df.columns[-1] == C.TARGET       # target is last
    assert clean_df.isnull().sum().sum() == 0


def test_split_shapes(clean_df):
    X_train, X_test, y_train, y_test = data.split(clean_df)
    assert len(X_train) + len(X_test) == len(clean_df)
    assert C.TARGET not in X_train.columns


# --------------------------------------------------------------------------- #
# Physics features
# --------------------------------------------------------------------------- #
def test_annular_velocity_monotonic():
    v = features.annular_velocity([100, 200, 300])
    assert np.all(np.diff(v) > 0)                 # velocity rises with flow
    assert np.all(v > 0)


def test_physics_features_present(clean_df):
    feat = features.add_physics_features(clean_df)
    for col in ["Annular velocity", "Transport index",
                "Carrying capacity index", "Rotation factor"]:
        assert col in feat.columns


# --------------------------------------------------------------------------- #
# Zones
# --------------------------------------------------------------------------- #
def test_zone_classification_boundaries():
    labels = zones.classify([0.0, 0.05, 0.10, 0.15, 0.30])
    assert list(labels) == ["Efficient", "Efficient", "Marginal",
                            "Marginal", "Poor"]


def test_zone_summary_totals(clean_df):
    summary = zones.zone_summary(clean_df)
    assert summary["n_runs"].sum() == len(clean_df)


# --------------------------------------------------------------------------- #
# Models + optimisation (fast smoke tests)
# --------------------------------------------------------------------------- #
def test_zoo_predicts(clean_df):
    X_train, X_test, y_train, y_test = data.split(clean_df)
    rf = models.model_zoo()["RandomForest"]
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    assert preds.shape == y_test.shape
    assert np.all(np.isfinite(preds))


def test_optimizer_does_not_worsen(clean_df):
    X_train, _, y_train, _ = data.split(clean_df)
    rf = models.model_zoo()["RandomForest"].fit(X_train, y_train)
    row = clean_df.drop(columns=[C.TARGET]).iloc[0]
    best_settings, best_conc = optimization.optimize_row(rf, row, pct_range=0.2)
    baseline = float(rf.predict(pd.DataFrame([row]))[0])
    # The optimum over a neighbourhood that includes the point itself can only
    # be <= the baseline prediction.
    assert best_conc <= baseline + 1e-6
