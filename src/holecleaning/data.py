"""
Data access layer: load the raw experimental table, clean it into a canonical
modelling frame, and produce reproducible train/test splits.

The raw file is the University-of-Tulsa cuttings-transport data set (Yu et al.):
116 flow-loop runs recording fluid rheology, wellbore geometry, and drilling
parameters against the measured downhole cuttings *Concentration* (a stationary
cuttings-bed volume fraction, 0-1).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw(path=C.RAW_DATA) -> pd.DataFrame:
    """Read the raw experimental CSV exactly as delivered."""
    return pd.read_csv(path, header=0)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return the canonical modelling frame.

    Steps
    -----
    1. Rename ``Density, ppg`` -> ``Density``.
    2. Drop the row index (``Test No.``) and the redundant, perfectly collinear
       plastic-viscosity column (``PV``); see ``config.DROP_COLUMNS``.
    3. Re-order columns so the feature block is stable and the target is last.
    """
    df = df.rename(columns=C.RENAME_COLUMNS)
    df = df.drop(columns=[c for c in C.DROP_COLUMNS if c in df.columns])
    ordered = [c for c in C.FEATURE_ORDER if c in df.columns] + [C.TARGET]
    return df.loc[:, ordered].reset_index(drop=True)


def load_clean(path=C.RAW_DATA, save_to=C.PROCESSED_DATA) -> pd.DataFrame:
    """Load and clean in one call, optionally persisting the processed frame."""
    df = clean(load_raw(path))
    if save_to is not None:
        df.to_csv(save_to, index=False)
    return df


def split(
    df: pd.DataFrame,
    target: str = C.TARGET,
    test_size: float = C.TEST_SIZE,
    random_state: int = C.RANDOM_STATE,
):
    """Deterministic feature/target train-test split.

    Returns ``X_train, X_test, y_train, y_test`` to match the scikit-learn
    convention used throughout the code base.
    """
    features = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        features, y, test_size=test_size, random_state=random_state
    )


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Enriched summary table: standard describe() plus unique-value counts,
    which matters here because most inputs are discrete design-of-experiment
    levels rather than continuous measurements."""
    desc = df.describe().T
    desc["n_unique"] = df.nunique()
    desc["dtype"] = df.dtypes.astype(str)
    return desc.round(3)
