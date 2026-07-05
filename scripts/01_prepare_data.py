"""
Stage 1 — Data preparation.

Loads the raw Tulsa flow-loop experiments, cleans them into the canonical
modelling frame, adds the physics-informed features, and writes both processed
tables to ``data/processed/``.

Run:  python scripts/01_prepare_data.py
"""

import _bootstrap  # noqa: F401

from holecleaning import config as C
from holecleaning import data, features


def main() -> None:
    raw = data.load_raw()
    print(f"Raw data: {raw.shape[0]} runs × {raw.shape[1]} columns")

    clean = data.load_clean()  # cleans and saves to processed/cleaned_data.csv
    print(f"Cleaned data written -> {C.PROCESSED_DATA.relative_to(C.ROOT_DIR)}")

    featured = features.add_physics_features(clean)
    featured.to_csv(C.FEATURED_DATA, index=False)
    print(f"Featured data written -> {C.FEATURED_DATA.relative_to(C.ROOT_DIR)}")

    print("\nSummary of the cleaned modelling frame:")
    print(data.describe(clean).to_string())


if __name__ == "__main__":
    main()
