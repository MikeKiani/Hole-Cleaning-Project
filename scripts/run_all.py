"""
Run the entire workflow end-to-end, in order.

    python scripts/run_all.py

Each stage is a standalone module too, so you can run them individually if you
only need to regenerate part of the analysis.
"""

import _bootstrap  # noqa: F401

import importlib
import time

STAGES = [
    ("01_prepare_data", "Data preparation"),
    ("02_run_eda", "Exploratory analysis (incl. density & velocity)"),
    ("03_train_models", "Benchmark, tune, evaluate models"),
    ("04_optimize", "Drilling-parameter optimisation"),
    ("05_find_zones", "Hole-cleaning zone / feed-zone mapping"),
]


def main() -> None:
    for module_name, label in STAGES:
        print("\n" + "=" * 72)
        print(f"STAGE  {module_name}  —  {label}")
        print("=" * 72)
        t0 = time.time()
        module = importlib.import_module(module_name)
        module.main()
        print(f"[{module_name} finished in {time.time() - t0:.1f}s]")
    print("\nAll stages complete. See reports/figures, reports/tables and models/.")


if __name__ == "__main__":
    main()
