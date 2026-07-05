"""
Stage 4 — Drilling-parameter optimisation.

For every experimental run, searches a ±20 % neighbourhood of the controllable
parameters (flow rate, pipe rotation, ROP) for the setting that minimises
predicted cuttings concentration, then compares the achievable minimum against
the originally observed concentration.

Produces:
  * reports/tables/optimized_concentration.csv
  * reports/figures/optimization_performance.png

Run:  python scripts/04_optimize.py   (requires stage 3 to have run)
"""

import _bootstrap  # noqa: F401

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from holecleaning import config as C
from holecleaning import data, evaluation, optimization


def main() -> None:
    evaluation.apply_style()
    df = data.load_clean()

    model_names = ["randomforest", "gradientboosting", "stacked"]
    loaded = {}
    for n in model_names:
        path = C.MODEL_DIR / f"{n}.joblib"
        if path.exists():
            loaded[n] = joblib.load(path)
    if not loaded:
        raise SystemExit("No trained models found — run 03_train_models.py first.")

    comparison = pd.DataFrame({"Original": df[C.TARGET]})
    for name, model in loaded.items():
        print(f"Optimising with {name} ...")
        comparison[f"{name}_optimized"] = optimization.optimize_dataset(
            model, df, pct_range=0.2)

    comparison = comparison.abs().sort_values("Original").reset_index(drop=True)
    comparison.to_csv(C.TABLE_DIR / "optimized_concentration.csv", index=False)

    # ---- Plot: original vs optimised for each model ----------------------- #
    opt_cols = [c for c in comparison.columns if c != "Original"]
    fig, axes = plt.subplots(1, len(opt_cols), figsize=(6 * len(opt_cols), 5.5),
                             sharey=True)
    if len(opt_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, opt_cols):
        ax.scatter(comparison.index, comparison["Original"], s=20, alpha=0.7,
                   label="Observed")
        ax.scatter(comparison.index, comparison[col], s=20, alpha=0.7,
                   label="Optimised", color="#2a9d8f")
        reduction = (1 - comparison[col].mean() / comparison["Original"].mean())
        ax.set_title(f"{col.replace('_optimized','')}\n"
                     f"mean reduction ≈ {reduction:.0%}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Run (sorted by observed concentration)")
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Cuttings concentration")
    fig.suptitle("Optimisation performance — achievable concentration reduction",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(C.FIGURE_DIR / "optimization_performance.png", bbox_inches="tight")

    print("\nMean concentration — observed vs optimised:")
    summary = comparison.mean().round(4)
    print(summary.to_string())
    print("\noptimized_concentration.csv and optimization_performance.png written")


if __name__ == "__main__":
    main()
