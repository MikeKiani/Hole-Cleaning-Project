"""
Stage 3 — Model benchmark, tuning, evaluation and persistence.

Cross-validates the full model zoo, tunes the leading tree ensembles, builds a
stacked ensemble, and evaluates the winners with parity, residual, learning-curve
and permutation-importance plots. Fitted models are saved to ``models/``.

Run:  python scripts/03_train_models.py
"""

import _bootstrap  # noqa: F401

import json

import joblib
import pandas as pd

from holecleaning import config as C
from holecleaning import data, evaluation, models


def main() -> None:
    evaluation.apply_style()
    df = data.load_clean()
    X_train, X_test, y_train, y_test = data.split(df)

    # ---- 1. Benchmark the whole zoo --------------------------------------- #
    zoo = models.model_zoo()
    zoo["Stacked"] = models.build_stack()
    bench = evaluation.benchmark(zoo, X_train, y_train)
    bench.to_csv(C.TABLE_DIR / "model_benchmark.csv", index=False)
    evaluation.plot_benchmark(bench)
    print("Cross-validated benchmark (train folds):")
    print(bench.to_string(index=False))

    # ---- 2. Tune the top tree ensembles ----------------------------------- #
    spaces = models.SearchSpaces()
    from sklearn.ensemble import (GradientBoostingRegressor,
                                  RandomForestRegressor)

    rf_search = models.random_search(
        RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1),
        spaces.random_forest, X_train, y_train, n_iter=30)
    gb_search = models.random_search(
        GradientBoostingRegressor(random_state=C.RANDOM_STATE),
        spaces.gradient_boosting, X_train, y_train, n_iter=30)

    tuned = {
        "RandomForest": rf_search.best_estimator_,
        "GradientBoosting": gb_search.best_estimator_,
        "Stacked": models.build_stack(),
    }
    tuned_params = {
        "RandomForest": rf_search.best_params_,
        "GradientBoosting": gb_search.best_params_,
    }
    # numpy scalars aren't JSON-serialisable; coerce to native types while
    # preserving int-vs-float (``.item()`` returns the correct Python scalar).
    def _native(v):
        return v.item() if hasattr(v, "item") else v

    tuned_params = {k: {kk: _native(vv) for kk, vv in v.items()}
                    for k, v in tuned_params.items()}
    with open(C.MODEL_DIR / "tuned_hyperparams.json", "w") as fh:
        json.dump(tuned_params, fh, indent=2)
    print("\nBest RF params:", tuned_params["RandomForest"])
    print("Best GB params:", tuned_params["GradientBoosting"])

    # ---- 3. Evaluate and persist the winners ------------------------------ #
    holdout_rows = []
    for name, model in tuned.items():
        rep = evaluation.evaluate_fitted(model, X_train, y_train, X_test, y_test)
        holdout_rows.append(rep.loc["test"].rename(name))

        tr_pred = model.predict(X_train)
        te_pred = model.predict(X_test)
        evaluation.plot_parity(y_train, tr_pred, y_test, te_pred,
                               model_name=name,
                               filename=f"parity_{name.lower()}.png")
        evaluation.plot_residuals(y_test, te_pred, model_name=name,
                                  filename=f"residuals_{name.lower()}.png")
        evaluation.plot_learning_curve(model, df.drop(columns=[C.TARGET]),
                                       df[C.TARGET], model_name=name,
                                       filename=f"learning_curve_{name.lower()}.png")
        joblib.dump(model, C.MODEL_DIR / f"{name.lower()}.joblib")
        print(f"  saved model + plots for {name}")

    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(C.TABLE_DIR / "holdout_metrics.csv")
    print("\nHold-out test metrics:")
    print(holdout.round(4).to_string())

    # ---- 4. Permutation importance on the champion ------------------------ #
    champion_name = holdout["RMSE"].idxmin()
    champion = tuned[champion_name]
    _, imp = evaluation.plot_permutation_importance(
        champion, df.drop(columns=[C.TARGET]), df[C.TARGET],
        model_name=champion_name, filename="permutation_importance.png")
    imp.to_csv(C.TABLE_DIR / "permutation_importance.csv", header=["importance"])
    joblib.dump(champion, C.MODEL_DIR / "champion.joblib")
    with open(C.MODEL_DIR / "champion.txt", "w") as fh:
        fh.write(champion_name)
    print(f"\nChampion model: {champion_name}  (saved as models/champion.joblib)")
    print("Permutation importance:")
    print(imp.round(4).to_string())


if __name__ == "__main__":
    main()
