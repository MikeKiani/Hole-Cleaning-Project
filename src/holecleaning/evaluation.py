"""
Model evaluation and visualization.

Provides three things:
  1. A consistent metric set (MAE, RMSE, R²).
  2. A cross-validated benchmark that ranks the whole model zoo fairly.
  3. A library of publication-quality plots (parity, residuals, learning
     curves, permutation importance) that all share the project style.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_validate, learning_curve

from . import config as C


def apply_style(style=C.STYLE) -> None:
    """Apply the shared seaborn/matplotlib style once per script."""
    sns.set_theme(context=style.context, style=style.style, palette=style.palette)
    plt.rcParams["figure.dpi"] = style.dpi
    plt.rcParams["savefig.dpi"] = style.dpi
    plt.rcParams["axes.titleweight"] = "bold"


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def metrics(y_true, y_pred) -> dict[str, float]:
    """MAE (also as %), RMSE and R² in one dict."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAE_pct": mean_absolute_error(y_true, y_pred) * 100.0,
        "RMSE": _rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Cross-validated benchmark
# --------------------------------------------------------------------------- #
def benchmark(models: dict, X, y, cv=C.CV_FOLDS) -> pd.DataFrame:
    """Cross-validate every model in ``models`` and return a ranked table.

    Reports the mean and standard deviation of test-fold RMSE, MAE and R², so
    the comparison reflects generalisation rather than a single lucky split.
    """
    rows = []
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    for name, model in models.items():
        cvres = cross_validate(
            model, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
        )
        rows.append(
            {
                "Model": name,
                "RMSE": -cvres["test_rmse"].mean(),
                "RMSE_std": cvres["test_rmse"].std(),
                "MAE": -cvres["test_mae"].mean(),
                "R2": cvres["test_r2"].mean(),
                "R2_std": cvres["test_r2"].std(),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("RMSE")
        .reset_index(drop=True)
        .round(4)
    )


def evaluate_fitted(model, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Train/test metric table for an already-instantiated model."""
    model.fit(X_train, y_train)
    tr = metrics(y_train, model.predict(X_train))
    te = metrics(y_test, model.predict(X_test))
    return pd.DataFrame({"train": tr, "test": te}).T.round(4)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save(fig, filename):
    if filename:
        path = C.FIGURE_DIR / filename
        fig.savefig(path, bbox_inches="tight")
        return path
    return None


def plot_benchmark(bench: pd.DataFrame, filename="model_benchmark.png"):
    """Horizontal bar chart of cross-validated RMSE with error bars."""
    fig, ax = plt.subplots(figsize=(9, 6))
    order = bench.sort_values("RMSE", ascending=True)
    colors = sns.color_palette("crest", len(order))
    ax.barh(order["Model"], order["RMSE"], xerr=order["RMSE_std"],
            color=colors, edgecolor="black", linewidth=0.6)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated RMSE (lower is better)")
    ax.set_title("Model benchmark — 5-fold cross-validation", **C.STYLE.title_kw)
    for i, (rmse, r2) in enumerate(zip(order["RMSE"], order["R2"])):
        ax.text(rmse + order["RMSE_std"].max() * 0.15, i,
                f"R²={r2:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_parity(y_train, tr_pred, y_test, te_pred, model_name="",
                filename=None):
    """Observed-vs-predicted parity plot for train and test in one figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(13, 5.5))
    fig.suptitle(model_name, fontsize=15, fontweight="bold")
    lo = float(min(np.min(y_train), np.min(y_test))) - 0.02
    hi = float(max(np.max(y_train), np.max(y_test))) + 0.02

    for ax, yt, yp, title in [
        (ax1, y_train, tr_pred, "Training set"),
        (ax2, y_test, te_pred, "Test set"),
    ]:
        ax.plot([lo, hi], [lo, hi], "--k", lw=2, label="Perfect prediction")
        ax.scatter(yt, yp, s=28, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Observed concentration")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
        r2 = r2_score(yt, yp)
        m = mean_absolute_error(yt, yp)
        ax.text(0.05, 0.9, f"R² = {r2:.3f}\nMAE = {m*100:.2f}%",
                transform=ax.transAxes, va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax1.set_ylabel("Predicted concentration")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_residuals(y_test, te_pred, model_name="", filename=None):
    """Residual diagnostics: residual-vs-fitted and residual distribution."""
    resid = np.asarray(y_test) - np.asarray(te_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.axhline(0, color="k", ls="--", lw=1)
    ax1.scatter(te_pred, resid, s=28, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax1.set_xlabel("Predicted concentration")
    ax1.set_ylabel("Residual (obs − pred)")
    ax1.set_title("Residuals vs fitted", fontsize=12, fontweight="bold")
    sns.histplot(resid, kde=True, ax=ax2, color="#4c72b0")
    ax2.axvline(0, color="k", ls="--", lw=1)
    ax2.set_xlabel("Residual")
    ax2.set_title("Residual distribution", fontsize=12, fontweight="bold")
    fig.suptitle(model_name, fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_learning_curve(model, X, y, model_name="", cv=C.CV_FOLDS, filename=None):
    """Learning curve to expose over/under-fitting given only 116 samples."""
    sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.25, 1.0, 6),
        scoring="neg_root_mean_squared_error",
    )
    train_m = -train_scores.mean(axis=1)
    val_m = -val_scores.mean(axis=1)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sizes, train_m, "o-", label="Training RMSE")
    ax.plot(sizes, val_m, "s-", label="Validation RMSE")
    ax.fill_between(sizes, train_m - train_scores.std(axis=1),
                    train_m + train_scores.std(axis=1), alpha=0.15)
    ax.fill_between(sizes, val_m - val_scores.std(axis=1),
                    val_m + val_scores.std(axis=1), alpha=0.15)
    ax.set_xlabel("Training-set size")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Learning curve — {model_name}", **C.STYLE.title_kw)
    ax.legend()
    fig.tight_layout()
    _save(fig, filename)
    return fig


def plot_permutation_importance(model, X, y, model_name="", filename=None):
    """Model-agnostic permutation importance (used in place of SHAP, which is
    not available offline). Reports the mean RMSE increase when each feature is
    shuffled — a fair, model-independent attribution."""
    model.fit(X, y)
    result = permutation_importance(
        model, X, y, n_repeats=30, random_state=C.RANDOM_STATE,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    imp = (
        pd.Series(result.importances_mean, index=X.columns)
        .sort_values(ascending=True)
    )
    err = pd.Series(result.importances_std, index=X.columns).loc[imp.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(imp.index, imp.values, xerr=err.values,
            color=sns.color_palette("flare", len(imp)),
            edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Increase in RMSE when feature is permuted")
    ax.set_title(f"Permutation importance — {model_name}", **C.STYLE.title_kw)
    fig.tight_layout()
    _save(fig, filename)
    return fig, imp.sort_values(ascending=False)
