"""
Exploratory data analysis.

Correlation structure, univariate distributions, target relationships and a
dual feature-ranking (Pearson vs. Gini) that reproduces and extends the original
study's ranking step.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

from . import config as C


def _save(fig, filename):
    if filename:
        fig.savefig(C.FIGURE_DIR / filename, bbox_inches="tight")


def correlation_matrix(df: pd.DataFrame, filename="correlation_matrix.png"):
    """Lower-triangle Pearson correlation heatmap."""
    corr = df.corr(numeric_only=True).round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, center=0, mask=mask, linewidths=0.5,
                cmap="RdYlGn", vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation matrix", **C.STYLE.title_kw)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, filename)
    return fig, corr


def feature_distributions(df: pd.DataFrame, filename="feature_distributions.png"):
    """Grid of histograms for every column, target included."""
    cols = list(df.columns)
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.2 * nrow))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(df[col], kde=True, ax=ax, color="#4c72b0")
        unit = C.UNITS.get(col, "")
        ax.set_title(f"{col}" + (f" [{unit}]" if unit and unit != "-" else ""),
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("")
    for ax in axes.ravel()[len(cols):]:
        ax.set_visible(False)
    fig.suptitle("Feature & target distributions", fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def target_relationships(df: pd.DataFrame, target=C.TARGET,
                         filename="target_relationships.png"):
    """Trend of mean concentration against each input — reveals the monotone
    drivers (flow rate down, inclination up, etc.)."""
    features = [c for c in df.columns if c != target]
    ncol = 3
    nrow = int(np.ceil(len(features) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3.4 * nrow))
    for ax, col in zip(axes.ravel(), features):
        grp = df.groupby(col)[target].mean()
        ax.plot(grp.index, grp.values, "o-", color="#c44e52")
        ax.set_xlabel(f"{col} [{C.UNITS.get(col, '')}]")
        ax.set_ylabel(f"mean {target}")
        ax.set_title(col, fontsize=11, fontweight="bold")
    for ax in axes.ravel()[len(features):]:
        ax.set_visible(False)
    fig.suptitle("Mean cuttings concentration vs. each operating variable",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, filename)
    return fig


def feature_ranking(df: pd.DataFrame, target=C.TARGET,
                    filename="feature_ranking.png") -> pd.DataFrame:
    """Dual ranking: absolute Pearson correlation and Random-Forest Gini
    importance, shown side by side. Agreement between the two raises confidence
    in the identified drivers."""
    X = df.drop(columns=[target])
    y = df[target]

    pearson = X.apply(lambda s: np.corrcoef(s, y)[0, 1]).abs()
    rf = RandomForestRegressor(n_estimators=400, random_state=C.RANDOM_STATE,
                               n_jobs=-1).fit(X, y)
    gini = pd.Series(rf.feature_importances_, index=X.columns)

    ranking = pd.DataFrame({
        "Pearson_|r|": pearson,
        "Gini_importance": gini,
    }).sort_values("Gini_importance", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    p = pearson.sort_values()
    g = gini.sort_values()
    ax1.barh(p.index, p.values, color=sns.color_palette("Blues_r", len(p)),
             edgecolor="black", linewidth=0.5)
    ax1.set_title("Pearson |r| with target", fontsize=12, fontweight="bold")
    ax2.barh(g.index, g.values, color=sns.color_palette("Greens_r", len(g)),
             edgecolor="black", linewidth=0.5)
    ax2.set_title("Random-Forest Gini importance", fontsize=12, fontweight="bold")
    fig.suptitle("Feature ranking — two independent methods",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, filename)
    return ranking.round(4)
