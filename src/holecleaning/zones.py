"""
Hole-cleaning zone analysis  ("feed-zone" identification).

The optimiser answers "what is the best setting for one run?". This module
answers the complementary operational question: "across the operating envelope,
*where* does hole cleaning break down?".

A trained regressor is swept across a 2-D grid of the two dominant controllable
drivers (annular velocity and inclination by default). Every grid point is
predicted and then binned into three operational zones:

    Efficient  (<= 5 %)   — cuttings are transported out; no bed forms.
    Marginal   (<= 15 %)  — a thin, mobile bed; manageable with rotation/sweeps.
    Poor       (> 15 %)   — a stationary cuttings bed accumulates. These are the
                            "feed zones" that continuously feed cuttings into the
                            annulus faster than they are removed, driving pack-off,
                            stuck-pipe and high-ECD risk.

The output is both a coloured zone map and a table of the poor-cleaning regions,
which is the actionable deliverable for a drilling programme.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from . import config as C
from .features import annular_velocity


def classify(concentration) -> np.ndarray:
    """Map a concentration (scalar or array) to a zone label."""
    c = np.asarray(concentration, dtype=float)
    eff = C.ZONE_THRESHOLDS["efficient_max"]
    mar = C.ZONE_THRESHOLDS["marginal_max"]
    labels = np.where(c <= eff, "Efficient",
                      np.where(c <= mar, "Marginal", "Poor"))
    return labels


def zone_summary(df: pd.DataFrame, target=C.TARGET) -> pd.DataFrame:
    """Count and share of observed runs falling in each zone."""
    z = pd.Series(classify(df[target]), name="Zone")
    summary = (
        z.value_counts()
        .reindex(C.ZONE_LABELS)
        .fillna(0)
        .astype(int)
        .rename("n_runs")
        .to_frame()
    )
    summary["share_%"] = (summary["n_runs"] / summary["n_runs"].sum() * 100).round(1)
    return summary


def build_zone_map(
    model,
    df: pd.DataFrame,
    x_feature: str = "Flow rate",
    y_feature: str = "Inclination",
    resolution: int = 60,
    fixed: dict | None = None,
):
    """Predict concentration over an ``x_feature`` × ``y_feature`` grid.

    All other features are held at the study median (overridable via ``fixed``)
    so the map isolates the effect of the two swept variables. Returns a tidy
    grid dataframe plus the meshgrid arrays for plotting.
    """
    features = list(df.drop(columns=[C.TARGET]).columns)
    medians = df[features].median()
    if fixed:
        medians.update(pd.Series(fixed))

    xs = np.linspace(df[x_feature].min(), df[x_feature].max(), resolution)
    ys = np.linspace(df[y_feature].min(), df[y_feature].max(), resolution)
    XX, YY = np.meshgrid(xs, ys)

    grid = pd.DataFrame(
        np.tile(medians.values, (XX.size, 1)), columns=features
    )
    grid[x_feature] = XX.ravel()
    grid[y_feature] = YY.ravel()

    grid["Concentration"] = model.predict(grid[features])
    grid["Zone"] = classify(grid["Concentration"])
    ZZ = grid["Concentration"].values.reshape(XX.shape)
    return grid, XX, YY, ZZ


def plot_zone_map(
    XX, YY, ZZ,
    x_feature="Flow rate", y_feature="Inclination",
    model_name="", overlay_df=None, filename="hole_cleaning_zone_map.png",
    add_velocity_axis=True,
):
    """Filled-contour zone map with the three cleaning zones colour-coded."""
    eff = C.ZONE_THRESHOLDS["efficient_max"]
    mar = C.ZONE_THRESHOLDS["marginal_max"]
    cmap = ListedColormap([C.ZONE_COLORS[z] for z in C.ZONE_LABELS])
    bounds = [ZZ.min() - 1e-6, eff, mar, ZZ.max() + 1e-6]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 7))
    cf = ax.contourf(XX, YY, ZZ, levels=bounds, cmap=cmap, norm=norm, alpha=0.9)
    cont = ax.contour(XX, YY, ZZ, levels=[eff, mar], colors="black",
                      linewidths=1.2, linestyles="--")
    ax.clabel(cont, fmt={eff: f"{eff:.0%}", mar: f"{mar:.0%}"}, fontsize=9)

    if overlay_df is not None:
        ax.scatter(overlay_df[x_feature], overlay_df[y_feature],
                   c="black", s=14, alpha=0.5, label="Experimental runs")
        ax.legend(loc="upper right")

    ax.set_xlabel(f"{x_feature} [{C.UNITS.get(x_feature, '')}]")
    ax.set_ylabel(f"{y_feature} [{C.UNITS.get(y_feature, '')}]")
    ax.set_title(f"Hole-cleaning zone map — {model_name}", **C.STYLE.title_kw)

    cbar = fig.colorbar(cf, ax=ax, boundaries=bounds, ticks=[])
    for frac, label in zip([0.16, 0.5, 0.84], C.ZONE_LABELS):
        cbar.ax.text(1.5, frac, label, rotation=90, va="center", ha="left",
                     transform=cbar.ax.transAxes, fontweight="bold")

    # Optional twin axis translating flow rate -> annular velocity.
    if add_velocity_axis and x_feature == "Flow rate":
        secax = ax.secondary_xaxis(
            "top",
            functions=(lambda q: annular_velocity(q), lambda v: v),
        )
        secax.set_xlabel("Annular velocity [ft/s]")

    fig.tight_layout()
    if filename:
        fig.savefig(C.FIGURE_DIR / filename, bbox_inches="tight")
    return fig


def extract_feed_zones(grid: pd.DataFrame, x_feature: str, y_feature: str
                       ) -> pd.DataFrame:
    """Summarise the poor-cleaning ("feed") region of a zone-map grid: its
    extent in each swept variable and how severe it gets."""
    poor = grid[grid["Zone"] == "Poor"]
    if poor.empty:
        return pd.DataFrame(
            [{"note": "No poor-cleaning zone within the swept envelope."}]
        )
    return pd.DataFrame([{
        f"{x_feature}_min": round(poor[x_feature].min(), 2),
        f"{x_feature}_max": round(poor[x_feature].max(), 2),
        f"{y_feature}_min": round(poor[y_feature].min(), 2),
        f"{y_feature}_max": round(poor[y_feature].max(), 2),
        "peak_concentration": round(poor["Concentration"].max(), 3),
        "mean_concentration": round(poor["Concentration"].mean(), 3),
        "grid_share_%": round(len(poor) / len(grid) * 100, 1),
    }])
