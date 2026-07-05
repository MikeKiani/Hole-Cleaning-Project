"""
Stage 2 — Exploratory analysis, including the density and fluid-velocity studies.

Produces:
  * correlation_matrix.png
  * feature_distributions.png
  * target_relationships.png
  * feature_ranking.png            (+ reports/tables/feature_ranking.csv)
  * density_analysis.png           (+ reports/tables/density_analysis.csv)
  * fluid_velocity_analysis.png    (+ reports/tables/velocity_analysis.csv)

Run:  python scripts/02_run_eda.py
"""

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import seaborn as sns

from holecleaning import config as C
from holecleaning import data, eda, evaluation, features


def density_analysis(df):
    tbl = features.density_table(df)
    tbl.to_csv(C.TABLE_DIR / "density_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(tbl["Density"].astype(str), tbl["mean_concentration"],
           yerr=tbl["std_concentration"].fillna(0), capsize=6,
           color=sns.color_palette("mako", len(tbl)), edgecolor="black")
    ax.set_xlabel(f"Mud weight / density [{C.UNITS['Density']}]")
    ax.set_ylabel("Mean cuttings concentration")
    ax.set_title("Density analysis — heavier mud lifts more cuttings",
                 **C.STYLE.title_kw)
    for i, (m, n) in enumerate(zip(tbl["mean_concentration"], tbl["n_runs"])):
        ax.text(i, m, f"{m:.3f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.FIGURE_DIR / "density_analysis.png", bbox_inches="tight")
    print("density_analysis.png written")
    return tbl


def fluid_velocity_analysis(df):
    tbl = features.velocity_table(df)
    tbl.to_csv(C.TABLE_DIR / "velocity_analysis.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(tbl["annular_velocity_ft_s"], tbl["mean_concentration"],
                yerr=tbl["std_concentration"].fillna(0), fmt="o-",
                capsize=5, color="#1f77b4", lw=2, markersize=8)
    ax.set_xlabel("Annular velocity [ft/s]")
    ax.set_ylabel("Mean cuttings concentration")
    ax.set_title("Fluid-velocity analysis — the primary cleaning lever",
                 **C.STYLE.title_kw)
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda v: v / C.GPM_TO_FT3S * C.GEOMETRY.annular_area_ft2,
                   lambda q: features.annular_velocity(q)),
    )
    secax.set_xlabel("Flow rate [gpm]")
    for _, r in tbl.iterrows():
        ax.annotate(f"{int(r['Flow rate'])} gpm",
                    (r["annular_velocity_ft_s"], r["mean_concentration"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURE_DIR / "fluid_velocity_analysis.png", bbox_inches="tight")
    print("fluid_velocity_analysis.png written")
    print(f"\nAssumed geometry: {C.GEOMETRY.label} "
          f"({C.GEOMETRY.hole_diameter_in} in × {C.GEOMETRY.pipe_diameter_in} in, "
          f"A = {C.GEOMETRY.annular_area_ft2:.4f} ft²)")
    return tbl


def main() -> None:
    evaluation.apply_style()
    df = data.load_clean()

    eda.correlation_matrix(df)
    print("correlation_matrix.png written")
    eda.feature_distributions(df)
    print("feature_distributions.png written")
    eda.target_relationships(df)
    print("target_relationships.png written")

    ranking = eda.feature_ranking(df)
    ranking.to_csv(C.TABLE_DIR / "feature_ranking.csv")
    print("feature_ranking.png written\n")
    print("Feature ranking (Pearson vs Gini):")
    print(ranking.to_string())
    print()

    print("=" * 60)
    print("DENSITY ANALYSIS")
    print("=" * 60)
    print(density_analysis(df).to_string(index=False))
    print()

    print("=" * 60)
    print("FLUID-VELOCITY ANALYSIS")
    print("=" * 60)
    print(fluid_velocity_analysis(df).to_string(index=False))


if __name__ == "__main__":
    main()
