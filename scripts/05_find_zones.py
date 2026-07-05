"""
Stage 5 — Hole-cleaning zone maps and feed-zone identification.

Sweeps the champion model across pairs of dominant drivers to build coloured
zone maps (Efficient / Marginal / Poor) and extracts the poor-cleaning
"feed zones" — the operating regions where a stationary cuttings bed is
predicted to accumulate.

Produces:
  * reports/figures/hole_cleaning_zone_map_<x>_<y>.png
  * reports/tables/zone_summary.csv
  * reports/tables/feed_zones.csv

Run:  python scripts/05_find_zones.py   (requires stage 3 to have run)
"""

import _bootstrap  # noqa: F401

import joblib
import pandas as pd

from holecleaning import config as C
from holecleaning import data, evaluation, zones


def main() -> None:
    evaluation.apply_style()
    df = data.load_clean()

    champion_path = C.MODEL_DIR / "champion.joblib"
    if not champion_path.exists():
        raise SystemExit("champion.joblib not found — run 03_train_models.py first.")
    model = joblib.load(champion_path)
    champ_name = (C.MODEL_DIR / "champion.txt").read_text().strip()
    model.fit(df.drop(columns=[C.TARGET]), df[C.TARGET])

    # ---- Observed zone distribution --------------------------------------- #
    summary = zones.zone_summary(df)
    summary.to_csv(C.TABLE_DIR / "zone_summary.csv")
    print("Observed hole-cleaning zone distribution:")
    print(summary.to_string())
    print()

    # ---- Zone maps over the two most operationally relevant driver pairs -- #
    pairs = [
        ("Flow rate", "Inclination"),
        ("Flow rate", "Density"),
    ]
    feed_rows = []
    for x_feat, y_feat in pairs:
        grid, XX, YY, ZZ = zones.build_zone_map(model, df, x_feat, y_feat)
        fname = f"hole_cleaning_zone_map_{x_feat}_{y_feat}.png".replace(" ", "")
        zones.plot_zone_map(XX, YY, ZZ, x_feature=x_feat, y_feature=y_feat,
                            model_name=champ_name, overlay_df=df, filename=fname)
        feed = zones.extract_feed_zones(grid, x_feat, y_feat)
        feed.insert(0, "swept_axes", f"{x_feat} × {y_feat}")
        feed_rows.append(feed)
        print(f"Zone map written: {fname}")

    feed_zones = pd.concat(feed_rows, ignore_index=True)
    feed_zones.to_csv(C.TABLE_DIR / "feed_zones.csv", index=False)
    print("\nIdentified feed (poor-cleaning) zones:")
    print(feed_zones.to_string(index=False))


if __name__ == "__main__":
    main()
