"""
Central configuration for the hole-cleaning ML workflow.

Everything that is a *choice* (a path, a threshold, an assumed piece of flow-loop
geometry, a random seed) lives here so that the rest of the code base contains
logic only. Changing an assumption is a one-line edit in this file rather than a
hunt through notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
# ``config.py`` lives at  <root>/src/holecleaning/config.py, so the project root
# is three parents up. Resolving here means scripts work regardless of the
# directory they are launched from.
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA: Path = DATA_DIR / "raw" / "hole_cleaning_experiments.csv"
PROCESSED_DATA: Path = DATA_DIR / "processed" / "cleaned_data.csv"
FEATURED_DATA: Path = DATA_DIR / "processed" / "featured_data.csv"

REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURE_DIR: Path = REPORTS_DIR / "figures"
TABLE_DIR: Path = REPORTS_DIR / "tables"
MODEL_DIR: Path = ROOT_DIR / "models"

for _d in (DATA_DIR / "processed", FIGURE_DIR, TABLE_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Modelling constants
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.25
CV_FOLDS: int = 5

TARGET: str = "Concentration"

# Columns dropped during cleaning and the reason why.
#   * "Test No."  -> an index, carries no physical information.
#   * "PV"        -> plastic viscosity is perfectly collinear with yield point
#                    in this design of experiments (PV in {1,10,20} maps 1-to-1
#                    onto YP in {0,20,40}), so it is redundant. Keeping both
#                    would inflate importance estimates and destabilise linear
#                    models. YP is retained as the rheology representative.
DROP_COLUMNS: tuple[str, ...] = ("Test No.", "PV")

RENAME_COLUMNS: dict[str, str] = {"Density, ppg": "Density"}

# Canonical feature order used everywhere downstream.
FEATURE_ORDER: tuple[str, ...] = (
    "Density",
    "YP",
    "Temperature",
    "ROP",
    "Pipe rotation",
    "Flow rate",
    "Inclination",
    "Eccentricity",
)

# Human-readable units for plotting/reporting.
UNITS: dict[str, str] = {
    "Density": "ppg",
    "YP": "lbf/100ft²",
    "Temperature": "°F",
    "ROP": "ft/hr",
    "Pipe rotation": "rpm",
    "Flow rate": "gpm",
    "Inclination": "deg",
    "Eccentricity": "-",
    "Concentration": "fraction",
}

# --------------------------------------------------------------------------- #
# Hole-cleaning risk zones
# --------------------------------------------------------------------------- #
# Cuttings concentration is a stationary-bed volume fraction. These thresholds
# turn the continuous prediction into an actionable operational classification.
# They are engineering conventions for this study, not universal constants, and
# are exposed here so they can be tuned to a given wellbore's tolerance.
ZONE_THRESHOLDS: dict[str, float] = {
    "efficient_max": 0.05,   # <= 5 %  -> Efficient transport
    "marginal_max": 0.15,    # <= 15 % -> Marginal / watch
    # > 0.15  -> Poor cleaning (bed-accumulation / "feed" zone)
}
ZONE_LABELS: tuple[str, ...] = ("Efficient", "Marginal", "Poor")
ZONE_COLORS: dict[str, str] = {
    "Efficient": "#2a9d8f",
    "Marginal": "#e9c46a",
    "Poor": "#e76f51",
}

# --------------------------------------------------------------------------- #
# Flow-loop geometry (for fluid-velocity analysis)
# --------------------------------------------------------------------------- #
# The source experiments (Yu et al., Univ. of Tulsa low-pressure flow loop) were
# run in a fixed annular test section. Because the geometry is constant across
# every run, annular velocity is directly proportional to the reported flow
# rate. To express velocity in physical units (ft/s) rather than an arbitrary
# proxy we adopt the representative test-section geometry below. These values
# are an explicit, transparent assumption; changing them rescales the velocity
# axis but not any of the correlations or model results.
@dataclass(frozen=True)
class AnnulusGeometry:
    hole_diameter_in: float = 8.0     # cased test-section ID
    pipe_diameter_in: float = 4.5     # drill-pipe OD
    label: str = "Tulsa low-pressure flow loop (representative)"

    @property
    def annular_area_ft2(self) -> float:
        """Cross-sectional flow area of the annulus, ft²."""
        import math

        d_hole_ft = self.hole_diameter_in / 12.0
        d_pipe_ft = self.pipe_diameter_in / 12.0
        return math.pi / 4.0 * (d_hole_ft**2 - d_pipe_ft**2)


GEOMETRY = AnnulusGeometry()

# gpm -> ft³/s : 1 US gallon = 0.133681 ft³, per minute -> /60
GPM_TO_FT3S: float = 0.133681 / 60.0


@dataclass
class PlotStyle:
    """Shared matplotlib styling so every figure looks like it belongs to the
    same report."""

    context: str = "notebook"
    style: str = "whitegrid"
    palette: str = "deep"
    dpi: int = 130
    figsize: tuple[float, float] = (9.0, 6.0)
    title_kw: dict = field(
        default_factory=lambda: {"fontsize": 14, "fontweight": "bold"}
    )


STYLE = PlotStyle()
