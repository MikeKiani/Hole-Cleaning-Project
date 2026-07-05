"""
Physics-informed feature engineering for cuttings transport.

The raw inputs are the operating variables an engineer reads at surface. Cuttings
transport, however, is governed by *derived* quantities — chiefly the annular
fluid velocity and its balance against the cuttings' tendency to settle. This
module adds a small, interpretable set of engineered features grounded in
cuttings-transport theory so that both the analyst and the models can reason in
those terms.

None of these features are required by the baseline models (which work directly
on the raw inputs); they are additive and are what powers the density and
fluid-velocity analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def annular_velocity(flow_rate_gpm, geometry=C.GEOMETRY) -> np.ndarray:
    """Bulk annular velocity in ft/s.

    v = Q / A, with Q converted from gpm to ft³/s and A the annular
    cross-section from the (assumed, constant) flow-loop geometry. Because the
    geometry is fixed across the study, this is a faithful *linear* rescaling of
    flow rate into physical velocity units.
    """
    q_ft3s = np.asarray(flow_rate_gpm, dtype=float) * C.GPM_TO_FT3S
    return q_ft3s / geometry.annular_area_ft2


def add_physics_features(df: pd.DataFrame, geometry=C.GEOMETRY) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered cuttings-transport features.

    Added columns
    -------------
    Annular velocity
        Bulk fluid velocity in the annulus (ft/s).
    Transport index
        Annular velocity scaled by inclination severity. Cuttings settle onto
        the low side of a deviated hole, so the *effective* lift decreases as
        the hole approaches horizontal; this multiplies velocity by cos(90-θ)
        as a first-order lift-efficiency factor.
    Carrying capacity index (CCI)
        A dimensionless proxy combining velocity, mud weight and rheology in the
        spirit of the classic API carrying-capacity index. Higher CCI implies
        better hole cleaning. Kept as a *relative* index, not an absolute API
        value, since the constants depend on units not fixed here.
    Rotation factor
        Pipe rotation mobilises the cuttings bed; encoded as a simple on/off
        enhancement to lift because the study uses only two rotation levels.
    """
    out = df.copy()

    out["Annular velocity"] = annular_velocity(out["Flow rate"], geometry)

    incl_rad = np.deg2rad(out["Inclination"])
    lift_efficiency = np.cos(np.deg2rad(90.0) - incl_rad)  # 1 at vertical-ish
    # Guard against zero when perfectly horizontal.
    lift_efficiency = np.clip(lift_efficiency, 1e-3, None)
    out["Transport index"] = out["Annular velocity"] * lift_efficiency

    # Relative carrying-capacity index: velocity * mud weight * (rheology / 100)
    # normalised to its own mean so it reads as "x times the study average".
    cci = out["Annular velocity"] * out["Density"] * (1.0 + out["YP"] / 100.0)
    out["Carrying capacity index"] = cci / cci.mean()

    out["Rotation factor"] = np.where(out["Pipe rotation"] > 0, 1.0, 0.0)

    return out


def velocity_table(df: pd.DataFrame, geometry=C.GEOMETRY) -> pd.DataFrame:
    """Map each unique flow-rate level to its annular velocity and the mean
    observed concentration at that level — the backbone of the fluid-velocity
    analysis."""
    tmp = df.copy()
    tmp["Annular velocity"] = annular_velocity(tmp["Flow rate"], geometry)
    grp = (
        tmp.groupby("Flow rate")
        .agg(
            annular_velocity_ft_s=("Annular velocity", "first"),
            mean_concentration=(C.TARGET, "mean"),
            std_concentration=(C.TARGET, "std"),
            n_runs=(C.TARGET, "size"),
        )
        .reset_index()
        .sort_values("Flow rate")
    )
    return grp.round(4)


def density_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean cuttings concentration grouped by mud-weight level — the backbone of
    the density analysis."""
    grp = (
        df.groupby("Density")
        .agg(
            mean_concentration=(C.TARGET, "mean"),
            std_concentration=(C.TARGET, "std"),
            min_concentration=(C.TARGET, "min"),
            max_concentration=(C.TARGET, "max"),
            n_runs=(C.TARGET, "size"),
        )
        .reset_index()
        .sort_values("Density")
    )
    return grp.round(4)
