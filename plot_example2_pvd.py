"""PVD-based broken-axis plots for Case II simulations.

Produces per-simulation figures showing binned fracture quantities (average
pressure, critical pressure for slip, slip tendency) on the same broken time
axis used by :func:`plot_seismic_moment_and_flux.plot_seismic_moment_and_flux`.

Run directly::

    python case2_pvd_plots.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import porepy as pp

import run_example_2
from load_pvd_data import load_from_pvd
from plot_seismic_moment_and_flux import get_intervals
from plot_pvd_quantities import (
    Panel,
    Quantity,
    build_panels,
    collect_pvd_inputs,
    fracture_subdomain_id_map,
    plot_broken_axis_panels,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_BINS_PER_INTERVAL = 4
# Fracture IDs present in Case II.
FRACTURE_IDS = ["Injection", "Production"]
# Output directory for saved figures.
FIGURES_DIR = Path("figures") / "pvd_quantities"

# Quantities to read from the fracture-dimension PVD cell data.
# The VTU files must contain these arrays (they are written by data_to_export).
#
# Each entry may be either:
#   - a shorthand tuple  ``(pvd_quantity_name, ylabel)``
#   - a :class:`~plot_pvd_quantities.Panel` with one or more
#     :class:`~plot_pvd_quantities.Quantity` objects for derived quantities,
#     per-fracture min/max bands, etc.


def _critical_pressure(sd_data: dict) -> np.ndarray:
    """Cell-wise critical pore pressure for slip (Pa).

    Uses the Mohr-Coulomb criterion:
    ``p_crit = -(t_n + |tau| / mu)`` where ``t_n`` is the normal contact
    traction (negative = compression).

    The friction coefficient ``mu`` is **not** assumed; it is back-computed
    from the saved ``contact_traction_in_Pa`` and ``slip_tendency`` arrays
    using the definition ``slip_tendency = |tau| / (|t_n| * mu)``
    → ``mu = |tau| / (|t_n| * slip_tendency)``.  In PorePy the coefficient
    is spatially uniform (set once from the initialization run), so the
    median over all in-contact cells gives an exact recovery (std ≈ 0).

    Open fracture cells (``t_n >= 0``) are masked to NaN: the Mohr-Coulomb
    formula is only meaningful where the surfaces are in contact.

    Parameters:
        sd_data: Per-subdomain data dict with keys
            ``"contact_traction_in_Pa"`` (T, n_cells, nd) and
            ``"slip_tendency"`` (T, n_cells).

    Returns:
        Array of shape ``(T, n_cells)`` with NaN for open cells.
    """
    t = sd_data["contact_traction_in_Pa"]  # (T, n_cells, nd)
    st = sd_data["slip_tendency"]  # (T, n_cells)
    t_n = t[..., -1]  # normal traction (T, n_cells), < 0 when in contact
    tau = np.linalg.norm(t[..., :-1], axis=-1)  # shear magnitude (T, n_cells)

    # Back-compute mu from in-contact cells where slip tendency is defined.
    in_contact = t_n < 0
    valid = in_contact & (st > 1e-6)
    mu_vals = np.where(valid, tau / (-t_n * st), np.nan)
    mu = float(np.nanmedian(mu_vals))  # uniform constant → exact recovery

    p_crit = np.where(in_contact, -(t_n + tau / mu), np.nan)
    return p_crit


def _flux_vector_to_magnitude(sd_data: dict) -> np.ndarray:
    """Convert a fluid flux vector to its magnitude."""
    return np.linalg.norm(sd_data["fluid_flux_vector"], axis=-1)


PVD_QUANTITIES: list[Panel | tuple[str, str]] = [
    # ("temperature", "Avg. temperature (K)"),
    # Panel(  Fracture flux!
    #     "Fluid flux magnitude (kg/(m²·s))",
    #     [
    #         Quantity(
    #             "fluid_flux_magnitude",
    #             inputs=["fluid_flux_vector"],
    #             fn=_flux_vector_to_magnitude,
    #         )
    #     ],
    # ),
    Panel(
        "Slip tendency (−)",
        [
            # Quantity("mean", inputs=["slip_tendency"]),
            # Quantity(
            #     "min", inputs=["slip_tendency"], spatial_fn=np.nanmin, linestyle="--"
            # ),
            Quantity(
                "max", inputs=["slip_tendency"], spatial_fn=np.nanmax, linestyle=":"
            ),
        ],
    ),
    Panel(
        "Fracture pressure (Pa)",
        [
            Quantity("mean", inputs=["pressure"]),
            # Quantity("min", inputs=["pressure"], spatial_fn=np.nanmin, linestyle="--"),
            # Quantity("max", inputs=["pressure"], spatial_fn=np.nanmax, linestyle=":"),
        ],
    ),
    Panel(
        "Critical pressure for slip (Pa)",
        [
            Quantity(
                "min",
                inputs=["contact_traction_in_Pa", "slip_tendency"],
                fn=_critical_pressure,
                spatial_fn=np.nanmin,
            ),
        ],
    ),
]

# Dimension of the fracture subdomains (nd-1).
FRACTURE_DIM: int = 2


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def make_case2_plots(
    fracture_ids: list[str] = FRACTURE_IDS,
    num_bins_per_interval: int = NUM_BINS_PER_INTERVAL,
    shut_in_duration: float = run_example_2.SHUT_IN_DURATION,
    pvd_quantities: list[Panel | tuple[str, str]] = PVD_QUANTITIES,
    fracture_subdomain_ids: dict[int, str] | None = None,
    out_dir: Path | str = FIGURES_DIR,
    binned: bool = True,
) -> None:
    """Generate broken-axis panels for every Case II simulation.

    Iterates over the parameter grid defined in :mod:`run_example_2`
    (``boundary_velocities`` × ``production_periods`` × ``with_production``)
    and produces one figure per simulation.  All quantities are read from the
    PVD/VTU files written by the simulation.

    Parameters:
        fracture_ids: Fracture IDs to use as series labels, in the same order
            as the sorted subdomain IDs found in the PVD data.
        num_bins_per_interval: Number of equal bins per sub-interval.
        shut_in_duration: Shut-in duration in seconds.
        pvd_quantities: List of ``(pvd_quantity_name, ylabel)`` pairs to plot
            from the fracture-dimension cell data.
        fracture_subdomain_ids: Explicit ``{subdomain_id: label}`` mapping.
            If ``None`` (default), the IDs are resolved automatically by
            sorting the subdomain IDs found in the PVD data and pairing them
            with *fracture_ids* in that order.
        out_dir: Directory where figures are saved.
        binned: If ``True`` (default) each interval is time-averaged into
            *num_bins_per_interval* equal buckets.  If ``False`` every
            converged timestep is plotted at its real time position.
    """
    out_dir = Path(out_dir)
    n_bins = num_bins_per_interval
    qty_suffix = "_pvd_quantities" if binned else "_pvd_quantities_raw"

    for with_prod in run_example_2.with_production:
        for velocity in run_example_2.boundary_velocities:
            for period in run_example_2.production_periods:
                production_period = period * pp.YEAR
                simulation_name, folder_name, _, file_name, title = (
                    run_example_2.names_from_params(velocity, period, with_prod)
                )

                pvd_dir = Path(folder_name)
                if not pvd_dir.exists():
                    print(f"Skipping {simulation_name}: PVD directory not found.")
                    continue

                # Use the fracture CSV only to determine the time range actually
                # simulated (so intervals are trimmed to available data).
                frac_csv = (
                    Path(folder_name + "_saved_data")
                    / "well_monitoring"
                    / f"{file_name}_fractures.csv"
                )
                try:
                    max_time = pd.read_csv(frac_csv)["time"].max()
                except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
                    max_time = run_example_2.FINAL_TIME

                intervals = get_intervals(production_period, shut_in_duration, max_time)
                if not intervals:
                    print(f"Skipping {simulation_name}: no completed intervals.")
                    continue

                try:
                    subdomains, _ = load_from_pvd(
                        pvd_dir,
                        collect_pvd_inputs(pvd_quantities) + ["subdomain_id"],
                        pvd_name=file_name,
                        subdomain_dimensions=[FRACTURE_DIM],
                    )
                    pvd_dim_data = subdomains.get(FRACTURE_DIM)
                except Exception as exc:
                    print(f"  PVD load failed for {simulation_name}: {exc}")
                    pvd_dim_data = None

                if pvd_dim_data is None:
                    print(f"Skipping {simulation_name}: no fracture PVD data.")
                    continue

                if fracture_subdomain_ids is None:
                    try:
                        resolved_ids = fracture_subdomain_id_map(
                            pvd_dim_data, fracture_ids
                        )
                    except ValueError as exc:
                        print(f"  Subdomain ID mapping failed: {exc}")
                        continue
                else:
                    resolved_ids = fracture_subdomain_ids

                panels = build_panels(
                    pvd_dim_data,
                    resolved_ids,
                    intervals,
                    n_bins,
                    pvd_quantities,
                    binned=binned,
                )

                if not panels:
                    print(f"No panels to plot for {simulation_name}, skipping.")
                    continue

                # --- Combined figure (all fractures) ---
                # out_path = out_dir / f"{simulation_name}{qty_suffix}.png"
                # plot_broken_axis_panels(
                #     panels=panels,
                #     intervals=intervals,
                #     n_bins=n_bins,
                #     title=title,
                #     out_path=out_path,
                # )

                # --- Per-fracture figures (PVD already in memory) ---
                for sd_id, frac_name in resolved_ids.items():
                    # For now, skip injection.
                    if frac_name.lower() == "injection":
                        continue
                    single_id_map = {sd_id: frac_name}
                    frac_panels = build_panels(
                        pvd_dim_data,
                        single_id_map,
                        intervals,
                        n_bins,
                        pvd_quantities,
                        binned=binned,
                    )
                    if not frac_panels:
                        continue
                    frac_slug = frac_name.lower().replace(" ", "_")
                    frac_out = (
                        out_dir / f"{simulation_name}_{frac_slug}{qty_suffix}.png"
                    )
                    plot_broken_axis_panels(
                        panels=frac_panels,
                        intervals=intervals,
                        n_bins=n_bins,
                        title=title,  # f"{title} — {frac_name}",
                        out_path=frac_out,
                    )


if __name__ == "__main__":
    make_case2_plots(binned=False)
