"""Standalone plotting for example_3's well/fracture monitoring CSVs.

Reads directly from the ``well_monitoring/*.csv`` files written by
``CosoExporter.save_results()`` -- no model re-instantiation needed, so this can be
pointed at a run's output folder at any time (including while it is still running).

``exporting.py``'s ``plot_well_monitoring``/``plot_flow_rate_and_fracture_displacement``
were written for the synthetic two-fracture geometries of examples 2/4: they hardcode a
"conductive"/"blocking" two-fracture labeling and cap colors at three. Example 3 uses
``FaultPlaneGeometry`` with real fault-plane data -- currently 4 active faults after
``exclude_faults``, but that count is a configuration choice (``exclude_faults`` in
``run_example_3.py``), not a fixed property of the geometry. The functions here instead
auto-discover fracture and well names from the CSV, so they keep working unchanged if
faults are added, removed, or renamed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from exporting import plot_fracture_displacement


def _monotonic_time_mask(time: pd.Series) -> np.ndarray:
    """Row indices of a monotonically increasing subsequence of ``time``.

    Guards against non-monotonic entries in the collected data (e.g. from a
    rejected/retried time step attempt). Mirrors the cleanup already done ad hoc in
    ``exporting.plot_flow_rate_and_fracture_displacement``.
    """
    accepted: list[float] = []
    inds: list[int] = []
    for i, t in enumerate(time):
        while accepted and t < accepted[-1]:
            accepted.pop()
            inds.pop()
        accepted.append(t)
        inds.append(i)
    return np.array(inds)


def load_monitoring_data(
    csv_dir: Path | str, file_base: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the well and fracture monitoring CSVs written by ``save_results()``."""
    csv_dir = Path(csv_dir)
    well_df = pd.read_csv(csv_dir / f"{file_base}.csv")
    fracture_df = pd.read_csv(csv_dir / f"{file_base}_fractures.csv")
    return well_df, fracture_df


def plot_all_fracture_displacements(
    csv_dir: Path | str, file_base: str, title: str | None = None, semilogy: bool = True
):
    """Displacement jump vs time for every fracture found in the CSV.

    Thin wrapper around ``exporting.plot_fracture_displacement``, which is already
    fracture-count-agnostic (it discovers ``fracture_id`` values from the CSV).
    """
    return plot_fracture_displacement(csv_dir, file_base, title=title, semilogy=semilogy)


def plot_wells(csv_dir: Path | str, file_base: str, title: str | None = None) -> None:
    """Per-well pressure/temperature and fluid_flux/enthalpy_flux plots.

    Standalone equivalent of ``CosoExporter.plot_well_monitoring``'s per-well loop,
    without needing a live model instance: well names and whether a temperature trace
    exists are auto-detected from the CSV columns/values instead of being read off
    ``self.injection_well_names``/``self.production_well_names``/``hasattr(self, ...)``.
    """
    csv_dir = Path(csv_dir)
    well_df, _ = load_monitoring_data(csv_dir, file_base)
    well_names = well_df["well_name"].unique().tolist()
    has_temperature = "temperature" in well_df.columns

    if has_temperature:
        variable_pairs = (("pressure", "temperature"), ("enthalpy_flux", "fluid_flux"))
        unit_pairs = ((r"Pa", r"K"), (r"W/m$^2$", r"kg/s"))
    else:
        variable_pairs = (("pressure",), ("fluid_flux",))
        unit_pairs = ((r"Pa",), (r"kg/s",))

    for name in well_names:
        data = well_df[well_df["well_name"] == name].copy()
        inds = _monotonic_time_mask(data["time"])
        data = data.iloc[inds].reset_index(drop=True)
        prefix = "injection" if "Injection" in name else "production"

        for var_pair, unit_pair in zip(variable_pairs, unit_pairs):
            fig, ax1 = plt.subplots()
            color1 = "tab:blue"
            var_name0 = var_pair[0].replace("_", " ").capitalize()
            ax1.set_xlabel("Time (d)")
            ax1.set_ylabel(f"{var_name0} ({unit_pair[0]})", color=color1)
            ax1.plot(
                data["time"] / 86400.0,
                data[var_pair[0]],
                color=color1,
                label=var_pair[0],
            )
            ax1.tick_params(axis="y", labelcolor=color1)
            suffix = "_fluxes" if "flux" in var_pair[0] else ""
            plt_name = f"{prefix}_{name}"

            if len(var_pair) > 1:
                ax2 = ax1.twinx()
                color2 = "tab:red"
                var_name1 = var_pair[1].replace("_", " ").capitalize()
                ax2.plot(
                    data["time"] / 86400.0,
                    data[var_pair[1]],
                    color=color2,
                    label=var_pair[1],
                )
                ax2.tick_params(axis="y", labelcolor=color2)
                ax2.set_ylabel(f"{var_name1} ({unit_pair[1]})")
                lines1, _ = ax1.get_legend_handles_labels()
                lines2, _ = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, [var_name0, var_name1], loc="best")
                if "flux" in var_pair[1]:
                    suffix = "_fluxes"

            plt.title(title if title is not None else f"Well data for {plt_name}")
            fig.tight_layout()
            output_path = csv_dir / f"{file_base}_{plt_name}{suffix}.png"
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Plot saved to: {output_path}")


def plot_flux_and_fracture_displacements(
    csv_dir: Path | str,
    well_name: str,
    file_base: str,
    title: str | None = None,
    semilogy: bool = True,
    out_path: Path | str | None = None,
) -> None:
    """Well fluid flux (one axis) vs displacement jump of every fracture (other axis).

    Generalizes ``exporting.plot_flow_rate_and_fracture_displacement``, which hardcodes
    a 2-fracture "conductive"/"blocking" labeling and a 3-color cap tailored to the
    synthetic two-fracture geometries of examples 2/4. Here, fracture names and colors
    scale to however many fractures are present in the CSV -- nothing assumes a
    particular count, so this keeps working if the active fault set changes.
    """
    csv_dir = Path(csv_dir)
    well_df, fracture_df = load_monitoring_data(csv_dir, file_base)

    well_data = well_df[well_df["well_name"] == well_name].copy()
    if well_data.empty:
        raise ValueError(f"No data found for well {well_name}.")
    well_data["fluid_flux_numeric"] = well_data["fluid_flux"].apply(
        lambda x: float(str(x).strip("[]"))
    )
    inds = _monotonic_time_mask(well_data["time"])
    well_data = well_data.iloc[inds].reset_index(drop=True)
    sign = 1 if "Production" in well_name else -1
    well_data["fluid_flux_numeric"] *= sign
    time = well_data["time"]

    fracture_names = fracture_df["fracture_id"].unique().tolist()
    cmap = plt.get_cmap("tab10")

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color1 = "tab:blue"
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Fluid flux (kg/s)", color=color1, fontsize=12)
    ax1.tick_params(axis="y", labelcolor=color1)
    line1 = ax1.plot(
        time,
        well_data["fluid_flux_numeric"],
        color=color1,
        linewidth=2,
        label=f"Fluid flux ({well_name})",
    )

    ax2 = ax1.twinx()
    ax2.set_ylabel("Displacement jump (m)", fontsize=12)
    for i, fracture_name in enumerate(fracture_names):
        subset = fracture_df[fracture_df["fracture_id"] == fracture_name]
        subset = subset.iloc[inds].reset_index(drop=True)
        color = cmap(i % 10)
        plot_fn = ax2.semilogy if semilogy else ax2.plot
        plot_fn(
            time,
            subset["displacement_jump_increment"],
            color=color,
            linewidth=2,
            label=fracture_name,
        )

    lines1 = line1
    lines2 = ax2.get_lines()
    ax1.legend(
        lines1 + lines2,
        [l.get_label() for l in lines1 + lines2],
        loc="best",
        fontsize=9,
        framealpha=0.9,
        edgecolor="black",
    )
    plt.title(
        title
        if title is not None
        else f"{well_name}: fluid flux and fracture displacement",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = (
        Path(out_path)
        if out_path is not None
        else csv_dir / f"{file_base}_flux_and_displacement_{well_name}.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {output_path}")


def plot_example_3_results(
    velocity: float,
    period: float,
    with_production: bool = True,
    use_iterative_solver: bool = True,
    semilogy: bool = True,
) -> None:
    """Generate all monitoring plots for one example_3 (velocity, period) run.

    Looks up the run's folder naming the same way ``run_example_3.py`` does, so this
    can be called with the same parameters used to launch a run.
    """
    from run_example_3 import names_from_params

    _, folder_name, _, file_name, title = names_from_params(
        velocity, period, with_production, use_iterative_solver
    )
    csv_dir = Path(f"{folder_name}_saved_data") / "well_monitoring"

    plot_wells(csv_dir, file_name, title=title)
    plot_all_fracture_displacements(csv_dir, file_name, title=title, semilogy=semilogy)

    well_df, _ = load_monitoring_data(csv_dir, file_name)
    well_names = well_df["well_name"].unique().tolist()
    injection_wells = [w for w in well_names if "Injection" in w] or well_names[:1]
    for well_name in injection_wells:
        plot_flux_and_fracture_displacements(
            csv_dir, well_name, file_name, title=title, semilogy=semilogy
        )


if __name__ == "__main__":
    plot_example_3_results(velocity=0.0, period=1.0, with_production=True)
