from pathlib import Path

import numpy as np
import pandas as pd
import porepy as pp
from matplotlib import pyplot as plt

import run_example_2
import run_example_4

NUM_BINS_PER_INTERVAL = 5
SHUT_IN_DURATION = 3 * pp.DAY
FINAL_TIME = 6 * pp.YEAR
PRODUCTION_WELL = "2 Production well"
CASES_TO_PLOT = [2]


def get_intervals(
    production_period: float,
    shut_in_duration: float = SHUT_IN_DURATION,
    final_time: float = FINAL_TIME,
) -> list[tuple[float, float, str]]:
    """Return list of (start, end, kind) for every sub-interval.

    kind is ``'production'`` or ``'shut_in'``.
    """
    n = int(np.ceil(final_time / production_period))
    intervals = []
    for i in np.arange(1, n + 1):
        prod_start = (i - 1) * production_period
        prod_end = i * production_period - shut_in_duration
        intervals.append((prod_start, prod_end, "production"))
        intervals.append((prod_end, i * production_period, "shut_in"))
    return intervals


def _monotone(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filter to strictly monotone-increasing times and non-NaN values.

    Also removes the t=0 initial condition, which is set by initialisation and does
    not correspond to a converged time-stepped solution.
    """
    valid = ~np.isnan(values) & (times > 0)
    times, values = times[valid], values[valid]
    mono = np.concatenate([[True], np.diff(times) > 0])
    return times[mono], values[mono]


def bin_slip_rate(
    frac_df: pd.DataFrame,
    fracture_id: str,
    t_start: float,
    t_end: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average slip rate (m/s) in each bin for one fracture and one interval.

    Returns ``(bin_edges, rates)`` where ``rates`` has length ``n_bins``.
    """
    df = frac_df[frac_df["fracture_id"] == fracture_id]
    times, jumps = _monotone(df["time"].values, df["displacement_jump"].values)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    rates = np.full(n_bins, np.nan)
    for k in range(n_bins):
        t0, t1 = bin_edges[k], bin_edges[k + 1]
        mask = (times >= t0) & (times <= t1)
        t_bin, v_bin = times[mask], jumps[mask]
        dt_bin = t_bin[-1] - t_bin[0] if len(t_bin) >= 2 else 0.0
        if len(t_bin) >= 2 and dt_bin > 0:
            rates[k] = (v_bin[-1] - v_bin[0]) / dt_bin
        elif len(t_bin) >= 1:
            rates[k] = 0.0
    return bin_edges, rates


def bin_fluid_flux(
    well_df: pd.DataFrame,
    well_name: str,
    t_start: float,
    t_end: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean fluid flux (kg/s) in each bin for one well and one interval.

    Returns ``(bin_edges, avg_flux)`` where ``avg_flux`` has length ``n_bins``.
    """
    df = well_df[well_df["well_name"] == well_name]
    times, fluxes = _monotone(df["time"].values, df["fluid_flux"].values)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    avg_flux = np.full(n_bins, np.nan)
    for k in range(n_bins):
        t0, t1 = bin_edges[k], bin_edges[k + 1]
        mask = (times >= t0) & (times <= t1)
        if mask.sum() > 0:
            avg_flux[k] = np.nanmean(fluxes[mask])
    return bin_edges, avg_flux


def plot_slip_rate_and_flux_in_bins(
    csv_dir: Path | str,
    file_base: str,
    production_period: float,
    fracture_ids: list[str] | None = None,
    num_bins_per_interval: int = NUM_BINS_PER_INTERVAL,
    shut_in_duration: float = SHUT_IN_DURATION,
    final_time: float = FINAL_TIME,
    production_well: str = PRODUCTION_WELL,
    title: str | None = None,
    out_path: Path | str | None = None,
) -> None:
    """Plot binned slip rate (left axis) and production flux (right axis) on a broken
    time axis, using a twin-y layout.

    The time axis is broken so that every production sub-interval and every
    shut-in sub-interval occupies the same visual width regardless of actual
    duration.  Within production intervals the x-tick labels are in days;
    within shut-in intervals they are in hours.

    Parameters
    ----------
    csv_dir:
        Directory containing ``{file_base}.csv`` and ``{file_base}_fractures.csv``.
    file_base:
        Base name for the CSV files (e.g. ``"example_4"``).
    production_period:
        Duration of one full production cycle in seconds.
    fracture_ids:
        Fracture IDs to include.  Defaults to all fractures in the CSV.
    num_bins_per_interval:
        Number of equal bins per sub-interval.
    shut_in_duration:
        Duration of each shut-in period in seconds.
    final_time:
        Simulation end time in seconds (used to infer number of cycles).
    production_well:
        Well name string used to filter the well CSV.
    title:
        Optional figure title.
    out_path:
        If given, save the figure here instead of showing it.
    """
    csv_dir = Path(csv_dir)
    frac_df = pd.read_csv(csv_dir / f"{file_base}_fractures.csv")
    well_df = pd.read_csv(csv_dir / f"{file_base}.csv")

    if fracture_ids is None:
        fracture_ids = sorted(frac_df["fracture_id"].unique())

    intervals = get_intervals(production_period, shut_in_duration, final_time)
    max_time = frac_df["time"].max()
    intervals = [(s, e, k) for s, e, k in intervals if s < max_time]
    n_intervals = len(intervals)
    n_bins = num_bins_per_interval
    total_bins = n_intervals * n_bins

    # Global bar-center x positions (integer indices).
    x_all = np.arange(total_bins)

    # --- Collect binned data ---
    all_slip = {fid: np.full(total_bins, np.nan) for fid in fracture_ids}
    all_flux = np.full(total_bins, np.nan)

    for iv, (t_start, t_end, kind) in enumerate(intervals):
        sl = slice(iv * n_bins, (iv + 1) * n_bins)
        for fid in fracture_ids:
            _, rates = bin_slip_rate(frac_df, fid, t_start, t_end, n_bins)
            all_slip[fid][sl] = rates
        _, fluxes = bin_fluid_flux(well_df, production_well, t_start, t_end, n_bins)
        all_flux[sl] = fluxes

    # --- Build broken-axis tick labels ---
    xtick_pos: list[float] = []
    xtick_lbl: list[str] = []
    for iv, (t_start, t_end, kind) in enumerate(intervals):
        duration = t_end - t_start
        for k in range(n_bins + 1):
            x_edge = iv * n_bins + k - 0.5
            local_t = k / n_bins * duration
            lbl = (
                f"{local_t / pp.DAY:.0f}d"
                if kind == "production"
                else f"{local_t / 3600:.1f}h"
            )
            if not xtick_pos or abs(xtick_pos[-1] - x_edge) > 1e-9:
                xtick_pos.append(x_edge)
                xtick_lbl.append(lbl)

    # --- Figure: single axes with twinx ---
    fig_width = max(10, n_intervals * n_bins * 0.35)
    fig, ax1 = plt.subplots(figsize=(fig_width, 4))
    ax2 = ax1.twinx()

    # Slip-rate bar colours: one per fracture from tab10.
    frac_colors = plt.cm.tab10(np.linspace(0, 0.9, len(fracture_ids)))

    # Background shading and vertical separators.
    for iv, (t_start, t_end, kind) in enumerate(intervals):
        x_left = iv * n_bins - 0.5
        x_right = (iv + 1) * n_bins - 0.5
        bg = "#d6e8f7" if kind == "production" else "#fde8ce"
        ax1.axvspan(x_left, x_right, color=bg, alpha=0.6, zorder=0)
    for iv in range(1, n_intervals):
        ax1.axvline(
            iv * n_bins - 0.5, color="gray", linewidth=0.7, linestyle="--", zorder=1
        )

    # --- Left axis: slip rate bars ---
    bar_w = 0.7 / max(len(fracture_ids), 1)
    offsets = np.linspace(-0.35 + bar_w / 2, 0.35 - bar_w / 2, len(fracture_ids))
    slip_handles = []
    for fi, fid in enumerate(fracture_ids):
        bars = ax1.bar(
            x_all + offsets[fi],
            all_slip[fid],
            width=bar_w,
            color=frac_colors[fi],
            label=f"Slip rate – {fid}",
            zorder=2,
        )
        slip_handles.append(bars)
    ax1.set_ylabel("Slip rate (m/s)")
    ax1.axhline(0, color="black", linewidth=0.6)

    # --- Right axis: fluid flux line (bin mid-points) ---
    (flux_line,) = ax2.plot(
        x_all,
        all_flux,
        color="tab:green",
        linewidth=1.5,
        marker="o",
        markersize=4,
        label=f"Fluid flux – {production_well}",
        zorder=3,
    )
    ax2.set_ylabel("Fluid flux (kg/s)")

    # --- x-axis ticks ---
    ax1.set_xticks(xtick_pos)
    ax1.set_xticklabels(xtick_lbl, fontsize=6, rotation=45, ha="right")

    # Interval-type annotations along the top.
    for iv, (t_start, t_end, kind) in enumerate(intervals):
        center = iv * n_bins + n_bins / 2 - 0.5
        lbl = "Prod." if kind == "production" else "Shut-in"
        ax1.annotate(
            lbl,
            xy=(center, 1.0),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#1f5f9e" if kind == "production" else "#b85c00",
        )

    # Combined legend (slip handles + flux handle on ax1 so it stays on top).
    legend_handles = list(slip_handles) + [flux_line]
    legend_labels = [f"Slip rate – {fid}" for fid in fracture_ids] + [
        f"Fluid flux – {production_well}"
    ]
    ax1.legend(legend_handles, legend_labels, loc="upper left", fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11)

    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    if 4 in CASES_TO_PLOT:
        # --- Example 4 (Case_I) ---
        for velocity in run_example_4.boundary_velocities:
            for well_name, well_endpoint in run_example_4.cases:
                for period in run_example_4.production_periods:
                    for thermal_expansion in run_example_4.thermal_expansions:
                        production_period = period * pp.YEAR
                        simulation_name, folder_name, _, file_name, title = (
                            run_example_4.names_from_params(
                                velocity, period, well_name, thermal_expansion
                            )
                        )
                        csv_dir = Path(folder_name + "_saved_data") / "well_monitoring"
                        if not (csv_dir / f"{file_name}_fractures.csv").exists():
                            print(
                                f"Missing fracture CSV for {simulation_name}, skipping."
                            )
                            continue
                        out_path = (
                            Path("figures")
                            / "slip_rate_bins"
                            / f"{simulation_name}_slip_rate_bins.png"
                        )
                        plot_slip_rate_and_flux_in_bins(
                            csv_dir=csv_dir,
                            file_base=file_name,
                            production_period=production_period,
                            fracture_ids=["Fracture 2", "Fracture 3"],
                            num_bins_per_interval=NUM_BINS_PER_INTERVAL,
                            final_time=6 * pp.YEAR,
                            title=title,
                            out_path=out_path,
                        )
    if 2 in CASES_TO_PLOT:
        # --- Example 2 (Case_II) ---
        for velocity in run_example_2.boundary_velocities:
            for period in run_example_2.production_periods:
                production_period = period * pp.YEAR
                simulation_name, folder_name, _, file_name, title = (
                    run_example_2.names_from_params(velocity, period)
                )
                csv_dir = Path(folder_name + "_saved_data") / "well_monitoring"
                if not (csv_dir / f"{file_name}_fractures.csv").exists():
                    print(f"Missing fracture CSV for {simulation_name}, skipping.")
                    continue
                out_path = (
                    Path("figures")
                    / "slip_rate_bins"
                    / f"{simulation_name}_slip_rate_bins.png"
                )
                plot_slip_rate_and_flux_in_bins(
                    csv_dir=csv_dir,
                    file_base=file_name,
                    production_period=production_period,
                    fracture_ids=["Fracture 2"],
                    num_bins_per_interval=NUM_BINS_PER_INTERVAL,
                    final_time=10 * pp.YEAR,
                    title=title,
                    out_path=out_path,
                )
