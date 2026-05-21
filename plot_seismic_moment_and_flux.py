from pathlib import Path

import numpy as np
import pandas as pd
import porepy as pp
from matplotlib import pyplot as plt

import run_example_2
import run_example_4


CASES_TO_PLOT = [2]


def get_intervals(
    production_period: float,
    shut_in_duration: float,
    final_time: float,
) -> list[tuple[float, float, str]]:
    """Return list of (start, end, kind) for every sub-interval.

    kind is ``'production'`` or ``'shut_in'``.

    Note: when :class:`SmoothWellTransitions` is active, the last
    ``well_transition_duration`` seconds of each shut-in window are a pressure-ramp
    period (Dirichlet BC, non-zero flux).  Those seconds are still labelled
    ``'shut_in'`` here because this function does not receive the transition duration.
    The effect is confined to the final shut-in bin and is negligible when the
    transition duration is short relative to the bin width.
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

    Non-converged time steps are saved before the solver retries, so a failed step at
    time T appears *before* the eventual converged step at T (or at an intermediate
    sub-step). A simple ``np.diff(times) > 0`` filter would keep the non-converged
    value and discard the converged one. Instead we use a sliding-window approach:
    when time goes backwards, pop the last accepted point and replace it with the
    current one, so only the most recent (converged) value at each time is retained.
    """
    valid = ~np.isnan(values) & (times > 0)
    times, values = times[valid], values[valid]
    if len(times) == 0:
        return times, values
    accepted_times = [times[0]]
    accepted_values = [values[0]]
    for t, v in zip(times[1:], values[1:]):
        if t < accepted_times[-1]:
            # Time went backwards — the previous point was from a failed step; discard.
            accepted_times.pop()
            accepted_values.pop()
        accepted_times.append(t)
        accepted_values.append(v)
    return np.array(accepted_times), np.array(accepted_values)


def bin_moment_rate(
    frac_df: pd.DataFrame,
    fracture_id: str,
    t_start: float,
    t_end: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average seismic moment rate (N·m/s) in each bin for one fracture and one interval.

    Returns ``(bin_edges, rates)`` where ``rates`` has length ``n_bins``.
    """
    df = frac_df[frac_df["fracture_id"] == fracture_id]
    times, jumps = _monotone(df["time"].values, df["seismic_moment"].values)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    rates = np.full(n_bins, np.nan)
    for k in range(n_bins):
        t0, t1 = bin_edges[k], bin_edges[k + 1]
        # Inclusive on both sides: we need the value at t0 as well as t1 to compute
        # the rate across the bin.  The bin edges coincide with schedule checkpoints,
        # so t0 and t1 are always present in the data.
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
        # Strict left inequality: each checkpoint belongs to the bin that ends at it.
        left = times > t0 if k > 0 or t_start > 0 else times >= t0
        mask = left & (times <= t1)
        if mask.sum() > 0:
            avg_flux[k] = np.nanmean(fluxes[mask])
    return bin_edges, avg_flux


def plot_seismic_moment_and_flux(
    frac_df: pd.DataFrame,
    fracture_id: str,
    column: str,
    t_start: float,
    t_end: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of a scalar fracture-CSV column in each bin for one fracture and interval.

    Unlike :func:`bin_moment_rate` (which computes a *rate* by differencing), this
    function returns the plain time-average of the column values within each bin.
    Suitable for columns like ``average_pressure`` or ``average_critical_pressure``
    that are already instantaneous scalars.

    Parameters:
        frac_df: Fracture monitoring DataFrame (from ``*_fractures.csv``).
        fracture_id: Fracture ID to filter on.
        column: Column name to aggregate.
        t_start: Bin-interval start time (seconds).
        t_end: Bin-interval end time (seconds).
        n_bins: Number of equal bins.

    Returns:
        ``(bin_edges, avg_values)`` where ``avg_values`` has length ``n_bins``.
    """
    df = frac_df[frac_df["fracture_id"] == fracture_id]
    times, values = _monotone(df["time"].values, df[column].values)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    avg_values = np.full(n_bins, np.nan)
    for k in range(n_bins):
        t0, t1 = bin_edges[k], bin_edges[k + 1]
        left = times > t0 if k > 0 or t_start > 0 else times >= t0
        mask = left & (times <= t1)
        if mask.sum() > 0:
            avg_values[k] = np.nanmean(values[mask])
    return bin_edges, avg_values


def bin_fracture_pvd(
    pvd_dim_data: dict[str, np.ndarray],
    subdomain_id: int,
    quantity: str,
    t_start: float,
    t_end: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of a scalar PVD quantity in each bin for one fracture subdomain.

    Reads cell-wise data from a ``split_by_subdomain`` result, computes the
    unweighted spatial mean across all cells at each time step, then time-averages
    within each bin.  Use this for quantities not captured in the monitoring CSV
    (e.g., historical runs or spatially-resolved fields).

    Parameters:
        pvd_dim_data: The fracture-dimension entry returned by
            :func:`~load_pvd_data.load_from_pvd`, i.e.
            ``subdomains[nd - 1]``.  Must have been loaded with ``quantity``
            and ``"subdomain_id"`` (or ``"fracture_name"``) included in the
            ``quantities`` list so that :func:`~load_pvd_data.split_by_subdomain`
            can partition the cells.
        subdomain_id: Integer subdomain id to extract (key in the dict returned
            by :func:`~load_pvd_data.split_by_subdomain`).
        quantity: Name of the scalar cell-data array to aggregate.
        t_start: Bin-interval start time (seconds).
        t_end: Bin-interval end time (seconds).
        n_bins: Number of equal bins.

    Returns:
        ``(bin_edges, avg_values)`` where ``avg_values`` has length ``n_bins``.
        Returns all-NaN ``avg_values`` if ``quantity`` is absent or
        ``subdomain_id`` is not found.

    Note:
        The spatial average is *unweighted* (plain mean over cells).  To obtain
        a volume-weighted average, export ``cell_volumes`` from PorePy and pass
        them as an additional quantity, then compute the weighted mean manually.
    """
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    avg_values = np.full(n_bins, np.nan)

    per_sd = split_by_subdomain(pvd_dim_data)
    if subdomain_id not in per_sd or quantity not in per_sd[subdomain_id]:
        return bin_edges, avg_values

    sd_data = per_sd[subdomain_id]
    times = sd_data["times"]  # shape (T,)
    arr = sd_data[quantity]  # shape (T, n_cells) for scalars

    # Spatial mean across cells at each time step.
    spatial_mean = np.nanmean(arr, axis=1)  # shape (T,)

    for k in range(n_bins):
        t0, t1 = bin_edges[k], bin_edges[k + 1]
        left = times > t0 if k > 0 or t_start > 0 else times >= t0
        mask = left & (times <= t1)
        if mask.sum() > 0:
            avg_values[k] = np.nanmean(spatial_mean[mask])
    return bin_edges, avg_values


def plot_seismic_moment_and_flux(
    csv_dir: Path | str,
    file_base: str,
    production_period: float,
    shut_in_duration: float,
    final_time: float,
    production_well: str,
    fracture_ids: list[str] | None = None,
    num_bins_per_interval: int = 4,
    title: str | None = None,
    out_path: Path | str | None = None,
    symlog: bool = False,
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
    shut_in_duration:
        Duration of each shut-in period in seconds.
    final_time:
        Simulation end time in seconds (used to infer number of cycles).
    production_well:
        Well name string used to filter the well CSV.
    fracture_ids:
        Fracture IDs to include.  Defaults to all fractures in the CSV.
    num_bins_per_interval:
        Number of equal bins per sub-interval.
    title:
        Optional figure title.
    out_path:
        If given, save the figure here instead of showing it.
    symlog:
        If ``True``, use a symmetric-log scale on the moment-rate axis (handles
        zeros and negative values via a linear region near zero).  If ``False``
        (default), use a plain log scale with non-positive values clipped to the
        smallest positive value present so they are not silently dropped.
    """
    csv_dir = Path(csv_dir)
    try:
        frac_df = pd.read_csv(csv_dir / f"{file_base}_fractures.csv")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print(f"Skipping {csv_dir}: fracture CSV not ready yet.")
        return
    try:
        well_df = pd.read_csv(csv_dir / f"{file_base}.csv")
        has_well_data = (
            len(well_df) > 0 and production_well in well_df["well_name"].values
        )
    except (pd.errors.EmptyDataError, FileNotFoundError):
        well_df = None
        has_well_data = False

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
    all_moment_rate = {fid: np.full(total_bins, np.nan) for fid in fracture_ids}
    all_flux = np.full(total_bins, np.nan)

    for iv, (t_start, t_end, kind) in enumerate(intervals):
        sl = slice(iv * n_bins, (iv + 1) * n_bins)
        for fid in fracture_ids:
            _, rates = bin_moment_rate(frac_df, fid, t_start, t_end, n_bins)
            all_moment_rate[fid][sl] = rates
        if has_well_data:
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

    # Moment-rate dot colours: one per fracture from tab10.
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

    # --- Left axis: moment rate dots (one per fracture, centred in each bin) ---
    moment_handles = []
    offsets = np.linspace(-0.2, 0.2, max(len(fracture_ids), 1))
    for fi, fid in enumerate(fracture_ids):
        _mr_label = f"Moment rate – {fid}" if len(fracture_ids) > 1 else "Moment rate"
        (dot_line,) = ax1.plot(
            x_all + offsets[fi],
            all_moment_rate[fid],
            color=frac_colors[fi],
            linestyle="none",
            marker="o",
            markersize=6,
            label=_mr_label,
            zorder=3,
        )
        moment_handles.append(dot_line)
    ax1.set_ylabel("Seismic moment rate (N·m/s)")

    # Apply log or symlog scale to the moment-rate axis.
    if symlog:
        all_finite = np.concatenate(
            [v[np.isfinite(v)] for v in all_moment_rate.values()]
        )
        linthresh = (
            float(np.percentile(np.abs(all_finite[all_finite != 0]), 10))
            if all_finite[all_finite != 0].size
            else 1.0
        )
        ax1.set_yscale("symlog", linthresh=linthresh)
    else:
        # Clip non-positive values to the smallest positive value so log works.
        all_positive = np.concatenate(
            [v[v > 0] for v in all_moment_rate.values() if np.any(v > 0)]
        )
        floor = float(all_positive.min()) if all_positive.size else 1.0
        for fid in fracture_ids:
            all_moment_rate[fid] = np.where(
                all_moment_rate[fid] > 0, all_moment_rate[fid], floor
            )
        # Re-plot with clipped data (replace existing line data).
        for fi, fid in enumerate(fracture_ids):
            moment_handles[fi].set_ydata(all_moment_rate[fid])
        ax1.set_yscale("log")

    ax1.axhline(0, color="black", linewidth=0.6)

    # --- Right axis: fluid flux bars (only when well data is available) ---
    if has_well_data:
        flux_bars = ax2.bar(
            x_all,
            all_flux,
            width=0.6,
            color="tab:green",
            alpha=0.5,
            label="Production well flux",
            zorder=2,
        )
        ax2.set_ylabel("Fluid flux (kg/s)")
    else:
        flux_bars = None
        ax2.set_visible(False)

    # --- x-axis ticks and limits ---
    ax1.set_xlim(-0.5, n_intervals * n_bins - 0.5)
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

    # Combined legend (moment handles + flux handle on ax1 so it stays on top).
    legend_handles = list(moment_handles)
    legend_labels = [
        f"Moment rate – {fid}" if len(fracture_ids) > 1 else "Moment rate"
        for fid in fracture_ids
    ]
    if flux_bars is not None:
        legend_handles.append(flux_bars)
        legend_labels.append("Production well flux")
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
                        plot_seismic_moment_and_flux(
                            csv_dir=csv_dir,
                            file_base=file_name,
                            production_period=production_period,
                            fracture_ids=["Fracture 2", "Fracture 3"],
                            num_bins_per_interval=1,
                            shut_in_duration=run_example_4.SHUT_IN_DURATION,
                            final_time=run_example_4.FINAL_TIME,
                            production_well=run_example_4.PRODUCTION_WELL,
                            title=title,
                            out_path=out_path,
                        )
    if 2 in CASES_TO_PLOT:
        # --- Example 2 (Case_II) ---
        for with_prod in run_example_2.with_production:
            for velocity in run_example_2.boundary_velocities:
                for period in run_example_2.production_periods:
                    production_period = period * pp.YEAR
                    simulation_name, folder_name, _, file_name, title = (
                        run_example_2.names_from_params(velocity, period, with_prod)
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
                    plot_seismic_moment_and_flux(
                        csv_dir=csv_dir,
                        file_base=file_name,
                        production_period=production_period,
                        fracture_ids=["Fracture 2"],
                        num_bins_per_interval=run_example_2.NUM_BINS_PER_INTERVAL,
                        shut_in_duration=run_example_2.SHUT_IN_DURATION,
                        final_time=run_example_2.FINAL_TIME,
                        production_well=run_example_2.PRODUCTION_WELL,
                        title=title,
                        out_path=out_path,
                    )
