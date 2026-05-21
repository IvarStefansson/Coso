"""Broken-axis plotting of per-fracture quantities from CSV and PVD data.

This module provides:

- Binning helpers that work with either the monitoring CSV or PVD cell data.
- A low-level broken-axis layout helper (:func:`broken_axis_layout`) that builds a
    gridspec figure with the same visual style as :func:`plot_seismic_moment_and_flux`.
- A high-level :func:`plot_broken_axis_panels` that takes pre-binned data and renders
    one subplot per panel.

Shared utilities (``get_intervals``, ``_monotone``) are imported directly from
:mod:`plot_seismic_moment_and_flux` to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import porepy as pp
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from load_pvd_data import split_by_subdomain
from plot_seismic_moment_and_flux import _monotone, get_intervals

# ---------------------------------------------------------------------------
# Panel / Quantity specification types
# ---------------------------------------------------------------------------


@dataclass
class Quantity:
    """One data series within a :class:`Panel`.

    Parameters:
        name: Series label used in the legend (and in the combined
            ``"{fracture} – {name}"`` label when multiple fractures and
            quantities share a panel).
        inputs: Names of raw PVD cell-data arrays required by *fn*
            (passed verbatim to :func:`~load_pvd_data.load_from_pvd`).
        fn: Cell-wise transform applied to the per-subdomain data dict
            before spatial aggregation.  Receives a dict with key
            ``"times"`` (shape ``(T,)``) and one key per entry in
            *inputs* (shape ``(T, n_cells[, nd])``); must return an
            array of shape ``(T, n_cells)``.  ``None`` (default) means
            use ``inputs[0]`` directly (identity transform).
        spatial_fn: Reduces the ``(n_cells,)`` cell array at each
            timestep to a scalar.  Defaults to :func:`numpy.nanmean`.
            Use e.g. ``numpy.nanmin`` / ``numpy.nanmax`` for extremes.
    """

    name: str
    inputs: list[str]
    fn: Callable[[dict[str, np.ndarray]], np.ndarray] | None = None
    spatial_fn: Callable[[np.ndarray], float] = field(default=np.nanmean)
    color: str | tuple | None = None
    linestyle: str = "-"
    marker: str = ""
    markersize: float = 3.0


@dataclass
class Panel:
    """One subplot axis grouping one or more :class:`Quantity` series.

    Parameters:
        ylabel: Y-axis label for the subplot.
        quantities: Series to draw on the shared axis.  Multiple
            quantities produce multiple lines, colour-coded and labelled.
    """

    ylabel: str
    quantities: list[Quantity]


#: Internal rendered panel consumed by :func:`plot_broken_axis_panels`.
#: Built from :class:`Panel` objects by :func:`build_panels`.


@dataclass
class _RenderedPanel:
    ylabel: str
    #: ``{series_label: (x_positions, y_values)}``.
    #: For binned panels x is integer bin indices; for raw panels x is : continuous
    # time-mapped positions within each interval.
    data: dict[str, tuple[np.ndarray, np.ndarray]]
    #: Per-series plot kwargs: ``color``, ``linestyle``, ``marker``, ``markersize``.
    styles: dict[str, dict]


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------


def bin_scalar_csv(
    df: pd.DataFrame,
    row_filter: dict[str, Any],
    column: str,
    t_start: float,
    t_end: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-average a scalar CSV column in each bin.

    Unlike :func:`~plot_seismic_moment_and_flux.bin_moment_rate` (which computes a
    *rate* by differencing), this returns the plain time-mean of ``column`` values
    within each bin.  Suitable for instantaneous scalars such as ``average_pressure`` or
    ``average_critical_pressure``.

    Parameters:
        df: Monitoring DataFrame (fracture or well CSV). row_filter: Dict of
            ``{column_name: value}`` used to subset ``df`` before binning (e.g.
            ``{"fracture_id": "Fracture 2"}``).
        column: Column name to aggregate. t_start: Bin-interval start time (seconds).
        t_end: Bin-interval end time (seconds). n_bins: Number of equal bins.

    Returns:
        ``(bin_edges, avg_values)`` where ``avg_values`` has length ``n_bins``.
    """
    mask = np.ones(len(df), dtype=bool)
    for col, val in row_filter.items():
        mask &= df[col] == val
    subset = df[mask]
    times, values = _monotone(subset["time"].values, subset[column].values)

    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    avg_values = np.full(n_bins, np.nan)
    for k in range(n_bins):
        t0, t1 = bin_edges[k], bin_edges[k + 1]
        left = times > t0 if k > 0 or t_start > 0 else times >= t0
        m = left & (times <= t1)
        if m.sum() > 0:
            avg_values[k] = np.nanmean(values[m])
    return bin_edges, avg_values


def bin_pvd_quantity(
    pvd_dim_data: dict[str, np.ndarray],
    subdomain_id: int,
    quantity: str,
    t_start: float,
    t_end: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Time-average a PVD cell quantity (spatially averaged) in each bin.

    Reads cell-wise data from a :func:`~load_pvd_data.load_from_pvd` result, computes
    the unweighted spatial mean across cells at each time step, then time-averages
    within each bin.

    Parameters:
        pvd_dim_data: The fracture-dimension entry from
            :func:`~load_pvd_data.load_from_pvd`, i.e. ``subdomains[nd - 1]``. Must have
            been loaded with ``quantity`` and ``"subdomain_id"`` in the ``quantities``
            list so that :func:`~load_pvd_data.split_by_subdomain` can partition the
            cells.
        subdomain_id: Integer subdomain id to extract.
        quantity: Name of the scalar cell-data array.
        t_start: Bin-interval start time (seconds).
        t_end: Bin-interval end time (seconds).
        n_bins: Number of equal bins.

    Returns:
        ``(bin_edges, avg_values)`` where ``avg_values`` has length ``n_bins``.
        Returns all-NaN ``avg_values`` if ``quantity`` or ``subdomain_id`` is
        absent.

    Note:
        The spatial average is *unweighted* (plain cell mean).
    """
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    avg_values = np.full(n_bins, np.nan)

    per_sd = split_by_subdomain(pvd_dim_data)
    if subdomain_id not in per_sd or quantity not in per_sd[subdomain_id]:
        return bin_edges, avg_values

    sd_data = per_sd[subdomain_id]
    times = sd_data["times"]  # (T,)
    arr = sd_data[quantity]  # (T, n_cells)
    spatial_mean = np.nanmean(arr, axis=1)  # (T,)

    for k in range(n_bins):
        t0, t1 = bin_edges[k], bin_edges[k + 1]
        left = times > t0 if k > 0 or t_start > 0 else times >= t0
        m = left & (times <= t1)
        if m.sum() > 0:
            avg_values[k] = np.nanmean(spatial_mean[m])
    return bin_edges, avg_values


def fracture_subdomain_id_map(
    pvd_dim_data: dict[str, np.ndarray],
    fracture_names: list[str],
) -> dict[int, str]:
    """Map fracture names to subdomain IDs by their sorted order.

    PorePy assigns integer subdomain IDs that are not known in advance, but their
    relative order (when sorted) is stable across runs and matches the order in which
    fractures appear in the mixed-dimensional grid.  This helper recovers that mapping
    without requiring the caller to hard-code IDs.

    Parameters:
        pvd_dim_data: Fracture-dimension entry from
            :func:`~load_pvd_data.load_from_pvd` (must include ``"subdomain_id"`` in the
            loaded quantities).
        fracture_names: Fracture display names **in the same order** as the sorted
            subdomain IDs (i.e. name[0] → smallest id, name[1] → second-smallest id, …).

    Returns:
        ``{subdomain_id: fracture_name}`` mapping.

    Raises:
        ValueError: If the number of names does not match the number of distinct
            subdomain IDs found in *pvd_dim_data*.

    Example::

        subdomains, _ = load_from_pvd(pvd_dir, ["p", "subdomain_id"],
                                      subdomain_dimensions=[2])
        id_map = fracture_subdomain_id_map(
            subdomains[2], ["Fracture 2", "Fracture 3"]
        )
        # id_map == {7: "Fracture 2", 12: "Fracture 3"}  (ids are illustrative)
    """
    per_sd = split_by_subdomain(pvd_dim_data)
    sorted_ids = sorted(per_sd.keys())
    if len(sorted_ids) != len(fracture_names):
        raise ValueError(
            f"Found {len(sorted_ids)} subdomain IDs {sorted_ids} but "
            f"{len(fracture_names)} fracture names were provided: {fracture_names}"
        )
    return dict(zip(sorted_ids, fracture_names))


# ---------------------------------------------------------------------------
# Quantity evaluation and binning
# ---------------------------------------------------------------------------


def _normalize_panel_spec(spec: Panel | tuple[str, str]) -> Panel:
    """Convert a ``(pvd_name, ylabel)`` shorthand to a full :class:`Panel`."""
    if isinstance(spec, Panel):
        return spec
    pvd_name, ylabel = spec
    return Panel(ylabel=ylabel, quantities=[Quantity(name=pvd_name, inputs=[pvd_name])])


def collect_pvd_inputs(specs: list[Panel | tuple[str, str]]) -> list[str]:
    """Return the deduplicated list of raw PVD array names needed by *specs*.

    Pass the result to :func:`~load_pvd_data.load_from_pvd` as the ``quantities``
    argument (alongside ``"subdomain_id"``).
    """
    seen: set[str] = set()
    result: list[str] = []
    for spec in specs:
        for qty in _normalize_panel_spec(spec).quantities:
            for inp in qty.inputs:
                if inp not in seen:
                    seen.add(inp)
                    result.append(inp)
    return result


def _evaluate_quantity_spatial(
    sd_data: dict[str, np.ndarray],
    quantity: Quantity,
) -> np.ndarray:
    """Apply *quantity.fn* then *quantity.spatial_fn* → ``(T,)`` series.

    Parameters:
        sd_data: Per-subdomain data dict (keys: ``"times"`` plus loaded quantities), as
            returned by :func:`~load_pvd_data.split_by_subdomain`.
        quantity: The :class:`Quantity` to evaluate.

    Returns:
        Array of shape ``(T,)`` — one scalar per timestep.
    """
    if quantity.fn is not None:
        cell_arr = quantity.fn(sd_data)  # (T, n_cells)
    else:
        cell_arr = sd_data[quantity.inputs[0]]  # (T, n_cells)
    return np.array([quantity.spatial_fn(cell_arr[t]) for t in range(len(cell_arr))])


def bin_quantity(
    sd_data: dict[str, np.ndarray],
    quantity: Quantity,
    t_start: float,
    t_end: float,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a :class:`Quantity` on subdomain data and bin in time.

    Parameters:
        sd_data: Per-subdomain data dict from :func:`~load_pvd_data.split_by_subdomain`.
        quantity: The :class:`Quantity` to evaluate.
        t_start: Bin-interval start time (seconds).
        t_end: Bin-interval end time (seconds).
        n_bins: Number of equal bins.

    Returns:
        ``(bin_edges, avg_values)`` where *avg_values* has length *n_bins*.
    """
    times = sd_data["times"]
    spatial = _evaluate_quantity_spatial(sd_data, quantity)
    bin_edges = np.linspace(t_start, t_end, n_bins + 1)
    avg_values = np.full(n_bins, np.nan)
    for k in range(n_bins):
        t0, t1 = bin_edges[k], bin_edges[k + 1]
        left = times > t0 if k > 0 or t_start > 0 else times >= t0
        m = left & (times <= t1)
        if m.sum() > 0:
            avg_values[k] = np.nanmean(spatial[m])
    return bin_edges, avg_values


def build_panels(
    pvd_dim_data: dict[str, np.ndarray],
    subdomain_id_map: dict[int, str],
    intervals: list[tuple[float, float, str]],
    n_bins: int,
    specs: list[Panel | tuple[str, str]],
    binned: bool = True,
) -> list[_RenderedPanel]:
    """Bin (or collect raw) PVD quantities and assemble rendered panel specs.

    Accepts both :class:`Panel` objects and ``(pvd_quantity_name, ylabel)`` shorthand
    tuples.  Calls :func:`~load_pvd_data.split_by_subdomain` once and reuses the result
    across all panels and quantities.

    Series labels follow these rules:

    - One quantity + multiple fractures → fracture name as label
    - Multiple quantities + one fracture → quantity name as label
    - Multiple quantities + multiple fractures → ``"{fracture} – {qty.name}"``

    Parameters:
        pvd_dim_data: Fracture-dimension entry from
            :func:`~load_pvd_data.load_from_pvd` (must include
            ``"subdomain_id"`` and all arrays referenced by *specs*).
        subdomain_id_map: ``{subdomain_id: fracture_label}`` mapping, e.g.
            from :func:`fracture_subdomain_id_map`.
        intervals: Output of :func:`~plot_seismic_moment_and_flux.get_intervals`.
        n_bins: Number of bins per sub-interval (also controls x-axis spacing for raw
            panels).
        specs: List of :class:`Panel` objects or ``(pvd_name, ylabel)`` tuples.
        binned: If ``True`` (default) each interval is divided into *n_bins* equal time
            buckets and the spatially-averaged value is time-averaged within each
            bucket.  If ``False`` every converged timestep is plotted at its real time
            position (mapped continuously within the interval's x-axis range).

    Returns:
        List of :class:`_RenderedPanel` ready for
        :func:`plot_broken_axis_panels`.
    """
    total_bins = len(intervals) * n_bins
    per_sd = split_by_subdomain(pvd_dim_data)
    multi_fracture = len(subdomain_id_map) > 1
    rendered: list[_RenderedPanel] = []

    # Stable per-fracture colour (sorted subdomain ids → tab10 in order).
    tab10 = plt.cm.tab10(np.linspace(0, 0.9, 10))
    frac_color: dict[int, Any] = {
        sd_id: tab10[i % 10] for i, sd_id in enumerate(sorted(subdomain_id_map.keys()))
    }

    for spec in specs:
        panel = _normalize_panel_spec(spec)
        multi_qty = len(panel.quantities) > 1
        data: dict[str, np.ndarray] = {}
        styles: dict[str, dict] = {}

        for sd_id, frac_name in subdomain_id_map.items():
            if sd_id not in per_sd:
                continue
            sd_data = per_sd[sd_id]

            for qi, qty in enumerate(panel.quantities):
                if any(inp not in sd_data for inp in qty.inputs):
                    continue

                if binned:
                    y_all = np.full(total_bins, np.nan)
                    for iv, (t0, t1, _) in enumerate(intervals):
                        sl = slice(iv * n_bins, (iv + 1) * n_bins)
                        _, v = bin_quantity(sd_data, qty, t0, t1, n_bins)
                        y_all[sl] = v
                    x_arr = np.arange(total_bins, dtype=float)
                    y_arr = y_all
                else:
                    spatial = _evaluate_quantity_spatial(sd_data, qty)  # (T,)
                    times_arr = sd_data["times"]
                    x_parts: list[np.ndarray] = []
                    y_parts: list[np.ndarray] = []
                    for iv, (t0, t1, _) in enumerate(intervals):
                        left = times_arr > t0 if iv > 0 or t0 > 0 else times_arr >= t0
                        mask = left & (times_arr <= t1)
                        if mask.sum() == 0:
                            continue
                        t_local = times_arr[mask]
                        # Map real time linearly to the interval's x-axis slot.
                        x_local = iv * n_bins + (t_local - t0) / (t1 - t0) * n_bins
                        x_parts.append(x_local)
                        y_parts.append(spatial[mask])
                    if x_parts:
                        x_arr = np.concatenate(x_parts)
                        y_arr = np.concatenate(y_parts)
                    else:
                        x_arr = np.array([], dtype=float)
                        y_arr = np.array([], dtype=float)

                if multi_fracture and multi_qty:
                    label = f"{frac_name} \u2013 {qty.name}"
                elif multi_fracture:
                    label = frac_name
                else:
                    label = qty.name
                data[label] = (x_arr, y_arr)

                # Color precedence: explicit → fracture auto (multi-frac) → qty auto.
                if qty.color is not None:
                    c = qty.color
                elif multi_fracture:
                    c = frac_color[sd_id]
                else:
                    c = tab10[qi % 10]
                styles[label] = dict(
                    color=c,
                    linestyle=qty.linestyle,
                    marker=qty.marker,
                    markersize=qty.markersize,
                )

        if data:
            rendered.append(
                _RenderedPanel(ylabel=panel.ylabel, data=data, styles=styles)
            )

    return rendered


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def broken_axis_layout(
    intervals: list[tuple[float, float, str]],
    n_bins: int,
    n_panels: int,
    main_height: float = 3.0,
    panel_height: float = 3.0,
    hspace: float = 0.35,
) -> tuple[Figure, list[Axes], np.ndarray, list[float], list[str]]:
    """Create a gridspec figure with the standard broken-axis layout.

    Returns the figure, a list of ``n_panels`` axes (top-to-bottom), the integer
    x-positions for each bin, and the pre-computed tick positions and labels for the
    broken time axis.

    Parameters:
        intervals: Output of :func:`~plot_seismic_moment_and_flux.get_intervals`.
        n_bins: Number of bins per sub-interval.
        n_panels: Total number of subplot rows to create.
        main_height: Height ratio of the first (main) panel.
        panel_height: Height ratio of each additional panel.
        hspace: Vertical spacing between panels (``gridspec`` ``hspace``).

    Returns:
        ``(fig, axes, x_all, xtick_pos, xtick_lbl)``
    """
    n_intervals = len(intervals)
    total_bins = n_intervals * n_bins
    fig_width = max(10, n_intervals * n_bins * 0.35)

    heights = [main_height] + [panel_height] * (n_panels - 1)
    fig = plt.figure(figsize=(fig_width, sum(heights)))
    gs = fig.add_gridspec(n_panels, 1, height_ratios=heights, hspace=hspace)
    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]

    x_all = np.arange(total_bins)

    # Build broken-axis tick labels.
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

    return fig, axes, x_all, xtick_pos, xtick_lbl


def apply_broken_axis_style(
    ax: Axes,
    intervals: list[tuple[float, float, str]],
    n_bins: int,
    xtick_pos: list[float],
    xtick_lbl: list[str],
    annotate_kinds: bool = False,
) -> None:
    """Apply the standard broken-axis styling to an axes.

    Sets x-limits, tick positions/labels, interval background shading, and vertical
    separators.  Optionally annotates each interval with a "Prod." / "Shut-in" label
    along the top edge (useful for the topmost panel only).

    Parameters:
        ax: Axes to style.
        intervals: Output of :func:`~plot_seismic_moment_and_flux.get_intervals`.
        n_bins: Number of bins per sub-interval.
        xtick_pos: Pre-computed tick positions (from :func:`broken_axis_layout`).
        xtick_lbl: Pre-computed tick labels (from :func:`broken_axis_layout`).
        annotate_kinds: If ``True``, annotate each interval with its kind label.
    """
    n_intervals = len(intervals)
    ax.set_xlim(-0.5, n_intervals * n_bins - 0.5)
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl, fontsize=6, rotation=45, ha="right")

    for iv, (_, _, kind) in enumerate(intervals):
        x_left = iv * n_bins - 0.5
        x_right = (iv + 1) * n_bins - 0.5
        bg = "#d6e8f7" if kind == "production" else "#fde8ce"
        ax.axvspan(x_left, x_right, color=bg, alpha=0.6, zorder=0)
    for iv in range(1, n_intervals):
        ax.axvline(
            iv * n_bins - 0.5, color="gray", linewidth=0.7, linestyle="--", zorder=1
        )

    if annotate_kinds:
        for iv, (_, _, kind) in enumerate(intervals):
            center = iv * n_bins + n_bins / 2 - 0.5
            lbl = "Prod." if kind == "production" else "Shut-in"
            ax.annotate(
                lbl,
                xy=(center, 1.0),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#1f5f9e" if kind == "production" else "#b85c00",
            )


# ---------------------------------------------------------------------------
# High-level plotting function
# ---------------------------------------------------------------------------


def plot_broken_axis_panels(
    panels: list[_RenderedPanel],
    intervals: list[tuple[float, float, str]],
    n_bins: int,
    title: str | None = None,
    out_path: Path | str | None = None,
    main_height: float = 3.0,
    panel_height: float = 3.0,
) -> Figure:
    """Plot one or more data series on a broken-axis figure with multiple panels.

    Each panel is a separate subplot sharing the same broken time axis. The first panel
    receives the "Prod." / "Shut-in" interval annotations.

    Parameters:
        panels: List of ``(ylabel, data_dict)`` tuples.  Each ``data_dict``
            maps a series label (e.g. fracture name) to a 1-D array of length
            ``len(intervals) * n_bins``, pre-filled with :func:`bin_scalar_csv`
            or :func:`bin_pvd_quantity`.
        intervals: Output of :func:`~plot_seismic_moment_and_flux.get_intervals`.
        n_bins: Number of bins per sub-interval.
        title: Optional figure suptitle.
        out_path: If given, save to this path instead of showing.
        main_height: Height ratio of the first panel.
        panel_height: Height ratio of each subsequent panel.

    Returns:
        The :class:`~matplotlib.figure.Figure`.

    Example::

        from plot_seismic_moment_and_flux import get_intervals, bin_fluid_flux
        import pandas as pd

        frac_df = pd.read_csv("well_monitoring/example_2_fractures.csv")
        intervals = get_intervals(production_period)
        n_bins = 4
        total_bins = len(intervals) * n_bins

        panels = []
        for col, ylabel in [
            ("average_pressure", "Avg. pressure (Pa)"),
            ("average_critical_pressure", "Critical pressure (Pa)"),
            ("slip_tendency", "Slip tendency (−)"),
        ]:
            data = {}
            for fid in ["Fracture 2"]:
                vals = np.full(total_bins, np.nan)
                for iv, (t0, t1, _) in enumerate(intervals):
                    sl = slice(iv * n_bins, (iv + 1) * n_bins)
                    _, vals[sl] = bin_scalar_csv(
                        frac_df, {"fracture_id": fid}, col, t0, t1, n_bins
                    )
                data[fid] = vals
            panels.append((ylabel, data))

        plot_broken_axis_panels(panels, intervals, n_bins, title="Case II")
    """
    n_panels = len(panels)
    fig, axes, x_all, xtick_pos, xtick_lbl = broken_axis_layout(
        intervals, n_bins, n_panels, main_height=main_height, panel_height=panel_height
    )

    for row_idx, rp in enumerate(panels):
        ax = axes[row_idx]
        is_top = row_idx == 0
        apply_broken_axis_style(
            ax, intervals, n_bins, xtick_pos, xtick_lbl, annotate_kinds=is_top
        )
        for label, (x_vals, y_vals) in rp.data.items():
            s = rp.styles.get(label, {})
            ax.plot(
                x_vals,
                y_vals,
                color=s.get("color"),
                linestyle=s.get("linestyle", "-"),
                linewidth=1.5,
                marker=s.get("marker", "o"),
                markersize=s.get("markersize", 3),
                label=label if len(rp.data) > 1 else None,
            )
        ax.set_ylabel(rp.ylabel, fontsize=8)
        if len(rp.data) > 1:
            ax.legend(fontsize=7, loc="upper left")

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

    return fig
