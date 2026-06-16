"""Preprocessing / inspection script for fault-plane point clouds.

Usage
-----
Run from the Coso workspace root::

    python preprocess_fault_planes.py [--config PATH] [--dirs DIR [DIR ...]] [--out PATH]

Options
-------
--config PATH
    Path to a JSON extension-config file (default: fault_extensions.json if it
    exists in the current directory, otherwise no extensions are applied).
--dirs DIR [DIR ...]
    One or more directories of CSV point clouds to load (default:
    UiB_SURF_faults point_cloud_clusters).
--exclude STEM [STEM ...]
    Fault stem names to skip (e.g. --exclude 0003 0027).
--out PATH
    Output VTU file path (default: data/fault_planes_preview.vtu).

Workflow
--------
1. Run the script once to load all fault planes and export a VTU preview.
2. Open ``data/fault_planes_preview.vtu`` in ParaView to inspect fracture
   positions, sizes, and orientations.
3. Edit ``fault_extensions.json`` to configure extensions for specific faults
   (see examples below).
4. Re-run with ``--config fault_extensions.json`` to preview the result.
5. Once satisfied, pass ``fault_extension_config`` and ``fault_plane_dirs`` as
   model params to ``FaultPlaneGeometry``.

Extension config format (fault_extensions.json)
------------------------------------------------
::

    {
        "0003": {"type": "uniform", "scale": 1.5},
        "0027": {"type": "directional", "axis": [1, 0, 0], "amount": 500.0}
    }

Types
~~~~~
uniform
    Scale all four corners away from the centroid by ``scale`` (e.g. 1.5
    increases linear dimensions by 50 %).
directional
    Shift the two corners most aligned with ``axis`` by ``amount`` metres.
to_intersect
    **Not yet implemented.**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import meshio
import numpy as np
import porepy as pp
from porepy.fracs.fracture_network_3d import FractureNetwork3d

# ---------------------------------------------------------------------------
# Ensure the Coso source directory is on sys.path so local imports work.
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from fault_planes import apply_extensions, load_fault_planes, load_point_clouds  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fracture_dims(frac: pp.PlaneFracture) -> tuple[float, float]:
    """Return approximate width and height of a planar fracture rectangle."""
    pts = frac.pts  # (3, 4)
    # Edge vectors from corner 0.
    e1 = pts[:, 1] - pts[:, 0]
    e2 = pts[:, 3] - pts[:, 0]
    return float(np.linalg.norm(e1)), float(np.linalg.norm(e2))


def _export_point_clouds_vtu(
    clouds: list[np.ndarray],
    names: list[str],
    out_path: Path,
) -> None:
    """Export raw (transformed) point clouds to a single VTU file.

    Each point is tagged with its fault index (``fault_id``) so individual
    faults can be filtered in ParaView using a *Threshold* filter on that
    array.  A string-valued ``fault_name`` cell array is also written for
    reference.
    """
    pts_list = [c.T for c in clouds]  # list of (N_i, 3)
    pts_all = np.vstack(pts_list)  # (total_N, 3)
    n_total = pts_all.shape[0]

    # Vertex cells – one cell per point.
    cells = [("vertex", np.arange(n_total, dtype=np.int64).reshape(-1, 1))]

    # Per-point fault index.
    fault_id = np.concatenate(
        [np.full(c.shape[1], i, dtype=np.int32) for i, c in enumerate(clouds)]
    )

    mesh = meshio.Mesh(
        points=pts_all,
        cells=cells,
        point_data={"fault_id": fault_id},
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(out_path)


def _print_summary(
    fractures: list[pp.PlaneFracture],
    names: list[str],
    source_dirs: list[str],
    n_pts_per_frac: list[int],
) -> None:
    header = (
        f"{'Name':<10}  {'Dir':<26}  {'N pts':>6}  "
        f"{'cx [m]':>9}  {'cy [m]':>9}  {'cz [m]':>9}  "
        f"{'W [m]':>8}  {'H [m]':>8}  "
        f"{'nx':>7}  {'ny':>7}  {'nz':>7}"
    )
    print()
    print(header)
    print("-" * len(header))
    for frac, name, src, n in zip(fractures, names, source_dirs, n_pts_per_frac):
        cx, cy, cz = frac.center.ravel()
        w, h = _fracture_dims(frac)
        nx, ny, nz = frac.normal.ravel()
        print(
            f"{name:<10}  {src:<26}  {n:>6}  "
            f"{cx:>9.1f}  {cy:>9.1f}  {cz:>9.1f}  "
            f"{w:>8.1f}  {h:>8.1f}  "
            f"{nx:>7.3f}  {ny:>7.3f}  {nz:>7.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect and preprocess fault-plane point clouds."
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to fault_extensions.json (default: auto-detect in cwd).",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        metavar="DIR",
        default=["UiB_SURF_faults", "point_cloud_clusters"],
        help="Directories containing CSV point clouds.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        metavar="STEM",
        default=[],
        help="Fault stem names to exclude.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default="data/fault_planes_preview.vtu",
        help="Output VTU file for fracture rectangles (default: data/fault_planes_preview.vtu).",
    )
    parser.add_argument(
        "--out-clouds",
        metavar="PATH",
        default="data/fault_planes_clouds.vtu",
        help="Output VTU file for raw point clouds (default: data/fault_planes_clouds.vtu).",
    )
    args = parser.parse_args()

    root = _script_dir

    # ------------------------------------------------------------------ load
    fractures: list[pp.PlaneFracture] = []
    names: list[str] = []
    source_dirs: list[str] = []
    n_pts_per_frac: list[int] = []
    all_clouds: list[np.ndarray] = []
    all_cloud_names: list[str] = []

    for d in args.dirs:
        directory = root / d
        if not directory.is_dir():
            print(f"Warning: directory not found, skipping: {directory}")
            continue

        clouds, cloud_names = load_point_clouds(directory, exclude=args.exclude)
        all_clouds.extend(clouds)
        all_cloud_names.extend(cloud_names)
        n_pts_per_frac.extend(c.shape[1] for c in clouds)

        batch = load_fault_planes(directory, exclude=args.exclude)
        fractures.extend(batch)
        names.extend(frac.fault_name for frac in batch)
        source_dirs.extend([d] * len(batch))

    if not fractures:
        print("No fault planes loaded. Check --dirs and --exclude.")
        sys.exit(1)

    # --------------------------------------------------------------- extend
    config_path: Path | None = None
    if args.config:
        config_path = Path(args.config)
    else:
        default_cfg = root / "fault_extensions.json"
        if default_cfg.exists():
            import json

            with open(default_cfg) as f:
                cfg = json.load(f)
            if cfg:  # Only apply if non-empty.
                config_path = default_cfg

    if config_path is not None:
        print(f"Applying extensions from: {config_path}")
        fractures = apply_extensions(fractures, names, config_path)
    else:
        print("No extension config applied (fault_extensions.json is empty or absent).")

    # --------------------------------------------------------- print summary
    _print_summary(fractures, names, source_dirs, n_pts_per_frac)

    # ----------------------------------------------------------- export VTU
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a minimal domain enclosing all fractures for export.
    all_pts = np.hstack([f.pts for f in fractures])  # (3, total_corners)
    pad = 1.0
    domain = pp.Domain(
        bounding_box={
            "xmin": all_pts[0].min() - pad,
            "xmax": all_pts[0].max() + pad,
            "ymin": all_pts[1].min() - pad,
            "ymax": all_pts[1].max() + pad,
            "zmin": all_pts[2].min() - pad,
            "zmax": all_pts[2].max() + pad,
        }
    )
    network = FractureNetwork3d(fractures=fractures, domain=domain)
    network.to_file(out_path)
    print(f"Exported {len(fractures)} fracture rectangle(s) to: {out_path.resolve()}")

    # ------------------------------------------------ export raw point clouds
    clouds_path = Path(args.out_clouds)
    _export_point_clouds_vtu(all_clouds, all_cloud_names, clouds_path)
    print(f"Exported {len(all_clouds)} point cloud(s) to:   {clouds_path.resolve()}")

    # ------------------------------------------------------------- guidance
    print(
        "\nNext steps:"
        "\n  1. Open both VTU files in ParaView to inspect fracture positions"
        "\n     relative to the raw point clouds."
        "\n     Use 'fault_id' scalar on the point cloud to filter individual faults."
        "\n  2. Edit fault_extensions.json to configure extensions, e.g.:"
        '\n       {"0003": {"type": "uniform", "scale": 1.5},'
        '\n        "0027": {"type": "directional", "axis": [1,0,0], "amount": 500}}'
        "\n  3. Re-run:  python preprocess_fault_planes.py --config fault_extensions.json"
        "\n  4. Once satisfied, use FaultPlaneGeometry in your model with params:"
        '\n       "fault_plane_dirs": ["UiB_SURF_faults", "point_cloud_clusters"],'
        '\n       "fault_extension_config": "fault_extensions.json"'
    )


if __name__ == "__main__":
    main()
