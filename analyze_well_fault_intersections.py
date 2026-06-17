"""Rank fault planes by extension needed to intersect well trajectories.

This script is intentionally separate from preprocess_fault_planes.py so it can
be reused for all wells and for different model setups without cluttering the
geometry-export workflow.

What it computes
----------------
For each selected well and each loaded fault rectangle:
1. Intersect the well segment with the fracture plane.
2. If the segment intersects the plane, compute the minimum in-plane extension
   needed for the rectangle to include the intersection point.

Reported metrics:
- min_extension_distance_m:
    Euclidean in-plane distance (m) from current rectangle boundary to the
    well/plane intersection point (0 if already intersecting).
- uniform_scale:
    Uniform scale factor around rectangle center required to include the point.
- directional_axis:
    Suggested unit vector `[ax, ay, az]` for the `"directional"` extension type.
- directional_amount_m:
    Suggested extension amount (m) along `directional_axis`.

"No feasible in-plane extension" means the well segment does not intersect the
fracture plane at all (so enlarging the rectangle in-plane cannot help).

Usage examples
--------------
python analyze_well_fault_intersections.py
python analyze_well_fault_intersections.py --dirs point_cloud_clusters
python analyze_well_fault_intersections.py --wells 68-20RD
python analyze_well_fault_intersections.py --report data/well_fault_intersections.json
python analyze_well_fault_intersections.py --report-csv data/well_fault_intersections.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import porepy as pp

# Keep local imports robust when run from any cwd.
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from fault_planes import apply_extensions, load_fault_planes  # noqa: E402


def _build_well_segment(
    well_sheet: str,
    well_file: Path,
    top_z: float = 0.0,
    straight_wells: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build well trajectory segment consistent with CosoGeometry.set_well_network()."""
    with pd.ExcelFile(well_file) as xls:
        df = pd.read_excel(xls, sheet_name=well_sheet, usecols=[3, 4, 5], skiprows=1)

    # File order is y, x, z -> convert to x, y, z.
    df = df.iloc[:, [1, 0, 2]]
    pts = df.values.T.astype(float)

    # Flip z to Coso convention.
    pts[2] *= -1

    pt0 = pts[:, 0].copy()
    pt0[2] = top_z

    if straight_wells:
        pt1 = pts[:, np.argmin(pts[2])].copy()
        return pt0, pt1

    # Polyline wells are reduced to top + last point for this analysis helper.
    # This keeps behaviour simple and conservative for intersection ranking.
    pt1 = pts[:, -1].copy()
    return pt0, pt1


def _required_extension_distance(
    frac: pp.PlaneFracture,
    p: np.ndarray,
) -> tuple[float, float]:
    """Return (min_in_plane_distance, uniform_scale) to include point p."""
    pts3 = frac.pts

    e1 = pts3[:, 1] - pts3[:, 0]
    e2 = pts3[:, 3] - pts3[:, 0]
    L1 = np.linalg.norm(e1)
    L2 = np.linalg.norm(e2)

    if L1 < 1e-12 or L2 < 1e-12:
        return float("inf"), float("inf")

    u1 = e1 / L1
    u2 = e2 / L2

    center = 0.5 * (pts3[:, 0] + pts3[:, 2])
    d = p - center
    x = d @ u1
    y = d @ u2

    hx, hy = L1 / 2.0, L2 / 2.0
    dx = max(abs(x) - hx, 0.0)
    dy = max(abs(y) - hy, 0.0)

    min_dist = float(np.hypot(dx, dy))
    uniform_scale = float(max(abs(x) / hx, abs(y) / hy, 1.0))
    return min_dist, uniform_scale


def _suggest_directional_extension(
    frac: pp.PlaneFracture,
    p: np.ndarray,
) -> tuple[list[float], float]:
    """Suggest (`axis`, `amount`) for the `"directional"` extension mode.

    The axis is chosen as the in-plane direction from rectangle center toward the
    well/plane intersection point. The amount is the support-function shortfall along
    that axis, i.e. the minimum shift required so the translated "front" edge reaches
    the point under the current directional-extension rule.
    """
    pts = frac.pts
    center = 0.5 * (pts[:, 0] + pts[:, 2])

    # Project vector center->p into the fracture plane.
    n = frac.normal.ravel()
    n = n / np.linalg.norm(n)
    v = p - center
    v_plane = v - (v @ n) * n

    if np.linalg.norm(v_plane) < 1e-12:
        # Fallback: use first rectangle edge direction if point is near center.
        edge = pts[:, 1] - pts[:, 0]
        if np.linalg.norm(edge) < 1e-12:
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = edge / np.linalg.norm(edge)
    else:
        axis = v_plane / np.linalg.norm(v_plane)

    # Support-function gap along axis.
    support_now = float(np.max(axis @ pts))
    support_target = float(axis @ p)
    amount = max(0.0, support_target - support_now)

    return axis.tolist(), amount


def _analyze_one_well(
    well_name: str,
    p_top: np.ndarray,
    p_bot: np.ndarray,
    fractures: list[pp.PlaneFracture],
    fracture_ids: list[str],
) -> dict:
    """Analyze all fractures for one well segment."""
    v = p_bot - p_top
    results = []

    for fid, frac in zip(fracture_ids, fractures):
        n = frac.normal.ravel()
        p0 = frac.pts[:, 0]
        denom = float(n @ v)

        if abs(denom) < 1e-10:
            results.append(
                {
                    "fracture": fid,
                    "feasible": False,
                    "reason": "well_segment_parallel_to_plane",
                    "min_extension_distance_m": None,
                    "uniform_scale": None,
                    "already_intersects": False,
                }
            )
            continue

        t = float(-(n @ (p_top - p0)) / denom)
        if t < 0.0 or t > 1.0:
            results.append(
                {
                    "fracture": fid,
                    "feasible": False,
                    "reason": "plane_intersection_outside_well_segment",
                    "min_extension_distance_m": None,
                    "uniform_scale": None,
                    "already_intersects": False,
                }
            )
            continue

        p_int = p_top + t * v
        min_dist, scale = _required_extension_distance(frac, p_int)
        axis, amount = _suggest_directional_extension(frac, p_int)
        results.append(
            {
                "fracture": fid,
                "feasible": True,
                "reason": "ok",
                "min_extension_distance_m": min_dist,
                "uniform_scale": scale,
                "directional_axis": axis,
                "directional_amount_m": amount,
                "already_intersects": bool(min_dist < 1e-8),
                "segment_t": t,
            }
        )

    feasible = [r for r in results if r["feasible"]]
    feasible.sort(key=lambda r: r["min_extension_distance_m"])

    best = feasible[0] if feasible else None
    return {
        "well": well_name,
        "p_top": p_top.tolist(),
        "p_bottom": p_bot.tolist(),
        "best": best,
        "ranked_feasible": feasible,
        "infeasible": [r for r in results if not r["feasible"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank fault planes by extension needed to intersect wells."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["point_cloud_clusters"],
        help="Directories containing fault CSV point clouds.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Fault stems to exclude.",
    )
    parser.add_argument(
        "--config",
        default="fault_extensions.json",
        help="Extension config JSON (applied before analysis).",
    )
    parser.add_argument(
        "--well-file",
        default="data/wellbores.xlsx",
        help="Path to wellbores Excel file.",
    )
    parser.add_argument(
        "--wells",
        nargs="+",
        default=["68-20RD", "16A-20", "16B-20"],
        help="Well sheet names to analyze.",
    )
    parser.add_argument(
        "--top-z",
        type=float,
        default=0.0,
        help="Top z used for synthetic top well point (default 0.0).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON output path for full results.",
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        help="Optional CSV output path (flat table, one row per well/fracture pair).",
    )
    args = parser.parse_args()

    root = _script_dir

    fractures: list[pp.PlaneFracture] = []
    fracture_ids: list[str] = []

    for d in args.dirs:
        directory = root / d
        batch = load_fault_planes(directory, exclude=args.exclude)
        fractures.extend(batch)
        fracture_ids.extend([f"{d}/{f.fault_name}" for f in batch])

    if not fractures:
        raise ValueError("No fractures loaded. Check --dirs and --exclude.")

    config_path = root / args.config
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if cfg:
            # apply_extensions keys by stem; preserve ids in fracture_ids.
            stems = [fid.split("/")[-1] for fid in fracture_ids]
            fractures = apply_extensions(fractures, stems, config_path)

    well_file = root / args.well_file

    all_results = []
    for well in args.wells:
        p_top, p_bot = _build_well_segment(
            well, well_file, top_z=args.top_z, straight_wells=True
        )
        res = _analyze_one_well(well, p_top, p_bot, fractures, fracture_ids)
        all_results.append(res)

    # Console summary.
    for res in all_results:
        print()
        print(f"Well: {res['well']}")
        best = res["best"]
        if best is None:
            print(
                "  No feasible in-plane extension (segment does not meet any fracture plane)."
            )
            continue

        print(
            "  Best fracture: "
            f"{best['fracture']} | min_extension={best['min_extension_distance_m']:.2f} m "
            f"| uniform_scale={best['uniform_scale']:.3f} "
            f"| already_intersects={best['already_intersects']}"
        )
        print("  Top 5 feasible:")
        for r in res["ranked_feasible"][:5]:
            print(
                "    "
                f"{r['fracture']}: dist={r['min_extension_distance_m']:.2f} m, "
                f"scale={r['uniform_scale']:.3f}, already={r['already_intersects']}"
            )

    if args.report:
        report_path = root / args.report
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({"results": all_results}, f, indent=2)
        print(f"\nWrote report: {report_path}")

    if args.report_csv:
        rows: list[dict] = []
        for res in all_results:
            well = res["well"]
            p_top = res["p_top"]
            p_bottom = res["p_bottom"]

            for rank, r in enumerate(res["ranked_feasible"], start=1):
                rows.append(
                    {
                        "well": well,
                        "fracture": r["fracture"],
                        "feasible": True,
                        "reason": r["reason"],
                        "rank_feasible": rank,
                        "min_extension_distance_m": r["min_extension_distance_m"],
                        "uniform_scale": r["uniform_scale"],
                        "directional_axis_x": r["directional_axis"][0],
                        "directional_axis_y": r["directional_axis"][1],
                        "directional_axis_z": r["directional_axis"][2],
                        "directional_amount_m": r["directional_amount_m"],
                        "already_intersects": r["already_intersects"],
                        "segment_t": r.get("segment_t"),
                        "p_top_x": p_top[0],
                        "p_top_y": p_top[1],
                        "p_top_z": p_top[2],
                        "p_bottom_x": p_bottom[0],
                        "p_bottom_y": p_bottom[1],
                        "p_bottom_z": p_bottom[2],
                    }
                )

            for r in res["infeasible"]:
                rows.append(
                    {
                        "well": well,
                        "fracture": r["fracture"],
                        "feasible": False,
                        "reason": r["reason"],
                        "rank_feasible": None,
                        "min_extension_distance_m": None,
                        "uniform_scale": None,
                        "directional_axis_x": None,
                        "directional_axis_y": None,
                        "directional_axis_z": None,
                        "directional_amount_m": None,
                        "already_intersects": False,
                        "segment_t": None,
                        "p_top_x": p_top[0],
                        "p_top_y": p_top[1],
                        "p_top_z": p_top[2],
                        "p_bottom_x": p_bottom[0],
                        "p_bottom_y": p_bottom[1],
                        "p_bottom_z": p_bottom[2],
                    }
                )

        csv_path = root / args.report_csv
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Wrote CSV report: {csv_path}")


if __name__ == "__main__":
    main()
