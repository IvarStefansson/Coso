"""Inspect fault 0005's bounding rectangle (length axis, low-y end) and its
current proximity/intersection with fault 0001, to decide how to extend 0005
so the 0005-0001 intersection segment is less marginal.

Fault 0005 is the fracture intersected by two wells (well_num 1 and 2); see
check_fault_0001_well_intersections.py and project-coso-geometry-facts memory.
"""

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # this script's relative paths (point_cloud_clusters/, etc.)
# and geometry.py's sys.path[0]-based data lookups assume the repo root as cwd.

import fault_planes as fp

EXCLUDE = [
    "0002",
    "0004",
    "0006",
    "0007",
    "0008",
    "0009",
    "0010",
    "0011",
    "0012",
]


def rectangle_frame(corners):
    """corners: (3,4) ccw. Returns origin, u (edge 0->1), v (edge 0->3)."""
    origin = corners[:, 0]
    u = corners[:, 1] - origin
    v = corners[:, 3] - origin
    return origin, u, v


def point_in_rectangle_param(corners, point):
    """Express *point* (assumed on the rectangle's plane) as origin + a*u + b*v.

    Returns (a, b). Point is inside the rectangle iff 0<=a<=1 and 0<=b<=1.
    """
    origin, u, v = rectangle_frame(corners)
    M = np.column_stack([u, v])  # (3,2)
    sol, *_ = np.linalg.lstsq(M, point - origin, rcond=None)
    return sol[0], sol[1]


def plane_intersection_line(c1, c2):
    """Infinite-plane intersection line between the planes of rectangles c1, c2.

    Returns (point_on_line, direction) or None if planes are ~parallel.
    """
    centroid1 = c1.mean(axis=1)
    centroid2 = c2.mean(axis=1)
    _, n1 = fp.fit_plane_svd(c1)
    _, n2 = fp.fit_plane_svd(c2)
    d = np.cross(n1, n2)
    norm_d = np.linalg.norm(d)
    if norm_d < 1e-9:
        return None
    d = d / norm_d
    # Solve for a point on both planes: p = centroid1 + s*n1_perp... use the
    # standard 2-plane intersection formula.
    # Plane i: n_i . (x - centroid_i) = 0  =>  n_i . x = n_i . centroid_i
    A = np.array([n1, n2, d])
    b = np.array([n1 @ centroid1, n2 @ centroid2, 0.0])
    p0 = np.linalg.solve(A, b)
    return p0, d


def segment_on_line_inside_rectangle(corners, p0, d, t_lo=-1e5, t_hi=1e5, n_samples=200001):
    """Sample t in [t_lo, t_hi] along p0 + t*d and find the range where the
    point falls inside *corners*' rectangle. Returns (t_min, t_max) or None."""
    ts = np.linspace(t_lo, t_hi, n_samples)
    pts = p0[:, None] + np.outer(d, ts)
    origin, u, v = rectangle_frame(corners)
    M = np.column_stack([u, v])
    Minv = np.linalg.pinv(M)
    ab = Minv @ (pts - origin[:, None])  # (2, n_samples)
    inside = (ab[0] >= 0) & (ab[0] <= 1) & (ab[1] >= 0) & (ab[1] <= 1)
    if not inside.any():
        return None
    idx = np.where(inside)[0]
    return ts[idx[0]], ts[idx[-1]]


def report_overlap(c_0005, c_0001, label):
    line = plane_intersection_line(c_0005, c_0001)
    if line is None:
        print(f"[{label}] Planes are ~parallel, no intersection line.")
        return
    p0, d = line
    seg5 = segment_on_line_inside_rectangle(c_0005, p0, d)
    seg1 = segment_on_line_inside_rectangle(c_0001, p0, d)
    print(f"\n--- {label} ---")
    print(f"  0005 covers line param range: {seg5}")
    print(f"  0001 covers line param range: {seg1}")
    if seg5 is None or seg1 is None:
        print("  No in-rectangle overlap at all (rectangles' extent doesn't reach the line).")
        return
    lo = max(seg5[0], seg1[0])
    hi = min(seg5[1], seg1[1])
    if hi <= lo:
        print(f"  NO OVERLAP (gap = {lo - hi:.2f} m)")
    else:
        print(f"  OVERLAP LENGTH: {hi - lo:.2f} m  (segment from t={lo:.2f} to t={hi:.2f})")


def main():
    fractures = fp.load_fault_planes("point_cloud_clusters", exclude=EXCLUDE)
    names = [f.fault_name for f in fractures]

    idx_0005 = names.index("0005")
    idx_0001 = names.index("0001")

    extended = fp.apply_extensions(fractures, names, "fault_extensions.json")
    c_0005 = extended[idx_0005].pts
    c_0001 = extended[idx_0001].pts

    report_overlap(c_0005, c_0001, "CURRENT config")

    # Determine 0005's length axis and which end-cap (side 0-1 vs side 2-3)
    # has the lower y coordinate.
    centroid = c_0005.mean(axis=1)
    e01 = c_0005[:, 1] - c_0005[:, 0]
    e03 = c_0005[:, 3] - c_0005[:, 0]
    if np.linalg.norm(e03) > np.linalg.norm(e01):
        length_dir = e03
        end_caps = {"side(0,1)": (0, 1), "side(2,3)": (2, 3)}
    else:
        length_dir = e01
        end_caps = {"side(0,3)": (0, 3), "side(1,2)": (1, 2)}
    length_unit = length_dir / np.linalg.norm(length_dir)
    print(f"\n0005 length direction (unit): {length_unit}")

    for name, (i, j) in end_caps.items():
        y_mean = 0.5 * (c_0005[1, i] + c_0005[1, j])
        proj = 0.5 * ((c_0005[:, i] - centroid) @ length_unit + (c_0005[:, j] - centroid) @ length_unit)
        print(f"  end-cap {name} (corners {i},{j}): mean y = {y_mean:.2f}, "
              f"mean projection onto length axis = {proj:.2f}")

    print("\nSigned distance of each 0005 corner to 0001's plane (current config):")
    centroid_0001, normal_0001 = fp.fit_plane_svd(c_0001)
    for i in range(4):
        signed = np.dot(c_0005[:, i] - centroid_0001, normal_0001)
        print(f"  corner {i} (y={c_0005[1,i]:.1f}): {signed:.2f} m")

    # Try candidate extra extensions of 0005 along the length axis, on the
    # low-y end, and see how the overlap length responds.
    import fault_planes as fp2  # re-import alias for clarity below

    raw_fractures = fp.load_fault_planes("point_cloud_clusters", exclude=EXCLUDE)
    raw_names = [f.fault_name for f in raw_fractures]
    raw_0005 = raw_fractures[raw_names.index("0005")]

    with open("fault_extensions.json") as f:
        import json
        base_config = json.load(f)

    low_y_axis = -length_unit if end_caps_low_y_is_negative(c_0005, end_caps, length_unit, centroid) else length_unit

    for amount in [0, 50, 100, 150, 200, 300]:
        specs = list(base_config.get("0005", [])) if isinstance(base_config.get("0005"), list) else (
            [base_config["0005"]] if "0005" in base_config else []
        )
        if amount > 0:
            specs = specs + [{"type": "directional", "axis": list(low_y_axis), "amount": float(amount)}]
        frac = raw_0005
        for spec in specs:
            frac = fp.apply_extension(frac, spec)
        report_overlap(frac.pts, c_0001, f"0005 length-extended by {amount} m (low-y end)")


def end_caps_low_y_is_negative(c_0005, end_caps, length_unit, centroid):
    """Return True if the low-y end-cap sits at negative projection onto length_unit."""
    items = list(end_caps.items())
    (name_a, (ia, ja)), (name_b, (ib, jb)) = items
    y_a = 0.5 * (c_0005[1, ia] + c_0005[1, ja])
    y_b = 0.5 * (c_0005[1, ib] + c_0005[1, jb])
    proj_a = 0.5 * ((c_0005[:, ia] - centroid) @ length_unit + (c_0005[:, ja] - centroid) @ length_unit)
    proj_b = 0.5 * ((c_0005[:, ib] - centroid) @ length_unit + (c_0005[:, jb] - centroid) @ length_unit)
    low_y_name, low_y_proj = (name_a, proj_a) if y_a < y_b else (name_b, proj_b)
    return low_y_proj < 0


if __name__ == "__main__":
    main()
