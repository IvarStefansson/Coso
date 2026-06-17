"""Utilities for loading and processing fault-plane point clouds.

Each CSV file contains a header ``x,y,z`` followed by rows of UTM coordinates
(easting, northing, elevation in metres).  The functions here

1. transform those UTM coordinates to the Coso-local coordinate system
   (see ``transform_coords``),
2. fit a best-fit plane to the resulting point cloud via SVD
   (see ``fit_plane_svd``),
3. construct the minimum-area oriented bounding rectangle in that plane and
   return it as a :class:`porepy.PlaneFracture` (see ``bounding_rectangle`` and
   ``load_fault_planes``), and
4. apply user-requested geometric extensions so that fractures form a more
   connected network (see ``apply_extensions``).

Coordinate system
-----------------
The Coso local system is centred on well 77-7:

    Easting_ref  = 4.278743722753740e+05  m
    Northing_ref = 3.987675558142440e+06  m
    Depth_ref    = -1.344168000000000e+03  m  (elevation of reference point)

The transformation (equivalent to Joanna's Matlab script, but converting to
negative-down z as used throughout the Coso models) is:

    x_coso [m] =  data.x - Easting_ref
    y_coso [m] =  data.y - Northing_ref
    z_coso [m] =  data.z + Depth_ref      # = data.z - 1344.168 m

"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import porepy as pp
from scipy.spatial import ConvexHull

# ---------------------------------------------------------------------------
# Reference-point constants
# ---------------------------------------------------------------------------

EASTING_REF: float = 4.278743722753740e05  # m
NORTHING_REF: float = 3.987675558142440e06  # m
DEPTH_REF: float = -1.344168000000000e03  # m  (elevation of well 77-7)


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------


def transform_coords(df: pd.DataFrame) -> np.ndarray:
    """Transform UTM point-cloud coordinates to Coso-local metres.

    Parameters:
        df: DataFrame with columns ``x`` (easting), ``y`` (northing),
            ``z`` (elevation, negative below surface), all in metres.

    Returns:
        Array of shape (3, N) with columns [x_coso, y_coso, z_coso].
        z_coso is negative-down, consistent with the rest of the Coso models.
    """
    x = df["x"].to_numpy() - EASTING_REF
    y = df["y"].to_numpy() - NORTHING_REF
    # data.z is elevation (negative underground); adding DEPTH_REF (which is also
    # negative) shifts to depth below reference point, giving negative z.
    z = df["z"].to_numpy() + DEPTH_REF
    return np.vstack([x, y, z])


# ---------------------------------------------------------------------------
# Plane fitting
# ---------------------------------------------------------------------------


def fit_plane_svd(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a best-fit plane to a 3-D point cloud via SVD.

    Parameters:
        pts: Array of shape (3, N) with the point cloud.

    Returns:
        centroid: Shape (3,) – mean of the point cloud.
        normal:   Shape (3,) – unit normal of the fitted plane.  The sign is
                  chosen so that the normal points towards +z (upward) when
                  possible.
    """
    centroid = pts.mean(axis=1)
    centered = pts - centroid[:, np.newaxis]
    # SVD of the (3, N) centred matrix.  Left singular vectors (columns of U)
    # live in 3-D space; the one with the *smallest* singular value gives the
    # direction of least variance, i.e., the plane normal.
    U, _, _ = np.linalg.svd(centered, full_matrices=True)
    normal = U[:, -1]  # shape (3,)
    # Ensure upward-pointing normal (positive z component).
    if normal[2] < 0:
        normal = -normal
    return centroid, normal


# ---------------------------------------------------------------------------
# Minimum-area bounding rectangle (rotating calipers)
# ---------------------------------------------------------------------------


def _min_area_rect_2d(pts_2d: np.ndarray) -> np.ndarray:
    """Minimum-area enclosing rectangle for a 2-D point set.

    Uses the rotating-calipers algorithm on the convex hull: for each hull
    edge, compute the bounding rectangle aligned with that edge and keep the
    one with the smallest area.

    Parameters:
        pts_2d: Shape (2, N) – 2-D point cloud.

    Returns:
        corners: Shape (2, 4) – four corners of the rectangle in
                 counter-clockwise order.
    """
    pts_T = pts_2d.T  # (N, 2) for ConvexHull
    if pts_T.shape[0] < 3:
        # Degenerate: fall back to axis-aligned bbox.
        x_min, x_max = pts_2d[0].min(), pts_2d[0].max()
        y_min, y_max = pts_2d[1].min(), pts_2d[1].max()
        return np.array([[x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max]])

    hull = ConvexHull(pts_T)
    hull_pts = pts_T[hull.vertices]  # (K, 2), counter-clockwise
    n = len(hull_pts)

    min_area = np.inf
    best: np.ndarray | None = None

    for i in range(n):
        # Unit vector along this hull edge.
        edge = hull_pts[(i + 1) % n] - hull_pts[i]
        length = np.linalg.norm(edge)
        if length < 1e-12:
            continue
        u = edge / length  # along edge
        v = np.array([-u[1], u[0]])  # perpendicular (90° ccw)

        # Project all hull points onto (u, v) axes.
        proj_u = hull_pts @ u
        proj_v = hull_pts @ v
        u_min, u_max = proj_u.min(), proj_u.max()
        v_min, v_max = proj_v.min(), proj_v.max()

        area = (u_max - u_min) * (v_max - v_min)
        if area < min_area:
            min_area = area
            # Four corners in 2-D (counter-clockwise).
            best = np.column_stack(
                [
                    u_min * u + v_min * v,
                    u_max * u + v_min * v,
                    u_max * u + v_max * v,
                    u_min * u + v_max * v,
                ]
            )  # (2, 4)

    assert best is not None
    return best


# ---------------------------------------------------------------------------
# Bounding rectangle
# ---------------------------------------------------------------------------


def bounding_rectangle(
    pts: np.ndarray,
    centroid: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Compute the minimum-area oriented bounding rectangle for a planar point cloud.

    The rectangle is found by the rotating-calipers algorithm applied to the
    convex hull of the 2-D projected points.  This yields the smallest
    enclosing rectangle and naturally aligns with the dominant orientation of
    the point cloud, regardless of how the local 2-D axes happen to be defined.

    Parameters:
        pts:      Shape (3, N) – point cloud (already in Coso coordinates).
        centroid: Shape (3,)   – centroid / origin of the local plane frame.
        normal:   Shape (3,)   – unit normal of the plane.

    Returns:
        corners: Shape (3, 4) – the four corners of the rectangle, ordered
                 counter-clockwise when viewed from the direction of *normal*.
    """
    # Rotation matrix R: first two rows span the plane, third is along normal.
    # check_planar=False because the cloud is only approximately planar.
    R = pp.map_geometry.project_plane_matrix(pts, normal=normal, check_planar=False)

    # Project to 2-D local coordinates.
    local_2d = (R @ (pts - centroid[:, np.newaxis]))[:2]  # (2, N)

    # Minimum-area bounding rectangle in the 2-D frame.
    corners_2d_local = _min_area_rect_2d(local_2d)  # (2, 4)

    # Lift back to 3-D (z = 0 in local frame → on the fitted plane).
    corners_3d_local = np.vstack([corners_2d_local, np.zeros(4)])  # (3, 4)

    # Rotate back to world 3-D and translate to centroid.
    corners_3d = R.T @ corners_3d_local + centroid[:, np.newaxis]
    return corners_3d


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_point_clouds(
    directory: str | Path,
    exclude: list[str] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Load and transform point clouds from a directory of CSV files.

    Parameters:
        directory: Path to the directory containing the CSV files.
        exclude:   Stem names (without extension) to skip.

    Returns:
        Tuple of (point_clouds, names) where each element of *point_clouds* is
        an array of shape (3, N) in Coso-local metres, and *names* contains
        the corresponding file stem.
    """
    exclude_set: set[str] = set(exclude) if exclude else set()
    directory = Path(directory)
    clouds: list[np.ndarray] = []
    names: list[str] = []

    for csv_path in sorted(directory.glob("*.csv")):
        stem = csv_path.stem
        if stem in exclude_set:
            continue
        df = pd.read_csv(csv_path)
        clouds.append(transform_coords(df))
        names.append(stem)

    return clouds, names


def load_fault_planes(
    directory: str | Path,
    exclude: list[str] | None = None,
) -> list[pp.PlaneFracture]:
    """Load fault-plane point clouds from a directory of CSV files.

    Each ``*.csv`` file is expected to have a header ``x,y,z`` followed by rows of UTM
    coordinates (metres).  The function transforms each cloud to Coso-local coordinates,
    fits a plane, and returns an axis-aligned bounding-rectangle
    :class:`~porepy.PlaneFracture`.

    Parameters:
        directory: Path to the directory containing the CSV files.
        exclude:   Stem names (without extension) to skip, e.g. ``["0003", "0027"]``.

    Returns:
        List of :class:`~porepy.PlaneFracture` objects, one per CSV file,
        tagged with ``{"fault_name": <stem>}``.
    """
    exclude_set: set[str] = set(exclude) if exclude else set()
    directory = Path(directory)
    fractures: list[pp.PlaneFracture] = []

    for csv_path in sorted(directory.glob("*.csv")):
        stem = csv_path.stem
        if stem in exclude_set:
            continue

        df = pd.read_csv(csv_path)
        pts = transform_coords(df)
        centroid, normal = fit_plane_svd(pts)
        corners = bounding_rectangle(pts, centroid, normal)

        frac = pp.PlaneFracture(corners, check_convexity=False)
        frac.fault_name = stem
        fractures.append(frac)

    return fractures


# ---------------------------------------------------------------------------
# Extension utilities
# ---------------------------------------------------------------------------


def _uniform_extension(frac: pp.PlaneFracture, scale: float) -> pp.PlaneFracture:
    """Scale the four corners of *frac* away from its centroid by *scale*."""
    centroid = frac.center.ravel()
    new_pts = centroid[:, np.newaxis] + scale * (frac.pts - centroid[:, np.newaxis])
    new_frac = pp.PlaneFracture(new_pts, check_convexity=False)
    new_frac.fault_name = frac.fault_name
    return new_frac


def _directional_extension(
    frac: pp.PlaneFracture,
    axis: list[float],
    amount: float,
) -> pp.PlaneFracture:
    """Extend the two corners most aligned with *axis* by *amount* metres.

    Parameters:
        frac:   The fracture to extend.
        axis:   Direction vector (need not be unit length). The two corners with the
            largest positive projection onto this axis will be shifted in the plane.
            The remaining
            two corners will be left unchanged.
        amount: Distance in metres to shift the two leading corners.

    Returns:
        A new :class:`~porepy.PlaneFracture` with the extended corners.
    """
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)

    centroid = frac.center.ravel()
    corners = frac.pts.copy()  # (3, 4)
    vecs = corners - centroid[:, np.newaxis]  # vectors from centroid to corners

    # Dot product of each corner vector with the axis direction.
    dots = ax @ vecs  # shape (4,)

    # Shift the two corners with the largest dot product.
    top2 = np.argsort(dots)[-2:]
    # The shift prolongs the fracture in-plane by adding a patch of width ~amount at the
    # leading edge.  The shift direction is parallel to the axis but perpendicular to
    # the normal, i.e., in the plane of the fracture.
    along_edge = np.cross(frac.normal.ravel(), ax)  # in-plane direction along the edge
    # in-plane direction perpendicular to edge (i.e., outward normal to leading edge),
    # sign ensures outward.
    shift_direction = -np.cross(frac.normal.ravel(), along_edge)
    shift_direction = shift_direction / np.linalg.norm(shift_direction)  # unit vector
    corners[:, top2] += amount * shift_direction[:, np.newaxis]

    new_frac = pp.PlaneFracture(corners, check_convexity=False)
    new_frac.fault_name = frac.fault_name
    return new_frac


def apply_extension(frac: pp.PlaneFracture, spec: dict) -> pp.PlaneFracture:
    """Apply a single extension specification to a fracture.

    Parameters:
        frac: The fracture to extend.
        spec: A dict with key ``"type"`` and type-specific parameters:

            ``"uniform"``
                Scale all corners away from the centroid.
                Requires ``"scale"`` (float, e.g. 1.5 doubles the size).

            ``"directional"``
                Shift the two leading corners along a given axis.
                Requires ``"axis"`` (list of 3 floats) and
                ``"amount"`` (float, metres).

            ``"to_intersect"``
                **Not yet implemented** – extend an edge until it meets a
                neighbouring fracture's plane.  Raises ``NotImplementedError``.

    Returns:
        A new :class:`~porepy.PlaneFracture` with the extension applied.
    """
    kind = spec["type"]
    if kind == "uniform":
        return _uniform_extension(frac, float(spec["scale"]))
    elif kind == "directional":
        return _directional_extension(frac, spec["axis"], float(spec["amount"]))
    elif kind == "to_intersect":
        raise NotImplementedError(
            "Extension type 'to_intersect' is not yet implemented."
        )
    else:
        raise ValueError(f"Unknown extension type: {kind!r}")


def apply_extensions(
    fractures: list[pp.PlaneFracture],
    names: list[str],
    config_path: str | Path,
) -> list[pp.PlaneFracture]:
    """Apply per-fracture extension specs from a JSON config file.

    Parameters:
        fractures:   List of fractures (one per fault plane).
        names:       Fault stem names corresponding to each fracture (same order).
        config_path: Path to a JSON file mapping stem names to extension specs.
                     Fractures whose names do not appear in the config are
                     returned unchanged.

    Returns:
        A new list of fractures with extensions applied where configured.
    """
    with open(config_path) as f:
        config: dict = json.load(f)

    result = []
    for frac, name in zip(fractures, names):
        if name in config:
            result.append(apply_extension(frac, config[name]))
        else:
            result.append(frac)
    return result
