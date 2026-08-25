"""Tests for fault_planes.py's index-order-robust rectangle geometry helpers.

pp.PlaneFracture re-sorts its 4 corners by polar angle around the centroid on
every construction, so corner index 0 is not stable across repeated
apply_extension calls (see fault_planes.rectangle_edge_axes docstring). These
tests confirm rectangle_edge_axes/classify_rectangle_ends are robust to that
reordering, and that resolve_directional_axis/apply_extension wire it up
correctly, including against the real Coso fault-plane data.
"""

from __future__ import annotations

import numpy as np
import pytest
import porepy as pp

import fault_planes as fp


class _FakeFrac:
    """Minimal stand-in exposing only what the rectangle helpers read."""

    def __init__(self, pts: np.ndarray):
        self.pts = pts

    @property
    def center(self):
        return self.pts.mean(axis=1, keepdims=True)


def _axis_aligned_rectangle_pts() -> np.ndarray:
    # 10 (x) by 3 (y), in the z=0 plane. Long axis = x, short axis = y.
    return np.array(
        [
            [0.0, 10.0, 10.0, 0.0],
            [0.0, 0.0, 3.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )


def test_rectangle_edge_axes_axis_aligned():
    long_axis, short_axis = fp.rectangle_edge_axes(_FakeFrac(_axis_aligned_rectangle_pts()))
    assert np.allclose(np.abs(long_axis), [1, 0, 0], atol=1e-8)
    assert np.allclose(np.abs(short_axis), [0, 1, 0], atol=1e-8)


def test_rectangle_edge_axes_robust_to_corner_rotation():
    base = _axis_aligned_rectangle_pts()
    baseline_long, baseline_short = fp.rectangle_edge_axes(_FakeFrac(base))
    for shift in [1, 2, 3]:
        rotated = np.roll(base, shift, axis=1)
        long_axis, short_axis = fp.rectangle_edge_axes(_FakeFrac(rotated))
        assert np.allclose(np.abs(long_axis), np.abs(baseline_long), atol=1e-8)
        assert np.allclose(np.abs(short_axis), np.abs(baseline_short), atol=1e-8)


def test_rectangle_edge_axes_tilted_3d():
    # Rotate the axis-aligned rectangle into an arbitrary tilted plane.
    base = _axis_aligned_rectangle_pts()
    theta = 0.4
    R = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    tilted = R @ base
    long_axis, short_axis = fp.rectangle_edge_axes(_FakeFrac(tilted))
    expected_long = R @ np.array([1.0, 0, 0])
    expected_short = R @ np.array([0, 1.0, 0])
    assert np.allclose(np.abs(long_axis), np.abs(expected_long), atol=1e-8)
    assert np.allclose(np.abs(short_axis), np.abs(expected_short), atol=1e-8)


def test_rectangle_edge_axes_near_square_raises():
    pts = np.array(
        [
            [0.0, 10.0, 10.0, 0.0],
            [0.0, 0.0, 9.99, 9.99],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    with pytest.raises(ValueError, match="near-square"):
        fp.rectangle_edge_axes(_FakeFrac(pts))


def test_rectangle_edge_axes_wrong_corner_count_raises():
    pts = np.array([[0.0, 1.0, 2.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="4-corner"):
        fp.rectangle_edge_axes(_FakeFrac(pts))


def test_classify_rectangle_ends_axis_aligned():
    frac = _FakeFrac(_axis_aligned_rectangle_pts())
    long_axis, _ = fp.rectangle_edge_axes(frac)
    ends = fp.classify_rectangle_ends(frac, long_axis, "x")
    assert ends["low"] @ np.array([1.0, 0, 0]) < 0
    assert ends["high"] @ np.array([1.0, 0, 0]) > 0
    assert np.allclose(ends["low"], -ends["high"])


def test_classify_rectangle_ends_robust_to_corner_rotation():
    base = _axis_aligned_rectangle_pts()
    long_axis, _ = fp.rectangle_edge_axes(_FakeFrac(base))
    baseline = fp.classify_rectangle_ends(_FakeFrac(base), long_axis, "x")
    for shift in [1, 2, 3]:
        rotated = np.roll(base, shift, axis=1)
        rotated_long_axis, _ = fp.rectangle_edge_axes(_FakeFrac(rotated))
        ends = fp.classify_rectangle_ends(_FakeFrac(rotated), rotated_long_axis, "x")
        assert np.allclose(ends["low"], baseline["low"], atol=1e-8) or np.allclose(
            ends["low"], baseline["high"], atol=1e-8
        )
        # Regardless of which sign the axis itself came back as, "low" must
        # always point toward negative x and "high" toward positive x.
        assert ends["low"] @ np.array([1.0, 0, 0]) < 0
        assert ends["high"] @ np.array([1.0, 0, 0]) > 0


def test_classify_rectangle_ends_tie_raises():
    # Short axis (y) barely varies in x -- picking coordinate "x" here ties.
    frac = _FakeFrac(_axis_aligned_rectangle_pts())
    _, short_axis = fp.rectangle_edge_axes(frac)
    with pytest.raises(ValueError, match="tie"):
        fp.classify_rectangle_ends(frac, short_axis, "x")


def test_classify_rectangle_ends_bad_coordinate_raises():
    frac = _FakeFrac(_axis_aligned_rectangle_pts())
    long_axis, _ = fp.rectangle_edge_axes(frac)
    with pytest.raises(ValueError, match="coordinate"):
        fp.classify_rectangle_ends(frac, long_axis, "w")


# --- resolve_directional_axis -----------------------------------------------


def _real_rectangle() -> pp.PlaneFracture:
    return pp.PlaneFracture(_axis_aligned_rectangle_pts(), check_convexity=False)


def test_resolve_directional_axis_raw_form():
    frac = _real_rectangle()
    spec = {"type": "directional", "axis": [1.0, 2.0, 3.0], "amount": 5.0}
    axis = fp.resolve_directional_axis(frac, spec)
    np.testing.assert_array_equal(axis, [1.0, 2.0, 3.0])


def test_resolve_directional_axis_symbolic_form():
    frac = _real_rectangle()
    spec = {
        "type": "directional",
        "reference": "long",
        "toward": {"coordinate": "x", "sign": "high"},
        "amount": 5.0,
    }
    axis = fp.resolve_directional_axis(frac, spec)
    assert axis @ np.array([1.0, 0, 0]) > 0
    assert np.isclose(np.linalg.norm(axis), 1.0)


def test_resolve_directional_axis_both_forms_raises():
    frac = _real_rectangle()
    spec = {
        "axis": [1.0, 0.0, 0.0],
        "reference": "long",
        "toward": {"coordinate": "x", "sign": "high"},
    }
    with pytest.raises(ValueError, match="not both"):
        fp.resolve_directional_axis(frac, spec)


def test_resolve_directional_axis_neither_form_raises():
    frac = _real_rectangle()
    with pytest.raises(ValueError, match="needs either"):
        fp.resolve_directional_axis(frac, {})


def test_resolve_directional_axis_unknown_reference_raises():
    frac = _real_rectangle()
    spec = {"reference": "medium", "toward": {"coordinate": "x", "sign": "high"}}
    with pytest.raises(ValueError, match="'long' or 'short'"):
        fp.resolve_directional_axis(frac, spec)


def _sorted_corners(pts: np.ndarray) -> np.ndarray:
    """Corners sorted lexicographically, for order-independent comparison.

    pp.PlaneFracture re-sorts its corners by polar angle on every
    construction, and which corner ends up at index 0 can differ between two
    constructions of the SAME physical rectangle if they were reached via
    slightly different floating-point paths (e.g. different chains of
    apply_extension calls) -- exactly the reordering rectangle_edge_axes/
    classify_rectangle_ends are designed to be robust to. Comparing corner
    SETS (not raw column order) is the correct way to check two fracture
    constructions represent the same geometry.
    """
    order = np.lexsort(pts[::-1])
    return pts[:, order]


# --- Regression against the real Coso fault-plane data -----------------------


def test_symbolic_spec_reproduces_hand_derived_0005_vector():
    exclude = [
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
    fractures = fp.load_fault_planes("point_cloud_clusters", exclude=exclude)
    names = [f.fault_name for f in fractures]
    raw_0005 = fractures[names.index("0005")]

    # First existing spec (unchanged) applied to the raw fracture, as in
    # fault_extensions.json's "0005" list.
    first_spec = {"type": "directional", "axis": [-2, -1, 0], "amount": 150.0}
    after_first = fp.apply_extension(raw_0005, first_spec)

    hand_derived_spec = {
        "type": "directional",
        "axis": [0.35554499791190625, -0.876308509745568, -0.3250709925652],
        "amount": 150.0,
    }
    symbolic_spec = {
        "type": "directional",
        "reference": "long",
        "toward": {"coordinate": "y", "sign": "low"},
        "amount": 150.0,
    }

    via_hand_derived = fp.apply_extension(after_first, hand_derived_spec)
    via_symbolic = fp.apply_extension(after_first, symbolic_spec)

    # Compare corner SETS, not raw column order -- see _sorted_corners.
    np.testing.assert_allclose(
        _sorted_corners(via_hand_derived.pts), _sorted_corners(via_symbolic.pts), atol=1e-6
    )


def test_fault_extensions_json_end_to_end_matches_previous_hand_derived_result():
    """fault_extensions.json now uses symbolic specs for 0005/0001; confirm the
    full apply_extensions pipeline still produces the same geometry as the
    original hand-derived raw-vector specs it replaced."""
    exclude = [
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
    fractures = fp.load_fault_planes("point_cloud_clusters", exclude=exclude)
    names = [f.fault_name for f in fractures]

    current = fp.apply_extensions(fractures, names, "fault_extensions.json")

    original_config = {
        "0003": {
            "type": "directional",
            "axis": [0.346, -0.9157, -0.204],
            "amount": 200.0,
        },
        "0005": [
            {"type": "directional", "axis": [-2, -1, 0], "amount": 150.0},
            {
                "type": "directional",
                "axis": [
                    0.35554499791190625,
                    -0.876308509745568,
                    -0.3250709925652,
                ],
                "amount": 150.0,
            },
        ],
        "0001": [
            {
                "type": "directional",
                "axis": [482.640, 174.715, 1462.550],
                "amount": -100.0,
            },
            {
                "type": "directional",
                "axis": [-482.640, -174.715, -1462.550],
                "amount": -200.0,
            },
        ],
    }
    original = []
    for frac, name in zip(fractures, names):
        if name in original_config:
            specs = original_config[name]
            if isinstance(specs, dict):
                specs = [specs]
            for spec in specs:
                frac = fp.apply_extension(frac, spec)
        original.append(frac)

    for cur, orig, name in zip(current, original, names):
        np.testing.assert_allclose(
            _sorted_corners(cur.pts),
            _sorted_corners(orig.pts),
            atol=1e-3,
            err_msg=f"mismatch for fault {name}",
        )
