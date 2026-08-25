"""Tests for fault_planes.apply_extension/apply_extensions: backward
compatibility of the raw-"axis" form, and the new symbolic-axis chaining
behavior (a later spec in a list resolves against the already-extended
shape, not the original)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import porepy as pp

import fault_planes as fp


def _rectangle() -> pp.PlaneFracture:
    pts = np.array(
        [
            [0.0, 10.0, 10.0, 0.0],
            [0.0, 0.0, 3.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    frac = pp.PlaneFracture(pts, check_convexity=False)
    frac.fault_name = "test"
    return frac


def test_uniform_extension_unchanged():
    frac = _rectangle()
    spec = {"type": "uniform", "scale": 2.0}
    scaled = fp.apply_extension(frac, spec)
    centroid = frac.center.ravel()
    expected = centroid[:, None] + 2.0 * (frac.pts - centroid[:, None])
    # PlaneFracture re-sorts corners, so compare as sets via centroid+radii.
    np.testing.assert_allclose(
        np.sort(np.linalg.norm(scaled.pts - centroid[:, None], axis=0)),
        np.sort(np.linalg.norm(expected - centroid[:, None], axis=0)),
        atol=1e-8,
    )


def test_directional_raw_axis_unchanged_by_refactor():
    frac = _rectangle()
    spec = {"type": "directional", "axis": [1.0, 0.0, 0.0], "amount": 5.0}
    extended = fp.apply_extension(frac, spec)
    # Long axis (x) should have grown from 10 to 15.
    long_axis, _ = fp.rectangle_edge_axes(extended)
    assert np.isclose(np.abs(long_axis @ np.array([1.0, 0, 0])), 1.0)
    xs = extended.pts[0]
    assert np.isclose(xs.max() - xs.min(), 15.0, atol=1e-8)


def test_directional_symbolic_axis():
    frac = _rectangle()
    spec = {
        "type": "directional",
        "reference": "long",
        "toward": {"coordinate": "x", "sign": "high"},
        "amount": 5.0,
    }
    extended = fp.apply_extension(frac, spec)
    xs = extended.pts[0]
    assert np.isclose(xs.max() - xs.min(), 15.0, atol=1e-8)
    assert np.isclose(xs.min(), 0.0, atol=1e-8)


def test_chained_symbolic_specs_resolve_against_current_shape():
    """Two sequential symbolic specs (mirrors fault_extensions.json's
    list-of-specs pattern): the second spec's "reference": "long" must
    resolve against the already-extended shape from the first spec."""
    frac = _rectangle()
    grow_x = {
        "type": "directional",
        "reference": "long",
        "toward": {"coordinate": "x", "sign": "high"},
        "amount": 50.0,
    }
    # After growing x from 10 to 60, x becomes the (even more) long axis;
    # y (3) stays short. A second spec targeting "short" should now extend y.
    after_first = fp.apply_extension(frac, grow_x)
    xs = after_first.pts[0]
    assert np.isclose(xs.max() - xs.min(), 60.0, atol=1e-8)

    grow_short = {
        "type": "directional",
        "reference": "short",
        "toward": {"coordinate": "y", "sign": "high"},
        "amount": 7.0,
    }
    after_second = fp.apply_extension(after_first, grow_short)
    ys = after_second.pts[1]
    assert np.isclose(ys.max() - ys.min(), 10.0, atol=1e-8)
    # x extent from the first spec must be untouched by the second.
    xs2 = after_second.pts[0]
    assert np.isclose(xs2.max() - xs2.min(), 60.0, atol=1e-8)


def test_apply_extensions_mixed_forms_and_passthrough(tmp_path):
    fractures = [_rectangle()]
    fractures[0].fault_name = "A"
    names = ["A"]

    config = {
        "A": {
            "type": "directional",
            "reference": "long",
            "toward": {"coordinate": "x", "sign": "high"},
            "amount": 5.0,
        },
        # "B" not present among `names` -- must not raise, simply unused.
        "B": {"type": "uniform", "scale": 1.5},
    }
    config_path = tmp_path / "extensions.json"
    config_path.write_text(json.dumps(config))

    result = fp.apply_extensions(fractures, names, config_path)
    assert len(result) == 1
    xs = result[0].pts[0]
    assert np.isclose(xs.max() - xs.min(), 15.0, atol=1e-8)


def test_apply_extension_unknown_type_raises():
    frac = _rectangle()
    with pytest.raises(ValueError, match="Unknown extension type"):
        fp.apply_extension(frac, {"type": "bogus"})


def test_apply_extension_to_intersect_not_implemented():
    frac = _rectangle()
    with pytest.raises(NotImplementedError):
        fp.apply_extension(frac, {"type": "to_intersect"})
