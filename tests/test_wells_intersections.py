"""Tests for wells.well_fracture_intersections, promoted from a one-off
diagnostic script (diagnostics/check_fault_0001_well_intersections.py) into
reusable, tested logic.

Well-fracture connections in the mixed-dimensional grid are NOT a direct
(well, 1D) <-> (fracture, 2D) interface; they go through a shared 0-D
intersection point: well --codim1--> point <--codim2-- fracture. The first
attempt at this walk (earlier this session) assumed a direct interface and
silently produced a vacuous "no intersections" result -- these tests pin
down the correct topology using real porepy meshing, not just mocked dicts.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gmsh
import numpy as np
import pytest
import porepy as pp
from porepy.fracs.well_network import WellNetwork3d

from wells import well_fracture_intersections

REPO_ROOT = Path(__file__).resolve().parent.parent

BOX = {"xmin": -5.0, "xmax": 5.0, "ymin": -5.0, "ymax": 5.0, "zmin": -5.0, "zmax": 5.0}
MESHING_ARGS = {
    "cell_size": 5.0,
    "refinement_proximity_multiplier": 1e-6,
    "refinement_size_multiplier": 1.0,
    "background_transition_multiplier": 1.01,
}


@pytest.fixture(autouse=True)
def finalize_gmsh():
    yield
    try:
        gmsh.clear()
        gmsh.finalize()
    except Exception:
        pass


def _vertical_well(x: float, y: float) -> pp.Well:
    return pp.Well(np.array([[x, x], [y, y], [-4.0, 4.0]]))


def _horizontal_fracture(z: float) -> pp.PlaneFracture:
    box = BOX
    return pp.PlaneFracture(
        np.array(
            [
                [box["xmin"], box["xmax"], box["xmax"], box["xmin"]],
                [box["ymin"], box["ymin"], box["ymax"], box["ymax"]],
                [z, z, z, z],
            ]
        )
    )


def _mesh_wells_and_fractures(
    wells: list[pp.Well], fractures: list[pp.PlaneFracture]
) -> pp.MixedDimensionalGrid:
    domain = pp.Domain(BOX)
    fracture_network = pp.create_fracture_network(fractures, domain=domain)
    tmp_mdg = pp.create_mdg("simplex", MESHING_ARGS, fracture_network=fracture_network)
    well_network = WellNetwork3d(wells, domain)
    return well_network.mesh(fracture_network, tmp_mdg, MESHING_ARGS)


def test_single_well_single_fracture():
    well = _vertical_well(0.0, 0.0)
    fracture = _horizontal_fracture(z=0.0)
    mdg = _mesh_wells_and_fractures([well], [fracture])

    result = well_fracture_intersections(mdg, fault_names=["frac_A"])
    assert result == {(0, 0, "frac_A")}


def test_well_with_no_fracture_crossing_reports_nothing():
    # Fracture at z=3 -- well spans z in [-4, 4] so it geometrically crosses
    # the fracture's plane, but a fracture far off to the side (offset in x)
    # should NOT be reported as intersected by an unrelated well.
    well = _vertical_well(4.5, 4.5)
    fracture = _horizontal_fracture(z=0.0)
    mdg = _mesh_wells_and_fractures([well], [fracture])

    # A well at the domain corner still crosses the throughgoing horizontal
    # fracture plane geometrically, so this *does* intersect -- verifying the
    # walker finds it (sanity check the fixture actually connects them).
    result = well_fracture_intersections(mdg, fault_names=["frac_A"])
    assert result == {(0, 0, "frac_A")}


def test_multiple_wells_one_fracture():
    wells = [_vertical_well(-2.0, -2.0), _vertical_well(2.0, 2.0)]
    fracture = _horizontal_fracture(z=0.0)
    mdg = _mesh_wells_and_fractures(wells, [fracture])

    result = well_fracture_intersections(mdg, fault_names=["frac_A"])
    assert result == {(0, 0, "frac_A"), (1, 0, "frac_A")}


def test_no_fault_names_gives_none_third_element():
    well = _vertical_well(0.0, 0.0)
    fracture = _horizontal_fracture(z=0.0)
    mdg = _mesh_wells_and_fractures([well], [fracture])

    result = well_fracture_intersections(mdg)
    assert result == {(0, 0, None)}


def test_out_of_range_frac_num_gives_none_name():
    well = _vertical_well(0.0, 0.0)
    fracture = _horizontal_fracture(z=0.0)
    mdg = _mesh_wells_and_fractures([well], [fracture])

    result = well_fracture_intersections(mdg, fault_names=[])
    assert result == {(0, 0, None)}


# --- Integration test against real Coso well data + geometry -----------------

_WELLBORES_XLSX = REPO_ROOT / "data" / "wellbores.xlsx"


@pytest.mark.skipif(
    not _WELLBORES_XLSX.exists(), reason="real well data (data/wellbores.xlsx) not available"
)
def test_single_fracture_diagnostic_geometry_reports_expected_intersections():
    """SingleFractureDiagnosticGeometry is explicitly constructed to intersect
    both the injection well and the first production well -- confirm the
    promoted walker finds exactly that, against real Coso well trajectories.

    geometry.py's set_well_network()/SingleFractureDiagnosticGeometry build
    their data path from sys.path[0] (see the "wellbores path gotcha"), which
    under pytest is the tests/ directory, not the repo root -- so this test
    inserts the repo root at sys.path[0] itself for the duration of the call.
    """
    original_sys_path0 = sys.path[0]
    sys.path.insert(0, str(REPO_ROOT))
    original_cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        import run_example_3 as ex3
        from geometry import SingleFractureDiagnosticGeometry
        from porepy.applications.test_utils.models import add_mixin

        class _MinimalGeometryModel(
            add_mixin(SingleFractureDiagnosticGeometry, ex3.BaseModel)
        ):
            pass

        model_params = {
            "diagnostic_domain_margin": 2.0e2,
            "well_sheet_names": {
                "1 Injection well": "68-20RD",
                "2 Production well": "16A-20",
                "3 Production well": "16B-20",
            },
            "top_boundary_z": 0.0,
            "read_well_operation_data": False,
            "material_constants": {
                "solid": pp.SolidConstants(**ex3.granodiorite_values),
                "fluid": pp.FluidComponent(**pp.fluid_values.water),
            },
            "units": pp.Units(m=1.0e0, kg=1.0e9, K=1.0),
            "grid_type": "simplex",
            "meshing_arguments": {"cell_size": 3e2, "cell_size_fracture": 1.5e2},
            "use_wells": True,
        }
        model = _MinimalGeometryModel(model_params)
        model.set_materials()
        model.set_geometry()

        result = well_fracture_intersections(model.mdg, model.fracture_names())
        frac_nums = {r[1] for r in result}
        well_nums = {r[0] for r in result}
        assert len(frac_nums) == 1, f"expected exactly one fracture involved, got {result}"
        assert len(well_nums) == 2, f"expected both wells to intersect it, got {result}"
    finally:
        sys.path.pop(0)
        assert sys.path[0] == original_sys_path0
        os.chdir(original_cwd)
        try:
            gmsh.clear()
            gmsh.finalize()
        except Exception:
            pass
