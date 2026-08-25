"""Standalone, geometry-only reproduction script for the well-fracture meshing crash:

    File ".../porepy/fracs/well_network.py", line 675, in _add_well_fracture_interfaces
        g_high = mdg.subdomains(dim=mdg.dim_max() - 1)[frac_inds[0]]
    IndexError: list index out of range

which is preceded by (and, per investigation this session, plausibly caused by):

    UserWarning: Found N fractures outside the domain boundary

Builds the exact geometry MainModel currently uses in run_example_3.py (same fault
exclusions/extensions, mesh refinement, well setup), but skips materials/physics/
solving entirely -- only set_materials() + set_geometry() run, so this is cheap to
iterate on compared to a full run_example_3.py loop.

Needs the real well data (params["read_well_operation_data"] / the wellbores.xlsx
file referenced in geometry.py's set_well_network) to reach the actual well-meshing
step, which isn't available in every environment -- hence "run yourself".

Usage:
    python diagnose_well_fracture_geometry.py [--no-constraints] [--extensions PATH]

    --no-constraints   Build the geometry WITHOUT ConstraintsCapcrockAndReservoirDepth
                        mixed in (i.e. the last known-working configuration), to check
                        whether that mixin is what's causing fractures to be dropped.
    --extensions PATH  Use an alternate fault_extensions.json (e.g. one without the
                        fault 0001 shrink entry) to check whether that shrink affects
                        the outcome. Defaults to fault_extensions.json.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np
import porepy as pp

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)  # this script's relative paths (point_cloud_clusters/, etc.)
# and geometry.py's sys.path[0]-based data lookups assume the repo root as cwd.

import run_example_3 as ex3
from geometry import FaultPlaneGeometry
from material_parameters import granodiorite_values
from physical_model import HagenPoiseuilleWellPermeability, PhysicalModel
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    HydrostaticBoundaryPressureValues,
    LithostaticBoundaryStressValues,
    ThermalGradientBoundaryTemperatureValues,
)
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting
from solution_strategy import SolutionStrategy
from wells import _WellDataBase
from exporting import CosoExporter, GeometryExporting


class NoConstraintsBaseModel(
    FractureDeformationExporting,
    GeometryExporting,
    CosoExporter,
    # ConstraintsCapcrockAndReservoirDepth intentionally omitted -- see --no-constraints.
    FaultPlaneGeometry,
    HagenPoiseuilleWellPermeability,
    pp.constitutive_laws.CubicLawPermeability,
    SolutionStrategy,
    pp.models.solution_strategy.ContactIndicators,
    _WellDataBase,
    HydrostaticBoundaryPressureValues,
    ThermalGradientBoundaryTemperatureValues,
    LithostaticBoundaryStressValues,
    PhysicalModel,
):
    """Mirrors run_example_3.BaseModel's mixin stack, minus
    ConstraintsCapcrockAndReservoirDepth, for A/B comparison."""


class NoConstraintsMainModel(
    ex3.CosoBackgroundValues,
    ex3.SmoothWellTransitions,
    ex3.NeumannWellBCsFromSchedule,
    ex3.CopyInitialCondition,
    ex3.WellBoundaryConditions,
    ex3.CosoBoundaryConditionsDisplacement,
    ex3.HeterogeneousPermeabilitySpecification,
    ex3.HeterogeneousFrictionCoefficient,
    NoConstraintsBaseModel,
):
    """Same mixin stack as run_example_3.MainModel, minus the constraints geometry."""


def build_model_params(fault_extension_config: str, use_wells: bool = True) -> dict:
    """Geometry-relevant subset of run_example_3.py's current MainModel params.

    Keep this in sync with run_example_3.py's __main__ loop (domain_size,
    fracture_size, refinement, cell_size*, exclude_faults) if those change --
    they're copied here rather than imported since they're local variables inside
    that script's loop, not module-level constants. USE_CONSTRAINTS and
    USE_ITERATIVE_SOLVER *are* module-level constants and are read directly from
    ex3, so those two stay in sync automatically.
    """
    domain_size = 8.0e3
    fracture_size = 6e2
    refinement = 10.0 if ex3.USE_ITERATIVE_SOLVER else 3.0
    if not ex3.USE_CONSTRAINTS:
        refinement /= 3.8
    cell_size = 15e2 * refinement
    cell_size_fracture = 0.4 * fracture_size * refinement

    water_values = {**pp.fluid_values.water, "thermal_expansion": 7.8745e-04}
    solid_values = copy.deepcopy(granodiorite_values)

    return {
        "fault_plane_dirs": ["point_cloud_clusters"],
        "fault_extension_config": fault_extension_config,
        "exclude_faults": [
            # # "0001",  # kept active here, matching run_example_3.py's current state.
            # "0002",
            # # "0003",
            # "0004",
            # "0006",
            # "0007",
            # "0008",
            # "0009",
            # "0010",
            # "0011",
            # "0012",
        ],
        "well_sheet_names": {
            "1 Injection well": "68-20RD",
            "2 Production well": "16A-20",
            "3 Production well": "16B-20",
        },
        "top_boundary_z": 0.0,
        "read_well_operation_data": False,
        "plot_title": "diagnose_well_fracture_geometry",
        "domain_sizes": np.array([domain_size, domain_size, 3e3]),
        "material_constants": {
            "solid": pp.SolidConstants(**solid_values),
            "fluid": pp.FluidComponent(**water_values),
        },
        "units": pp.Units(m=1.0e0, kg=1.0e9, K=1.0),
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size": cell_size,
            "cell_size_fracture": cell_size_fracture,
        },
        "file_name": "diagnose",
        "data_folder_name": "diagnose_saved_data",
        "use_wells": use_wells,
        "reference_variable_values": pp.ReferenceVariableValues(
            temperature=pp.Celsius_to_Kelvin(20),
            pressure=pp.ATMOSPHERIC_PRESSURE,
        ),
        "fracture_file": "coords.txt",
        "folder_name": "diagnose_folder",
        "initialization": False,
        "lithostatic_stress_multipliers": np.array([0.62, 1.55, 1.0]),
        "fracture_params": {
            "fracture_major_axes": np.array(
                (
                    fracture_size / domain_size,
                    fracture_size / domain_size,
                    fracture_size / domain_size,
                )
            ),
        },
        "heterogeneous_permeability": True,
        "darcy_flux_discretization": "tpfa",
        "fourier_flux_discretization": "tpfa",
        "use_ic_interpolation": True,
        "boundary_displacement_velocity": 0.0,
        "surface_temperature": pp.Celsius_to_Kelvin(20.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-constraints",
        action="store_true",
        help="Build without ConstraintsCapcrockAndReservoirDepth mixed in.",
    )
    parser.add_argument(
        "--extensions",
        default="fault_extensions.json",
        help="Path to the fault-extensions config to use (default: fault_extensions.json).",
    )
    args = parser.parse_args()

    model_class = NoConstraintsMainModel if args.no_constraints else ex3.MainModel
    print(f"Model class: {model_class.__name__}")
    print(f"Fault extensions config: {args.extensions}")

    model_params = build_model_params(args.extensions)
    model = model_class(model_params)
    model.set_materials()
    model.set_domain()
    model.set_fractures()
    fracture_names = [
        getattr(f, "fault_name", f"<constraint {i}>")
        for i, f in enumerate(model._fractures)
    ]
    print(f"\nFractures before meshing ({len(fracture_names)}): {fracture_names}")
    print(f"Domain bounding box: {model._domain.bounding_box}")

    # This is where "Found N fractures outside the domain boundary" would fire, and
    # (with use_wells=True and real well data available) where the well-fracture
    # interface IndexError would eventually surface.
    model.set_geometry()
    model.initialize_data_saving()

    subdomains_2d = model.mdg.subdomains(dim=2)
    print(f"\n2d subdomains after meshing ({len(subdomains_2d)}):")
    for sd in subdomains_2d:
        print(f"  frac_num={sd.frac_num}  num_cells={sd.num_cells}")

    # Meshing constraints (e.g. from ConstraintsCapcrockAndReservoirDepth) legitimately
    # don't get their own 2d subdomain, so account for those before flagging a mismatch.
    expected_subdomains = getattr(model, "_num_fractures", len(fracture_names))
    if len(subdomains_2d) != expected_subdomains:
        print(
            f"\n*** MISMATCH: {len(fracture_names)} fractures went in "
            f"({expected_subdomains} expected to become real subdomains), "
            f"{len(subdomains_2d)} 2d subdomains came out. This is almost certainly "
            "the root cause of the well_network.py IndexError -- frac_inds computed "
            "from the original fracture list no longer aligns with mdg.subdomains(dim=2)."
        )
    else:
        print("\nFracture/subdomain count is as expected -- no mismatch detected here.")

    print("\nSUCCESS: set_geometry() completed without raising.")
    print(
        "Wrote to the following files:"
        f"\n  {model_params['folder_name']}/{model_params['file_name']}.vtu"
        f"\n  {model_params['folder_name']}/{model_params['file_name']}.pvd"
    )


if __name__ == "__main__":
    main()
