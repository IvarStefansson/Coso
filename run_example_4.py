from typing import Any, Callable
import porepy as pp
import numpy as np
from geometry import ConceptualGeometry
from material_parameters import granodiorite_values

from physical_model import PhysicalModel
from boundary_conditions import (
    CosoBoundaryConditions,
    InitialCosoBoundaryConditions,
)
from solution_strategy import SolutionStrategy
from initial_conditions import InitialConditionFromDepth, CopyInitialCondition
from exporting import CosoExporter, IterationExporting
from wells import WellDataConceptual
from porepy.numerics.nonlinear import line_search
import diff_tpfa
import logging
import sys
import copy

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class BaseModel(
    IterationExporting,
    CosoExporter,
    ConceptualGeometry,
    diff_tpfa.DarcysLawAd,
    pp.constitutive_laws.CubicLawPermeability,
    SolutionStrategy,  # Precedence over pp.models.solution_strategy.ContactIndicators
    pp.models.solution_strategy.ContactIndicators,
    WellDataConceptual,
    PhysicalModel,
):
    """Model for the Coso geothermal reservoir."""


inds = slice(0, 1) if isinstance(PhysicalModel, pp.Thermoporomechanics) else slice(0)
# for method, dof in zip(diff_tpfa.methods[inds], diff_tpfa.dofs[inds]):
#     diff_tpfa.override_methods(BaseModel, method, dof, diff_tpfa.sort_criterion)


class InitializationModel(
    InitialConditionFromDepth,  # Commenting yields zero ICs, consistent with
    InitialCosoBoundaryConditions,  # zero BCs for initialization.
    CosoBoundaryConditions,
    BaseModel,
):
    """Initialization model for the Coso geothermal reservoir."""


class MainModel(CopyInitialCondition, CosoBoundaryConditions, BaseModel):
    """Main model for the Coso geothermal reservoir."""


class ConstraintLineSearchNonlinearSolver(
    # line_search.ConstrainUpdateLineSearch,
    # line_search.ConstrainVariableLineSearch,
    line_search.ConstraintLineSearch,
    line_search.SplineInterpolationLineSearch,
    line_search.LineSearchNewtonSolver,
):
    pass


if __name__ == "__main__":
    fast = 1 == 11  # Set to 1 for fast run, 0 for full run
    # Define the time parameters
    logger.info("Starting the simulation")
    dt = 1 * pp.DAY
    # injection_start_time = 10e3
    # Include dt to make sure itis included in the time steps which are exported.
    schedule = np.array([0, 1, 10, 20, 50, 80, 111, 112, 113, 114, 115, 116]) * pp.DAY
    if fast:
        injection_start_time = 10  # 0.2 * pp.YEAR
        schedule = np.arange(2) * 10.0  # pp.HOUR
    # schedule += injection_start_time
    # Add the initial time step to the schedule
    # schedule = np.insert(schedule, 0, 0)  # Initial time step at 0
    time_manager = pp.TimeManager(
        schedule=schedule,
        dt_init=dt,
        dt_min_max=(1, pp.YEAR),
        iter_max=20,
        iter_optimal_range=(5, 12),
        iter_relax_factors=(0.5, 2.0),
        recomp_factor=0.2,
        recomp_max=5,
    )
    dt_init = 2e7  # * pp.YEAR
    time_manager_init = pp.TimeManager(
        [0, 2 * dt_init],
        dt_init=dt_init,
        dt_min_max=(1, 2 * dt_init),
        constant_dt=False,
    )
    cell_size = 12e2
    if fast:
        cell_size = 2e3
    init_granodiorite_values = copy.deepcopy(granodiorite_values)
    init_granodiorite_values.update(
        {
            "maximum_elastic_fracture_opening": 0.0,
            "dilation_angle": 0.0,
            "permeability": 1e-10,
            "thermal_conductivity": 1e5,
            "residual_aperture": 1e-0,
        }
    )
    folder_name = "conceptual"
    fracture_shift_distance = 0  # Must be int for filename formatting
    file_name = "example_4"
    # if fast:
    #     file_name += "_fast"
    model_params_init = {
        "material_constants": {
            "solid": pp.SolidConstants(**init_granodiorite_values),
            "fluid": pp.FluidComponent(**pp.fluid_values.water),
            "numerical": pp.NumericalConstants(characteristic_displacement=1e-2),  # type: ignore[arg-type]
        },
        "units": pp.Units(m=1.0, kg=1.0e5, K=1.0),
        "time_manager": time_manager_init,
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size": cell_size,
            "cell_size_fracture": 0.6 * cell_size,
        },
        "file_name": f"{file_name}_initialize",
        "data_folder_name": f"{file_name}_saved_data",
        "adaptive_indicator_scaling": 1,  # Scale the indicator adaptively to increase robustness
        "use_wells": False,
        "reference_variable_values": pp.ReferenceVariableValues(
            temperature=300.0,
            pressure=pp.BAR,
        ),
        "fracture_file": "coords.txt",
        "folder_name": folder_name,
        "initialization": True,
    }
    if fast:
        model_params_init["data_folder_name"] = "saved_data_fast_runs"
    model_params = copy.deepcopy(model_params_init)
    model_params.update(
        {
            "time_manager": time_manager,
            "file_name": file_name,
            "use_wells": True,
            "reference_from_initial": True,
            "material_constants": {
                "solid": pp.SolidConstants(**granodiorite_values),
                "fluid": pp.FluidComponent(**pp.fluid_values.water),
                "numerical": pp.NumericalConstants(characteristic_displacement=1e-2),  # type: ignore[arg-type]
            },
        }
    )

    # Create the model
    init_model = InitializationModel(model_params_init)
    solver_params = {
        "nl_convergence_tol_res": 1e1,
        "nl_convergence_tol": 5e-5,  # Seems to be the best we can do with current condition number
        "max_iterations": 40,
        "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
        "local_line_search": 1,
        "global_line_search": 1,
        "residual_line_search_interval_size": 1e-3,
        # "linear_solver": "scipy_sparse",
    }
    if 10 == 11:
        init_model.prepare_simulation()
        m = init_model
        ma = m.mdg.subdomains(dim=3)
        m.porosity(ma).value(m.equation_system)
        m.fluid.density(ma).value(m.equation_system)
    else:
        pp.run_time_dependent_model(init_model, solver_params)
        model = MainModel(model_params)
        model.initialization_model = init_model
        solver_params.update(
            {
                "local_line_search": 1,
                "nl_convergence_tol_res": 5e0,
            }
        )
        pp.run_time_dependent_model(model, solver_params)
        model.plot_well_monitoring()
