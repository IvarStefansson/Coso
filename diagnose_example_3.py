"""Fast diagnostic harness for iterating on example 3's solver / time-stepper setup.

Reuses the real ``InitializationModel`` / ``MainModel`` class stack from
``run_example_3.py`` (same physics, boundary conditions, wells, exporters, and
``nl_params.py`` solver setup) unmodified, with ``SingleFractureDiagnosticGeometry``
grafted on in place of ``FaultPlaneGeometry`` via ``add_mixin`` so meshing and
solves are cheap. Does not modify run_example_3.py.

The init -> main flow (friction-coefficient extraction from the initialization
run, the manual grid/energy-balance/AD warm-up before the init ModelRunner call)
mirrors run_example_3.py's __main__ block line for line, since that logic isn't
factored into a reusable function there. Keep this loosely in sync if that block
changes materially.

No sweep here by design -- a single run per call. Wrap run_diagnostic() in your
own loop if a sweep is needed later, following run_example_3.py's pattern.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import porepy as pp
from porepy.applications.test_utils.models import add_mixin

import run_example_3 as ex3
from geometry import SingleFractureDiagnosticGeometry
from nl_params import set_nonlinear_solver
from time_stepping import reset_time_io, time_steppers_coso
from porepy.viz.data_saving_model_mixin import (
    FractureDeformationExporting,
    IterationExporting,
    ResidualExporting,
)

logger = logging.getLogger("diagnose_example_3")
logging.getLogger("porepy.numerics.solvers.convergence_check").setLevel(logging.DEBUG)


class DiagAdditions(
    IterationExporting, ResidualExporting, SingleFractureDiagnosticGeometry
):
    pass


DiagInitializationModel = add_mixin(DiagAdditions, ex3.InitializationModel)
DiagMainModel = add_mixin(DiagAdditions, ex3.MainModel)


def run_diagnostic(
    *,
    # --- schedule / time stepper ---
    dt: float = 1e2,
    production_period_years: float = 0.02,
    final_time_years: float = 0.04,
    num_bins_per_interval: int = 2,
    time_stepper_kwargs: Optional[dict] = None,
    time_stepper: Optional[pp.time_stepper.TimeStepper] = None,
    time_stepper_init: Optional[pp.time_stepper.TimeStepper] = None,
    # --- mesh ---
    cell_size: float = 6e2,
    cell_size_fracture: float = 3.0e2,
    # --- physics / wells ---
    with_production: bool = True,
    velocity: float = 0.0,
    # --- solver ---
    iterative_linear_solver: bool = False,
    # --- init phase ---
    run_init: bool = True,
    friction_coefficient: float = 0.6,
    # --- saving (override freely) ---
    folder_name: str = "diagnostics/single_fracture",
    folder_name_init: Optional[str] = None,
    file_name: str = "diagnostic_example_3",
    # --- escape hatches for anything else ---
    model_params_overrides: Optional[dict] = None,
    init_params_overrides: Optional[dict] = None,
):
    """Run one fast init(+main) simulation on the single-fracture diagnostic
    geometry, using example 3's real model classes and solver setup.

    Returns:
        (init_model, model) -- init_model is None if run_init=False.
    """
    logging.basicConfig(level=logging.INFO)
    folder_name_init = folder_name_init or f"{folder_name}_initialization"

    schedule, neumann_intervals, is_transition = ex3.create_schedule(
        production_period_years * pp.YEAR,
        final_time=final_time_years * pp.YEAR,
        num_bins_per_interval=num_bins_per_interval,
        transition_duration=ex3.transition_duration,
        initial_shutin_duration=0.0,
        with_production=with_production,
    )

    time_stepper_kwargs = time_stepper_kwargs or {}
    time_stepper_kwargs.setdefault("is_transition", is_transition)
    default_ts, default_ts_init = time_steppers_coso(
        schedule, dt, **time_stepper_kwargs
    )
    time_stepper = time_stepper if time_stepper is not None else default_ts
    time_stepper_init = (
        time_stepper_init if time_stepper_init is not None else default_ts_init
    )

    water_values = {
        **pp.fluid_values.water,
        "thermal_expansion": 7.8745e-04,
    }
    solid_values_init = {
        **ex3.granodiorite_values,
        "permeability": ex3.granodiorite_values["permeability"] * 1e2,
    }
    solid_values_local = {
        **ex3.granodiorite_values,
        "friction_coefficient": friction_coefficient,
    }

    model_params_init = {
        "diagnostic_domain_margin": 2.0e2,
        "well_sheet_names": {
            "1 Injection well": "68-20RD",
            "2 Production well": "16A-20",
            "3 Production well": "16B-20",
        },
        "top_boundary_z": 0.0,
        "read_well_operation_data": False,
        "material_constants": {
            "solid": pp.SolidConstants(**solid_values_init),
            "fluid": pp.FluidComponent(**water_values),
        },
        "units": pp.Units(m=1.0e0, kg=1.0e9, K=1.0),
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size": cell_size,
            "cell_size_fracture": cell_size_fracture,
        },
        "file_name": file_name,
        "data_folder_name": f"{folder_name}_saved_data",
        "adaptive_indicator_scaling": True,
        "use_wells": True,
        "reference_variable_values": pp.ReferenceVariableValues(
            temperature=pp.Celsius_to_Kelvin(20),
            pressure=pp.ATMOSPHERIC_PRESSURE,
        ),
        "folder_name": folder_name_init,
        "initialization": True,
        "lithostatic_stress_multipliers": np.array([0.62, 1.55, 1.0]),
        "heterogeneous_permeability": False,
        "darcy_flux_discretization": "tpfa",
        "fourier_flux_discretization": "tpfa",
        "use_ic_interpolation": True,
        "boundary_displacement_velocity": velocity / pp.YEAR,
        "surface_temperature": pp.Celsius_to_Kelvin(20.0),
    }
    if init_params_overrides:
        model_params_init.update(init_params_overrides)

    model_params = {**model_params_init}
    if with_production:
        injection_p = np.full(schedule.shape, 0.3 * pp.MEGA * pp.PASCAL)
        production_p = np.full(schedule.shape, pp.ATMOSPHERIC_PRESSURE)
        injection_t = np.full(schedule.shape, pp.Celsius_to_Kelvin(40))
        production_t = np.full(schedule.shape, 475)
        for name in DiagMainModel.injection_well_names.fget(None):
            model_params[f"{name}_pressures"] = injection_p
            model_params[f"{name}_temperatures"] = injection_t
        for name in DiagMainModel.production_well_names.fget(None):
            model_params[f"{name}_pressures"] = production_p
            model_params[f"{name}_temperatures"] = production_t

    init_model = None
    t0_init = t1_init = time.perf_counter()
    if run_init:
        init_model = DiagInitializationModel(model_params_init)
        nonlinear_solver = set_nonlinear_solver(
            iterative_linear_solver=iterative_linear_solver
        )
        init_model.prepare_simulation()
        # Manual grid/energy-balance/AD warm-up, mirrors run_example_3.py. That
        # warm-up targets dim=1 grids from fracture-fracture intersections in
        # the real multi-fault network (not wells -- use_wells is False here,
        # same as in run_example_3.py's init model). The single-fracture
        # diagnostic geometry has no such intersections, so skip when absent.
        dim1_grids = init_model.mdg.subdomains(dim=1)
        if dim1_grids:
            grids = [dim1_grids[0]]
            init_model.energy_balance_equation(grids)
            variables, _ = init_model.equation_system.variable_indexer.filter_by_tags(
                model=init_model,
                tags=[
                    pp.solvers.VariableTag(
                        name="temperature", defined_on=pp.solvers.OnLowerDimensions()
                    )
                ],
            )
            var_indexer = init_model.equation_system.variable_indexer.construct_restricted_indexer(
                [variables[0]]
            )
            init_model.volume_integral(integrand=pp.ad.Scalar(1), grids=grids, dim=1)
            init_model.solid_internal_energy(grids)
            op = init_model.fluid_internal_energy(grids)
            init_model.equation_system.evaluate(
                op, derivative=True, variable_indexer=var_indexer
            )

        t0_init = time.perf_counter()
        reset_time_io()
        pp.ModelRunner(
            init_model,
            nonlinear_solver=nonlinear_solver,
            params={"prepare_simulation": False},
            time_stepper=time_stepper_init,
        ).run()
        t1_init = time.perf_counter()

        sds = init_model.mdg.subdomains(dim=2)
        traction = init_model.evaluate_and_scale(sds, "contact_traction", "-").reshape(
            (3, -1), order="F"
        )
        friction_coefficient = (
            np.max(np.linalg.norm(traction[:-1], axis=0) / np.abs(traction[-1, :]))
            + 0.001
        )
        logger.info(
            "Determined friction coefficient from initialization: %.4f",
            friction_coefficient,
        )
        solid_values_local = {
            **ex3.granodiorite_values,
            "friction_coefficient": friction_coefficient,
        }

    model_params.update(
        {
            "file_name": file_name,
            "folder_name": folder_name,
            "initialization": False,
            "use_wells": with_production,
            "reference_from_initial": run_init,
            "material_constants": {
                "solid": pp.SolidConstants(**solid_values_local),
                "fluid": pp.FluidComponent(**water_values),
            },
            "neumann_intervals": neumann_intervals,
            "well_transition_duration": ex3.transition_duration,
        }
    )
    if model_params_overrides:
        model_params.update(model_params_overrides)

    model = DiagMainModel(model_params)
    if init_model is not None:
        model.initialization_model = init_model
    nonlinear_solver = set_nonlinear_solver(
        iterative_linear_solver=iterative_linear_solver
    )
    t0_main = time.perf_counter()
    reset_time_io()
    pp.ModelRunner(
        model, nonlinear_solver=nonlinear_solver, time_stepper=time_stepper
    ).run()
    t1_main = time.perf_counter()

    logger.info(
        "init: %d steps in %.1fs | main: %d steps in %.1fs",
        init_model.time_data.time_index_successful if init_model is not None else 0,
        t1_init - t0_init,
        model.time_data.time_index_successful,
        t1_main - t0_main,
    )
    return init_model, model


if __name__ == "__main__":
    run_diagnostic()
