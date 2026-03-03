import re
import time
import pandas
import porepy as pp
import numpy as np
from geometry import CoolingGeometry
from material_parameters import granodiorite_values

from physical_model import (
    PhysicalModel,
    HeterogeneousPermeabilitySpecification,
    FluidExtensions,
)
from boundary_conditions import (
    CosoBoundaryConditionsDisplacement,
    OnlyInjectionWellNeumannBCsFromSchedule,
)

from porepy.applications.boundary_conditions.model_boundary_conditions import (
    BoundaryConditionsMechanicsNeumann,
    HydrostaticBoundaryPressureValues,
    LithostaticBoundaryStressValues,
    ThermalGradientBoundaryTemperatureValues,
)
from porepy.applications.initial_conditions.model_initial_conditions import (
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
)
from porepy.examples.geothermal_reservoir import WellBoundaryConditions
from porepy.viz.data_saving_model_mixin import (
    IterationExporting,
    FractureDeformationExporting,
    ResidualExporting,
)
from solution_strategy import SolutionStrategy
from initial_conditions import CopyInitialCondition
from exporting import CosoExporter, GeometryExporting, summarize_slip_onset_times
from wells import WellDataConceptual
from porepy.numerics.nonlinear import line_search
import diff_tpfa
import logging
import sys
import copy
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class BaseModel(
    FractureDeformationExporting,
    # IterationExporting,
    GeometryExporting,
    CosoExporter,
    CoolingGeometry,
    # pp.poromechanics.TpsaPoromechanicsMixin,
    # diff_tpfa.DarcysLawAdEverywhere,
    # FluidExtensions,
    HeterogeneousPermeabilitySpecification,
    pp.constitutive_laws.CubicLawPermeability,
    SolutionStrategy,  # Precedence over pp.models.solution_strategy.ContactIndicators
    pp.models.solution_strategy.ContactIndicators,
    WellDataConceptual,
    HydrostaticBoundaryPressureValues,
    ThermalGradientBoundaryTemperatureValues,
    LithostaticBoundaryStressValues,
    PhysicalModel,
):
    """Model for the Coso geothermal reservoir."""

    def create_plot_title(self) -> str:
        """Generate a formatted plot title from folder name and simulation parameters.

        Extracts simulation metadata from folder name (strike angle, fracture index, well status)
        and formats it as a human-readable title. Fracture indices are incremented by 1 for
        display purposes (0-indexed internally, 1-indexed for display).

        Returns:
            str | None: Formatted plot title with well status, strike angle, and fracture info.

        Example:
            Input folder_name: "case_II_with_wells_strike_35_tilted_fracture_0"
            Output: "with wells strike 35 tilted fracture, Tilted, Strike: 1"
        """
        fn = self.params["plot_title"]
        return fn

        fn = self.params["folder_name"]

        # The split returns ["example_7"] and [-1] gives "example_7" unchanged
        suffix = fn.split("temperatures")[-1]
        suffix = suffix.replace("_", " ").strip()

        match = re.search(r"fracture\s*(\d+)", suffix)
        if match:
            fracture_index = int(match.group(1))
            new_index = fracture_index + 1
            suffix = re.sub(r"fracture\s*\d+", f"Fracture: {new_index}", suffix)
        title = suffix.replace(" tilted", ", Tilted").replace("strike", "Strike:")
        # Replace "high" and "low" with T_{h} and T_{l} for temperature notation
        # Wrap in $...$ for proper LaTeX subscript rendering in matplotlib
        title = re.sub(r"high\s*(\d+)", r"$T_{h}$=\1 °C", title)
        title = re.sub(r"low\s*(\d+)", r"$T_{l}$=\1 °C", title)

        return title


class InitializationModel(
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
    BoundaryConditionsMechanicsNeumann,
    # pp.constitutive_laws.ConstantPorosity,
    BaseModel,
):
    """Initialization model for the Coso geothermal reservoir."""


class MainModel(
    OnlyInjectionWellNeumannBCsFromSchedule,
    CopyInitialCondition,
    WellBoundaryConditions,
    CosoBoundaryConditionsDisplacement,
    BaseModel,
):
    """Main model for the Coso geothermal reservoir."""


class ConstraintLineSearchNonlinearSolver(
    line_search.ConstraintLineSearch,
    line_search.SplineInterpolationLineSearch,
    line_search.LineSearchNewtonSolver,
):
    pass


if __name__ == "__main__":
    tp = True

    fracture_strike_angles = np.array(
        [
            -42,
            # -43,
            # -41,
            # 39,
            # 40,
            # 35,
        ]
    )
    strike = fracture_strike_angles[0]
    # granodiorite_values["permeability"] = 1e-16
    granodiorite_values["residual_aperture"] = 8.0e-4  # just changed from 3e-4
    granodiorite_values["dilation_angle"] = 0  # np.deg2rad(1)
    granodiorite_values["porosity"] = 0.02

    init_granodiorite_values = copy.deepcopy(granodiorite_values)
    init_granodiorite_values.update(
        {
            "permeability": 5e-16,
            "friction_coefficient": 16,
        }
    )
    injection_temperature_values = [
        (30, 5),
        # (30, 30),
        # (5, 5),
    ]
    angle_index = 1
    boundary_velocities = [0.0, 2.0e-6, 5.0e-6]

    controls = ["Pressure", "Rate"]
    injection_fluxes = {}
    all_slip_onset_times = {}

    for velocity in boundary_velocities:
        v_key = str(int(velocity))
        injection_fluxes[v_key] = {}
        for high, low in injection_temperature_values:
            for control in controls:
                tic = time.time()
                t_key = f"t_high_{high}_low_{low}"

                suffix = f"{control}_controlled_{t_key}_velocity_{velocity:.1e}"

                logger.info(f"Starting the simulation with suffix {suffix}")
                dt = 5e2
                period = pp.YEAR / 2
                transition_time = pp.DAY
                # period = 1e4
                # transition_time = 5e3
                # Alternate between warm and cold injection every period, with a short
                # transition time in between to allow for convergence without too large
                # jumps in the solution. The schedule defines the time points at which
                # the boundary conditions change.
                schedule = np.array(
                    [
                        0,
                        1 * period - transition_time,  # End of first warm injection
                        1 * period,  # Start of first cold period
                        2 * period - transition_time,  # End of first cold period
                        2 * period,  # Start of 2nd warm injection
                        3 * period - transition_time,  # End of 2nd warm injection
                        3 * period,  # Start of second cold period
                        4 * period - transition_time,  # End of second cold period
                        4 * period,
                        5 * period - transition_time,  # End of third warm injection
                        5 * period,
                        6 * period - transition_time,  # End of third cold period
                        6 * period,  # Start of fourth warm injection
                        7 * period - transition_time,
                        7 * period,  # Start of fourth cold period
                        8 * period - transition_time,  # End of fourth cold period
                    ],
                )
                if control == "Rate":
                    # We will use Neumann in the cold periods of the schedule.
                    neumann_intervals = [
                        (1 * period, 2 * period),
                        (3 * period, 4 * period),
                        (5 * period, 6 * period),
                        (7 * period, 8 * period),
                    ]

                else:
                    neumann_intervals = []
                # schedule += injection_start_time
                # Add the initial time step to the schedule
                # schedule = np.insert(schedule, 0, 0)  # Initial time step at 0
                time_manager = pp.TimeManager(
                    schedule=schedule,
                    dt_init=dt,
                    dt_min_max=(1e-3, max(dt, period / 5)),
                    iter_max=20,
                    iter_optimal_range=(5, 12),
                    iter_relax_factors=(0.5, 2.0),
                    recomp_factor=0.2,
                    recomp_max=10,
                )
                dt_init = 2e9  # * pp.YEAR
                time_manager_init = pp.TimeManager(
                    [0, 4 * dt_init],
                    dt_init=dt_init,
                    dt_min_max=(1, 2 * dt_init),
                    constant_dt=False,
                )
                fracture_size = 6e2
                # fracture_size = 1e3
                cell_size = 14e2

                folder_name = Path("case_III") / suffix
                folder_name_init = str(folder_name) + "_initialization"
                file_name = "example_8"
                data_folder_name = Path(str(folder_name) + "_saved_data")
                strike_angles = np.deg2rad([45, 45, 45])
                strike_angles[angle_index] = np.deg2rad(strike)
                title = (
                    f"{control} controlled, Velocity={velocity:.1e} m/y, "
                    + f"$T_{{h}}$={high}°C, $T_{{l}}$={low}°C"
                )

                model_params_init = {
                    "plot_title": title,
                    "domain_sizes": np.array([6, 6, 6]) * 1e3,
                    "material_constants": {
                        "solid": pp.SolidConstants(**init_granodiorite_values),
                        "fluid": pp.FluidComponent(**pp.fluid_values.water),
                    },
                    "units": pp.Units(m=1.0, kg=1.0e0, K=1.0),
                    "time_manager": time_manager_init,
                    "grid_type": "simplex",
                    "meshing_arguments": {
                        "cell_size": cell_size,
                        "cell_size_fracture": 0.3 * fracture_size,
                    },
                    "file_name": file_name,
                    "data_folder_name": str(data_folder_name),
                    "adaptive_indicator_scaling": True,
                    "initialization": True,
                    "use_wells": False,
                    "reference_variable_values": pp.ReferenceVariableValues(
                        temperature=pp.Celsius_to_Kelvin(20),
                        #     pressure=pp.BAR,
                    ),
                    "thermal_gradient": 70e-3,  # Test 7e-2
                    "fracture_file": "coords.txt",
                    "folder_name": folder_name_init,
                    "lithostatic_stress_multipliers": np.array([0.62, 1.55, 1.0]),
                    "fracture_params": {
                        "fracture_major_axes": np.array(
                            (fracture_size, 1.3 * fracture_size, fracture_size)
                        ),
                        "strike_angles": strike_angles,
                        "num_points": 8,
                        # "dx": 6e2,
                    },
                    "darcy_flux_discretization": "tpfa" if tp else "mpfa",
                    "fourier_flux_discretization": "tpfa" if tp else "mpfa",
                    "heterogeneous_permeability": False,
                    "use_ic_interpolation": True,
                    "boundary_displacement_velocity_scaling": velocity / pp.YEAR,
                }
                model_params = copy.deepcopy(model_params_init)
                # Reduce target pressure in favour of displacement BC as driving force?
                injection_pressures = np.full(schedule.shape, 0.3 * pp.MEGA * pp.PASCAL)
                # injection_pressures = np.full(schedule.shape, 10* pp.ATMOSPHERIC_PRESSURE)
                # injection_pressures[0] = .5 * pp.MEGA * pp.PASCAL  # Initial pressure
                # injection_pressures[:1] = 1 * pp.ATMOSPHERIC_PRESSURE

                production_pressures = np.full(schedule.shape, pp.ATMOSPHERIC_PRESSURE)

                # production_pressures = 0.5 * pp.ATMOSPHERIC_PRESSURE

                # Alternate between warm and cold injection every period, with a short
                # transition time in between. This corresponds to injection temperatures
                # of alternating two consequtive values in the schedule, with the first
                # value being the higher one.
                if control == "Rate":
                    injection_temperatures = np.full(
                        schedule.shape, pp.Celsius_to_Kelvin(high)
                    )
                else:
                    injection_temperatures = pp.Celsius_to_Kelvin(
                        np.tile([high, high, low, low], schedule.size // 4)
                    )

                production_temperatures = np.full(
                    schedule.shape, pp.Celsius_to_Kelvin(150)
                )
                # Can be refined to have different schedules for each well.
                for name in MainModel.injection_well_names.fget(None):
                    model_params[f"{name}_pressures"] = injection_pressures
                    model_params[f"{name}_temperatures"] = injection_temperatures

                    val = (
                        1e10
                        if control == "Pressure"
                        else injection_fluxes[v_key][t_key]
                    )
                    model_params[f"{name}_darcy_fluxes"] = val
                for name in MainModel.production_well_names.fget(None):
                    model_params[f"{name}_pressures"] = production_pressures
                    model_params[f"{name}_temperatures"] = production_temperatures
                    # Should not be used, but needs to be set. Set to high value to
                    # reveal if it is used by mistake.
                    model_params[f"{name}_darcy_fluxes"] = 1e10

                # Create the model
                solver_params = {
                    "nl_convergence_tol_res": 1e-1,
                    "nl_convergence_tol": 1e-3,
                    "nl_divergence_tol": 1e20,
                    "max_iterations": 20,
                    "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
                    "local_line_search": 1,
                    "global_line_search": 1,
                    "residual_line_search_interval_size": 1e-3,
                    "constraint_violation_tolerance": 1e-3,
                }

                init_model = InitializationModel(model_params_init)

                pp.run_time_dependent_model(init_model, solver_params)
                # Analyze the initialization results to set friction coefficient
                sds = init_model.mdg.subdomains(dim=2)
                traction = init_model.evaluate_and_scale(
                    sds, "contact_traction", "-"
                ).reshape((3, -1), order="F")
                friction_coeff = (
                    np.max(
                        np.linalg.norm(traction[:-1], axis=0) / np.abs(traction[-1, :])
                    )
                    + 0.005
                )
                # solver_params["max_iterations"] = 15
                granodiorite_values["friction_coefficient"] = friction_coeff
                # Save granodiorite values for main model
                df = pandas.DataFrame.from_dict(granodiorite_values, orient="index")
                output_path = data_folder_name / "granodiorite_values.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, header=False)
                model_params.update(
                    {
                        "time_manager": time_manager,
                        "file_name": file_name,
                        "folder_name": str(folder_name),
                        "use_wells": True,
                        "initialization": False,
                        "reference_from_initial": True,
                        "material_constants": {
                            "solid": pp.SolidConstants(**granodiorite_values),
                            "fluid": pp.FluidComponent(**pp.fluid_values.water),
                        },
                        "neumann_intervals": neumann_intervals,
                    }
                )

                model = MainModel(model_params)  # Load from initialization from file
                model.initialization_model = init_model
                solver_params.update(
                    {
                        "local_line_search": 1,
                        "global_line_search": 1,
                        "nl_convergence_tol_res": 1e0,
                        "residual_line_search_interval_size": 2e-3,
                    }
                )
                pp.run_time_dependent_model(model, solver_params)
                df = model.plot_well_monitoring(temperature_schedule=schedule)
                toc = time.time()
                logger.info(
                    f"Simulation with suffix {suffix} completed in {toc - tic:.2f} seconds."
                )
                # Retrieve injection flux at end of simulation if pressure controlled,
                # for use in next simulation with rate control.
                if control == "Pressure":
                    df = model.well_monitoring_data
                    well_data = df[df["well_name"] == model.injection_well_names[0]]
                    injection_fluxes[v_key][t_key] = well_data["darcy_flux"].iloc[-1][0]
                    logger.info(
                        f"Final injection flux for pressure controlled simulation: {injection_fluxes[v_key][t_key]:.2e} m/s"
                    )
                # Retrieve slip onset times.
                slip_onset_times = model.slip_onset_times()
                all_slip_onset_times[suffix] = slip_onset_times

    # Save injection fluxes for all simulations in a CSV file.
    injection_fluxes_df = pandas.DataFrame(injection_fluxes)
    injection_fluxes_output_path = data_folder_name / "injection_fluxes.csv"
    injection_fluxes_df.to_csv(injection_fluxes_output_path, index=False)
    summarize_slip_onset_times(
        all_slip_onset_times,
        model.fracture_names(),
        boundary_velocities,
        output_file=data_folder_name / "slip_onset_times.csv",
    )
