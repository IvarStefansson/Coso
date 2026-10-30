"""
Real fractures enabled in run_example_3.py (0000, 0001, 0003, 0005) + one artificial
fracture through the well tips. Wells are trimmed below the fractures they touch.

Not yet run.

PorePy: bce2db427c0a20b9bacc799a90f12b83f5ec3d19
pp_solvers: 1ba92eae72e0fa2c337a4d016a62a32578ccec55

Command (from coso dir):
python runscript_3.py > runscript_3.stdout 2> runscript_3.stderr

"""

from collections import defaultdict
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import porepy as pp

from boundary_conditions import (
    CosoBoundaryConditionsDisplacement,
    NeumannWellBCsFromSchedule,
    SmoothWellTransitions,
)
from exporting import CosoExporter, GeometryExporting
from geometry import FaultPlaneGeometry
from material_parameters import granodiorite_values
from nl_params import set_nonlinear_solver
from physical_model import (
    CosoBackgroundValues,
    HagenPoiseuilleWellPermeability,
    HeterogeneousPermeabilitySpecification,
    PhysicalModel,
)
from porepy.applications.boundary_conditions.model_boundary_conditions import (
    HydrostaticBoundaryPressureValues,
    LithostaticBoundaryStressValues,
    ThermalGradientBoundaryTemperatureValues,
)
from porepy.applications.initial_conditions.model_initial_conditions import (
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
)
from porepy.examples.geothermal_reservoir import (
    WellBCType,
    WellBoundaryConditions,
    WellInterval,
)
from porepy.models.contact_mechanics import (
    ScaledProjectionTangentialContactMechanicsEquation,
)
from porepy.viz.data_saving_model_mixin import FractureDeformationExporting
from solution_strategy import SolutionStrategy
from time_stepping import name_schedule_intervals, time_steppers_coso
from wells import _WellDataBase
from porepy.applications.initialization.initialization import (
    initialization_pipeline,
    default_rate_tolerances,
    default_initialization_time_manager,
)

# ----------------------------------------------------------------------------------
# Run settings
# ----------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
FILE_NAME = Path(__file__).stem
OUTPUT_FOLDER = REPO_ROOT / FILE_NAME

FINAL_TIME = 2 * pp.YEAR
PRODUCTION_PERIOD = 1 * pp.YEAR
SHUT_IN_DURATION = 3 * pp.DAY
INITIAL_SHUTIN_DURATION = 2 * pp.DAY  # Zero-flux period at t=0.
TRANSITION_DURATION = 2 * pp.HOUR  # Length of the Neumann <-> Dirichlet ramps.
NUM_BINS_PER_INTERVAL = 3  # Extra output times per production / shut-in interval.

BOUNDARY_STRAIN_RATE = 1.0e-6  # [1/year], tectonic extension in x.
cell_size_multiplier = 1  # 1 corresponds to original COSO
CELL_SIZE_BOUNDARY = 318.0 * cell_size_multiplier  # [m]
CELL_SIZE_FRACTURE = 76.0 * cell_size_multiplier  # [m]
CELL_SIZE_MIN = 38.0 * cell_size_multiplier  # [m]

SOLVER_SCHEME = "monolithic"
TIME_STEPPER_PARAMS = {
    "iter_optimal_range": (10, 15),
    "iter_relax_factors": (0.6, 2.5),
    "recomp_factor": 0.2,
    "max_attempts": 10,
    "dt_start_transition": 10,
    "dt_max_divisor": 2.0,
    "dt_min": 1e-2,
}


# ----------------------------------------------------------------------------------
# Geometry: faults enabled in run_example_3.py plus an artificial well-tip fracture
# ----------------------------------------------------------------------------------
# Of the real Coso fault planes (point_cloud_clusters), keep those enabled in
# run_example_3.py: 0000, 0001, 0003 and 0005.
EXCLUDE_FAULTS = [
    # "0000",
    # "0001",
    "0002",
    # "0003",
    "0004",
    "0005",
    "0006",
    "0007",
    "0008",
    "0009",
    "0010",
    "0011",
    "0012",
]


class TrimmedWellsFaultGeometry(FaultPlaneGeometry):
    """Faults 0000/0001/0003/0005 and an artificial well-tip fracture.

    Wells are trimmed below the fractures they touch.

    The artificial fracture is a rectangle in the plane through the three well bottom
    tips, with sides along strike (horizontal) and dip, enlarged by ``fracture_scale``
    around the bounding rectangle of the tips. The domain covers the wells and both
    fractures. Each (straight) well is then shortened so that it ends
    ``well_depth_margin`` metres (measured along the well) beyond its deepest
    intersection with a fracture; wells that intersect no fracture are left untouched.
    """

    fracture_scale = 1.3
    domain_padding_xy = 2.0e3
    domain_padding_z = 1.5e3
    well_depth_margin = 100.0  # [m]

    def _well_lines(self) -> np.ndarray:
        """Wellhead and bottom tip of each well, shape ``(num_wells, 3, 2)``."""
        lines = []
        with pd.ExcelFile(f"{sys.path[0]}/data/wellbores.xlsx") as xls:
            for name in self.well_names:
                df = pd.read_excel(xls, sheet_name=name, usecols=[3, 4, 5], skiprows=1)
                # The file stores y, x, z with depth positive down.
                pts = df.iloc[:, [1, 0, 2]].values.T
                pts[2] *= -1
                tip = pts[:, np.argmin(pts[2])]
                head = np.array([pts[0, 0], pts[1, 0], 0.0])
                lines.append(np.column_stack([head, tip]))
        return np.array(lines)

    def _well_tip_fracture(self) -> pp.PlaneFracture:
        tips = self._well_lines()[:, :, 1].T  # (3, num_wells)
        origin = tips.mean(axis=1)
        normal = np.cross(tips[:, 1] - tips[:, 0], tips[:, 2] - tips[:, 0])
        # In-plane directions: strike (horizontal) and dip (steepest descent).
        strike = np.cross(normal, [0.0, 0.0, 1.0])
        if np.linalg.norm(strike) == 0.0:
            raise ValueError("The well tips span a horizontal plane.")
        strike /= np.linalg.norm(strike)
        dip = np.cross(normal, strike)
        dip /= np.linalg.norm(dip)
        # Bounding rectangle of the tips in (strike, dip) coordinates.
        local = np.array([strike, dip]) @ (tips - origin[:, None])
        center = (local.max(axis=1) + local.min(axis=1)) / 2
        half = self.fracture_scale * (local.max(axis=1) - local.min(axis=1)) / 2
        signs = np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]])
        corners_local = center[:, None] + half[:, None] * signs
        corners = origin[:, None] + np.array([strike, dip]).T @ corners_local
        return pp.PlaneFracture(corners)

    def set_fractures(self) -> None:
        super().set_fractures()  # Real faults (with extensions, if configured).
        self._fractures.append(self._well_tip_fracture())
        self._fault_names = [*self._fault_names, "well_tip_fracture"]

    def set_domain(self) -> None:
        self.set_fractures()
        points = np.hstack(
            [*self._well_lines(), *(frac.pts for frac in self._fractures)]
        )
        pad_xy, pad_z = self.domain_padding_xy, self.domain_padding_z
        self._domain = pp.Domain(
            bounding_box={
                "xmin": points[0].min() - pad_xy,
                "xmax": points[0].max() + pad_xy,
                "ymin": points[1].min() - pad_xy,
                "ymax": points[1].max() + pad_xy,
                "zmin": points[2].min() - pad_z,
                "zmax": self.params.get("top_boundary_z", 0.0),
            }
        )

    @staticmethod
    def _segment_fracture_intersection(
        p0: np.ndarray, p1: np.ndarray, corners: np.ndarray
    ) -> float | None:
        """Parameter t in [0, 1] where p0 + t (p1 - p0) crosses the fracture, if any."""
        e1, e2 = corners[:, 1] - corners[:, 0], corners[:, 3] - corners[:, 0]
        normal = np.cross(e1, e2)
        direction = p1 - p0
        denom = normal @ direction
        if abs(denom) < 1e-12:
            return None
        t = normal @ (corners[:, 0] - p0) / denom
        if not -1e-9 <= t <= 1.0 + 1e-9:
            return None
        # In-plane check on the (parallelogram) fracture.
        rel = p0 + t * direction - corners[:, 0]
        coeffs = np.linalg.lstsq(np.column_stack([e1, e2]), rel, rcond=None)[0]
        inside = np.all((coeffs >= -1e-9) & (coeffs <= 1.0 + 1e-9))
        return min(t, 1.0) if inside else None

    def set_well_network(self) -> None:
        super().set_well_network()
        if not self.params.get("use_wells", True):
            return
        # Rebuild the wells with shortened trajectories.
        wells = []
        for well in self.well_network.wells:
            pts = well.pts
            top, bottom = pts[:, 0], pts[:, -1]
            ts = [
                self._segment_fracture_intersection(top, bottom, frac.pts)
                for frac in self._fractures
            ]
            ts = [t for t in ts if t is not None]
            if ts:
                length = np.linalg.norm(bottom - top)
                t_end = min(1.0, max(ts) + self.well_depth_margin / length)
                pts = np.column_stack([top, top + t_end * (bottom - top)])
            wells.append(pp.Well(pts, tags=dict(well.tags)))
        self.well_network = pp.WellNetwork3d(domain=self._domain, wells=wells)


# ----------------------------------------------------------------------------------
# Initial state and tectonic loading (the initial state is refined by the
# initialization pipeline)
# ----------------------------------------------------------------------------------
class DepthDisplacementInitialAndBoundaryConditions(CosoBoundaryConditionsDisplacement):
    """Start from the depth-dependent displacement of ``CosoBackgroundValues``.

    Dirichlet displacement on all outer faces: the initial depth-dependent field plus
    a uniform extension in x, ``u_x = strain_rate * (x - x_min) * t``. The same field
    is used as the initial displacement in the domain.
    """

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        return self.displacement_from_depth(sd.cell_centers).ravel("F")

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        if boundary_grid.dim < self.nd - 1:
            return super().bc_values_displacement(boundary_grid)
        coords = boundary_grid.cell_centers
        values = self.displacement_from_depth(coords)
        x_from_boundary = coords[0] - self.domain.bounding_box["xmin"]
        values += np.outer(self.boundary_displacement_velocity, x_from_boundary) * (
            self.time_manager.time
        )
        return values.ravel("F")


# ----------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------
class CosoModel(
    # Initial conditions and boundary conditions. The order matters: the well mixins
    # (SmoothWellTransitions, NeumannWellBCsFromSchedule) modify the values and BC types
    # of the mixins below them via super(), e.g. WellBoundaryConditions and the generic
    # boundary conditions, so they must come first.
    CosoBackgroundValues,  # Geotherm, hydrostatic pressure, depth displacement.
    SmoothWellTransitions,  # Ramps between shut-in and production.
    NeumannWellBCsFromSchedule,  # Zero-flux wells during shut-in.
    InitialConditionHydrostaticPressureValues,
    InitialConditionThermalGradientTemperatureValues,
    WellBoundaryConditions,
    DepthDisplacementInitialAndBoundaryConditions,
    HeterogeneousPermeabilitySpecification,
    # Geometry, constitutive laws, solution strategy, exporting.
    FractureDeformationExporting,
    GeometryExporting,
    CosoExporter,
    TrimmedWellsFaultGeometry,
    HagenPoiseuilleWellPermeability,
    pp.constitutive_laws.CubicLawPermeability,
    SolutionStrategy,  # Precedence over pp.models.solution_strategy.ContactIndicators
    pp.models.solution_strategy.ContactIndicators,
    _WellDataBase,
    HydrostaticBoundaryPressureValues,
    ThermalGradientBoundaryTemperatureValues,
    LithostaticBoundaryStressValues,
    ScaledProjectionTangentialContactMechanicsEquation,
    PhysicalModel,  # Gravity + porepy's thermo-poromechanics.
):
    """Coso reservoir with faults 0000/0001/0003/0005 and a well-tip fracture."""

    @property
    def time_step_indices(self) -> np.ndarray:
        # Index 1 (the step before the latest) is read by CosoExporter.collect_data()
        # for the fracture displacement-jump increment.
        return np.array([0, 1])


def create_schedule(
    production_period: float,
    shut_in_duration: float = 3 * pp.DAY,
    final_time: float = 10 * pp.YEAR,
    num_bins_per_interval: int | None = None,
    transition_duration: float = 0.0,
    initial_shutin_duration: float = 0.0,
    with_production: bool = True,
) -> tuple[np.ndarray, Sequence[tuple[float, float]], np.ndarray]:
    """Build a simulation time schedule with alternating production and shut-in periods.

    Each cycle consists of a production sub-interval of length ``production_period -
    shut_in_duration`` followed by a shut-in sub-interval of length
    ``shut_in_duration``.  The schedule always starts at t=0 and ends at the last cycle
    boundary that is needed to cover ``final_time``.

    Shut-in periods correspond to intervals where Neumann (zero-flux) well boundary
    conditions are applied instead of prescribed pressures.  The returned
    ``neumann_intervals`` list marks exactly these shut-in windows; :func:`well_schedule`
    turns them into the Neumann intervals of the well schedule.

    If ``num_bins_per_interval`` is given, equally-spaced bin-boundary times are
    inserted into the schedule for *both* the production and shut-in sub-intervals of
    every cycle.  This ensures that the time integrator always outputs a state at each
    bin edge, so that post-processing (slip-rate / flux binning) can read off values at
    exactly the right times without interpolation.

    Parameters:
        production_period: Duration of one full cycle (production + shut-in) in seconds.
        shut_in_duration: Duration of the shut-in portion at the end of each cycle in
            seconds (default: 3 days).
        final_time: Total simulation end time in seconds (default: 10 years).  The
            number of cycles is rounded up so the schedule covers at least this long.
        num_bins_per_interval: Number of equal bins per sub-interval (production *and*
            shut-in).  When given, ``num_bins_per_interval + 1`` evenly-spaced boundary
            times are added to the schedule for each sub-interval.
        initial_shutin_duration: Duration of an optional zero-flux shut-in period at
            the very start of the simulation (seconds, default 0 = disabled).  When
            non-zero, Neumann BCs are active on all well grids for
            ``(0, initial_shutin_duration - transition_duration]``, then the existing
            production-ramp machinery linearly ramps the well-head pressure from the
            cached no-flow pressure to the scheduled target over the final
            ``transition_duration`` seconds of the initial shut-in.  Must be strictly
            greater than ``transition_duration`` when ``transition_duration > 0``.

    Returns:
        tuple: A tuple containing:
            - schedule (np.ndarray): Sorted array of time checkpoints (seconds),
              starting at 0 and ending at the last cycle boundary.
            - neumann_intervals (list[tuple[float, float]]): One ``(start, end)``
              tuple per shut-in period, with zero-flux well BCs. Input of
              :func:`well_schedule`.
            - is_transition (np.ndarray): Boolean mask aligned with ``schedule``,
              True where the checkpoint is a "mandatory" one (BC type switch or
              ramp start/end) and False where it is a purely cosmetic bin-boundary
              inserted by ``num_bins_per_interval`` for post-processing. Time steps
              starting right after a mandatory checkpoint tend to be numerically
              stiff (BC discontinuity), while steps after a bin-only checkpoint
              continue the same smooth physics as before it. Intended for
              :func:`time_stepping.time_steppers_coso` to pick a small ``dt_start``
              only where the physics actually changes.
    """
    if not with_production:
        # No need to add transition checkpoints if there is no production, since the BC
        # type never changes.
        transition_duration = 0.0
    if transition_duration > shut_in_duration:
        raise ValueError(
            f"transition_duration ({transition_duration:.3g} s) must not exceed "
            f"shut_in_duration ({shut_in_duration:.3g} s). The shut-in ramp and the "
            "production ramp would otherwise overlap."
        )

    # Number of complete cycles required to reach final_time.
    n = int(np.ceil(final_time / production_period))

    # For each cycle i (1-indexed), add the shut-in start (= production end) and the
    # cycle end.  These are the mandatory checkpoint times that delimit each sub-
    # interval boundary.  t=0 is prepended separately below. If a transition duration is
    # set, also insert, for each cycle:
    #   - the time of full shut-in (shut-in start + transition duration), when the
    #     shut-in flux ramp has finished, and
    #   - the production ramp start (cycle end - transition duration), which is also the
    #     end of the Neumann interval.
    # This ensures that the time integrator always steps at these times where the BC
    # type or target value changes, so the transitions are well-resolved and consistent
    # across runs, and the Neumann->Dirichlet switch always starts at an exact schedule
    # boundary.
    checkpoints = []
    for i in np.arange(1, n + 1):
        checkpoints.append(i * production_period - shut_in_duration)
        if transition_duration > 0.0:
            checkpoints.append(
                i * production_period - shut_in_duration + transition_duration
            )
            checkpoints.append(i * production_period - transition_duration)
        checkpoints.append(i * production_period)
    schedule = np.array(checkpoints)

    # Each Neumann interval covers the pure zero-flux part of the shut-in window:
    #   start = shut-in start  (= i*T - d_shutin)
    #   end   = transition start (= i*T - d_trans)
    # The final ``transition_duration`` seconds of each shut-in period are NOT included
    # in the Neumann interval; they form the production ramp window during which the BC
    # type is already Dirichlet but the well-head pressure is ramped from the last
    # reservoir pressure toward the scheduled target.  When transition_duration == 0,
    # end == i*T and the full shut-in window is Neumann.
    neumann_intervals = [
        (
            i * production_period - shut_in_duration,
            i * production_period - transition_duration,
        )
        for i in np.arange(1, n + 1)
    ]

    # Prepend t=0 so the time manager starts from the initial condition.
    schedule = np.insert(schedule, 0, 0.0)

    # If requested, prepend an initial shut-in Neumann interval and insert its mandatory
    # checkpoints into the schedule. The pure zero-flux period runs from t=0 to
    # t=(initial_shutin_duration - transition_duration); the final transition_duration
    # seconds of the initial shut-in are the production ramp window, handled
    # automatically by SmoothWellTransitions (see boundary_conditions.py), which reads
    # the end of each Neumann interval as a production start.
    if initial_shutin_duration > 0.0:
        if transition_duration > 0.0 and initial_shutin_duration <= transition_duration:
            raise ValueError(
                f"initial_shutin_duration ({initial_shutin_duration:.3g} s) must be "
                f"strictly greater than transition_duration ({transition_duration:.3g} s) "
                "so that at least one converged zero-flux step populates the pressure cache."
            )
        ramp_start = initial_shutin_duration - transition_duration
        extra: list[float] = [initial_shutin_duration]
        if transition_duration > 0.0:
            extra.append(ramp_start)
        schedule = np.unique(np.concatenate([schedule, extra]))
        neumann_intervals = [(0.0, ramp_start)] + neumann_intervals

    # All checkpoints up to this point are "mandatory": they mark an actual BC type
    # switch or ramp start/end. Anything added below (bin boundaries) is purely for
    # post-processing and does not correspond to a change in physics.
    mandatory_schedule = schedule

    if num_bins_per_interval is not None:
        bin_times = []
        for i in np.arange(1, n + 1):
            # Production sub-interval: from the end of the previous cycle to the
            # start of this cycle's shut-in.
            prod_start = (i - 1) * production_period
            prod_end = i * production_period - shut_in_duration
            bin_times.append(
                np.linspace(prod_start, prod_end, num_bins_per_interval + 1)
            )
            # Shut-in sub-interval: from shut-in start to cycle end.
            shut_start = prod_end
            shut_end = i * production_period
            bin_times.append(
                np.linspace(shut_start, shut_end, num_bins_per_interval + 1)
            )
        # Merge bin boundaries with the existing mandatory checkpoints and deduplicate.
        schedule = np.unique(np.concatenate([schedule] + bin_times))

    is_transition = np.isin(schedule, mandatory_schedule)

    return schedule, neumann_intervals, is_transition


def well_schedule(
    schedule: np.ndarray, neumann_intervals: list[tuple[float, float]]
) -> dict[str, list[WellInterval]]:
    """Well schedule: constant protocol values, Neumann in the shut-in windows.

    Each ``neumann_intervals`` entry ``(start, end)`` becomes a Neumann interval of
    the well schedule; the well is Dirichlet in between.
    """
    t_start = float(schedule[0])
    bc_types = {t_start: WellBCType.DIRICHLET}
    for start, end in neumann_intervals:
        bc_types[float(start)] = WellBCType.NEUMANN
        bc_types[float(end)] = WellBCType.DIRICHLET

    def intervals(**values: float) -> list[WellInterval]:
        return [
            WellInterval(t, values, bc_type) for t, bc_type in sorted(bc_types.items())
        ]

    schedules = {}
    for name in CosoModel.injection_well_names.fget(None):
        schedules[name] = intervals(
            pressures=0.3 * pp.MEGA * pp.PASCAL,
            temperatures=pp.Celsius_to_Kelvin(40),
        )
    for name in CosoModel.production_well_names.fget(None):
        schedules[name] = intervals(
            pressures=pp.ATMOSPHERIC_PRESSURE,
            temperatures=475.0,  # Expected T at depth.
        )
    return schedules


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pp_solvers.porepy_integration").setLevel(logging.WARNING)

    schedule, neumann_intervals, is_transition = create_schedule(
        PRODUCTION_PERIOD,
        shut_in_duration=SHUT_IN_DURATION,
        final_time=FINAL_TIME,
        num_bins_per_interval=NUM_BINS_PER_INTERVAL,
        transition_duration=TRANSITION_DURATION,
        initial_shutin_duration=INITIAL_SHUTIN_DURATION,
    )
    time_stepper, _ = time_steppers_coso(
        schedule,
        dt=1e3,
        is_transition=is_transition,
        interval_names=name_schedule_intervals(
            schedule, neumann_intervals, TRANSITION_DURATION
        ),
        **TIME_STEPPER_PARAMS,
    )

    # Re-fitted thermal expansion for the Coso T range (see physical_model.py).
    water_values = {**pp.fluid_values.water, "thermal_expansion": 7.8745e-04}
    model_params = {
        "file_name": FILE_NAME,
        "folder_name": str(OUTPUT_FOLDER),
        "data_folder_name": f"{OUTPUT_FOLDER}_saved_data",
        "solver_statistics_file_name": "solver_statistics.json",
        "time_manager": time_stepper.time_manager,
        "material_constants": {
            "solid": pp.SolidConstants(**granodiorite_values),
            "fluid": pp.FluidComponent(**water_values),
        },
        "units": pp.Units(m=1.0e0, kg=1.0e9, K=1.0),
        "reference_variable_values": pp.ReferenceVariableValues(
            temperature=pp.Celsius_to_Kelvin(20),
            pressure=pp.ATMOSPHERIC_PRESSURE,
        ),
        "surface_temperature": pp.Celsius_to_Kelvin(20.0),
        "lithostatic_stress_multipliers": np.array([0.62, 1.55, 1.0]),
        "fault_plane_dirs": ["point_cloud_clusters"],
        "fault_extension_config": "fault_extensions.json",
        "exclude_faults": EXCLUDE_FAULTS,
        "top_boundary_z": 0.0,
        "grid_type": "simplex",
        "meshing_arguments": {
            "cell_size_boundary": CELL_SIZE_BOUNDARY,
            "cell_size_fracture": CELL_SIZE_FRACTURE,
            "cell_size_min": CELL_SIZE_MIN,
        },
        "adaptive_indicator_scaling": 1,
        "heterogeneous_permeability": True,
        "darcy_flux_discretization": "mpfa",
        "fourier_flux_discretization": "mpfa",
        "boundary_displacement_velocity": BOUNDARY_STRAIN_RATE / pp.YEAR,
        "use_wells": True,
        "well_transition_duration": TRANSITION_DURATION,
        "well_schedule": well_schedule(schedule, neumann_intervals),
    }

    model = CosoModel(model_params)

    # The initialization and the main run each need their own nonlinear solver
    # instance: the solver depends on the model's grids, which are rebuilt at every
    # prepare_simulation call (the initialization pipeline prepares the model several
    # times). One solver cannot be shared between the two.
    initialization_pipeline(
        model,
        nonlinear_solver=set_nonlinear_solver(
            iterative_linear_solver=True,
            system_amg=True,
            scheme=SOLVER_SCHEME,
            nonlinear_solver_params={
                "nl_convergence_res_atol": 1e-5,
                "nl_convergence_inc_atol": 1e-5,
            },
        ),
        rate_tolerances=default_rate_tolerances(model),
        residual_tolerances=defaultdict(lambda: 1e-5),
        time_manager=default_initialization_time_manager(),
        # Initialize with the first well interval only (Neumann, i.e. the wells are shut
        # in), held constant in time. The initialization clock runs far beyond the
        # first interval, so the full schedule would switch the wells from Neumann to
        # Dirichlet in the middle of the initialization.
        initialization_params={
            "well_schedule": {
                well: intervals[:1]
                for well, intervals in model_params["well_schedule"].items()
            }
        },
    )

    t0 = time.perf_counter()
    pp.ModelRunner(
        model,
        # The initialization pipeline has already prepared the model; preparing again
        # would discard the initialized state.
        params={"prepare_simulation": False},
        nonlinear_solver=set_nonlinear_solver(
            iterative_linear_solver=True,
            system_amg=True,
            scheme=SOLVER_SCHEME,
            nonlinear_solver_params={
                "nl_convergence_res_atol": 1e-5,
                "nl_convergence_inc_atol": 1e-8,
            },
        ),
        time_stepper=time_stepper,
    ).run()
    model.save_results()
    print(
        f"{model.time_manager.time_index} time steps, {time.perf_counter() - t0:.1f} s"
    )
