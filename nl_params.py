import logging
from pathlib import Path

import porepy as pp
import pp_solvers

REPO_ROOT = Path(__file__).resolve().parent
# Snapshot of the linear system (matrix, rhs, indexer bookkeeping) each solver last
# attempted, overwritten on every solve. Cheap safety net for crashes that kill the
# whole process before any Python-level except can run (e.g. the native heap-
# corruption abort inside HYPRE's BoomerAMGCreateS -- see debugging session on
# run_example_3.py, ~time step 38): whatever landed here right before such a crash
# is the exact input that triggered it. See
# IterativeLinearSolver._dump_linear_system_for_debugging in pp_solvers.
LINEAR_SYSTEM_DUMP_DIR = REPO_ROOT / "linear_system_dumps"

# Enable the ML solver selector for the monolithic scheme. Off by default, matching the
# `main` branch, where the selector was likewise present but switched off.
USE_SOLVER_SELECTOR = False

try:
    # These are only needed for the iterative-solver preconditioner factory
    # (th_linear_solver_factory / set_nonlinear_solver(iterative_linear_solver=True)).
    # Guarded so that importing this module still works with a pp_solvers version
    # where this API has drifted, as long as the iterative solver path isn't used.
    from pp_solvers.equation_variable_groups import (
        DefaultEquationVariableGroups,
        EquationVariableGroup,
    )
    from pp_solvers.preconditioners import (
        GMRES,
        ILU,
        CompositePreconditioner,
        DiagonalInverter,
        FieldSplit,
        FieldSplitSchur,
        AMG,
        Identity,
        LinearSolverConfiguration,
        PythonPermutationWrapper,
    )
except ImportError:
    pass

solver_params = {
    "nl_convergence_res_atol": 1e-1,
    "nl_convergence_inc_atol": 1e-4,
    "nl_divergence_res_atol": 1e20,
    "nl_max_iterations": 20,
    "nonlinear_solver": pp.solvers.ConstraintLineSearchNonlinearSolver,
    "local_line_search": 1,
    "global_line_search": 0,
    "residual_line_search_interval_size": 1e-3,
    "constraint_violation_tolerance": 1e-3,
    # "linear_solver": "scipy_sparse",
}

# Parameters for the monolithic nonlinear solver. Same values as `solver_params` above,
# minus its "nonlinear_solver" key: under the current porepy API the solver class is
# instantiated directly and this dict is handed to it as `params=`, so that key would be
# dead state.
monolithic_solver_params = {
    "nl_convergence_res_atol": 1e-1,
    "nl_convergence_inc_atol": 1e-4,
    "nl_divergence_res_atol": 1e20,
    "nl_max_iterations": 20,
    "local_line_search": 1,
    "global_line_search": 0,
    "residual_line_search_interval_size": 1e-3,
    "constraint_violation_tolerance": 1e-3,
}


def th_linear_solver_factory():
    interface_flow_groups: list[EquationVariableGroup] = [
        DefaultEquationVariableGroups.interface_darcy_flux_group,
        DefaultEquationVariableGroups.well_flux_group,
    ]
    interface_energy_groups: list[EquationVariableGroup] = [
        DefaultEquationVariableGroups.interface_enthalpy_flux_group,
        DefaultEquationVariableGroups.interface_fourier_flux_group,
        DefaultEquationVariableGroups.well_enthalpy_flux_group,
    ]
    mass_balance_groups: list[EquationVariableGroup] = [
        DefaultEquationVariableGroups.mass_balance_pressure_matrix_group,
        DefaultEquationVariableGroups.mass_balance_pressure_fractures_group,
        DefaultEquationVariableGroups.mass_balance_pressure_intersections_group,
    ]
    energy_balance_groups: list[EquationVariableGroup] = [
        EquationVariableGroup(
            equation_tag=pp.solvers.DefaultEquationTags.energy_balance,
            variable_tag=pp.solvers.DefaultVariableTags.temperature,
        ).restricted(pp.solvers.OnAmbientDimension()),
        EquationVariableGroup(
            equation_tag=pp.solvers.DefaultEquationTags.energy_balance,
            variable_tag=pp.solvers.DefaultVariableTags.temperature,
        ).restricted(pp.solvers.OnFractures()),
        EquationVariableGroup(
            equation_tag=pp.solvers.DefaultEquationTags.energy_balance,
            variable_tag=pp.solvers.DefaultVariableTags.temperature,
        ).restricted(pp.solvers.OnLowerDimensions()),
    ]

    solver = GMRES(
        preconditioner=FieldSplitSchur(
            subsolver=ILU(groups=interface_flow_groups, key="interface_flow"),
            approximate_inverter=DiagonalInverter(),
            complement_solver=FieldSplitSchur(
                subsolver=ILU(groups=interface_energy_groups, key="interface_energy"),
                approximate_inverter=DiagonalInverter(),
                complement_solver=CompositePreconditioner(
                    subsolvers=[
                        FieldSplit(
                            subsolvers=[
                                Identity(
                                    groups=energy_balance_groups, key="cpr0_energy"
                                ),
                                AMG(groups=mass_balance_groups, key="cpr0_mass"),
                            ]
                        ),
                        ILU(
                            groups=energy_balance_groups + mass_balance_groups,
                            key="cpr1",
                        ),
                    ]
                ),
            ),
        )
    )
    return LinearSolverConfiguration(
        transformations=[
            # ScaleSpecificVolume(groups=energy_balance_groups),
            # ScaleSpecificVolume(groups=mass_balance_groups),
        ],
        solver=solver,
    )


def _monolithic_nonlinear_solver(iterative_linear_solver: bool):
    """Build the monolithic solver: one Newton over the fully coupled THM system.

    Ported from the `main` branch, where the same scheme was assembled as
    `add_mixin(pp_solvers.IterativeSolverMixin, Model)` plus a
    `model_params["linear_solver"]` dict pointing at `pp_solvers.thm_factory`. Neither
    the mixin nor `pp_solvers.LinearSolverParams` exists any more; the linear solver is
    now an object handed to the nonlinear solver, and the solver selector is one of its
    constructor arguments.

    Contrast with the sequential mechanics/pT split built by
    `set_nonlinear_solver(scheme="sequential")`, which solves the two blocks in turn.

    """
    linear_solver = None  # -> porepy's default LinearSolverDirect("pypardiso").
    if iterative_linear_solver:
        # Imported lazily: solver_configurations constructs an ML SolverSelector at
        # import time, which the direct-solver path has no reason to pay for.
        from solver_configurations import (
            linear_solver_params,
            linear_solver_selector_options,
            solver_selector,
        )

        options = linear_solver_params
        if USE_SOLVER_SELECTOR:
            options = {**linear_solver_params, **linear_solver_selector_options}
        linear_solver = pp_solvers.IterativeLinearSolver(
            configuration_factory=pp_solvers.thm_factory,
            solver_options=options,
            solver_selector=solver_selector if USE_SOLVER_SELECTOR else None,
            debug_dump_dir=LINEAR_SYSTEM_DUMP_DIR / "monolithic",
        )
    # ConstraintLineSearchNonlinearSolver operates on the full system by construction
    # (it rejects equation_tags/variable_tags), which is exactly the monolithic case.
    # The tailored contact line search needs the model to provide opening_indicator and
    # sliding_indicator, i.e. to mix in pp.models.solution_strategy.ContactIndicators.
    return pp.solvers.ConstraintLineSearchNonlinearSolver(
        params=monolithic_solver_params, linear_solver=linear_solver
    )


def set_nonlinear_solver(iterative_linear_solver=False, scheme="sequential"):
    """Build the nonlinear solver for the Coso models.

    Parameters:
        iterative_linear_solver: If True, use pp_solvers' PETSc-preconditioned Krylov
            solvers instead of a direct solve.
        scheme: Either "sequential" (default) for the mechanics/pT split, or
            "monolithic" for a single Newton over the fully coupled system.

    """
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("porepy.numerics.solvers.nonlinear_solvers").setLevel(
        logging.WARNING
    )
    logging.getLogger("porepy.numerics.solvers.linear_solvers.linear_solver").setLevel(
        logging.WARNING
    )

    if scheme == "monolithic":
        return _monolithic_nonlinear_solver(iterative_linear_solver)
    if scheme != "sequential":
        raise ValueError(
            f"Unknown nonlinear solver scheme {scheme!r}. "
            "Expected 'sequential' or 'monolithic'."
        )

    if iterative_linear_solver:
        linear_solver_mechanics = pp_solvers.IterativeLinearSolver(
            configuration_factory=pp_solvers.momentum_balance_factory,
            debug_dump_dir=LINEAR_SYSTEM_DUMP_DIR / "mechanics",
        )
        linear_solver_pT = pp_solvers.IterativeLinearSolver(
            delete_matrices=False,
            configuration_factory=th_linear_solver_factory,
            debug_dump_dir=LINEAR_SYSTEM_DUMP_DIR / "pT",
            solver_options={
                # "gmres": {
                #     "ksp_monitor": None,
                #     "ksp_max_it": 10,
                # },
                # "cpr0_energy": {
                #     "ksp_monitor": None,
                # },
                # "cpr0_mass": {
                #     "ksp_monitor": None,
                # },
                "cpr1": {
                    # "ksp_monitor": None,
                    "pc_type": "hypre",
                    "pc_hypre_type": "ilu",
                    "pc_hypre_ilu_maxiter": 1,
                    "pc_hypre_ilu_local_reordering": 1,
                },
            },
        )
    else:
        linear_solver_mechanics = None
        linear_solver_pT = None

    return pp.solvers.SequentialNonlinearSolver(
        max_iterations=25,
        convergence_criteria=pp.solvers.assemble_default_convergence_criteria(
            is_nonlinear_problem=True,
            inc_atol=1e-1,
            inc_rtol=1e-4,
            res_atol=1e-4,
            res_rtol=1e-1,
            metric=pp.EuclideanMetric(),
        ),
        divergence_criteria=pp.solvers.assemble_default_divergence_criteria(
            is_nonlinear_problem=True,
            max_iterations=25,
            inc_div_atol=1e20,
            res_div_atol=1e20,
            metric=pp.EuclideanMetric(),
        ),
        subsolvers=[
            pp.solvers.NewtonSolver(
                linear_solver=linear_solver_mechanics,
                params={
                    "nl_max_iterations": 25,
                    "nl_convergence_inc_atol": 1e-4,
                    "nl_convergence_res_atol": 1e-4,
                    "nl_divergence_inc_atol": 1e20,
                    "nl_divergence_res_atol": 1e20,
                },
                equation_tags=[
                    pp.solvers.DefaultEquationTags.momentum_balance,
                    pp.solvers.DefaultEquationTags.interface_force_balance,
                    pp.solvers.DefaultEquationTags.normal_fracture_deformation,
                    pp.solvers.DefaultEquationTags.tangential_fracture_deformation,
                ],
                variable_tags=[
                    pp.solvers.DefaultVariableTags.displacement,
                    pp.solvers.DefaultVariableTags.interface_displacement,
                    pp.solvers.DefaultVariableTags.contact_traction,
                ],
            ),
            pp.solvers.NewtonSolver(
                linear_solver=linear_solver_pT,
                params={
                    "nl_max_iterations": 25,
                    "nl_convergence_inc_atol": 1e-4,
                    "nl_convergence_res_atol": 1e-4,
                    "nl_divergence_inc_atol": 1e20,
                    "nl_divergence_res_atol": 1e20,
                    "global_line_search": 1,
                    "local_line_search": 0,
                },
                equation_tags=[
                    pp.solvers.DefaultEquationTags.mass_balance,
                    pp.solvers.DefaultEquationTags.interface_darcy_flux,
                    pp.solvers.DefaultEquationTags.well_flux,
                    pp.solvers.DefaultEquationTags.energy_balance,
                    pp.solvers.DefaultEquationTags.interface_fourier_flux,
                    pp.solvers.DefaultEquationTags.interface_enthalpy_flux,
                    pp.solvers.DefaultEquationTags.well_enthalpy_flux,
                ],
                variable_tags=[
                    pp.solvers.DefaultVariableTags.pressure,
                    pp.solvers.DefaultVariableTags.interface_darcy_flux,
                    pp.solvers.DefaultVariableTags.well_flux,
                    pp.solvers.DefaultVariableTags.temperature,
                    pp.solvers.DefaultVariableTags.interface_fourier_flux,
                    pp.solvers.DefaultVariableTags.interface_enthalpy_flux,
                    pp.solvers.DefaultVariableTags.well_enthalpy_flux,
                ],
            ),
        ],
    )
