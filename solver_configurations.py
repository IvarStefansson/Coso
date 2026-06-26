from porepy.numerics.nonlinear.line_search import ConstraintLineSearchNonlinearSolver
import pp_solvers
from pp_solvers.solver_selection import (
    SolverSpace,
    SolverSelector,
    NumericalChoices,
    CategoricalChoices,
    assemble_default_performance_predictor,
)

# The solver space scheme mirrors the options format that we use without solver
# selection (e.g. see `thm_factory` function docstring):
# {
#     "solver_key": {
#         "petsc_key": "petsc_value"
#     },
# }
# But now we can use CategoricalChoices and NumericalChoices that describe ranges of
# options.

solver_space = SolverSpace(
    {
        "options": {
            "gmres": {
                "ksp_gmres_restart": NumericalChoices([400]),
            },
            "mechanics_amg": {
                "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                    [0.5, 0.6, 0.7, 0.8, 0.9]
                ),
            },
            "cpr0_energy": {
                "pc_type": CategoricalChoices(["pbjacobi", "none"]),
            },
            "cpr0_mass": {
                "pc_hypre_boomeramg_strong_threshold": NumericalChoices(
                    [0.5, 0.6, 0.7, 0.8, 0.9]
                ),
            },
            "cpr1": {
                "pc_type": CategoricalChoices(["pbjacobi", "sor", "ilu"]),
            },
        }
    }
)
solver_selector = SolverSelector(
    solver_space=solver_space,
    performance_predictor=assemble_default_performance_predictor(),
)
linear_solver_selector_params = pp_solvers.LinearSolverParams(
    # Pass the solver selector to enable it.
    solver_selector=solver_selector,
    # Optionally, pass the manual options. They are merged with the solver
    # selector's output, but the solver selector takes priority on conflicts.
    options={
        "gmres": {
            "ksp_monitor": None,
        }
    },
)
linear_solver_params = {
    # Options for mechanics.
    # "mechanics_amg": {
    #     "pc_hypre_boomeramg_strong_threshold": 0.9,
    #     "pc_hypre_boomeramg_smooth_type": "ilu",
    #     "pc_hypre_boomeramg_ilu_level": 1,
    #     # "ksp_type": "gmres",
    #     # "ksp_rtol": 1e-6,
    #     # "ksp_monitor": None,
    #     "pc_hypre_boomeramg_ilu_drop_tol": 1e-5,
    # },
    # # Options for the temperature block in the first stage of CPR.
    # "cpr0_energy": {"pc_type": "hypre", "pc_hypre_type": "ilu"},
    # # "cpr0_energy": {"pc_type": "ilu"},
    # "cpr1": {
    #     "pc_type": "hypre",
    #     "pc_hypre_type": "ilu",
    #     "pc_hypre_ilu_level": 1,
    #     # "pc_hypre_ilu_drop_tol": 1e-5,
    # },
    "gmres": {
        # Options for the outer solver
        # "ksp_monitor": None,
        # "ksp_type": "fgmres",
        "ksp_max_it": 400,
        # "ksp_gmres_classicalgramschmidt": False,
    },
    # "cpr_composite": {
    #     "ksp_type": "gmres",
    #     "ksp_rtol": 1e-10,
    # # },
    # "interface_flow": {
    #     "pc_type": "none",
    #     # "pc_hypre_type": "ilu",
    #     "ksp_monitor": None,
    #     "ksp_type": "gmres",
    #     "ksp_rtol": 1e-12,
    # },
}

solver_params = {
    "nl_convergence_res_atol": 1e-1,
    "nl_convergence_inc_atol": 1e-4,
    "nl_divergence_res_atol": 1e20,
    "nl_max_iterations": 20,
    "nonlinear_solver": ConstraintLineSearchNonlinearSolver,
    "local_line_search": 1,
    "global_line_search": 0,
    "residual_line_search_interval_size": 1e-3,
    "constraint_violation_tolerance": 1e-3,
    # "linear_solver": "scipy_sparse",
}
