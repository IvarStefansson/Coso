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
linear_solver_selector_params = {}

linear_solver_params = {}
