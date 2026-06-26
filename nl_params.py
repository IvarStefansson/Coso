from porepy.numerics.nonlinear.line_search import ConstraintLineSearchNonlinearSolver

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
