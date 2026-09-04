"""Tests pinning down how nl_params.set_nonlinear_solver wires its two schemes.

The monolithic scheme is a port of the `main` branch's setup, which was written
against APIs that no longer exist (`pp_solvers.IterativeSolverMixin`,
`pp_solvers.LinearSolverParams`, a `model_params["linear_solver"]` dict). These tests
pin the ported wiring -- that the monolithic solver really is assembled from
`pp_solvers.thm_factory` and that the solver selector is passed the new way -- and,
just as importantly, that adding the `scheme` switch did not disturb the sequential
split that is still the default.

Solver objects only: no meshing, no model, so the whole file runs in well under a
second.
"""

from __future__ import annotations

import porepy as pp
import pytest

import nl_params

pp_solvers = pytest.importorskip("pp_solvers")


def test_default_is_sequential():
    """The new `scheme` parameter must default to the pre-existing behaviour."""
    solver = nl_params.set_nonlinear_solver()
    assert isinstance(solver, pp.solvers.SequentialNonlinearSolver)


def test_sequential_structure_unchanged():
    solver = nl_params.set_nonlinear_solver()
    assert len(solver.subsolvers) == 2
    mechanics, pT = solver.subsolvers
    assert isinstance(mechanics, pp.solvers.NewtonSolver)
    assert isinstance(pT, pp.solvers.NewtonSolver)
    assert pp.solvers.DefaultEquationTags.momentum_balance in mechanics.equation_tags
    assert (
        pp.solvers.DefaultEquationTags.normal_fracture_deformation
        in mechanics.equation_tags
    )
    assert pp.solvers.DefaultEquationTags.mass_balance in pT.equation_tags
    assert pp.solvers.DefaultEquationTags.energy_balance in pT.equation_tags
    # The two blocks must not overlap, or the sequential scheme is not a split.
    assert not set(mechanics.equation_tags) & set(pT.equation_tags)


def test_sequential_iterative_factories():
    solver = nl_params.set_nonlinear_solver(iterative_linear_solver=True)
    mechanics, pT = solver.subsolvers
    assert (
        mechanics.linear_solver.configuration_factory
        is pp_solvers.momentum_balance_factory
    )
    assert pT.linear_solver.configuration_factory is nl_params.th_linear_solver_factory


def test_monolithic_is_constraint_line_search():
    solver = nl_params.set_nonlinear_solver(scheme="monolithic")
    assert isinstance(solver, pp.solvers.ConstraintLineSearchNonlinearSolver)
    assert not isinstance(solver, pp.solvers.SequentialNonlinearSolver)
    # Monolithic means the solver is not restricted to a subset of the system.
    assert not solver.equation_tags
    assert not solver.variable_tags


def test_monolithic_uses_thm_factory():
    """The one assertion that says main's monolithic preconditioner is back."""
    import solver_configurations

    solver = nl_params.set_nonlinear_solver(
        iterative_linear_solver=True, scheme="monolithic"
    )
    linear_solver = solver.linear_solver
    assert isinstance(linear_solver, pp_solvers.IterativeLinearSolver)
    assert linear_solver.configuration_factory is pp_solvers.thm_factory
    assert linear_solver.solver_options == solver_configurations.linear_solver_params


def test_monolithic_direct_by_default():
    """Without the iterative solver the monolithic scheme is a direct-solve reference."""
    solver = nl_params.set_nonlinear_solver(
        iterative_linear_solver=False, scheme="monolithic"
    )
    assert isinstance(solver.linear_solver, pp.solvers.LinearSolverDirect)


def test_monolithic_selector_wiring(monkeypatch):
    """`LinearSolverParams(solver_selector=...)` is now a constructor kwarg."""
    import solver_configurations

    off = nl_params.set_nonlinear_solver(
        iterative_linear_solver=True, scheme="monolithic"
    )
    assert off.linear_solver.solver_selector is None

    monkeypatch.setattr(nl_params, "USE_SOLVER_SELECTOR", True)
    on = nl_params.set_nonlinear_solver(
        iterative_linear_solver=True, scheme="monolithic"
    )
    assert on.linear_solver.solver_selector is solver_configurations.solver_selector
    # The selector's manual options are merged in on top of the plain ones.
    assert on.linear_solver.solver_options["gmres"]["ksp_monitor"] is None


def test_monolithic_dump_dir_distinct():
    """A monolithic run must not overwrite a sequential run's crash evidence."""
    monolithic = nl_params.set_nonlinear_solver(
        iterative_linear_solver=True, scheme="monolithic"
    ).linear_solver.debug_dump_dir
    sequential = nl_params.set_nonlinear_solver(iterative_linear_solver=True)
    dirs = {s.linear_solver.debug_dump_dir for s in sequential.subsolvers}

    assert monolithic.parent == nl_params.LINEAR_SYSTEM_DUMP_DIR
    assert monolithic not in dirs


def test_monolithic_params_carry_line_search():
    params = nl_params.monolithic_solver_params
    assert params["local_line_search"] == 1
    # Dead under the current API: the class is instantiated directly, so a
    # "nonlinear_solver" key handed to it as params= would silently do nothing.
    assert "nonlinear_solver" not in params


def test_unknown_scheme_raises():
    with pytest.raises(ValueError) as excinfo:
        nl_params.set_nonlinear_solver(scheme="bogus")
    message = str(excinfo.value)
    assert "sequential" in message and "monolithic" in message
