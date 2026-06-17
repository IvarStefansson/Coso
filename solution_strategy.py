import logging
import time
import warnings
from functools import partial

import numpy as np
import porepy as pp
import scipy.sparse as sps
from porepy.applications.discretizations.flux_discretization import FluxDiscretization

logger = logging.getLogger(__name__)


def clip_decorator(a: float, b: float):
    """Decorator factory to mask the output to the interval [a, b] via two Heaviside functions."""
    if b <= a:
        raise ValueError("b must be greater than a")

    def decorator(func):
        def wrapper(*args, **kwargs):
            f = func(*args, **kwargs)
            f_heaviside = pp.ad.Function(
                partial(pp.ad.functions.heaviside, 0.0),
                "heaviside_function_for_clipping",
            )
            return (
                f
                - f_heaviside(f - pp.ad.Scalar(b)) * (f - pp.ad.Scalar(b))
                + f_heaviside(pp.ad.Scalar(a) - f) * (pp.ad.Scalar(a) - f)
            )

        return wrapper

    return decorator


class SolutionStrategy(FluxDiscretization):
    def initial_condition(self) -> None:
        """Set initial conditions for the model.
        This method sets the initial conditions for the model by reading well data and
        setting the initial values for pressure and displacement.

        """
        # First read input from the well data files. This is used when setting initial
        # BCs.
        if hasattr(self, "read_well_data"):
            self.read_well_data()
        # self.fracture_locking_variable = pp.ad.Scalar(0)
        super().initial_condition()

    def solve_linear_system(self) -> np.ndarray:
        """Solve linear system.

        Default method is a direct solver. The linear solver is chosen in the
        initialize_linear_solver of this model. Implemented options are
            - scipy.sparse.spsolve with and without call to umfpack
            - pypardiso.spsolve

        See also:
            :meth:`initialize_linear_solver`

        Returns:
            np.ndarray: Solution vector.

        """
        A, b = self.linear_system
        t_0 = time.time()
        logger.debug(f"Max element in A {np.max(np.abs(A)):.2e}")
        logger.debug(
            f"""Max {np.max(np.sum(np.abs(A), axis=1)):.2e} and min
            {np.min(np.sum(np.abs(A), axis=1)):.2e} A sum."""
        )

        solver = self.linear_solver
        if solver == "pypardiso":
            # This is the default option which is invoked unless explicitly overridden
            # by the user. We need to check if the pypardiso package is available.
            try:
                from pypardiso import spsolve as sparse_solver  # type: ignore
            except ImportError:
                # Fall back on the standard scipy sparse solver.
                sparse_solver = sps.linalg.spsolve
                warnings.warn(
                    """PyPardiso could not be imported,
                    falling back on scipy.sparse.linalg.spsolve"""
                )
            try:
                x = sparse_solver(A, b)
            except Exception as e:
                warnings.warn(
                    f"Pypardiso solver failed with error {e}, returning NaN solution."
                )
                x = np.full(b.shape, np.nan)
        elif solver == "umfpack":
            # Following may be needed:
            # A.indices = A.indices.astype(np.int64)
            # A.indptr = A.indptr.astype(np.int64)
            x = sps.linalg.spsolve(A, b, use_umfpack=True)
        elif solver == "scipy_sparse":
            x = sps.linalg.spsolve(A, b)
        else:
            raise ValueError(
                f"AbstractModel does not know how to apply the linear solver {solver}"
            )
        x = np.atleast_1d(x)
        logger.info(f"Solved linear system in {time.time() - t_0:.2e} seconds.")
        return x

    def update_discretization_parameters(self) -> None:
        """Set default (unitary/zero) parameters for the energy problem.

        The parameter fields of the data dictionaries are updated for all subdomains and
        interfaces (of codimension 1). The data to be set is related to:

        - The temperature diffusion, e.g., the thermal conductivity and boundary
          conditions for the temperature. This applies to subdomains and interfaces.
        - Boundary conditions for the conductive heat flux. This applies to subdomains
          only.

        """
        super().update_discretization_parameters()
        inverter = self.params.get("mpfa_inverter", "numba")
        for _, data in self.mdg.subdomains(return_data=True):
            pp.initialize_data(
                data,
                self.fourier_keyword,
                {
                    "mpfa_inverter": inverter,
                },
            )
            pp.initialize_data(
                data,
                self.darcy_keyword,
                {
                    "mpfa_inverter": inverter,
                },
            )


class Foo:
    def before_nonlinear_iteration(self) -> None:
        super().before_nonlinear_iteration()
        # Update fracture locking variable. Positive values mean that the fractures are
        # not locked. Should always be zero during the first time step, then zero during
        # the first iteration, and then increase with the number of iterations.
        self.fracture_locking_variable.set_value(
            self.nonlinear_solver_statistics.num_iterations
            * (self.time_manager.time_index > 1)
        )

    def normal_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        nd_vec_to_normal = self.normal_component(subdomains)
        # The normal component of the contact traction and the displacement jump.
        u_n: pp.ad.Operator = nd_vec_to_normal @ self.displacement_jump(subdomains)
        eq = super().normal_fracture_deformation_equation(subdomains)

        h = self.locking_operator(subdomains)
        eq = h * eq + (u_n - self.fracture_gap(subdomains)) * (pp.ad.Scalar(1) - h)
        eq.set_name("normal_fracture_deformation_equation_with_locking")
        return eq

    def tangential_fracture_deformation_equation(
        self, subdomains: list[pp.Grid]
    ) -> pp.ad.Operator:
        nd_vec_to_tangential = self.tangential_component(subdomains)
        # The tangential component of the contact traction and the displacement jump.
        u_t: pp.ad.Operator = nd_vec_to_tangential @ self.displacement_jump(subdomains)
        eq = super().tangential_fracture_deformation_equation(subdomains)

        h = self.locking_operator(subdomains)
        eq = h * eq + u_t * (pp.ad.Scalar(1) - h)
        eq.set_name("tangential_fracture_deformation_equation_with_locking")
        return eq

    def sliding_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        h = self.locking_operator(subdomains)
        return h * super().sliding_indicator(subdomains) + (pp.ad.Scalar(1) - h)

    def opening_indicator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        h = self.locking_operator(subdomains)
        return h * super().opening_indicator(subdomains) + (pp.ad.Scalar(1) - h)

    def locking_operator(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Locking indicator for fractures.

        Parameters:
            subdomains: List of fracture subdomains.

        """
        f_heaviside = pp.ad.Function(partial(pp.ad.heaviside, 0), "heaviside_function")
        return pp.ad.Scalar(0)  # f_heaviside(self.fracture_locking_variable)

    @clip_decorator(1e-5, 1.0 - 1e-5)
    def porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return super().porosity(subdomains)

    @clip_decorator(1e-25, 1.0)
    def permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        return super().permeability(subdomains)
