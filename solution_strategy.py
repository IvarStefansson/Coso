import logging
import time
import numpy as np
import scipy.sparse as sps
import warnings
from typing import Union
import porepy as pp
from functools import partial
from functools import wraps


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


class SolutionStrategy:
    def initial_condition(self) -> None:
        """Set initial conditions for the model.
        This method sets the initial conditions for the model by reading well data and
        setting the initial values for pressure and displacement.

        """
        # First read input from the well data files. This is used when setting initial
        # BCs.
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


class Foo:
    def before_nonlinear_iteration(self) -> None:
        super().before_nonlinear_iteration()
        # Update fracture locking variable. Positive values mean that the fractures are
        # not locked. Should always be zero during the first time step, then zero during
        # the first iteration, and then increase with the number of iterations.
        self.fracture_locking_variable.set_value(
            self.nonlinear_solver_statistics.num_iteration
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


# def lopsided_smooth_heaviside(zero_value: float, eps_pos: float = 1e-3, eps_neg: float = 1e-3, var: FloatType) -> FloatType:
#     r"""Lopsided smooth (regularized) version of the Heaviside function.

#     The analytical expression for the lopsided smooth version Heaviside function reads:
#         ``H_eps(x) = 1-exp(x/eps_pos) for ``x >= 0``,
#         ``H_eps(x) = exp(x/eps_neg) for ``x < 0``,
#     with its derivative smoothly approximating the Dirac delta function:
#         ``d(H(x))/dx = delta_eps = (1/eps_pos) * exp(-x/eps_pos) for ``x >= 0``,
#         ``d(H(x))/dx = delta_eps = (1/eps_neg) * exp(x/eps_neg) for ``x < 0``.
#     The special case of either ``eps_pos = 0`` or ``eps_neg = 0`` reverts to the
#     one-sided smooth Heaviside function, i.e.,
#         ``H_eps(x) = 1 for ``x >= 0, eps_pos = 0``, or
#         ``H_eps(x) = 0 for ``x < 0, eps_neg = 0``.

#     Parameters:
#         zero_value: Value of the Heaviside function at zero. Typically, this is
#             set to 0, 0.5 or 1.
#         eps_pos (optional): Regularization parameter for positive values. The function
#             will converge to the Heaviside function in the limit when ``eps_pos --> 0``.
#             The default is ``1e-3``.
#         eps_neg (optional): Regularization parameter for negative values. The function
#             will converge to the Heaviside function in the limit when ``eps_neg --> 0``.
#             The default is ``1e-3``.
#         var: Input array.

#     Returns:
#         Regularized heaviside function (and its Jacobian if applicable) in form of a
#         AdArray or ndarray (depending on the input).

#     """
#     if isinstance(var, AdArray):
#         val = np.zeros_like(var.val)
#         jac_vals = np.zeros_like(var.val)

#         pos_inds = var.val > 0
#         neg_inds = var.val < 0
#         zero_inds = np.isclose(var.val, 0.0)

#         # Positive values
#         if eps_pos > 0:
#             val[pos_inds] = 1 - np.exp(-var.val[pos_inds] / eps_pos)
#             jac_vals[pos_inds] = (1 / eps_pos) * np.exp(-var.val[pos_inds] / eps_pos)
#         else:
#             val[pos_inds] = 1.0
#             jac_vals[pos_inds] = 0.0

#         # Negative values
#         if eps_neg > 0:
#             val[neg_inds] = np.exp(var.val[neg_inds] / eps_neg)
#             jac_vals[neg_inds] = (1 / eps_neg) * np.exp(var.val[neg_inds] / eps_neg)
#         else:
#             val[neg_inds] = 0.0
#             jac_vals[neg_inds] = 0.0

#         # Zero values
#         val[zero_inds] = zero_value
#         jac_vals[zero_inds] = 0.0

#         jac = var._diagvec_mul_jac(jac_vals)
#         return AdArray(val, jac)
#     else:
#         val = np.zeros_like(var)
#         pos_inds = var > 0
#         neg_inds = var < 0
#         zero_inds = np.isclose(var, 0.0)

#         # Positive values
#         if eps_pos > 0:
#             val[pos_inds] = 1 - np.exp(-var[pos_inds] / eps_pos)
#         else:
#             val[pos_inds] = 1.0

#         # Negative values
#         if eps_neg > 0:
#             val[neg_inds] = np.exp(var[neg_inds] / eps_neg)
#         else:
#             val[neg_inds] = 0.0

#         # Zero values
#         val[zero_inds] = zero_value

#         return val


# def smooth_clip(min_val: float, max_val: float, eps: float = 1e-3, var: pp) -> FloatType:
#     """Smooth clipping function.

#     The smooth clipping function smoothly limits the input variable to be within
#     the range [min_val, max_val] using a regularization parameter eps.

#     Parameters:
#         min_val: Minimum value for clipping.
#         max_val: Maximum value for clipping.
#         eps (optional): Regularization parameter. The function will converge to
#             hard clipping in the limit when eps --> 0. The default is 1e-3.
#         var: Input array.

#     Returns:
#         Smoothly clipped function (and its Jacobian if applicable) in form of a
#         AdArray or ndarray (depending on the input).

#     """
#     ''' Helper function for smooth clipping '''
#     # The clipping is created as the product of two lopsided smooth Heaviside functions.
#     zero_value = 0.0  # Want H(0) = 0 for the endpoints.
#     # Smoothing on the positive side of both functions. I.e., the function smoothly
#     # approaches the value of var inside the interval [min_val, max_val].
#     eps_pos = eps
#     # One-sided for lower side of both functions. I.e., the function is exactly zero below min_val
#     eps_neg = 0.0

#     f_lower = pp.ad.Function(partial(lopsided_smooth_heaviside, zero_value, eps_pos, eps_neg), "heaviside_smooth_lower")
#     f_upper = pp.ad.Function(partial(lopsided_smooth_heaviside, zero_value, eps_pos, eps_neg), "heaviside_smooth_upper")
#     return
