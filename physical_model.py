from typing import Callable

import numpy as np
import porepy as pp
from scipy.optimize import minimize_scalar


def fit_thermal_expansion(
    depth_max: float,
    temperature_at_depth: Callable[[np.ndarray], np.ndarray],
    density: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rho_ref: float,
    T_ref: float,
    p_ref: float = 0.0,
    n_points: int = 2000,
    c_T_bounds: tuple[float, float] = (1e-5, 5e-3),
) -> float:
    r"""Fit an effective thermal-expansion coefficient :math:`c_T` for an exponential
    fluid-density law over a given depth range.

    The exponential law used by
    :class:`~porepy.models.fluid_property_library.FluidDensityFromPressureAndTemperature`
    is

    .. math::

        \rho(T, p) = \rho_{\rm ref}
                     \exp\!\bigl[c_p(p - p_{\rm ref}) - c_T(T - T_{\rm ref})\bigr]

    A constant :math:`c_T` cannot reproduce the strongly nonlinear variation of water
    density with temperature.  This function finds the single value of :math:`c_T` that
    minimises the depth-weighted mean-squared error between the exponential law and a
    reference density function ``density(T, p)`` sampled uniformly along the depth
    column.

    Depth-weighting (uniform sampling in depth rather than temperature) is important
    here because the steep shallow geotherm (e.g. 150 mK/m) concentrates many
    temperature values near the surface, which would otherwise bias the fit toward the
    low-temperature end of the range.

    Only the thermal part of the exponential is fitted.  The pressure part
    :math:`c_p(p - p_{\rm ref})` is removed from the reference density before fitting
    by evaluating ``density`` at the fixed reference pressure ``p_ref``, so the result
    is independent of the compressibility.

    All arguments must use **consistent units** (SI or model units, but not mixed).

    Parameters:
        depth_max: Maximum depth [m] over which to sample the geotherm.
        temperature_at_depth: Callable ``T = f(depths)`` returning a 1-D array of
            temperatures at the supplied depth array (shape ``(n,)``).  Must accept
            and return values in the same unit system as ``rho_ref`` and ``T_ref``.
        density: Callable ``rho = f(T, p)`` returning fluid density from 1-D arrays
            of temperature and pressure.  Should correspond to the reference density
            model (e.g. IAPWS-IF97 lookup table or the Kroll polynomial).
        rho_ref: Reference density :math:`\rho_0` at ``(T_ref, p_ref)`` [kg/m³], i.e.
            the prefactor in the exponential law.
        T_ref: Reference temperature :math:`T_{\rm ref}` [K].
        p_ref: Reference pressure at which ``density`` is evaluated for fitting.
            Defaults to 0 (gauge pressure), which removes the compressibility
            contribution and isolates the thermal part of the fit.
        n_points: Number of evenly-spaced depth samples used for fitting.  Higher
            values improve accuracy at the cost of more ``density`` evaluations.
            Defaults to 2000.
        c_T_bounds: Search interval ``(c_T_min, c_T_max)`` [K⁻¹] passed to the
            bounded scalar minimiser.  Defaults to ``(1e-5, 5e-3)``, which covers
            the range from near-incompressible (cold) to strongly expansive (hot)
            water.

    Returns:
        Depth-weighted optimal :math:`c_T` [K⁻¹].

    Raises:
        ValueError: If ``c_T_bounds[0] >= c_T_bounds[1]`` or ``n_points < 2``.

    Notes:
        - The fit minimises :math:`\sum_i [\rho_{\rm ref} \exp(-c_T (T_i - T_{\rm ref}))
          - \rho_{\rm ref}(T_i, p_{\rm ref})]^2` using
          :func:`scipy.optimize.minimize_scalar` with Brent's method.
        - For highly nonlinear temperature profiles the optimal :math:`c_T` may
          overestimate density near the surface and underestimate it at depth (or
          vice versa); inspect the residuals if accuracy across the full range is
          critical.

    Example:
        Fit :math:`c_T` for a bilinear Coso geotherm and an IAPWS-derived lookup
        table::

            import numpy as np

            T_tab = np.array([20, 60, 100, 140, 180, 220, 240]) + 273.15
            rho_tab = np.array([998.2, 983.2, 958.4, 926.1, 886.9, 840.3, 815.0])

            def T_profile(depths):
                return np.where(depths < 1100,
                                293.15 + 0.150 * depths,
                                293.15 + 0.150 * 1100 + 0.020 * (depths - 1100))

            def rho_lookup(T, p):
                return np.interp(T, T_tab, rho_tab)

            rho_ref = float(np.interp(323.15, T_tab, rho_tab))  # at 50 °C
            c_T = fit_thermal_expansion(
                depth_max=4000.0,
                temperature_at_depth=T_profile,
                density=rho_lookup,
                rho_ref=rho_ref,
                T_ref=323.15,
            )
    """
    if c_T_bounds[0] >= c_T_bounds[1]:
        raise ValueError(f"c_T_bounds must satisfy lower < upper, got {c_T_bounds}.")
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}.")

    depths = np.linspace(0.0, depth_max, n_points)
    T_profile = temperature_at_depth(depths)
    # Reference density at each depth, evaluated at p_ref to isolate the thermal
    # contribution (pressure part cancels because p == p_ref everywhere).
    p_profile = np.full_like(T_profile, p_ref)
    rho_reference = density(T_profile, p_profile)

    def _mse(c_T: float) -> float:
        rho_exp = rho_ref * np.exp(-c_T * (T_profile - T_ref))
        return float(np.mean((rho_exp - rho_reference) ** 2))

    result = minimize_scalar(_mse, bounds=c_T_bounds, method="bounded")
    return float(result.x)  # type: ignore[union-attr]


class HeterogeneousWellRadius:
    def grid_aperture(self, grid: pp.Grid) -> np.ndarray:
        """Get the aperture of a single grid.

        Parameters:
            grid: Grid for which to compute the aperture.

        Returns:
            Aperture for each cell in the grid.

        """
        aperture = np.ones(grid.num_cells)
        if grid.dim < self.nd:
            if self.is_well_grid(grid):
                # This is a well. The aperture is the well radius.
                aperture *= self.well_radius([grid]).value(self.equation_system)
            else:
                aperture = self.solid.residual_aperture * aperture
        else:
            # For the matrix, the aperture is one, but needs to be scaled by the
            # length units.
            aperture = self.units.convert_units(aperture, "m")
        return aperture


class PolynomialFluidDensityNp:
    """Mixin providing a polynomial numpy density for use in hydrostatic integration.

    Implements the Kroll et al. polynomial:

    .. math::

        \\rho_0(T) = 1000 - 0.07\\,\\Delta T - 0.0002\\,\\Delta T^2 \\;\\text{[kg/m³]}

    corrected for pressure compressibility as
    :math:`\\rho = \\rho_0(T)\\,[1 + c_p(p - p_{\\rm atm})]`.

    This is the form used in the (currently disabled) ``___density_of_phase``
    implementation.  Mix in instead of :class:`CosoBackgroundValues` to switch
    the hydrostatic integration to the polynomial law.
    """

    def _fluid_density_np(self, T: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Polynomial fluid density as a function of temperature and pressure.

        Parameters:
            T: Temperature array in model units [K].
            p: Pressure array in model units [Pa].

        Returns:
            Fluid density array in model units [kg/m³].
        """
        dT = T - pp.Celsius_to_Kelvin(20)
        rho_0 = (
            self.units.convert_units(1000.0, "kg*m^-3")
            - self.units.convert_units(0.07, "kg*m^-3*K^-1") * dT
            - self.units.convert_units(0.0002, "kg*m^-3*K^-2") * dT**2
        )
        p_ref = self.units.convert_units(pp.ATMOSPHERIC_PRESSURE, "Pa")
        c = self.fluid.reference_component.compressibility
        return rho_0 * (1.0 + c * (p - p_ref))


class CosoBackgroundValues:
    def _fluid_density_np(self, T: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Fluid density as a function of temperature and pressure (numpy version).

        Matches ``pp.constitutive_laws.FluidDensityFromPressureAndTemperature``, the
        law used by ``pp.Thermoporomechanics``:

        .. math::

            \\rho(T, p) = \\rho_0 \\exp\\!\\left[c_p(p - p_{\\rm ref})
                          - c_T(T - T_{\\rm ref})\\right]

        The material constants are taken from ``self.fluid.reference_component``; the
        reference state is read from ``self.reference_variable_values`` (already in
        model units).

        Parameters:
            T: Temperature array in model units [K].
            p: Pressure array in model units [Pa].

        Returns:
            Fluid density array in model units [kg/m³].
        """
        rho_0 = self.fluid.reference_component.density
        c_p = self.fluid.reference_component.compressibility
        c_T = self.fluid.reference_component.thermal_expansion
        p_ref = self.reference_variable_values.pressure
        T_ref = self.reference_variable_values.temperature
        return rho_0 * np.exp(c_p * (p - p_ref) - c_T * (T - T_ref))

    def hydrostatic_pressure(self, depth: np.ndarray) -> np.ndarray:
        r"""Compute hydrostatic pressure by integrating :math:`dp/dz = \rho(T,p)\,g`.

        Temperature at depth is obtained from :meth:`temperature_at_depth` and fluid
        density is evaluated as a function of both *T* and *p* via
        :meth:`_fluid_density_np`, making the integration self-consistent.

        Parameters:
            depth: Array of depths [m] at which to compute hydrostatic pressure.

        Returns:
            Array of hydrostatic pressure values at the given depths.
        """
        # Delegate to parent (constant-density formula) for empty depth arrays; avoids
        # corner cases in max() and solve_ivp below.
        super_val = super().hydrostatic_pressure(depth)
        if depth.size == 0:
            return super_val

        from scipy.integrate import solve_ivp

        # Gravity magnitude in model units [model-m * s^-2].
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")
        # Atmospheric pressure in model units: initial condition p(z=0) = p_atm.
        p_surface = self.units.convert_units(pp.ATMOSPHERIC_PRESSURE, "Pa")

        # Integrate from the surface (z=0) down to the deepest requested point.
        z_max = float(np.max(depth))
        # All points are at the surface; return p_atm everywhere.
        if z_max == 0.0:
            return np.full_like(depth, p_surface, dtype=float)

        def rhs(z: float, p_vec: list) -> list:
            # Temperature at the current depth, from the piecewise-linear geotherm.
            T = self.temperature_at_depth(np.array([z]))[0]
            # Density at (T, p): matches FluidDensityFromPressureAndTemperature used by
            # the AD model, so the hydrostatic state is self-consistent with the solver.
            rho = self._fluid_density_np(np.array([T]), np.array([p_vec[0]]))[0]
            # dp/dz = rho * g  (z positive downward, p increases with depth).
            return [rho * g]

        sol = solve_ivp(
            rhs,
            t_span=(0.0, z_max),  # integrate from surface to maximum depth
            y0=[p_surface],  # initial condition: atmospheric pressure at z=0
            method="RK45",  # explicit Runge-Kutta of order 4(5)
            dense_output=True,  # build a continuous solution for arbitrary depths
            rtol=1e-6,  # relative tolerance
            atol=self.units.convert_units(1e0, "Pa"),  # absolute tolerance: 1 Pa
        )
        # for z1 in [1600, 1800, 2000]:

        #     def rhs_in(z: float, p_vec: list) -> list:
        #         # Temperature at the current depth, from the piecewise-linear geotherm.
        #         T = pp.Celsius_to_Kelvin(
        #             40
        #         )  #  self.params[f"{self.injection_well_names[0]}_temperature"][0]
        #         # Density at (T, p): matches FluidDensityFromPressureAndTemperature used by
        #         # the AD model, so the hydrostatic state is self-consistent with the solver.
        #         rho = self._fluid_density_np(np.array([T]), np.array([p_vec[0]]))[0]
        #         # dp/dz = rho * g  (z positive downward, p increases with depth).
        #         return [rho * g]

        #     sol_in = solve_ivp(
        #         rhs_in,
        #         t_span=(0.0, z1),  # integrate from surface to maximum depth
        #         y0=[p_surface],  # initial condition: atmospheric pressure at z=0
        #         method="RK45",  # explicit Runge-Kutta of order 4(5)
        #         dense_output=True,  # build a continuous solution for arbitrary depths
        #         rtol=1e-6,  # relative tolerance
        #         atol=self.units.convert_units(1e0, "Pa"),  # absolute tolerance: 1 Pa
        #     )
        #     p_est = self.units.convert_units(sol.sol(z1)[0], "Pa", to_si=True)
        #     p_in_est = self.units.convert_units(sol_in.sol(z1)[0], "Pa", to_si=True)
        #     print(
        #         f"Comparing hydrostatic pressure at depth {z1:.0f} m for polynomial and constant-density laws:"
        #     )
        #     print(f"Depth {z1:.0f} m: p(polynomial) = {p_est:.2e} Pa")
        #     print(
        #         f"Depth {z1:.0f} m: p(constant rho injection) = "
        #         f"{p_in_est:.2e} Pa. Difference = "
        #         f"{p_est - p_in_est:.2e} Pa"
        #     )
        #     for z0 in np.arange(1600, 2200, 200):
        #         # Do some comparisons.
        #         def rhs0(z: float, p_vec: list) -> list:
        #             # Temperature at the current depth, from the piecewise-linear geotherm.
        #             T = self.temperature_at_depth(np.array([z0]))[0]
        #             # Density at (T, p): matches FluidDensityFromPressureAndTemperature used by
        #             # the AD model, so the hydrostatic state is self-consistent with the solver.
        #             rho = self._fluid_density_np(np.array([T]), np.array([p_vec[0]]))[0]
        #             # dp/dz = rho * g  (z positive downward, p increases with depth).
        #             return [rho * g]

        #         sol0 = solve_ivp(
        #             rhs0,
        #             t_span=(0.0, z0),  # integrate from surface to maximum depth
        #             y0=[p_surface],  # initial condition: atmospheric pressure at z=0
        #             method="RK45",  # explicit Runge-Kutta of order 4(5)
        #             dense_output=True,  # build a continuous solution for arbitrary depths
        #             rtol=1e-6,  # relative tolerance
        #             atol=self.units.convert_units(
        #                 1e0, "Pa"
        #             ),  # absolute tolerance: 1 Pa
        #         )
        #         p_0 = self.units.convert_units(sol0.sol(z1)[0], "Pa", to_si=True)

        #         print(
        #             f"p(const rho z={z0:.0f} m) = {p_0:.2e} Pa, difference = {p_est - p_0:.2e} Pa"
        #         )

        # Evaluate the continuous solution at every requested depth in one vectorised
        # call; sol.sol(depth) has shape (1, n), so take row 0.
        return sol.sol(depth)[0]

    def temperature_at_depth(self, depth: np.ndarray) -> np.ndarray:
        """Compute temperature at given depths.

        Parameters:
            depth: Array of depths at which to compute temperature.

        Returns:
            Array of temperature values at the given depths.
        """
        # Steep gradient of 245 K/km first 1.1 km, then 5.6 K/km below that, according
        # to Kroll et al.
        gradient1 = self.units.convert_units(245e-3, "K*m^-1")
        gradient2 = self.units.convert_units(5.6e-3, "K*m^-1")
        #         The Feedback Between Stress, Faulting, and Fluid Flow:
        #  Lessons from the Coso Geothermal Field, CA, USA
        # Nicholas C. Davatzes1
        #  and Stephen H. Hickman2
        gradient1 = self.units.convert_units(150e-3, "K*m^-1")
        gradient2 = self.units.convert_units(20e-3, "K*m^-1")
        temperature = np.where(
            depth < 1100,
            self.surface_temperature + gradient1 * depth,
            self.surface_temperature + gradient1 * 1100 + gradient2 * (depth - 1100),
        )
        return temperature

    def displacement_from_depth(self, coords: np.ndarray) -> np.ndarray:
        """Depth-dependent displacement.

        The x and y displacements are proportional to the distance from the center of
        the domain in the xy-plane and to depth. The z displacement is proportional to
        the depth.

        Parameters:
            coords ``shape=(3, num_cells)``: Coordinates.

        Returns:
            Array with displacement values. Shape (nd, num_cells).

        """
        # Displacement gradient set to .1 mm/km.
        # For the moment, we use a constant gradient in each direction.
        gradient = (
            self.parameters.get(
                "lithostatic_stress_multipliers", np.array([1.0, 1.0, 1.0])
            )
            * (
                self.solid.density * (1 - self.solid.porosity)
                + self.fluid.reference_component.density * self.solid.porosity
            )
            * pp.GRAVITY_ACCELERATION
            / self.bulk_modulus(None)._value
        )
        # compute from center of domain in the xy-plane
        box = self.domain.bounding_box
        center = np.array([box["xmin"] + box["xmax"], box["ymin"] + box["ymax"]]) / 2
        # Normalise by box size in the xy-plane

        distances = (center[:, np.newaxis] - coords[:2, :]) / np.array(
            [box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]]
        )[:, np.newaxis]
        # Pad with one for the z-coordinate
        distances = np.vstack([distances, np.ones_like(distances[0])])
        # Compute product of gradient, distances and depth
        return gradient[:, np.newaxis] * distances * self.depth(coords)

    def displacement_from_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Displacement from coordinates.

        The x and y displacements are proportional to the distance from the center of
        the domain in the xy-plane. The z displacement is proportional to the depth.

        Parameters:
            coords ``shape=(3, num_cells)``: Coordinates.

        Returns:
            Array with displacement values. Shape (nd, num_cells).

        """
        # if (

        # ):
        #     # If the time is before the well protocol offset, set displacement to zero.
        #     return np.zeros_like(coords)
        values = self.displacement_from_depth(coords)
        # Set velocity for all cells. The displacement is scaled with the x-coordinate
        # and time.
        if self.params["use_wells"]:
            x_scaling = coords[0] - self.domain.bounding_box["xmin"]
            offset_time = 20 * pp.YEAR if self.time_data.time > 0 else 0
            values += np.outer(self.boundary_displacement_velocity, x_scaling) * (
                self.time_data.time + offset_time
            )
        return values


class HeterogeneousPermeabilitySpecification:
    def matrix_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability [m^2].

        Parameters:
            subdomains: List of subdomains where the permeability is defined.

        Returns:
            Cell-wise permeability operator [m^2].

        """
        if not self.params.get("heterogeneous_permeability", False):
            return super().matrix_permeability(subdomains)
        if len(subdomains) == 0:
            return pp.wrap_as_dense_ad_array(0, size=0)
        permeability = self.permeability_vals(subdomains)
        return self.isotropic_second_order_tensor(subdomains, permeability)

    def permeability_vals(self, subdomains: list[pp.Grid]) -> np.ndarray:
        cc = np.hstack([sd.cell_centers for sd in subdomains])
        size = sum(sd.num_cells for sd in subdomains)
        vals = self.solid.permeability * np.ones(size)
        high_perm_zones = self._high_perm_zones(cc)
        # Set permeability in the high permeability zones.
        vals[high_perm_zones] *= 2000
        permeability = pp.wrap_as_dense_ad_array(
            vals, size, name="heterogeneous_permeability"
        )
        return permeability

    def _high_perm_zones(self, cc: np.ndarray) -> np.ndarray:
        """Identify high permeability zones. Set to central box of size 1/3 of domain.

        Parameters:
            cc: Cell centers.

        Returns:
            Boolean array indicating which cells are in the high permeability zones.
        """
        dx, dy, dz = self.domain_sizes()
        caprock_depth = self.caprock_depth()
        reservoir_depth = self.reservoir_depth()
        # zone_x = np.logical_and(cc[0, :] > dx / 3, cc[0, :] < 2 * dx / 3)
        # zone_y = np.logical_and(cc[1, :] > dy / 3, cc[1, :] < 2 * dy / 3)
        zone_z = np.logical_and(-cc[2, :] > caprock_depth, -cc[2, :] < reservoir_depth)
        return zone_z
        # return np.logical_and.reduce((zone_x, zone_y, zone_z))


class HeterogeneousFrictionCoefficient:
    """Mixin giving each fracture its own (back-computed) friction coefficient.

    Follows the same pattern as :class:`HeterogeneousPermeabilitySpecification`
    above: a plain mixin sharing ``self`` with the model, gated by
    ``params["heterogeneous_friction_coefficient"]``. Expects
    ``params["friction_coefficients"]`` to be a ``{str(frac_num): value}`` mapping
    covering every fracture, e.g. from
    :func:`run_example_3.compute_friction_coefficients_each_fracture`.
    """

    def friction_coefficient(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Friction coefficient.

        Parameters:
            subdomains: List of subdomains where the friction coefficient is defined.

        Returns:
            Cell-wise friction coefficient operator.

        """
        if not self.params.get("heterogeneous_friction_coefficient", False):
            return super().friction_coefficient(subdomains)
        if len(subdomains) == 0:
            return pp.wrap_as_dense_ad_array(0, size=0)
        friction_coefficients = self.params["friction_coefficients"]
        vals = []
        for sd in subdomains:
            key = str(sd.frac_num)
            if key not in friction_coefficients:
                raise KeyError(
                    f"No heterogeneous friction coefficient found for fracture "
                    f"frac_num={sd.frac_num}. 'friction_coefficients' must cover "
                    "every fracture subdomain passed to friction_coefficient()."
                )
            vals.append(np.full(sd.num_cells, friction_coefficients[key]))
        return pp.wrap_as_dense_ad_array(
            np.hstack(vals), name="heterogeneous_friction_coefficient"
        )


class FluidExtensions:
    def viscosity_of_phase(
        self, phase: pp.Phase[pp.FluidComponent]
    ) -> pp.compositional.compositional_mixins.ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a viscosity exponential law for the fluid's phase.

        .. math::
            \\mu = 2.414e-5 * 10^(247.8 / (T - 140))


        The reference viscosity and the pressure coefficient are taken from the material
        constants of the reference component, while the reference pressure is accessible
        by mixin; a typical implementation will provide this in a variable class.

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def mu(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            A = pp.ad.Scalar(
                self.units.convert_units(2.414e-5, "Pa * s"),
                "reference_fluid_viscosity",
            )
            B = pp.ad.Scalar(
                self.units.convert_units(247.8, "K"),
                "temperature_coefficient_for_viscosity",
            )
            C = pp.ad.Scalar(
                self.units.convert_units(140, "K"), "temperature_offset_for_viscosity"
            )
            T = self.temperature(domains)
            mu_ = A * pp.ad.Scalar(10, "exponent_base") ** (B / (T - C))
            return mu_

        return mu

    def ___density_of_phase(self, phase: pp.Phase) -> pp.ExtendedDomainFunctionType:
        """Mixin method for :class:`~porepy.compositional.compositional_mixins.
        FluidMixin` to provide a density exponential law for the fluid's phase.

        .. math::
            \\rho = \\rho_0 \\exp \\left[ c_p \\left(p - p_0\\right) \\right]

        The reference density and the compressibility are taken from the material
        constants of the reference component, while the reference pressure is accessible
        by mixin; a typical implementation will provide this in a variable class.

        Parameters:
            phase: The single fluid phase.

        Returns:
            A function representing above expression on some domains.

        """

        def rho(domains: pp.SubdomainsOrBoundaries) -> pp.ad.Operator:
            temperature = self.temperature(domains) - pp.ad.Scalar(
                pp.Celsius_to_Kelvin(20), "temperature_for_density"
            )
            rho_ref = (
                pp.ad.Scalar(
                    self.units.convert_units(1000, "kg*m^-3"), "reference_fluid_density"
                )
                - pp.ad.Scalar(
                    self.units.convert_units(0.07, "kg*m^-3*K^-1"),
                    "temperature_coefficient_for_density",
                )
                * temperature
                - pp.ad.Scalar(
                    self.units.convert_units(0.0002, "kg*m^-3*K^-2"),
                    "temperature_squared_coefficient_for_density",
                )
                * temperature
                * temperature
            )
            dp = self.pressure(domains).perturbation_from_reference()

            # Wrap compressibility from fluid class as matrix (left multiplication with dp).
            c = self.fluid_compressibility(domains)
            rho_ = rho_ref * (pp.ad.Scalar(1) + c * dp)
            rho_.set_name("fluid_density_from_pressure_and_temperature")
            return rho_

        return rho


class HagenPoiseuilleWellPermeability:
    """Mixin that replaces the Hagen-Poiseuille permeability for well subdomains.

    For 1-D well grids the permeability is set according to the Hagen-Poiseuille
    law for laminar pipe flow:

    .. math::
        k = r_w^2 / 8

    where :math:`r_w` is the well radius (``self.solid.well_radius``). All other
    intersection subdomains are handled by the parent class.

    """

    def intersection_permeability(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Permeability for intersection subdomains [m^2].

        Well subdomains receive the Hagen-Poiseuille permeability; all others fall
        back to the parent implementation. The two contributions are assembled via
        subdomain projections, following the same pattern as
        :meth:`~porepy.models.constitutive_laws.DimensionDependentPermeability.
        permeability`.

        Parameters:
            subdomains: Intersection subdomains (dimension < nd - 1).

        Returns:
            Cell-wise permeability operator [m^2].

        """
        if len(subdomains) == 0:
            return super().intersection_permeability(subdomains)

        projection = pp.ad.SubdomainProjections(subdomains, dim=9)
        wells = [sd for sd in subdomains if self.is_well_grid(sd)]
        non_wells = [sd for sd in subdomains if not self.is_well_grid(sd)]

        # solid.well_radius is already in model units (same convention as
        # solid.permeability used in ConstantPermeability).
        r_w = self.solid.well_radius
        well_perm_scalar = r_w**2 / 8.0

        permeability = pp.wrap_as_dense_ad_array(
            0,
            size=sum(sd.num_cells for sd in subdomains)
            * 9,  # 9 for second-order tensor in 3D
            name="intersection_permeability",
        )
        if len(wells) > 0:
            well_size = sum(sd.num_cells for sd in wells)
            well_perm = pp.wrap_as_dense_ad_array(
                well_perm_scalar,
                size=well_size,
                name="hagen_poiseuille_well_permeability",
            )
            well_tensor = self.isotropic_second_order_tensor(wells, well_perm)
            permeability = (
                permeability + projection.cell_prolongation(wells) @ well_tensor
            )
        if len(non_wells) > 0:
            permeability = permeability + projection.cell_prolongation(
                non_wells
            ) @ super().intersection_permeability(non_wells)
        return permeability


class PhysicalModel(
    pp.constitutive_laws.GravityForce,
    # pp.SinglePhaseFlow,
    pp.Thermoporomechanics,
    # pp.Poromechanics,
    # pp.MomentumBalance,
):
    """Model for the Coso geothermal reservoir."""

    def matrix_porosity(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Porosity [-].

        Parameters:
            subdomains: List of subdomains where the porosity is defined.

        Returns:
            Cell-wise porosity operator [-].

        """
        # Inherit poromechanical porosity from base class.
        phi = super().matrix_porosity(subdomains)
        f_max = pp.ad.Function(pp.ad.maximum, "maximum_function")
        f_max.set_name("max for porosity")
        eps = 1e-8
        lower_bounded = f_max(phi, pp.ad.Scalar(eps))
        m = pp.ad.Scalar(-1)
        both_bounded = m * f_max(phi * m, pp.ad.Scalar(eps - 1))
        return both_bounded
