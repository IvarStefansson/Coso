from typing import cast

import numpy as np
import porepy as pp


class CosoBackgroundValues:
    def hydrostatic_pressure(self, depth: np.ndarray) -> np.ndarray:
        r"""Compute hydrostatic pressure at given depths.

        According to Exploring Whether Subsurface Fluid Production Can Minimize
        Triggered Seismicity in Geothermal Fields K. A. Kroll, H. Wu, P. Fu, the
        pressure gradient is 7.19 MPa/km

        Parameters:
            depth: Array of depths at which to compute hydrostatic pressure.

        Returns:
            Array of hydrostatic pressure values at the given depths.

        """
        gradient = self.units.convert_units(7.19e3, "Pa/m")
        pressure = gradient * depth + self.units.convert_units(
            pp.ATMOSPHERIC_PRESSURE, units="Pa"
        )
        return pressure

    def temperature_at_depth(self, depth: np.ndarray) -> np.ndarray:
        """Compute temperature at given depths.

        Parameters:
            depth: Array of depths at which to compute temperature.

        Returns:
            Array of temperature values at the given depths.
        """
        # Steep gradient of 245 K/km first 1.1 km, then 5.6 K/km below that, according
        # to Kroll et al.
        surface_temperature = self.params.get(
            "surface_temperature", self.units.convert_units(20, "Celsius")
        )
        gradient1 = self.units.convert_units(245, "K/km")
        gradient2 = self.units.convert_units(5.6, "K/km")
        temperature = np.where(
            depth < 1100,
            surface_temperature + gradient1 * depth,
            surface_temperature + gradient1 * 1100 + gradient2 * (depth - 1100),
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
            offset_time = 20 * pp.YEAR if self.time_manager.time > 0 else 0
            values += np.outer(self.boundary_displacement_velocity, x_scaling) * (
                self.time_manager.time + offset_time
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

    def caprock_depth(self) -> float:
        """Depth of the caprock.

        Returns:
            Depth of the caprock in meters.
        """
        return self.params.get("caprock_depth", 1.1e3)

    def reservoir_depth(self) -> float:
        """Depth of the reservoir.

        Returns:
            Depth of the reservoir in meters.
        """
        return self.params.get("reservoir_depth", 3e3)


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
            dp = self.perturbation_from_reference("pressure", domains)

            # Wrap compressibility from fluid class as matrix (left multiplication with dp).
            c = self.fluid_compressibility(domains)
            rho_ = rho_ref * (pp.ad.Scalar(1) + c * dp)
            rho_.set_name("fluid_density_from_pressure_and_temperature")
            return rho_

        return rho


class PhysicalModel(
    # CosoBackgroundValues,
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
