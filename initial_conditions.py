from functools import partial

import numpy as np
import porepy as pp
from scipy.interpolate import interp1d


class InitialConditionFromDepth:
    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return self.hydrostatic_pressure(self.depth(sd.cell_centers))

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return self.temperature_at_depth(self.depth(sd.cell_centers))


class CopyInitialCondition:
    """Copy the initial conditions from the parent class.

    This is used to copy the initial conditions from the parent class to the child class.
    """

    def boundary_displacement_from_initialization(
        self, boundary_grid: pp.BoundaryGrid
    ) -> np.ndarray:
        """Displacement from initialization model.

        Parameters:
            boundary_grid: Boundary grid for which to get displacement values.

        Returns:
            Array with displacement values. Shape (nd * num_cells,).

        """
        sds = [self.find_initialization_grid(boundary_grid.parent)]
        mod = self.initialization_model
        u = mod.displacement(sds)

        # Discretization
        discr = mod.stress_discretization(sds)
        # Boundary conditions
        bc = mod.combine_boundary_operators_mechanical_stress(
            subdomains=sds,
        )

        # Compute the pseudo-trace of the displacement
        # Note that this is not the real trace, as this only holds for particular
        # choices of boundary condtions
        u_faces_ad = (
            discr.bound_displacement_cell() @ u + discr.bound_displacement_face() @ bc
        )
        if hasattr(self, "pressure"):
            # Add contribution from pressure
            p = mod.pressure(sds).perturbation_from_reference()
            u_faces_ad += discr.bound_pressure(mod.darcy_keyword) @ p
        if hasattr(self, "temperature"):
            # Add contribution from temperature
            T = mod.temperature(sds).perturbation_from_reference()
            u_faces_ad += discr.bound_pressure(mod.enthalpy_keyword) @ T
        u_faces = mod.equation_system.evaluate(u_faces_ad)
        assert isinstance(u_faces, np.ndarray)
        return boundary_grid.projection(nd=self.nd) @ u_faces.ravel("F")

    def well_related_domain(self, g: pp.Grid) -> bool:
        """Check if the grid is related to a well.

        Parameters:
            g: Grid to check.

        Returns:
            True if the grid is related to a well, False otherwise.
        """
        if isinstance(g, pp.MortarGrid):
            # Skip mortar domains if their primary subdomain is a well
            sd = self.mdg.interface_to_subdomain_pair(g)[0]
        else:
            sd = g
        if self.is_well_grid(sd):
            return True
        elif sd.tags.get("parent_well_index", -1) > -1:
            return True
        elif sd.dim == 0:
            return True
        return False

    def find_initialization_grid(self, g: pp.Grid) -> pp.Grid:
        """Find the corresponding grid in the initialization model.

        Parameters:
            g: Grid in the current model.

        Returns:
            Corresponding grid in the initialization model.

        Raises:
            ValueError: If the grid is not found in the initialization model.
        """

        def check(other):
            if other.num_cells != g.num_cells:
                return False
            return np.allclose(other.cell_centers, g.cell_centers, atol=1e-5)

        if isinstance(g, pp.MortarGrid):
            for intf in self.initialization_model.mdg.interfaces():
                if check(intf):
                    return intf
        else:
            for sd in self.initialization_model.mdg.subdomains():
                if check(sd):
                    return sd
        raise ValueError(
            f"Subdomain {sd} not found in initialization model {self.initialization_model}"
        )

    def copy_initial_conditions(self, sd: pp.Grid, variable: str) -> np.ndarray:
        """Copy initial conditions from the initialization model.

        Parameters:
            sd: Subdomain to copy from.
            variable: Variable to copy.

        Returns:
            Array with initial condition values.

        """
        g = self.find_initialization_grid(sd)
        # Get the variable values from the initialization model
        variables = self.initialization_model.equation_system.get_variables(
            [variable], [g]
        )
        return self.initialization_model.equation_system.get_variable_values(
            variables, time_step_index=0
        )

    def interpolate_initial_condition(
        self, sd: pp.Grid, variable: str, fallback_vals: np.ndarray
    ) -> np.ndarray:
        """Interpolate initial conditions as a function of depth from all grids of the
        initialization model.

        Parameters:
            sd: Subdomain to copy from.
            variable: Variable to copy.
        Returns:
            Array with initial condition values.

        """
        im = self.initialization_model
        var = im.equation_system.get_variables([variable])
        vals = np.hstack([v.value(im.equation_system) for v in var])
        depths = np.hstack([im.depth(v.domain.cell_centers) for v in var])
        depths_sorted_indices = np.argsort(depths)
        depths_sorted = depths[depths_sorted_indices]
        vals_sorted = vals[depths_sorted_indices]
        # Remove duplicates
        unique_depths, unique_indices = np.unique(depths_sorted, return_index=True)
        depths_sorted = unique_depths
        vals_sorted = vals_sorted[unique_indices]
        # Interpolation function
        f = interp1d(
            depths_sorted,
            vals_sorted,
            kind="linear",
            fill_value="extrapolate",
        )
        depths_sd = self.depth(sd.cell_centers)
        val = f(depths_sd)
        # For depths outside the range of the initialization model, use the fallback values.
        mask_lower = depths_sd < depths_sorted.min()
        mask_upper = depths_sd > depths_sorted.max()
        val[mask_lower] = fallback_vals[mask_lower]
        val[mask_upper] = fallback_vals[mask_upper]
        return val

    @property
    def use_ic_interpolation(self) -> bool:
        return self.params.get("use_ic_interpolation", False)

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        if self.well_related_domain(sd):
            vals = self.hydrostatic_pressure(self.depth(sd.cell_centers))
            if self.use_ic_interpolation:
                vals = self.interpolate_initial_condition(sd, "pressure", vals)
            return vals
        else:
            return self.copy_initial_conditions(sd, "pressure")

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        return self.copy_initial_conditions(sd, "u")

    def ic_values_contact_traction(self, sd: pp.Grid) -> np.ndarray:
        return self.copy_initial_conditions(sd, "contact_traction")

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        if self.well_related_domain(sd):
            vals = self.temperature_at_depth(self.depth(sd.cell_centers))
            if self.use_ic_interpolation:
                vals = self.interpolate_initial_condition(sd, "temperature", vals)
            return vals
        else:
            return self.copy_initial_conditions(sd, "temperature")

    def ic_values_interface_fourier_flux(self, sd: pp.Grid) -> np.ndarray:
        if self.well_related_domain(sd):
            return super().ic_values_interface_fourier_flux(sd)
        else:
            return self.copy_initial_conditions(sd, "interface_fourier_flux")

    def ic_values_interface_enthalpy_flux(self, sd: pp.Grid) -> np.ndarray:
        if self.well_related_domain(sd):
            return super().ic_values_interface_enthalpy_flux(sd)
        else:
            return self.copy_initial_conditions(sd, "interface_enthalpy_flux")

    def ic_values_interface_darcy_flux(self, sd: pp.Grid) -> np.ndarray:
        if self.well_related_domain(sd):
            return super().ic_values_interface_darcy_flux(sd)
        else:
            return self.copy_initial_conditions(sd, "interface_darcy_flux")

    def ic_values_interface_displacement(self, sd: pp.Grid) -> np.ndarray:
        return self.copy_initial_conditions(sd, "u_interface")

    def reference_displacement_jump(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Reference displacement jump [m].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise reference displacement jump.

        """
        vals = []
        model = self.initialization_model
        for sd in subdomains:
            other = self.find_initialization_grid(sd)
            vals.append(
                model.plastic_displacement_jump([other]).value(model.equation_system)
            )
        return pp.ad.DenseArray(np.hstack(vals), name="reference_displacement_jump")

    def shear_dilation_gap(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Shear dilation [m].

        Parameters:
            subdomains: List of fracture subdomains.

        Returns:
            Cell-wise shear dilation.

        """
        angle: pp.ad.Operator = self.dilation_angle(subdomains)
        f_norm = pp.ad.Function(
            partial(pp.ad.functions.l2_norm, self.nd - 1), "norm_function"
        )
        f_tan = pp.ad.Function(pp.ad.functions.tan, "tan_function")
        shear_dilation: pp.ad.Operator = f_tan(angle) * f_norm(
            self.tangential_component(subdomains)
            @ (
                self.plastic_displacement_jump(subdomains)
                - self.reference_displacement_jump(subdomains)
            )
        )

        shear_dilation.set_name("shear_dilation")
        return shear_dilation
