import numpy as np
import porepy as pp


class InitialCondition(pp.PorePyModel):
    def initial_condition(self) -> None:
        """Set initial conditions for the model.
        This method sets the initial conditions for the model by reading well data and
        setting the initial values for pressure and displacement.

        """
        # First read input from the well data files. This is used when setting initial
        # BCs.
        self.read_well_data()
        super().initial_condition()

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        return self.hydrostatic_pressure(sd.cell_centers)

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        depth = self.displacement_from_depth(sd.cell_centers).ravel("F")
        coords = self.displacement_from_coordinates(sd.cell_centers).ravel("F")
        return depth  # + coords

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        return self.temperature_from_depth(sd.cell_centers)

    def temperature_from_depth(self, coords: np.ndarray) -> np.ndarray:
        """Depth-dependent temperature.

        Parameters:
            coords: Coordinates.

        Returns:
            Array with temperature values.

        """
        # Thermal gradient set to 73 K/km, https://gdr.openei.org/submissions/787
        # For the moment, we use a constant gradient.
        gradient = self.units.convert_units(7.3e-2, "K*m^-1")
        temperature = self.reference_variable_values.temperature
        return temperature + gradient * self.depth(coords)

    def displacement_from_depth(self, coords: np.ndarray) -> np.ndarray:
        """Depth-dependent displacement.

        The x and y displacements are proportional to the distance from the center of the domain
        in the xy-plane and to depth. The z displacement is proportional to the depth.

        Parameters:
            coords ``shape=(3, num_cells)``: Coordinates.

        Returns:
            Array with displacement values. Shape (nd, num_cells).

        """
        # Displacement gradient set to .1 mm/km.
        # For the moment, we use a constant gradient in each direction.
        gradient = (
            np.array([0.8, 1.2, 1.0])
            * self.solid.density
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

        The x and y displacements are proportional to the distance from the center of the domain
        in the xy-plane. The z displacement is proportional to the depth.

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


class CopyInitialCondition(InitialCondition):
    """Copy the initial conditions from the parent class.

    This is used to copy the initial conditions from the parent class to the child class.
    """

    def well_related_domain(self, d):
        if isinstance(d, pp.MortarGrid):
            # Skip mortar domains if their primary subdomain is a well
            g = self.mdg.interface_to_subdomain_pair(d)[0]
        else:
            g = d
        if self.is_well(g):
            # Skip well equations
            return True
        elif g.tags.get("parent_well_index", -1) > -1:
            # Skip well equations
            return True
        return False

    def copy_initial_conditions(self, sd: pp.Grid, variable: str) -> np.ndarray:
        def check(sd, g):
            if sd.num_cells != g.num_cells:
                return False
            return np.allclose(sd.cell_centers, g.cell_centers, atol=1e-5)

        if isinstance(sd, pp.MortarGrid):
            for g in self.initialization_model.mdg.interfaces():
                if check(sd, g):
                    found = True
                    break
        else:
            for g in self.initialization_model.mdg.subdomains():
                if check(sd, g):
                    found = True
                    break
        if not found:
            raise ValueError(
                f"Subdomain {sd} not found in initialization model {self.initialization_model}"
            )
        # Get the variable values from the initialization model
        variables = self.initialization_model.equation_system.get_variables(
            [variable], [g]
        )
        return self.initialization_model.equation_system.get_variable_values(
            variables, time_step_index=0
        )

    def ic_values_pressure(self, sd: pp.Grid) -> np.ndarray:
        if self.well_related_domain(sd):
            return self.hydrostatic_pressure(sd.cell_centers)
        else:
            return self.copy_initial_conditions(sd, "pressure")

    def ic_values_displacement(self, sd: pp.Grid) -> np.ndarray:
        return self.copy_initial_conditions(sd, "u")

    def ic_values_contact_traction(self, sd: pp.Grid) -> np.ndarray:
        return self.copy_initial_conditions(sd, "contact_traction")

    def ic_values_temperature(self, sd: pp.Grid) -> np.ndarray:
        if self.well_related_domain(sd):
            return self.temperature_from_depth(sd.cell_centers)
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
