from typing import Callable
import numpy as np
import porepy as pp


class InitialCosoBoundaryConditions:
    """Boundary conditions for the Coso geothermal reservoir model.

    We impose a displacement value scaling linearly with x and time on all boundaries
    except the top, where we impose a zero normal traction.
    """

    hydrostatic_pressure: Callable[[np.ndarray], np.ndarray]

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Temperature boundary values.

        Parameters:
            boundary_grid: Boundary grid for which temperature values are to be returned.

        Returns:
            Array of temperature boundary values for each cell in the boundary grid.
        """
        return (
            self.temperature_at_depth(self.depth(boundary_grid.cell_centers))
            * self.initialization_time_dependency_factor()
        )

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure boundary values.

        Parameters:
            boundary_grid: Boundary grid for which pressure values are to be returned.

        Returns:
            Array of pressure boundary values for each cell in the boundary grid.
        """
        return (
            self.hydrostatic_pressure(self.depth(boundary_grid.cell_centers))
            * self.initialization_time_dependency_factor()
        )

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Displacement values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """

        #
        # if self.time_manager.time > 1e-5:
        #     values = self.displacement_from_coordinates(boundary_grid.cell_centers)
        # else:
        values = np.zeros((self.nd, boundary_grid.num_cells))
        return values.ravel("F") * self.initialization_time_dependency_factor()

    def initialization_time_dependency_factor(self) -> float:
        val = self.time_manager.time / self.time_manager.time_final
        self.ad_time_step_factor.set_value(val)
        return 1  # self.time_manager.time > 0


class CosoBoundaryConditionsDisplacement:
    """Boundary conditions for the Coso geothermal reservoir model.

    We impose a displacement value scaling linearly with x and time on all boundaries
    except the top, where we impose a zero normal traction.
    """

    hydrostatic_pressure: Callable[[np.ndarray], np.ndarray]

    def bc_values_displacement(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Displacement values.

        Parameters:
            boundary_grid: Boundary grid for which boundary values are to be returned.

        Returns:
            Array of boundary values, with one value for each dimension of the
                problem, for each face in the subdomain.

        """

        if boundary_grid.dim < self.nd - 1:
            return super().bc_values_displacement(boundary_grid)
        values = self.boundary_displacement_from_initialization(boundary_grid)
        # Set velocity for all cells. The displacement is scaled with the x-coordinate
        # and time.
        if self.params["use_wells"]:
            coords = boundary_grid.cell_centers
            x_scaling = coords[0] - self.domain.bounding_box["xmin"]
            y_scaling = coords[1] - self.domain.bounding_box["ymin"]

            offset_time = 0 * pp.YEAR if self.time_manager.time > 0 else 0
            vals_time = np.outer(self.boundary_displacement_velocity, y_scaling) * (
                self.time_manager.time + offset_time
            )
            values += vals_time.ravel("F")
        return values

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for mechanics.

        Dirichlet boundary conditions are defined on all boundaries except the top.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        dir_sides = self.domain_boundary_sides(sd).all_bf
        bc = pp.BoundaryConditionVectorial(sd, dir_sides, "dir")
        bc.internal_to_dirichlet(sd)
        return bc

    @property
    def boundary_displacement_velocity(self) -> np.ndarray:
        """Displacement velocity on the boundary [m*s^-1*m^-1].

        Returns:
            Array of displacement velocities of shape (nd,).
        """
        # Relative displacement rate according to the following paper: Satellite
        # observations of surface deformation at the Coso Geothermal Field, California
        """Surface deformation time series and rates are identified at the Coso
        Geothermal Field (CGF) and surrounding areas by applying interferometric
        synthetic aperture radar (InSAR) to satellite scenes from Envisat (June 2004 ̶
        October 2010) and Sentinel (November 2014–April 2018). The measurements are done
        in the line of sight (LOS) to each satellite, within an area of size ~450 km2,
        at the locations of hundreds of thousands permanent and distributed scatterers.
        Thirty descending (satellite moves north to south) and 45 ascending (south to
        north) images were used from Envisat, and 63 descending and 65 ascending from
        Sentinel. A decomposition into average vertical and east horizontal components
        is also performed in more than 35,000 100-m pixels where both types of LOS
        measurements are available. The main observations at CGF include: (1) a
        subsidence area of size ~70 km2, with a maximum subsidence of –27.6 mm/year for
        the Envisat period and lower maximum subsidence of –19.1 mm/year for the
        Sentinel period; (2) eastward movements in the western part of the subsidence
        area, with Envisat maximum of +23.9 mm/year and a lower Sentinel maximum of
        +15.9 mm/year; (3) westward displacements in the eastern part of the subsidence
        area, with Envisat maximum of ̶ 14.2 mm/year and Sentinel maximum of –11.9
        mm/year; (4) very good agreement of the InSAR observations with leveling survey
        data; (5) earthquake clusters in the subsidence area and hypocentral
        cross-sections showing clusters at various depths and migration in time; and (6)
        good predictions of the overall geothermal resource, based on poroelastic
        modeling using both leveling and InSAR data. The ultimate goal of the project is
        to provide geothermal operators with tools that can be used in reservoir
        management."""
        # The relative displacement rate is approx. 24 + 14 mm/year.
        # The side length of the Coso Geothermal Field is roughly 10 km.
        rate = (38 * pp.MILLI * pp.METER / (10 * (pp.KILO * pp.METER)) / pp.YEAR) * 10
        return np.array([0.0, -1.0, 0.0]) * self.units.convert_units(rate, "s^-1")


class CosoBoundaryConditions:
    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure boundary values.

        Parameters:
            boundary_grid: Boundary grid for which pressure values are to be returned.

        Returns:
            Array of pressure boundary values for each cell in the boundary grid.
        """
        if boundary_grid.dim == 2:
            return self.hydrostatic_pressure(self.depth(boundary_grid.cell_centers))
        # Else, we are at the top of a well. The pressure is set to the well head
        # pressure.
        # Get the parent well of the subdomain.
        sd = boundary_grid.parent
        parent_well = self.parent_well(sd)
        vals = np.zeros(boundary_grid.num_cells)
        domain_sides = self.domain_boundary_sides(boundary_grid)
        if parent_well is None or not self.wells_active():
            # We are not at a well or not in the schedule yet.
            vals[domain_sides.top] = self.reference_variable_values.pressure
        else:
            # Get the well data for the well.
            well_data = parent_well.data
            # Get the well head pressure for the well.
            vals[domain_sides.top] = well_data["well_head_pressure"].values[
                self.well_protocol_index(well_data)
            ]
        return vals

    def bc_values_temperature(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Temperature boundary values.

        Parameters:
            boundary_grid: Boundary grid for which temperature values are to be returned.

        Returns:
            Array of temperature boundary values for each cell in the boundary grid.
        """
        if boundary_grid.dim == 2:
            return self.temperature_at_depth(self.depth(boundary_grid.cell_centers))
        # Get the parent well of the subdomain.
        sd = boundary_grid.parent
        parent_well = self.parent_well(sd)
        vals = np.zeros(boundary_grid.num_cells)
        ind = self.domain_boundary_sides(boundary_grid).top
        if parent_well is None or not self.wells_active():
            # We are not at a well or not in the schedule yet.
            vals[ind] = self.reference_variable_values.temperature
        else:
            # Get the well data for the well.
            well_data = parent_well.data
            # Get the temperature for the well.
            vals[ind] = well_data["temperature"].values[
                self.well_protocol_index(well_data)
            ]
        return vals

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Darcy flux.

        For the 3d domain, Dirichlet conditions are defined on all boundaries. For
        injection wells, Neumann conditions are imposed. Dirichlet conditions are
        set on production wells.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        return self._bc_type_diffusion(sd)

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Boundary condition type for Fourier flux.

        For the 3d domain, Dirichlet conditions are defined on all boundaries. For
        injection wells, Neumann conditions are imposed. Dirichlet conditions are
        set on production wells.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        parent_well = self.parent_well(sd)
        if (
            self.is_injection_well(parent_well) or self.is_production_well(parent_well)
        ) and not self.wells_active():
            return pp.BoundaryCondition(sd, domain_sides.top, "neu")

        return pp.BoundaryCondition(sd, domain_sides.all_bf, "dir")

    def _bc_type_diffusion(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        parent_well = self.parent_well(sd)
        if self.is_injection_well(parent_well):
            # This is an injection well subdomain. It always has Neumann conditions,
            # albeit with different values depending on whether we are in the schedule
            # or not.
            neumann_sides = domain_sides.top
            return pp.BoundaryCondition(sd, neumann_sides, "neu")
        if self.is_production_well(parent_well) and not self.wells_active():
            # This is a production well subdomain, but we are not in the schedule
            # yet. Set (zero) Neumann conditions on the top boundary. If we are in the
            # schedule, we will set Dirichlet conditions on the top boundary.
            neumann_sides = domain_sides.top
            bc = pp.BoundaryCondition(sd, neumann_sides, "neu")
            return bc

        dirichlet_sides = domain_sides.all_bf
        bc = pp.BoundaryCondition(sd, dirichlet_sides, "dir")
        return bc

    def bc_values_darcy_flux(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Darcy flux boundary values.

        Parameters:
            boundary_grid: Boundary grid for which Darcy flux values are to be returned.

        Returns:
            Array of Darcy flux boundary values for each cell in the grid.
        """
        # Only need to set nonzero values for the injection wells.
        sd = boundary_grid.parent
        # Get the parent well of the subdomain.
        parent_well = self.parent_well(sd)
        vals = np.zeros(boundary_grid.num_cells)

        if self.is_injection_well(parent_well):
            domain_sides = self.domain_boundary_sides(boundary_grid)
            neumann_sides = domain_sides.top
            if not np.any(neumann_sides):
                return vals
            # Get the well data for the well.
            well_data = parent_well.data
            # Get the mass rate for the well.

            if not self.wells_active():
                return np.zeros(boundary_grid.num_cells)
            mass_rate = well_data["mass_rate"].values[
                self.well_protocol_index(well_data)
            ]
            # Get the Darcy flux values for the well.
            darcy_flux = mass_rate / self.equation_system.evaluate(
                self.advection_weight_mass_balance([boundary_grid])
            )
            # if self.well_protocol_index() < 2:
            #     # If we are before the first well protocol, we assume a zero flux.
            #     darcy_flux = np.zeros_like(darcy_flux)
            vals[neumann_sides] = -darcy_flux[neumann_sides]
        return vals


class NeumannWellBCsFromSchedule(pp.PorePyModel):
    """Class defining Neumann BCs on well grids during the first time interval."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Return Neumann BC for Darcy flux on well grids during first time interval.

        Parameters:
            sd: The subdomain for which to return the BC type.

        Returns:
            The boundary condition type for Darcy flux on the given subdomain.
        """
        if self.is_well_grid(sd) and self.neumann_bcs_active():
            # Before start of injection, impose Neumann BCs on well grids. A zero-flux
            # condition is imposed by default when no BC values are specified. The <=
            # comparison ensures that the BCs are kept as Neumann as long as the time
            # step is within the first time interval [0, t1], where t1 is the start time
            # of injection, consistent with the implicit time stepping employed in
            # PorePy.
            domain_sides = self.domain_boundary_sides(sd)
            inds = domain_sides.top if self.nd == 3 else domain_sides.north
            return pp.BoundaryCondition(sd, inds, "neu")
        else:
            return super().bc_type_darcy_flux(sd)  # type: ignore[misc]

    def bc_type_fourier_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Return Neumann BC for Fourier flux on well grids during first time interval.

        Parameters:
            sd: The subdomain for which to return the BC type.

        Returns:
            The boundary condition type for Fourier flux on the given subdomain.
        """
        if self.is_well_grid(sd) and self.neumann_bcs_active():
            # Before start of injection, impose Neumann BCs on well grids. A zero-flux
            # condition is imposed by default when no BC values are specified. The <=
            # comparison ensures that the BCs are kept as Neumann as long as the time
            # step is within the first time interval [0, t1], where t1 is the start time
            # of injection, consistent with the implicit time stepping employed in
            # PorePy.
            domain_sides = self.domain_boundary_sides(sd)
            inds = domain_sides.top if self.nd == 3 else domain_sides.north
            return pp.BoundaryCondition(sd, inds, "neu")
        else:
            return super().bc_type_fourier_flux(sd)  # type: ignore[misc]

    def neumann_bcs_active(self) -> bool:
        """Check if Neumann BCs on well grids are active.

        Returns:
            True if the current time is within the first time interval, False otherwise.
        """
        is_neumann = False
        if self.params.get("neumann_intervals") is not None:
            neumann_intervals = self.params["neumann_intervals"]
            current_time = self.time_manager.time
            for interval in neumann_intervals:
                if interval[0] < current_time <= interval[1]:
                    is_neumann = True
                    break
        return is_neumann
