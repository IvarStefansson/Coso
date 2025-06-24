from typing import Callable
import numpy as np
import porepy as pp
import pandas as pd
import sys


class CosoBoundaryConditions(pp.PorePyModel):
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

        # TODO: Replace by initial condition, which can be pre-computed to match
        # background stress.
        values = self.displacement_from_coordinates(boundary_grid.cell_centers)

        return values.ravel("F")

    def bc_type_mechanics(self, sd: pp.Grid) -> pp.BoundaryConditionVectorial:
        """Boundary condition type for mechanics.

        Dirichlet boundary conditions are defined on all boundaries except the top.

        Parameters:
            sd: Subdomain for which to define boundary conditions.

        Returns:
            bc: Boundary condition object.

        """
        domain_sides = self.domain_boundary_sides(sd)
        dir_sides = (
            domain_sides.bottom
            + domain_sides.east
            + domain_sides.west
            + domain_sides.south
            + domain_sides.north
        )
        dir_sides = domain_sides.all_bf
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
        in the line of sight (LOS) to each satellite, within an area of size~ 450 km2,
        at the locations of hundreds of thousands permanent and distributed scatterers.
        Thirty descending (satellite moves north to south) and 45 ascending (south to
        north) images were used from Envisat, and 63 descending and 65 ascending from
        Sentinel. A decomposition into average vertical and east horizontal components
        is also performed in more than 35,000 100-m pixels where both types of LOS
        measurements are available. The main observations at CGF include:(1) a
        subsidence area of size~ 70 km2, with a maximum subsidence of–27.6 mm/year for
        the Envisat period and lower maximum subsidence of–19.1 mm/year for the Sentinel
        period;(2) eastward movements in the western part of the subsidence area, with
        Envisat maximum of+ 23.9 mm/year and a lower Sentinel maximum of+ 15.9
        mm/year;(3) westward displacements in the eastern part of the subsidence area,
        with Envisat maximum of ̶ 14.2 mm/year and Sentinel maximum of–11.9 mm/year;(4)
        very good agreement of the InSAR observations with leveling survey data;(5)
        earthquake clusters in the subsidence area and hypocentral cross-sections
        showing clusters at various depths and migration in time; and (6) good
        predictions of the overall geothermal resource, based on poroelastic modeling
        using both leveling and InSAR data. The ultimate goal of the project is to
        provide geothermal operators with tools that can be used in reservoir
        management."""
        # The relative displacement rate is approx. 24 + 14 mm/year.
        # The area of the Coso Geothermal Field is 450 km2, which is 450e6 m2, so the
        # displacement rate per unit length is 38e-3 m / sqrt(450e6 m/year.)
        rate = (
            38
            * pp.MILLI
            * pp.METER
            / (450 * (pp.KILO * pp.METER) ** 2) ** 0.5
            / pp.YEAR
        )
        return np.array([1.0, 1.0, 0.0]) * self.units.convert_units(rate, "s^-1")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Pressure boundary values.

        Parameters:
            boundary_grid: Boundary grid for which pressure values are to be returned.

        Returns:
            Array of pressure boundary values for each cell in the boundary grid.
        """
        if boundary_grid.dim == 2:
            return self.hydrostatic_pressure(boundary_grid.cell_centers)
        # Else, we are at the top of a well. The pressure is set to the well head
        # pressure.
        # Get the parent well of the subdomain.
        sd = boundary_grid.parent
        parent_well = self._parent_well(sd)
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
                self.well_protocol_index()
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
            return self.temperature_from_depth(boundary_grid.cell_centers)
        # Get the parent well of the subdomain.
        sd = boundary_grid.parent
        parent_well = self._parent_well(sd)
        vals = np.zeros(boundary_grid.num_cells)
        ind = self.domain_boundary_sides(boundary_grid).top
        if parent_well is None or not self.wells_active():
            # We are not at a well or not in the schedule yet.
            vals[ind] = self.reference_variable_values.temperature
        else:
            # Get the well data for the well.
            well_data = parent_well.data
            # Get the temperature for the well.
            vals[ind] = well_data["temperature"].values[self.well_protocol_index()]
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
        parent_well = self._parent_well(sd)
        if (
            self.is_injection_well(parent_well) or self.is_production_well(parent_well)
        ) and not self.wells_active():
            return pp.BoundaryCondition(sd, domain_sides.top, "neu")

        return pp.BoundaryCondition(sd, domain_sides.all_bf, "dir")

    def _bc_type_diffusion(self, sd: pp.Grid) -> pp.BoundaryCondition:
        domain_sides = self.domain_boundary_sides(sd)
        parent_well = self._parent_well(sd)
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

    def _parent_well(self, sd: pp.Grid) -> pp.Well:
        """Get the parent well of a well subdomain.

        Parameters:
            sd: Subdomain for which to get the parent well.

        Returns:
            The parent well of the subdomain.

        """
        if sd.dim == 1 and "parent_well_index" in sd.tags:
            return self.well_network.wells[sd.tags["parent_well_index"]]
        return None

    def read_well_data(self) -> None:
        """Read well data from file.

        This function reads the well data from the file and assigns it to the
        well_network attribute of the class.

        """
        if len(self.well_network.wells) == 0:
            return
        psig2Pa = 6894.76
        lb2kg = 0.45359237

        def farenheit2kelvin(farenheit: float) -> float:
            """Convert Fahrenheit to Kelvin."""
            return (farenheit - 32) * 5 / 9 + 273.15

        for well_type in ["injection", "production"]:
            pth = sys.path[0]
            fn = f"{pth}/Coso data/{well_type}.csv"
            # Read the CSV file into a DataFrame
            # Use the first row as the header (header=0)
            # Use the first column as the index (index_col=0)
            # Parse dates in the first column (parse_dates=[0])
            # Set the date format to day-month-year (dayfirst=True)
            data = pd.read_csv(fn, header=0)
            data["well_head_pressure"] = psig2Pa * data["WHP_psig_"]
            if well_type == "injection":
                key = "CumInj_24hr"
            else:
                key = "CumMass_24hr"
                data["p1_pressure"] = psig2Pa * data["P1_Prod"]
            # The 1e6 factor is because, for whatever reason, the mass rate is given in
            # 1e6 lb/day.
            data["mass_rate"] = lb2kg * data[key] / pp.DAY * 1e6
            data["temperature"] = farenheit2kelvin(data["Temp_F_"])

            data["Date"] = pd.to_datetime(data["Date"], format="%d-%b-%Y")
            # Set the date as the index
            data.set_index("Date", inplace=True)
            for well_name in getattr(self, f"{well_type}_well_names"):
                # Get the data for the well
                well_data = data[data["Wellname"] == well_name]
                # Get the well object from the well network
                wells = self.well_network.wells
                # Find the well whose "well_name" tag is "well_name"
                well = next(
                    (w for w in wells if w.tags["well_name"] == well_name), None
                )
                if well is None:
                    raise ValueError(f"Well {well_name} not found in well network.")
                well.data = well_data

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
        parent_well = self._parent_well(sd)
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
            mass_rate = well_data["mass_rate"].values[self.well_protocol_index()]
            # Get the Darcy flux values for the well.
            darcy_flux = mass_rate / self.equation_system.evaluate(
                self.advection_weight_mass_balance([boundary_grid])
            )
            # if self.well_protocol_index() < 2:
            #     # If we are before the first well protocol, we assume a zero flux.
            #     darcy_flux = np.zeros_like(darcy_flux)
            vals[neumann_sides] = -darcy_flux[neumann_sides]
        return vals

    def well_protocol_offset(self) -> int:
        """Get the offset of the well protocol.

        Returns:
            The offset of the well protocol.
        """
        return 1

    def wells_active(self) -> bool:
        return (
            self.time_manager.time
            > self.time_manager.schedule[self.well_protocol_offset()]
        )

    def well_protocol_index(self) -> int:
        """Get the index of the well protocol.

        Returns:
            The index of the well protocol.
        """
        # The schedule is 0, start_time, end day 0, end day 1, end day 2, ...
        # Find the index of the current time in the schedule.
        return int(
            np.searchsorted(self.time_manager.schedule, self.time_manager.time)
            - self.well_protocol_offset()
            - 1
        )

    def is_injection_well(self, well: pp.Well | None) -> bool:
        """Check if the well is an injection well.

        Parameters:
            well: Well object.

        Returns:
            True if the well is an injection well, False otherwise.
        """
        return well is not None and well.tags["well_name"] in self.injection_well_names

    def is_production_well(self, well: pp.Well | None) -> bool:
        """Check if the well is a production well.

        Parameters:
            well: Well object.

        Returns:
            True if the well is a production well, False otherwise.
        """
        return well is not None and well.tags["well_name"] in self.production_well_names
