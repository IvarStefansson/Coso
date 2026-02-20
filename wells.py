import porepy as pp
import numpy as np
import pandas as pd
import sys


class _WellDataBase:
    def parent_well(self, sd: pp.Grid) -> pp.Well:
        """Get the parent well of a well subdomain.

        Parameters:
            sd: Subdomain for which to get the parent well.

        Returns:
            The parent well of the subdomain.

        """
        if sd.dim == 1 and "parent_well_index" in sd.tags:
            return self.well_network.wells[sd.tags["parent_well_index"]]
        return None

    def wells_active(self) -> bool:
        return (
            self.time_manager.time
            > self.time_manager.schedule[self.well_protocol_offset()]
        )

    def well_protocol_index(self, data) -> int:
        """Get the index of the well protocol.

        The index is offset by two compared to the csv files. One for the header row,
        and one because Python indexing starts at time zero.

        Parameters:
            data: Well data.

        Returns:
            The index of the well protocol.
        """
        # The schedule is 0, start_time, end day 0, end day 1, end day 2, ...
        # Find the index of the current time in the schedule.
        return int(
            np.searchsorted(np.arange(data.shape[0]) * pp.DAY, self.time_manager.time)
            - self.well_protocol_offset()
            - 1
        )

    def well_protocol_offset(self) -> int:
        """Get the offset of the well protocol.

        Returns:
            The offset of the well protocol.
        """
        return 0

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

    @property
    def well_names(self) -> list:
        """List of well names."""
        return self.injection_well_names + self.production_well_names

    @property
    def injection_well_names(self) -> list:
        """List of injection well names."""
        return ["68-20RD"]

    @property
    def production_well_names(self) -> list:
        """List of production well names."""
        return ["16A-20", "16B-20"]


class WellDataCoso(_WellDataBase):
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

    def open_well_cells(self, subdomains) -> pp.ad.Operator:
        """Open well cells in the given subdomains.

        Parameters:
            subdomains: Subdomains where to open well cells.
        """
        z = {
            "68-20RD": np.array([0.603, 1.970]),
            "16A-20": np.array([2.1, 2.961]),
            "16B-20": np.array([2.484, 3.196]),
        }
        z["16B-20"] -= 0.2  # Ad hoc adjustment
        z = {name: self.units.convert_units(-z * pp.KILO, "m") for name, z in z.items()}
        all_vals = []
        for sd in subdomains:
            well = self.parent_well(sd)
            assert well is not None, f"No well found for subdomain {sd}"
            z_vals = z[well.tags["well_name"]]
            ind = np.zeros(sd.num_cells, dtype=int)
            where = np.logical_and(
                sd.cell_centers[2, :] <= z_vals[0], sd.cell_centers[2, :] >= z_vals[1]
            )
            ind[where] = 1
            sd.tags["open_well_cells"] = ind
            all_vals.append(ind)
        return pp.ad.DenseArray(np.hstack(all_vals))


class WellDataConceptual(_WellDataBase):
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

    @property
    def injection_well_names(self) -> list:
        """List of injection well names."""
        return ["1 Injection well"]  # 68-20RD

    @property
    def production_well_names(self) -> list:
        """List of production well names."""
        return ["2 Production well"]  # "16A-20"

    def open_well_cells(self, subdomains) -> pp.ad.Operator:
        """Open well cells in the given subdomains.

        Parameters:
            subdomains: Subdomains where to open well cells.
        """
        if len(subdomains) == 0:
            return pp.ad.DenseArray(np.array([]), "open_well_cells")
        all_vals = []
        for sd in subdomains:
            if not self.is_well_grid(sd):
                vals = np.ones(sd.num_cells, dtype=int)
            else:
                vals = self.set_open_well_cells(sd)
            all_vals.append(vals)
        all_vals = np.hstack(all_vals)
        return pp.ad.DenseArray(all_vals, "open_well_cells")

    # def update_time_dependent_ad_arrays(self) -> None:
    #     """Update time-dependent ad arrays."""
    #     super().update_time_dependent_ad_arrays()
    #     for sd, data in self.mdg.subdomains(return_data=True):
    #         if not self.is_well_grid(sd):
    #             data["open_well_cells"] = np.ones(sd.num_cells, dtype=int)
    #         else:
    #             data["open_well_cells"] = self.set_open_well_cells(sd)

    def set_open_well_cells(self, sd) -> np.ndarray:
        """Open well cells in the given subdomains.

        Parameters:
            subdomains: Subdomains where to open well cells.
        """
        z = {
            "1 Injection well": np.array([-5.0, 0.0]),
        }
        z = {name: self.units.convert_units(z * pp.KILO, "m") for name, z in z.items()}
        well = self.parent_well(sd)
        assert well is not None, f"No well found for subdomain {sd}"
        ind = np.ones(sd.num_cells, dtype=int)
        if "Injection" in well.tags["well_name"]:  # == "16A-20":
            coord_vals = z[well.tags["well_name"]]
            axis = 1
            where = np.logical_and(
                sd.cell_centers[axis, :] <= coord_vals[0],
                sd.cell_centers[axis, :] >= coord_vals[1],
            )
            ind[where] = 0
        return ind

    # def well_flux_equation(self, interfaces: list[pp.MortarGrid]) -> pp.ad.Operator:
    #     """Equation relating the well flux to the difference between well and formation
    #     pressure.

    #     For details, see Lie: An introduction to reservoir simulation using MATLAB/GNU
    #     Octave, 2019, Section 4.3.

    #     Parameters:
    #         interfaces: List of interfaces where the well fluxes are defined.

    #     Returns:
    #         Cell-wise well flux operator, units [kg * m^{nd-1} * s^-2].

    #     """
    #     eq = super().well_flux_equation(interfaces)
    #     subdomains = self.interfaces_to_subdomains(interfaces)
    #     projection = pp.ad.MortarProjections(self.mdg, subdomains, interfaces)
    #     ind = projection.primary_to_mortar_avg() @ self.open_well_cells(subdomains)
    #     new_eq = self.well_flux(interfaces) * (pp.ad.Scalar(1) - ind) + eq * ind
    #     new_eq.set_name("well_flux_equation")

    #     return new_eq
