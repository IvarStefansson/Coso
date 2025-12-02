import logging
from typing import Callable, Sequence
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import porepy as pp
import os


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CosoExporter:
    evaluate_and_scale: Callable[
        [Sequence[pp.Grid] | Sequence[pp.MortarGrid], str, str], np.ndarray
    ]

    def data_to_export(self) -> list:
        """Returns data for exporting.

        Returns:
            A list of tuples (subdomain, variable name, variable values).
        """
        data = super().data_to_export()
        sds = self.mdg.subdomains(dim=self.nd - 1)
        cell_offsets = np.cumsum([0] + [sd.num_cells * self.nd for sd in sds])
        displacement_jump = self.evaluate_and_scale(sds, "displacement_jump", "m")
        char = self.evaluate_and_scale(sds, "characteristic_contact_traction", "Pa")
        traction = self.evaluate_and_scale(sds, "contact_traction", "-")
        # Loop over the fracture subdomains
        for id, sd in enumerate(sds):
            # Export the displacement jump.
            data.append(
                (
                    sd,
                    "displacement_jump",
                    displacement_jump[cell_offsets[id] : cell_offsets[id + 1]],
                )
            )
            # Export the slip tendency, defined as the ratio of the shear traction to
            # the normal traction.
            traction_loc = traction[cell_offsets[id] : cell_offsets[id + 1]].reshape(
                (3, -1), order="F"
            )
            zero_inds = np.isclose(traction_loc[-1], 0)
            traction_loc[-1, zero_inds] = -1
            traction_loc[0, zero_inds] = 1
            slip_tendency = np.linalg.norm(traction_loc[:-1], axis=0) / np.abs(
                traction_loc[-1]
            )
            data.append((sd, "slip_tendency", slip_tendency))

            data.append((sd, "traction", traction_loc.ravel("F") * char))
            # residual = self.equation_system.assemble(evaluate_jacobian=False)
            # data.append((sd, "residual", residual))

        for sd in self.mdg.subdomains(dim=1):
            if self.parent_well(sd) is not None:
                if sd.tags.get("open_well_cells", False) is False:
                    self.open_well_cells([sd])
                data.append((sd, "open_well_cells", sd.tags["open_well_cells"]))
        return data

    def collect_data(self) -> dict:
        """Collect data for well monitoring.

        Returns:
            A dictionary with well names as keys and another dictionary as value.
            The inner dictionary has variable names as keys and their values.
        """
        data = {}
        sds = self.mdg.subdomains(dim=self.nd - 2)
        face_offsets = np.cumsum([0] + [sd.num_faces for sd in sds])
        pressure = self.evaluate_and_scale(sds, "pressure_trace", "Pa")
        has_t = hasattr(self, "temperature_trace")
        if has_t:
            temperature = self.evaluate_and_scale(sds, "temperature_trace", "K")
            enthalpy_f = self.evaluate_and_scale(sds, "enthalpy_flux", "W * m^-2")

        darcy_f = self.evaluate_and_scale(sds, "darcy_flux", "m^3 *s ^-1")
        fluid_f = self.evaluate_and_scale(sds, "fluid_flux", "kg * s^ -1")
        for id, sd in enumerate(sds):
            if not self.is_well(sd):
                # Skip non-well subdomains
                continue
            top_faces = self.domain_boundary_sides(sd).top
            if sum(top_faces) == 0:
                # Skip subdomains without a top face
                continue
            parent_well = self.parent_well(sd)
            # Get the face normals for the top face. We stay consistent with the pp
            # convention of having positive outward fluxes.
            flux_sign = np.sign(sd.face_normals[-1, top_faces])

            data[parent_well.tags["well_name"]] = {
                "pressure": pressure[face_offsets[id] : face_offsets[id + 1]][
                    top_faces
                ][0],
                "darcy_flux": darcy_f[face_offsets[id] : face_offsets[id + 1]][
                    top_faces
                ][0]
                * flux_sign,
                "fluid_flux": (
                    fluid_f[face_offsets[id] : face_offsets[id + 1]][top_faces][0]
                    * flux_sign
                ),
            }
            if has_t:
                data[parent_well.tags["well_name"]].update(
                    {
                        "temperature": temperature[
                            face_offsets[id] : face_offsets[id + 1]
                        ][top_faces][0],
                        "enthalpy_flux": enthalpy_f[
                            face_offsets[id] : face_offsets[id + 1]
                        ][top_faces][0]
                        * flux_sign,
                    }
                )

        return data

    def save_results(self) -> None:
        """Save results to a file.

        The results are saved in a CSV file in the specified folder. The file contains
        time, well name, and the monitored variables."""
        records = []
        for time, data in zip(self.time_manager.schedule[1:], self.results):
            for well_name, variables in data.items():
                record = {"time": time, "well_name": well_name}
                record.update(variables)
                records.append(record)
        df = pd.DataFrame(records)
        folder_name = self.params["data_folder_name"] + "/well_monitoring"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        df.to_csv(
            f"{folder_name}/{self.params['file_name']}.csv",
            index=False,
        )
        self.well_monitoring_data = df

    def plot_well_monitoring(self) -> None:
        """Plot well monitoring data for a given well."""
        if not hasattr(self, "well_monitoring_data"):
            self.save_results()
        well_names = self.injection_well_names + self.production_well_names
        df = self.well_monitoring_data
        if hasattr(self, "temperature_trace"):
            variables = (("pressure", "temperature"), ("enthalpy_flux", "fluid_flux"))
            units = ((r"Pa", r"K"), (r"W/m$^2$", r"kg/s"))
        else:
            variables = (("pressure",), ("fluid_flux",))
            units = ((r"Pa",), (r"kg/s",))
        for name in well_names:
            # Get the data for the specified well
            well_data = df[df["well_name"] == name]
            if well_data.empty:
                raise ValueError(f"No data found for well {name}.")
            # Plot pressure, temperature, and fluxes over time
            for var_pair, unit_pair in zip(variables, units):
                fig, ax1 = plt.subplots()

                # First variable on left y-axis
                color1 = "tab:blue"
                ax1.set_xlabel("Time (d)")
                # Replace underscore with space for better readability and capitalize
                # the first letter of the variable name.
                var_name0 = var_pair[0].replace("_", " ").capitalize()
                ax1.set_ylabel(f"{var_name0} ({unit_pair[0]})", color=color1)
                ax1.plot(
                    well_data["time"] / pp.DAY,
                    well_data[var_pair[0]],
                    color=color1,
                    label=var_pair[0],
                )
                ax1.tick_params(axis="y", labelcolor=color1)
                prefix = (
                    "injection" if name in self.injection_well_names else "production"
                )
                suffix = "_fluxes" if "flux" in var_pair[0] else ""
                plt_name = f"{prefix}_{name}"

                if len(var_pair) == 1:
                    plt.title(f"Well Data for {plt_name}")
                    plt.legend()
                    plt.savefig(
                        f"{self.params['data_folder_name']}/well_monitoring/"
                        + f"{self.params['file_name']}_{plt_name}{suffix}.png"
                    )
                    continue
                # Second variable on right y-axis
                ax2 = ax1.twinx()
                color2 = "tab:red"
                ax2.set_ylabel(var_pair[1], color=color2)
                ax2.plot(
                    well_data["time"] / pp.DAY,
                    well_data[var_pair[1]],
                    color=color2,
                    label=var_pair[1],
                )
                ax2.tick_params(axis="y", labelcolor=color2)
                var_name1 = var_pair[1].replace("_", " ").capitalize()
                ax2.set_ylabel(f"{var_name1} ({unit_pair[1]})")
                # Add legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, [var_name0, var_name1], loc="best")

                # plt.tight_layout()
                suffix = "_fluxes" if "flux" in var_pair[1] else ""

                plt.title(f"Well Data for {plt_name}")
                plt.savefig(
                    f"{self.params['data_folder_name']}/well_monitoring/"
                    + f"{self.params['file_name']}_{plt_name}{suffix}.png"
                )


class IterationExporting:
    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        # Setting export_constants_separately to False facilitates operations such as
        # filtering by dimension in ParaView and is done here for illustrative purposes.
        self.iteration_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"] + "_iterations",
            folder_name=self.params["folder_name"],
            export_constants_separately=False,
        )

        geometry_exporter = pp.Exporter(
            self.mdg,
            file_name=self.params["file_name"],
            folder_name=self.params["folder_name"] + "/geometry",
            export_constants_separately=False,
        )
        geometry_exporter.write_vtu(time_dependent=False)

    def prepare_simulation(self):
        """Prepare simulation.

        This method is called before the simulation starts. It initializes the
        iteration exporter and writes the initial geometry to a vtu file.

        """
        self.ad_time_step_factor = pp.ad.Scalar(1.0)  # Value adapted elsewhere
        super().prepare_simulation()
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()

    def data_to_export_iteration(self):
        """Returns data for iteration exporting.

        Returns:
            Any type compatible with data argument of pp.Exporter().write_vtu().

        """
        # The following is a slightly modified copy of the method
        # data_to_export() from DataSavingMixin.
        return self.data_to_export()

    def save_data_iteration(self):
        """Export current solution to vtu files.

        This method is typically called by after_nonlinear_iteration.

        Having a separate exporter for iterations avoids distinguishing between
        iterations and time steps in the regular exporter's history (used for
        export_pvd).

        """
        # To make sure the nonlinear iteration index does not interfere with the
        # time part, we multiply the latter by the next power of ten above the
        # maximum number of nonlinear iterations. Default value set to 10 in
        # accordance with the default value used in NewtonSolver
        n = self.params.get("max_iterations", 10)
        p = round(np.log10(n))
        r = 10**p
        if r <= n:
            r = 10 ** (p + 1)
        self.iteration_exporter.write_vtu(
            self.data_to_export_iteration(),
            time_dependent=True,
            time_step=self.nonlinear_solver_statistics.num_iteration
            + r * self.time_manager.time_index,
        )

    def after_nonlinear_iteration(self, solution_vector: np.ndarray) -> None:
        """Integrate iteration export into simulation workflow.

        Order of operations is important, super call distributes the solution to
        iterate subdictionary.

        """
        super().after_nonlinear_iteration(solution_vector)
        self.save_data_iteration()
        self.iteration_exporter.write_pvd()
