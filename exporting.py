import logging
from typing import Callable, Sequence
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import porepy as pp
import os


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from porepy.applications.convergence_analysis import ConvergenceAnalysis


def plot_flow_rate_and_fracture_displacement(
    csv_dir: str = "example_4_saved_data/well_monitoring",
    well_name: str = "16A-20",
) -> None:
    """
    Plot fluid flux from production well and displacement jump of first two fractures.

    This function creates a plot with two y-axes:
    - Left axis: fluid_flux (kg/s) from the production well vs time
    - Right axis: displacement_jump (m) for the first two fractures vs time

    Parameters:
    -----------
    csv_dir : str
        Path to directory containing the CSV files
    well_name : str
        Name of the production well to plot (default: "16A-20")
    """
    # Read CSV files
    well_data = pd.read_csv(os.path.join(csv_dir, "example_4.csv"))
    fracture_data = pd.read_csv(os.path.join(csv_dir, "example_4_fractures.csv"))

    # Process well data
    well_data_filtered = well_data[well_data["well_name"] == well_name].copy()

    # Parse fluid_flux - it's stored as string like "[-0.]" or "[value]"
    well_data_filtered["fluid_flux_numeric"] = well_data_filtered["fluid_flux"].apply(
        lambda x: float(str(x).strip("[]"))
    )
    time = well_data_filtered["time"]
    # Process time data. Times may not be strictly monotonic. This is due to saving of
    # diverged time steps. We sort the data by time to ensure correct plotting.
    # Identify non-monotonic times.
    accepted_times = [time.iloc[0]]
    inds = [0]
    for i, t in enumerate(time.iloc[1:], start=1):
        if t > accepted_times[-1]:
            accepted_times.append(t)
            inds.append(i)
    time = pd.Series(accepted_times)
    well_data_filtered = well_data_filtered.iloc[inds].reset_index(drop=True)
    # Dynamically get the first two unique fracture IDs, preserving order of first occurrence
    fracture_ids = fracture_data["fracture_id"].unique()[:2].tolist()
    fracture_data_filtered = fracture_data[
        fracture_data["fracture_id"].isin(fracture_ids)
    ].copy()

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot fluid flux on left axis
    color1 = "tab:blue"
    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Fluid Flux (kg/s)", color=color1, fontsize=12)
    line1 = ax1.plot(
        time,
        well_data_filtered["fluid_flux_numeric"],
        color=color1,
        linewidth=2,
        label=f"Fluid Flux ({well_name})",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    # Create right axis for fracture displacement
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    color3 = "tab:green"
    ax2.set_ylabel("Displacement Jump (m)", fontsize=12)
    fracture_mapping = {fracture_ids[0]: "not connected", fracture_ids[1]: "connected"}
    # Plot displacement jump for each fracture
    for fracture_id, color in zip(fracture_ids, [color2, color3]):
        fracture_subset = fracture_data_filtered[
            fracture_data_filtered["fracture_id"] == fracture_id
        ].sort_values("time")
        fracture_subset = fracture_subset.iloc[inds].reset_index(drop=True)
        ax2.plot(
            time,
            fracture_subset["displacement_jump"],
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"Displacement Jump on fracture {fracture_id} ({fracture_mapping[fracture_id]})",
        )

    ax2.tick_params(axis="y")

    # Create combined legend
    lines1 = line1
    lines2 = ax2.get_lines()
    ax1.legend(
        lines1 + lines2,
        [l.get_label() for l in lines1 + lines2],
        loc="best",
        fontsize=10,
        framealpha=0.9,
        edgecolor="black",
    )

    plt.title(
        "Production Well Fluid Flux and Fracture Displacement Jump vs Time",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    # Save the plot
    output_path = os.path.join(csv_dir, "flow_rate_displacement_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    return fig


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
        for sd in self.mdg.subdomains():
            depth = self.depth(sd.cell_centers)
            data.append(
                (
                    sd,
                    "depth",
                    self.units.convert_units(depth, "m"),
                )
            )
            data.append(
                (
                    sd,
                    "hydrostatic_pressure",
                    self.units.convert_units(self.hydrostatic_pressure(depth), "Pa"),
                )
            )
        for sd in self.mdg.subdomains():
            if not self.is_well_grid(sd):
                continue
            if hasattr(self, "open_well_cells"):
                vals = self.evaluate_and_scale([sd], "open_well_cells", "-")
                data.append((sd, "open_well_cells", vals))

            well = self.parent_well(sd)
            well_tag = int(well.tags["well_name"][:2])
            data.append((sd, "well_name", np.full(sd.num_cells, well_tag)))
        return data

    #     sds = self.mdg.subdomains(dim=self.nd - 1)
    #     cell_offsets = np.cumsum([0] + [sd.num_cells * self.nd for sd in sds])
    #     displacement_jump = self.evaluate_and_scale(sds, "displacement_jump", "m")
    #     char = self.evaluate_and_scale(sds, "characteristic_contact_traction", "Pa")
    #     traction = self.evaluate_and_scale(sds, "contact_traction", "-")
    #     # Loop over the fracture subdomains
    #     for id, sd in enumerate(sds):
    #         # Export the displacement jump.
    #         data.append(
    #             (
    #                 sd,
    #                 "displacement_jump",
    #                 displacement_jump[cell_offsets[id] : cell_offsets[id + 1]],
    #             )
    #         )
    #         # Export the slip tendency, defined as the ratio of the shear traction to
    #         # the normal traction.
    #         traction_loc = traction[cell_offsets[id] : cell_offsets[id + 1]].reshape(
    #             (3, -1), order="F"
    #         )
    #         zero_inds = np.isclose(traction_loc[-1], 0)
    #         traction_loc[-1, zero_inds] = -1
    #         traction_loc[0, zero_inds] = 1
    #         slip_tendency = np.linalg.norm(traction_loc[:-1], axis=0) / np.abs(
    #             traction_loc[-1]
    #         )
    #         data.append((sd, "slip_tendency", slip_tendency))

    #         data.append((sd, "traction", traction_loc.ravel("F") * char))
    #         # residual = self.equation_system.assemble(evaluate_jacobian=False)
    #         # data.append((sd, "residual", residual))

    #     for sd in self.mdg.subdomains(dim=1):
    #         if self.parent_well(sd) is not None:
    #             if sd.tags.get("open_well_cells", False) is False:
    #                 self.open_well_cells([sd])
    #             data.append((sd, "open_well_cells", sd.tags["open_well_cells"]))
    #     return data

    def collect_data(self) -> dict:
        """Collect data for well monitoring.

        Returns:
            A dictionary with well names as keys and another dictionary as value.
            The inner dictionary has variable names as keys and their values.
        """
        if not hasattr(self, "_data_collection_times"):
            self._data_collection_times = []
        self._data_collection_times.append(self.time_manager.time)
        data = super().collect_data()
        if data is None:
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
            if not self.is_well_grid(sd):
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
        fracs = self.mdg.subdomains(dim=self.nd - 1)
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in fracs])
        displacement_jump = self.evaluate_and_scale(fracs, "displacement_jump", "m")
        for id, sd in enumerate(fracs):
            key = "fracture_" + str(id)
            data[key] = {
                "displacement_jump": ConvergenceAnalysis.lp_norm(
                    displacement_jump[cell_offsets[id] : cell_offsets[id + 1]],
                    fracs[id].cell_volumes,
                )
            }
        return data

    def save_results(self) -> None:
        """Save results to a file.

        The results are saved in a CSV file in the specified folder. The file contains
        time, well name, and the monitored variables."""
        well_records = []
        fracture_records = []
        for time, data in zip(self._data_collection_times, self.results):
            for name, variables in data.items():
                if name.startswith("fracture_"):
                    record = {"time": time, "fracture_id": name}
                    record.update(variables)
                    fracture_records.append(record)
                else:
                    record = {"time": time, "well_name": name}
                    record.update(variables)
                    well_records.append(record)
        df = pd.DataFrame(well_records)
        folder_name = self.params["data_folder_name"] + "/well_monitoring"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        df.to_csv(
            f"{folder_name}/{self.params['file_name']}.csv",
            index=False,
        )
        df_fractures = pd.DataFrame(fracture_records)
        df_fractures.to_csv(
            f"{folder_name}/{self.params['file_name']}_fractures.csv",
            index=False,
        )
        self.well_monitoring_data = df
        self.fracture_monitoring_data = df_fractures

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
        plot_flow_rate_and_fracture_displacement(
            csv_dir=f"{self.params['data_folder_name']}/well_monitoring",
            well_name=self.production_well_names[0],
        )


class GeometryExporting:
    def initialize_data_saving(self):
        """Initialize iteration exporter."""
        super().initialize_data_saving()
        # Setting export_constants_separately to False facilitates operations such as
        # filtering by dimension in ParaView and is done here for illustrative purposes.

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

    def save_data_time_step(self) -> None:
        super().save_data_time_step()
        t = self.time_manager.time  # current time
        scheduled = self.time_manager.schedule[1:]  # scheduled times except t_init
        if not any(np.isclose(t, scheduled)):  # Invert logic of super.
            collected_data = self.collect_data()
            if collected_data is not None:
                self.results.append(collected_data)


if __name__ == "__main__":
    plot_flow_rate_and_fracture_displacement()
