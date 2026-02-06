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

DISPLACEMENT_JUMP_Y_MIN = 1e-10


def plot_flow_rate_and_fracture_displacement(
    csv_dir: str,
    well_name: str,
    file_base: str,
    fracture_names,
    title=None,
) -> None:
    """
    Plot fluid flux from production well and displacement jump of first two fractures.

    This function creates a plot with two y-axes:
    - Left axis: fluid_flux (kg/s) from the production well vs time
    - Right axis: displacement_jump (m) for the first two fractures vs time

    Parameters:
        csv_dir: Path to directory containing the CSV files.
        well_name: Name of the production well to plot (default: "16A-20").
    """
    # Read CSV files
    well_data = pd.read_csv(os.path.join(csv_dir, f"{file_base}.csv"))
    fracture_data = pd.read_csv(os.path.join(csv_dir, f"{file_base}_fractures.csv"))

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

    i0 = 1  # Initial condition does not correspond to a converged solution.
    accepted_times = [time.iloc[i0]]
    inds = [i0]
    for i, t in enumerate(time.iloc[i0:], start=i0):
        if t > accepted_times[-1]:
            accepted_times.append(t)
            inds.append(i)
    time = pd.Series(accepted_times)
    well_data_filtered = well_data_filtered.iloc[inds].reset_index(drop=True)
    # Dynamically get the first two unique fracture IDs, preserving order of first
    # occurrence. We pick out all but the first occurence, which corresponds to the
    # injection fracture.
    fracture_data_filtered = fracture_data[
        fracture_data["fracture_id"].isin(fracture_names)
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
        label="Fluid Flux in Production Well",
    )
    ax1.tick_params(axis="y", labelcolor=color1)

    # Create right axis for fracture displacement
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    color3 = "tab:green"
    color4 = "tab:purple"
    colors = [color2, color3]
    ax2.set_ylabel("Displacement Jump (m)", fontsize=12)
    if "example_4" in file_base:
        fracture_mapping = {
            fracture_names[0]: "connected",
            fracture_names[1]: "not connected",
        }
        if "long_well" in file_base:
            fracture_mapping[fracture_names[0]] = "connected"

    else:
        fracture_mapping = {
            fracture_names[0]: "conductive",
            fracture_names[1]: "blocking",
        }
        colors.append(color4)
    # Plot displacement jump for each fracture
    for fracture_name, color in zip(fracture_names, colors):
        fracture_subset = fracture_data_filtered[
            fracture_data_filtered["fracture_id"] == fracture_name
        ].sort_values("time")
        fracture_subset = fracture_subset.iloc[inds].reset_index(drop=True)
        ax2.semilogy(
            time,
            np.clip(
                fracture_subset["displacement_jump"],
                a_min=DISPLACEMENT_JUMP_Y_MIN,
                a_max=None,
            ),
            color=color,
            linewidth=2,
            markersize=4,
            label=f"Displacement Jump on {fracture_name}",  # ({fracture_mapping[fracture_name]})",
        )

    ax2.tick_params(axis="y")
    ax2.set_ylim(bottom=DISPLACEMENT_JUMP_Y_MIN)

    # Create combined legend
    lines1 = line1
    lines2 = ax2.get_lines()
    ax1.legend(
        lines1 + lines2,
        [l.get_label() for l in lines1 + lines2],
        loc="lower right",
        fontsize=10,
        framealpha=0.9,
        edgecolor="black",
    )
    if title is None:
        title = "Production Well Fluid Flux and Fracture Displacement Jump"
    plt.title(
        title,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    # Save the plot
    output_path = os.path.join(csv_dir, "flow_rate_and_displacement_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    return fig


def plot_fracture_displacement(
    csv_dir: str,
    file_base: str,
    title=None,
) -> None:
    """
    Plot displacement jump of fractures over time (for simulations without wells).

    This function creates a plot showing the displacement_jump (m) for fractures vs time.

    Parameters:
        csv_dir: Path to directory containing the CSV files.
        file_base: Base name for the CSV file (default: "example_3").
    """
    # Read CSV file for fractures
    fracture_data = pd.read_csv(os.path.join(csv_dir, f"{file_base}_fractures.csv"))

    # Get time data from fracture data
    time = fracture_data["time"].unique()
    time = pd.Series(np.sort(time))

    # Process time data to handle non-monotonic times (due to diverged time steps)
    # Initial condition does not correspond to a converged solution.
    i0 = 1
    accepted_times = [time.iloc[i0]]

    for i, t in enumerate(time.iloc[i0:], start=i0):
        if t > accepted_times[-1]:
            accepted_times.append(t)

    time = pd.Series(accepted_times)

    # Get unique fracture IDs, preserving order of first occurrence
    fracture_names = fracture_data["fracture_id"].unique().tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set up colors for different fractures
    color_map = {
        0: "tab:orange",
        1: "tab:green",
        2: "tab:purple",
        3: "tab:red",
    }

    # Determine fracture mapping based on file_base
    if "example_4" in file_base:
        fracture_mapping = {}
        for idx, fname in enumerate(fracture_names):
            if idx == 0:
                fracture_mapping[fname] = "connected"
            else:
                fracture_mapping[fname] = "not connected"
        if "long_well" in file_base and len(fracture_names) > 0:
            fracture_mapping[fracture_names[0]] = "connected"
    else:
        fracture_mapping = {}
        labels = ["conductive", "blocking"]
        for idx, fname in enumerate(fracture_names):
            if idx < len(labels):
                fracture_mapping[fname] = labels[idx]
            else:
                fracture_mapping[fname] = f"fracture_{idx}"

    # Plot displacement jump for each fracture
    for idx, fracture_name in enumerate(fracture_names):
        fracture_subset = fracture_data[
            fracture_data["fracture_id"] == fracture_name
        ].sort_values("time")

        # Filter to accepted times
        fracture_subset = fracture_subset[fracture_subset["time"].isin(accepted_times)]

        color = color_map.get(idx, f"C{idx}")
        ax.semilogy(
            fracture_subset["time"],
            np.clip(
                fracture_subset["displacement_jump"],
                a_min=DISPLACEMENT_JUMP_Y_MIN,
                a_max=None,
            ),
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=f"{fracture_name}",  # ({fracture_mapping.get(fracture_name, '')})",
        )

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Displacement Jump (m)", fontsize=12)
    ax.set_ylim(bottom=DISPLACEMENT_JUMP_Y_MIN)
    ax.legend(loc="best", fontsize=10, framealpha=0.9, edgecolor="black")
    ax.grid(True, alpha=0.3)
    if title is None:
        title = "Fracture Displacement Jump Over Time"

    plt.title(
        title,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()

    # Save the plot
    output_path = os.path.join(csv_dir, "fracture_displacement_plot.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    return fig


class CosoExporter:
    evaluate_and_scale: Callable[
        [Sequence[pp.Grid] | Sequence[pp.MortarGrid], str, str], np.ndarray
    ]

    def _face_scalar_to_cell_vector(
        self, sd: pp.Grid, face_scalar: np.ndarray
    ) -> np.ndarray:
        """Convert face-wise scalar to cell-wise averaged vector.

        Given a scalar value at each face and the face normals, compute a vector
        for each cell by averaging the normal-weighted scalars over the cell's faces.
        Area weighting is removed by dividing by face areas.

        Parameters:
            sd: Grid
            face_scalar: Scalar values at each face, shape (num_faces,)

        Returns:
            Cell-wise vector data, shape (sd.dim, sd.num_cells)
        """
        # Remove area weighting: scalar_vector = (scalar / face_areas) * face_normals
        scalar_unweighted = face_scalar / sd.face_areas
        face_vectors = sd.face_normals * scalar_unweighted  # Shape: (sd.dim, num_faces)
        f2c = sd.divergence(self.nd)
        f2c.data = np.abs(f2c.data)  # Face-to-cell connectivity matrix
        cell_vector = f2c @ face_vectors.ravel("F")
        # Infer number of faces from the csr matrix f2c
        faces_per_cell = np.diff(f2c.indptr)
        result = cell_vector / faces_per_cell
        return result

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
            if sd.dim == self.nd - 1:
                data.append(
                    (
                        sd,
                        "fracture_name",
                        np.full(sd.num_cells, sd.frac_num),
                    )
                )
            if hasattr(self, "hydrostatic_pressure"):
                data.append(
                    (
                        sd,
                        "hydrostatic_pressure",
                        self.units.convert_units(
                            self.hydrostatic_pressure(depth), "Pa"
                        ),
                    )
                )
            # Export face-wise fluxes as cell-averaged vectors
            if hasattr(self, "fluid_flux"):
                if sd.dim > 0:
                    flux = self.evaluate_and_scale(
                        [sd], "fluid_flux", "kg * s^-1 * m^-2"
                    )
                    cell_flux = self._face_scalar_to_cell_vector(sd, flux)
                else:
                    cell_flux = np.zeros((self.nd, sd.num_cells))
                data.append((sd, "fluid_flux_vector", cell_flux.ravel("F")))
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
        displacement_jump = self.evaluate_and_scale(
            fracs, "displacement_jump", "m"
        ).reshape((self.nd, -1), order="F")
        jump_norm = np.linalg.norm(displacement_jump, axis=0)
        for id, sd in enumerate(fracs):
            key = self.fracture_names()[sd.frac_num]
            data[key] = {
                "displacement_jump": ConvergenceAnalysis.lp_norm(
                    jump_norm[cell_offsets[id] : cell_offsets[id + 1]],
                    fracs[id].cell_volumes,
                    p=1,
                )
                / np.sum(fracs[id].cell_volumes),
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
                if name.startswith("Fracture"):
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
        if not self.params["use_wells"]:
            plot_fracture_displacement(
                csv_dir=f"{self.params['data_folder_name']}/well_monitoring",
                file_base=self.params["file_name"],
                title=self.create_plot_title(),

            )
            return
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
                # Ensure the output directory exists
                if not os.path.exists(
                    f"{self.params['data_folder_name']}/well_monitoring/"
                ):
                    os.makedirs(f"{self.params['data_folder_name']}/well_monitoring/")
                plt.savefig(
                    f"{self.params['data_folder_name']}/well_monitoring/"
                    + f"{self.params['file_name']}_{plt_name}{suffix}.png"
                )
        start = (
            1
            if np.isclose(self.params["fracture_params"]["strike_angles"][0], 0)
            else 0
        )
        plot_flow_rate_and_fracture_displacement(
            csv_dir=f"{self.params['data_folder_name']}/well_monitoring",
            well_name=self.production_well_names[0],
            file_base=self.params["file_name"],
            fracture_names=self.fracture_names()[start:],
            title=self.create_plot_title(),
        )

    def create_plot_title(self) -> str | None:
        return None


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
    # suffix = "_long_well"
    # # suffix = ""
    # csv_dir = f"short_cold{suffix}_saved_data/well_monitoring"
    # well_name = "2 Production well"
    # file_base = f"example_4{suffix}"
    file_base = "example_7"

    with_well_file_names = []
    without_well_file_names = [
        "case_II_without_wells_strike_45_tilted_fracture_0_saved_data/well_monitoring",
        "case_II_without_wells_strike_45_tilted_fracture_1_saved_data/well_monitoring",
    ]
    titles = [
        "Strike 45°, Tilted Fracture 1",
        "Strike 45°, Tilted Fracture 2",
    ]
    for file_name in with_well_file_names:
        plot_flow_rate_and_fracture_displacement(
            csv_dir=file_name, well_name=well_name, file_base=file_base
        )

    for file_name, title in zip(without_well_file_names, titles):
        plot_fracture_displacement(
            csv_dir=file_name,
            file_base=file_base,
            title=title,
        )
