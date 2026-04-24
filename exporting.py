import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import porepy as pp
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from porepy.applications.convergence_analysis import ConvergenceAnalysis

JUMP_RANGE = (1e-8, 5e-4)


def plot_flow_rate_and_fracture_displacement(
    csv_dir: Path | str,
    well_name: str,
    file_base: str,
    fracture_names,
    title=None,
    temperature_schedule=None,
    out_path=None,
    semilogy=False,
    create_legend=True,
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
    separate_legend = True
    csv_dir = Path(csv_dir)
    # Read CSV files
    well_data = pd.read_csv(csv_dir / f"{file_base}.csv")
    fracture_data = pd.read_csv(csv_dir / f"{file_base}_fractures.csv")

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

    i0 = 0  # Initial condition does not correspond to a converged solution.
    accepted_times = [time.iloc[i0]]
    inds = [i0]
    for i, t in enumerate(time.iloc[i0 + 1 :], start=i0 + 1):
        if t < accepted_times[-1]:
            accepted_times.pop(-1)
            inds.pop(-1)
        accepted_times.append(t)
        inds.append(i)
    # Remove first time point if it corresponds to the initial condition (time=0) which does not
    # correspond to a converged solution.
    if accepted_times[0] == 0:
        accepted_times.pop(0)
        inds.pop(0)

    time = pd.Series(accepted_times)
    well_data_filtered = well_data_filtered.iloc[inds].reset_index(drop=True)
    sign = 1 if "Production" in well_name else -1
    well_data_filtered["fluid_flux_numeric"] *= sign
    # Dynamically get the first two unique fracture IDs, preserving order of first
    # occurrence. We pick out all but the first occurence, which corresponds to the
    # injection fracture.
    fracture_data_filtered = fracture_data[
        fracture_data["fracture_id"].isin(fracture_names)
    ].copy()

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Plot fluid flux on left axis
    flux_colors = ["tab:red", "tab:purple", "tab:blue", "tab:purple"]
    flux_linestyles = ["-", "--", "-.", "--"]
    color1 = "tab:blue"

    ax1.set_xlabel("Time (s)", fontsize=12)
    ax1.set_ylabel("Fluid Flux (kg/s)", color=color1, fontsize=12)

    ax1.tick_params(axis="y", labelcolor=color1)

    # Create right axis for fracture displacement
    ax2 = ax1.twinx()
    color2 = "tab:orange"
    color3 = "tab:green"
    color4 = "tab:gray"
    colors = [color2, color3, color4]
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
        ]
        fracture_subset = fracture_subset.iloc[inds].reset_index(drop=True)
        if semilogy:
            ax2.semilogy(
                time,
                fracture_subset["displacement_jump"],
                color=color,
                linewidth=2,
                markersize=4,
                label=f"Displacement Jump on {fracture_name}",
            )
            ax2.tick_params(axis="y")

        else:
            ax2.plot(
                time,
                fracture_subset["displacement_jump"],
                color=color,
                linewidth=2,
                markersize=4,
                label=f"Displacement Jump on {fracture_name}",
            )
            ax2.tick_params(axis="y")
        ax2.set_ylim(bottom=JUMP_RANGE[0], top=JUMP_RANGE[1])

    if temperature_schedule is None:
        line1 = ax1.plot(
            time,
            well_data_filtered["fluid_flux_numeric"],
            color=color1,
            linewidth=2,
            label="Fluid Flux in Production Well",
        )
    else:
        # Plot temperature with alternating colors for different schedule steps. We use
        # red-blue-red-blue-... colors for the temperature to visually distinguish the
        # schedule steps.
        # line1 = ax1.plot(
        #     time,
        #     well_data_filtered["fluid_flux_numeric"],
        #     color=color1,
        #     linewidth=2,
        #     label="Fluid Flux in Production Well",
        # )
        for i in range(len(temperature_schedule) - 1):
            step_mask = (time >= temperature_schedule[i]) & (
                time <= temperature_schedule[i + 1]
            )
            ax1.plot(
                time[step_mask],
                well_data_filtered["fluid_flux_numeric"][step_mask],
                color=color1,  # flux_colors[0],#[i % len(flux_colors)],
                linestyle=flux_linestyles[i % len(flux_linestyles)],
                linewidth=2,
            )
        line1 = ax1.get_lines()[: len(temperature_schedule) - 1]
    # Create combined legend
    # Create custom legend handle with red and blue colors for fluid flux
    if temperature_schedule is not None:
        # red_line = Line2D([0], [0], color='tab:red', linewidth=2)
        # purple_line = Line2D([0], [0], color='tab:purple', linewidth=2)
        blue_line = Line2D([0], [0], color="tab:blue", linewidth=2)
        fluid_flux_handle = blue_line
        fluid_flux_label = "Fluid Flux in Production Well"
        lines2 = ax2.get_lines()
        if not separate_legend:
            ax1.legend(
                [fluid_flux_handle] + lines2,
                [fluid_flux_label] + [l.get_label() for l in lines2],
                loc="lower right",
                fontsize=10,
                framealpha=0.9,
                edgecolor="black",
                handler_map={tuple: HandlerTuple(ndivide=None)},
            )
    else:
        lines1 = line1
        lines2 = ax2.get_lines()
        if not separate_legend:
            ax1.legend(
                lines1 + lines2,
                [l.get_label() for l in lines1 + lines2],
                loc="upper left",
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
    if out_path is not None:
        output_path = Path(out_path)
    else:
        output_path = csv_dir / "flow_rate_and_displacement_plot.png"
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    # Create separate figure with legend only
    if create_legend:
        fig_legend = plt.figure(figsize=(7, 0.5))
        ax_legend = fig_legend.add_subplot(111)
        ax_legend.axis("off")
        if temperature_schedule is not None:
            handles = [fluid_flux_handle] + lines2
            labels = [fluid_flux_label] + [l.get_label() for l in lines2]
        else:
            handles = lines1 + lines2
            labels = [l.get_label() for l in lines1 + lines2]
        ax_legend.legend(
            handles,
            labels,
            loc="center",
            fontsize=10,
            framealpha=0.9,
            edgecolor="black",
        )
        legend_out_path = output_path.parent / f"{output_path.stem}_legend.png"
        fig_legend.savefig(legend_out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    return fig


def plot_fracture_displacement(
    csv_dir: Path | str,
    file_base: str,
    title=None,
    semilogy=False,
) -> None:
    """
    Plot displacement jump of fractures over time (for simulations without wells).

    This function creates a plot showing the displacement_jump (m) for fractures vs time.

    Parameters:
        csv_dir: Path to directory containing the CSV files.
        file_base: Base name for the CSV file (default: "example_3").
    """
    csv_dir = Path(csv_dir)
    # Read CSV file for fractures
    fracture_data = pd.read_csv(csv_dir / f"{file_base}_fractures.csv")

    # Get time data from fracture data
    time = fracture_data["time"].unique()
    time = pd.Series(np.sort(time))

    # Process time data to handle non-monotonic times (due to diverged time steps)
    # Initial condition does not correspond to a converged solution.
    i0 = 1
    accepted_times = [time.iloc[i0]]

    for i, t in enumerate(time.iloc[i0:], start=i0):
        if t < accepted_times[-1]:
            accepted_times.pop(-1)
        accepted_times.append(t)

    time = pd.Series(accepted_times)

    # Get unique fracture IDs, preserving order of first occurrence
    fracture_names = fracture_data["fracture_id"].unique().tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))

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
        fracture_subset = fracture_data[fracture_data["fracture_id"] == fracture_name]

        # Filter to accepted times
        fracture_subset = fracture_subset[fracture_subset["time"].isin(accepted_times)]

        color = color_map.get(idx, f"C{idx}")
        if semilogy:
            ax.semilogy(
                fracture_subset["time"],
                fracture_subset["displacement_jump"],
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                label=f"{fracture_name}",
            )

        else:
            ax.plot(
                fracture_subset["time"],
                fracture_subset["displacement_jump"],
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                label=f"{fracture_name}",
            )
        ax.set_ylim(bottom=JUMP_RANGE[0], top=JUMP_RANGE[1])

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Displacement Jump (m)", fontsize=12)
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
    output_path = csv_dir / "fracture_displacement_plot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    return fig


def summarize_slip_onset_times(
    slip_onset_times: dict,
    fracture_names,
    velocities,
    output_file: Path | str | None = None,
) -> None:
    """Save a summary of slip onset times for different simulations."""
    output_string = ""

    for fracture in fracture_names:
        # Create table with rows corresponding to boundary velocity and columns
        # corresponding to long/short well. The entries are the slip onset times for
        # this fracture.
        output_string += f"Fracture: {fracture}\n"
        for velocity in velocities:
            output_string += f"Velocity: {velocity}\n"
            for simulation, times in slip_onset_times.items():
                if f"velocity_{velocity:.0e}_" in simulation and fracture in times:
                    output_string += f"  {simulation}: {times[fracture]:.2f} seconds\n"
                elif f"velocity_{velocity:.0e}_" in simulation:
                    output_string += f"  {simulation}: Not found\n"

    if output_file is not None:
        # Create directory if it doesn't exist
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_string)
    else:
        print(output_string)


class CosoExporter:
    mdg: pp.MixedDimensionalGrid
    nd: int
    evaluate_and_scale: Callable[
        [Sequence[pp.Grid] | Sequence[pp.MortarGrid], str, str], np.ndarray
    ]

    def slip_onset_times(self) -> dict:
        """Compute sliding onset time for each fracture."""
        sliding_onset_times = {}
        for time, data in zip(self._data_collection_times, self.results):
            for name, variables in data.items():
                if not name.startswith("Fracture"):
                    continue
                jump = variables["displacement_jump"]
                if jump > JUMP_RANGE[0]:
                    if name not in sliding_onset_times:
                        sliding_onset_times[name] = time
        return sliding_onset_times

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
            tensor = self.evaluate_and_scale([sd], "permeability", "m ^ 2")
            perm = tensor.reshape(9, -1, order="F")
            data.append((sd, "permeability_tensor", perm[0]))
            depth = self.depth(sd.cell_centers)
            data.append(
                (
                    sd,
                    "depth",
                    self.units.convert_units(depth, "m", to_si=True),
                )
            )
            density = self.fluid.density([sd]).value(self.equation_system)
            data.append(
                (
                    sd,
                    "density",
                    self.units.convert_units(density, "kg * m^-3", to_si=True),
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
            elif sd.dim == self.nd:
                data.append(
                    (
                        sd,
                        "porosity",
                        self.evaluate_and_scale([sd], "porosity", "-"),
                    )
                )
            if hasattr(self, "hydrostatic_pressure"):
                data.append(
                    (
                        sd,
                        "hydrostatic_pressure",
                        self.units.convert_units(
                            self.hydrostatic_pressure(depth), "Pa", to_si=True
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
                data.append((sd, "well_name", np.full(sd.num_cells, -1)))
            else:
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
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in sds])
        pressure_trace = self.evaluate_and_scale(sds, "pressure_trace", "Pa")
        pressure = self.evaluate_and_scale(sds, "pressure", "Pa")

        has_t = hasattr(self, "temperature_trace")
        if has_t:
            temperature = self.evaluate_and_scale(sds, "temperature", "K")
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
            flux_sign = np.sign(sd.face_normals[-1, top_faces])[0]

            data[parent_well.tags["well_name"]] = {
                "pressure": float(
                    pressure_trace[face_offsets[id] : face_offsets[id + 1]][top_faces][
                        0
                    ]
                ),
                "darcy_flux": float(
                    darcy_f[face_offsets[id] : face_offsets[id + 1]][top_faces][0]
                    * flux_sign
                ),
                "fluid_flux": float(
                    fluid_f[face_offsets[id] : face_offsets[id + 1]][top_faces][0]
                    * flux_sign
                ),
            }
            if has_t:
                top_cell = np.argsort(self.depth(sd.cell_centers))[0]
                data[parent_well.tags["well_name"]].update(
                    {
                        "temperature": float(
                            temperature[cell_offsets[id] : cell_offsets[id + 1]][
                                top_cell
                            ]
                        ),
                        "enthalpy_flux": float(
                            enthalpy_f[face_offsets[id] : face_offsets[id + 1]][
                                top_faces
                            ][0]
                            * flux_sign
                        ),
                    }
                )
        fracs = self.mdg.subdomains(dim=self.nd - 1)
        cell_offsets = np.cumsum([0] + [sd.num_cells for sd in fracs])

        displacement_jump = self.evaluate_and_scale(
            fracs, "displacement_jump", "m"
        ).reshape((self.nd, -1), order="F")
        jump_norm = np.linalg.norm(displacement_jump, axis=0)
        friction_coefficient = self.evaluate_and_scale(
            fracs, "friction_coefficient", ""
        )
        traction = self.evaluate_and_scale(fracs, "contact_traction", "-").reshape(
            (self.nd, -1), order="F"
        )
        slip_tendency = self.compute_slip_tendency(traction, friction_coefficient)
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
            data[key]["slip_tendency"] = np.nanmean(
                slip_tendency[cell_offsets[id] : cell_offsets[id + 1]]
            )
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
        folder_path = Path(self.params["data_folder_name"]) / "well_monitoring"
        folder_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            folder_path / f"{self.params['file_name']}.csv",
            index=False,
        )
        df_fractures = pd.DataFrame(fracture_records)
        df_fractures.to_csv(
            folder_path / f"{self.params['file_name']}_fractures.csv",
            index=False,
        )
        self.well_monitoring_data = df
        self.fracture_monitoring_data = df_fractures

    def plot_well_monitoring(self, temperature_schedule=None) -> None:
        """Plot well monitoring data for a given well."""
        if not hasattr(self, "well_monitoring_data"):
            self.save_results()
        if not self.params["use_wells"]:
            plot_fracture_displacement(
                csv_dir=Path(self.params["data_folder_name"]) / "well_monitoring",
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
                    output_path = (
                        Path(self.params["data_folder_name"])
                        / "well_monitoring"
                        / f"{self.params['file_name']}_{plt_name}{suffix}.png"
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(output_path)
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
                # Ensure the output directory exists and save
                output_path = (
                    Path(self.params["data_folder_name"])
                    / "well_monitoring"
                    / f"{self.params['file_name']}_{plt_name}{suffix}.png"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(output_path)
        if "example_4" in self.params["file_name"]:
            start = 1
        else:
            start = (
                1
                if np.isclose(self.params["fracture_params"]["strike_angles"][0], 0)
                else 0
            )
        well_ind = 1 if "example_8" in self.params["file_name"] else 0
        plot_flow_rate_and_fracture_displacement(
            csv_dir=Path(self.params["data_folder_name"]) / "well_monitoring",
            well_name=self.well_names[well_ind],
            file_base=self.params["file_name"],
            fracture_names=self.fracture_names()[start:],
            title=self.create_plot_title(),
            temperature_schedule=temperature_schedule,
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
            folder_name=str(Path(self.params["folder_name"]) / "geometry"),
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

    with_well_file_names = [
        "case_III_with_wells_strike_-41_tilted_fracture_1_saved_data/well_monitoring",
    ]
    with_well_titles = [
        "Strike 41°, Tilted Fracture 2",
    ]
    without_well_file_names = [
        "case_II_without_wells_strike_45_tilted_fracture_0_saved_data/well_monitoring",
        "case_II_without_wells_strike_45_tilted_fracture_1_saved_data/well_monitoring",
    ]
    titles = [
        "Strike 45°, Tilted Fracture 1",
        "Strike 45°, Tilted Fracture 2",
    ]
    transition_time = pp.HOUR
    # Alternate between warm and cold injection every period, with a short
    # transition time in between to allow for convergence without too large
    # jumps in the solution. The schedule defines the time points at which
    # the boundary conditions change.
    period = 100 * pp.DAY
    schedule = np.array(
        [
            0,
            1 * period - transition_time,  # End of first warm injection
            1 * period,  # Start of first cold period
            2 * period - transition_time,  # End of first cold period
            2 * period,  # Start of second warm injection
            3 * period - transition_time,  # End of second warm injection
            3 * period,  # Start of second cold period
            4 * period - transition_time,  # End of second cold period
            # 4 * period,
        ],
    )
    for file_name in with_well_file_names:
        # plot_flow_rate_and_fracture_displacement(
        #     csv_dir=file_name, well_name="2 Production well", file_base="example_8",
        #     fracture_names=["Fracture 1", "Fracture 2", "Fracture 3"],
        #     title=with_well_titles[0],
        #     temperature_schedule=schedule,
        # )
        plot_flow_rate_and_fracture_displacement(
            csv_dir=file_name,
            well_name="1 Injection well",
            file_base="example_8",
            fracture_names=["Fracture 1", "Fracture 2", "Fracture 3"],
            title=with_well_titles[0],
            temperature_schedule=schedule,
        )

    # for file_name, title in zip(without_well_file_names, titles):
    #     plot_fracture_displacement(
    #         csv_dir=file_name,
    #         file_base=file_base,
    #         title=title,
    #     )
