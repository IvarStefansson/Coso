from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import porepy as pp
from run_example_4 import (
    boundary_velocities,
    cases,
    production_periods,
    thermal_expansions,
    names_from_params,
    create_schedule,
)
from exporting import plot_flow_rate_and_fracture_displacement


if __name__ == "__main__":
    create_legend = True  # Only needed once
    fracture_names = ["Fracture 2", "Fracture 3"]
    for velocity in boundary_velocities:
        for well_name, well_endpoint in cases:
            for period in production_periods:
                for thermal_expansion in thermal_expansions[
                    :1
                ]:  # Only plot for non-zero thermal expansion
                    simulation_name, folder_name, folder_name_init, file_name, title = (
                        names_from_params(
                            velocity, period, well_name, thermal_expansion
                        )
                    )
                    data = pd.read_csv(
                        f"{folder_name}_saved_data/well_monitoring/example_4_fractures.csv"
                    )
                    times = data["time"].values
                    plot_flow_rate_and_fracture_displacement(
                        csv_dir=folder_name + "_saved_data/well_monitoring/",
                        well_name="2 Production well",
                        file_base="example_4",
                        fracture_names=fracture_names,
                        title=title,
                        out_path="figures/Case I/semilogy/"
                        + simulation_name
                        + "_flow_rate_and_displacement.png",
                        semilogy=True,
                        create_legend=create_legend,
                    )
                    create_legend = False
