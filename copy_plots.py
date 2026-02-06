import os
import shutil


if __name__ == "__main__":

    fracture_strike_angles =(
        [
            45,
            40,
            35,
        ]
    )


    well_options = [
        True,
        False,
    ]
    angle_indices = [
        0,  # Injection fracture
        1,  # Central fracture
        2,  # Production fracture
    ]
    for angle_index in angle_indices:
        for strike in fracture_strike_angles:
            if strike == 45 and angle_index != 0:
                continue  # Only run the 45 degree case for the injection fracture
            for has_wells in well_options:
                suffix = f"_strike_{int(strike)}_tilted_fracture_{angle_index}"

                if has_wells:
                    prefix = "case_II_with_wells"
                    destination = f"plots/case_II/with_wells{suffix}.png"

                    source = f"{prefix}{suffix}/well_monitoring/flow_rate_and_displacement_plot.png"
                else:
                    prefix = "case_II_without_wells"
                    source = f"{prefix}{suffix}/well_monitoring/fracture_displacement_plot.png"
                    destination = f"plots/case_II/without_wells{suffix}.png"
                if os.path.exists(source):
                    print(f"Copying {source} to {destination}")
                    shutil.copyfile(source, destination)
                else:
                    print(f"Source file {source} does not exist. Skipping.")