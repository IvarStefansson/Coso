# Sources:
#
# [1] C.A. Morrow and D.A. Lockner (2006): Physical properties of Two Core Samples from
#  Well 34-9RD2 at the Coso Geothermal Field, California
# [2] Nicholas C. Davatzes1 and Stephen H. Hickman2 (2010): The Feedback Between Stress,
# Faulting, and Fluid Flow
import os

import pandas as pd
import porepy as pp

bulk_modulus_granodiorite = 2.69e10  # [1], Table 4
youngs_modulus_granodiorite = 7.4e10  # [1], Table 4


def compute_shear_modulus(bulk_modulus, youngs_modulus):
    """From bulk and Young's modulus, compute the shear modulus."""
    return 3 * bulk_modulus * youngs_modulus / (9 * bulk_modulus - youngs_modulus)


def compute_lame_lambda(bulk_modulus, youngs_modulus):
    """From bulk and Young's modulus, compute Lame's first parameter."""
    return (
        3
        * bulk_modulus
        * (3 * bulk_modulus - youngs_modulus)
        / (9 * bulk_modulus - youngs_modulus)
    )


granodiorite_values = {
    "density": 2650,  # [1], p. 5
    "porosity": 0.01,  # [1], p. 5
    "friction_coefficient": 0.9,  # [1], p. 6 Range: 0.81-1.19!
    "lame_lambda": compute_lame_lambda(
        bulk_modulus_granodiorite, youngs_modulus_granodiorite
    ),
    "shear_modulus": compute_shear_modulus(
        bulk_modulus_granodiorite, youngs_modulus_granodiorite
    ),
    "biot_coefficient": 0.5,  # TODO: Find a reference and a value
    "dilation_angle": 0.1,  # TODO: Find a reference and a value
    "permeability": 2e-16,  # [2], p. 1
    "fracture_gap": 0e-4,  # TODO: Find a reference and a value
    "residual_aperture": 1e-3,  # TODO: Find a reference and a value
    "well_radius": 1.5e-1,  # TODO: Find a reference and a value. Not too far off if
    # LineID can be trusted. Increase to 0.2 m, to avoid pressure drop between wells?
    "normal_permeability": 1e-8,  # TODO: Reconsider model.
    "fracture_normal_stiffness": 1e9,  # TODO: Find a reference and a value
    "maximum_elastic_fracture_opening": 0e-3,  # TODO: Find a reference and a value
}
copy_names = ["thermal_conductivity", "specific_heat_capacity", "thermal_expansion"]
for name in copy_names:
    granodiorite_values[name] = pp.solid_values.granite[name]


def write_to_csv(file_name, constants: pp.compositional.materials.Constants):
    """Write the material parameters and their units to a CSV file."""
    # The constants object has a SI_units attribute that contains the units for each property.
    # Each parameter is an attribute of the constants object with a numeric value.
    # We write a csv file with the parameter names, their values, and their units.
    data = {
        "Parameter": [],
        "Value": [],
        "Unit": [],
    }
    for name, unit in constants.SI_units.items():
        data["Parameter"].append(name)
        data["Value"].append(getattr(constants, name))
        data["Unit"].append(unit)
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)


if __name__ == "__main__":
    # Write the material parameters to a CSV file.
    folder = "material_parameters"
    if not os.path.exists(folder):
        os.makedirs(folder)
    write_to_csv(f"{folder}/granodiorite.csv", pp.SolidConstants(**granodiorite_values))
    write_to_csv(f"{folder}/water.csv", pp.FluidComponent(**pp.fluid_values.water))
    print("Material parameters written to CSV files.")
