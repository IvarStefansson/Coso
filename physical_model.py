from typing import Callable
import numpy as np
import porepy as pp


class Hydrostatic(pp.PorePyModel):
    depth: Callable[[np.ndarray], np.ndarray]

    def hydrostatic_pressure(self, coords) -> np.ndarray:
        p_top = self.reference_variable_values.pressure

        # Hydrostatic pressure at the top of the domain
        rho = self.fluid.reference_component.density
        g = self.units.convert_units(pp.GRAVITY_ACCELERATION, "m*s^-2")

        # Hydrostatic pressure
        p = p_top + rho * g * self.depth(coords)
        return p


class PhysicalModel(
    Hydrostatic,
    pp.constitutive_laws.CubicLawPermeability,
    pp.constitutive_laws.GravityForce,
    pp.Thermoporomechanics,
    # pp.Poromechanics,
):
    """Model for the Coso geothermal reservoir."""
