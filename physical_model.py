import numpy as np
import porepy as pp


class CosoBackgroundValues:
    def displacement_from_depth(self, coords: np.ndarray) -> np.ndarray:
        """Depth-dependent displacement.

        The x and y displacements are proportional to the distance from the center of the domain
        in the xy-plane and to depth. The z displacement is proportional to the depth.

        Parameters:
            coords ``shape=(3, num_cells)``: Coordinates.

        Returns:
            Array with displacement values. Shape (nd, num_cells).

        """
        # Displacement gradient set to .1 mm/km.
        # For the moment, we use a constant gradient in each direction.
        gradient = (
            self.parameters.get(
                "lithostatic_stress_multipliers", np.array([1.0, 1.0, 1.0])
            )
            * (
                self.solid.density * (1 - self.solid.porosity)
                + self.fluid.reference_component.density * self.solid.porosity
            )
            * pp.GRAVITY_ACCELERATION
            / self.bulk_modulus(None)._value
        )
        # compute from center of domain in the xy-plane
        box = self.domain.bounding_box
        center = np.array([box["xmin"] + box["xmax"], box["ymin"] + box["ymax"]]) / 2
        # Normalise by box size in the xy-plane

        distances = (center[:, np.newaxis] - coords[:2, :]) / np.array(
            [box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]]
        )[:, np.newaxis]
        # Pad with one for the z-coordinate
        distances = np.vstack([distances, np.ones_like(distances[0])])
        # Compute product of gradient, distances and depth
        return gradient[:, np.newaxis] * distances * self.depth(coords)

    def displacement_from_coordinates(self, coords: np.ndarray) -> np.ndarray:
        """Displacement from coordinates.

        The x and y displacements are proportional to the distance from the center of
        the domain in the xy-plane. The z displacement is proportional to the depth.

        Parameters:
            coords ``shape=(3, num_cells)``: Coordinates.

        Returns:
            Array with displacement values. Shape (nd, num_cells).

        """
        # if (

        # ):
        #     # If the time is before the well protocol offset, set displacement to zero.
        #     return np.zeros_like(coords)
        values = self.displacement_from_depth(coords)
        # Set velocity for all cells. The displacement is scaled with the x-coordinate
        # and time.
        if self.params["use_wells"]:
            x_scaling = coords[0] - self.domain.bounding_box["xmin"]
            offset_time = 20 * pp.YEAR if self.time_manager.time > 0 else 0
            values += np.outer(self.boundary_displacement_velocity, x_scaling) * (
                self.time_manager.time + offset_time
            )
        return values


class PhysicalModel(
    # CosoBackgroundValues,
    pp.constitutive_laws.GravityForce,
    # pp.SinglePhaseFlow,
    pp.Thermoporomechanics,
    # pp.Poromechanics,
    # pp.MomentumBalance,
):
    """Model for the Coso geothermal reservoir."""
