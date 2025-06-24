import numpy as np
import porepy as pp
import pandas as pd
from save_fracture_coords import (
    easting_northing_offset_to_strike_angle_dip_angle,
)
import sys


class CosoGeometry(pp.PorePyModel):
    def set_domain(self):
        pth = sys.path[0]
        self._fn_temp = pp.fracture_importer.network_3d_from_csv(
            f"{pth}/data/coords.txt"
        )
        # self._fn_temp.impose_external_boundary()
        # Increase domain size to avoid boundary effects
        box = self._fn_temp.domain.bounding_box
        for direction, val in zip(["x", "y", "z"], [1e3, 1e3, 1e2]):
            box[f"{direction}min"] -= val
            box[f"{direction}max"] += val
        self._domain = pp.Domain(bounding_box=box)

    def set_fractures(self):
        self._fractures = self._fn_temp.fractures

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

    def set_well_network(self) -> None:
        """Assign well network class."""
        if not self.params.get("use_wells", True):
            return super().set_well_network()
        # Read well network from file.
        wells = []
        pth = sys.path[0]
        # The file is in the same directory as this script.
        fn = f"{pth}/data/wellbores.xlsx"
        for name in self.well_names:
            # Read columns 4-6 (counting from 1), which are the x, y, z coordinates,
            # from the "name" sheet, skipping the header.
            with pd.ExcelFile(fn) as xls:
                df = pd.read_excel(xls, sheet_name=name, usecols=[3, 4, 5], skiprows=1)
                # For whatever reason, the ordering is y, x, z in the file.
                df = df.iloc[:, [1, 0, 2]]
                pts = df.values.T
                # Flip the z-coordinate to match the coordinate system.
                pts[2] *= -1
                # Add a point at the surface with the same x and y coordinates as the
                # first point in the wellbore.
                pt0 = pts[:, 0].copy().reshape((3, 1))
                pt0[2] = self._domain.bounding_box["zmax"]
                pts = np.hstack([pt0, pts])
                # Create a well object.
                well = pp.Well(pts, tags={"well_name": name})
                wells.append(well)

        self.well_network = pp.WellNetwork3d(
            domain=self._domain, wells=wells, parameters={"mesh_size": 25.0}
        )

    def depth(self, coords: np.ndarray) -> np.ndarray:
        """Depth from the surface.

        Parameters:
            coords: Coordinates.

        Returns:
            Array with depth values.

        """
        return -coords[2]


class EllipticFractureGeometry(CosoGeometry):
    def set_domain(self):
        box = {
            "xmin": -3e3,
            "xmax": 3e3,
            "ymin": -6e3,
            "ymax": 0e3,
            "zmin": -4e3,
            "zmax": 0e2,
        }

        self._domain = pp.Domain(bounding_box=box)

    def set_fractures(self):
        # -1.865*easting - 1.521*northing - 1.819 = z
        # Center: easting, northing, offset
        easting, northing, offset = -1.865, -1.521, 1.819

        center = self.units.convert_units(
            np.array([0.0387, -2.8576, -2.4575]) * pp.KILO * pp.METER, "m"
        )
        strike, dip = easting_northing_offset_to_strike_angle_dip_angle(
            easting, northing, offset
        )
        # Create a single elliptic fracture
        num_points = 10
        self._fractures = [
            pp.create_elliptic_fracture(
                center=center,
                strike_angle=strike,
                dip_angle=dip,
                major_axis=7.5e2,
                minor_axis=7.5e2,
                major_axis_angle=0,
                num_points=num_points,
            )
        ]
        # -0.903*easting +1.738*northing + 7.939 = z
        easting, northing, offset = -0.903, 1.738, -7.939
        center = self.units.convert_units(
            np.array([0.5592, -2.9615, -2.2861]) * pp.KILO * pp.METER, "m"
        )

        strike, dip = easting_northing_offset_to_strike_angle_dip_angle(
            easting, northing, offset
        )
        # Create a single elliptic fracture
        self._fractures.append(
            pp.create_elliptic_fracture(
                center=center,
                strike_angle=strike,
                dip_angle=dip,
                major_axis=750,
                minor_axis=750,
                major_axis_angle=0,
                num_points=num_points,
            )
        )


if __name__ == "__main__":

    class MockModel(
        EllipticFractureGeometry, pp.fluid_mass_balance.SinglePhaseFlow
    ): ...

    m = MockModel(
        {
            "grid_type": "simplex",
            "meshing_arguments": {"cell_size": 40e2, "cell_size_fracture": 1e2},
            "file_name": "elliptic_fractures",
            "folder_name": "visualization/geometry",
            "use_wells": True,
        }
    )
    m.prepare_simulation()

    print("Simulation prepared successfully.")
