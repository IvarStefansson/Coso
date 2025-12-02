import os

import numpy as np
import porepy as pp
import pandas as pd
from save_fracture_coords import (
    easting_northing_offset_to_strike_angle_dip_angle,
)
from exporting import CosoExporter
import sys
from wells import WellDataConceptual, WellDataCoso
from porepy.applications.md_grids.model_geometries import SubsurfaceCuboidDomain


class CosoGeometry:
    def set_domain(self):
        pth = sys.path[0]
        self._fn_temp = pp.fracture_importer.network_3d_from_csv(
            f"{pth}/data/{self.params.get('fracture_file', 'coords.txt')}",
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


def plane_equation_to_strike_dip(
    A: float, B: float, C: float, D: float
) -> tuple[float, float]:
    """Convert plane equation coefficients to strike and dip angles.

    Given the coefficients of a plane equation in the form of Ax + By + Cz + D = 0,
    this function calculates the strike angle and dip angle of the plane.

    Parameters:
        A: Coefficient for x.
        B: Coefficient for y.
        C: Coefficient for z.
        D: Constant term.
    Returns:
        A tuple containing the strike angle and dip angle.
        strike_angle: Line of rotation for the dip. Given as angle in radians from the
            x-direction.
        dip_angle: Dip angle in radians, i.e., rotation around the strike direction.
    """
    # Calculate strike angle
    strike = np.arctan2(B, A)
    # Convert from angle with horizontal to angle with x-direction
    strike = strike - np.pi / 2
    # Calculate dip angle
    dip = np.arctan2(np.sqrt(A**2 + B**2), C)
    return strike, dip


class FractureGeometry2(EllipticFractureGeometry):
    def set_domain(self):
        s = 3.0e3
        box = {
            "xmin": 3e2 - s,
            "xmax": 3e2 + s,
            "ymin": -3e3 - s,
            "ymax": s - 3e3,
            "zmin": -5e3,
            "zmax": 0e2,
        }
        self._domain = pp.Domain(bounding_box=box)

    def set_fractures(self):
        # Shut-in plane: -0.990*x + -0.098*y + 0.102*z + -0.019 = 0
        # Non-shut-in plane: -0.974*x + 0.112*y + 0.196*z + 0.839 = 0
        # easting +northing + offset = z
        # Center: easting, northing, offset
        dz = -0.102
        easting, northing, offset = -0.990 / dz, -0.098 / dz, -0.019 / dz
        x_0, y_0 = 0.0387, -2.8576
        z_0 = offset + easting * x_0 + northing * y_0
        center = self.units.convert_units(
            np.array([x_0, y_0, z_0]) * pp.KILO * pp.METER, "m"
        )
        strike, dip = easting_northing_offset_to_strike_angle_dip_angle(
            easting, northing, offset
        )
        A, B, C, D = -0.990, -0.098, 0.102, -0.019
        strike, dip = plane_equation_to_strike_dip(A, B, C, D)
        # Create a single elliptic fracture
        num_points = 12
        self._fractures = [
            pp.create_elliptic_fracture(
                center=center,
                strike_angle=strike,
                dip_angle=dip,
                major_axis=5.0e2,
                minor_axis=5.0e2,
                major_axis_angle=0,
                num_points=num_points,
            )
        ]
        dz = -0.196
        easting, northing, offset = -0.974 / dz, 0.112 / dz, 0.839 / dz
        A, B, C, D = -0.974, 0.112, 0.196, 0.839
        x_1, y_1 = 0.2, -2.9
        z_1 = offset + easting * x_1 + northing * y_1
        center = self.units.convert_units(
            np.array([x_1, y_1, z_1]) * pp.KILO * pp.METER, "m"
        )

        strike, dip = easting_northing_offset_to_strike_angle_dip_angle(
            easting, northing, offset
        )
        strike, dip = plane_equation_to_strike_dip(A, B, C, D)
        # Create a single elliptic fracture
        self._fractures.append(
            pp.create_elliptic_fracture(
                center=center,
                strike_angle=strike,
                dip_angle=dip,
                major_axis=5.0e2,
                minor_axis=5.0e2,
                major_axis_angle=0,
                num_points=num_points,
            )
        )
        center_injection = self.units.convert_units(
            np.array([1.1, -3.4, -1.9]) * pp.KILO * pp.METER, "m"
        )
        # Create a single elliptic fracture
        self._fractures.append(
            pp.create_elliptic_fracture(
                center=center_injection,
                strike_angle=0,
                dip_angle=0,
                major_axis=3.0e2,
                minor_axis=3.0e2,
                major_axis_angle=0,
                num_points=num_points,
            )
        )


class ConceptualGeometry(SubsurfaceCuboidDomain):
    def set_well_network(self) -> None:
        """Assign well network class."""
        if not self.params.get("use_wells", True):
            return super().set_well_network()
        # Read well network from file.
        wells = []
        x_in = np.full((1, 3), -1)
        y_in = np.array([[2, 2, -2]])
        z_in = np.array([[0, -2, -2]])
        pts = np.vstack([x_in, y_in, z_in]) * pp.KILO * pp.METER
        name = self.injection_well_names[0]
        # Create a well object.
        well = pp.Well(pts, tags={"well_name": name})
        wells.append(well)
        x_prod = np.full((1, 2), 1.0)
        y_prod = np.array([[0, 0]])
        z_prod = np.array([[0, -3]])
        pts = np.vstack([x_prod, y_prod, z_prod]) * pp.KILO * pp.METER
        name = self.production_well_names[0]
        # Create a well object.
        well = pp.Well(pts, tags={"well_name": name})
        wells.append(well)
        self.well_network = pp.WellNetwork3d(
            domain=self._domain, wells=wells, parameters={"mesh_size": 25.0}
        )

    def set_domain(self):
        s = 3.0e3
        box = {
            "xmin": -s,
            "xmax": s,
            "ymin": -s,
            "ymax": s,
            "zmin": -5e3,
            "zmax": 0e2,
        }
        self._domain = pp.Domain(bounding_box=box)

    def set_fractures(self):
        center = self.units.convert_units(
            np.array([-1.0, 1.0, -2.0]) * pp.KILO * pp.METER, "m"
        )
        # Create a single elliptic fracture
        num_points = 10
        self._fractures = [
            pp.create_elliptic_fracture(
                center=center,
                strike_angle=np.pi / 4,
                dip_angle=np.pi / 2,
                major_axis=5.0e2,
                minor_axis=5.0e2,
                major_axis_angle=0,
                num_points=num_points,
            )
        ]
        center = self.units.convert_units(
            np.array([-1.0, -1.0, -2.0]) * pp.KILO * pp.METER, "m"
        )

        self._fractures.append(
            pp.create_elliptic_fracture(
                center=center,
                strike_angle=np.pi / 4,
                dip_angle=np.pi / 2,
                major_axis=5.0e2,
                minor_axis=5.0e2,
                major_axis_angle=0,
                num_points=num_points,
            )
        )
        center_injection = self.units.convert_units(
            np.array([1.0, 0, -2.0]) * pp.KILO * pp.METER, "m"
        )
        # Create a single elliptic fracture
        self._fractures.append(
            pp.create_elliptic_fracture(
                center=center_injection,
                strike_angle=0,
                dip_angle=0,
                major_axis=4.0e2,
                minor_axis=4.0e2,
                major_axis_angle=0,
                num_points=num_points,
            )
        )


if __name__ == "__main__":

    class MockModel(
        ConceptualGeometry, WellDataConceptual, CosoExporter, pp.MomentumBalance
    ): ...

    m = MockModel(
        {
            "grid_type": "simplex",
            "meshing_arguments": {"cell_size": 5e3, "cell_size_fracture": 5e2},
            "file_name": "elliptic_fractures",
            "folder_name": "visualization/geometry",
            "use_wells": True,
        }
    )
    m.prepare_simulation()
    for sd in m.mdg.subdomains(dim=0):
        print(sd.tags)
        print(sd.cell_centers)
    print("Simulation prepared successfully.")
