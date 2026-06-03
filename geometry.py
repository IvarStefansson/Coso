import os
import sys

import numpy as np
import pandas as pd
import porepy as pp
from porepy.applications.md_grids.model_geometries import TwoEllipticFractures3d

from exporting import CosoExporter
from save_fracture_coords import easting_northing_offset_to_strike_angle_dip_angle
from wells import WellDataConceptual, WellDataCoso


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


class ConceptualGeometry(TwoEllipticFractures3d):
    def set_well_network(self) -> None:
        """Assign well network class."""
        if not self.params.get("use_wells", True):
            return super().set_well_network()
        # Read well network from file.
        dx = self.domain_sizes()[0] / 4
        y_mid = self.domain_sizes()[1] / 2
        dy = self.domain_sizes()[1] / 8
        z = -self.domain_sizes()[2]
        x = self.domain_sizes()[0] / 8 * 3

        x_prod = np.full((1, 3), x)
        y_prod = np.array(
            [
                [
                    y_mid + dy + 10,
                    y_mid + dy + 10,
                    self.params["production_well_y_endpoint"],
                ]
            ]
        )
        z_top = 0  # self._domain.bounding_box["zmax"]
        z_prod = np.array([[z_top, z / 2, z / 2]])
        pts_prod = np.vstack([x_prod, y_prod, z_prod])
        # Create a well object.
        well_prod = pp.Well(pts_prod, tags={"well_name": self.production_well_names[0]})
        x_inj = np.full((1, 2), x + dx)
        y_inj = np.array([[y_mid + 2, y_mid - 1]])
        z_inj = np.array([[z_top, 3 / 4 * z]])
        pts_inj = np.vstack([x_inj, y_inj, z_inj])
        # Create a well object.
        well_inj = pp.Well(
            pts_inj, tags={"well_name": self.injection_well_names[0]}
        )  # 68 20rd
        wells = [well_inj, well_prod]
        self.well_network = pp.WellNetwork3d(
            domain=self._domain,
            wells=wells,
            parameters={"mesh_size": self.params["meshing_arguments"]["cell_size"] / 2},
        )

    def fracture_names(self) -> list[str]:
        return ["Fracture 1", "Fracture 2", "Fracture 3"]

    def set_fractures(self):
        # The order of the fractures is used elsewhere, inferred from sd.frac_num if
        # we operate on grids. Make sure the order is
        #  0 injection,
        #  1 production,
        #  2 production or passive, depending on length of well.
        dx = self.domain_sizes()[0] / 4
        y_mid = self.domain_sizes()[1] / 2
        dy = self.domain_sizes()[1] / 8
        z = -self.domain_sizes()[2] / 2
        x = self.domain_sizes()[0] / 8 * 3

        center_injection = np.array([x + dx, y_mid, z])
        # Create a single elliptic fracture
        self._fractures = [  # First the injection fracture
            pp.EllipticFracture(
                center=center_injection,
                strike_angle=np.pi / 2,
                dip_angle=np.pi / 2,
                major_axis=self.fracture_major_axes[2],
                minor_axis=self.fracture_minor_axes[2],
                major_axis_angle=0,
            )
        ]

        center = np.array([x, y_mid + dy, z])
        # Production fracture, always connected.
        self._fractures.append(
            pp.EllipticFracture(
                center=center,
                strike_angle=np.pi / 4,
                dip_angle=np.pi / 2,
                major_axis=self.fracture_major_axes[0],
                minor_axis=self.fracture_minor_axes[0],
                major_axis_angle=0,
            )
        )
        center = np.array([x, y_mid - dy, z])
        # Production or passive fracture, connected for long well.
        self._fractures.append(
            pp.EllipticFracture(
                center=center,
                strike_angle=-np.pi / 4,
                dip_angle=np.pi / 2,
                major_axis=self.fracture_major_axes[1],
                minor_axis=self.fracture_minor_axes[1],
                major_axis_angle=0,
            )
        )


class ConceptualGeometryTwoFractures(ConceptualGeometry):
    def caprock_depth(self) -> float:
        """Depth of the caprock.

        Returns:
            Depth of the caprock in meters.
        """
        return self.params.get("caprock_depth", 1.1e3)

    def reservoir_depth(self) -> float:
        """Depth of the reservoir.

        Returns:
            Depth of the reservoir in meters.
        """
        return self.params.get("reservoir_depth", 3e3)

    def fracture_names(self) -> list[str]:
        return ["Fracture 1", "Fracture 2"]

    def set_fractures(self):
        """Keep first fracture and create second fracture at same y location but different x location."""
        # Call the parent class's set_fractures method
        super().set_fractures()
        # Remove the second and third fracture
        self._fractures = self._fractures[:1]
        y_mid = self.domain_sizes()[1] / 2
        z = -self.domain_sizes()[2] / 2
        x = self.domain_sizes()[0] / 8 * 3
        center = np.array([x, y_mid, z])
        # Production fracture, always connected.
        self._fractures.append(
            pp.EllipticFracture(
                center=center,
                strike_angle=np.pi / 4,
                dip_angle=np.pi / 2,
                major_axis=self.fracture_major_axes[0],
                minor_axis=self.fracture_minor_axes[0],
                major_axis_angle=0,
            )
        )

    def set_well_network(self) -> None:
        """Assign well network class."""
        if not self.params.get("use_wells", True):
            return super().set_well_network()
        dx = self.domain_sizes()[0] / 4
        y_mid = self.domain_sizes()[1] / 2
        z = -self.domain_sizes()[2]
        x = self.domain_sizes()[0] / 8 * 3

        x_prod = np.full((1, 3), x)
        y_prod = np.array(
            [
                [
                    y_mid + 2,
                    y_mid - 1,
                    self.params["production_well_y_endpoint"],
                ]
            ]
        )
        z_top = 0
        z_prod = np.array([[z_top, z / 2, z / 2]])
        pts_prod = np.vstack([x_prod, y_prod, z_prod])
        # Create a well object.
        well_prod = pp.Well(pts_prod, tags={"well_name": self.production_well_names[0]})
        x_inj = np.full((1, 2), x + dx)
        y_inj = np.array([[y_mid + 2, y_mid - 1]])
        z_inj = np.array([[z_top, 3 / 4 * z]])
        pts_inj = np.vstack([x_inj, y_inj, z_inj])
        # Create a well object.
        well_inj = pp.Well(
            pts_inj, tags={"well_name": self.injection_well_names[0]}
        )  # 68 20rd
        wells = [well_inj, well_prod]
        self.well_network = pp.WellNetwork3d(
            domain=self._domain,
            wells=wells,
            parameters={"mesh_size": self.params["meshing_arguments"]["cell_size"] / 2},
        )


class ConstraintsCapcrockAndReservoirDepth:
    def set_fractures(self):
        """Keep first fracture and create second fracture at same y location but different x location."""
        # Call the parent class's set_fractures method
        super().set_fractures()
        # Set throughgoing fractures at the two depths
        caprock_depth = self.caprock_depth()
        reservoir_depth = self.reservoir_depth()
        # Store length of fractures as was.
        self._num_fractures = len(self._fractures)
        for z in [caprock_depth, reservoir_depth]:
            center = np.array(
                [self.domain_sizes()[0] / 2, self.domain_sizes()[1] / 2, -z]
            )
            self._fractures.append(
                pp.EllipticFracture(
                    center=center,
                    strike_angle=np.pi / 4,
                    dip_angle=np.pi / 2,
                    major_axis=self.fracture_major_axes[0],
                    minor_axis=self.fracture_minor_axes[0],
                    major_axis_angle=0,
                )
            )

    def meshing_kwargs(self) -> dict:
        """Ensure the two additional fractures are marked as constraints for meshing."""
        kwargs = super().meshing_kwargs()
        constraints = kwargs.get("constraints", [])
        # Add if not already present, to avoid duplicates if this method is called
        # multiple times.
        constraints = set(constraints)
        for i in range(self._num_fractures, self._num_fractures + 2):
            constraints.add(i)
        kwargs["constraints"] = list(constraints)
        return kwargs


class SubmergedDomain:
    def depth(self, coords: np.ndarray) -> np.ndarray:
        """Depth from the surface.

        Parameters:
            coords: Coordinates.
        Returns:
            Array with depth values.

        """
        return -coords[2]

    def set_domain(self) -> None:
        """Set the cubic domain."""
        x_size, y_size, z_size = self.domain_sizes()
        top_surface = self.params.get("top_surface", -1.1e3)
        box = {
            "xmin": 0.0,
            "xmax": x_size,
            "ymin": 0.0,
            "ymax": y_size,
            "zmin": top_surface - z_size,
            "zmax": top_surface,
        }
        self._domain = pp.Domain(box)


class LargeGeometry(TwoEllipticFractures3d):
    def set_well_network(self) -> None:
        """Assign well network class."""
        if not self.params.get("use_wells", True):
            return super().set_well_network()
        x = self.domain_sizes()[0] / 2
        z = -self.domain_sizes()[2] * 1 / 2
        y_max = self.domain_sizes()[1]
        x_prod = np.array([[x - 5e2, x - 5e2, x + 5e2]])
        y_prod = np.full((1, 3), y_max - 3e3)
        z_prod = np.array([[0, z, z]])
        pts_prod = np.vstack([x_prod, y_prod, z_prod])
        # Create a well object.
        well_prod = pp.Well(pts_prod, tags={"well_name": self.production_well_names[0]})
        x_inj = np.array([[x - 5e2, x - 5e2, x + 5e2]])
        y_inj = np.full((1, 3), y_max - 5e3)
        z_inj = np.array([[0, z, z]])
        pts_inj = np.vstack([x_inj, y_inj, z_inj])
        # Create a well object.
        well_inj = pp.Well(
            pts_inj, tags={"well_name": self.injection_well_names[0]}
        )  # 68 20rd
        wells = [well_inj, well_prod]
        self.well_network = pp.WellNetwork3d(
            domain=self._domain, wells=wells, parameters={"mesh_size": 250.0}
        )

    def fracture_names(self) -> list[str]:
        return ["Fracture 1", "Fracture 2", "Fracture 3", "Fracture 4"]

    def set_fractures(self):
        # The order of the fractures is used elsewhere, inferred from sd.frac_num if
        # we operate on grids. Make sure the order is
        #  0 injection,
        #  1 production,
        #  2 in reservoir between injection and production,
        #  3 outside reservoir.

        dy = 1e3
        x = self.domain_sizes()[0] / 2
        z = -self.domain_sizes()[2] * 1 / 2
        y_max = self.domain_sizes()[1]

        # Injection fracture
        num_points = 15
        params = self.params["fracture_params"]
        major_axes = params.get("fracture_major_axes", np.full((4,), 7.5e2))
        minor_axes = np.full((4,), 7.5e2)
        angle = params["strike_angle"]
        self._fractures = [
            pp.create_elliptic_fracture(
                center=np.array([x, y_max - 5 * dy, z]),
                strike_angle=np.pi / 2,
                dip_angle=np.pi / 2,
                major_axis=major_axes[0],
                minor_axis=minor_axes[0],
                major_axis_angle=0,
                num_points=num_points,
            )
        ]
        # Production fracture
        self._fractures.append(
            pp.create_elliptic_fracture(
                center=np.array([x, y_max - 3 * dy, z]),
                strike_angle=angle,
                dip_angle=np.pi / 2,
                major_axis=major_axes[1],
                minor_axis=minor_axes[1],
                major_axis_angle=0,
                num_points=num_points,
            )
        )
        # Fracture between injection and production
        self._fractures.append(
            pp.create_elliptic_fracture(
                center=np.array([x, y_max - 4 * dy, z]),
                strike_angle=angle,
                dip_angle=np.pi / 2,
                major_axis=major_axes[2],
                minor_axis=minor_axes[2],
                major_axis_angle=0,
                num_points=num_points,
            )
        )
        # Passive fracture outside reservoir.
        self._fractures.append(
            pp.create_elliptic_fracture(
                center=np.array([x, 3 * dy, z]),
                strike_angle=angle,
                dip_angle=np.pi / 2,
                major_axis=major_axes[3],
                minor_axis=minor_axes[3],
                major_axis_angle=0,
                num_points=num_points,
            )
        )


class CoolingGeometry(TwoEllipticFractures3d):
    def fracture_names(self) -> list[str]:
        return ["Fracture 1", "Fracture 2", "Fracture 3"]

    def set_fractures(self):
        # The order of the fractures is used elsewhere, inferred from sd.frac_num if
        # we operate on grids. Make sure the order is
        #  0 injection,
        #  1 production,
        #  2 production or passive, depending on length of well.
        params = self.params["fracture_params"]

        dx = params.get("dx", self.domain_sizes()[0] / 14)
        y = self.domain_sizes()[1] / 2
        z = -self.domain_sizes()[2] / 2
        x = self.domain_sizes()[0] * 6 / 14

        center_injection = np.array([x, y, z])
        r = params.get("fracture_major_axes")
        num_points = params.get("num_points", 14)

        self._fractures = [  # First the injection fracture
            pp.create_elliptic_fracture(
                center=center_injection,
                strike_angle=params["strike_angles"][0],
                dip_angle=np.pi / 2,
                major_axis=r[0],
                minor_axis=r[0],
                major_axis_angle=0,
                num_points=num_points,
            )
        ]
        center = np.array([x + dx + params.get("center_fracture_x_offset", 0), y, z])
        # Center fracture
        self._fractures.append(
            pp.create_elliptic_fracture(
                center=center,
                strike_angle=params["strike_angles"][1],
                dip_angle=np.pi / 2,
                major_axis=r[1],
                minor_axis=r[1],
                major_axis_angle=0,
                num_points=num_points,
            )
        )
        center = np.array([x + 2 * dx, y, z])
        # Production fracture, always connected.
        self._fractures.append(
            pp.create_elliptic_fracture(
                center=center,
                strike_angle=params["strike_angles"][2],
                dip_angle=np.pi / 2,
                major_axis=r[2],
                minor_axis=r[2],
                major_axis_angle=0,
                num_points=num_points,
            )
        )

    def set_well_network(self) -> None:
        """Assign well network class."""
        if not self.params.get("use_wells", True):
            return super().set_well_network()
        dx = self.domain_sizes()[0] / 14
        y = self.domain_sizes()[1] / 2
        dy = self.domain_sizes()[1] / 2000
        z = -self.domain_sizes()[2] / 2
        x = self.domain_sizes()[0] * 6 / 14
        x_inj = np.array([[x, x, x]])
        x_prod = x_inj + 2 * dx
        y_both = np.array([[y + dy, y + dy, y - dy]])
        z_both = np.array([[0, z, z]])
        pts_prod = np.vstack([x_prod, y_both, z_both])
        # Create a well object.
        well_prod = pp.Well(pts_prod, tags={"well_name": self.production_well_names[0]})
        # y_inj = np.full((1, 3), y_max - 5e3)
        # z_inj = np.array([[0, z, z]])
        pts_inj = np.vstack([x_inj, y_both, z_both])

        x_inj = np.array([[x, x]])
        x_prod = x_inj + 2 * dx
        y_both = np.array([[y + 2 * dy, y - dy]])
        z_both = np.array([[0, 1.5 * z]])
        pts_prod = np.vstack([x_prod, y_both, z_both])
        # Create a well object.
        well_prod = pp.Well(pts_prod, tags={"well_name": self.production_well_names[0]})
        # y_inj = np.full((1, 3), y_max - 5e3)
        # z_inj = np.array([[0, z, z]])
        pts_inj = np.vstack([x_inj, y_both, z_both])
        # Create a well object.
        well_inj = pp.Well(
            pts_inj, tags={"well_name": self.injection_well_names[0]}
        )  # 68 20rd
        wells = [well_inj, well_prod]
        self.well_network = pp.WellNetwork3d(
            domain=self._domain, wells=wells, parameters={"mesh_size": 150.0}
        )


if __name__ == "__main__":

    class MockModel(
        CoolingGeometry, WellDataConceptual, CosoExporter, pp.MomentumBalance
    ): ...

    fracture_size = 1e3
    strike = 40  # degrees
    m = MockModel(
        {
            "grid_type": "simplex",
            "meshing_arguments": {"cell_size": 2e3, "cell_size_fracture": 5e2},
            "meshing_kwargs": {
                "refinement_buffer": 0.25,
                "farfield_transition": 0.10,
                "refinement_threshold": 0.15,
            },
            "file_name": "elliptic_fractures",
            "folder_name": "visualization/geometry",
            "use_wells": False,
            "domain_sizes": np.array([5e3, 5e3, 4e3]),
            "fracture_params": {
                "fracture_major_axes": np.array(
                    (fracture_size, fracture_size, fracture_size, fracture_size)
                ),
                "strike_angles": np.deg2rad([0, strike, 45]),
            },
        }
    )
    m.prepare_simulation()
    for sd in m.mdg.subdomains(dim=0):
        print(sd.tags)
        print(sd.cell_centers)
    print("Simulation prepared successfully.")
