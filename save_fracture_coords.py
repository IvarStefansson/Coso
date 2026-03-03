import numpy as np

"""
This script generates and saves fracture coordinates within a specified domain.

Functions:
    planes_to_csv_coords(domain, easting, northing, offset):
        Generates coordinates for planes based on the given domain, easting, northing, and offset.

Variables:
    domain (np.ndarray): The initial domain boundaries reshaped and scaled.
    fracture_box (np.ndarray): The fracture box boundaries derived from the domain.
    f0 (np.ndarray): Coordinates for the first fracture plane.
    f1 (np.ndarray): Coordinates for the second fracture plane.
    coords (np.ndarray): Stacked coordinates of all fractures.

Usage:
    The script calculates the coordinates of fractures within a specified domain and saves them to a file named "data/coords.txt".
"""


def planes_to_csv_coords(domain, easting, northing, offset):
    """
    Generate coordinates for planes and format them for CSV output.

    Parameters:
        domain (numpy.ndarray): A 2x2 array defining the domain boundaries.
        easting (float): The easting coefficient for the plane equation.
        northing (float): The northing coefficient for the plane equation.
        offset (float): The offset for the plane equation.

    Returns:
    numpy.ndarray: A 1D array of coordinates formatted for CSV output.
    """
    x = np.linspace(domain[0, 0], domain[0, 1], 2)
    y = np.linspace(domain[1, 0], domain[1, 1], 2)
    E, N = np.meshgrid(x, y)

    # Calculate z coordinates
    Z = -(easting * E + northing * N + offset)
    # Ravel and stack the coordinates into an 3 by 4 array
    coords = np.array([E.ravel("F"), N.ravel("F"), Z.ravel("F")])
    return coords.ravel("F")


def easting_northing_offset_to_strike_angle_dip_angle(
    easting: float, northing: float, offset: float
) -> tuple[float, float]:
    """Convert easting, northing, and offset to strike angle and dip angle.

    Given the coefficients of a plane equation in the form of Ax + By + Cz + D = z,
    this function calculates the strike angle and dip angle of the plane.

    Parameters:
        easting: The easting coefficient for the plane equation.
        northing: The northing coefficient for the plane equation.
        offset: The offset for the plane equation.

    Returns:
        A tuple containing the strike angle and dip angle.
    """
    # Calculate strike angle
    strike = np.arctan2(northing, easting)
    # Calculate dip angle
    dip = np.arctan2(np.sqrt(easting**2 + northing**2), offset)
    return strike, dip


domain = np.array([-5, -40, -35, 20, -20, 2]).reshape((1, -1)) * 100.0


# fracture_box = np.array([[0, 15], [-35, -25]]) * 100.0
fracture_box = np.array([[domain[0, 0], domain[0, 3]], [domain[0, 1], domain[0, 4]]])
if __name__ == "__main__":
    f0 = planes_to_csv_coords(fracture_box, -1.865, -1.521, -1819)
    f1 = planes_to_csv_coords(fracture_box, -0.903, 1.738, 7939)
    # Extend domain to get internal fractures surrounded by a bigger box.
    domain[0, 0:2] -= 1e3
    domain[0, 3:5] += 1e3
    # Stack fractures
    coords = np.vstack([f0, f1])
    # Save domain and fractures to one file
    with open("data/coords.txt", "w") as f:
        np.savetxt(f, domain, delimiter=",")
        np.savetxt(f, coords, delimiter=",")
