import matplotlib.pyplot as plt
import numpy as np


def plot_plane(easting, northing, offset, ax, color="blue", label=""):
    # Generate x and y coordinates
    t, b = 2, -4
    x = np.linspace(b, t, 2)
    y = np.linspace(b, t, 2)
    E, N = np.meshgrid(x, y)

    # Calculate z coordinates
    Z = easting * E + northing * N + offset
    print(E, N, Z)
    # Ravel and stack the coordinates into an 3 by 4 array
    coords = np.array([E.ravel(), N.ravel(), Z.ravel()]).T
    print(coords)
    # Save the coordinates to a file
    np.savetxt("data/coords.txt", coords, delimiter=",", header="Easting, Northing, Z")
    # Plot the plane
    ax.plot_surface(E, N, Z, alpha=0.5, color=color, label=label)

    # Set labels and title
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.set_zlabel("Z")
    # ax.set_title("Plane: z = easting + northing + offset")


# Shut-in seismicity fracture plane: -1.865*easting - 1.521*northing - 1.819 = z
# Fracture plane for non-shut-in periods: -0.903*easting +1.738*northing + 7.939 = z
# Plot the two planes
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

plot_plane(-1.865, -1.521, -1.819, ax, "blue", label="shut-in")
plot_plane(-0.903, 1.738, 7.939, ax, "red", label="non-shut-in")

# Add legend
ax.legend()
# Save the plot
plt.savefig("figures/fracture_planes.png")
plt.show()
