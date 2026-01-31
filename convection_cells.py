#!/usr/bin/env python3
"""
Fracture convection onset + characteristic velocity scale (Darcy) using:
- Rayleigh–Darcy onset criterion: Ra_crit = 4*pi^2
- Temperature difference along the fracture: ΔT = G * H
- Cubic-law permeability: K = b^2 / 12

Onset (aperture thickness b):
Ra = (rho*g*beta*G*H/(12*mu*alpha)) * b^3
=> b_crit(H) = [12*mu*alpha*Ra_crit / (rho*g*beta*G*H)]^(1/3)

Velocity scale once convection exists:
v_scale ~ (K/mu) * (rho*g*beta*ΔT/H) = (K/mu) * (rho*g*beta*G)
=> v_scale(b) = (rho*g*beta*G/(12*mu)) * b^2

Interpretation:
- v_scale is a characteristic Darcy velocity for convection rolls.
- For a fracture, porosity ~1 so pore velocity ~ Darcy velocity.
"""

import numpy as np
import matplotlib.pyplot as plt

# Physical / model parameters
g = 9.81
Ra_crit = 4 * np.pi**2  # 39.48...

# Water properties near ~20°C
rho = 998.0  # kg/m^3
mu = 1.0e-3  # Pa*s
beta = 2.1e-4  # 1/K
alpha = 1.4e-7  # m^2/s (thermal diffusivity)

# Thermal gradient along fracture
G = 0.05  # K/m


def b_crit(H, rho=rho, mu=mu, beta=beta, alpha=alpha, g=g, G=G, Ra_crit=Ra_crit):
    """Critical fracture aperture b [m] for onset as a function of fracture
    length/height H [m]."""
    H = np.asarray(H, dtype=float)
    return (12.0 * mu * alpha * Ra_crit / (rho * g * beta * G * H)) ** (1.0 / 3.0)


def v_scale_from_b(b, rho=rho, mu=mu, beta=beta, g=g, G=G):
    """Characteristic Darcy velocity scale [m/s] for convection in a cubic-law
    fracture.

    v ~ (rho*g*beta*G/(12*mu)) * b^2
    """
    b = np.asarray(b, dtype=float)
    return (rho * g * beta * G / (12.0 * mu)) * b**2


def period_from_H_v(H, v):
    """Estimate convection cell period [s] from fracture height H [m] and
    characteristic velocity v [m/s].

    Assumes fluid circulates a distance ~ 2*H (up and down).
    """
    H = np.asarray(H, dtype=float)
    v = np.asarray(v, dtype=float)
    return 2.0 * H / v


def plot_b_crit_vs_H(H_vals, bcrit_vals):
    """Plot critical aperture vs fracture height."""
    plt.figure(figsize=(7, 4.5))
    plt.loglog(H_vals, bcrit_vals * 1e3, lw=2)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("Fracture height/length H (m)")
    plt.ylabel("Critical aperture b_crit (mm)")
    plt.title("Onset of convection in a fracture (Rayleigh–Darcy + cubic law)")
    plt.tight_layout()


def plot_velocity_vs_H(H_vals, v_at_onset):
    """Plot velocity scale at onset vs fracture height."""
    plt.figure(figsize=(7, 4.5))
    plt.loglog(H_vals, v_at_onset, lw=2, color="darkorange")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("Fracture height/length H (m)")
    plt.ylabel("Velocity scale at onset v_scale(b_crit) (m/s)")
    plt.title("Characteristic Darcy velocity scale evaluated at onset")
    plt.tight_layout()


def plot_period_vs_H(H_vals, period_at_onset):
    """Plot convection period at onset vs fracture height."""
    plt.figure(figsize=(7, 4.5))
    plt.loglog(H_vals, period_at_onset / 3600, lw=2, color="purple")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("Fracture height/length H (m)")
    plt.ylabel("Convection period at onset (hours)")
    plt.title("Convection cell period evaluated at onset")
    plt.tight_layout()


def plot_velocity_vs_aperture():
    """Plot velocity scale vs aperture for a range of apertures."""
    b_vals = np.logspace(-5, -1, 200)  # 10 µm to 10 cm
    v_vals = v_scale_from_b(b_vals)

    plt.figure(figsize=(7, 4.5))
    plt.loglog(b_vals * 1e3, v_vals, lw=2, color="seagreen")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("Aperture b (mm)")
    plt.ylabel("Velocity scale v_scale(b) (m/s)")
    plt.title("Velocity scale vs aperture (cubic-law fracture)")
    plt.tight_layout()


def main():
    # Range of fracture heights (H) and compute b_crit(H)
    H_vals = np.logspace(-1, 2, 800)  # 0.1 m to 100 m
    bcrit_vals = b_crit(H_vals)  # meters

    # Velocity evaluated at onset (i.e., using b = b_crit(H))
    v_at_onset = v_scale_from_b(bcrit_vals)  # m/s

    # Period of convection cell at onset
    period_at_onset = period_from_H_v(H_vals, v_at_onset)  # seconds

    # Print sample values
    print("Sample values:")
    for H in [2**i for i in range(-1, 10)]:  # 0.25 m to 512 m
        bc = float(b_crit(H))
        v = float(v_scale_from_b(bc))
        T = float(period_from_H_v(H, v))
        print(
            f"H = {H:>6.2f} m  ->  b_crit = {bc * 1e3:>7.3f} mm  |  "
            f"v_scale = {v:>9.3e} m/s  |  period = {T:>9.1f} s ({T / 3600:>6.2f} hr)"
        )

    # Generate plots
    plot_b_crit_vs_H(H_vals, bcrit_vals)
    plot_velocity_vs_H(H_vals, v_at_onset)
    plot_period_vs_H(H_vals, period_at_onset)
    plot_velocity_vs_aperture()

    plt.show()


if __name__ == "__main__":
    main()
