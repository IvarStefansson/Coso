from iapws import IAPWS97
import numpy as np
from physical_model import fit_thermal_expansion


def iapws_density(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """IAPWS-IF97 saturated liquid density [kg/m³]. T in K, p ignored."""
    return np.array([IAPWS97(T=t, x=0).rho for t in T])


def temperature_at_depth(depths: np.ndarray) -> np.ndarray:
    g1, g2 = 150e-3, 20e-3  # K/m  (Davatzes & Hickman)
    return np.where(
        depths < 1100, 293.15 + g1 * depths, 293.15 + g1 * 1100 + g2 * (depths - 1100)
    )


T_ref = 20 + 273.15  # K  (reference_variable_values temperature)
p_ref = 101325.0  # Pa (atmospheric, == reference pressure)
rho_ref = IAPWS97(T=T_ref, P=p_ref * 1e-6).rho

c_T = fit_thermal_expansion(
    depth_max=3000.0,
    temperature_at_depth=temperature_at_depth,
    density=iapws_density,
    rho_ref=rho_ref,
    T_ref=T_ref,
    p_ref=p_ref,
)
print(f"Optimal c_T = {c_T:.4e} K^-1")
