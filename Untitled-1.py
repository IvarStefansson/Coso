def mu(T):
    A = 2.414e-5
    B = 247.8

    C = 140

    mu_ = A * 10 ** (B / (T - C))
    return mu_


def Celsius_to_Kelvin(T_C):
    return T_C + 273.15


def rho(temperature, pressure):
    T = temperature - Celsius_to_Kelvin(20)

    rho_ref = 1000 - 0.07 * T - 0.0002 * T**2
    dp = pressure - 1e5

    # Wrap compressibility from fluid class as matrix (left multiplication with dp).
    c = 4.5e-10
    rho_ = rho_ref * (1 + c * dp)
    rho_.set_name("fluid_density_from_pressure_and_temperature")
    return rho_
