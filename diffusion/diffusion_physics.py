"""
Physics functions for vacancy diffusion in BCC alloys.

Pure calculations — no ASE, no MLIP, no file I/O. Takes raw numbers
from DiffusionResult and computes the diffusion coefficient D(T).

The diffusion coefficient for vacancy-mediated diffusion is:

    D(T) = f_0 * a^2 * nu_0 * c_v(T) * exp(-E_m / k_B T)

where:
    f_0   = geometric correlation factor (0.7272 for BCC)
    a     = nearest-neighbor jump distance = a_lat * sqrt(3) / 2
    nu_0  = attempt frequency (Debye frequency from elastic constants)
    c_v   = vacancy concentration = exp(-E_f^V / k_B T)
    E_m   = migration barrier from NEB
    k_B   = Boltzmann constant
    T     = temperature in Kelvin

References:
    - Mehrer, H. "Diffusion in Solids" (Springer, 2007)
    - Anderson, O.L. J. Phys. Chem. Solids 24, 909 (1963) — Debye temp from elastic constants
"""

import numpy as np
from typing import Union, Optional

# ------------------------------------------------------------------ #
#  Physical constants (SI)                                             #
# ------------------------------------------------------------------ #

K_BOLTZMANN = 1.380649e-23      # J/K
H_PLANCK = 6.62607015e-34       # J*s
EV_TO_JOULE = 1.602176634e-19   # J/eV
EV_TO_KJ_PER_MOL = 96.485      # kJ/(mol*eV)
ANGSTROM_TO_METER = 1e-10       # m/A
GPA_TO_PA = 1e9                 # Pa/GPa

# BCC lattice constants
F_0_BCC = 0.7272                # Correlation factor for BCC
Z_BCC = 8                       # Coordination number for BCC


# ------------------------------------------------------------------ #
#  Vacancy concentration                                               #
# ------------------------------------------------------------------ #

def vacancy_concentration(E_f_eV: float, T_K: float) -> float:
    """
    Equilibrium vacancy concentration c_v(T).

        c_v = exp(-E_f^V / k_B T)

    Parameters
    ----------
    E_f_eV : float
        Vacancy formation energy in eV.
    T_K : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Vacancy concentration (dimensionless, 0 to 1).
    """
    if T_K <= 0:
        raise ValueError(f"Temperature must be positive, got {T_K} K")
    return np.exp(-E_f_eV * EV_TO_JOULE / (K_BOLTZMANN * T_K))


# ------------------------------------------------------------------ #
#  Jump distance                                                       #
# ------------------------------------------------------------------ #

def jump_distance_m(lattice_parameter_A: float) -> float:
    """
    Nearest-neighbor jump distance in BCC lattice.

        a_jump = a_lat * sqrt(3) / 2

    Parameters
    ----------
    lattice_parameter_A : float
        Cubic lattice parameter in Angstrom.

    Returns
    -------
    float
        Jump distance in meters.
    """
    return lattice_parameter_A * np.sqrt(3) / 2 * ANGSTROM_TO_METER


# ------------------------------------------------------------------ #
#  Attempt frequency from elastic constants                            #
# ------------------------------------------------------------------ #

def debye_frequency(
    C11_GPa: float,
    C12_GPa: float,
    C44_GPa: float,
    density_kg_m3: float,
    volume_A3: float,
    n_atoms: int,
) -> float:
    """
    Debye frequency from cubic elastic constants (Anderson method).

    Steps:
    1. Bulk modulus:    B = (C11 + 2*C12) / 3
    2. Shear modulus:   G = Voigt-Reuss-Hill average
       G_Voigt = (C11 - C12 + 3*C44) / 5
       G_Reuss = 5*(C11 - C12)*C44 / (4*C44 + 3*(C11 - C12))
       G = (G_Voigt + G_Reuss) / 2
    3. Sound velocities:
       v_l = sqrt((B + 4G/3) / rho)     (longitudinal)
       v_t = sqrt(G / rho)               (transverse)
    4. Mean sound velocity:
       v_m = [ (1/3) * (1/v_l^3 + 2/v_t^3) ]^(-1/3)
    5. Debye frequency:
       nu_D = (v_m / (2*pi)) * (6*pi^2 * n/V)^(1/3)

    Parameters
    ----------
    C11_GPa, C12_GPa, C44_GPa : float
        Cubic elastic constants in GPa.
    density_kg_m3 : float
        Mass density in kg/m^3.
    volume_A3 : float
        Supercell volume in Angstrom^3.
    n_atoms : int
        Number of atoms in the supercell.

    Returns
    -------
    float
        Debye frequency in Hz.
    """
    # Convert to Pa
    C11 = C11_GPa * GPA_TO_PA
    C12 = C12_GPa * GPA_TO_PA
    C44 = C44_GPa * GPA_TO_PA
    rho = density_kg_m3

    # Bulk modulus
    B = (C11 + 2 * C12) / 3

    # Shear modulus (Voigt-Reuss-Hill average)
    G_voigt = (C11 - C12 + 3 * C44) / 5
    denom = 4 * C44 + 3 * (C11 - C12)
    if abs(denom) < 1e-10:
        raise ValueError("Degenerate elastic constants: 4*C44 + 3*(C11-C12) ~ 0")
    G_reuss = 5 * (C11 - C12) * C44 / denom
    G = (G_voigt + G_reuss) / 2

    # Sound velocities (m/s)
    v_l = np.sqrt((B + 4 * G / 3) / rho)
    v_t = np.sqrt(G / rho)

    # Mean sound velocity
    v_m = (1 / 3 * (1 / v_l**3 + 2 / v_t**3)) ** (-1 / 3)

    # Atomic number density (1/m^3)
    V_m3 = volume_A3 * ANGSTROM_TO_METER**3
    number_density = n_atoms / V_m3

    # Debye frequency
    nu_D = (v_m / (2 * np.pi)) * (6 * np.pi**2 * number_density) ** (1 / 3)

    return nu_D


# ------------------------------------------------------------------ #
#  Diffusion coefficient                                               #
# ------------------------------------------------------------------ #

def diffusion_coefficient(
    E_m_eV: float,
    E_f_eV: float,
    nu_0_Hz: float,
    jump_distance_A: float,
    T_K: Union[float, np.ndarray],
    f_0: float = F_0_BCC,
) -> Union[float, np.ndarray]:
    """
    Vacancy diffusion coefficient D(T).

        D = f_0 * a^2 * nu_0 * exp(-(E_f + E_m) / k_B T)

    Parameters
    ----------
    E_m_eV : float
        Migration barrier in eV (from NEB).
    E_f_eV : float
        Vacancy formation energy in eV.
    nu_0_Hz : float
        Attempt frequency in Hz (from Debye frequency).
    jump_distance_A : float
        Actual jump distance in Angstrom (from relaxed NEB endpoints).
    T_K : float or array
        Temperature(s) in Kelvin.
    f_0 : float, optional
        Correlation factor (default: 0.7272 for BCC).

    Returns
    -------
    float or ndarray
        Diffusion coefficient in m^2/s.
    """
    T_K = np.asarray(T_K, dtype=float)
    if np.any(T_K <= 0):
        raise ValueError("Temperature must be positive")

    a = jump_distance_A * ANGSTROM_TO_METER
    Q_eV = E_f_eV + E_m_eV  # Total activation energy
    Q_J = Q_eV * EV_TO_JOULE

    D = f_0 * a**2 * nu_0_Hz * np.exp(-Q_J / (K_BOLTZMANN * T_K))

    return float(D) if D.ndim == 0 else D


# ------------------------------------------------------------------ #
#  Convenience: compute D(T) directly from DiffusionResult             #
# ------------------------------------------------------------------ #

def diffusion_coefficient_from_result(
    result,  # DiffusionResult (no import to keep this module dependency-free)
    T_K: Union[float, np.ndarray],
    f_0: float = F_0_BCC,
) -> Union[float, np.ndarray]:
    """
    Compute D(T) directly from a DiffusionResult object.

    Extracts all needed quantities (barrier, E_f, elastic constants,
    jump distance, density) and computes D.

    Parameters
    ----------
    result : DiffusionResult
        Output from DiffusionOracle.calculate().
    T_K : float or array
        Temperature(s) in Kelvin.
    f_0 : float, optional
        Correlation factor (default: 0.7272 for BCC).

    Returns
    -------
    float or ndarray
        Diffusion coefficient in m^2/s.
    """
    nu_0 = debye_frequency(
        C11_GPa=result.C11_GPa,
        C12_GPa=result.C12_GPa,
        C44_GPa=result.C44_GPa,
        density_kg_m3=result.density_kg_m3,
        volume_A3=result.volume_A3,
        n_atoms=result.n_atoms,
    )

    return diffusion_coefficient(
        E_m_eV=result.forward_barrier_eV,
        E_f_eV=result.vacancy_formation_energy_eV,
        nu_0_Hz=nu_0,
        jump_distance_A=result.jump_distance_A,
        T_K=T_K,
        f_0=f_0,
    )


def activation_energy_eV(result) -> float:
    """Total activation energy Q = E_f^V + E_m (eV)."""
    return result.vacancy_formation_energy_eV + result.forward_barrier_eV


def activation_energy_kJ_per_mol(result) -> float:
    """Total activation energy Q = E_f^V + E_m (kJ/mol)."""
    return activation_energy_eV(result) * EV_TO_KJ_PER_MOL


def prefactor_m2_per_s(result, f_0: float = F_0_BCC) -> float:
    """
    Diffusion prefactor D_0 = f_0 * a^2 * nu_0 (m^2/s).

    D(T) = D_0 * exp(-Q / k_B T)
    """
    nu_0 = debye_frequency(
        C11_GPa=result.C11_GPa,
        C12_GPa=result.C12_GPa,
        C44_GPa=result.C44_GPa,
        density_kg_m3=result.density_kg_m3,
        volume_A3=result.volume_A3,
        n_atoms=result.n_atoms,
    )
    a = result.jump_distance_A * ANGSTROM_TO_METER
    return f_0 * a**2 * nu_0