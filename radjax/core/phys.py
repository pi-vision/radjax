"""
Disk physics calculations

This module provides:
  - CO abundance thresholding (explicit chemistry knobs),
  - Keplerian + pressure-supported azimuthal velocity,
  - Vertical hydrostatic number density,
  - Column integration,
  - Helper terms (v_K^2, pressure gradient) and a vector lift to (vx,vy,vz).

Conventions
-----------
- r, z in AU
- Masses in grams (consistent with G)
- Temperatures in K
- Velocities in m/s
- Columns in cm^-2; number densities in cm^-3

These functions are intentionally independent of any config/dataclass so they
can be called from different model families (e.g., models/broken_power_law.py).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# constants module must define: G [cgs], au [cm], kk [erg/K]
from .consts import G, au, kk


# ----------------------------------------------------------------------------- #
# Kinematics
# ----------------------------------------------------------------------------- #
def velocity_profile(
    z: jnp.ndarray,
    r: jnp.ndarray,
    nd: jnp.ndarray,
    temperature: jnp.ndarray,
    *,
    M_star: float,
    m_mol_h: float,
    pressure_correction: bool = False,
) -> jnp.ndarray:
    """
    Azimuthal velocity v_phi(z, r). Optionally includes pressure support.

    Parameters
    ----------
    z, r : jnp.ndarray
        Height and radius grids [AU].
    nd : jnp.ndarray
        Number density n_H2 [cm^-3].
    temperature : jnp.ndarray
        Temperature [K].
    M_star : float
        Stellar mass [g].
    m_mol_h : float
        Molecular mass [g].
    pressure_correction : bool
        If True, adds sqrt(v_K^2 + v_p^2). Else sqrt(v_K^2) only.

    Returns
    -------
    jnp.ndarray
        v_phi [m/s], broadcastable to r (and z).
    """
    if pressure_correction:
        v = jnp.sqrt(
            vsq_keplerian_vertical(z, r, M_star) +
            vsq_pressure_grad(r, nd, temperature, m_mol_h)
        )
        return jnp.nan_to_num(v)
    else:
        return jnp.sqrt(vsq_keplerian_vertical(z, r, M_star))

# ----------------------------------------------------------------------------- #
# Vertical structure
# ----------------------------------------------------------------------------- #
def number_density_profile(
    z: jnp.ndarray,
    r: jnp.ndarray,
    temperature: jnp.ndarray,
    *,
    gamma: float,
    r_in_au: float,
    r_c_au: float,
    M_gas: float,
    M_star: float,
    m_mol_h: float,
) -> jnp.ndarray:
    """
    Hydrostatic n_H2(z, r) using discrete integration in z.

    Surface density:
        Σ(r) ∝ (r/r_c)^(-γ) * exp[-(r/r_c)^(2−γ)]
    Then integrate hydrostatic balance along z with trapezoid-like averaging.

    Parameters
    ----------
    z, r : jnp.ndarray
        Height and radius grids [AU]. z varies along axis=0.
    temperature : jnp.ndarray
        Temperature [K], broadcastable to (z, r).
    gamma : float
    r_in_au : float
    r_c_au : float
    M_gas : float
    M_star : float
    m_mol_h : float

    Returns
    -------
    jnp.ndarray
        n_H2 [cm^-3] matching (z, r) broadcasting.
    """
    sigma_0 = (2.0 - gamma) * (M_gas / (2.0 * jnp.pi * (r_c_au * au) ** 2)) * jnp.exp(r_in_au / r_c_au) ** (2.0 - gamma)
    sigma   = sigma_0 * (r[0] / r_c_au) ** (-gamma) * jnp.exp(-(r[0] / r_c_au) ** (2.0 - gamma))

    dz      = jnp.diff(z * au, axis=0)
    dlogrho = (au**(-2) * G * M_star * z / (r**2 + z**2) ** 1.5) * (m_mol_h / (kk * temperature))
    dlogrho = -0.5 * (dlogrho[:-1] + dlogrho[1:]) - jnp.log(temperature[1:] / temperature[:-1]) / dz

    logrho  = jnp.cumsum(dlogrho * dz, axis=0)
    logrho  = jnp.pad(logrho, pad_width=[(1, 0), (0, 0)])
    rho     = jnp.exp(logrho)
    rho     = sigma * rho / (1.0 + jnp.sum(rho[1:] * dz, axis=0, keepdims=True))
    return rho / m_mol_h


def surface_density(z: jnp.ndarray, nd: jnp.ndarray) -> jnp.ndarray:
    """
    Column N_H2(z, r) [cm^-2] by cumulative sum (rectangle rule).

    Parameters
    ----------
    z : jnp.ndarray
        Height grid [AU]; uniform spacing assumed along axis=0.
    nd : jnp.ndarray
        n_H2 [cm^-3].

    Returns
    -------
    jnp.ndarray
        Column density with same (r, ...) shape as nd (excluding z-axis).
    """
    dz = jnp.diff(z * au, axis=0)[0, 0]
    return jnp.cumsum(nd[::-1] * dz, axis=0)[::-1]


def vsq_keplerian_vertical(
    z: jnp.ndarray,
    r: jnp.ndarray,
    M_star: float,
) -> jnp.ndarray:
    """
    Squared Keplerian velocity with vertical correction.

    v_K^2 = G M_star r^2 / (r^2 + z^2)^(3/2)

    Parameters
    ----------
    z : jnp.ndarray
        Height grid [AU].
    r : jnp.ndarray
        Radius grid [AU].
    M_star : float
        Stellar mass [g].

    Returns
    -------
    jnp.ndarray
        Squared speed [m^2/s^2].
    """
    return G * M_star * (r * au) ** 2 / ((r * au) ** 2 + (z * au) ** 2) ** 1.5

def vsq_pressure_grad(
    r: jnp.ndarray,
    nd: jnp.ndarray,
    temperature: jnp.ndarray,
    m_mol_h: float,
) -> jnp.ndarray:
    """
    Pressure support contribution in squared-velocity units:
        (r / ρ) * dP/dr

    Parameters
    ----------
    r : jnp.ndarray
        Radius grid [AU].
    nd : jnp.ndarray
        Number density n_H2 [1/cm^3].
    temperature : jnp.ndarray
        Temperature [K].
    m_mol : float
        Molecular mass [g] (e.g. mean molecular weight of H2/CO).
        
    Returns
    -------
    jnp.ndarray
        Squared-velocity term [m^2/s^2] shaped like r.
    """
    rho = m_mol_h * nd
    dr  = jnp.diff(r * au, axis=1)[0, 0]
    P   = nd * kk * temperature
    dP  = (P[:, 1:] - P[:, :-1]) / dr
    dP  = jnp.pad(dP, pad_width=[(0, 0), (1, 0)])
    return ((r * au) / rho) * dP


def azimuthal_velocity(
    coords: jnp.ndarray,
    v_phi: jnp.ndarray,
) -> jnp.ndarray:
    """
    Lift scalar azimuthal speed into 3D velocity components.

    Parameters
    ----------
    coords : jnp.ndarray
        Array of shape (..., 3) containing disk-frame coordinates [AU].
        Only x=coords[...,0] and y=coords[...,1] are used.
    v_phi : jnp.ndarray
        Scalar azimuthal speed [m/s], broadcastable to coords[...,0].

    Returns
    -------
    jnp.ndarray
        Velocity field (..., 3) [m/s], with components (vx, vy, vz=0).
    """
    ray_r = jnp.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2)
    ray_r = jnp.where(ray_r == 0.0, 1.0, ray_r)  # avoid NaNs at r=0
    cosp  = coords[..., 1] / ray_r
    sinp  = coords[..., 0] / ray_r
    vy    = v_phi * cosp
    vx    = v_phi * sinp
    vz    = jnp.zeros_like(v_phi)
    return jnp.stack([vy, vx, vz], axis=-1)


# ----------------------------------------------------------------------------- #
# Pre-jitted wrappers (same signatures; mark bools static where needed)
# ----------------------------------------------------------------------------- #
velocity_profile_jit        = jax.jit(velocity_profile,      static_argnames=("pressure_correction",))
number_density_profile_jit  = jax.jit(number_density_profile)
surface_density_jit         = jax.jit(surface_density)
vsq_keplerian_vertical_jit  = jax.jit(vsq_keplerian_vertical)
vsq_pressure_grad_jit       = jax.jit(vsq_pressure_grad)
azimuthal_velocity_jit      = jax.jit(azimuthal_velocity)
