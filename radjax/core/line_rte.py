"""
Core radiative transfer routines for line emission.

This module provides:
- Einstein coefficients utilities for a chosen transition
- Partition-based level populations (n_up, n_dn)
- A differentiable radiative transfer integrator for spectral cubes

All functions are JAX-friendly and can be vmapped/pmapped.
"""
from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp

from .grid import expand_dims
from .consts import (
    cc,          # speed of light      [cm/s]
    hh,          # Planck constant     [erg*s]
    kk,          # Boltzmann constant  [erg/K]
    ghz,         # GHz -> Hz (or consistent internal units)
    pc,          # parsec [cm]
    m_co,        # molecular mass of CO [g] (or consistent units)
    m_mol_h      # mean molecular mass of hydrogen gass [g] (or consistent units)
)

def einstein_coefficients(
    energy_levels: jnp.ndarray,
    radiative_transitions: jnp.ndarray,
    transition: int = 3,
) -> Tuple[float, float, float, float]:
    """
    Compute Einstein coefficients for a specific radiative transition.

    Parameters
    ----------
    energy_levels : jnp.ndarray
        Shape (num_levels, 4). Columns:
        [LEVEL, ENERGIES(cm^-1), WEIGHT, J]
    radiative_transitions : jnp.ndarray
        Shape (num_levels-1, 6). Columns:
        [TRANS, UP, LOW, EINSTEINA(s^-1), FREQ(GHz), E_u(K)]
    transition : int, optional
        1-based index of the line (e.g., 3 for CO(3-2)), by default 3.

    Returns
    -------
    (nu0, a_ud, b_ud, b_du) : Tuple[float, float, float, float]
        nu0 : line rest frequency (in your internal frequency units)
        a_ud : Einstein A (spontaneous emission)
        b_ud : Einstein B (stimulated emission, up->down)
        b_du : Einstein B (absorption, down->up)

    Notes
    -----
    - gratio = weight_UP / weight_LOW
    - b_ud = (c^2 / (2 h nu0^3)) * a_ud
    - b_du = b_ud * gratio
    
    See: RADMC-3D line RT docs for background:
    https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/lineradtrans.html
    """
    # extract up and down indices from radiative transition table (e.g. 4,3 for second CO line at 345GHz)
    line_index = transition - 1
    nu0 = radiative_transitions[line_index, 4] * ghz
    up_idx = jnp.array(radiative_transitions[line_index, 1] - 1, int)
    dn_idx = jnp.array(radiative_transitions[line_index, 2] - 1, int)

    gratio = energy_levels[up_idx, 2] / energy_levels[dn_idx, 2]
    a_ud = radiative_transitions[line_index, 3]
    b_ud = (cc**2 / (2 * hh * nu0**3)) * a_ud
    b_du = b_ud * gratio
    return nu0, a_ud, b_ud, b_du

def n_up_down(
    gas_nd: jnp.ndarray,
    gas_t: jnp.ndarray,
    energy_levels: jnp.ndarray,
    radiative_transitions: jnp.ndarray,
    transition: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    LTE level populations for the chosen transition.

    Parameters
    ----------
    gas_nd : jnp.ndarray
        Number density (same shape as gas_t). 
        example: (num_pix, num_pix, n_ray)
    gas_t : jnp.ndarray
        Temperature in Kelvin (same shape as gas_nd). 
        example: (npix, npix, nray)
    energy_levels : jnp.ndarray
        Shape (num_levels, 4). See `einstein_coefficients`.
    radiative_transitions : jnp.ndarray
        Shape (num_levels-1, 6). See `einstein_coefficients`.
    transition : int, optional
        1-based index of the line, by default 3.

    Returns
    -------
    n_up, n_dn : Tuple[jnp.ndarray, jnp.ndarray]
        Up- and down-level number densities (same shape as gas_t/gas_nd).
    """
    # Temperature grid for partition function (log-spaced)
    ntemp, temp0, temp1 = 1000, 0.1, 100000.0
    partition_temp = temp0 * (temp1 / temp0) ** (jnp.arange(ntemp) / (ntemp - 1))

    # Partition function from available levels
    dendivk = (hh * cc / kk) * jnp.diff(energy_levels[:, 1])
    dummy = jnp.exp(-dendivk[None] / partition_temp[:, None]) * (
        energy_levels[1:, 2] / energy_levels[:-1, 2]
    )
    partition_fn = energy_levels[0, 2] + jnp.sum(jnp.cumprod(dummy, axis=-1), axis=-1)
    pfunc = jnp.interp(gas_t, partition_temp, partition_fn)

    # Identify up/down indices for chosen transition
    line_index = transition - 1
    up_idx = jnp.array(radiative_transitions[line_index, 1] - 1, int)
    dn_idx = jnp.array(radiative_transitions[line_index, 2] - 1, int)

    dendivk_up = (hh * cc / kk) * (energy_levels[up_idx, 1] - energy_levels[0, 1])
    dendivk_dn = (hh * cc / kk) * (energy_levels[dn_idx, 1] - energy_levels[0, 1])
    levelweight_up = energy_levels[up_idx, 2]
    levelweight_dn = energy_levels[dn_idx, 2]

    n_up = (gas_nd / pfunc) * jnp.exp(-dendivk_up / gas_t) * levelweight_up
    n_dn = (gas_nd / pfunc) * jnp.exp(-dendivk_dn / gas_t) * levelweight_dn
    return n_up, n_dn

def compute_spectral_cube(
    camera_freqs: jnp.ndarray,
    gas_v: jnp.ndarray,
    alpha_tot: jnp.ndarray,
    n_up: jnp.ndarray,
    n_dn: jnp.ndarray,
    a_ud: float,
    b_ud: float,
    b_du: float,
    ray_coords: jnp.ndarray,
    obs_dir: jnp.ndarray,
    nu0: float,
    pixel_area: float,
) -> jnp.ndarray:
    """
    Perform radiative transfer along rays to produce an image cube. Output units are Jy/pixel.

    Parameters
    ----------
    camera_freqs : jnp.ndarray
        Frequencies along the spectral axis, shape (nfreq,).
    gas_v : jnp.ndarray
        Gas velocity field [cm/s], shape (npix, npix, nray, 3).
    alpha_tot : jnp.ndarray
        Total line broadening [cm/s], shape matching gas_t / spatial field (npix, npix, nray,.
    n_up, n_dn : jnp.ndarray
        Upper/lower level populations, shape matching spatial field (npix, npix, nray,.
    a_ud, b_ud, b_du : float
        Einstein coefficients for the chosen transition.
    ray_coords : jnp.ndarray
        Ray coordinates along last-but-one axis, shape (npix, npix, nray, 3).
    obs_dir : jnp.ndarray
        Unit vector of the line of sight, shape (3,).
    nu0 : float
        Line rest frequency.
    pixel_area : float
        Pixel solid-angle * distance^2 in cm^2 at 1 pc (used for Jy conversion).

    Returns
    -------
    image_fluxes_jy : jnp.ndarray
        Image plane fluxes, shape (nfreq, ...) in Jy/pixel.

    Notes
    -----
    Second order integration of the source
    More info: https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/imagesspectra.html#sec-second-order
    """
    # Compute doppler shift
    # Note: doppler positive means moving toward observer, hence the minus sign
    doppler = (1.0/cc) * jnp.sum(obs_dir * gas_v, axis=-1)

    # Define a vector line profile over multiple camera frequencies 
    # The line profile is a Gaussian with width (alpha) and shifted by doppler
    dnu = expand_dims(camera_freqs, alpha_tot.ndim + 1, axis=-1) - nu0 - doppler * nu0
    line_profile = (cc / (alpha_tot * nu0 * jnp.sqrt(jnp.pi))) * jnp.exp(
        -(cc * dnu / (nu0 * alpha_tot)) ** 2
    )
    
    # Compute emissivity (j_nu) and extinction (alpha_nu) for radiative transfer
    #     h nu_0                                          h nu_0
    # j = ------ n_up A_ud * phi(omega, nu)   ;  alpha = ------ ( n_down B_du - n_up B_ud ) * phi(omega, nu)
    #     4 pi                                            4 pi
    const = hh * nu0 / (4 * np.pi)
    j_nu  = const * n_up * a_ud  * line_profile
    a_nu  = const * (n_dn * b_du - n_up * b_ud) * line_profile

    # Ray trace through the volume to compute image intensities
    ray_ds = jnp.sqrt(jnp.sum(jnp.diff(ray_coords, axis=-2) ** 2, axis=-1))
    dtau = 0.5 * (a_nu[...,1:] + a_nu[...,:-1]) * ray_ds

    # First order interpolation of the source
    source_1st = 0.5 * (j_nu[...,1:] + j_nu[...,:-1]) * ray_ds
    
    # Second-order integration
    s_nu = j_nu / (a_nu + 1e-30)   # Radmc3d has +1e-99 but this results in nans
    beta = (dtau - 1 + jnp.exp(-dtau)) / (dtau + 1e-30)
    beta = jnp.where(dtau > 1e-6, beta, 0.5*dtau)
    source_2nd = (1 - jnp.exp(-dtau) - beta) * s_nu[...,:-1] + beta * s_nu[...,1:]
    source_2nd = jnp.where(source_2nd < source_1st, source_2nd, source_1st)

    pad_width = [(0, 0)] * (dtau.ndim - 1) + [(1, 0)]
    attenuation = jnp.exp(-jnp.cumsum(jnp.pad(dtau, pad_width), axis=-1))[...,:-1]
    intensity = (source_2nd * attenuation).sum(axis=-1)

    # Conversion from erg/s/cm/cm/ster to Jy/pixel
    image_fluxes_jy =  pixel_area / pc**2 * 1e23 * intensity
    return image_fluxes_jy

# Vectorize over frequency axis on a single device
compute_spectral_cube_vmap = jax.vmap(
    compute_spectral_cube,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None),
)

# Parallelize over frequency axis across devices
compute_spectral_cube_pmap = jax.pmap(
    compute_spectral_cube,
    axis_name="freq",
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None),
)

def alpha_total(
    v_turb: float,
    gas_t: jnp.ndarray,
    m_mol: float = m_co,
) -> jnp.ndarray:
    """
    Total line broadening from thermal + (scaled) turbulent velocities.

    Parameters
    ----------
    v_turb : float
        Dimensionless turbulence parameter (scales local sound speed).
    gas_t : jnp.ndarray
        Temperature (K). Shape: (npix, npix, nray)
    m_mol : float, optional
        Molecular mass (defaults to CO).

    Returns
    -------
    alpha_tot : jnp.ndarray
        Total broadening [cm/s].

    Notes
    -----
    Note that m_mol_h is the mean molecular weight of the hydrogen background gas ~2.34 * m_h
    """
    alpha_therm_sq = 2 * kk * gas_t / m_mol
    cs_sq = 2 * kk * gas_t / m_mol_h
    alpha_tot = jnp.sqrt(alpha_therm_sq + (v_turb**2) * cs_sq)
    return alpha_tot


__all__ = [
    "einstein_coefficients",
    "n_up_down",
    "compute_spectral_cube",
    "compute_spectral_cube_vmap",
    "compute_spectral_cube_pmap",
    "alpha_total",
]
