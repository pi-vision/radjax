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
    pc,          # parsec [cm]
    m_co,        # molecular mass of CO [g] (or consistent units)
    m_mol_h      # mean molecular mass of hydrogen gass [g] (or consistent units)
)


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
        Unit vector of the line of sight (towards the observer), shape (3,) .
    nu0 : float
        Line rest frequency.
    pixel_area : float
        Pixel solid-angle * distance^2 in cm^2 (used for Jy conversion).

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
    doppler = -(1.0/cc) * jnp.sum(obs_dir * gas_v, axis=-1)

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
    const = hh * nu0 / (4 * jnp.pi)
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

# ----------------------------------------------------------------------------- #
# JIT wrappers & vectorized ops
# ----------------------------------------------------------------------------- #

# Parallelize over frequency axis across devices
compute_spectral_cube_pmap = jax.pmap(
    compute_spectral_cube,
    axis_name="freq",
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None),
)
# Vectorize over frequency axis on a single device
compute_spectral_cube_vmap = jax.vmap(
    compute_spectral_cube,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None),
)



__all__ = [
    "einstein_coefficients",
    "n_up_down",
    "compute_spectral_cube",
    "compute_spectral_cube_vmap",
    "compute_spectral_cube_pmap",
    "alpha_total",
]
