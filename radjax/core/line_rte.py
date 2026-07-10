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
import matplotlib.pyplot as plt


from .grid import expand_dims
from .consts import (
    cc,          # speed of light      [cm/s]
    hh,          # Planck constant     [erg*s]
    kk,          # Boltzmann constant  [erg/K]
    pc,          # parsec [cm]
    m_co,        # molecular mass of CO [g] (or consistent units)
    m_mol_h,      # mean molecular mass of hydrogen gass [g] (or consistent units)
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
    distance_pc: float,
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
        Total line broadening [cm/s], shape matching gas_t / spatial field (npix, npix, nray,)
    n_up, n_dn : jnp.ndarray
        Upper/lower level populations, shape matching spatial field (npix, npix, nray,)
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
    dust_profile : jnp.ndarray, optional
        Dust extinction profile [cm^-1], shape matching spatial field (npix, npix, nray,)

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
    # obs_dir points toward the observer, so obs_dir·v > 0 means approaching
    # gas, which must appear blueshifted: nu_peak = nu0 * (1 + v_los/c).
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
    const = hh * nu0 / (4 * jnp.pi)
    j_nu  = const * n_up * a_ud  * line_profile
    a_nu  = const * (n_dn * b_du - n_up * b_ud) * line_profile

    # Ray trace through the volume to compute image intensities
    ray_ds = jnp.sqrt(jnp.sum(jnp.diff(ray_coords, axis=-2) ** 2, axis=-1))
    dtau = 0.5 * (a_nu[...,1:] + a_nu[...,:-1]) * ray_ds

    # First order interpolation of the source
    source_1st = 0.5 * (j_nu[...,1:] + j_nu[...,:-1]) * ray_ds

    # Second-order integration: I = beta * S_near + (1 - e^-dtau - beta) * S_far,
    # with index :-1 the near (observer-side) end of each segment. In the optically
    # thick limit beta -> 1 so the observer sees the near-side source function.
    s_nu = j_nu / (a_nu + 1e-30)   # Radmc3d has +1e-99 but this results in nans
    beta = (dtau - 1 + jnp.exp(-dtau)) / (dtau + 1e-30)
    beta = jnp.where(dtau > 1e-6, beta, 0.5*dtau)
    source_2nd = beta * s_nu[...,:-1] + (1 - jnp.exp(-dtau) - beta) * s_nu[...,1:]
    source_2nd = jnp.where(source_2nd < source_1st, source_2nd, source_1st)

    pad_width = [(0, 0)] * (dtau.ndim - 1) + [(1, 0)]
    attenuation = jnp.exp(-jnp.cumsum(jnp.pad(dtau, pad_width), axis=-1))[...,:-1]
    intensity = (source_2nd * attenuation).sum(axis=-1)

    # Conversion from erg/s/cm²/Hz/sr to Jy/pixel
    image_fluxes_jy = pixel_area / (distance_pc * pc)**2 * 1e23 * intensity
    return image_fluxes_jy


def planck_nu(nu, T):
    """Planck function B_nu(T) in erg/s/cm^2/Hz/sr."""
    exp_factor = (hh * nu) / (kk * T + 1e-30)
    return (2 * hh * nu**3 / cc**2) / (jnp.exp(exp_factor) - 1.0 + 1e-30)



def compute_dust_opacity(freq_hz):
    """
    Dust opacity for protoplanetary disks in sub-mm
    
    Parameters:
    -----------
    freq_hz : array
        Frequency in Hz
    
    Returns:
    --------
    kappa : array
        Dust opacity in cm²/g
    """
    nu0 = 3e11  # 300 GHz (1 mm) reference frequency
    kappa0 = 2.3  # cm²/g at 1 mm
    beta = 1.0  # opacity index for disks with grain growth
    
    kappa = kappa0 * (freq_hz / nu0)**beta
    return kappa



def compute_spectral_cube_with_dust(
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
    distance_pc: float,
    dust_alpha: jnp.ndarray,
    dust_temp: jnp.ndarray,
) -> jnp.ndarray:
    # When called via vmap, camera_freqs is a scalar (one frequency per call).
    # Avoid storing separate gas/dust intermediates — compute j_tot and a_tot directly
    # to minimize peak memory when all frequency channels are materialized simultaneously.

    # Gas line profile (doppler shift along LOS; approaching gas is blueshifted)
    doppler = (1.0 / cc) * jnp.sum(obs_dir * gas_v, axis=-1)
    dnu = camera_freqs - nu0 - doppler * nu0
    line_profile = (cc / (alpha_tot * nu0 * jnp.sqrt(jnp.pi))) * jnp.exp(
        -(cc * dnu / (nu0 * alpha_tot)) ** 2
    )

    const = hh * nu0 / (4 * jnp.pi)

    # Dust extinction coefficient: dust_alpha [g/cm³] × κ(ν) [cm²/g] → [cm⁻¹]
    # kappa_dust is a scalar at this frequency; no extra array allocation needed.
    kappa_dust = compute_dust_opacity(camera_freqs)
    a_dust = dust_alpha * kappa_dust  # (H, W, N)

    # Fused total emissivity and extinction — avoids storing j_gas, a_gas, j_dust separately
    j_tot = const * n_up * a_ud * line_profile + a_dust * planck_nu(camera_freqs, dust_temp)
    a_tot = const * (n_dn * b_du - n_up * b_ud) * line_profile + a_dust

    # Ray tracing
    ray_ds = jnp.sqrt(jnp.sum(jnp.diff(ray_coords, axis=-2) ** 2, axis=-1))
    dtau = 0.5 * (a_tot[..., 1:] + a_tot[..., :-1]) * ray_ds
    source_1st = 0.5 * (j_tot[..., 1:] + j_tot[..., :-1]) * ray_ds

    s_nu = j_tot / (a_tot + 1e-30)
    beta = (dtau - 1 + jnp.exp(-dtau)) / (dtau + 1e-30)
    beta = jnp.where(dtau > 1e-6, beta, 0.5 * dtau)
    source_2nd = beta * s_nu[..., :-1] + (1 - jnp.exp(-dtau) - beta) * s_nu[..., 1:]
    source_2nd = jnp.where(source_2nd < source_1st, source_2nd, source_1st)

    pad_width = [(0, 0)] * (dtau.ndim - 1) + [(1, 0)]
    attenuation = jnp.exp(-jnp.cumsum(jnp.pad(dtau, pad_width), axis=-1))[..., :-1]
    intensity = (source_2nd * attenuation).sum(axis=-1)

    return pixel_area / (distance_pc * pc)**2 * 1e23 * intensity



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


def compute_emission_height_cube(
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
    cutoff: float = 1e-13,
    ) -> jnp.ndarray:
    """
    Perform radiative transfer along rays to produce an cube of the z-coordinate of the emission height.

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
    cutoff : float
        Minimum intensity cutoff to consider a pixel valid (not background diffusion)
    """
        # Compute doppler shift (approaching gas is blueshifted)
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
    const = hh * nu0 / (4 * jnp.pi)
    j_nu  = const * n_up * a_ud  * line_profile
    a_nu  = const * (n_dn * b_du - n_up * b_ud) * line_profile

    # Ray trace through the volume
    ray_ds = jnp.sqrt(jnp.sum(jnp.diff(ray_coords, axis=-2) ** 2, axis=-1))
    dtau = 0.5 * (a_nu[...,1:] + a_nu[...,:-1]) * ray_ds

    z_raw = ray_coords[..., 2]
    is_forward = z_raw[0, 0, 0] > z_raw[0, 0, -1]
    # jnp.where instead of Python if/else so this is safe inside jax.lax.scan.
    dtau  = jnp.where(is_forward, dtau,  jnp.flip(dtau,  axis=-1))
    j_nu  = jnp.where(is_forward, j_nu,  jnp.flip(j_nu,  axis=-1))
    a_nu  = jnp.where(is_forward, a_nu,  jnp.flip(a_nu,  axis=-1))
    z_raw = jnp.where(is_forward, z_raw, jnp.flip(z_raw, axis=-1))
    
    # Mid-point Z-coordinates for the segments
    z_mid = 0.5 * (z_raw[..., 1:] + z_raw[..., :-1])

    # Second-order Source term (from your spectral cube function)
    s_nu = j_nu / (a_nu + 1e-30)
    beta = (dtau - 1 + jnp.exp(-dtau)) / (dtau + 1e-30)
    beta = jnp.where(dtau > 1e-6, beta, 0.5*dtau)
    source_term = beta * s_nu[...,:-1] + (1 - jnp.exp(-dtau) - beta) * s_nu[...,1:]

    # Calculate Attenuation (e^-tau) along the ray
    pad_width = [(0, 0)] * (dtau.ndim - 1) + [(1, 0)]
    attenuation = jnp.exp(-jnp.cumsum(jnp.pad(dtau, pad_width), axis=-1))[...,:-1]

    # Weighted contribution of each segment to the total intensity
    # Shape: (nfreq, npix, npix, nray-1)
    emission_weight = source_term * attenuation
    # Total intensity (the denominator)
    total_intensity = emission_weight.sum(axis=-1)
    # Weighted Z (the numerator)
    avg_z = (emission_weight * z_mid).sum(axis=-1) / (total_intensity + 1e-30)

    # Clean up optically thin pixels where no significant emission occurred
    return jnp.where(total_intensity > cutoff, avg_z, jnp.nan)


def compute_tau1_cube(
    camera_freqs: jnp.ndarray,
    gas_v: jnp.ndarray,
    alpha_tot: jnp.ndarray,
    n_up: jnp.ndarray,
    n_dn: jnp.ndarray,
    b_ud: float,
    b_du: float,
    ray_coords: jnp.ndarray,
    obs_dir: jnp.ndarray,
    nu0: float,
    ) -> jnp.ndarray:
    """
    Perform radiative transfer along rays to produce an cube with emitting height of tau=1 surface. 

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
    b_ud, b_du : float
        Einstein coefficients for the chosen transition.
    ray_coords : jnp.ndarray
        Ray coordinates along last-but-one axis, shape (npix, npix, nray, 3).
    obs_dir : jnp.ndarray
        Unit vector of the line of sight (towards the observer), shape (3,) .
    nu0 : float
        Line rest frequency.
    """

    doppler = (1.0/cc) * jnp.sum(obs_dir * gas_v, axis=-1)
    dnu = expand_dims(camera_freqs, alpha_tot.ndim + 1, axis=-1) - nu0 - doppler * nu0
    line_profile = (cc / (alpha_tot * nu0 * jnp.sqrt(jnp.pi))) * jnp.exp(
        -(cc * dnu / (nu0 * alpha_tot)) ** 2
    )
    # Compute emissivity (j_nu) and extinction (alpha_nu) for radiative transfer
    #     h nu_0                                          h nu_0
    # j = ------ n_up A_ud * phi(omega, nu)   ;  alpha = ------ ( n_down B_du - n_up B_ud ) * phi(omega, nu)
    #     4 pi                                            4 pi
    const = hh * nu0 / (4 * jnp.pi)
    a_nu  = const * (n_dn * b_du - n_up * b_ud) * line_profile

    # Ray trace through the volume to compute image intensities
    ray_ds = jnp.sqrt(jnp.sum(jnp.diff(ray_coords, axis=-2) ** 2, axis=-1))
    dtau = 0.5 * (a_nu[...,1:] + a_nu[...,:-1]) * ray_ds

    z_raw = ray_coords[..., 2] # (200, 200, 20)
    is_forward = z_raw[0, 0, 0] > z_raw[0, 0, -1]
    # Use jnp.where instead of a Python if/else so this is safe inside
    # jax.lax.scan / jit where is_forward is a traced boolean.
    dtau_ordered = jnp.where(is_forward, dtau, jnp.flip(dtau, axis=-1))
    z_ordered    = jnp.where(is_forward, z_raw, jnp.flip(z_raw, axis=-1))

    tau_sum = jnp.cumsum(dtau_ordered, axis=-1)
    # Find the FIRST point where tau >= 1
    mask = (tau_sum >= 1.0)
    # argmax returns the first index where mask is True
    first_tau_1_idx = jnp.argmax(mask, axis=-1)

    # Map back to Z
    z_surface = jnp.take_along_axis(z_ordered[None, ...], first_tau_1_idx[..., None] + 1, axis=-1).squeeze(-1)

    # Handle rays that never hit tau=1 (optically thin)
    has_reached_tau_1 = jnp.any(mask, axis=-1)
    z_final = jnp.where(has_reached_tau_1, z_surface, jnp.nan)
    return z_final

# ----------------------------------------------------------------------------- #
# JIT wrappers & vectorized ops
# ----------------------------------------------------------------------------- #

# Parallelize over frequency axis across devices
compute_spectral_cube_pmap = jax.pmap(
    compute_spectral_cube,
    axis_name="freq",
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None),
)
# Vectorize over frequency axis on a single device
compute_spectral_cube_vmap = jax.vmap(
    compute_spectral_cube,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None),
)

compute_tau1_cube_pmap = jax.pmap(
    compute_tau1_cube,
    axis_name="freq",
    in_axes=(0, None, None, None, None, None, None, None, None, None),
)

compute_tau1_cube_vmap = jax.vmap(
    compute_tau1_cube,
    in_axes=(0, None, None, None, None, None, None, None, None, None),
)

compute_emission_height_cube_pmap = jax.pmap(
    compute_emission_height_cube,
    axis_name="freq",
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None),
)   

compute_emission_height_cube_vmap = jax.vmap(
    compute_emission_height_cube,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None),
)

# The dust kernel expects a *scalar* frequency, so pmap over the device axis
# must wrap an inner vmap over the per-device frequency axis.
compute_spectral_cube_dust_pmap = jax.pmap(
    jax.vmap(
        compute_spectral_cube_with_dust,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
    ),
    axis_name="freq",
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
)

compute_spectral_cube_dust_vmap = jax.vmap(
    compute_spectral_cube_with_dust,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None, None, None, None),
)


def compute_spectral_cube_dust_scan(
    camera_freqs, gas_v, alpha_tot, n_up, n_dn,
    a_ud, b_ud, b_du, ray_coords, obs_dir, nu0, pixel_area,
    distance_pc, dust_alpha, dust_temp,
):
    """
    Memory-efficient alternative to vmap: process one frequency channel at a time
    using jax.lax.scan. Peak memory is O(H*W*N) instead of O(F*H*W*N).
    Slower than vmap but avoids OOM on large cubes.
    """
    def scan_fn(_, freq):
        image = compute_spectral_cube_with_dust(
            freq, gas_v, alpha_tot, n_up, n_dn,
            a_ud, b_ud, b_du, ray_coords, obs_dir, nu0, pixel_area,
            distance_pc, dust_alpha, dust_temp,
        )
        return None, image

    _, images = jax.lax.scan(scan_fn, None, camera_freqs)
    return images



__all__ = [
    "compute_spectral_cube",
    "compute_spectral_cube_vmap",
    "compute_spectral_cube_pmap",
    "compute_spectral_cube_with_dust",
    "compute_spectral_cube_dust_vmap",
    "compute_spectral_cube_dust_pmap",
    "compute_spectral_cube_dust_scan",
    "compute_tau1_cube",
    "compute_tau1_cube_vmap",
    "compute_tau1_cube_pmap",
    "compute_emission_height_cube",
    "compute_emission_height_cube_vmap",
    "compute_emission_height_cube_pmap",
    "alpha_total",
    "planck_nu",
    "compute_dust_opacity",
]
