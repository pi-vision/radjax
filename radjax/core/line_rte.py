"""
Core radiative transfer routines for line emission.

This module provides:
- Einstein coefficients utilities for a chosen transition
- Partition-based level populations (n_up, n_dn)
- A differentiable radiative transfer integrator for spectral cubes

All functions are JAX-friendly and can be vmapped/pmapped.
"""
from __future__ import annotations

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





def _segment_emission(j_nu: jnp.ndarray, a_nu: jnp.ndarray, ray_ds: jnp.ndarray):
    """
    Exact constant-property radiative transfer solution per ray segment.

    For each segment the emissivity/extinction endpoints are combined into an
    extinction-weighted source function
        S_seg = (j1 + j2) / (a1 + a2)
    and the segment's emergent contribution (before attenuation by material in
    front of it) is the exact solution for a uniform slab:
        contrib = S_seg * (1 - exp(-dtau)).

    Why extinction-weighted instead of endpoint-interpolated (the previous
    RADMC-3D-style second-order scheme): when a segment straddles a sharp
    tau=1 surface, the observer-side endpoint sits in near-vacuum where
    S = j/a is ~0/0, and any scheme that weights toward that endpoint in the
    optically thick limit inherits a garbage source function — producing O(1)
    banding/streaking that flips with the sub-segment phase of the sampling
    grid. The extinction-weighted S_seg instead limits to j2/a2 = B(T) of the
    dense side (temperature is smooth across the surface even though opacity
    is not), making the emergent intensity insensitive to where the surface
    sits within the segment. In smooth regions S_seg equals the midpoint
    source function to O(ds^2), and for optically thin segments
    contrib -> 0.5*(j1+j2)*ds (the trapezoid rule), so nothing is lost in the
    line wings.

    Parameters
    ----------
    j_nu, a_nu : jnp.ndarray
        Emissivity and extinction at the sample points, shape (..., nray).
        Ordering along the last axis must be observer-side first.
    ray_ds : jnp.ndarray
        Segment lengths [cm], shape (..., nray-1). Non-uniform spacing is fine.

    Returns
    -------
    contrib : jnp.ndarray
        Per-segment emitted intensity S_seg*(1-exp(-dtau)), shape (..., nray-1).
    dtau : jnp.ndarray
        Per-segment optical depth, shape (..., nray-1).
    attenuation : jnp.ndarray
        exp(-tau) accumulated between each segment and the observer,
        shape (..., nray-1).
    """
    a_sum = a_nu[..., 1:] + a_nu[..., :-1]
    dtau = 0.5 * a_sum * ray_ds
    s_seg = (j_nu[..., 1:] + j_nu[..., :-1]) / (a_sum + 1e-30)
    contrib = s_seg * (-jnp.expm1(-dtau))   # expm1 keeps the small-dtau limit exact

    pad_width = [(0, 0)] * (dtau.ndim - 1) + [(1, 0)]
    attenuation = jnp.exp(-jnp.cumsum(jnp.pad(dtau, pad_width), axis=-1))[..., :-1]
    return contrib, dtau, attenuation


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
    Perform radiative transfer along rays to produce a spectral image cube (Jy/pixel).
    
    Parameters
    ----------
    camera_freqs : jnp.ndarray
        Frequencies along the spectral axis; shape ``(nfreq,)``.
    gas_v : jnp.ndarray
        Gas velocity field [cm/s]; shape ``(npix, npix, nray, 3)``.
    alpha_tot : jnp.ndarray
        Total line broadening [cm/s], shape matching gas_t / spatial field (npix, npix, nray,)
    n_up, n_dn : jnp.ndarray
        Upper/lower level populations, shape matching spatial field (npix, npix, nray,)
    a_ud, b_ud, b_du : float
        Einstein coefficients for the chosen transition.
    ray_coords : jnp.ndarray
        Ray coordinates; shape ``(npix, npix, nray, 3)``. The second-to-last axis is the
        marching index along each ray.
    obs_dir : jnp.ndarray
        Unit vector of the line of sight (towards the observer); shape ``(3,)``.
    nu0 : float
        Line rest frequency [Hz].
    pixel_area : float
        Pixel solid-angle * distance^2 in cm^2 (used for Jy conversion).
    dust_profile : jnp.ndarray, optional
        Dust extinction profile [cm^-1], shape matching spatial field (npix, npix, nray,)

    Returns
    -------
    jnp.ndarray
        Image-plane fluxes; shape ``(nfreq, npix, npix)`` in Jy/pixel.

    Notes
    -----
    Per-segment integration uses the exact constant-property slab solution
    with an extinction-weighted source function (see _segment_emission).
    This replaced the RADMC-3D-style second-order endpoint interpolation,
    which produced O(1) banding artifacts on segments straddling a sharp
    tau=1 surface (near endpoint in vacuum => garbage source function).
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
    contrib, _, attenuation = _segment_emission(j_nu, a_nu, ray_ds)
    intensity = (contrib * attenuation).sum(axis=-1)

    # Conversion from erg/s/cm²/Hz/sr to Jy/pixel
    image_fluxes_jy = pixel_area / (distance_pc * pc)**2 * 1e23 * intensity
    return image_fluxes_jy


def compute_spectral_cube_radmc3d(
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
    Perform radiative transfer along rays using the RADMC-3D second-order scheme.

    This is the original RADMC-3D-style second-order source function integration.
    Use this when backward-compatibility with previous RADMC-3D runs is needed.
    For new work, prefer ``compute_spectral_cube`` (segmented integrator) which
    avoids banding artifacts near sharp opacity surfaces.

    Parameters
    ----------
    Same as ``compute_spectral_cube``.

    Returns
    -------
    jnp.ndarray
        Image-plane fluxes; shape ``(nfreq, npix, npix)`` in Jy/pixel.

    Notes
    -----
    Implements the second-order scheme from the RADMC-3D manual:
    https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/imagesspectra.html#sec-second-order
    """
    doppler = (1.0 / cc) * jnp.sum(obs_dir * gas_v, axis=-1)
    dnu = expand_dims(camera_freqs, alpha_tot.ndim + 1, axis=-1) - nu0 - doppler * nu0
    line_profile = (cc / (alpha_tot * nu0 * jnp.sqrt(jnp.pi))) * jnp.exp(
        -(cc * dnu / (nu0 * alpha_tot)) ** 2
    )
    const = hh * nu0 / (4 * jnp.pi)
    j_nu = const * n_up * a_ud * line_profile
    a_nu = const * (n_dn * b_du - n_up * b_ud) * line_profile

    ray_ds = jnp.sqrt(jnp.sum(jnp.diff(ray_coords, axis=-2) ** 2, axis=-1))
    dtau = 0.5 * (a_nu[..., 1:] + a_nu[..., :-1]) * ray_ds
    source_1st = 0.5 * (j_nu[..., 1:] + j_nu[..., :-1]) * ray_ds
    s_nu = j_nu / (a_nu + 1e-30)
    beta = (dtau - 1 + jnp.exp(-dtau)) / (dtau + 1e-30)
    beta = jnp.where(dtau > 1e-6, beta, 0.5 * dtau)
    source_2nd = (1 - jnp.exp(-dtau) - beta) * s_nu[..., :-1] + beta * s_nu[..., 1:]
    source_2nd = jnp.where(source_2nd < source_1st, source_2nd, source_1st)
    pad_width = [(0, 0)] * (dtau.ndim - 1) + [(1, 0)]
    attenuation = jnp.exp(-jnp.cumsum(jnp.pad(dtau, pad_width), axis=-1))[..., :-1]
    intensity = (source_2nd * attenuation).sum(axis=-1)
    image_fluxes_jy = pixel_area / (distance_pc * pc) ** 2 * 1e23 * intensity
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

    # Ray tracing: exact constant-property segment solution (see _segment_emission)
    ray_ds = jnp.sqrt(jnp.sum(jnp.diff(ray_coords, axis=-2) ** 2, axis=-1))
    contrib, _, attenuation = _segment_emission(j_tot, a_tot, ray_ds)
    intensity = (contrib * attenuation).sum(axis=-1)

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
        Temperature [K]; shape ``(npix, npix, nray)`` (or broadcastable).
    m_mol : float, optional
        Molecular mass [g] (defaults to CO).

    Returns
    -------
    alpha_tot : jnp.ndarray
        Total broadening [cm/s].

    Notes
    -----
    ``m_mol_h`` is the mean molecular mass of the background gas (~2.34 * m_H).
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

    z_raw = ray_coords[..., 2]
    is_forward = z_raw[0, 0, 0] > z_raw[0, 0, -1]
    # jnp.where instead of Python if/else so this is safe inside jax.lax.scan.
    ray_ds = jnp.where(is_forward, ray_ds, jnp.flip(ray_ds, axis=-1))
    j_nu   = jnp.where(is_forward, j_nu,   jnp.flip(j_nu,   axis=-1))
    a_nu   = jnp.where(is_forward, a_nu,   jnp.flip(a_nu,   axis=-1))
    z_raw  = jnp.where(is_forward, z_raw,  jnp.flip(z_raw,  axis=-1))

    # Mid-point Z-coordinates for the segments
    z_mid = 0.5 * (z_raw[..., 1:] + z_raw[..., :-1])

    # Exact constant-property segment solution (see _segment_emission)
    contrib, _, attenuation = _segment_emission(j_nu, a_nu, ray_ds)

    # Weighted contribution of each segment to the total intensity
    # Shape: (nfreq, npix, npix, nray-1)
    emission_weight = contrib * attenuation
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
    # Find the FIRST segment whose cumulative tau crosses 1
    mask = (tau_sum >= 1.0)
    # argmax returns the first index where mask is True
    first_tau_1_idx = jnp.argmax(mask, axis=-1)

    # Sub-segment interpolation: locate where tau=1 falls WITHIN the crossing
    # segment by linear interpolation in cumulative tau, instead of snapping
    # to the far sample. The nearest-sample version quantised the surface to
    # the ray sampling grid, producing staircased tau1 maps / masks whenever
    # a segment's dtau was large (optically thick discs).
    tau_after  = jnp.take_along_axis(tau_sum,      first_tau_1_idx[..., None], axis=-1).squeeze(-1)
    dtau_seg   = jnp.take_along_axis(dtau_ordered, first_tau_1_idx[..., None], axis=-1).squeeze(-1)
    tau_before = tau_after - dtau_seg
    frac = jnp.clip((1.0 - tau_before) / (dtau_seg + 1e-30), 0.0, 1.0)

    # Map back to Z: segment k runs from sample k (near) to sample k+1 (far)
    z_near = jnp.take_along_axis(z_ordered[None, ...], first_tau_1_idx[..., None],     axis=-1).squeeze(-1)
    z_far  = jnp.take_along_axis(z_ordered[None, ...], first_tau_1_idx[..., None] + 1, axis=-1).squeeze(-1)
    z_surface = z_near + frac * (z_far - z_near)

    # Handle rays that never hit tau=1 (optically thin)
    has_reached_tau_1 = jnp.any(mask, axis=-1)
    z_final = jnp.where(has_reached_tau_1, z_surface, jnp.nan)
    return z_final

# ----------------------------------------------------------------------------- #
# JIT wrappers & vectorized ops
# ----------------------------------------------------------------------------- #

_CUBE_IN_AXES = (0, None, None, None, None, None, None, None, None, None, None, None, None)

# Segmented integrator (default)
compute_spectral_cube_pmap = jax.pmap(
    compute_spectral_cube,
    axis_name="freq",
    in_axes=_CUBE_IN_AXES,
)
compute_spectral_cube_vmap = jax.vmap(
    compute_spectral_cube,
    in_axes=_CUBE_IN_AXES,
)

# RADMC-3D second-order integrator
compute_spectral_cube_radmc3d_pmap = jax.pmap(
    compute_spectral_cube_radmc3d,
    axis_name="freq",
    in_axes=_CUBE_IN_AXES,
)
compute_spectral_cube_radmc3d_vmap = jax.vmap(
    compute_spectral_cube_radmc3d,
    in_axes=_CUBE_IN_AXES,
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
    "compute_spectral_cube_radmc3d",
    "compute_spectral_cube_radmc3d_vmap",
    "compute_spectral_cube_radmc3d_pmap",
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
