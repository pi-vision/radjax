"""Dust functions for radiative transfer calculations, for adding dust rings and spirals to disk. 
Maybe this can be software engineered into a better place later. :) 
"""

from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import gofish

from .consts import (
    cc,          # speed of light      [cm/s]
    hh,          # Planck constant     [erg*s]
    kk,          # Boltzmann constant  [erg/K]
    pc,          # parsec [cm]
    m_co,        # molecular mass of CO [g] (or consistent units)
    m_mol_h,      # mean molecular mass of hydrogen gass [g] (or consistent units)
    au
)


def ray_coords_to_disk(ray_coords, center=jnp.array([0.0,0.0,0.0]), incl=0.0, pa=0.0):
    """
    Convert ray world coordinates to intrinsic disk coordinates.
    
    Parameters
    ----------
    ray_coords : (H,W,N,3) array
        world coordinates of each sample along each ray
    center : (3,) array
        disk center in world coordinates
    incl : float
        disk inclination in degrees (0 = face-on)
    pa : float
        position angle of disk major axis in degrees (CCW from +x)
    
    Returns
    -------
    r_disk, z_disk, theta_disk : (H,W,N) arrays
        intrinsic disk coordinates
    """
    # Shift to disk center
    pos = ray_coords
    x, y, z = pos[...,0]/au-center[0], pos[...,1]/au-center[1], pos[...,2]/au-center[2]

    # --- Undo PA rotation (sky-plane rotation) ---
    pa_rad = jnp.deg2rad(-pa)  # negative to rotate back to disk frame
    cos_pa, sin_pa = jnp.cos(pa_rad), jnp.sin(pa_rad)
    x_pa = x * cos_pa + y * sin_pa
    y_pa = -x * sin_pa + y * cos_pa

    # --- Undo inclination (tilt around x axis) ---
    incl_rad = jnp.deg2rad(incl)
    x_disk = x_pa
    y_disk = y_pa * jnp.cos(incl_rad) + z * jnp.sin(incl_rad)
    z_disk = -y_pa * jnp.sin(incl_rad) + z * jnp.cos(incl_rad)

    # Cylindrical coordinates
    r_disk = jnp.sqrt(x_disk**2 + y_disk**2)
    theta_disk = jnp.arctan2(y_disk, x_disk)

    return r_disk, z_disk, theta_disk



def create_3d_dust_ring(
    ray_coords: jnp.ndarray,  # (H, W, N, 3)
    radius: float,
    width: float,
    thickness: float,
    peak_density: float,      # peak dust mass density [g/cm³]
    inclination: float,   # degrees
    pa: float,            # degrees
    center: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]),
):

    r_disk, z_disk, _ = ray_coords_to_disk(ray_coords, incl=inclination, pa=pa, center=center)

    radial_part = jnp.exp(-0.5 * ((r_disk - radius) / (0.5 * width))**2)
    vertical_part = jnp.exp(-0.5 * (z_disk / (0.5 * thickness))**2)

    density = peak_density * radial_part * vertical_part

    return density


def get_dust_temperature(
    ray_coords,
    disk_params,
    temp_func,
    posang,
    incl,
    center=jnp.array([0.0, 0.0, 0.0]),
):
    """
    Evaluate disk temperature on ray coordinates using intrinsic disk (r, z).
    """

    r_disk, z_disk, _ = ray_coords_to_disk(ray_coords, incl=incl, pa=posang, center=center)

    # --- temperature law (pure disk physics) ---
    temp_profile = temp_func(
        jnp.abs(z_disk),
        r_disk,
        disk_params
    )

    return temp_profile


