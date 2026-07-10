"""Dust functions for radiative transfer calculations, for adding dust rings and spirals to disk.
Maybe this can be software engineered into a better place later. :)

Coordinate convention
---------------------
Ray coordinates produced by `sensor.rays_alma_projection` are in the *disk frame*:
the disk midplane is the world z = 0 plane and the disk symmetry axis is world z-hat.
Inclination / position angle are applied to the *camera* when the rays are built,
never to the disk. Therefore no derotation is needed here: cylindrical disk
coordinates follow directly from the world coordinates.

The optional `tilt`/`tilt_pa` arguments below describe a ring that is genuinely
misaligned *with respect to the gas disk* (e.g. a warped inner ring). They default
to zero and must NOT be set to the observation inclination / position angle.
"""

from __future__ import annotations
from typing import Tuple

import jax
import jax.numpy as jnp

from .consts import au


def ray_coords_to_disk(ray_coords, center=jnp.array([0.0, 0.0, 0.0]), tilt=0.0, tilt_pa=0.0):
    """
    Convert ray world coordinates (already disk-frame, in cm) to cylindrical disk
    coordinates in au.

    Parameters
    ----------
    ray_coords : (H,W,N,3) array
        world coordinates of each sample along each ray [cm]
    center : (3,) array
        structure center in world coordinates [au]
    tilt : float
        tilt of the structure relative to the gas-disk midplane in degrees
        (0 = coplanar with the gas disk). NOT the observation inclination.
    tilt_pa : float
        position angle of the tilt axis in the disk plane in degrees
        (CCW from +x). NOT the observation position angle.

    Returns
    -------
    r_disk, z_disk, theta_disk : (H,W,N) arrays
        cylindrical coordinates relative to the (possibly tilted) structure plane [au]
    """
    x = ray_coords[..., 0] / au - center[0]
    y = ray_coords[..., 1] / au - center[1]
    z = ray_coords[..., 2] / au - center[2]

    # Optional misalignment of the structure relative to the gas disk:
    # rotate by -tilt_pa about z, tilt by -tilt about the (new) x axis.
    pa_rad = jnp.deg2rad(tilt_pa)
    cos_pa, sin_pa = jnp.cos(pa_rad), jnp.sin(pa_rad)
    x_pa = x * cos_pa + y * sin_pa
    y_pa = -x * sin_pa + y * cos_pa

    tilt_rad = jnp.deg2rad(tilt)
    x_d = x_pa
    y_d = y_pa * jnp.cos(tilt_rad) + z * jnp.sin(tilt_rad)
    z_d = -y_pa * jnp.sin(tilt_rad) + z * jnp.cos(tilt_rad)

    r_disk = jnp.sqrt(x_d**2 + y_d**2)
    theta_disk = jnp.arctan2(y_d, x_d)

    return r_disk, z_d, theta_disk


def create_3d_dust_ring(
    ray_coords: jnp.ndarray,  # (H, W, N, 3)
    radius: float,            # [au]
    width: float,             # full Gaussian width (2 sigma) [au]
    thickness: float,         # full Gaussian thickness (2 sigma) [au]
    peak_density: float,      # peak dust mass density [g/cm³]
    tilt: float = 0.0,        # degrees, misalignment w.r.t. the gas disk (usually 0)
    tilt_pa: float = 0.0,     # degrees
    center: jnp.ndarray = jnp.array([0.0, 0.0, 0.0]),
):
    """
    Gaussian dust ring in the gas-disk midplane (or tilted by `tilt` relative to it).

    Note: ray_coords are already disk-frame; do NOT pass the observation
    inclination/position angle here (see module docstring).
    """
    r_disk, z_disk, _ = ray_coords_to_disk(ray_coords, center=center, tilt=tilt, tilt_pa=tilt_pa)

    radial_part = jnp.exp(-0.5 * ((r_disk - radius) / (0.5 * width))**2)
    vertical_part = jnp.exp(-0.5 * (z_disk / (0.5 * thickness))**2)

    density = peak_density * radial_part * vertical_part

    return density


def get_dust_temperature(
    ray_coords,
    disk_params,
    temp_func,
    center=jnp.array([0.0, 0.0, 0.0]),
):
    """
    Evaluate disk temperature on ray coordinates using disk-frame (r, z) [au].

    `temp_func` has signature temp_func(z, r, params) with z, r in au.
    """
    r_disk, z_disk, _ = ray_coords_to_disk(ray_coords, center=center)

    temp_profile = temp_func(
        jnp.abs(z_disk),
        r_disk,
        disk_params
    )

    return temp_profile
