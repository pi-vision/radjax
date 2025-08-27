"""
ALMA FITS cube loading & preparation (lives in core to avoid extra packages).

This module wraps a FITS spectral cube (via `imagecube`) and derives:
- rest-frame frequencies using V_LSR,
- sky-plane grids (x_sky, y_sky) in arcsec,
- beam kernel scaled to per-pixel units,
- spectral bookkeeping (resolution, width, counts).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import jax.numpy as jnp

from gofish import imagecube

ArrayLike = Union[np.ndarray, jnp.ndarray]

@dataclass
class PreparedALMACube:
    cube: object
    data: ArrayLike            # (nchan, ny, nx)
    velocities: ArrayLike      # (nchan,) m/s
    freqs: ArrayLike           # (nchan,) rest-frame Hz
    nu0: float
    num_freqs: int
    delta_v_ms: float
    width_kms: float
    npix: int
    x_sky: ArrayLike           # (ny, nx) arcsec
    y_sky: ArrayLike           # (ny, nx) arcsec
    beam_kernel: ArrayLike     # scaled by beams_per_pix


def prepare_alma_cube(
    data_path: str,
    metadata: "ALMASensorMetadata",
    *,
    to_jax: bool = True,
    imagecube_kwargs: Optional[dict] = None,
) -> PreparedALMACube:
    """
    Load a FITS spectral cube and derive common ALMA analysis products.

    Parameters
    ----------
    data_path : str
        Path to the FITS cube.
    metadata : ALMASensorMetadata
        Includes fov_as (arcsec), velocity_range (m/s), vlsr (m/s).
    to_jax : bool, default True
        Return JAX arrays for numeric outputs when True; NumPy otherwise.
    imagecube_kwargs : dict, optional
        Extra kwargs forwarded to `imagecube(...)`.

    Returns
    -------
    PreparedALMACube
    """

    ic_kwargs = dict(FOV=getattr(metadata, "fov", None), velocity_range=getattr(metadata, "velocity_range"))
    if imagecube_kwargs:
        ic_kwargs.update(imagecube_kwargs)

    cube = imagecube(data_path, **ic_kwargs)

    # spectra & axes
    freqs = cube.velocity_to_restframe_frequency(vlsr=getattr(metadata, "vlsr"))
    velocities = cube.velax     # m/s
    data = cube.data            # (nchan, ny, nx)
    xaxis = cube.xaxis          # arcsec
    yaxis = cube.yaxis          # arcsec

    # sky grids
    backend = jnp if to_jax else np
    x_sky, y_sky = backend.meshgrid(xaxis, yaxis, indexing="xy")

    # stats
    nu0 = float(cube.nu0)
    npix = int(cube.nxpix)
    width_kms = (float(velocities[-1]) - float(velocities[0])) / 1000.0
    delta_v_ms = float(velocities[1] - velocities[0])
    num_freqs = int(len(freqs))

    # beam (import here to avoid hard dependency at module import time)
    from . import sensor
    kernel = sensor.beam(cube.dpix, cube.bmaj, cube.bmin, cube.bpa)
    kernel = jnp.asarray(kernel) if to_jax else np.asarray(kernel)
    beam_kernel = cube.beams_per_pix * kernel

    # cast
    if to_jax:
        data = jnp.asarray(data)
        velocities = jnp.asarray(velocities)
        freqs = jnp.asarray(freqs)
    else:
        data = np.asarray(data)
        velocities = np.asarray(velocities)
        freqs = np.asarray(freqs)

    return PreparedALMACube(
        cube=cube,
        data=data,
        velocities=velocities,
        freqs=freqs,
        nu0=nu0,
        num_freqs=num_freqs,
        delta_v_ms=delta_v_ms,
        width_kms=width_kms,
        npix=npix,
        x_sky=x_sky,
        y_sky=y_sky,
        beam_kernel=beam_kernel,
    )


__all__ = ["PreparedALMACube", "prepare_alma_cube"]
