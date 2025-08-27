"""
Sensor and projection utilities.

This module defines simple camera/projection dataclasses and helpers to:
- construct frequency grids around a spectral line,
- build ALMA-style and orthographic ray bundles through a disk volume,
- integrate (project) a scalar volume along rays,
- build a 2D Gaussian beam kernel and convolve image stacks.

Conventions
-----------
- Angles passed to rotation helpers are in **degrees** (matches `grid.rotate_*`).
- Coordinates are in **code units** unless explicitly noted. For sky-plane helpers
  that accept arcseconds and parsecs we convert using constants from `consts`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Tuple, Any, Dict

from flax import struct
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from . import grid
from .consts import arcsec, au, pc, cc

import yaml

ArrayLike = Union[np.ndarray, jnp.ndarray]


# ----------------------------------------------------------------------------- #
# Projections
# ----------------------------------------------------------------------------- #
@struct.dataclass
class ObservationParams:
    """
    Params describing an observation dataset or projection config.
    Stored in a compact YAML-friendly format.

    Attributes
    ----------
    name : str
        Human-readable dataset name.
    distance : float
        Source distance in parsecs.
    fov : float
        Field of view in arcseconds.
    velocity_range : Tuple[float, float]
        Min/max velocity range in m/s.
    vlsr : float
        Local Standard of Rest velocity in m/s.
    nray : int
        Number of samples along each ray.
    incl : float
        Inclination angle [deg].
    phi : float
        Azimuthal angle [deg].
    posang : float
        Position angle [deg].
    z_width: float
        Physical width of the slab in AU (rays go from ± z_width/2)
    """
    name: str
    distance: float
    fov: float
    velocity_range: Tuple[float, float]
    vlsr: float
    nray: int
    incl: float
    phi: float
    posang: float
    z_width: float             
    

def orthographic_projection(
    npix: int,
    nray: int,
    incl: float,
    phi: float,
    posang: float,
    fov: float,
    z_width: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build parallel (orthographic) rays through a slab of thickness `z_width`.
    """
    obs_dir = grid.rotate_coords(jnp.array([0.0, 0.0, 1.0]), incl, phi, posang)

    # Regular grid in camera plane and along z (in world coords)
    ray_x = jnp.linspace(-0.5, 0.5, npix) * fov * (npix - 1) / npix
    ray_y = jnp.linspace(-0.5, 0.5, npix) * fov * (npix - 1) / npix
    ray_z = jnp.linspace(-0.5, 0.5, nray) * z_width / obs_dir[2]

    coords_grid = jnp.stack(jnp.meshgrid(ray_x, ray_y, ray_z, indexing="xy"), axis=-1).reshape(npix * npix, nray, 3)
    ray_coords = grid.rotate_coords(coords_grid, incl, phi, posang)

    # Center the slab around z=0 by shifting along obs_dir
    xy_plane = jnp.stack(
        jnp.meshgrid(ray_x, ray_y, jnp.array([0.0]), indexing="xy"), axis=-1
    ).reshape(npix * npix, 1, 3)
    xy_plane_rot = grid.rotate_coords(xy_plane, incl, phi, posang)
    s = xy_plane_rot[..., 2] / obs_dir[2]
    ray_coords = ray_coords - s[..., None] * obs_dir

    # Far → near ordering
    ray_coords = ray_coords[:, ::-1]
    return ray_coords, obs_dir

orthographic_projection_jit = jax.jit(
    orthographic_projection, 
    static_argnames=("npix", "nray")
)

def rays_alma_projection(
    x_sky: ArrayLike,
    y_sky: ArrayLike,
    distance: float,
    nray: int,
    incl: float,
    phi: float,
    posang: float,
    z_width: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Construct pinhole rays through a finite-thickness disk slab.

    Geometry
    --------
    Rays originate from a virtual pinhole located at ``distance * obs_dir`` and
    pass through image-plane coordinates ``(x_sky, y_sky)`` (in arcsec). Start
    and stop points are the intersections with two planes bounding the slab at
    ``z = ± z_width/2`` (in AU, measured along the world Z-axis after applying
    the viewing geometry).

    Parameters
    ----------
    x_sky, y_sky : 2D arrays (arcsec)
        Sky-plane coordinate grids. Must have identical shapes (ny, nx).
    distance : float (pc)
        Source distance in parsecs. Sets the pinhole location along the LOS.
    nray : int
        Number of sample points along each ray (discretization along depth).
    incl, phi : float (deg)
        Viewing angles applied via ``grid.rotate_coords_angles``. ``incl`` is the
        tilt from +Z; ``phi`` is the azimuthal rotation about +Z.
    posang : float (deg)
        Additional rotation **about the line of sight** (roll) applied after
        orienting by (incl, phi). Positive values rotate clockwise on the sky.
    z_width : float (au)
        Physical slab thickness; rays start on +z_half and end on −z_half.

    Returns
    -------
    ray_coords : (ny, nx, nray, 3) jnp.ndarray
        Sampled world-space points along each ray from far→near.
    obs_dir : (3,) jnp.ndarray
        Unit line-of-sight vector in world coordinates after applying (incl, phi).

    Notes
    -----
    - Angles are in **degrees**.
    - ``x_sky, y_sky`` are in **arcseconds**; conversions use constants in ``consts``.
    - Output coordinates are in the code's **world units** (same units as your grids).
    """
    x_sky = jnp.asarray(x_sky)
    y_sky = jnp.asarray(y_sky)
    ny, nx = x_sky.shape

    # LOS in world coords after (incl, phi)
    obs_dir = jnp.squeeze(grid.rotate_coords_angles(jnp.array([0.0, 0.0, 1.0]), incl, phi))  # (3,)

    # Pixel-to-ray direction in camera frame (arcsec → radians via arcsec constant)
    rho = jnp.sqrt(x_sky**2 + y_sky**2) * arcsec
    theta = jnp.arctan2(y_sky, x_sky)
    ray_x_dir = rho * jnp.cos(theta)
    ray_y_dir = rho * jnp.sin(theta)
    ray_z_dir = jnp.ones_like(ray_x_dir)

    # Stack as (ny*nx, 3) and align to world; then roll about obs_dir by -posang
    ray_dir = jnp.stack((ray_y_dir, ray_x_dir, ray_z_dir), axis=-1).reshape(ny * nx, 3)
    ray_dir = grid.rotate_coords_angles(ray_dir, incl, -phi)
    ray_dir = grid.rotate_coords_vector(ray_dir, obs_dir, -posang)

    # Intersections with z = ± z_width/2 planes (world coords)
    d_pc = distance * pc
    zhalf = 0.5 * z_width * au
    s = (zhalf - d_pc * obs_dir[2]) / ray_dir[..., 2]
    t = (-zhalf - d_pc * obs_dir[2]) / ray_dir[..., 2]
    ray_start = d_pc * obs_dir + s[..., None] * ray_dir
    ray_stop  = d_pc * obs_dir + t[..., None] * ray_dir

    ray_coords = jnp.linspace(ray_start, ray_stop, nray, axis=1).reshape(ny, nx, nray, 3)
    return ray_coords, obs_dir


def rays_from_params(
    meta: ObservationParams, 
    x_sky: ArrayLike, 
    y_sky: ArrayLike
):
    return rays_alma_projection_jit(
        jnp.asarray(x_sky),
        jnp.asarray(y_sky),
        float(meta.distance),
        int(meta.nray),
        float(meta.incl),
        float(meta.phi),
        float(meta.posang),
        float(meta.z_width),
    )


def params_from_yaml(filename: str | Path) -> ObservationParams:
    """
    Load only the `observation` block from a YAML file.

    Parameters
    ----------
    filename : str | Path
        Path to a YAML that contains at least an `observation: {...}` section.

    Returns
    -------
    ObservationParams
        Parsed observation params.

    Raises
    ------
    KeyError
        If the YAML does not contain an `observation` section.
    """
    with open(Path(filename), "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}
    if "observation" not in cfg:
        raise KeyError("YAML is missing required `observation` section.")

    o = cfg["observation"]
    # allow optional nested name override; fall back to top-level name if present
    name = o.get("name", cfg.get("name", ""))

    return ObservationParams(
        name=name,
        distance=float(o["distance"]),
        fov=float(o["fov"]),
        velocity_range=(float(o["velocity_range"][0]), float(o["velocity_range"][1])),
        vlsr=float(o["vlsr"]),
        nray=int(o["nray"]),
        incl=float(o["incl"]),
        phi=float(o["phi"]),
        posang=float(o["posang"]),
        z_width=float(o["z_width"]),
    )


def params_to_yaml(obs: ObservationParams, filename: str | Path) -> None:
    """
    Write only the `observation` block into a unified YAML, preserving everything else.

    Behavior
    --------
    - If `filename` exists:
        • Load the file
        • Replace the entire `observation` section with `obs`
        • Leave other sections (e.g., `disk`, chemistry, paths) untouched
        • If top-level `name` is empty/missing and `obs.name` is non-empty, set `name`
    - If it does not exist:
        • Create a new YAML with just `name` (if provided) and `observation`

    Parameters
    ----------
    obs : ObservationParams
        Observation params to persist.
    filename : str | Path
        Destination YAML path.
    """
    path = Path(filename)
    doc: Dict[str, Any] = {}
    if path.exists():
        with open(path, "r") as f:
            doc = yaml.safe_load(f) or {}

    # build fresh observation block from `obs`
    obs_block = {
        "distance": float(obs.distance),
        "fov": float(obs.fov),
        "velocity_range": [float(obs.velocity_range[0]), float(obs.velocity_range[1])],
        "vlsr": float(obs.vlsr),
        "nray": int(obs.nray),
        "incl": float(obs.incl),
        "phi": float(obs.phi),
        "posang": float(obs.posang),
        "z_width": float(obs.z_width),
    }

    # replace observation, preserve everything else
    doc["observation"] = obs_block

    # set top-level name if missing/empty and obs.name is available
    if not doc.get("name") and obs.name:
        doc["name"] = obs.name

    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)



def print_params(params: "ObservationParams") -> None:
    """
    Nicely formatted summary of ALMA sensor params.

    Parameters
    ----------
    params : ObservationParams
        Params object loaded via `params_from_yaml(...)`.
    """
    print("Dataset name:              {}".format(params.name))
    print("Distance (pc):            {:8.2f}".format(params.distance))
    print("Field of view (arcsec):   {:8.3f}".format(params.fov))
    print("Velocity range (m/s):     {:8.1f} → {:8.1f}".format(*params.velocity_range))
    print("VLSR (m/s):               {:8.2f}".format(params.vlsr))
    print("Number of rays:           {:8d}".format(params.nray))
    print("Inclination (deg):        {:8.2f}".format(params.incl))
    print("Phi (deg):                {:8.2f}".format(params.phi))
    print("Position angle (deg):     {:8.2f}".format(params.posang))
    print("Slab thickness z_width (au): {:8.2f}".format(params.z_width))

    
# ----------------------------------------------------------------------------- #
# Frequencies
# ----------------------------------------------------------------------------- #
def compute_camera_freqs(
    linelam: int,
    width_kms: float,
    nu0: float,
    v_sys: float = 0.0,
    num_subfreq: int = 1,
    subfreq_width: Optional[float] = None,
) -> jnp.ndarray:
    """
    Build a (possibly sub-sampled) frequency grid around line center.
    """
    # Doppler offsets: map linear velocity window into frequency offsets
    # v positive (toward observer) reduces frequency → (1 - v/c)
    v = (2 * jnp.arange(linelam) / (linelam - 1) - 1.0) * width_kms  # km/s
    camera_freqs = nu0 * (1.0 - (v_sys * 1e5) / cc - (v * 1e5) / cc)

    if num_subfreq > 1:
        if subfreq_width is None:
            raise ValueError("If num_subfreq > 1, you must provide subfreq_width.")
        # Create num_subfreq points spanning [−Δ/2, +Δ/2] around each coarse center
        sub = jnp.linspace(-0.5, 0.5, num_subfreq) * float(subfreq_width)
        camera_freqs = camera_freqs[:, None] + sub[None, :]

    return camera_freqs


# ----------------------------------------------------------------------------- #
# Volume projection
# ----------------------------------------------------------------------------- #
def project_volume(volume: jnp.ndarray, coords: jnp.ndarray, bbox: jnp.ndarray) -> jnp.ndarray:
    """
    Integrate a scalar field along rays using midpoint rule.
    """
    ds = jnp.sqrt(jnp.sum(jnp.diff(coords, axis=-2) ** 2, axis=-1))  # (..., nray-1)

    # Interpolate values at ray points and midpoint integrate
    vals = grid.interpolate_scalar(volume, coords, bbox)  # (..., nray)
    mid = vals[..., :-1] + jnp.diff(vals, axis=-1) / 2.0
    projection = jnp.sum(mid * ds, axis=-1)
    return projection


# ----------------------------------------------------------------------------- #
# Beam & convolution
# ----------------------------------------------------------------------------- #
def beam(
    dpix: float,
    bmaj: float,
    bmin: float,
    bpa: float,
    scale: float = 1.0,
    x_c: float = 0.0,
    y_c: float = 0.0,
) -> jnp.ndarray:
    """
    Build a 2D Gaussian beam kernel.
    """
    from astropy.convolution import Gaussian2DKernel

    sigma_maj = scale * bmaj / dpix / 2.355
    sigma_min = scale * bmin / dpix / 2.355
    kernel = np.asarray(Gaussian2DKernel(x_stddev=sigma_min, y_stddev=sigma_maj, theta=np.radians(bpa)).array)
    return jnp.asarray(kernel)

    
# ----------------------------------------------------------------------------- #
# JIT wrappers & vectorized ops
# ----------------------------------------------------------------------------- #

rays_alma_projection_jit = jax.jit(rays_alma_projection, static_argnames=("nray",),)

# Convolve a stack of images with the same kernel (vectorized over first axis)
fftconvolve_vmap = jax.vmap(lambda x, k: jsp.signal.fftconvolve(x, k, mode="same"), in_axes=(0, None))


