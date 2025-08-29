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

from . import grid, phys
from . import chemistry as chem
from . import line_rte
from .consts import arcsec, au, pc, cc

import yaml

ArrayLike = Union[jnp.ndarray, jnp.ndarray]


# ----------------------------------------------------------------------------- #
# Projections
# ----------------------------------------------------------------------------- #
@struct.dataclass
class RayBundle:
    nx: int
    ny: int
    coords_xyz: jnp.ndarray   # (H, W, N, 3), world XYZ along each ray
    pixel_area: jnp.ndarray   # (H, W), projected area per pixel in [sr] or [cm^2] on image plane
    obs_dir: jnp.ndarray      # (3,), unit vector from source to observer
    
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
    x_sky: "ArrayLike",          # [arcsec], shape (ny, nx)
    y_sky: "ArrayLike",          # [arcsec], shape (ny, nx)
    distance: float,             # [pc]
    nray: int,
    incl: float,                 # [deg]
    phi: float,                  # [deg]
    posang: float,               # [deg], roll about LOS
    z_width: float,              # [au]
    fov_as: float,               # [arcsec], total field of view on a side
) -> "RayBundle":
    """
    Construct pinhole rays through a finite-thickness disk slab and return a RayBundle.
    Always uses the provided FOV (arcsec) to compute a constant pixel_area (cm^2).

    Returns
    -------
    RayBundle with:
      - nx, ny: int
      - coords_xyz: (ny, nx, nray, 3) world-space samples along each ray (far → near)
      - pixel_area: (ny, nx) constant map of pixel area in cm^2
      - obs_dir:    (3,) unit LOS in world coordinates after (incl, phi)
    """
    # ---- inputs & shapes
    x_sky = jnp.asarray(x_sky)  # [arcsec]
    y_sky = jnp.asarray(y_sky)  # [arcsec]
    ny, nx = x_sky.shape
    if y_sky.shape != (ny, nx):
        raise ValueError("x_sky and y_sky must have identical shapes (ny, nx).")
    npix = int(max(ny, nx))

    # ---- LOS after (incl, phi)
    obs_dir = jnp.squeeze(
        grid.rotate_coords_angles(jnp.array([0.0, 0.0, 1.0]), incl, phi)
    )  # (3,)

    # ---- camera-frame directions (arcsec → rad)
    rho   = jnp.sqrt(x_sky**2 + y_sky**2) * arcsec
    theta = jnp.arctan2(y_sky, x_sky)
    ray_x_dir = rho * jnp.cos(theta)
    ray_y_dir = rho * jnp.sin(theta)
    ray_z_dir = jnp.ones_like(ray_x_dir)

    # stack (ny*nx, 3), align to world; roll by -posang about LOS
    ray_dir = jnp.stack((ray_y_dir, ray_x_dir, ray_z_dir), axis=-1).reshape(ny * nx, 3)
    ray_dir = grid.rotate_coords_angles(ray_dir, incl, -phi)
    ray_dir = grid. rotate_coords_vector(ray_dir, obs_dir, -posang)

    # ---- intersections with z = ± z_width/2 (world coords in cm)
    d_cm     = distance * pc
    zhalf_cm = 0.5 * z_width * au
    denom = ray_dir[..., 2]
    denom = jnp.where(jnp.abs(denom) < 1e-30, jnp.sign(denom) * 1e-30, denom)

    s = ( zhalf_cm - d_cm * obs_dir[2]) / denom
    t = (-zhalf_cm - d_cm * obs_dir[2]) / denom

    ray_start = d_cm * obs_dir + s[..., None] * ray_dir  # (ny*nx, 3)
    ray_stop  = d_cm * obs_dir + t[..., None] * ray_dir  # (ny*nx, 3)

    ray_coords = jnp.linspace(ray_start, ray_stop, nray, axis=1)  # (ny*nx, nray, 3)
    ray_coords = ray_coords.reshape(ny, nx, nray, 3)

    # ---- pixel area from FOV (constant over the grid), in cm^2
    # fov on the image plane at distance d: fov_cm = 2 * d * tan((fov_as*arcsec)/2)
    fov_rad = fov_as * arcsec
    fov_cm  = 2.0 * d_cm * jnp.tan(fov_rad / 2.0)
    pixel_area = (fov_cm / float(npix)) ** 2

    rays = RayBundle(
        nx=nx, 
        ny=ny, 
        coords_xyz=ray_coords, 
        pixel_area=pixel_area, 
        obs_dir=obs_dir
    )
    
    return rays


def rays_from_params(
    obs_params: ObservationParams, 
    x_sky: ArrayLike, 
    y_sky: ArrayLike
):
    return rays_alma_projection_jit(
        jnp.asarray(x_sky),
        jnp.asarray(y_sky),
        float(obs_params.distance),
        int(obs_params.nray),
        float(obs_params.incl),
        float(obs_params.phi),
        float(obs_params.posang),
        float(obs_params.z_width),
        float(obs_params.fov),
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
# Volume sampling/ projection / rendering
# ----------------------------------------------------------------------------- #
def render_cube(
    rays: "RayBundle",
    nd_ray: jnp.ndarray,           # (H, W, N)
    temperature_ray: jnp.ndarray,  # (H, W, N)
    velocity_ray: jnp.ndarray,     # (H, W, N, 3)
    *,
    nu0: float,
    freqs: jnp.ndarray,            # (F,) [Hz]
    v_turb: float,
    mol: "MolecularData",
    backend: str = "vmap",         # {"vmap", "pmap", "none"}
) -> jnp.ndarray:
    """
    Render I(ν, y, x) using pre-sampled ray fields and line data in `mol`.

    Parameters
    ----------
    rays : RayBundle
        nx, ny, coords_xyz, pixel_area, obs_dir describing ray geometry.
    nd_ray : (H, W, N)
        Number density along rays.
    temperature_ray : (H, W, N)
        Temperature along rays [K].
    velocity_ray : (H, W, N, 3)
        3D velocity vectors along rays.
    nu0: float, 
        Central frequency, e.g. from alma_cube.nu0
    freqs : (F,)
        Frequency channels [Hz].
    v_turb : float
        Microturbulent velocity (ensure units consistent with opacity kernel).
    mol : MolecularData
        Energy levels, transitions, and Einstein coefficients for one line.
    backend : {"vmap", "pmap", "none"}, default="vmap"
        Which compute backend to use for the spectral cube solver:
          - "vmap" : run vectorized over frequency (default, usually fastest single-device)
          - "pmap" : parallelize across multiple devices (if available)
          - "none" : plain per-frequency loop (slow, but simplest)

    Returns
    -------
    cube : (nfreq, ny, nx) jnp.ndarray
        Spectral cube (NaNs sanitized).
    """

    from radjax.core.parallel import shard_with_padding

    # LTE level populations
    n_up, n_dn = chem.n_up_down(
        nd_ray, temperature_ray,
        mol.energy_levels, mol.radiative_transitions,
        transition=mol.transition,
    )

    # Line opacity
    alpha_tot = line_rte.alpha_total(v_turb, temperature_ray)

    # Choose backend
    if backend == "pmap":
        compute_fn = line_rte.compute_spectral_cube_pmap
        freqs = shard_with_padding(freqs)  # (ndev, F_per_dev)
    elif backend == "vmap":
        compute_fn = line_rte.compute_spectral_cube_vmap
    elif backend == "none":
        compute_fn = line_rte.compute_spectral_cube
    else:
        raise ValueError(f"Unknown backend={backend!r}. Must be 'vmap', 'pmap', or 'none'.")
    
    images = compute_fn(
        freqs, velocity_ray, alpha_tot, n_up, n_dn,
        mol.a_ud, mol.b_ud, mol.b_du,
        rays.coords_xyz, rays.obs_dir, nu0, rays.pixel_area
    )
    images = np.nan_to_num(images).reshape(-1, rays.ny, rays.nx)[:freqs.size]
    
    return images



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

def sample_symmetric_disk__along_rays(
    rays: "RayBundle",
    bbox: jnp.ndarray,                  # shape (2,2): [[zmin,zmax],[rmin,rmax]] in cm
    co_nd: jnp.ndarray,                 # (Nz, Nr)
    temperature: jnp.ndarray,           # (Nz, Nr)
    v_phi: jnp.ndarray,             # (Nz, Nr), azimuthal speed in disk frame
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Interpolate **axisymmetric** disk fields (z–r grids) along a bundle of rays.

    Assumptions
    -----------
    - The disk is mirror-symmetric and axisymmetric; fields are defined on a 2D (z, r) grid.
    - Velocity is purely azimuthal (vφ); we reconstruct 3D velocity vectors along the rays.
    - `rays` provides world-space sample coordinates; only `coords_xyz` is used here.

    Parameters
    ----------
    rays : RayBundle
        Container with:
          - coords_xyz : (H, W, N, 3) world-space samples along each ray (far → near)
          - pixel_area : (H, W) (unused here)
          - obs_dir    : (3,) (unused here)
    bbox : jnp.ndarray, shape (2,2)
        Bounding box in cm, [[zmin, zmax], [rmin, rmax]], for cylindrical interpolation.
    co_nd : jnp.ndarray, shape (Nz, Nr)
        CO number density grid in the (z, r) plane.
    temperature : jnp.ndarray, shape (Nz, Nr)
        Temperature grid in the (z, r) plane.
    v_phi : jnp.ndarray, shape (Nz, Nr)
        Azimuthal velocity (scalar speed) grid in the (z, r) plane.


    Returns
    -------
    nd_ray : jnp.ndarray, shape (H, W, N)
        CO number density interpolated along rays.
    temperature_ray : jnp.ndarray, shape (H, W, N)
        Temperature interpolated along rays.
    velocity_ray : jnp.ndarray, shape (H, W, N, 3)
        3D velocity vectors along each ray, reconstructed from vφ.
    """
    ray_sph = grid.cartesian_to_spherical(rays.coords_xyz)   # (H, W, N, 3)
    ray_zr  = grid.spherical_to_zr(ray_sph)                 # (H, W, N, 2): (z, r)

    nd_ray   = grid.interpolate_scalar(co_nd, ray_zr, bbox)             # (H, W, N)
    temperature_ray = grid.interpolate_scalar(temperature, ray_zr, bbox, cval=1e-10) # (H, W, N)
    v_phi_ray = grid.interpolate_scalar(v_phi, ray_zr, bbox)             # (H, W, N)

    # Convert azimuthal scalar speed to 3D velocity vectors along the rays
    velocity_ray = phys.azimuthal_velocity(rays.coords_xyz, v_phi_ray)      # (H, W, N, 3)

    return nd_ray, temperature_ray, velocity_ray


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
    kernel = jnp.asarray(Gaussian2DKernel(x_stddev=sigma_min, y_stddev=sigma_maj, theta=np.radians(bpa)).array)
    return jnp.asarray(kernel)

    
# ----------------------------------------------------------------------------- #
# JIT wrappers & vectorized ops
# ----------------------------------------------------------------------------- #

rays_alma_projection_jit = jax.jit(rays_alma_projection, static_argnames=("nray",),)

# Convolve a stack of images with the same kernel (vectorized over first axis)
fftconvolve_vmap = jax.vmap(lambda x, k: jsp.signal.fftconvolve(x, k, mode="same"), in_axes=(0, None))


