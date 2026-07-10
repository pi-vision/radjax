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
import matplotlib.pyplot as plt

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
    pixel_area: jnp.ndarray   # (H, W), projected pixel area in cm^2 at the source plane
    obs_dir: jnp.ndarray      # (3,), unit vector from source to observer
    distance_pc: float        # source distance [pc], used for solid-angle flux conversion
    
@struct.dataclass
class ObservationParams:
    """
    Observation / projection configuration.

    Three mutually exclusive modes:
    - REAL:      provide velocity_range=(vmin, vmax) [m/s] and vlsr [m/s]  (FITS cube)
    - SYNTHETIC: provide velocity_width_kms [km/s] only                     (synthetic cube)
    - MS:        provide vlsr [m/s] only; channels come from the MS          (measurement set)

    Common fields:
      name:      dataset identifier
      distance:  [pc]
      fov:       [arcsec]
      nray:      samples along each ray
      incl, phi, posang: [deg]
      z_width:   [AU] (rays traverse ± z_width/2)
    """
    # Common
    name: str
    distance: float
    fov: float
    nray: int = 100
    incl: float = 0.0
    phi: float = 0.0
    posang: float = 0.0
    z_width: float = 400.0

    # REAL mode fields
    velocity_range: Optional[Tuple[float, float]] = None  # (vmin, vmax) in m/s
    vlsr: Optional[float] = None                          # m/s

    # SYNTHETIC mode field
    # NOTE: interpreted as a HALF-width — rendered channels span
    # [-velocity_width_kms, +velocity_width_kms] (see compute_camera_freqs).
    velocity_width_kms: Optional[float] = None            # km/s

    # ---- validations & mode logic ----
    def __post_init__(self):
        has_range = self.velocity_range is not None
        has_vlsr  = self.vlsr is not None
        has_width = self.velocity_width_kms is not None

        # Enforce XOR modes
        if (has_range or has_vlsr) and has_width:
            raise ValueError("Provide EITHER (velocity_range & vlsr) for REAL mode OR velocity_width_kms for SYNTHETIC mode, not both.")
        if has_width:
            # SYNTHETIC mode
            if has_range or has_vlsr:
                raise ValueError("In SYNTHETIC mode, do not provide velocity_range or vlsr.")
            if self.velocity_width_kms <= 0:
                raise ValueError("velocity_width_kms must be positive.")
        elif has_range and has_vlsr:
            # REAL mode (FITS cube)
            vmin, vmax = self.velocity_range
            if not isinstance(vmin, (int, float)) or not isinstance(vmax, (int, float)):
                raise TypeError("velocity_range must be a tuple of floats (vmin, vmax) in m/s.")
            if vmax <= vmin:
                raise ValueError("velocity_range must satisfy vmax > vmin.")
        elif has_vlsr and not has_range:
            # MS mode: vlsr only, channels come from the measurement set
            pass
        else:
            raise ValueError(
                "Provide one of: (a) velocity_range + vlsr for FITS/REAL mode, "
                "(b) velocity_width_kms for SYNTHETIC mode, "
                "or (c) vlsr only for MS mode."
            )

        # Basic sanity (shared)
        # Ray sampling starts at z = +z_width/2, which is only the observer-near
        # side for incl < 90 deg; the RT attenuation ordering assumes this.
        if not (0.0 <= self.incl < 90.0):
            raise ValueError("incl must be in [0, 90) degrees (attenuation ordering assumes the observer is above the midplane).")
        if self.distance <= 0:
            raise ValueError("distance must be positive [pc].")
        if self.fov <= 0:
            raise ValueError("fov must be positive [arcsec].")
        if self.nray <= 0:
            raise ValueError("nray must be positive.")
        if self.z_width <= 0:
            raise ValueError("z_width must be positive [AU].")

    # ---- mode flags ----
    @property
    def is_synthetic(self) -> bool:
        return self.velocity_width_kms is not None

    @property
    def is_ms(self) -> bool:
        return self.vlsr is not None and self.velocity_range is None and self.velocity_width_kms is None

    @property
    def is_real(self) -> bool:
        return self.velocity_range is not None

    # ---- convenience getters ----
    @property
    def velocity_width_ms(self) -> float:
        """Return spectral span in m/s (REAL: derived from range; SYNTHETIC: from width)."""
        if self.is_synthetic:
            return float(self.velocity_width_kms) * 1_000.0
        vmin, vmax = self.velocity_range  # type: ignore
        return float(vmax - vmin)

    def velocity_center_ms(self) -> float:
        """
        Center velocity for grid construction:
        - REAL: vlsr
        - SYNTHETIC: 0 m/s (by convention; adjust if you want a non-zero center)
        """
        return 0.0 if self.is_synthetic else float(self.vlsr)  # type: ignore

    def velocity_bounds_ms(self) -> Tuple[float, float]:
        """
        Bounds for channel construction:
        - REAL: return (vmin, vmax)
        - SYNTHETIC: symmetric around 0, i.e., (-width, +width), matching
          compute_camera_freqs which treats velocity_width_kms as a half-width.
        """
        if self.is_synthetic:
            half = self.velocity_width_ms
            return (-half, +half)
        return self.velocity_range  # type: ignore


@struct.dataclass
class ObsParams:
    incl: float
    posang: float
    n_freqs: int
    velocity_width_kms: float
    v_sys: float
    beam_size: float
    distance: float
    nray: int
    fov: float
    z_width: float
    phi: float
    noise: float

    def validate(self):
        return self
    

def obs_from_yaml(source: Union[str, Path, Dict[str, Any]]) -> ObsParams:
    """
    Load ObsParams from the `observation:` section of a YAML (or dict).
    """

    if isinstance(source, (str, Path)):
        with open(Path(source), "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = dict(source)

    if "observation" not in cfg:
        raise KeyError("YAML is missing required `observation` section.")

    o = dict(cfg["observation"])

    # coerce floats
    float_keys = [
        "incl",
        "posang",
        "velocity_width_kms",
        "v_sys",
        "beam_size",
        "distance",
        "fov",
        "phi",
        "noise",
        "z_width",
    ]

    for k in float_keys:
        if k in o and o[k] is not None:
            o[k] = float(o[k])

    # integers
    if "n_freqs" in o:
        o["n_freqs"] = int(o["n_freqs"])
    if "nray" in o:
        o["nray"] = int(o["nray"])

    obs = ObsParams(**o).validate()

    return obs


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
    Always uses the provided FOV (arcsec) to compute a constant pixel_area (sr).

    Returns
    -------
    RayBundle with:
      - nx, ny: int
      - coords_xyz: (ny, nx, nray, 3) world-space samples along each ray (near → far, index 0 = top/near, index N-1 = bottom/far)
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
    # Same (incl, phi) rotation as obs_dir above — a sign mismatch here makes the
    # camera look away from the disk for any phi != 0.
    ray_dir = grid.rotate_coords_angles(ray_dir, incl, phi)
    ray_dir = grid. rotate_coords_vector(ray_dir, obs_dir, -posang)

    # ---- intersections with z = ± z_width/2 (world coords in cm)
    d_cm     = distance * pc
    zhalf_cm = 0.5 * z_width * au
    denom = ray_dir[..., 2]
    # sign(0) == 0, so build the sign explicitly to keep exactly edge-on rays finite
    denom_sign = jnp.where(denom >= 0, 1.0, -1.0)
    denom = denom_sign * jnp.maximum(jnp.abs(denom), 1e-30)

    s = ( zhalf_cm - d_cm * obs_dir[2]) / denom
    t = (-zhalf_cm - d_cm * obs_dir[2]) / denom

    ray_start = d_cm * obs_dir + s[..., None] * ray_dir  # (ny*nx, 3)
    ray_stop  = d_cm * obs_dir + t[..., None] * ray_dir  # (ny*nx, 3)

    ray_coords = jnp.linspace(ray_start, ray_stop, nray, axis=1)  # (ny*nx, nray, 3)
    ray_coords = ray_coords.reshape(ny, nx, nray, 3)

    # ---- pixel solid angle [sr], distance-independent
    fov_rad    = fov_as * arcsec
    pixel_area = (2.0 * jnp.tan(fov_rad / 2.0) / float(npix)) ** 2

    rays = RayBundle(
        nx=nx,
        ny=ny,
        coords_xyz=ray_coords,
        pixel_area=pixel_area,
        obs_dir=obs_dir,
        distance_pc=distance
    )
    
    return rays

def rays_simulation_projection(
    x_sky: "ArrayLike",          # [cm], shape (ny, nx)
    y_sky: "ArrayLike",          # [cm], shape (ny, nx)
    distance: float,             # [pc]
    nray: int,
    incl: float,                 # [deg]
    phi: float,                  # [deg]
    posang: float,               # [deg], roll about LOS
    z_width: float,              # [au]
    fov_cm: float,               # [cm], total field of view on a side
) -> "RayBundle":
    raise NotImplementedError("rays_simulation_projection is not yet implemented.")

# TODO: rename to rays_from_alma_params since we might not use rays_alma_projection. Or otherwise remove alma from rays_alma_projection
def rays_from_params(
    obs_params: ObservationParams,
    x_sky: ArrayLike,
    y_sky: ArrayLike
):
    return rays_alma_projection_jit(
        jnp.asarray(x_sky),
        jnp.asarray(y_sky),
        obs_params.distance,
        obs_params.nray,
        obs_params.incl,
        obs_params.phi,
        obs_params.posang,
        obs_params.z_width,
        obs_params.fov,
    )

    Parameters
    ----------
    rays_base : RayBundle
        Canonical ray bundle constructed at reference geometry (e.g. zero inclination).
    incl_deg : float
        Inclination angle in degrees.
    phi_deg : float
        Azimuthal rotation angle in degrees.
    posang_deg : float
        Position angle (roll about the line of sight) in degrees.

    Returns
    -------
    RayBundle
        A new RayBundle with identical shape and pixel area as ``rays_base``,
        but with rotated coordinates and updated line-of-sight direction.

    Notes
    -----
    - The base coordinates are assumed to be in world space for some reference
      angles.
    - Rotations follow the same convention as :func:`rays_alma_projection`:
      
      * ``obs_dir = rotate([0,0,1], incl, phi)``  
      * ``coords  = rotate(coords, incl, -phi)``  
      * ``coords  = rotate_about(obs_dir, coords, -posang)``  

    Use this when inclination or position angle are part of the parameter
    vector ``θ``, but the overall camera grid (distance, FOV, ``nray``, ``z_width``)
    remains fixed.
    """
    # LOS after (incl, phi)
    obs_dir = jnp.squeeze(grid.rotate_coords_angles(jnp.array([0.0, 0.0, 1.0]), incl_deg, phi_deg))  # (3,)

    # Rotate coordinates
    coords = rays_base.coords_xyz.reshape(-1, 3)
    coords = grid.rotate_coords_angles(coords, incl_deg, -phi_deg)
    coords = grid.rotate_coords_vector(coords, obs_dir, -posang_deg)
    coords = coords.reshape(rays_base.coords_xyz.shape)

    # Rebuild RayBundle (keep nx, ny, nray, pixel_area; update coords & obs_dir)
    return rays_base.replace(coords_xyz=coords, obs_dir=obs_dir)

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

    # --- choose mode ---
    if "velocity_range" in o and "vlsr" in o:
        # REAL mode
        vmin, vmax = o["velocity_range"]
        return ObservationParams(
            name=name,
            distance=float(o["distance"]),
            fov=float(o["fov"]),
            velocity_range=(float(vmin), float(vmax)),
            vlsr=float(o["vlsr"]),
            nray=int(o["nray"]),
            incl=float(o["incl"]),
            phi=float(o["phi"]),
            posang=float(o["posang"]),
            z_width=float(o["z_width"]),
        )
    elif "velocity_width_kms" in o:
        # SYNTHETIC mode
        return ObservationParams(
            name=name,
            distance=float(o["distance"]),
            fov=float(o["fov"]),
            velocity_width_kms=float(o["velocity_width_kms"]),
            nray=int(o["nray"]),
            incl=float(o["incl"]),
            phi=float(o["phi"]),
            posang=float(o["posang"]),
            z_width=float(o["z_width"]),
        )
    elif "vlsr" in o:
        # MS mode: channels come from the measurement set; only vlsr needed
        return ObservationParams(
            name=name,
            distance=float(o["distance"]),
            fov=float(o["fov"]),
            vlsr=float(o["vlsr"]),
            nray=int(o["nray"]),
            incl=float(o["incl"]),
            phi=float(o["phi"]),
            posang=float(o["posang"]),
            z_width=float(o["z_width"]),
        )
    else:
        raise KeyError(
            "Observation block must contain one of: "
            "(a) `velocity_range` + `vlsr` for FITS/REAL mode, "
            "(b) `velocity_width_kms` for SYNTHETIC mode, "
            "or (c) `vlsr` only for MS mode."
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

    # --- Build observation block depending on mode ---
    obs_block: Dict[str, Any] = {
        "distance": float(obs.distance),
        "fov": float(obs.fov),
        "nray": int(obs.nray),
        "incl": float(obs.incl),
        "phi": float(obs.phi),
        "posang": float(obs.posang),
        "z_width": float(obs.z_width),
    }

    if obs.is_real:
        vmin, vmax = obs.velocity_range
        obs_block.update({
            "velocity_range": [float(vmin), float(vmax)],
            "vlsr": float(obs.vlsr),
        })
    else:  # synthetic
        obs_block.update({
            "velocity_width_kms": float(obs.velocity_width_kms),
        })

    # replace observation, preserve everything else
    doc["observation"] = obs_block

    # set top-level name if missing/empty and obs.name is available
    if not doc.get("name") and obs.name:
        doc["name"] = obs.name

    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def print_params(params: "ObservationParams") -> None:
    """
    Nicely formatted summary of observation parameters.
    Handles both REAL (velocity_range + vlsr) and SYNTHETIC (velocity_width_kms) modes.
    """
    print("=" * 50)
    print(f" Dataset name:            {params.name}")
    print(f" Mode:                    {'REAL (observed cube)' if params.is_real else 'SYNTHETIC (toy cube)'}")
    print("-" * 50)
    print(f" Distance (pc):           {params.distance:8.2f}")
    print(f" Field of view (arcsec):  {params.fov:8.3f}")

    if params.is_real:
        vmin, vmax = params.velocity_range
        print(f" Velocity range (m/s):    {vmin:8.1f} → {vmax:8.1f}")
        print(f" VLSR (m/s):              {params.vlsr:8.2f}")
    else:  # synthetic
        print(f" Velocity width (km/s):   {params.velocity_width_kms:8.2f}")
        bounds = params.velocity_bounds_ms()
        print(f" Velocity bounds (m/s):   {bounds[0]:8.1f} → {bounds[1]:8.1f}")
        print(f" Center velocity (m/s):   {params.velocity_center_ms():8.2f}")

    print("-" * 50)
    print(f" Number of rays:          {params.nray:8d}")
    print(f" Inclination (deg):       {params.incl:8.2f}")
    print(f" Phi (deg):               {params.phi:8.2f}")
    print(f" Position angle (deg):    {params.posang:8.2f}")
    print(f" Slab thickness z_width (AU): {params.z_width:8.2f}")
    print("=" * 50)


    
# ----------------------------------------------------------------------------- #
# Frequencies
# ----------------------------------------------------------------------------- #
def compute_camera_freqs(
    num_freqs: int,
    half_width_kms: float = None,
    nu0: float = None,
    v_sys: float = 0.0,
    num_subfreq: int = 1,
    subfreq_width: Optional[float] = None,
    width_kms: Optional[float] = None,  # deprecated alias for half_width_kms
) -> jnp.ndarray:
    """
    Build a (possibly sub-sampled) frequency grid around line center.

    Channels span velocities in [-half_width_kms, +half_width_kms] (i.e. the
    TOTAL velocity coverage is 2 * half_width_kms). `width_kms` is accepted as
    a deprecated alias with the same half-width meaning.
    """
    if half_width_kms is None:
        half_width_kms = width_kms
    if half_width_kms is None or nu0 is None:
        raise TypeError("compute_camera_freqs requires half_width_kms and nu0.")
    # Doppler offsets: radio convention, v positive = receding = lower frequency
    v = (2 * jnp.arange(num_freqs) / (num_freqs - 1) - 1.0) * half_width_kms  # km/s
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
    nfreq = freqs.shape[0]
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
        rays.coords_xyz, rays.obs_dir, nu0, rays.pixel_area, rays.distance_pc
    )
    images = jnp.clip(jnp.nan_to_num(images).reshape(-1, rays.ny, rays.nx)[:nfreq], 0.0)
    
    return images


def render_cube_with_dust(
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
    dust_alpha: jnp.ndarray,    # (H, W, N)
    dust_temp_profile: jnp.ndarray,  # (H, W, N)
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

    # Build eval_freqs (prepend a continuum sentinel) before any sharding
    freq_diff = freqs[1] - freqs[0]
    distant_freq = freqs[0] - 50 * freq_diff
    eval_freqs = jnp.concatenate([jnp.array([distant_freq]), freqs])
    neval = eval_freqs.shape[0]

    # Choose backend
    if backend == "pmap":
        compute_fn = line_rte.compute_spectral_cube_dust_pmap
        eval_freqs = shard_with_padding(eval_freqs)  # (ndev, F_per_dev)
    elif backend == "vmap":
        compute_fn = line_rte.compute_spectral_cube_dust_vmap
    elif backend in ("scan", "none"):
        # 'none' maps to scan: processes one channel at a time without vmap/pmap.
        compute_fn = line_rte.compute_spectral_cube_dust_scan
    else:
        raise ValueError(f"Unknown backend={backend!r}. Must be 'vmap', 'pmap', 'scan', or 'none'.")

    images = compute_fn(
        eval_freqs, velocity_ray, alpha_tot, n_up, n_dn,
        mol.a_ud, mol.b_ud, mol.b_du,
        rays.coords_xyz, rays.obs_dir, nu0, rays.pixel_area, rays.distance_pc,
        dust_alpha, dust_temp_profile
    )

    images = jnp.nan_to_num(images).reshape(-1, rays.ny, rays.nx)[:neval]

    # images[0] is the continuum channel (at distant_freq, far from the line).
    # Subtract it from each line channel and clip negatives.
    continuum = images[0]
    images = jnp.clip(images[1:] - continuum, 0.0)

    return images



def render_tau1_cube(
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
    Render Tau=1 surface using pre-sampled ray fields and line data in `mol`.

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
        Cube with z coordinate of Tau=1 surface (NaNs sanitized).
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
        compute_fn = line_rte.compute_tau1_cube_pmap
        freqs = shard_with_padding(freqs)  # (ndev, F_per_dev)
    elif backend == "vmap":
        compute_fn = line_rte.compute_tau1_cube_vmap
    elif backend == "none":
        compute_fn = line_rte.compute_tau1_cube
    else:
        raise ValueError(f"Unknown backend={backend!r}. Must be 'vmap', 'pmap', or 'none'.")
    
    images = compute_fn(
        freqs, velocity_ray, alpha_tot, n_up, n_dn,
        mol.b_ud, mol.b_du,
        rays.coords_xyz, rays.obs_dir, nu0
    )
    
    return images


def render_emission_height_cube(
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
    cutoff_intensity: float = 1e-14
) -> jnp.ndarray:
    """
    Render Tau=1 surface using pre-sampled ray fields and line data in `mol`.

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
        Cube with z coordinate of Tau=1 surface (NaNs sanitized).
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
        compute_fn = line_rte.compute_emission_height_cube_pmap
        freqs = shard_with_padding(freqs)  # (ndev, F_per_dev)
    elif backend == "vmap":
        compute_fn = line_rte.compute_emission_height_cube_vmap
    elif backend == "none":
        compute_fn = line_rte.compute_emission_height_cube
    else:
        raise ValueError(f"Unknown backend={backend!r}. Must be 'vmap', 'pmap', or 'none'.")
    
    z_cube = compute_fn(
        freqs, velocity_ray, alpha_tot, n_up, n_dn,
        mol.a_ud, mol.b_ud, mol.b_du,
        rays.coords_xyz, rays.obs_dir, nu0, cutoff=cutoff_intensity
    )

    return z_cube



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
          - coords_xyz : (H, W, N, 3) world-space samples along each ray (near → far, index 0 = top/near, index N-1 = bottom/far)
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


