"""
Parametric protoplanetary-disk components (broken power-law temperature).

Conventions
-----------
r, z in AU (unless explicitly multiplied by `au`);
temperatures in K;
velocities in m/s;
masses in grams (consistent with G).
"""

from __future__ import annotations
from typing import Any, Dict, Literal, Optional, Union
from pathlib import Path
import dataclasses as dc

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from flax import struct
import yaml

# local constants
from ..core.consts import M_sun, au, m_mol_h
from ..core import phys
from ..core import grid
from ..core import sensor
from ..core import chemistry as chem
from ..core.inference import SamplerState

@struct.dataclass
class BaseDisk:
    """
    Container for disk grids and background H₂ fields.

    Attributes
    ----------
    z : jnp.ndarray
        Vertical grid [AU], shape (Nz,).
    r : jnp.ndarray
        Radial grid [AU], shape (Nr,).
    z_r_mesh : jnp.ndarray
        Stacked meshgrid (z, r) coordinates, shape (Nz, Nr, 2).
    bbox : jnp.ndarray
        2×2 array of [(z_min, z_max), (r_min, r_max)] in cgs [cm].

    h2_nd : jnp.ndarray, optional
        Local H2 number density n_H2(z,r) [cm^-3], shape (Nz, Nr).
    h2_N : jnp.ndarray, optional
        Vertical H2 column density N_H2(r) [cm^-2], shape (Nr,).
    """

    z: jnp.ndarray
    r: jnp.ndarray
    z_r_mesh: jnp.ndarray
    bbox: jnp.ndarray
    h2_nd: Optional[jnp.ndarray] = None
    h2_N: Optional[jnp.ndarray] = None

def print_base_params(base: BaseDisk) -> None:
    """Pretty summary of BaseDisk grids and H₂ fields."""
    print("BaseDisk (grids + baseline H₂ fields):")
    print(f"  r grid (AU):                  {base.r.shape} "
          f"[{base.r.min():.2f} → {base.r.max():.2f}]")
    print(f"  z grid (AU):                  {base.z.shape} "
          f"[{base.z.min():.2f} → {base.z.max():.2f}]")
    print("")
    if base.h2_nd is not None:
        print(f"  H₂ number density n_H2 (cm^-3): {base.h2_nd.shape}, "
              f"range {base.h2_nd.min():.2e} → {base.h2_nd.max():.2e}")
    if base.h2_N is not None:
        print(f"  H₂ column density N_H2 (cm^-2): {base.h2_N.shape}, "
              f"range {base.h2_N.min():.2e} → {base.h2_N.max():.2e}")
    print("")
    print(f"  Bounding box [cm]:            {base.bbox.tolist()}")

@struct.dataclass
class DiskParams:
    """
    Parametric disk parameters.

    Thermal:
      T_mid1, T_atm1 : temperature normalizations at r = r_scale.
      q, q_in        : radial exponents (use q_in for r <= r_break).
      z_q0           : height scale for temperature peak.
      delta          : mid/atm blending exponent.

    Radial / surface density:
      r_scale, r_in, r_break, log_r_c, gamma, M_gas.

    Dynamics / kinematics:
      M_star         : stellar mass.
      v_turb         : dimensionless scaling of local sound speed

    Model grid (resolution / bounds):
      resolution     : z/r grid resolution (int).
      z_min, z_max   : vertical bounds.
      r_min, r_max   : radial bounds.
    """
    # Thermal
    T_mid1: float
    T_atm1: float
    q: float
    q_in: float
    z_q0: float
    delta: float

    # Radial / surface density
    r_scale: float
    r_in: float
    r_break: float
    log_r_c: float
    gamma: float
    M_gas: float  # stored in grams internally

    # Dynamics / kinematics
    M_star: float  # grams internally
    v_turb: float  # dimensionless (scales local sound speed)

    # Model grid
    resolution: int
    z_min: float
    z_max: float
    r_min: float
    r_max: float

    # --- new ring fields ---
    ring_radii: jnp.ndarray = dc.field(default_factory=lambda: jnp.empty(0))
    ring_widths: jnp.ndarray = dc.field(default_factory=lambda: jnp.empty(0))
    ring_densities: jnp.ndarray = dc.field(default_factory=lambda: jnp.empty(0))
    ring_heights: jnp.ndarray = dc.field(default_factory=lambda: jnp.empty(0))

    # Rotation direction: +1 = CCW in disk plane, -1 = CW in disk plane.
    # Disks described as "CCW on the sky" in ALMA papers (e.g. HD163296) need -1.
    rotation_flip: float = 1.0

    def validate(self) -> "DiskParams":
        """Lightweight validation on creation."""
        if self.r_scale <= 0: raise ValueError("r_scale must be > 0")
        if self.T_mid1 <= 0 or self.T_atm1 <= 0: raise ValueError("temperatures must be > 0")
        if self.M_star <= 0: raise ValueError("M_star must be > 0")
        if self.M_gas  <= 0: raise ValueError("M_gas must be > 0")
        if self.resolution <= 0: raise ValueError("resolution must be > 0")
        if not (self.z_max >= self.z_min): raise ValueError("z_max must be ≥ z_min")
        if not (self.r_max >= self.r_min): raise ValueError("r_max must be ≥ r_min")
        return self


def disk_from_yaml(source: Union[str, Path, Dict[str, Any]]) -> DiskParams:
    """
    Load DiskParams from the `disk:` section of a YAML (or dict).

    Unit conversions applied to `disk:`:
      M_star, M_gas: [M_sun] → grams
    """
    if isinstance(source, (str, Path)):
        with open(Path(source), "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = dict(source)

    if "disk" not in cfg:
        raise KeyError("YAML is missing required `disk` section.")

    d = dict(cfg["disk"])

    # --- extract rings before dataclass construction ---
    rings = d.pop("rings", [])

    # unit conversions
    d["M_star"] = float(d["M_star"]) * M_sun
    d["M_gas"]  = float(d["M_gas"])  * M_sun

    float_keys = [
        "T_mid1","T_atm1","q","q_in","z_q0","delta",
        "r_scale","r_in","r_break","log_r_c","gamma",
        "M_gas","M_star","v_turb","rotation_flip","z_min","z_max","r_min","r_max",
    ]

    for k in float_keys:
        if k in d and d[k] is not None:
            d[k] = float(d[k])

    d["resolution"] = int(d["resolution"])

    # ---- convert rings BEFORE constructing DiskParams ----
    if rings:
        ring_radii = jnp.array([r["radius"] for r in rings])
        ring_widths = jnp.array([r["width"] for r in rings])
        ring_densities = jnp.array([r["density"] for r in rings])
        ring_heights = jnp.array([r["height"] for r in rings])
    else:
        ring_radii = jnp.empty(0)
        ring_widths = jnp.empty(0)
        ring_densities = jnp.empty(0)
        ring_heights = jnp.empty(0)

    disk = DiskParams(
        **d,
        ring_radii=ring_radii,
        ring_widths=ring_widths,
        ring_densities=ring_densities,
        ring_heights=ring_heights,
    ).validate()

    return disk


def params_to_yaml_path(params: DiskParams, path: str | Path) -> None:
    """
    Write back only the `disk:` block, preserving other YAML sections.
    If the YAML file does not exist, it will be created.

    Reverse conversions:
      M_star, M_gas: grams → [M_sun]
      v_turb:        m/s   → [km/s]
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    doc: Dict[str, Any] = {}
    if p.exists():
        with open(p, "r") as f:
            doc = yaml.safe_load(f) or {}

    disk_out = {
        # thermal
        "T_mid1": float(params.T_mid1),
        "T_atm1": float(params.T_atm1),
        "q": float(params.q),
        "q_in": float(params.q_in),
        "r_break": float(params.r_break),
        "log_r_c": float(params.log_r_c),
        "gamma": float(params.gamma),
    
        # mass content (back to M_sun)
        "M_star": float(params.M_star / M_sun),
        "M_gas": float(params.M_gas / M_sun),
    
        # geometry / scaling
        "r_in": float(params.r_in),
        "r_scale": float(params.r_scale),
        "z_q0": float(params.z_q0),
        "delta": float(params.delta),
    
        # kinematics
        "v_turb": float(params.v_turb),  # dimensionless
    
        # model grid
        "resolution": int(params.resolution),
        "z_min": float(params.z_min),
        "z_max": float(params.z_max),
        "r_min": float(params.r_min),
        "r_max": float(params.r_max),
    }


    doc["disk"] = disk_out
    with open(p, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)

def print_params(params: "DiskParams") -> None:
    """
    Nicely formatted summary of parametric disk parameters.

    Parameters
    ----------
    params : DiskParams
        Disk parameter object loaded via `disk_from_yaml(...)`.
    """
    print("Disk parameters (parametric broken power law model):")
    print("  Midplane T_norm (K):       {:8.2f}".format(params.T_mid1))
    print("  Atmosphere T_norm (K):     {:8.2f}".format(params.T_atm1))
    print("  q (outer exponent):        {:8.3f}".format(params.q))
    print("  q_in (inner exponent):     {:8.3f}".format(params.q_in))
    print("  r_break (AU):              {:8.2f}".format(params.r_break))
    print("  log(r_c [AU]):             {:8.2f}".format(params.log_r_c))
    print("  gamma (Σ slope):           {:8.3f}".format(params.gamma))
    print()
    print("  Stellar mass (M_sun):      {:8.3f}".format(params.M_star / M_sun))
    print("  Gas mass (M_sun):          {:8.3f}".format(params.M_gas / M_sun))
    print()
    print("  r_in (AU):                 {:8.2f}".format(params.r_in))
    print("  r_scale (AU):              {:8.2f}".format(params.r_scale))
    print("  z_q0 (AU):                 {:8.2f}".format(params.z_q0))
    print("  delta (blend exp):         {:8.3f}".format(params.delta))
    print()
    print("  Turbulence parameter:      {:8.3f}".format(params.v_turb))
    print()
    print("  Resolution (z/r):          {:8d}".format(params.resolution))
    print("  z range (AU):              {:8.2f} → {:8.2f}".format(params.z_min, params.z_max))
    print("  r range (AU):              {:8.2f} → {:8.2f}".format(params.r_min, params.r_max))

def co_disk_from_params(
    disk_params: "DiskParams",
    chem_params: "ChemistryParams",
    *,
    pressure_correction: bool = True,
):
    """
    Build grids and compute core CO disk params fields from parameter sets.

    Parameters
    ----------
    disk_param : DiskParams
        Parametric disk parameters.
    chem_params : ChemistryParams
        Chemistry thresholds and tracer mass.
    pressure_correction : bool, optional
        If True, include pressure support in azimuthal velocity.

    Returns
    -------
    temperature : jnp.ndarray
        temperature(z,r) [K], shape (resolution, resolution).
    velocity : jnp.ndarray
        v_phi(z,r) [m/s], same shape.
    co_nd : jnp.ndarray
        n_CO(z,r) [cm^-3], same shape.
    base_disk : PreparedDisk
        Small container with (z, r) grids and `bbox` in cgs.
    """
    # --- grids from disk_params
    z = jnp.linspace(disk_params.z_min, disk_params.z_max, int(disk_params.resolution))
    r = jnp.linspace(disk_params.r_min, disk_params.r_max, int(disk_params.resolution))
    z_mesh, r_mesh = jnp.meshgrid(z, r, indexing="ij")

    # bounding box in cgs (cm)
    bbox = jnp.array([
        (au * disk_params.z_min, au * disk_params.z_max),
        (au * disk_params.r_min, au * disk_params.r_max),
    ])

    # --- temperature (broken power-law)
    temperature = temperature_profile(z_mesh, r_mesh, disk_params)

    # --- hydrostatic number density 
    h2_nd = phys.number_density_profile(
        z_mesh, r_mesh, temperature,
        gamma=disk_params.gamma,
        r_in_au=disk_params.r_in,
        r_c_au=10.0 ** disk_params.log_r_c,
        M_gas=disk_params.M_gas,
        M_star=disk_params.M_star,
        m_mol_h=m_mol_h,
    )

    # --- azimuthal velocity; rotation_flip controls CW vs CCW in disk plane
    v_phi = phys.velocity_profile(
        z_mesh, r_mesh, h2_nd, temperature,
        M_star=disk_params.M_star,
        m_mol_h=m_mol_h,
        pressure_correction=pressure_correction,
    )
    v_phi = v_phi * disk_params.rotation_flip

    # --- column density & CO abundance
    h2_N = phys.surface_density(z_mesh, h2_nd)  # [cm^-2]
    Xco  = chem.co_abundance_profile(
        h2_N, temperature,
        freezeout=chem_params.freezeout,
        N_dissoc=chem_params.N_dissoc,
        N_desorp=chem_params.N_desorp,
        co_abundance=chem_params.co_abundance,
    )
    co_nd = Xco * h2_nd

    base_disk = BaseDisk(
        z=z,
        r=r,
        z_r_mesh=jnp.stack((z_mesh, r_mesh), axis=-1),
        bbox=bbox,
        h2_nd=h2_nd,
        h2_N=h2_N,
    )

    return temperature, v_phi, co_nd, base_disk

def temperature_profile(z: jnp.ndarray, r: jnp.ndarray, params: DiskParams) -> jnp.ndarray:
    """
    Temperature T(z, r) with piecewise-q radial scaling and mid/atm blending.

    Both inner (r <= r_break) and outer branches are anchored at r_scale, so
    they are generally discontinuous at r_break (unless q_in == q).

    Parameters
    ----------
    z, r : jnp.ndarray
        Height and radius grids [AU], broadcastable to a common shape.
    params : DiskParams
        Holds thermal/geometry/blending parameters.

    Returns
    -------
    jnp.ndarray
        Temperature [K] with shape broadcast(z, r).

    See Also
    --------
    continuous_temperature_profile : variant that anchors the inner branch at
        r_break so T is continuous across the break.
    """
    r_safe = jnp.maximum(r, 1e-3)
    q_eff = jnp.where(r_safe <= params.r_break, params.q_in, params.q)
    T_mid = params.T_mid1 * (r_safe / params.r_scale) ** q_eff
    T_atm = params.T_atm1 * (r_safe / params.r_scale) ** q_eff
    z_q   = params.z_q0 * (r_safe / params.r_scale) ** 1.3
    cos_arg = jnp.clip(jnp.pi * jnp.abs(z) / (2.0 * z_q), 0.0, 0.5 * jnp.pi)
    return jnp.where(
        jnp.abs(z) < z_q,
        T_atm + (T_mid - T_atm) * jnp.cos(cos_arg) ** (2.0 * params.delta),
        T_atm,
    )


def continuous_temperature_profile(z: jnp.ndarray, r: jnp.ndarray, params: DiskParams) -> jnp.ndarray:
    """
    Temperature T(z, r) — continuous broken power-law variant.

    The inner branch is anchored at r_break rather than r_scale, so T is
    continuous across the break:
        r > r_break: T ∝ (r / r_scale)^q
        r ≤ r_break: T ∝ (r_break / r_scale)^q × (r / r_break)^q_in

    Parameters
    ----------
    z, r : jnp.ndarray
        Height and radius grids [AU], broadcastable to a common shape.
    params : DiskParams
        Holds thermal/geometry/blending parameters.

    Returns
    -------
    jnp.ndarray
        Temperature [K] with shape broadcast(z, r).

    See Also
    --------
    temperature_profile : the default (discontinuous) variant.
    """
    r_safe = jnp.maximum(r, 1e-3)
    scale_outer = (r_safe / params.r_scale) ** params.q
    # Clamp r_break: a non-positive r_break is physically meaningless but
    # would produce NaN (negative base to fractional power) in the untaken
    # where-branch, poisoning gradients.
    rb_safe = jnp.maximum(params.r_break, 1e-3)
    scale_inner = (rb_safe / params.r_scale) ** params.q * (r_safe / rb_safe) ** params.q_in
    radial_scale = jnp.where(r_safe <= params.r_break, scale_inner, scale_outer)
    T_mid = params.T_mid1 * radial_scale
    T_atm = params.T_atm1 * radial_scale
    z_q   = params.z_q0 * (r_safe / params.r_scale) ** 1.3
    cos_arg = jnp.clip(jnp.pi * jnp.abs(z) / (2.0 * z_q), 0.0, 0.5 * jnp.pi)
    return jnp.where(
        jnp.abs(z) < z_q,
        T_atm + (T_mid - T_atm) * jnp.cos(cos_arg) ** (2.0 * params.delta),
        T_atm,
    )

def forward_model_with_rays(    
    *,
    disk_params: DiskParams, 
    chem_params: ChemistryParams,
    mol: MolecularData,
    rays: RayBundle ,     
    freqs: jnp.ndarray,       
    beam_kernel: Optional[jnp.ndarray] = None,
    output: Literal["image","vis"] = "image",
    backend: Literal["vmap","pmap","none"] = "vmap",
    use_pressure_correction: bool = True,
):
    """
    Forward-model a spectral cube from parametric disk parameters using a
    preconstructed RayBundle. This is the fastest inference mode, since the
    ray geometry does not change between evaluations.

    Parameters
    ----------
    disk_params : DiskParams
        Parametric disk parameters for the current model evaluation.
    chem_params : ChemistryParams
        Chemistry thresholds and tracer abundance settings.
    mol : MolecularData
        Molecular line data (energy levels, transitions, Einstein coefficients).
    rays : RayBundle
        Prebuilt camera rays describing the projection geometry.
    freqs : jnp.ndarray of shape (F,)
        Frequency grid in Hz at which to compute the spectral cube.
    beam_kernel : jnp.ndarray of shape (ny, nx), optional
        2D beam kernel. If provided, each frequency channel is convolved with it.
        If ``None`` (default), raw model images are returned.
    output : {"image", "vis"}, default="image"
        Select output mode:
          * ``"image"`` – return spectral cube (optionally beam-convolved).
          * ``"vis"`` – placeholder, not implemented.
    backend : {"vmap", "pmap", "none"}, default="vmap"
        Backend for the radiative transfer solver:
          * ``"vmap"`` – vectorized over frequency (fastest single-device).
          * ``"pmap"`` – parallelize across devices (multi-GPU/TPU).
          * ``"none"`` – plain Python loop (slowest, debugging only).
    use_pressure_correction : bool, default=True
        If True, include pressure support in azimuthal velocity.

    Returns
    -------
    jnp.ndarray
        Spectral cube with shape (F, ny, nx).
        If ``output="image"``: returns beam-convolved cube if a kernel is provided,
        else raw cube.
        If ``output="vis"``: raises ``NotImplementedError``.

    Raises
    ------
    NotImplementedError
        If ``output="vis"`` is selected (visibility sampling not implemented).
    ValueError
        If ``output`` is not one of {"image", "vis"}.

    Notes
    -----
    For MCMC, use this function when the ray geometry is fixed, i.e. inclination,
    position angle, and related viewing parameters are *not* part of the parameter
    vector ``θ``. If geometry is variable, use
    :func:`forward_model_rotate_rays` instead.
    """
    # 1) Disk fields on (z,r) grid
    temperature, v_phi, co_nd, base_disk = co_disk_from_params(
        disk_params=disk_params,
        chem_params=chem_params,
        pressure_correction=use_pressure_correction,
    )

    # 2) Sample fields along rays
    nd_ray, temperature_ray, velocity_ray = sensor.sample_symmetric_disk__along_rays(
        rays=rays,
        bbox=base_disk.bbox,
        co_nd=co_nd,
        temperature=temperature,
        v_phi=v_phi,
    )

    # 3) Radiative transfer -> raw cube
    images = sensor.render_cube(
        rays=rays,
        nd_ray=nd_ray,
        temperature_ray=temperature_ray,
        velocity_ray=velocity_ray,
        nu0=mol.nu0,
        freqs=freqs,
        v_turb=disk_params.v_turb,
        mol=mol,
        backend=backend,
    )

    # 4) Post-processing
    if output == "image":
        return sensor.fftconvolve_vmap(images, beam_kernel) if beam_kernel is not None else images

    if output == "vis":
        raise NotImplementedError(
            "Visibility sampling not implemented yet. "
            "Planned: per-channel FFT to UV grid + (u,v) sampling from measured baselines."
        )

    raise ValueError("output must be 'image' or 'vis'")


def forward_model_rotate_rays(
    *,
    disk_params: DiskParams,
    chem_params: ChemistryParams,
    mol: MolecularData,
    rays_base: RayBundle,
    incl_deg: float,
    phi_deg: float,
    posang_deg: float,
    freqs: jnp.ndarray,
    beam_kernel: Optional[jnp.ndarray] = None,
    output: Literal["image","vis"] = "image",
    backend: Literal["vmap","pmap","none"] = "vmap",
    use_pressure_correction: bool = True,
) -> jnp.ndarray:
    """
    Forward-model a spectral cube from parametric disk parameters by rotating
    a canonical RayBundle according to new viewing angles
    (inclination, azimuthal rotation, and position angle).

    This mode is intended for inference when geometric parameters (e.g.
    inclination, position angle) are part of the sampled parameter vector ``θ``.

    Parameters
    ----------
    disk_params : DiskParams
        Parametric disk parameters for the current model evaluation.
    chem_params : ChemistryParams
        Chemistry thresholds and tracer abundance settings.
    mol : MolecularData
        Molecular line data (energy levels, transitions, Einstein coefficients).
    rays_base : RayBundle
        Canonical camera rays built once at reference geometry (e.g. zero inclination).
        Geometry is updated each call by rotation, avoiding a full ray rebuild.
    incl_deg : float
        Inclination angle in degrees.
    phi_deg : float
        Azimuthal rotation angle in degrees.
    posang_deg : float
        Position angle (roll about line of sight) in degrees.
    freqs : jnp.ndarray of shape (F,)
        Frequency grid in Hz at which to compute the spectral cube.
    beam_kernel : jnp.ndarray of shape (ny, nx), optional
        2D beam kernel. If provided, each frequency channel is convolved with it.
        If ``None`` (default), raw model images are returned.
    output : {"image", "vis"}, default="image"
        Select output mode:
          * ``"image"`` – return spectral cube (optionally beam-convolved).
          * ``"vis"`` – placeholder, not implemented.
    backend : {"vmap", "pmap", "none"}, default="vmap"
        Backend for the radiative transfer solver:
          * ``"vmap"`` – vectorized over frequency (fastest single-device).
          * ``"pmap"`` – parallelize across devices (multi-GPU/TPU).
          * ``"none"`` – plain Python loop (slowest, debugging only).
    use_pressure_correction : bool, default=True
        If True, include pressure support in azimuthal velocity.

    Returns
    -------
    jnp.ndarray
        Spectral cube with shape (F, ny, nx).
        If ``output="image"``: returns beam-convolved cube if a kernel is provided,
        else raw cube.
        If ``output="vis"``: raises ``NotImplementedError``.

    Raises
    ------
    NotImplementedError
        If ``output="vis"`` is selected (visibility sampling not implemented).
    ValueError
        If ``output`` is not one of {"image", "vis"}.

    Notes
    -----
    For MCMC, use this function when the geometry (inclination, position angle)
    is included in the parameter vector ``θ``. This avoids recomputing the full
    ray grid (distance, FOV, z_width) and only rotates a precomputed canonical
    RayBundle. If camera scale parameters also vary, a full ray rebuild is required
    (see :func:`sensor.rays_from_params`).
    """
    rays = grid.rotate_rays(
        rays_base,
        incl_deg=incl_deg,
        phi_deg=phi_deg,
        posang_deg=posang_deg,
    )
    return forward_model_with_rays(
        disk_params=disk_params,
        chem_params=chem_params,
        mol=mol,
        rays=rays,
        freqs=freqs,
        beam_kernel=beam_kernel,
        output=output,
        backend=backend,
        use_pressure_correction=use_pressure_correction,
    )
