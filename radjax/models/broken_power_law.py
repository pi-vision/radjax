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
from typing import Any, Dict
from pathlib import Path
import dataclasses as dc

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from flax import struct
import yaml

# local constants
from ..core.consts import M_sun


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
    v_turb: float  # m/s internally

    # Model grid
    resolution: int
    z_min: float
    z_max: float
    r_min: float
    r_max: float

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

def params_from_yaml(source: Union[str, Path, Dict[str, Any]]) -> DiskParams:
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

    # conversions to internal units
    d["M_star"] = float(d["M_star"]) * M_sun
    d["M_gas"]  = float(d["M_gas"])  * M_sun

    # coerce numerics
    float_keys = [
        "T_mid1","T_atm1","q","q_in","z_q0","delta",
        "r_scale","r_in","r_break","log_r_c","gamma",
        "M_gas","M_star","v_turb","z_min","z_max","r_min","r_max",
    ]
    for k in float_keys:
        if k in d and d[k] is not None:
            d[k] = float(d[k])
    d["resolution"] = int(d["resolution"])

    return DiskParams(**d).validate()


def params_to_yaml_path(params: DiskParams, path: str | Path) -> None:
    """
    Write back only the `disk:` block, preserving other YAML sections.

    Reverse conversions:
      M_star, M_gas: grams → [M_sun]
      v_turb:        m/s   → [km/s]
    """
    p = Path(path)
    doc: Dict[str, Any] = {}
    if p.exists():
        with open(p, "r") as f:
            doc = yaml.safe_load(f) or {}

    disk_out = {
        # thermal
        "T_mid1": params.T_mid1,
        "T_atm1": params.T_atm1,
        "q": params.q,
        "q_in": params.q_in,
        "r_break": params.r_break,
        "log_r_c": params.log_r_c,
        "gamma": params.gamma,
        # mass content (back to M_sun)
        "M_star": params.M_star / M_sun,
        "M_gas": params.M_gas / M_sun,
        # geometry / scaling
        "r_in": params.r_in,
        "r_scale": params.r_scale,
        "z_q0": params.z_q0,
        "delta": params.delta,
        # kinematics
        "v_turb": params.v_turb / 1.0e3,
        # model grid
        "resolution": int(params.resolution),
        "z_min": params.z_min,
        "z_max": params.z_max,
        "r_min": params.r_min,
        "r_max": params.r_max,
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
        Disk parameter object loaded via `params_from_yaml(...)`.
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

    
def temperature_profile(z: jnp.ndarray, r: jnp.ndarray, params: DiskParams) -> jnp.ndarray:
    """
    Temperature T(z, r) with piecewise-q radial scaling and mid/atm blending.

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
    """
    q_eff = jnp.where(r <= params.r_break, params.q_in, params.q)
    T_mid = params.T_mid1 * (r / params.r_scale) ** q_eff
    T_atm = params.T_atm1 * (r / params.r_scale) ** q_eff
    z_q   = params.z_q0 * (r / params.r_scale) ** 1.3
    return jnp.where(
        z < z_q,
        T_atm + (T_mid - T_atm) * jnp.cos(jnp.pi * z / (2.0 * z_q)) ** (2.0 * params.delta),
        T_atm,
    )

# Pre-jitted temp
temperature_profile_jit = jax.jit(temperature_profile)