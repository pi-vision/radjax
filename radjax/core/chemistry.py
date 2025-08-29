"""
Disk chemistry and parameter container (matches yaml `chemistry:` block).

YAML schema (example)
---------------------
chemistry:
  molecule: CO
  line_index: 2
  line_name: "CO(2-1)"
  co_abundance: 1.0e-4
  freezeout: 19.0
  N_dissoc: 1.256e21
  N_desorp: null
  molecular_table: "data/molecular_tables/molecule_12c16o.inp"

Conventions 
------------
- freezeout : K (freezeout temperature threshold)
- N_dissoc  : cm^-2 (dissociation threshold column)
- N_desorp  : cm^-2 (optional desorption/reintroduction threshold column; can be None)
- co_abundance : dimensionless fraction
- line_index : integer J-upper (e.g., 2 for CO(2-1), 3 for CO(3-2))
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from flax import struct
import yaml

import numpy as np
import importlib
import importlib.resources as pkg_resources

from .consts import (
    cc,          # speed of light      [cm/s]
    hh,          # Planck constant     [erg*s]
    kk,          # Boltzmann constant  [erg/K]
    ghz,         # GHz -> Hz (or consistent internal units)
)

@struct.dataclass
class MolecularData:
    transition: int                      # 1-based J-upper index (e.g., 2 for CO J=2-1): 
    energy_levels: jnp.ndarray
    radiative_transitions: jnp.ndarray
    a_ud: jnp.ndarray
    b_ud: jnp.ndarray
    b_du: jnp.ndarray
    """
    Load molecular tables and Einstein coefficients for the transition specified
    in a ChemistryParams object.

    This function reads:
      - `chem_params.molecular_table` : path to the RADMC-3D style molecular table file
      - `chem_params.line_index`      : 1-based transition index (e.g., 2 for CO J=2-1)

    It returns a MolecularData container bundling:
      - full energy level table
      - full radiative transitions table
      - selected line rest frequency (`nu0`)
      - Einstein A (a_ud) and B coefficients (b_ud, b_du) for the chosen transition
      - transition : int, 1-based index of the line (e.g., 3 for CO(3-2)).

    Parameters
    ----------
    chem_params : ChemistryParams
        Chemistry configuration containing the table path and transition index.

    Returns
    -------
    mol : MolecularData
        Immutable JAX-friendly container with molecular tables and
        coefficients for the selected transition.
    """
    
def load_molecular_tables(chem_params: "ChemistryParams") -> MolecularData:
    """
    Load CO molecular data from a .inp table and build a MolecularData struct for the transition specified.

    Parameters
    ----------
    chem_params: ChemistryParams
    This function reads:
      - `chem_params.molecular_table` : path to the .inp molecular table file
      - `chem_params.line_index`      : 1-based transition index (e.g., 2 for CO J=2-1)

    Returns
    -------
    mol : MolecularData
        Container with energy levels, radiative transitions,
        and Einstein coefficients for the chosen transition.

    Notes
    -----
    See e.g. tables below from molecule_12c16o.inp
    CO Molecular tables can be found here: https://home.strw.leidenuniv.nl/~moldata/CO.html
    ------------------------------------
    LEVEL   ENERGIES(cm^-1)   WEIGHT     J
      3     11.534919938	    5.0	     2
      4     23.069512649	    7.0	     3
    ---------------------------------------------------------
    TRANS   UP   LOW  EINSTEINA(s^-1)  FREQ(GHz)       E_u(K)
      3     4     3   2.497e-06        345.7959899     33.19
    """
    relpath = chem_params.molecular_table

    pkg = importlib.import_module('radjax')
    pkg_root = Path(pkg.__file__).resolve().parent.parent
    
    # Get a real filesystem path to the packaged resource
    with pkg_resources.as_file(pkg_root / relpath) as p:
        table_path = str(p)   
        
    # Load raw arrays
    energy_levels = np.loadtxt(table_path, skiprows=7, max_rows=41)
    radiative_transitions = np.loadtxt(table_path, skiprows=51, max_rows=40)

    # Compute Einstein A, B coefficients for this transition
    a_ud, b_ud, b_du = einstein_coefficients(
        energy_levels, radiative_transitions, transition=chem_params.line_index
    )

    # Wrap in MolecularData
    return MolecularData(
        transition=chem_params.line_index,
        energy_levels=energy_levels,
        radiative_transitions=radiative_transitions,
        a_ud=a_ud,
        b_ud=b_ud,
        b_du=b_du,
    )

def einstein_coefficients(
    energy_levels: jnp.ndarray,
    radiative_transitions: jnp.ndarray,
    transition: int,
) -> Tuple[float, float, float, float]:
    """
    Compute Einstein coefficients for a specific radiative transition.

    Parameters
    ----------
    energy_levels : jnp.ndarray
        Shape (num_levels, 4). Columns:
        [LEVEL, ENERGIES(cm^-1), WEIGHT, J]
    radiative_transitions : jnp.ndarray
        Shape (num_levels-1, 6). Columns:
        [TRANS, UP, LOW, EINSTEINA(s^-1), FREQ(GHz), E_u(K)]
    transition : int,
        1-based index of the line (e.g., 3 for CO(3-2)).

    Returns
    -------
    (nu0, a_ud, b_ud, b_du) : Tuple[float, float, float, float]
        nu0 : line rest frequency (in your internal frequency units)
        a_ud : Einstein A (spontaneous emission)
        b_ud : Einstein B (stimulated emission, up->down)
        b_du : Einstein B (absorption, down->up)

    Notes
    -----
    - gratio = weight_UP / weight_LOW
    - b_ud = (c^2 / (2 h nu0^3)) * a_ud
    - b_du = b_ud * gratio
    
    See: RADMC-3D line RT docs for background:
    https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/lineradtrans.html
    """
    # extract up and down indices from radiative transition table (e.g. 4,3 for second CO line at 345GHz)
    line_index = transition - 1
    nu0 = radiative_transitions[line_index, 4] * ghz
    up_idx = jnp.array(radiative_transitions[line_index, 1] - 1, int)
    dn_idx = jnp.array(radiative_transitions[line_index, 2] - 1, int)

    gratio = energy_levels[up_idx, 2] / energy_levels[dn_idx, 2]
    a_ud = radiative_transitions[line_index, 3]
    b_ud = (cc**2 / (2 * hh * nu0**3)) * a_ud
    b_du = b_ud * gratio
    return a_ud, b_ud, b_du

def n_up_down(
    gas_nd: jnp.ndarray,
    gas_t: jnp.ndarray,
    energy_levels: jnp.ndarray,
    radiative_transitions: jnp.ndarray,
    transition: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    LTE level populations for the chosen transition.

    Parameters
    ----------
    gas_nd : jnp.ndarray
        Number density (same shape as gas_t). 
        example: (num_pix, num_pix, n_ray)
    gas_t : jnp.ndarray
        Temperature in Kelvin (same shape as gas_nd). 
        example: (npix, npix, nray)
    energy_levels : jnp.ndarray
        Shape (num_levels, 4). See `einstein_coefficients`.
    radiative_transitions : jnp.ndarray
        Shape (num_levels-1, 6). See `einstein_coefficients`.
    transition : int, optional
        1-based index of the line, by default 3.

    Returns
    -------
    n_up, n_dn : Tuple[jnp.ndarray, jnp.ndarray]
        Up- and down-level number densities (same shape as gas_t/gas_nd).
    """
    # Temperature grid for partition function (log-spaced)
    ntemp, temp0, temp1 = 1000, 0.1, 100000.0
    partition_temp = temp0 * (temp1 / temp0) ** (jnp.arange(ntemp) / (ntemp - 1))

    # Partition function from available levels
    dendivk = (hh * cc / kk) * jnp.diff(energy_levels[:, 1])
    dummy = jnp.exp(-dendivk[None] / partition_temp[:, None]) * (
        energy_levels[1:, 2] / energy_levels[:-1, 2]
    )
    partition_fn = energy_levels[0, 2] + jnp.sum(jnp.cumprod(dummy, axis=-1), axis=-1)
    pfunc = jnp.interp(gas_t, partition_temp, partition_fn)

    # Identify up/down indices for chosen transition
    line_index = transition - 1
    up_idx = jnp.array(radiative_transitions[line_index, 1] - 1, int)
    dn_idx = jnp.array(radiative_transitions[line_index, 2] - 1, int)

    dendivk_up = (hh * cc / kk) * (energy_levels[up_idx, 1] - energy_levels[0, 1])
    dendivk_dn = (hh * cc / kk) * (energy_levels[dn_idx, 1] - energy_levels[0, 1])
    levelweight_up = energy_levels[up_idx, 2]
    levelweight_dn = energy_levels[dn_idx, 2]

    n_up = (gas_nd / pfunc) * jnp.exp(-dendivk_up / gas_t) * levelweight_up
    n_dn = (gas_nd / pfunc) * jnp.exp(-dendivk_dn / gas_t) * levelweight_dn
    return n_up, n_dn

    
@struct.dataclass
class ChemistryParams:
    """
    Immutable container for line-selection & simple CO chemistry knobs.
    """
    molecule: str
    line_index: int
    line_name: str
    co_abundance: float
    freezeout: float
    N_dissoc: float
    N_desorp: Optional[float]  # <-- new optional field
    molecular_table: str

    def validate(self) -> "ChemistryParams":
        """Lightweight validation on creation."""
        if not self.molecule:
            raise ValueError("molecule must be a non-empty string")
        if not isinstance(self.line_index, int) or self.line_index < 0:
            raise ValueError("line_index must be a non-negative integer")
        if not self.line_name:
            raise ValueError("line_name must be a non-empty string")
        if self.co_abundance < 0:
            raise ValueError("co_abundance must be ≥ 0")
        if self.freezeout <= 0:
            raise ValueError("freezeout must be > 0 (K)")
        if self.N_dissoc < 0:
            raise ValueError("N_dissoc must be ≥ 0 (cm^-2)")
        if self.N_desorp is not None and self.N_desorp < 0:
            raise ValueError("N_desorp must be ≥ 0 (cm^-2) or None")
        if not self.molecular_table:
            raise ValueError("molecular_table must be a non-empty path")
        return self


# ----------------------------------------------------------------------------- #
# YAML helpers (operate only on the `chemistry:` block of a YAML)
# ----------------------------------------------------------------------------- #
def _chemistry_yaml_to_params_dict(cfg: dict) -> dict:
    if "chemistry" not in cfg:
        raise KeyError("YAML is missing required `chemistry` section.")
    c = dict(cfg["chemistry"])  # shallow copy

    # Coerce numerics
    for k in ["co_abundance", "freezeout", "N_dissoc", "N_desorp"]:
        if k in c and c[k] is not None:
            c[k] = float(c[k])

    # Integers
    if "line_index" in c and c["line_index"] is not None:
        c["line_index"] = int(c["line_index"])

    # Ensure required string fields are str
    for k in ["molecule", "line_name", "molecular_table"]:
        if k in c and c[k] is not None:
            c[k] = str(c[k])

    # Keep only declared fields
    allowed = {
        "molecule", "line_index", "line_name",
        "co_abundance", "freezeout",
        "N_dissoc", "N_desorp",
        "molecular_table",
    }
    return {k: c.get(k, None) for k in allowed}


def chemistry_from_yaml_path(path: str | Path) -> ChemistryParams:
    """
    Load ChemistryParams from YAML
    """
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    d = _chemistry_yaml_to_params_dict(cfg)
    return ChemistryParams(**d).validate()



def chemistry_to_yaml_path(params: ChemistryParams, path: str | Path) -> None:
    p = Path(path)
    doc: Dict[str, Any] = {}
    if p.exists():
        with open(p, "r") as f:
            doc = yaml.safe_load(f) or {}

    chem_out = {
        "molecule": params.molecule,
        "line_index": int(params.line_index),
        "line_name": params.line_name,
        "co_abundance": float(params.co_abundance),
        "freezeout": float(params.freezeout),
        "N_dissoc": float(params.N_dissoc),
        "N_desorp": None if params.N_desorp is None else float(params.N_desorp),
        "molecular_table": params.molecular_table,
    }

    doc["chemistry"] = chem_out
    with open(p, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def print_params(p: ChemistryParams) -> None:
    """Pretty, compact summary of ChemistryParams in aligned format."""
    print("Chemistry parameters (line selection & thresholds):")
    print(f"  Molecule:                   {p.molecule}")
    print(f"  Line index (J upper):       {p.line_index}")
    print(f"  Line name:                  {p.line_name}")
    print("")
    print(f"  CO abundance fraction:      {p.co_abundance:.3e}")
    print(f"  Freezeout threshold (K):    {p.freezeout:.2f}")
    print(f"  Dissoc. threshold (cm^-2):  {p.N_dissoc:.3e}")
    print(
        f"  Desorp. threshold (cm^-2):  "
        f"{'None' if p.N_desorp is None else f'{p.N_desorp:.3e}'}"
    )
    print("")
    print(f"  Molecular table:            {p.molecular_table}")

def co_abundance_profile(
    h2_N: jnp.ndarray,
    temperature: jnp.ndarray,
    *,
    freezeout: float,
    N_dissoc: float,
    N_desorp: float | None,
    co_abundance: float,
) -> jnp.ndarray:
    """
    CO abundance X_CO(N_H2, T) with warm/cold thresholding.

    Parameters
    ----------
    h2_N : jnp.ndarray
        H2 column (or proxy) [cm^-2].
    temperature : jnp.ndarray
        Temperature [K], broadcastable to h2_N.
    freezeout : float
        Freezeout threshold [K].
    N_dissoc : float
        Dissociation threshold column [cm^-2] for warm branch.
    N_desorp : float | None
        Optional upper limit for reintroduction band in cold branch [cm^-2].
        If None, only the warm branch applies.
    co_abundance : float
        CO abundance fraction to assign inside active regions.

    Returns
    -------
    Xco: jnp.ndarray
        Abundance field (dimensionless), same shape as inputs.
    """
    warm = jnp.bitwise_and(temperature > freezeout, 0.706 * h2_N > N_dissoc)

    if N_desorp is None or jnp.isinf(N_desorp):
        cold = jnp.zeros_like(warm, dtype=bool)
    else:
        cold = jnp.bitwise_and(
            temperature <= freezeout,
            jnp.bitwise_and(0.706 * h2_N > N_dissoc, 0.706 * h2_N < N_desorp),
        )

    active = jnp.bitwise_or(warm, cold)
    Xco = jnp.where(active, co_abundance, 0.0)
    return Xco


# ----------------------------------------------------------------------------- #
# Pre-jitted wrappers (same signatures; mark bools static where needed)
# ----------------------------------------------------------------------------- #
co_abundance_profile_jit    = jax.jit(co_abundance_profile, static_argnames=("freezeout", "co_abundance"))