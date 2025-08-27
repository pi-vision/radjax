"""
Chemistry parameter container.

Conventions
-----------
- m_mol: g (mass of tracer molecule, e.g., mean molecular mass for H2/CO)
- freezeout: K
- N_dissoc, N_desorp: cm^-2
- co_abundance: dimensionless fraction
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import jax
from flax import struct
import yaml


@struct.dataclass
class ChemistryParams:
    """
    Immutable parameter container for simple CO chemistry thresholding.

    Attributes
    ----------
    m_mol : float
        Molecular mass [g] (e.g., mean molecular weight).
    co_abundance : float
        CO abundance fraction (dimensionless).
    freezeout : float
        Freezeout threshold temperature [K].
    N_dissoc : float
        Dissociation threshold column density [cm^-2].
    N_desorp : float
        Desorption/reintroduction threshold column density [cm^-2].
    """

    m_mol: float
    co_abundance: float
    freezeout: float
    N_dissoc: float
    N_desorp: float

    def validate(self) -> "ChemistryParams":
        """Lightweight validation on creation."""
        if self.m_mol <= 0: raise ValueError("m_mol must be > 0")
        if self.co_abundance < 0: raise ValueError("co_abundance must be ≥ 0")
        if self.freezeout <= 0: raise ValueError("freezeout must be > 0")
        if self.N_dissoc < 0: raise ValueError("N_dissoc must be ≥ 0")
        if self.N_desorp < 0: raise ValueError("N_desorp must be ≥ 0 (or +inf)")
        return self


# ----------------------------------------------------------------------------- #
# YAML helpers (operate only on the `chemistry:` block of a unified YAML)
# ----------------------------------------------------------------------------- #
def _chemistry_yaml_to_params_dict(cfg: dict) -> dict:
    """
    Extract the `chemistry` block from a unified YAML.
    Expects cfg["chemistry"] to exist.
    """
    if "chemistry" not in cfg:
        raise KeyError("YAML is missing required `chemistry` section.")
    d = dict(cfg["chemistry"])  # shallow copy

    # coerce to float where appropriate
    for k in ["m_mol", "co_abundance", "freezeout", "N_dissoc", "N_desorp"]:
        if k in d and d[k] is not None:
            d[k] = float(d[k])

    return d


def chemistry_from_yaml_path(path: str | Path) -> ChemistryParams:
    """
    Load ChemistryParams from the **chemistry** section of a unified YAML.
    """
    with open(Path(path), "r") as f:
        cfg = yaml.safe_load(f) or {}
    d = _chemistry_yaml_to_params_dict(cfg)
    return ChemistryParams(**d).validate()


def chemistry_to_yaml_path(params: ChemistryParams, path: str | Path) -> None:
    """
    Write back **only** the `chemistry` block, preserving everything else.
    """
    p = Path(path)
    doc: Dict[str, Any] = {}
    if p.exists():
        with open(p, "r") as f:
            doc = yaml.safe_load(f) or {}

    chem_out = {
        "m_mol": params.m_mol,
        "co_abundance": params.co_abundance,
        "freezeout": params.freezeout,
        "N_dissoc": params.N_dissoc,
        "N_desorp": params.N_desorp,
    }

    # drop None
    chem_out = {k: v for k, v in chem_out.items() if v is not None}

    doc["chemistry"] = chem_out
    with open(p, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)


def print_chemistry(p: ChemistryParams) -> None:
    """Pretty-print a compact, human-readable summary of ChemistryParams."""
    lines = [
        "ChemistryParams(",
        f"  m_mol={p.m_mol:.3e} g, co_abundance={p.co_abundance}",
        f"  freezeout={p.freezeout} K, N_dissoc={p.N_dissoc} cm^-2, N_desorp={p.N_desorp} cm^-2",
        ")",
    ]
    print("\n".join(lines))
