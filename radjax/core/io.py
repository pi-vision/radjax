"""
I/O utilities for RadJAX.

This module provides routines to load molecular data tables
(e.g. CO energy levels and radiative transitions) from external input files. 
These tables supply the fundamental spectroscopic information needed for line radiative transfer.
"""
from __future__ import annotations
from typing import Tuple
from pathlib import Path

import jax.numpy as jnp
import numpy as np


def load_molecular_tables(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load molecular information. 
    
    path: str,
        path to molecule_12c16o.inp
   
    See e.g. tables below from molecule_12c16o.inp
    More information: https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/lineradtrans.html
    CO Molecular tables can be found here: https://home.strw.leidenuniv.nl/~moldata/CO.html
    ------------------------------------
    LEVEL   ENERGIES(cm^-1)   WEIGHT   J
      3     11.534919938	    5.0	     2
      4     23.069512649	    7.0	     3
    ---------------------------------------------------------
    TRANS   UP   LOW  EINSTEINA(s^-1)  FREQ(GHz)       E_u(K)
      3     4     3   2.497e-06        345.7959899     33.19
     """
    energy_levels = np.loadtxt(path, skiprows=7, max_rows=41)
    radiative_transitions = np.loadtxt(path, skiprows=51, max_rows=40)
    return energy_levels, radiative_transitions

def read_spherical_amr(path: str | Path) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Load a spherical AMR grid from a RADMC-3D-style `amr_grid.inp`.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to `amr_grid.inp`.

    Returns
    -------
    r, theta, phi : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        1D arrays with bin edges (lengths are nr+1, nth+1, nph+1).
    """
    p = Path(path)
    nr, nth, nph = np.loadtxt(p, skiprows=5, max_rows=1, dtype=int)
    r = np.loadtxt(p, skiprows=6, max_rows=nr + 1)
    theta = np.loadtxt(p, skiprows=6 + nr + 1, max_rows=nth + 1)
    phi = np.loadtxt(p, skiprows=6 + nr + nth + 2, max_rows=nph + 1)
    return jnp.asarray(r), jnp.asarray(theta), jnp.asarray(phi)

__all__ = [
    "load_molecular_tables", 
    "read_spherical_amr",
]