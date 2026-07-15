"""
RadJAX: JAX-based line radiative transfer for protoplanetary disks.

GPU-accelerated, differentiable modeling of molecular line emission for
protoplanetary disks (e.g., ALMA). This package provides forward models,
I/O utilities, inference helpers, and visualization tools.

Core modules exposed at top level
---------------------------------
- line_rte         : Differentiable line radiative transfer solver
- grid             : Physical / computational grids
- sensor           : Ray geometry, sampling, and instrument model
- visibilities     : Visibility-domain utilities
- alma_io          : ALMA FITS/measurement set I/O helpers
- casa_io          : CASA measurement set visibility I/O
- vis_forward_model: NUFFT visibility-domain forward model + chi²
- consts           : Physical constants
- phys             : Physics utilities
- chemistry        : Abundance / chemistry helpers
- inference        : Inference utilities (e.g., MCMC)
- parallel         : JAX pmap/vmap helpers
- utils            : Misc utilities
- visualization    : Plotting helpers
"""

from .core import (
    line_rte,
    grid,
    sensor,
    visibilities,
    alma_io,
    casa_io,
    vis_forward_model,
    consts,
    phys,
    chemistry,
    inference,
    parallel,
    utils,
    visualization,
)

from .models import broken_power_law

__version__ = "0.1.0"

__all__ = [
    
    # Core
    "line_rte",
    "grid",
    "sensor",
    "visibilities",
    "alma_io",
    "casa_io",
    "vis_forward_model",
    "consts",
    "phys",
    "chemistry",
    "inference",
    "parallel",
    "utils",
    "visualization",

    # Models
    "broken_power_law",
]
