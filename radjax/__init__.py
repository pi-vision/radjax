"""
RadJAX: JAX-based line radiative transfer for protoplanetary disks.

This package provides fast, differentiable solvers for line radiative transfer,
parametric disk models, and inference tools for astrophysical applications.

High-level API:
    from radjax import XXX

For advanced usage, explore submodules:
    radjax.core
    radjax.models
    radjax.inference
    radjax.viz
"""
# --- Core / physics ---
from .core import line_rte
from .core import io

# --- Models ---

# --- Inference ---

# --- Visualization ---

# --- Package metadata ---
__version__ = "0.1.0"
__all__ = ["line_rte", "io", "grid"]