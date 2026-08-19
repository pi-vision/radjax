# radjax/core/__init__.py
"""
Core subpackage for RadJAX.
"""

from . import (
    line_rte,
    grid,
    sensor,
    visibilities,
    alma_io,
    consts,
    phys,
    chemistry,
    inference,
    network,
    parallel,
    utils,
    visualization,
)

try:
    from . import casa_io, vis_forward_model
except ImportError:
    pass  # requires radjax[vis]: pip install radjax[vis]

__all__ = [
    "line_rte",
    "grid",
    "sensor",
    "visibilities",
    "alma_io",
    "consts",
    "phys",
    "chemistry",
    "inference",
    "network",
    "parallel",
    "utils",
    "visualization",
]
