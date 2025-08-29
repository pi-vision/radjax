# radjax/core/parallel.py
from __future__ import annotations
from typing import Tuple, Any
import jax
import jax.numpy as jnp

def shard(xs: Any) -> Any:
    """
    Split data into shards for multiple devices along the first dimension.
    Works on pytrees. First dimension must be divisible by local_device_count().
    """
    ndev = jax.local_device_count()
    def _reshape(x):
        return x.reshape((ndev, -1) + x.shape[1:])
    return jax.tree_map(_reshape, xs)

def shard_with_padding(xs, pad_value=0):
    """
    Pads and reshapes xs for sharding across devices.
    Returns (sharded_xs, original_length).
    """
    n_devices = jax.local_device_count()

    def pad_and_reshape(x):
        orig_len = x.shape[0]
        pad_len = (-orig_len) % n_devices
        if pad_len > 0:
            pad_width = [(0, pad_len)] + [(0, 0)] * (x.ndim - 1)
            x = jnp.pad(x, pad_width, constant_values=pad_value)
        x = x.reshape((n_devices, -1) + x.shape[1:])
        return x

    # Assume all arrays have the same first dimension
    original_length = jax.tree_util.tree_leaves(xs)[0].shape[0]

    sharded = jax.tree_map(pad_and_reshape, xs)
    return sharded
