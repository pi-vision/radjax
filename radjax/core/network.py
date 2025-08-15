"""
Neural networks for RadJAX (Flax/JAX).

Provides:
- PixelPredictor: small MLP wrapper for per-pixel predictions with positional encodings
- MLP: generic feedforward block with optional skip
- Positional encodings: standard positional encoding
- shard: helper to reshape batches across multiple local devices
"""

from __future__ import annotations
from typing import Any, Callable, Tuple, Union

import functools
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

PRNGLike = Union[int, jax.Array]

class PixelPredictor(nn.Module):
    """
    Pixel-wise predictor with positional encoding.

    Parameters
    ----------
    scale : float
        Feature scaling for coordinates before encoding (e.g., bring into [-1, 1]).
    posenc_deg : int, optional
        Number of Fourier frequency bands for positional encoding (0 disables PE),
        by default 3.
    net_depth : int, optional
        Number of hidden layers in the MLP, by default 4.
    net_width : int, optional
        Hidden layer width, by default 64.
    activation : Callable[..., Any], optional
        Activation function for hidden layers, by default ``flax.linen.relu``.
    out_channel : int, optional
        Number of output channels (if 1, the last dimension is squeezed), by default 1.
    out_activation : Callable[..., Any], optional
        Final activation applied to the raw output (e.g., ``sigmoid``), by default
        ``flax.linen.sigmoid``.
    do_skip : bool, optional
        If True, concatenates a skip connection halfway through the MLP, by default True.
    min_val : float, optional
        Minimum of the output range after activation, by default 0.0.
    max_val : float, optional
        Maximum of the output range after activation, by default 1.0.
    offset : float, optional
        Value subtracted from the raw network output before ``out_activation``,
        by default 0.0.
    """
    scale: float
    posenc_deg: int = 3
    net_depth: int = 4
    net_width: int = 64
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    out_activation: Callable[..., Any] = nn.sigmoid
    do_skip: bool = True
    min_val: float = 0.0
    max_val: float = 1.0
    offset: float = 0.0
    
    def init_params(self, coords: jnp.ndarray, seed: PRNGLike = 1) -> dict:
        """Initialize parameters from coordinates and a seed/key.
        
        Parameters
        ----------
        coords : jnp.ndarray
            Input coordinates, shape [..., D].
        seed : int or jax.Array, optional
            PRNG seed or key, by default 1.

        Returns
        -------
        dict
            Flax parameters dictionary.
        """
        key = seed if isinstance(seed, jax.Array) else jax.random.PRNGKey(int(seed))
        params = self.init(key, coords)["params"]
        return params

    def init_state(
        self,
        params: dict,
        num_iters: int = 5_000,
        lr_init: float = 1e-4,
        lr_final: float = 1e-6,
    ) -> train_state.TrainState:
        """
        Create a TrainState with Adam and a polynomial LR decay over `num_iters`.

        Parameters
        ----------
        params : dict
            Model parameters.
        num_iters : int, optional
            Total steps for the LR schedule, by default 5000.
        lr_init : float, optional
            Initial learning rate, by default 1e-4.
        lr_final : float, optional
            Final learning rate, by default 1e-6.

        Returns
        -------
        flax.training.train_state.TrainState
            Training state with optimizer and schedule.
        """
        lr_schedule = optax.polynomial_schedule(
            init_value=lr_init, end_value=lr_final, power=1, transition_steps=num_iters
        )
        tx = optax.adam(learning_rate=lr_schedule)
        return train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx)
    
    @nn.compact
    def __call__(self, coords):
        """
        Forward pass producing pixel outputs.

        Parameters
        ----------
        coords : jnp.ndarray
            Input coordinates/features, shape [..., D].

        Returns
        -------
        jnp.ndarray
            Network outputs with shape [..., out_channel], or [...,] if
            `out_channel == 1`.
        """
        mlp = MLP(
            net_depth=self.net_depth,
            net_width=self.net_width,
            activation=self.activation,
            out_channel=self.out_channel,
            do_skip=self.do_skip,
        )

        if self.posenc_deg == 0:
            pe = coords / self.scale
        else:
            pe = posenc(coords / self.scale, self.posenc_deg)

        net_output = mlp(pe)
        if self.out_channel == 1:
            net_output = net_output[..., 0] 

        # Range mapping with offset then activation
        pixels = self.out_activation(net_output - self.offset) * (self.max_val - self.min_val) + self.min_val
        return pixels
        
class MLP(nn.Module):
    """
    Simple MLP with optional mid-way skip connection.

    Args:
        net_depth: Number of hidden layers.
        net_width: Hidden width per layer.
        activation: Nonlinearity.
        out_channel: Output channels.
        do_skip: If True, concatenates the input midway through the stack.
    """

    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
  
    @nn.compact
    def __call__(self, x):
        """A simple Multi-Layer Preceptron (MLP) network

        Parameters
        ----------
        x: jnp.ndarray(float32), 
            [batch_size * n_samples, feature], points.

        Returns
        -------
        out: jnp.ndarray(float32), 
            [batch_size * n_samples, out_channel].
        """
        dense_layer = functools.partial(nn.Dense, kernel_init=jax.nn.initializers.he_uniform())

        if self.do_skip:
            skip_layer = self.net_depth // 2

        inputs = x
        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.activation(x)
            if self.do_skip:
                if i % skip_layer == 0 and i > 0:
                    x = jnp.concatenate([x, inputs], axis=-1)
        out = dense_layer(self.out_channel)(x)

        return out

def posenc(x: jnp.ndarray, deg: int) -> jnp.ndarray:
    """
    Concatenate `x` with a positional encoding of `x` with degree `deg`.
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Parameters
    ----------
    x : jnp.ndarray
        Input in radians (ideally scaled to [-pi, pi]), shape [..., D].
    deg : int
        Number of frequency bands (0 disables encoding).

    Returns
    -------
    jnp.ndarray
        Encoded features, shape [..., D + 2*D*deg].
    """
    if deg == 0:
        return x
    scales = jnp.array([2**i for i in range(deg)])
    xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1])
    four_feat = safe_sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))
    return jnp.concatenate([x] + [four_feat], axis=-1)
    
def shard(xs):
    """Split data into shards for multiple devices along the first dimension."""
    return jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)


def safe_sin(x: jnp.ndarray) -> jnp.ndarray:
    """
    Numerically safer sine with large-input wrapping.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.

    Returns
    -------
    jnp.ndarray
        sin(x mod 100π), same shape as input.
    """
    return jnp.sin(x % (100 * jnp.pi))