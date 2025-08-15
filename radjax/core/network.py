import jax
import flax
import functools
import optax
from jax import numpy as jnp
from flax import linen as nn
from typing import Any, Callable
from flax.training import train_state, checkpoints

class PixelPredictor(nn.Module):
    scale: float   # Scaling of the input dimension axis (e.g. to make input range [-1,1] or [0,1] along the different dimensions)
    posenc_deg: int = 3
    posenc_var: float = 0.0
    net_depth: int = 4
    net_width: int = 64
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    out_activation: Callable[..., Any] = nn.sigmoid
    do_skip: bool = True
    min_val: float = 0.0
    max_val: float = 1.0
    offset: float = 0.0
    
    def init_params(self, coords, seed=1):
        params = self.init(jax.random.PRNGKey(seed), coords)['params']
        return params

    def init_state(self, params, num_iters=5000, lr_init=1e-4, lr_final=1e-6, checkpoint_dir=''):
        lr = optax.polynomial_schedule(lr_init, lr_final, 1, num_iters)
        tx = optax.adam(learning_rate=lr)
        state = train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx) 
        return state
    
    @nn.compact
    def __call__(self, coords):
        pixel_mlp = MLP(self.net_depth, self.net_width, self.activation, self.out_channel, self.do_skip)
        def predict_pixels(coords):
            
            if self.posenc_var > 0:
                input = integrated_posenc(coords / self.scale, self.posenc_deg, self.posenc_var)
            else:
                input = posenc(coords / self.scale, self.posenc_deg)
            
            net_output = pixel_mlp(input)
            if self.out_channel == 1:
                net_output = net_output[..., 0]

        
            pixels = self.out_activation(net_output-self.offset)*(self.max_val - self.min_val) + self.min_val
            return pixels
        pixels = predict_pixels(coords)
        return pixels
        
class MLP(nn.Module):
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
        net_depth: int, 
            the depth of the first part of MLP.
        net_width: int, 
            the width of the first part of MLP.
        activation: function, 
            the activation function used in the MLP.
        out_channel: 
            int, the number of alpha_channels.
        do_skip: boolean, 
            whether or not to use a skip connection

        Returns
        -------
        out: jnp.ndarray(float32), 
            [batch_size * n_samples, out_channel].
        """
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.he_uniform())

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

def posenc(x, deg):
    """
    Concatenate `x` with a positional encoding of `x` with degree `deg`.
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Parameters
    ----------
    x: jnp.ndarray, 
        variables to be encoded. Note that x should be in [-pi, pi].
    deg: int, 
        the degree of the encoding.

    Returns
    -------
    encoded: jnp.ndarray, 
        encoded variables.
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
    
safe_sin = lambda x: jnp.sin(x % (100 * jnp.pi))

