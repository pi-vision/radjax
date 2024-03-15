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
    net_width: int = 128
    activation: Callable[..., Any] = nn.sigmoid
    out_channel: int = 1
    out_activation: Callable[..., Any] = nn.sigmoid
    do_skip: bool = True
    sigmoid_offset: float = 10.0
    
    def init_params(self, coords, seed=1):
        params = self.init(jax.random.PRNGKey(seed), coords)['params']
        return params.unfreeze() # TODO(pratul): this unfreeze feels sketchy

    def init_state(self, params, num_iters=5000, lr_init=1e-4, lr_final=1e-6, checkpoint_dir=''):
        lr = optax.polynomial_schedule(lr_init, lr_final, 1, num_iters)
        tx = optax.adam(learning_rate=lr)
        state = train_state.TrainState.create(apply_fn=self.apply, params=params, tx=tx) 
        # Replicate state for multiple gpus
        state = flax.jax_utils.replicate(state)
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
        
            offset = self.sigmoid_offset if self.out_activation.__name__ == 'sigmoid' else 0
            pixels = self.out_activation(net_output - offset)
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


def integrated_posenc(x, max_deg, x_cov, min_deg=0):
    """
    Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1].

    Parameters
    ----------
    x: jnp.ndarray, variables to be encoded. Should
      be in [-pi, pi]. 
    max_deg: int, the max degree of the encoding.
    x_cov: jnp.ndarray, covariance matricfes for `x`.
    min_deg: int, the min degree of the encoding. default=0.

    Returns
    -------
    encoded: jnp.ndarray, encoded variables.
    """
    if jnp.isscalar(x_cov):
        x_cov = jnp.full_like(x, x_cov)
    scales = 2**jnp.arange(min_deg, max_deg)
    shape = list(x.shape[:-1]) + [-1]
    y = jnp.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = jnp.reshape(x_cov[..., None, :] * scales[:, None]**2, shape)

    return expected_sin(
      jnp.concatenate([y, y + 0.5 * jnp.pi], axis=-1),
      jnp.concatenate([y_var] * 2, axis=-1))

def expected_sin(x, x_var):
    # When the variance is wide, shrink sin towards zero.
    y = jnp.exp(-0.5 * x_var) * jnp.sin(x)
    return y

    
def shard(xs):
    """Split data into shards for multiple devices along the first dimension."""
    return jax.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)

"""
def shard(xs, overlap=0):
    n_gpus = jax.local_device_count()
    if xs.shape[0] % n_gpus != overlap:
        raise AttributeError
    if overlap == 0:
        output = jax.tree_map(lambda x: x.reshape((n_gpus, -1) + x.shape[1:]), xs)    
    elif overlap > 0:
        output = jax.tree_map(lambda x: x.reshape((n_gpus, x.shape[0] // n_gpus) + x.shape[1:]), xs[:-overlap])
        output = jnp.concatenate((output, jnp.concatenate((output[1:,0], xs[-1][None]))[:,None]), axis=1)

    return output
"""   

safe_sin = lambda x: jnp.sin(x % (100 * jnp.pi))

