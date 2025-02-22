import jax
import jax.numpy as jnp
import jax.scipy as jsp

import numpy as np

# local imports
import sensor
import grid
import line_rte
import parametric_disk 
from consts import *
from network import shard
from functools import partial

###############################################################################
# Example: Setting up a "Sampler" dictionary and a set of functions that
# operate on that data to compute priors, likelihoods, etc.

# Dictionary for index-param name:
DISK_PARAM_INDEX = {
    'T_mid1': 0,
    'T_atm1': 1,
    'q': 2,
    'q_in': 3,
    'r_break': 4,
    'M_star': 5,
    'gamma': 6,
    'r_in': 7,
    'log_r_c': 8,
    'M_gas': 9,
    'v_turb': 10,
    'co_abundance': 11,
    'N_dissoc': 12,
    'N_desorp': 13,
    'z_q0': 14,
    'transition': 15,
    'm_mol': 16,
    'freezeout': 17,
    'delta': 18,
    'r_scale': 19
}

# Reverse key-value pair for DISK_PARAM_INDEX
REV_DISK_PARAM_INDEX = {index: name for name, index in DISK_PARAM_INDEX.items()}
###############################################################################

def create_sampler(
    z_disk, r_disk, bbox, pixel_area, ray_coords, ray_coords_polar, obs_dir,
    beam, energy_levels, radiative_transitions, sigma, disk_params, posterior_params, stride=None    
):
    """
    Return a dict containing all the static data needed for the MCMC sampling.
    """
    param_names = list(posterior_params.keys())

    # Pre-compute line RTE constants
    nu0, a_ud, b_ud, b_du = line_rte.einstein_coefficients(
        energy_levels, radiative_transitions, transition=disk_params['transition']
    )

    sampler_dict = {
        'z_disk': z_disk,
        'r_disk': r_disk,
        'bbox': bbox,
        'pixel_area': pixel_area,
        'ray_coords': ray_coords,
        'ray_coords_polar': ray_coords_polar,
        'obs_dir': obs_dir,
        'beam': beam,
        'energy_levels': energy_levels,
        'radiative_transitions': radiative_transitions,
        'sigma': sigma,
        'disk_params': disk_params,
        'posterior_params': posterior_params,
        'stride': stride,
        'nu0': nu0,
        'a_ud': a_ud,
        'b_ud': b_ud,
        'b_du': b_du
    }
    return sampler_dict

def sample_p0(nwalkers, posterior_params, seed=None):
    """
    Sample initial positions for MCMC walkers uniformly from `p0_bounds`.
    """
    np.random.seed(seed)
    ndim = len(posterior_params)
    p0 = np.empty((nwalkers, ndim), dtype=float)

    # The integer keys might be 0..N-1 or any order, so let's collect them in sorted order:
    sorted_indices = sorted(posterior_params.keys())
    
    for col, i in enumerate(sorted_indices):
        lo, hi = posterior_params[i]["p0_range"]
        p0[:, col] = np.random.uniform(lo, hi, size=nwalkers)

    return p0

def logprior(params, sampler_dict):
    """
    Check that 'params' is within the bounds. If so, return 0; else -inf.
    """
    posterior_params = sampler_dict['posterior_params']
    sorted_indices = sorted(posterior_params.keys())

    # uniform prior
    lp = 0.0
    for col, i in enumerate(sorted_indices):
        lo, hi = posterior_params[i]["bounds"]
        in_bounds = jnp.logical_and(params[col] >= lo, params[col] <= hi)
        lp += jnp.where(in_bounds, 0.0, -jnp.inf)
    return lp


def loglikelihood(params, y, freqs, stride, sampler_dict):
    """
    Evaluate the log-likelihood by building the disk model and comparing
    to the data.
    """
    disk_params = sampler_dict['disk_params'].copy()
    posterior_params = sampler_dict['posterior_params']
    sorted_indices = sorted(posterior_params.keys())
    
    # For each col in the 'params' array, figure out which disk param index it corresponds to:
    for col, param_index in enumerate(sorted_indices):
        name = REV_DISK_PARAM_INDEX[param_index]
#         print(f'updated {disk_params[name]} to {params[col]}')
        disk_params[name] = params[col]
        

    # 1) Build temperature field
    temperature = parametric_disk.temperature_profile(
        sampler_dict['z_disk'], sampler_dict['r_disk'], disk_params
    )

    # 2) Build H2 number density
    nd_h2 = parametric_disk.number_density_profile(
        sampler_dict['z_disk'],
        sampler_dict['r_disk'],
        temperature,
        disk_params['gamma'],
        disk_params['r_in'],
        10**disk_params['log_r_c'],
        disk_params['M_gas'],
        disk_params['M_star'],
        disk_params['m_mol']
    )

    # 3) Build velocities
    velocity_az = -parametric_disk.velocity_profile(
        sampler_dict['z_disk'], sampler_dict['r_disk'],
        nd_h2, temperature, disk_params
    )

    # 4) Compute column integrated density N_h2
    N_h2 = parametric_disk.surface_density(sampler_dict['z_disk'], nd_h2)

    # 5) Build the CO abundance
    abundance_co = parametric_disk.co_abundance_profile(N_h2, temperature, disk_params)
    nd_co = abundance_co * nd_h2

    # 6) Interpolate onto ray-coordinates
    gas_nd = grid.interpolate_scalar(
        nd_co, sampler_dict['ray_coords_polar'], sampler_dict['bbox']
    )
    gas_t = grid.interpolate_scalar(
        temperature, sampler_dict['ray_coords_polar'], sampler_dict['bbox'], cval=1e-10
    )
    gas_v_az = grid.interpolate_scalar(
        velocity_az, sampler_dict['ray_coords_polar'], sampler_dict['bbox']
    )
    gas_v = parametric_disk.azimuthal_velocity(sampler_dict['ray_coords'], gas_v_az)

    # 7) Einstein coefficients
    n_up, n_dn = line_rte.n_up_down(
        gas_nd,
        gas_t,
        sampler_dict['energy_levels'],
        sampler_dict['radiative_transitions'],
        transition=disk_params['transition']
    )
    alpha_tot = line_rte.alpha_total_co(disk_params['v_turb'], gas_t)

    # 8) Compute the model data cube
    model_cube = line_rte.compute_spectral_cube_pmap(
        shard(freqs), gas_v, alpha_tot, n_up, n_dn,
        sampler_dict['a_ud'], sampler_dict['b_ud'], sampler_dict['b_du'],
        sampler_dict['ray_coords'], sampler_dict['obs_dir'],
        sampler_dict['nu0'], sampler_dict['pixel_area']
    )

    # 9) Convolve model with the beam
    model_cube = jnp.nan_to_num(model_cube).reshape(freqs.size, *sampler_dict['ray_coords'].shape[:2])
    model_cube = sensor.fftconvolve_vmap(model_cube, sampler_dict['beam'])

    # 10) Evaluate log-likelihood
    if stride is not None:
        model_cube = model_cube[:, ::stride, ::stride]
        y = y[:, ::stride, ::stride]

    sigma = sampler_dict['sigma']
    logpdf = jsp.stats.norm.logpdf(y, model_cube, sigma)
    return jnp.sum(logpdf)

@partial(jax.jit, static_argnames=['stride']) # TODO: seperate static components from sampler_dict to seperate arg
def logprob(params, y, freqs, sampler_dict, stride=None):
    """
    jitted version for logprior + loglikelihood.
    """
    return logprior(params, sampler_dict) + loglikelihood(params, y, freqs, stride, sampler_dict)


