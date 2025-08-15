import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
import functools
import sensor, grid, line_rte, parametric_disk
from consts import *
from network import shard
import copy

@jax.tree_util.register_pytree_node_class
class DiskSampler(object):
    def __init__(self, z_disk, r_disk, bbox, pixel_area, ray_coords, ray_coords_polar, obs_dir, beam, 
                 energy_levels, radiative_transitions, sigma, disk, stride, params):
        self.z_disk = z_disk
        self.r_disk = r_disk
        self.bbox = bbox
        self.pixel_area = pixel_area
        self.ray_coords = ray_coords
        self.ray_coords_polar = ray_coords_polar
        self.obs_dir = obs_dir
        self.beam = beam
        self.energy_levels = energy_levels
        self.radiative_transitions = radiative_transitions
        self.sigma = sigma
        self.disk = disk
        self.stride = stride
        self.params = params
        self.param_names = list(params.keys())
        self.bounds = np.array(list(params.values())).T
        self.nu0, self.a_ud, self.b_ud, self.b_du = line_rte.einstein_coefficients(energy_levels, radiative_transitions, transition=disk.transition)

    def sample_p0(self, params, nwalkers, seed=None):
        np.random.seed(seed)
        print('setting inital seed to: {}'.format(np.random.get_state()[1][0]))
        ndim = len(params)
        return np.random.uniform(*self.bounds, (nwalkers, ndim))

    def logprior(self, x):
        condition = jnp.logical_and(x>=self.bounds[0],x<=self.bounds[1])
        return jnp.sum(jnp.where(condition, 0, -jnp.inf))

    def loglikelihood(self, x, y, freqs):
        disk = self.disk
        disk.update(self.param_names, x)
        
        temperature = disk.temperature_profile(self.z_disk, self.r_disk)
        nd_h2 = parametric_disk.number_density_profile(
            self.z_disk, self.r_disk, temperature, disk.gamma, disk.r_in, 10**disk.log_r_c, disk.M_gas, disk.M_star, disk.m_mol
        )
        velocity_az = -disk.velocity(self.z_disk, self.r_disk, nd_h2, temperature)
        
        # Compute the column integrated density N(h2) (units of 1/cm^2)
        N_h2 = parametric_disk.surface_density(self.z_disk, nd_h2)
        
        # Compute the relative abundance of co
        abundance_co = disk.co_abundance_profile(N_h2, temperature)
        nd_co = abundance_co * nd_h2
        
        # Interpolate dataset along the ray coordinates
        gas_nd = grid.interpolate_scalar(nd_co, self.ray_coords_polar, self.bbox)
        gas_t  = grid.interpolate_scalar(temperature, self.ray_coords_polar, self.bbox, cval=1e-10)
        gas_v_az = grid.interpolate_scalar(velocity_az, self.ray_coords_polar, self.bbox)
        gas_v = parametric_disk.azimuthal_velocity(self.ray_coords, gas_v_az)
        
        # Einstein coefficient for spontaneous emission from level u to level d
        n_up, n_dn = line_rte.n_up_down(gas_nd, gas_t, self.energy_levels, self.radiative_transitions, transition=disk.transition)
        alpha_tot = line_rte.alpha_total_co(disk.v_turb, gas_t)

        model = jnp.nan_to_num(line_rte.compute_spectral_cube(
            freqs, gas_v, alpha_tot, n_up, n_dn, self.a_ud, self.b_ud, self.b_du, self.ray_coords, self.obs_dir, self.nu0, self.pixel_area
        ))
        model = sensor.fftconvolve_vmap(model, self.beam)
        logpdf = jsp.stats.norm.logpdf(y[:,::self.stride,::self.stride], model[:,::self.stride,::self.stride], self.sigma)
        return jnp.sum(logpdf)
        
    @jax.jit
    def logprob(self, x, y, freqs):
        return self.logprior(x) + self.loglikelihood(x, y, freqs)
    
    def tree_flatten(self):
        children = (self.z_disk, self.r_disk, self.bbox, self.pixel_area, self.ray_coords, 
                    self.ray_coords_polar, self.obs_dir, self.beam, self.energy_levels, 
                    self.radiative_transitions, self.sigma, self.disk)
        aux_data = {'stride': self.stride, 'params': self.params}  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data) 

@jax.tree_util.register_pytree_node_class
class DiskSamplerTurbulence(object):
    def __init__(self, z_disk, r_disk, bbox, pixel_area, ray_coords, ray_coords_polar, obs_dir, beam, 
                 energy_levels, radiative_transitions, sigma, disk, params):
        self.z_disk = z_disk
        self.r_disk = r_disk
        self.bbox = bbox
        self.pixel_area = pixel_area
        self.ray_coords = ray_coords
        self.ray_coords_polar = ray_coords_polar
        self.obs_dir = obs_dir
        self.beam = beam
        self.energy_levels = energy_levels
        self.radiative_transitions = radiative_transitions
        self.sigma = sigma
        self.disk = disk
        self.params = params
        self.param_names = list(params.keys())
        self.bounds = np.array([p['bounds'] for p in params.values()]).T
        self.param_ranges = np.array([p['p0_range'] for p in params.values()]).T
        self.nu0, self.a_ud, self.b_ud, self.b_du = line_rte.einstein_coefficients(energy_levels, radiative_transitions, transition=disk.transition)

    def sample_p0(self, params, nwalkers, seed=None):
        np.random.seed(seed)
        print('setting inital seed to: {}'.format(np.random.get_state()[1][0]))
        param_ranges = np.array([p['p0_range'] for p in params.values()]).T
        ndim = len(params)
        return np.random.uniform(*param_ranges, (nwalkers, ndim))

    def logprior(self, x):
        condition = jnp.logical_and(x>=self.bounds[0],x<=self.bounds[1])
        return jnp.sum(jnp.where(condition, 0, -jnp.inf))

    def loglikelihood(self, x, y, freqs):
        disk = self.disk
        disk.__dict__.update(dict(zip(self.param_names, x)))
        
        temperature = disk.temperature_profile(self.z_disk, self.r_disk)
        nd_h2 = parametric_disk.number_density_profile(self.z_disk, self.r_disk, temperature, disk.gamma, disk.r_in, 10**disk.log_r_c, disk.M_gas, disk.M_star, disk.m_mol)
        velocity_az = -disk.velocity(self.z_disk, self.r_disk, nd_h2, temperature)
        
        # Compute the column integrated density N(h2) (units of 1/cm^2)
        N_h2 = parametric_disk.surface_density(self.z_disk, nd_h2)
        
        # Compute the relative abundance of co
        abundance_co = disk.co_abundance_profile(N_h2, temperature)
        nd_co = abundance_co * nd_h2
        
        # Interpolate dataset along the ray coordinates
        gas_nd = grid.interpolate_scalar(nd_co, self.ray_coords_polar, self.bbox)
        gas_t  = grid.interpolate_scalar(temperature, self.ray_coords_polar, self.bbox, cval=1e-10)
        gas_v_az = grid.interpolate_scalar(velocity_az, self.ray_coords_polar, self.bbox)
        gas_v = parametric_disk.azimuthal_velocity(self.ray_coords, gas_v_az)
        
        # Einstein coefficient for spontaneous emission from level u to level d
        n_up, n_dn = line_rte.n_up_down(gas_nd, gas_t, self.energy_levels, self.radiative_transitions, transition=disk.transition)
        alpha_tot = line_rte.alpha_total_co(disk.v_turb, gas_t)

        model = jnp.nan_to_num(line_rte.compute_spectral_cube(
            freqs, gas_v, alpha_tot, n_up, n_dn, self.a_ud, self.b_ud, self.b_du, self.ray_coords, self.obs_dir, self.nu0, self.pixel_area
        ))
        model = sensor.fftconvolve_vmap(model, self.beam)
        logpdf = jsp.stats.norm.logpdf(y, model, self.sigma)
        return jnp.sum(logpdf)
        
    @jax.jit
    def logprob(self, x, y, freqs):
        return self.logprior(x) + self.loglikelihood(x, y, freqs)
    
    def tree_flatten(self):
        children = (self.z_disk, self.r_disk, self.bbox, self.pixel_area, self.ray_coords, 
                    self.ray_coords_polar, self.obs_dir, self.beam, self.energy_levels, 
                    self.radiative_transitions, self.sigma, self.disk)
        aux_data = {'params': self.params}  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)