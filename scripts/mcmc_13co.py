import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from gofish import imagecube
from ruamel.yaml import YAML

sys.path.insert(0, '../')
import visualization, sensor, grid, line_rte, parametric_disk, inference
from network import shard
from consts import *
import emcee
from functools import partial
import time

NUM_WALKERS = 60
NUM_WARMUP_STEPS = 80
NUM_STEPS = 1200

isotope_name = '13CO'
ABUNDANCE_FACTOR = 1/69 # 13CO
STRIDE = 3
noise_sigma = 0.00143


################################################################################
# 6) Define the rendering functions in a functional style
################################################################################
def render_cube_pinhole(ray_coords, pixel_area, disk_params, nd_co, temperature, velocity_az, bbox, freqs, nu0, obs_dir, molecular_table):
    ray_coords_sph = grid.cartesian_to_spherical(ray_coords)
    ray_coords_polar = grid.spherical_to_zr(ray_coords_sph)

    # Interpolate dataset along the ray coordinates
    gas_nd = grid.interpolate_scalar(nd_co, ray_coords_polar, bbox)
    gas_t  = grid.interpolate_scalar(temperature, ray_coords_polar, bbox, cval=1e-10)
    gas_v_az = grid.interpolate_scalar(velocity_az, ray_coords_polar, bbox)
    gas_v = parametric_disk.azimuthal_velocity(ray_coords, gas_v_az)

    # Load molecular data from disk_params
    energy_levels, radiative_transitions = line_rte.load_molecular_tables(path=molecular_table)
    _, a_ud, b_ud, b_du = line_rte.einstein_coefficients(
        energy_levels, radiative_transitions, transition=disk_params['transition']
    )
    n_up, n_dn = line_rte.n_up_down(
        gas_nd, gas_t, energy_levels, radiative_transitions, transition=disk_params['transition']
    )
    alpha_tot = line_rte.alpha_total_co(disk_params['v_turb'], gas_t)

    print(f'computing pinhole spectral cube for a total of {freqs.size} frequencies')
    images = line_rte.compute_spectral_cube_pmap(
        shard(freqs), gas_v, alpha_tot, n_up, n_dn,
        a_ud, b_ud, b_du, ray_coords, obs_dir, nu0, pixel_area
    )
    images = np.nan_to_num(images).reshape(freqs.size, *ray_coords.shape[:2])
    return images

################################################################################
# 1) Create the disk_params dictionary (instead of DiskFlaherty object).
################################################################################
disk_params = parametric_disk.create_disk_params(
    q=-0.27,
    q_in=-0.57,
    r_break=70,
    log_r_c=2.3,
    v_turb=0.06,
    T_atm1=87.0,
    gamma=1.0,
    T_mid1=17.8,
    M_star=2.3 * M_sun,
    r_in=11.0,
    M_gas=0.09 * M_sun,
    co_abundance=ABUNDANCE_FACTOR*1e-4,
    N_dissoc=0.79 * 1.59e21,
    N_desorp=-np.inf,
    freezeout=19.0,
    r_scale=150.0,
    z_q0=70.0,
    delta=1,
    transition=2,
    m_mol=2.37 * m_h,
)

################################################################################
# 2) Build the disk grid (z and r) for computing temperature, densities, etc.
################################################################################
resolution = 500
z_min, z_max = 0, 200
r_min, r_max = disk_params['r_in'], 800
z_disk, r_disk = jnp.meshgrid(
    jnp.linspace(z_min, z_max, resolution), 
    jnp.linspace(r_min, r_max, resolution), 
    indexing='ij'
)
molecular_table = "/nfs/rhea.dgp/u8/d/len/code/radjax_updated/molecular_tables/molecule_12c16o.inp"

################################################################################
# 3) Use the functional approach to compute temperature, number density, velocity
################################################################################
temperature = parametric_disk.temperature_profile(z_disk, r_disk, disk_params)

nd_h2 = parametric_disk.number_density_profile(
    z_disk, 
    r_disk, 
    temperature,
    disk_params['gamma'],
    disk_params['r_in'],
    10 ** disk_params['log_r_c'],
    disk_params['M_gas'],
    disk_params['M_star'],
    disk_params['m_mol']
)

velocity_az = parametric_disk.velocity_profile(
    z_disk, 
    r_disk, 
    nd_h2, 
    temperature, 
    disk_params
)


################################################################################
# 4) Compute column density, CO abundance, etc. (still purely functional)
################################################################################
N_h2 = parametric_disk.surface_density(z_disk, nd_h2)

abundance_co = parametric_disk.co_abundance_profile(N_h2, temperature, disk_params)
nd_co = abundance_co * nd_h2

################################################################################
# 5) Prepare bounding box for interpolation
################################################################################
bbox_disk = jnp.array([(au*z_min, au*z_max), (au*r_min, au*r_max)])

################################################################################
# 7) Example usage: generating pinhole and orthographic projections
################################################################################
# Pinhole Projection Setup
fov_as = 10.0  # arcsecs
# data_path = '~/code/radjax/data/alma/HD163296_CO_highres_cen.cm.fits'
data_path = f'/scratch/ondemand28/len/data/radjax/MAPS/HD_163296_{isotope_name}_220GHz.0.2arcsec.image.fits'
velocity_range = (2000, 10000) # m/s
vlsr = 5770 # systematic velocity of target

print("Loaded data cube:", data_path.split('/')[-1])

data_cube = imagecube(data_path, FOV=fov_as, velocity_range=velocity_range)

freqs = data_cube.velocity_to_restframe_frequency(vlsr=vlsr)
velocities = data_cube.velax
data = data_cube.data

print(f'Data shape: {data.shape}')

x_sky, y_sky = np.meshgrid(data_cube.xaxis, data_cube.yaxis, indexing='xy')
nu0 = data_cube.nu0  # e.g. 230.538 GHz
npix = data_cube.nxpix
width_kms = (velocities[-1] - velocities[0]) / 1000.0
delta_v_ms = velocities[1] - velocities[0]
# if num_freqs_resample > 1:
#     delta_v_ms = velocities[1] - velocities[0]
# else: 
#     delta_v_ms = np.inf
print(f'velocity resolution: {delta_v_ms} m/s')

# Shift frequencies so that center is ~nu0
# freqs += nu0 - freqs.mean()

projection_pinhole = sensor.PinholeProjection(
    name=f'HD163296_{isotope_name}',
    x_sky=x_sky,
    y_sky=y_sky,
    distance=122.0,
    nray=100,
    incl=47.5,
    phi=0.0,
    posang=312.0,
    z_width=2 * z_max,
    freqs=freqs
)

fov_rad = fov_as * arcsec
fov = 2 * projection_pinhole.distance * pc * np.tan(fov_rad / 2.0)
pixel_area_pinhole = (fov / npix) ** 2

ray_coords_pinhole, obs_dir_pinhole = sensor.pinhole_disk_projection(
    projection_pinhole.x_sky,
    projection_pinhole.y_sky,
    projection_pinhole.distance,
    projection_pinhole.nray,
    projection_pinhole.incl,
    projection_pinhole.phi,
    projection_pinhole.posang,
    projection_pinhole.z_width
)

beam = data_cube.beams_per_pix * sensor.beam(
    data_cube.dpix, data_cube.bmaj, data_cube.bmin, data_cube.bpa
)

images = render_cube_pinhole(
    ray_coords_pinhole,
    pixel_area_pinhole,
    disk_params,
    nd_co,
    temperature,
    -velocity_az,  # note sign convention
    bbox_disk,
    freqs=freqs,
    nu0=nu0,
    obs_dir=obs_dir_pinhole,
    molecular_table=molecular_table
)
images_blurred = sensor.fftconvolve_vmap(images, beam)


param_index_dict = inference.DISK_PARAM_INDEX
################################################################################
# 1) Set up the single-parameter prior for v_turb
################################################################################
posterior_params = {
    param_index_dict['q']: {
        'bounds': [-0.5, 0.0],
        'p0_range': [-0.3, -0.1],
    },
    param_index_dict['log_r_c']: {
        'bounds': [1.75, 2.50],
        'p0_range': [2.00, 2.30],
    },
    param_index_dict['v_turb']: {
        'bounds': [0.0, 1.0],
        'p0_range': [0.05, 0.15],
    },
    param_index_dict['T_atm1']: {
        'bounds': [40.0, 150.0],
        'p0_range': [60.0, 110.0],
    },
    param_index_dict['T_mid1']: {
        'bounds': [5.0, 40.0],
        'p0_range': [10.0, 30.0],
    },
    param_index_dict['r_in']: {
        'bounds': [1.0, 20.0],
        'p0_range': [5.0, 15.0],
    },
    param_index_dict['r_break']: {
        'bounds': [15.0, 115.0],
        'p0_range': [50.0, 80.0],
    },
    param_index_dict['q_in']: {
        'bounds': [-1.0, 0.0],
        'p0_range': [-0.7, -0.3],
    }
}

# Our disk_params dictionary (already created), e.g.:
disk_params = parametric_disk.create_disk_params(
    T_mid1=17.8,
    T_atm1=87.0,
    q=-0.27,
    q_in=-0.57,
    r_break=70,
    M_star=2.3 * M_sun,
    gamma=1.0,
    r_in=11.0,
    log_r_c=2.3,
    M_gas=0.09 * M_sun,
    v_turb=0.3,  # Just an initial guess; the MCMC will sample this
    co_abundance=ABUNDANCE_FACTOR * 1e-4,
    N_dissoc=0.79 * 1.59e21,
    N_desorp=-np.inf,
    freezeout=19.0,
    r_scale=150.0,
    z_q0=70.0,
    delta=1,
    transition=2,
    m_mol=2.37 * m_h
)

ray_coords_sph = grid.cartesian_to_spherical(ray_coords_pinhole)
ray_coords_polar = grid.spherical_to_zr(ray_coords_sph)
energy_levels, radiative_transitions = line_rte.load_molecular_tables(
    path="/nfs/rhea.dgp/u8/d/len/code/radjax_updated/molecular_tables/molecule_12c16o.inp"
)

################################################################################
# 2) Create the sampler_dict for the new functional inference
################################################################################
sampler_dict = inference.create_sampler(
    z_disk=z_disk,
    r_disk=r_disk,
    bbox=bbox_disk,
    pixel_area=pixel_area_pinhole,
    ray_coords=ray_coords_pinhole,
    ray_coords_polar=ray_coords_polar,
    obs_dir=obs_dir_pinhole,
    beam=beam,
    energy_levels=energy_levels,
    radiative_transitions=radiative_transitions,
    sigma=noise_sigma,
    disk_params=disk_params,
    posterior_params=posterior_params,
    stride=3
)

ndim = len(list(posterior_params.keys()))    
nwalkers = NUM_WALKERS
nsteps = NUM_STEPS # Example number of MCMC steps
num_iter_restart = NUM_WARMUP_STEPS
restart_cluster_size = 1e-2

run_name = f"MAPS_{isotope_name}_{nsteps}_iter_stride{sampler_dict['stride']}_MCMC_sampler"
print(f"Run name: {run_name}")

save_dir = f"../mcmc_output/runs/{run_name}"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = f"{save_dir}/{run_name}.h5"
print(f"Saving outputs to {save_path}")

# Save mcmc hyperparameters to log file
log_file = f"{save_dir}/log.txt"
with open(log_file, 'w') as f:
    f.write(f"Run name: {run_name}\n")
    f.write(f"Number of walkers: {nwalkers}\n")
    f.write(f"Number of steps: {nsteps}\n")
    f.write(f"Stride: {sampler_dict['stride']}\n")
    f.write(f"Restart cluster size: {restart_cluster_size}\n")
    f.write(f"Posterior parameters:\n")
    for param, info in posterior_params.items():
        f.write(f"{param}: {info}\n")

# Sample initial positions p0 from the param_bounds
p0 = inference.sample_p0(nwalkers, posterior_params, seed=1337)

backend = emcee.backends.HDFBackend(save_path)
backend.reset(nwalkers, ndim)


def logprob_wrapper(params):
    """
    A small wrapper that calls the functional logprob with the 
    arguments we need: the data (images_ground_truth_blurred) 
    and frequencies, plus the sampler_dict.
    """
    return inference.logprob(params, data, freqs, sampler_dict, sampler_dict['stride'])

sampler = emcee.EnsembleSampler(
    nwalkers=nwalkers,
    ndim=ndim,
    log_prob_fn=logprob_wrapper,
    backend=backend
)


print("Running MCMC Warmup...")
warmup_sampler = sampler.run_mcmc(p0, num_iter_restart, progress=True)
flat_samples = sampler.get_chain(flat=True)
warmup_medians = np.median(flat_samples, axis=0)
p1 = np.random.normal(warmup_medians,  restart_cluster_size * np.abs(warmup_medians), size=(nwalkers, ndim))

print("Finished Warm up. Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

print("Running MCMC:", np.mean(sampler.acceptance_fraction))
start_time_mcmc = time.time()
sampler.run_mcmc(p1, nsteps, progress=True)
end_time_mcmc = time.time()
mcmc_time = end_time_mcmc - start_time_mcmc
print(f"MCMC completed in {mcmc_time:.2f} seconds.")
print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))


#### Load the MCMC results

backend = emcee.backends.HDFBackend(save_path)
samples = backend.get_chain(flat=True)

flat_samples = backend.get_chain(discard=backend.iteration//2, thin=1, flat=True)

ndim = flat_samples.shape[-1]

param_index_dict = inference.DISK_PARAM_INDEX
posterior_params = {
    param_index_dict['q']: {
        'bounds': [-0.5, 0.0],
        'p0_range': [-0.3, -0.1],
    },
    param_index_dict['log_r_c']: {
        'bounds': [1.75, 2.50],
        'p0_range': [2.00, 2.30],
    },
    param_index_dict['v_turb']: {
        'bounds': [0.0, 1.0],
        'p0_range': [0.05, 0.15],
    },
    param_index_dict['T_atm1']: {
        'bounds': [40.0, 150.0],
        'p0_range': [60.0, 110.0],
    },
    param_index_dict['T_mid1']: {
        'bounds': [5.0, 40.0],
        'p0_range': [10.0, 30.0],
    },
    param_index_dict['r_in']: {
        'bounds': [1.0, 20.0],
        'p0_range': [5.0, 15.0],
    },
    param_index_dict['r_break']: {
        'bounds': [15.0, 115.0],
        'p0_range': [50.0, 80.0],
    },
    param_index_dict['q_in']: {
        'bounds': [-1.0, 0.0],
        'p0_range': [-0.7, -0.3],
    }
}

param_labels = [
    inference.REV_DISK_PARAM_INDEX[idx] for idx in sorted(posterior_params.keys())
]

chosen_values = np.median(flat_samples, axis=0)
print("Chosen parameter values (from MCMC) with their Flaherty's values:")

disk_params_mcmc = parametric_disk.create_disk_params(
    q=-0.27,
    q_in=-0.57,
    r_break=70,
    log_r_c=2.3,
    v_turb=0.06,
    T_atm1=87.0,
    gamma=1.0,
    T_mid1=17.8,
    M_star=2.3 * M_sun,
    r_in=11.0,
    M_gas=0.09 * M_sun,
    co_abundance=ABUNDANCE_FACTOR*1e-4,
    N_dissoc=0.79 * 1.59e21,
    N_desorp=-np.inf,
    freezeout=19.0,
    r_scale=150.0,
    z_q0=70.0,
    delta=1,
    transition=2,
    m_mol=2.37 * m_h,
)

for label, value in zip(param_labels, chosen_values):
    disk_params_mcmc[label] = value
    print(f"Diskparam MCMC for {label} updated to: {disk_params_mcmc[label]}")
    
ground_truth = {
    "q": -0.27,
    "log_r_c": 2.3,
    "v_turb": 0.06,
    "T_atm1": 87.0,
    "T_mid1": 17.8,
    "r_in": 11.0,
    "r_break": 70.0,
    "q_in": -0.57
}

chain = backend.get_chain()

fig, axes = plt.subplots(1, ndim, figsize=(14,3))
for i in range(ndim):
    axes[i].plot(chain[...,i])
    axes[i].set_title(param_labels[i])

plt.tight_layout()
fig.savefig(f"{save_dir}/walkers_plot.png", dpi=300)

import corner

print("Flat samples shape:", flat_samples.shape)  # should be ((nsteps-100)*nwalkers, ndim)


truth_values = [ground_truth[label] for label in param_labels]

# corner plot
fig = corner.corner(
    flat_samples,
    labels=param_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    truths=truth_values,        # <--- overlay true values
    truth_color="red",          # optional color
    title_kwargs={"fontsize": 12}
)
fig.savefig(f"{save_dir}/corner_plot.png", dpi=300)

# Recompute temperature and densities with updated v_turb
temperature_mcmc = parametric_disk.temperature_profile(z_disk, r_disk, disk_params_mcmc)
nd_h2_mcmc = parametric_disk.number_density_profile(
    z_disk,
    r_disk,
    temperature_mcmc,
    disk_params_mcmc["gamma"],
    disk_params_mcmc["r_in"],
    10 ** disk_params_mcmc["log_r_c"],
    disk_params_mcmc["M_gas"],
    disk_params_mcmc["M_star"],
    disk_params_mcmc["m_mol"]
)

r_min_mcmc, r_max_mcmc = disk_params_mcmc['r_in'], 800
z_disk_mcmc, r_disk_mcmc = jnp.meshgrid(
    jnp.linspace(z_min, z_max, resolution), 
    jnp.linspace(r_min_mcmc, r_max_mcmc, resolution), 
    indexing='ij'
)

velocity_az_mcmc = -parametric_disk.velocity_profile(
    z_disk, r_disk_mcmc, nd_h2_mcmc, temperature_mcmc, disk_params_mcmc
)

N_h2_mcmc = parametric_disk.surface_density(z_disk, nd_h2_mcmc)
abundance_co_mcmc = parametric_disk.co_abundance_profile(N_h2_mcmc, temperature_mcmc, disk_params_mcmc)
nd_co_mcmc = abundance_co_mcmc * nd_h2_mcmc

bbox_disk_mcmc = jnp.array([(au*z_min, au*z_max), (au*r_min_mcmc, au*r_max_mcmc)])

projection_pinhole_mcmc = sensor.PinholeProjection(
    name=f'HD163296_{isotope_name}_MCMC',
    x_sky=x_sky,
    y_sky=y_sky,
    distance=122.0,
    nray=100,
    incl=47.5,
    phi=0.0,
    posang=312.0,
    z_width=2 * z_max,
    freqs=freqs
)

fov_rad = fov_as * arcsec
fov = 2 * projection_pinhole_mcmc.distance * pc * np.tan(fov_rad / 2.0)
pixel_area_pinhole_mcmc = (fov / npix) ** 2

ray_coords_pinhole_mcmc, obs_dir_pinhole_mcmc = sensor.pinhole_disk_projection(
    projection_pinhole_mcmc.x_sky,
    projection_pinhole_mcmc.y_sky,
    projection_pinhole_mcmc.distance,
    projection_pinhole_mcmc.nray,
    projection_pinhole_mcmc.incl,
    projection_pinhole_mcmc.phi,
    projection_pinhole_mcmc.posang,
    projection_pinhole_mcmc.z_width
)

beam = data_cube.beams_per_pix * sensor.beam(
    data_cube.dpix, data_cube.bmaj, data_cube.bmin, data_cube.bpa
)

# # Render new cube with the updated v_turb
images_mcmc = render_cube_pinhole(
    ray_coords=ray_coords_pinhole_mcmc,
    pixel_area=pixel_area_pinhole_mcmc,
    disk_params=disk_params_mcmc,       # pass the dictionary instead of a disk object
    nd_co=nd_co_mcmc,
    temperature=temperature_mcmc,
    velocity_az=velocity_az_mcmc,
    bbox=bbox_disk_mcmc,
    freqs=freqs,
    nu0=nu0,
    obs_dir=obs_dir_pinhole_mcmc,
    molecular_table=molecular_table
)

images_mcmc_blurred = sensor.fftconvolve_vmap(images_mcmc, beam)

residual_mcmc = images_blurred - images_mcmc_blurred
chi2_mcmc = np.sum(((images_blurred - images_mcmc_blurred) / noise_sigma)**2)
residual_sum_mcmc = np.sum(np.abs(residual_mcmc))
residual_mean_mcmc = np.mean(np.abs(residual_mcmc))

n_data = np.prod(data.shape)
chi_squared_over_n_data = chi2_mcmc / n_data

print(f"MCMC Chi-squared: {chi2_mcmc:.2f}, Chi-squared/N_data = {chi_squared_over_n_data}")
print(f"MCMC Residual Sum: {residual_sum_mcmc:.5f}")
print(f"MCMC Residual Mean: {residual_mean_mcmc:.5f}")