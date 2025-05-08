# benchmark.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from network import shard

# Import your project-specific modules
from gofish import imagecube
import sensor, parametric_disk_old as parametric_disk, grid, line_rte
from consts import *

print(f"Using GPU: {jax.devices()}")
# --------------------
def render_cube(ray_coords, pixel_area, disk, nd_co, temperature, velocity_az, bbox, freqs, nu0):
    ray_coords_sph = grid.cartesian_to_spherical(ray_coords)
    ray_coords_polar = grid.spherical_to_zr(ray_coords_sph)
    
    # Interpolate dataset along the ray coordinates
    gas_nd = grid.interpolate_scalar(nd_co, ray_coords_polar, bbox)
    gas_t  = grid.interpolate_scalar(temperature, ray_coords_polar, bbox, cval=1e-10)
    gas_v_az = grid.interpolate_scalar(velocity_az, ray_coords_polar, bbox)
    gas_v = parametric_disk.azimuthal_velocity(ray_coords, gas_v_az)
    
    # Einstein coefficient for spontaneous emission from level u to level d
    energy_levels, radiative_transitions = line_rte.load_molecular_tables(path=disk.molecule_table)
    _, a_ud, b_ud, b_du = line_rte.einstein_coefficients(energy_levels, radiative_transitions, transition=disk.transition)
    n_up, n_dn = line_rte.n_up_down(gas_nd, gas_t, energy_levels, radiative_transitions, transition=disk.transition)

    alpha_tot = line_rte.alpha_total_co(disk.v_turb, gas_t)

    # print('computing spectral cube for a total of {} frequencies'.format(freqs.size))
    images = line_rte.compute_spectral_cube_pmap(shard(freqs), gas_v, alpha_tot, n_up, n_dn, a_ud, b_ud, b_du, ray_coords, obs_dir, nu0, pixel_area)
    images = np.nan_to_num(images).reshape(freqs.size, *ray_coords.shape[:2])
    return images

# --------------------
# Environment setup
# --------------------
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # Prevent JAX memory preallocation

# --------------------
# Load data
# --------------------
data_path = '/scratch/ondemand28/len/data/radjax/MAPS/HD_163296_CO_220GHz.0.2arcsec.image.fits'
fov_as = 11.0 
velocity_range = (2000, 10000) # m/s
vlsr = 5760
isotop_abundance = 1

# Initialize image cube
data_cube = imagecube(data_path, FOV=fov_as, velocity_range=velocity_range)
freqs = data_cube.velocity_to_restframe_frequency(vlsr=vlsr)
velocities = data_cube.velax
data = data_cube.data
x_sky, y_sky = np.meshgrid(data_cube.xaxis, data_cube.yaxis, indexing='xy')
nu0 = data_cube.nu0
npix = data_cube.nxpix
width_kms = (velocities[-1] - velocities[0]) / 1000.0
delta_v_ms = velocities[1] - velocities[0]
beam = data_cube.beams_per_pix * sensor.beam(data_cube.dpix, data_cube.bmaj, data_cube.bmin, data_cube.bpa)

print(f'Using {len(freqs)} channels for cube of shape: {data.shape}')

# Disk parameters
disk = parametric_disk.DiskFlaherty(
    name='HD163296_CO(2-1)',
    T_mid1=18.8216,
    T_atm1=63.5251,
    q=-0.2474,
    q_in=-0.5563,
    r_break=78.5606,
    r_in=11.1852,
    log_r_c=2.2229,
    v_turb=0.6528,
    gamma=1.0,
    M_star=2.3 * M_sun,
    M_gas=0.09 * M_sun,
    co_abundance=1e-4 * isotop_abundance,
    N_dissoc=0.79 * 1.59e21,
    N_desorp=-np.inf,
    freezeout=19.0,
    r_scale=150.0,
    z_q0=70.0,
    delta=1,
    transition=2,
    m_mol=2.37 * m_h,
    molecule_table='./molecular_tables/molecule_12c16o.inp',
)

# Sensor setup
projection = sensor.PinholeProjection(
    name=disk.name,
    x_sky=x_sky,
    y_sky=y_sky,
    distance=122.0, 
    nray=100,
    incl=47.5,
    phi=0.0,
    posang=312.0,
    z_width=400.0,
    freqs=freqs
)

bbox_disk = jnp.array([(au*0, au*200), (au*disk.r_in, au*800)])
fov_rad = fov_as * arcsec
fov = 2 * projection.distance * pc * np.tan(fov_rad/2.0)
pixel_area = (fov/npix)**2
ray_coords, obs_dir = sensor.pinhole_disk_projection(
    projection.x_sky, projection.y_sky, projection.distance, projection.nray, 
    projection.incl, projection.phi, projection.posang, projection.z_width
)

# --------------------
# Benchmark helper
# --------------------
def benchmark_function(func, n_runs=10):
    times = []
    for _ in range(n_runs):
        start = time.time()
        result = func()
        if isinstance(result, (jax.Array, jnp.ndarray)):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if isinstance(r, (jax.Array, jnp.ndarray)):
                    r.block_until_ready()
        end = time.time()
        times.append(end - start)
    times = np.array(times)  # Ignore the first run
    print(f"Times (seconds): {times}")
    print(f"Mean (excl. initial run): {times[1:].mean():.6f} s, Std (excl. initial run): {times[1:].std():.6f} s")
    return times

# --------------------
# Setup disk fields benchmark
# --------------------
def setup_disk_fields():
    resolution = 500
    z_min, z_max = 0, 200
    r_min, r_max = disk.r_in, 800
    z_disk, r_disk = jnp.meshgrid(
        jnp.linspace(z_min, z_max, resolution),
        jnp.linspace(r_min, r_max, resolution),
        indexing='ij'
    )
    temperature = disk.temperature_profile(z_disk, r_disk)
    nd_h2 = parametric_disk.number_density_profile(
        z_disk, r_disk, temperature, disk.gamma, disk.r_in, 10**disk.log_r_c, disk.M_gas, disk.M_star, disk.m_mol
    )
    velocity_az = disk.velocity(z_disk, r_disk, nd_h2, temperature)
    N_h2 = parametric_disk.surface_density(z_disk, nd_h2)
    sigma_h2 = N_h2 * m_h
    abundance_co = disk.co_abundance_profile(N_h2, temperature)
    nd_co = abundance_co * nd_h2
    return temperature, nd_h2, velocity_az, N_h2, sigma_h2, abundance_co, nd_co

# --------------------
# Render cube benchmark
# --------------------
def render_disk_images():
    images = render_cube(
        ray_coords, pixel_area, disk, nd_co, temperature, -velocity_az, bbox_disk, freqs, nu0
    )
    return images

# --------------------
# Run Benchmarks
# --------------------
if __name__ == "__main__":
    print("Benchmarking disk setup (temperature and co fields)...")
    disk_setup_times = benchmark_function(setup_disk_fields, n_runs=10)

    # Setup disk fields once for rendering benchmark
    temperature, nd_h2, velocity_az, N_h2, sigma_h2, abundance_co, nd_co = setup_disk_fields()

    print("\nBenchmarking render_cube...")
    render_times = benchmark_function(render_disk_images, n_runs=10)
