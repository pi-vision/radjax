"""
Validation plots for radjax/core/casa_io.py and vis_forward_model.py.

Reads the MAPS HD163296 CO 2-1 measurement set, renders the Flaherty
parametric model, runs the NUFFT forward model, and saves diagnostic plots:
  - vis_diagnostics.png       amplitude vs UV distance + chi²
  - dirty_comparison.png      data vs model dirty images per channel
  - selfconsistency.png       model image vs dirty(NUFFT(model))

Saves figures to /scratch/ondemand28/len/data/radjax/casa_io_validation/

Run with:
  python scripts/validate_casa_io.py [--gpu]
"""
import os, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()

if not args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from radjax.core.casa_io import read_ms, knn_sigma, VisibilityData
from radjax.core.vis_forward_model import image_to_vis, plot_vis_diagnostics, plot_dirty_comparison, plot_selfconsistency
from radjax.core import sensor, chemistry as chem
from radjax.models.broken_power_law import disk_from_yaml, forward_model_with_rays

C    = 2.99792458e8
NU0  = 230538e6
OUT  = "/scratch/ondemand28/len/data/radjax/casa_io_validation"
MS   = "/scratch/ondemand28/len/data/radjax/MAPS_uv/HD_163296_CO_220GHz_custom_0.1kms.bin_30s.ms.cvel"
PARAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data/HD163296/Flaherty/HD163296_CO_highres_cen.cm.gaia.params.yaml")

os.makedirs(OUT, exist_ok=True)

print("Reading MS...")
vis_obs = read_ms(MS)

_lam_ms  = C / vis_obs.freq
_uvd_lam = np.sqrt((vis_obs.uvw[:,0:1]/_lam_ms)**2 + (vis_obs.uvw[:,1:2]/_lam_ms)**2)
_B_min   = _uvd_lam[_uvd_lam[:, np.argmin(vis_obs.freq)] > 0, np.argmin(vis_obs.freq)].min()
_B_max   = _uvd_lam[:, np.argmax(vis_obs.freq)].max()
_fov_rad = 1.0 / _B_min
_res_rad = 1.0 / _B_max
_npix    = int(2 ** np.ceil(np.log2(np.ceil(_fov_rad / _res_rad * 2))))
_fov_as  = np.degrees(_fov_rad) * 3600
_dpix_as = _fov_as / _npix
dpix_rad = np.radians(_dpix_as / 3600)
print(f"  MS-derived grid: npix={_npix}, dpix={_dpix_as*1e3:.1f} mas, FOV={_fov_as:.2f}\"")

_npix_render = 256
_fov_render_as = _npix_render * _dpix_as
_half_render = _fov_render_as / 2
xax = np.linspace( _half_render, -_half_render, _npix_render)
yax = np.linspace(-_half_render,  _half_render, _npix_render)
x_sky, y_sky = np.meshgrid(xax, yax, indexing="xy")
print(f"  Render grid: {_npix_render}px × {_dpix_as*1e3:.1f} mas = {_fov_render_as:.2f}\"")

disk_params = disk_from_yaml(PARAMS)
chem_params = chem.chemistry_from_yaml_path(PARAMS)
mol         = chem.load_molecular_tables(chem_params)
obs_params  = sensor.params_from_yaml(PARAMS)
vlsr        = float(obs_params.vlsr)

idx = np.arange(0, vis_obs.nchan, 10)
vis_sub = VisibilityData(
    uvw    = vis_obs.uvw,
    data   = vis_obs.data[:, idx],
    weight = vis_obs.weight[:, idx],
    freq   = vis_obs.freq[idx],
    flag   = vis_obs.flag[:, idx],
)
print(f"  vis_sub: {vis_sub.nrows} rows x {vis_sub.nchan} channels")

print("Computing knn sigma (k=50)...")
t0 = time.time()
vis_sub.sigma_knn = knn_sigma(vis_sub, k=50)
print(f"  done in {time.time()-t0:.1f}s  sigma median={np.median(vis_sub.sigma_knn):.4f} Jy")

print("Rendering model cube...")
f_rf = vis_sub.freq + NU0 * vlsr / C
rays = sensor.rays_from_params(obs_params, jnp.array(x_sky), jnp.array(y_sky))
t0 = time.time()
cube = forward_model_with_rays(disk_params=disk_params, chem_params=chem_params,
                               mol=mol, rays=rays, freqs=jnp.array(f_rf))
cube.block_until_ready()
print(f"  render {time.time()-t0:.1f}s  cube peak={np.max(np.asarray(cube)):.4f} Jy/pix")

print("Running NUFFT...")
cube_np = np.asarray(cube)
t0 = time.time()
vis_model = np.asarray(image_to_vis(cube_np, vis_sub.uvw, vis_sub.freq, dpix_rad))
print(f"  NUFFT {time.time()-t0:.1f}s")

plot_vis_diagnostics(vis_sub, vis_model, f"{OUT}/vis_diagnostics.png")
print(f"  → vis_diagnostics.png")

plot_dirty_comparison(vis_sub, vis_model, dpix_rad, f"{OUT}/dirty_comparison.png", channels=[14,15,16,17,18])
print(f"  → dirty_comparison.png")

plot_selfconsistency(cube_np, vis_sub, vis_model, dpix_rad, f"{OUT}/selfconsistency.png", channels=[14,15,16])
print(f"  → selfconsistency.png")

print("Done.")
