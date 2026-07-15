"""
Validation plots for radjax/core/casa_io.py (NUFFT visibility forward model).

Saves figures to /scratch/ondemand28/len/data/radjax/casa_io_validation/

Run with:
  python scripts/validate_casa_io.py [--gpu]
"""

import argparse, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
args = parser.parse_args()

if not args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from radjax.core.casa_io import read_ms, VisibilityData
from radjax.core.vis_forward_model import image_to_vis, chi2_vis
from radjax.core import sensor
from radjax.core import chemistry as chem
from radjax.models.broken_power_law import disk_from_yaml, forward_model_with_rays

OUT = "/scratch/ondemand28/len/data/radjax/casa_io_validation"
os.makedirs(OUT, exist_ok=True)

PARAMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data/HD163296/Flaherty/HD163296_CO_highres_cen.cm.gaia.params.yaml")
MS     = "/scratch/ondemand28/len/data/radjax/MAPS_uv/HD_163296_CO_220GHz_custom_0.1kms.bin_30s.ms.cvel"
C      = 2.99792458e8
NU0    = 230538e6   # CO 2-1 rest Hz

rng = np.random.default_rng(42)

# image_to_vis convention: FITS layout, East at col=0 (left), increasing col = West.
# Standard radio FT: V(u,v) = F * exp(-2πi*(u*l + v*m))
# Point at col=N/2-dx (East): k2 = (N/2-dx) - N/2 = -dx  → V = exp(-2πi*u*dx*dpix)
# Point at row=N/2+dy (North): k1 = +dy                   → V = exp(-2πi*v*dy*dpix)

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: point source at phase center → V = 1+0j everywhere
# ─────────────────────────────────────────────────────────────────────────────
print("Test 1: point source at phase center")
N = 128
dpix_rad = 0.04 * np.pi / 180 / 3600
uvw = rng.uniform(-1e5, 1e5, (500, 3))
freq = np.array([230538e6])
lam = C / 230538e6
uvdist = np.sqrt(uvw[:,0]**2 + uvw[:,1]**2) / lam

img = np.zeros((1, N, N)); img[0, N//2, N//2] = 1.0
v1 = image_to_vis(img, uvw, freq, dpix_rad)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(uvdist/1e3, v1[:,0].real, s=6, alpha=0.6, label="real (expected: 1.0)")
axes[0].scatter(uvdist/1e3, v1[:,0].imag, s=6, alpha=0.6, label="imag (expected: 0.0)", color="C1")
axes[0].axhline(1.0, color="k", ls="--", lw=1)
axes[0].axhline(0.0, color="gray", ls="--", lw=1)
axes[0].set_xlabel("Baseline [kλ]"); axes[0].set_ylabel("Visibility [Jy]")
axes[0].set_title("Point source at phase center → flat V = 1+0j")
axes[0].legend(fontsize=8); axes[0].set_ylim(-0.1, 1.2)
err1 = np.abs(v1[:,0] - 1.0).max()

# Test 2: RA offset source — phase ramp in u
# FITS convention: East at col=0 (left). Offset +dx East → col = N/2 - dx (left of centre).
# Expected: V = exp(-2πi * u * dx * dpix_rad)
dx = 7
img2 = np.zeros((1, N, N)); img2[0, N//2, N//2 - dx] = 1.0   # East of centre (FITS col)
v2 = image_to_vis(img2, uvw, freq, dpix_rad)
uu = uvw[:,0] / lam
pred2 = np.exp(-1j * 2*np.pi * uu * dx * dpix_rad)

axes[1].scatter(uu/1e3, v2[:,0].real, s=6, alpha=0.6, label="computed real")
axes[1].scatter(uu/1e3, pred2.real,   s=6, alpha=0.3, marker="x", label="predicted real", color="C0")
axes[1].scatter(uu/1e3, v2[:,0].imag, s=6, alpha=0.6, label="computed imag", color="C1")
axes[1].scatter(uu/1e3, pred2.imag,   s=6, alpha=0.3, marker="x", label="predicted imag", color="C1")
axes[1].set_xlabel("u [kλ]"); axes[1].set_ylabel("Visibility [Jy]")
axes[1].set_title(f"Point source offset +{dx} pix in RA → phase ramp")
axes[1].legend(fontsize=7)
err2 = np.abs(v2[:,0] - pred2).max()

fig.suptitle(f"NUFFT sanity checks  |T1 err|={err1:.2e}  |T2 err|={err2:.2e}", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/sanity_nufft.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  T1 max err={err1:.2e}  T2 max err={err2:.2e}  → saved sanity_nufft.png")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: row offset (Dec) — phase ramp in v
# ─────────────────────────────────────────────────────────────────────────────
print("Test 3: row offset in Dec direction")
dy = 5
img3 = np.zeros((1, N, N)); img3[0, N//2 + dy, N//2] = 1.0
v3 = image_to_vis(img3, uvw, freq, dpix_rad)
vv = uvw[:,1] / lam
pred3 = np.exp(-1j * 2*np.pi * vv * dy * dpix_rad)
err3 = np.abs(v3[:,0] - pred3).max()

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(vv/1e3, v3[:,0].real, s=6, alpha=0.6, label="computed real")
ax.scatter(vv/1e3, pred3.real,   s=6, alpha=0.3, marker="x", label="predicted real", color="C0")
ax.scatter(vv/1e3, v3[:,0].imag, s=6, alpha=0.6, label="computed imag", color="C1")
ax.scatter(vv/1e3, pred3.imag,   s=6, alpha=0.3, marker="x", label="predicted imag", color="C1")
ax.set_xlabel("v [kλ]"); ax.set_ylabel("Visibility [Jy]")
ax.set_title(f"Point source offset +{dy} pix in Dec → phase ramp in v\n|err|_max={err3:.2e}")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/sanity_nufft_dec.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  T3 max err={err3:.2e}  → saved sanity_nufft_dec.png")

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Extended source — compare NUFFT to numpy FFT + bilinear interpolation
# (should agree at short baselines; NUFFT is exact, FFT+interp has aliasing)
# ─────────────────────────────────────────────────────────────────────────────
print("Test 4: compare NUFFT vs FFT + bilinear for a Gaussian source")
N = 128
sigma_pix = 10.0
yi, xi = np.mgrid[:N, :N]
img_g = np.zeros((1, N, N))
img_g[0] = np.exp(-((xi - N/2)**2 + (yi - N/2)**2) / (2 * sigma_pix**2))
img_g /= img_g.sum()   # normalise: zero-spacing V = 1

# NUFFT
uvw_test = rng.uniform(-5e4, 5e4, (300, 3))
v_nufft = image_to_vis(img_g, uvw_test, freq, dpix_rad)[:,0]

# FFT + bilinear interpolation (baseline method)
fft_img = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img_g[0])))
uu_t = uvw_test[:,0] / lam;  vv_t = uvw_test[:,1] / lam
# convert to pixel coords in FFT image (DC at N//2, N//2)
u_pix = uu_t * dpix_rad * N + N//2
v_pix = vv_t * dpix_rad * N + N//2

def bilinear(arr, r, c):
    r0, c0 = np.floor(r).astype(int), np.floor(c).astype(int)
    dr, dc = r - r0, c - c0
    r0 = np.clip(r0, 0, N-2); c0 = np.clip(c0, 0, N-2)
    return (arr[r0, c0]*(1-dr)*(1-dc) + arr[r0+1, c0]*dr*(1-dc) +
            arr[r0, c0+1]*(1-dr)*dc   + arr[r0+1, c0+1]*dr*dc)

v_fft = bilinear(fft_img, v_pix, u_pix)   # note: row=v, col=u in FFT image

uvdist_t = np.sqrt(uvw_test[:,0]**2 + uvw_test[:,1]**2) / lam
sort_idx = np.argsort(uvdist_t)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(uvdist_t[sort_idx]/1e3, np.abs(v_nufft)[sort_idx], "o", ms=4, alpha=0.7, label="NUFFT |V|")
axes[0].plot(uvdist_t[sort_idx]/1e3, np.abs(v_fft)[sort_idx],   "x", ms=4, alpha=0.7, label="FFT+bilinear |V|", color="C1")
axes[0].set_xlabel("Baseline [kλ]"); axes[0].set_ylabel("|V| [Jy]")
axes[0].set_title("NUFFT vs FFT+bilinear: Gaussian source amplitude")
axes[0].legend()

diff = np.abs(v_nufft - v_fft)
axes[1].plot(uvdist_t[sort_idx]/1e3, diff[sort_idx], "o", ms=4, alpha=0.7, color="C2")
axes[1].set_xlabel("Baseline [kλ]"); axes[1].set_ylabel("|NUFFT - FFT+bilinear|")
axes[1].set_title("Difference (NUFFT is exact; FFT+bilinear has aliasing at long baselines)")
plt.tight_layout()
plt.savefig(f"{OUT}/nufft_vs_fft_bilinear.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → saved nufft_vs_fft_bilinear.png")

# ─────────────────────────────────────────────────────────────────────────────
# Test 5: end-to-end with real MS + Flaherty parametric model
# ─────────────────────────────────────────────────────────────────────────────
print("Test 5: end-to-end: Flaherty model → NUFFT → chi² vs MAPS MS")

MAPS_FITS = "/scratch/ondemand28/len/data/radjax/HD_163296_CO_220GHz.0.15arcsec.image.fits"
from gofish import imagecube as _imagecube
_cube_fits_full = _imagecube(MAPS_FITS, FOV=10.5)
# Crop to 256×256 pixels centred on phase centre (rendering 526×526 OOMs on GPU)
_dpix_fits = float(_cube_fits_full.dpix)   # 20 mas
_half = 128
_cen  = _cube_fits_full.nxpix // 2
_sl   = slice(_cen - _half, _cen + _half)
_xax  = _cube_fits_full.xaxis[_sl]
_yax  = _cube_fits_full.yaxis[_sl]
x_sky_fits, y_sky_fits = np.meshgrid(_xax, _yax, indexing="xy")
dpix_fits     = _dpix_fits
dpix_rad_fits = dpix_fits * np.pi / 180 / 3600
fits_ext      = [_xax[0], _xax[-1], _yax[0], _yax[-1]]  # East left

disk_params = disk_from_yaml(PARAMS)
chem_params = chem.chemistry_from_yaml_path(PARAMS)
mol         = chem.load_molecular_tables(chem_params)
obs_params  = sensor.params_from_yaml(PARAMS)
vlsr        = float(obs_params.vlsr)

vis_obs = read_ms(MS)
idx = np.arange(0, vis_obs.nchan, 10)
vis_sub = VisibilityData(uvw=vis_obs.uvw, data=vis_obs.data[:,idx],
                         weight=vis_obs.weight[:,idx], freq=vis_obs.freq[idx], flag=vis_obs.flag[:,idx])

# LSRK → rest-frame frequencies for the renderer
f_rf = vis_sub.freq + NU0 * vlsr / C

# Use gofish sky grid: East at col=0, increasing col = West (FITS convention)
rays = sensor.rays_from_params(obs_params, jnp.array(x_sky_fits), jnp.array(y_sky_fits))

t0 = time.time()
cube = forward_model_with_rays(disk_params=disk_params, chem_params=chem_params,
                               mol=mol, rays=rays, freqs=jnp.array(f_rf))
cube.block_until_ready()
t_render = time.time() - t0

t0 = time.time()
vis_model = image_to_vis(np.asarray(cube), vis_sub.uvw, vis_sub.freq, dpix_rad_fits)
t_nufft = time.time() - t0

chi2 = chi2_vis(vis_sub, vis_model)
ndata = int(np.sum(vis_sub.weight > 0))
red_chi2 = chi2 / ndata

print(f"  render {t_render:.1f}s | NUFFT {t_nufft:.1f}s | red χ²={red_chi2:.3f}")

uu_all = vis_sub.uvw[:,0][:,None] / (C / vis_sub.freq[None,:])
vv_all = vis_sub.uvw[:,1][:,None] / (C / vis_sub.freq[None,:])
uvdist_all = np.sqrt(uu_all**2 + vv_all**2).ravel()

bins = np.logspace(4, 7, 60)
bin_idx = np.digitize(uvdist_all, bins)
obs_amp   = np.abs(vis_sub.data).ravel()
model_amp = np.abs(vis_model).ravel()
obs_binned   = [obs_amp[bin_idx == i].mean()   if (bin_idx == i).any() else np.nan for i in range(1, len(bins))]
model_binned = [model_amp[bin_idx == i].mean() if (bin_idx == i).any() else np.nan for i in range(1, len(bins))]
bc = 0.5*(bins[:-1]+bins[1:])

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

ax = axes[0]
ax.loglog(bc, obs_binned,   "o-", ms=4, label="Data (MAPS CO 2-1)")
ax.loglog(bc, model_binned, "s-", ms=4, label="Model (Flaherty params)", color="C1")
ax.set_xlabel("Baseline [λ]"); ax.set_ylabel("Mean |V| [Jy]")
ax.set_title("|V| vs baseline length"); ax.legend(fontsize=9)

obs_chan_amp   = np.abs(vis_sub.data).mean(0)
model_chan_amp = np.abs(vis_model).mean(0)
v_lsrk_chan = C * (1 - vis_sub.freq / NU0) / 1e3
ax = axes[1]
ax.plot(v_lsrk_chan, obs_chan_amp,   label="Data mean |V|")
ax.plot(v_lsrk_chan, model_chan_amp, label="Model mean |V|", color="C1")
ax.set_xlabel("LSRK velocity [km/s]"); ax.set_ylabel("Mean |V| [Jy]")
ax.set_title("Mean visibility amplitude per channel"); ax.legend(fontsize=9)

# Dirty image: col=-u convention, display with gofish extent (East on left)
ax = axes[2]
chan_mid  = vis_sub.nchan // 2
nimg      = 2 * _half
dpix_d    = dpix_fits * np.pi / 180 / 3600
uu_c = (vis_sub.uvw[:,0] / (C / vis_sub.freq[chan_mid])).astype(np.float64)
vv_c = (vis_sub.uvw[:,1] / (C / vis_sub.freq[chan_mid])).astype(np.float64)
dirty = np.zeros((nimg, nimg), dtype=complex)
wtgrd = np.zeros((nimg, nimg), dtype=float)
for sgn, dat in [(+1, vis_sub.data[:,chan_mid]), (-1, np.conj(vis_sub.data[:,chan_mid]))]:
    up = (-sgn * uu_c * dpix_d * nimg + nimg//2).astype(int)   # col = -u (FITS East-left)
    vp = ( sgn * vv_c * dpix_d * nimg + nimg//2).astype(int)
    msk = (up>=0)&(up<nimg)&(vp>=0)&(vp<nimg)
    np.add.at(dirty, (vp[msk], up[msk]), dat[msk])
    np.add.at(wtgrd, (vp[msk], up[msk]), 1.0)
s = wtgrd > 0; dirty[s] /= wtgrd[s]
dirty_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(dirty))).real
vlsr_chan = v_lsrk_chan[chan_mid]
vmax_d = np.nanmax(np.abs(dirty_img))
im = ax.imshow(dirty_img, origin="lower", extent=fits_ext,
               cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)
ax.set_xlabel("ΔRA [arcsec]"); ax.set_ylabel("ΔDec [arcsec]")
ax.set_title(f"Dirty image (data, v={vlsr_chan:.1f} km/s)\n[col=−u, East on left]")
plt.colorbar(im, ax=ax, label="Jy/pix")

fig.suptitle(
    f"End-to-end: Flaherty → NUFFT → MAPS MS  |  red χ²={red_chi2:.2f}  "
    f"|  render={t_render:.1f}s, NUFFT={t_nufft:.1f}s",
    fontsize=11
)
plt.tight_layout()
plt.savefig(f"{OUT}/e2e_flaherty_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  → saved e2e_flaherty_maps.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(f"T1 point source at phase center  |err|={err1:.2e}  {'PASS' if err1 < 1e-4 else 'FAIL'}")
print(f"T2 RA offset +{dx} pix           |err|={err2:.2e}  {'PASS' if err2 < 1e-4 else 'FAIL'}")
print(f"T3 Dec offset +{dy} pix          |err|={err3:.2e}  {'PASS' if err3 < 1e-4 else 'FAIL'}")
print(f"T5 end-to-end red χ²             {red_chi2:.3f} (expect ~few for unoptimized params)")
print()
print(f"Figures saved to: {OUT}/")
