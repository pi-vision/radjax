# Visibility module context — radjax/core/casa_io.py

Picked up from a multi-session thread. Everything below is the authoritative current state.

---

## What exists

### `radjax/core/casa_io.py` (new, untracked)

Full NUFFT-based visibility forward model. Key objects:

```python
VisibilityData        # dataclass: uvw (nrows,3), data (nrows,nchan) complex64,
                      #   weight (nrows,nchan), freq (nchan,), flag (nrows,nchan)
read_ms(ms_path)      # loads CASA MS → Stokes I (XX+YY)/2, zeros flagged weights
image_to_vis(image_cube, uvw, freq, dpix_rad, eps=1e-6)  # NUFFT forward model
chi2_vis(obs, vis_model)  # weighted chi²
```

### `scripts/validate_casa_io.py` (new, untracked)

Runs 4 sanity checks + end-to-end test. All PASS at ~4e-8 error.
Saves figures to `/scratch/ondemand28/len/data/radjax/casa_io_validation/`.

Neither file has been committed yet (user said no git for now).

---

## Key conventions (hard-won, fully verified)

### Sky grid / image orientation

**Use the gofish sky grid** for any real-data rendering:

```python
from gofish import imagecube
cube_fits = imagecube(FITS_PATH, FOV=fov_as)
xax, yax  = cube_fits.xaxis, cube_fits.yaxis
x_sky, y_sky = np.meshgrid(xax, yax, indexing='xy')
```

`xax` decreases left-to-right (East at col=0, West at col=N-1). This is the FITS/CDELT convention. Rendering on this grid gives a model image where:
- **increasing col = West** (decreasing RA)
- **increasing row = North**
- **East is at col=0** (left side of array)

Display with `extent=[xax[0], xax[-1], yax[0], yax[-1]]` — East lands on the left with no `invert_xaxis()` call needed.

### NUFFT sign convention (`image_to_vis`)

finufft type-2, `isign=-1`, with **`y = -uu * scale`** (not +uu):

```python
x = ( vv[:, c] * scale)   # row axis = Dec = +v  
y = (-uu[:, c] * scale)   # col axis = RA  = -u  (negated because increasing col = West)
```

This is correct for the gofish grid. Derivation: a point at col=N/2−dx (East offset +dx) gives
`V = exp(−2πi · u · dx · dpix_rad)` = standard radio FT. ✓

**Wrong for a symmetric grid** (px = (arange−N/2)·dpix, col increases = East) — that would need y=+uu. Don't use symmetric grids with real data.

### Dirty image convention

**`col = −u`**, display with gofish extent (East on left):

```python
for sgn, data in [(+1, vis_1d), (-1, np.conj(vis_1d))]:
    up = (-sgn * uu * dpix_rad * npix + npix//2).astype(int)  # col = -u
    vp = ( sgn * vv * dpix_rad * npix + npix//2).astype(int)  # row = +v
    ...
dirty_img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid))).real
```

Display with `extent=[xax[0], xax[-1], yax[0], yax[-1]]` — no `invert_xaxis()`.

---

## Data paths

| Item | Path |
|------|------|
| MAPS CO 2-1 FITS | `/scratch/ondemand28/len/data/radjax/HD_163296_CO_220GHz.0.15arcsec.image.fits` |
| MAPS MS (cvel) | `/scratch/ondemand28/len/data/radjax/MAPS_uv/HD_163296_CO_220GHz_custom_0.1kms.bin_30s.ms.cvel` |
| Flaherty params YAML | `data/HD163296/Flaherty/HD163296_CO_highres_cen.cm.gaia.params.yaml` |
| Validation figures | `/scratch/ondemand28/len/data/radjax/casa_io_validation/` |

MAPS MS: 374k rows, 308 channels at 0.1 km/s, ~30s binning. Freq range ≈ 230.32–230.74 GHz LSRK. Stokes I from XX+YY.

---

## End-to-end numbers (Flaherty params vs MAPS MS)

- Rendered 256×256 at 20 mas/pix (gofish crop), 31 channels (every 10th), nray=100
- Render time: ~4.3 s, NUFFT time: ~0.8 s
- **Reduced χ² = 9.6** (unoptimized Flaherty params on MAPS data — expected to be large; Flaherty was fit to a different dataset)

---

## Open question raised at end of session

**Variance per UV bin** — need to diagnose whether χ²=9.6 is from model mismatch or miscalibrated MS weights.

Two diagnostics to add to `validate_casa_io.py`:

1. **Empirical scatter vs weight**: use line-free channels (no signal), compute Re(V) scatter per UV bin, compare to `1/sqrt(mean_weight)` per bin. If they match, weights are correct.

2. **Binned χ²**: plot `chi²_per_bin / N_per_bin` vs UV distance. Should be flat at 1.0 if weights are correct and model matches. Deviations tell you which UV scales are driving the mismatch.

Implementation sketch:

```python
# line-free channels: channels far from v_sys where |V| ≈ 0
line_free_mask = np.abs(v_lsrk - vlsr_kms) > 3.0   # > 3 km/s from systemic
vis_lf = vis_obs.data[:, line_free_mask]             # (nrows, n_lf)
w_lf   = vis_obs.weight[:, line_free_mask]

# per-baseline scatter
empirical_var = np.var(vis_lf.real, axis=1)          # (nrows,) — per-row noise
mean_w        = w_lf.mean(axis=1)                     # (nrows,) — mean weight per row
# should have empirical_var ≈ 1/(2 * mean_w)  [factor 2 from real part only]

# bin by UV distance
lam  = CC / vis_obs.freq.mean()
uvd  = np.sqrt(vis_obs.uvw[:,0]**2 + vis_obs.uvw[:,1]**2) / lam
bins = np.logspace(4, 7, 40)
# ... bin and compare empirical_var vs 1/mean_w per bin
```

---

## What is NOT done yet

- [ ] Add binned-variance / binned-chi² diagnostic to `validate_casa_io.py`
- [ ] Commit `casa_io.py`, `__init__.py`, `radjax/CLAUDE.md`, `validate_casa_io.py`
- [ ] Wire `image_to_vis` into MCMC (`scripts/mcmc_co_gaia.py`) — replace image-plane chi² with visibility chi²
- [ ] Wire into neural field training (`scripts/gap_disk_nn_temp.py`) — replace FITS loss with MS loss
- [ ] Read the MAPS MS WEIGHT_SPECTRUM column (per-channel weights) instead of WEIGHT (per-pol only) — may improve weight accuracy
