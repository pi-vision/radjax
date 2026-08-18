"""
Visibility-domain forward model: image cube -> model visibilities -> chi².

Key objects
-----------
image_to_vis : NUFFT forward model (image cube → model visibilities).
chi2_vis     : weighted chi² in visibility space.

Format-agnostic: takes plain (uvw, freq) arrays, so it works with
`VisibilityData` from any reader (`casa_io.read_ms`, a future UVFITS reader, ...).

Unit conventions
----------------
- Image cube in Jy/pixel (radjax output, post pixel_area fix).
- Pixel size in radians for the NUFFT.
- uvw in meters, freq in Hz — matches `casa_io.VisibilityData`.

NUFFT notes
-----------
image_to_vis uses finufft type-2 (uniform → non-uniform), which evaluates
the continuous FT of the model image at each observed (u,v) baseline without
FFT + bilinear interpolation. The (u,v) coordinates are converted to radians/pixel:
    omega = 2*pi * u_wavelengths * dpix_rad
finufft type-2 stores its input with DC at array index (N/2, N/2), matching
our image layout — pass the image directly without any fftshift.
No cell_size² scaling is needed because our image is already in Jy/pixel
(not Jy/sr as in MPoL).

Backend: jax-finufft (not the plain `finufft` package). Same underlying FINUFFT
C++ library and numerics -- verified to match plain `finufft.nufft2d2` to ~1e-14
relative precision, and its gradient verified against finite differences (~1e-6
relative, consistent with finite-difference step size). The difference is that
jax-finufft registers this as a proper JAX primitive with custom_vjp rules, so
`jax.grad` can trace through it -- required for gradient-based neural temperature
recovery (radjax/core/neural.py's train_step), unlike plain `finufft` which
operates on numpy arrays and is opaque to JAX autodiff. This currently installed
build of jax-finufft has no GPU (cufinufft) backend, only CPU, so this function
explicitly runs on the CPU device via `jax.default_device` -- the same place the
NUFFT step already ran before (finufft is numpy/CPU-only too), so this is not a
new performance regression, just now differentiable.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from .casa_io import CC, VisibilityData

_CPU_DEVICE = jax.devices("cpu")[0]


def image_to_vis(
    image_cube,
    uvw: np.ndarray,
    freq: np.ndarray,
    dpix_rad: float,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Compute model visibilities from an image cube using finufft type-2 (NUFFT).

    Uses Kaiser-Bessel window interpolation (finufft) rather than FFT + bilinear,
    avoiding aliasing artifacts from interpolation in a dense FFT grid. JAX-
    differentiable (see module docstring) -- `image_cube` may be a JAX array
    that is part of an active autodiff trace (e.g. a neural field's rendered
    cube); gradients flow through this function correctly.

    Convention
    ----------
    - Image in Jy/pixel; no cell_size^2 scaling needed (already folded in).
    - Phase center at pixel (N//2, N//2); finufft type-2 expects DC at that
      index, so pass the image directly without fftshift.
    - (u,v) in wavelengths -> radians/pixel: omega = 2*pi * u_wavelengths * dpix_rad.
    - (u,v) are channel-dependent since u = D_meters / lambda(freq).

    Parameters
    ----------
    image_cube : (nchan, npix, npix) float or complex, numpy or JAX array
        Model image in Jy/pixel. Phase center at pixel (npix//2, npix//2).
    uvw : (nrows, 3) float
        Baseline vectors [m].
    freq : (nchan,) float
        Channel frequencies [Hz].
    dpix_rad : float
        Image pixel size [rad].
    eps : float
        NUFFT accuracy tolerance (default 1e-6).

    Returns
    -------
    vis_model : (nrows, nchan) complex128 JAX array
        Model visibilities in Jy, on the CPU device (see module docstring).
    """
    nchan = image_cube.shape[0]

    lam   = CC / jnp.asarray(freq)               # (nchan,) [m]
    uu    = jnp.asarray(uvw[:, 0:1]) / lam       # (nrows, nchan) [wavelengths], East
    vv    = jnp.asarray(uvw[:, 1:2]) / lam       # (nrows, nchan), North
    scale = 2.0 * jnp.pi * dpix_rad              # wavelengths -> rad/pix

    image_cube = jnp.asarray(image_cube).astype(jnp.complex128)

    with jax.default_device(_CPU_DEVICE):
        image_cube = jax.device_put(image_cube, _CPU_DEVICE)
        uu = jax.device_put(uu, _CPU_DEVICE)
        vv = jax.device_put(vv, _CPU_DEVICE)

        import jax_finufft  # optional dep: pip install radjax[vis]
        vis_channels = []
        for c in range(nchan):
            # finufft type-2 with DC at (N//2, N//2), isign=-1:
            #   out[j] = sum_{k1,k2} f[k1+N/2, k2+N/2] * exp(-i*(x[j]*k1 + y[j]*k2))
            # Image convention (FITS / gofish): increasing row = North (+m), increasing col = West (-l).
            # East is at col=0 (left), West at col=N-1 (right) — CDELT_RA < 0.
            # Standard radio FT: V(u,v) = F * exp(-2πi*(u*l + v*m))
            # With k2 = col - N/2 and l = -(k2)*dpix (West = +col → negative RA):
            #   x[j] = 2π*v*dpix  (Dec/row, positive North)
            #   y[j] = -2π*u*dpix (RA/col, negated because increasing col = West = -RA)
            x = ( vv[:, c] * scale).astype(jnp.float64)
            y = (-uu[:, c] * scale).astype(jnp.float64)
            vis_channels.append(jax_finufft.nufft2(image_cube[c], x, y, iflag=-1, eps=eps))
        vis_model = jnp.stack(vis_channels, axis=1)   # (nrows, nchan)

    return vis_model


def chi2_vis(
    obs: VisibilityData,
    vis_model,
):
    """
    Weighted chi² between observed and model visibilities.

    chi² = Σ_{i,c} weight[i,c] × |data[i,c] - vis_model[i,c]|²

    Flagged samples (weight=0) contribute zero automatically.

    JAX-differentiable: uses `jnp` ops throughout and does not cast to a
    Python `float`, so this can be called on a traced `vis_model` (e.g. inside
    `jax.grad` during neural-field training) as well as on concrete arrays
    (e.g. MCMC, where the caller can wrap the result in `float(...)` itself --
    see `scripts/mcmc_co_gaia_vis.py`). Bare `np.sum`/`np.abs` would raise
    `jax.errors.TracerArrayConversionError` under tracing; `jnp` ops do not.

    Parameters
    ----------
    obs : VisibilityData
    vis_model : (nrows, nchan) complex, numpy or JAX array

    Returns
    -------
    float or JAX scalar
        A JAX scalar if `vis_model` is a JAX array (possibly traced); a
        numpy/Python scalar if `vis_model` is a plain numpy array.
    """
    data, weight = obs.data, obs.weight
    if isinstance(vis_model, jax.Array) and not isinstance(vis_model, jax.core.Tracer):
        # image_to_vis's output is CPU-resident (jax-finufft has no GPU backend yet).
        # obs.data/weight are plain numpy and would otherwise get implicitly placed on
        # JAX's default device (GPU) the moment they touch a jnp op below, causing a
        # "Received incompatible devices" error on this subtraction. Under an active
        # jax.grad/jit trace, vis_model is a Tracer (no concrete .devices()) and XLA
        # handles any needed transfer itself, so this only applies to eager calls.
        target_device = next(iter(vis_model.devices()))
        data = jax.device_put(data, target_device)
        weight = jax.device_put(weight, target_device)
    residual = data - vis_model
    return jnp.sum(weight * jnp.abs(residual) ** 2)


def pad_cube(cube: np.ndarray, npix_full: int) -> np.ndarray:
    """
    Zero-pad a model image cube to a larger spatial size.

    Embeds the cube in the centre of a (nchan, npix_full, npix_full) array.
    The pixel size (dpix_rad) is unchanged, so the NUFFT coordinates are
    unaffected — only the FOV increases.  The sky outside the rendered region
    is zero, which is correct when the disk emission is confined to the
    inner npix pixels.

    Parameters
    ----------
    cube     : (nchan, npix, npix) float — rendered model cube
    npix_full: int — target spatial size (must be >= npix)

    Returns
    -------
    (nchan, npix_full, npix_full) float, dtype preserved
    """
    nchan, npix, _ = cube.shape
    if npix_full < npix:
        raise ValueError(f"npix_full={npix_full} must be >= cube npix={npix}")
    if npix_full == npix:
        return cube
    out = np.zeros((nchan, npix_full, npix_full), dtype=cube.dtype)
    pad = (npix_full - npix) // 2
    out[:, pad:pad + npix, pad:pad + npix] = cube
    return out


def _dirty_image(uu_c, vv_c, vis_c, npix, dpix_rad):
    """Natural-weight dirty image for one channel (nearest-neighbour gridding + IFFT)."""
    dirty = np.zeros((npix, npix), dtype=complex)
    wtgrd = np.zeros((npix, npix))
    for sgn, dat in [(+1, vis_c), (-1, np.conj(vis_c))]:
        col = np.round(-sgn * uu_c * dpix_rad * npix + npix / 2).astype(int)
        row = np.round( sgn * vv_c * dpix_rad * npix + npix / 2).astype(int)
        ok  = (col >= 0) & (col < npix) & (row >= 0) & (row < npix)
        np.add.at(dirty, (row[ok], col[ok]), dat[ok])
        np.add.at(wtgrd, (row[ok], col[ok]), 1.0)
    s = wtgrd > 0
    dirty[s] /= wtgrd[s]
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(dirty))).real


def plot_dirty_comparison(
    vis: VisibilityData,
    vis_model: np.ndarray,
    dpix_rad: float,
    out_path: str | Path,
    nchans: int = 5,
    npix: int = 256,
    channels: list | None = None,
) -> None:
    """
    Test 1: dirty images of data, model, and residual for several channels.

    For each channel, grids the observed and model visibilities onto the same
    UV grid using natural weighting and IFFTs to produce dirty images. The
    residual (data - model) should look like noise if the model is correct.
    Orientation errors (flipped axis, wrong PA) are immediately visible.

    Parameters
    ----------
    vis       : VisibilityData
    vis_model : (nrows, nchan) complex — output of image_to_vis
    dpix_rad  : float — pixel size [rad] (must match the rendered cube)
    out_path  : path for the saved figure
    nchans    : number of channels to show, evenly spaced (default 5)
    npix      : dirty image size in pixels (default 256)
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nchan    = vis.nchan
    chan_idx = np.array(channels) if channels is not None else \
               np.round(np.linspace(0, nchan - 1, nchans)).astype(int)
    nchans   = len(chan_idx)

    lam = CC / vis.freq                  # (nchan,) [m]
    uu  = vis.uvw[:, 0:1] / lam         # (nrows, nchan) [wavelengths]
    vv  = vis.uvw[:, 1:2] / lam

    dpix_as  = dpix_rad * (180 / np.pi * 3600)
    half_as  = npix / 2 * dpix_as
    ext      = [half_as, -half_as, -half_as, half_as]   # East left, North top

    fig, axes = plt.subplots(nchans, 3, figsize=(10, 3 * nchans))
    if nchans == 1:
        axes = axes[np.newaxis, :]

    for row, c in enumerate(chan_idx):
        uu_c = uu[:, c].astype(np.float64)
        vv_c = vv[:, c].astype(np.float64)

        d_dirty = _dirty_image(uu_c, vv_c, vis.data[:, c],      npix, dpix_rad)
        m_dirty = _dirty_image(uu_c, vv_c, vis_model[:, c],     npix, dpix_rad)
        r_dirty = _dirty_image(uu_c, vv_c, vis.data[:, c] - vis_model[:, c], npix, dpix_rad)

        vmax = np.nanpercentile(np.abs(d_dirty), 99)

        for ax, img, title in zip(axes[row], [d_dirty, m_dirty, r_dirty],
                                  ["Data", "Model (NUFFT)", "Residual"]):
            im = ax.imshow(img, origin="lower", extent=ext,
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set(xlabel="ΔRA [″]", ylabel="ΔDec [″]")
            v_kms = CC * (1 - vis.freq[c] / (CC / 1.3e-3)) / 1e3  # rough LSRK
            ax.set_title(f"{title}  ch={c}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jy/beam")

    fig.suptitle("Dirty image: data vs model vs residual", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_selfconsistency(
    cube: np.ndarray,
    vis: VisibilityData,
    vis_model: np.ndarray,
    dpix_rad: float,
    out_path: str | Path,
    nchans: int = 3,
    channels: list | None = None,
) -> None:
    """
    Test 2: self-consistency check — model image vs dirty image of model visibilities.

    Renders the model image directly and reconstructs it by dirty-imaging the
    NUFFT model visibilities. Both should show the same disk structure (the dirty
    image is PSF-blurred, but orientation, scale, and center must match). A sign
    error or FOV bug in the NUFFT will produce a scrambled or mirrored dirty image.

    Parameters
    ----------
    cube      : (nchan, npix_r, npix_r) float — model image from the renderer
    vis       : VisibilityData
    vis_model : (nrows, nchan) complex — output of image_to_vis
    dpix_rad  : float — pixel size [rad] (same for cube and dirty image)
    out_path  : path for the saved figure
    nchans    : number of channels to show, evenly spaced (default 3)
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    nchan_cube = cube.shape[0]
    npix_cube  = cube.shape[1]
    chan_idx   = np.array(channels) if channels is not None else \
                 np.round(np.linspace(0, nchan_cube - 1, nchans)).astype(int)
    nchans     = len(chan_idx)

    lam = CC / vis.freq
    uu  = vis.uvw[:, 0:1] / lam
    vv  = vis.uvw[:, 1:2] / lam

    dpix_as = dpix_rad * (180 / np.pi * 3600)
    half_as = npix_cube / 2 * dpix_as
    ext     = [half_as, -half_as, -half_as, half_as]

    fig, axes = plt.subplots(nchans, 3, figsize=(10, 3 * nchans))
    if nchans == 1:
        axes = axes[np.newaxis, :]

    for row, c in enumerate(chan_idx):
        model_img = cube[c]
        m_dirty   = _dirty_image(uu[:, c].astype(np.float64),
                                 vv[:, c].astype(np.float64),
                                 vis_model[:, c], npix_cube, dpix_rad)
        diff      = model_img - m_dirty

        vmax = np.nanpercentile(np.abs(model_img), 99.5)

        for ax, img, title in zip(axes[row],
                                  [model_img, m_dirty, diff],
                                  ["Model image (direct)", "Dirty (NUFFT→grid→IFFT)", "Difference"]):
            im = ax.imshow(img, origin="lower", extent=ext,
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set(xlabel="ΔRA [″]", ylabel="ΔDec [″]")
            ax.set_title(f"{title}  ch={c}", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jy/pix")

    fig.suptitle("Self-consistency: model image vs dirty image of NUFFT visibilities\n"
                 "(structures should match; dirty is PSF-blurred but not mirrored/scrambled)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_vis_diagnostics(
    vis: VisibilityData,
    vis_model: np.ndarray,
    out_path: str | Path,
    nbins: int = 80,
) -> None:
    """
    Save a 3-panel diagnostic figure: data |V|, model |V|, and sigma_knn vs UV distance.

    All quantities are binned by UV distance in kλ (using per-channel baselines).
    Flagged samples are excluded.

    Parameters
    ----------
    vis       : VisibilityData (sigma_knn populated for the third panel)
    vis_model : (nrows, nchan) complex — output of image_to_vis
    out_path  : path for the saved figure (.png or .pdf)
    nbins     : number of UV-distance bins (default 80)
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-(row, chan) UV distance in kλ
    lam = CC / vis.freq                         # (nchan,) [m/cycle]
    uu  = vis.uvw[:, 0:1] / lam                # (nrows, nchan) [wavelengths]
    vv  = vis.uvw[:, 1:2] / lam
    uvd = np.sqrt(uu ** 2 + vv ** 2) / 1e3    # (nrows, nchan) [kλ]

    mask = ~vis.flag                            # True = unflagged
    uvd_f  = uvd[mask]
    data_f = np.abs(vis.data[mask])
    modl_f = np.abs(vis_model[mask])

    bins    = np.linspace(uvd_f.min(), uvd_f.max(), nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bidx    = np.clip(np.digitize(uvd_f, bins) - 1, 0, nbins - 1)

    def _bin_mean(values):
        counts = np.bincount(bidx, minlength=nbins).astype(float)
        sums   = np.bincount(bidx, weights=values, minlength=nbins)
        return np.where(counts > 0, sums / counts, np.nan)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=False)

    axes[0].plot(centers, _bin_mean(data_f), "k-", lw=1.5)
    axes[0].set(xlabel="UV distance [kλ]", ylabel="|V| [Jy]", title="Data")

    axes[1].plot(centers, _bin_mean(modl_f), "C0-", lw=1.5)
    axes[1].set(xlabel="UV distance [kλ]", ylabel="|V| [Jy]", title="Model (NUFFT)")

    if vis.sigma_knn is not None:
        sig_f = vis.sigma_knn[mask]
        axes[2].plot(centers, _bin_mean(sig_f), "C3-", lw=1.5)
        axes[2].set(xlabel="UV distance [kλ]", ylabel="σ [Jy]", title="KNN σ per visibility")
    else:
        axes[2].text(0.5, 0.5, "sigma_knn not computed",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("KNN σ per visibility")

    # chi² per sample, binned by UV distance
    w_f    = vis.weight[mask]
    chi2_f = w_f * np.abs((vis.data[mask] - vis_model[mask])) ** 2
    axes[3].plot(centers, _bin_mean(chi2_f), "C2-", lw=1.5)
    axes[3].axhline(1.0, color="k", lw=0.8, ls="--", label="χ²=1")
    axes[3].set(xlabel="UV distance [kλ]", ylabel="χ² per visibility", title="χ² vs UV distance")
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


__all__ = ["image_to_vis", "chi2_vis", "pad_cube", "plot_dirty_comparison", "plot_selfconsistency", "plot_vis_diagnostics"]
