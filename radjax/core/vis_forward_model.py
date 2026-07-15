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
"""
from __future__ import annotations

import numpy as np

from .casa_io import CC, VisibilityData


def image_to_vis(
    image_cube: np.ndarray,
    uvw: np.ndarray,
    freq: np.ndarray,
    dpix_rad: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute model visibilities from an image cube using finufft type-2 (NUFFT).

    Uses Kaiser-Bessel window interpolation (finufft) rather than FFT + bilinear,
    avoiding aliasing artifacts from interpolation in a dense FFT grid.

    Convention
    ----------
    - Image in Jy/pixel; no cell_size^2 scaling needed (already folded in).
    - Phase center at pixel (N//2, N//2); finufft type-2 expects DC at that
      index, so pass the image directly without fftshift.
    - (u,v) in wavelengths -> radians/pixel: omega = 2*pi * u_wavelengths * dpix_rad.
    - (u,v) are channel-dependent since u = D_meters / lambda(freq).

    Parameters
    ----------
    image_cube : (nchan, npix, npix) float
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
    vis_model : (nrows, nchan) complex128
        Model visibilities in Jy.
    """
    import finufft

    nchan, npix, _ = image_cube.shape
    nrows = uvw.shape[0]

    lam   = CC / freq                  # (nchan,) [m]
    uu    = uvw[:, 0:1] / lam          # (nrows, nchan) [wavelengths], East
    vv    = uvw[:, 1:2] / lam          # (nrows, nchan), North
    scale = 2.0 * np.pi * dpix_rad    # wavelengths -> rad/pix

    vis_model = np.zeros((nrows, nchan), dtype=np.complex128)

    for c in range(nchan):
        # finufft type-2 with DC at (N//2, N//2), isign=-1:
        #   out[j] = sum_{k1,k2} f[k1+N/2, k2+N/2] * exp(-i*(x[j]*k1 + y[j]*k2))
        # Image convention (FITS / gofish): increasing row = North (+m), increasing col = West (-l).
        # East is at col=0 (left), West at col=N-1 (right) — CDELT_RA < 0.
        # Standard radio FT: V(u,v) = F * exp(-2πi*(u*l + v*m))
        # With k2 = col - N/2 and l = -(k2)*dpix (West = +col → negative RA):
        #   x[j] = 2π*v*dpix  (Dec/row, positive North)
        #   y[j] = -2π*u*dpix (RA/col, negated because increasing col = West = -RA)
        img = image_cube[c].astype(np.complex128)
        x = ( vv[:, c] * scale).astype(np.float64)
        y = (-uu[:, c] * scale).astype(np.float64)
        vis_model[:, c] = finufft.nufft2d2(x, y, img, isign=-1, eps=eps)

    return vis_model


def chi2_vis(
    obs: VisibilityData,
    vis_model: np.ndarray,
) -> float:
    """
    Weighted chi² between observed and model visibilities.

    chi² = Σ_{i,c} weight[i,c] × |data[i,c] - vis_model[i,c]|²

    Flagged samples (weight=0) contribute zero automatically.

    Parameters
    ----------
    obs : VisibilityData
    vis_model : (nrows, nchan) complex

    Returns
    -------
    float
    """
    residual = obs.data - vis_model
    return float(np.sum(obs.weight * np.abs(residual) ** 2))


__all__ = ["image_to_vis", "chi2_vis"]
