"""
CASA measurement set I/O and visibility-domain forward model.

Key objects
-----------
VisibilityData : dataclass holding Stokes I visibilities from an MS.
read_ms        : load and average an MS to Stokes I.
image_to_vis   : NUFFT forward model (image cube → model visibilities).
chi2_vis       : weighted chi² in visibility space.

Unit conventions
----------------
- UVW in meters (as stored in the MS).
- Frequencies in Hz.
- Visibilities in Jy (DATA column of the MS).
- Weights are 1/sigma² per Jy (thermal weights from the WEIGHT column).
- Image cube in Jy/pixel (radjax output, post pixel_area fix).
- Pixel size in radians for the NUFFT.

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

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CC = 2.99792458e8   # speed of light [m/s]


@dataclass
class VisibilityData:
    """
    Stokes I visibilities from a CASA measurement set.

    Attributes
    ----------
    uvw    : (nrows, 3) float64  — baseline vectors [m]
    data   : (nrows, nchan) complex64  — Stokes I visibilities [Jy]
    weight : (nrows, nchan) float32  — weights 1/sigma² [1/Jy²]
    freq   : (nchan,) float64  — channel frequencies [Hz]
    flag   : (nrows, nchan) bool  — True = flagged/bad
    """
    uvw:    np.ndarray
    data:   np.ndarray
    weight: np.ndarray
    freq:   np.ndarray
    flag:   np.ndarray

    @property
    def nrows(self) -> int:
        return self.data.shape[0]

    @property
    def nchan(self) -> int:
        return self.data.shape[1]

    def uv_wavelengths(self):
        """
        Per-channel (u, v) in wavelengths.

        Returns
        -------
        uu, vv : each (nrows, nchan) float64
        """
        lam = CC / self.freq           # (nchan,) [m]
        uu  = self.uvw[:, 0:1] / lam  # (nrows, 1) / (nchan,) → (nrows, nchan)
        vv  = self.uvw[:, 1:2] / lam
        return uu, vv


def read_ms(
    ms_path: str | Path,
    data_column: str = "DATA",
    spw_id: int = 0,
) -> VisibilityData:
    """
    Load Stokes I visibilities from a CASA measurement set.

    Averages XX and YY polarizations to Stokes I:
      I = (XX + YY) / 2,  weight_I = w_XX + w_YY

    A (row, channel) sample is flagged if either polarization is flagged;
    flagged samples have their weight set to zero.

    Parameters
    ----------
    ms_path : str or Path
    data_column : str
        Column to read: 'DATA', 'CORRECTED_DATA', or 'MODEL_DATA'.
    spw_id : int
        Spectral window index (0 for single-SPW datasets).

    Returns
    -------
    VisibilityData
    """
    try:
        from casacore.tables import table as casatable
    except ImportError:
        raise ImportError(
            "python-casacore is required to read CASA MS files: pip install python-casacore"
        )

    ms_path = str(ms_path)

    ms      = casatable(ms_path, readonly=True, ack=False)
    data    = ms.getcol(data_column)   # (nrows, nchan, npol) complex64
    uvw     = ms.getcol('UVW')         # (nrows, 3) float64, meters
    weight  = ms.getcol('WEIGHT')      # (nrows, npol) float32 — per-pol, NOT per-channel
    flag    = ms.getcol('FLAG')        # (nrows, nchan, npol) bool
    ms.close()

    spw_tab = casatable(ms_path + '/SPECTRAL_WINDOW', readonly=True, ack=False)
    freq    = spw_tab.getcol('CHAN_FREQ')[spw_id].astype(np.float64)   # (nchan,) Hz
    spw_tab.close()

    # Average XX + YY → Stokes I
    data_I   = 0.5 * (data[..., 0] + data[..., 1])    # (nrows, nchan) complex64
    flag_I   = flag[..., 0] | flag[..., 1]             # (nrows, nchan) bool

    # WEIGHT is per-polarization, broadcast over channels
    nchan    = freq.shape[0]
    weight_I = (weight[:, 0] + weight[:, 1])[:, None]  # (nrows, 1) float32
    weight_I = np.broadcast_to(weight_I, (uvw.shape[0], nchan)).copy()

    # Zero out flagged samples
    weight_I[flag_I] = 0.0

    return VisibilityData(
        uvw    = uvw,
        data   = data_I,
        weight = weight_I,
        freq   = freq,
        flag   = flag_I,
    )


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


__all__ = ["VisibilityData", "read_ms", "image_to_vis", "chi2_vis"]
