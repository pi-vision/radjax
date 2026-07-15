"""
CASA measurement set I/O.

Key objects
-----------
VisibilityData : dataclass holding Stokes I visibilities from an MS.
read_ms        : load and average an MS to Stokes I.

Unit conventions
----------------
- UVW in meters (as stored in the MS).
- Frequencies in Hz.
- Visibilities in Jy (DATA column of the MS).
- Weights are 1/sigma² per Jy (thermal weights from the WEIGHT column).

See `vis_forward_model.py` for the NUFFT forward model (`image_to_vis`) and
visibility chi² (`chi2_vis`) that consume this data.
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


__all__ = ["VisibilityData", "read_ms"]
