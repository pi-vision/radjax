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
    uvw       : (nrows, 3) float64  — baseline vectors [m]
    data      : (nrows, nchan) complex64  — Stokes I visibilities [Jy]
    weight    : (nrows, nchan) float32  — weights 1/sigma² [1/Jy²]
    freq      : (nchan,) float64  — channel frequencies [Hz]
    flag      : (nrows, nchan) bool  — True = flagged/bad
    sigma_knn : (nrows, nchan) float32 or None  — per-visibility noise estimate [Jy]
    """
    uvw:       np.ndarray
    data:      np.ndarray
    weight:    np.ndarray
    freq:      np.ndarray
    flag:      np.ndarray
    sigma_knn: np.ndarray | None = None

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

    def freq_rest(self, vlsr: float, nu0: float) -> np.ndarray:
        """
        Convert stored LSRK channel frequencies to rest-frame frequencies.

        Parameters
        ----------
        vlsr : float
            Systemic velocity [m/s], from observation params YAML.
        nu0 : float
            Rest frequency of the line [Hz], e.g. 230538e6 for CO 2-1.

        Returns
        -------
        (nchan,) float64 — rest-frame frequencies [Hz]
        """
        return self.freq + nu0 * vlsr / CC


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

    # Average XX + YY → Stokes I, then conjugate.
    # CASA stores visibilities as the complex conjugate of the TMS-standard convention
    # V(u,v) = ∫ I(l,m) exp(-2πi(ul+vm)) dl dm.  Conjugating the data restores the
    # standard convention without touching UVW.
    # Reference: https://mpol-dev.github.io/MPoL/units-and-conventions.html
    data_I   = np.conj(0.5 * (data[..., 0] + data[..., 1]))    # (nrows, nchan) complex64
    flag_I   = flag[..., 0] | flag[..., 1]             # (nrows, nchan) bool

    # WEIGHT is per-polarization, broadcast over channels
    nchan    = freq.shape[0]
    weight_I = (weight[:, 0] + weight[:, 1])[:, None]  # (nrows, 1) float32
    weight_I = np.broadcast_to(weight_I, (uvw.shape[0], nchan)).copy()

    # Zero out flagged samples
    weight_I[flag_I] = 0.0

    # Sort channels to increasing frequency order
    if len(freq) > 1 and freq[1] < freq[0]:
        order   = np.argsort(freq)
        freq    = freq[order]
        data_I  = data_I[:, order]
        flag_I  = flag_I[:, order]
        weight_I = weight_I[:, order]

    return VisibilityData(
        uvw    = uvw,
        data   = data_I,
        weight = weight_I,
        freq   = freq,
        flag   = flag_I,
    )


def save_vis(vis: VisibilityData, path: str | Path) -> None:
    """
    Save a VisibilityData to a compressed .npz file.

    All array fields are stored; sigma_knn is included if present.
    Load back with :func:`load_vis`.

    Parameters
    ----------
    vis  : VisibilityData
    path : str or Path
        Output path. The .npz extension is appended automatically if absent.
    """
    arrays = dict(
        uvw    = vis.uvw,
        data   = vis.data,
        weight = vis.weight,
        freq   = vis.freq,
        flag   = vis.flag,
    )
    if vis.sigma_knn is not None:
        arrays["sigma_knn"] = vis.sigma_knn
    np.savez(path, **arrays)


def load_vis(path: str | Path) -> VisibilityData:
    """
    Load a VisibilityData from a .npz file written by :func:`save_vis`.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    VisibilityData
        sigma_knn is populated if it was saved, otherwise None.
    """
    d = np.load(path)
    return VisibilityData(
        uvw       = d["uvw"],
        data      = d["data"],
        weight    = d["weight"],
        freq      = d["freq"],
        flag      = d["flag"],
        sigma_knn = d["sigma_knn"] if "sigma_knn" in d else None,
    )


def knn_sigma(vis: VisibilityData, k: int = 50) -> np.ndarray:
    """
    Estimate per-visibility noise sigma using k-nearest-neighbour std in UV space.

    For each baseline row, finds the k closest rows by (u, v) distance [m]
    and returns std(Re(V)) across those neighbours per spectral channel.
    KNN is performed in metres (not wavelengths), so the same neighbours are
    used for all channels. This follows Flaherty et al. 2015, who used k=70.

    Parameters
    ----------
    vis : VisibilityData
    k : int
        Number of nearest neighbours (default 50).

    Returns
    -------
    sigma : (nrows, nchan) float32
        Per-visibility noise estimate [Jy].

    Notes
    -----
    Known bias: neighbours are close in UV space but not the *same* baseline,
    so if the source has any real spatial structure (e.g. a compact continuum
    core whose visibility amplitude falls off with baseline length), nearby-
    but-distinct baselines sample slightly different true amplitudes. That
    structure variance leaks into the estimate as if it were noise, biasing
    it high. Empirically on HD163296 MAPS CO(2-1) data (k=70, restricted to
    line-free channels) this was ~15-20% higher than the channel-based
    estimators below (`sigma_from_channel_diff`, `sigma_from_block_mean`),
    which compare the *same* baseline across frequency and so are immune to
    this particular bias. Cross-check against those before trusting kNN sigma
    on a new dataset.
    """
    from scipy.spatial import KDTree

    k = min(k, vis.nrows - 1)
    uv   = vis.uvw[:, :2]                      # (nrows, 2) [m]
    tree = KDTree(uv)
    _, idx = tree.query(uv, k=k + 1)           # k+1 so we can drop the self-hit
    idx = idx[:, 1:]                            # (nrows, k)

    nrows, nchan = vis.data.shape
    sigma = np.empty((nrows, nchan), dtype=np.float32)
    batch = 2048
    for i in range(0, nrows, batch):
        nb = vis.data[idx[i : i + batch]]       # (batch, k, nchan) complex64
        sigma[i : i + batch] = np.std(nb.real, axis=1, ddof=1)
    return sigma


def sigma_from_weight(vis: VisibilityData) -> np.ndarray:
    """
    Per-visibility sigma implied by the MS WEIGHT column: sigma = 1/sqrt(weight).

    This is CASA's own thermal-noise estimate -- computed online from
    integration time and system temperature via the radiometer equation, then
    propagated through calibration -- not an empirical estimate from the
    visibility scatter itself (contrast with the sigma_from_* functions below,
    and with `knn_sigma`).

    WEIGHT is stored per row, not per channel, in the MS; `read_ms` already
    broadcasts it identically across all channels for a row (see VisibilityData
    docs), so every column of the returned array is identical except where a
    specific channel was individually flagged.

    Parameters
    ----------
    vis : VisibilityData

    Returns
    -------
    sigma : (nrows, nchan) float64
        Per-visibility noise estimate [Jy]. inf where weight == 0 (flagged).

    Example
    -------
    >>> vis = read_ms(ms_path)
    >>> sigma = sigma_from_weight(vis)
    >>> sigma[:, 0]   # one value per row (constant across unflagged channels)
    """
    with np.errstate(divide="ignore"):
        return 1.0 / np.sqrt(vis.weight)


def _split_indices(indices) -> tuple[np.ndarray, np.ndarray]:
    """Split a single channel-index array in half: first half = block A, second half = block B."""
    indices = np.asarray(indices)
    if len(indices) % 2 != 0:
        raise ValueError(
            f"indices must have even length (equal-size first/second halves), got {len(indices)}"
        )
    n = len(indices) // 2
    return indices[:n], indices[n:]


def sigma_from_channel_diff(
    vis: VisibilityData,
    indices,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-row sigma from positionally-paired differences of two same-length
    channel blocks believed to carry the same underlying signal (e.g. equal-size
    windows on either side of a line, both clear of line flux).

    ``indices`` is split in half: the first half is block A, the second half
    is block B, paired by position. For each row, D_k = V[:, A[k]] - V[:, B[k]]
    compares the *same* baseline at two different frequencies, so any real
    source structure (which does not depend on the tiny frequency offset
    between A and B) cancels exactly, leaving close to pure thermal noise.
    Since D = A - B with A, B independent and equal-variance, Var(D) =
    2*sigma^2, hence the sqrt(2) correction below. This is the most robust of
    the estimators here -- it is immune to the structure-contamination bias
    that affects `knn_sigma`.

    Rows with any flagged sample among ``indices`` come back as NaN.

    Parameters
    ----------
    vis : VisibilityData
    indices : array-like of int, even length
        Channel indices to use; split in half into two equal-size blocks,
        paired by position. E.g. the first and last 20 channels of a
        spectral window: ``np.concatenate([np.arange(20), np.arange(vis.nchan - 20, vis.nchan)])``.

    Returns
    -------
    sigma_real, sigma_imag : (nrows,) float64
        Per-row noise estimate [Jy] for the real and imaginary components.

    Example
    -------
    >>> vis = read_ms(ms_path)
    >>> indices = np.concatenate([np.arange(20), np.arange(vis.nchan - 20, vis.nchan)])
    >>> sigma_real, sigma_imag = sigma_from_channel_diff(vis, indices)
    >>> weight_row = 1.0 / sigma_real**2   # broadcast across the fitted channels for chi2_vis
    """
    idx_a, idx_b = _split_indices(indices)

    flagged = np.any(vis.flag[:, np.concatenate([idx_a, idx_b])], axis=1)
    diff = vis.data[:, idx_a] - vis.data[:, idx_b]

    sigma_real = np.std(diff.real, axis=1, ddof=1) / np.sqrt(2)
    sigma_imag = np.std(diff.imag, axis=1, ddof=1) / np.sqrt(2)
    sigma_real[flagged] = np.nan
    sigma_imag[flagged] = np.nan
    return sigma_real, sigma_imag


def sigma_from_block_mean(
    vis: VisibilityData,
    indices,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-row sigma from two channel blocks, each mean-subtracted separately
    before pooling (Flaherty et al. 2017/2018 style: "dispersion around each
    baseline in line-free channels").

    ``indices`` is split in half into two blocks A and B. Each block's own
    mean is removed first (`a - mean(a)`, `b - mean(b)`) so that a small real
    signal offset *between* the two blocks does not artificially inflate the
    noise estimate -- only within-block scatter contributes. The residuals
    are then pooled and a single std is taken.

    Rows with any flagged sample among ``indices`` come back as NaN.

    Parameters
    ----------
    vis : VisibilityData
    indices : array-like of int, even length
        Channel indices believed to be free of the signal of interest; split
        in half into two blocks (see `sigma_from_channel_diff`).

    Returns
    -------
    sigma_real, sigma_imag : (nrows,) float64
        Per-row noise estimate [Jy] for the real and imaginary components.

    Example
    -------
    >>> vis = read_ms(ms_path)
    >>> indices = np.concatenate([np.arange(30), np.arange(vis.nchan - 30, vis.nchan)])
    >>> sigma_real, sigma_imag = sigma_from_block_mean(vis, indices)
    """
    idx_a, idx_b = _split_indices(indices)

    flagged = np.any(vis.flag[:, np.concatenate([idx_a, idx_b])], axis=1)
    a = vis.data[:, idx_a]
    b = vis.data[:, idx_b]
    a_resid = a - np.mean(a, axis=1, keepdims=True)
    b_resid = b - np.mean(b, axis=1, keepdims=True)
    residual = np.concatenate([a_resid, b_resid], axis=1)

    sigma_real = np.std(residual.real, axis=1, ddof=1)
    sigma_imag = np.std(residual.imag, axis=1, ddof=1)
    sigma_real[flagged] = np.nan
    sigma_imag[flagged] = np.nan
    return sigma_real, sigma_imag


def compare_sigma_estimates(
    vis: VisibilityData,
    indices,
    knn_k: int = 70,
    verbose: bool = True,
) -> dict:
    """
    Run all four per-visibility sigma estimators on the same reference
    channels and report their medians side by side.

    Useful as a one-shot sanity check before committing to a sigma for MCMC:
    if `weight`, `block_mean`, and `channel_diff` agree (they should, to a few
    percent, if the MS WEIGHT column is reasonably well calibrated), that's
    good evidence the estimate is trustworthy. If `knn` disagrees with the
    other three, suspect UV-space structure contamination (see `knn_sigma`
    notes) rather than a problem with the channel-based estimators.

    Parameters
    ----------
    vis : VisibilityData
    indices : array-like of int, even length
        Channel indices believed to be free of the signal of interest; split
        in half into two blocks for `sigma_from_channel_diff` / `sigma_from_block_mean`
        (see those docstrings), and used as-is (unsplit) to restrict `knn_sigma`.
    knn_k : int
        Number of UV-space nearest neighbours for the kNN estimate (default
        70, following Flaherty et al. 2015).
    verbose : bool
        Print a summary table of median sigma per method (default True).

    Returns
    -------
    dict with keys:
        "weight"            : (nrows, nchan) sigma from the MS WEIGHT column
        "block_mean_real"   : (nrows,) sigma from `sigma_from_block_mean` (real)
        "block_mean_imag"   : (nrows,) sigma from `sigma_from_block_mean` (imag)
        "channel_diff_real" : (nrows,) sigma from `sigma_from_channel_diff` (real)
        "channel_diff_imag" : (nrows,) sigma from `sigma_from_channel_diff` (imag)
        "knn"               : (nrows, len(indices)) sigma from `knn_sigma`,
                               restricted to the given channels

    Example
    -------
    >>> vis = read_ms(ms_path)
    >>> indices = np.concatenate([np.arange(20), np.arange(vis.nchan - 20, vis.nchan)])
    >>> results = compare_sigma_estimates(vis, indices)
    method                                     median sigma [Jy]
    MS WEIGHT column                                      0.84155
    block-mean-subtracted (real)                          0.84689
    channel differences (real)                            0.84966
    kNN (k=70)                                            0.97578
    """
    indices = np.asarray(indices)

    sigma_weight = sigma_from_weight(vis)
    sigma_bm_re, sigma_bm_im = sigma_from_block_mean(vis, indices)
    sigma_cd_re, sigma_cd_im = sigma_from_channel_diff(vis, indices)

    vis_ref = VisibilityData(
        uvw    = vis.uvw,
        data   = vis.data[:, indices],
        weight = vis.weight[:, indices],
        freq   = vis.freq[indices],
        flag   = vis.flag[:, indices],
    )
    sigma_knn = knn_sigma(vis_ref, k=knn_k)

    results = {
        "weight": sigma_weight,
        "block_mean_real": sigma_bm_re,
        "block_mean_imag": sigma_bm_im,
        "channel_diff_real": sigma_cd_re,
        "channel_diff_imag": sigma_cd_im,
        "knn": sigma_knn,
    }

    if verbose:
        print(f"{'method':40s}{'median sigma [Jy]':>20s}")
        print(f"{'MS WEIGHT column':40s}{np.nanmedian(sigma_weight):>20.5f}")
        print(f"{'block-mean-subtracted (real)':40s}{np.nanmedian(sigma_bm_re):>20.5f}")
        print(f"{'channel differences (real)':40s}{np.nanmedian(sigma_cd_re):>20.5f}")
        print(f"{'kNN (k={})'.format(knn_k):40s}{np.nanmedian(sigma_knn):>20.5f}")

    return results


def load_ms_dataset(
    yaml_path: str | Path,
    data_column: str = "DATA",
    spw_id: int = 0,
    channel_stride: int = 1,
) -> tuple:
    """
    Load a CASA measurement set using paths and geometry from a params YAML.

    Everything is derived automatically:
    - ``ms_path``        → reads the MS
    - ``B_max``          → dpix_rad (Nyquist pixel size)
    - ``observation.fov`` → npix
    - ``observation.vlsr`` + molecular table → f_rest (rest-frame frequencies for the renderer)

    ``vis.freq`` (LSRK) is used by ``image_to_vis`` for baseline lengths.
    ``f_rest`` is used by the renderer to map channels to disk-frame velocities.

    Parameters
    ----------
    yaml_path : str or Path
    data_column : str
        MS column to read ('DATA', 'CORRECTED_DATA', 'MODEL_DATA').
    spw_id : int
        Spectral window index.
    channel_stride : int
        Keep every Nth channel (1 = all channels).

    Returns
    -------
    vis      : VisibilityData  — LSRK visibilities (use vis.freq for NUFFT)
    dpix_rad : float           — Nyquist pixel size [rad]
    npix     : int             — pixels per side
    x_sky    : (npix, npix) float64  — RA offset grid [arcsec], East at col=0
    y_sky    : (npix, npix) float64  — Dec offset grid [arcsec], South at row=0
    f_rest   : (nchan,) float64      — rest-frame channel frequencies [Hz] for the renderer
    """
    import yaml as _yaml
    from .chemistry import chemistry_from_yaml_path, load_molecular_tables

    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = _yaml.safe_load(f)

    ms_path    = Path(cfg["ms_path"])
    if not ms_path.is_absolute():
        ms_path = yaml_path.parent / ms_path
    fov_arcsec = float(cfg["observation"]["fov"])
    vlsr       = float(cfg["observation"]["vlsr"])

    print(f"Reading MS: {ms_path.name}")
    vis = read_ms(str(ms_path), data_column=data_column, spw_id=spw_id)

    if channel_stride > 1:
        idx = np.arange(0, vis.nchan, channel_stride)
        vis = VisibilityData(
            uvw    = vis.uvw,
            data   = vis.data[:, idx],
            weight = vis.weight[:, idx],
            freq   = vis.freq[idx],
            flag   = vis.flag[:, idx],
        )

    lam = CC / vis.freq
    uvd = np.sqrt((vis.uvw[:, 0:1] / lam) ** 2 + (vis.uvw[:, 1:2] / lam) ** 2)
    B_max = uvd.max()
    B_min = uvd[uvd > 0].min()

    dpix_rad    = 1.0 / (2.0 * B_max)
    dpix_arcsec = np.degrees(dpix_rad) * 3600
    npix        = int(np.ceil(fov_arcsec / dpix_arcsec))

    half  = fov_arcsec / 2.0
    xax   = np.linspace( half, -half, npix)
    yax   = np.linspace(-half,  half, npix)
    x_sky, y_sky = np.meshgrid(xax, yax, indexing="xy")

    # Rest-frame frequencies: LSRK + nu0*vlsr/c
    # nu0 is read from the molecular table specified in the YAML chemistry section.
    chem   = chemistry_from_yaml_path(yaml_path)
    mol    = load_molecular_tables(chem)
    nu0    = float(mol.nu0)
    f_rest = vis.freq_rest(vlsr, nu0)

    beam_mas     = np.degrees(1.0 / B_max) * 3600 * 1e3
    max_scale_as = np.degrees(1.0 / B_min) * 3600
    print(f"  {vis.nrows} rows × {vis.nchan} channels")
    print(f"  dpix = {dpix_arcsec * 1e3:.1f} mas  (Nyquist from B_max = {B_max / 1e3:.0f} kλ)")
    print(f"  beam ≈ {beam_mas:.0f} mas  |  max detectable scale = {max_scale_as:.1f}\"")
    print(f"  fov = {fov_arcsec:.1f}\"  →  npix = {npix}")
    print(f"  nu0 = {nu0 / 1e9:.4f} GHz  |  vlsr = {vlsr:.0f} m/s  →  f_rest shift = {nu0 * vlsr / CC / 1e6:.2f} MHz")

    return vis, dpix_rad, npix, x_sky, y_sky, f_rest


__all__ = [
    "VisibilityData", "read_ms", "save_vis", "load_vis", "load_ms_dataset",
    "knn_sigma", "sigma_from_weight", "sigma_from_channel_diff",
    "sigma_from_block_mean", "compare_sigma_estimates",
]
