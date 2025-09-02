import os, yaml
import numpy as np
from astropy.io import fits

def dump_yaml(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"✔️ Saved YAML → {path}")

# Helper: pretty preview (first ~80 lines)
def preview_yaml(data, max_lines=80):
    s = yaml.safe_dump(data, sort_keys=False)
    print("\n".join(s.splitlines()[:max_lines]))

def yaml_safe(obj):
    """Recursively convert numpy/jax scalars and arrays to built-in types."""
    if isinstance(obj, dict):
        return {k: yaml_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [yaml_safe(v) for v in obj]
    elif isinstance(obj, (np.generic,)):  # numpy scalar
        if np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        else:
            return float(obj)
    elif hasattr(obj, "item") and callable(obj.item):
        # catches JAX DeviceArray scalars too
        try:
            return obj.item()
        except Exception:
            return obj
    return obj

def save_synthetic_observation(
    filepath,
    cube_obs,
    freqs,
    nu0=None,
    sigma=None,
    noise_type=None,
    seed=None,
    extra_hdr=None,
):
    """
    Save a synthetic spectral cube and frequency axis to a FITS file with metadata.

    Parameters
    ----------
    filepath : str
        Path to output FITS file.
    cube_obs : ndarray
        Observed data cube, shape (nfreq, ny, nx).
    freqs : ndarray
        Frequency axis (Hz), shape (nfreq,).
    nu0 : float, optional
        Line rest frequency (Hz).
    sigma : float, optional
        Noise standard deviation used to generate the cube.
    noise_type : str, optional
        Noise generator identifier, e.g. "jax.random.normal".
    seed : int, optional
        Random seed used for noise generation.
    extra_hdr : dict, optional
        Extra header entries to include.

    Returns
    -------
    str
        Path to saved FITS file.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

    hdr = fits.Header()
    hdr["NFREQ"] = cube_obs.shape[0]
    hdr["NPIX"]  = cube_obs.shape[1]
    if sigma is not None:
        hdr["SIGMA"] = float(sigma)
    if nu0 is not None:
        hdr["NU0_HZ"] = float(nu0)
    if noise_type is not None:
        hdr["NOISEGEN"] = str(noise_type)
    if seed is not None:
        hdr["SEED"] = int(seed)
    if extra_hdr is not None:
        for k, v in extra_hdr.items():
            hdr[str(k).upper()] = v

    hdu0 = fits.PrimaryHDU(np.asarray(cube_obs), header=hdr)
    hdu1 = fits.ImageHDU(np.asarray(freqs), name="FREQS")
    fits.HDUList([hdu0, hdu1]).writeto(filepath, overwrite=True)

    print(f"✔️ Saved FITS → {filepath}")


def load_synthetic_observation(filepath):
    """
    Load a synthetic observation saved by `save_synthetic_observation`.

    Parameters
    ----------
    filepath : str
        Path to the FITS file.

    Returns
    -------
    cube : np.ndarray
        Data cube with shape (nfreq, ny, nx).
    freqs : np.ndarray
        Frequency axis (Hz) with shape (nfreq,).
    meta : dict
        Dictionary with parsed metadata:
          - "nfreq" : int
          - "npix"  : int
          - "sigma" : float or None
          - "nu0": float or None
          - "noisegen": str or None
          - "seed"  : int or None
          - "header": astropy.io.fits.Header (full primary header)
    """
    with fits.open(filepath) as hdul:
        # Primary HDU → cube
        cube = np.asarray(hdul[0].data)
        hdr  = hdul[0].header.copy()

        # Frequency extension
        if "FREQS" in hdul:
            freqs = np.asarray(hdul["FREQS"].data)
        else:
            # fallback: first image extension if present
            if len(hdul) > 1 and isinstance(hdul[1], fits.ImageHDU):
                freqs = np.asarray(hdul[1].data)
            else:
                raise KeyError("FREQS extension not found in FITS file.")

    # Build meta (use header if available, otherwise infer from data)
    meta = {
        "nfreq": int(hdr.get("NFREQ", cube.shape[0] if cube is not None else len(freqs))),
        "npix":  int(hdr.get("NPIX", cube.shape[-1] if cube is not None else 0)),
        "sigma": (float(hdr["SIGMA"]) if "SIGMA" in hdr else None),
        "nu0_hz": (float(hdr["NU0"]) if "NU0" in hdr else None),
        "noisegen": hdr.get("NOISEGEN", None),
        "seed": (int(hdr["SEED"]) if "SEED" in hdr else None),
        "header": hdr,
    }

    return cube, freqs, meta 