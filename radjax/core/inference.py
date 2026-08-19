from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
import yaml

from ..core.utils import yaml_safe
if TYPE_CHECKING:
    from .casa_io import VisibilityData

Array = jnp.ndarray
@struct.dataclass
class SamplerState:
    obs_params: Any
    chem_params: Any
    disk_params: Any
    base_disk: Any
    rays: Any
    mol: Any
    beam: Any     
    sigma: float = 1.0  # sigma can be scalar or an array matching the data shape
    adapter: Any = None 
    use_pressure_correction: bool = True


def append_inference_to_yaml(filepath, mcmc_params, results=None):
    """
    Append MCMC inference parameters and (optionally) results to an existing YAML file.

    Parameters
    ----------
    filepath : str
        Path to an existing YAML file created from disk/chemistry/observation params.
    mcmc_params : dict
        Dictionary of inference parameters.
        Example::

            {
                "method": "emcee",
                "numpy_seed": 123,
                "nwalkers": 64,
                "nburn": 2_000,
                "nsteps": 10_000,
                "sampled_params": ["v_turb"],
                "bounds": {"v_turb": [1e-3, 1e-1]}
            }

    results : dict, optional
        Dictionary of results to store.
        Example::

            {
                "v_turb_true": 0.01,
                "v_turb_map": 0.012,
                "v_turb_med": 0.011,
                "v_turb_ci68": [0.009, 0.013],
                "rms_map": 3.4e-3
            }

    Returns
    -------
    dict
        The merged YAML dictionary (with safe types).
    """
    # Load existing YAML
    try:
        with open(filepath, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        config = {}

    # Sanitize values
    mcmc_params = yaml_safe(mcmc_params)
    results = yaml_safe(results) if results is not None else None

    # Insert inference section
    config["inference"] = {"mcmc": mcmc_params}
    if results is not None:
        config["inference"]["results"] = results

    # Save back
    with open(filepath, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


# =============================================================================
# Visibility-domain likelihood
#
# Model- and dataset-independent: operates on a VisibilityData + model
# visibility array, so any MCMC script (any disk model, any MS) can import
# these directly rather than re-deriving the same likelihood.
#
# Derivation
# ----------
# For MeasurementSet row i and channel k, let
#
#     r_{i,k} = V_obs_{i,k} - V_model_{i,k}
#
# (Stokes I is already averaged over polarization in `casa_io.read_ms`, so
# there is no separate correlation index here -- CASA's per-polarization
# WEIGHT/SIGMA columns are already combined into `VisibilityData.weight`.)
#
# Assume circularly-symmetric Gaussian thermal noise, independent between
# real and imaginary parts, with the SAME standard deviation sigma_i for both
# (Convention A -- see below):
#
#     Re(r_{i,k}) ~ N(0, sigma_i^2),   Im(r_{i,k}) ~ N(0, sigma_i^2)
#
# The likelihood of one complex visibility is the product of two independent
# 1D Gaussians:
#
#     p(V_obs_{i,k} | theta) = 1/(2*pi*sigma_i^2) * exp[-(Re(r)^2 + Im(r)^2) / (2*sigma_i^2)]
#
# Summing log-probabilities over all unflagged (row, channel) pairs:
#
#     logL(theta) = -1/2 * sum_{i,k} [ |r_{i,k}|^2 / sigma_i^2  +  2*log(2*pi*sigma_i^2) ]
#
# where |r|^2 = Re(r)^2 + Im(r)^2. sigma (and therefore the normalization
# term 2*log(2*pi*sigma^2)) does not depend on theta, so it is a constant
# offset that does not affect the MCMC posterior shape and can be dropped:
#
#     logL(theta) = -1/2 * sum_{i,k} |r_{i,k}|^2 / sigma_i^2  + const
#                 = -1/2 * sum_{i,k} weight_i * |r_{i,k}|^2      (weight = 1/sigma^2)
#
# which is exactly `-0.5 * chi2_vis(obs, vis_model)`.
#
# Convention A vs. B -- do not guess this, verify it (see check_sigma_convention)
# --------------------------------------------------------------------------
# The formula above assumes CASA's WEIGHT/SIGMA is the standard deviation of
# EACH Cartesian component (Convention A). Some pipelines instead define
# sigma as the standard deviation of the complex modulus itself (Convention
# B), for which sigma_complex^2 = E[|n|^2] = 2*sigma_component^2, i.e.
# sigma_component = sigma_complex / sqrt(2), which would change the formula
# to logL = -sum |r|^2 / sigma_complex^2 (no factor of 1/2 sigma^2, since the
# missing sqrt(2) is folded into sigma_complex). Getting this wrong silently
# rescales chi^2 by a factor of 2. `check_sigma_convention` tests this
# directly on real residuals rather than assuming: for HD163296 MAPS CO(2-1),
# this was verified empirically (`casa_io.compare_sigma_estimates`: WEIGHT
# column, channel-difference, and block-mean sigma estimators all agree, and
# sigma_real == sigma_imag to 4 decimal places), confirming Convention A.
# =============================================================================


def visibility_log_likelihood(obs: VisibilityData, vis_model: np.ndarray) -> float:
    """
    Complex Gaussian visibility log-likelihood (Convention A, see module docstring above).

    logL(theta) = -1/2 * sum_{i,k} weight_{i,k} * |V_obs_{i,k} - V_model_{i,k}|^2

    Flagged samples have weight=0 in `obs.weight` and are excluded automatically.
    The additive normalization constant (independent of theta) is dropped, so
    this is only valid for comparing/sampling theta at fixed sigma -- not for
    comparing likelihoods across different sigma choices or different datasets.

    Parameters
    ----------
    obs : VisibilityData
        Observed visibilities; `obs.weight` must be 1/sigma^2 per real/imaginary
        component (Convention A -- verify with `check_sigma_convention` if unsure).
    vis_model : (nrows, nchan) complex
        Model visibilities, e.g. from `vis_forward_model.image_to_vis`.

    Returns
    -------
    float
        logL(theta), up to an additive theta-independent constant.

    Example
    -------
    >>> def log_prob(theta):
    ...     lp = log_prior(theta)
    ...     if not np.isfinite(lp):
    ...         return -np.inf
    ...     vis_model = image_to_vis(render(theta), vis.uvw, vis.freq, dpix_rad)
    ...     return lp + visibility_log_likelihood(vis, vis_model)
    """
    from .vis_forward_model import chi2_vis  # optional dep: pip install radjax[vis]
    return -0.5 * chi2_vis(obs, vis_model)


def reduced_chi2_vis(obs: VisibilityData, vis_model: np.ndarray, n_params: int) -> float:
    """
    Reduced chi-square goodness-of-fit diagnostic for a visibility-domain fit.

    chi^2 = sum_{i,k} weight_{i,k} * |V_obs_{i,k} - V_model_{i,k}|^2  (= chi2_vis(obs, vis_model))

    Each complex visibility contributes 2 real degrees of freedom (Re and Im),
    so for N_complex unflagged (row, channel) pairs and n_params fitted
    parameters, the approximate degrees of freedom are

        nu = 2 * N_complex - n_params

    and chi^2_nu = chi^2 / nu. At a good fit, with correctly-scaled sigma and
    approximately independent residuals, chi^2_nu ~= 1. Substantially above 1
    suggests sigma is too small (or the model/calibration is inadequate);
    substantially below 1 suggests sigma is too large.

    Parameters
    ----------
    obs : VisibilityData
    vis_model : (nrows, nchan) complex
    n_params : int
        Number of fitted (theta) parameters.

    Returns
    -------
    float
        chi^2_nu
    """
    from .vis_forward_model import chi2_vis  # optional dep: pip install radjax[vis]
    chi2 = chi2_vis(obs, vis_model)
    n_complex = int(np.sum(obs.weight > 0))
    nu = 2 * n_complex - n_params
    return chi2 / nu


def check_sigma_convention(obs: VisibilityData, vis_model: np.ndarray) -> dict:
    """
    Empirically check whether `obs.weight` follows Convention A (sigma is the
    standard deviation of each real/imaginary component -- what
    `visibility_log_likelihood` assumes) rather than Convention B (sigma of
    the complex modulus). See the module-level docstring above for the
    distinction; do not assume which one applies without checking.

    Computes the standardized residuals

        z_real = Re(V_obs - V_model) / sigma,   z_imag = Im(V_obs - V_model) / sigma

    (sigma = 1/sqrt(weight)) over all unflagged samples. Under Convention A,
    at a reasonable model fit, both should have mean ~= 0 and std ~= 1. If the
    stds instead cluster near sqrt(2) ~= 1.41, `obs.weight` is more likely
    Convention B and should be rescaled by a factor of 2 before use in
    `visibility_log_likelihood` / `chi2_vis`.

    Parameters
    ----------
    obs : VisibilityData
    vis_model : (nrows, nchan) complex
        Model visibilities at a reasonable-fit theta (residuals should be
        close to pure noise -- checking this at a poor fit will show inflated
        std from model mismatch, not from a sigma-convention error).

    Returns
    -------
    dict with keys "z_real_mean", "z_real_std", "z_imag_mean", "z_imag_std"
    """
    valid = obs.weight > 0
    sigma = 1.0 / np.sqrt(obs.weight[valid])
    residual = obs.data[valid] - vis_model[valid]
    z_real = residual.real / sigma
    z_imag = residual.imag / sigma
    return {
        "z_real_mean": float(np.mean(z_real)),
        "z_real_std": float(np.std(z_real, ddof=1)),
        "z_imag_mean": float(np.mean(z_imag)),
        "z_imag_std": float(np.std(z_imag, ddof=1)),
    }


