import jax
import jax.numpy as jnp
import numpy as np
import grid
from consts import *

def einstein_coefficients(energy_levels, radiative_transitions, transition=3):
    """
    a_ud is the Einstein coefficient for spontaneous emission from level u to level d.
    b_ud and b_du are the Einstein-B-coefficients

    Parameters
    ----------
    energy_levels: np.array
        shape=(num_levels, 4);  columns = LEVEL   ENERGIES(cm^-1)   WEIGHT   J
    radiative_transitions: np.array
        shape=(num_levels-1, 6); colums = TRANS   UP   LOW  EINSTEINA(s^-1)  FREQ(GHz)       E_u(K)
    transition: int,
        default=3, for CO(3-2) at ~345 GHz
        
    Notes
    -----
    Einstein coeffiecients for spontaneous emission from level d to level u
    gratio = weight_UP / weight_LOW = 7/5 for 345 GHz
    a_ud = 2.497e-06 for 345.7959899 GHz
    More info: https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/lineradtrans.html
    """
    # extract up and down indices from radiative transition table (e.g. 4,3 for second CO line at 345GHz)
    line_index = transition - 1
    nu0 = radiative_transitions[line_index,4] * ghz
    up_idx, dn_idx = jnp.array(radiative_transitions[line_index,1]-1, int), jnp.array(radiative_transitions[line_index,2]-1, int)
    gratio = energy_levels[up_idx,2] / energy_levels[dn_idx,2]
    a_ud = radiative_transitions[line_index, 3]
    b_ud = (cc**2/(2*hh*nu0**3)) * a_ud
    b_du = b_ud * gratio
    return nu0, a_ud, b_ud, b_du

def n_up_down(gas_nd, gas_t, energy_levels, radiative_transitions, transition=3):
    """
    Parameters
    ----------
    gas_nd: np.array
        Number density
    gas_t: np.array
        Temperature
    energy_levels: np.array
        shape=(num_levels, 4);  columns = LEVEL   ENERGIES(cm^-1)   WEIGHT   J
    radiative_transitions: np.array
        shape=(num_levels-1, 6); colums = TRANS   UP   LOW  EINSTEINA(s^-1)  FREQ(GHz)       E_u(K)
    transition: int,
        default=3, for CO(3-2) at ~345 GHz
    """
    # Create a temperature point
    ntemp, temp0, temp1 = 1000, 0.1, 100000
    partition_temp = temp0 * (temp1/temp0)**(jnp.arange(ntemp)/(ntemp-1))
    
    # compute the partition function based on all available levels for this molecule.
    dendivk = (hh*cc/kk) * jnp.diff(energy_levels[:,1])
    dummy = jnp.exp(-dendivk[None]/partition_temp[:,None]) * (energy_levels[1:,2] / energy_levels[:-1,2])
    partition_fn = energy_levels[0,2] + jnp.sum(jnp.cumprod(dummy, axis=-1), axis=-1)
    pfunc = jnp.interp(gas_t, partition_temp, partition_fn)
    
    # Construct the partition function by linear interpolation in the table
    line_index = transition - 1
    up_idx, dn_idx = jnp.array(radiative_transitions[line_index,1]-1, int), jnp.array(radiative_transitions[line_index,2]-1, int)
    dendivk_up = (hh*cc/kk) * (energy_levels[up_idx,1] - energy_levels[0,1])
    dendivk_dn = (hh*cc/kk) * (energy_levels[dn_idx,1] - energy_levels[0,1])
    levelweight_up = energy_levels[up_idx,2]
    levelweight_dn = energy_levels[dn_idx,2]
    
    n_up = (gas_nd / pfunc) * jnp.exp(-dendivk_up/gas_t) * levelweight_up
    n_dn = (gas_nd / pfunc) * jnp.exp(-dendivk_dn/gas_t) * levelweight_dn
    
    return n_up, n_dn

def compute_spectral_cube(camera_freqs, gas_v, alpha_tot, n_up, n_dn, a_ud, b_ud, b_du, ray_coords, obs_dir, nu0, pixel_area):
    """
    Compute radiative tranfer for gaussian emission lines. 
    Output units for image fluxes are Jy/pixel.
    """
    # Compute doppler shift
    # Note: doppler positive means moving toward observer, hence the minus sign
    doppler = (1.0/cc) * jnp.sum(obs_dir * gas_v, axis=-1)

    # Define a vector line profile over multiple camera frequencies 
    # The line profile is a Gaussian with width (alpha) and shifted by doppler
    dnu = grid.expand_dims(camera_freqs, alpha_tot.ndim+1, axis=-1) - nu0 - doppler*nu0
    line_profile = (cc/(alpha_tot*nu0*jnp.sqrt(jnp.pi))) * jnp.exp(-(cc*dnu/(nu0*alpha_tot))**2)
    
    # Compute emissivity (j_nu) and extinction (alpha_nu) for radiative transfer
    #     h nu_0                                          h nu_0
    # j = ------ n_up A_ud * phi(omega, nu)   ;  alpha = ------ ( n_down B_du - n_up B_ud ) * phi(omega, nu)
    #     4 pi                                            4 pi
    const = hh*nu0 / (4*np.pi)
    j_nu  = const * n_up * a_ud  * line_profile
    a_nu  = const * (n_dn * b_du - n_up * b_ud) * line_profile

    # Ray trace through the volume to compute image intensities
    ray_ds = jnp.sqrt(jnp.sum(jnp.diff(ray_coords, axis=-2)**2, axis=-1))
    dtau = 0.5 * (a_nu[...,1:] + a_nu[...,:-1]) * ray_ds

    # First order interpolation of the source
    source_1st = 0.5 * (j_nu[...,1:] + j_nu[...,:-1]) * ray_ds
    
    # Second order integration of the source
    # Notes: https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/imagesspectra.html#sec-second-order
    s_nu = j_nu / (a_nu + 1e-30)   # Radmc3d has +1e-99 but this results in nans
    beta = (dtau - 1 + jnp.exp(-dtau)) / (dtau + 1e-30)
    beta = jnp.where(dtau > 1e-6, beta, 0.5*dtau)
    source_2nd = (1 - jnp.exp(-dtau) - beta) * s_nu[...,:-1] + beta * s_nu[...,1:]
    source_2nd = jnp.where(source_2nd < source_1st, source_2nd, source_1st)

    pad_width = [(0,0)]*(dtau.ndim-1) + [(1,0)]
    attenuation = jnp.exp(-jnp.cumsum(jnp.pad(dtau, pad_width), axis=-1))[...,:-1]
    intensity = (source_2nd * attenuation).sum(axis=-1)

    # Conversion from erg/s/cm/cm/ster to Jy/pixel
    image_fluxes_jy =  pixel_area / pc**2 * 1e23 * intensity
    return image_fluxes_jy

# Define vector mapping for speedup on single gpu
compute_spectral_cube_vmap = jax.vmap(
    compute_spectral_cube,
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None)
)

# Define a parallel mapping over several gpus
compute_spectral_cube_pmap = jax.pmap(
    compute_spectral_cube, axis_name='freq', 
    in_axes=(0, None, None, None, None, None, None, None, None, None, None, None)
)

def alpha_total_co(v_turb, gas_t, m_mol_deprecated=None):
    """
    Compute the total line broadening due to thermal and turbulence velocities

    Parameters
    ----------
    v_turb: float
        turbulence broadening scaled by the speed of sound.
    gas_t: jnp.array
        Temperature field (K)
    m_mol: float,
        Molecular weight
    """
    alpha_therm_sq = 2*kk*gas_t / m_co
    cs_sq = 2*kk*gas_t / m_mol  # local speed of sound
    alpha_tot = jnp.sqrt(alpha_therm_sq + v_turb**2 * cs_sq)
    return alpha_tot
    
def load_molecular_tables(path):
    """
    Load molecular information
    
    path: str,
        path to molecule_12c16o.inp
   
    See e.g. tables below from molecule_12c16o.inp
    More information: https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/lineradtrans.html
    ------------------------------------
    LEVEL   ENERGIES(cm^-1)   WEIGHT   J
      3     11.534919938	    5.0	     2
      4     23.069512649	    7.0	     3
    ---------------------------------------------------------
    TRANS   UP   LOW  EINSTEINA(s^-1)  FREQ(GHz)       E_u(K)
      3     4     3   2.497e-06        345.7959899     33.19
     """
    energy_levels = np.loadtxt(path, skiprows=7, max_rows=41)
    radiative_transitions = np.loadtxt(path, skiprows=51, max_rows=40)
    return energy_levels, radiative_transitions