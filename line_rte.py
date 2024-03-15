import jax.numpy as jnp
import numpy as np
import grid
from consts import *

def einstein_coefficients(energy_levels, radiative_transitions, line_index=2):
    """
    a_ud is the Einstein coefficient for spontaneous emission from level u to level d.
    b_ud and b_du are the Einstein-B-coefficients

    Parameters
    ----------
    energy_levels: np.array
        shape=(num_levels, 4);  columns = LEVEL   ENERGIES(cm^-1)   WEIGHT   J
    radiative_transitions: np.array
        shape=(num_levels-1, 6); colums = TRANS   UP   LOW  EINSTEINA(s^-1)  FREQ(GHz)       E_u(K)
    line_index: int,
        default=2, for CO at ~345 GHz
        
    Notes
    -----
    Einstein coeffiecients for spontaneous emission from level d to level u
    gratio = weight_UP / weight_LOW = 7/5 for 345 GHz
    a_ud = 2.497e-06 for 345.7959899 GHz
    More info: https://www.ita.uni-heidelberg.de/~dullemond/software/radmc-3d/manual_radmc3d/lineradtrans.html
    """
    # extract up and down indices from radiative transition table (e.g. 4,3 for second CO line at 345GHz)
    nu0 = radiative_transitions[line_index,4] * ghz
    up_idx, dn_idx = int(radiative_transitions[line_index,1]-1), int(radiative_transitions[line_index,2]-1)
    gratio = energy_levels[up_idx,2] / energy_levels[dn_idx,2]
    a_ud = radiative_transitions[line_index, 3]
    b_ud = (cc**2/(2*hh*nu0**3)) * a_ud
    b_du = b_ud * gratio
    return nu0, a_ud, b_ud, b_du

def n_up_down(gas_nd, gas_t, energy_levels, radiative_transitions, line_index=2):
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
    line_index: int,
        default=2, for CO at ~345 GHz
    """
    
    # Create a temperature point
    ntemp, temp0, temp1 = 1000, 0.1, 100000
    partition_temp = temp0 * (temp1/temp0)**(np.arange(ntemp)/(ntemp-1))
    
    # compute the partition function based on all available levels for this molecule.
    dendivk = (hh*cc/kk) * np.diff(energy_levels[:,1])
    dummy = np.exp(-dendivk[None]/partition_temp[:,None]) * (energy_levels[1:,2] / energy_levels[:-1,2])
    partition_fn = energy_levels[0,2] + np.sum(np.cumproduct(dummy, axis=-1), axis=-1)
    
    # Construct the partition function by linear interpolation in the table
    dendivk1 = (hh*cc/kk) * (energy_levels[:,1] - energy_levels[0,1])
    levelweight = energy_levels[:,2]

    # Construct vector equation for multiple grid points
    dendivk1 = grid.expand_dims(dendivk1, gas_t.ndim+1, axis=-1)
    levelweight = grid.expand_dims(levelweight, gas_t.ndim+1, axis=-1)
    
    pfunc = jnp.interp(gas_t, partition_temp, partition_fn)
    levelpop = (gas_nd / pfunc) * np.exp(-dendivk1/gas_t) * levelweight
    up_idx, dn_idx = int(radiative_transitions[line_index,1]-1), int(radiative_transitions[line_index,2]-1)
    n_up, n_dn = levelpop[up_idx], levelpop[dn_idx]
    return n_up, n_dn


def compute_spectral_cube(camera_freqs, gas_v, gas_t, n_up, n_dn, a_ud, b_ud, b_du, ray_coords, obs_dir, nu0, pixel_area, alpha_turb=0.0):
    """
    Compute radiative tranfer for gaussian emission lines. 
    Output units for image fluxes are Jy.
    """
    # Compute spectral width
    alpha_therm = jnp.sqrt(2*kk*gas_t/m_mol)
    alpha_tot = jnp.sqrt(alpha_turb**2 + alpha_therm**2)
    
    # Compute doppler shift
    # Note: doppler positive means moving toward observer, hence the minus sign
    doppler = (1.0/cc) * jnp.sum(obs_dir * gas_v, axis=-1)

    # Define a vector line profile over multiple camera frequencies 
    # The line profile is a Gaussian with width (alpha) and shifted by doppler
    dnu = grid.expand_dims(camera_freqs, gas_t.ndim+1, axis=-1) - nu0 - doppler*nu0
    line_profile = (cc/(alpha_tot*nu0*jnp.sqrt(jnp.pi))) * jnp.exp(-(cc*dnu/(nu0*alpha_tot))**2)


    
    # Compute emissivity (j_nu) and extinction (alpha_nu) for radiative transfer
    #     h nu_0                                          h nu_0
    # j = ------ n_up A_ud * phi(omega, nu)   ;  alpha = ------ ( n_down B_du - n_up B_ud ) * phi(omega, nu)
    #     4 pi                                            4 pi
    const = hh*nu0 / (4*np.pi)
    j_nu  = const * n_up * a_ud  * line_profile
    a_nu  = const * (n_dn * b_du - n_up * b_ud) * line_profile

    # Ray trace through the volume to compute image intensities
    ray_ds = jnp.sqrt(jnp.sum(jnp.diff(ray_coords, axis=1)**2, axis=-1))
    dtau = 0.5 * (a_nu[...,1:] + a_nu[...,:-1]) * ray_ds

    beta = (dtau - 1 + jnp.exp(-dtau)) / dtau
    beta = jnp.where(dtau > 1e-6, beta, 0.5*dtau)
    
    # First order interpolation of the source
    source_1st = 0.5 * (j_nu[...,1:] + j_nu[...,:-1]) * ray_ds
    
    # Second order interpolation of the source
    s_nu = j_nu / (a_nu + 1e-99)
    source_2nd = (1 - jnp.exp(-dtau) - beta) * s_nu[...,:-1] + beta * s_nu[...,1:]
    source_2nd = jnp.where(source_2nd < source_1st, source_2nd, source_1st)
    intensity = (source_1st * jnp.exp(-jnp.cumsum(dtau, axis=-1))).sum(axis=-1)

    # Conversion from erg/s/cm/cm/ster to Jy/pixel
    image_fluxes_jy =  pixel_area / pc**2 * 1e23 * intensity
    return image_fluxes_jy


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

def compute_partition_function(molecule_file):
    return