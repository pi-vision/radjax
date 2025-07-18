import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from consts import *  # Replace with actual imports as needed

###############################################################################
# Replacing DiskFlaherty class, by define a parameter dictionary and 
# a set of functions that operate on these parameters.
###############################################################################

def create_disk_params(
    T_mid1, T_atm1, q, q_in, r_break, M_star, gamma, r_in, log_r_c, M_gas,
    v_turb, co_abundance, N_dissoc, N_desorp, z_q0, transition, m_mol,
    freezeout, delta, r_scale, v_sys = 0
):
    """
    Returns a dict containing the disk parameters formerly used in DiskFlaherty.
    """
    return {
        'T_mid1': T_mid1,
        'T_atm1': T_atm1,
        'q': q,
        'q_in': q_in,
        'r_break': r_break,
        'M_star': M_star,
        'gamma': gamma,
        'r_in': r_in,
        'log_r_c': log_r_c,
        'M_gas': M_gas,
        'v_turb': v_turb,
        'co_abundance': co_abundance,
        'N_dissoc': N_dissoc,
        'N_desorp': N_desorp,
        'z_q0': z_q0,
        'transition': transition,
        'm_mol': m_mol,
        'freezeout': freezeout,
        'delta': delta,
        'r_scale': r_scale,
        'v_sys': v_sys
    }

def temperature_profile(z, r, params):
    """
    Computes the temperature profile using parameters in the `params` dict.
    """
    # Choose exponent q depending on whether r <= r_break
    q = jnp.where(r <= params['r_break'], params['q_in'], params['q'])
    T_mid = params['T_mid1'] * (r / params['r_scale']) ** q
    T_atm = params['T_atm1'] * (r / params['r_scale']) ** q
    
    # The parameter z_q is the height at which the gas temperature is maximum
    z_q = params['z_q0'] * (r / params['r_scale'])**1.3

    # Blend T_mid and T_atm with a cosine function
    temperature = jnp.where(
        z < z_q,
        T_atm + (T_mid - T_atm) * jnp.cos(jnp.pi * z / (2 * z_q))**(2*params['delta']),
        T_atm
    )
    return temperature

def co_abundance_profile(N_h2, temperature, params):
    """
    Computes the CO abundance profile given N(h2), temperature, and disk params.
    """
    # Regions of CO depletion or reintroduction
    co_dissoc = jnp.bitwise_and(
        temperature > params['freezeout'],
        0.706 * N_h2 > params['N_dissoc']
    )
    co_desorp = jnp.bitwise_and(
        temperature <= params['freezeout'],
        jnp.bitwise_and(0.706 * N_h2 > params['N_dissoc'], 0.706 * N_h2 < params['N_desorp'])
    )

    co_region = jnp.bitwise_or(co_dissoc, co_desorp)
    abundance = jnp.where(co_region, params['co_abundance'], 0.0)
    return abundance

def velocity_profile(z, r, nd, temperature, params, pressure_correction=False):
    """
    Computes the (azimuthal) Keplerian velocity, ignoring pressure corrections.
    """
    if pressure_correction:
        v = jnp.sqrt(vsq_keplerian_vertical(z, r, params['M_star']) + vsq_pressure_grad(r, nd, temperature, params['m_mol']))
        # If there is a large negative pressure gradient this sqrt might produce nans
        v = jnp.nan_to_num(v)
    else:
        v = jnp.sqrt(vsq_keplerian_vertical(z, r, params['M_star']))
    return v

###############################################################################
# Utility functions for the disk model (mostly 
###############################################################################

def number_density_profile(z, r, temperature, gamma, r_in, r_c, M_gas, M_star, m_mol):
    """
    Compute the H2 number density as a function of z, r, and disk params.
    This is the same logic from your original code, but now as a function.
    """
    sigma_0 = (2 - gamma) * (
        M_gas / (2 * jnp.pi * (r_c * au) ** 2)
    ) * jnp.exp(r_in / r_c) ** (2 - gamma)
    sigma = sigma_0 * (r[0] / r_c) ** (-gamma) * jnp.exp(-(r[0] / r_c) ** (2 - gamma))

    dz = jnp.diff(z * au, axis=0)
    dlogrho = (
        au**(-2) * G * M_star * z / (r**2 + z**2) ** (3 / 2)
    ) * (m_mol / (kk * temperature))
    dlogrho = (
        -0.5 * (dlogrho[:-1] + dlogrho[1:])
        - jnp.log(temperature[1:] / temperature[:-1]) / dz
    )

    logrho = jnp.cumsum(dlogrho * dz, axis=0)
    logrho = jnp.pad(logrho, pad_width=[(1, 0), (0, 0)])
    rho = jnp.exp(logrho)
    rho = sigma * rho / (1 + jnp.sum(rho[1:] * dz, axis=0, keepdims=True))
    numberdens = rho / m_mol
    return numberdens

def surface_density(z, nd):
    """
    Compute the column-integrated density N(h2) [1/cm^2].
    """
    dz = jnp.diff(z * au, axis=0)[0, 0]  # watch out for shape handling
    N = jnp.cumsum(nd[::-1] * dz, axis=0)[::-1]
    return N

def vsq_keplerian_vertical(z, r, M):
    """
    Modified Keplerian velocity to account for height above the midplane.
    """
    return G * M * (r * au) ** 2 / ((r * au) ** 2 + (z * au) ** 2) ** (3 / 2)

def vsq_pressure_grad(r, nd, temperature, m_mol):
    rho = m_mol * nd
    dr = jnp.diff(r * au, axis=1)[0, 0]
    pressure = nd * kk * temperature
    pressure_grad = (pressure[:, 1:] - pressure[:, :-1]) / dr
    pressure_grad = jnp.pad(pressure_grad, pad_width=[(0, 0), (1, 0)])
    return ((r * au) / rho) * pressure_grad

def azimuthal_velocity(coords, v_phi):
    """
    Convert the scalar (azimuthal) velocity into a 3D velocity field
    in the plane of the disk. 
    """
    ray_r = jnp.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2)
    cosp = coords[..., 1] / ray_r
    sinp = coords[..., 0] / ray_r
    vy = v_phi * cosp
    vx = v_phi * sinp
    vz = jnp.zeros_like(v_phi)
    return jnp.stack([vy, vx, vz], axis=-1)
