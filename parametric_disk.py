from collections import namedtuple
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from consts import *

Disk = namedtuple('Disk', 
                  ['T_mid1', 'T_atm1', 'q', 'M_star', 'gamma', 'r_in', 
                   'r_c', 'M_gas', 'm_mol', 'alpha_turb', 'co_abundance', 
                   'N_dissoc', 'freezeout', 'molecule_table'])

def temperature_profile(z, r, T_mid1, T_atm1, q, M_star, m_mol):
    # Temperature at the mid-plane is a power law with 2 parameters
    # Atmospheric tempertature is also a radial power law with 1 additional parameter
    # Typical values, T_mid1=200K, q=0.55, for Taurus protostars similar to those in the CO survey (Andrews et al. 2009)
    # Value range for T_atm1=500-1500K
    T_mid = T_mid1 * (r)**(-q)
    T_atm = T_atm1 * (r)**(-q)
    
    # The temperature profile is a combination of T_mid, T_atm 
    # delta, z_q describe the steepness of the profile and the hight at which 
    # the disk transitions to the atmopheric value. H_p is the pressure scale hight
    # derived from the midplane tempertaure
    H_p = (r * au)**(3/2) * (kk*T_mid / (G*M_star*m_mol))**0.5
    delta, z_q = 2, 4*H_p/au
    temperature = jnp.where(jnp.abs(z)<z_q, T_mid + (T_atm - T_mid)*jnp.sin(jnp.pi*z/(2*z_q))**(2*delta), T_atm)
    return temperature

def number_density_profile(z, r, temperature, gamma, r_in, r_c, M_gas, M_star, m_mol):
    sigma_0 = (2-gamma) * (M_gas / (2*jnp.pi*(r_c*au)**2)) * jnp.exp(r_in / r_c)**(2-gamma)
    sigma = sigma_0 * (r[0] / r_c)**(-gamma) * jnp.exp(-(r[0]/r_c)**(2-gamma))
    dz = jnp.diff(z * au, axis=0)
    dlogrho = ( au**(-2) * G*M_star*z/(r**2 + z**2)**(3/2) ) * (m_mol / (kk * temperature))
    dlogrho = -0.5*(dlogrho[:-1] + dlogrho[1:]) - jnp.log(temperature[1:]/temperature[:-1])/dz
    logrho = jnp.cumsum(dlogrho*dz, axis=0)
    logrho = jnp.pad(logrho, pad_width=[(1,0), (0,0)])
    rho = jnp.exp(logrho)
    rho = sigma * rho / (1+jnp.sum(rho[1:]*dz, axis=0, keepdims=True))
    numberdens = rho/m_mol
    return numberdens
    
def density_temperature_profiles(disk, z_min, z_max, r_min, r_max, resolution):
    
    z, r = jnp.meshgrid(jnp.linspace(z_min, z_max, resolution), jnp.linspace(r_min, r_max, resolution), indexing='ij')
    
    temperature = temperature_profile(z, r, disk.T_mid1, disk.T_atm1, disk.q, disk.M_star, disk.m_mol)
    numberdens_h2 = number_density_profile(z, r, temperature, disk.gamma, disk.r_in, disk.r_c, disk.M_gas, disk.M_star, disk.m_mol)

    # Compute the column integrated density N(h2) (units of 1/cm^2)
    dz = jnp.diff(z * au, axis=0)[0,0]
    N_h2 = jnp.cumsum(numberdens_h2[::-1]*dz, axis=0)[::-1]

    # CO chemestry, compute the relative abundance of co 
    # Below T=20K there is CO depletion via freeze-out onto dust grains 
    # At the upper layers of the disk, N_h2>N_dissoc, there is CO photodissociation due to radiation
    abundance_co = jnp.where(jnp.bitwise_and(temperature>disk.freezeout, N_h2>disk.N_dissoc), disk.co_abundance, 0.0)
    numberdens_co = abundance_co * numberdens_h2
    return numberdens_co, temperature

def keplerian_velocity(coords, M_star):
    ray_x, ray_y, ray_z = coords[...,0], coords[...,1], coords[...,2]
    ray_r = jnp.sqrt(ray_x**2 + ray_y**2)
    omega = jnp.sqrt(G * M_star / jnp.sqrt(ray_r**2 + ray_z**2))
    cosp = ray_x/ray_r
    sinp = ray_y/ray_r
    vx, vy, vz = omega*cosp, omega*sinp, jnp.zeros_like(omega)
    velocity = jnp.stack([vx, vy, vz], axis=-1)
    return velocity

