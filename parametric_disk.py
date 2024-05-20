import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from consts import *

class DiskFlaherty(object):
    """
    A parameteric disk model based on Flaherty et al. (2015)
    
    Parameters
    ----------
    molecule_table: path to the molecular table (e.g. molecule_12c16o.inp)
    transition: rotational transition index (e.g. 3 is CO(3-2) at ~345 GHz)
    freezeout: Kelvin. Freezeout temperature of the molecule (e.g. 20K for CO)
    delta: describe the steepness of the profile (multiplies the power of the sinusoid)
    r_scale: scaling in AU (default = 150AU)

    Notes
    -----
    https://iopscience.iop.org/article/10.1088/0004-637X/813/2/99
    """
    def __init__(self, name: str, T_mid1: float, T_atm1: float, q: float, M_star: float, gamma: float, r_in: float, 
                 r_c: float, M_gas: float, v_turb: float, co_abundance: float, N_dissoc: float, z_q0: float, 
                 molecule_table: str, transition: int, m_mol: float = 2.37*m_h, freezeout: float = 20.0, delta: float = 1, r_scale: float = 150):
        self.name = name
        self.T_mid1 = T_mid1
        self.T_atm1 = T_atm1
        self.q = q
        self.M_star = M_star
        self.gamma = gamma
        self.r_in = r_in
        self.r_c = r_c
        self.M_gas = M_gas
        self.v_turb = v_turb
        self.co_abundance = co_abundance
        self.N_dissoc = N_dissoc
        self.z_q0 = z_q0
        self.molecule_table = molecule_table
        self.transition = transition
        self.m_mol = m_mol
        self.freezeout = freezeout
        self.delta = delta
        self.r_scale = r_scale

    def temperature_profile(self, z, r):
        T_mid = self.T_mid1 * (r/self.r_scale)**(self.q)
        T_atm = self.T_atm1 * (r/self.r_scale)**(self.q)
    
        # The parameter Zq is the height above the midplane at which the gas temperature reaches its maximum value
        z_q = self.z_q0 * (r/self.r_scale)**1.3
        
        # The temperature profile is a combination of T_mid, T_atm 
        temperature = jnp.where(z<z_q, T_atm + (T_mid-T_atm)*jnp.cos(jnp.pi*z/(2*z_q))**(2*self.delta), T_atm)
        return temperature

    def co_abundance_profile(self, N_h2, temperature):
        co_region = jnp.bitwise_and(temperature>self.freezeout, 0.706*N_h2>self.N_dissoc)
        abundance = jnp.where(co_region, self.co_abundance, 0.0)
        return abundance 

    def velocity(self, z, r, nd, temperature):
        """Modified Keplerian velocity to account for height and gas pressure above the midplane"""
        v = jnp.sqrt( vsq_keplerian_vertical(z, r, self.M_star) + vsq_pressure_grad(r, nd, temperature, self.m_mol) )
        return v


class DiskWilliams(object):
    """
    A parameteric disk model based on Williams et al. (2014)
    
    Parameters
    ----------
    molecule_table: path to the molecular table (e.g. molecule_12c16o.inp)
    transition: rotational transition index (e.g. 3 is CO(3-2) at ~345 GHz)
    freezeout: Kelvin. Freezeout temperature of the molecule (e.g. 20K for CO)
    delta: describe the steepness of the profile (multiplies the power of the sinusoid)
    r_scale: scaling in AU (default = 150AU)

    Notes
    -----
    https://iopscience.iop.org/article/10.1088/0004-637X/788/1/59
    """
    def __init__(self, name: str, T_mid1: float, T_atm1: float, q: float, M_star: float, gamma: float, r_in: float, 
                 r_c: float, M_gas: float, alpha_turb: float, co_abundance: float, N_dissoc: float, 
                 molecule_table: str, transition: int, m_mol: float = 2.37*m_h, freezeout: float = 20.0, delta: float = 2, r_scale: float = 1.0):
        self.name = name
        self.T_mid1 = T_mid1
        self.T_atm1 = T_atm1
        self.q = q
        self.M_star = M_star
        self.gamma = gamma
        self.r_in = r_in
        self.r_c = r_c
        self.M_gas = M_gas
        self.alpha_turb = alpha_turb
        self.co_abundance = co_abundance
        self.N_dissoc = N_dissoc
        self.z_q0 = z_q0
        self.molecule_table = molecule_table
        self.transition = transition
        self.m_mol = m_mol
        self.freezeout = freezeout
        self.delta = delta
        self.r_scale = r_scale

    def temperature_profile(self, z, r):
        T_mid = self.T_mid1 * (r/self.r_scale)**(-self.q)
        T_atm = self.T_atm1 * (r/self.r_scale)**(-self.q)
    
        # The temperature profile is a combination of T_mid, T_atm 
        # z_q describes the hight at which the disk transitions to the atmopheric value.
        # H_p is the pressure scale hight derived from the midplane tempertaure
        H_p = (kk * T_mid * (r * au)**3 / (G*M_star*m_mol))**0.5
        z_q = 4*H_p/au
        
        # The temperature profile is a combination of T_mid, T_atm 
        temperature = np.where(np.abs(z)<z_q, T_mid + (T_atm - T_mid)*jnp.sin(jnp.pi*z/(2*z_q))**(2*self.delta), T_atm)
        return temperature

    def co_abundance_profile(self, N_h2, temperature):
        co_region = jnp.bitwise_and(temperature>self.freezeout, N_h2>self.N_dissoc)
        abundance = jnp.where(co_region, self.co_abundance, 0.0)
        return abundance 

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

def surface_density(z, nd):
    """Compute the column integrated density N(h2) (units of 1/cm^2)"""
    dz = jnp.diff(z * au, axis=0)[0,0]
    N = jnp.cumsum(nd[::-1]*dz, axis=0)[::-1]
    return N

def vsq_keplerian_thin_disk(r, M):
    """Keplerian velocity of a thin disk"""
    return G*M/(r*au)
    
def vsq_keplerian_vertical(z, r, M):
    """Modified Keplerian velocity to account for height and gas pressure above the midplane"""
    return G*M*(r*au)**2 / ((r*au)**2 + (z*au)**2)**(3/2) 

def vsq_pressure_grad(r, nd, temperature, m_mol):
    """Modified Keplerian velocity to account for height and gas pressure above the midplane"""
    rho = m_mol * nd # mass density
    dr = jnp.diff(r * au, axis=1)[0,0]
    pressure_grad = (nd[:,1:]*kk*temperature[:,1:] - nd[:,:-1]*kk*temperature[:,:-1]) / dr
    pressure_grad = jnp.pad(pressure_grad, pad_width=[(0,0), (1,0)])
    return ((r*au)/rho)*pressure_grad

def azimuthal_velocity(coords, v_phi):
    ray_x, ray_y, ray_z = coords[...,0], coords[...,1], coords[...,2]
    ray_r = jnp.sqrt(ray_x**2 + ray_y**2)
    cosp = ray_x/ray_r
    sinp = ray_y/ray_r
    vx, vy, vz = v_phi*cosp, v_phi*sinp, jnp.zeros_like(v_phi)
    velocity = jnp.stack([vx, vy, vz], axis=-1)
    return velocity
    
def v_Keplerian(x, y, r, v_az):
    cosp = ray_x/ray_r
    sinp = ray_y/ray_r
    vx, vy, vz = v*cosp, v*sinp, jnp.zeros_like(v)
    velocity = jnp.stack([vx, vy, vz], axis=-1)
    return velocity

@jax.jit
def render_cube_jit(T_mid1, T_atm1, q, M_star, m_mol, delta, r_scale, gamma, r_in, r_c, M_gas, freezeout, N_dissoc, co_abundance,
                    freqs, n_up, n_dn, a_ud, b_ud, b_du, ray_coords, obs_dir, nu0, pixel_area, alpha_turb,
                    z_disk, r_disk, coords_polar, bbox_cyl, bbox):
    temperature = parametric_disk.temperature_profile_williams(z_disk, r_disk, T_mid1, T_atm1, q, M_star, m_mol, delta, r_scale)
    numberdens_h2 = parametric_disk.number_density_profile(z_disk, r_disk, temperature, disk.gamma, disk.r_in, disk.r_c, disk.M_gas, disk.M_star, disk.m_mol)
    
    # Compute the column integrated density N(h2) (units of 1/cm^2)
    dz = jnp.diff(z_disk * au, axis=0)[0,0]
    N_h2 = jnp.cumsum(numberdens_h2[::-1]*dz, axis=0)[::-1]
    
    co_region = jnp.bitwise_and(temperature>freezeout, N_h2>N_dissoc)
    abundance_co = jnp.where(co_region, co_abundance, 0.0)
    numberdens_co = abundance_co * numberdens_h2
    
    # Interpolate the data on spherical coordinates
    numberdens_co_sph = grid.interpolate_scalar(numberdens_co, coords_polar, bbox=bbox_cyl)
    temperature_sph = grid.interpolate_scalar(temperature, coords_polar, bbox=bbox_cyl)

    # Interpolate dataset along the ray coordinates
    gas_nd = grid.interpolate_scalar(numberdens_co_sph, ray_coords_sph, bbox)
    gas_t  = grid.interpolate_scalar(temperature_sph, ray_coords_sph, bbox, cval=1e-10)
    gas_v = parametric_disk.keplerian_velocity(ray_coords, M_star)

    images = jnp.nan_to_num(line_rte.compute_spectral_cube(freqs, gas_v, gas_t, n_up, n_dn, a_ud, b_ud, b_du, ray_coords, obs_dir, nu0, pixel_area, alpha_turb))
    return images
