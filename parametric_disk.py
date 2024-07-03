import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from consts import *

@jax.tree_util.register_pytree_node_class
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
    def __init__(self, T_mid1: float, T_atm1: float, q: float, q_in: float, r_break: float, M_star: float, gamma: float, 
                 r_in: float, log_r_c: float, M_gas: float, v_turb: float, co_abundance: float, N_dissoc_lo: 
                 float, N_dissoc_hi: float, z_q0: float, transition: int, m_mol: float, freezeout: float, delta: float, 
                 r_scale: float, molecule_table: str, name: str = 'DiskFlaherty'):
        self.T_mid1 = T_mid1
        self.T_atm1 = T_atm1
        self.q = q
        self.q_in = q_in
        self.r_break = r_break
        self.M_star = M_star
        self.gamma = gamma
        self.r_in = r_in
        self.log_r_c = log_r_c
        self.M_gas = M_gas
        self.v_turb = v_turb
        self.co_abundance = co_abundance
        self.N_dissoc_lo = N_dissoc_lo
        self.N_dissoc_hi = N_dissoc_hi
        self.z_q0 = z_q0
        self.molecule_table = molecule_table
        self.transition = transition
        self.m_mol = m_mol
        self.freezeout = freezeout
        self.delta = delta
        self.r_scale = r_scale        
        self.name = name

    def temperature_profile(self, z, r):
        q = jnp.where(r<=self.r_break, self.q_in, self.q)
        T_mid = self.T_mid1 * (r/self.r_scale)**q
        T_atm = self.T_atm1 * (r/self.r_scale)**q
    
        # The parameter Zq is the height above the midplane at which the gas temperature reaches its maximum value
        z_q = self.z_q0 * (r/self.r_scale)**1.3
        
        # The temperature profile is a combination of T_mid, T_atm 
        temperature = jnp.where(z<z_q, T_atm + (T_mid-T_atm)*jnp.cos(jnp.pi*z/(2*z_q))**(2*self.delta), T_atm)
        return temperature

    def co_abundance_profile(self, N_h2, temperature):
        
        co_region = jnp.bitwise_and(
            temperature>self.freezeout, 
            jnp.bitwise_and(0.706*N_h2>self.N_dissoc_lo, 0.706*N_h2<self.N_dissoc_hi)
        )
        abundance = jnp.where(co_region, self.co_abundance, 0.0)
        return abundance 

    def velocity(self, z, r, nd, temperature):
        """Modified Keplerian velocity to account for height and gas pressure above the midplane"""
        # v = jnp.sqrt( vsq_keplerian_vertical(z, r, self.M_star) + vsq_pressure_grad(r, nd, temperature, self.m_mol) )
        v = jnp.sqrt( vsq_keplerian_vertical(z, r, self.M_star) )
        # If there is a large negative pressure gradient this sqrt might produce nans
        # v = jnp.nan_to_num(v)
        return v
        
    def tree_flatten(self):
        children = (self.T_mid1,self.T_atm1, self.q, self.q_in, self.r_break, self.M_star, self.gamma, 
                    self.r_in, self.log_r_c, self.M_gas, self.v_turb, self.co_abundance, self.N_dissoc_lo, 
                    self.N_dissoc_hi, self.z_q0, self.transition, self.m_mol, self.freezeout, self.delta, self.r_scale)
        aux_data = {'molecule_table': self.molecule_table, 'name': self.name}  # static values
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def update(self, params, values):
        self.__dict__.update(dict(zip(params, values)))
        
    def __copy__(self, name=None):
        name = self.name if name is None else name
        return DiskFlaherty(
            self.T_mid1,self.T_atm1, self.q,  self.q_in, self.r_break, self.M_star, self.gamma, self.r_in, self.log_r_c, 
            self.M_gas, self.v_turb, self.co_abundance, self.N_dissoc, self.z_q0, self.transition, 
            self.m_mol, self.freezeout, self.delta, self.r_scale, self.molecule_table, name)
        
    def __deepcopy__(self, memo, name=None):
        name = self.name if name is None else name
        return DiskFlaherty(copy.deepcopy(
            self.T_mid1,self.T_atm1, self.q,  self.q_in, self.r_break, self.M_star, self.gamma, self.r_in, self.log_r_c, 
            self.M_gas, self.v_turb, self.co_abundance, self.N_dissoc_lo, self.N_dissoc_hi, self.z_q0, self.transition, 
            self.m_mol, self.freezeout, self.delta, self.r_scale, self.molecule_table, name, memo))

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
    pressure = nd*kk*temperature
    pressure_grad = (pressure[:,1:] - pressure[:,:-1]) / dr
    pressure_grad = jnp.pad(pressure_grad, pad_width=[(0,0), (1,0)])
    return ((r*au)/rho)*pressure_grad

def azimuthal_velocity(coords, v_phi):
    ray_r = jnp.sqrt(coords[...,0]**2 + coords[...,1]**2)
    cosp = coords[...,1]/ray_r
    sinp = coords[...,0]/ray_r
    vy, vx, vz = v_phi*cosp, v_phi*sinp, jnp.zeros_like(v_phi)
    velocity = jnp.stack([vy, vx, vz], axis=-1)
    return velocity