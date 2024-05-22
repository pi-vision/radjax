import jax.numpy as jnp
import scipy
import jax.scipy as jscp
import numpy as np

def read_spherical_amr(path):
    """
    Load spherical grid from .inp file
    
    path: str,
        path to amr_grid.inp
    """
    nr, nth, nph = np.loadtxt(path, skiprows=5, max_rows=1, dtype='int')
    r = np.loadtxt(path, skiprows=6, max_rows=nr+1)
    theta = np.loadtxt(path, skiprows=6+nr+1, max_rows=nth+1)
    phi = np.loadtxt(path, skiprows=6+nr+nth+2, max_rows=nph+1)
    return r, theta, phi

def cartesian_to_spherical(coords, logr=False):
    r = jnp.sqrt(coords[...,0]**2 + coords[...,1]**2 + coords[...,2]**2)
    r = jnp.log10(r) if logr==True else r
    rho = jnp.sqrt(coords[...,0]**2 + coords[...,1]**2)
    theta = jnp.arctan2(rho, coords[...,2])
    phi = (jnp.arctan2(coords[...,1], coords[...,0]) + 2*jnp.pi) % (2*jnp.pi)
    coords_sph = jnp.stack([phi, theta, r], axis=-1)
    return coords_sph

def spherical_vec_to_cartesian(vector, coords, bbox):
    r = jnp.sqrt(coords[...,0]**2 + coords[...,1]**2 + coords[...,2]**2)
    rho = jnp.sqrt(coords[...,0]**2 + coords[...,1]**2)
    cosp = coords[...,0]/rho
    sinp = coords[...,1]/rho
    cost = coords[...,2]/r
    sint = rho/r
    dummy = sint * vector[...,0] + cost * vector[...,1]
    vx   = cosp * dummy - sinp * vector[...,2]
    vy   = sinp * dummy + cosp * vector[...,2]
    vz   = cost * vector[...,0] - sint * vector[...,1]
    coords_cart = jnp.stack([vx, vy, vz], axis=-1)
    return coords_cart

def spherical_to_zr(ray_coords_sph):
    """Polar coordinates with azimuthal symmetry (phi) and mirror symmetry in vertical coordinate (z)"""
    r_polar = ray_coords_sph[...,2] * jnp.sin(ray_coords_sph[...,1])
    z_polar = jnp.abs(ray_coords_sph[...,2] * jnp.cos(ray_coords_sph[...,1]))
    ray_coords_polar = jnp.stack((z_polar, r_polar), axis=-1)
    return ray_coords_polar
    
def rotate_coords(coords, incl, phi, posang):
    rot_matrix = scipy.spatial.transform.Rotation.from_euler('zxz', [posang, incl, -phi], degrees=True).as_matrix()
    coords_rot = jnp.rollaxis(jnp.matmul(rot_matrix, jnp.rollaxis(jnp.array(coords), -1, 1)), -1, 1)
    return coords_rot

def world_to_image_coords(coords, bbox, npix):
    return (coords - bbox[:,0]) * (npix - 1)/(bbox[:,1]-bbox[:,0])
    
def interpolate_scalar(volume, coords, bbox, cval=0.0, order=1):
    image_coords = jnp.moveaxis(world_to_image_coords(coords, bbox, jnp.array(volume.shape)), -1, 0)
    interpolated_data = jscp.ndimage.map_coordinates(volume, image_coords, order=int(order), cval=float(cval))
    return interpolated_data

def interpolate_vector(vector_field, coords_cart, coords_sph, bbox, cval=0.0, order=1):
    """
    Vector dimension along last axis. 
    Input vector is spherical interpolated to Cartesian coordinates.
    """
    v_sph = jnp.stack([interpolate_scalar(vector_field[...,i], coords_sph, bbox, cval, order) for i in range(3)], axis=-1)
    v_cart = spherical_vec_to_cartesian(v_sph, coords_cart, bbox)
    return v_cart
    
def expand_dims(x, ndim, axis=0):
    for i in range(ndim - jnp.array(x).ndim):
        x = jnp.expand_dims(x, axis=min(axis, jnp.array(x).ndim))
    return x