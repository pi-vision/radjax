from dataclasses import dataclass
import jax
import jax.scipy as jsp
import jax.numpy as jnp
import numpy as np
import grid
from consts import *

@dataclass
class OrthographicProjection:
    name: str
    npix: int
    nray: int
    incl: float
    phi: float
    posang: float
    fov: float
    r_min: float
    r_max: float
    theta_min: float
    theta_max: float
    linelam: float
    width_kms: float
    v_sys: float = 0.0

@dataclass
class PinholeProjection:
    name: str
    x_sky: float
    y_sky: float
    distance: float
    nray: int
    incl: float
    phi: float
    posang: float
    z_width: float
    freqs: float

def compute_camera_freqs(linenlam, width_kms, nu0, v_sys=0.0, num_subfreq=1, subfreq_width=None):
    """
    width_kms: float,
        Width of the spectral window
    linenlam: int,
        Spectral resolution
    nu0: float, 
        Central frequency.
    v_sys: float,
        systematic velocity of the disk [km/s] (zero-point of the freqency grid). Default is 0 results in a grid centered around nu0.
    """
    camera_freqs = nu0 * (1 - v_sys*1e5/cc - (2*jnp.arange(linenlam)/(linenlam-1)-1)*width_kms*1e5/cc)
    if num_subfreq > 1:
        assert subfreq_width, "if num_subfreq>1, subfreq_width should be specified"
        camera_freqs = jnp.linspace(camera_freqs-subfreq_width/2, camera_freqs+subfreq_width/2, num_subfreq, axis=1)
    return camera_freqs


def pinhole_disk_projection(x_sky, y_sky, distance, nray, incl, phi, posang, z_width):
    """
    Pinhole ray projection
    The rays are constrained to start and terminate within the disk of thickness z_width

    Parameters
    ----------
    x_sky, y_sky: two dimensional arrays,
        on sky coordinates in [arcsecs]
    distance: float,
        distance in [pc]
    nray: int,
        Number of point along each ray. 
    incl: float, 
        inclination angle [deg]
    phi: float,
        azimuthal angle [deg]
    posang: float,
        camera roll angle in [deg]
    z_width: float,
        Width of the disk in [au]
    """
    ny, nx = x_sky.shape
    obs_dir = jnp.squeeze(grid.rotate_coords_angles(jnp.array([0,0,1]), incl, phi))

    rho = jnp.sqrt(x_sky**2 + y_sky**2) * arcsec
    theta = jnp.arctan2(y_sky, x_sky)
    
    ray_x_dir = rho * jnp.cos(theta)
    ray_y_dir = rho * jnp.sin(theta)
    ray_z_dir = jnp.ones_like(ray_x_dir)
    
    ray_dir = jnp.stack((ray_y_dir, ray_x_dir, ray_z_dir), axis=-1).reshape(ny*nx, 3)
    ray_dir = grid.rotate_coords_vector(grid.rotate_coords_angles(ray_dir, incl, -phi), obs_dir, -posang)
    
    s = (0.5 * z_width*au - distance*pc * obs_dir[2] ) / ray_dir[...,2]
    t = (-0.5 * z_width*au - distance*pc * obs_dir[2] ) / ray_dir[...,2]
    ray_start = distance*pc * obs_dir + s[...,None] * ray_dir
    ray_stop = distance*pc * obs_dir + t[...,None] * ray_dir
    
    ray_coords = jnp.linspace(ray_start, ray_stop, nray, axis=1).reshape(ny, nx, nray, 3)
    return ray_coords, obs_dir

def orthographic_disk_projection(npix, nray, incl, phi, posang, fov, r_min, r_max, theta_min, theta_max):
    """
    Parallel ray projection --> Observer at infinity
    The rays are constrained to start and terminate within a disk described by (r_min, r_max), (theta_min, theta_max)

    Parameters
    ----------
    npix: int,
        Number of pixels on camera x,y axis
    nray: int,
        Number of point along each ray. 
    incl: float, 
        inclination angle [deg]
    phi: float,
        azimuthal angle [deg]
    posang: float,
        camera roll angle in [deg]
    fov: float,
        field of view in grid units
    r_min: float,
        minimum radius of the disk 
    r_max: float,
        maximum radius of the disk
    theta_min: float,
        minimum inclination of the disk [radians] (note the units differ from view angle!)
    theta_max: float,
        maximum inclination of the disk [radians] (note the units differ from view angle!)
    """
    obs_dir = grid.rotate_coords([0,0,1], incl, phi, posang)
    
    # XY Plane coordinates
    ray_x = jnp.linspace(-0.5, 0.5, npix) * fov * (npix-1)/npix
    ray_y = jnp.linspace(-0.5, 0.5, npix) * fov * (npix-1)/npix
    xy_plane = jnp.stack(jnp.meshgrid(ray_x, ray_y, jnp.array([0.0]), indexing='xy'), axis=-1).reshape(npix*npix, 1, 3)
    xy_plane_rot = jnp.squeeze(grid.rotate_coords(xy_plane, incl, phi, posang))
    
    # Solve for the intersection of the observation vector with the sphere r=r_max
    # This ends up being a quadratic expression
    r_sq = jnp.sum(xy_plane_rot**2, axis=-1)
    a = jnp.sum(obs_dir**2)
    b = 2 * jnp.sum(xy_plane_rot*obs_dir, axis=-1)
    c = r_sq - r_max**2
    s = (-b+jnp.sqrt(b**2 - 4*a*c)) / (2*a)
    t = (-b-jnp.sqrt(b**2 - 4*a*c)) / (2*a)
    ray_start = jnp.array(xy_plane_rot + s[:,None] * obs_dir[None,:])
    ray_stop = jnp.array(xy_plane_rot + t[:,None] * obs_dir[None,:])
    
    # Intersection with {theta_min, theta_max} cones
    a_min = (obs_dir[0]**2 + obs_dir[1]**2)*jnp.cos(theta_min)**2 - obs_dir[2]**2*jnp.sin(theta_min)**2
    a_max = (obs_dir[0]**2 + obs_dir[1]**2)*jnp.cos(theta_max)**2 - obs_dir[2]**2*jnp.sin(theta_max)**2
    b_min = 2*(ray_start[...,0]*obs_dir[0] + ray_start[...,1]*obs_dir[1])*jnp.cos(theta_min)**2 - 2*ray_start[...,2]*obs_dir[2]*jnp.sin(theta_min)**2
    b_max = 2*(ray_start[...,0]*obs_dir[0] + ray_start[...,1]*obs_dir[1])*jnp.cos(theta_max)**2 - 2*ray_start[...,2]*obs_dir[2]*jnp.sin(theta_max)**2
    c_min = (ray_start[...,0]**2 + ray_start[...,1]**2)*jnp.cos(theta_min)**2 - ray_start[...,2]**2*jnp.sin(theta_min)**2
    c_max = (ray_start[...,0]**2 + ray_start[...,1]**2)*jnp.cos(theta_max)**2 - ray_start[...,2]**2*jnp.sin(theta_max)**2
    s = (-b_max+jnp.sqrt(b_max**2 - 4*a_max*c_max)) / (2*a_max)
    t = (-b_min-jnp.sqrt(b_min**2 - 4*a_min*c_min)) / (2*a_min)
    cone_bottom = ray_start + s[:,None] * obs_dir[None,:]
    cone_top = ray_start + t[:,None] * obs_dir[None,:]
    
    # Project start/stop points that intersect the top/bottom cones
    ray_start_theta = grid.cartesian_to_spherical(ray_start)[:,1]
    ray_stop_theta = grid.cartesian_to_spherical(ray_stop)[:,1]
    ray_start.at[ray_start_theta<theta_min,:].set(cone_top[ray_start_theta<theta_min])
    ray_start.at[ray_start_theta>theta_max,:].set(cone_bottom[ray_start_theta>theta_max])
    ray_stop.at[ray_stop_theta<theta_min,:].set(cone_top[ray_stop_theta<theta_min])
    ray_stop.at[ray_stop_theta>theta_max,:].set(cone_bottom[ray_stop_theta>theta_max])
    
    # Remove start/stop points that are both outside the angle range
    ray_start.at[jnp.bitwise_and(ray_start_theta<theta_min, ray_stop_theta<theta_min),:].set(jnp.nan)
    ray_start.at[jnp.bitwise_and(ray_start_theta>theta_max, ray_stop_theta>theta_max),:].set(jnp.nan)
    ray_stop.at[jnp.bitwise_and(ray_start_theta<theta_min, ray_stop_theta<theta_min),:].set(jnp.nan)
    ray_stop.at[jnp.bitwise_and(ray_start_theta>theta_max, ray_stop_theta>theta_max),:].set(jnp.nan)
    
    # Project start/stop points that intersect the r_min sphere
    # This ends up being a quadratic expression
    ray_start_r = grid.cartesian_to_spherical(ray_start)[:,2]
    ray_stop_r = grid.cartesian_to_spherical(ray_stop)[:,2]
    c = r_sq - r_min**2
    s = (-b+jnp.sqrt(b**2 - 4*a*c)) / (2*a)
    t = (-b-jnp.sqrt(b**2 - 4*a*c)) / (2*a)
    ray_start.at[ray_start_r<r_min,:].set(xy_plane_rot[ray_start_r<r_min,:] + t[ray_start_r<r_min,None] * obs_dir[None,:])
    ray_stop.at[ray_stop_r<r_min,:].set(xy_plane_rot[ray_stop_r<r_min,:] + s[ray_stop_r<r_min,None] * obs_dir[None,:])
    
    # Remove points that are within r_min 
    remove_indices = jnp.bitwise_and(ray_start_r<r_min, ray_stop_r<r_min)
    ray_start.at[remove_indices,:].set(jnp.nan)
    ray_stop.at[remove_indices,:].set(jnp.nan)
    
    ray_coords = jnp.linspace(ray_start, ray_stop, nray, axis=1)
    return ray_coords, obs_dir

def orthographic_projection(npix, nray, incl, phi, posang, fov, z_width):
    """
    Parallel ray projection --> Observer at infinity

    Parameters
    ----------
    npix: int,
        Number of pixels on camera x,y axis
    nray: int,
        Number of point along each ray. 
    incl: float, 
        inclination angle [deg]
    phi: float,
        azimuthal angle [deg]
    posang: float,
        camera roll angle in [deg]
    fov: float,
        field of view in grid units
    z_width: float, 
        Width of the disk
    """
    obs_dir = grid.rotate_coords([0,0,1], incl, phi, posang)
    
    # Ray coordinates
    ray_x = jnp.linspace(-0.5, 0.5, npix) * fov * (npix-1)/npix
    ray_y = jnp.linspace(-0.5, 0.5, npix) * fov * (npix-1)/npix
    ray_z = jnp.linspace(-0.5, 0.5, nray) * z_width / obs_dir[2]
    coords_grid = jnp.stack(jnp.meshgrid(ray_x, ray_y, ray_z, indexing='xy'), axis=-1).reshape(npix*npix, nray, 3)
    ray_coords = grid.rotate_coords(coords_grid, incl, phi, posang)

    # Center the z-coordinates to be within the disk width
    xy_plane = jnp.stack(jnp.meshgrid(ray_x, ray_y, jnp.array([0.0]), indexing='xy'), axis=-1).reshape(npix*npix, 1, 3)
    xy_plane_rot = grid.rotate_coords(xy_plane, incl, phi, posang)
    s = xy_plane_rot[...,2] / obs_dir[2]
    ray_coords = ray_coords - s[...,None] * obs_dir
    
    # Reverse the order of the coordinates
    ray_coords = ray_coords[:,::-1] 
    
    return ray_coords, obs_dir

def project_volume(volume, coords, bbox):
    ds = jnp.sqrt(jnp.diff(coords[...,0])**2 + jnp.diff(coords[...,1])**2 + jnp.diff(coords[...,2])**2 )
    volume_interp = grid.interpolate_coords(volume, coords, bbox)
    volume_zero_order = volume_interp[:,:-1] + np.diff(volume_interp, axis=-1) / 2.0
    projection = jnp.sum(volume_zero_order * ds, axis=-1)
    return projection

def beam(dpix, bmaj, bmin, bpa, scale=1.0, x_c=0.0, y_c=0.0):
    """›
    """
    from astropy.convolution import Gaussian2DKernel
    
    bmaj = scale * bmaj / dpix / 2.355
    bmin = scale * bmin / dpix / 2.355
    kernel = jnp.array(Gaussian2DKernel(x_stddev=bmin, y_stddev=bmaj, theta=np.radians(bpa)).array)
    
    return kernel
    
fftconvolve_vmap = jax.vmap(lambda x, kernel: jsp.signal.fftconvolve(x, kernel, mode='same'), (0, None))