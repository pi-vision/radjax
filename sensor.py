from dataclasses import dataclass
import jax
import jax.scipy as jsp
import jax.numpy as jnp
import numpy as np
import grid
from consts import *

@dataclass
class Projection:
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

def make_gaussian(size, fov_rad, fwhm_max, fwhm_min, theta, frac=1.0, x_c=0.0, y_c=0.0):
    """
    Notes
    -----
    This function is modeled after eht-imaging blur_gauss function: 
    https://github.com/achael/eht-imaging/blob/9ebd83345b62dbcee1fe5267c6e27b786acfe9ff/ehtim/image.py#L1342
    """
    psize_rad = fov_rad/size
    x = jnp.linspace(-(fov_rad-psize_rad)/2.0, (fov_rad-psize_rad)/2.0, size) - x_c
    y = jnp.linspace(-(fov_rad-psize_rad)/2.0, (fov_rad-psize_rad)/2.0, size) - y_c
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    sigma_maj = fwhm_max / (2. * jnp.sqrt(2. * jnp.log(2.)))
    sigma_min = fwhm_min / (2. * jnp.sqrt(2. * jnp.log(2.)))
    cth = jnp.cos(theta)
    sth = jnp.sin(theta)
    gaussian = jnp.exp(-(yy * cth + xx * sth)**2 / (2 * (frac * sigma_maj)**2) - (xx * cth - yy * sth)**2 / (2 * (frac * sigma_min)**2))
    gaussian = gaussian / jnp.sum(gaussian)
    return gaussian

def gaussian_beam(size, fov_rad, beamsize):
    beam_area = jnp.pi * beamsize * beamsize / (4*jnp.log(2))
    beams_per_pix = (fov_rad/size)**2 / beam_area
    beam = beams_per_pix * make_gaussian(size, fov_rad, beamsize, beamsize, 0, 1.0, 0.0, 0.0)
    return beam
    
fftconvolve_vmap = jax.vmap(lambda x, kernel: jsp.signal.fftconvolve(x, kernel, mode='same'), (0, None))