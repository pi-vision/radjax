import jax.numpy as jnp
import grid
from consts import *

def compute_camera_freqs(linenlam, width_kms, nu0, num_subfreq=1, subfreq_width=None, v_kms=0.0):
    """
    width_kms: float,
        Width of the spectral window
    linenlam: int,
        Spectral resolution
    nu0: float, 
        Central frequency.
    v_kms: float,
        Zero-point of the freqency grid. Default is 0 results in a grid centered around nu0.
    """
    camera_freqs = nu0 * (1 - v_kms*1e5/cc - (2*jnp.arange(linenlam)/(linenlam-1)-1)*width_kms*1e5/cc)
    if num_subfreq > 1:
        assert subfreq_width, "if num_subfreq>1, subfreq_width should be specified"
        camera_freqs = jnp.linspace(camera_freqs-subfreq_width/2, camera_freqs+subfreq_width/2, num_subfreq, axis=1)
    return camera_freqs
    
def orthographic_projection(npix, nray, incl, phi, posang, fov, z_width):
    """Observer at infinity
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

