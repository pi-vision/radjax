import numpy as np
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)  # ensure float64 so we can use tight tolerances

import radjax.core.grid  as grid# your module
arcsec = np.pi / (180.0 * 3600.0)           # rad/arcsec
au     = 1.495978707e11                     # m
pc     = 3.085677581491367e16               # m

def pinhole_disk_projection_new(x_sky, y_sky, distance, nray, incl, phi, posang, z_width):
    """
    New path: uses JAX-native rotate functions in radjax.core.grid.
    """
    x_sky = jnp.asarray(x_sky, dtype=jnp.float64)
    y_sky = jnp.asarray(y_sky, dtype=jnp.float64)

    ny, nx = x_sky.shape
    obs_dir = jnp.squeeze(grid.rotate_coords_angles(jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64), incl, phi))

    rho   = jnp.sqrt(x_sky**2 + y_sky**2) * arcsec
    theta = jnp.arctan2(y_sky, x_sky)

    ray_x_dir = rho * jnp.cos(theta)
    ray_y_dir = rho * jnp.sin(theta)
    ray_z_dir = jnp.ones_like(ray_x_dir)

    # stack as (ny*nx, 3)
    ray_dir = jnp.stack((ray_y_dir, ray_x_dir, ray_z_dir), axis=-1).reshape(ny * nx, 3)

    # rotate directions: first XZ (incl, -phi), then axis-angle around obs_dir by -posang
    ray_dir = grid.rotate_coords_angles(ray_dir, incl, -phi)
    ray_dir = grid.rotate_coords_vector(ray_dir, obs_dir, -posang)

    # intersect with slab of thickness z_width centered on origin along obs_dir
    z_half = 0.5 * z_width * au
    cam_z  = distance * pc * obs_dir[2]

    s = ( z_half - cam_z) / ray_dir[..., 2]
    t = (-z_half - cam_z) / ray_dir[..., 2]

    ray_start = distance * pc * obs_dir + s[..., None] * ray_dir
    ray_stop  = distance * pc * obs_dir + t[..., None] * ray_dir

    # sample along each ray: (ny*nx, nray, 3) -> (ny, nx, nray, 3)
    ray_coords = jnp.linspace(ray_start, ray_stop, nray, axis=1).reshape(ny, nx, nray, 3)
    return ray_coords, obs_dir


def pinhole_disk_projection_old(x_sky, y_sky, distance, nray, incl, phi, posang, z_width):
    """
    Old path: uses the *_old rotation functions exactly.
    Note: rotate_coords_angles_old on a (3,) input returns extra dims; we squeeze.
    """
    x_sky = jnp.asarray(x_sky, dtype=jnp.float64)
    y_sky = jnp.asarray(y_sky, dtype=jnp.float64)

    ny, nx = x_sky.shape
    obs_dir_old = jnp.squeeze(grid.rotate_coords_angles_old(jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64), incl, phi))

    rho   = jnp.sqrt(x_sky**2 + y_sky**2) * arcsec
    theta = jnp.arctan2(y_sky, x_sky)

    ray_x_dir = rho * jnp.cos(theta)
    ray_y_dir = rho * jnp.sin(theta)
    ray_z_dir = jnp.ones_like(ray_x_dir)

    ray_dir = jnp.stack((ray_y_dir, ray_x_dir, ray_z_dir), axis=-1).reshape(ny * nx, 3)

    ray_dir = grid.rotate_coords_angles_old(ray_dir, incl, -phi)
    ray_dir = grid.rotate_coords_vector_old(ray_dir, obs_dir_old, -posang)

    z_half = 0.5 * z_width * au
    cam_z  = distance * pc * obs_dir_old[2]

    s = ( z_half - cam_z) / ray_dir[..., 2]
    t = (-z_half - cam_z) / ray_dir[..., 2]

    ray_start = distance * pc * obs_dir_old + s[..., None] * ray_dir
    ray_stop  = distance * pc * obs_dir_old + t[..., None] * ray_dir

    ray_coords = jnp.linspace(ray_start, ray_stop, nray, axis=1).reshape(ny, nx, nray, 3)
    return ray_coords, obs_dir_old


def max_err(a, b):
    A = np.asarray(a); B = np.asarray(b)
    return np.max(np.abs(A - B))

def main():
    # Build a tiny on-sky grid in arcsec
    ny, nx = 16, 20
    fov = 8.0  # arcsec full width
    yv, xv = np.linspace(-fov/2, fov/2, ny), np.linspace(-fov/2, fov/2, nx)
    x_sky, y_sky = np.meshgrid(xv, yv)  # arcsec

    # Camera / disk params
    distance = 100.0   # pc
    nray     = 17
    incl     = 47.0
    phi      = -121.3
    posang   = 33.4
    z_width  = 40.0    # au

    rc_new,  od_new  = pinhole_disk_projection_new(x_sky, y_sky, distance, nray, incl, phi, posang, z_width)
    rc_old,  od_old  = pinhole_disk_projection_old(x_sky, y_sky, distance, nray, incl, phi, posang, z_width)

    e_obs  = max_err(od_new, od_old)
    e_rays = max_err(rc_new, rc_old)

    print(f"raycoords shape old: {rc_old.shape}, raycoords shape new: {rc_new.shape}")
    print(f"mean diff raycoords: {(rc_new-rc_old).mean()}")
    print(f"raycoords old: {rc_old[0,0,0,:]}, raycoords new: {rc_new[0,0,0,:]}")
    print(f"raycoords diff: {rc_old[0,0,0,:]-rc_new[0,0,0,:]}")
    print(f"obsdir old: {od_old}, obsdir new: {od_new}")
    print("obs_dir   max|Δ| =", e_obs)
    print("ray_coords max|Δ| =", e_rays)

    # Tight tolerances in float64
    assert e_obs  < 5e-12, f"obs_dir mismatch: {e_obs}"
    assert e_rays < 5e-12, f"ray_coords mismatch: {e_rays}"
    print("pinhole_disk_projection (new vs _old): PASS")

if __name__ == "__main__":
    main()