"""
Grid, coordinate transforms, and interpolation utilities for RadJAX.

Notes
-----
- Interpolation uses `jax.scipy.ndimage.map_coordinates` so it is JAX-compatible.
- Rotations use `scipy.spatial.transform.Rotation`, which returns NumPy arrays; we
  convert to `jnp` before applying. Avoid JIT-wrapping these rotation builders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import jax.scipy as jscp
from scipy.spatial.transform import Rotation as R


# -----------------------------------------------------------------------------
# Coordinate transforms
# -----------------------------------------------------------------------------
def cartesian_to_spherical(coords: jnp.ndarray, logr: bool = False) -> jnp.ndarray:
    """
    Convert Cartesian coordinates to spherical (phi, theta, r).

    Parameters
    ----------
    coords : (..., 3) jnp.ndarray
        [x, y, z] in the last axis.
    logr : bool, optional
        If True, return log10(r) instead of r.

    Returns
    -------
    coords_sph : (..., 3) jnp.ndarray
        [phi, theta, r] with phi in [0, 2π), theta in [0, π].
    """
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    r_out = jnp.log10(jnp.clip(r, min=1e-300)) if logr else r
    rho = jnp.sqrt(x**2 + y**2)
    theta = jnp.arctan2(rho, z)
    phi = (jnp.arctan2(y, x) + 2 * jnp.pi) % (2 * jnp.pi)
    return jnp.stack([phi, theta, r_out], axis=-1)

def spherical_vec_to_cartesian(vector: jnp.ndarray, coords: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a vector given in spherical basis (e_phi, e_theta, e_r) to Cartesian.

    Parameters
    ----------
    vector : (..., 3) jnp.ndarray
        Vector components in spherical basis [v_phi, v_theta, v_r].
    coords : (..., 3) jnp.ndarray
        Cartesian coordinates [x, y, z] at which the basis is defined.

    Returns
    -------
    v_cart : (..., 3) jnp.ndarray
        Cartesian components [vx, vy, vz].
    """
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    r = jnp.sqrt(x**2 + y**2 + z**2)
    rho = jnp.sqrt(x**2 + y**2)

    # Avoid singularities at rho=0
    eps = 1e-30
    cosp = x / jnp.clip(rho, min=eps)
    sinp = y / jnp.clip(rho, min=eps)
    cost = z / jnp.clip(r, min=eps)
    sint = rho / jnp.clip(r, min=eps)

    v_phi, v_theta, v_r = vector[..., 0], vector[..., 1], vector[..., 2]
    dummy = sint * v_theta + cost * v_r
    vx = cosp * dummy - sinp * v_phi
    vy = sinp * dummy + cosp * v_phi
    vz = cost * v_theta - sint * v_r
    return jnp.stack([vx, vy, vz], axis=-1)

def spherical_to_zr(ray_coords_sph: jnp.ndarray) -> jnp.ndarray:
    """
    Map spherical coordinates to polar (z, r) for axisymmetric problems.

    Parameters
    ----------
    ray_coords_sph : (..., 3) jnp.ndarray
        [phi, theta, r] (phi unused).

    Returns
    -------
    ray_coords_polar : (..., 2) jnp.ndarray
        [z, r_cyl] where z>=0 (mirror symmetry applied).
    """
    theta = ray_coords_sph[..., 1]
    r = ray_coords_sph[..., 2]
    r_cyl = r * jnp.sin(theta)
    z = jnp.abs(r * jnp.cos(theta))
    return jnp.stack((z, r_cyl), axis=-1)

# -----------------------------------------------------------------------------
# Rotations (JAX-native)
# -----------------------------------------------------------------------------
def _skew(v: jnp.ndarray) -> jnp.ndarray:
    """Skew-symmetric matrix [v]_x for cross products."""
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    zeros = jnp.zeros_like(vx)
    return jnp.stack(
        [
            jnp.stack([zeros, -vz, vy], axis=-1),
            jnp.stack([vz, zeros, -vx], axis=-1),
            jnp.stack([-vy, vx, zeros], axis=-1),
        ],
        axis=-2,
    )  # (..., 3, 3)


def _normalize(v: jnp.ndarray, eps: float = 1e-30) -> jnp.ndarray:
    return v / jnp.clip(jnp.linalg.norm(v, axis=-1, keepdims=True), min=eps)


def _rodrigues(axis: jnp.ndarray, angle_rad: jnp.ndarray) -> jnp.ndarray:
    """
    Rodrigues' rotation formula: R = I + sinθ [k]_x + (1-cosθ) [k]_x^2.

    Supports broadcasting over the leading dimensions of `axis` and `angle_rad`.
    """
    k = _normalize(axis)
    K = _skew(k)
    sin_t = jnp.sin(angle_rad)[..., None, None]
    cos_t = jnp.cos(angle_rad)[..., None, None]
    I = jnp.eye(3, dtype=axis.dtype)
    return I + sin_t * K + (1.0 - cos_t) * (K @ K)


def _rotz(angle_rad: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    return jnp.stack(
        [
            jnp.stack([c, -s, jnp.zeros_like(c)], axis=-1),
            jnp.stack([s, c, jnp.zeros_like(c)], axis=-1),
            jnp.stack([jnp.zeros_like(c), jnp.zeros_like(c), jnp.ones_like(c)], axis=-1),
        ],
        axis=-2,
    )


def _rotx(angle_rad: jnp.ndarray) -> jnp.ndarray:
    c, s = jnp.cos(angle_rad), jnp.sin(angle_rad)
    return jnp.stack(
        [
            jnp.stack([jnp.ones_like(c), jnp.zeros_like(c), jnp.zeros_like(c)], axis=-1),
            jnp.stack([jnp.zeros_like(c), c, -s], axis=-1),
            jnp.stack([jnp.zeros_like(c), s, c], axis=-1),
        ],
        axis=-2,
    )


def _rotzxz(phi1: jnp.ndarray, theta: jnp.ndarray, phi2: jnp.ndarray) -> jnp.ndarray:
    """ZXZ Euler rotation matrix: Rz(phi1) @ Rx(theta) @ Rz(phi2)."""
    return _rotz(phi1) @ _rotx(theta) @ _rotz(phi2)


def _roxz(theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    """XZ Euler rotation matrix: Rx(theta) @ Rz(phi)."""
    return _rotx(theta) @ _rotz(phi)


def rotate_coords(coords: jnp.ndarray, incl: float, phi: float, posang: float) -> jnp.ndarray:
    """
    Apply ZXZ Euler rotation (degrees): (posang, incl, -phi).

    Parameters
    ----------
    coords : (..., 3) jnp.ndarray
        Cartesian coordinates to rotate.
    incl, phi, posang : float
        Angles in degrees.

    Returns
    -------
    coords_rot : (..., 3) jnp.ndarray
        Rotated coordinates.
    """
    # ZXZ intrinsic like SciPy: Rz(posang) @ Rx(incl) @ Rz(-phi)
    Rmat = _rotzxz(
        jnp.deg2rad(jnp.asarray(posang)),
        jnp.deg2rad(jnp.asarray(incl)),
        jnp.deg2rad(-jnp.asarray(phi)),
    )  # (3,3)
    coords = jnp.asarray(coords)
    # (...,3) @ (3,3)^T → (...,3)
    return coords @ Rmat.T


def rotate_coords_angles(coords: jnp.ndarray, incl: float, phi: float) -> jnp.ndarray:
    """
    Apply XZ Euler rotation (degrees): (incl, phi).

    Parameters
    ----------
    coords : (..., 3) jnp.ndarray
    incl, phi : float
        Angles in degrees.

    Returns
    -------
    coords_rot : (..., 3) jnp.ndarray
    """
    # XZ intrinsic like SciPy: Rx(incl) @ Rz(phi)
    Rmat = _roxz(jnp.deg2rad(jnp.asarray(incl)), jnp.deg2rad(jnp.asarray(phi)))  # (3,3)
    coords = jnp.asarray(coords)
    return coords @ Rmat.T


def rotate_coords_vector(coords: jnp.ndarray, vector: jnp.ndarray, angle: float) -> jnp.ndarray:
    """
    Rotate points around an axis `vector` by `angle` degrees using Rodrigues' formula.

    Parameters
    ----------
    coords : (..., 3) jnp.ndarray
        Points to rotate.
    vector : (3,) jnp.ndarray
        Rotation axis (need not be unit length).
    angle : float
        Rotation angle in degrees.

    Returns
    -------
    coords_rot : (..., 3) jnp.ndarray
    """
    # Axis-angle (Rodrigues) with unit axis; SciPy expects rotvec in radians
    Rmat = _rodrigues(jnp.asarray(vector, dtype=jnp.result_type(coords)), jnp.deg2rad(jnp.asarray(angle)))
    coords = jnp.asarray(coords)
    return coords @ Rmat.T


# -----------------------------------------------------------------------------
# Image/world mapping & interpolation
# -----------------------------------------------------------------------------
def world_to_image_coords(coords: jnp.ndarray, bbox: jnp.ndarray, npix: jnp.ndarray) -> jnp.ndarray:
    """
    Affine-map world coordinates to image index space.

    Parameters
    ----------
    coords : (..., D) jnp.ndarray
        World coordinates.
    bbox : (D, 2) jnp.ndarray
        [min, max] per dimension in world coordinates.
    npix : (D,) jnp.ndarray
        Number of pixels (or samples) per dimension.

    Returns
    -------
    idx_coords : (..., D) jnp.ndarray
        Image coordinates in pixel index space.
    """
    return (coords - bbox[:, 0]) * (npix - 1) / (bbox[:, 1] - bbox[:, 0])


def interpolate_scalar(
    volume: jnp.ndarray,
    coords: jnp.ndarray,
    bbox: jnp.ndarray,
    cval: float = 0.0,
    order: int = 1,
) -> jnp.ndarray:
    """
    Interpolate a scalar field at arbitrary coordinates.

    Parameters
    ----------
    volume : (N1, N2, ..., ND) jnp.ndarray
        Scalar field.
    coords : (..., D) jnp.ndarray
        World coordinates for sampling.
    bbox : (D, 2) jnp.ndarray
        World bounds matching `volume`.
    cval : float, optional
        Fill value outside bounds, by default 0.0.
    order : int, optional
        Spline order (0–5). `1` is linear, by default 1.

    Returns
    -------
    values : (...) jnp.ndarray
        Interpolated values.
    """
    image_coords = jnp.moveaxis(
        world_to_image_coords(coords, bbox, jnp.array(volume.shape)), -1, 0
    )
    return jscp.ndimage.map_coordinates(
        volume, image_coords, order=int(order), cval=float(cval)
    )


def interpolate_vector(
    vector_field: jnp.ndarray,
    coords_cart: jnp.ndarray,
    coords_sph: jnp.ndarray,
    bbox: jnp.ndarray,
    cval: float = 0.0,
    order: int = 1,
) -> jnp.ndarray:
    """
    Interpolate a vector field given in spherical components, and convert to Cartesian.

    Parameters
    ----------
    vector_field : (..., 3) jnp.ndarray
        Vector components in spherical basis [v_phi, v_theta, v_r].
    coords_cart : (..., 3) jnp.ndarray
        Target Cartesian coordinates.
    coords_sph : (..., 3) jnp.ndarray
        Spherical coordinates corresponding to `coords_cart`.
    bbox : (3, 2) jnp.ndarray
        World bounds for the field.
    cval : float, optional
        Fill value outside bounds, by default 0.0.
    order : int, optional
        Interpolation spline order, by default 1 (linear).

    Returns
    -------
    v_cart : (..., 3) jnp.ndarray
        Interpolated vector in Cartesian basis.
    """
    v_sph = jnp.stack(
        [
            interpolate_scalar(vector_field[..., i], coords_sph, bbox, cval, order)
            for i in range(3)
        ],
        axis=-1,
    )
    return spherical_vec_to_cartesian(v_sph, coords_cart, bbox)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def expand_dims(x: jnp.ndarray, ndim: int, axis: int = 0) -> jnp.ndarray:
    """
    Expand dimensions of `x` until it reaches `ndim` total dimensions.

    Parameters
    ----------
    x : jnp.ndarray
        Input array.
    ndim : int
        Target number of dimensions.
    axis : int, optional
        Insert axis position (clamped to current ndim), by default 0.

    Returns
    -------
    y : jnp.ndarray
        Broadcast-ready array with `ndim` dimensions.
    """
    y = x
    for _ in range(ndim - jnp.asarray(y).ndim):
        y = jnp.expand_dims(y, axis=min(axis, jnp.asarray(y).ndim))
    return y
    
def rotate_coords_old(coords, incl, phi, posang):
    rot_matrix = scipy.spatial.transform.Rotation.from_euler('zxz', [posang, incl, -phi], degrees=True).as_matrix()
    coords_rot = jnp.rollaxis(jnp.matmul(rot_matrix, jnp.rollaxis(jnp.array(coords), -1, 1)), -1, 1)
    return coords_rot

def rotate_coords_angles_old(coords, incl, phi):
    rot_matrix = scipy.spatial.transform.Rotation.from_euler('xz', [incl, phi], degrees=True).as_matrix()
    coords_rot = jnp.rollaxis(jnp.matmul(rot_matrix, jnp.atleast_3d(jnp.rollaxis(coords, -1, 1))), -1, 1)
    return coords_rot
    
def rotate_coords_vector_old(coords, vector, angle):
    rot_matrix = scipy.spatial.transform.Rotation.from_rotvec(angle*vector, degrees=True).as_matrix()
    coords_rot = jnp.rollaxis(jnp.matmul(rot_matrix, jnp.atleast_3d(jnp.rollaxis(coords, -1, 1))), -1, 1)
    return coords_rot

__all__ = [
    "read_spherical_amr",
    "cartesian_to_spherical",
    "spherical_vec_to_cartesian",
    "spherical_to_zr",
    "rotate_coords",
    "rotate_coords_angles",
    "rotate_coords_vector",
    "world_to_image_coords",
    "interpolate_scalar",
    "interpolate_vector",
    "expand_dims",
]