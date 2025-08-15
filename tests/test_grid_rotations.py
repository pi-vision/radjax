import numpy as np
import jax.numpy as jnp
import pytest
from scipy.spatial.transform import Rotation as R
from radjax.core.grid import rotate_coords, rotate_coords_angles, rotate_coords_vector

# New JAX-native implementations under test
from radjax.core.grid import (
    rotate_coords,
    rotate_coords_angles,
    rotate_coords_vector,
)

# -------------------------
# Reference (old) functions
# -------------------------


def test_rotate_coords_matches_old(shape=(10,3), angles=(12.3, 45.0, -7.0)):
    posang, incl, phi = angles
    rng = np.random.default_rng(0)
    coords = jnp.asarray(rng.standard_normal(shape))

    new_out = rotate_coords(coords, incl=incl, phi=phi, posang=posang)

    # SciPy matrix (same parameters as the legacy path)
    Rmat = R.from_euler("zxz", [posang, incl, -phi], degrees=True).as_matrix()

    assert_matches_either(new_out, coords, Rmat, msg="ZXZ")

def test_rotate_coords_angles_matches_old(shape=(12,3), angles=(10.0, -30.0)):
    incl, phi = angles
    rng = np.random.default_rng(1)
    coords = jnp.asarray(rng.standard_normal(shape))

    new_out = rotate_coords_angles(coords, incl=incl, phi=phi)
    Rmat = R.from_euler("xz", [incl, phi], degrees=True).as_matrix()

    assert_matches_either(new_out, coords, Rmat, msg="XZ")


# -------------------------
# Helpers
# -------------------------

def _rng_points(shape, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal(shape).astype(np.float64)
    return jnp.asarray(pts)

def assert_matches_either(new_out, coords, Rmat, *, msg=""):
    """Pass if new_out matches coords @ R^T (row) OR (R @ coords^T)^T (col)."""
    row_pred = jnp.asarray(coords) @ jnp.asarray(Rmat).T
    col_pred = (jnp.asarray(Rmat) @ jnp.asarray(coords).T).T

    a = np.asarray(new_out)
    b_row = np.asarray(row_pred)
    b_col = np.asarray(col_pred)

    ok_row = np.allclose(a, b_row, rtol=RTOL, atol=ATOL)
    ok_col = np.allclose(a, b_col, rtol=RTOL, atol=ATOL)
    if ok_row or ok_col:
        return

    # Helpful diagnostics if both fail
    diff_row = np.max(np.abs(a - b_row))
    diff_col = np.max(np.abs(a - b_col))
    raise AssertionError(
        f"{msg} neither form matched within tol "
        f"(max|Δ| row={diff_row:.3e}, col={diff_col:.3e}; "
        f"rtol={RTOL}, atol={ATOL})"
    )
ATOL = 1e-8
RTOL = 1e-8

# ---------------------------
# Round-trip: ZXZ (rotate_coords)
# ---------------------------
@pytest.mark.parametrize("shape", [(10, 3), (4, 5, 3)])
@pytest.mark.parametrize("angles", [(12.3, 45.0, -7.0), (30.0, 0.0, 10.0)])
def test_roundtrip_rotate_coords(shape, angles):
    posang, incl, phi = angles
    pts = _rng_points(shape, seed=1)

    fwd = rotate_coords(pts, incl=incl, phi=phi, posang=posang)

    # Inverse: Rz(phi) @ Rx(-incl) @ Rz(-posang)
    inv = rotate_coords(fwd, incl=-incl, phi=posang, posang=phi)

    np.testing.assert_allclose(np.asarray(inv), np.asarray(pts), rtol=RTOL, atol=ATOL)

# ---------------------------
# Round-trip: XZ (rotate_coords_angles)
# ---------------------------
@pytest.mark.parametrize("shape", [(12, 3), (3, 4, 3)])
@pytest.mark.parametrize("angles", [(0.0, 0.0), (10.0, -30.0), (90.0, 45.0), (12.3, 123.4)])
def test_roundtrip_rotate_coords_angles(shape, angles):
    incl, phi = angles
    pts = _rng_points(shape, seed=2)

    fwd = rotate_coords_angles(pts, incl=incl, phi=phi)

    # Build the same rotation matrix as the forward path (Rx(incl) @ Rz(phi))
    Rmat = R.from_euler("xz", [incl, phi], degrees=True).as_matrix()

    # Forward uses row-vector convention: pts @ R^T
    # Inverse is multiply by R (since (R^T)^-1 = R)
    back = fwd @ jnp.asarray(Rmat)

    np.testing.assert_allclose(np.asarray(back), np.asarray(pts), rtol=RTOL, atol=ATOL)

# ---------------------------
# Round-trip: Axis–angle (rotate_coords_vector)
# ---------------------------
@pytest.mark.parametrize("shape", [(9, 3), (2, 3, 3)])
@pytest.mark.parametrize("axis, angle_deg", [
    ([1.0, 0.0, 0.0], 0.0),
    ([1.0, 0.0, 0.0], 90.0),
    ([0.0, 1.0, 0.0], 45.0),
    ([0.0, 0.0, 1.0], 123.4),
    ([1.0, 2.0, 3.0], 33.0),
])
def test_roundtrip_rotate_coords_vector(shape, axis, angle_deg):
    pts = _rng_points(shape, seed=3)
    axis = jnp.asarray(axis, dtype=pts.dtype)

    fwd = rotate_coords_vector(pts, vector=axis, angle=angle_deg)
    back = rotate_coords_vector(fwd, vector=axis, angle=-angle_deg)

    np.testing.assert_allclose(np.asarray(back), np.asarray(pts), rtol=RTOL, atol=ATOL)


# -------------------------
# Tests: ZXZ Euler rotation
# -------------------------

@pytest.mark.parametrize("shape", [(10, 3), (4, 5, 3)])
@pytest.mark.parametrize("angles", [(0.0, 0.0, 0.0), (30.0, 45.0, -10.0), (90.0, 0.0, 12.3), (12.3, 89.9, 123.4)])
def test_rotate_coords_matches_old(shape, angles):
    coords = _rng_points(shape, seed=1)
    posang, incl, phi = angles  # note the JAX impl uses (posang, incl, -phi) internally

    new_out = rotate_coords(coords, incl=incl, phi=phi, posang=posang)
    ref_out = rotate_coords_old(coords, incl=incl, phi=phi, posang=posang)

    np.testing.assert_allclose(np.asarray(new_out), np.asarray(ref_out), rtol=RTOL, atol=ATOL)

# Identity check
def test_rotate_coords_identity():
    coords = _rng_points((7, 3), seed=2)
    out = rotate_coords(coords, incl=0.0, phi=0.0, posang=0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(coords), rtol=RTOL, atol=ATOL)

# Tests: XZ Euler rotation
# -------------------------

@pytest.mark.parametrize("shape", [(12, 3), (2, 3, 3)])
@pytest.mark.parametrize("angles", [(0.0, 0.0), (10.0, -30.0), (90.0, 45.0), (12.3, 123.4)])
def test_rotate_coords_angles_matches_old(shape, angles):
    coords = _rng_points(shape, seed=3)
    incl, phi = angles

    new_out = rotate_coords_angles(coords, incl=incl, phi=phi)
    ref_out = rotate_coords_angles_old(coords, incl=incl, phi=phi)

    np.testing.assert_allclose(np.asarray(new_out), np.asarray(ref_out), rtol=RTOL, atol=ATOL)

def test_rotate_coords_angles_identity():
    coords = _rng_points((5, 3), seed=4)
    out = rotate_coords_angles(coords, incl=0.0, phi=0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(coords), rtol=RTOL, atol=ATOL)


# -------------------------
# Tests: Axis-angle rotation
# -------------------------

@pytest.mark.parametrize("shape", [(9, 3), (3, 4, 3)])
@pytest.mark.parametrize(
    "axis, angle_deg",
    [
        ([1.0, 0.0, 0.0], 0.0),
        ([1.0, 0.0, 0.0], 90.0),
        ([0.0, 1.0, 0.0], 45.0),
        ([0.0, 0.0, 1.0], 123.4),
        ([1.0, 2.0, 3.0], 33.0),  # will be normalized for OLD to avoid scale bug
    ],
)
def test_rotate_coords_vector_matches_old_with_unit_axis(shape, axis, angle_deg):
    coords = _rng_points(shape, seed=5)
    axis = np.asarray(axis, dtype=np.float64)
    # Normalize axis so OLD behavior (angle scaled by |axis|) matches the new function:
    axis_unit = axis / (np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else 1.0)

    new_out = rotate_coords_vector(coords, vector=axis_unit, angle=angle_deg)
    ref_out = rotate_coords_vector_old(coords, vector=axis_unit, angle=angle_deg)

    np.testing.assert_allclose(np.asarray(new_out), np.asarray(ref_out), rtol=RTOL, atol=ATOL)

def test_rotate_coords_vector_identity():
    coords = _rng_points((6, 3), seed=6)
    axis = np.array([0.0, 0.0, 1.0])
    out = rotate_coords_vector(coords, vector=axis, angle=0.0)
    np.testing.assert_allclose(np.asarray(out), np.asarray(coords), rtol=RTOL, atol=ATOL)
