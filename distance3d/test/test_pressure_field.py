import numpy as np
from distance3d import pressure_field, utils
from numpy.testing import assert_array_almost_equal
from pytest import approx


def test_intersect_halfplanes():
    halfplanes = [
        pressure_field.HalfPlane(np.array([0.5, 0.0]), np.array([-1.0, 0.0])),
        pressure_field.HalfPlane(np.array([0.0, 0.5]), np.array([0.0, -1.0])),
        pressure_field.HalfPlane(np.array([-0.5, 0.0]), np.array([1.0, 0.0])),
        pressure_field.HalfPlane(np.array([0.0, -0.5]), np.array([0.0, 1.0])),
    ]
    polygon, _ = pressure_field.intersect_halfplanes(halfplanes)
    expected_polygon = np.array([
        [0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, -0.5],
        [0.5, -0.5]
    ])
    assert_array_almost_equal(polygon, expected_polygon)


def test_plane_projection():
    random_state = np.random.RandomState(3)
    plane_hnf = random_state.randn(4)
    plane_hnf = np.array([0.0, 1.0, 1.0, 0.5])
    plane_hnf[:3] = utils.norm_vector(plane_hnf[:3])
    x, y = utils.plane_basis_from_normal(plane_hnf[:3])
    plane2world = np.column_stack((x, y, plane_hnf[:3]))
    plane_offset = plane_hnf[:3] * plane_hnf[3]

    cart2plane, plane2cart, plane2cart_offset = pressure_field.plane_projection(plane_hnf)

    for _ in range(10):
        x_plane_cart = random_state.randn(3)
        x_cart = plane2world.dot(x_plane_cart) + plane_offset
        x_plane = cart2plane.dot(x_cart)
        x_cart2 = plane2cart.dot(x_plane) + plane2cart_offset
        print(x_plane_cart)
        print(x_cart)
        print(x_cart2)
        assert approx(np.linalg.norm(x_cart - x_cart2)) == abs(plane_hnf[:3].dot(x_cart) - plane_hnf[:3].dot(x_cart2))
