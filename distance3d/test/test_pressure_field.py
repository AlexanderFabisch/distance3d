import numpy as np
from distance3d import pressure_field
from numpy.testing import assert_array_almost_equal


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
