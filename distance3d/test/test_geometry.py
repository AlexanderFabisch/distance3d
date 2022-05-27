import numpy as np
from distance3d.geometry import box_extreme_along_direction
from numpy.testing import assert_array_almost_equal


def test_box_extreme_along_direction():
    search_direction = np.array([1, 0, 0])
    extreme = box_extreme_along_direction(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)

    search_direction = np.array([0, 1, 0])
    extreme = box_extreme_along_direction(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)

    search_direction = np.array([0, 0, 1])
    extreme = box_extreme_along_direction(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)
