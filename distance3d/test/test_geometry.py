import numpy as np
from distance3d.geometry import support_function_box, support_function_cone
from distance3d.random import rand_cone
from numpy.testing import assert_array_almost_equal


def test_box_extreme_along_direction():
    search_direction = np.array([1, 0, 0], dtype=float)
    extreme = support_function_box(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)

    search_direction = np.array([0, 1, 0], dtype=float)
    extreme = support_function_box(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)

    search_direction = np.array([0, 0, 1], dtype=float)
    extreme = support_function_box(
        search_direction, np.eye(4), np.ones(1))
    assert_array_almost_equal(extreme, search_direction)


def test_cone_support_function():
    random_state = np.random.RandomState(2323)
    cone2origin, radius, height = rand_cone(random_state)

    p = support_function_cone(cone2origin[:3, 2], cone2origin, radius, height)
    assert_array_almost_equal(
        p, cone2origin[:3, 3] + height * cone2origin[:3, 2])

    p = support_function_cone(cone2origin[:3, 1], cone2origin, radius, height)
    assert_array_almost_equal(
        p, cone2origin[:3, 3] + radius * cone2origin[:3, 1])

    p = support_function_cone(cone2origin[:3, 0], cone2origin, radius, height)
    assert_array_almost_equal(
        p, cone2origin[:3, 3] + radius * cone2origin[:3, 0])
