import numpy as np
from distance3d.utils import norm_vector
from numpy.testing import assert_array_almost_equal
from pytest import approx


def test_norm_vector():
    assert_array_almost_equal(norm_vector(np.array([0, 1, 0], dtype=float)), [0, 1, 0])
    assert_array_almost_equal(norm_vector(np.array([0, 2, 0], dtype=float)), [0, 1, 0])
    assert_array_almost_equal(norm_vector(np.array([0, 0, 0], dtype=float)), [0, 0, 0])
    assert approx(1.0) == np.linalg.norm(norm_vector(np.array([1, -2.3, 1.3], dtype=float)))
