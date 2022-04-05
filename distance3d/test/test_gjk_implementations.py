import numpy as np
from distance3d import gjk, colliders
from distance3d._gjk_python import distance_subalgorithm_python
from distance3d._gjk import distance_subalgorithm
from numpy.testing import assert_array_almost_equal
from pytest import approx


def test_compare_python_and_c():
    random_state = np.random.RandomState(10)
    for _ in range(10):
        collider1 = colliders.Convex(random_state.randn(100, 3) + 5.0)
        collider2 = colliders.Convex(random_state.randn(100, 3) - 5.0)
        dc, cp1c, cp2c, sc = gjk.gjk_with_simplex(
            collider1, collider2,
            distance_subalgorithm=distance_subalgorithm)
        dp, cp1p, cp2p, sp = gjk.gjk_with_simplex(
            collider1, collider2,
            distance_subalgorithm=distance_subalgorithm_python)
        assert approx(dc) == dp
        assert_array_almost_equal(cp1c, cp1p)
        assert_array_almost_equal(cp2c, cp2p)
        assert_array_almost_equal(sc, sp)
