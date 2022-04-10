import numpy as np
from distance3d import gjk, epa
from numpy.testing import assert_array_almost_equal


def test_epa():
    vertices = np.array([
        [1.76405235, 0.40015721, 0.97873798],
        [2.2408932, 1.86755799, -0.97727788],
        [0.4105985, 0.14404357, 1.45427351],
        [0.33367433, 1.49407907, -0.20515826],
        [0.3130677, -0.85409574, -2.55298982],
        [2.26975462, -1.45436567, 0.04575852],
        [-0.18718385, 1.53277921, 1.46935877]])
    vertices2 = np.array([
        [-2.32605299, -0.31242692, 0.07599278],
        [0.88503416, 1.23786508, -0.467683],
        [-0.64755927, -1.01306774, -1.50037412],
        [-2.05152671, 1.98626062, -0.59000837],
        [-0.78333082, -1.21731013, 0.69713417],
        [-1.95915437, -0.17725505, -0.97582275],
        [0.04164598, -0.47531991, -1.26098837],
        [-0.37343875, 0.4638171, -0.01383896],
        [-0.04278462, -0.59883687, -0.44309735]])
    dist, p1, p2, simplex = gjk.gjk_with_simplex(
        gjk.Convex(vertices), gjk.Convex(vertices2))
    assert_array_almost_equal(p1, p2)
    mtv, _, success = epa.epa(
        simplex, gjk.Convex(vertices), gjk.Convex(vertices2))
    assert success
    assert_array_almost_equal(mtv, np.array([-0.387287,  0.179576, -0.176204]))
