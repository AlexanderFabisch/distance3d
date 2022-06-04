import numpy as np
from distance3d import colliders, mpr, gjk, epa


def test_intersection_mpr_vs_gjk():
    random_state = np.random.RandomState(24)

    for _ in range(50):
        vertices1 = random_state.rand(6, 3) * np.array([[2, 2, 2]])
        convex1 = colliders.Convex(vertices1)

        vertices2 = random_state.rand(6, 3) * np.array([[1, 1, 1]])
        convex2 = colliders.Convex(vertices2)

        dist, closest_point1, closest_point2, _ = gjk.gjk_with_simplex(
            convex1, convex2)
        gjk_intersection = dist < 1e-16
        mpr_intersection = mpr.mpr_intersection(convex1, convex2)
        assert gjk_intersection == mpr_intersection
