import pickle
import numpy as np
from distance3d import gjk, colliders


def test_gjk_intersection_vs_gjk():
    random_state = np.random.RandomState(27)

    for i in range(1000000):
        print(i)
        vertices1 = random_state.rand(6, 3) * np.array([[2, 2, 2]])
        vertices2 = random_state.rand(6, 3) * np.array([[1, 1, 1]])

        if random_state.rand() > 0.5:
            vertices2[3] = vertices1[0]
        if random_state.rand() > 0.75:
            vertices2[2] = 0.5 * (vertices1[1] + vertices1[2])
        if random_state.rand() > 0.875:
            vertices2[1] = (vertices1[3] + vertices1[4] + vertices1[5]) / 3.0
        if random_state.rand() > 0.9375:
            vertices2[0] = (vertices1[0] + vertices1[3] + vertices1[4] + vertices1[5]) / 4.0

        convex1 = colliders.ConvexHullVertices(vertices1)
        convex2 = colliders.ConvexHullVertices(vertices2)

        dist_original, closest_point1_original, closest_point2_original, _ = gjk.gjk_distance_original(convex1, convex2)
        dist_jolt, closest_point1_jolt, closest_point2_jolt, _ = gjk.gjk_distance_jolt(convex1, convex2)
        gjk_intersection_original = dist_original < 1e-10
        gjk_intersection_libccd = gjk.gjk_intersection_libccd(convex1, convex2)
        gjk_intersection_jolt1 = dist_jolt < 1e-10
        gjk_intersection_jolt2 = gjk.gjk_intersection_jolt(convex1, convex2)
        np.set_printoptions(precision=50)
        any_diff = (gjk_intersection_original != gjk_intersection_libccd
                    or gjk_intersection_original != gjk_intersection_jolt1
                    or gjk_intersection_original != gjk_intersection_jolt2)
        if any_diff:
            print(f"{vertices1=}\n{vertices2=}"
                  f"\n{dist_original=}\n{dist_jolt=}\n"
                  f"{gjk_intersection_libccd=}\n{gjk_intersection_jolt2=}")
            with open("testdata.pickle", "wb") as f:
                pickle.dump((vertices1, vertices2), f)
            break


test_gjk_intersection_vs_gjk()
