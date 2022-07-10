import pickle
import numpy as np
from distance3d import gjk, colliders, random


collider_classes = {
    "sphere": (colliders.Sphere, random.rand_sphere),
    "ellipsoid": (colliders.Ellipsoid, random.rand_ellipsoid),
    "capsule": (colliders.Capsule, random.rand_capsule),
    "disk": (colliders.Disk, random.rand_circle),
    "ellipse": (colliders.Ellipse, random.rand_ellipse),
    "cone": (colliders.Cone, random.rand_cone),
    "cylinder": (colliders.Cylinder, random.rand_cylinder),
    "box": (colliders.Box, random.rand_box),
    "mesh": (colliders.MeshGraph, random.randn_convex),
}


def test_gjk_intersection_vs_gjk():
    random_state = np.random.RandomState(29)

    collider_names = list(collider_classes.keys())

    for i in range(1000000):
        print(i)

        collider_name1 = collider_names[random_state.randint(len(collider_names))]
        Class, random_function = collider_classes[collider_name1]
        args1 = random_function(random_state)
        collider1 = Class(*args1)
        collider_name2 = collider_names[random_state.randint(len(collider_names))]
        Class, random_function = collider_classes[collider_name2]
        args2 = random_function(random_state)
        collider2 = Class(*args2)

        dist_original, closest_point1_original, closest_point2_original, _ = gjk.gjk_distance_original(collider1, collider2)
        dist_jolt, closest_point1_jolt, closest_point2_jolt, _ = gjk.gjk_distance_jolt(collider1, collider2)
        gjk_intersection_original = dist_original < 1e-5
        gjk_intersection_libccd = gjk.gjk_intersection_libccd(collider1, collider2)
        gjk_intersection_jolt1 = dist_jolt < 1e-5
        gjk_intersection_jolt2 = gjk.gjk_intersection_jolt(collider1, collider2)

        any_diff = (gjk_intersection_original != gjk_intersection_libccd
                    or gjk_intersection_original != gjk_intersection_jolt1
                    or gjk_intersection_original != gjk_intersection_jolt2)
        if any_diff:
            print(f"{dist_original=}\n{dist_jolt=}\n"
                  f"{gjk_intersection_libccd=}\n{gjk_intersection_jolt2=}")
            save_colliders(collider_name1, args1, collider_name2, args2)
            break


def save_colliders(collider_name1, args1, collider_name2, args2):
    with open("testdata.pickle", "wb") as f:
        pickle.dump((collider_name1, args1, collider_name2, args2), f)


def random_vertices(random_state):
    np.set_printoptions(precision=50)
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
    return convex1, convex2, vertices1, vertices2


test_gjk_intersection_vs_gjk()
