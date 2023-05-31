import numpy as np
from distance3d import colliders, gjk, geometry, random, distance
from pytest import approx
from numpy.testing import assert_array_almost_equal


def test_compare_gjk_intersection_flavours_with_random_shapes():
    random_state = np.random.RandomState(84)
    shape_names = list(["sphere", "ellipsoid", "capsule", "cone", "cylinder", "box"])
    k = 100

    for i in range(k):
        print(i)

        shape1 = shape_names[random_state.randint(len(shape_names))]
        args1 = random.RANDOM_GENERATORS[shape1](random_state)
        shape2 = shape_names[random_state.randint(len(shape_names))]
        args2 = random.RANDOM_GENERATORS[shape2](random_state)
        collider1 = colliders.COLLIDERS[shape1](*args1)
        collider2 = colliders.COLLIDERS[shape2](*args2)

        intersection_jolt = gjk.gjk_intersection_jolt(collider1, collider2)
        intersection_libccd = gjk.gjk_intersection_libccd(collider1, collider2)
        intersection_nesterov = gjk.gjk_nesterov_accelerated(collider1, collider2, use_nesterov_acceleration=False)[0]
        intersection_nesterov_primitives = gjk.gjk_nesterov_accelerated_primitives(collider1, collider2, use_nesterov_acceleration=False)[0]

        assert intersection_jolt == intersection_libccd
        assert intersection_nesterov == intersection_libccd
        assert intersection_nesterov_primitives == intersection_libccd


def test_compare_gjk_intersection_flavours_with_random_shapes_with_nesterov_acceleration():
    random_state = np.random.RandomState(84)
    shape_names = list(["sphere", "ellipsoid", "capsule", "cone", "cylinder", "box"])
    k = 100

    for i in range(k):
        print(i)

        shape1 = shape_names[random_state.randint(len(shape_names))]
        args1 = random.RANDOM_GENERATORS[shape1](random_state)
        shape2 = shape_names[random_state.randint(len(shape_names))]
        args2 = random.RANDOM_GENERATORS[shape2](random_state)
        collider1 = colliders.COLLIDERS[shape1](*args1)
        collider2 = colliders.COLLIDERS[shape2](*args2)

        intersection_jolt = gjk.gjk_intersection_jolt(collider1, collider2)
        intersection_libccd = gjk.gjk_intersection_libccd(collider1, collider2)
        intersection_nesterov = gjk.gjk_nesterov_accelerated(collider1, collider2, use_nesterov_acceleration=False)[0]
        intersection_nesterov_primitives = gjk.gjk_nesterov_accelerated_primitives(collider1, collider2, use_nesterov_acceleration=True)[0]

        assert intersection_jolt == intersection_libccd
        assert intersection_nesterov == intersection_libccd
        assert intersection_nesterov_primitives == intersection_libccd
