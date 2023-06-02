import numpy as np
from _pytest.python_api import approx
from numpy.testing import assert_array_almost_equal

from distance3d import colliders, gjk, geometry, random, distance
from distance3d.gjk._gjk_nesterov_accelerated import gjk_nesterov_accelerated_iterations, gjk_nesterov_accelerated
from distance3d.gjk._gjk_nesterov_accelerated_primitives import gjk_nesterov_accelerated_primitives_iterations, \
    gjk_nesterov_accelerated_primitives


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


def test_iterations():
    sphere1 = colliders.Sphere(center=np.array([0, 0, 0], dtype=float), radius=1.0)

    assert gjk_nesterov_accelerated_iterations(sphere1, sphere1) == 0
    assert gjk_nesterov_accelerated_primitives_iterations(sphere1, sphere1) == 0


def test_to_many_iterations():
    A2B = np.eye(4)
    A2B[:3, 3] = np.array([3, 0, 0])

    cylinder1 = colliders.Cylinder(cylinder2origin=np.eye(4), radius=1.0, length=0.5)
    cylinder2 = colliders.Cylinder(cylinder2origin=A2B, radius=1.0, length=0.5)

    assert not gjk_nesterov_accelerated(cylinder1, cylinder2, max_interations=1)[0]
    assert not gjk_nesterov_accelerated_primitives(cylinder1, cylinder2, max_interations=1)[0]
