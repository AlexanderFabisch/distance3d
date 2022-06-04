import numpy as np
from distance3d.mesh import MeshHillClimbingSupportFunction, MeshSupportFunction
from distance3d import random
from numpy.testing import assert_array_almost_equal


def test_support_function():
    random_state = np.random.RandomState(2323)

    for _ in range(10):
        mesh2origin, vertices, triangles = random.randn_convex(random_state)
        support_function1 = MeshSupportFunction(mesh2origin, vertices, triangles)
        support_function2 = MeshHillClimbingSupportFunction(mesh2origin, vertices, triangles)
        for _ in range(20):
            search_direction = random.randn_direction(random_state)
            idx1, point1 = support_function1(search_direction)
            idx2, point2 = support_function2(search_direction)
            assert idx1 == idx2
            assert_array_almost_equal(point1, point2)
