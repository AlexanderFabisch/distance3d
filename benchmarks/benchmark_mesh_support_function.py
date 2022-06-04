import timeit
import time
from functools import partial
import numpy as np
from distance3d import random
from distance3d.mesh import MeshSupportFunction, MeshHillClimbingSupportFunction


def run_support_function(random_state, support_function):
    search_direction = random.randn_direction(random_state)
    support_function(search_direction)


random_state = np.random.RandomState(0)
mesh2origin, vertices, triangles = random.randn_convex(random_state, n_vertices=50000)

support_function = MeshSupportFunction(mesh2origin, vertices, triangles)
times = timeit.repeat(partial(
    run_support_function, random_state=random_state,
    support_function=support_function), repeat=10, number=1000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

random_state = np.random.RandomState(0)
mesh2origin, vertices, triangles = random.randn_convex(random_state, n_vertices=50000)

start = time.time()
support_function = MeshHillClimbingSupportFunction(mesh2origin, vertices, triangles)
end = time.time()
print(end - start)
times = timeit.repeat(partial(
    run_support_function, random_state=random_state,
    support_function=support_function), repeat=10, number=1000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
