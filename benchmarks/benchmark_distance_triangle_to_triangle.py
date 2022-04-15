import timeit
from functools import partial
import numpy as np
from distance3d.distance import triangle_to_triangle
from distance3d import random, gjk, geometry


random_state = np.random.RandomState(0)
triangle_points = random.randn_triangle(random_state) * 0.5


def distance_random_triangle_to_triangle(random_state, triangle_points):
    triangle_points2 = random.randn_triangle(random_state)
    triangle_to_triangle(triangle_points2, triangle_points)


times = timeit.repeat(partial(
    distance_random_triangle_to_triangle, random_state=random_state,
    triangle_points=triangle_points), repeat=10, number=200)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")


random_state = np.random.RandomState(0)
triangle_points = random.randn_triangle(random_state) * 0.5


def distance_random_triangle_to_triangle_gjk(random_state, triangle_points):
    triangle_points2 = random.randn_triangle(random_state)
    gjk.gjk(triangle_points2, triangle_points)


times = timeit.repeat(partial(
    distance_random_triangle_to_triangle_gjk, random_state=random_state,
    triangle_points=triangle_points), repeat=10, number=200)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
