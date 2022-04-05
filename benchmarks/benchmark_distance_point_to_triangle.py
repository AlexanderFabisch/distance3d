import timeit
from functools import partial
import numpy as np
from distance3d.distance import point_to_triangle
from distance3d import random


random_state = np.random.RandomState(0)
triangle_points = random.randn_triangle(random_state) * 0.5


def distance_random_point_to_triangle(random_state, triangle_points):
    point = random_state.randn(3)
    point_to_triangle(point, triangle_points)


times = timeit.repeat(partial(
    distance_random_point_to_triangle, random_state=random_state,
    triangle_points=triangle_points), repeat=10, number=10000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
