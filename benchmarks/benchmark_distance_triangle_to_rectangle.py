import timeit
from functools import partial
import numpy as np
from distance3d.distance import triangle_to_rectangle
from distance3d import random, gjk, geometry


random_state = np.random.RandomState(0)
triangle_points = random.randn_triangle(random_state) * 0.5


def distance_random_triangle_to_rectangle(random_state, triangle_points):
    rectangle_center, rectangle_axes, rectangle_lengths = random.randn_rectangle(
        random_state)
    triangle_to_rectangle(triangle_points, rectangle_center, rectangle_axes, rectangle_lengths)


times = timeit.repeat(partial(
    distance_random_triangle_to_rectangle, random_state=random_state,
    triangle_points=triangle_points), repeat=10, number=200)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")


random_state = np.random.RandomState(0)
triangle_points = random.randn_triangle(random_state) * 0.5


def distance_random_triangle_to_rectangle_gjk(random_state, triangle_points):
    rectangle_center, rectangle_axes, rectangle_lengths = random.randn_rectangle(
        random_state)
    rectangle_points = geometry.convert_rectangle_to_vertices(
        rectangle_center, rectangle_axes, rectangle_lengths)
    gjk.gjk(triangle_points, rectangle_points)


times = timeit.repeat(partial(
    distance_random_triangle_to_rectangle_gjk, random_state=random_state,
    triangle_points=triangle_points), repeat=10, number=200)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
