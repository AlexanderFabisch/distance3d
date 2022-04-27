import timeit
from functools import partial
import numpy as np
from distance3d.containment import cylinder_aabb
from distance3d.geometry import cylinder_extreme_along_direction
from distance3d import random


random_state = np.random.RandomState(0)
cylinder2origin, radius, length = random.rand_cylinder(random_state)


def containment_cylinder_aabb(cylinder2origin, radius, length):
    cylinder_aabb(cylinder2origin, radius, length)


times = timeit.repeat(partial(
    containment_cylinder_aabb, cylinder2origin=cylinder2origin, radius=radius,
    length=length), repeat=10, number=10000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")


XM = np.array([-1.0, 0.0, 0.0])
YM = np.array([0.0, -1.0, 0.0])
ZM = np.array([0.0, 0.0, -1.0])
XP = np.array([1.0, 0.0, 0.0])
YP = np.array([0.0, 1.0, 0.0])
ZP = np.array([0.0, 0.0, 1.0])


def cylinder_aabb_slow(cylinder2origin, radius, length):
    negative_vertices = np.vstack((
        cylinder_extreme_along_direction(XM, cylinder2origin, radius, length),
        cylinder_extreme_along_direction(YM, cylinder2origin, radius, length),
        cylinder_extreme_along_direction(ZM, cylinder2origin, radius, length),
    ))
    mins = np.min(negative_vertices, axis=0)
    positive_vertices = np.vstack((
        cylinder_extreme_along_direction(XP, cylinder2origin, radius, length),
        cylinder_extreme_along_direction(YP, cylinder2origin, radius, length),
        cylinder_extreme_along_direction(ZP, cylinder2origin, radius, length),
    ))
    maxs = np.max(positive_vertices, axis=0)
    return mins, maxs


def containment_cylinder_aabb_slow(cylinder2origin, radius, length):
    cylinder_aabb_slow(cylinder2origin, radius, length)


times = timeit.repeat(partial(
    containment_cylinder_aabb_slow, cylinder2origin=cylinder2origin,
    radius=radius, length=length), repeat=10, number=10000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
