import timeit
from functools import partial
import numpy as np
from distance3d.containment import cylinder_aabb
from distance3d.geometry import support_function_cylinder
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
        support_function_cylinder(XM, cylinder2origin, radius, length),
        support_function_cylinder(YM, cylinder2origin, radius, length),
        support_function_cylinder(ZM, cylinder2origin, radius, length),
    ))
    mins = np.min(negative_vertices, axis=0)
    positive_vertices = np.vstack((
        support_function_cylinder(XP, cylinder2origin, radius, length),
        support_function_cylinder(YP, cylinder2origin, radius, length),
        support_function_cylinder(ZP, cylinder2origin, radius, length),
    ))
    maxs = np.max(positive_vertices, axis=0)
    return mins, maxs


def containment_cylinder_aabb_slow(cylinder2origin, radius, length):
    cylinder_aabb_slow(cylinder2origin, radius, length)


times = timeit.repeat(partial(
    containment_cylinder_aabb_slow, cylinder2origin=cylinder2origin,
    radius=radius, length=length), repeat=10, number=10000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
