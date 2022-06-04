import timeit
from functools import partial
import numpy as np
from distance3d.containment import capsule_aabb
from distance3d.geometry import support_function_capsule
from distance3d import random


random_state = np.random.RandomState(0)
capsule2origin, radius, height = random.rand_capsule(random_state)


def containment_capsule_aabb(capsule2origin, radius, height):
    capsule_aabb(capsule2origin, radius, height)


times = timeit.repeat(partial(
    containment_capsule_aabb, capsule2origin=capsule2origin, radius=radius,
    height=height), repeat=10, number=10000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")


XM = np.array([-1.0, 0.0, 0.0])
YM = np.array([0.0, -1.0, 0.0])
ZM = np.array([0.0, 0.0, -1.0])
XP = np.array([1.0, 0.0, 0.0])
YP = np.array([0.0, 1.0, 0.0])
ZP = np.array([0.0, 0.0, 1.0])


def capsule_aabb_slow(capsule2origin, radius, height):
    negative_vertices = np.vstack((
        support_function_capsule(XM, capsule2origin, radius, height),
        support_function_capsule(YM, capsule2origin, radius, height),
        support_function_capsule(ZM, capsule2origin, radius, height),
    ))
    mins = np.min(negative_vertices, axis=0)
    positive_vertices = np.vstack((
        support_function_capsule(XP, capsule2origin, radius, height),
        support_function_capsule(YP, capsule2origin, radius, height),
        support_function_capsule(ZP, capsule2origin, radius, height),
    ))
    maxs = np.max(positive_vertices, axis=0)
    return mins, maxs


def containment_capsule_aabb_slow(capsule2origin, radius, height):
    capsule_aabb_slow(capsule2origin, radius, height)


times = timeit.repeat(partial(
    containment_capsule_aabb_slow, capsule2origin=capsule2origin,
    radius=radius, height=height), repeat=10, number=10000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
