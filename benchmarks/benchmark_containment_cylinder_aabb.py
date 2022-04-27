import timeit
from functools import partial
import numpy as np
from distance3d.containment import cylinder_aabb
from distance3d import random


random_state = np.random.RandomState(0)


def containment_cylinder_aabb(random_state):
    cylinder2origin, radius, length = random.rand_cylinder(random_state)
    cylinder_aabb(cylinder2origin, radius, length)


times = timeit.repeat(partial(
    containment_cylinder_aabb, random_state=random_state),
    repeat=10, number=10000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
