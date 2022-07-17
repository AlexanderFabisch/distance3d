import timeit
from functools import partial
import numpy as np
from distance3d.geometry import barycentric_coordinates_tetrahedron


random_state = np.random.RandomState(0)


def random_barycentric_coordinates_tetrahedron(random_state):
    p = random_state.randn(3)
    tetrahedron_points = random_state.randn(4, 3)
    barycentric_coordinates_tetrahedron(p, tetrahedron_points)


times = timeit.repeat(partial(
    random_barycentric_coordinates_tetrahedron, random_state=random_state),
    repeat=10, number=100000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
