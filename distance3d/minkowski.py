import numba
import numpy as np


class Simplex:
    def __init__(self):
        self.v = np.empty((4, 3))  # Vertices of the Minkowski difference
        self.v1 = np.empty((4, 3))  # Vertices of collider 1
        self.v2 = np.empty((4, 3))  # Vertices of collider 2
        self.n_points = 0

    def __len__(self):
        return self.n_points

    def add_point(self, v, v1, v2):
        self.v[self.n_points] = v
        self.v1[self.n_points] = v1
        self.v2[self.n_points] = v2
        self.n_points += 1


def support_function(collider1, collider2, search_direction):
    v1 = collider1.support_function(search_direction)
    v2 = collider2.support_function(-search_direction)
    return make_support_point(v1, v2)


@numba.njit(cache=True)
def make_support_point(v1, v2):
    return v1 - v2, v1, v2


def minkowski_sum(vertices1, vertices2):
    """Minkowski sum of two sets of vertices.

    Parameters
    ----------
    vertices1 : array, shape (n_vertices1, 3)
        First set of vertices.

    vertices2 : array, shape (n_vertices2, 3)
        Second set of vertices.

    Returns
    -------
    ms : array, shape (n_vertices1 * n_vertices2, 3)
        Sums of all pairs of vertices from first and second set.
    """
    return np.array([v1 + v2 for v1 in vertices1 for v2 in vertices2])
