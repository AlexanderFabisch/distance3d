import numba
import numpy as np
from ..utils import norm_vector, EPSILON


# replaces from numba.np.extensions import cross2d, which seems to have a bug
# when called with NUMBA_DISABLE_JIT=1
@numba.njit(
    numba.float64(numba.float64[::1], numba.float64[::1]),
    cache=True)
def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


@numba.njit(
    numba.float64[::1](numba.float64[::1], numba.float64[::1]),
    cache=True)
def intersect_two_halfplanes(halfplane1, halfplane2):
    denom = cross2d(halfplane1[2:], halfplane2[2:])
    if abs(denom) < EPSILON:
        return np.empty(0, dtype=np.dtype("float"))
    t = cross2d((halfplane2[:2] - halfplane1[:2]), halfplane2[2:]) / denom
    return halfplane1[:2] + halfplane1[2:] * t


@numba.njit(
    numba.bool_(numba.float64[::1], numba.float64[::1]),
    cache=True)
def point_outside_of_halfplane(halfplane, point):
    return cross2d(halfplane[2:], point - halfplane[:2]) < -EPSILON


@numba.njit(
    numba.float64[:, :](numba.float64[:, ::1]), cache=True)
def intersect_halfplanes(halfplanes):
    """Find polygon points by halfplane intersection.

    Parameters
    ----------
    halfplanes : array, shape (n_halfplanes, 4)
        Halfplanes in contact plane. Each halfplane is defined by a point
        p and a direction pq.

    Returns
    -------
    points : list
        Points of the polygon.
    """
    # reserve more space than required, there might be duplicates
    points = np.empty((2 * len(halfplanes), 2))
    n_intersections = 0
    for i in range(len(halfplanes)):
        for j in range(i + 1, len(halfplanes)):
            p = intersect_two_halfplanes(halfplanes[i], halfplanes[j])
            if len(p) == 0:  # parallel halfplanes
                continue
            valid = True
            for k in range(len(halfplanes)):
                if k != i and k != j and point_outside_of_halfplane(
                        halfplanes[k], p):
                    valid = False
                    break
            if valid:
                points[n_intersections] = p
                n_intersections += 1
    assert n_intersections < len(points)
    return points[:n_intersections]


def plot_halfplanes_and_intersections(halfplanes, points=None, xlim=None, ylim=None):  # pragma: no cover
    import matplotlib.pyplot as plt
    if points is None:
        scale = 1.0
    else:
        center = np.mean(points, axis=0)
        max_distance = max(np.linalg.norm(points - center, axis=1))
        scale = 10.0 * max_distance
    plt.figure()
    ax = plt.subplot(111, aspect="equal")
    for i, halfplane in enumerate(halfplanes):
        c = "r" if i < len(halfplanes) // 2 else "b"
        plot_halfplane(halfplane, ax, c, 0.5, scale)
    if points is not None:
        colors = ["r", "g", "b", "orange", "magenta", "brown", "k"][:len(points)]
        if len(colors) < len(points):
            colors.extend(["k"] * (len(points) - len(colors)))
        plt.scatter(points[:, 0], points[:, 1], c=colors, s=100)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.show()


def plot_halfplane(halfplane, ax, c, alpha, scale):  # pragma: no cover
    line = (halfplane[:2] + np.linspace(-scale, scale, 101)[:, np.newaxis]
            * norm_vector(halfplane[2:]))
    ax.plot(line[:, 0], line[:, 1], lw=3, c=c, alpha=alpha)
    normal2d = np.array([-halfplane[3], halfplane[2]])
    for p in line[::10]:
        normal = (p + np.linspace(0.0, 0.1 * scale, 101)[:, np.newaxis]
                  * norm_vector(normal2d))
        ax.plot(normal[:, 0], normal[:, 1], c=c, alpha=0.5 * alpha)
