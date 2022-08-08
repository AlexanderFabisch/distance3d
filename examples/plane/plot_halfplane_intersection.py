"""
======================
Halfplane Intersection
======================
"""
print(__doc__)

import numpy as np
from distance3d import pressure_field, benchmark


# TODO remove or reuse function
# source: https://cp-algorithms.com/geometry/halfplane-intersection.html#direct-implementation
from distance3d.pressure_field._halfplanes import intersect_two_halfplanes, point_outside_of_halfplane
from collections import deque
def intersect_halfplanes2(halfplanes):  # TODO can we modify this to work with parallel lines?
    angles = np.arctan2(halfplanes[:, 3], halfplanes[:, 2])
    halfplanes = halfplanes[np.argsort(angles)]

    result = deque()
    for hp in halfplanes:
        try:
            while len(result) >= 2 and point_outside_of_halfplane(hp, intersect_two_halfplanes(result[-1], result[-2])):
                result.pop()
            while len(result) >= 2 and point_outside_of_halfplane(hp, intersect_two_halfplanes(result[0], result[1])):
                result.popleft()
            result.append(hp)
        except TypeError:
            #if len(result) > 0 and abs(cross2d(hp[2:], result[-1][2:])) < 1e-6:
            # Opposite parallel halfplanes that ended up checked against each other
            if np.dot(hp[2:], result[-1][2:]) < 0.0:
                return None
            # Same direction halfplane: keep only the leftmost halfplane
            if point_outside_of_halfplane(hp, result[-1][:2]):
                result.pop()
                result.append(hp)
            else:
                continue

    while len(result) >= 3:
        try:
            if point_outside_of_halfplane(result[0], intersect_two_halfplanes(result[-1], result[-2])):
                result.pop()
            else:
                break
        except TypeError:
            result.pop()
    while len(result) >= 3 and point_outside_of_halfplane(result[-1], intersect_two_halfplanes(result[0], result[1])):
        result.popleft()

    if len(result) < 3:
        return None
    else:
        polygon = np.row_stack([intersect_two_halfplanes(result[i], result[(i + 1) % len(result)])
                                for i in range(len(result))])
        return polygon


random_state = np.random.RandomState(0)
n_halfplanes = 10
x = random_state.rand(n_halfplanes) * 2.0 * np.pi
p = np.column_stack((np.cos(x), np.sin(x)))
p += random_state.rand(*p.shape) * 0.1
angles = np.arctan2(p[:, 1], p[:, 0])
p = p[np.argsort(angles)]
pq = p[np.arange(n_halfplanes)] - p[np.arange(n_halfplanes) - 1]
halfplanes = np.hstack((p, pq))

#"""
halfplanes = np.array([[-0.00373259, -0.09755581,  0.01122989,  0.00524249],
       [-0.00650511, -0.09267489,  0.00277252, -0.00488092],
       [ 0.0074973 , -0.09231332, -0.01400241, -0.00036157],
       [-0.02538515, -0.10766395,  0.02257121,  0.010537  ],
       [-0.00981839, -0.0689421 , -0.03087142, -0.02461552],
       [-0.04068981, -0.09355762,  0.01530466, -0.01410634],
       [-0.00281394, -0.09712695, -0.00700445,  0.02818485]])
#"""
#pressure_field.plot_halfplanes_and_intersections(
#    halfplanes, halfplanes[:, :2], xlim=(-2, 2), ylim=(-2, 2))

timer = benchmark.Timer()
timer.start("intersect_halfplanes")
polygon = np.asarray(pressure_field.intersect_halfplanes(halfplanes))
print(f"{timer.stop('intersect_halfplanes')} s")
print(polygon)
timer.start("intersect_halfplanes2")
polygon2 = np.asarray(intersect_halfplanes2(halfplanes))
print(f"{timer.stop('intersect_halfplanes2')} s")
print(polygon2)
pressure_field.plot_halfplanes_and_intersections(
    halfplanes, polygon, xlim=(-2, 2), ylim=(-2, 2))
pressure_field.plot_halfplanes_and_intersections(
    halfplanes, polygon2, xlim=(-2, 2), ylim=(-2, 2))
