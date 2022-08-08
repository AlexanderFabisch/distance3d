"""
======================
Halfplane Intersection
======================
"""
print(__doc__)

import numpy as np
from distance3d import pressure_field, benchmark


# TODO remove or reuse function
from collections import deque
from distance3d.pressure_field._halfplanes import intersect_two_halfplanes, point_outside_of_halfplane
def intersect_halfplanes2(halfplanes):  # TODO can we modify this to work with parallel lines?
    dq = deque()
    for hp in halfplanes:
        while len(dq) >= 2 and point_outside_of_halfplane(hp, intersect_two_halfplanes(dq[-1], dq[-2])):
            dq.pop()
        while len(dq) >= 2 and point_outside_of_halfplane(hp, intersect_two_halfplanes(dq[0], dq[1])):
            dq.popleft()
        dq.append(hp)

    while len(dq) >= 3 and point_outside_of_halfplane(dq[0], intersect_two_halfplanes(dq[-1], dq[-2])):
        dq.pop()
    while len(dq) >= 3 and point_outside_of_halfplane(dq[-1], intersect_two_halfplanes(dq[0], dq[1])):
        dq.popleft()

    if len(dq) < 3:
        return None
    else:
        polygon = np.row_stack([intersect_two_halfplanes(dq[i], dq[(i + 1) % len(dq)])
                                for i in range(len(dq))])
        return polygon


random_state = np.random.RandomState(0)
n_halfplanes = 100
x = random_state.rand(n_halfplanes) * 2.0 * np.pi
p = np.column_stack((np.cos(x), np.sin(x)))
p += random_state.rand(*p.shape) * 0.1
angles = np.arctan2(p[:, 1], p[:, 0])
p = p[np.argsort(angles)]
pq = p[np.arange(n_halfplanes)] - p[np.arange(n_halfplanes) - 1]
halfplanes = np.hstack((p, pq))

timer = benchmark.Timer()
timer.start("intersect_halfplanes")
polygon = np.asarray(intersect_halfplanes2(halfplanes))
print(f"{timer.stop('intersect_halfplanes')} s")
timer.start("intersect_halfplanes2")
polygon2 = np.asarray(pressure_field.intersect_halfplanes(halfplanes))
print(f"{timer.stop('intersect_halfplanes2')} s")
print(polygon)
print(polygon2)
pressure_field.plot_halfplanes_and_intersections(
    halfplanes, polygon, xlim=(min(p[:, 0]) - 1, max(p[:, 0]) + 1),
    ylim=(min(p[:, 1]) - 1, max(p[:, 1]) + 1))
pressure_field.plot_halfplanes_and_intersections(
    halfplanes, polygon2, xlim=(min(p[:, 0]) - 1, max(p[:, 0]) + 1),
    ylim=(min(p[:, 1]) - 1, max(p[:, 1]) + 1))
