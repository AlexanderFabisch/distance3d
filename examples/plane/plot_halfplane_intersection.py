"""
======================
Halfplane Intersection
======================
"""
print(__doc__)

import numpy as np
from distance3d import pressure_field, benchmark


from distance3d.pressure_field._halfplanes import intersect_two_halfplanes, point_outside_of_halfplane, cross2d
def intersect_halfplanes(halfplanes):
    # source: https://cp-algorithms.com/geometry/halfplane-intersection.html#direct-implementation
    angles = np.arctan2(halfplanes[:, 3], halfplanes[:, 2])
    halfplanes = halfplanes[np.argsort(angles)]

    result = []
    for hp in halfplanes:
        while len(result) >= 2:
            p = intersect_two_halfplanes(result[-2], result[-1])
            if p is not None and point_outside_of_halfplane(hp, p):
                del result[-1]
            else:
                break
        while len(result) >= 2:
            p = intersect_two_halfplanes(result[0], result[1])
            if p is not None and point_outside_of_halfplane(hp, p):
                del result[0]
            else:
                break

        parallel_halfplanes = len(result) > 0 and abs(cross2d(result[-1][2:], hp[2:])) < 1e-6
        if parallel_halfplanes:
            opposite = np.dot(hp[2:], result[-1][2:]) < 0.0
            if opposite:
                return None
            new_halfplane_leftmost = point_outside_of_halfplane(hp, result[-1][:2])
            if new_halfplane_leftmost:
                del result[-1]
                result.append(hp)
            else:
                continue
        else:
            result.append(hp)

    while len(result) >= 3:
        p = intersect_two_halfplanes(result[-2], result[-1])
        assert p is not None
        if point_outside_of_halfplane(result[0], p):
            del result[-1]
        else:
            break

    while len(result) >= 3:
        p = intersect_two_halfplanes(result[0], result[1])
        assert p is not None
        if point_outside_of_halfplane(result[-1], p):
            del result[0]
        else:
            break

    if len(result) < 3:
        return None
    else:
        polygon = [
            intersect_two_halfplanes(result[i], result[(i + 1) % len(result)])
            for i in range(len(result))]
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

timer = benchmark.Timer()
timer.start("pressure_field.intersect_halfplanes")
polygon = np.asarray(pressure_field.intersect_halfplanes(halfplanes))
print(f"{timer.stop('pressure_field.intersect_halfplanes')} s")
print(polygon)
timer.start("intersect_halfplanes")
polygon2 = np.asarray(intersect_halfplanes(halfplanes))
print(f"{timer.stop('intersect_halfplanes')} s")
print(polygon2)
pressure_field.plot_halfplanes_and_intersections(
    halfplanes, polygon, xlim=(-2, 2), ylim=(-2, 2))
pressure_field.plot_halfplanes_and_intersections(
    halfplanes, polygon2, xlim=(-2, 2), ylim=(-2, 2))
