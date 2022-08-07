"""
======================
Halfplane Intersection
======================
"""
print(__doc__)

import numpy as np
from distance3d import pressure_field


random_state = np.random.RandomState(0)
n_halfplanes = 10
x = random_state.rand(n_halfplanes) * 2.0 * np.pi
p = np.column_stack((np.cos(x), np.sin(x)))
p += random_state.randn(*p.shape) * 0.1
angles = np.arctan2(p[:, 1], p[:, 0])
p = p[np.argsort(angles)]
pq = p[np.arange(n_halfplanes)] - p[np.arange(n_halfplanes) - 1]
halfplanes = np.hstack((p, pq))
polygon = np.asarray(pressure_field.intersect_halfplanes(halfplanes))
pressure_field.plot_halfplanes_and_intersections(
    halfplanes, polygon, xlim=(min(p[:, 0]) - 1, max(p[:, 0]) + 1),
    ylim=(min(p[:, 1]) - 1, max(p[:, 1]) + 1))
