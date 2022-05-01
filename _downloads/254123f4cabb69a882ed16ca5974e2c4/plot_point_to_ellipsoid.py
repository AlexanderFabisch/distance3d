"""
================================
Distance from point to ellipsoid
================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_ellipsoid
from distance3d import random, plotting


random_state = np.random.RandomState(3)
ellipsoid2origin, radii = random.rand_ellipsoid(random_state, min_radius=0.5)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(1500):
    point = random.randn_point(random_state)
    start = time.time()
    dist, contact_point = point_to_ellipsoid(point, ellipsoid2origin, radii)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 50:
        continue
    plotting.plot_segment(ax, point, contact_point, lw=1)
print(f"{accumulated_time=}")

ppu.plot_ellipsoid(ax=ax, A2B=ellipsoid2origin, radii=radii, wireframe=False, alpha=0.5)
pt.plot_transform(ax=ax, A2B=ellipsoid2origin, s=0.1)

plt.show()
