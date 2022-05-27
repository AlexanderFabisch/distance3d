"""
====================================
Distance between ellipsoids with GJK
====================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d import gjk, colliders
from distance3d import random
from distance3d import plotting


random_state = np.random.RandomState(0)
ellipsoid2origin, radii = random.rand_ellipsoid(random_state, min_radius=0.2)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(700):
    ellipsoid2origin2, radii2 = random.rand_ellipsoid(random_state, min_radius=0.2)
    start = time.time()
    c1 = colliders.Ellipsoid(ellipsoid2origin, radii)
    c2 = colliders.Ellipsoid(ellipsoid2origin2, radii2)
    dist, closest_point_capsule, closest_point_capsule2, _ = gjk.gjk_with_simplex(c1, c2)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 5:
        continue
    if i == 0:
        vertices1 = np.array(c1.vertices_)
        ax.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], color="r")
        vertices2 = np.array(c2.vertices_)
        ax.scatter(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], color="g")
    plotting.plot_segment(
        ax, closest_point_capsule, closest_point_capsule2, c="k", lw=1)
    ppu.plot_ellipsoid(
        ax=ax, A2B=ellipsoid2origin2, radii=radii2, wireframe=False, alpha=0.5)
print(f"{accumulated_time=}")

ppu.plot_ellipsoid(
    ax=ax, A2B=ellipsoid2origin, radii=radii, wireframe=False, alpha=0.5,
    color="yellow")
plt.show()
