"""
=================================
Distance between spheres with GJK
=================================
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
center, radius = random.rand_sphere(random_state, 0.2, 1.0)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(1000):
    center2, radius2 = random.rand_sphere(random_state, 1.0, 1.0)
    start = time.time()
    s1 = colliders.Sphere(center, radius)
    s2 = colliders.Sphere(center2, radius2)
    dist, contact_point_sphere, contact_point_sphere2, _ = gjk.gjk_with_simplex(s1, s2)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 5:
        continue
    if i == 0:
        vertices1 = np.array(s1.vertices_)
        ax.scatter(vertices1[:, 0], vertices1[:, 1], vertices1[:, 2], color="r")
        vertices2 = np.array(s2.vertices_)
        ax.scatter(vertices2[:, 0], vertices2[:, 1], vertices2[:, 2], color="g")
    plotting.plot_segment(
        ax, contact_point_sphere, contact_point_sphere2, c="k", lw=1)
    ppu.plot_sphere(ax, p=center2, radius=radius2, wireframe=False, alpha=0.2)
print(f"{accumulated_time=}")

ppu.plot_sphere(
    ax, p=center, radius=radius, color="yellow", wireframe=False, alpha=0.5)
plt.show()
