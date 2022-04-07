"""TODO"""
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
import pytransform3d.transformations as pt
from distance3d import gjk, colliders
from distance3d import random
from distance3d import plotting


random_state = np.random.RandomState(0)
capsule2origin, radius, height = random.rand_capsule(random_state, 0.2, 0.4, 1.0)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(700):
    capsule2origin2, radius2, height2 = random.rand_capsule(random_state, 1.0, 0.3, 1.0)
    start = time.time()
    c1 = colliders.Capsule(capsule2origin, radius, height)
    c2 = colliders.Capsule(capsule2origin2, radius2, height2)
    dist, contact_point_capsule, contact_point_capsule2, _ = gjk.gjk_with_simplex(c1, c2)
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
    plotting.plot_segment(ax, contact_point_capsule, contact_point_capsule2)
    pt.plot_transform(ax=ax, A2B=capsule2origin2, s=0.1)
    ppu.plot_capsule(ax=ax, A2B=capsule2origin2, radius=radius2, height=height2, wireframe=False, alpha=0.5)
print(f"{accumulated_time=}")

pt.plot_transform(ax=ax, A2B=capsule2origin, s=0.1)
ppu.plot_capsule(ax=ax, A2B=capsule2origin, radius=radius, height=height, wireframe=False, alpha=0.5, color="red")
plt.show()
