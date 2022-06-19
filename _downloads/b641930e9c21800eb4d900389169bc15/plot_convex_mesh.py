"""
=======================================
Distance between convex meshes with GJK
=======================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d import random, plotting, gjk, colliders


random_state = np.random.RandomState(0)
mesh2origin, vertices, triangles = random.randn_convex(random_state, center_scale=0.0)
points = pt.transform(mesh2origin, pt.vectors_to_points(vertices))[:, :3]

ax = ppu.make_3d_axis(ax_s=12)

accumulated_time = 0.0
for i in range(4000):
    mesh2origin2, vertices2, triangles2 = random.randn_convex(random_state, center_scale=10.0)
    points2 = pt.transform(mesh2origin2, pt.vectors_to_points(vertices2))[:, :3]
    start = time.time()
    dist, p1, p2, _ = gjk.gjk(colliders.ConvexHullVertices(points), colliders.ConvexHullVertices(points2))
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 20:
        continue
    plotting.plot_segment(ax, p1, p2, c="k", lw=1)
    plotting.plot_convex(ax, mesh2origin2, vertices2, triangles2)
print(f"{accumulated_time=}")

plotting.plot_convex(ax, mesh2origin, vertices, triangles)
plt.show()
