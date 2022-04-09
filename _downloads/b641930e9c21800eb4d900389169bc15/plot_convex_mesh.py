"""
=======================================
Distance between convex meshes with GJK
=======================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d import random, plotting, gjk


random_state = np.random.RandomState(0)
vertices, faces = random.randn_convex(random_state, center_scale=0.0)


ax = ppu.make_3d_axis(ax_s=12)

accumulated_time = 0.0
for i in range(4000):
    vertices2, faces2 = random.randn_convex(random_state, center_scale=10.0)
    start = time.time()
    dist, p1, p2 = gjk.gjk(vertices, vertices2)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 20:
        continue
    plotting.plot_segment(ax, p1, p2, c="k", lw=1)
    plotting.plot_convex(ax, vertices2, faces2)
print(f"{accumulated_time=}")

plotting.plot_convex(ax, vertices, faces)
plt.show()
