"""
===================================
Distance between cylinders with GJK
===================================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d import gjk, colliders
from distance3d import random
from distance3d import plotting


random_state = np.random.RandomState(1)
cylinder2origin, radius, length = random.rand_cylinder(random_state, 0.5, 1.0, 0.0)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(700):
    cylinder2origin2, radius2, length2 = random.rand_cylinder(random_state)
    start = time.time()
    c1 = colliders.Cylinder(cylinder2origin, radius, length)
    c2 = colliders.Cylinder(cylinder2origin2, radius2, length2)
    dist, contact_point_cylinder, contact_point_cylinder2, _ = gjk.gjk_with_simplex(c1, c2)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 5:
        continue
    plotting.plot_segment(
        ax, contact_point_cylinder, contact_point_cylinder2, c="k", lw=1)
    ppu.plot_cylinder(ax, A2B=cylinder2origin2, radius=radius2, length=length2,
                      wireframe=False, color="b", alpha=0.2)
print(f"{accumulated_time=}")

ppu.plot_cylinder(ax, A2B=cylinder2origin, radius=radius, length=length,
                  wireframe=False, alpha=0.5)
plt.show()
