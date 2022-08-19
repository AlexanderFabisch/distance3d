"""
==================================
Distance between ellipses with GJK
==================================
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
center, axes, radii = random.rand_ellipse(random_state)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(700):
    center2, axes2, radii2 = random.rand_ellipse(random_state)
    start = time.time()
    c1 = colliders.Ellipse(center, axes, radii)
    c2 = colliders.Ellipse(center2, axes2, radii2)
    dist, closest_point_capsule, closest_point_capsule2, _ = gjk.gjk(c1, c2)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 5:
        continue
    plotting.plot_segment(
        ax, closest_point_capsule, closest_point_capsule2, c="k", lw=1)
    plotting.plot_ellipse(
        ax=ax, center=center2, axes=axes2, radii=radii2, surface_alpha=0.5)
print(f"{accumulated_time=}")

plotting.plot_ellipse(
    ax=ax, center=center, axes=axes, radii=radii, surface_alpha=0.5,
    color="yellow")
plt.show()
