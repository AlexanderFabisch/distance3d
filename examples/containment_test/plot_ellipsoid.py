"""
=====================
Ellipsoid Containment
=====================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d import random, containment_test


random_state = np.random.RandomState(0)
ellipsoid2origin = np.eye(4)
radii = np.array([0.5, 0.5, 0.9])

ax = ppu.make_3d_axis(ax_s=3)
points = random_state.rand(100000, 3)
points[:, 0] -= 0.5
points[:, 0] *= 2.0
points[:, 2] -= 0.5
points[:, 2] *= 2.0
start = time.time()
contained = containment_test.points_in_ellipsoid(points, ellipsoid2origin, radii)
stop = time.time()
print(f"{stop - start} s")
ax.scatter(points[::10, 0], points[::10, 1], points[::10, 2], c=contained[::10])
ppu.plot_ellipsoid(ax=ax, A2B=ellipsoid2origin, radii=radii, wireframe=True, color="r")
ppu.plot_ellipsoid(ax=ax, A2B=ellipsoid2origin, radii=radii, wireframe=False, alpha=0.5)
plt.show()
