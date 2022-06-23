"""
================
Mesh Containment
================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d import containment_test, random, plotting


random_state = np.random.RandomState(1)
mesh2origin, vertices, triangles = random.randn_convex(random_state, center_scale=0.0, radius_scale=10.0)

ax = ppu.make_3d_axis(ax_s=3)
points = random_state.rand(10000, 3)
points[:, 0] -= 0.5
points[:, 0] *= 2.0
points[:, 2] -= 0.5
points[:, 2] *= 2.0
start = time.time()
contained = containment_test.points_in_mesh(points, mesh2origin, vertices, triangles)
stop = time.time()
print(f"{stop - start} s")
ax.scatter(points[::10, 0], points[::10, 1], points[::10, 2], c=contained[::10])
plotting.plot_convex(ax=ax, mesh2origin=mesh2origin, vertices=vertices, triangles=triangles)
plt.show()
