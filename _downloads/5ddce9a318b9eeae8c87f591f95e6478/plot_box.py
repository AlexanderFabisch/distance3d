"""
===============
Box Containment
===============
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d import containment_test


random_state = np.random.RandomState(0)
box2origin = np.eye(4)
size = np.ones(3)

ax = ppu.make_3d_axis(ax_s=3)
points = random_state.rand(100000, 3)
points[:, 0] -= 0.5
points[:, 0] *= 2.0
points[:, 2] -= 0.5
points[:, 2] *= 2.0
start = time.time()
contained = containment_test.points_in_box(points, box2origin, size)
stop = time.time()
print(f"{stop - start} s")
ax.scatter(points[::10, 0], points[::10, 1], points[::10, 2], c=contained[::10])
ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=True, color="r")
ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=False, alpha=0.5)
plt.show()
