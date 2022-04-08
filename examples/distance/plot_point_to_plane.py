"""
============================
Distance from point to plane
============================
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as pr
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_plane


random_state = np.random.RandomState(0)
# TODO move to random
plane_point = random_state.randn(3)
plane_normal = pr.norm_vector(random_state.randn(3))

ax = ppu.make_3d_axis(ax_s=2)

for i in range(15):
    point = random_state.randn(3)
    dist, contact_point = point_to_plane(point, plane_point, plane_normal)
    print(dist)
    points = np.vstack((point, contact_point))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.plot(points[:, 0], points[:, 1], points[:, 2])

# TODO how to plot a plane? part of pytransform3d?
ppu.plot_vector(ax=ax, start=plane_point, direction=plane_normal)

plt.show()
