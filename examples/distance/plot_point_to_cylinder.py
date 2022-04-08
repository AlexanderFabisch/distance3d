"""
===============================
Distance from point to cylinder
===============================
"""
print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_cylinder
from distance3d import random


random_state = np.random.RandomState(3)
cylinder2origin, radius, length = random.rand_cylinder(random_state, min_radius=0.5, min_length=1.0)

ax = ppu.make_3d_axis(ax_s=2)

for i in range(50):
    point = random_state.randn(3)
    dist, contact_point = point_to_cylinder(point, cylinder2origin, radius, length)
    print(dist)
    points = np.vstack((point, contact_point))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.plot(points[:, 0], points[:, 1], points[:, 2])

ppu.plot_cylinder(ax=ax, A2B=cylinder2origin, radius=radius, length=length, wireframe=False, alpha=0.5)
pt.plot_transform(ax=ax, A2B=cylinder2origin, s=0.1)

plt.show()
