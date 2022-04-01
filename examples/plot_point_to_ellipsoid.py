import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_ellipsoid


random_state = np.random.RandomState(3)
# TODO move to random
ellipsoid2origin = pt.random_transform(random_state)
radii = 1 + random_state.rand(3)

ax = ppu.make_3d_axis(ax_s=2)

for i in range(50):
    point = random_state.randn(3)
    dist, contact_point = point_to_ellipsoid(point, ellipsoid2origin, radii)
    print(dist)
    points = np.vstack((point, contact_point))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.plot(points[:, 0], points[:, 1], points[:, 2])

ppu.plot_ellipsoid(ax=ax, A2B=ellipsoid2origin, radii=radii, wireframe=False, alpha=0.5)
pt.plot_transform(ax=ax, A2B=ellipsoid2origin, s=0.1)

plt.show()
