import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.rotations as pr
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_disk
from distance3d import plotting


random_state = np.random.RandomState(0)
point = np.array([-0.2, -0.6, -0.4])
center = np.array([0.3, 0.4, 0.5])
radius = 0.5
normal = pr.norm_vector(np.array([0.5, -0.3, -0.4]))

ax = ppu.make_3d_axis(ax_s=1)
for i in range(25):
    point = random_state.randn(3)
    distance, contact_point = point_to_disk(point, center, radius, normal)
    print(f"{distance}")
    points = np.vstack((point, contact_point))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.plot(points[:, 0], points[:, 1], points[:, 2])

ax.scatter(center[0], center[1], center[2])
plotting.plot_circle(ax, center, radius, normal)
plt.show()
