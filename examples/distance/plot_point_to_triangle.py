import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_triangle
from distance3d import random
from distance3d import plotting


random_state = np.random.RandomState(0)
triangle_points = random.randn_triangle(random_state) * 0.5

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(40000):
    point = random_state.randn(3)
    start = time.time()
    dist, contact_point = point_to_triangle(point, triangle_points)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 100:
        continue
    points = np.vstack((point, contact_point))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.plot(points[:, 0], points[:, 1], points[:, 2])
print(f"{accumulated_time=}")

plotting.plot_triangle(ax, triangle_points)
plt.show()
