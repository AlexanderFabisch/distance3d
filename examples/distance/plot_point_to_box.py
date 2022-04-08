"""
==========================
Distance from point to box
==========================
"""
print(__doc__)
import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_box
from distance3d import random
from distance3d import plotting


random_state = np.random.RandomState(2)
box2origin, size = random.rand_box(random_state, size_scale=5.0)

ax = ppu.make_3d_axis(ax_s=2)

accumulated_time = 0.0
for i in range(3500):
    point = random_state.randn(3)
    start = time.time()
    dist, contact_point = point_to_box(point, box2origin, size)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(ax, point, contact_point)
print(f"{accumulated_time=}")

ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=False, alpha=0.5)
pt.plot_transform(ax=ax, A2B=box2origin, s=0.1)
plt.show()
