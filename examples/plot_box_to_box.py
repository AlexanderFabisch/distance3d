import time
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d import gjk, random, geometry, plotting


random_state = np.random.RandomState(0)
box2origin, size = random.rand_box(random_state, 0.1, 3)
vertices = geometry.convert_box_to_vertices(box2origin, size)

ax = ppu.make_3d_axis(ax_s=3)

accumulated_time = 0.0
for i in range(100):
    box2origin2, size2 = random.rand_box(random_state, 2, 1)
    vertices2 = geometry.convert_box_to_vertices(box2origin2, size2)
    start = time.time()
    dist, contact_point_box, contact_point_box2 = gjk.gjk(vertices, vertices2)
    end = time.time()
    accumulated_time += end - start
    print(dist)
    if i > 10:
        continue
    plotting.plot_segment(ax, contact_point_box, contact_point_box2)
    ppu.plot_box(ax=ax, A2B=box2origin2, size=size2, wireframe=False, alpha=0.5)
print(f"{accumulated_time=}")

ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=False, alpha=0.5)
plt.show()
