"""Distance from line segment to box."""
import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pt
import pytransform3d.plot_utils as ppu
from distance3d.distance import line_segment_to_box


random_state = np.random.RandomState(2)
box2origin = np.eye(4)
size = np.ones(3)

ax = ppu.make_3d_axis(ax_s=2)

for i in range(10):
    offset = random_state.randn()
    segment_start = offset + random_state.randn(3) / 2
    segment_end = offset + random_state.randn(3) / 2
    dist, contact_point_segment, contact_point_box = line_segment_to_box(
        segment_start, segment_end, box2origin, size)
    print(dist)
    segment = np.vstack((segment_start, segment_end))
    points = np.vstack((contact_point_segment, contact_point_box))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.plot(points[:, 0], points[:, 1], points[:, 2])
    ax.scatter(segment[:, 0], segment[:, 1], segment[:, 2])
    ax.plot(segment[:, 0], segment[:, 1], segment[:, 2])

ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=False, alpha=0.5)
pt.plot_transform(ax=ax, A2B=box2origin, s=0.1)

plt.show()
