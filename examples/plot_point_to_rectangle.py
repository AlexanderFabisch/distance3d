import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.plot_utils as ppu
from distance3d.distance import point_to_rectangle
from distance3d import geometry
from distance3d import plotting


random_state = np.random.RandomState(6)
rectangle_center, rectangle_axes, rectangle_lengths = geometry.randn_rectangle(
    random_state, length_scale=5.0)

ax = ppu.make_3d_axis(ax_s=2)

for i in range(15):
    point = random_state.randn(3)
    dist, contact_point = point_to_rectangle(point, rectangle_center, rectangle_axes, rectangle_lengths)
    print(dist)
    plotting.plot_segment(ax, point, contact_point)

plotting.plot_rectangle(ax, rectangle_center, rectangle_axes, rectangle_lengths)
plt.show()
