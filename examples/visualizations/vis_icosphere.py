"""
===================
Visualize Icosphere
===================
"""
print(__doc__)
import numpy as np
import pytransform3d.visualizer as pv
from distance3d import mesh, visualization, benchmark


timer = benchmark.Timer()
timer.start("make_icosphere")
vertices, triangles = mesh.make_icosphere(0.1 * np.ones(3), 0.1, 4)
print(timer.stop("make_icosphere"))

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
mesh = visualization.Mesh(np.eye(4), vertices, triangles)
mesh.add_artist(fig)

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
