"""
==========================
Visualize Tetrahedral Mesh
==========================
"""
print(__doc__)
import numpy as np
import pytransform3d.visualizer as pv
from distance3d import io, visualization


vertices, tetrahedra = io.load_tetrahedral_mesh("test/data/insole.vtk")

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
mesh = visualization.TetraMesh(np.eye(4), vertices, tetrahedra, c=(1, 0, 0))
mesh.add_artist(fig)
fig.view_init()
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
