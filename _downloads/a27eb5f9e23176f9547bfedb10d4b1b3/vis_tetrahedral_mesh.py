"""
==========================
Visualize Tetrahedral Mesh
==========================
"""
print(__doc__)

import os
import numpy as np
import pytransform3d.visualizer as pv
from distance3d import io, visualization

BASE_DIR = "test/data/"
data_dir = BASE_DIR
search_path = ".."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "distance3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)
filename = os.path.join(data_dir, "insole.vtk")
vertices, tetrahedra = io.load_tetrahedral_mesh(filename)

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
mesh = visualization.TetraMesh(np.eye(4), vertices, tetrahedra, c=(1, 0, 0))
mesh.add_artist(fig)
fig.view_init()
if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
