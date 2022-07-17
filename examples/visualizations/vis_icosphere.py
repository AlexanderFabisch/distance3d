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
timer.start("make_triangular_icosphere")
vertices1, triangles = mesh.make_triangular_icosphere(0.1 * np.ones(3), 0.1, 4)
print(timer.stop("make_triangular_icosphere"))
timer.start("make_tetrahedral_icosphere")
vertices2, tetrahedra = mesh.make_tetrahedral_icosphere(0.1 * np.ones(3), 0.15, 4)
print(timer.stop("make_tetrahedral_icosphere"))

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
tri_mesh = visualization.Mesh(np.eye(4), vertices1, triangles)
tri_mesh.add_artist(fig)
tetra_mesh = visualization.TetraMesh(np.eye(4), vertices2, tetrahedra)
tetra_mesh.add_artist(fig)

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
