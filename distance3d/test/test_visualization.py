import numpy as np
from distance3d import random, visualization


def test_mesh():
    random_state = np.random.RandomState(3)
    mesh = visualization.Mesh(*random.randn_convex(random_state), c=(1, 0, 0))
    geoms = mesh.geometries
    assert len(geoms) == 1
    assert all(np.asarray(geoms[0].vertex_colors[0]) == (1, 0, 0))
