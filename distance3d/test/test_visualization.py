import numpy as np
from distance3d import random, visualization, io
from numpy.testing import assert_array_almost_equal


def test_mesh():
    random_state = np.random.RandomState(3)
    mesh = visualization.Mesh(*random.randn_convex(random_state), c=(1, 0, 0))
    geoms = mesh.geometries
    assert len(geoms) == 1
    assert all(np.asarray(geoms[0].vertex_colors[0]) == (1, 0, 0))


def test_tetra_mesh():
    vertices, tetrahedra = io.load_tetrahedral_mesh("test/data/insole.vtk")
    mesh = visualization.TetraMesh(np.eye(4), vertices, tetrahedra, c=(1, 0, 0))
    geoms = mesh.geometries
    assert len(geoms) == 1
    assert all(np.asarray(geoms[0].vertex_colors[0]) == (1, 0, 0))
    assert len(geoms[0].vertices) == 88
    assert len(geoms[0].tetras) == 189
    assert_array_almost_equal(
        np.mean(geoms[0].vertices, axis=0), [0.148573, 0.060693, 0.004949])
    mesh2origin = np.eye(4)
    mesh2origin[:3, 3] = -0.148573, -0.060693, -0.004949
    mesh.set_data(mesh2origin)
    assert_array_almost_equal(np.mean(geoms[0].vertices, axis=0), [0, 0, 0])
