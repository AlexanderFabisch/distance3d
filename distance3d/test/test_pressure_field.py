import numpy as np
from distance3d import pressure_field, containment
from numpy.testing import assert_array_almost_equal


def test_intersect_halfplanes():
    halfplanes = np.array([
        [0.5, 0.0, 0.0, 1.0],
        [0.0, 0.5, -1.0, 0.0],
        [-0.5, 0.0, 0.0, -1.0],
        [0.0, -0.5, 1.0, 0.0]
    ])
    polygon = pressure_field.intersect_halfplanes(halfplanes)
    polygon = [p.tolist() for p in polygon]
    expected_polygon = [
        [0.5, 0.5],
        [-0.5, 0.5],
        [-0.5, -0.5],
        [0.5, -0.5]
    ]
    for p in expected_polygon:
        assert p in polygon


def test_center_of_mass_tetrahedral_mesh():
    center = np.array([0.0, 0.2, -0.3])
    vertices, tetrahedra, _ = pressure_field.make_tetrahedral_icosphere(
        center, 0.2)
    tetrahedra_points = vertices[tetrahedra]
    com = pressure_field.center_of_mass_tetrahedral_mesh(tetrahedra_points)
    assert_array_almost_equal(com, center)


def test_tetrahedral_mesh_aabbs():
    center = np.array([0.0, 0.2, -0.3])
    vertices, tetrahedra, _ = pressure_field.make_tetrahedral_icosphere(
        center, 0.2, 2)
    tetrahedra_points = vertices[tetrahedra]
    aabbs = pressure_field.tetrahedral_mesh_aabbs(tetrahedra_points)
    for i in range(len(tetrahedra_points)):
        mins, maxs = containment.axis_aligned_bounding_box(tetrahedra_points[i])
        assert_array_almost_equal(mins, aabbs[i, :, 0])
        assert_array_almost_equal(maxs, aabbs[i, :, 1])
