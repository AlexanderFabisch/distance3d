import numpy as np
from distance3d import pressure_field, containment
from numpy.testing import assert_array_almost_equal


def test_contact_forces():
    mesh12origin = np.eye(4)
    vertices1, tetrahedra1, potentials1 = pressure_field.make_tetrahedral_icosphere(
        np.array([0.0, 0.0, 0.01]), 0.15, 1)
    mesh22origin = np.eye(4)

    vertices2, tetrahedra2, potentials2 = pressure_field.make_tetrahedral_icosphere(
        np.array([0.0, 0.0, 0.4]), 0.15, 1)
    intersection, wrench12, wrench21 = pressure_field.contact_forces(
        mesh12origin, vertices1, tetrahedra1, potentials1,
        mesh22origin, vertices2, tetrahedra2, potentials2)
    assert not intersection
    assert_array_almost_equal(wrench12, np.zeros(6))
    assert_array_almost_equal(wrench21, np.zeros(6))

    vertices2, tetrahedra2, potentials2 = pressure_field.make_tetrahedral_icosphere(
        np.array([0.0, 0.0, 0.3]), 0.15, 1)
    intersection, wrench12, wrench21 = pressure_field.contact_forces(
        mesh12origin, vertices1, tetrahedra1, potentials1,
        mesh22origin, vertices2, tetrahedra2, potentials2)
    assert intersection
    assert_array_almost_equal(
        wrench12, np.array([0.0, 0.0, 1.120994e-06, 0.0, 0.0, 0.0]))
    assert_array_almost_equal(
        wrench21, np.array([0.0, 0.0, -1.120994e-06, 0.0, 0.0, 0.0]))

    vertices2, tetrahedra2, potentials2 = pressure_field.make_tetrahedral_icosphere(
        np.array([0.0, 0.0, 0.15]), 0.15, 1)
    intersection, wrench12, wrench21 = pressure_field.contact_forces(
        mesh12origin, vertices1, tetrahedra1, potentials1,
        mesh22origin, vertices2, tetrahedra2, potentials2)
    assert intersection
    assert_array_almost_equal(
        wrench12, np.array([0.0, 0.0, 0.00166866, 0.0, 0.0, 0.0]))
    assert_array_almost_equal(
        wrench21, np.array([0.0, 0.0, -0.00166866, 0.0, 0.0, 0.0]))


def test_contact_forces_and_details():
    mesh12origin = np.eye(4)
    vertices1, tetrahedra1, potentials1 = pressure_field.make_tetrahedral_icosphere(
        np.array([0.0, 0.0, 0.01]), 0.15, 1)
    mesh22origin = np.eye(4)
    vertices2, tetrahedra2, potentials2 = pressure_field.make_tetrahedral_icosphere(
        np.array([0.0, 0.0, 0.3]), 0.15, 1)
    intersection, wrench12, wrench21, details = pressure_field.contact_forces(
        mesh12origin, vertices1, tetrahedra1, potentials1,
        mesh22origin, vertices2, tetrahedra2, potentials2, return_details=True)
    assert "contact_polygons" in details
    assert len(details["contact_polygons"]) == 7
    assert "contact_polygon_triangles" in details
    assert len(details["contact_polygon_triangles"]) == 7
    assert "contact_planes" in details
    assert len(details["contact_planes"]) == 7
    assert "intersecting_tetrahedra1" in details
    assert len(details["intersecting_tetrahedra1"]) == 7
    assert "intersecting_tetrahedra2" in details
    assert len(details["intersecting_tetrahedra2"]) == 7
    assert "contact_coms" in details
    assert len(details["contact_coms"]) == 7
    assert "contact_forces" in details
    assert len(details["contact_forces"]) == 7
    assert "contact_areas" in details
    assert len(details["contact_areas"]) == 7
    assert "pressures" in details
    assert len(details["pressures"]) == 7
    assert "contact_point" in details



def test_intersect_halfplanes():
    halfplanes = np.array([
        [1.0, 0.0, 0.0, 1.0],
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
