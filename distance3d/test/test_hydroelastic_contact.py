import numpy as np
from distance3d import hydroelastic_contact, containment
from numpy.testing import assert_array_almost_equal


def test_contact_forces():
    rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(
        np.array([0.0, 0.0, 0.01]), 0.15, 1)

    rigid_body2 = hydroelastic_contact.RigidBody.make_sphere(
        np.array([0.0, 0.0, 0.4]), 0.15, 1)
    intersection, wrench12, wrench21 = hydroelastic_contact.contact_forces(
        rigid_body1, rigid_body2)
    assert not intersection
    assert_array_almost_equal(wrench12, np.zeros(6))
    assert_array_almost_equal(wrench21, np.zeros(6))

    rigid_body2 = hydroelastic_contact.RigidBody.make_sphere(np.array([0.0, 0.0, 0.3]), 0.15, 1)
    intersection, wrench12, wrench21 = hydroelastic_contact.contact_forces(
        rigid_body1, rigid_body2)
    assert intersection
    assert_array_almost_equal(
        wrench12, np.array([0.0, 0.0, 1.120994e-06, 0.0, 0.0, 0.0]))
    assert_array_almost_equal(
        wrench21, np.array([0.0, 0.0, -1.120994e-06, 0.0, 0.0, 0.0]))

    rigid_body2 = hydroelastic_contact.RigidBody.make_sphere(
        np.array([0.0, 0.0, 0.15]), 0.15, 1)
    intersection, wrench12, wrench21 = hydroelastic_contact.contact_forces(
        rigid_body1, rigid_body2)
    assert intersection
    assert_array_almost_equal(
        wrench12, np.array([0.0, 0.0, 0.00166866, 0.0, 0.0, 0.0]))
    assert_array_almost_equal(
        wrench21, np.array([0.0, 0.0, -0.00166866, 0.0, 0.0, 0.0]))


def test_contact_forces_and_details():
    rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(
        np.array([0.0, 0.0, 0.01]), 0.15, 1)
    rigid_body2 = hydroelastic_contact.RigidBody.make_sphere(
        np.array([0.0, 0.0, 0.3]), 0.15, 1)
    intersection, wrench12, wrench21, details = hydroelastic_contact.contact_forces(
        rigid_body1, rigid_body2, return_details=True)
    assert "contact_polygons" in details
    n_intersections = len(details["contact_polygons"])
    assert "contact_polygon_triangles" in details
    assert len(details["contact_polygon_triangles"]) == n_intersections
    assert "contact_planes" in details
    assert len(details["contact_planes"]) == n_intersections
    assert "intersecting_tetrahedra1" in details
    assert len(details["intersecting_tetrahedra1"]) == n_intersections
    assert "intersecting_tetrahedra2" in details
    assert len(details["intersecting_tetrahedra2"]) == n_intersections
    assert "contact_coms" in details
    assert len(details["contact_coms"]) == n_intersections
    assert "contact_forces" in details
    assert len(details["contact_forces"]) == n_intersections
    assert "contact_areas" in details
    assert len(details["contact_areas"]) == n_intersections
    assert "pressures" in details
    assert len(details["pressures"]) == n_intersections
    assert "contact_point" in details


def test_rigid_body_transforms():
    cube12origin = np.eye(4)
    cube22origin = np.eye(4)
    cube1_22origin = np.copy(cube12origin)
    cube1_22origin[:3, 3] += 1.0

    rigid_body1 = hydroelastic_contact.RigidBody.make_cube(cube12origin, 0.1)
    cube22origin[:3, 3] = np.array([0.0, 0.0, 0.08])
    rigid_body2 = hydroelastic_contact.RigidBody.make_cube(cube22origin, 0.1)

    rigid_body1.express_in(rigid_body2.body2origin_)
    tetras1_in_2 = rigid_body1.tetrahedra_points  # measure in cube2 frame

    rigid_body1.express_in(cube12origin)
    rigid_body1.body2origin_ = cube1_22origin  # move forward in origin frame

    rigid_body1.express_in(rigid_body2.body2origin_)
    tetras1_in_2_2 = rigid_body1.tetrahedra_points  # measure in cube frame

    assert_array_almost_equal(tetras1_in_2 + 1, tetras1_in_2_2)


def test_intersect_halfplanes():
    halfplanes = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0, 1.0],
        [0.0, 0.5, -1.0, 0.0],
        [-0.5, 0.0, 0.0, -1.0],
        [0.0, -0.5, 1.0, 0.0]
    ])
    polygon = hydroelastic_contact.intersect_halfplanes(halfplanes)
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
    vertices, tetrahedra, _ = hydroelastic_contact.make_tetrahedral_icosphere(
        center, 0.2)
    tetrahedra_points = vertices[tetrahedra]
    com = hydroelastic_contact.center_of_mass_tetrahedral_mesh(tetrahedra_points)
    assert_array_almost_equal(com, center)


def test_tetrahedral_mesh_aabbs():
    center = np.array([0.0, 0.2, -0.3])
    vertices, tetrahedra, _ = hydroelastic_contact.make_tetrahedral_icosphere(
        center, 0.2, 2)
    tetrahedra_points = vertices[tetrahedra]
    aabbs = hydroelastic_contact.tetrahedral_mesh_aabbs(tetrahedra_points)
    for i in range(len(tetrahedra_points)):
        mins, maxs = containment.axis_aligned_bounding_box(tetrahedra_points[i])
        assert_array_almost_equal(mins, aabbs[i, :, 0])
        assert_array_almost_equal(maxs, aabbs[i, :, 1])
