import numpy as np
from distance3d import hydroelastic_contact, containment
from numpy.testing import assert_array_almost_equal
from pytest import approx


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


def test_compute_contact_plane_overlap():
    sphere = hydroelastic_contact.RigidBody.make_cylinder(
        np.eye(4), 0.1, 1.0, resolution_hint=0.1)
    tetrahedron = sphere.tetrahedra_points[0]
    epsilon = np.array([0.0, 0.0, 0.0, 1.0])
    X = hydroelastic_contact.barycentric_transforms(
        tetrahedron.reshape(1, 4, 3))[0]
    intersection, contact_info = hydroelastic_contact.intersect_tetrahedron_pair(
        tetrahedron, epsilon, X, tetrahedron, epsilon, X)
    assert intersection


def test_compute_contact_force_overlap():
    sphere = hydroelastic_contact.RigidBody.make_sphere(
        np.array([0.1, 0.2, 0.1]), 1.0, order=2)
    sphere.youngs_modulus = 1.0

    tetrahedron = sphere.tetrahedra_points[0]
    epsilon = np.array([0.0, 0.0, 0.0, 1.0])
    X = hydroelastic_contact.barycentric_transforms(
        tetrahedron.reshape(1, 4, 3))[0]
    intersection, contact_info = hydroelastic_contact.intersect_tetrahedron_pair(
        tetrahedron, epsilon, X, tetrahedron, epsilon, X)
    contact_plane_hnf, contact_polygon = contact_info
    intersection_com, force_vector, total_area, triangles = hydroelastic_contact.compute_contact_force(
        tetrahedron, epsilon, contact_plane_hnf, contact_polygon)
    assert total_area == 0.0
    assert_array_almost_equal(force_vector, np.zeros(3))


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
    sphere = hydroelastic_contact.RigidBody.make_sphere(center, 0.2)
    sphere.express_in(np.eye(4))
    com = hydroelastic_contact.center_of_mass_tetrahedral_mesh(sphere.tetrahedra_points)
    assert_array_almost_equal(com, center)

    radii = np.array([1.0, 2.0, 3.0])
    ellipsoid = hydroelastic_contact.RigidBody.make_ellipsoid(np.eye(4), radii)
    com = hydroelastic_contact.center_of_mass_tetrahedral_mesh(ellipsoid.tetrahedra_points)
    assert_array_almost_equal(com, np.zeros(3))

    size = 1.0
    cube = hydroelastic_contact.RigidBody.make_cube(np.eye(4), size)
    com = hydroelastic_contact.center_of_mass_tetrahedral_mesh(cube.tetrahedra_points)
    assert_array_almost_equal(com, np.zeros(3))

    size = np.array([1.0, 2.0, 3.0])
    box = hydroelastic_contact.RigidBody.make_box(np.eye(4), size)
    com = hydroelastic_contact.center_of_mass_tetrahedral_mesh(box.tetrahedra_points)
    assert_array_almost_equal(com, np.zeros(3))

    long_cylinder = hydroelastic_contact.RigidBody.make_cylinder(
        np.eye(4), 0.1, 1.0, resolution_hint=0.1)
    com = hydroelastic_contact.center_of_mass_tetrahedral_mesh(
        long_cylinder.tetrahedra_points)
    assert_array_almost_equal(com, np.zeros(3))

    medium_cylinder = hydroelastic_contact.RigidBody.make_cylinder(
        np.eye(4), 0.1, 0.2, resolution_hint=0.1)
    com = hydroelastic_contact.center_of_mass_tetrahedral_mesh(
        medium_cylinder.tetrahedra_points)
    assert_array_almost_equal(com, np.zeros(3))

    short_cylinder = hydroelastic_contact.RigidBody.make_cylinder(
        np.eye(4), 0.1, 0.1, resolution_hint=0.1)
    com = hydroelastic_contact.center_of_mass_tetrahedral_mesh(
        short_cylinder.tetrahedra_points)
    assert_array_almost_equal(com, np.zeros(3))

    capsule = hydroelastic_contact.RigidBody.make_capsule(
        np.eye(4), 0.1, 1.0, resolution_hint=0.1)
    com = hydroelastic_contact.center_of_mass_tetrahedral_mesh(capsule.tetrahedra_points)
    assert_array_almost_equal(com, np.zeros(3))


def test_tetrahedral_mesh_aabbs():
    center = np.array([0.0, 0.2, -0.3])
    rb = hydroelastic_contact.RigidBody.make_sphere(center, 0.2, 2)
    rb.express_in(np.eye(4))
    aabbs = hydroelastic_contact.tetrahedral_mesh_aabbs(rb.tetrahedra_points)
    for i in range(len(rb.tetrahedra_points)):
        mins, maxs = containment.axis_aligned_bounding_box(rb.tetrahedra_points[i])
        assert_array_almost_equal(mins, aabbs[i, :, 0])
        assert_array_almost_equal(maxs, aabbs[i, :, 1])


def test_tetrahedral_mesh_volumes():
    center = np.array([1.0, 2.0, 3.0])
    radius = 1.0
    sphere = hydroelastic_contact.RigidBody.make_sphere(center, radius)
    sphere.express_in(np.eye(4))
    V = hydroelastic_contact.tetrahedral_mesh_volumes(sphere.tetrahedra_points)
    sphere_volume = 4.0 / 3.0 * np.pi * radius ** 3
    assert approx(np.sum(V), abs=1e-2) == sphere_volume

    radii = np.array([1.0, 2.0, 3.0])
    ellipsoid = hydroelastic_contact.RigidBody.make_ellipsoid(np.eye(4), radii, 5)
    V = hydroelastic_contact.tetrahedral_mesh_volumes(ellipsoid.tetrahedra_points)
    ellipsoid_volume = 4.0 / 3.0 * np.pi * np.product(radii)
    assert approx(np.sum(V), abs=1e-1) == ellipsoid_volume

    size = 1.0
    cube = hydroelastic_contact.RigidBody.make_cube(np.eye(4), size)
    V = hydroelastic_contact.tetrahedral_mesh_volumes(cube.tetrahedra_points)
    cube_volume = size ** 3
    assert approx(np.sum(V)) == cube_volume

    size = np.array([1.0, 2.0, 3.0])
    box = hydroelastic_contact.RigidBody.make_box(np.eye(4), size)
    V = hydroelastic_contact.tetrahedral_mesh_volumes(box.tetrahedra_points)
    box_volume = np.product(size)
    assert approx(np.sum(V)) == box_volume

    radius = 0.1
    length = 1.0
    long_cylinder = hydroelastic_contact.RigidBody.make_cylinder(
        np.eye(4), radius, length, resolution_hint=0.01)
    V = hydroelastic_contact.tetrahedral_mesh_volumes(long_cylinder.tetrahedra_points)
    long_cylinder_volume = np.pi * radius ** 2 * length
    assert approx(np.sum(V), abs=1e-2) == long_cylinder_volume

    radius = 0.1
    length = 0.2
    medium_cylinder = hydroelastic_contact.RigidBody.make_cylinder(
        np.eye(4), radius, length, resolution_hint=0.01)
    V = hydroelastic_contact.tetrahedral_mesh_volumes(medium_cylinder.tetrahedra_points)
    medium_cylinder_volume = np.pi * radius ** 2 * length
    assert approx(np.sum(V), abs=1e-2) == medium_cylinder_volume

    radius = 0.1
    length = 0.1
    short_cylinder = hydroelastic_contact.RigidBody.make_cylinder(
        np.eye(4), radius, length, resolution_hint=0.01)
    V = hydroelastic_contact.tetrahedral_mesh_volumes(short_cylinder.tetrahedra_points)
    short_cylinder_volume = np.pi * radius ** 2 * length
    assert approx(np.sum(V), abs=1e-2) == short_cylinder_volume

    radius = 0.1
    height = 0.0
    capsule = hydroelastic_contact.RigidBody.make_capsule(
        np.eye(4), radius, height, resolution_hint=0.01)
    V = hydroelastic_contact.tetrahedral_mesh_volumes(capsule.tetrahedra_points)
    capsule_volume = np.pi * radius ** 2 * length + 4.0 / 3.0 * np.pi * radius ** 3
    assert approx(np.sum(V), abs=1e-2) == capsule_volume


def test_youngs_modulus():
    rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(np.array([0.5, 0, 0]), 0.15, 2)
    rigid_body2 = hydroelastic_contact.RigidBody.make_sphere(np.array([0.5, 0.12, 0]), 0.15, 2)

    # Setting the young's modulus to 1 should be the same as not setting the young's modulus.

    intersection1, wrench12_1, wrench21_1, _ = hydroelastic_contact.contact_forces(
        rigid_body1, rigid_body2, return_details=True)

    rigid_body1.youngs_modulus = 1
    rigid_body2.youngs_modulus = 1

    intersection_2, wrench12_2, wrench21_2, _ = hydroelastic_contact.contact_forces(
        rigid_body1, rigid_body2, return_details=True)

    assert intersection1 == intersection_2
    assert_array_almost_equal(wrench12_1, wrench12_2)
    assert_array_almost_equal(wrench21_1, wrench21_2)

    rigid_body1.youngs_modulus = 10
    rigid_body2.youngs_modulus = 0.1

    intersection_3, wrench12_3, _, _ = hydroelastic_contact.contact_forces(
        rigid_body1, rigid_body2, return_details=True)

    assert intersection_3

    expected_vals = [0, 0.00032, 0, 0, 0, 0]

    assert_array_almost_equal(wrench12_3, expected_vals, decimal=4)
    len_2 = np.linalg.norm(wrench12_3)
    len_3 = np.linalg.norm(wrench12_2)
    assert len_2 < len_3

