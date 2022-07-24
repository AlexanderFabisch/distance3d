import aabbtree
import numba
import numpy as np
from .colliders import ConvexHullVertices
from .mpr import mpr_penetration
from .gjk import gjk_intersection
from .distance import line_segment_to_plane
from .mesh import tetrahedral_mesh_aabbs, center_of_mass_tetrahedral_mesh
from .geometry import barycentric_coordinates_tetrahedron
from .utils import transform_point, invert_transform


def contact_forces(
        mesh12origin, vertices1_in_mesh1, tetrahedra1, potentials1,
        mesh22origin, vertices2_in_mesh2, tetrahedra2, potentials2,
        return_details=False):
    # We transform vertices of mesh1 to mesh2 frame to be able to reuse the AABB
    # tree of mesh2.
    origin2mesh2 = invert_transform(mesh22origin)
    mesh12mesh2 = np.dot(origin2mesh2, mesh12origin)
    vertices1_in_mesh2 = np.dot(
        vertices1_in_mesh1, mesh12mesh2[:3, :3].T) + mesh12mesh2[np.newaxis, :3, 3]

    # TODO we can also use the pressure functions for this. does it work with concave objects? which one is faster?
    c1 = ConvexHullVertices(vertices1_in_mesh2)
    c2 = ConvexHullVertices(vertices2_in_mesh2)
    intersection, depth, normal, contact_point = mpr_penetration(c1, c2)
    if not intersection:
        if return_details:
            return intersection, None, None, None
        else:
            return intersection, None, None

    # When two objects with pressure functions p1(*), p2(*) intersect, there is
    # a surface S inside the space of intersection at which the values of p1 and
    # p2 are equal. After identifying this surface, we then define the total force
    # exerted by one object on another [..].
    # Source: https://www.ekzhang.com/assets/pdf/Hydroelastics.pdf

    broad_overlapping_indices1, broad_overlapping_indices2 = _check_aabbs_of_tetrahedra(
        vertices1_in_mesh2, tetrahedra1, vertices2_in_mesh2, tetrahedra2)

    broad_overlapping_indices1 = np.asarray(broad_overlapping_indices1, dtype=int)
    broad_overlapping_indices2 = np.asarray(broad_overlapping_indices2, dtype=int)
    broad_overlapping_indices1, broad_overlapping_indices2 = _check_tetrahedra_intersect_contact_plane(
        vertices1_in_mesh2, tetrahedra1, vertices2_in_mesh2, tetrahedra2,
        contact_point, normal, broad_overlapping_indices1,
        broad_overlapping_indices2)

    forces1, forces2 = _contact_surface(
        vertices1_in_mesh2, tetrahedra1, potentials1,
        vertices2_in_mesh2, tetrahedra2, potentials2,
        broad_overlapping_indices1, broad_overlapping_indices2,
        contact_point, normal)

    # TODO
    #com1 = center_of_mass_tetrahedral_mesh(mesh12origin, vertices1_in_mesh2, tetrahedra1)

    normal_in_world = mesh22origin[:3, :3].dot(normal)
    force12 = sum([forces1[f][0] for f in forces1]) * normal_in_world
    wrench12 = np.hstack((force12, np.zeros(3)))
    force21 = sum([forces2[f][0] for f in forces2]) * -normal_in_world
    wrench21 = np.hstack((force21, np.zeros(3)))

    if return_details:
        details = {
            "object1_pressures": np.array([forces1[f][0] for f in forces1]),
            "object2_pressures": np.array([forces2[f][0] for f in forces2]),
            "object1_coms": np.array([forces1[f][1] for f in forces1]),
            "object2_coms": np.array([forces2[f][1] for f in forces2]),
            "object1_polys": [forces1[f][2] for f in forces1],
            "object2_polys": [forces2[f][2] for f in forces2],
            "contact_point": transform_point(mesh22origin, contact_point)
        }
        return intersection, wrench12, wrench21, details
    else:
        return intersection, wrench12, wrench21


def _contact_surface(
        vertices1_in_mesh2, tetrahedra1, potentials1,
        vertices2_in_mesh2, tetrahedra2, potentials2,
        broad_overlapping_indices1, broad_overlapping_indices2,
        contact_point, normal):
    forces1 = dict()
    forces2 = dict()
    last1 = -1
    for i in range(len(broad_overlapping_indices1)):
        idx1 = broad_overlapping_indices1[i]
        idx2 = broad_overlapping_indices2[i]

        if idx1 != last1:
            tetra1 = vertices1_in_mesh2[tetrahedra1[idx1]]
            t1 = ConvexHullVertices(tetra1)
            poly1 = contact_plane_projection(contact_point, normal, tetra1)
            area1 = polygon_area(poly1)
            p1 = np.mean(poly1, axis=0)
            c1 = barycentric_coordinates_tetrahedron(p1, tetra1)
            pressure1 = c1.dot(potentials1[tetrahedra1[idx1]])

        # TODO tetra-tetra intersection to compute triangle, something with halfplanes?
        # instead we try to compute surface for each object individually
        tetra2 = vertices2_in_mesh2[tetrahedra2[idx2]]
        t2 = ConvexHullVertices(tetra2)
        if gjk_intersection(t1, t2):
            # TODO compute triangle projection on contact surface, compute
            # area and use it as a weight for the pressure in integral
            poly2 = contact_plane_projection(contact_point, normal, tetra2)
            area2 = polygon_area(poly2)
            p2 = np.mean(poly2, axis=0)
            c2 = barycentric_coordinates_tetrahedron(p2, tetra2)
            pressure2 = c2.dot(potentials2[tetrahedra2[idx2]])

            forces1[idx1] = (area1 * pressure1, p1, poly1)
            forces2[idx2] = (area2 * pressure2, p2, poly2)
    return forces1, forces2


def _check_aabbs_of_tetrahedra(vertices1_in_mesh2, tetrahedra1, vertices2_in_mesh2, tetrahedra2):
    """Initial check of bounding boxes of tetrahedra."""
    aabbs1 = tetrahedral_mesh_aabbs(vertices1_in_mesh2, tetrahedra1)
    aabbs2 = tetrahedral_mesh_aabbs(vertices2_in_mesh2, tetrahedra2)
    broad_overlapping_indices1 = []
    broad_overlapping_indices2 = []
    tree2 = aabbtree.AABBTree()
    for j, aabb in enumerate(aabbs2):
        tree2.add(aabbtree.AABB(aabb), j)
    for i, aabb in enumerate(aabbs1):
        new_indices2 = tree2.overlap_values(aabbtree.AABB(aabb))
        broad_overlapping_indices2.extend(new_indices2)
        broad_overlapping_indices1.extend([i] * len(new_indices2))
    return broad_overlapping_indices1, broad_overlapping_indices2


@numba.njit(cache=True)
def _check_tetrahedra_intersect_contact_plane(
        vertices1_in_mesh2, tetrahedra1, vertices2_in_mesh2, tetrahedra2,
        contact_point, normal, broad_overlapping_indices1,
        broad_overlapping_indices2):
    """Check if the tetrahedra actually intersect the contact plane."""
    candidates1 = tetrahedra1[broad_overlapping_indices1]
    candidates2 = tetrahedra2[broad_overlapping_indices2]
    keep1 = intersecting_tetrahedra(vertices1_in_mesh2, candidates1, contact_point, normal)
    keep2 = intersecting_tetrahedra(vertices2_in_mesh2, candidates2, contact_point, normal)
    keep = np.logical_and(keep1, keep2)
    broad_overlapping_indices1 = broad_overlapping_indices1[keep]
    broad_overlapping_indices2 = broad_overlapping_indices2[keep]
    return broad_overlapping_indices1, broad_overlapping_indices2


@numba.njit(
    numba.float64[:](numba.float64[:, ::1], numba.float64[:], numba.float64[:]),
    cache=True)
def points_to_plane_signed(points, plane_point, plane_normal):
    return np.dot(points - np.ascontiguousarray(plane_point).reshape(1, -1), plane_normal)


@numba.njit(
    numba.boolean[:](numba.float64[:, :], numba.int64[:, :], numba.float64[:], numba.float64[:]),
    cache=True)
def intersecting_tetrahedra(vertices, tetrahedra, contact_point, normal):
    candidates = np.empty(len(tetrahedra), dtype=np.dtype("bool"))
    for i, tetrahedron in enumerate(tetrahedra):
        d = points_to_plane_signed(vertices[tetrahedron], contact_point, normal)
        candidates[i] = np.sign(min(d)) != np.sign(max(d))
    return candidates


def contact_plane_projection(plane_point, plane_normal, tetrahedron_points):
    d = points_to_plane_signed(tetrahedron_points, plane_point, plane_normal)
    neg = np.where(d < 0)[0]
    pos = np.where(d >= 0)[0]
    triangle_points = []
    for n in neg:
        for p in pos:
            triangle_points.append(
                line_segment_to_plane(
                    tetrahedron_points[n], tetrahedron_points[p],
                    plane_point, plane_normal)[2])
    triangle_points = np.asarray(triangle_points)
    return triangle_points


@numba.njit(cache=True)
def polygon_area(points):
    if len(points) == 3:
        return 0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
    else:
        assert len(points) == 4
        return 0.5 * (
            np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
            + np.linalg.norm(np.cross(points[1] - points[3], points[2] - points[3])))


def barycentric_transform(vertices):  # TODO is there a faster implementation possible?
    """Returns X. X.dot(coords) = (r, 1), where r is a Cartesian vector."""
    # NOTE that in the original paper it is not obvious that we have to take
    # the inverse
    return np.linalg.pinv(np.vstack((vertices.T, np.ones((1, len(vertices))))))


def contact_plane(tetrahedron1, tetrahedron2, epsilon1, epsilon2):
    X1 = barycentric_transform(tetrahedron1)
    X2 = barycentric_transform(tetrahedron2)
    plane_hnf = epsilon1.dot(X1) - epsilon2.dot(X2)  # TODO Young's modulus, see Eq. 16 of paper
    plane_hnf /= np.linalg.norm(plane_hnf[:3])
    plane_hnf[3] *= -1  # TODO where does the -1 come from?
    return plane_hnf
