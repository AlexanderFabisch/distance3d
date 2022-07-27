"""Pressure field model for contact wrenches."""
from collections import deque
import functools
import math
import aabbtree
import numba
import numpy as np
from .colliders import ConvexHullVertices
from .mpr import mpr_penetration
from .gjk import gjk_intersection
from .distance import line_segment_to_plane
from .mesh import tetrahedral_mesh_aabbs, center_of_mass_tetrahedral_mesh
from .geometry import barycentric_coordinates_tetrahedron
from .utils import transform_point, invert_transform, norm_vector, plane_basis_from_normal, EPSILON


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

    # When two objects with pressure functions p1(*), p2(*) intersect, there is
    # a surface S inside the space of intersection at which the values of p1 and
    # p2 are equal. After identifying this surface, we then define the total force
    # exerted by one object on another [..].
    # Source: https://www.ekzhang.com/assets/pdf/Hydroelastics.pdf

    broad_overlapping_indices1, broad_overlapping_indices2 = _check_aabbs_of_tetrahedra(
        vertices1_in_mesh2, tetrahedra1, vertices2_in_mesh2, tetrahedra2)

    # TODO
    #com1 = center_of_mass_tetrahedral_mesh(mesh22origin, vertices1_in_mesh2, tetrahedra1)
    #com2 = center_of_mass_tetrahedral_mesh(mesh22origin, vertices2_in_mesh2, tetrahedra2)

    intersection = False
    total_force_vector = np.zeros(3)
    contact_polygons = []
    contact_coms = []
    contact_forces = []
    contact_areas = []
    contact_planes = []
    intersecting_tetrahedra1 = []
    intersecting_tetrahedra2 = []
    previous_i = -1
    for i, j in zip(broad_overlapping_indices1, broad_overlapping_indices2):
        if i != previous_i:
            tetrahedron1 = vertices1_in_mesh2[tetrahedra1[i]]
            epsilon1 = potentials1[tetrahedra1[i]]
        previous_i = i

        tetrahedron2 = vertices2_in_mesh2[tetrahedra2[j]]
        epsilon2 = potentials2[tetrahedra2[j]]
        contact_plane_hnf = contact_plane(
            tetrahedron1, tetrahedron2, epsilon1, epsilon2)
        if not check_tetrahedra_intersect_contact_plane(
                tetrahedron1, tetrahedron2, contact_plane_hnf):
            continue

        """
        if i * len(tetrahedra2) + j > 8 * len(tetrahedra2) + 91:
            print(i, j)
            import pytransform3d.visualizer as pv
            from distance3d.visualization import TetraMesh, Tetrahedron
            fig = pv.figure()
            TetraMesh(mesh22origin, vertices1_in_mesh2, tetrahedra1).add_artist(fig)
            TetraMesh(mesh22origin, vertices2_in_mesh2, tetrahedra2).add_artist(fig)
            Tetrahedron(tetrahedron1).add_artist(fig)
            Tetrahedron(tetrahedron2).add_artist(fig)
            fig.plot_plane(normal=contact_plane_hnf[:3], d=contact_plane_hnf[3])
            fig.show()
        """

        debug = False
        #debug = i == 8 and j == 95
        contact_polygon = compute_contact_polygon(
            tetrahedron1, tetrahedron2, contact_plane_hnf, debug=debug)
        if contact_polygon is None:
            continue
        intersection = True

        #assert np.dot(contact_plane_hnf[:3], mesh22origin[:3, :3].dot(com1) + mesh22origin[:3, 3]) - np.dot(contact_plane_hnf[:3], mesh22origin[:3, :3].dot(com2) + mesh22origin[:3, 3]) >= 0.0  # Otherwise contact_plane_hnf[:3] *= -1
        intersection_com, force_vector, area = contact_force(
            tetrahedron1, epsilon1, contact_plane_hnf, contact_polygon)

        total_force_vector += force_vector
        # TODO use intersection com to compute torque

        contact_polygons.append(contact_polygon)
        contact_coms.append(intersection_com)
        contact_forces.append(force_vector)
        contact_areas.append(area)
        contact_planes.append(contact_plane_hnf)
        intersecting_tetrahedra1.append(tetrahedron1)
        intersecting_tetrahedra2.append(tetrahedron2)

    force_in_world = mesh22origin[:3, :3].dot(total_force_vector)
    wrench21 = np.hstack((force_in_world, np.zeros(3)))
    wrench12 = -wrench21

    if return_details:
        if intersection:
            details = make_details(
                contact_areas, contact_coms, contact_forces, contact_planes,
                contact_polygons, intersecting_tetrahedra1,
                intersecting_tetrahedra2, mesh22origin)
        else:
            details = {}
        return intersection, wrench12, wrench21, details
    else:
        return intersection, wrench12, wrench21


def make_details(
        contact_areas, contact_coms, contact_forces, contact_planes,
        contact_polygons, intersecting_tetrahedra1, intersecting_tetrahedra2,
        mesh22origin):
    contact_polygons = [contact_polygon.dot(mesh22origin[:3, :3].T) + mesh22origin[:3, 3]
                        for contact_polygon in contact_polygons]
    contact_coms = np.asarray(contact_coms)
    contact_coms = contact_coms.dot(mesh22origin[:3, :3].T) + mesh22origin[:3, 3]
    contact_forces = np.asarray(contact_forces)
    contact_forces = contact_forces.dot(mesh22origin[:3, :3].T)
    contact_areas = np.asarray(contact_areas)
    contact_point = np.sum(
        contact_coms * contact_areas[:, np.newaxis],
        axis=0) / sum(contact_areas)
    contact_point = mesh22origin[:3, :3].dot(contact_point) + mesh22origin[:3, 3]
    contact_planes = np.asarray(contact_planes)
    plane_points = contact_planes[:, :3] * contact_planes[:, 3, np.newaxis]
    plane_points = plane_points.dot(mesh22origin[:3, :3].T) + mesh22origin[:3, 3]
    plane_normals = contact_planes[:, :3].dot(mesh22origin[:3, :3].T)
    intersecting_tetrahedra1 = np.asarray(intersecting_tetrahedra1)
    n_intersections = len(intersecting_tetrahedra1)
    intersecting_tetrahedra1 = (
            intersecting_tetrahedra1.reshape(
                n_intersections * 4, 3).dot(mesh22origin[:3, :3].T)
            + mesh22origin[:3, 3]).reshape(n_intersections, 4, 3)
    intersecting_tetrahedra2 = np.asarray(intersecting_tetrahedra2)
    intersecting_tetrahedra2 = (
            intersecting_tetrahedra2.reshape(
                n_intersections * 4, 3).dot(mesh22origin[:3, :3].T)
            + mesh22origin[:3, 3]).reshape(n_intersections, 4, 3)
    details = {
        "contact_polygons": contact_polygons,
        "contact_coms": contact_coms,
        "contact_forces": contact_forces,
        "contact_areas": contact_areas,
        "contact_point": contact_point,
        "plane_points": plane_points,
        "plane_normals": plane_normals,
        "intersecting_tetrahedra1": intersecting_tetrahedra1,
        "intersecting_tetrahedra2": intersecting_tetrahedra2,
    }
    return details


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
    # NOTE in order to obtain proper Hesse normal form of the contact plane
    # we have to multiply the scalar by -1, since it appears as -d in the
    # equation np.dot(normal, x) - d = 0. However, it appears as
    # np.dot(normal, x) + d = 0 in the paper (page 7).
    plane_hnf[3] *= -1
    return plane_hnf


def check_tetrahedra_intersect_contact_plane(tetrahedron1, tetrahedron2, contact_plane_hnf, epsilon=1e-6):
    plane_distances1 = tetrahedron1.dot(contact_plane_hnf[:3]) - contact_plane_hnf[3]
    plane_distances2 = tetrahedron2.dot(contact_plane_hnf[:3]) - contact_plane_hnf[3]
    return (
        min(plane_distances1) < -epsilon
        and max(plane_distances1) > epsilon
        and min(plane_distances2) < -epsilon
        and max(plane_distances2) > epsilon)


def compute_contact_polygon(tetrahedron1, tetrahedron2, contact_plane_hnf, debug=False):
    cart2plane, plane2cart, plane2cart_offset = plane_projection(contact_plane_hnf)
    #print(contact_plane_hnf[:3].dot(plane2cart.dot(np.ones(2)) + plane2cart_offset) - contact_plane_hnf[3])
    halfplanes = (make_halfplanes(tetrahedron1, cart2plane, plane2cart_offset)
                  + make_halfplanes(tetrahedron2, cart2plane, plane2cart_offset))
    unique_halfplanes = remove_duplicates(halfplanes)
    poly, poly_halfplanes = intersect_halfplanes(unique_halfplanes)

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.subplot(111, aspect="equal")
        colors = "rb"
        for i, halfplane in enumerate(halfplanes):
            halfplane.plot(ax, colors[i // 4], 0.1)
        for i, halfplane in enumerate(unique_halfplanes):
            halfplane.plot(ax, "orange", 0.5)
        for i, halfplane in enumerate(poly_halfplanes):
            halfplane.plot(ax, "g", 1.0)
        if poly is not None:
            plt.scatter(poly[:, 0], poly[:, 1], s=100)
        plt.show()

    if poly is None:
        return poly
    else:
        return np.row_stack([plane2cart.dot(p) + plane2cart_offset for p in poly])


def compute_contact_polygon2(tetrahedron1, tetrahedron2, contact_plane_hnf, debug=False):
    cart2plane, plane2cart, plane2cart_offset = plane_projection(contact_plane_hnf)
    halfplanes = (make_halfplanes(tetrahedron1, cart2plane, plane2cart_offset)
                  + make_halfplanes(tetrahedron2, cart2plane, plane2cart_offset))
    validated_points = intersect_halfplanes2(halfplanes)

    if len(validated_points) == 0:
        poly = None
    else:
        poly = np.row_stack([
            plane2cart.dot(p) + plane2cart_offset for p in validated_points])

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.subplot(111, aspect="equal")
        colors = "rb"
        for i, halfplane in enumerate(halfplanes):
            halfplane.plot(ax, colors[i // 4], 0.1)
        if poly is not None:
            plt.scatter(poly[:, 0], poly[:, 1], s=100)
        plt.show()

    return poly


def intersect_halfplanes2(halfplanes):
    points = []
    for i in range(len(halfplanes)):
        for j in range(i, len(halfplanes)):
            p = halfplanes[i].intersect(halfplanes[j])
            if all(np.isfinite(p)):
                points.append(p)
    validated_points = []
    for point in points:
        valid = True
        for hp in halfplanes:
            if hp.outside_of(point):
                valid = False
                break
        if valid:
            validated_points.append(point)
    return validated_points


class HalfPlane:
    def __init__(self, p, normal2d):
        self.p = p
        self.pq = norm_vector(np.array([normal2d[1], -normal2d[0]]))
        self.normal2d = normal2d
        self.angle = math.atan2(self.pq[1], self.pq[0])

    def outside_of(self, point):
        return float(np.cross(self.pq, point - self.p)) < -EPSILON

    def intersect(self, halfplane):
        denom = np.cross(self.pq, halfplane.pq)
        if abs(denom) < EPSILON:
            return np.array([np.inf, np.inf])
        alpha = np.cross((halfplane.p - self.p), halfplane.pq) / denom
        return self.p + self.pq * alpha

    def plot(self, ax, c, alpha):
        line = self.p + np.linspace(-3.0, 3.0, 101)[:, np.newaxis] * norm_vector(self.pq)
        ax.plot(line[:, 0], line[:, 1], lw=3, c=c, alpha=alpha)
        normal = self.p + np.linspace(0.0, 1.0, 101)[:, np.newaxis] * norm_vector(self.normal2d)
        ax.plot(normal[:, 0], normal[:, 1], c=c, alpha=alpha)


def cmp_halfplanes(halfplane1, halfplane2):
    if abs(halfplane1.angle - halfplane2.angle) < EPSILON:
        return float(np.cross(halfplane1.pq, halfplane2.p - halfplane1.p))
    return halfplane1.angle - halfplane2.angle


def plane_projection(plane_hnf):  # TODO check sign of d
    """Find a 2x3 projection from the plane onto two dimensions, along with an inverse 3x2 projection that has the following properties:

    1. cart2plane * plane2cart = I
    2. plane_hnf[:3]' * (plane2cart * x + plane2cart_offset) - plane_hnf[3] = 0

    Source: https://github.com/ekzhang/hydroelastics/blob/d2c1e02aa1dd7e791212bdb930d80dee221bff1a/src/forces.jl#L152
    (MIT license)
    """
    assert abs(1.0 - np.linalg.norm(plane_hnf[:3])) < 10 * EPSILON, repr(np.linalg.norm(plane_hnf[:3]))
    if abs(plane_hnf[0]) > 1e-3:
        cart2plane = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        plane2cart = np.array(
            [[-plane_hnf[1] / plane_hnf[0], -plane_hnf[2] / plane_hnf[0]],
             [1.0, 0.0],
             [0.0, 1.0]])
        plane2cart_offset = np.array([plane_hnf[3] / plane_hnf[0], 0.0, 0.0])
    elif abs(plane_hnf[1]) > 1e-3:
        cart2plane = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        plane2cart = np.array([
            [1.0, 0.0],
            [-plane_hnf[0] / plane_hnf[1], -plane_hnf[2] / plane_hnf[1]],
            [0.0, 1.0]
        ])
        plane2cart_offset = np.array([0.0, plane_hnf[3] / plane_hnf[1], 0.0])
    else:
        assert abs(plane_hnf[2]) > 1e-3
        cart2plane = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        plane2cart = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [-plane_hnf[0] / plane_hnf[2], -plane_hnf[1] / plane_hnf[2]]
        ])
        plane2cart_offset = np.array([0.0, 0.0, plane_hnf[3] / plane_hnf[2]])
    return cart2plane, plane2cart, plane2cart_offset


def make_halfplanes(tetrahedron, cart2plane, plane2cart_offset):
    halfplanes = []
    X = barycentric_transform(tetrahedron)
    for i in range(4):
        halfspace = X[i]
        normal2d = cart2plane.dot(halfspace[:3])
        norm = np.linalg.norm(normal2d)
        if norm > 1e-9:
            p = normal2d * (-halfspace[3] - halfspace[:3].dot(plane2cart_offset)) / np.dot(normal2d, normal2d)
            halfplanes.append(HalfPlane(p, normal2d))
    return halfplanes


def remove_duplicates(halfplanes):
    compare_key = functools.cmp_to_key(cmp_halfplanes)
    halfplanes = sorted(halfplanes, key=compare_key)
    result = []
    for hp in halfplanes:  # TODO this is not the right way to handle this!
        if len(result) == 0 or abs(result[-1].angle - hp.angle) > 1e-12:
            result.append(hp)
    return result


def intersect_halfplanes(halfplanes):
    dq = deque()
    for hp in halfplanes:
        while len(dq) >= 2 and hp.outside_of(dq[-1].intersect(dq[-2])):
            dq.pop()
        while len(dq) >= 2 and hp.outside_of(dq[0].intersect(dq[1])):
            dq.popleft()
        dq.append(hp)

    while len(dq) >= 3 and dq[0].outside_of(dq[-1].intersect(dq[-2])):
        dq.pop()
    while len(dq) >= 3 and dq[-1].outside_of(dq[0].intersect(dq[1])):
        dq.popleft()

    if len(dq) < 3:
        return None, []
    else:
        polygon = np.row_stack([dq[i].intersect(dq[(i + 1) % len(dq)])
                                for i in range(len(dq))])
        return polygon, list(dq)


def contact_force(tetrahedron, epsilon, contact_plane_hnf, contact_polygon):
    normal = contact_plane_hnf[:3]

    total_force = 0.0
    intersection_com = np.zeros(3)
    total_area = 0.0

    X = np.vstack((tetrahedron.T, np.ones((1, 4))))
    for i in range(2, len(contact_polygon)):
        vertices = contact_polygon[np.array([0, i - 1, i], dtype=int)]  # TODO check
        com = np.hstack((np.mean(vertices, axis=0), (1,)))
        res = np.linalg.solve(X, com)
        pressure = sum(res * epsilon)
        area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
        total_force += pressure * area
        total_area += area
        intersection_com += area * com[:3]

    intersection_com /= total_area
    force_vector = total_force * normal
    return intersection_com, force_vector, total_area
