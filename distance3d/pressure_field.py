"""Pressure field model for contact wrenches."""
from collections import deque
import math
import aabbtree
import numba
from numba.np.extensions import cross2d
import numpy as np
from scipy.spatial import Delaunay
from .mesh import tetrahedral_mesh_aabbs, center_of_mass_tetrahedral_mesh
from .utils import invert_transform, norm_vector, plane_basis_from_normal, EPSILON
from .benchmark import Timer


def contact_forces(
        mesh12origin, vertices1_in_mesh1, tetrahedra1, potentials1,
        mesh22origin, vertices2_in_mesh2, tetrahedra2, potentials2,
        return_details=False):
    timer = Timer()

    timer.start("transformation")
    # We transform vertices of mesh1 to mesh2 frame to be able to reuse the AABB
    # tree of mesh2.
    origin2mesh2 = invert_transform(mesh22origin)
    mesh12mesh2 = np.dot(origin2mesh2, mesh12origin)
    vertices1_in_mesh2 = np.dot(
        vertices1_in_mesh1, mesh12mesh2[:3, :3].T) + mesh12mesh2[np.newaxis, :3, 3]
    print(f"transformation: {timer.stop('transformation')}")

    # When two objects with pressure functions p1(*), p2(*) intersect, there is
    # a surface S inside the space of intersection at which the values of p1 and
    # p2 are equal. After identifying this surface, we then define the total force
    # exerted by one object on another [..].
    # Source: https://www.ekzhang.com/assets/pdf/Hydroelastics.pdf

    timer.start("broad phase")
    broad_overlapping_indices1, broad_overlapping_indices2 = check_aabbs_of_tetrahedra(
        vertices1_in_mesh2, tetrahedra1, vertices2_in_mesh2, tetrahedra2)
    print(f"broad phase: {timer.stop('broad phase')}")

    # TODO
    #com1 = center_of_mass_tetrahedral_mesh(mesh22origin, vertices1_in_mesh2, tetrahedra1)
    #com2 = center_of_mass_tetrahedral_mesh(mesh22origin, vertices2_in_mesh2, tetrahedra2)

    intersection = False
    total_force_vector = np.zeros(3)
    contact_polygons = []
    contact_polygon_triangles = []
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

        timer.start("contact_plane")
        contact_plane_hnf = contact_plane(
            tetrahedron1, tetrahedron2, epsilon1, epsilon2)
        timer.stop_and_add_to_total("contact_plane")
        if not check_tetrahedra_intersect_contact_plane(
                tetrahedron1, tetrahedron2, contact_plane_hnf):
            continue

        timer.start("compute_contact_polygon")
        contact_polygon, triangles = compute_contact_polygon(
            tetrahedron1, tetrahedron2, contact_plane_hnf)
        timer.stop_and_add_to_total("compute_contact_polygon")
        if contact_polygon is None:
            continue
        intersection = True

        timer.start("compute_contact_force")
        intersection_com, force_vector, area = compute_contact_force(
            tetrahedron1, epsilon1, contact_plane_hnf, contact_polygon,
            triangles)
        timer.stop_and_add_to_total("compute_contact_force")

        total_force_vector += force_vector
        # TODO use intersection com to compute torque

        contact_polygons.append(contact_polygon)
        contact_polygon_triangles.append(triangles)
        contact_coms.append(intersection_com)
        contact_forces.append(force_vector)
        contact_areas.append(area)
        contact_planes.append(contact_plane_hnf)
        intersecting_tetrahedra1.append(tetrahedron1)
        intersecting_tetrahedra2.append(tetrahedron2)

    print(timer.total_time_)

    force_in_world = mesh22origin[:3, :3].dot(total_force_vector)
    wrench21 = np.hstack((force_in_world, np.zeros(3)))
    wrench12 = -wrench21

    if return_details:
        if intersection:
            details = make_details(
                contact_areas, contact_coms, contact_forces, contact_planes,
                contact_polygons, contact_polygon_triangles,
                intersecting_tetrahedra1, intersecting_tetrahedra2,
                mesh22origin)
        else:
            details = {}
        return intersection, wrench12, wrench21, details
    else:
        return intersection, wrench12, wrench21


def make_details(
        contact_areas, contact_coms, contact_forces, contact_planes,
        contact_polygons, contact_polygon_triangles, intersecting_tetrahedra1,
        intersecting_tetrahedra2, mesh22origin):
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
        "contact_polygon_triangles": contact_polygon_triangles,
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


def check_aabbs_of_tetrahedra(vertices1_in_mesh2, tetrahedra1, vertices2_in_mesh2, tetrahedra2):
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


@numba.njit(
    numba.float64[:](numba.float64[:, ::1], numba.float64[:], numba.float64[:]),
    cache=True)
def points_to_plane_signed(points, plane_point, plane_normal):
    return np.dot(points - np.ascontiguousarray(plane_point).reshape(1, -1), plane_normal)


@numba.njit(cache=True)  # TODO can we use this?
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
    numba.boolean[:](numba.float64[:, :], numba.int64[:, :], numba.float64[:], numba.float64[:]),
    cache=True)
def intersecting_tetrahedra(vertices, tetrahedra, contact_point, normal):
    candidates = np.empty(len(tetrahedra), dtype=np.dtype("bool"))
    for i, tetrahedron in enumerate(tetrahedra):
        d = points_to_plane_signed(vertices[tetrahedron], contact_point, normal)
        candidates[i] = np.sign(min(d)) != np.sign(max(d))
    return candidates


@numba.njit(cache=True)
def barycentric_transform(vertices):  # TODO is there a faster implementation possible?
    """Returns X. X.dot(coords) = (r, 1), where r is a Cartesian vector."""
    # NOTE that in the original paper it is not obvious that we have to take
    # the inverse
    return np.linalg.pinv(np.vstack((vertices.T, np.ones((1, 4)))))


@numba.njit(cache=True)
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


@numba.njit(cache=True)
def check_tetrahedra_intersect_contact_plane(tetrahedron1, tetrahedron2, contact_plane_hnf, epsilon=1e-6):
    plane_distances1 = tetrahedron1.dot(contact_plane_hnf[:3]) - contact_plane_hnf[3]
    plane_distances2 = tetrahedron2.dot(contact_plane_hnf[:3]) - contact_plane_hnf[3]
    return (
        min(plane_distances1) < -epsilon
        and max(plane_distances1) > epsilon
        and min(plane_distances2) < -epsilon
        and max(plane_distances2) > epsilon)


def compute_contact_polygon(tetrahedron1, tetrahedron2, contact_plane_hnf, debug=False):
    timer = Timer()
    cart2plane = np.row_stack(plane_basis_from_normal(contact_plane_hnf[:3]))
    halfplanes = (make_halfplanes(tetrahedron1, contact_plane_hnf, cart2plane)
                  + make_halfplanes(tetrahedron2, contact_plane_hnf, cart2plane))
    timer.start("halfplanes")
    poly = intersect_halfplanes(halfplanes)
    print(f"{timer.stop('halfplanes')}")

    if debug:
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.subplot(111, aspect="equal")
        colors = "rb"
        for i, halfplane in enumerate(halfplanes):
            plot_halfplane(halfplane, ax, colors[i // 4], 0.1)
        if poly is not None:
            plt.scatter(poly[:, 0], poly[:, 1], s=100)
        plt.show()

    if poly is None:
        return None, None
    else:
        ch = Delaunay(poly)
        triangles = ch.simplices
        plane2cart = cart2plane.T
        plane_point = contact_plane_hnf[:3] * contact_plane_hnf[3]
        return np.row_stack([plane2cart.dot(p) + plane_point for p in poly]), triangles


@numba.experimental.jitclass(
    [("p", numba.float64[:]),
     ("pq", numba.float64[:]),
     ("normal2d", numba.float64[:]),
     ("angle", numba.float64)]
)
class HalfPlane:
    def __init__(self, p, pq, normal2d):
        self.p = p
        self.pq = pq
        self.normal2d = normal2d
        self.angle = math.atan2(self.pq[1], self.pq[0])

    def outside_of(self, point):
        return cross2d(self.pq, point - self.p) < -EPSILON

    def intersect(self, p2, pq2):
        denom = cross2d(self.pq, pq2)
        if np.abs(denom) < EPSILON:
            raise ValueError("Parallel halfplanes")
        alpha = cross2d((p2 - self.p), pq2) / denom
        return self.p + self.pq * alpha


def plot_halfplane(self, ax, c, alpha):
    line = self.p + np.linspace(-3.0, 3.0, 101)[:, np.newaxis] * norm_vector(self.pq)
    ax.plot(line[:, 0], line[:, 1], lw=3, c=c, alpha=alpha)
    normal = self.p + np.linspace(0.0, 1.0, 101)[:, np.newaxis] * norm_vector(self.normal2d)
    ax.plot(normal[:, 0], normal[:, 1], c=c, alpha=alpha)


TRIANGLES = np.array([[2, 1, 0], [2, 3, 1], [2, 0, 3], [1, 3, 0]], dtype=int)
LINE_SEGMENTS = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
TRIANGLE_LINE_SEGMENTS = np.array([triangle[LINE_SEGMENTS] for triangle in TRIANGLES], dtype=int)


def make_halfplanes(tetrahedron_points, plane_hnf, cart2plane):
    plane_normal = plane_hnf[:3]
    d = plane_hnf[3]
    plane_point = plane_normal * d

    P, d_signs, directions = _precompute_edge_intersections(
        d, plane_normal, tetrahedron_points)

    halfplanes = []
    for i, triangle in enumerate(TRIANGLES):
        intersection_points = []
        for line_segment in TRIANGLE_LINE_SEGMENTS[i]:
            i = min(line_segment)
            j = max(line_segment)
            if d_signs[i] != d_signs[j]:
                intersection_points.append(P[i, j])

        if len(intersection_points) != 2:  # TODO what if 3 points?
            continue

        # normal pointing inwards
        normal = np.cross(directions[triangle[1], triangle[0]],
                          directions[triangle[2], triangle[0]])

        normal2d = cart2plane.dot(normal)
        intersection_points = np.row_stack(intersection_points)
        intersection_points -= plane_point[np.newaxis]
        intersection_points = intersection_points.dot(cart2plane.T)

        p, q = intersection_points
        pq = q - p
        if np.cross(pq, normal2d) < 0:
            p = q
            pq *= -1.0
        halfplanes.append(HalfPlane(p, pq, normal2d))
    return halfplanes


@numba.njit(cache=True)
def _precompute_edge_intersections(d, plane_normal, tetrahedron_points):
    directions = np.empty((4, 4, 3), np.dtype("float"))
    for i in range(4):
        for j in range(4):
            directions[i, j] = tetrahedron_points[j] - tetrahedron_points[i]
    unnormalized_distances = d - np.dot(tetrahedron_points, plane_normal)
    d_signs = np.sign(unnormalized_distances)
    P = np.empty((4, 4, 3), np.dtype("float"))
    for i in range(4):
        for j in range(i + 1, 4):  # only fill upper triangle
            normal_direction = np.dot(directions[i, j], plane_normal)
            if normal_direction != 0.0:
                t = unnormalized_distances[i] / normal_direction
                P[i, j] = tetrahedron_points[i] + t * directions[i, j]
    return P, d_signs, directions


def make_halfplanes2(tetrahedron, cart2plane, plane2cart_offset):  # TODO can we fix this?
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


def intersect_halfplanes(halfplanes):
    points = []
    for i in range(len(halfplanes)):
        for j in range(i + 1, len(halfplanes)):
            try:
                p = halfplanes[i].intersect(halfplanes[j].p, halfplanes[j].pq)
            except ValueError:
                continue  # parallel halfplanes
            valid = True
            for hp in halfplanes:
                if hp.outside_of(p):
                    valid = False
                    break
            if valid:
                points.append(p)
    if len(points) < 3:
        return None
    return np.asarray(points)


def intersect_halfplanes2(halfplanes):  # TODO can we modify this to work with parallel lines?
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


@numba.njit(cache=True)
def compute_contact_force(
        tetrahedron, epsilon, contact_plane_hnf, contact_polygon, triangles):
    normal = contact_plane_hnf[:3]

    total_force = 0.0
    intersection_com = np.zeros(3)
    total_area = 0.0

    X = np.vstack((tetrahedron.T, np.ones((1, 4))))
    com = np.empty(4, dtype=np.dtype("float"))
    com[3] = 1.0
    for triangle in triangles:
        vertices = contact_polygon[triangle]
        com[:3] = (vertices[0] + vertices[1] + vertices[2]) / 3.0
        res = np.linalg.solve(X, com)
        pressure = sum(res * epsilon)
        area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0],
                                             vertices[2] - vertices[0]))
        total_force += pressure * area
        total_area += area
        intersection_com += area * com[:3]

    intersection_com /= total_area
    force_vector = total_force * normal
    return intersection_com, force_vector, total_area
