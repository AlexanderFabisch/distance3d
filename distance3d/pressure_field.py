"""Pressure field model for contact wrenches."""
import math
from collections import deque
import aabbtree
import numba
import numpy as np
from .utils import (
    invert_transform, transform_points, transform_directions, norm_vector,
    plane_basis_from_normal, adjoint_from_transform, EPSILON)
from .benchmark import Timer


class RigidBody:
    def __init__(self, mesh2origin, vertices_in_mesh, tetrahedra, potentials):
        self.mesh2origin = mesh2origin
        self.vertices_in_mesh = vertices_in_mesh
        self.tetrahedra = tetrahedra
        self.potentials = potentials

        self._tetrahedra_points = None

    @property
    def tetrahedra_points(self):
        if self._tetrahedra_points is None:
            self._tetrahedra_points = self.vertices_in_mesh[self.tetrahedra]
        return self._tetrahedra_points

    @property
    def tetrahedra_potentials(self):
        return self.potentials[self.tetrahedra]

    def transform(self, old2new):
        self.vertices_in_mesh = transform_points(old2new, self.vertices_in_mesh)
        self._tetrahedra_points = None


class ContactSurface:
    def __init__(
            self, frame2world, intersection, contact_planes, contact_polygons,
            contact_polygon_triangles, intersecting_tetrahedra1,
            intersecting_tetrahedra2):
        self.frame2world = frame2world
        self.intersection = intersection
        self.contact_planes = contact_planes
        self.contact_polygons = contact_polygons
        self.contact_polygon_triangles = contact_polygon_triangles
        self.contact_forces = contact_forces
        self.intersecting_tetrahedra1 = intersecting_tetrahedra1
        self.intersecting_tetrahedra2 = intersecting_tetrahedra2

        self.contact_areas = None
        self.contact_coms = None
        self.contact_forces = None

    def add_polygon_info(self, contact_areas, contact_coms, contact_forces):
        self.contact_areas = contact_areas
        self.contact_coms = contact_coms
        self.contact_forces = contact_forces

    def _transform_to_world(self, frame2world):
        self.contact_polygons = [transform_points(frame2world, contact_polygon)
                                 for contact_polygon in self.contact_polygons]
        plane_points = self.contact_planes[:, :3] * self.contact_planes[:, 3, np.newaxis]
        plane_points = transform_points(self.frame2world, plane_points)
        plane_normals = transform_directions(
            self.frame2world, self.contact_planes[:, :3])
        plane_distances = np.sum(plane_points * plane_normals, axis=1)
        self.contact_planes = np.hstack((plane_normals, plane_distances.reshape(-1, 1)))
        if self.contact_coms is not None:
            self.contact_coms = transform_points(
                self.frame2world, np.asarray(self.contact_coms))
            self.contact_forces = transform_directions(
                self.frame2world, np.asarray(self.contact_forces))

    def make_details(self, tetrahedra_points1, tetrahedra_points2):
        self._transform_to_world(self.frame2world)
        n_intersections = len(self.intersecting_tetrahedra1)
        intersecting_tetrahedra1 = tetrahedra_points1[np.asarray(self.intersecting_tetrahedra1, dtype=int)]
        intersecting_tetrahedra1 = transform_points(
            self.frame2world, intersecting_tetrahedra1.reshape(n_intersections * 4, 3)
        ).reshape(n_intersections, 4, 3)
        intersecting_tetrahedra2 = tetrahedra_points2[np.asarray(self.intersecting_tetrahedra2, dtype=int)]
        intersecting_tetrahedra2 = transform_points(
            self.frame2world, intersecting_tetrahedra2.reshape(n_intersections * 4, 3)
        ).reshape(n_intersections, 4, 3)
        details = {
            "contact_polygons": self.contact_polygons,
            "contact_polygon_triangles": self.contact_polygon_triangles,
            "contact_planes": self.contact_planes,
            "intersecting_tetrahedra1": intersecting_tetrahedra1,
            "intersecting_tetrahedra2": intersecting_tetrahedra2,
        }
        if self.contact_coms is not None:
            details["contact_coms"] = self.contact_coms
            details["contact_forces"] = self.contact_forces
            self.contact_areas = np.asarray(self.contact_areas)
            details["contact_areas"] = self.contact_areas
            pressures = np.linalg.norm(self.contact_forces, axis=1) / self.contact_areas
            details["pressures"] = pressures
            contact_point = np.sum(
                self.contact_coms * self.contact_areas[:, np.newaxis],
                axis=0) / sum(self.contact_areas)
            details["contact_point"] = contact_point
        return details


def contact_forces(
        mesh12origin, vertices1_in_mesh1, tetrahedra1, potentials1,
        mesh22origin, vertices2_in_mesh2, tetrahedra2, potentials2,
        return_details=False, timer=None):
    if timer is None:
        timer = Timer()

    rigid_body1 = RigidBody(mesh12origin, vertices1_in_mesh1, tetrahedra1, potentials1)
    rigid_body2 = RigidBody(mesh22origin, vertices2_in_mesh2, tetrahedra2, potentials2)

    contact_surface = find_contact_plane(rigid_body1, rigid_body2, timer)

    timer.start("accumulate_wrenches")
    wrench12_in_world, wrench21_in_world = accumulate_wrenches(
        contact_surface, rigid_body1, rigid_body2)
    timer.stop_and_add_to_total("accumulate_wrenches")

    if return_details:
        timer.start("make_details")
        if contact_surface.intersection:
            details = contact_surface.make_details(
                rigid_body1.tetrahedra_points, rigid_body2.tetrahedra_points)
        else:
            details = {}
        timer.stop_and_add_to_total("make_details")
        return contact_surface.intersection, wrench12_in_world, wrench21_in_world, details
    else:
        return contact_surface.intersection, wrench12_in_world, wrench21_in_world


def find_contact_plane(rigid_body1, rigid_body2, timer=None):
    if timer is None:
        timer = Timer()

    timer.start("transformation")
    # We transform vertices of mesh1 to mesh2 frame to be able to reuse the
    # AABB tree of mesh2.
    rigid_body1.transform(mesh12mesh2(rigid_body1.mesh2origin, rigid_body2.mesh2origin))
    tetrahedra_points1 = rigid_body1.tetrahedra_points
    tetrahedra_points2 = rigid_body2.tetrahedra_points
    timer.stop_and_add_to_total("transformation")

    timer.start("broad_phase_tetrahedra")
    broad_tetrahedra1, broad_tetrahedra2, broad_pairs = broad_phase_tetrahedra(
        rigid_body1, rigid_body2)
    timer.stop_and_add_to_total("broad_phase_tetrahedra")

    timer.start("barycentric_transform")
    unique_indices1 = np.unique(broad_tetrahedra1)
    unique_indices2 = np.unique(broad_tetrahedra2)
    X1 = barycentric_transforms(tetrahedra_points1[unique_indices1])
    X2 = barycentric_transforms(tetrahedra_points2[unique_indices2])
    X1 = {j: X1[i] for i, j in enumerate(unique_indices1)}
    X2 = {j: X2[i] for i, j in enumerate(unique_indices2)}
    timer.stop_and_add_to_total("barycentric_transform")

    timer.start("intersect_pairs")
    intersection_result = intersect_pairs(
        broad_pairs, tetrahedra_points1, tetrahedra_points2,
        X1, X2, rigid_body1.tetrahedra_potentials,
        rigid_body2.tetrahedra_potentials)
    contact_surface = ContactSurface(rigid_body2.mesh2origin, *intersection_result)
    timer.stop_and_add_to_total("intersect_pairs")

    return contact_surface


def mesh12mesh2(mesh12origin, mesh22origin):
    origin2mesh2 = invert_transform(mesh22origin)
    mesh12mesh2 = np.dot(origin2mesh2, mesh12origin)
    return mesh12mesh2


def broad_phase_tetrahedra(rigid_body1, rigid_body2):
    """Broad phase collision detection of tetrahedra."""

    """
    # TODO fix broad phase for cube vs. sphere
    # TODO speed up broad phase
    # TODO store result in RigidBody
    aabbs1 = tetrahedral_mesh_aabbs(rigid_body1.tetrahedra_points)
    aabbs2 = tetrahedral_mesh_aabbs(rigid_body2.tetrahedra_points)
    tree2 = aabbtree.AABBTree()
    for j, aabb in enumerate(aabbs2):
        tree2.add(aabbtree.AABB(aabb), j)
    broad_tetrahedra1 = []
    broad_tetrahedra2 = []
    for i, aabb in enumerate(aabbs1):
        new_indices2 = tree2.overlap_values(aabbtree.AABB(aabb))
        broad_tetrahedra2.extend(new_indices2)
        broad_tetrahedra1.extend([i] * len(new_indices2))
    broad_pairs = zip(broad_tetrahedra1, broad_tetrahedra2)
    """

    # FIXME workaround for broad phase bug:
    from itertools import product
    broad_tetrahedra1 = np.array(list(range(len(rigid_body1.tetrahedra))), dtype=int)
    broad_tetrahedra2 = np.array(list(range(len(rigid_body2.tetrahedra))), dtype=int)
    broad_pairs = list(product(broad_tetrahedra1, broad_tetrahedra2))
    return broad_tetrahedra1, broad_tetrahedra2, broad_pairs


def accumulate_wrenches(contact_surface, rigid_body1, rigid_body2):
    tetrahedra_potentials1 = rigid_body1.tetrahedra_potentials
    n_contacts = len(contact_surface.intersecting_tetrahedra1)
    contact_coms = np.empty((n_contacts, 3), dtype=float)
    contact_forces = np.empty((n_contacts, 3), dtype=float)
    contact_areas = np.empty(n_contacts, dtype=float)
    for intersection_idx in range(n_contacts):
        i = contact_surface.intersecting_tetrahedra1[intersection_idx]
        contact_plane_hnf = contact_surface.contact_planes[intersection_idx]
        contact_polygon = contact_surface.contact_polygons[intersection_idx]
        triangles = contact_surface.contact_polygon_triangles[intersection_idx]

        com, force, area = compute_contact_force(
            rigid_body1.tetrahedra_points[i], tetrahedra_potentials1[i],
            contact_plane_hnf, contact_polygon, triangles)

        contact_coms[intersection_idx] = com
        contact_forces[intersection_idx] = force
        contact_areas[intersection_idx] = area

    com1_in_mesh2 = center_of_mass_tetrahedral_mesh(rigid_body1.tetrahedra_points)
    com2_in_mesh2 = center_of_mass_tetrahedral_mesh(rigid_body2.tetrahedra_points)
    total_force_21 = np.sum(contact_forces, axis=0)
    com_minus_com1 = contact_coms - com1_in_mesh2
    com_minus_com2 = contact_coms - com2_in_mesh2
    torques_21 = np.vstack([
        np.cross(com_diff, force)
        for com_diff, force in zip(com_minus_com1, contact_forces)])
    total_torque_21 = np.sum(torques_21, axis=0)
    torques_12 = np.vstack([
        np.cross(com_diff, -force)
        for com_diff, force in zip(com_minus_com2, contact_forces)])
    total_torque_12 = np.sum(torques_12, axis=0)
    wrench12_in_world, wrench21_in_world = _transform_wrenches(
        contact_surface.frame2world, total_force_21, total_torque_12, total_torque_21)
    contact_surface.add_polygon_info(contact_areas, contact_coms, contact_forces)
    return wrench12_in_world, wrench21_in_world


@numba.njit(cache=True)
def _transform_wrenches(mesh22origin, total_force_21, total_torque_12, total_torque_21):
    wrench21 = np.hstack((total_force_21, total_torque_21))
    wrench12 = np.hstack((-total_force_21, total_torque_12))
    mesh22origin_adjoint = adjoint_from_transform(mesh22origin)
    wrench21_in_world = mesh22origin_adjoint.T.dot(wrench21)
    wrench12_in_world = mesh22origin_adjoint.T.dot(wrench12)
    return wrench12_in_world, wrench21_in_world


def intersect_pairs(
        broad_pairs, tetrahedra_points1, tetrahedra_points2, X1, X2,
        epsilon1, epsilon2):
    intersection = False
    contact_planes = []
    contact_polygons = []
    contact_polygon_triangles = []
    intersecting_tetrahedra1 = []
    intersecting_tetrahedra2 = []
    for i, j in broad_pairs:
        intersecting, contact_details = intersect_two_tetrahedra(
            tetrahedra_points1[i], epsilon1[i], X1[i],
            tetrahedra_points2[j], epsilon2[j], X2[j])
        if intersecting:
            intersection = True
        else:
            continue

        contact_plane_hnf, contact_polygon, triangles = contact_details

        intersecting_tetrahedra1.append(i)
        intersecting_tetrahedra2.append(j)
        contact_planes.append(contact_plane_hnf)
        contact_polygons.append(contact_polygon)
        contact_polygon_triangles.append(triangles)
    contact_planes = np.vstack(contact_planes)
    return (intersection, contact_planes, contact_polygons,
            contact_polygon_triangles, intersecting_tetrahedra1,
            intersecting_tetrahedra2)


@numba.njit(cache=True)
def intersect_two_tetrahedra(tetrahedron1, epsilon1, X1, tetrahedron2, epsilon2, X2):
    contact_plane_hnf = contact_plane(X1, X2, epsilon1, epsilon2)
    if not check_tetrahedra_intersect_contact_plane(
            tetrahedron1, tetrahedron2, contact_plane_hnf):
        return False, None

    contact_polygon, triangles = compute_contact_polygon(
        tetrahedron1, tetrahedron2, contact_plane_hnf)
    if contact_polygon is None:
        return False, None

    return True, (contact_plane_hnf, contact_polygon, triangles)


def barycentric_transforms(tetrahedra_points):
    """Returns X. X.dot(coords) = (r, 1), where r is a Cartesian vector."""
    # NOTE that in the original paper it is not obvious that we have to take
    # the inverse
    return np.linalg.pinv(np.hstack((tetrahedra_points.transpose((0, 2, 1)),
                                     np.ones((len(tetrahedra_points), 1, 4)))))


@numba.njit(cache=True)
def contact_plane(X1, X2, epsilon1, epsilon2):
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


@numba.njit(cache=True)
def compute_contact_polygon(tetrahedron1, tetrahedron2, contact_plane_hnf):
    cart2plane = np.row_stack(plane_basis_from_normal(contact_plane_hnf[:3]))
    halfplanes = np.vstack((
        make_halfplanes(tetrahedron1, contact_plane_hnf, cart2plane),
        make_halfplanes(tetrahedron2, contact_plane_hnf, cart2plane)))
    poly = intersect_halfplanes(halfplanes)

    if poly is None:
        return None, None

    points = np.empty((len(poly), 2))
    for i in range(len(poly)):
        points[i] = poly[i]

    center = np.array([np.mean(points[:, 0]), np.mean(points[:, 1])])
    poly_centered = points - center
    angles = np.arctan2(poly_centered[:, 1], poly_centered[:, 0])
    points = points[np.argsort(angles)]
    # this approach sometimes results in duplicate points, remove them
    unique_points = np.empty((len(points), 2))
    n_unique_points = 0
    for j in range(len(points)):
        if j == len(points) - 1 or np.linalg.norm(points[j + 1] - points[j]) > 10.0 * EPSILON:
            unique_points[n_unique_points] = points[j]
            n_unique_points += 1

    """
    import matplotlib.pyplot as plt
    plt.figure()
    ax = plt.subplot(111, aspect="equal")
    colors = "rb"
    for i, halfplane in enumerate(halfplanes):
        plot_halfplane(halfplane, ax, colors[i // 4], 0.1)
    plt.scatter(
        points[:, 0], points[:, 1],
        c=["r", "g", "b", "orange", "magenta", "brown", "k"][:len(points)],
        s=100)
    plt.show()
    #"""

    triangles = tesselate_ordered_polygon(unique_points[:n_unique_points])

    poly3d = cartesian_intersection_polygon(unique_points[:n_unique_points], cart2plane, contact_plane_hnf)
    return poly3d, triangles


# replaces from numba.np.extensions import cross2d, which seems to have a bug
# when called with NUMBA_DISABLE_JIT=1
@numba.njit(cache=True)
def cross2d(a, b):
    return a[0] * b[1] - a[1] * b[0]


@numba.njit(cache=True)
def point_outside_of_halfplane(p, pq, point):
    return cross2d(pq, point - p) < -EPSILON


@numba.njit(cache=True)
def intersect_two_halfplanes(p1, pq1, p2, pq2):
    denom = cross2d(pq1, pq2)
    if np.abs(denom) < EPSILON:
        raise ValueError("Parallel halfplanes")
    alpha = cross2d((p2 - p1), pq2) / denom
    return p1 + pq1 * alpha


def plot_halfplane(ppq, ax, c, alpha):
    line = ppq[:2] + np.linspace(-3.0, 3.0, 101)[:, np.newaxis] * norm_vector(ppq[2:])
    ax.plot(line[:, 0], line[:, 1], lw=3, c=c, alpha=alpha)
    normal2d = np.array([-ppq[3], ppq[2]])
    normal = ppq[:2] + np.linspace(0.0, 1.0, 101)[:, np.newaxis] * norm_vector(normal2d)
    ax.plot(normal[:, 0], normal[:, 1], c=c, alpha=alpha)


TRIANGLES = np.array([[2, 1, 0], [2, 3, 1], [2, 0, 3], [1, 3, 0]], dtype=int)
LINE_SEGMENTS = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
TRIANGLE_LINE_SEGMENTS = np.array([triangle[LINE_SEGMENTS] for triangle in TRIANGLES], dtype=int)


@numba.njit(cache=True)
def make_halfplanes(tetrahedron_points, plane_hnf, cart2plane):
    plane_normal = plane_hnf[:3]
    d = plane_hnf[3]
    plane_point = plane_normal * d

    P, d_signs, directions = _precompute_edge_intersections(
        d, plane_normal, tetrahedron_points)

    halfplanes = np.empty((4, 4))
    hp_idx = 0
    isect_points = np.empty((2, 3))
    for i, triangle in enumerate(TRIANGLES):
        intersection_points = []
        for line_segment in TRIANGLE_LINE_SEGMENTS[i]:
            i = min(line_segment)
            j = max(line_segment)
            if d_signs[i] != d_signs[j]:
                intersection_points.append(P[i, j])

        if len(intersection_points) != 2:  # TODO what if 3 points?
            continue

        isect_points[0] = intersection_points[0]
        isect_points[1] = intersection_points[1]
        halfplanes[hp_idx, :2], halfplanes[hp_idx, 2:] = make_halfplane(
            cart2plane, directions, isect_points, plane_point, triangle)
        hp_idx += 1
    return halfplanes[:hp_idx]


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


@numba.njit(cache=True)
def make_halfplane(
        cart2plane, directions, intersection_points, plane_point, triangle):
    # normal pointing inwards
    normal = np.cross(directions[triangle[1], triangle[0]],
                      directions[triangle[2], triangle[0]])
    normal2d = cart2plane.dot(normal)
    intersection_points -= plane_point
    intersection_points = intersection_points.dot(cart2plane.T)
    p, q = intersection_points
    pq = q - p
    if cross2d(pq, normal2d) < 0:
        p = q
        pq *= -1.0
    return p, pq


def make_halfplanes2(tetrahedron, cart2plane, plane2cart_offset):  # TODO can we fix this?
    halfplanes = []
    X = barycentric_transform(tetrahedron)
    for i in range(4):
        halfspace = X[i]
        normal2d = cart2plane.dot(halfspace[:3])
        norm = np.linalg.norm(normal2d)
        if norm > 1e-9:
            p = normal2d * (-halfspace[3] - halfspace[:3].dot(plane2cart_offset)) / np.dot(normal2d, normal2d)
            halfplanes.append((p, normal2d))
    return halfplanes


@numba.njit(cache=True)
def intersect_halfplanes(halfplanes):
    points = []
    for i in range(len(halfplanes)):
        for j in range(i + 1, len(halfplanes)):
            try:
                p = intersect_two_halfplanes(
                    halfplanes[i, :2], halfplanes[i, 2:],
                    halfplanes[j, :2], halfplanes[j, 2:])
            except Exception:
                continue  # parallel halfplanes
            valid = True
            for k in range(len(halfplanes)):
                if k != i and k != j and point_outside_of_halfplane(
                        halfplanes[k, :2], halfplanes[k, 2:], p):
                    valid = False
                    break
            if valid:
                points.append(p)
    if len(points) < 3:
        return None
    return points


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
def tesselate_ordered_polygon(poly):
    triangles = np.empty((len(poly) - 2, 3), dtype=np.dtype("int"))
    for i in range(len(poly) - 2):
        triangles[i] = (0, i + 1, i + 2)
    return triangles


@numba.njit(cache=True)
def cartesian_intersection_polygon(poly, cart2plane, contact_plane_hnf):
    plane2cart = cart2plane.T
    plane_point = contact_plane_hnf[:3] * contact_plane_hnf[3]
    poly3d = poly.dot(plane2cart.T) + plane_point
    return poly3d


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
        pressure = np.sum(res * epsilon)
        area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0],
                                             vertices[2] - vertices[0]))
        total_force += pressure * area
        total_area += area
        intersection_com += area * com[:3]

    intersection_com /= total_area
    force_vector = total_force * normal
    return intersection_com, force_vector, total_area


def make_triangular_icosphere(center, radius, order=4):
    """Creates a triangular icosphere mesh.

    Source: https://observablehq.com/@mourner/fast-icosphere-mesh

    Parameters
    ----------
    center : array, shape (3,)
        Center of the sphere.

    radius : float
        Radius of the sphere.

    order : int, optional (default: 4)
        Number of subdivisions of initial 20 triangles.

    Returns
    -------
    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    triangles : array, shape (n_triangles, 3)
        Indices of vertices that form triangles of the mesh.
    """
    f = (1 + 5 ** 0.5) / 2
    vertices = np.zeros(((10 * 4 ** order + 2), 3))
    vertices[:12] = np.array([
        [-1, f, 0], [1, f, 0], [-1, -f, 0],
        [1, -f, 0], [0, -1, f], [0, 1, f],
        [0, -1, -f], [0, 1, -f], [f, 0, -1],
        [f, 0, 1], [-f, 0, -1], [-f, 0, 1]])
    triangles = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [11, 10, 2],
        [5, 11, 4], [1, 5, 9], [7, 1, 8], [10, 7, 6], [3, 9, 4], [3, 4, 2],
        [3, 2, 6], [3, 6, 8], [3, 8, 9], [9, 8, 1], [4, 9, 5], [2, 4, 11],
        [6, 2, 10], [8, 6, 7]
    ], dtype=int)
    v = 12
    mid_cache = dict()  # midpoint vertices cache to avoid duplicating shared vertices

    def add_mid_point(a, b, mid_cache, v):
        # Cantor's pairing function
        key = math.floor((a + b) * (a + b + 1) / 2) + min(a, b)
        i = mid_cache.get(key, None)
        if i is not None:
            del mid_cache[key]
            return i, v
        mid_cache[key] = v
        vertices[v] = 0.5 * (vertices[a] + vertices[b])
        i = v
        v += 1
        return i, v

    # repeatedly subdivide each triangle into 4 triangles
    triangles_prev = triangles
    for i in range(order):
        triangles = np.empty(
            (4 * triangles.shape[0], triangles.shape[1]),
            dtype=int)
        for k in range(len(triangles_prev)):
            v1, v2, v3 = triangles_prev[k]
            a, v = add_mid_point(v1, v2, mid_cache, v)
            b, v = add_mid_point(v2, v3, mid_cache, v)
            c, v = add_mid_point(v3, v1, mid_cache, v)
            t = k * 4
            triangles[t] = v1, a, c
            triangles[t + 1] = v2, b, a
            triangles[t + 2] = v3, c, b
            triangles[t + 3] = a, b, c
        triangles_prev = triangles

    vertices /= 1.0 / radius * np.linalg.norm(vertices, axis=1)[:, np.newaxis]
    vertices += center[np.newaxis]
    return vertices, triangles


def make_tetrahedral_icosphere(center, radius, order=4):
    """Creates a tetrahedral icosphere mesh.

    Source: https://github.com/ekzhang/hydroelastics

    Parameters
    ----------
    center : array, shape (3,)
        Center of the sphere.

    radius : float
        Radius of the sphere.

    order : int, optional (default: 4)
        Number of subdivisions of initial 20 triangles.

    Returns
    -------
    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    tetrahedra : array, shape (n_tetrahedra, 4)
        Indices of vertices that form tetrahedra of the mesh.

    potentials : array, shape (n_vertices, 3)
        Potential of each vertex.
    """
    vertices, triangles = make_triangular_icosphere(center, radius, order)
    center_idx = len(vertices)
    vertices = np.vstack((vertices, center[np.newaxis]))
    tetrahedra = np.hstack(
        (triangles, center_idx * np.ones((len(triangles), 1), dtype=int)))
    potentials = np.zeros(len(vertices))
    potentials[-1] = radius
    return vertices, tetrahedra, potentials


def make_tetrahedral_cube(size):
    """Creates a tetrahedral cube mesh.

    Parameters
    ----------
    size : float
        Length of the edges in each dimension.

    Returns
    -------
    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    tetrahedra : array, shape (n_tetrahedra, 4)
        Indices of vertices that form tetrahedra of the mesh.

    potentials : array, shape (n_vertices, 3)
        Potential of each vertex.
    """
    vertices = size * np.array([
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0]
    ])
    tetrahedra = np.array([
        [0, 2, 6, 8],
        [0, 4, 5, 8],
        [0, 1, 2, 8],
        [1, 3, 2, 8],
        [1, 5, 7, 8],
        [1, 7, 3, 8],
        [5, 1, 0, 8],
        [5, 6, 7, 8],
        [6, 2, 3, 8],
        [6, 3, 7, 8],
        [6, 4, 0, 8],
        [6, 5, 4, 8],
    ], dtype=int)
    potentials = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, size / 2.0], dtype=float)
    return vertices, tetrahedra, potentials


def center_of_mass_tetrahedral_mesh(tetrahedra_points):
    """Compute center of mass of a tetrahedral mesh.

    Assumes uniform density.

    Parameters
    ----------
    tetrahedra_points : array, shape (n_tetrahedra, 4, 3)
        Points that form the tetrahedra.

    Returns
    -------
    com : array, shape (3,)
        Center of mass.
    """
    volumes = _tetrahedral_mesh_volumes(tetrahedra_points)
    centers = tetrahedra_points.mean(axis=1)
    return np.dot(volumes, centers) / np.sum(volumes)


def _tetrahedral_mesh_volumes(tetrahedra_points):
    """Compute volumes of tetrahedra.

    Parameters
    ----------
    tetrahedra_points : array, shape (n_tetrahedra, 4, 3)
        Points that form the tetrahedra.

    Returns
    -------
    volumes : array, shape (n_tetrahedra,)
        Volumes of tetrahedra.
    """
    tetrahedra_edges = tetrahedra_points[:, 1:] - tetrahedra_points[:, np.newaxis, 0]
    return np.abs(np.sum(
        np.cross(tetrahedra_edges[:, 0], tetrahedra_edges[:, 1])
        * tetrahedra_edges[:, 2], axis=1)) / 6.0


def tetrahedral_mesh_aabbs(tetrahedra_points):
    """Compute axis-aligned bounding boxes of tetrahedra.

    Parameters
    ----------
    tetrahedra_points : array, shape (n_tetrahedra, 4, 3)
        Points that form the tetrahedra.

    Returns
    -------
    aabbs : array, shape (n_tetrahedra, 3, 2)
        Axis-aligned bounding boxes of tetrahedra.
    """
    mins = np.min(tetrahedra_points, axis=1)
    maxs = np.max(tetrahedra_points, axis=1)
    aabbs = np.dstack((mins, maxs))
    return aabbs
