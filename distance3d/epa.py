"""Expanding polytope algorithm (EPA) for collision resolution after GJK."""
import numpy as np
from .utils import norm_vector


EDGES_PER_FACE = 3


def epa(simplex, collider1, collider2, max_iter=64, max_loose_edges=32, max_faces=64, epsilon=1e-8):
    """Expanding Polytope Algorithm (EPA).

    Find minimum translation vector to resolve collision.

    Based on Kevin Moran's implementation:
    https://github.com/kevinmoran/GJK/blob/b38d923d268629f30b44c3cf6d4f9974bbcdb0d3/GJK.h
    (MIT License or Unlicense)

    Parameters
    ----------
    simplex : array, shape (4, 3)
        Simplex of Minkowski distances obtained by GJK.

    collider1 : Collider
        Convex collider 1.

    collider2 : Collider
        Convex collider 2.

    max_iter : int, optional (default: 64)
        Maximum number of iterations.

    max_loose_edges : int, optional (default: 32)
        Maximum number of loose edges per iteration.

    max_faces : int, optional (default: 64)
        Maximum number of faces in polytope.

    epsilon : float, optional (default: 1e-8)
        Floating point tolerance.

    Returns
    -------
    mtv : array, shape (3,)
        Minimum translation vector to be added to the second set of vertices
        or subtracted from the first set of vertices to resolve the collision.
        The norm of this vector is the penetration depth and the direction is
        the contact normal.

    faces : array, shape (n_faces, 4, 3)
        Faces in Minkowski difference space, defined by 3 points and its
        normal.

    success : bool
        EPA converged before maximum number of iterations was reached.
    """
    polytope = Polytope(simplex, max_faces, epsilon)
    loose_edges = LooseEdges(max_loose_edges, epsilon)

    for iteration in range(max_iter):
        min_dist, closest_face = polytope.find_face_closest_to_origin()

        search_direction = closest_face[3]
        new_vertex1 = collider1.support_function(search_direction)
        new_vertex2 = collider2.support_function(-search_direction)
        new_point = new_vertex1 - new_vertex2

        if np.dot(new_point, search_direction) - min_dist < epsilon:
            # Convergence: new point is not significantly further from origin.
            mtv = closest_face[3] * np.dot(new_point, search_direction)
            return mtv, polytope.get_all_faces(), True

        loose_edges.find_triangles_facing_point_and_store_loose_edges(
            polytope, new_point)
        polytope.extend_with_point(loose_edges, new_point)

    # Return most recent closest point
    mtv = closest_face[3] * np.dot(closest_face[0], closest_face[3])
    return mtv, polytope.get_all_faces(), False


class Polytope:
    """Array of faces, each with 3 vertices and a normal."""
    def __init__(self, simplex, max_faces, epsilon):
        self.max_faces = max_faces
        self.epsilon = epsilon
        self.faces = np.zeros((self.max_faces, 4, 3))
        self.n_faces = self._initialize_from_simplex(simplex)

    def _initialize_from_simplex(self, simplex):
        self.faces[0, :3] = simplex[:3]  # ABC
        self.faces[1, :3] = simplex[np.array((0, 2, 3), dtype=int)]  # ACD
        self.faces[2, :3] = simplex[np.array((0, 3, 1), dtype=int)]  # ADB
        self.faces[3, :3] = simplex[np.array((1, 3, 2), dtype=int)]  # BDC
        n_faces = 4
        for i in range(n_faces):
            self.compute_normal(i)
        return n_faces

    def compute_normal(self, face_idx):
        self.faces[face_idx, 3] = norm_vector(
            np.cross(self.faces[face_idx, 1] - self.faces[face_idx, 0],
                     self.faces[face_idx, 2] - self.faces[face_idx, 0]))

    def find_face_closest_to_origin(self):
        dists = np.sum(
            self.faces[:self.n_faces, 0] * self.faces[:self.n_faces, 3],
            axis=1)
        closest_face = np.argmin(dists)
        return dists[closest_face], self.faces[closest_face]

    def get_edge(self, face_idx, edge_idx):
        return np.vstack([self.faces[face_idx, edge_idx],
                          self.faces[face_idx, (edge_idx + 1) % 3]])

    def get_all_faces(self):
        return self.faces[:self.n_faces]

    def remove_face(self, i):
        self.faces[i] = self.faces[self.n_faces - 1]
        self.n_faces -= 1

    def triangle_faces_point(self, i, new_points):
        return np.dot(self.faces[i, 3],
                      new_points - self.faces[i, 0]) > self.epsilon

    def extend_with_point(self, loose_edges, new_point):
        for i in range(loose_edges.n_loose_edges):
            assert self.n_faces < self.max_faces
            if self.n_faces >= self.max_faces:
                break
            self.faces[self.n_faces, :2] = loose_edges.loose_edges[i]
            self.faces[self.n_faces, 2] = new_point
            self.compute_normal(self.n_faces)
            if np.linalg.norm(self.faces[self.n_faces, 3]) < 0.5:  # TODO is this the right solution?
                continue
            self.fix_ccw_normal_direction(self.n_faces)
            self.n_faces += 1

    def fix_ccw_normal_direction(self, face_idx, bias=1e-6):
        """Correct wrong normal direction to maintain CCW winding."""
        # Use bias in case dot result is only slightly < 0 (because origin is on face)
        if np.dot(self.faces[face_idx, 0], self.faces[face_idx, 3]) + bias < 0.0:
            temp = self.faces[face_idx, 0]
            self.faces[face_idx, 0] = self.faces[face_idx, 1]
            self.faces[face_idx, 1] = temp
            self.faces[face_idx, 3] = -self.faces[face_idx, 3]


class LooseEdges:
    """Keep track of edges we need to fix after removing faces."""
    def __init__(self, max_loose_edges, epsilon):
        self.max_loose_edges = max_loose_edges
        self.epsilon = epsilon
        self.loose_edges = np.zeros((self.max_loose_edges, 2, 3))
        self.n_loose_edges = 0

    def find_triangles_facing_point_and_store_loose_edges(self, faces, new_point):
        self.n_loose_edges = 0
        i = 0
        while i < faces.n_faces:
            if faces.triangle_faces_point(i, new_point):
                self.add_removed_triangles_edges_to_list(faces, i)
                faces.remove_face(i)
                i -= 1
            i += 1

    def add_removed_triangles_edges_to_list(self, faces, i):
        for j in range(EDGES_PER_FACE):
            current_edge = faces.get_edge(i, j)
            found_edge = False
            k = 0
            while k < self.n_loose_edges:
                if self.edge_already_in_list(k, current_edge):
                    # This assumes edge can only be shared by 2 triangles
                    # (which should be true). This also assumes shared edge
                    # will be reversed in the triangles (which should be true
                    # provided every triangle is wound CCW).
                    self.overwrite_edge_with_last_edge(k)
                    found_edge = True
                    # Exit loop because edge can only be shared once
                    break
                k += 1

            if not found_edge:
                success = self.add_edge_to_list(current_edge)
                if not success:
                    break

    def edge_already_in_list(self, edge_idx, current_edge):
        return (np.linalg.norm(self.loose_edges[edge_idx, 1] - current_edge[0]) < self.epsilon and
                np.linalg.norm(self.loose_edges[edge_idx, 0] - current_edge[1]) < self.epsilon)

    def add_edge_to_list(self, edge):
        if self.n_loose_edges >= self.max_loose_edges:
            return False
        self.loose_edges[self.n_loose_edges] = edge
        self.n_loose_edges += 1
        return True

    def overwrite_edge_with_last_edge(self, edge_idx):
        self.loose_edges[edge_idx] = self.loose_edges[self.n_loose_edges - 1]
        self.n_loose_edges -= 1
