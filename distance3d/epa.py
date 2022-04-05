import numpy as np
from .colliders import Convex
from .utils import norm_vector


EDGES_PER_FACE = 3


def epa_vertices(simplex, vertices1, vertices2, epsilon=1e-8):
    """Expanding Polytope Algorithm (EPA).

    Find minimum translation vector to resolve collision.
    Based on https://github.com/kevinmoran/GJK/blob/master/GJK.h

    Parameters
    ----------
    simplex : array-like, shape (4, 3)
        Simplex of Minkowski distances obtained by GJK.

    vertices1 : array, shape (n_vertices1, 3)
        Vertices of the first convex shape.

    vertices2 : array, shape (n_vertices2, 3)
        Vertices of the second convex shape.

    epsilon : float, optional (default: 1e-8)
        Floating point tolerance

    Returns
    -------
    mtv : array, shape (3,)
        Minimum translation vector to be added to the second set of vertices
        or subtracted from the first set of vertices to resolve the collision.
        The norm of this vector is the penetration depth and the direction is
        the contact normal.
    """
    return epa(simplex, Convex(vertices1), Convex(vertices2), epsilon=epsilon)[0]


def epa(simplex, collider1, collider2, max_iter=64, max_loose_edges=32, max_faces=64, epsilon=1e-8):
    """Expanding Polytope Algorithm (EPA).

    Find minimum translation vector to resolve collision.
    Based on https://github.com/kevinmoran/GJK/blob/master/GJK.h

    Parameters
    ----------
    simplex : array-like, shape (4, 3)
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
    # Array of faces, each with 3 vertices and a normal
    faces = np.zeros((max_faces, 4, 3))
    # Keep track of edges we need to fix after removing faces
    loose_edges = np.zeros((max_loose_edges, 2, 3))

    n_faces = _simplex_to_faces(simplex, faces)

    for iteration in range(max_iter):
        min_dist, closest_face = _face_closest_to_origin(faces, n_faces)

        # Search normal to face that is closest to origin
        search_direction = faces[closest_face, 3]
        _, new_vertex1 = collider1.support_function(search_direction)
        _, new_vertex2 = collider2.support_function(-search_direction)
        new_point = new_vertex1 - new_vertex2

        if np.dot(new_point, search_direction) - min_dist < epsilon:
            # Convergence: new point is not significantly further from origin.
            # Dot product vertex with normal to resolve collision along normal.
            mtv = faces[closest_face, 3] * np.dot(new_point, search_direction)
            return mtv, faces[:n_faces], True

        n_faces, n_loose_edges = _find_loose_edges(faces, loose_edges, n_faces, new_point, max_loose_edges, epsilon)
        n_faces = _extend_polytope(faces, loose_edges, n_faces, n_loose_edges, new_point, max_faces)

    # Return most recent closest point
    mtv = faces[closest_face, 3] * np.dot(faces[closest_face, 0], faces[closest_face, 3])
    return mtv, faces[:n_faces], False


def _simplex_to_faces(simplex, faces):
    """Initialize with final simplex from GJK."""
    # ABC
    faces[0, :3] = simplex[:3]
    faces[0, 3] = norm_vector(np.cross(simplex[1] - simplex[0], simplex[2] - simplex[0]))
    # ACD
    faces[1, 0] = simplex[0]
    faces[1, 1] = simplex[2]
    faces[1, 2] = simplex[3]
    faces[1, 3] = norm_vector(np.cross(simplex[2] - simplex[0], simplex[3] - simplex[0]))
    # ADB
    faces[2, 0] = simplex[0]
    faces[2, 1] = simplex[3]
    faces[2, 2] = simplex[1]
    faces[2, 3] = norm_vector(np.cross(simplex[3] - simplex[0], simplex[1] - simplex[0]))
    # BDC
    faces[3, 0] = simplex[1]
    faces[3, 1] = simplex[3]
    faces[3, 2] = simplex[2]
    faces[3, 3] = norm_vector(np.cross(simplex[3] - simplex[1], simplex[2] - simplex[1]))
    return 4


def _face_closest_to_origin(faces, n_faces):
    """Find face that is closest to origin."""
    dists = np.sum(faces[:n_faces, 0] * faces[:n_faces, 3], axis=1)
    closest_face = np.argmin(dists)
    return dists[closest_face], closest_face


def _find_loose_edges(faces, loose_edges, n_faces, new_points, max_loose_edges, epsilon):
    """Find all triangles that are facing p and store loose edges."""
    n_loose_edges = 0
    i = 0
    while i < n_faces:
        triangle_i_faces_p = np.dot(faces[i, 3], new_points - faces[i, 0]) > epsilon
        if triangle_i_faces_p:  # Remove it
            # Add removed triangle's edges to loose edge list.
            # If it is already there, remove it (both triangles it belonged to are gone)
            for j in range(EDGES_PER_FACE):
                current_edge = np.vstack([faces[i, j],
                                          faces[i, (j + 1) % 3]])
                found_edge = False
                k = 0
                while k < n_loose_edges:  # Check if current edge is already in list
                    if (np.linalg.norm(loose_edges[k, 1] - current_edge[0]) < epsilon and
                            np.linalg.norm(loose_edges[k, 0] - current_edge[1]) < epsilon):
                        # Edge is already in the list, remove it
                        # THIS ASSUMES EDGE CAN ONLY BE SHARED BY 2 TRIANGLES (which should be true)
                        # THIS ALSO ASSUMES SHARED EDGE WILL BE REVERSED IN THE TRIANGLES (which
                        # should be true provided every triangle is wound CCW)
                        # Overwrite current edge with last edge in list
                        loose_edges[k] = loose_edges[n_loose_edges - 1]
                        n_loose_edges -= 1
                        found_edge = True
                        # Exit loop because edge can only be shared once
                        break
                    k += 1

                if not found_edge:  # Add current edge to list
                    assert n_loose_edges < max_loose_edges
                    if n_loose_edges >= max_loose_edges:
                        break
                    loose_edges[n_loose_edges] = current_edge
                    n_loose_edges += 1

            # Remove triangle i from list
            faces[i] = faces[n_faces - 1]
            n_faces -= 1
            i -= 1
        i += 1
    return n_faces, n_loose_edges


def _extend_polytope(faces, loose_edges, n_faces, n_loose_edges, new_points, max_faces):
    """Reconstruct polytope with p added."""
    for i in range(n_loose_edges):
        assert n_faces < max_faces
        if n_faces >= max_faces:
            break
        faces[n_faces, :2] = loose_edges[i]
        faces[n_faces, 2] = new_points
        faces[n_faces, 3] = norm_vector(
            np.cross(faces[n_faces, 0] - faces[n_faces, 1],
                     faces[n_faces, 0] - faces[n_faces, 2]))
        if np.linalg.norm(faces[n_faces, 3]) < 0.5:  # TODO is this the right solution?
            continue
        _fix_ccw_normal_direction(faces, n_faces)
        n_faces += 1
    return n_faces


def _fix_ccw_normal_direction(faces, face_idx, bias=1e-6):
    """Correct wrong normal direction to maintain CCW winding."""
    # Use bias in case dot result is only slightly < 0 (because origin is on face)
    if np.dot(faces[face_idx, 0], faces[face_idx, 3]) + bias < 0.0:
        temp = faces[face_idx, 0]
        faces[face_idx, 0] = faces[face_idx, 1]
        faces[face_idx, 1] = temp
        faces[face_idx, 3] = -faces[face_idx, 3]
