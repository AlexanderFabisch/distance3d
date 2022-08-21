import math
import numpy as np


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
    # midpoint vertices cache to avoid duplicating shared vertices
    mid_cache = dict()

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
    for _ in range(order):
        triangles = np.empty(
            (4 * triangles.shape[0], triangles.shape[1]),
            dtype=int)
        for k, triangle in enumerate(triangles_prev):
            v1, v2, v3 = triangle
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
    potentials = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, size / 2.0],
        dtype=float)
    return vertices, tetrahedra, potentials


def make_tetrahedral_box(size):
    """TODO

    Source: Drake (https://github.com/RobotLocomotion/drake/blob/6b4664c2b4c4a7f52d24d16898d0bc2cc7f2b893/geometry/proximity/make_box_mesh.cc#L245),
    BSD 3-clause
    """
    mesh_vertices = []
    v = np.empty((2, 2, 2), dtype=float)
    half_size = 0.5 * size
    for i in range(2):
        x = -half_size[0] if i == 0 else half_size[0]
        for j in range(2):
            y = -half_size[1] if j == 0 else half_size[1]
            for k in range(2):
                z = -half_size[2] if k == 0 else half_size[2]
                v[i, j, k] = len(mesh_vertices)
                mesh_vertices.append([x, y, z])
    m = np.empty((2, 2, 2), dtype=float)
    min_half_size = min(half_size)
    relative_tolerance = 1e-14 * max(1.0, min_half_size)
    half_central_Ma = half_size - min_half_size
    half_central_Ma[half_central_Ma <= relative_tolerance] = 0.0
    for i in range(2):
        x = -half_central_Ma[0] if i == 0 else half_central_Ma[0]
        for j in range(2):
            y = -half_central_Ma[1] if j == 0 else half_central_Ma[1]
            for k in range(2):
                z = -half_central_Ma[2] if k == 0 else half_central_Ma[2]
                duplicate_in_i = i == 1 and half_central_Ma[0] == 0.0
                duplicate_in_j = j == 1 and half_central_Ma[1] == 0.0
                duplicate_in_k = k == 1 and half_central_Ma[2] == 0.0
                m[i, j, k] = m[0, j, k] if duplicate_in_i else (m[i, 0, k] if duplicate_in_j else (m[i, j, 0] if duplicate_in_k else len(mesh_vertices)))
                if not duplicate_in_i and not duplicate_in_j and not duplicate_in_k:
                    mesh_vertices.append([x, y, z])
    mesh_vertices = np.array(mesh_vertices)
    assert len(mesh_vertices) <= 12

    mesh_elements = []
    mesh_elements.extend(_split_to_tetrahedra(
        m[1][0][0], m[1][1][0], m[1][1][1], m[1][0][1],
        v[1][0][0], v[1][1][0], v[1][1][1], v[1][0][1]
    ))
    mesh_elements.extend(_split_to_tetrahedra(
        m[0][0][0], m[0][0][1], m[0][1][1], m[0][1][0],
        v[0][0][0], v[0][0][1], v[0][1][1], v[0][1][0]
    ))
    mesh_elements.extend(_split_to_tetrahedra(
        m[0][1][0], m[0][1][1], m[1][1][1], m[1][1][0],
        v[0][1][0], v[0][1][1], v[1][1][1], v[1][1][0]
    ))
    mesh_elements.extend(_split_to_tetrahedra(
        m[0][0][0], m[1][0][0], m[1][0][1], m[0][0][1],
        v[0][0][0], v[1][0][0], v[1][0][1], v[0][0][1]
    ))
    mesh_elements.extend(_split_to_tetrahedra(
        m[0][0][1], m[1][0][1], m[1][1][1], m[0][1][1],
        v[0][0][1], v[1][0][1], v[1][1][1], v[0][1][1]
    ))
    mesh_elements.extend(_split_to_tetrahedra(
        m[0][0][0], m[0][1][0], m[1][1][0], m[1][0][0],
        v[0][0][0], v[0][1][0], v[1][1][0], v[1][0][0]
    ))
    mesh_elements = np.array(mesh_elements, dtype=int)

    potentials = np.zeros(len(mesh_vertices))
    potentials[8:] = min_half_size

    return mesh_vertices, mesh_elements, potentials


def _split_to_tetrahedra(v0, v1, v2, v3, v4, v5, v6, v7):
    elements = []
    previous = v1
    for next in [v2, v3, v7, v4, v5, v1]:
        if len({previous, next, v0, v6}) == 4:
            elements.append([previous, next, v0, v6])
        previous = next
    return elements
