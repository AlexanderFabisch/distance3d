import enum
import math
import numpy as np


def make_triangular_icosphere(center, radius, order=4):
    """Create a triangular icosphere mesh.

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


def make_tetrahedral_sphere(radius, order=4):
    """Create a tetrahedral icosphere mesh.

    Parameters
    ----------
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
    vertices, triangles = make_triangular_icosphere(np.zeros(3), radius, order)
    center_idx = len(vertices)
    vertices = np.vstack((vertices, np.zeros((1, 3))))
    tetrahedra = np.hstack(
        (triangles, center_idx * np.ones((len(triangles), 1), dtype=int)))
    potentials = np.zeros(len(vertices))
    potentials[-1] = radius
    return vertices, tetrahedra, potentials


def make_tetrahedral_ellipsoid(radii, order=4):
    """Create a tetrahedral ellipsoid mesh.

    Parameters
    ----------
    radii : array, shape (3,)
        Radii of the ellipsoid.

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
    vertices, triangles = make_triangular_icosphere(np.zeros(3), 1.0, order)
    vertices *= radii[np.newaxis]
    center_idx = len(vertices)
    vertices = np.vstack((vertices, np.zeros((1, 3))))
    tetrahedra = np.hstack(
        (triangles, center_idx * np.ones((len(triangles), 1), dtype=int)))
    potentials = np.zeros(len(vertices))
    potentials[-1] = min(radii)
    return vertices, tetrahedra, potentials


def make_tetrahedral_cube(size):
    """Create a tetrahedral cube mesh.

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
    """Create a tetrahedral box mesh.

    Source: Drake (https://github.com/RobotLocomotion/drake/blob/6b4664c2b4c4a7f52d24d16898d0bc2cc7f2b893/geometry/proximity/make_box_mesh.cc#L245),
    BSD 3-clause

    Parameters
    ----------
    size : array, shape (3,)
        Lengths of the edges in each dimension.

    Returns
    -------
    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    tetrahedra : array, shape (n_tetrahedra, 4)
        Indices of vertices that form tetrahedra of the mesh.

    potentials : array, shape (n_vertices, 3)
        Potential of each vertex.
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
    half_central = half_size - min_half_size
    half_central[half_central <= relative_tolerance] = 0.0
    for i in range(2):
        x = -half_central[0] if i == 0 else half_central[0]
        for j in range(2):
            y = -half_central[1] if j == 0 else half_central[1]
            for k in range(2):
                z = -half_central[2] if k == 0 else half_central[2]
                duplicate_in_i = i == 1 and half_central[0] == 0.0
                duplicate_in_j = j == 1 and half_central[1] == 0.0
                duplicate_in_k = k == 1 and half_central[2] == 0.0
                if duplicate_in_i:
                    m[i, j, k] = m[0, j, k]
                elif duplicate_in_j:
                    m[i, j, k] = m[i, 0, k]
                elif duplicate_in_k:
                    m[i, j, k] = m[i, j, 0]
                else:
                    m[i, j, k] = len(mesh_vertices)

                if not duplicate_in_i and not duplicate_in_j and not duplicate_in_k:
                    mesh_vertices.append([x, y, z])
    mesh_vertices = np.array(mesh_vertices)
    assert len(mesh_vertices) <= 12

    mesh_elements = []
    mesh_elements.extend(_split_to_tetrahedra(
        m[1, 0, 0], m[1, 1, 0], m[1, 1, 1], m[1, 0, 1],
        v[1, 0, 0], v[1, 1, 0], v[1, 1, 1], v[1, 0, 1]
    ))
    mesh_elements.extend(_split_to_tetrahedra(
        m[0, 0, 0], m[0, 0, 1], m[0, 1, 1], m[0, 1, 0],
        v[0, 0, 0], v[0, 0, 1], v[0, 1, 1], v[0, 1, 0]
    ))
    mesh_elements.extend(_split_to_tetrahedra(
        m[0, 1, 0], m[0, 1, 1], m[1, 1, 1], m[1, 1, 0],
        v[0, 1, 0], v[0, 1, 1], v[1, 1, 1], v[1, 1, 0]
    ))
    mesh_elements.extend(_split_to_tetrahedra(
        m[0, 0, 0], m[1, 0, 0], m[1, 0, 1], m[0, 0, 1],
        v[0, 0, 0], v[1, 0, 0], v[1, 0, 1], v[0, 0, 1]
    ))
    mesh_elements.extend(_split_to_tetrahedra(
        m[0, 0, 1], m[1, 0, 1], m[1, 1, 1], m[0, 1, 1],
        v[0, 0, 1], v[1, 0, 1], v[1, 1, 1], v[0, 1, 1]
    ))
    mesh_elements.extend(_split_to_tetrahedra(
        m[0, 0, 0], m[0, 1, 0], m[1, 1, 0], m[1, 0, 0],
        v[0, 0, 0], v[0, 1, 0], v[1, 1, 0], v[1, 0, 0]
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


def make_tetrahedral_cylinder(radius, length, resolution_hint):
    """Create a tetrahedral cylinder mesh.

    Source: Drake (https://github.com/RobotLocomotion/drake/blob/665f178d7c61edef1a4961cc44bb320062224944/geometry/proximity/make_cylinder_mesh.cc#L345),
    BSD 3-clause

    Parameters
    ----------
    radius : float
        Radius of the cylinder.

    length : float
        Length of the cylinder.

    resolution_hint : float
        Controls the fineness of the tetrahedral mesh. The coarsest mesh
        that produces desirable results will allow simulation to run as
        efficiently as possible. The circles of the cylinder will have
        2 * pi * radius / resolution_hint edges.

    Returns
    -------
    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    tetrahedra : array, shape (n_tetrahedra, 4)
        Indices of vertices that form tetrahedra of the mesh.

    potentials : array, shape (n_vertices, 3)
        Potential of each vertex.
    """
    top_z = 0.5 * length
    bottom_z = -top_z
    tolerance = 1e-14 * max(1.0, min(top_z, radius))
    cylinder_class = CylinderClass.Medium
    if top_z - radius > tolerance:
        cylinder_class = CylinderClass.Long
    elif radius - top_z > tolerance:
        cylinder_class = CylinderClass.Short

    n_vertices_per_circle = max(3, math.ceil(2.0 * np.pi * radius / resolution_hint))

    mesh_vertices = []

    bottom_center = len(mesh_vertices)
    mesh_vertices.append(np.array([0.0, 0.0, bottom_z]))
    top_center = len(mesh_vertices)
    mesh_vertices.append(np.array([0.0, 0.0, top_z]))

    bottom = []
    top = []
    angle_step = 2.0 * np.pi / n_vertices_per_circle
    for i in range(n_vertices_per_circle):
        x = radius * np.cos(angle_step * i)
        y = radius * np.sin(angle_step * i)
        bottom.append(len(mesh_vertices))
        mesh_vertices.append(np.array([x, y, bottom_z]))
        top.append(len(mesh_vertices))
        mesh_vertices.append(np.array([x, y, top_z]))

    n_outer_vertices = len(mesh_vertices)

    potentials = [0.0] * n_outer_vertices

    if cylinder_class == CylinderClass.Long:
        mesh_elements = _calc_long_cylinder_volume_mesh_with_ma(
            radius, length, n_vertices_per_circle, bottom_center, bottom,
            top_center, top, mesh_vertices, potentials)
    elif cylinder_class == CylinderClass.Medium:
        mesh_elements = _calc_medium_cylinder_volume_mesh_with_ma(
            radius, n_vertices_per_circle, bottom_center, bottom, top_center,
            top, mesh_vertices, potentials)
    else:
        assert cylinder_class == CylinderClass.Short
        mesh_elements = _calc_short_cylinder_volume_mesh_with_ma(
            radius, length, n_vertices_per_circle, bottom_center, bottom,
            top_center, top, mesh_vertices, potentials)

    return (np.array(mesh_vertices), np.array(mesh_elements, dtype=int),
            np.array(potentials))


class CylinderClass(enum.Enum):
    Long = 0
    Medium = 1
    Short = 2


def _calc_long_cylinder_volume_mesh_with_ma(
        radius, length, n_vertices_per_circle, bottom_center, bottom,
        top_center, top, mesh_vertices, potentials):
    medial = []
    offset_distance = radius
    top_z = 0.5 * length
    offset_top_z = top_z - offset_distance
    offset_bottom_z = -offset_top_z
    medial.append(len(mesh_vertices))
    mesh_vertices.append(np.array([0.0, 0.0, offset_bottom_z]))
    potentials.append(radius)
    medial.append(len(mesh_vertices))
    mesh_vertices.append(np.array([0.0, 0.0, offset_top_z]))
    potentials.append(radius)

    mesh_elements = []
    i = n_vertices_per_circle - 1
    for j in range(n_vertices_per_circle):
        mesh_elements.append([bottom_center, bottom[i], bottom[j], medial[0]])
        mesh_elements.append([top_center, top[j], top[i], medial[1]])
        mesh_elements.extend(_split_triangular_prism_to_tetrahedra(
            medial[0], bottom[i], bottom[j], medial[1], top[i], top[j]))
        i = j

    return mesh_elements


def _calc_medium_cylinder_volume_mesh_with_ma(
        radius, n_vertices_per_circle, bottom_center, bottom, top_center, top,
        mesh_vertices, potentials):
    medial = len(mesh_vertices)
    mesh_vertices.append(np.array([0.0, 0.0, 0.0]))
    potentials.append(radius)

    mesh_elements = []
    i = n_vertices_per_circle - 1
    for j in range(n_vertices_per_circle):
        mesh_elements.append([bottom_center, bottom[i], bottom[j], medial])
        mesh_elements.append([top_center, top[j], top[i], medial])
        mesh_elements.extend(_split_pyramid_to_tetrahedra(
            top[i], top[j], bottom[j], bottom[i], medial))
        i = j

    return mesh_elements


def _calc_short_cylinder_volume_mesh_with_ma(
        radius, length, n_vertices_per_circle, bottom_center, bottom,
        top_center, top, mesh_vertices, potentials):
    center = len(mesh_vertices)
    mesh_vertices.append(np.array([0.0, 0.0, 0.0]))
    half_length = 0.5 * length
    potentials.append(half_length)

    medial = []
    medial_radius = radius - half_length
    scale_cylinder_radius_to_medial_circle = medial_radius / radius
    for i in range(n_vertices_per_circle):
        x = mesh_vertices[bottom[i]][0] * scale_cylinder_radius_to_medial_circle
        y = mesh_vertices[bottom[i]][1] * scale_cylinder_radius_to_medial_circle
        medial.append(len(mesh_vertices))
        mesh_vertices.append(np.array([x, y, 0.0]))
        potentials.append(half_length)

    mesh_elements = []
    i = n_vertices_per_circle - 1
    for j in range(n_vertices_per_circle):
        mesh_elements.extend(_split_triangular_prism_to_tetrahedra(
            bottom_center, bottom[i], bottom[j], center, medial[i], medial[j]))
        mesh_elements.extend(_split_triangular_prism_to_tetrahedra(
            center, medial[i], medial[j], top_center, top[i], top[j]))
        mesh_elements.extend(_split_triangular_prism_to_tetrahedra(
            bottom[i], medial[i], top[i], bottom[j], medial[j], top[j]))
        i = j

    return mesh_elements


def _split_triangular_prism_to_tetrahedra(v0, v1, v2, v3, v4, v5):
    elements = []
    previous = v3
    for next in [v4, v1, v2]:
        elements.append([previous, next, v0, v5])
        previous = next
    return elements


def _split_pyramid_to_tetrahedra(v0, v1, v2, v3, v4):
    elements = []
    previous = v3
    for next in [v4, v1]:
        elements.append([previous, next, v0, v2])
        previous = next
    return elements


def make_tetrahedral_capsule(radius, height, resolution_hint):
    """Create a tetrahedral capsule mesh.

    Source: Drake (https://github.com/RobotLocomotion/drake/blob/903019faf53b771e8e2fe81222ffd74eae2dc85c/geometry/proximity/make_capsule_mesh.cc#L14),
    BSD 3-clause

    Parameters
    ----------
    radius : float
        Radius of the capsule.

    height : float
        Height of the capsule.

    resolution_hint : float
        Controls the fineness of the tetrahedral mesh. The coarsest mesh
        that produces desirable results will allow simulation to run as
        efficiently as possible. The circles of the cylinder and great circles
        of each hemisphere will have 2 * pi * radius / resolution_hint edges.

    Returns
    -------
    vertices : array, shape (n_vertices, 3)
        Vertices of the mesh.

    tetrahedra : array, shape (n_tetrahedra, 4)
        Indices of vertices that form tetrahedra of the mesh.

    potentials : array, shape (n_vertices, 3)
        Potential of each vertex.
    """
    medial_top_z = 0.5 * height
    medial_bottom_z = -medial_top_z
    top_z = medial_top_z + radius
    bottom_z = -top_z

    n_vertices_per_circle = int(np.clip(2.0 * np.pi * radius / resolution_hint,
                                        3.0, 706.0))
    n_circles_per_cap = n_vertices_per_circle // 2

    mesh_vertices = []

    medial_top = len(mesh_vertices)
    mesh_vertices.append(np.array([0.0, 0.0, medial_top_z]))
    medial_bottom = len(mesh_vertices)
    mesh_vertices.append(np.array([0.0, 0.0, medial_bottom_z]))
    top = len(mesh_vertices)
    mesh_vertices.append(np.array([0.0, 0.0, top_z]))
    bottom = len(mesh_vertices)
    mesh_vertices.append(np.array([0.0, 0.0, bottom_z]))

    top_cap = []
    bottom_cap = []

    theta_step = 0.5 * np.pi / n_circles_per_cap
    phi_step = 2.0 * np.pi / n_vertices_per_circle

    for i in range(n_circles_per_cap):
        theta = 0.5 * np.pi - i * theta_step
        s = np.sin(theta)
        top_circle_z = radius * np.cos(theta) + medial_top_z
        bottom_circle_z = -top_circle_z
        for j in range(n_vertices_per_circle):
            phi = j * phi_step
            x = radius * s * np.cos(phi)
            y = radius * s * np.sin(phi)

            top_cap.append(len(mesh_vertices))
            mesh_vertices.append(np.array([x, y, top_circle_z]))
            bottom_cap.append(len(mesh_vertices))
            mesh_vertices.append(np.array([x, y, bottom_circle_z]))

    mesh_elements = []
    for i in range(n_circles_per_cap - 1):
        for j in range(n_vertices_per_circle):
            j1 = (j + 1) % n_vertices_per_circle
            mesh_elements.extend(_split_pyramid_to_tetrahedra(
                top_cap[(i + 1) * n_vertices_per_circle + j],
                top_cap[(i + 1) * n_vertices_per_circle + j1],
                top_cap[i * n_vertices_per_circle + j1],
                top_cap[i * n_vertices_per_circle + j],
                medial_top
            ))
            mesh_elements.extend(_split_pyramid_to_tetrahedra(
                bottom_cap[i * n_vertices_per_circle + j],
                bottom_cap[i * n_vertices_per_circle + j1],
                bottom_cap[(i + 1) * n_vertices_per_circle + j1],
                bottom_cap[(i + 1) * n_vertices_per_circle + j],
                medial_bottom
            ))

    last_circle_offset = (n_circles_per_cap - 1) * n_vertices_per_circle
    for j in range(n_vertices_per_circle):
        j1 = (j + 1) % n_vertices_per_circle
        mesh_elements.append([top, top_cap[last_circle_offset + j1],
                              top_cap[last_circle_offset + j], medial_top])
        mesh_elements.append([bottom, bottom_cap[last_circle_offset + j],
                              bottom_cap[last_circle_offset + j1],
                              medial_bottom])
        mesh_elements.extend(_split_triangular_prism_to_tetrahedra(
            medial_bottom, bottom_cap[j], bottom_cap[j1], medial_top,
            top_cap[j], top_cap[j1]))

    potentials = np.zeros(len(mesh_vertices))
    potentials[:2] = radius

    return (np.array(mesh_vertices), np.array(mesh_elements, dtype=int),
            potentials)
