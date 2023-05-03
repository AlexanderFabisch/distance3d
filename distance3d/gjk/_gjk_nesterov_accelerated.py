import numpy as np
import numba

from ..colliders import MeshGraph, Capsule, Sphere, Box, Ellipse, Cone, Cylinder, Ellipsoid
from ..utils import norm_vector, EPSILON


def gjk_nesterov_accelerated_intersection(collider1, collider2, ray_guess=None):
    """
    Parameters
    ----------
    collider1 : ConvexCollider
        Convex collider 1.

    collider2 : ConvexCollider
        Convex collider 2.

    Returns
    -------
    contact : bool
        Shapes collide
    """
    return gjk_nesterov_accelerated(collider1, collider2, ray_guess)[0]


def gjk_nesterov_accelerated_distance(collider1, collider2, ray_guess=None):
    """
    Parameters
    ----------
    collider1 : ConvexCollider
        Convex collider 1.

    collider2 : ConvexCollider
        Convex collider 2.

    Returns
    -------
    contact : bool
        Shapes collide
    """
    return max(gjk_nesterov_accelerated(collider1, collider2, ray_guess)[1], 0.0)


def gjk_nesterov_accelerated_iterations(collider1, collider2, ray_guess=None):
    """
    Parameters
    ----------
    collider1 : ConvexCollider
        Convex collider 1.

    collider2 : ConvexCollider
        Convex collider 2.

    Returns
    -------
    contact : bool
        Shapes collide
    """
    return gjk_nesterov_accelerated(collider1, collider2, ray_guess)[3]


def gjk_nesterov_accelerated(collider0, collider1, ray_guess=None, max_interations=128, upper_bound=1.79769e+308, tolerance=1e-6):
    """
    Parameters
    ----------
    collider0 : ConvexCollider
        Convex collider 1.

    collider1 : ConvexCollider
        Convex collider 2.

    Returns
    -------
    contact : bool
        Shapes collide

    distance : float
        Distance between shapes

    simplex :
        Distance between shapes
    """

    # ------ Initialize Variables ------
    use_nesterov_acceleration = False

    # normalize_support_direction is for soem reason only needed when both colliders are an mesh.
    normalize_support_direction = type(collider0) == MeshGraph and type(collider1) == MeshGraph

    # Infaltion is only used with spheres and capsules
    inflation = 0.0
    if type(collider0) == Sphere or type(collider0) == Capsule:
        inflation += collider0.radius

    if type(collider1) == Sphere or type(collider1) == Capsule:
        inflation += collider1.radius

    upper_bound += inflation

    alpha = 0.0

    inside = False
    simplex = np.empty([4, 3, 3], dtype=float)
    simplex_len = 0
    distance = 0.0

    ray = np.array([1.0, 0.0, 0.0])  # x in paper
    if ray_guess is not None:
        ray = ray_guess

    ray_len = np.linalg.norm(ray)
    if ray_len < tolerance:
        ray = np.array([-1.0, 0.0, 0.0])
        ray_len = 1

    ray_dir = ray  # d in paper
    support_point = np.array((ray, ray, ray))  # s in paper

    iterations = 0
    for k in range(max_interations):
        iterations = k
        if ray_len < tolerance:
            distance = -inflation
            inside = True
            break

        if use_nesterov_acceleration:
            ray_dir = nesterov_direction(k, normalize_support_direction, ray, ray_dir, support_point)
        else:
            ray_dir = ray

        s0, s1 = support_function(-ray_dir, collider0, collider1)

        distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, support_point, converged = iteration(
            alpha, distance, inflation, inside, k, ray, ray_dir, ray_len,
            simplex, simplex_len, tolerance, upper_bound,
            use_nesterov_acceleration, s0, s1)
        if converged:
            break

    return inside, distance, simplex, iterations


@numba.njit(
    numba.float64[::1](
        numba.int64, numba.bool_, numba.float64[::1], numba.float64[::1],
        numba.float64[:, ::1]
    ),
    cache=True)
def nesterov_direction(k, normalize_support_direction, ray, ray_dir, support_point):
    if normalize_support_direction:
        momentum = (k + 2) / (k + 3)
        y = momentum * ray + (1.0 - momentum) * support_point[0]
        ray_dir = momentum * norm_vector(ray_dir) + (1.0 - momentum) * norm_vector(y)
    else:
        momentum = (k + 1) / (k + 3)
        y = momentum * ray + (1.0 - momentum) * support_point[0]
        ray_dir = momentum * ray_dir + (1.0 - momentum) * y

    return ray_dir


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.int64))(
        numba.float64[:, :, ::1], numba.int64, numba.float64[::1]
    ),
    cache=True)
def origin_to_point(simplex, a_index, a):
    simplex[0] = a
    return np.copy(a), 1


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.int64))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.float64[::1],
        numba.float64[::1], numba.float64[::1], numba.float64
    ),
    cache=True)
def origin_to_segment(simplex, a_index, b_index, a, b, ab, ab_dot_a0):
    ray = (ab.dot(b) * a + ab_dot_a0 * b) / ab.dot(ab)
    simplex[0], simplex[1] = b, a
    return ray, 2


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.int64, numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.int64,
        numba.float64[::1], numba.float64
    ),
    cache=True)
def origin_to_triangle(simplex, a, b, c, abc, abc_dot_a0):
    if abc_dot_a0 == 0:
        simplex[0], simplex[1], simplex[2] = c, b, a
        return np.zeros(3), 3, True

    if abc_dot_a0 > 0:
        simplex[0], simplex[1], simplex[2] = c, b, a
    else:
        simplex[0], simplex[1], simplex[2] = np.copy(b), np.copy(c), np.copy(a)

    ray = -abc_dot_a0 / abc.dot(abc) * abc
    return ray, 3, False


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.int64, numba.bool_))(
        numba.float64[:, :, ::1]
    ),
    cache=True)
def project_line_origin(line):
    # A is the last point we added.
    a_index = 1
    b_index = 0

    a = line[a_index, 0]
    b = line[b_index, 0]

    ab = b - a
    d = np.dot(ab, -a)

    if d == 0:
        # Two extremely unlikely cases:
        #  - AB is orthogonal to A: should never happen because it means the support
        #    function did not do any progress and GJK should have stopped.
        #  - A == origin
        # In any case, A is the closest to the origin
        ray, simplex_len = origin_to_point(line, a_index, a)
        return ray, simplex_len, np.all(a == 0.0)
    if d < 0:
        ray, simplex_len = origin_to_point(line, a_index, a)
    else:
        ray, simplex_len = origin_to_segment(line, a_index, b_index, a, b, ab, d)

    return ray, simplex_len, False


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.float64[::1],
        numba.float64[::1], numba.float64[::1]
    ),
    cache=True)
def t_b(triangle, a_index, b_index, a, b, ab):
    towards_b = ab.dot(-a)
    if towards_b < 0:
        return origin_to_point(triangle, a_index, a)
    else:
        return origin_to_segment(triangle, a_index, b_index, a, b, ab, towards_b)


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.int64, numba.bool_))(
        numba.float64[:, :, ::1]
    ),
    cache=True)
def project_triangle_origin(triangle):
    # A is the last point we added.
    a_index = 2
    b_index = 1
    c_index = 0

    a = triangle[a_index, 0]
    b = triangle[b_index, 0]
    c = triangle[c_index, 0]

    ab = b - a
    ac = c - a
    abc = np.cross(ab, ac)

    edge_ac2o = np.cross(abc, ac).dot(-a)

    if edge_ac2o >= 0:

        towards_c = ac.dot(-a)
        if towards_c >= 0:
            ray, simplex_len = origin_to_segment(triangle, a_index, c_index, a, c, ac, towards_c)
        else:
            ray, simplex_len = t_b(triangle, a_index, b_index, a, b, ab)
    else:

        edge_ab2o = np.cross(ab, abc).dot(-a)
        if edge_ab2o >= 0:
            ray, simplex_len = t_b(triangle, a_index, b_index, a, b, ab)
        else:
            return origin_to_triangle(triangle, a, b, c, abc, abc.dot(-a))

    return ray, simplex_len, False


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.float64[::1]
    ),
    cache=True)
def region_a(simplex, a_index, a):
    return origin_to_point(simplex, a_index, a)


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.float64[::1],
        numba.float64[::1], numba.float64
    ),
    cache=True)
def region_ab(simplex, a_index, b_index, a, b, ba_aa):
    return origin_to_segment(simplex, a_index, b_index, a, b, b - a, -ba_aa)


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.float64[::1],
        numba.float64[::1], numba.float64
    ),
    cache=True)
def region_ac(simplex, a_index, c_index, a, c, ca_aa):
    return origin_to_segment(simplex, a_index, c_index, a, c, c - a, -ca_aa)


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.float64[::1],
        numba.float64[::1], numba.float64
    ),
    cache=True)
def region_ad(simplex, a_index, d_index, a, d, da_aa):
    return origin_to_segment(simplex, a_index, d_index, a, d, d - a, -da_aa)


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.int64,
        numba.float64[::1], numba.float64[::1], numba.float64[::1],
        numba.float64[::1]
    ),
    cache=True)
def region_abc(simplex, a_index, b_index, c_index, a, b, c, a_cross_b):
    return origin_to_triangle(simplex, a, b, c, np.cross(b - a, c - a), -c.dot(a_cross_b))[:2]


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.int64,
        numba.float64[::1], numba.float64[::1], numba.float64[::1],
        numba.float64[::1]
    ),
    cache=True)
def region_acd(simplex, a_index, c_index, d_index, a, c, d, a_cross_c):
    return origin_to_triangle(simplex, a, c, d, np.cross(c - a, d - a), -d.dot(a_cross_c))[:2]


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.int64,
        numba.float64[::1], numba.float64[::1], numba.float64[::1],
        numba.float64[::1]
    ),
    cache=True)
def region_adb(simplex, a_index, d_index, b_index, a, d, b, a_cross_b):
    return origin_to_triangle(simplex, a, d, b, np.cross(d - a, b - a), d.dot(a_cross_b))[:2]


@numba.njit(
    numba.bool_(numba.float64, numba.float64, numba.float64, numba.float64),
    cache=True)
def check_convergence(alpha, omega, ray_len, tolerance):
    alpha = max(alpha, omega)
    diff = ray_len - alpha
    return (diff - tolerance * ray_len) <= 0


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.int64, numba.bool_))(
        numba.float64[:, :, ::1]
    ),
    cache=True)
def project_tetra_to_origin(tetra):
    a_index = 3
    b_index = 2
    c_index = 1
    d_index = 0

    a = tetra[a_index, 0]
    b = tetra[b_index, 0]
    c = tetra[c_index, 0]
    d = tetra[d_index, 0]

    aa = a.dot(a)

    da = d.dot(a)
    db = d.dot(b)
    dc = d.dot(c)
    dd = d.dot(d)
    da_aa = da - aa

    ca = c.dot(a)
    cb = c.dot(b)
    cc = c.dot(c)
    ca_aa = ca - aa

    ba = b.dot(a)
    bb = b.dot(b)
    bc = cb
    bd = db
    ba_aa = ba - aa
    ba_ca = ba - ca
    ca_da = ca - da
    da_ba = da - ba

    a_cross_b = np.cross(a, b)
    a_cross_c = np.cross(a, c)

    if ba_aa <= 0:
        if -d.dot(a_cross_b) <= 0:
            if ba * da_ba + bd * ba_aa - bb * da_aa <= 0:
                if da_aa <= 0:
                    if ba * ba_ca + bb * ca_aa - bc * ba_aa <= 0:
                        ray, simplex_len = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                    else:
                        ray, simplex_len = region_ab(tetra, a_index, b_index, a, b, ba_aa)
                else:
                    if ba * ba_ca + bb * ca_aa - bc * ba_aa <= 0:
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                            if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                                ray, simplex_len = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                            else:
                                ray, simplex_len = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                        else:
                            ray, simplex_len = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                    else:
                        ray, simplex_len = region_ab(tetra, a_index, b_index, a, b, ba_aa)
            else:
                if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                    ray, simplex_len = region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                else:
                    if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex_len = region_ad(tetra, a_index, d_index, a, d, da_aa)
                        else:
                            ray, simplex_len = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                    else:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex_len = region_ad(tetra, a_index, d_index, a, d, da_aa)
                        else:
                            ray, simplex_len = region_ac(tetra, a_index, c_index, a, c, ca_aa)
        else:
            if c.dot(a_cross_b) <= 0:
                if ba * ba_ca + bb * ca_aa - bc * ba_aa <= 0:
                    if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                        if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                            ray, simplex_len = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                        else:
                            ray, simplex_len = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                    else:
                        ray, simplex_len = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                else:
                    ray, simplex_len = region_ad(tetra, a_index, d_index, a, d, da_aa)
            else:
                if d.dot(a_cross_c) <= 0:
                    if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex_len = region_ad(tetra, a_index, d_index, a, d, da_aa)
                        else:
                            ray, simplex_len = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                    else:
                        if ca_aa <= 0:
                            ray, simplex_len = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                        else:
                            ray, simplex_len = region_ad(tetra, a_index, d_index, a, d, da_aa)
                else:
                    return np.zeros(3), 4, True
    else:
        if ca_aa <= 0:
            if d.dot(a_cross_c) <= 0:
                if da_aa <= 0:
                    if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                                ray, simplex_len = region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                            else:
                                ray, simplex_len = region_ad(tetra, a_index, d_index, a, d, da_aa)
                        else:
                            ray, simplex_len = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                    else:
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                            ray, simplex_len = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                        else:
                            ray, simplex_len = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                else:
                    if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                        if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                            ray, simplex_len = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                        else:
                            ray, simplex_len = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                    else:
                        if c.dot(a_cross_b):
                            ray, simplex_len = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                        else:
                            ray, simplex_len = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
            else:
                if c.dot(a_cross_b) <= 0:
                    if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                        ray, simplex_len = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                    else:
                        ray, simplex_len = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                else:
                    if -d.dot(a_cross_b) <= 0:
                        if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                            ray, simplex_len = region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                        else:
                            ray, simplex_len = region_ad(tetra, a_index, d_index, a, d, da_aa)
                    else:
                        return np.zeros(3), 4, True
        else:
            if da_aa <= 0:
                if -d.dot(a_cross_b) <= 0:
                    if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                        if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                            ray, simplex_len = region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                        else:
                            ray, simplex_len = region_ad(tetra, a_index, d_index, a, d, da_aa)
                    else:
                        if d.dot(a_cross_c) <= 0:
                            ray, simplex_len = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                        else:
                            if c.dot(a_cross_b) <= 0:  # ???
                                ray, simplex_len = region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                            else:
                                ray, simplex_len= region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                else:
                    if d.dot(a_cross_c) <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex_len = region_ad(tetra, a_index, d_index, a, d, da_aa)
                        else:
                            ray, simplex_len = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                    else:
                        return np.zeros(3), 4, True
            else:
                ray, simplex_len = region_a(tetra, a_index, a)
    return ray, simplex_len, False


@numba.njit(
    numba.types.Tuple((
            numba.float64, numba.bool_, numba.float64[::1], numba.float64,
            numba.int64, numba.bool_, numba.bool_))(
        numba.float64, numba.float64, numba.float64, numba.bool_, numba.int64,
        numba.float64[::1], numba.float64[::1], numba.float64,
        numba.float64[:, :, ::1], numba.int64, numba.float64, numba.float64,
        numba.bool_, numba.float64[::1], numba.float64[::1]
    ),
    cache=True)
def iteration(alpha, distance, inflation, inside, k, ray, ray_dir, ray_len,
              simplex, simplex_len, tolerance, upper_bound,
              use_nesterov_acceleration, s0, s1):
    simplex[simplex_len, 0] = s0 - s1
    simplex[simplex_len, 1] = s0
    simplex[simplex_len, 2] = s1
    support_point = simplex[simplex_len]
    simplex_len += 1

    omega = ray_dir.dot(support_point[0]) / np.linalg.norm(ray_dir)
    if omega > upper_bound:
        distance = omega - inflation
        inside = False
        return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, support_point, True

    if use_nesterov_acceleration:
        frank_wolfe_duality_gap = 2 * ray.dot(ray - support_point[0])
        if frank_wolfe_duality_gap - tolerance <= 0:
            use_nesterov_acceleration = False
            simplex_len -= 1
            return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, support_point, False

    cv_check_passed = check_convergence(alpha, omega, ray_len, tolerance)
    if k > 0 and cv_check_passed:
        if k > 0:
            simplex_len -= 1
        if use_nesterov_acceleration:
            use_nesterov_acceleration = False
            return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, support_point, False
        distance = ray_len - inflation

        if distance < tolerance:
            inside = True
        return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, support_point, True

    assert 1 <= simplex_len <= 4
    if simplex_len == 1:
        ray = np.copy(support_point[0])
    elif simplex_len == 2:
        ray, simplex_len, inside = project_line_origin(simplex)
    elif simplex_len == 3:
        ray, simplex_len, inside = project_triangle_origin(simplex)
    elif simplex_len == 4:
        ray, simplex_len, inside = project_tetra_to_origin(simplex)

    if not inside:
        ray_len = np.linalg.norm(ray)
    if inside or ray_len == 0:
        distance = -inflation - 1.
        inside = True
        return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, support_point, True

    return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, support_point, False


def support_function(dir, collider0, collider1):
    oR1 = np.dot(collider0.frame()[:3, :3].T, collider1.frame()[:3, :3])
    ot1 = np.dot(collider0.frame()[:3, :3].T, collider1.center() - collider0.frame()[:3, 3])

    support0, found0 = select_support(dir, collider0)

    support1, found1 = select_support(np.dot(-oR1.T, dir), collider1)
    support1 = np.dot(oR1, support1) + ot1

    if found0 and found1:
        return support0, support1

    return collider0.support_function(dir), collider1.support_function(-dir)


def select_support(dir, collider):
    if type(collider) == Sphere:
        return sphere_support(), True

    if type(collider) == Capsule:
        return capsule_support(dir, collider), True

    if type(collider) == Box:
        return box_support(dir, collider), True

    if type(collider) == Ellipsoid:
        return ellipsoid_support(dir, collider), True

    if type(collider) == Cone:
        return cone_support(dir, collider), True

    if type(collider) == Cylinder:
        return cylinder_support(dir, collider), True

    # Type not found
    return np.array([0.0, 0.0, 0.0]), False


def sphere_support():
    return np.array([0.0, 0.0, 0.0])


def capsule_support(dir, capsule):
    support = np.array([0.0, 0.0, 0.0])
    if dir[2] > 0:
        support[2] = capsule.height / 2
    else:
        support[2] = -capsule.height / 2

    return support


def box_support(dir, box):
    inflate = 1.0
    if (dir == 0).any():
        inflate = 1.00000001

    support = np.array([0.0, 0.0, 0.0])
    for i in range(0, 3):
        if dir[i] > 0:
            support[i] = inflate * (box.size[i] / 2)
        else:
            support[i] = -inflate * (box.size[i] / 2)

    return support


def ellipsoid_support(dir, ellipsoid):
    a2 = ellipsoid.radii[0] * ellipsoid.radii[0]
    b2 = ellipsoid.radii[1] * ellipsoid.radii[1]
    c2 = ellipsoid.radii[2] * ellipsoid.radii[2]

    v = np.array([a2 * dir[0], b2 * dir[1], c2 * dir[2]])
    d = np.sqrt(v.dot(dir))

    return v / d


def cone_support(dir, cone):
    support = np.array([0.0, 0.0, 0.0])

    inflate = 1.00001
    h = cone.height / 2
    r = cone.radius

    if (dir[:2] == 0).all():
        dir[0] = 0.0
        dir[1] = 0.0

        if dir[2] > 0:
            support[2] = h
        else:
            support[2] = -inflate * h
        return support

    zdist = dir[0] * dir[0] + dir[1] * dir[1]
    len = zdist + dir[2] * dir[2]
    zdist = np.sqrt(zdist)

    if dir[2] <= 0:
        rad = r / zdist
        support[:2] = rad * dir[:2]
        support[2] = -h
        return support

    len = np.sqrt(len)
    sin_a = r / np.sqrt(r * r + 4 * h * h)

    if dir[2] > len * sin_a:
        support = np.array([0.0, 0.0, h])
        return support

    rad = r / zdist
    support[:2] = rad * dir[:2]
    support[2] = -h
    return support


def cylinder_support(dir, cylinder):
    support = np.array([0.0, 0.0, 0.0])

    inflate = 1.00001

    half_h = cylinder.length / 2
    r = cylinder.radius

    if (dir[:2] == np.array([0.0, 0.0])).all():
        half_h *= inflate

    if dir[2] > 0:
        support[2] = half_h
    elif dir[2] < 0:
        support[2] = -half_h
    else:
        support[2] = 0
        r *= inflate

    if (dir[:2] == np.array([0.0, 0.0])).all():
        support[0] = 0.0
        support[1] = 0.0
    else:
        support[:2] = dir[:2] / np.linalg.norm(dir[:2]) * r

    return support







