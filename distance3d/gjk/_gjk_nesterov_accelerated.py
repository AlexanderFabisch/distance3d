import numpy as np
import numba

from ..colliders import MeshGraph
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


def gjk_nesterov_accelerated(collider1, collider2, ray_guess=None, max_interations=100, upper_bound=1000000000, tolerance=1e-6):
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

    distance : float
        Distance between shapes

    simplex :
        Distance between shapes
    """

    # ------ Initialize Variables ------
    use_nesterov_acceleration = True
    normalize_support_direction = type(collider1) == MeshGraph and type(collider2) == MeshGraph

    inflation = 0

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
        ray = np.array([1.0, 0.0, 0.0])
        ray_len = 1

    ray_dir = ray  # d in paper
    support_point = np.array((ray, ray, ray))  # s in paper

    for k in range(max_interations):
        if ray_len < tolerance:
            distance = -inflation
            inside = True
            break

        if use_nesterov_acceleration:
            ray_dir = nesterov_direction(k, normalize_support_direction, ray, ray_dir, support_point)
        else:
            ray_dir = ray

        s0 = collider1.support_function(-ray_dir)
        s1 = collider2.support_function(ray_dir)

        simplex[simplex_len, 0] = s0 - s1
        simplex[simplex_len, 1] = s0
        simplex[simplex_len, 2] = s1
        support_point = simplex[simplex_len]
        simplex_len += 1

        distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, converged = iteration(
            alpha, distance, inflation, inside, k, ray, ray_dir, ray_len,
            simplex, simplex_len, support_point, tolerance, upper_bound,
            use_nesterov_acceleration)
        if converged:
            break

    return inside, distance, simplex


@numba.njit(
    numba.float64[::1](
        numba.int64, numba.bool_, numba.float64[::1], numba.float64[::1],
        numba.float64[:, ::1]
    ),
    cache=True)
def nesterov_direction(k, normalize_support_direction, ray, ray_dir, support_point):
    momentum = (k + 1) / (k + 3)
    minv = (1.0 - momentum)
    y = momentum * ray + minv * support_point[0]
    ray_dir = momentum * ray_dir + minv * y
    if normalize_support_direction:
        ray_dir = norm_vector(ray_dir)
    return ray_dir


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.int64))(
        numba.float64[:, :, ::1], numba.int64, numba.float64[::1]
    ),
    cache=True)
def origin_to_point(simplex, a_index, a):
    simplex[0] = simplex[a_index]
    return a, 1


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.int64))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.float64[::1],
        numba.float64[::1], numba.float64[::1], numba.float64
    ),
    cache=True)
def origin_to_segment(simplex, a_index, b_index, a, b, ab, ab_dot_a0):
    ray = (ab.dot(b) * a + ab_dot_a0 * b) / ab.dot(ab)
    # TODO check if swapping works correctly
    simplex[0] = simplex[b_index]
    simplex[1] = simplex[a_index]
    return ray, 2


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.int64, numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.int64,
        numba.float64[::1], numba.float64[::1], numba.float64[::1],
        numba.float64[::1], numba.float64
    ),
    cache=True)
def origin_to_triangle(simplex, a_index, b_index, c_index, a, b, c, abc, abc_dot_a0):
    # TODO check if swapping works correctly
    if abc_dot_a0 == 0:
        simplex[0] = simplex[c_index]
        simplex[1] = simplex[b_index]
        simplex[2] = simplex[a_index]

        return np.zeros(3), 3, True

    if abc_dot_a0 > 0:
        simplex[0] = simplex[c_index]
        simplex[1] = simplex[b_index]
        simplex[2] = simplex[a_index]
    else:
        simplex[0] = simplex[b_index]
        simplex[1] = simplex[c_index]
        simplex[2] = simplex[a_index]

    abc_sq_norm = abc.dot(abc)
    ray = -abc_dot_a0 * abc
    if abc_sq_norm >= EPSILON:
        ray /= abc_sq_norm
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
            return origin_to_triangle(triangle, a_index, b_index, c_index, a, b, c, abc, abc.dot(-a))

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
    return origin_to_triangle(simplex, a_index, b_index, c_index, a, b, c, np.cross(b - a, c - a), -c.dot(a_cross_b))[:2]


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.int64,
        numba.float64[::1], numba.float64[::1], numba.float64[::1],
        numba.float64[::1]
    ),
    cache=True)
def region_acd(simplex, a_index, c_index, d_index, a, c, d, a_cross_c):
    return origin_to_triangle(simplex, a_index, c_index, d_index, a, c, d, np.cross(c - a, d - a), -d.dot(a_cross_c))[:2]


@numba.njit(
    numba.types.Tuple((numba.float64[::1], numba.bool_))(
        numba.float64[:, :, ::1], numba.int64, numba.int64, numba.int64,
        numba.float64[::1], numba.float64[::1], numba.float64[::1],
        numba.float64[::1]
    ),
    cache=True)
def region_adb(simplex, a_index, d_index, b_index, a, d, b, a_cross_b):
    return origin_to_triangle(simplex, a_index, d_index, b_index, a, d, b, np.cross(d - a, b - a), d.dot(a_cross_b))[:2]


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
        numba.float64[:, :, ::1], numba.int64, numba.float64[:, ::1],
        numba.float64, numba.float64, numba.bool_
    ),
    cache=True)
def iteration(alpha, distance, inflation, inside, k, ray, ray_dir, ray_len,
              simplex, simplex_len, support_point, tolerance, upper_bound,
              use_nesterov_acceleration):
    omega = ray_dir.dot(support_point[0]) / np.linalg.norm(ray_dir)
    if omega > upper_bound:
        distance = omega - inflation
        inside = False
        return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, True

    if use_nesterov_acceleration:
        frank_wolfe_duality_gap = 2 * ray.dot(ray - support_point[0])
        if frank_wolfe_duality_gap - tolerance <= 0:
            use_nesterov_acceleration = False
            simplex_len -= 1
            return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, False

    cv_check_passed = check_convergence(alpha, omega, ray_len, tolerance)
    if k > 0 and cv_check_passed:
        if k > 0:
            simplex_len -= 1
        if use_nesterov_acceleration:
            use_nesterov_acceleration = False
            return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, False
        distance = ray_len - inflation

        if distance < tolerance:
            inside = True
        return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, True

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
        distance = -inflation
        inside = True
        return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, True

    return distance, inside, ray, ray_len, simplex_len, use_nesterov_acceleration, False


def compare_gjk_intersection_flavours_with_random_shapes():
    from distance3d import colliders
    from distance3d.gjk import gjk_intersection_jolt
    from distance3d.gjk import gjk_intersection_libccd
    from distance3d import random

    random_state = np.random.RandomState(84)

    shape_names = list(colliders.COLLIDERS.keys())
    for i in range(100):
        print(i)

        shape1 = shape_names[random_state.randint(len(shape_names))]
        args1 = random.RANDOM_GENERATORS[shape1](random_state)
        shape2 = shape_names[random_state.randint(len(shape_names))]
        args2 = random.RANDOM_GENERATORS[shape2](random_state)
        collider1 = colliders.COLLIDERS[shape1](*args1)
        collider2 = colliders.COLLIDERS[shape2](*args2)


        intersection_jolt = gjk_intersection_jolt(collider1, collider2)
        intersection_libccd = gjk_intersection_libccd(collider1, collider2)

        if i == 68:
            print("break")

        intersection_nesterov = gjk_nesterov_accelerated_intersection(collider1, collider2)
        assert intersection_jolt == intersection_libccd
        assert intersection_jolt == intersection_nesterov
        assert intersection_nesterov == intersection_libccd


if __name__ == "__main__":
    compare_gjk_intersection_flavours_with_random_shapes()
