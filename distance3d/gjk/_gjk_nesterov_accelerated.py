import numpy as np
import numba

from ..colliders import MeshGraph
from ..utils import norm_vector


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


def gjk_nesterov_accelerated(collider1, collider2, ray_guess=None):
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
    max_interations = 100
    upper_bound = 1000000000
    tolerance = 1e-6

    use_nesterov_acceleration = True
    normalize_support_direction = type(collider1) == MeshGraph and type(collider2) == MeshGraph

    inflation = 0

    alpha = 0

    inside = False
    simplex = []
    distance = 0

    ray = np.array([1.0, 0.0, 0.0])  # x in paper
    if ray_guess is not None:
        ray = ray_guess

    ray_len = np.linalg.norm(ray)
    if ray_len < tolerance:
        ray = np.array([1.0, 0.0, 0.0])
        ray_len = 1

    ray_dir = ray  # d in paper
    support_point = np.array([ray, ray, ray])  # s in paper

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
        support_point = np.array([s0 - s1, s0, s1])
        simplex.append(support_point)

        distance, inside, ray, ray_len, simplex, use_nesterov_acceleration, converged = iteration(
            alpha, distance, inflation, inside, k, ray, ray_dir, ray_len,
            simplex, support_point, tolerance, upper_bound,
            use_nesterov_acceleration)
        if converged:
            break

    return inside, distance, simplex


@numba.njit(cache=True)
def nesterov_direction(k, normalize_support_direction, ray, ray_dir, support_point):
    momentum = (k + 1) / (k + 3)
    y = momentum * ray + (1 - momentum) * support_point[0]
    ray_dir = momentum * ray_dir + (1 - momentum) * y
    if normalize_support_direction:
        ray_dir = norm_vector(ray_dir)
    return ray_dir


@numba.njit(cache=True)
def iteration(alpha, distance, inflation, inside, k, ray, ray_dir, ray_len,
              simplex, support_point, tolerance, upper_bound,
              use_nesterov_acceleration):
    omega = ray_dir.dot(support_point[0]) / np.linalg.norm(ray_dir)
    if omega > upper_bound:
        distance = omega - inflation
        inside = False
        return distance, inside, ray, ray_len, simplex, use_nesterov_acceleration, True
    if use_nesterov_acceleration:
        frank_wolfe_duality_gap = 2 * ray.dot(ray - support_point[0])
        if frank_wolfe_duality_gap - tolerance <= 0:
            use_nesterov_acceleration = False
            simplex.pop()
            return distance, inside, ray, ray_len, simplex, use_nesterov_acceleration, False
    cv_check_passed = check_convergence(alpha, omega, ray_len, tolerance)
    if k > 0 and cv_check_passed:
        if k > 0:
            simplex.pop()
        if use_nesterov_acceleration:
            use_nesterov_acceleration = False
            return distance, inside, ray, ray_len, simplex, use_nesterov_acceleration, False
        distance = ray_len - inflation

        if distance < tolerance:
            inside = True
        return distance, inside, ray, ray_len, simplex, use_nesterov_acceleration, True

    assert 1 <= len(simplex) <= 4
    if len(simplex) == 1:
        ray = support_point[0]
    elif len(simplex) == 2:
        ray, simplex, inside = project_line_origin(simplex)
    elif len(simplex) == 3:
        ray, simplex, inside = project_triangle_origin(simplex)
    elif len(simplex) == 4:
        ray, simplex, inside = project_tetra_to_origin(simplex)
    if not inside:
        ray_len = np.linalg.norm(ray)
    if inside or ray_len == 0:
        distance = -inflation
        inside = True
        return distance, inside, ray, ray_len, simplex, use_nesterov_acceleration, True

    return distance, inside, ray, ray_len, simplex, use_nesterov_acceleration, False


@numba.njit(cache=True)
def origin_to_point(simplex, a_index, a):
    return a, [simplex[a_index]]


@numba.njit(cache=True)
def origin_to_segment(simplex, a_index, b_index, a, b, ab, ab_dot_a0):
    ray = (ab.dot(b) * a + ab_dot_a0 * b) / ab.dot(ab)
    simplex = [simplex[b_index], simplex[a_index]]
    return ray, simplex


@numba.njit(cache=True)
def project_line_origin(line):
    # A is the last point we added.
    a_index = 1
    b_index = 0

    a = line[a_index][0]
    b = line[b_index][0]

    ab = b - a
    d = np.dot(ab, -a)

    if d == 0:
        # Two extremely unlikely cases:
        #  - AB is orthogonal to A: should never happen because it means the support
        #    function did not do any progress and GJK should have stopped.
        #  - A == origin
        # In any case, A is the closest to the origin
        ray, simplex = origin_to_point(line, a_index, a)
        return ray, simplex, np.all(a == 0.0)
    if d < 0:
        ray, simplex = origin_to_point(line, a_index, a)
    else:
        ray, simplex = origin_to_segment(line, a_index, b_index, a, b, ab, d)

    return ray, simplex, False


@numba.njit(cache=True)
def region_a(simplex, a_index, a):
    return origin_to_point(simplex, a_index, a)


@numba.njit(cache=True)
def project_triangle_origin(triangle):
    # A is the last point we added.
    a_index = 2
    b_index = 1
    c_index = 0

    a = triangle[a_index][0]
    b = triangle[b_index][0]
    c = triangle[c_index][0]

    ab = b - a
    ac = c - a
    abc = np.cross(ab, ac)

    edge_ac2o = np.cross(abc, ac).dot(-a)

    if edge_ac2o >= 0:

        towards_c = ac.dot(-a)
        if towards_c >= 0:
            ray, simplex = origin_to_segment(triangle, a_index, c_index, a, c, ac, towards_c)
        else:
            ray, simplex = t_b(triangle, a_index, b_index, a, b, ab)
    else:

        edge_ab2o = np.cross(ab, abc).dot(-a)
        if edge_ab2o >= 0:
            ray, simplex = t_b(triangle, a_index, b_index, a, b, ab)
        else:
            return origin_to_triangle(triangle, a_index, b_index, c_index, a, b, c, abc, abc.dot(-a))

    return ray, simplex, False


@numba.njit(cache=True)
def t_b(triangle, a_index, b_index, a, b, ab):
    towards_b = ab.dot(-a)
    if towards_b < 0:
        return origin_to_point(triangle, a_index, a)
    else:
        return origin_to_segment(triangle, a_index, b_index, a, b, ab, towards_b)


@numba.njit(cache=True)
def origin_to_triangle(simplex, a_index, b_index, c_index, a, b, c, abc, abc_dot_a0):
    if abc_dot_a0 == 0:
        simplex = [simplex[c_index], simplex[b_index], simplex[a_index]]
        ray = np.zeros(3)
        return ray, simplex, True

    if abc_dot_a0 > 0:
        simplex = [simplex[c_index], simplex[b_index], simplex[a_index]]
    else:
        simplex = [simplex[b_index], simplex[c_index], simplex[a_index]]

    ray = -abc_dot_a0 / abc.dot(abc) * abc
    return ray, simplex, False


@numba.njit(cache=True)
def region_abc(simplex, a_index, b_index, c_index, a, b, c, a_cross_b):
    return origin_to_triangle(simplex, a_index, b_index, c_index, a, b, c, np.cross(b - a, c - a), -c.dot(a_cross_b))[:2]


@numba.njit(cache=True)
def region_acd(simplex, a_index, c_index, d_index, a, c, d, a_cross_c):
    return origin_to_triangle(simplex, a_index, c_index, d_index, a, c, d, np.cross(c - a, d - a), -d.dot(a_cross_c))[:2]


@numba.njit(cache=True)
def region_adb(simplex, a_index, d_index, b_index, a, d, b, a_cross_b):
    return origin_to_triangle(simplex, a_index, d_index, b_index, a, d, b, np.cross(d - a, b - a), d.dot(a_cross_b))[:2]


@numba.njit(cache=True)
def region_ab(simplex, a_index, b_index, a, b, ba_aa):
    return origin_to_segment(simplex, a_index, b_index, a, b, b - a, -ba_aa)


@numba.njit(cache=True)
def region_ac(simplex, a_index, c_index, a, c, ca_aa):
    return origin_to_segment(simplex, a_index, c_index, a, c, c - a, -ca_aa)


@numba.njit(cache=True)
def region_ad(simplex, a_index, d_index, a, d, da_aa):
    return origin_to_segment(simplex, a_index, d_index, a, d, d - a, -da_aa)


@numba.njit(cache=True)
def check_convergence(alpha, omega, ray_len, tolerance):
    alpha = max(alpha, omega)
    diff = ray_len - alpha

    check_passed = (diff - tolerance * ray_len) <= 0

    return check_passed


@numba.njit(cache=True)
def project_tetra_to_origin(tetra):
    a_index = 3
    b_index = 2
    c_index = 1
    d_index = 0

    a = tetra[a_index][0]
    b = tetra[b_index][0]
    c = tetra[c_index][0]
    d = tetra[d_index][0]

    aa = a.dot(a)

    da = d.dot(a)
    db = d.dot(b)
    dc = d.dot(c)
    dd = d.dot(d)
    da_aa = da - aa

    ca = c.dot(a)
    cb = c.dot(b)
    cc = c.dot(c)
    cd = dc
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
                        ray, simplex = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                    else:
                        ray, simplex = region_ab(tetra, a_index, b_index, a, b, ba_aa)
                else:
                    if ba * ba_ca + bb * ca_aa - bc * ba_aa <= 0:
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                            if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                                ray, simplex = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                            else:
                                ray, simplex = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                        else:
                            ray, simplex = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                    else:
                        ray, simplex = region_ab(tetra, a_index, b_index, a, b, ba_aa)
            else:
                if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                    ray, simplex = region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                else:
                    if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex = region_ad(tetra, a_index, d_index, a, d, da_aa)
                        else:
                            ray, simplex = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                    else:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex = region_ad(tetra, a_index, d_index, a, d, da_aa)
                        else:
                            ray, simplex = region_ac(tetra, a_index, c_index, a, c, ca_aa)
        else:
            if c.dot(a_cross_b) <= 0:
                if ba * ba_ca + bb * ca_aa - bc * ba_aa <= 0:
                    if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                        if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                            ray, simplex = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                        else:
                            ray, simplex = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                    else:
                        ray, simplex = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                else:
                    ray, simplex = region_ad(tetra, a_index, d_index, a, d, da_aa)
            else:
                if d.dot(a_cross_c) <= 0:
                    if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex = region_ad(tetra, a_index, d_index, a, d, da_aa)
                        else:
                            ray, simplex = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                    else:
                        if ca_aa <= 0:
                            ray, simplex = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                        else:
                            ray, simplex = region_ad(tetra, a_index, d_index, a, d, da_aa)
                else:
                    return np.zeros(3), tetra, True
    else:
        if ca_aa <= 0:
            if d.dot(a_cross_c) <= 0:
                if da_aa <= 0:
                    if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                                ray, simplex = region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                            else:
                                ray, simplex = region_ad(tetra, a_index, d_index, a, d, da_aa)
                        else:
                            ray, simplex = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                    else:
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                            ray, simplex = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                        else:
                            ray, simplex = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                else:
                    if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                        if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                            ray, simplex = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                        else:
                            ray, simplex = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                    else:
                        if c.dot(a_cross_b):
                            ray, simplex = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                        else:
                            ray, simplex = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
            else:
                if c.dot(a_cross_b) <= 0:
                    if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                        ray, simplex = region_ac(tetra, a_index, c_index, a, c, ca_aa)
                    else:
                        ray, simplex = region_abc(tetra, a_index, b_index, c_index, a, b, c, a_cross_b)
                else:
                    if -d.dot(a_cross_b) <= 0:
                        if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                            ray, simplex = region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                        else:
                            ray, simplex = region_ad(tetra, a_index, d_index, a, d, da_aa)
                    else:
                        return np.zeros(3), tetra, True
        else:
            if da_aa <= 0:
                if -d.dot(a_cross_b) <= 0:
                    if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                        if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                            ray, simplex = region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                        else:
                            ray, simplex = region_ad(tetra, a_index, d_index, a, d, da_aa)
                    else:
                        if d.dot(a_cross_c) <= 0:
                            ray, simplex = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                        else:
                            if c.dot(a_cross_b) <= 0:  # ???
                                ray, simplex = region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                            else:
                                ray, simplex = region_adb(tetra, a_index, d_index, b_index, a, d, b, a_cross_b)
                else:
                    if d.dot(a_cross_c) <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex = region_ad(tetra, a_index, d_index, a, d, da_aa)
                        else:
                            ray, simplex = region_acd(tetra, a_index, c_index, d_index, a, c, d, a_cross_c)
                    else:
                        return np.zeros(3), tetra, True
            else:
                ray, simplex = region_a(tetra, a_index, a)
    return ray, simplex, False


"""
def project_line_origin(line):
    a = line[0]
    b = line[1]

    d = b - a
    l = squared_norm(d)

    if l > 0:
        t = -a.dot(d)
        if t >= l:
            return [1, 0], squared_norm(b)

        elif t <= 0:
            return [0, 1], squared_norm(a)

        else:
            x = t / l
            y = 1 - x
            return [x, y], squared_norm((a + b) * y)

    return [0, 0], 0


def project_triangle_origin(triangle):
    a = triangle[0]
    b = triangle[1]
    c = triangle[2]

    next_index = [1, 2, 0]
    dl = [a - b, b - c, c - a]
    n = np.cross(dl[0], dl[1])
    l = squared_norm(n)

    parameterization = [0, 0, 0]
    sqr_distance = 0

    if l > 0:
        min_dist = -1
        for i in range(3):
            if triangle[i].dot(np.cross(dl[i], n)) > 0:
                j = next_index[i]
                line_parameterization, line_sqr_distance = project_line_origin([triangle[i], triangle[j]])

                if min_dist < 0 or line_sqr_distance < min_dist:
                    min_dist = line_sqr_distance

                    parameterization[i] = line_parameterization[0]
                    parameterization[j] = line_parameterization[1]
                    parameterization[next_index[j]] = 0

        if min_dist < 0:
            d = a.dot(n)
            s = np.square(l)
            o_to_project = n * (d / l)
            min_dist = squared_norm(o_to_project)

            parameterization[0] = norm_vector(np.cross(dl[1], b - o_to_project)) / s
            parameterization[1] = norm_vector(np.cross(dl[2], c - o_to_project)) / s
            parameterization[2] = 1 - parameterization[0] - parameterization[1]

        sqr_distance = min_dist

    return parameterization, sqr_distance


def project_tetra_origin(tetra):
    def triple(a, b, c):
        return a.dot(np.cross(b, c))

    a = tetra[0]
    b = tetra[1]
    c = tetra[2]
    d = tetra[3]

    next_index = [1, 2, 0]
    dl = [a - b, b - d, c - d]
    vl = triple(dl[0], dl[1], dl[2])
    ng = (vl * np.cross(a.dot(b - c), a - b)) <= 0

    parameterization = [0, 0, 0, 0]
    sqr_distance = 0

    if ng and abs(vl) > 0:
        min_dist = -1

        for i in range(3):
            j = next_index[i]
            s = vl * d.dot(np.cross(dl[i], dl[j]))
            if s > 0:
                triangle_parameterization, triangle_sqr_distance = project_triangle_origin([tetra[i], tetra[i], d])

                if min_dist < 0 or triangle_sqr_distance < min_dist:
                    min_dist = triangle_sqr_distance

                    parameterization[i] = triangle_parameterization[0]
                    parameterization[j] = triangle_parameterization[1]
                    parameterization[next_index[j]] = 0
                    parameterization[3] = triangle_parameterization[2]

        if min_dist < 0:
            min_dist = 0

            parameterization[0] = triple(c, b, d) / vl
            parameterization[1] = triple(a, c, d) / vl
            parameterization[2] = triple(b, a, d) / vl
            parameterization[4] = 1 - parameterization[0] + parameterization[1] + parameterization[2]

        sqr_distance = min_dist

    elif not ng:
        triangle_parameterization, triangle_sqr_distance = project_triangle_origin([a, b, c])
        parameterization = [triangle_parameterization[0], triangle_parameterization[1], triangle_parameterization[2], 0]
        sqr_distance = triangle_sqr_distance

    return parameterization, sqr_distance


def get_closest_points(simplex):
    if len(simplex) == 1:
        return simplex[0][1], simplex[0][2]

    if len(simplex) == 2:
        a = simplex[0][0]
        a0 = simplex[0][1]
        a1 = simplex[0][2]

        b = simplex[1][0]
        b0 = simplex[1][1]
        b1 = simplex[1][2]

        n = b - a
        la = n.dot(-a)
        if la <= 0:
            return a0, a1

        lb = squared_norm(n)
        if la > lb:
            return b0, b1

        lb = la / lb
        la = 1 - lb
        return la * a0 + lb * b0, la * a1 + lb * b1

    parameterization = None
    if len(simplex) == 3:
        parameterization, _ = project_triangle_origin(simplex)

    elif len(simplex) == 4:
        parameterization, _ = project_tetra_origin(simplex)

    p0 = 0
    p1 = 0
    for i in range(len(simplex)):
        p0 += parameterization[i] * simplex[i][1]
        p1 += parameterization[i] * simplex[i][2]

    return p0, p1
"""
