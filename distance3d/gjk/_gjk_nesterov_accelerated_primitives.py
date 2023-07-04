import numpy as np
import numba

from ..colliders import Capsule, Sphere, Box, Cylinder, Ellipsoid


def gjk_nesterov_accelerated_primitives_intersection(collider0, collider1):
    """Nesterov-accelerated GJK.

    This implementation is based on the Paper
    "Collision Detection Accelerated: An Optimization Perspective"
    https://lmontaut.github.io/nesterov-gjk.github.io/
    and highly inspired by the C++ implementation in of the Authors:
    https://github.com/humanoid-path-planner/hpp-fcl/blob/devel/src/narrowphase/gjk.cpp

    This implemetation can only be used with the following Collider Types:
    "Sphere", "Capsule", "Box", "Ellipsoid" and "Cylinder".

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
    """
    return gjk_nesterov_accelerated_primitives(collider0, collider1)[0]


def gjk_nesterov_accelerated_primitives_distance(collider0, collider1):
    """Nesterov-accelerated GJK.

    This implementation is based on the Paper
    "Collision Detection Accelerated: An Optimization Perspective"
    https://lmontaut.github.io/nesterov-gjk.github.io/
    and highly inspired by the C++ implementation in of the Authors:
    https://github.com/humanoid-path-planner/hpp-fcl/blob/devel/src/narrowphase/gjk.cpp

    This implemetation can only be used with the following Collider Types:
    "Sphere", "Capsule", "Box", "Ellipsoid" and "Cylinder".

    Parameters
    ----------
    collider0 : ConvexCollider
        Convex collider 1.

    collider1 : ConvexCollider
        Convex collider 2.

    Returns
    -------
    distance : float
        The distance between the colliders. The distance is zero if they collide.
    """
    return max(gjk_nesterov_accelerated_primitives(collider0, collider1)[1], 0.0)


def gjk_nesterov_accelerated_primitives_iterations(collider1, collider2):
    """
    Nesterov accelerated gjk approach. This implementation is based on the Paper
    "Collision Detection Accelerated: An Optimization Perspective"
    https://lmontaut.github.io/nesterov-gjk.github.io/
    and highly inspired by the C++ implementation in of the Authors:
    https://github.com/humanoid-path-planner/hpp-fcl/blob/devel/src/narrowphase/gjk.cpp

    This implemetation can only be used with the following Collider Types:
    "Sphere", "Capsule", "Box", "Ellipsoid" and "Cylinder".

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
    return gjk_nesterov_accelerated_primitives(collider1, collider2)[3]


def get_minkowski_diff(collider0, collider1):
    # Tuple data
    # - 0 type0 (int)
    # - 1 data0 (vec3)
    # - 2 type1 (int)
    # - 3 data1 (vec3
    # - 4 oR1  (mat4)
    # - 5 ot1  (vec3)

    data0, type0 = get_data_from_collider(collider0)
    data1, type1 = get_data_from_collider(collider1)

    collider02origin = collider0.collider2origin()
    collider12origin = collider1.collider2origin()
    oR1 = np.dot(collider02origin[:3, :3].T, collider12origin[:3, :3])
    ot1 = np.dot(collider02origin[:3, :3].T, collider1.center() - collider02origin[:3, 3])

    minkowski_diff = (type0, data0, type1, data1, oR1, ot1)
    return minkowski_diff


def get_data_from_collider(collider):
    if type(collider) == Sphere:
        return np.array([0.0, 0.0, 0.0]), 0

    if type(collider) == Capsule:
        h = collider.height / 2
        return np.array([h, 0.0, 0.0]), 1

    if type(collider) == Box:
        s = collider.size / 2
        return s, 2

    if type(collider) == Ellipsoid:
        a2 = collider.radii[0] * collider.radii[0]
        b2 = collider.radii[1] * collider.radii[1]
        c2 = collider.radii[2] * collider.radii[2]
        return np.array([a2, b2, c2]), 3

    assert type(collider) == Cylinder
    h = collider.length / 2
    r = collider.radius
    return np.array([h, r, 0.0]), 4


def gjk_nesterov_accelerated_primitives(
        collider0, collider1, max_interations=128, upper_bound=1.79769e+308,
        tolerance=1e-6, use_nesterov_acceleration=False):
    minkowski_diff = get_minkowski_diff(collider0, collider1)

    # Inflation is only used with spheres and capsules
    inflation = 0.0
    if type(collider0) == Sphere or type(collider0) == Capsule:
        inflation += collider0.radius

    if type(collider1) == Sphere or type(collider1) == Capsule:
        inflation += collider1.radius

    return run_gjk_nesterov_accelerated(minkowski_diff, inflation, max_interations, upper_bound, tolerance,
                                        use_nesterov_acceleration)


@numba.njit(cache=True)
def run_gjk_nesterov_accelerated(
        minkowski_diff, inflation, max_interations, upper_bound, tolerance,
        use_nesterov_acceleration):
    # ------ Initialize Variables ------

    upper_bound += inflation

    alpha = 0.0

    inside = False
    simplex = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    simplex_len = 0
    distance = 0.0

    ray = np.array([1.0, 0.0, 0.0])  # x in paper
    ray_len = 1

    ray_dir = ray  # d in paper
    support_point = ray.copy()  # s in paper

    i = 0
    while i < max_interations:

        if ray_len < tolerance:
            distance = -inflation
            inside = True
            break

        if use_nesterov_acceleration:
            momentum = (i + 1) / (i + 3)
            y = momentum * ray + (1.0 - momentum) * support_point
            ray_dir = momentum * ray_dir + (1.0 - momentum) * y
        else:
            ray_dir = ray

        s0, s1 = support_function(-ray_dir, minkowski_diff)

        simplex[simplex_len] = s0 - s1
        support_point = simplex[simplex_len]
        simplex_len += 1

        omega = ray_dir.dot(support_point) / np.linalg.norm(ray_dir)
        if omega > upper_bound:
            distance = omega - inflation
            inside = False
            break

        if use_nesterov_acceleration:
            frank_wolfe_duality_gap = 2 * ray.dot(ray - support_point)
            if frank_wolfe_duality_gap - tolerance <= 0:
                use_nesterov_acceleration = False
                simplex_len -= 1
                continue

        # Check convergence
        alpha = max(alpha, omega)
        diff = ray_len - alpha
        cv_check_passed = (diff - tolerance * ray_len) <= 0

        if i > 0 and cv_check_passed:
            simplex_len -= 1
            if use_nesterov_acceleration:
                use_nesterov_acceleration = False
                continue
            distance = ray_len - inflation

            inside = distance < tolerance
            break

        assert 1 <= simplex_len <= 4
        if simplex_len == 1:
            ray = np.copy(support_point)
        elif simplex_len == 2:
            ray, simplex_len, inside = project_line_origin(simplex)
        elif simplex_len == 3:
            ray, simplex_len, inside = project_triangle_origin(simplex)
        else:
            ray, simplex_len, inside = project_tetra_to_origin(simplex)

        if not inside:
            ray_len = np.linalg.norm(ray)
        if inside or ray_len == 0:
            distance = -inflation - 1.
            inside = True
            break

        i += 1

    return inside, distance, simplex, i


@numba.njit(cache=True)
def origin_to_point(simplex, a):
    simplex[0] = np.copy(a)
    return np.copy(a), 1


@numba.njit(cache=True)
def origin_to_segment(simplex, a, b, ab, ab_dot_a0):
    ray = (ab.dot(b) * a + ab_dot_a0 * b) / ab.dot(ab)
    simplex[0], simplex[1] = np.copy(b), np.copy(a)
    return ray, 2


@numba.njit(cache=True)
def origin_to_triangle(simplex, a, b, c, abc, abc_dot_a0):
    if abc_dot_a0 == 0:
        simplex[0], simplex[1], simplex[2] = np.copy(c), np.copy(b), np.copy(a)
        return np.zeros(3), 3, True

    if abc_dot_a0 > 0:
        simplex[0], simplex[1], simplex[2] = np.copy(c), np.copy(b), np.copy(a)
    else:
        simplex[0], simplex[1], simplex[2] = np.copy(b), np.copy(c), np.copy(a)

    ray = -abc_dot_a0 / abc.dot(abc) * abc
    return ray, 3, False


@numba.njit(cache=True)
def project_line_origin(line):
    # A is the last point we added.
    a_index = 1
    b_index = 0

    a = line[a_index]
    b = line[b_index]

    ab = b - a
    d = np.dot(ab, -a)

    if d == 0:
        # Two extremely unlikely cases:
        #  - AB is orthogonal to A: should never happen because it means the support
        #    function did not do any progress and GJK should have stopped.
        #  - A == origin
        # In any case, A is the closest to the origin
        ray, simplex_len = origin_to_point(line, a)
        return ray, simplex_len, np.all(a == 0.0)
    if d < 0:
        ray, simplex_len = origin_to_point(line, a)
    else:
        ray, simplex_len = origin_to_segment(line, a, b, ab, d)

    return ray, simplex_len, False


@numba.njit(cache=True)
def t_b(triangle, a, b, ab):
    towards_b = ab.dot(-a)
    if towards_b < 0:
        return origin_to_point(triangle, a)
    else:
        return origin_to_segment(triangle, a, b, ab, towards_b)


@numba.njit(cache=True)
def project_triangle_origin(triangle):
    # A is the last point we added.
    a_index = 2
    b_index = 1
    c_index = 0

    a = triangle[a_index]
    b = triangle[b_index]
    c = triangle[c_index]

    ab = b - a
    ac = c - a
    abc = np.cross(ab, ac)

    edge_ac2o = np.cross(abc, ac).dot(-a)

    if edge_ac2o >= 0:

        towards_c = ac.dot(-a)
        if towards_c >= 0:
            ray, simplex_len = origin_to_segment(triangle, a, c, ac, towards_c)
        else:
            ray, simplex_len = t_b(triangle, a, b, ab)
    else:

        edge_ab2o = np.cross(ab, abc).dot(-a)
        if edge_ab2o >= 0:
            ray, simplex_len = t_b(triangle, a, b, ab)
        else:
            return origin_to_triangle(triangle, a, b, c, abc, abc.dot(-a))

    return ray, simplex_len, False


@numba.njit(cache=True)
def region_a(simplex, a):
    return origin_to_point(simplex, a)


@numba.njit(cache=True)
def region_ab(simplex, a, b, ba_aa):
    return origin_to_segment(simplex, a, b, b - a, -ba_aa)


@numba.njit(cache=True)
def region_ac(simplex, a, c, ca_aa):
    return origin_to_segment(simplex, a, c, c - a, -ca_aa)


@numba.njit(cache=True)
def region_ad(simplex, a, d, da_aa):
    return origin_to_segment(simplex, a, d, d - a, -da_aa)


@numba.njit(cache=True)
def region_abc(simplex, a, b, c, a_cross_b):
    return origin_to_triangle(simplex, a, b, c, np.cross(b - a, c - a), -c.dot(a_cross_b))[:2]


@numba.njit(cache=True)
def region_acd(simplex, a, c, d, a_cross_c):
    return origin_to_triangle(simplex, a, c, d, np.cross(c - a, d - a), -d.dot(a_cross_c))[:2]


@numba.njit(cache=True)
def region_adb(simplex, a, d, b, a_cross_b):
    return origin_to_triangle(simplex, a, d, b, np.cross(d - a, b - a), d.dot(a_cross_b))[:2]


@numba.njit(cache=True)
def project_tetra_to_origin(tetra):
    a_index = 3
    b_index = 2
    c_index = 1
    d_index = 0

    a = tetra[a_index]
    b = tetra[b_index]
    c = tetra[c_index]
    d = tetra[d_index]

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
                        ray, simplex_len = region_abc(tetra, a, b, c, a_cross_b)
                    else:
                        ray, simplex_len = region_ab(tetra, a, b, ba_aa)
                else:
                    if ba * ba_ca + bb * ca_aa - bc * ba_aa <= 0:
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                            if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                                ray, simplex_len = region_acd(tetra, a, c, d, a_cross_c)
                            else:
                                ray, simplex_len = region_ac(tetra, a, c, ca_aa)
                        else:
                            ray, simplex_len = region_abc(tetra, a, b, c, a_cross_b)
                    else:
                        ray, simplex_len = region_ab(tetra, a, b, ba_aa)
            else:
                if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                    ray, simplex_len = region_adb(tetra, a, d, b, a_cross_b)
                else:
                    if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex_len = region_ad(tetra, a, d, da_aa)
                        else:
                            ray, simplex_len = region_acd(tetra, a, c, d, a_cross_c)
                    else:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex_len = region_ad(tetra, a, d, da_aa)
                        else:
                            ray, simplex_len = region_ac(tetra, a, c, ca_aa)
        else:
            if c.dot(a_cross_b) <= 0:
                if ba * ba_ca + bb * ca_aa - bc * ba_aa <= 0:
                    if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                        if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                            ray, simplex_len = region_acd(tetra, a, c, d, a_cross_c)
                        else:
                            ray, simplex_len = region_ac(tetra, a, c, ca_aa)
                    else:
                        ray, simplex_len = region_abc(tetra, a, b, c, a_cross_b)
                else:
                    ray, simplex_len = region_ad(tetra, a, d, da_aa)
            else:
                if d.dot(a_cross_c) <= 0:
                    if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex_len = region_ad(tetra, a, d, da_aa)
                        else:
                            ray, simplex_len = region_acd(tetra, a, c, d, a_cross_c)
                    else:
                        if ca_aa <= 0:
                            ray, simplex_len = region_ac(tetra, a, c, ca_aa)
                        else:
                            ray, simplex_len = region_ad(tetra, a, d, da_aa)
                else:
                    return np.zeros(3), 4, True
    else:
        if ca_aa <= 0:
            if d.dot(a_cross_c) <= 0:
                if da_aa <= 0:
                    if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                                ray, simplex_len = region_adb(tetra, a, d, b, a_cross_b)
                            else:
                                ray, simplex_len = region_ad(tetra, a, d, da_aa)
                        else:
                            ray, simplex_len = region_acd(tetra, a, c, d, a_cross_c)
                    else:
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                            ray, simplex_len = region_ac(tetra, a, c, ca_aa)
                        else:
                            ray, simplex_len = region_abc(tetra, a, b, c, a_cross_b)
                else:
                    if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                        if ca * ca_da + cc * da_aa - dc * ca_aa <= 0:
                            ray, simplex_len = region_acd(tetra, a, c, d, a_cross_c)
                        else:
                            ray, simplex_len = region_ac(tetra, a, c, ca_aa)
                    else:
                        if c.dot(a_cross_b):
                            ray, simplex_len = region_abc(tetra, a, b, c, a_cross_b)
                        else:
                            ray, simplex_len = region_acd(tetra, a, c, d, a_cross_c)
            else:
                if c.dot(a_cross_b) <= 0:
                    if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                        ray, simplex_len = region_ac(tetra, a, c, ca_aa)
                    else:
                        ray, simplex_len = region_abc(tetra, a, b, c, a_cross_b)
                else:
                    if -d.dot(a_cross_b) <= 0:
                        if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                            ray, simplex_len = region_adb(tetra, a, d, b, a_cross_b)
                        else:
                            ray, simplex_len = region_ad(tetra, a, d, da_aa)
                    else:
                        return np.zeros(3), 4, True
        else:
            if da_aa <= 0:
                if -d.dot(a_cross_b) <= 0:
                    if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                        if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                            ray, simplex_len = region_adb(tetra, a, d, b, a_cross_b)
                        else:
                            ray, simplex_len = region_ad(tetra, a, d, da_aa)
                    else:
                        if d.dot(a_cross_c) <= 0:
                            ray, simplex_len = region_acd(tetra, a, c, d, a_cross_c)
                        else:
                            ray, simplex_len = region_adb(tetra, a, d, b, a_cross_b)
                else:
                    if d.dot(a_cross_c) <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            ray, simplex_len = region_ad(tetra, a, d, da_aa)
                        else:
                            ray, simplex_len = region_acd(tetra, a, c, d, a_cross_c)
                    else:
                        return np.zeros(3), 4, True
            else:
                ray, simplex_len = region_a(tetra, a)
    return ray, simplex_len, False


@numba.njit(cache=True)
def support_function(dir, minkowski_diff):
    oR1 = minkowski_diff[4]
    ot1 = minkowski_diff[5]

    support0 = select_support(dir, minkowski_diff[0], minkowski_diff[1])

    support1 = select_support(np.dot(-oR1.T, dir), minkowski_diff[2], minkowski_diff[3])
    support1 = np.dot(oR1, support1) + ot1

    return support0, support1


@numba.njit(cache=True)
def select_support(dir, type, data):
    if type == 0:
        return sphere_support()

    if type == 1:
        return capsule_support(dir, data)

    if type == 2:
        return box_support(dir, data)

    if type == 3:
        return ellipsoid_support(dir, data)

    assert type == 4
    return cylinder_support(dir, data)


@numba.njit(cache=True)
def sphere_support():
    return np.array([0.0, 0.0, 0.0])


@numba.njit(cache=True)
def capsule_support(dir, data):
    support = np.array([0.0, 0.0, 0.0])
    if dir[2] > 0:
        support[2] = data[0]
    else:
        support[2] = -data[0]

    return support


@numba.njit(cache=True)
def box_support(dir, data):
    inflate = 1.0
    if (dir == 0).any():
        inflate = 1.00000001

    support = np.array([0.0, 0.0, 0.0])
    for i in range(0, 3):
        if dir[i] > 0:
            support[i] = inflate * data[i]
        else:
            support[i] = -inflate * data[i]

    return support


@numba.njit(cache=True)
def ellipsoid_support(dir, data):
    v = data * dir
    d = np.sqrt(v.dot(dir))

    return v / d


@numba.njit(cache=True)
def cylinder_support(dir, data):
    support = np.array([0.0, 0.0, 0.0])

    inflate = 1.00001

    h = data[0]
    r = data[1]

    if (dir[:2] == np.array([0.0, 0.0])).all():
        h *= inflate

    if dir[2] > 0:
        support[2] = h
    elif dir[2] < 0:
        support[2] = -h
    else:
        support[2] = 0
        r *= inflate

    if (dir[:2] == np.array([0.0, 0.0])).all():
        support[0] = 0.0
        support[1] = 0.0
    else:
        support[:2] = dir[:2] / np.linalg.norm(dir[:2]) * r

    return support
