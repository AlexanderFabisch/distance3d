import numpy as np

from distance3d import colliders, gjk, random
from distance3d.colliders import MeshGraph


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
    distance : float
        Distance between shapes

    closest_point_1

    closest_point_2

    simplex

    """
    inside, dist, simplex = gjk_nesterov_accelerated(collider1, collider2, ray_guess)

    return dist, None, None, simplex


# ----- Math helper functions -----
def norm(v):
    return np.linalg.norm(v)


def squared_norm(a):
    return np.abs(np.square(a)).sum()


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


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
    omega = 0

    inside = False
    simplex = []
    distance = 0

    ray = np.array([1.0, 0.0, 0.0])  # x in paper
    if ray_guess is not None:
        ray = ray_guess

    ray_len = norm(ray)
    if ray_len < tolerance:
        ray = np.array([1.0, 0.0, 0.0])
        ray_len = 1

    ray_dir = ray  # d in paper
    support_point = ray  # s in paper

    # ------ Define Functions to structure Code ------
    def check_convergence():
        nonlocal alpha

        alpha = max(alpha, omega)
        diff = ray_len - alpha

        check_passed = (diff - tolerance * ray_len) <= 0

        return check_passed

    def origen_to_point(a):
        nonlocal ray, simplex
        ray = a
        simplex = [a]

    def origen_to_segment(a, b, ab, ab_dot_a0):
        nonlocal ray, simplex
        ray = (ab.dot(b) * a + ab_dot_a0 * b) / squared_norm(ab)
        simplex = [b, a]

    def origen_to_triangle(a, b, c, abc, abc_dot_a0):
        nonlocal ray, simplex

        if abc_dot_a0 == 0:
            simplex = [c, b, a]
            ray = [0, 0, 0]
            return True

        if abc_dot_a0 > 0:
            simplex = [c, b, a]
        else:
            simplex = [b, c, a]

        ray = -abc_dot_a0 / squared_norm(abc) * abc
        return False

    def project_line_origen(line):
        # A is the last point we added.
        a = line[1]
        b = line[0]

        ab = b - a
        d = np.dot(ab, -a)

        if d == 0:
            # Two extremely unlikely cases:
            #  - AB is orthogonal to A: should never happen because it means the support
            #    function did not do any progress and GJK should have stopped.
            #  - A == origin
            # In any case, A is the closest to the origin
            origen_to_point(a)
            return (a == np.array([0, 0, 0])).all()
        if d < 0:
            origen_to_point(a)
        else:
            origen_to_segment(a, b, ab, d)

        return False

    def project_triangle_origen(triangle):
        # A is the last point we added.
        a = triangle[2]
        b = triangle[1]
        c = triangle[0]

        ab = b - a
        ac = c - a
        abc = np.cross(ab, ac)

        edge_ac2o = np.cross(abc, ac).dot(-a)

        def t_b():
            towards_b = ab.dot(-a)
            if towards_b < 0:
                origen_to_point(a)
            else:
                origen_to_segment(a, b, ab, towards_b)

        if edge_ac2o >= 0:

            towards_c = ac.dot(-a)
            if towards_c >= 0:
                origen_to_segment(a, c, ac, towards_c)
            else:
                t_b()
        else:

            edge_ab2o = np.cross(ab, abc).dot(-a)
            if edge_ab2o >= 0:
                t_b()
            else:
                return origen_to_triangle(a, b, c, abc, abc.dot(-a))

        return False

    def project_tetra_to_origen(tetra):
        a = tetra[3]
        b = tetra[2]
        c = tetra[1]
        d = tetra[0]

        aa = squared_norm(a)

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

        def region_inside():
            nonlocal ray
            ray = [0, 0, 0]
            return True

        def region_abc():
            origen_to_triangle(a, b, c, np.cross(b - a, c - a), -c.dot(a_cross_b))

        def region_acd():
            origen_to_triangle(a, c, d, np.cross(c - a, d - a), -d.dot(a_cross_c))

        def region_adb():
            origen_to_triangle(a, d, b, np.cross(d - a, b - a), d.dot(a_cross_b))

        def region_ab():
            origen_to_segment(a, b, b - a, -ba_aa)

        def region_ac():
            origen_to_segment(a, c, c - a, -ca_aa)

        def region_ad():
            origen_to_segment(a, d, d - a, -da_aa)

        def region_a():
            origen_to_point(a)

        if ba_aa <= 0:
            if -d.dot(a_cross_b) <= 0:
                if ba * da_ba + bd * ba_aa - bb * da_aa <= 0:
                    if da_aa <= 0:
                        if ba * ba_ca + bb * ca_aa - bc * ba_aa <= 0:
                            region_abc()
                        else:
                            region_ab()
                    else:
                        if ba * ba_ca + bb * ca_aa - bc * ba_aa <= 0:
                            if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                                if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                                    region_acd()
                                else:
                                    region_ac()
                            else:
                                region_abc()
                        else:
                            region_ab()
                else:
                    if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                        region_adb()
                    else:
                        if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                            if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                                region_ad()
                            else:
                                region_acd()
                        else:
                            if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                                region_ad()
                            else:
                                region_ac()
            else:
                if c.dot(a_cross_b) <= 0:
                    if ba * ba_ca + bb * ca_aa - bc * ba_aa <= 0:
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                            if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                                region_acd()
                            else:
                                region_ac()
                        else:
                            region_abc()
                    else:
                        region_ad()
                else:
                    if d.dot(a_cross_c) <= 0:
                        if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                            if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                                region_ad()
                            else:
                                region_acd()
                        else:
                            if ca_aa <= 0:
                                region_ac()
                            else:
                                region_ad()
                    else:
                        region_inside()
        else:
            if ca_aa <= 0:
                if d.dot(a_cross_c) <= 0:
                    if da_aa <= 0:
                        if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                            if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                                if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                                    region_adb()
                                else:
                                    region_ad()
                            else:
                                region_acd()
                        else:
                            if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                                region_ac()
                            else:
                                region_abc()
                    else:
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                            if ca * ca_da + cc * da_aa - cd * ca_aa <= 0:
                                region_acd()
                            else:
                                region_ac()
                        else:
                            if c.dot(a_cross_b):
                                region_abc()
                            else:
                                region_acd()
                else:
                    if c.dot(a_cross_b) <= 0:
                        if ca * ba_ca + cb * ca_aa - cc * ba_aa <= 0:
                            region_ac()
                        else:
                            region_abc()
                    else:
                        if -d.dot(a_cross_b) <= 0:
                            if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                                region_adb()
                            else:
                                region_ad()
                        else:
                            region_inside()
            else:
                if da_aa <= 0:
                    if -d.dot(a_cross_b) <= 0:
                        if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                            if da * da_ba + dd * ba_aa - db * da_aa <= 0:
                                region_adb()
                            else:
                                region_ad()
                        else:
                            if d.dot(a_cross_c) <= 0:
                                region_acd()
                            else:
                                if c.dot(a_cross_b) <= 0:  # ???
                                    region_adb()
                                else:
                                    region_adb()
                    else:
                        if d.dot(a_cross_c) <= 0:
                            if da * ca_da + dc * da_aa - dd * ca_aa <= 0:
                                region_ad()
                            else:
                                region_acd()
                        else:
                            region_inside()
                else:
                    region_a()
        return False

    # ----- Actual Algorthm -----

    for k in range(max_interations):

        if ray_len < tolerance:
            distance = -inflation
            inside = True
            break

        if use_nesterov_acceleration:
            momentum = (k + 1) / (k + 3)
            y = momentum * ray + (1 - momentum) * support_point
            ray_dir = momentum * ray_dir + (1 - momentum) * y

            if normalize_support_direction:
                ray_dir = normalize(ray_dir)

        else:
            ray_dir = ray

        support_point = np.array(collider1.support_function(-ray_dir) - collider2.support_function(ray_dir))
        simplex.append(support_point)

        omega = ray_dir.dot(support_point) / norm(ray_dir)
        if omega > upper_bound:
            distance = omega - inflation
            inside = False
            break

        if use_nesterov_acceleration:
            frank_wolfe_duality_gap = 2 * ray.dot(ray - support_point)
            if frank_wolfe_duality_gap - tolerance <= 0:
                use_nesterov_acceleration = False
                simplex.pop()
                continue

        cv_check_passed = check_convergence()
        if k > 0 and cv_check_passed:
            if k > 0:
                simplex.pop()
            if use_nesterov_acceleration:
                use_nesterov_acceleration = False
                continue
            distance = ray_len - inflation

            if distance < tolerance:
                inside = True
            break

        if len(simplex) == 1:
            ray = support_point
        elif len(simplex) == 2:
            inside = project_line_origen(simplex)
        elif len(simplex) == 3:
            inside = project_triangle_origen(simplex)
        elif len(simplex) == 4:
            inside = project_tetra_to_origen(simplex)
        else:
            print("LOGIC ERROR: invalid simplex len")

        if not inside:
            ray_len = norm(ray)

        if inside or ray_len == 0:
            distance = -inflation
            inside = True
            break

    return inside, distance, simplex
