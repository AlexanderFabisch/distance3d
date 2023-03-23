import numpy as np
from distance3d import colliders, gjk, geometry, random, distance
from pytest import approx
from numpy.testing import assert_array_almost_equal

from distance3d.colliders import ConvexHullVertices

simplex = []
ray = None
dir = None
w = None


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def gjk_nesterov_accelerated(s1, s2):
    """
    Parameters
    ----------
    s1 : ConvexCollider
        Convex collider 1.

    s2 : ConvexCollider
        Convex collider 2.

    Returns
    -------
    contact : bool
    """

    global simplex, ray, dir, w

    max_interations = 100
    upper_bound = 1000
    tolerance = 0.1
    use_nesterov_acceleration = True
    simplex = []
    inside = False

    ray = np.array(s1.center() - s2.center())  # x in paper
    rl = np.linalg.norm(ray)
    if rl < tolerance:
        ray = np.array([-1, 0, 0])
        rl = 1

    dir = ray   # d in paper
    old_dir = ray
    w = ray     # s in paper

    for k in range(max_interations):

        if rl < tolerance:
            return True

        old_dir = dir
        if use_nesterov_acceleration:
            momentum = (k + 1) / (k + 3)
            y = momentum * ray + (1 - momentum) * w
            dir = momentum * dir + (1 - momentum) * y
        else:
            dir = ray

        print("dir", dir)
        w = np.array(s1.support_function(-dir) - s2.support_function(dir))
        print("w: ", w)
        simplex.append(w)

        frank_wolfe_duality_gap = 2 * ray.dot(ray - w)
        if frank_wolfe_duality_gap - tolerance <= 0:
            if (old_dir == ray).all():
                return True
            else:
                simplex.pop()
                use_nesterov_acceleration = False

        if len(simplex) == 1:
            ray = w
        elif len(simplex) == 2:
            inside = project_line_origen(simplex)
        elif len(simplex) == 3:
            inside = project_triangle_origen(simplex)
        elif len(simplex) == 4:
            inside = project_tetra_to_origen(simplex)

        if not inside:
            rl = np.linalg.norm(ray)

        if inside or rl == 0:
            return True

    return False


def origen_to_point(a):
    global ray, simplex
    ray = a
    simplex = [a]


def origen_to_segment(a, b, ab, ab_dot_a0):
    global ray, simplex
    ray = (ab.dot(b) * a + ab_dot_a0 * b) / np.linalg.norm(ab)
    simplex = [b, a]


def origen_to_triangle(a, b, c, abc, abc_dot_a0):
    global ray, simplex

    if abc_dot_a0 == 0:
        simplex = [c, b, a]
        ray = [0, 0, 0]
        return True

    if abc_dot_a0 > 0:
        simplex = [c, b, a]
    else:
        simplex = [b, c, a]

    ray = -abc_dot_a0 / np.linalg.norm(abc) * abc
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

    aa = np.linalg.norm(a)

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
        global ray
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
                    if da * ca_da + dc * da_aa - dd * ca_aa:
                        if da * da_ba + dd * ba_aa - db * da_aa:
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


def run_gjk_nesterov_accelerated_boxes():
    box2origin = np.eye(4)
    size = np.ones(3)
    box_collider = colliders.Box(box2origin, size)

    # complete overlap
    contact = gjk_nesterov_accelerated(
        box_collider, box_collider)
    assert contact

    # touching faces, edges, or points
    for dim1 in range(3):
        for dim2 in range(3):
            for dim3 in range(3):
                for sign1 in [-1, 1]:
                    for sign2 in [-1, 1]:
                        for sign3 in [-1, 1]:
                            box2origin2 = np.eye(4)
                            box2origin2[dim1, 3] = sign1
                            box2origin2[dim2, 3] = sign2
                            box2origin2[dim3, 3] = sign3
                            size2 = np.ones(3)
                            box_collider2 = colliders.Box(box2origin2, size2)

                            contact = gjk_nesterov_accelerated(
                                box_collider, box_collider2)
                            assert contact

    box2origin = np.array([
        [-0.29265666, -0.76990535, 0.56709596, 0.1867558],
        [0.93923897, -0.12018753, 0.32153556, -0.09772779],
        [-0.17939408, 0.62673815, 0.75829879, 0.09500884],
        [0., 0., 0., 1.]])
    size = np.array([2.89098828, 1.15032456, 2.37517511])
    box_collider = colliders.Box(box2origin, size)

    box2origin2 = np.array([
        [-0.29265666, -0.76990535, 0.56709596, 3.73511598],
        [0.93923897, -0.12018753, 0.32153556, -1.95455576],
        [-0.17939408, 0.62673815, 0.75829879, 1.90017684],
        [0., 0., 0., 1.]])
    size2 = np.array([0.96366276, 0.38344152, 0.79172504])
    box_collider2 = colliders.Box(box2origin2, size2)

    contact = gjk_nesterov_accelerated(box_collider, box_collider2)
    assert not contact

run_gjk_nesterov_accelerated_boxes()
