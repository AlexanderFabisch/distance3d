import numpy as np

from distance3d import colliders, gjk, geometry, random, distance
from distance3d.colliders import MeshGraph

simplex = []
ray = None
dir = None
omega = None
tolerance = None
rl = None
alpha = None


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def squared_norm(a):
    return np.abs(np.square(a)).sum()


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

    global simplex, ray, dir, omega, tolerance, rl, alpha

    max_interations = 100
    upper_bound = 1000000000
    tolerance = 1e-6
    use_nesterov_acceleration = True
    normalize_support_direction = type(s1) == MeshGraph and type(s2) == MeshGraph
    simplex = []
    inside = False
    alpha = 0

    ray = np.array([1., 0., 0.])  # x in paper
    rl = np.linalg.norm(ray)
    if rl < tolerance:
        ray = np.array([-1, 0, 0])
        rl = 1

    dir = ray  # d in paper
    w = ray  # s in paper

    for k in range(max_interations):

        if rl < tolerance:
            return True

        if use_nesterov_acceleration:
            momentum = (k + 1) / (k + 3)
            y = momentum * ray + (1 - momentum) * w
            dir = momentum * dir + (1 - momentum) * y

            if normalize_support_direction:
                dir /= np.linalg.norm(dir)

        else:
            dir = ray

        w = np.array(s1.support_function(-dir) - s2.support_function(dir))
        """
        for t in simplex:
            if (t == w).all():
                print("w in simplex")
                """

        simplex.append(w)

        omega = dir.dot(w) / np.linalg.norm(dir)
        if omega > upper_bound:
            return False

        if use_nesterov_acceleration:
            frank_wolfe_duality_gap = 2 * ray.dot(ray - w)
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
            distance = rl

            if distance < tolerance:
                return True
            break

        if len(simplex) == 1:
            ray = w
        elif len(simplex) == 2:
            inside = project_line_origen(simplex)
        elif len(simplex) == 3:
            inside = project_triangle_origen(simplex)
        elif len(simplex) == 4:
            inside = project_tetra_to_origen(simplex)
        else:
            print("simplex to big")

        if not inside:
            rl = np.linalg.norm(ray)

        if inside or rl == 0:
            return True

    return False


def check_convergence():
    global omega, tolerance, rl, alpha

    alpha = max(alpha, omega)
    diff = rl - alpha

    check_passed = (diff - tolerance * rl) <= 0

    return check_passed


def origen_to_point(a):
    global ray, simplex
    ray = a
    simplex = [a]


def origen_to_segment(a, b, ab, ab_dot_a0):
    global ray, simplex
    ray = (ab.dot(b) * a + ab_dot_a0 * b) / squared_norm(ab)
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


def run():
    random_state = np.random.RandomState(84)
    shape_names = list(colliders.COLLIDERS.keys())

    not_the_same_counter = 0
    k = 1000000
    skip = 0

    for i in range(k):
        if i % 1000 == 0:
            print(i)


        shape1 = shape_names[random_state.randint(len(shape_names))]
        args1 = random.RANDOM_GENERATORS[shape1](random_state)
        shape2 = shape_names[random_state.randint(len(shape_names))]
        args2 = random.RANDOM_GENERATORS[shape2](random_state)
        collider1 = colliders.COLLIDERS[shape1](*args1)
        collider2 = colliders.COLLIDERS[shape2](*args2)

        intersection_jolt = gjk.gjk_intersection_jolt(collider1, collider2)
        intersection_libccd = gjk.gjk_intersection_libccd(collider1, collider2)

        if i == 254:
            print("Debug")

        intersection_nesterov = gjk_nesterov_accelerated(collider1, collider2)
        assert intersection_jolt == intersection_libccd

        same = intersection_nesterov == intersection_libccd

        if not same:
            if skip != 0:
                skip -= 1
                continue

            not_the_same_counter += 1
            print(f"Not the Same: {i}")


            import pytransform3d.visualizer as pv
            fig = pv.figure()
            fig.view_init()
            collider1.make_artist()
            collider2.make_artist()
            collider1.artist_.add_artist(fig)
            collider2.artist_.add_artist(fig)
            fig.show()
            break


    print(not_the_same_counter / k)


run()
