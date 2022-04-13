import numpy as np
import pytransform3d.transformations as pt
from distance3d.distance import (
    point_to_line, point_to_line_segment, point_to_plane, point_to_triangle,
    point_to_box, point_to_circle, point_to_disk, point_to_cylinder,
    point_to_ellipsoid, line_to_line, line_to_box,
    line_segment_to_line_segment, line_segment_to_triangle,
    line_segment_to_box, triangle_to_triangle, rectangle_to_rectangle,
    rectangle_to_box)
from distance3d.geometry import convert_box_to_face
from pytest import approx
from numpy.testing import assert_array_almost_equal


def test_point_to_line():
    line_point = np.array([0, 0, 0])
    line_direction = np.array([1, 0, 0])

    distance, contact_point_line = point_to_line(
        line_point, line_point, line_direction)
    assert distance == 0.0
    assert_array_almost_equal(contact_point_line, line_point)

    point = np.array([1, 1, 0])
    distance, contact_point_line = point_to_line(
        point, line_point, line_direction)
    assert distance == 1.0
    assert_array_almost_equal(contact_point_line, np.array([1, 0, 0]))

    point = np.array([1, -1, 0])
    distance, contact_point_line = point_to_line(
        point, line_point, line_direction)
    assert distance == 1.0
    assert_array_almost_equal(contact_point_line, np.array([1, 0, 0]))


def test_point_to_line_segment():
    segment_start = np.array([0, 0, 0])
    segment_end = np.array([1, 0, 0])

    distance, contact_point_line = point_to_line_segment(
        0.5 * (segment_start + segment_end), segment_start, segment_end)
    assert distance == 0.0
    assert_array_almost_equal(
        contact_point_line, 0.5 * (segment_start + segment_end))

    distance, contact_point_line = point_to_line_segment(
        segment_start, segment_start, segment_end)
    assert distance == 0.0
    assert_array_almost_equal(contact_point_line, segment_start)

    distance, contact_point_line = point_to_line_segment(
        segment_end, segment_start, segment_end)
    assert distance == 0.0
    assert_array_almost_equal(contact_point_line, segment_end)

    distance, contact_point_line = point_to_line_segment(
        np.array([-1, 0, 0]), segment_start, segment_end)
    assert distance == 1.0
    assert_array_almost_equal(contact_point_line, segment_start)

    distance, contact_point_line = point_to_line_segment(
        np.array([2, 0, 0]), segment_start, segment_end)
    assert distance == 1.0
    assert_array_almost_equal(contact_point_line, segment_end)


def test_point_to_plane():
    point = np.array([0, 0, 0])
    plane_point = np.array([0, 0, 0])
    plane_normal = np.array([0, 0, 1])
    dist, closest_point_on_plane = point_to_plane(
        point, plane_point, plane_normal)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point_on_plane, np.array([0, 0, 0]))

    point = np.array([0, 0, 1])
    plane_point = np.array([0, 0, 0])
    plane_normal = np.array([0, 0, 1])
    dist, closest_point_on_plane = point_to_plane(
        point, plane_point, plane_normal)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point_on_plane, np.array([0, 0, 0]))

    point = np.array([0, 0, -1])
    plane_point = np.array([0, 0, 0])
    plane_normal = np.array([0, 0, 1])
    dist, closest_point_on_plane = point_to_plane(
        point, plane_point, plane_normal)
    assert approx(dist) == 1
    assert_array_almost_equal(closest_point_on_plane, np.array([0, 0, 0]))


def test_point_to_triangle():
    for i in range(3):
        all_dims = [0, 1, 2]
        all_dims.remove(i)

        # example in x-y plane:
        # 0-----1
        #   \   |
        #     \ |
        #       2
        triangle_points = np.zeros((3, 3))
        triangle_points[0, all_dims[1]] = 1
        triangle_points[1, all_dims[0]] = 1
        triangle_points[1, all_dims[1]] = 1
        triangle_points[2, all_dims[0]] = 1

        point = np.array([0, 0, 0], dtype=float)
        dist, closest_point = point_to_triangle(point, triangle_points)
        expected = np.zeros(3)
        expected[all_dims[0]] = 0.5
        expected[all_dims[1]] = 0.5
        assert_array_almost_equal(closest_point, expected)
        assert_array_almost_equal(dist, np.sqrt(0.5))

        point[all_dims[0]] = 1.5
        point[all_dims[1]] = 0.5
        dist, closest_point = point_to_triangle(point, triangle_points)
        expected[all_dims[0]] = 1.0
        expected[all_dims[1]] = 0.5
        assert_array_almost_equal(closest_point, expected)
        assert_array_almost_equal(dist, 0.5)

        point[all_dims[0]] = 0.5
        point[all_dims[1]] = 1.5
        dist, closest_point = point_to_triangle(point, triangle_points)
        expected[all_dims[0]] = 0.5
        expected[all_dims[1]] = 1.0
        assert_array_almost_equal(closest_point, expected)
        assert_array_almost_equal(dist, 0.5)

        point[all_dims[0]] = -0.25
        point[all_dims[1]] = 1.5
        dist, closest_point = point_to_triangle(point, triangle_points)
        assert_array_almost_equal(closest_point, triangle_points[0])
        assert_array_almost_equal(dist, 0.559017)

        point[all_dims[0]] = 1.25
        point[all_dims[1]] = 1.5
        dist, closest_point = point_to_triangle(point, triangle_points)
        assert_array_almost_equal(closest_point, triangle_points[1])
        assert_array_almost_equal(dist, 0.559017)

        point[all_dims[0]] = 1.25
        point[all_dims[1]] = -0.5
        dist, closest_point = point_to_triangle(point, triangle_points)
        assert_array_almost_equal(closest_point, triangle_points[2])
        assert_array_almost_equal(dist, 0.559017)

        point[all_dims[0]] = 0.5
        point[all_dims[1]] = 0.5
        dist, closest_point = point_to_triangle(point, triangle_points)
        assert_array_almost_equal(closest_point, point)
        assert_array_almost_equal(dist, 0.0)

        point[all_dims[0]] = 0.75
        point[all_dims[1]] = 0.75
        dist, closest_point = point_to_triangle(point, triangle_points)
        assert_array_almost_equal(closest_point, point)
        assert_array_almost_equal(dist, 0.0)

        expected_closest_point_triangle = np.copy(point)
        point[i] = 0.5
        dist, closest_point = point_to_triangle(point, triangle_points)
        assert_array_almost_equal(
            closest_point, expected_closest_point_triangle)
        assert_array_almost_equal(dist, 0.5)

        point[i] = -0.5
        dist, closest_point = point_to_triangle(point, triangle_points)
        assert_array_almost_equal(
            closest_point, expected_closest_point_triangle)
        assert_array_almost_equal(dist, 0.5)


def test_point_to_box():
    point = np.array([0, 0, 0])

    box2origin = np.eye(4)
    size = np.array([1, 1, 1])

    dist, closest_point_box = point_to_box(point, box2origin, size)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point_box, np.array([0, 0, 0]))

    point = np.array([0.5, 0.5, 0.5])
    dist, closest_point_box = point_to_box(point, box2origin, size)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point_box, point)

    point = np.array([1, 1, 1])
    dist, closest_point_box = point_to_box(point, box2origin, size)
    assert approx(dist) == np.sqrt(0.75)
    assert_array_almost_equal(closest_point_box, np.array([0.5, 0.5, 0.5]))


def test_point_to_circle():
    point = np.array([0, 0, 0])
    center = np.array([0, 0, 0])
    normal = np.array([0, 0, 1])
    radius = 1.0
    dist, closest_point_circle = point_to_circle(point, center, radius, normal)
    assert approx(dist) == 1.0
    assert approx(np.linalg.norm(closest_point_circle - center)) == 1.0

    dist, closest_point_circle2 = point_to_circle(
        closest_point_circle, center, radius, normal)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point_circle2, closest_point_circle)

    point = np.array([0, 0, -1])
    dist, closest_point_circle = point_to_circle(
        point, center, radius, normal)
    assert approx(dist) == np.sqrt(2)
    assert approx(np.linalg.norm(closest_point_circle - center)) == 1.0


def test_point_to_disk():
    point = np.array([0, 0, 0])
    center = np.array([0, 0, 0])
    normal = np.array([0, 0, 1])
    radius = 1.0
    dist, closest_point_circle = point_to_disk(point, center, radius, normal)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point_circle, point)

    point = np.array([0, 0, 1])
    dist, closest_point_circle = point_to_disk(point, center, radius, normal)
    assert approx(dist, abs=1e-7) == 1.0
    assert_array_almost_equal(closest_point_circle, np.array([0, 0, 0]))

    point = np.array([0, 1, 1])
    dist, closest_point_circle = point_to_disk(point, center, radius, normal)
    assert approx(dist, abs=1e-7) == 1.0
    assert_array_almost_equal(closest_point_circle, np.array([0, 1, 0]))

    point = np.array([0, 2, -1])
    dist, closest_point_circle = point_to_disk(point, center, radius, normal)
    assert approx(dist, abs=1e-7) == np.sqrt(2)
    assert_array_almost_equal(closest_point_circle, np.array([0, 1, 0]))


def test_point_to_cylinder():
    point = np.array([0, 0, 0])
    cylinder2origin = np.eye(4)
    radius = 1.0
    length = 1.0
    dist, closest_point_cylinder = point_to_cylinder(
        point, cylinder2origin, radius, length)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point_cylinder, np.array([0, 0, 0]))

    point = np.array([0, 0.5, 0])
    cylinder2origin = np.eye(4)
    radius = 1.0
    length = 1.0
    dist, closest_point_cylinder = point_to_cylinder(
        point, cylinder2origin, radius, length)
    assert approx(dist, abs=1e-7) == 0.0
    assert_array_almost_equal(closest_point_cylinder, np.array([0, 0.5, 0]))

    point = np.array([0.5, 0, 0])
    cylinder2origin = np.eye(4)
    radius = 1.0
    length = 1.0
    dist, closest_point_cylinder = point_to_cylinder(
        point, cylinder2origin, radius, length)
    assert approx(dist, abs=1e-7) == 0.0
    assert_array_almost_equal(closest_point_cylinder, np.array([0.5, 0, 0]))

    point = np.array([0.5, 0, -1])
    cylinder2origin = np.eye(4)
    radius = 1.0
    length = 1.0
    dist, closest_point_cylinder = point_to_cylinder(
        point, cylinder2origin, radius, length)
    assert approx(dist, abs=1e-7) == 0.5
    assert_array_almost_equal(closest_point_cylinder, np.array([0.5, 0, -0.5]))


def test_point_to_ellipsoid():
    random_state = np.random.RandomState(323)
    ellipsoid2origin = pt.random_transform(random_state)
    radii = random_state.rand(3)

    dist, closest_point_ellipsoid = point_to_ellipsoid(
        ellipsoid2origin[:3, 3], ellipsoid2origin, radii)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point_ellipsoid, ellipsoid2origin[:3, 3])

    point = ellipsoid2origin[:3, 3] + radii[0] * ellipsoid2origin[:3, 0]
    dist, closest_point_ellipsoid = point_to_ellipsoid(
        point, ellipsoid2origin, radii)
    assert approx(dist, abs=1e-7) == 0.0
    assert_array_almost_equal(closest_point_ellipsoid, point)

    for i in range(3):
        point = ellipsoid2origin[:3, 3] + 2 * radii[i] * ellipsoid2origin[:3, i]
        dist, closest_point_ellipsoid = point_to_ellipsoid(
            point, ellipsoid2origin, radii)
        assert approx(dist, abs=1e-7) == radii[i]
        assert_array_almost_equal(
            closest_point_ellipsoid,
            ellipsoid2origin[:3, 3] + radii[i] * ellipsoid2origin[:3, i])


def test_line_to_line():
    line_point1 = np.array([1.76405235, 0.40015721, 0.97873798])
    line_direction1 = np.array([0.72840603, 0.6070528, -0.31766579])

    line_point2 = np.array([0.95008842, -0.15135721, -0.10321885])
    line_direction2 = np.array([0.27049077, 0.09489186, 0.95803459])
    dist, contact_point1, contact_point2 = line_to_line(
        line_point1, line_direction1, line_point2, line_direction2)
    assert approx(dist, abs=1e-6) == 0.037258912990233976
    assert_array_almost_equal(
        contact_point1, [1.29010346, 0.00516871, 1.18543225])
    assert_array_almost_equal(
        contact_point2, [1.31292374, -0.02406961, 1.1818852])

    line_point2 = np.array([0.48431215, 0.57914048, -0.18158257])
    line_direction2 = np.array([0.94975868, -0.25220293, 0.18534331])
    dist, contact_point1, contact_point2 = line_to_line(
        line_point1, line_direction1, line_point2, line_direction2)
    assert approx(dist) == 0.8691087137089082
    assert_array_almost_equal(
        contact_point1, [2.00358931, 0.59978706, 0.87427331])
    assert_array_almost_equal(
        contact_point2, [2.03568155, 0.1671833, 0.12116374])

    line_point2 = np.array([-2.55298982, 0.6536186, 0.8644362])
    line_direction2 = np.array([0.72840603, 0.6070528, -0.31766579])
    dist, contact_point1, contact_point2 = line_to_line(
        line_point1, line_direction1, line_point2, line_direction2)
    assert approx(dist) == 3.1600265753342143
    assert_array_almost_equal(
        contact_point1, [-0.38793973, -1.39331068, 1.91724512])
    assert_array_almost_equal(
        contact_point2, [-2.55298982, 0.6536186, 0.8644362])

    line_point2 = np.array([1.46274045, 1.53502913, 0.56644004])
    line_direction2 = np.array([0.72840603, 0.6070528, -0.31766579])
    dist, contact_point1, contact_point2 = line_to_line(
        line_point1, line_direction1, line_point2, line_direction2)
    assert approx(dist) == 1.0900482924723882
    assert_array_almost_equal(
        contact_point1, [2.20140388, 0.76464551, 0.78800423])
    assert_array_almost_equal(
        contact_point2, [1.46274045, 1.53502913, 0.56644004])

    line_point2 = np.array([-0.34791215, 0.15634897, 1.23029068])
    line_direction2 = np.array([0.72840603, 0.6070528, -0.31766579])
    dist, contact_point1, contact_point2 = line_to_line(
        line_point1, line_direction1, line_point2, line_direction2)
    assert approx(dist) == 1.2096957224924787
    assert_array_almost_equal(
        contact_point1, [0.47748201, -0.67206913, 1.53982529])
    assert_array_almost_equal(
        contact_point2, [-0.34791215, 0.15634897, 1.23029068])


def test_line_to_box():
    box2origin = np.eye(4)
    size = np.array([1, 1, 1])

    for i in range(3):
        for sign in [-1, 1]:
            # parallel to box edge on i-axis without contact
            line_point = np.array([1, 1, 1])
            line_point[i] = 0
            line_direction = np.array([0, 0, 0])
            line_direction[i] = sign
            dist, closest_point_line, closest_point_box = line_to_box(
                line_point, line_direction, box2origin, size)
            assert approx(dist) == np.sqrt(0.5)
            assert approx(np.linalg.norm(closest_point_box - closest_point_line)) == np.sqrt(0.5)
            assert -0.5 <= closest_point_box[0] <= 0.5
            assert -0.5 <= closest_point_box[1] <= 0.5
            assert -0.5 <= closest_point_box[2] <= 0.5

    # line in 3D without contact
    for i in range(3):
        for sign in [-1, 1]:
            for sign_i in [-1, 1]:
                for sign_j in [-1, 1]:
                    for sign_k in [-1, 1]:
                        line_point = np.array([0, 0, 0])
                        line_point[i] = sign * 2
                        line_direction = np.array([sign_i, sign_j, sign_k])
                        dist, closest_point_line, closest_point_box = line_to_box(
                            line_point, line_direction, box2origin, size)
                        assert approx(dist) == 0.8164965809277259
                        assert approx(np.linalg.norm(closest_point_box - closest_point_line)) == 0.8164965809277259
                        assert -0.5 <= closest_point_box[0] <= 0.5
                        assert -0.5 <= closest_point_box[1] <= 0.5
                        assert -0.5 <= closest_point_box[2] <= 0.5

    for i in range(3):
        # parallel to box face on i-axis without contact
        line_point = np.array([0, 0, 0])
        j = (i - 1) % 3
        line_point[j] = 1
        line_direction = np.array([0, 0, 0])
        line_direction[i] = 1
        dist, closest_point_line, closest_point_box = line_to_box(
            line_point, line_direction, box2origin, size)
        assert approx(dist) == 0.5
        assert approx(np.linalg.norm(closest_point_box - closest_point_line)) == 0.5
        assert -0.5 <= closest_point_box[0] <= 0.5
        assert -0.5 <= closest_point_box[1] <= 0.5
        assert -0.5 <= closest_point_box[2] <= 0.5

    for i in range(3):
        # parallel to box face on i-axis with contact
        line_point = np.array([0, 0, 0])
        j = (i - 1) % 3
        line_point[j] = 0.5
        line_direction = np.array([0, 0, 0])
        line_direction[i] = 1
        dist, closest_point_line, closest_point_box = line_to_box(
            line_point, line_direction, box2origin, size)
        assert approx(dist) == 0
        assert approx(np.linalg.norm(closest_point_box - closest_point_line)) == 0
        assert -0.5 <= closest_point_box[0] <= 0.5
        assert -0.5 <= closest_point_box[1] <= 0.5
        assert -0.5 <= closest_point_box[2] <= 0.5

    for i in range(3):
        # passes through box face on i-axis
        line_point = np.array([0, 0, 0])
        line_point[i] = 1
        line_direction = np.array([0, 0, 0])
        line_direction[i] = 1
        dist, closest_point_line, closest_point_box = line_to_box(
            line_point, line_direction, box2origin, size)
        assert approx(dist) == 0.0
        expected = np.array([0, 0, 0])
        expected[i] = 0.5
        assert_array_almost_equal(closest_point_line, closest_point_box)
        assert -0.5 <= closest_point_line[0] <= 0.5
        assert -0.5 <= closest_point_line[1] <= 0.5
        assert -0.5 <= closest_point_line[2] <= 0.5

    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            # not parallel to any edge in i-j plane without contact
            line_point = np.array([0, 0, 0])
            line_point[i] = 1
            line_point[j] = 1
            line_direction = np.array([0, 0, 0])
            line_direction[i] = -1
            line_direction[j] = 1
            dist, closest_point_line, closest_point_box = line_to_box(
                line_point, line_direction, box2origin, size)
            assert approx(dist) == np.sqrt(0.5)
            expected = np.array([0, 0, 0])
            expected[i] = 1
            expected[j] = 1
            assert_array_almost_equal(closest_point_line, expected)
            assert_array_almost_equal(closest_point_box, 0.5 * expected)

    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            for sign in [-1, 1]:
                # parallel to box face in i-j-plane without contact
                line_point = np.array([0, 0, 0])
                k = [0, 1, 2]
                k.remove(i)
                k.remove(j)
                k = k[0]
                line_point[k] = sign
                line_direction = np.array([0, 0, 0])
                line_direction[i] = sign
                line_direction[j] = sign
                dist, closest_point_line, closest_point_box = line_to_box(
                    line_point, line_direction, box2origin, size)
                assert approx(dist) == 0.5
                # multiple solutions, this is a regression test
                expected = sign * np.array([0.5, 0.5, 0.5])
                assert_array_almost_equal(closest_point_box, expected)
                expected[k] = sign
                assert_array_almost_equal(closest_point_line, expected)


def test_line_segment_to_line_segment():
    segment_start1 = np.array([0, 0, 0])
    segment_end1 = np.array([1, 0, 0])
    dist, closest_point1, closest_point2 = line_segment_to_line_segment(
        segment_start1, segment_end1, segment_start1, segment_end1)
    assert approx(dist) == 0

    segment_start2 = np.array([0, 1, 0])
    segment_end2 = np.array([1, 1, 0])
    dist, closest_point1, closest_point2 = line_segment_to_line_segment(
        segment_start1, segment_end1, segment_start2, segment_end2)
    assert approx(dist) == 1
    assert approx(np.linalg.norm(closest_point2 - closest_point1)) == 1

    segment_start2 = np.array([1, 0, 0])
    segment_end2 = np.array([2, 0, 0])
    dist, closest_point1, closest_point2 = line_segment_to_line_segment(
        segment_start1, segment_end1, segment_start2, segment_end2)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point1, np.array([1, 0, 0]))
    assert_array_almost_equal(closest_point1, closest_point2)

    segment_start2 = np.array([0.5, -0.5, 0])
    segment_end2 = np.array([0.5, 0.5, 0])
    dist, closest_point1, closest_point2 = line_segment_to_line_segment(
        segment_start1, segment_end1, segment_start2, segment_end2)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point1, np.array([0.5, 0, 0]))
    assert_array_almost_equal(closest_point1, closest_point2)

    segment_start2 = np.array([-0.5, -0.5, 0])
    segment_end2 = np.array([-0.5, 0.5, 0])
    dist, closest_point1, closest_point2 = line_segment_to_line_segment(
        segment_start1, segment_end1, segment_start2, segment_end2)
    assert approx(dist) == 0.5
    assert_array_almost_equal(closest_point1, np.array([0, 0, 0]))
    assert_array_almost_equal(closest_point2, np.array([-0.5, 0, 0]))

    segment_start2 = np.array([0.5, 0, 0])
    segment_end2 = np.array([0.5, 0, 0])
    dist, closest_point1, closest_point2 = line_segment_to_line_segment(
        segment_start1, segment_end1, segment_start2, segment_end2)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point1, np.array([0.5, 0, 0]))
    assert_array_almost_equal(closest_point1, closest_point2)
    dist, closest_point1, closest_point2 = line_segment_to_line_segment(
        segment_start2, segment_end2, segment_start1, segment_end1)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point1, np.array([0.5, 0, 0]))
    assert_array_almost_equal(closest_point1, closest_point2)


def test_line_segment_to_triangle():
    triangle_points = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=float)
    segment_start = np.array([1, 0, 0], dtype=float)
    segment_end = np.array([1, 1, 0], dtype=float)
    dist, contact_point_segment, contact_point_triangle = \
        line_segment_to_triangle(segment_start, segment_end, triangle_points)
    assert_array_almost_equal(
        contact_point_segment, np.array([1, 1, 0]))
    assert_array_almost_equal(
        contact_point_triangle, np.array([0, 1, 0]))
    assert dist == 1.0


def test_line_segment_to_box():
    box2origin = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ])
    size = np.array([1, 1, 1])

    segment_start = np.array([2, 0, 0], dtype=float)
    segment_end = np.array([0, 2, 0])
    dist, closest_point_segment, closest_point_box = line_segment_to_box(
        segment_start, segment_end, box2origin, size)
    assert approx(dist) == np.sqrt(0.5)
    assert_array_almost_equal(closest_point_segment, np.array([1, 1, 0]))
    assert_array_almost_equal(closest_point_box, np.array([0.5, 0.5, 0]))

    segment_end = np.array([3, 0, 0])
    dist, closest_point_segment, closest_point_box = line_segment_to_box(
        segment_start, segment_end, box2origin, size)
    assert approx(dist) == 1.5
    assert_array_almost_equal(closest_point_segment, np.array([2, 0, 0]))
    assert_array_almost_equal(closest_point_box, np.array([0.5, 0, 0]))

    segment_start = np.array([-2, 0, 0], dtype=float)
    dist, closest_point_segment, closest_point_box = line_segment_to_box(
        segment_start, segment_end, box2origin, size)
    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point_segment, np.array([0.5, 0, 0]))
    assert_array_almost_equal(closest_point_box, closest_point_segment)

    segment_end = np.array([-1, 0, 0])
    dist, closest_point_segment, closest_point_box = line_segment_to_box(
        segment_start, segment_end, box2origin, size)
    assert approx(dist) == 0.5
    assert_array_almost_equal(closest_point_segment, np.array([-1, 0, 0]))
    assert_array_almost_equal(closest_point_box, np.array([-0.5, 0, 0]))


def test_triangel_to_triangle():
    triangle_points = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=float)
    dist, closest_point1, closest_point2 = triangle_to_triangle(
        triangle_points, triangle_points)
    assert approx(dist) == 0
    assert_array_almost_equal(closest_point1, closest_point2)

    triangle_points2 = np.copy(triangle_points)
    triangle_points2[:, 0] += 1.5
    dist, closest_point1, closest_point2 = triangle_to_triangle(
        triangle_points, triangle_points2)
    assert approx(dist) == 0.5
    assert_array_almost_equal(closest_point1, np.array([1, 1, 0]))
    assert_array_almost_equal(closest_point2, np.array([1.5, 1, 0]))


def test_rectangle_to_rectangle():
    rectangle_center1 = np.array([0, 0, 0])
    rectangle_axes1 = np.array([[1, 0, 0], [0, 1, 0]])
    rectangle_lengths1 = np.array([1, 1])
    rectangle_center2 = np.array([0, 0, 1])
    rectangle_axes2 = np.array([[1, 0, 0], [0, 1, 0]])
    rectangle_lengths2 = np.array([1, 1])

    dist, closest_point_rectangle1, closest_point_rectangle2 = rectangle_to_rectangle(
        rectangle_center1, rectangle_axes1, rectangle_lengths1,
        rectangle_center2, rectangle_axes2, rectangle_lengths2
    )
    assert approx(dist) == 1
    # choose one of many solutions:
    assert_array_almost_equal(
        closest_point_rectangle1, np.array([-0.5, -0.5, 0]))
    assert_array_almost_equal(
        closest_point_rectangle2, np.array([-0.5, -0.5, 1]))


def test_rectangle_to_box():
    box2origin = np.eye(4)
    size = np.array([1, 1, 1])

    for i in range(3):
        for sign in [-1, 1]:
            face_center, face_axes, face_lengths = convert_box_to_face(
                box2origin, size, i, sign)
            dist, closest_point_rectangle, closest_point_box = rectangle_to_box(
                face_center, face_axes, face_lengths, box2origin, size)
            assert approx(dist) == 0
            assert_array_almost_equal(
                closest_point_rectangle, closest_point_box)

            face_center = box2origin[:3, 3] + sign * size[i] * box2origin[:3, i]
            dist, closest_point_rectangle, closest_point_box = rectangle_to_box(
                face_center, face_axes, face_lengths, box2origin, size)
            assert approx(dist) == 0.5
            assert approx(np.linalg.norm(closest_point_box - closest_point_rectangle)) == 0.5
