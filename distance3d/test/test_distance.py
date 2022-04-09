import numpy as np
from distance3d.distance import (
    point_to_line, point_to_line_segment, line_to_line, line_to_box)
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
    line_point = np.array([0, 1, 0])
    line_direction = np.array([1, 0, 0])
    box2origin = np.eye(4)
    size = np.array([1, 1, 1])

    dist, closest_point_line, closest_point_box = line_to_box(
        line_point, line_direction, box2origin, size)

    assert approx(dist) == 0.5
    # multiple solutions, this is a regression test
    assert_array_almost_equal(closest_point_line, np.array([0.5, 1, 0]))
    assert_array_almost_equal(closest_point_box, np.array([0.5, 0.5, 0]))

    line_point = np.array([1, 0, 0])
    line_direction = np.array([1, 0, 0])
    box2origin = np.eye(4)
    size = np.array([1, 1, 1])

    dist, closest_point_line, closest_point_box = line_to_box(
        line_point, line_direction, box2origin, size)

    assert approx(dist) == 0.0
    assert_array_almost_equal(closest_point_line, np.array([0.5, 0, 0]))
    assert_array_almost_equal(closest_point_box, np.array([0.5, 0, 0]))

    line_point = np.array([1, 0, 0])
    line_direction = np.array([0, 1, 0])
    box2origin = np.eye(4)
    size = np.array([1, 1, 1])

    dist, closest_point_line, closest_point_box = line_to_box(
        line_point, line_direction, box2origin, size)

    assert approx(dist) == 0.5
    # multiple solutions, this is a regression test
    assert_array_almost_equal(closest_point_line, np.array([1, 0.5, 0]))
    assert_array_almost_equal(closest_point_box, np.array([0.5, 0.5, 0]))

    line_point = np.array([1, 1, 0])
    line_direction = np.array([-1, 1, 0])
    box2origin = np.eye(4)
    size = np.array([1, 1, 1])

    dist, closest_point_line, closest_point_box = line_to_box(
        line_point, line_direction, box2origin, size)

    assert approx(dist) == np.sqrt(0.5)
    assert_array_almost_equal(closest_point_line, np.array([1, 1, 0]))
    assert_array_almost_equal(closest_point_box, np.array([0.5, 0.5, 0]))
