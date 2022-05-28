"""Distance between line and box.

Implementation adapted from 3D Game Engine Design by David H. Eberly.

Geometric Tools, Inc.
http://www.geometrictools.com
Copyright (c) 1998-2006.  All Rights Reserved

The Wild Magic Version 4 Foundation Library source code is supplied
under the terms of the license agreement
(http://www.geometrictools.com/License/Wm4FoundationLicense.pdf)
and may not be copied or disclosed except in accordance with the terms
of that agreement.
"""
import math

import numpy as np


def _line_to_box(line_point, line_direction, box2origin, size):
    box_half_size = 0.5 * size

    # compute coordinates of line in box coordinate system
    origin2box = np.linalg.inv(box2origin)
    point_in_box = origin2box[:3, 3] + origin2box[:3, :3].dot(line_point)
    direction_in_box = origin2box[:3, :3].dot(line_direction)

    # Apply reflections so that direction vector has nonnegative components.
    direction_sign = np.ones(3)
    for i in range(3):
        if direction_in_box[i] < 0.0:
            point_in_box[i] = -point_in_box[i]
            direction_in_box[i] = -direction_in_box[i]
            direction_sign[i] = -1.0

    if direction_in_box[0] > 0.0:
        if direction_in_box[1] > 0.0:
            if direction_in_box[2] > 0.0:  # (+,+,+)
                sqr_dist, line_parameter = _case_no_zeros(
                    point_in_box, direction_in_box, box_half_size)
            else:  # (+,+,0)
                sqr_dist, line_parameter = _case_0(
                    0, 1, 2, point_in_box, direction_in_box, box_half_size)
        else:
            if direction_in_box[2] > 0.0:  # (+,0,+)
                sqr_dist, line_parameter = _case_0(
                    0, 2, 1, point_in_box, direction_in_box, box_half_size)
            else:  # (+,0,0)
                sqr_dist, line_parameter = _case_00(
                    0, 1, 2, point_in_box, direction_in_box, box_half_size)
    else:
        if direction_in_box[1] > 0.0:
            if direction_in_box[2] > 0.0:  # (0,+,+)
                sqr_dist, line_parameter = _case_0(
                    1, 2, 0, point_in_box, direction_in_box, box_half_size)
            else:  # (0,+,0)
                sqr_dist, line_parameter = _case_00(
                    1, 0, 2, point_in_box, direction_in_box, box_half_size)
        else:
            if direction_in_box[2] > 0.0:  # (0,0,+)
                sqr_dist, line_parameter = _case_00(
                    2, 0, 1, point_in_box, direction_in_box, box_half_size)
            else:  # (0,0,0)
                sqr_dist = _case_000(point_in_box, box_half_size)
                line_parameter = 0.0

    # compute closest point on line
    closest_point_line = line_point + line_parameter * line_direction

    # compute closest point on box
    # undo the reflections applied previously
    closest_point_box = box2origin[:3, 3] + box2origin[:3, :3].dot(
        direction_sign * point_in_box)

    return (math.sqrt(sqr_dist), closest_point_line, closest_point_box,
            line_parameter)


def _case_no_zeros(point_in_box, direction_in_box, box_half_size):
    point_m_edge = point_in_box - box_half_size

    prod_dx_py = direction_in_box[0] * point_m_edge[1]
    prod_dy_px = direction_in_box[1] * point_m_edge[0]

    if prod_dy_px >= prod_dx_py:
        prod_dz_px = direction_in_box[2] * point_m_edge[0]
        prod_dx_pz = direction_in_box[0] * point_m_edge[2]
        if prod_dz_px >= prod_dx_pz:
            # line intersects x = e0
            sqr_dist, line_parameter = _box_face(
                0, 1, 2, point_in_box, direction_in_box, point_m_edge,
                box_half_size)
        else:
            # line intersects z = e2
            sqr_dist, line_parameter = _box_face(
                2, 0, 1, point_in_box, direction_in_box, point_m_edge,
                box_half_size)
    else:
        prod_dz_py = direction_in_box[2] * point_m_edge[1]
        prod_dy_pz = direction_in_box[1] * point_m_edge[2]
        if prod_dz_py >= prod_dy_pz:
            # line intersects y = e1
            sqr_dist, line_parameter = _box_face(
                1, 2, 0, point_in_box, direction_in_box, point_m_edge,
                box_half_size)
        else:
            # line intersects z = e2
            sqr_dist, line_parameter = _box_face(
                2, 0, 1, point_in_box, direction_in_box, point_m_edge,
                box_half_size)
    return sqr_dist, line_parameter


def _box_face(i0, i1, i2, point_in_box, direction_in_box, point_m_edge, box_half_size):
    sqr_dist = 0.0
    point_p_edge = np.zeros(3)

    point_p_edge[i1] = point_in_box[i1] + box_half_size[i1]
    point_p_edge[i2] = point_in_box[i2] + box_half_size[i2]
    if direction_in_box[i0] * point_p_edge[i1] >= direction_in_box[i1] * point_m_edge[i0]:
        if direction_in_box[i0] * point_p_edge[i2] >= direction_in_box[i2] * point_m_edge[i0]:
            # v[i1] >= -e[i1], v[i2] >= -e[i2] (distance = 0)
            point_in_box[i0] = box_half_size[i0]
            point_in_box[i1] -= direction_in_box[i1] * point_m_edge[i0] / direction_in_box[i0]
            point_in_box[i2] -= direction_in_box[i2] * point_m_edge[i0] / direction_in_box[i0]
            line_parameter = -point_m_edge[i0] / direction_in_box[i0]
        else:
            # v[i1] >= -e[i1], v[i2] < -e[i2]
            l_sqr = (direction_in_box[i0] * direction_in_box[i0]
                     + direction_in_box[i2] * direction_in_box[i2])
            tmp = (l_sqr * point_p_edge[i1]
                   - direction_in_box[i1] * (direction_in_box[i0] * point_m_edge[i0]
                                             + direction_in_box[i2] * point_p_edge[i2]))
            if tmp <= 2.0 * l_sqr * box_half_size[i1]:
                t = tmp / l_sqr
                l_sqr += direction_in_box[i1] * direction_in_box[i1]
                tmp = point_p_edge[i1] - t
                delta = (direction_in_box[i0] * point_m_edge[i0]
                         + direction_in_box[i1] * tmp
                         + direction_in_box[i2] * point_p_edge[i2])
                line_parameter = -delta / l_sqr
                sqr_dist += (point_m_edge[i0] * point_m_edge[i0]
                             + tmp * tmp
                             + point_p_edge[i2] * point_p_edge[i2]
                             + delta * line_parameter)

                point_in_box[i0] = box_half_size[i0]
                point_in_box[i1] = t - box_half_size[i1]
                point_in_box[i2] = -box_half_size[i2]
            else:
                l_sqr += direction_in_box[i1] * direction_in_box[i1]
                delta = (direction_in_box[i0] * point_m_edge[i0]
                         + direction_in_box[i1] * point_m_edge[i1]
                         + direction_in_box[i2] * point_p_edge[i2])
                line_parameter = -delta / l_sqr
                sqr_dist += (point_m_edge[i0] * point_m_edge[i0]
                             + point_m_edge[i1] * point_m_edge[i1]
                             + point_p_edge[i2] * point_p_edge[i2]
                             + delta * line_parameter)

                point_in_box[i0] = box_half_size[i0]
                point_in_box[i1] = box_half_size[i1]
                point_in_box[i2] = -box_half_size[i2]
    else:
        if direction_in_box[i0] * point_p_edge[i2] >= direction_in_box[i2] * point_m_edge[i0]:
            # v[i1] < -e[i1], v[i2] >= -e[i2]
            l_sqr = (direction_in_box[i0] * direction_in_box[i0]
                     + direction_in_box[i1] * direction_in_box[i1])
            tmp = (l_sqr * point_p_edge[i2]
                   - direction_in_box[i2] * (direction_in_box[i0] * point_m_edge[i0]
                                             + direction_in_box[i1] * point_p_edge[i1]))
            if tmp <= 2.0 * l_sqr * box_half_size[i2]:
                t = tmp / l_sqr
                l_sqr += direction_in_box[i2] * direction_in_box[i2]
                tmp = point_p_edge[i2] - t
                delta = (direction_in_box[i0] * point_m_edge[i0]
                         + direction_in_box[i1] * point_p_edge[i1]
                         + direction_in_box[i2] * tmp)
                line_parameter = -delta / l_sqr
                sqr_dist += (point_m_edge[i0] * point_m_edge[i0]
                             + point_p_edge[i1] * point_p_edge[i1]
                             + tmp * tmp + delta * line_parameter)

                point_in_box[i0] = box_half_size[i0]
                point_in_box[i1] = -box_half_size[i1]
                point_in_box[i2] = t - box_half_size[i2]
            else:
                l_sqr += direction_in_box[i2] * direction_in_box[i2]
                delta = (direction_in_box[i0] * point_m_edge[i0]
                         + direction_in_box[i1] * point_p_edge[i1]
                         + direction_in_box[i2] * point_m_edge[i2])
                line_parameter = -delta / l_sqr
                sqr_dist += (point_m_edge[i0] * point_m_edge[i0]
                             + point_p_edge[i1] * point_p_edge[i1]
                             + point_m_edge[i2] * point_m_edge[i2]
                             + delta * line_parameter)

                point_in_box[i0] = box_half_size[i0]
                point_in_box[i1] = -box_half_size[i1]
                point_in_box[i2] = box_half_size[i2]
        else:
            # v[i1] < -e[i1], v[i2] < -e[i2]
            l_sqr = (direction_in_box[i0] * direction_in_box[i0]
                     + direction_in_box[i2] * direction_in_box[i2])
            tmp = (l_sqr * point_p_edge[i1]
                   - direction_in_box[i1] * (direction_in_box[i0] * point_m_edge[i0]
                                             + direction_in_box[i2] * point_p_edge[i2]))
            if tmp >= 0.0:
                # v[i1]-edge is closest
                if tmp <= 2.0 * l_sqr * box_half_size[i1]:
                    t = tmp / l_sqr
                    l_sqr += direction_in_box[i1] * direction_in_box[i1]
                    tmp = point_p_edge[i1] - t
                    delta = (direction_in_box[i0] * point_m_edge[i0]
                             + direction_in_box[i1] * tmp
                             + direction_in_box[i2] * point_p_edge[i2])
                    line_parameter = -delta / l_sqr
                    sqr_dist += (point_m_edge[i0] * point_m_edge[i0]
                                 + tmp * tmp
                                 + point_p_edge[i2] * point_p_edge[i2]
                                 + delta * line_parameter)

                    point_in_box[i0] = box_half_size[i0]
                    point_in_box[i1] = t - box_half_size[i1]
                    point_in_box[i2] = -box_half_size[i2]
                else:
                    l_sqr += direction_in_box[i1] * direction_in_box[i1]
                    delta = (direction_in_box[i0] * point_m_edge[i0]
                             + direction_in_box[i1] * point_m_edge[i1]
                             + direction_in_box[i2] * point_p_edge[i2])
                    line_parameter = -delta / l_sqr
                    sqr_dist += (point_m_edge[i0] * point_m_edge[i0]
                                 + point_m_edge[i1] * point_m_edge[i1]
                                 + point_p_edge[i2] * point_p_edge[i2]
                                 + delta * line_parameter)

                    point_in_box[i0] = box_half_size[i0]
                    point_in_box[i1] = box_half_size[i1]
                    point_in_box[i2] = -box_half_size[i2]
            else:
                l_sqr = (direction_in_box[i0] * direction_in_box[i0]
                         + direction_in_box[i1] * direction_in_box[i1])
                tmp = (l_sqr * point_p_edge[i2]
                       - direction_in_box[i2] * (direction_in_box[i0] * point_m_edge[i0]
                                                 + direction_in_box[i1] * point_p_edge[i1]))
                if tmp >= 0.0:
                    # v[i2]-edge is closest
                    if tmp <= 2.0 * l_sqr * box_half_size[i2]:
                        t = tmp / l_sqr
                        l_sqr += direction_in_box[i2] * direction_in_box[i2]
                        tmp = point_p_edge[i2] - t
                        delta = (direction_in_box[i0] * point_m_edge[i0]
                                 + direction_in_box[i1] * point_p_edge[i1]
                                 + direction_in_box[i2] * tmp)
                        line_parameter = -delta / l_sqr
                        sqr_dist += (point_m_edge[i0] * point_m_edge[i0]
                                     + point_p_edge[i1] * point_p_edge[i1]
                                     + tmp * tmp + delta * line_parameter)

                        point_in_box[i0] = box_half_size[i0]
                        point_in_box[i1] = -box_half_size[i1]
                        point_in_box[i2] = t - box_half_size[i2]
                    else:
                        l_sqr += direction_in_box[i2] * direction_in_box[i2]
                        delta = (direction_in_box[i0] * point_m_edge[i0]
                                 + direction_in_box[i1] * point_p_edge[i1]
                                 + direction_in_box[i2] * point_m_edge[i2])
                        line_parameter = -delta / l_sqr
                        sqr_dist += (point_m_edge[i0] * point_m_edge[i0]
                                     + point_p_edge[i1] * point_p_edge[i1]
                                     + point_m_edge[i2] * point_m_edge[i2]
                                     + delta * line_parameter)

                        point_in_box[i0] = box_half_size[i0]
                        point_in_box[i1] = -box_half_size[i1]
                        point_in_box[i2] = box_half_size[i2]
                else:
                    # (v[i1],v[i2])-corner is closest
                    l_sqr += direction_in_box[i2] * direction_in_box[i2]
                    delta = (direction_in_box[i0] * point_m_edge[i0]
                             + direction_in_box[i1] * point_p_edge[i1]
                             + direction_in_box[i2] * point_p_edge[i2])
                    line_parameter = -delta / l_sqr
                    sqr_dist += (point_m_edge[i0] * point_m_edge[i0]
                                 + point_p_edge[i1] * point_p_edge[i1]
                                 + point_p_edge[i2] * point_p_edge[i2]
                                 + delta * line_parameter)

                    point_in_box[i0] = box_half_size[i0]
                    point_in_box[i1] = -box_half_size[i1]
                    point_in_box[i2] = -box_half_size[i2]

    return sqr_dist, line_parameter


def _case_0(i0, i1, i2, point_in_box, direction_in_box, box_half_size):
    sqr_dist = 0.0
    point_m_edge0 = point_in_box[i0] - box_half_size[i0]
    point_m_edge1 = point_in_box[i1] - box_half_size[i1]
    prod0 = direction_in_box[i1] * point_m_edge0
    prod1 = direction_in_box[i0] * point_m_edge1

    if prod0 >= prod1:
        # line intersects P[i0] = e[i0]
        point_in_box[i0] = box_half_size[i0]

        point_p_edge1 = point_in_box[i1] + box_half_size[i1]
        delta = prod0 - direction_in_box[i0] * point_p_edge1
        if delta >= 0.0:
            inv_l_sqr = 1.0 / (direction_in_box[i0] * direction_in_box[i0]
                               + direction_in_box[i1] * direction_in_box[i1])
            sqr_dist += delta * delta * inv_l_sqr
            point_in_box[i1] = -box_half_size[i1]
            line_parameter = -(direction_in_box[i0] * point_m_edge0
                               + direction_in_box[i1] * point_p_edge1) * inv_l_sqr
        else:
            inv = 1.0 / direction_in_box[i0]
            point_in_box[i1] -= prod0 * inv
            line_parameter = -point_m_edge0 * inv
    else:
        # line intersects P[i1] = e[i1]
        point_in_box[i1] = box_half_size[i1]

        point_p_edge0 = point_in_box[i0] + box_half_size[i0]
        delta = prod1 - direction_in_box[i1] * point_p_edge0
        if delta >= 0.0:
            inv_l_sqr = 1.0 / (direction_in_box[i0] * direction_in_box[i0]
                               + direction_in_box[i1] * direction_in_box[i1])
            sqr_dist += delta * delta * inv_l_sqr
            point_in_box[i0] = -box_half_size[i0]
            line_parameter = -(direction_in_box[i0] * point_p_edge0
                               + direction_in_box[i1] * point_m_edge1) * inv_l_sqr
        else:
            inv = 1.0 / direction_in_box[i1]
            point_in_box[i0] -= prod1 * inv
            line_parameter = -point_m_edge1 * inv

    if point_in_box[i2] < -box_half_size[i2]:
        delta = point_in_box[i2] + box_half_size[i2]
        sqr_dist += delta * delta
        point_in_box[i2] = -box_half_size[i2]
    elif point_in_box[i2] > box_half_size[i2]:
        delta = point_in_box[i2] - box_half_size[i2]
        sqr_dist += delta * delta
        point_in_box[i2] = box_half_size[i2]

    return sqr_dist, line_parameter


def _case_00(i0, i1, i2, point_in_box, direction_in_box, box_half_size):
    line_parameter = (box_half_size[i0] - point_in_box[i0]) / direction_in_box[i0]
    point_in_box[i0] = box_half_size[i0]
    box_half_size_i12 = box_half_size[[i1, i2]]
    new_point_in_box = np.clip(
        point_in_box[[i1, i2]], -box_half_size_i12, box_half_size_i12)
    deltas = point_in_box[[i1, i2]] - new_point_in_box
    point_in_box[[i1, i2]] = new_point_in_box
    return np.dot(deltas, deltas), line_parameter


def _case_000(point_in_box, box_half_size):
    new_point_in_box = np.clip(point_in_box, -box_half_size, box_half_size)
    deltas = point_in_box - new_point_in_box
    point_in_box[:] = new_point_in_box
    return np.dot(deltas, deltas)
