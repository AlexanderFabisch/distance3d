import math

import numpy as np


def _line_to_box(line_point, line_direction, box2origin, size, origin2box=None):
    box_half_size = 0.5 * size

    # compute coordinates of line in box coordinate system
    if origin2box is None:
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
    contact_point_line = line_point + line_parameter * line_direction

    # compute closest point on box
    # undo the reflections applied previously
    contact_point_box = box2origin[:3, 3] + box2origin[:3, :3].dot(
        direction_sign * point_in_box)

    return math.sqrt(sqr_dist), contact_point_line, contact_point_box, line_parameter


def _case_no_zeros(point_in_box, direction_in_box, box_half_size):
    kPmE = point_in_box - box_half_size

    fProdDxPy = direction_in_box[0] * kPmE[1]
    fProdDyPx = direction_in_box[1] * kPmE[0]

    if fProdDyPx >= fProdDxPy:
        fProdDzPx = direction_in_box[2] * kPmE[0]
        fProdDxPz = direction_in_box[0] * kPmE[2]
        if fProdDzPx >= fProdDxPz:
            # line intersects x = e0
            sqr_dist, line_parameter = _box_face(0, 1, 2, point_in_box, direction_in_box, kPmE, box_half_size)
        else:
            # line intersects z = e2
            sqr_dist, line_parameter = _box_face(2, 0, 1, point_in_box, direction_in_box, kPmE, box_half_size)
    else:
        fProdDzPy = direction_in_box[2] * kPmE[1]
        fProdDyPz = direction_in_box[1] * kPmE[2]
        if fProdDzPy >= fProdDyPz:
            # line intersects y = e1
            sqr_dist, line_parameter = _box_face(1, 2, 0, point_in_box, direction_in_box, kPmE, box_half_size)
        else:
            # line intersects z = e2
            sqr_dist, line_parameter = _box_face(2, 0, 1, point_in_box, direction_in_box, kPmE, box_half_size)
    return sqr_dist, line_parameter


def _box_face(i0, i1, i2, point_in_box, direction_in_box, rkPmE, box_half_size):
    sqr_dist = 0.0
    kPpE = np.zeros(3)

    kPpE[i1] = point_in_box[i1] + box_half_size[i1]
    kPpE[i2] = point_in_box[i2] + box_half_size[i2]
    if direction_in_box[i0] * kPpE[i1] >= direction_in_box[i1] * rkPmE[i0]:
        if direction_in_box[i0] * kPpE[i2] >= direction_in_box[i2] * rkPmE[i0]:
            # v[i1] >= -e[i1], v[i2] >= -e[i2] (distance = 0)
            point_in_box[i0] = box_half_size[i0]
            point_in_box[i1] -= direction_in_box[i1] * rkPmE[i0] / direction_in_box[i0]
            point_in_box[i2] -= direction_in_box[i2] * rkPmE[i0] / direction_in_box[i0]
            line_parameter = -rkPmE[i0] / direction_in_box[i0]
        else:
            # v[i1] >= -e[i1], v[i2] < -e[i2]
            fLSqr = direction_in_box[i0] * direction_in_box[i0] + direction_in_box[i2] * direction_in_box[i2]
            fTmp = fLSqr * kPpE[i1] - direction_in_box[i1] * (direction_in_box[i0] * rkPmE[i0] + direction_in_box[i2] * kPpE[i2])
            if fTmp <= 2.0 * fLSqr * box_half_size[i1]:
                fT = fTmp / fLSqr
                fLSqr += direction_in_box[i1] * direction_in_box[i1]
                fTmp = kPpE[i1] - fT
                delta = direction_in_box[i0] * rkPmE[i0] + direction_in_box[i1] * fTmp + direction_in_box[i2] * kPpE[i2]
                line_parameter = -delta / fLSqr
                sqr_dist += rkPmE[i0] * rkPmE[i0] + fTmp * fTmp + kPpE[i2] * kPpE[i2] + delta * line_parameter

                point_in_box[i0] = box_half_size[i0]
                point_in_box[i1] = fT - box_half_size[i1]
                point_in_box[i2] = -box_half_size[i2]
            else:
                fLSqr += direction_in_box[i1] * direction_in_box[i1]
                delta = direction_in_box[i0] * rkPmE[i0] + direction_in_box[i1] * rkPmE[i1] + direction_in_box[i2] * kPpE[i2]
                line_parameter = -delta / fLSqr
                sqr_dist += rkPmE[i0] * rkPmE[i0] + rkPmE[i1] * rkPmE[i1] + kPpE[i2] * kPpE[i2] + delta * line_parameter

                point_in_box[i0] = box_half_size[i0]
                point_in_box[i1] = box_half_size[i1]
                point_in_box[i2] = -box_half_size[i2]
    else:
        if direction_in_box[i0] * kPpE[i2] >= direction_in_box[i2] * rkPmE[i0]:
            # v[i1] < -e[i1], v[i2] >= -e[i2]
            fLSqr = direction_in_box[i0] * direction_in_box[i0] + direction_in_box[i1] * direction_in_box[i1]
            fTmp = fLSqr * kPpE[i2] - direction_in_box[i2] * (direction_in_box[i0] * rkPmE[i0] + direction_in_box[i1] * kPpE[i1])
            if fTmp <= 2.0 * fLSqr * box_half_size[i2]:
                fT = fTmp / fLSqr
                fLSqr += direction_in_box[i2] * direction_in_box[i2]
                fTmp = kPpE[i2] - fT
                delta = direction_in_box[i0] * rkPmE[i0] + direction_in_box[i1] * kPpE[i1] + direction_in_box[i2] * fTmp
                line_parameter = -delta / fLSqr
                sqr_dist += rkPmE[i0] * rkPmE[i0] + kPpE[i1] * kPpE[i1] + fTmp * fTmp + delta * line_parameter

                point_in_box[i0] = box_half_size[i0]
                point_in_box[i1] = -box_half_size[i1]
                point_in_box[i2] = fT - box_half_size[i2]
            else:
                fLSqr += direction_in_box[i2] * direction_in_box[i2]
                delta = direction_in_box[i0] * rkPmE[i0] + direction_in_box[i1] * kPpE[i1] + direction_in_box[i2] * rkPmE[i2]
                line_parameter = -delta / fLSqr
                sqr_dist += rkPmE[i0] * rkPmE[i0] + kPpE[i1] * kPpE[i1] + rkPmE[i2] * rkPmE[i2] + delta * line_parameter

                point_in_box[i0] = box_half_size[i0]
                point_in_box[i1] = -box_half_size[i1]
                point_in_box[i2] = box_half_size[i2]
        else:
            # v[i1] < -e[i1], v[i2] < -e[i2]
            fLSqr = direction_in_box[i0] * direction_in_box[i0] + direction_in_box[i2] * direction_in_box[i2]
            fTmp = fLSqr * kPpE[i1] - direction_in_box[i1] * (direction_in_box[i0] * rkPmE[i0] + direction_in_box[i2] * kPpE[i2])
            if fTmp >= 0.0:
                # v[i1]-edge is closest
                if fTmp <= 2.0 * fLSqr * box_half_size[i1]:
                    fT = fTmp / fLSqr
                    fLSqr += direction_in_box[i1] * direction_in_box[i1]
                    fTmp = kPpE[i1] - fT
                    delta = direction_in_box[i0] * rkPmE[i0] + direction_in_box[i1] * fTmp + direction_in_box[i2] * kPpE[i2]
                    line_parameter = -delta / fLSqr
                    sqr_dist += rkPmE[i0] * rkPmE[i0] + fTmp * fTmp + kPpE[i2] * kPpE[i2] + delta * line_parameter

                    point_in_box[i0] = box_half_size[i0]
                    point_in_box[i1] = fT - box_half_size[i1]
                    point_in_box[i2] = -box_half_size[i2]
                else:
                    fLSqr += direction_in_box[i1] * direction_in_box[i1]
                    delta = direction_in_box[i0] * rkPmE[i0] + direction_in_box[i1] * rkPmE[i1] + direction_in_box[i2] * kPpE[i2]
                    line_parameter = -delta / fLSqr
                    sqr_dist += rkPmE[i0] * rkPmE[i0] + rkPmE[i1] * rkPmE[i1] + kPpE[i2] * kPpE[
                        i2] + delta * line_parameter

                    point_in_box[i0] = box_half_size[i0]
                    point_in_box[i1] = box_half_size[i1]
                    point_in_box[i2] = -box_half_size[i2]
            else:
                fLSqr = direction_in_box[i0] * direction_in_box[i0] + direction_in_box[i1] * direction_in_box[i1]
                fTmp = fLSqr * kPpE[i2] - direction_in_box[i2] * (direction_in_box[i0] * rkPmE[i0] + direction_in_box[i1] * kPpE[i1])
                if fTmp >= 0.0:
                    # v[i2]-edge is closest
                    if fTmp <= 2.0 * fLSqr * box_half_size[i2]:
                        fT = fTmp / fLSqr
                        fLSqr += direction_in_box[i2] * direction_in_box[i2]
                        fTmp = kPpE[i2] - fT
                        delta = direction_in_box[i0] * rkPmE[i0] + direction_in_box[i1] * kPpE[i1] + direction_in_box[i2] * fTmp
                        line_parameter = -delta / fLSqr
                        sqr_dist += rkPmE[i0] * rkPmE[i0] + kPpE[i1] * kPpE[i1] + fTmp * fTmp + delta * line_parameter

                        point_in_box[i0] = box_half_size[i0]
                        point_in_box[i1] = -box_half_size[i1]
                        point_in_box[i2] = fT - box_half_size[i2]
                    else:
                        fLSqr += direction_in_box[i2] * direction_in_box[i2]
                        delta = direction_in_box[i0] * rkPmE[i0] + direction_in_box[i1] * kPpE[i1] + direction_in_box[i2] * rkPmE[i2]
                        line_parameter = -delta / fLSqr
                        sqr_dist += rkPmE[i0] * rkPmE[i0] + kPpE[i1] * kPpE[i1] + rkPmE[i2] * rkPmE[
                            i2] + delta * line_parameter

                        point_in_box[i0] = box_half_size[i0]
                        point_in_box[i1] = -box_half_size[i1]
                        point_in_box[i2] = box_half_size[i2]
                else:
                    # (v[i1],v[i2])-corner is closest
                    fLSqr += direction_in_box[i2] * direction_in_box[i2]
                    delta = direction_in_box[i0] * rkPmE[i0] + direction_in_box[i1] * kPpE[i1] + direction_in_box[i2] * kPpE[i2]
                    line_parameter = -delta / fLSqr
                    sqr_dist += rkPmE[i0] * rkPmE[i0] + kPpE[i1] * kPpE[i1] + kPpE[i2] * kPpE[i2] + delta * line_parameter

                    point_in_box[i0] = box_half_size[i0]
                    point_in_box[i1] = -box_half_size[i1]
                    point_in_box[i2] = -box_half_size[i2]

    return sqr_dist, line_parameter


def _case_0(i0, i1, i2, point_in_box, direction_in_box, box_half_size):
    sqr_dist = 0.0
    fPmE0 = point_in_box[i0] - box_half_size[i0]
    fPmE1 = point_in_box[i1] - box_half_size[i1]
    fProd0 = direction_in_box[i1] * fPmE0
    fProd1 = direction_in_box[i0] * fPmE1

    if fProd0 >= fProd1:
        # line intersects P[i0] = e[i0]
        point_in_box[i0] = box_half_size[i0]

        fPpE1 = point_in_box[i1] + box_half_size[i1]
        delta = fProd0 - direction_in_box[i0] * fPpE1
        if delta >= 0.0:
            fInvLSqr = 1.0 / (direction_in_box[i0] * direction_in_box[i0] + direction_in_box[i1] * direction_in_box[i1])
            sqr_dist += delta * delta * fInvLSqr
            point_in_box[i1] = -box_half_size[i1]
            line_parameter = -(direction_in_box[i0] * fPmE0 + direction_in_box[i1] * fPpE1) * fInvLSqr
        else:
            fInv = 1.0 / direction_in_box[i0]
            point_in_box[i1] -= fProd0 * fInv
            line_parameter = -fPmE0 * fInv
    else:
        # line intersects P[i1] = e[i1]
        point_in_box[i1] = box_half_size[i1]

        fPpE0 = point_in_box[i0] + box_half_size[i0]
        delta = fProd1 - direction_in_box[i1] * fPpE0
        if delta >= 0.0:
            fInvLSqr = 1.0 / (direction_in_box[i0] * direction_in_box[i0] + direction_in_box[i1] * direction_in_box[i1])
            sqr_dist += delta * delta * fInvLSqr
            point_in_box[i0] = -box_half_size[i0]
            line_parameter = -(direction_in_box[i0] * fPpE0 + direction_in_box[i1] * fPmE1) * fInvLSqr
        else:
            fInv = 1.0 / direction_in_box[i1]
            point_in_box[i0] -= fProd1 * fInv
            line_parameter = -fPmE1 * fInv

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
    new_point_in_box = np.clip(point_in_box[[i1, i2]], -box_half_size_i12, box_half_size_i12)
    deltas = point_in_box[[i1, i2]] - new_point_in_box
    point_in_box[[i1, i2]] = new_point_in_box
    return np.dot(deltas, deltas), line_parameter


def _case_000(point_in_box, box_half_size):
    new_point_in_box = np.clip(point_in_box, -box_half_size, box_half_size)
    deltas = point_in_box - new_point_in_box
    point_in_box[:] = new_point_in_box
    return np.dot(deltas, deltas)