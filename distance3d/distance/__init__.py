"""Distance calculation of specific geometric shapes.

Note that for distance calculations to triangles, rectangles, boxes, cylinders,
spheres, capsules, and convex meshes GJK is usually the fastest option if
applicable. GJK is not applicable if one of the geometric objects is a line
or a plane and GJK is slower when one of the objects is a point or a line
segment.

In the following table an X indicates that the distance computation between
two geometric objects is implemented. G means the pair of shapes is covered
by GJK.

.. raw:: html

    <table>
      <tr>
        <th></th>
        <th style="transform:rotate(315deg);height: 130px;">point</th>
        <th style="transform:rotate(315deg);height: 130px;">line</th>
        <th style="transform:rotate(315deg);height: 130px;">line segment</th>
        <th style="transform:rotate(315deg);height: 130px;">plane</th>
        <th style="transform:rotate(315deg);height: 130px;">triangle</th>
        <th style="transform:rotate(315deg);height: 130px;">rectangle</th>
        <th style="transform:rotate(315deg);height: 130px;">circle</th>
        <th style="transform:rotate(315deg);height: 130px;">disk</th>
        <th style="transform:rotate(315deg);height: 130px;">box</th>
        <th style="transform:rotate(315deg);height: 130px;">ellipsoid</th>
        <th style="transform:rotate(315deg);height: 130px;">cylinder</th>
      </tr>
      <tr>
        <td>point</td>
        <td>-</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
      </tr>
      <tr>
        <td>line</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>-</td>
        <td>X</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>line segment</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>G</td>
        <td>X</td>
        <td>G</td>
        <td>G</td>
      </tr>
      <tr>
        <td>plane</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>-</td>
        <td>-</td>
        <td>X</td>
        <td>X</td>
        <td>-</td>
      </tr>
      <tr>
        <td>triangle</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>-</td>
        <td>G</td>
        <td>G</td>
        <td>G</td>
        <td>G</td>
      </tr>
      <tr>
        <td>rectangle</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>G</td>
        <td>-</td>
        <td>G</td>
        <td>G</td>
        <td>G</td>
        <td>G</td>
      </tr>
      <tr>
        <td>circle</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>disk</td>
        <td>X</td>
        <td>-</td>
        <td>G</td>
        <td>-</td>
        <td>G</td>
        <td>G</td>
        <td>-</td>
        <td>X</td>
        <td>G</td>
        <td>G</td>
        <td>G</td>
      </tr>
      <tr>
        <td>box</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>X</td>
        <td>G</td>
        <td>G</td>
        <td>-</td>
        <td>G</td>
        <td>G</td>
        <td>G</td>
        <td>G</td>
      </tr>
      <tr>
        <td>ellipsoid</td>
        <td>X</td>
        <td>-</td>
        <td>G</td>
        <td>X</td>
        <td>G</td>
        <td>G</td>
        <td>-</td>
        <td>G</td>
        <td>G</td>
        <td>G</td>
        <td>G</td>
      </tr>
      <tr>
        <td>cylinder</td>
        <td>X</td>
        <td>-</td>
        <td>G</td>
        <td>-</td>
        <td>G</td>
        <td>G</td>
        <td>-</td>
        <td>G</td>
        <td>G</td>
        <td>G</td>
        <td>G</td>
      </tr>
    </table>
"""
from ._line import (
    point_to_line, line_to_line, point_to_line_segment, line_to_line_segment,
    line_segment_to_line_segment)
from ._plane import (
    point_to_plane, line_to_plane, line_segment_to_plane, plane_to_plane,
    plane_to_triangle, plane_to_rectangle, plane_to_box, plane_to_ellipsoid)
from ._triangle import (
    point_to_triangle, line_to_triangle, line_segment_to_triangle,
    triangle_to_triangle, triangle_to_rectangle)
from ._rectangle import (
    point_to_rectangle, line_to_rectangle, line_segment_to_rectangle,
    rectangle_to_rectangle)
from ._disk import point_to_disk, disk_to_disk
from ._circle import point_to_circle, line_to_circle, line_segment_to_circle
from ._box import (
    point_to_box, line_to_box, line_segment_to_box, rectangle_to_box)
from ._ellipsoid import point_to_ellipsoid
from ._cylinder import point_to_cylinder


__all__ = [
    "point_to_line",
    "point_to_line_segment",
    "point_to_plane",
    "point_to_triangle",
    "point_to_rectangle",
    "point_to_disk",
    "point_to_circle",
    "point_to_box",
    "point_to_ellipsoid",
    "point_to_cylinder",
    "line_to_line",
    "line_to_line_segment",
    "line_to_plane",
    "line_to_triangle",
    "line_to_rectangle",
    "line_to_circle",
    "line_to_box",
    "line_segment_to_line_segment",
    "line_segment_to_plane",
    "line_segment_to_triangle",
    "line_segment_to_rectangle",
    "line_segment_to_circle",
    "line_segment_to_box",
    "plane_to_plane",
    "plane_to_triangle",
    "plane_to_rectangle",
    "plane_to_box",
    "plane_to_ellipsoid",
    "triangle_to_triangle",
    "triangle_to_rectangle",
    "rectangle_to_rectangle",
    "rectangle_to_box",
    "disk_to_disk",
]
