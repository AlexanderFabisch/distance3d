from ._line import (
    point_to_line, line_to_line, point_to_line_segment, line_to_line_segment,
    line_segment_to_line_segment, _line_to_line_segment)
from ._triangle import (
    point_to_triangle, line_to_triangle, line_segment_to_triangle,
    triangle_to_triangle, triangle_to_rectangle)
from ._rectangle import (
    point_to_rectangle, line_to_rectangle, line_segment_to_rectangle,
    rectangle_to_rectangle)
from ._box import (
    point_to_box, line_to_box, line_segment_to_box, rectangle_to_box)
from ._plane import point_to_plane
from ._cylinder import point_to_cylinder
from ._ellipsoid import point_to_ellipsoid
from ._circle import point_to_circle
from ._disk import point_to_disk
