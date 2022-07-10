"""Gilbert-Johnson-Keerthi (GJK) for distance calculation of convex shapes.

The GJK algorithm only works for convex shapes. Concave objects have to be
decomposed into convex shapes first.

This module contains several flavours of the GJK algorithm. Some of them
only detect collisions (intersections) and some of them calculate distances
between separated objects as well.
"""
from ._gjk_original import gjk_distance_original
from ._gjk_libccd import gjk_intersection_libccd
from ._gjk_jolt import gjk_distance_jolt, gjk_intersection_jolt


# Aliases
gjk = gjk_distance_jolt
gjk_distance = gjk_distance_jolt
gjk_intersection = gjk_intersection_jolt


__all__ = [
    "gjk_distance_original",
    "gjk_intersection_libccd",
    "gjk_distance_jolt",
    "gjk_intersection_jolt",
    "gjk_distance",
    "gjk_intersection",
    "gjk"
]
