"""Gilbert-Johnson-Keerthi (GJK) for distance calculation of convex shapes.

The GJK algorithm only works for convex shapes. Concave objects have to be
decomposed into convex shapes first.

This module contains several flavours of the GJK algorithm. Some of them
only detect collisions (intersections) and some of them calculate distances
between separated objects as well.

The original publication describing the algorithm is:

E.G. Gilbert, D.W. Johnson, S.S. Keerthi: A fast procedure for computing
the distance between complex objects in three-dimensional space, IEEE
Journal on Robotics and Automation (1988),
https://graphics.stanford.edu/courses/cs448b-00-winter/papers/gilbert.pdf
"""
from ._gjk_original import gjk_distance_original
from ._gjk_libccd import gjk_intersection_libccd
from ._gjk_jolt import gjk_distance_jolt, gjk_intersection_jolt
from ._gjk_nesterov_accelerated import gjk_nesterov_accelerated_intersection, gjk_nesterov_accelerated_distance, \
    gjk_nesterov_accelerated
from ._gjk_nesterov_accelerated_primitives import gjk_nesterov_accelerated_primitives_distance, \
    gjk_nesterov_accelerated_primitives_intersection, \
    gjk_nesterov_accelerated_primitives

# Aliases
gjk = gjk_distance_jolt
gjk_distance = gjk_distance_jolt
gjk_intersection = gjk_intersection_jolt

__all__ = [
    "gjk_distance_original",
    "gjk_intersection_libccd",
    "gjk_distance_jolt",
    "gjk_intersection_jolt",
    "gjk_nesterov_accelerated_distance",
    "gjk_nesterov_accelerated_intersection",
    "gjk_nesterov_accelerated_primitives_distance",
    "gjk_nesterov_accelerated_primitives_intersection",

    "gjk_distance",
    "gjk_intersection",
    "gjk",

    # For benchmarking
    "gjk_nesterov_accelerated",
    "gjk_nesterov_accelerated_primitives",
]
