"""Gilbert-Johnson-Keerthi (GJK) for distance calculation of convex shapes."""
from ._gjk_distance import gjk, gjk_with_simplex
from ._gjk_intersection import gjk_intersection


__all__ = ["gjk", "gjk_with_simplex", "gjk_intersection"]
