import numpy as np


def barycentric_transforms(tetrahedra_points):
    """Returns X. X.dot(coords) = (r, 1), where r is a Cartesian vector."""
    # NOTE that in the original paper it is not obvious that we have to take
    # the inverse
    return np.linalg.pinv(np.hstack((tetrahedra_points.transpose((0, 2, 1)),
                                     np.ones((len(tetrahedra_points), 1, 4)))))
