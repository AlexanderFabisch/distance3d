from .utils import norm_vector


def randn_point(random_state):
    """Sample 3D point from standard normal distribution.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    Returns
    -------
    point : array, shape (3,)
        3D Point sampled from standard normal distribution.
    """
    return random_state.randn(3)


def randn_direction(random_state):
    """Sample 3D direction from standard normal distribution and normalize it.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    Returns
    -------
    direction : array, shape (3,)
        3D direction: 3D vector of unit length.
    """
    return norm_vector(random_state.randn(3))


def randn_line(random_state):
    """Sample 3D line.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    Returns
    -------
    line_point : array, shape (3,)
        3D Point sampled from standard normal distribution.

    line_direction : array, shape (3,)
        3D direction: 3D vector of unit length.
    """
    line_point = randn_point(random_state)
    line_direction = randn_direction(random_state)
    return line_point, line_direction


def randn_line_segment(random_state):
    """Sample 3D line segment.

    Parameters
    ----------
    random_state : np.random.RandomState
        Random number generator.

    Returns
    -------

    segment_start : array, shape (3,)
        Start point of segment sampled from a standard normal distribution.

    segment_end : array, shape (3,)
        End point of segment sampled from a standard normal distribution.
    """
    return randn_point(random_state), randn_point(random_state)
