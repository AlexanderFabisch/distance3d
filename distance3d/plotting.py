import numpy as np


def plot_line(ax, line_point, line_direction, length=10):
    """Plot line.

    Parameters
    ----------
    ax : Matplotlib 3d axis
        A matplotlib 3d axis.

    line_point : array, shape (3,)
        Point on line.

    line_direction : array, shape (3,)
        Direction of the line. This is assumed to be of unit length.

    length : float, optional (default: 10)
        Length of the line that should be displayed, since we cannot display
        an infinitely long line.
    """
    line = np.vstack([line_point + t * line_direction
                      for t in np.linspace(-length / 2, length / 2, 2)])
    ax.scatter(line_point[0], line_point[1], line_point[2])
    ax.plot(line[:, 0], line[:, 1], line[:, 2])


def plot_segment(ax, segment_start, segment_end):
    """Plot line.

    Parameters
    ----------
    ax : Matplotlib 3d axis
        A matplotlib 3d axis.

    segment_start : array, shape (3,)
        Start point of segment.

    segment_end : array, shape (3,)
        End point of segment.
    """
    points = np.vstack((segment_start, segment_end))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.plot(points[:, 0], points[:, 1], points[:, 2])
