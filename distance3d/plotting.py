import numpy as np
from mpl_toolkits import mplot3d
import pytransform3d.plot_utils as ppu


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


def plot_rectangle(ax, rectangle_center, rectangle_axes, rectangle_lengths, show_axes=True, surface_alpha=0.1):
    """Plot rectangle.

    Parameters
    ----------
    ax : Matplotlib 3d axis
        A matplotlib 3d axis.

    rectangle_center : array, shape (3,)
        Center point of the rectangle.

    rectangle_axes : array, shape (2, 3)
        Each row is a vector of unit length, indicating the direction of one
        axis of the rectangle. Both vectors are orthogonal.

    rectangle_lengths : array, shape (2,)
        Lengths of the two sides of the rectangle.

    show_axes : bool, optional (default: True)
        Show axes of the rectangle as arrows.

    surface_alpha : float, optional (default: 0.1)
        Alpha value of the rectangle surface.
    """
    rectangle_points = np.vstack((
        rectangle_center + -0.5 * rectangle_axes[0] * rectangle_lengths[0] + -0.5 * rectangle_axes[1] * rectangle_lengths[1],
        rectangle_center + 0.5 * rectangle_axes[0] * rectangle_lengths[0] + -0.5 * rectangle_axes[1] * rectangle_lengths[1],
        rectangle_center + 0.5 * rectangle_axes[0] * rectangle_lengths[0] + 0.5 * rectangle_axes[1] * rectangle_lengths[1],
        rectangle_center + -0.5 * rectangle_axes[0] * rectangle_lengths[0] + 0.5 * rectangle_axes[1] * rectangle_lengths[1],
        rectangle_center + -0.5 * rectangle_axes[0] * rectangle_lengths[0] + -0.5 * rectangle_axes[1] * rectangle_lengths[1],
    ))

    vertices = np.vstack((
        rectangle_points[0], rectangle_points[1], rectangle_points[2],
        rectangle_points[0], rectangle_points[2], rectangle_points[3]))
    try:  # Matplotlib < 3.5
        rectangle = mplot3d.art3d.Poly3DCollection(vertices)
    except ValueError:  # Matplotlib >= 3.5
        rectangle = mplot3d.art3d.Poly3DCollection(vertices.reshape(2, 3, 3))
    rectangle.set_alpha(surface_alpha)
    ax.add_collection3d(rectangle)

    ax.plot(rectangle_points[:, 0], rectangle_points[:, 1], rectangle_points[:, 2])
    ax.scatter(rectangle_center[0], rectangle_center[1], rectangle_center[2])
    if show_axes:
        ppu.plot_vector(
            ax=ax, start=rectangle_center, direction=rectangle_axes[0],
            s=0.5 * rectangle_lengths[0], color="r")
        ppu.plot_vector(
            ax=ax, start=rectangle_center, direction=rectangle_axes[1],
            s=0.5 * rectangle_lengths[1], color="g")


def plot_triangle(ax, triangle_points, surface_alpha=0.1):
    """Plot triangle.

    Parameters
    ----------
    ax : Matplotlib 3d axis
        A matplotlib 3d axis.

    triangle_points : array, shape (3, 3)
        Each row contains a point of the triangle (A, B, C).

    surface_alpha : float, optional (default: 0.1)
        Alpha value of the rectangle surface.
    """
    try:  # Matplotlib < 3.5
        triangle = mplot3d.art3d.Poly3DCollection(triangle_points)
    except ValueError:  # Matplotlib >= 3.5
        triangle = mplot3d.art3d.Poly3DCollection(
            triangle_points.reshape(1, 3, 3))
    triangle.set_alpha(surface_alpha)
    ax.add_collection3d(triangle)

    triangle_points = np.vstack((triangle_points, [triangle_points[0]]))
    ax.plot(triangle_points[:, 0], triangle_points[:, 1], triangle_points[:, 2])
