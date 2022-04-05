import numpy as np
from mpl_toolkits import mplot3d
import pytransform3d.plot_utils as ppu
import pytransform3d.rotations as pr


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


def plot_circle(ax, center, radius, normal, show_normal=False):
    """Plot circle.

    Parameters
    ----------
    ax : Matplotlib 3d axis
        A matplotlib 3d axis.

    center : array, shape (3,)
        Center of the circle.

    radius : float
        Radius of the circle.

    normal : array, shape (3,)
        Normal to the plane in which the circle lies.

    show_normal : bool, optional (default: False)
        Display normal of the circle plane.
    """
    ax.scatter(center[0], center[1], center[2])
    u, v = pr.plane_basis_from_normal(normal)
    R = np.column_stack((u, v, normal))
    circle = np.array([center + R.dot(pr.matrix_from_angle(2, angle)).dot([radius, 0, 0])
                       for angle in np.linspace(0, 2 * np.pi, 20)])
    ax.plot(circle[:, 0], circle[:, 1], circle[:, 2])
    if show_normal:
        ppu.plot_vector(ax=ax, start=center, direction=normal, s=1.0)


def plot_aabb(ax, mins, maxs, alpha=1.0, color="r"):
    """Plot axis-aligned bounding box.

    Parameters
    ----------
    ax : Matplotlib 3d axis
        A matplotlib 3d axis.

    mins : array, shape (3,)
        Minimum values along axes.

    maxs : array, shape (3,)
        Maximum values along axes.

    alpha : float, optional (default: 1)
        Alpha value of edges.

    color : str
        Color of edges.
    """
    box2origin = np.eye(4)
    box2origin[:3, 3] = 0.5 * (mins + maxs)
    size = maxs - mins
    ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=True, alpha=alpha, color=color)


def plot_convex(ax, vertices, faces, alpha=0.5, color="b"):
    """Plot convex mesh.

    Parameters
    ----------
    ax : Matplotlib 3d axis
        A matplotlib 3d axis.

    vertices : array, shape (n_convex_points, 3)
        Vertices of the convex mesh.

    faces : array, shape (n_triangles, 3, 3)
        Vertices organized as triangles.

    alpha : float, optional (default: 0.5)
        Alpha value of faces.

    color : str
        Color of faces.
    """
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="red")
    surface = mplot3d.art3d.Poly3DCollection(faces)
    surface.set_facecolor(color)
    surface.set_alpha(alpha)
    ax.add_collection3d(surface)
    wireframe = mplot3d.art3d.Line3DCollection(faces)
    ax.add_collection3d(wireframe)
