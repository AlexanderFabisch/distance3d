import warnings
from pytransform3d import urdf
import pytransform3d.visualizer as pv
from .colliders import Cylinder, Sphere, Box, Convex


def get_colliders(tm, frame):
    """Get colliders.

    Parameters
    ----------
    tm : urdf.UrdfTransformManager
        Kinematic tree of the robot.

    frame : str
        Base frame.

    Returns
    -------
    TODO
    """
    if frame not in tm.nodes:
        raise KeyError("Unknown frame '%s'" % frame)

    colliders = []
    for obj in tm.collision_objects:
        A2B = tm.get_transform(obj.frame, frame)
        try:
            if isinstance(obj, urdf.Sphere):
                artist = pv.Sphere(radius=obj.radius, A2B=A2B)
                collider = Sphere(
                    center=A2B[:3, 3], radius=obj.radius, artist=artist)
            elif isinstance(obj, urdf.Box):
                artist = pv.Box(size=obj.size, A2B=A2B)
                collider = Box(A2B, obj.size, artist=artist)
            elif isinstance(obj, urdf.Cylinder):
                artist = pv.Cylinder(
                    length=obj.length, radius=obj.radius, A2B=A2B)
                collider = Cylinder(
                    cylinder2origin=A2B, radius=obj.radius, length=obj.length,
                    artist=artist)
            else:
                assert isinstance(obj, urdf.Mesh)
                artist = pv.Mesh(filename=obj.filename, s=obj.scale, A2B=A2B)
                collider = Convex.from_mesh(
                    obj.filename, A2B, obj.scale, artist=artist)
            colliders.append(collider)
        except RuntimeError as e:
            warnings.warn(str(e))

    return colliders
