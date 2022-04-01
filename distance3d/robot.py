import warnings
from pytransform3d import urdf
import pytransform3d.visualizer as pv
from .colliders import Cylinder, Sphere, Convex


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
                collider = Sphere(center=A2B[:3, 3], radius=obj.radius)
            elif isinstance(obj, urdf.Box):
                collider = Convex.from_box(A2B, obj.size)
            elif isinstance(obj, urdf.Cylinder):
                collider = Cylinder(
                    cylinder2origin=A2B, radius=obj.radius, length=obj.length)
            else:
                assert isinstance(obj, urdf.Mesh)
                collider = Convex.from_mesh(obj.filename, A2B, obj.scale)
            colliders.append(collider)
        except RuntimeError as e:
            warnings.warn(str(e))

    return colliders


def get_geometries(tm, frame):
    """Get geometries.

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

    collision_objects = _objects_to_artists(tm.collision_objects)

    for collision_object_frame, obj in collision_objects.items():
        A2B = tm.get_transform(collision_object_frame, frame)
        obj.set_data(A2B)

    geometries = []
    for obj in collision_objects.values():
        geometries += obj.geometries

    return geometries


def _objects_to_artists(objects):
    """Convert geometries from URDF to artists.

    Parameters
    ----------
    objects : list of Geometry
        Objects parsed from URDF.

    Returns
    -------
    artists : dict
        Mapping from frame names to artists.
    """
    artists = {}
    for obj in objects:
        if obj.color is None:
            color = None
        else:
            # we loose the alpha channel as it is not supported by Open3D
            color = (obj.color[0], obj.color[1], obj.color[2])
        try:
            if isinstance(obj, urdf.Sphere):
                artist = pv.Sphere(radius=obj.radius, c=color)
            elif isinstance(obj, urdf.Box):
                artist = pv.Box(obj.size, c=color)
            elif isinstance(obj, urdf.Cylinder):
                artist = pv.Cylinder(obj.length, obj.radius, c=color)
            else:
                assert isinstance(obj, urdf.Mesh)
                artist = pv.Mesh(obj.filename, s=obj.scale, c=color)
            artists[obj.frame] = artist
        except RuntimeError as e:
            warnings.warn(str(e))
    return artists
