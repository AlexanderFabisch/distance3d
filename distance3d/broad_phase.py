"""Broad-phase collision detection."""
import warnings

import numpy as np
from aabbtree import AABBTree
from pytransform3d import urdf

from .colliders import Sphere, Box, Cylinder, MeshGraph
from .urdf_utils import self_collision_whitelists


class BoundingVolumeHierarchy:
    """Bounding volume hierarchy (BVH) for broad phase collision detection.

    Wraps multiple colliders that are connected through transformations.
    In addition, these colliders are stored in an AABB tree for broad phase
    collision detection.

    Parameters
    ----------
    tm : pytransform3d.transform_manager.TransformManager
        Transform manager that stores the transformations.

    base_frame : str
        Name of the base frame in which colliders are represented.

    Attributes
    ----------
    aabbtree_ : AABBTree
        Tree of axis-aligned bounding boxes.

    colliders_ : dict
        Maps frames of collision objects to colliders.

    self_collision_whitelists_ : dict
        Whitelists for self-collision detection in case this BVH represents
        a robot.
    """
    def __init__(self, tm, base_frame):
        self.tm = tm
        self.base_frame = base_frame
        self.collider_frames = set()
        self.aabbtree_ = AABBTree()
        self.colliders_ = {}
        self.self_collision_whitelists_ = {}

    def fill_tree_with_colliders(
            self, tm, make_artists=False,
            fill_self_collision_whitelists=False):
        """Fill tree with colliders from URDF transform manager.

        Parameters
        ----------
        tm : pytransform3d.urdf.UrdfTransformManager
            Transform manager that has colliders.

        make_artists : bool, optional (default: False)
            Create artist for visualization for each collision object.

        fill_self_collision_whitelists : bool, optional (default: False)
            Fill whitelists for self collision detection. All collision
            objects connected to the current link, child, and parent links
            will be ignored.
        """
        for obj in tm.collision_objects:
            try:
                collider = self._make_collider(tm, obj, make_artists)
                self.add_collider(obj.frame, collider)
            except RuntimeError as e:
                warnings.warn(str(e))

        if fill_self_collision_whitelists:
            self.self_collision_whitelists_.update(
                self_collision_whitelists(tm))

    def _make_collider(self, tm, obj, make_artists):
        A2B = tm.get_transform(obj.frame, self.base_frame)
        if isinstance(obj, urdf.Sphere):
            collider = Sphere(center=A2B[:3, 3], radius=obj.radius)
        elif isinstance(obj, urdf.Box):
            collider = Box(A2B, obj.size)
        elif isinstance(obj, urdf.Cylinder):
            collider = Cylinder(
                cylinder2origin=A2B, radius=obj.radius,
                length=obj.length)
        else:
            assert isinstance(obj, urdf.Mesh)
            import open3d as o3d
            mesh = o3d.io.read_triangle_mesh(obj.filename)
            vertices = np.asarray(mesh.vertices) * obj.scale
            triangles = np.asarray(mesh.triangles)
            collider = MeshGraph(A2B, vertices, triangles)
        if make_artists:
            collider.make_artist()
        return collider

    def add_collider(self, frame, collider):
        """Add collider.

        Parameters
        ----------
        frame : Hashable
            Frame in which the collider is located.

        collider : ConvexCollider
            Collider.
        """
        self.collider_frames.add(frame)
        self.colliders_[frame] = collider
        self.aabbtree_.add(collider.aabb(), (frame, collider))

    def update_collider_poses(self):
        """Update poses of all colliders from transform manager."""
        self.aabbtree_ = AABBTree()
        for frame in self.colliders_:
            A2B = self.tm.get_transform(frame, self.base_frame)
            collider = self.colliders_[frame]
            collider.update_pose(A2B)
            self.aabbtree_.add(collider.aabb(), (frame, collider))

    def get_colliders(self):
        """Get all colliders.

        Returns
        -------
        colliders : list
            List of colliders.
        """
        return self.colliders_.values()

    def get_artists(self):
        """Get all artists.

        Returns
        -------
        artists : list
            List of artists.
        """
        return [collider.artist_ for collider in self.colliders_.values()
                if collider.artist_ is not None]

    def aabb_overlapping_colliders(self, collider, whitelist=()):
        """Get colliders with an overlapping AABB.

        This function performs broad phase collision detection with a bounding
        volume hierarchy, where the bounding volumes are axis-aligned bounding
        boxes.

        Parameters
        ----------
        collider : ConvexCollider
            Collider.

        whitelist : sequence
            Names of frames to which collisions are allowed.

        Returns
        -------
        colliders : dict
            Maps frame names to colliders with overlapping AABB.
        """
        aabb = collider.aabb()
        colliders = dict(self.aabbtree_.overlap_values(aabb))
        for frame in whitelist:
            colliders.pop(frame, None)
        return colliders

    def get_collider_frames(self):
        """Get collider frames.

        Returns
        -------
        collider_frames : set
            Collider frames.
        """
        return self.collider_frames
