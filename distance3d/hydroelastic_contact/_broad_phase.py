from distance3d.broad_phase import BoundingVolumeHierarchy
from pytransform3d import urdf
from distance3d.hydroelastic_contact import RigidBody


class HydroelasticBoundingVolumeHierarchy(BoundingVolumeHierarchy):
    """Hydroelastic Bounding volume hierarchy (BVH) for broad phase collision detection.

        This BVH works the same but uses the hydro-elastic Rigidbody as the Collider Object.

        Parameters
        ----------
        tm : pytransform3d.transform_manager.TransformManager
            Transform manager that stores the transformations.

        base_frame : str
            Name of the base frame in which colliders are represented.

        Attributes
        ----------
        aabbtree_ : AabbTree
            Tree of axis-aligned bounding boxes.

        colliders_ : dict
            Maps frames of collision objects to colliders.

        self_collision_whitelists_ : dict
            Whitelists for self-collision detection in case this BVH represents
            a robot.
    """

    def __init__(self, tm, base_frame):
        super().__init__(tm, base_frame)

    def _make_collider(self, tm, obj, make_artists):
        a2_b = tm.get_transform(obj.frame, self.base_frame)
        if isinstance(obj, urdf.Sphere):
            collider = RigidBody.make_sphere(a2_b[:3, 3], obj.radius, 2)
        elif isinstance(obj, urdf.Box):
            collider = RigidBody.make_box(a2_b, obj.size)
        elif isinstance(obj, urdf.Cylinder):
            collider = RigidBody.make_cylinder(a2_b, obj.radius, obj.length, resolution_hint=0.01)
        else:
            assert isinstance(obj, urdf.Mesh)
            print("Arbitrary mesh conversion is not implemented!! ")
            # TODO Arbitrary mesh conversion

        if make_artists:
            collider.make_artist()
        return collider

    # Only overwritten to change the Doku.
    def aabb_overlapping_colliders(self, collider, whitelist=()):
        """Get colliders with an overlapping AABB.

        This function performs broad phase collision detection with a bounding
        volume hierarchy, where the bounding volumes are axis-aligned bounding
        boxes.

        Parameters
        ----------
        collider : RigidBody
            RigidBody.

        whitelist : sequence
            Names of frames to which collisions are allowed.

        Returns
        -------
        colliders : dict
            Maps frame names to colliders with overlapping AABB.
        """
        return super().aabb_overlapping_colliders(collider, whitelist)
