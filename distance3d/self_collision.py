"""Self-collision detection."""
from . import mpr


def detect(bvh):
    """Detect self collisions of a robot represented by a BVH.

    Parameters
    ----------
    bvh : BoundingVolumeHierarchy
        Bounding volume hierarchy that contains colliders of a robot. Note that
        the attribute bvh.self_collision_whitelists_ has to be filled.
        Otherwise a collection of a collider with itself or direct neighbors
        will be considered a self collision!

    Returns
    -------
    contacts : dict
        Maps each collider frame to a boolean indicating whether it is in
        contact with another collider or not.
    """
    contacts = {}
    for frame, collider in bvh.colliders_.items():
        if frame in contacts:
            continue  # contact was detected before

        candidates = bvh.aabb_overlapping_colliders(
            collider, whitelist=bvh.self_collision_whitelists_[frame])

        contacts[frame] = False
        for frame2, collider2 in candidates.items():
            if mpr.mpr_intersection(collider, collider2):
                contacts[frame] = True
                contacts[frame2] = True
                break
    return contacts


def detect_any(bvh):
    """Detect self collisions of a robot represented by a BVH.

    This function aborts on first detected self collision.

    Parameters
    ----------
    bvh : BoundingVolumeHierarchy
        Bounding volume hierarchy that contains colliders of a robot. Note that
        the attribute bvh.self_collision_whitelists_ has to be filled.
        Otherwise a collection of a collider with itself or direct neighbors
        will be considered a self collision!

    Returns
    -------
    has_self_collision : bool
        Whether there is any self collision or not.
    """
    for frame, collider in bvh.colliders_.items():
        candidates = bvh.aabb_overlapping_colliders(
            collider, whitelist=bvh.self_collision_whitelists_[frame])

        for frame2, collider2 in candidates.items():
            if mpr.mpr_intersection(collider, collider2):
                return True
    return False
