from . import gjk


def detect(bvh, collision_margin=1e-3):
    """Detect self collisions of a robot represented by a BVH.

    Parameters
    ----------
    bvh : BoundingVolumeHierarchy
        Bounding volume hierarchy that contains colliders of a robot.

    collision_margin : float, optional (default: 0.001)
        Distance between colliders that is considered to be a collision.

    Returns
    -------
    contacts : dict
        Maps each collider frame to a boolean indicating whether it is in
        contact with another collider or not.
    """
    contacts = {}
    for frame, collider in bvh.colliders_.items():
        candidates = bvh.aabb_overlapping_colliders(
            collider, whitelist=bvh.self_collision_whitelists_[frame])
        contacts[frame] = False
        for frame2, collider2 in candidates.items():
            dist, _, _, _ = gjk.gjk_with_simplex(collider, collider2)
            if dist < collision_margin:
                contacts[frame] = True
                break
    return contacts
