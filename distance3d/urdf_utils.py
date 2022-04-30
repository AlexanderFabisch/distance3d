import re
import warnings


def self_collision_whitelists(tm):
    """Construct self-collision whitelist.

    Parameters
    ----------
    tm : pytransform3d.urdf.UrdfTransformManager
        Transform manager that has colliders.

    Returns
    -------
    whitelist : dict
        Whitelists for self-collision detection.
    """
    whitelist = {}
    link_info = LinkInfo(tm)
    for obj in tm.collision_objects:
        link_frame = link_info.link(obj.frame)
        parent_frame = link_info.parent_link(link_frame)
        child_frame = link_info.child_link(link_frame)
        collision_objects_link = link_info.collision_frames_attached_to_link(link_frame)
        collision_objects_parent = link_info.collision_frames_attached_to_link(parent_frame)
        collision_objects_child = link_info.collision_frames_attached_to_link(child_frame)
        whitelist[obj.frame] = (
                collision_objects_link + collision_objects_parent
                + collision_objects_child)
    return whitelist


class LinkInfo:
    """Collect information about links from a UrdfTransformManager.

    Parameters
    ----------
    tm : pytransform3d.urdf.UrdfTransformManager
        Transform manager.
    """
    def __init__(self, tm):
        self.tm = tm
        self.parent_links = {}
        self.child_links = {}
        for child, parent in tm.transforms:
            self.parent_links[child] = parent
            self.child_links[parent] = child
        # HACK uses naming convention from URDF parser to extract link
        #      of a collision object
        self.prog_match_link = re.compile(r"collision:(.*)\/.*")

    def link(self, frame):
        """Finds link frame for a given collision object frame."""
        result = self.prog_match_link.match(frame)
        if result is None:
            warnings.warn(
                f"Couldn't extract link of collision object at frame '{frame}'")
            return None
        else:
            link_frame = result.group(1)
            return link_frame

    def child_link(self, link_frame):
        """Get child of a link."""
        return self._connected_link(self.child_links, link_frame)

    def parent_link(self, link_frame):
        """Get parent of a link."""
        return self._connected_link(self.parent_links, link_frame)

    @staticmethod
    def _connected_link(relation_info, link_frame):
        return relation_info.get(link_frame, None)

    def collision_frames_attached_to_link(self, link_frame):
        """Get all collision frames attached to a link frame."""
        collision_frames = []
        # HACK uses naming convention from URDF parser to extract link
        #      of a collision object
        prog = re.compile(f"collision:{link_frame}" + r"\/.*")
        for node in self.tm.nodes:
            if prog.match(node):
                collision_frames.append(node)
        return collision_frames
