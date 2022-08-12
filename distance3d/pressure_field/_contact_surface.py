import numpy as np
from ..utils import transform_points, transform_directions


class ContactSurface:
    """Contact surface of tetrahedral meshes.

    Parameters
    ----------
    frame2world : array, shape (4, 4)
        Transforms vertices, points, and directions to world frame.

    intersection : bool
        Do both tetrahedral meshes intersect?

    contact_planes : array, shape (n_intersections, 4)
        Contact planes of intersection pairs in Hesse normal form.

    contact_polygons : list
        Contact polygons of intersection pairs.

    intersecting_tetrahedra1 : list
        Intersecting tetrahedron indices of first mesh.

    intersecting_tetrahedra2 : list
        Intersecting tetrahedron indices of second mesh.
    """
    def __init__(
            self, frame2world, intersection, contact_planes, contact_polygons,
            intersecting_tetrahedra1, intersecting_tetrahedra2):
        self.frame2world = frame2world
        self.intersection = intersection
        self.contact_planes = contact_planes
        self.contact_polygons = contact_polygons
        self.intersecting_tetrahedra1 = intersecting_tetrahedra1
        self.intersecting_tetrahedra2 = intersecting_tetrahedra2

        self.contact_areas = None
        self.contact_coms = None
        self.contact_forces = None
        self.contact_polygon_triangles = None

    def add_polygon_info(self, contact_areas, contact_coms, contact_forces, contact_polygon_triangles):
        """Add more info on contact polygons.

        Parameters
        ----------
        contact_areas : array, shape (n_intersections,)
            Areas of contact polygons.

        contact_coms : array, shape (n_intersections, 3)
            Center of mass of contact polygons.

        contact_forces : array, shape (n_intersections, 3)
            Contact forces of contact polygons.

        contact_polygon_triangles : list
            Vertex indices of triangles of contact polygons.
        """
        self.contact_areas = contact_areas
        self.contact_coms = contact_coms
        self.contact_forces = contact_forces
        self.contact_polygon_triangles = contact_polygon_triangles

    def make_details(self, tetrahedra_points1, tetrahedra_points2):
        """Summarize details of contact points in world frame.

        Parameters
        ----------
        tetrahedra_points1 : array, shape (n_tetrahedra, 4, 3)
            Points that form the first tetrahedra.

        tetrahedra_points2 : array, shape (n_tetrahedra, 4, 3)
            Points that form the second tetrahedra.

        Returns
        -------
        details : dict
            Detailed description of contact surface and pressure field.
        """
        intersecting_tetrahedra1, intersecting_tetrahedra2 = self._transform_to_world(
            self.frame2world, tetrahedra_points1, tetrahedra_points2)

        pressures = np.linalg.norm(self.contact_forces, axis=1) / self.contact_areas
        contact_point = np.sum(
            self.contact_coms * self.contact_areas[:, np.newaxis],
            axis=0) / sum(self.contact_areas)

        details = {
            "contact_polygons": self.contact_polygons,
            "contact_polygon_triangles": self.contact_polygon_triangles,
            "contact_planes": self.contact_planes,
            "intersecting_tetrahedra1": intersecting_tetrahedra1,
            "intersecting_tetrahedra2": intersecting_tetrahedra2,
            "contact_coms": self.contact_coms,
            "contact_forces": self.contact_forces,
            "contact_areas": self.contact_areas,
            "pressures": pressures,
            "contact_point": contact_point
        }
        return details

    def _transform_to_world(self, frame2world, tetrahedra_points1, tetrahedra_points2):
        self.contact_polygons = [transform_points(frame2world, contact_polygon)
                                 for contact_polygon in self.contact_polygons]
        plane_points = self.contact_planes[:, :3] * self.contact_planes[:, 3, np.newaxis]
        plane_points = transform_points(frame2world, plane_points)
        plane_normals = transform_directions(
            frame2world, self.contact_planes[:, :3])
        plane_distances = np.sum(plane_points * plane_normals, axis=1)
        self.contact_planes = np.hstack((plane_normals, plane_distances.reshape(-1, 1)))
        self.contact_coms = transform_points(
            frame2world, self.contact_coms)
        self.contact_forces = transform_directions(
            frame2world, self.contact_forces)
        n_intersections = len(self.intersecting_tetrahedra1)
        intersecting_tetrahedra1 = tetrahedra_points1[np.asarray(self.intersecting_tetrahedra1, dtype=int)]
        intersecting_tetrahedra1 = transform_points(
            self.frame2world, intersecting_tetrahedra1.reshape(n_intersections * 4, 3)
        ).reshape(n_intersections, 4, 3)
        intersecting_tetrahedra2 = tetrahedra_points2[np.asarray(self.intersecting_tetrahedra2, dtype=int)]
        intersecting_tetrahedra2 = transform_points(
            self.frame2world, intersecting_tetrahedra2.reshape(n_intersections * 4, 3)
        ).reshape(n_intersections, 4, 3)
        return intersecting_tetrahedra1, intersecting_tetrahedra2
