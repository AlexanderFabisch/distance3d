import numpy as np
from ..utils import transform_points, transform_directions


class ContactSurface:
    def __init__(
            self, frame2world, intersection, contact_planes, contact_polygons,
            contact_polygon_triangles, intersecting_tetrahedra1,
            intersecting_tetrahedra2):
        self.frame2world = frame2world
        self.intersection = intersection
        self.contact_planes = contact_planes
        self.contact_polygons = contact_polygons
        self.contact_polygon_triangles = contact_polygon_triangles
        self.intersecting_tetrahedra1 = intersecting_tetrahedra1
        self.intersecting_tetrahedra2 = intersecting_tetrahedra2

        self.contact_areas = None
        self.contact_coms = None
        self.contact_forces = None

    def add_polygon_info(self, contact_areas, contact_coms, contact_forces):
        self.contact_areas = contact_areas
        self.contact_coms = contact_coms
        self.contact_forces = contact_forces

    def _transform_to_world(self, frame2world):
        self.contact_polygons = [transform_points(frame2world, contact_polygon)
                                 for contact_polygon in self.contact_polygons]
        plane_points = self.contact_planes[:, :3] * self.contact_planes[:, 3, np.newaxis]
        plane_points = transform_points(frame2world, plane_points)
        plane_normals = transform_directions(
            frame2world, self.contact_planes[:, :3])
        plane_distances = np.sum(plane_points * plane_normals, axis=1)
        self.contact_planes = np.hstack((plane_normals, plane_distances.reshape(-1, 1)))
        self.contact_coms = transform_points(
            frame2world, np.asarray(self.contact_coms))
        self.contact_forces = transform_directions(
            frame2world, np.asarray(self.contact_forces))

    def make_details(self, tetrahedra_points1, tetrahedra_points2):
        self._transform_to_world(self.frame2world)
        n_intersections = len(self.intersecting_tetrahedra1)
        intersecting_tetrahedra1 = tetrahedra_points1[np.asarray(self.intersecting_tetrahedra1, dtype=int)]
        intersecting_tetrahedra1 = transform_points(
            self.frame2world, intersecting_tetrahedra1.reshape(n_intersections * 4, 3)
        ).reshape(n_intersections, 4, 3)
        intersecting_tetrahedra2 = tetrahedra_points2[np.asarray(self.intersecting_tetrahedra2, dtype=int)]
        intersecting_tetrahedra2 = transform_points(
            self.frame2world, intersecting_tetrahedra2.reshape(n_intersections * 4, 3)
        ).reshape(n_intersections, 4, 3)

        self.contact_areas = np.asarray(self.contact_areas)
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
