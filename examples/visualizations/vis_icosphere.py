"""
===================
Visualize Icosphere
===================
"""
print(__doc__)
import aabbtree
import numpy as np
import open3d as o3d
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pt
from distance3d import mesh, visualization, benchmark, mpr, colliders, geometry, io


timer = benchmark.Timer()
timer.start("make_tetrahedral_icosphere")
vertices1_in_mesh1, tetrahedra1 = mesh.make_tetrahedral_icosphere(0.1 * np.ones(3), 0.15, 2)
vertices2_in_mesh2, tetrahedra2 = mesh.make_tetrahedral_icosphere(0.25 * np.ones(3), 0.15, 2)
print(timer.stop("make_tetrahedral_icosphere"))
#vertices2_in_mesh2, tetrahedra2 = io.load_tetrahedral_mesh("test/data/insole.vtk")
mesh12origin = np.eye(4)
mesh22origin = np.eye(4)

# We transform vertices of mesh1 to mesh2 frame to be able to reuse the AABB
# tree of mesh2.
origin2mesh2 = pt.invert_transform(mesh22origin)
mesh12mesh2 = pt.concat(mesh12origin, origin2mesh2)
vertices1_in_mesh2 = pt.transform(mesh12mesh2, pt.vectors_to_points(vertices1_in_mesh1))[:, :3]

# TODO we can also use the pressure functions for this. does it work with concave objects? which one is faster?
c1 = colliders.ConvexHullVertices(vertices1_in_mesh2)
c2 = colliders.ConvexHullVertices(vertices2_in_mesh2)
timer.start("mpr_penetration")
intersection, depth, normal, contact_point = mpr.mpr_penetration(c1, c2)
assert intersection
print(timer.stop("mpr_penetration"))

# TODO refactor

def points_to_plane_signed(points, plane_point, plane_normal):
    return np.dot(points - plane_point.reshape(1, -1), plane_normal)

def intersecting_tetrahedra(vertices, tetrahedra, contact_point, normal):
    d = points_to_plane_signed(vertices, contact_point, normal)[tetrahedra]
    mins = np.min(d, axis=1)
    maxs = np.max(d, axis=1)
    return np.sign(mins) != np.sign(maxs)

from distance3d.distance import line_segment_to_plane
def contact_plane_projection(plane_point, plane_normal, tetrahedron_points):
    d = np.sign(points_to_plane_signed(tetrahedron_points, plane_point, plane_normal))
    neg = np.where(d < 0)[0]
    pos = np.where(d >= 0)[0]
    triangle_points = []
    for n in neg:
        for p in pos:
            triangle_points.append(
                line_segment_to_plane(
                    tetrahedron_points[n], tetrahedron_points[p],
                    plane_point, plane_normal)[2])
    assert len(triangle_points) >= 3, f"{triangle_points}"
    triangle_points = np.asarray(triangle_points)
    return triangle_points

def polygon_area(points):
    if len(points) == 3:
        return 0.5 * np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
    else:
        assert len(points) == 4
        return 0.5 * (
            np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0]))
            + np.linalg.norm(np.cross(points[1] - points[3], points[2] - points[3])))

# The pressure function assigns to each point in the interior of the object
# a nonnegative real number representing the pressure at that point, which
# is an intuitive notion of how much resistance a foreign body protruding
# into the object would experience at that point.
# Source: https://www.ekzhang.com/assets/pdf/Hydroelastics.pdf
# TODO general distance to surface
potentials1 = np.zeros(len(vertices1_in_mesh2))
potentials1[-1] = 0.15
potentials2 = np.zeros(len(vertices2_in_mesh2))
potentials2[-1] = 0.15

# When two objects with pressure functions p1(*), p2(*) intersect, there is
# a surface S inside the space of intersection at which the values of p1 and
# p2 are equal. After identifying this surface, we then define the total force
# exerted by one object on another [..].
# Source: https://www.ekzhang.com/assets/pdf/Hydroelastics.pdf

timer.start("prescreening")
# Initial check of bounding boxes of tetrahedra
aabbs1 = mesh.tetrahedral_mesh_aabbs(vertices1_in_mesh2, tetrahedra1)
aabbs2 = mesh.tetrahedral_mesh_aabbs(vertices2_in_mesh2, tetrahedra2)
broad_overlapping_indices1 = []
broad_overlapping_indices2 = []
tree2 = aabbtree.AABBTree()
for j, aabb in enumerate(aabbs2):
    tree2.add(aabbtree.AABB(aabb), j)
for i, aabb in enumerate(aabbs1):
    new_indices2 = tree2.overlap_values(aabbtree.AABB(aabb))
    broad_overlapping_indices2.extend(new_indices2)
    broad_overlapping_indices1.extend([i] * len(new_indices2))

# Check if the tetrahedra actually intersect the contact plane
broad_overlapping_indices1 = np.asarray(broad_overlapping_indices1, dtype=int)
broad_overlapping_indices2 = np.asarray(broad_overlapping_indices2, dtype=int)
candidates1 = tetrahedra1[broad_overlapping_indices1]
candidates2 = tetrahedra2[broad_overlapping_indices2]
keep1 = intersecting_tetrahedra(vertices1_in_mesh2, candidates1, contact_point, normal)
keep2 = intersecting_tetrahedra(vertices2_in_mesh2, candidates2, contact_point, normal)
keep = np.logical_and(keep1, keep2)
broad_overlapping_indices1 = broad_overlapping_indices1[keep]
broad_overlapping_indices2 = broad_overlapping_indices2[keep]
print(timer.stop("prescreening"))

# TODO the paper suggests computing surface area, com of the contact surface and p(com)
# How do we compute p(com)?
timer.start("compute pressures")
pressures1 = dict()
pressures2 = dict()
last1 = -1
for i in range(len(broad_overlapping_indices1)):
    idx1 = broad_overlapping_indices1[i]
    idx2 = broad_overlapping_indices2[i]

    if idx1 != last1:
        tetra1 = vertices1_in_mesh2[tetrahedra1[idx1]]
        t1 = colliders.ConvexHullVertices(tetra1)
        poly1 = contact_plane_projection(contact_point, normal, tetra1)
        area1 = polygon_area(poly1)
        p1 = np.mean(poly1, axis=0)
        c1 = geometry.barycentric_coordinates_tetrahedron(p1, tetra1)
        pressure1 = c1.dot(potentials1[tetrahedra1[idx1]])

    # TODO tetra-tetra intersection to compute triangle, something with halfplanes?
    # instead we try to compute surface for each object individually
    tetra2 = vertices2_in_mesh2[tetrahedra2[idx2]]
    t2 = colliders.ConvexHullVertices(tetra2)
    if mpr.mpr_intersection(t1, t2):
        # TODO compute triangle projection on contact surface, compute
        # area and use it as a weight for the pressure in integral
        poly2 = contact_plane_projection(contact_point, normal, tetra2)
        area2 = polygon_area(poly2)
        p2 = np.mean(poly2, axis=0)
        c2 = geometry.barycentric_coordinates_tetrahedron(p2, tetra2)
        pressure2 = c2.dot(potentials2[tetrahedra2[idx2]])

        pressures1[idx1] = (area1 * pressure1, p1, poly1)
        pressures2[idx2] = (area2 * pressure2, p2, poly2)
print(timer.stop("compute pressures"))

print(f"force 1: {sum([p[0] for p in pressures1.values()])}")
print(f"force 2: {sum([p[0] for p in pressures2.values()])}")

# TODO compute torque
timer.start("center_of_mass_tetrahedral_mesh")
com1 = mesh.center_of_mass_tetrahedral_mesh(mesh12origin, vertices1_in_mesh2, tetrahedra1)
print(timer.stop("center_of_mass_tetrahedral_mesh"))

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
tetra_mesh1 = visualization.TetraMesh(mesh22origin, vertices1_in_mesh2, tetrahedra1)
tetra_mesh1.add_artist(fig)
tetra_mesh2 = visualization.TetraMesh(mesh22origin, vertices2_in_mesh2, tetrahedra2)
tetra_mesh2.add_artist(fig)

max_pressure1 = max([p[0] for p in pressures1.values()])
P = []
c = []
for pressure, point, _ in pressures1.values():
    P.append(point)
    c.append((pressure / max_pressure1, 0, 0))
fig.scatter(P, s=0.003, c=c)
max_pressure2 = max([p[0] for p in pressures2.values()])
P = []
c = []
for pressure, point, _ in pressures2.values():
    P.append(point)
    c.append((0, pressure / max_pressure2, 0))
fig.scatter(P, s=0.003, c=c)

for _, _, points in pressures1.values():
    if len(points) == 3:
        triangles = np.array([[0, 1, 2], [2, 1, 0]], dtype=int)
    else:
        assert len(points) == 4
        triangles = np.array([[0, 1, 2], [2, 1, 0], [3, 1, 2], [2, 1, 3]], dtype=int)
    contact_surface_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector3iVector(triangles)
    )
    fig.add_geometry(contact_surface_mesh)

fig.view_init()

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
