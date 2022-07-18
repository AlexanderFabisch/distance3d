"""
===================
Visualize Icosphere
===================
"""
print(__doc__)
import aabbtree
import numpy as np
import pytransform3d.visualizer as pv
from distance3d import mesh, visualization, benchmark, mpr, colliders, geometry, io


timer = benchmark.Timer()
timer.start("make_tetrahedral_icosphere")
vertices1, tetrahedra1 = mesh.make_tetrahedral_icosphere(0.1 * np.ones(3), 0.15, 2)
vertices2, tetrahedra2 = mesh.make_tetrahedral_icosphere(0.25 * np.ones(3), 0.15, 2)
print(timer.stop("make_tetrahedral_icosphere"))
#vertices2, tetrahedra2 = io.load_tetrahedral_mesh("test/data/insole.vtk")
mesh2origin1 = np.eye(4)
mesh2origin2 = np.eye(4)

# TODO we can also use the pressure functions for this. does it work with concave objects? which one is faster?
# TODO mesh2origin
c1 = colliders.ConvexHullVertices(vertices1)
c2 = colliders.ConvexHullVertices(vertices2)
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
    return np.where(np.sign(mins) != np.sign(maxs))[0]

from distance3d.distance import point_to_plane
def point_in_plane(plane_point, plane_normal, tetrahedron_points):  # TODO triangle projection?
    return np.mean(np.array([point_to_plane(p, plane_point, plane_normal)[1]
                             for p in tetrahedron_points]), axis=0)

# TODO general distance to surface
potentials1 = np.zeros(len(vertices1))
potentials1[-1] = 0.15
potentials2 = np.zeros(len(vertices1))
potentials2[-1] = 0.15

timer.start("prescreening")
# TODO mesh2origin
indices1 = intersecting_tetrahedra(vertices1, tetrahedra1, contact_point, normal)
indices2 = intersecting_tetrahedra(vertices2, tetrahedra2, contact_point, normal)

aabbs1 = mesh.tetrahedral_mesh_aabbs(mesh2origin1, vertices1, tetrahedra1[indices1])
aabbs2 = mesh.tetrahedral_mesh_aabbs(mesh2origin2, vertices2, tetrahedra2[indices2])
broad_overlapping_pairs = dict()
tree2 = aabbtree.AABBTree()
for j, aabb in zip(indices2, aabbs2):
    tree2.add(aabbtree.AABB(aabb), j)
for i, aabb in zip(indices1, aabbs1):
    overlapping_indices2 = tree2.overlap_values(aabbtree.AABB(aabb))
    broad_overlapping_pairs[i] = overlapping_indices2
print(timer.stop("prescreening"))

# TODO the paper suggests computing surface area, com of the contact surface and p(com)
# How do we compute p(com)?
timer.start("compute pressures")
pressures1 = dict()
pressures2 = dict()
for idx1 in broad_overlapping_pairs.keys():
    tetra1 = vertices1[tetrahedra1[idx1]]
    t1 = colliders.ConvexHullVertices(tetra1)
    p1 = point_in_plane(contact_point, normal, tetra1)
    c1 = geometry.barycentric_coordinates_tetrahedron(p1, tetra1)
    pressure1 = c1.dot(potentials1[tetrahedra1[idx1]])
    for idx2 in broad_overlapping_pairs[idx1]:
        # TODO tetra-tetra intersection, something with halfplanes?
        tetra2 = vertices2[tetrahedra2[idx2]]
        t2 = colliders.ConvexHullVertices(tetra2)
        if mpr.mpr_intersection(t1, t2):
            # TODO compute triangle projection on contact surface, compute
            # area and use it as a weight for the pressure in integral
            p2 = point_in_plane(contact_point, normal, tetra2)
            c2 = geometry.barycentric_coordinates_tetrahedron(p2, tetra2)
            pressure2 = c2.dot(potentials2[tetrahedra2[idx2]])

            v1, _ = pressures1.get(idx1, [0.0, p1])
            pressures1[idx1] = [v1 + pressure1, p1]
            v2, _ = pressures2.get(idx2, [0.0, p2])
            pressures2[idx2] = [v2 + pressure2, p2]
print(timer.stop("compute pressures"))

print(f"force 1: {sum([p[0] for p in pressures1.values()])}")
print(f"force 2: {sum([p[0] for p in pressures2.values()])}")

# TODO compute torque
timer.start("center_of_mass_tetrahedral_mesh")
com1 = mesh.center_of_mass_tetrahedral_mesh(mesh2origin1, vertices1, tetrahedra1)
print(timer.stop("center_of_mass_tetrahedral_mesh"))

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
tetra_mesh1 = visualization.TetraMesh(mesh2origin1, vertices1, tetrahedra1)
tetra_mesh1.add_artist(fig)
tetra_mesh2 = visualization.TetraMesh(mesh2origin2, vertices2, tetrahedra2)
tetra_mesh2.add_artist(fig)
#fig.plot_plane(normal=normal, point_in_plane=contact_point)

max_pressure1 = max([p[0] for p in pressures1.values()])
P = []
c = []
for pressure, point in pressures1.values():
    P.append(point)
    c.append((pressure / max_pressure1, 0, 0))
fig.scatter(P, s=0.003, c=c)
max_pressure2 = max([p[0] for p in pressures2.values()])
P = []
c = []
for pressure, point in pressures2.values():
    P.append(point)
    c.append((0, pressure / max_pressure2, 0))
fig.scatter(P, s=0.003, c=c)

fig.view_init()

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
