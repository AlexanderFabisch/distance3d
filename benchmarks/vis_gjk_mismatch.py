import numpy as np
from distance3d import colliders, gjk
from distance3d import mesh, mpr
import pytransform3d.visualizer as pv
import pickle


with open("testdata.pickle", "rb") as f:
    vertices1, vertices2 = pickle.load(f)

convex1 = colliders.ConvexHullVertices(vertices1)
convex2 = colliders.ConvexHullVertices(vertices2)
intersection_libccd = gjk.gjk_intersection_libccd(convex1, convex2)
print(f"{intersection_libccd=}")
intersection_jolt = gjk.gjk_intersection_jolt(convex1, convex2)
print(f"{intersection_jolt=}")
intersection_mpr = mpr.mpr_intersection(convex1, convex2)
print(f"{intersection_mpr=}")
dist_original, cp1_original, cp2_original, _ = gjk.gjk_distance_original(convex1, convex2)
print(f"{dist_original=}, {cp1_original=}, {cp2_original=}")
dist_jolt, cp1_jolt, cp2_jolt, _ = gjk.gjk_distance_jolt(convex1, convex2)
print(f"{dist_jolt=}, {cp1_jolt=}, {cp2_jolt=}")
fig = pv.figure()
triangles1 = mesh.make_convex_mesh(vertices1)
convex1 = colliders.MeshGraph(np.eye(4), vertices1, triangles1)
triangles2 = mesh.make_convex_mesh(vertices2)
convex2 = colliders.MeshGraph(np.eye(4), vertices2, triangles2)
convex1.make_artist((1, 0, 0))
convex2.make_artist((0, 1, 0))
convex1.artist_.add_artist(fig)
convex2.artist_.add_artist(fig)
fig.show()
