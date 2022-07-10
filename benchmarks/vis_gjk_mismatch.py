import numpy as np
from numpy import array
from distance3d import colliders, gjk
import pytransform3d.visualizer as pv


"""
vertices1=array([[0.04984464, 1.18517908, 1.47875558],
                    [0.0193679 , 1.96458599, 0.09178721],
                    [1.13361017, 0.1989046 , 0.70067395],
                    [0.78892693, 0.77456573, 0.80539834],
                    [1.85983229, 1.88427538, 0.36048079],
                    [1.46102622, 1.3603625 , 1.76779677]])
vertices2=array([[0.58087507, 0.69877449, 0.25128472],
                    [0.81291854, 0.56290825, 0.31027156],
                    [0.37463453, 0.8218163 , 0.90536959],
                    [0.9182387 , 0.22945058, 0.14967991],
                    [0.76467999, 0.32994405, 0.92189327],
                    [0.39975794, 0.17801446, 0.09803848]])
"""

import pickle
with open("testdata.pickle", "rb") as f:
    vertices1, vertices2 = pickle.load(f)

"""
convex1 = colliders.Convex(vertices1)
convex2 = colliders.Convex(vertices2)
print(gjk.gjk_with_simplex(convex1, convex2))
#"""
from distance3d import mesh, mpr
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
