import pprint
from distance3d import colliders, gjk, mpr
import pytransform3d.visualizer as pv
import pickle


collider_classes = {
    "sphere": colliders.Sphere,
    "ellipsoid": colliders.Ellipsoid,
    "capsule": colliders.Capsule,
    "disk": colliders.Disk,
    "ellipse": colliders.Ellipse,
    "cone": colliders.Cone,
    "cylinder": colliders.Cylinder,
    "box": colliders.Box,
    "mesh": colliders.MeshGraph,
}


with open("testdata.pickle", "rb") as f:
    collider_name1, args1, collider_name2, args2 = pickle.load(f)
print(collider_name1)
pprint.pprint(args1)
print(collider_name2)
pprint.pprint(args2)

collider1 = collider_classes[collider_name1](*args1)
collider2 = collider_classes[collider_name2](*args2)
intersection_libccd = gjk.gjk_intersection_libccd(collider1, collider2)
print(f"{intersection_libccd=}")
intersection_jolt = gjk.gjk_intersection_jolt(collider1, collider2)
print(f"{intersection_jolt=}")
intersection_mpr = mpr.mpr_intersection(collider1, collider2)
print(f"{intersection_mpr=}")
dist_original, cp1_original, cp2_original, _ = gjk.gjk_distance_original(collider1, collider2)
print(f"{dist_original=}, {cp1_original=}, {cp2_original=}")
dist_jolt, cp1_jolt, cp2_jolt, _ = gjk.gjk_distance_jolt(collider1, collider2)
print(f"{dist_jolt=}, {cp1_jolt=}, {cp2_jolt=}")

fig = pv.figure()
collider1.make_artist((1, 0, 0))
collider2.make_artist((0, 1, 0))
collider1.artist_.add_artist(fig)
collider2.artist_.add_artist(fig)
fig.show()
