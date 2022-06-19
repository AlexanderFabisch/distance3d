"""
=====
Title
=====
"""
import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.transformations as pr
from grasp_metrics.hands import MiaHand
from distance3d.broad_phase import BoundingVolumeHierarchy
from distance3d.colliders import Box
from distance3d.mpr import mpr_penetration


hand = MiaHand()
joint_angles = hand.get_grasp_angles("lateral", 0.1)
for joint_name in joint_angles:
    hand.tm_.set_joint(joint_name, joint_angles[joint_name])

bvh = BoundingVolumeHierarchy(hand.tm_, hand.get_base_frame())
bvh.fill_tree_with_colliders(
    hand.tm_, make_artists=True, fill_self_collision_whitelists=True)

object_to_grasp = Box(pr.transform_from(R=np.eye(3), p=np.array([0.04, 0.14, 0.03])),
                      np.array([0.03, 0.025, 0.025]))
object_to_grasp.make_artist()
geometry = object_to_grasp.artist_.geometries[0]
aabb = geometry.get_axis_aligned_bounding_box()

fig = pv.figure()
for hand_collider in bvh.get_colliders():
    #hand_collider.artist_.add_artist(fig)
    geometry = hand_collider.artist_.geometries[0]
    points = geometry.sample_points_poisson_disk(500)
    fig.add_geometry(points)

for hand_collider in bvh.get_colliders():
    intersection, depth, direction, position = mpr_penetration(
        hand_collider, object_to_grasp)
    if intersection:
        fig.plot_vector(position, depth * direction, c=(1, 0, 0))

fig.add_geometry(aabb)
fig.show()
