"""
==========================
Self-collisions of a robot
==========================
"""
print(__doc__)
import os
import time
import numpy as np
from pytransform3d.urdf import UrdfTransformManager
import pytransform3d.visualizer as pv
from distance3d import colliders, self_collision


class AnimationCallback:
    def __init__(self, n_frames=100, detect_any=False):
        self.n_frames = n_frames
        self.total_time = 0.0
        self.detect_any = detect_any

    def __call__(self, step, tm, bvh, joint_names):
        if step == 0:
            self.total_time = 0.0

        angle = 2.0 * np.cos(2.0 * np.pi * (step / self.n_frames))
        for joint_name in joint_names:
            tm.set_joint(joint_name, angle)
        bvh.update_collider_poses()

        if self.detect_any:
            start = time.time()
            contacts = self_collision.detect_any(bvh)
            stop = time.time()
            self.total_time += stop - start

            for frame, collider in bvh.colliders_.items():
                geometry = collider.artist_.geometries[0]
                if contacts:
                    geometry.paint_uniform_color((1, 0, 0))
                else:
                    geometry.paint_uniform_color((0, 1, 0))
        else:
            start = time.time()
            contacts = self_collision.detect(bvh)
            stop = time.time()
            self.total_time += stop - start

            for frame, collider in bvh.colliders_.items():
                geometry = collider.artist_.geometries[0]
                if contacts[frame]:
                    geometry.paint_uniform_color((1, 0, 0))
                else:
                    geometry.paint_uniform_color((0, 1, 0))

        if step == self.n_frames - 1:
            print(f"Total time: {self.total_time}")

        return bvh.get_artists()


BASE_DIR = "test/data/"
data_dir = BASE_DIR
search_path = ".."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "distance3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)

tm = UrdfTransformManager()
filename = os.path.join(data_dir, "simple_mechanism.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=data_dir)

bvh = colliders.BoundingVolumeHierarchy(tm, "simple_mechanism")
bvh.fill_tree_with_colliders(
    tm, make_artists=True, fill_self_collision_whitelists=True)

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.2)
for artist in bvh.get_artists():
    artist.add_artist(fig)
fig.view_init()

n_frames = 500
animation_callback = AnimationCallback(n_frames=n_frames, detect_any=False)
if "__file__" in globals():
    fig.animate(animation_callback, n_frames, loop=True,
                fargs=(tm, bvh, ["joint1", "joint2", "joint3"]))
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
