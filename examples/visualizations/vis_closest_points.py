"""
=======================
Closest Points with GJK
=======================
"""
print(__doc__)

import time
import numpy as np
import pytransform3d.visualizer as pv
import pytransform3d.rotations as pr
import pytransform3d.transformations as pt
from distance3d import random, colliders, gjk


class AnimationCallback:
    def __init__(self, n_frames=100, verbose=0):
        self.n_frames = n_frames
        self.verbose = verbose
        self.total_time = 0.0

    def __call__(self, step, obj, env, connections):
        if step == 0:
            self.total_time = 0.0

        angle = np.pi * np.cos(2.0 * np.pi * (step / self.n_frames))
        obj2world = pt.transform_from(
            R=pr.active_matrix_from_angle(1, angle), p=np.zeros(3))
        obj.update_pose(obj2world)

        total_time = 0.0
        start = time.time()
        for collider, connection in zip(env, connections):
            _, cp1, cp2, _ = gjk.gjk_distance_original(obj, collider)
            connection.set_data(np.row_stack((cp1, cp2)))
        stop = time.time()
        total_time += stop - start

        self.total_time += total_time

        if step == self.n_frames - 1:
            print(f"Total time: {self.total_time}")

        return [obj.artist_] + connections


random_state = np.random.RandomState(5)

fig = pv.figure()

env = []
connections = []
for _ in range(10):
    cone2origin, radius, height = random.rand_cone(
        random_state, center_scale=5.0, min_height=0.5, min_radius=0.2)
    cone = colliders.Cone(
        cone2origin=cone2origin, radius=radius, height=height)
    env.append(cone)
    color = random_state.rand(3)
    cone.make_artist(color)
    cone.artist_.add_artist(fig)

    connection = pv.Line3D(np.zeros((2, 3)), c=color)
    connection.add_artist(fig)
    connections.append(connection)
for _ in range(10):
    center, radius = random.rand_sphere(random_state, center_scale=5.0)
    sphere = colliders.Sphere(center=center, radius=radius)
    env.append(sphere)
    color = random_state.rand(3)
    sphere.make_artist(color)
    sphere.artist_.add_artist(fig)

    connection = pv.Line3D(np.zeros((2, 3)), c=color)
    connection.add_artist(fig)
    connections.append(connection)
for _ in range(10):
    ellipsoid2origin, radii = random.rand_ellipsoid(
        random_state, center_scale=5.0, min_radius=0.2)
    ellipsoid = colliders.Ellipsoid(ellipsoid2origin, radii)
    env.append(ellipsoid)
    color = random_state.rand(3)
    ellipsoid.make_artist(color)
    ellipsoid.artist_.add_artist(fig)

    connection = pv.Line3D(np.zeros((2, 3)), c=color)
    connection.add_artist(fig)
    connections.append(connection)

capsule2origin, radius, height = random.rand_capsule(
    random_state, center_scale=0.0, radius_scale=1.0, height_scale=10.0)
obj = colliders.Capsule(capsule2origin, radius, height)
obj.make_artist()
obj.artist_.add_artist(fig)

fig.view_init()
n_frames = 200
animation_callback = AnimationCallback(n_frames=n_frames, verbose=0)
if "__file__" in globals():
    fig.animate(animation_callback, n_frames, loop=True,
                fargs=(obj, env, connections))
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
