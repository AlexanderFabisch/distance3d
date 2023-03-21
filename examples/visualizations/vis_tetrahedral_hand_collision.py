"""
============================================
Physical Simulation of Bouncing Ball on Hand
============================================

A rubber ball bouncing on a hand.
"""

import os
from pytransform3d.urdf import UrdfTransformManager

import numpy as np
import pytransform3d.visualizer as pv
from distance3d import hydroelastic_contact

from distance3d.hydroelastic_contact._broad_phase import HydroelasticBoundingVolumeHierarchy

dt = 0.001
g = np.array([9.81, 0, 0])

fig = pv.figure()

class PhysicsObject:
    def __init__(self, rigid_body, artist, mass, velocity, fixed=False):
        self.rigid_body = rigid_body
        self.fixed = fixed

        self.mass = mass

        self.acceleration = np.zeros(3, dtype=float)
        self.velocity = velocity
        self.artist_ = artist
        self.forces = []

    def step(self):
        if self.fixed:
            self.forces = []
            return

        gravity_force = g * self.mass
        self.forces.append(gravity_force)

        self.acceleration = np.add.reduce(self.forces) / self.mass
        self.velocity += dt * self.acceleration
        self.rigid_body.body2origin_[:3, 3] += self.velocity * dt
        self.artist_.set_data(self.rigid_body.body2origin_, self.rigid_body.vertices_,
                              self.rigid_body.tetrahedra_)

        self.forces = []

class AnimationCallback:
    def __init__(self, p_objects):
        self.p_objects = p_objects
        self.artists = []
        for p_object in p_objects:
            self.artists.append(p_object.artist_)

    def __call__(self, step):
        global xs, ys
        for i in range(len(self.p_objects)):
            for j in range(i + 1, len(self.p_objects)):
                intersection, wrench12, wrench21, details = hydroelastic_contact.contact_forces(
                    self.p_objects[i].rigid_body, self.p_objects[j].rigid_body, return_details=True)

                if intersection:
                    self.p_objects[i].forces.append(wrench21[:3])
                    self.p_objects[j].forces.append(wrench12[:3])

            self.p_objects[i].step()

            sum_vel = 0
            sum_acc = 0
            for p_object in self.p_objects:
                o_vel = np.sqrt(p_object.velocity[0] * p_object.velocity[0]
                                + p_object.velocity[1] * p_object.velocity[1]
                                + p_object.velocity[2] * p_object.velocity[2])
                o_acc = np.sqrt(p_object.acceleration[0] * p_object.acceleration[0]
                                + p_object.acceleration[1] * p_object.acceleration[1]
                                + p_object.acceleration[2] * p_object.acceleration[2])
                sum_vel += o_vel
                sum_acc += o_acc



        return self.artists


GPa = 100000000

BASE_DIR = "test/data/"
data_dir = BASE_DIR
search_path = ".."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "distance3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)

tm = UrdfTransformManager()
filename = os.path.join(data_dir, "mia_hand_description/urdf/mia_hand.urdf")
with open(filename, "r") as f: # Run: cd test/data/ && git clone git@github.com:aprilprojecteu/mia_hand_description.git
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=data_dir)

robot_bvh = HydroelasticBoundingVolumeHierarchy(tm, "mia_hand")
robot_bvh.fill_tree_with_colliders(tm, make_artists=True)

for rb in robot_bvh.get_colliders():
    rb.youngs_modulus = 100 * GPa

rb = hydroelastic_contact.RigidBody.make_sphere(np.array([-0.1, 0.01, 0.0]), 0.02, 1)
rb.youngs_modulus = 100 * GPa

p_objects = []
for collider in robot_bvh.get_colliders():
    p_objects.append(PhysicsObject(collider, collider.artist_, 100, np.array([0.0, 0.0, 0.0]), True))
    collider.artist_.add_artist(fig)

p_objects.append(PhysicsObject(rb, rb.artist_, 100, np.array([0.0, 0.0, 0.0]), False))
rb.artist_.add_artist(fig)

fig.view_init()

if "__file__" in globals():
    fig.animate(AnimationCallback(p_objects), 500, loop=True)
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
