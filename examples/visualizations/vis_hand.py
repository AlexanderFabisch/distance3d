import os
from pytransform3d.urdf import UrdfTransformManager

import distance3d.colliders
from distance3d import broad_phase
import numpy as np
import pytransform3d.visualizer as pv
from distance3d import visualization, hydroelastic_contact
import matplotlib.pyplot as plt
import time

from distance3d.hydroelastic_contact._tetra_mesh_creation import make_tetrahedral_sphere

# https://www.youtube.com/watch?v=h4OVbqfS6f4

dt = 0.001
g = np.array([9.81, 0, 0])
GPa = 100000000

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)

p_objects = []
plot = plt.figure()
ax = plot.add_subplot(1, 1, 1)
xs = []
vels = []
accs = []
# Format plot
plt.title('Velocity over Time')
plt.ylabel('Velocity')


class PhysicsObject:
    def __init__(self, rigid_body, artist, mass, velocity, fixed=False):
        self.rigid_body = rigid_body
        self.fixed = fixed

        self.mass = mass

        self.acceleration = np.zeros(3, dtype=float)
        self.velocity = velocity
        self.artist = artist
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
        self.artist.set_data(self.rigid_body.body2origin_, self.rigid_body.vertices_,
                             self.rigid_body.tetrahedra_)

        self.forces = []


class AnimationCallback:
    def __init__(self, p_objects):
        self.p_objects = p_objects
        self.artists = []
        for p_object in p_objects:
            self.artists.append(p_object.artist)

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

            xs.append(time.time())
            vels.append(sum_vel)
            accs.append(sum_acc / 100)

        return self.artists


BASE_DIR = "test/data/"
data_dir = BASE_DIR
search_path = ".."
while (not os.path.exists(data_dir) and
       os.path.dirname(search_path) != "distance3d"):
    search_path = os.path.join(search_path, "..")
    data_dir = os.path.join(search_path, BASE_DIR)

tm = UrdfTransformManager()
filename = os.path.join(data_dir, "mia_hand_description/urdf/mia_hand.urdf")
with open(filename, "r") as f:
    robot_urdf = f.read()
    tm.load_urdf(robot_urdf, mesh_path=data_dir)

robot_bvh = broad_phase.BoundingVolumeHierarchy(tm, "mia_hand")
robot_bvh.fill_tree_with_colliders(tm, make_artists=True)


def getSphereRB(sphere):
    rb = hydroelastic_contact.RigidBody.make_sphere(sphere.c, sphere.radius, 1)
    return rb


def getBoxRB(box):
    rb = hydroelastic_contact.RigidBody.make_box(box.box2origin, box.size)
    return rb


def getCylinderRB(cylinder):
    rb = hydroelastic_contact.RigidBody.make_cylinder(cylinder.cylinder2origin, cylinder.radius, cylinder.length)
    return rb

def make_object(rigid_body, mass, acc, fixed):
    artist = visualization.RigidBodyTetrahedralMesh(
        rigid_body.body2origin_, rigid_body.vertices_, rigid_body.tetrahedra_)
    artist.add_artist(fig)

    return PhysicsObject(rigid_body, artist, mass, acc, fixed)

for collider in robot_bvh.get_colliders():
    switchDict = {distance3d.colliders.Box: getBoxRB,
                  distance3d.colliders.Sphere: getSphereRB,
                  distance3d.colliders.Cylinder: getCylinderRB}
    rb = switchDict[type(collider)](collider)

    rb.youngs_modulus = 100 * GPa

    object = make_object(rb, 1, np.array([0.0, 0.0, 0.0]), True)
    p_objects.append(object)

rb = hydroelastic_contact.RigidBody.make_sphere(np.array([-0.1, 0.01, 0.0]), 0.02, 1)
rb.youngs_modulus = 100 * GPa
p_object1 = make_object(rb, 1, np.array([-0.1, 0.0, 0.0]), False)
p_objects.append(p_object1)

fig.view_init()

if "__file__" in globals():
    fig.animate(AnimationCallback(p_objects), 500, loop=True)
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")

ax.plot(xs, vels)
ax.plot(xs, accs)
plot.show()
