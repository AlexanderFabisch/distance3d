"""
==========================================================
Physical Simulation of Soft Balls Bouncing in a Wooden Box
==========================================================
"""

print(__doc__)
import numpy as np
import pytransform3d.visualizer as pv
from distance3d import visualization, hydroelastic_contact
import matplotlib.pyplot as plt
import time

dt = 0.001
g = np.array([0, -9.81, 0])

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
        self.artists_ = []
        for p_object in p_objects:
            self.artists_.append(p_object.artist_)

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

        return self.artists_


def make_object(rigid_body, mass, acc, fixed):
    artist = visualization.RigidBodyTetrahedralMesh(
        rigid_body.body2origin_, rigid_body.vertices_, rigid_body.tetrahedra_)
    artist.add_artist(fig)

    return PhysicsObject(rigid_body, artist, mass, acc, fixed)


cube2origin = np.eye(4)
rb_box1 = hydroelastic_contact.RigidBody.make_box(cube2origin, np.array([1.5, 0.5, 1.5]))

cube2origin2 = np.eye(4)
cube2origin2[:3, 3] = np.array([0.7, 1.0, 0.0])
rb_box2 = hydroelastic_contact.RigidBody.make_box(cube2origin2, np.array([0.2, 2.0, 1.5]))

cube2origin3 = np.eye(4)
cube2origin3[:3, 3] = np.array([-0.7, 1.0, 0.0])
rb_box3 = hydroelastic_contact.RigidBody.make_box(cube2origin3, np.array([0.2, 2.0, 1.5]))

cube2origin4 = np.eye(4)
cube2origin4[:3, 3] = np.array([0.0, 1.0, 0.7])
rb_box4 = hydroelastic_contact.RigidBody.make_box(cube2origin4, np.array([1.5, 2.0, 0.2]))

cube2origin5 = np.eye(4)
cube2origin5[:3, 3] = np.array([0.0, 1.0, -0.7])
rb_box5 = hydroelastic_contact.RigidBody.make_box(cube2origin5, np.array([1.5, 2.0, 0.2]))

rb1 = hydroelastic_contact.RigidBody.make_sphere(np.array([0.4, 0.5, 0.2]), 0.15, 1)
rb2 = hydroelastic_contact.RigidBody.make_sphere(np.array([-0.2, 1.8, -0.4]), 0.15, 1)
rb3 = hydroelastic_contact.RigidBody.make_sphere(np.array([0.2, 1.0, 0]), 0.15, 1)
rb4 = hydroelastic_contact.RigidBody.make_sphere(np.array([-0.4, 1.5, -0.4]), 0.15, 1)


GPa = 100000000
rb_box1.youngs_modulus = 100 * GPa
rb_box2.youngs_modulus = 100 * GPa
rb_box3.youngs_modulus = 100 * GPa
rb_box4.youngs_modulus = 100 * GPa
rb_box5.youngs_modulus = 100 * GPa

rb1.youngs_modulus = 0.1 * GPa
rb2.youngs_modulus = 0.1 * GPa
rb3.youngs_modulus = 0.1 * GPa
rb4.youngs_modulus = 0.1 * GPa

p_object_box1 = make_object(rb_box1, 100, np.array([0.0, 0.0, 0.0]), True)
p_object_box2 = make_object(rb_box2, 100, np.array([0.0, 0.0, 0.0]), True)
p_object_box3 = make_object(rb_box3, 100, np.array([0.0, 0.0, 0.0]), True)
p_object_box4 = make_object(rb_box4, 100, np.array([0.0, 0.0, 0.0]), True)
p_object_box5 = make_object(rb_box5, 100, np.array([0.0, 0.0, 0.0]), True)

p_object1 = make_object(rb1, 1, np.array([-0.1, 0.0, 0.0]), False)
p_object2 = make_object(rb2, 1, np.array([0.0, 0.0, 1.5]), False)
p_object3 = make_object(rb3, 1, np.array([2.0, 0.0, 0.0]), False)
p_object4 = make_object(rb4, 1, np.array([0.0, 0.0, -0.5]), False)

fig.view_init()

if "__file__" in globals():
    fig.animate(AnimationCallback([p_object_box1, p_object_box2, p_object_box3, p_object_box4, p_object_box5,
                                   p_object1, p_object2, p_object3, p_object4]), 500, loop=True)
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")


ax.plot(xs, vels)
ax.plot(xs, accs)
plot.show()
