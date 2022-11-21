"""
=================================================
Physical simulation exploring the young's modulus
=================================================
"""
# Scenario:
# A rubber ball bouncing on a wooden box

print(__doc__)
import numpy as np
import pytransform3d.visualizer as pv
from distance3d import visualization, hydroelastic_contact

dt = 0.001
g = np.array([0, -9.81, 0])

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)

p_objects = []


class PhysicsObject:
    def __init__(self, rigid_body, artist, mass, velocity, fixed=False):
        self.rigid_body = rigid_body
        self.fixed = fixed

        self.mass = mass

        self.acceleration = np.zeros(3, dtype=float)
        self.velocity = velocity
        self.artist = artist

    def step(self, forces=[]):
        if self.fixed:
            return

        gravity_force = g * self.mass
        forces.append(gravity_force)

        self.acceleration = np.add.reduce(forces) / self.mass
        self.velocity += dt * self.acceleration
        self.rigid_body.body2origin_[:3, 3] += self.velocity * dt
        self.artist.set_data(self.rigid_body.body2origin_, self.rigid_body.vertices_,
                             self.rigid_body.tetrahedra_)


class AnimationCallback:
    def __init__(self, p_object1, p_object2):
        self.p_object1 = p_object1
        self.p_object2 = p_object2
        self.contact_surfaces = []

        contact_surface = hydroelastic_contact.find_contact_surface(
            self.p_object1.rigid_body, self.p_object2.rigid_body)

        self.contact_surface = visualization.ContactSurface(
            contact_surface.frame2world,
            contact_surface.contact_polygons,
            contact_surface.contact_polygon_triangles,
            contact_surface.pressures)
        self.contact_surface.add_artist(fig)

    def __call__(self, step):
        intersection, wrench12, wrench21, details = hydroelastic_contact.contact_forces(
            self.p_object1.rigid_body, self.p_object2.rigid_body, return_details=True)

        if intersection:
            self.contact_surface.set_data(
                np.eye(4),
                details["contact_polygons"],
                details["contact_polygon_triangles"],
                details["pressures"])

        self.p_object1.step([wrench21[:3]])
        self.p_object2.step([wrench12[:3]])

        return [self.p_object1.artist, self.p_object2.artist, self.contact_surface]


def make_object(rigid_body, mass, acc, fixed):
    artist = visualization.RigidBodyTetrahedralMesh(
        rigid_body.body2origin_, rigid_body.vertices_, rigid_body.tetrahedra_)
    artist.add_artist(fig)

    return PhysicsObject(rigid_body, artist, mass, acc, fixed)


rb1 = hydroelastic_contact.RigidBody.make_box(np.eye(4), np.array([0.5, 0.5, 0.5]))
rb2 = hydroelastic_contact.RigidBody.make_sphere(np.array([0.0, 1.0, 0.0]), 0.15, 2)


GPa = 100000000
rb1.set_youngs_modulus(100*GPa)
rb2.set_youngs_modulus(0.1*GPa)

p_object1 = make_object(rb1, 100, np.array([0.0, 0.0, 0.0]), True)
p_object2 = make_object(rb2, 1, np.array([-0.1, 0.0, 0.0]), False)

fig.view_init()

if "__file__" in globals():
    fig.animate(AnimationCallback(p_object1, p_object2), 500, loop=True)
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
