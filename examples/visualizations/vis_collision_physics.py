"""
============================================
Physical simulation with collision detection
============================================
"""
print(__doc__)
import numpy as np
import pytransform3d.transformations as pt
import pytransform3d.visualizer as pv
from distance3d import random, gjk, epa, visualization


dt = 0.001
g = np.array([0, 0, -9.81])


class AnimationCallback:
    def __init__(self, mesh2origin2):
        self.dt = 0.01
        self.m2 = 1.0
        self.mesh2origin2 = mesh2origin2
        self._initial_state()

    def _initial_state(self):
        self.pose2 = np.copy(self.mesh2origin2)
        self.v2 = np.zeros(3)

    def __call__(self, step, artist1, artist2):
        if step == 0:
            self._initial_state()

        F_gravity = self.m2 * g

        self.a2 = F_gravity / self.m2
        self.v2 += self.dt * self.a2
        self.pose2[:3, 3] += self.dt * self.v2

        # Collision detection and resolution
        vertices1 = np.asarray(artist1.mesh.vertices)
        vertices2 = np.asarray(artist2.mesh.vertices)
        dist, _, _, simplex = gjk.gjk_with_simplex(gjk.Convex(vertices1), gjk.Convex(vertices2))
        if dist == 0.0:
            mtv = epa.epa_vertices(simplex, vertices1, vertices2)
            self.pose2[:3, 3] += mtv

        artist2.set_data(np.copy(self.pose2))
        return artist2


fig = pv.figure()
fig.plot_transform(np.eye(4), s=1)


random_state = np.random.RandomState(6)

mesh2origin, vertices1, triangles1 = random.randn_convex(
    random_state, n_points=10, center_scale=0.0, radius_scale=5)
artist1 = visualization.Mesh(mesh2origin, vertices1, triangles1)
artist1.add_artist(fig)

_, vertices2, triangles2 = random.randn_convex(
    random_state, n_points=10, center_scale=0.0, radius_scale=1)
mesh2origin2 = pt.transform_from(R=np.eye(3), p=[0, 0, 10])
artist2 = visualization.Mesh(mesh2origin2, vertices2, triangles2)
artist2.add_artist(fig)

fig.view_init()

if "__file__" in globals():
    fig.animate(AnimationCallback(mesh2origin2),
                500, loop=False, fargs=(artist1, artist2))
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
