import numpy as np
import open3d as o3d
import pytransform3d.transformations as pt
import pytransform3d.visualizer as pv
from distance3d import random, gjk, epa


class Mesh(pv.Artist):
    def __init__(self, mesh, A2B=np.eye(4)):
        self.mesh = mesh
        self.mesh.compute_vertex_normals()
        self.A2B = None
        self.set_data(A2B)

    def set_data(self, A2B):
        previous_A2B = self.A2B
        if previous_A2B is None:
            previous_A2B = np.eye(4)
        self.A2B = A2B

        self.mesh.transform(pt.invert_transform(previous_A2B, check=False))
        self.mesh.transform(self.A2B)

    @property
    def geometries(self):
        return [self.mesh]


dt = 0.001
g = np.array([0, 0, -9.81])


class AnimationCallback:
    def __init__(self, mesh22origin):
        self.dt = 0.01
        self.m2 = 1.0
        self.mesh22origin = mesh22origin
        self._initial_state()

    def _initial_state(self):
        self.pose2 = np.copy(mesh22origin)
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

vertices1, faces1, points1, triangles1 = random.randn_convex(
    random_state, n_points=100000, center_scale=0.0, std=1, return_indices=True)
mesh1 = o3d.geometry.TriangleMesh()
mesh1.vertices = o3d.utility.Vector3dVector(points1)
mesh1.triangles = o3d.utility.Vector3iVector(triangles1)
artist1 = Mesh(mesh1)
artist1.add_artist(fig)

vertices2, faces2, points2, triangles2 = random.randn_convex(
    random_state, n_points=100000, center_scale=0.0, std=0.1, return_indices=True)
mesh2 = o3d.geometry.TriangleMesh()
mesh2.vertices = o3d.utility.Vector3dVector(points2)
mesh2.triangles = o3d.utility.Vector3iVector(triangles2)
mesh22origin = pt.transform_from(R=np.eye(3), p=[0, 0, 10])
artist2 = Mesh(mesh2, mesh22origin)
artist2.add_artist(fig)

fig.view_init()

fig.animate(AnimationCallback(mesh22origin), 500, loop=True, fargs=(artist1, artist2))
fig.show()
