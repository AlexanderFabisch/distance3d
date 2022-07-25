"""
====================================
Visualize Intersection of Tetrahedra
====================================
"""
print(__doc__)
import numpy as np
import pytransform3d.visualizer as pv
from distance3d import mesh, pressure_field, utils


import math
class HalfPlane:
    def __init__(self, p, normal2d):
        self.p = p
        self.pq = utils.norm_vector(np.array([normal2d[1], -normal2d[0]]))
        self.normal2d = normal2d
        self.angle = math.atan(self.pq[1] / self.pq[0])

    def out(self, point):
        return float(np.cross(self.pq, point - self.p)) < 1e-9

    def isless(self, halfplane):
        if abs(self.angle - halfplane.angle) < 1e-6:
            return float(np.cross(self.pq, halfplane.p - self.p)) < 0.0
        return self.angle < halfplane.angle

    def isect(self, halfplane):
        alpha = np.cross((halfplane.p - self.p), halfplane.pq) / np.cross(
            self.pq, halfplane.pq)
        return self.p + self.pq * alpha


def remove_duplicates(halfplanes):
    angles = np.array([hp.angle for hp in halfplanes])
    indices = np.argsort(angles)
    halfplanes = [halfplanes[i] for i in indices]
    result = []
    for hp in halfplanes:
        if len(result) == 0 or abs(result[-1].angle - hp.angle) > 1e-12:
            result.append(hp)
    return result


from collections import deque
def intersect_halfplanes(halfplanes):
    halfplanes = remove_duplicates(halfplanes)
    dq = deque()
    for hp in halfplanes:
        while len(dq) >= 2 and hp.out(dq[-1].isect(dq[-2])):
            dq.pop()
        while len(dq) >= 2 and hp.out(dq[0].isect(dq[1])):
            dq.popleft()
        dq.append(hp)

    while len(dq) >= 3 and dq[0].out(dq[-1].isect(dq[-2])):
        dq.pop()
    while len(dq) >= 3 and dq[-1].out(dq[0].isect(dq[2])):
        dq.popleft()

    if len(dq) < 3:
        return None
    else:
        return np.row_stack([dq[i].isect(dq[(i + 1) % len(dq)])
                             for i in range(len(dq))])


vertices1, tetrahedra1 = mesh.make_tetrahedral_icosphere(np.array([0.1, 0.1, 0.1]), 1.0, order=2)
vertices2, tetrahedra2 = mesh.make_tetrahedral_icosphere(np.array([0.1, 0.1, 1.6]), 1.0, order=2)

tetrahedron1 = vertices1[tetrahedra1[257]]
tetrahedron2 = vertices2[tetrahedra2[310]]

epsilon1 = np.array([0.0, 0.0, 0.0, 1.0])
epsilon2 = np.array([0.0, 0.0, 0.0, 1.0])
plane_hnf = pressure_field.contact_plane(tetrahedron1, tetrahedron2, epsilon1, epsilon2)
assert pressure_field.check_tetrahedra_intersect_contact_plane(tetrahedron1, tetrahedron2, plane_hnf)

plane_normal = plane_hnf[:3]
plane_point = plane_hnf[:3] * plane_hnf[3]
x, y = utils.plane_basis_from_normal(plane_normal)
plane2origin = np.vstack((np.column_stack((x, y, plane_normal, plane_point)),
                          np.array([0.0, 0.0, 0.0, 1.0])))

cart2plane, plane2cart, plane2cart_offset = pressure_field.plane_projection(plane_hnf)

halfplanes = []
for tetrahedron in (tetrahedron1, tetrahedron2):
    X = pressure_field.barycentric_transform(tetrahedron)
    for i in range(4):
        halfspace = X[i]
        normal2d = cart2plane.dot(halfspace[:3])
        if np.linalg.norm(normal2d) > 1e-9:
            p = normal2d * (-halfspace[3] - halfspace[:3].dot(plane2cart_offset)) / np.dot(normal2d, normal2d)
            halfplanes.append(HalfPlane(p, normal2d))

poly = intersect_halfplanes(halfplanes)

poly3d = np.row_stack([plane2cart.dot(p) + plane2cart_offset for p in poly])

#"""
import matplotlib.pyplot as plt
plt.figure()
ax = plt.subplot(111, aspect="equal")
for halfplane in halfplanes:
    line = halfplane.p + np.linspace(-10.0, 10.0, 101)[:, np.newaxis] * halfplane.pq
    plt.plot(line[:, 0], line[:, 1], lw=3)
    normal = halfplane.p + np.linspace(0.0, 1.0, 101)[:, np.newaxis] * halfplane.normal2d
    plt.plot(normal[:, 0], normal[:, 1])
plt.scatter(poly[:, 0], poly[:, 1], s=100)
plt.show()
#"""

fig = pv.figure()
fig.scatter(tetrahedron1, s=0.01, c=(1, 0, 0))
fig.scatter(tetrahedron2, s=0.01, c=(0, 0, 1))
fig.plot_transform(np.eye(4), s=0.05)
fig.plot_plane(normal=plane_hnf[:3], d=plane_hnf[3])
fig.plot_transform(plane2origin, s=0.05)
fig.scatter(poly3d, s=0.01, c=(1, 0, 1))
fig.view_init()

if "__file__" in globals():
    fig.show()
else:
    fig.save_image("__open3d_rendered_image.jpg")
