"""
=========
AABB Tree
=========

Plot AABB tree.
"""
print(__doc__)
import numpy as np
import pytransform3d.plot_utils as ppu
import matplotlib.pyplot as plt
from distance3d import containment, random, plotting
import aabbtree


ax = ppu.make_3d_axis(3)
random_state = np.random.RandomState(0)

center, radius = random.rand_sphere(random_state)
mins, maxs = containment.sphere_aabb(center, radius)
ppu.plot_sphere(ax, radius, center, wireframe=False, alpha=0.5)
plotting.plot_aabb(ax, mins, maxs)
aabb1 = aabbtree.AABB(np.array([mins, maxs]).T)

box2origin, size = random.rand_box(random_state)
mins, maxs = containment.box_aabb(box2origin, size)
ppu.plot_box(ax=ax, A2B=box2origin, size=size, wireframe=False, alpha=0.5)
plotting.plot_aabb(ax, mins, maxs)
aabb2 = aabbtree.AABB(np.array([mins, maxs]).T)

cylinder2origin, radius, length = random.rand_cylinder(random_state)
mins, maxs = containment.cylinder_aabb(cylinder2origin, radius, length)
ppu.plot_cylinder(ax=ax, A2B=cylinder2origin, radius=radius, length=length, wireframe=False, alpha=0.5)
plotting.plot_aabb(ax, mins, maxs)
aabb3 = aabbtree.AABB(np.array([mins, maxs]).T)

capsule2origin, radius, height = random.rand_capsule(random_state)
mins, maxs = containment.capsule_aabb(capsule2origin, radius, height)
ppu.plot_capsule(ax, capsule2origin, height, radius, wireframe=False, alpha=0.5)
plotting.plot_aabb(ax, mins, maxs)
aabb4 = aabbtree.AABB(np.array([mins, maxs]).T)

ellipsoid2origin, radii = random.rand_ellipsoid(random_state)
mins, maxs = containment.ellipsoid_aabb(ellipsoid2origin, radii)
ppu.plot_ellipsoid(ax, radii, ellipsoid2origin, wireframe=False, alpha=0.5)
plotting.plot_aabb(ax, mins, maxs)
aabb5 = aabbtree.AABB(np.array([mins, maxs]).T)

tree = aabbtree.AABBTree()
tree.add(aabb1, "sphere")
tree.add(aabb2, "box")
tree.add(aabb3, "cylinder")
tree.add(aabb4, "capsule")
tree.add(aabb5, "ellipsoid")
plotting.plot_aabb_tree(ax, tree)
plt.show()
