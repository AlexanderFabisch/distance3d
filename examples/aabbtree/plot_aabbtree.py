from collections import deque
import numpy as np
import pytransform3d.plot_utils as ppu
import matplotlib.pyplot as plt
from pymanipulation import containment, aabbtree, geometry


ax = ppu.make_3d_axis(3)
random_state = np.random.RandomState(0)

center, radius = geometry.randn_sphere(random_state)
mins, maxs = containment.sphere_aabb(center, radius)
geometry.plot_sphere(ax, center, radius)
geometry.plot_aabb(ax, mins, maxs)
aabb1 = aabbtree.AABB(np.array([mins, maxs]).T)

box2origin, size = geometry.rand_box(random_state)
mins, maxs = containment.box_aabb(box2origin, size)
geometry.plot_box(ax, box2origin, size)
geometry.plot_aabb(ax, mins, maxs)
aabb2 = aabbtree.AABB(np.array([mins, maxs]).T)

cylinder2origin, radius, length = geometry.randn_cylinder(random_state)
mins, maxs = containment.cylinder_aabb(cylinder2origin, radius, length)
geometry.plot_cylinder(ax, cylinder2origin, radius, length)
geometry.plot_aabb(ax, mins, maxs)
aabb3 = aabbtree.AABB(np.array([mins, maxs]).T)

capsule2origin, radius, height = geometry.randn_capsule(random_state)
mins, maxs = containment.capsule_aabb(capsule2origin, radius, height)
geometry.plot_capsule(ax, capsule2origin, radius, height)
geometry.plot_aabb(ax, mins, maxs)
aabb4 = aabbtree.AABB(np.array([mins, maxs]).T)

tree = aabbtree.AABBTree()
tree.add(aabb1, "sphere")
tree.add(aabb2, "box")
tree.add(aabb3, "cylinder")
tree.add(aabb4, "capsule")
mins, maxs = np.array(tree.aabb.limits).T
nodes = deque()
nodes.append(tree)
while nodes:
    node = nodes.popleft()
    mins, maxs = np.array(node.aabb.limits).T
    geometry.plot_aabb(ax, mins, maxs, alpha=0.4)
    if not node.is_leaf:
        nodes.extend([node.left, node.right])
plt.show()
