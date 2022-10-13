"""
=================================================
Visualize Subdivision of the AABB Tree
=================================================
"""
print(__doc__)

import numpy as np
import pytransform3d.visualizer as pv
from distance3d import visualization, hydroelastic_contact


def get_leafs_of_node(node_index, nodes):
    if nodes[node_index, 3] == 1:
        return [node_index]
    else:
        leafs = get_leafs_of_node(nodes[node_index, 1], nodes)
        leafs.extend(get_leafs_of_node(nodes[node_index, 2], nodes))
        return leafs


rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(0 * np.ones(3), 0.15, 4)
# rigid_body1 = hydroelastic_contact.RigidBody.make_cube(np.eye(4), 0.15)

points = rigid_body1.tetrahedra_points
aabbs1 = hydroelastic_contact.tetrahedral_mesh_aabbs(points)

aabb_tree = hydroelastic_contact.AabbTree(aabbs1, "shuffle")

print(aabb_tree)

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
visualization.RigidBodyTetrahedralMesh(
    rigid_body1.body2origin_, rigid_body1.vertices_, rigid_body1.tetrahedra_).add_artist(fig)


def color_leaves(index, color):
    for i in get_leafs_of_node(index, aabb_tree.nodes):
        tetrahedron_points1 = points[i].dot(
            rigid_body1.body2origin_[:3, :3].T) + rigid_body1.body2origin_[:3, 3]
        visualization.Tetrahedron(tetrahedron_points1, c=color).add_artist(fig)


a = aabb_tree.nodes[aabb_tree.nodes[aabb_tree.root, 1], 1]
if a != -1:
    color_leaves(a, (1, 0, 0))

a = aabb_tree.nodes[aabb_tree.nodes[aabb_tree.root, 1], 2]
if a != -1:
    color_leaves(a, (1, 0, 0.3))

a = aabb_tree.nodes[aabb_tree.nodes[aabb_tree.root, 2], 1]
if a != -1:
    color_leaves(a, (0, 1, 0))

a = aabb_tree.nodes[aabb_tree.nodes[aabb_tree.root, 2], 2]
if a != -1:
    color_leaves(a, (0, 1, 0.3))


fig.view_init()
fig.show()
