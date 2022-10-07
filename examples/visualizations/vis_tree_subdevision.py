"""
=================================================
Visualize Subdevision of the AABB Tree
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


# rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(0 * np.ones(3), 0.15, 4)
rigid_body1 = hydroelastic_contact.RigidBody.make_cube(np.eye(4), 0.15)

aabbs1 = hydroelastic_contact.tetrahedral_mesh_aabbs(rigid_body1.tetrahedra_points)
root1, nodes1, aabbs1 = hydroelastic_contact.new_tree_from_aabbs(aabbs1)

hydroelastic_contact.print_aabb_tree(root1, nodes1)

fig = pv.figure()
fig.plot_transform(np.eye(4), s=0.1)
visualization.RigidBodyTetrahedralMesh(
    rigid_body1.body2origin_, rigid_body1.vertices_, rigid_body1.tetrahedra_).add_artist(fig)

for i in get_leafs_of_node(nodes1[nodes1[root1, 1], 1], nodes1):
    tetrahedron_points1 = rigid_body1.tetrahedra_points[i].dot(
        rigid_body1.body2origin_[:3, :3].T) + rigid_body1.body2origin_[:3, 3]
    visualization.Tetrahedron(tetrahedron_points1, c=(1, 0, 0)).add_artist(fig)

for i in get_leafs_of_node(nodes1[nodes1[root1, 1], 2], nodes1):
    tetrahedron_points1 = rigid_body1.tetrahedra_points[i].dot(
        rigid_body1.body2origin_[:3, :3].T) + rigid_body1.body2origin_[:3, 3]
    visualization.Tetrahedron(tetrahedron_points1, c=(1, 0, 0.7)).add_artist(fig)

for i in get_leafs_of_node(nodes1[nodes1[root1, 2], 1], nodes1):
    tetrahedron_points1 = rigid_body1.tetrahedra_points[i].dot(
        rigid_body1.body2origin_[:3, :3].T) + rigid_body1.body2origin_[:3, 3]
    visualization.Tetrahedron(tetrahedron_points1, c=(0, 1, 0)).add_artist(fig)

for i in get_leafs_of_node(nodes1[nodes1[root1, 2], 2], nodes1):
    tetrahedron_points1 = rigid_body1.tetrahedra_points[i].dot(
        rigid_body1.body2origin_[:3, :3].T) + rigid_body1.body2origin_[:3, 3]
    visualization.Tetrahedron(tetrahedron_points1, c=(0, 1, 0.7)).add_artist(fig)


fig.view_init()
fig.show()





