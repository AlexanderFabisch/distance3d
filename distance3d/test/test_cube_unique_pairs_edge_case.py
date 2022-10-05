import numpy as np
from distance3d import hydroelastic_contact
import pytransform3d.rotations as pr


def cube_unique_pairs(use_aabb_trees):
    rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(0.13 * np.ones(3), 0.15, 2)
    cube2origin = np.eye(4)
    cube2origin[:3, :3] = pr.active_matrix_from_extrinsic_euler_zyx([0.1, 0.3, 0.5])
    cube2origin[:3, 3] = 0.25 * np.ones(3)
    rigid_body2 = hydroelastic_contact.RigidBody.make_cube(cube2origin, 0.15)

    rigid_body1.express_in(rigid_body2.body2origin_)
    broad_tetrahedra1, broad_tetrahedra2, broad_pairs = hydroelastic_contact.broad_phase_tetrahedra(
        rigid_body1, rigid_body2, use_aabb_trees)

    assert len(broad_pairs) == 162


def test_cube_uique_pairs_brute_force(self):
    cube_unique_pairs(False)


def test_cube_uique_pairs_aabb_tree(self):
    cube_unique_pairs(True)
