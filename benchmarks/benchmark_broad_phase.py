import timeit

import aabbtree
import numpy as np
from functools import partial
from distance3d import hydroelastic_contact

random_state = np.random.RandomState(0)


def create_random_spheres(random_state, size):
    p = random_state.randn(3)
    rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(0 * np.ones(3), 0.15, size)
    rigid_body2 = hydroelastic_contact.RigidBody.make_sphere(p, 0.15, size)
    rigid_body1.express_in(rigid_body2.body2origin_)

    aabbs1 = hydroelastic_contact.tetrahedral_mesh_aabbs(rigid_body1.tetrahedra_points)
    aabbs2 = hydroelastic_contact.tetrahedral_mesh_aabbs(rigid_body2.tetrahedra_points)

    return aabbs1, aabbs2


def brute_force_aabbs(aabbs1, aabbs2):
    hydroelastic_contact._all_aabbs_overlap(aabbs1, aabbs2)

def aabb_tree(aabbs1, aabbs2):
    root1, nodes1, aabbs1 = hydroelastic_contact.new_tree_from_aabbs(aabbs1)
    root2, nodes2, aabbs2 = hydroelastic_contact.new_tree_from_aabbs(aabbs2)

    hydroelastic_contact.query_overlap_of_other_tree(root1, nodes1, aabbs1, root2, nodes2, aabbs2)


def old_aabb_tree(aabbs1, aabbs2):
    tree2 = aabbtree.AABBTree()
    for j, aabb in enumerate(aabbs2):
        tree2.add(aabbtree.AABB(aabb), j)
    broad_tetrahedra1 = []
    broad_tetrahedra2 = []
    for i, aabb in enumerate(aabbs1):
        new_indices2 = tree2.overlap_values(aabbtree.AABB(aabb), method='DFS', closed=False, unique=False)
        broad_tetrahedra2.extend(new_indices2)
        broad_tetrahedra1.extend([i] * len(new_indices2))


aabbs1, aabbs2 = create_random_spheres(random_state, 1)

repeat = 5
number = 5


times = timeit.repeat(partial(brute_force_aabbs, aabbs1=aabbs1, aabbs2=aabbs2), repeat=repeat, number=number)
print(f"Brute Force Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

root1, nodes1, aabbs1 = hydroelastic_contact.new_tree_from_aabbs(aabbs1)# root2, nodes2, aabbs2 = hydroelastic_contact.new_tree_from_aabbs(aabbs2)

times = timeit.repeat(partial(hydroelastic_contact.query_overlap_of_other_tree, root1, nodes1, aabbs1, root2, nodes2, aabbs2),repeat=repeat, number=number)
print(f"AABB Trees No Creation Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

times = timeit.repeat(partial(aabb_tree, aabbs1=aabbs1, aabbs2=aabbs2), repeat=repeat, number=number)
print(f"AABB Trees Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

times = timeit.repeat(partial(old_aabb_tree, aabbs1=aabbs1, aabbs2=aabbs2), repeat=repeat, number=number)
print(f"Old AABB Trees Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")


"""
AABB Trees Mean: 0.11631; Std. dev.: 0.00316
Brute Force Mean: 0.00656; Std. dev.: 0.00315

AABB Trees Mean: 0.08419; Std. dev.: 0.01048
Brute Force Mean: 0.00623; Std. dev.: 0.00291
"""