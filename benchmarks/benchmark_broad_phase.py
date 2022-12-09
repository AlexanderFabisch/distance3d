from functools import partial
import aabbtree
from distance3d import hydroelastic_contact
import timeit
import numpy as np
from distance3d.aabb_tree import all_aabbs_overlap, AabbTree
import matplotlib.pyplot as plt

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
    all_aabbs_overlap(aabbs1, aabbs2)


def aabb_tree(aabbs1, aabbs2):
    aabb_tree1 = AabbTree()
    aabb_tree1.insert_aabbs(aabbs1)
    aabb_tree2 = AabbTree()
    aabb_tree2.insert_aabbs(aabbs2)

    aabb_tree1.overlaps_aabb_tree(aabb_tree2)


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


aabbs1, aabbs2 = create_random_spheres(random_state, 3)

repeat = 5
number = 5

values = [[], [], []]
y_steps = [[], [], []]
step = 10
skip_point = 3.0
skip_data = [False, False, False]

for i in range(1, int(len(aabbs1) / step)):
    print(f"Nr {i * step} of {len(aabbs1)}")

    if not skip_data[0]:
        times = timeit.repeat(partial(brute_force_aabbs,
                                      aabbs1=aabbs1[(len(aabbs1) - i * step):],
                                      aabbs2=aabbs2[(len(aabbs1) - i * step):]),
                              repeat=repeat, number=number)
        print(f"Brute: Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
        values[0].append(np.mean(times))
        skip_data[0] = np.mean(times) > skip_point
        y_steps[0].append(i * step)

    if not skip_data[1]:
        times = timeit.repeat(partial(aabb_tree,
                                      aabbs1=aabbs1[(len(aabbs1) - i * step):],
                                      aabbs2=aabbs2[(len(aabbs1) - i * step):]),
                              repeat=repeat, number=number)

        print(f"New Tree: Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
        values[1].append(np.mean(times))
        skip_data[1] = np.mean(times) > skip_point
        y_steps[1].append(i * step)

    if not skip_data[2]:
        times = timeit.repeat(partial(old_aabb_tree,
                                      aabbs1=aabbs1[(len(aabbs1) - i * step):],
                                      aabbs2=aabbs2[(len(aabbs1) - i * step):]),
                              repeat=repeat, number=number)
        print(f"Old Tree: Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
        values[2].append(np.mean(times))
        skip_data[2] = np.mean(times) > skip_point
        y_steps[2].append(i * step)

plt.xlabel("aabb ammount")
plt.ylabel("time in sec")
plt.plot(y_steps[0], values[0], markersize=20, label="Brute")
plt.plot(y_steps[1], values[1], markersize=20, label="New AABB Tree")
plt.plot(y_steps[2], values[2], markersize=20, label="Old AABB Tree")
plt.legend()
plt.show()
