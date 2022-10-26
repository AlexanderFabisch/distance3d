from functools import partial

from distance3d import hydroelastic_contact
import timeit
import numpy as np

import matplotlib

from distance3d.aabb_tree import AabbTree

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(0 * np.ones(3), 0.15, 4)
aabbs1 = hydroelastic_contact.tetrahedral_mesh_aabbs(rigid_body1.tetrahedra_points)

values = [[], [], []]
y_steps = []
step = 100
skip_point = 3.0
skip_data = [False, False, False]

for i in range(int(len(aabbs1) / step)):
    print(f"Nr {i * step} of {len(aabbs1)}")
    y_steps.append(i * step)

    if not skip_data[0]:
        times = timeit.repeat(partial(AabbTree, aabbs=aabbs1[(len(aabbs1) - i*step):],
                                      pre_insertion_methode="none"), repeat=5, number=5)
        print(f"None: Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
        values[0].append(np.mean(times))
        skip_data[0] = np.mean(times) > skip_point

    if not skip_data[1]:
        times = timeit.repeat(partial(AabbTree, aabbs=aabbs1[(len(aabbs1) - i * step):],
                                      pre_insertion_methode="shuffle"), repeat=5, number=5)
        print(f"Shuffle: Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
        values[1].append(np.mean(times))
        skip_data[0] = np.mean(times) > skip_point

    if not skip_data[2]:
        times = timeit.repeat(partial(AabbTree, aabbs=aabbs1[(len(aabbs1) - i * step):],
                                      pre_insertion_methode="sort"), repeat=5, number=5)
        print(f"Sort: Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
        values[2].append(np.mean(times))
        skip_data[0] = np.mean(times) > skip_point

plt.plot(y_steps, values[0], markersize=20, label="AABB Tree none")
plt.plot(y_steps, values[1], markersize=20, label="AABB Tree shuffle")
plt.plot(y_steps, values[2], markersize=20, label="AABB Tree sort")
plt.legend()
plt.show()
