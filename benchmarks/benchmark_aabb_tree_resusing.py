from functools import partial

from distance3d import hydroelastic_contact
import timeit
import numpy as np

import matplotlib

from distance3d.hydroelastic_contact import _all_aabbs_overlap

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def create_random_spheres(random_state, size):
    p = random_state.randn(3)
    rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(0 * np.ones(3), 0.15, size)
    rigid_body2 = hydroelastic_contact.RigidBody.make_sphere(p, 0.15, size)
    rigid_body1.express_in(rigid_body2.body2origin_)

    return rigid_body1, rigid_body2


def brute(rb1, rb2):
    broad_tetrahedra1, broad_tetrahedra2, broad_pairs = _all_aabbs_overlap(rb1.aabbs, rb2.aabbs)


def aabb_tree(rb1, rb2):
    _, broad_tetrahedra1, broad_tetrahedra2, broad_pairs \
        = rb1.aabb_tree.overlaps_aabb_tree(rb2.aabb_tree)


random_state = np.random.RandomState(0)

values = [[], []]
y_steps = [[], []]
step = 10
skip_point = 3.0
skip_data = [False, False, False]

for i in range(int(100 / step)):
    print(f"Nr {i * step}")

    if not skip_data[0]:
        rb1, rb2 = create_random_spheres(random_state, 4)
        times = timeit.repeat(partial(brute, rb1=rb1, rb2=rb2), repeat=20, number=i)
        print(f"Brute: Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
        values[0].append(np.mean(times))
        skip_data[0] = np.mean(times) > skip_point
        y_steps[0].append(i * step)

    if not skip_data[1]:
        rb1, rb2 = create_random_spheres(random_state, 4)
        times = timeit.repeat(partial(brute, rb1=rb1, rb2=rb2), repeat=20, number=i)
        print(f"Tree: Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
        values[1].append(np.mean(times))
        skip_data[1] = np.mean(times) > skip_point
        y_steps[1].append(i * step)


plt.plot(y_steps[0], values[0], markersize=20, label="Brute")
plt.plot(y_steps[1], values[1], markersize=20, label="Tree")
plt.legend()
plt.show()