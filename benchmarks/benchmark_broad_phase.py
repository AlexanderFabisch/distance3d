import timeit
import numpy as np
from functools import partial
from distance3d import hydroelastic_contact

random_state = np.random.RandomState(0)


def random_broad_phase_collisions(random_state, use_aabb_trees):
    p = random_state.randn(3)
    rigid_body1 = hydroelastic_contact.RigidBody.make_sphere(0 * np.ones(3), 0.15, 2)
    rigid_body2 = hydroelastic_contact.RigidBody.make_sphere(p, 0.15, 2)

    rigid_body1.express_in(rigid_body2.body2origin_)
    hydroelastic_contact.broad_phase_tetrahedra(rigid_body1, rigid_body2, use_aabb_trees)


repeat = 10
number = 10
times = timeit.repeat(partial(
    random_broad_phase_collisions, random_state=random_state, use_aabb_trees=True),
    repeat=repeat, number=number)
print(f"AABB Trees Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

times = timeit.repeat(partial(
    random_broad_phase_collisions, random_state=random_state, use_aabb_trees=False),
    repeat=repeat, number=number)
print(f"Brute Force Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
