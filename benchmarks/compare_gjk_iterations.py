import json
import os
import numpy as np
from distance3d import colliders, random
from distance3d.gjk._gjk_original import gjk_distance_iterations
from distance3d.gjk._gjk_jolt import gjk_distance_jolt_iterations
from distance3d.gjk._gjk_nesterov_accelerated_primitives import gjk_nesterov_accelerated_primitives_iterations

iterations = 100
random_state = np.random.RandomState(84)
shape_names = ["sphere", "capsule", "cylinder"]

original_iteration_sum = 0
jolt_iteration_sum = 0
nasterov_iteration_sum = 0

for i in range(iterations):
    print("Case:", i)

    shape1 = shape_names[random_state.randint(len(shape_names))]
    args1 = random.RANDOM_GENERATORS[shape1](random_state)
    shape2 = shape_names[random_state.randint(len(shape_names))]
    args2 = random.RANDOM_GENERATORS[shape2](random_state)
    collider1 = colliders.COLLIDERS[shape1](*args1)
    collider2 = colliders.COLLIDERS[shape2](*args2)

    original_iterations = gjk_distance_iterations(collider1, collider2)
    jolt_iterations = gjk_distance_jolt_iterations(collider1, collider2)
    nasterov_iterations = gjk_nesterov_accelerated_primitives_iterations(collider1, collider2)

    print("Original Iterations:", original_iterations)
    print("Jolt Distance Iterations:", jolt_iterations)
    print("Nesterov Iterations", nasterov_iterations)

    original_iteration_sum += original_iterations
    jolt_iteration_sum += jolt_iterations
    nasterov_iteration_sum += nasterov_iterations

print("Original Iterations per Case", original_iteration_sum / iterations)
print("Jolt Distance Iterations per Case", jolt_iteration_sum / iterations)
print("Nesterov Iterations per Case", nasterov_iteration_sum / iterations)