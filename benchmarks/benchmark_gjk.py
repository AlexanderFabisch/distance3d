import timeit
import numpy as np
from distance3d import colliders, random
from distance3d.gjk import gjk_distance_original, gjk_intersection_jolt, gjk_distance_jolt, \
    gjk_nesterov_accelerated_primitives, gjk_nesterov_accelerated
from numba import config

iterations = 200
shapes = []
random_state = np.random.RandomState(84)
shape_names = list(["sphere", "ellipsoid", "capsule", "cone", "cylinder", "box"])

for _ in range(iterations):
    shape1 = shape_names[random_state.randint(len(shape_names))]
    args1 = random.RANDOM_GENERATORS[shape1](random_state)
    shape2 = shape_names[random_state.randint(len(shape_names))]
    args2 = random.RANDOM_GENERATORS[shape2](random_state)
    collider1 = colliders.COLLIDERS[shape1](*args1)
    collider2 = colliders.COLLIDERS[shape2](*args2)

    shapes.append((collider1, collider2))


def benchmark_original():
    for i in range(iterations):
        gjk_distance_original(shapes[i][0], shapes[i][1])


def benchmark_jolt_intersection():
    for i in range(iterations):
        gjk_intersection_jolt(shapes[i][0], shapes[i][1])


def benchmark_jolt_distance():
    for i in range(iterations):
        gjk_distance_jolt(shapes[i][0], shapes[i][1])


def benchmark_nesterov_accelerated():
    for i in range(iterations):
        gjk_nesterov_accelerated(shapes[i][0], shapes[i][1], use_nesterov_acceleration=False)


def benchmark_nesterov_accelerated_with_acceleration():
    for i in range(iterations):
        gjk_nesterov_accelerated(shapes[i][0], shapes[i][1], use_nesterov_acceleration=True)


def benchmark_nesterov_accelerated_primitives():
    for i in range(iterations):
        gjk_nesterov_accelerated_primitives(shapes[i][0], shapes[i][1], use_nesterov_acceleration=False)


def benchmark_nesterov_accelerated_primitives_with_acceleration():
    for i in range(iterations):
        gjk_nesterov_accelerated_primitives(shapes[i][0], shapes[i][1], use_nesterov_acceleration=True)


times = timeit.repeat(benchmark_original, repeat=10, number=1)
print(f"Original Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

print("\nJolt:")

config.DISABLE_JIT = True
times = timeit.repeat(benchmark_jolt_intersection, repeat=10, number=1)
print(f"Jolt intersection Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
config.DISABLE_JIT = False

times = timeit.repeat(benchmark_jolt_intersection, repeat=10, number=1)
print(f"Jolt intersection with Numba Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

config.DISABLE_JIT = True
times = timeit.repeat(benchmark_jolt_distance, repeat=10, number=1)
print(f"Jolt distance Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
config.DISABLE_JIT = False

times = timeit.repeat(benchmark_jolt_distance, repeat=10, number=1)
print(f"Jolt distance with Numba Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")


print("\nNesterov accelerated:")
times = timeit.repeat(benchmark_nesterov_accelerated, repeat=10, number=1)
print(f"Nesterov Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

config.DISABLE_JIT = True
times = timeit.repeat(benchmark_nesterov_accelerated, repeat=10, number=1)
print(f"Nesterov with Numba Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
config.DISABLE_JIT = False

times = timeit.repeat(benchmark_nesterov_accelerated_with_acceleration, repeat=10, number=1)
print(f"Nesterov with acceleration Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

config.DISABLE_JIT = True
times = timeit.repeat(benchmark_nesterov_accelerated_with_acceleration, repeat=10, number=1)
print(f"Nesterov with acceleration and Numba Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
config.DISABLE_JIT = False


print("\nNesterov accelerated Primitives:")
times = timeit.repeat(benchmark_nesterov_accelerated_primitives, repeat=10, number=1)
print(f"Nesterov Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

config.DISABLE_JIT = True
times = timeit.repeat(benchmark_nesterov_accelerated_primitives, repeat=10, number=1)
print(f"Nesterov with Numba Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
config.DISABLE_JIT = False

times = timeit.repeat(benchmark_nesterov_accelerated_primitives_with_acceleration, repeat=10, number=1)
print(f"Nesterov with acceleration Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

config.DISABLE_JIT = True
times = timeit.repeat(benchmark_nesterov_accelerated_primitives_with_acceleration, repeat=10, number=1)
print(f"Nesterov with acceleration and Numba Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
config.DISABLE_JIT = False
