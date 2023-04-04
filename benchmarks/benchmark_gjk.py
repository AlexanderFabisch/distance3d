# python -m cProfile -s tottime benchmarks/benchmark_gjk.py > profile
# python -m cProfile -o program.prof benchmarks/benchmark_gjk.py
import timeit
import numpy as np
from distance3d import colliders, random
from distance3d.gjk import gjk_nesterov_accelerated_intersection, gjk_distance_original, gjk_intersection_jolt
from numba import config

iterations = 10
shapes = []
random_state = np.random.RandomState(84)
shape_names = list(colliders.COLLIDERS.keys())

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


def benchmark_jolt():
    for i in range(iterations):
        gjk_intersection_jolt(shapes[i][0], shapes[i][1])


def benchmark_nesterov_accelerated():
    for i in range(iterations):
        gjk_nesterov_accelerated_intersection(shapes[i][0], shapes[i][1])


times = timeit.repeat(benchmark_original, repeat=10, number=1)
print(f"Original Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

config.DISABLE_JIT = True
times = timeit.repeat(benchmark_jolt, repeat=10, number=1)
print(f"Jolt Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
config.DISABLE_JIT = False

times = timeit.repeat(benchmark_jolt, repeat=10, number=1)
print(f"Jolt with Numba Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")

times = timeit.repeat(benchmark_nesterov_accelerated, repeat=10, number=1)
print(f"Nesterov Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
