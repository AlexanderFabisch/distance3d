# python -m cProfile -s tottime benchmarks/benchmark_gjk.py > profile
# python -m cProfile -o program.prof benchmarks/benchmark_gjk.py
import timeit
from functools import partial
import numpy as np
from distance3d import colliders
from distance3d.gjk import gjk


random_state = np.random.RandomState(32)
collider1 = colliders.ConvexHullVertices(random_state.randn(1000, 3))
collider2 = colliders.ConvexHullVertices(random_state.randn(1000, 3))
times = timeit.repeat(partial(gjk, collider1=collider1, collider2=collider2), repeat=10, number=1000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
