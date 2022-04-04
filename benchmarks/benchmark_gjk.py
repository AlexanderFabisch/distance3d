# python -m cProfile -s tottime benchmarks/benchmark_gjk.py > profile
# python -m cProfile -o program.prof benchmarks/benchmark_gjk.py
import timeit
from functools import partial
import numpy as np
from distance3d.gjk import gjk


random_state = np.random.RandomState(32)
vertices1 = random_state.randn(1000, 3)
vertices2 = random_state.randn(1000, 3)
times = timeit.repeat(partial(gjk, vertices1=vertices1, vertices2=vertices2), repeat=10, number=1000)
print(f"Mean: {np.mean(times):.5f}; Std. dev.: {np.std(times):.5f}")
