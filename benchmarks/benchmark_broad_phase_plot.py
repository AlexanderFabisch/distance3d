from functools import partial

import matplotlib

matplotlib.use('TkAgg')
from distance3d import hydroelastic_contact
import timeit
import numpy as np
from matplotlib import pyplot

"""
Brute Force Mean: 0.00287; Std. dev.: 0.00529
AABB Trees No Creation Mean: 0.00674; Std. dev.: 0.01290
AABB Trees Mean: 0.02675; Std. dev.: 0.00457
Old AABB Trees Mean: 0.24804; Std. dev.: 0.03056

Brute Force Mean: 0.00200; Std. dev.: 0.00282
AABB Trees No Creation Mean: 0.00239; Std. dev.: 0.00476
AABB Trees Mean: 0.19004; Std. dev.: 0.00541
Old AABB Trees Mean: 1.60006; Std. dev.: 0.01856

Brute Force Mean: 0.01090; Std. dev.: 0.00295
AABB Trees No Creation Mean: 0.00279; Std. dev.: 0.00552
AABB Trees Mean: 0.94646; Std. dev.: 0.00822

Brute Force Mean: 0.15937; Std. dev.: 0.00502
AABB Trees No Creation Mean: 0.00247; Std. dev.: 0.00490
AABB Trees Mean: 4.70398; Std. dev.: 0.06692

Brute Force Mean: 2.37237; Std. dev.: 0.02910
AABB Trees No Creation Mean: 0.00232; Std. dev.: 0.00460
AABB Trees Mean: 77.21539; Std. dev.: 1.11606

Brute Force Mean: 40.64899; Std. dev.: 1.14510
AABB Trees No Creation Mean: 0.00238; Std. dev.: 0.00471

AABB Trees No Creation Mean: 0.00243; Std. dev.: 0.00483

"""


values = [['Brute Force', [0.00287, 0.00200, 0.15937, 2.37237]],
          ['AABB Trees No Creation', [0.00274, 0.00239, 0.00279, 0.00247, 0.00232, 0.00238, 0.00243]],
          ['AABB Trees', [0.02675, 0.19004, 0.94646, 4.70398]],
          ['Old AABB Trees', [0.24804, 1.60006, 10.0]]]

for value in values:
    pyplot.plot(value[1], markersize=20, label=value[0])

pyplot.legend()
pyplot.show()

"""



"""

