import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



"""
Brute Force Mean: 0.00139; Std. dev.: 0.00268
AABB Trees No Creation Mean: 0.00265; Std. dev.: 0.00526
AABB Trees Mean: 0.03471; Std. dev.: 0.00207
Old AABB Trees Mean: 0.28774; Std. dev.: 0.00624

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

plt.plot([0.00139, 0.00200, 0.01090, 0.15937, 2.37237, 40.64899], label='Brute Force')
plt.plot([0.00265, 0.00239, 0.00279, 0.00247, 0.00232, 0.00244, 0.00310], label='AABB Trees No Creation')
plt.plot([0.03471, 0.06843, 0.94646, 4.70398, 77.21539], label='AABB Trees')
plt.plot([0.2877, 1.60006, 11.77460], label='Old AABB Trees')
plt.show()
