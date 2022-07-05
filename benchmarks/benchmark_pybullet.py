import numpy as np
import pybullet as pb
import pytransform3d.rotations as pr
from distance3d import random, colliders, benchmark, gjk, mpr


def test_benchmark_pybullet(random_state, n_collision_objects, gui=False):
    pcid = pb.connect(pb.GUI if gui else pb.DIRECT)
    collision_objects = []
    for _ in range(n_collision_objects):
        cylinder2origin, radius, length = random.rand_cylinder(
            random_state, center_scale=5, min_radius=0.1, min_length=0.5)
        pos = cylinder2origin[:3, 3]
        orn = pr.quaternion_xyzw_from_wxyz(
            pr.quaternion_from_matrix(cylinder2origin[:3, :3]))
        collision = pb.createCollisionShape(
            shapeType=pb.GEOM_CYLINDER, radius=radius, height=length,
            physicsClientId=pcid)
        multibody = pb.createMultiBody(
            baseMass=1, baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=collision, physicsClientId=pcid)
        pb.resetBasePositionAndOrientation(
            multibody, pos, orn, physicsClientId=pcid)
        collision_objects.append(multibody)

    distances = []

    timer = benchmark.Timer()
    timer.start("pybullet")
    #pb.performCollisionDetection(pcid)
    for c1 in collision_objects:
        for c2 in collision_objects:
            dist = pb.getClosestPoints(c1, c2, np.inf, physicsClientId=pcid)[0][8]
            #dist = len(pb.getContactPoints(c1, c2, physicsClientId=pcid)) > 0
            distances.append(dist)
    duration = timer.stop("pybullet")

    if gui:
        while True:
            import time
            time.sleep(1)

    pb.disconnect(physicsClientId=pcid)
    return duration, distances


def test_benchmark_distance3d(random_state, n_collision_objects, gui=False):
    collision_objects = []
    for _ in range(n_collision_objects):
        cylinder2origin, radius, length = random.rand_cylinder(
            random_state, center_scale=5, min_radius=0.1, min_length=0.5)
        collision_object = colliders.Cylinder(cylinder2origin, radius, length)
        collision_objects.append(collision_object)

    distances = []

    timer = benchmark.Timer()
    timer.start("distance3d")
    for c1 in collision_objects:
        for c2 in collision_objects:
            dist = gjk.gjk(c1, c2)[0]
            #dist = gjk.gjk_intersection(c1, c2)
            #dist = mpr.mpr_intersection(c1, c2)
            distances.append(dist)
    duration = timer.stop("distance3d")

    if gui:
        import pytransform3d.visualizer as pv
        fig = pv.figure()
        for c in collision_objects:
            c.make_artist()
            c.artist_.add_artist(fig)
        fig.view_init()
        fig.show()

    return duration, distances


n_collision_objects = 50
random_state = np.random.RandomState(31)
duration, distances = test_benchmark_pybullet(random_state, n_collision_objects, gui=False)
print(f"PyBullet: {duration}")
print(distances[:20])
random_state = np.random.RandomState(31)
duration, distances = test_benchmark_distance3d(random_state, n_collision_objects, gui=False)
print(f"distance3d: {duration}")
print(distances[:20])
