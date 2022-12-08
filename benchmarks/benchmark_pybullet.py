import numpy as np
import pybullet as pb
import pytransform3d.rotations as pr
from distance3d import random, colliders, benchmark, gjk, mpr


COLLISION_SHAPES = [
    "cylinder", "box", "capsule", "sphere"
]


def test_benchmark_pybullet(random_state, n_collision_objects, gui=False):
    pcid = pb.connect(pb.GUI if gui else pb.DIRECT)
    collision_objects = []
    for _ in range(n_collision_objects):
        shape_name = COLLISION_SHAPES[random_state.randint(len(COLLISION_SHAPES))]
        args = random.RANDOM_GENERATORS[shape_name](
            random_state, center_scale=5)
        collision, orn, pos = _make_collision_objects_in_pybullet(
            args, pcid, shape_name)
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


def _make_collision_objects_in_pybullet(args, pcid, shape_name):
    if shape_name == "cylinder":
        cylinder2origin, radius, length = args
        pos, orn = _pybullet_pos_orn(cylinder2origin)
        collision = pb.createCollisionShape(
            shapeType=pb.GEOM_CYLINDER, radius=radius, height=length,
            physicsClientId=pcid)
    elif shape_name == "capsule":
        capsule2origin, radius, height = args
        pos, orn = _pybullet_pos_orn(capsule2origin)
        collision = pb.createCollisionShape(
            shapeType=pb.GEOM_CAPSULE, radius=radius, height=height,
            physicsClientId=pcid)
    elif shape_name == "sphere":
        center, radius = args
        pos, orn = center, np.array([0.0, 0.0, 0.0, 1.0])
        collision = pb.createCollisionShape(
            shapeType=pb.GEOM_SPHERE, radius=radius,
            physicsClientId=pcid)
    else:
        assert shape_name == "box"
        box2origin, size = args
        pos, orn = _pybullet_pos_orn(box2origin)
        collision = pb.createCollisionShape(
            shapeType=pb.GEOM_BOX, halfExtents=0.5 * size,
            physicsClientId=pcid)
    return collision, orn, pos


def _pybullet_pos_orn(A2B):
    pos = A2B[:3, 3]
    orn = pr.quaternion_xyzw_from_wxyz(
        pr.quaternion_from_matrix(A2B[:3, :3]))
    return pos, orn


def test_benchmark_distance3d(random_state, n_collision_objects, gui=False):
    collision_objects = []
    for _ in range(n_collision_objects):
        shape_name = COLLISION_SHAPES[random_state.randint(len(COLLISION_SHAPES))]
        args = random.RANDOM_GENERATORS[shape_name](
            random_state, center_scale=5)
        collision_object = colliders.COLLIDERS[shape_name](*args)
        collision_objects.append(collision_object)

    distances = []

    timer = benchmark.Timer()
    timer.start("distance3d")
    for c1 in collision_objects:
        for c2 in collision_objects:
            dist = gjk.gjk_distance(c1, c2)[0]
            #dist = gjk.gjk_intersection(c1, c2)
            #dist = mpr.mpr_intersection(c1, c2)
            distances.append(dist)
    duration = timer.stop("distance3d")

    if gui:
        import pytransform3d.visualizer as pv
        fig = pv.figure()
        for c in collision_objects:
            c.make_artist()
            c.artist.add_artist(fig)
        fig.view_init()
        fig.show()

    return duration, distances


seed = 31
n_collision_objects = 300
duration, distances = test_benchmark_pybullet(
    np.random.RandomState(seed), n_collision_objects, gui=False)
print(f"PyBullet: {duration}")
print(distances[:20])
duration, distances = test_benchmark_distance3d(
    np.random.RandomState(seed), n_collision_objects, gui=False)
print(f"distance3d: {duration}")
print(distances[:20])
