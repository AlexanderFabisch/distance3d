import numpy as np
from pytransform3d.transform_manager import TransformManager
from distance3d.urdf_utils import fast_transform_manager_initialization
from numpy.testing import assert_array_almost_equal


def test_fast_transform_manager_initialization():
    tm = TransformManager()
    fast_transform_manager_initialization(tm, [1, 2, 3], "base")
    assert 1 in tm.nodes
    assert 2 in tm.nodes
    assert 3 in tm.nodes
    assert "base" in tm.nodes
    fast_transform_manager_initialization(tm, [4, 5], 1)
    assert 4 in tm.nodes
    assert 5 in tm.nodes
    five2base = tm.get_transform(5, "base")
    assert_array_almost_equal(five2base, np.eye(4))
