from distance3d.io import load_tetrahedral_mesh


def test_load_tetrahedral_mesh_insole():
    vertices, tetrahedra = load_tetrahedral_mesh("test/data/insole.vtk")
    assert vertices.shape[0] == 88
    assert vertices.shape[1] == 3
    assert tetrahedra.shape[0] == 189
    assert tetrahedra.shape[1] == 4
