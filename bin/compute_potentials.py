import argparse
import json
from distance3d import io, distance


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute potentials of tetrahedral mesh and store them in "
                    "JSON format.")
    parser.add_argument("triangle_mesh", type=str,
                        help="Triangle mesh file (e.g., .stl) that defines "
                             "the surface.")
    parser.add_argument("tetrahedral_mesh", type=str,
                        help="Tetrahedral mesh file (.vtk) that defines the "
                             "volume.")
    parser.add_argument("output", type=str,
                        help="Output file (.json) that will contain a list of "
                             "potentials of the vertices of the tetrahedral "
                             "mesh.")
    return parser.parse_args()


def main():
    args = parse_args()

    triangular_vertices, triangles = io.load_mesh(args.triangle_mesh)
    tetrahedral_vertices, tetrahedra = io.load_tetrahedral_mesh(args.tetrahedral_mesh)

    potentials = []
    for v in tetrahedral_vertices:
        distances = [distance.point_to_triangle(v, triangular_vertices[t])[0]
                     for t in triangles]
        min_distance = min(distances)
        potentials.append(min_distance)

    with open(args.output, "w") as f:
        json.dump(potentials, f)


if __name__ == "__main__":
    main()
