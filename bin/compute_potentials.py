import argparse
import json
from distance3d import io, distance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("triangle_mesh", type=str, help="Triangle mesh file")
    parser.add_argument("tetrahedral_mesh", type=str, help="Tetrahedral mesh file")
    parser.add_argument("output", type=str, help="Output file")
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
