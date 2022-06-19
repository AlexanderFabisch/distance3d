![continuous integration](https://github.com/AlexanderFabisch/distance3d/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/AlexanderFabisch/distance3d/branch/master/graph/badge.svg?token=GJE5ZMVVB8)](https://codecov.io/gh/AlexanderFabisch/distance3d)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6509736.svg)](https://doi.org/10.5281/zenodo.6509736)

# distance3d

Distance computation and collision detection in 3D.

<table>
<tr>
    <td>Robot collision detection</td>
    <td>Capsule collision detection</td>
</tr>
<tr>
    <td><img src="https://raw.githubusercontent.com/AlexanderFabisch/distance3d/master/doc/source/_static/robot_collision_detection.png" width=100% /></td>
    <td><img src="https://raw.githubusercontent.com/AlexanderFabisch/distance3d/master/doc/source/_static/capsule_collisions.png" width=100% /></td>
</tr>
<tr>
    <td><a href="https://github.com/AlexanderFabisch/distance3d/blob/master/examples/visualizations/vis_robot_collision_objects.py">vis_robot_collision_objects.py</a></td>
    <td><a href="https://github.com/AlexanderFabisch/distance3d/blob/master/examples/visualizations/vis_capsules_benchmark.py">vis_capsules_benchmark.py</a></td>
</tr>
</table>

## Features

* Collision detection and distance computation with GJK.
* Calculation of penetration depth with EPA.
* Collision detection and calculation of penetration depth with MPR.
* Various specific distance calculations for points, lines, line segments,
  planes, triangles, rectangles, circles, disks, boxes, cylinders, ellipsoids,
  ...
* Broad phase collision detection with bounding volume hierarchy (AABB tree).
* Self-collision detection for robots.

## Dependencies

distance3d relies on numba to speed up computations. numba in its latest
version requires at least Python 3.7 and NumPy 1.18. See [here](
https://numba.readthedocs.io/en/stable/user/installing.html#compatibility)
for current requirements. Required Python libraries will automatically be
installed during installation of distance3d.

## Installation

Install the package with

    pip install -e .

or from PyPI with

    pip install distance3d

## Unit Tests

Install dependencies with

    pip install -e .[test]

Run unit tests with

    NUMBA_DISABLE_JIT=1 pytest

You will find the coverage report in `htmlcov/index.html`.

## API Documentation

Install dependencies with

    pip install -e .[doc]

Build API documentation with

    cd doc
    make html

You will find the documentation in `doc/build/html/index.html`.

## Licenses

These implementations are mostly based on

* Christer Ericson: Real-Time Collision Detection, CRC Press, 2004.
* David H. Eberly: 3D Game Engine Design, CRC Press, 2006.

and accompanying implementations. These are marked as such.

The distance computation between a line and a circle is based on David Eberly's
implementation, Copyright (c) 1998-2022 David Eberly, Geometric Tools,
Redmond WA 98052, distributed under the Boost Software License, Version 1.0.

The GJK algorithm is a translation to Python of the translation to C of the
original Fortran implementation. The C implementation is from Diego Ruspini.
It is available from http://realtimecollisiondetection.net/files/gilbert.c

Some features related to the GJK algorithm have been inspired by
[Bullet](https://github.com/bulletphysics/bullet3/) (zlib license) and are
marked as such in the source code.

The EPA algorithm is adapted from
[Kevin Moran's GJK implementation](https://github.com/kevinmoran/GJK)
(MIT License or Unlicense).

The GJK intersection test and MPR algorithm are based on libccd (for details,
see https://github.com/danfis/libccd). For the original code the copyright is
of Daniel Fiser <danfis@danfis.cz>. It has been released under 3-clause BSD
license.

The translation to Python has been done by Alexander Fabisch and the glue
code around it is licensed under the 3-clause BSD license.
