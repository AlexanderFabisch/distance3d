# distance3d

Distance computation and collision detection in 3D.

<img src="https://raw.githubusercontent.com/AlexanderFabisch/distance3d/master/doc/source/_static/robot_collision_detection.png" width=50% />

## Installation

Install the package with

    pip install -e .

## Unit Tests

Install dependencies with

    pip install -e .[test]

Run unit tests with

    pytest --cov-report html --cov=distance3d distance3d

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

The GJK algorithm is a translation to Python of the translation to C of the
original Fortran implementation. The C implementation is from Diego Ruspini.
It is available from http://realtimecollisiondetection.net/files/gilbert.c

Some features related to the GJK algorithm have been inspired by
[Bullet](https://github.com/bulletphysics/bullet3/) (zlib license) and are
marked as such in the source code.

The EPA algorithm is adapted from
[Kevin Moran's GJK implementation](https://github.com/kevinmoran/GJK)
(MIT License or Unlicense).

The translation to Python has been done by Alexander Fabisch and the glue
code around it is licensed under the 3-clause BSD license.
