# distance3d

Distance computation in 3D.

## Installation

Install the package with

    pip install -e .

## Unit Tests

Install dependencies with

    pip install -e .[test]

Run unit tests with

    pytest --cov-report html --cov=distance3d distance3d

You will find the coverage report in `htmlcov/index.html`.

## Literature

These implementations are mostly based on

* Christer Ericson: Real-Time Collision Detection, CRC Press, 2004.
* David H. Eberly: 3D Game Engine Design, CRC Press, 2006.

Some features related to the GJK algorithm have been inspired by
[Bullet](https://github.com/bulletphysics/bullet3/) (zlib license) and are
marked as such in the source code.

The EPA algorithm is adapted from
[Kevin Moran's GJK implementation](https://github.com/kevinmoran/GJK)
(MIT License or Unlicense).
