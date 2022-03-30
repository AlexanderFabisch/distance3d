#!/usr/bin/env python
from setuptools import setup
import distance3d


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    setup(name="distance3d",
          version=distance3d.__version__,
          author="Alexander Fabisch",
          author_email="alexander.fabisch@dfki.de",
          url="https://github.com/AlexanderFabisch/distance3d",
          description="Distance computation in 3D.",
          long_description=long_description,
          long_description_content_type="text/markdown",
          classifiers=[
              "Programming Language :: Python :: 3",
              "License :: OSI Approved :: BSD License",
              "Operating System :: OS Independent",
              "Topic :: Scientific/Engineering :: Mathematics",
              "Topic :: Scientific/Engineering :: Visualization",
          ],
          license="BSD-3-Clause",
          packages=["distance3d"],
          install_requires=["numpy", "scipy", "matplotlib", "pytransform3d",
                            "open3d"],
          extras_require={
              "test": ["pytest", "pytest-cov"]
          })
