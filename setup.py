#!/usr/bin/env python
import warnings
from setuptools import setup, find_packages
from setuptools.extension import Extension


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    setup_config = dict(
        name="distance3d",
        version="0.0.0",  # TODO load version from file
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
        packages=find_packages(),
        install_requires=["numpy", "scipy", "matplotlib", "pytransform3d",
                          "open3d", "aabbtree"],
        extras_require={
          "test": ["pytest", "pytest-cov"]
        })
    try:
        from Cython.Build import cythonize
        import numpy
        extension = Extension(
            "distance3d._gjk",
            ["distance3d/_gjk.pyx"],
            include_dirs=[".", "distance3d", numpy.get_include()],
            extra_compile_args=["-O3", "-Wno-cpp", "-Wno-unused-function"],
            language="c",
            compiler_directives={"language_level": "3"},
            language_level=3,
        )
        cython_config = dict(
            ext_modules=cythonize([extension]),
            zip_safe=False,
        )
        setup_config.update(cython_config)
    except ImportError:
        warnings.warn("Cython or NumPy is not available. "
                      "Install it if you want fast DMPs.")

    setup(**setup_config)
