#!/usr/bin/env python
import warnings
from setuptools import setup


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    setup_config = dict(
        name="distance3d",
        version="0.0.0",  # TODO load from module
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

    try:
        from Cython.Build import cythonize
        import numpy
        cython_config = dict(
            ext_modules=cythonize("distance3d/_gjk.pyx"),
            zip_safe=False,
            compiler_directives={"language_level": "3"},
            include_dirs=[numpy.get_include()],
            extra_compile_args=[
                "-O3",
                "-Wno-cpp", "-Wno-unused-function"
            ]
        )
        setup_config.update(cython_config)
    except ImportError:
        warnings.warn("Cython or NumPy is not available.")
    setup(**setup_config)
