#!/usr/bin/env python
import os
from pathlib import Path

import numpy
from setuptools import Extension, setup
from Cython.Distutils import build_ext

# Allow user to specify a compiler; default to gcc if not provided
os.environ.setdefault("CC", "gcc")

source_location = Path("topocalc") / "core_c"

extensions = [
    Extension(
        "topocalc.topo_core",
        sources=[
            str(source_location / "topo_core.pyx"),
            str(source_location / "hor1d.c"),
        ],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-fopenmp", "-O3"],
        extra_link_args=["-fopenmp", "-O3"],
    )
]

setup(
    name="topocalc",
    cmdclass={"build_ext": build_ext},
    ext_modules = extensions
)
