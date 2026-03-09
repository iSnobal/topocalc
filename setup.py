#!/usr/bin/env python
import os
from pathlib import Path

import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

# Give user option to specify local compiler name
if "CC" not in os.environ:
    os.environ["CC"] = "gcc"

print("Compiler set to: " + os.environ["CC"])

extension_params = dict(
    extra_compile_args=[
        '-fopenmp',
        '-O3',
    ],
    extra_link_args=['-fopenmp'],
    include_dirs=[numpy.get_include()]
)

directives = {
    'language_level': "3str",
    'embedsignature': True,
    'boundscheck': False,
    'wraparound': False,
    'initializedcheck': False,
    'cdivision': True,
    'binding': True,
}


source_location = Path("topocalc") / "core_c"

extensions = [
    Extension(
        "topocalc.topo_core",
        sources=[
            str(source_location / "topo_core.pyx"),
        ],
        **extension_params
    )
]

setup(
    name="topocalc",
    ext_modules=cythonize(
        extensions,
        compiler_directives=directives,
        annotate=False,
    ),
)
