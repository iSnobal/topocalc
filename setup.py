#!/usr/bin/env python
import os
from pathlib import Path

import numpy
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except Exception:
    cythonize = None
    USE_CYTHON = False

print(f"Using Cython {USE_CYTHON}")
ext_suffix = ".pyx" if USE_CYTHON else ".c"

# Allow user to specify a compiler; default to gcc if not provided
os.environ.setdefault("CC", "gcc")

extension_params = dict(
    extra_compile_args=["-fopenmp", "-O3"],
    extra_link_args=["-fopenmp", "-O3"],
    include_dirs=[numpy.get_include()],
)

loc = Path("topocalc") / "core_c"
sources = [str(loc / f"topo_core{ext_suffix}"), str(loc / "hor1d.c")]

extensions = [
    Extension(
        "topocalc.core_c.topo_core",
        sources=sources,
        **extension_params,
    )
]

if USE_CYTHON:
    ext_modules = cythonize(
        extensions, compiler_directives={"language_level": "3"}
    )
else:
    ext_modules = extensions

setup(ext_modules=ext_modules)
