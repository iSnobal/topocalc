"""
C implementation of some radiation functions
"""
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import cython
import numpy as np
cimport numpy as np

include "illumination_angle.pyx"

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

cdef extern from "topo_core.h":
    void hor1f(int n, double *z, int *h)
    void hor1b(int n, double *z, int *h)
    void horval(int n, double *z, double delta, int *h, double *hcos)
    void hor2d(int n, int m, double *z, double delta, bint forward, double *hcos)

def c_hor1d(
    np.ndarray[double, ndim=1, mode="c"] z not None,
    double spacing,
    bint forward,
    np.ndarray[double, ndim=1, mode="c"] hcos not None
):
    """
    Call the function hor1f in hor1f.c

    Args:
        z: elevation array
        spacing: grid spacing
        forward: whether to process forward or backward
        hcos: cosine angle of horizon array (output)

    Returns
        hcos: cosine angle of horizon array changed in place
    """
    cdef:
        int n = z.shape[0]
        np.ndarray[double, ndim=1, mode="c"] z_arr
        np.ndarray[int, ndim=1, mode="c"] h

    # Ensure consistent memory layout
    z_arr = np.ascontiguousarray(z, dtype=np.float64)

    # Integer array for horizon index
    h = np.empty(n, dtype=np.int32)

    # Call the appropriate C function based on direction
    if forward:
        hor1f(n, &z_arr[0], &h[0])
    else:
        hor1b(n, &z_arr[0], &h[0])

    # Calculate horizon values
    horval(n, &z_arr[0], spacing, &h[0], &hcos[0])

def c_hor2d(
    np.ndarray[double, ndim=2, mode="c"] z not None,
    double spacing,
    bint forward,
    np.ndarray[double, ndim=2, mode="c"] hcos not None
):
    """
    Call the function hor2d in topo_core.c

    Args:
        z: elevation array
        spacing: grid spacing
        forward: whether to process forward or backward
        hcos: cosine angle of horizon array (output)

    Returns
        hcos: cosine angle of horizon array changed in place
    """
    cdef:
        int nrows = z.shape[0]
        int ncols = z.shape[1]
        np.ndarray[double, ndim=2, mode="c"] z_arr

    # Ensure consistent memory layout
    z_arr = np.ascontiguousarray(z, dtype=np.float64)

    # Call the hor2d C function
    hor2d(nrows, ncols, &z_arr[0,0], spacing, forward, &hcos[0,0])
