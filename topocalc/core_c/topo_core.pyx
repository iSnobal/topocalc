"""
C implementation of some radiation functions
"""
# distutils: language = c
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

include "illumination_angle.pyx"

# Initialize NumPy API
np.import_array()

cdef extern from "horizon.c":
    int horizon_1d(int n, double *z, int *h, bint forward)
    void horizon_cos(int n, double *z, double delta, int *h, double *hcos)
    void horizon_2d(int nrows, int ncols, double *z, double delta, bint forward, double *hcos)

def c_horizon_1d(double[::1] z, double spacing, bint forward, double[::1] hcos):
    """
    Computes 1D horizon indices and converts them into cosine values.

    Args:
        z (ndarray): A 1D elevation data array.
        spacing (float): The spacing value between grid points.
        forward (bool): A boolean flag indicating the direction of computation.
        hcos (ndarray): A 1D array result array to use
    """
    cdef int n = z.shape[0]

    # Integer buffer for indices
    cdef int[::1] h = np.empty(n, dtype=np.intc)

    # Compute horizon indices
    horizon_1d(n, &z[0], &h[0], forward)

    # Convert indices to cosine values
    horizon_cos(n, &z[0], spacing, &h[0], &hcos[0])

def c_horizon_2d(double[:, ::1] z, double spacing, bint forward, double[:, ::1] hcos):
    """
    Compute horizon value for a 2D array of elevations.

    Args:
        z (ndarray): A 1D elevation data array.
        spacing (float): The spacing value between grid points.
        forward (bool): A boolean flag indicating the direction of computation.
        hcos (ndarray): A 1D array result array to use
    """
    horizon_2d(z.shape[0], z.shape[1], &z[0, 0], spacing, forward, &hcos[0, 0])
