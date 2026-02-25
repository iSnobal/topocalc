import numpy as np
cimport numpy as np
from libc.math cimport cos, sin, sqrt, pi, acos
from cython.parallel import prange


def illumination_angle_c(
    np.ndarray[np.float64_t, ndim=2] slope,
    np.ndarray[np.float64_t, ndim=2] aspect,
    double azimuth,
    double c_theta
):
    """
    Cython implementation of illumination_angle calculation.

    Args:
        slope: 2D numpy array of sine of slope angles sin(S)
        aspect: 2D numpy array of aspect in radians from south
        azimuth: azimuth in degrees to the sun
        c_theta: cosine of the zenith angle

    Returns:
        2D numpy array of the cosine of the local illumination angle
    """
    cdef:
        int rows = slope.shape[0]
        int cols = slope.shape[1]
        np.ndarray[np.float64_t, ndim=2] mu = np.zeros((rows, cols), dtype=np.float64)
        double azimuth_rad = azimuth * pi / 180.0
        double s_theta = sin(acos(c_theta))
        double value
        int i, j

    # Parallel computation using OpenMP
    for i in prange(rows, nogil=True):
        for j in range(cols):
            ## Slope angles are passed in as sin(S)
            # Extract the cosine by using the identity:
            #   cos^2(s) = 1 - sin^2(s)
            # Second identity: 1^2 - sin^2(z) = (1 - sin(z)) * (1 + sin(z))
            #   cos(z) = sqrt((1 - sin(z)) * (1 + sin(z)))

            # Calculate illumination angle at pixel
            value = c_theta * sqrt((1.0 - slope[i, j]) * (1.0 + slope[i, j])) + \
                    s_theta * slope[i, j] * cos(azimuth_rad - aspect[i, j])

            # Ensure values between 0 and 1
            value = (value >= 0.0) * (value <= 1.0) * value + \
                    (value < 0.0) * 0.0 + \
                    (value > 1.0) * 1.0

            mu[i, j] = value

    return mu
