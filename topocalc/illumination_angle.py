import numpy as np

from topocalc.topo_core import illumination_angle_c


def illumination_angle(
    slope: np.ndarray,
    aspect: np.ndarray,
    azimuth: float,
    cos_z: float = None,
    zenith: float = None,
) -> np.ndarray:
    """
    Calculate the cosine of the local illumination angle over a DEM.
    Solves the following equation

    cos(ts) = cos(t0) * cos(S) + sin(t0) * sin(S) * cos(phi0 - A)

    where
        t0 is the illumination angle on a horizontal surface
        phi0 is the azimuth of illumination
        S is slope in radians
        A is aspect in radians

    Slope and aspect are expected to come from the IPW gradient command:
        * Slope is stored as sin(S) with range from 0 to 1
        * Aspect is stored as radians from south (aspect 0 is toward the south)
          with range from -pi to pi, with negative values to the west and positive
          values to the east.

    Args:
        slope: numpy array of sine of slope angles sin(S)
        aspect: numpy array of aspect in radians from south
        azimuth: azimuth in degrees to the sun -180..180 (comes from sunang)
        cos_z: cosine of the zenith angle 0..1 (comes from sunang)
        zenith: the solar zenith angle 0..90 degrees

    At least one of the cosz or zenith must be specified.  If both are
    specified the zenith is ignored

    Returns:
        mu: numpy matrix of the cosine of the local illumination angle cos(ts)

    This python function is an interpretation of the IPW shade() function and follows
    as close as possible. All equations are based on Dozier & Frew, 1990.
    'Rapid calculation of Terrain Parameters For Radiation Modeling From Digital
    Elevation Data,' IEEE Transactions on Geoscience and Remote Sensing
    """

    # process the options
    if cos_z is not None:
        if (cos_z <= 0) or (cos_z > 1):
            raise Exception("cos_z must be > 0 and <= 1")

        c_theta = cos_z

    elif zenith is not None:
        if (zenith < 0) or (zenith >= 90):
            raise Exception("Zenith must be >= 0 and < 90")

        zenith *= np.pi / 180.0  # in radians
        c_theta = np.cos(zenith)

    else:
        raise Exception("Must specify either cos_z or zenith")

    if np.max(np.abs(aspect)) > np.pi:
        raise Exception("Aspect is not in radians from south")

    if (azimuth > 180) or (azimuth < -180):
        raise Exception("Azimuth must be between -180 and 180 degrees")

    return illumination_angle_c(slope, aspect, azimuth, c_theta)
