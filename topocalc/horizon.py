import numpy as np
import numpy.typing as npt

from topocalc import topo_core
from topocalc.skew import adjust_spacing, skew


def skew_dem(
    dem: npt.NDArray, spacing: float, angle: float, transpose_input: bool = False
) -> tuple[npt.NDArray, float]:
    """
    Skew and transpose the dem for the given angle.
    Also calculate the new spacing given the skew.

    Arguments:
        dem:              DEM elevations
        spacing:          DEM grid spacing
        angle:            Skew angle
        transpose_input:  Whether to transpose before skewing

    Returns: (Tuple)
        - skew and transpose array
        - new spacing adjusted for angle
    """
    if transpose_input:
        dem = dem.transpose()

    spacing = adjust_spacing(spacing, np.abs(angle))
    t = skew(dem, angle, fill_min=True).transpose()

    return t, spacing


def horizon(azimuth: float, dem: npt.NDArray, spacing: float) -> npt.NDArray:
    """
    Calculate horizon angles for one direction. Horizon angles
    are based on Dozier and Frew 1990 and are adapted from the
    IPW C code.

    The coordinate system for the azimuth is 0 degrees is South,
    with positive angles through East and negative values
    through West. Azimuth values must be on the -180 -> 0 -> 180
    range.

    Arguments:
        azimuth:  Angle to find horizon's along this direction
        dem:      DEM elevations
        spacing:  DEM grid spacing

    Returns:
        Cosines of angles to the horizon
    """
    if dem.ndim != 2:
        raise ValueError("horizon input of dem is not a 2D array")

    if azimuth > 180 or azimuth < -180:
        raise ValueError("azimuth must be between -180 and 180 degrees")

    # Initial state for parameters
    is_forward = True
    transpose_output = False
    is_skewed = False
    skew_angle = 0.0

    # Determine transformation parameters
    if azimuth == 90:  # East
        skewed_dem = dem
        adjusted_spacing = spacing
        is_forward = True
    elif azimuth == -90:  # West
        skewed_dem = dem
        adjusted_spacing = spacing
        is_forward = False
    elif azimuth == 0:  # South
        skewed_dem = dem.transpose()
        adjusted_spacing = spacing
        is_forward = True
        transpose_output = True
    elif np.abs(azimuth) == 180:  # North
        skewed_dem = dem.transpose()
        adjusted_spacing = spacing
        is_forward = False
        transpose_output = True
    elif -45 <= azimuth <= 45:
        # South west through south east
        is_skewed = True
        skew_angle = azimuth
        is_forward = True
        skewed_dem, adjusted_spacing = skew_dem(dem, spacing, skew_angle)
    elif azimuth <= -135 or azimuth >= 135:
        # North west and North east
        is_skewed = True
        skew_angle = azimuth + 180 if azimuth < 0 else azimuth - 180
        is_forward = False
        skewed_dem, adjusted_spacing = skew_dem(dem, spacing, skew_angle)
    elif 45 < azimuth < 135:
        # South east through north east
        is_skewed = True
        transpose_output = True
        skew_angle = 90 - azimuth
        is_forward = True
        skewed_dem, adjusted_spacing = skew_dem(
            dem, spacing, skew_angle, transpose_input=True
        )
    elif -135 < azimuth < -45:
        # South west through north west
        is_skewed = True
        transpose_output = True
        skew_angle = -90 - azimuth
        is_forward = False
        skewed_dem, adjusted_spacing = skew_dem(
            dem, spacing, skew_angle, transpose_input=True
        )
    else:
        raise ValueError("azimuth not valid")

    # Ensure memory is contiguous and type is double for the C extension.
    elevations = np.require(skewed_dem, dtype=np.double, requirements=['C', 'A'])
    horizon_cos = np.zeros_like(elevations)

    topo_core.c_horizon_2d(
        elevations,
        np.double(adjusted_spacing),
        is_forward,
        horizon_cos
    )

    if is_skewed:
        horizon_cos = skew(horizon_cos.transpose(), skew_angle, fwd=False)

    if transpose_output:
        horizon_cos = horizon_cos.transpose()

    return horizon_cos


def pyhorizon(dem: npt.NDArray, dx: float) -> tuple[npt.NDArray, int]:
    """Pure python version of the horizon function.

    NOTE: this is fast for small dem's but quite slow
    for larger ones. This is mainly to show that it
    can be done with numpy but requires a bit more to
    remove the for loop over the rows. Also, this just
    calculates the horizon in one direction, need to implement
    the rest of the horizon function for calcuating the
    horizon at an angle.

    Args:
        dem (npt.NDArray): dem for the horizon
        dx (float): spacing for the dem

    Returns:
        [tuple]: cosine of the horizon angle and index
            to the horizon.
    """

    # needs to be a float
    if dem.dtype != np.float64:
        dem = dem.astype(np.float64)

    nrows, ncols = dem.shape
    hcos = np.zeros_like(dem)
    horizon_index = np.zeros_like(dem)

    # distance to each point
    # k=-1 because the distance to the point itself is 0
    distance = dx * np.cumsum(np.tri(ncols, ncols, k=-1), axis=0)
    col_index = np.arange(0, ncols)

    for n in range(nrows):
        surface = dem[n, :]

        m = np.repeat(surface.reshape(1, -1), ncols, axis=0)

        # height change
        height = np.tril(m.T - m)

        # slope
        slope = height / distance

        # horizon location
        hor = np.nanargmax(slope[:, :-1], axis=0)
        hor = np.append(hor, ncols - 1)
        hidx = hor.astype(int)

        horizon_height_diff = surface[hidx] - surface
        horizon_distance_diff = dx * (hor - col_index)

        new_horizon = horizon_height_diff / np.sqrt(
            horizon_height_diff ** 2 + horizon_distance_diff ** 2
        )

        new_horizon[new_horizon < 0] = 0
        new_horizon[np.isnan(new_horizon)] = 0

        hcos[n, :] = new_horizon
        horizon_index[n, :] = hidx

    return hcos, horizon_index
