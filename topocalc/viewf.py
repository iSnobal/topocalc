import numpy as np

from topocalc.gradient import gradient_d8
from topocalc.horizon import horizon


def viewf(
    dem: np.ndarray, spacing: float, nangles: int = 72
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the sky view factor of a dem.

    The sky view factor from equation 7b from Dozier and Frew 1990

    .. math::
        V_d \approx \frac{1}{2\pi} \int_{0}^{2\pi}\left [ cos(S) sin^2{H_\phi}
        + sin(S)cos(\phi-A) \times \left ( H_\phi - sin(H_\phi) cos(H_\phi)
        \right )\right ] d\phi

    terrain configuration factor (tvf) is defined as:
        (1 + cos(slope))/2 - sky view factor

    Based on the paper Dozier and Frew, 1990 and modified from
    the Image Processing Workbench code base (Frew, 1990). The
    Python version of sky view factor will be an almost exact
    replication of the IPW command `viewf` minus rounding errors
    from type and linear quantization.

    The horizon term, H, in equation above is updated to account for
    with-in pixel topography as demonstrated by Dozier (2021).
    See equation 2 in https://doi.org/10.1109/LGRS.2021.3125278

    Args:
        dem: numpy array for the DEM
        spacing: grid spacing of the DEM
        nangles: number of angles to estimate the horizon, defaults
                to 72 angles

    Returns:
        svf: sky view factor
        tcf: terrain configuration factor

    """  # noqa

    if dem.ndim != 2:
        raise ValueError("viewf input of dem is not a 2D array")

    if nangles < 16:
        raise ValueError("viewf number of angles should be 16 or greater")

    # calculate the gradient
    # The slope is returned as radians so convert to sin(S)
    slope, aspect = gradient_d8(dem, dx=spacing, dy=spacing, aspect_rad=True)
    sin_slope = np.sin(slope).astype(np.float32)
    tan_slope = np.tan(slope).astype(np.float32)

    # -180 is North
    angles = np.linspace(-180, 180, num=nangles, endpoint=False)

    # perform the integral
    cos_slope = np.sqrt((1 - sin_slope) * (1 + sin_slope))
    svf = np.zeros_like(sin_slope)
    for angle in angles:

        # horizon angles
        hcos = horizon(angle, dem, spacing)
        h = np.arccos(hcos)
        azimuth = np.radians(angle)

        # cosines of difference between horizon aspect and slope aspect
        cos_aspect = np.cos(azimuth - aspect)

        # update horizon for within-pixel topography (equation 2 in Dozier 2021)
        t = cos_aspect < 0
        h[t] = np.minimum(
            h[t],
            np.arccos(np.sqrt(1 - 1 / (1 + tan_slope[t] ** 2 * cos_aspect[t] ** 2))),
        )

        # integral in equation 7b
        intgrnd = cos_slope * np.sin(h) ** 2 + sin_slope * cos_aspect * (
            h - np.sin(h) * np.cos(h)
        )

        ind = intgrnd > 0
        svf[ind] = svf[ind] + intgrnd[ind]

    svf = svf / len(angles)

    tcf = (1 + cos_slope) / 2 - svf

    return svf, tcf
