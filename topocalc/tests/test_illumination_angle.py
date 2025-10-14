import unittest

import numpy as np

from topocalc.gradient import gradient_d8
from topocalc.illumination_angle import illumination_angle

ZENITH = 45
AZIMUTH = 0


class TestIlluminationAngle(unittest.TestCase):
    # with self.dx and self.dy equal to 1, the cardinal direction
    # slope values will be np.pi/4 as one of the differences will be zero
    dx = 1
    dy = 1

    def compare_dem(self, dem: np.ndarray) -> None:
        """
        Compare that illumination angles for given dem are as expected

        Args:
            dem: DEM
        """
        slope, asp = gradient_d8(dem, self.dx, self.dy, aspect_rad=True)

        zenith_rad = np.radians(ZENITH)
        # Math from illumination_angle()
        angle = np.cos(zenith_rad) * (
            np.sqrt((1 - slope[0][0]) * (1 + slope[0][0]))
        ) + (np.sin(zenith_rad) * slope[0][0] * np.cos(AZIMUTH - asp[0][0]))

        # Calculate for both possible method options
        mu = illumination_angle(slope, asp, AZIMUTH, zenith=ZENITH)
        mu_cos = illumination_angle(slope, asp, AZIMUTH, cos_z=np.cos(ZENITH * np.pi / 180))

        # All angles in the "North" case are less than 0
        # Correct to 0 as it is done in the method
        angle = 0 if angle < 0 else angle

        # Actual comparison
        self.assertTrue(np.all(mu == angle))
        np.testing.assert_equal(mu, mu_cos)

    def test_angle_cos_z_bounds_error(self):
        with self.assertRaises(Exception) as context:
            illumination_angle(0, 0, 0, cos_z=1.5)

        self.assertTrue("cos_z must be > 0 and <= 1" in str(context.exception))

    def test_angle_zenith_bounds_error(self):
        with self.assertRaises(Exception) as context:
            illumination_angle(0, 0, 0, zenith=-10)

        self.assertTrue("Zenith must be >= 0 and < 90" in str(context.exception))

    def test_angle_zenith_cos_z_not_specified_error(self):
        with self.assertRaises(Exception) as context:
            illumination_angle(0, 0, 0)

        self.assertTrue("Must specify either cos_z or zenith" in str(context.exception))

    def test_angle_aspect_degrees_error(self):
        with self.assertRaises(Exception) as context:
            illumination_angle(0, 100, 0, zenith=10)

        self.assertTrue("Aspect is not in radians from south" in str(context.exception))

    def test_angle_azimuth_value_error(self):
        with self.assertRaises(Exception) as context:
            illumination_angle(0, 0, 360, zenith=10)

        self.assertTrue(
            "Azimuth must be between -180 and 180 degrees" in str(context.exception)
        )

    def test_angle_west(self):
        self.compare_dem(np.tile(range(10), (10, 1)))

    def test_angle_north(self):
        self.compare_dem(np.tile(range(10), (10, 1)).transpose())

    def test_angle_east(self):
        self.compare_dem(np.fliplr(np.tile(range(10), (10, 1))))

    def test_angle_south(self):
        self.compare_dem(np.flipud(np.tile(range(10), (10, 1)).transpose()))
