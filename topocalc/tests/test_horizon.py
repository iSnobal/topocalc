import unittest
from pathlib import Path
import numpy as np
from topocalc.horizon import hor2d_c, horizon


class TestHorizon(unittest.TestCase):
    def test_horizon_dem_errors(self):
        dem = np.ones((10))
        with self.assertRaises(ValueError) as context:
            horizon(0, dem, 1)
        self.assertIn("horizon input of dem is not a 2D array", str(context.exception))

    def test_horizon_azimuth_errors(self):
        dem = np.ones((10, 1))
        with self.assertRaises(ValueError) as context:
            horizon(-200, dem, 1)
        self.assertIn(
            "azimuth must be between -180 and 180 degrees", str(context.exception)
        )

    def test_hor2dc_errors(self):
        dem = np.ones((10))
        with self.assertRaises(ValueError) as context:
            hor2d_c(dem, 1)
        self.assertIn("z is not a 2D array", str(context.exception))

    def test_hor2dc_type_errors(self):
        dem = np.float32(np.ones((10, 1)))

        with self.assertRaises(ValueError) as context:
            hor2d_c(dem, 1)

        self.assertIn("z must be of type double", str(context.exception))


class TestHorizonGold(unittest.TestCase):
    DX = 30

    def calc_gold_horizon(self, surface, gold_index):
        distance = self.DX * np.arange(len(surface))
        hgt = surface[gold_index] - surface
        d = distance[gold_index] - distance
        with np.errstate(invalid="ignore"):
            hcos = hgt / np.sqrt(hgt**2 + d**2)
        hcos[np.isnan(hcos)] = 0
        return hcos

    def assert_horizon(self, surf, gold_index):
        hcos_gold = self.calc_gold_horizon(surf, gold_index)
        hcos = horizon(90, surf.reshape(1, -1), self.DX)
        np.testing.assert_allclose(hcos_gold.reshape(1, -1), hcos, rtol=1e-7,
                    atol=1e-7)

    def test_horizon1(self):
        surf = np.array([100.0, 80, 75, 85, 70, 50, 64, 65, 85, 90])
        gold_index = np.array([0, 3, 3, 9, 9, 6, 8, 8, 9, 9])
        self.assert_horizon(surf, gold_index)

    def test_horizon2(self):

        surf = np.array([100.0, 80, 75, 85, 70, 80, 64, 65, 70, 90])
        gold_index = np.array([0, 3, 3, 9, 5, 9, 9, 9, 9, 9])

        self.assert_horizon(surf, gold_index)

    def test_horizon3(self):

        surf = np.array([0.0, 5, 7, 20, 18, 30, 30, 35, 20, 21])
        gold_index = np.array([3, 3, 3, 5, 5, 7, 7, 7, 9, 9])

        self.assert_horizon(surf, gold_index)


class TestHorizonLakesData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # NOTE Only runs test within a certain buffer away from the edge
        # This a typical use case, as one should prepare basin file that is
        # at least 0.2-0.5 km past the extent of the watershed to capture nearby terrain.
        # But presently, since we use a linear interpolation along the line,
        # the first few rows column data will always be in error.
        cls.DX = 50.0
        cls.EDGE_BUFFER = 15

        base_path = Path(__file__).resolve().parent
        cls.lakes_dir = base_path / "Lakes"

        cls.topo_path = cls.lakes_dir / "dem.npy"
        cls.lakes_path = cls.lakes_dir / "topo_horizons.npz"

        cls.dem = np.load(cls.topo_path)
        cls.lakes_data = np.load(cls.lakes_path)

    def test_lakes_horizon_files(self):
        keys = self.lakes_data.files

        for key in keys:
            azimuth = float(key.replace("az_", ""))

            with self.subTest(azimuth=azimuth):
                h_lakes = self.lakes_data[key]
                h_calc = horizon(azimuth, self.dem, self.DX)
                h_calc = np.maximum(h_calc, 0)

                b = self.EDGE_BUFFER
                h_lakes = h_lakes[b:-b, b:-b]
                h_calc = h_calc[b:-b, b:-b]

                np.testing.assert_allclose(
                    h_calc,
                    h_lakes,
                    rtol=1e-3,
                    atol=1e-3,
                    err_msg=f"Potential error at {azimuth} degrees.",
                )
