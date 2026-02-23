import unittest
import numpy as np
import os
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
        np.testing.assert_array_almost_equal(hcos_gold.reshape(1, -1), hcos, decimal=6)

    def test_horizon1(self):
        surf = np.array([100.0, 80, 75, 85, 70, 50, 64, 65, 85, 90])
        gold_index = np.array([0, 3, 3, 9, 9, 6, 8, 8, 9, 9])
        self.assert_horizon(surf, gold_index)


class TestHorizonLakes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.DX = 50.0
        cls.EDGE_BUFFER = 2

        base_path = os.path.dirname(os.path.abspath(__file__))
        cls.lakes_dir = os.path.join(base_path, "Lakes")
        cls.gold_dir = os.path.join(cls.lakes_dir, "horizon_files")
        topo_path = os.path.join(cls.lakes_dir, "topo.txt")

        if not os.path.exists(topo_path):
            raise FileNotFoundError(f"Missing test DEM: {topo_path}")

        cls.dem = np.loadtxt(topo_path)

    def test_lakes_horizon_files(self):
        gold_files = [
            f
            for f in os.listdir(self.gold_dir)
            if f.startswith("horizon_") and f.endswith(".txt")
        ]
        self.assertGreater(len(gold_files), 0, "No files found.")

        for filename in gold_files:
            azimuth = float(filename.split("_")[1].replace(".txt", ""))

            # NOTE Only runs test within a certain buffer away from the edge
            with self.subTest(azimuth=azimuth):
                h_gold = np.loadtxt(os.path.join(self.gold_dir, filename))
                h_calc = horizon(azimuth, self.dem, self.DX)
                b = self.EDGE_BUFFER
                h_gold_inner = h_gold[b:-b, b:-b]
                h_calc_inner = h_calc[b:-b, b:-b]

                np.testing.assert_array_almost_equal(
                    h_calc_inner,
                    h_gold_inner,
                    decimal=2,
                    err_msg=f"Potential error at {azimuth} degrees.",
                )


if __name__ == "__main__":
    unittest.main()
