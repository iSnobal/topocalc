#!/usr/bin/env python
import os
import unittest
from sys import platform

import numpy as np

from topocalc.viewf import viewf


class TestViewf(unittest.TestCase):
    """Tests for `viewf` package."""

    def test_theory_edge(self):
        """Test with infinite edge dem"""

        dem = np.ones((50, 50))
        dem[:, :25] = 100000

        svf, tvf = viewf(dem, spacing=10, nangles=360)

        # The top should all be ones with 100% sky view
        # OSX seems to have some difficulty with the edge
        # where linux does not. It is close to 1 but not quite
        if platform == "darwin":
            np.testing.assert_allclose(
                svf[:, :24], np.ones_like(svf[:, :24]), atol=1e-2
            )

            # The edge should be 50% or 0.5 svf
            np.testing.assert_allclose(
                svf[:, 25], 0.5 * np.ones_like(svf[:, 25]), atol=1e-2
            )

        else:
            np.testing.assert_array_equal(svf[:, :24], np.ones_like(svf[:, :24]))

            # The edge should be 50% or 0.5 svf
            np.testing.assert_allclose(
                svf[:, 25], 0.5 * np.ones_like(svf[:, 25]), atol=1e-3
            )

    def test_viewf_errors_dem(self):
        """Test viewf dem errors"""

        self.assertRaises(ValueError, viewf, np.ones(10), 10)

    def test_viewf_errors_angles(self):
        """Test viewf nangles errors"""

        self.assertRaises(ValueError, viewf, np.ones((10, 1)), 10, nangles=10)

    def test_viewf_errors_sin_slope(self):
        """Test viewf sin_slope errors"""

        self.assertRaises(
            ValueError, viewf, np.ones((10, 1)), 10, sin_slope=10 * np.ones((10, 1))
        )


class TestViewfLakes(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.spacing = 50.0
        self.nangles = 72
        self.tolerance = 0.05
        self.edge_buffer = 5

        base_path = os.path.dirname(os.path.abspath(__file__))
        lakes_dir = os.path.join(base_path, "Lakes")

        self.topo_path = os.path.join(lakes_dir, "dem.npy")
        self.gold_svf_path = os.path.join(lakes_dir, "sky_view.npy")

        if not os.path.exists(self.topo_path) or not os.path.exists(self.gold_svf_path):
            self.skipTest("Lakes test data not found.")

        self.dem = np.load(self.topo_path).astype("double")
        self.gold_svf = np.load(self.gold_svf_path).astype("double")

    def test_lakes_svf_match(self):

        svf_calc, _ = viewf(self.dem, spacing=self.spacing, nangles=self.nangles)

        # NOTE applying a small buffer to ignore edge errros
        b = self.edge_buffer
        svf_calc = svf_calc[b:-b, b:-b]
        gold_svf = self.gold_svf[b:-b, b:-b]
        svf_calc = np.clip(svf_calc, 0, 1)

        np.testing.assert_allclose(
            svf_calc,
            gold_svf,
            atol=self.tolerance,
            err_msg=f"Sky view calc exceeds tolerance of {self.tolerance}.",
        )
