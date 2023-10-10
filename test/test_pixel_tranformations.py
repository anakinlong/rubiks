import numpy as np
import unittest

from rubiks.constants import RUBIKS_PALETTE
from rubiks.palette_class import PaletteWeights
from rubiks.pixel_transformations import recolour_closest, recolour_closest_greyscale


class RecolourClosestTest(unittest.TestCase):
    """
    Test the recolour_closest function.
    """

    def setUp(self) -> None:

        self.pixel = np.array([200, 200, 200], dtype=np.uint8)
        self.palette = RUBIKS_PALETTE
        self.weights = PaletteWeights({colour_name: 1 for colour_name in self.palette.names})

    def test_regular(self) -> None:
        """
        Test that recolouring a pixel returns the expected colour.
        """
        np.testing.assert_array_equal(
            recolour_closest(self.pixel, self.palette, self.weights), np.array([255, 255, 255])
        )


class RecolourClosestGreyscaleTest(unittest.TestCase):
    """
    Test the recolour_closest_greyscale function.
    """

    def setUp(self) -> None:

        self.pixel = np.array([20, 200, 200], dtype=np.uint8)
        self.palette = RUBIKS_PALETTE
        self.weights = PaletteWeights({colour_name: 1 for colour_name in self.palette.names})

    def test_regular(self) -> None:
        """
        Test that recolouring a pixel returns the expected colour.
        """
        np.testing.assert_array_equal(
            recolour_closest_greyscale(self.pixel, self.palette, self.weights), np.array([0, 213, 255])
        )
