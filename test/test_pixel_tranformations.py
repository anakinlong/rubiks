import numpy as np
import unittest

from rubiks.constants import RUBIKS_PALETTE
from rubiks.pixel_class import Pixel
from rubiks.palette_class import PaletteWeights, CombinedPalette
from rubiks.pixel_transformations import recolour_closest, recolour_closest_greyscale, recolour_closest_combined


class RecolourClosestTest(unittest.TestCase):
    """
    Test the recolour_closest function.
    """

    def setUp(self) -> None:

        self.pixel = Pixel(np.array([200, 200, 200], dtype=np.uint8))
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

        self.pixel = Pixel(np.array([20, 200, 200], dtype=np.uint8))
        self.palette = RUBIKS_PALETTE
        self.weights = PaletteWeights({colour_name: 1 for colour_name in self.palette.names})

    def test_regular(self) -> None:
        """
        Test that recolouring a pixel returns the expected colour.
        """
        np.testing.assert_array_equal(
            recolour_closest_greyscale(self.pixel, self.palette, self.weights), np.array([0, 213, 255])
        )


class RecolourClosestCombinedTest(unittest.TestCase):
    """
    Test the recolour_closest_combined function.
    """

    def setUp(self) -> None:

        self.palette = RUBIKS_PALETTE
        self.weights = PaletteWeights({colour_name: 1 for colour_name in self.palette.names})
        self.combined_palette = CombinedPalette(self.palette)
        self.combined_weights = PaletteWeights({colour_name: 1 for colour_name in self.combined_palette.names})

    def test_regular_original(self) -> None:
        """
        Test that recolouring a pixel returns the expected colour when the expected colour is in the original palette.
        """
        pixel = Pixel(np.array([20, 200, 200], dtype=np.uint8))
        np.testing.assert_array_equal(
            recolour_closest_combined(pixel, self.palette, self.weights, self.combined_palette, self.combined_weights),
            np.array([0, 213, 255]),
        )

    def test_regular_combined(self) -> None:
        """
        Test that recolouring a pixel returns the expected colour when the expected colour is in the combined palette.
        """
        pixel = Pixel(np.array([200, 200, 200], dtype=np.uint8), (0, 0))
        np.testing.assert_array_equal(
            recolour_closest_combined(pixel, self.palette, self.weights, self.combined_palette, self.combined_weights),
            np.array([72, 155, 0]),
        )
