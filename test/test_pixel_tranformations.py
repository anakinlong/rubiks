import numpy as np
import unittest

from rubiks.constants import RUBIKS_PALETTE
from rubiks.pixel_transformations import recolour_closest_weighted


class RecolourClosestWeightedTest(unittest.TestCase):
    """
    Test the recolour_closest_weighted function.
    """

    def setUp(self) -> None:

        self.pixel = np.array([200, 200, 200], dtype=np.uint8)
        self.palette = RUBIKS_PALETTE
        self.weights = {colour_name: 1 for colour_name in self.palette.names}

    def test_regular(self) -> None:
        """
        Test that recolouring a pixel returns the expected colour.
        """
        np.testing.assert_array_equal(
            recolour_closest_weighted(self.pixel, self.palette, self.weights), np.array([255, 255, 255])
        )
