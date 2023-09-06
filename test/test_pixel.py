import numpy as np
import unittest

from rubiks.pixel_class import Pixel


class PixelTest(unittest.TestCase):
    """
    Test some use cases for the Pixel class.
    """

    def setUp(self) -> None:

        self.pixel = Pixel([1, 2, 3], coordinates=(0, 127))

    def test_x(self) -> None:
        """
        Test that the values property returns the expected result.
        """
        self.assertEqual(self.pixel.x, 0)

    def test_y(self) -> None:
        """
        Test that the values property returns the expected result.
        """
        self.assertEqual(self.pixel.y, 127)

    def test_array_type(self) -> None:
        """
        Test that supplying something other than an array or iterable raises an error when creating a Pixel.
        """
        with self.assertRaises(TypeError):
            Pixel(None)

    def test_array_element_type(self) -> None:
        """
        Test that supplying an array with a length not equal to 3 raises an error when creating a Pixel.
        """
        with self.assertRaises(ValueError):
            Pixel(np.asarray([1, 2, 3, 4]))
