import numpy as np
import unittest

from rubiks.palette_class import Palette, PaletteWeights
from rubiks.transformations import Transformation, NoneTransformation, RecolourClosest


class TransformationTest(unittest.TestCase):
    """
    Test the Transformation class.
    """

    def setUp(self) -> None:

        self.image = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
                [[6, 6, 6], [7, 7, 7], [8, 8, 8]],
            ],
            dtype=np.uint8,
        )

    def test_regular(self) -> None:
        """
        Test that attempting to transform an image using the base Transformation class raises a NotImplementedError.
        """
        with self.assertRaises(NotImplementedError):
            Transformation.transform_image(self.image)

    def test_inheritance(self) -> None:
        """
        Test that inheriting from Transformation without defining a _transform_pixel method results in a
        NotImplementedError when applying that transformation to an image.
        """

        class NewTransformation(Transformation):
            """
            A new transformation
            """

        with self.assertRaises(NotImplementedError):
            NewTransformation.transform_image(self.image)


class NoneTransformationTest(unittest.TestCase):
    """
    Test the NoneTransformation class.
    """

    def setUp(self) -> None:

        self.image = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
                [[6, 6, 6], [7, 7, 7], [8, 8, 8]],
            ],
            dtype=np.uint8,
        )
        self.palette = Palette()

    def test_regular(self) -> None:
        """
        Test that applying the NoneTransformation transformation on an image returns the same image.
        """
        np.testing.assert_array_equal(NoneTransformation.transform_image(self.image, self.palette), self.image)


class RecolourClosestTest(unittest.TestCase):
    """
    Test the RecolourClosest class.
    """

    def setUp(self) -> None:

        self.image = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
                [[6, 6, 6], [7, 7, 7], [8, 8, 8]],
            ],
            dtype=np.uint8,
        )
        self.palette = Palette()

    def test_regular(self) -> None:
        """
        Test that applying the RecolourClosest transformation on an image returns the correct result.
        """
        # TODO write this test once the class is finished

    def test_missing_weights(self) -> None:
        """
        Test that supplying PaletteWeights with one of the colours missing raises a ValueError.
        """
        invalid_weights = PaletteWeights({"white": 1})
        with self.assertRaises(ValueError):
            RecolourClosest.transform_image(self.image, self.palette, palette_weights=invalid_weights)

    def test_extra_weights(self) -> None:
        """
        Test that supplying PaletteWeights with one of the colours missing raises a ValueError.
        """
        invalid_weights = PaletteWeights({"white": 1, "black": 2, "extra": 3})
        with self.assertRaises(ValueError):
            RecolourClosest.transform_image(self.image, self.palette, palette_weights=invalid_weights)
