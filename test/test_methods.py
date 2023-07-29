import cv2
import numpy as np
import unittest

from rubiks.methods import rubiks_dimension_estimator, resize_image


class RubiksDimensionEstimatorTest(unittest.TestCase):
    """
    Test some use cases for the rubiks_dimension_estimator function.
    """

    def setUp(self) -> None:

        self.n_cubes = 400
        self.width = 3
        self.height = 2
        self.image = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
            ],
            dtype=np.uint8,
        )

    def test_regular_cube_size_3(self) -> None:
        """
        Test an ordinary use case for rubiks_dimension_estimator with a cube size of 3.
        """
        self.assertEqual(rubiks_dimension_estimator(100, cube_size=3, width=1920, height=1080), (39, 21))

    def test_regular_cube_size_4(self) -> None:
        """
        Test an ordinary use case for rubiks_dimension_estimator with a cube size of 4.
        """
        self.assertEqual(rubiks_dimension_estimator(100, cube_size=4, width=1920, height=1080), (52, 28))

    def test_width_height_image_equivalent(self) -> None:
        """
        Make sure that supplying a width and height is equivalent to supplying an image with those dimensions.
        """
        self.assertEqual(
            rubiks_dimension_estimator(self.n_cubes, width=self.width, height=self.height),
            rubiks_dimension_estimator(self.n_cubes, image=self.image),
        )

    def test_bad_arguments(self) -> None:
        """
        Test that supplying an incorrect set of keyword arguments raises a ValueError.
        """
        with self.assertRaises(ValueError):
            rubiks_dimension_estimator(self.n_cubes, width=self.width, image=self.image)


class ResizeImageTest(unittest.TestCase):
    """
    Test some use cases for the resize_image function.
    """

    def setUp(self) -> None:

        self.image = np.array(
            [
                [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
            ],
            dtype=np.uint8,
        )
        self.width = 3
        self.height = 2
        self.scale = 10

    def test_regular_enlarge(self) -> None:
        """
        Test a regular use case for resize_image where we are enlarging.
        """
        equivalent_image = np.array(
            [
                [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]],
                [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2]],
                [[3, 3, 3], [3, 3, 3], [4, 4, 4], [4, 4, 4], [5, 5, 5], [5, 5, 5]],
                [[3, 3, 3], [3, 3, 3], [4, 4, 4], [4, 4, 4], [5, 5, 5], [5, 5, 5]],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(resize_image(self.image, scale=2), equivalent_image)

    def test_regular_shrink(self) -> None:
        """
        Test a regular use case for resize_image where we are enlarging.
        """
        image = np.array(
            [
                [[0, 0, 0], [10, 10, 10], [20, 20, 20]],
                [[30, 30, 30], [40, 40, 40], [50, 50, 50]],
            ],
            dtype=np.uint8,
        )
        equivalent_image = np.array(
            [
                [[3, 3, 3], [17, 17, 17]],
                [[33, 33, 33], [47, 47, 47]],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(resize_image(image, new_width=2, new_height=2), equivalent_image)

    def test_scale_1(self) -> None:
        """
        Test that resizing with a scale of 1 returns the same image.
        """
        np.testing.assert_array_equal(resize_image(self.image, scale=1), self.image)

    def test_width_height_scale_equivalent(self) -> None:
        """
        Make sure that supplying a width and height is equivalent to supplying an equivalent scale.
        """
        np.testing.assert_array_equal(
            resize_image(self.image, scale=2), resize_image(self.image, new_width=6, new_height=4)
        )

    def test_enlarge_interpolation(self) -> None:
        """
        Test that INTER_NEAREST is used by default when enlarging.
        """
        equivalent_image = cv2.resize(self.image, (6, 4), interpolation=cv2.INTER_NEAREST)
        np.testing.assert_array_equal(resize_image(self.image, new_width=6, new_height=4), equivalent_image)

    def test_shrink_interpolation(self) -> None:
        """
        Test that INTER_AREA is used by default when shrinking.
        """
        equivalent_image = cv2.resize(self.image, (2, 2), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(resize_image(self.image, new_width=2, new_height=2), equivalent_image)

    def test_bad_arguments(self) -> None:
        """
        Test that supplying an incorrect set of keyword arguments raises a ValueError.
        """
        with self.assertRaises(ValueError):
            resize_image(self.image, new_width=self.width, scale=self.image)
