"""
Rubiks image transformations
"""

import numpy as np
import cv2
from typing import Any, Callable
from .pallete_class import Pallete


class Transformation:
    """
    An abstract transformation. Lays out the general flow of actual transformations.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize.
        """
        pass

    def transform_image(self, image: cv2.Mat, *args, **kwargs) -> cv2.Mat:
        """
        Main transform method for an image.
        :param image: an image.
        :return: a transformed image.
        """
        return self.__apply_to_all_pixels(image, self.transform_pixel, *args, **kwargs)

    def transform_pixel(self, pixel: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Main transform method for a pixel.
        :param pixel: a pixel.
        :return: a transformed pixel.
        """
        return pixel

    def __apply_to_all_pixels(
        self,
        image: cv2.Mat,
        transformation: Callable[[np.ndarray, Any], np.ndarray],
        *args,
        **kwargs
    ) -> cv2.Mat:
        """
        Apply a pixel transform method to all pixels in an image.
        :param image: an image.
        :param transformation: the method used to transform each pixel of the image.
        :return: the transformed image.
        """
        height, width = image.shape[:2]
        transformed_image = image.copy()
        for x in range(width):
            for y in range(height):
                # Find the current pixel:
                original_pixel = image[y, x]
                # Transform it:
                transformed_image[y, x] = transformation(
                    original_pixel, *args, **kwargs
                )


class PalleteTransformation(Transformation):
    """
    Generic transformation where the transformation of each pixel depends only on the pixel itself and a given pallete.
    """

    def __init__(self, pallete: Pallete) -> None:
        super().__init__()

        # Store the pallete:
        self.pallete = pallete

    def transform_image(self, image: cv2.Mat) -> cv2.Mat:
        return super().transform_image(image, self.pallete)

    def transform_pixel(self, pixel: np.ndarray) -> np.ndarray:
        return super().transform_pixel(pixel, self.pallete)

    def __apply_to_all_pixels(
        self,
        image: cv2.Mat,
        transformation: Callable[[np.ndarray, Pallete], np.ndarray],
        pallete: Pallete,
    ) -> cv2.Mat:
        return super().__apply_to_all_pixels(image, transformation, pallete)
