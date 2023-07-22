"""
Rubiks image transformations
"""

import numpy as np
import cv2
from typing import Callable


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
        """
        return self.__apply_to_all_pixels(image, self.transform_pixel, *args, **kwargs)

    def transform_pixel(self, pixel: np.array, *args, **kwargs) -> np.array:
        """
        Main transform method for a pixel.
        """
        return pixel

    def __apply_to_all_pixels(self, image: cv2.Mat, transformation: Callable, *args, **kwargs) -> cv2.Mat:
        """
        Apply a pixel transform method to all pixels in an image.
        """
        height, width = image.shape[:2]
        transformed_image = image.copy()
        for x in range(width):
            for y in range(height):
                # Find the current pixel:
                original_pixel = image[y, x]
                # Transform it:
                transformed_image[y, x] = self.transform_pixel(original_pixel, *args, **kwargs)
