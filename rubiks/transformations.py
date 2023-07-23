"""
Rubiks image transformations
"""

import numpy as np
import cv2
from typing import Any, Callable
from .palette_class import Palette, PaletteWeights


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
        **kwargs,
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


class RecolourClosest(Transformation):
    """
    Recolour each pixel to the colour in the palette that is closest geometrically.
    """

    def __init__(self, palette: Palette, weights: PaletteWeights | None = None) -> None:
        super().__init__()

        # Input validation:
        weights = self.__validate_weights(palette, weights)

        # Store the palette and colour weights:
        self.palette = palette
        self.weights = weights

    def transform_image(self, image: cv2.Mat) -> cv2.Mat:
        return super().transform_image(image, self.palette, self.weights)

    def transform_pixel(self, pixel: np.ndarray) -> np.ndarray:

        return super().transform_pixel(pixel, self.palette, self.weights)

    def __apply_to_all_pixels(
        self,
        image: cv2.Mat,
        transformation: Callable[[np.ndarray, Palette], np.ndarray],
        palette: Palette,
        weights: dict[str, float],
    ) -> cv2.Mat:
        return super().__apply_to_all_pixels(image, transformation, palette, weights)

    @staticmethod
    def __validate_weights(
        palette: Palette, weights: PaletteWeights | None
    ) -> PaletteWeights:
        """ """
        # If we haven't been given any weights, assign each colour a weight of 1:
        if weights is None:
            weights = PaletteWeights({colour_name: 1 for colour_name in palette.names})

        # Check that each colour in the palette has a weight:
        colours_with_no_weight = [
            colour_name
            for colour_name in palette.names
            if colour_name not in weights.names
        ]
        if colours_with_no_weight:
            raise ValueError(
                f"The following colours were not assigned weights:\n{colours_with_no_weight}"
            )

        # Check that only colours from the palette are in the weights:
        extra_colours = [
            colour_name
            for colour_name in weights.names
            if colour_name not in palette.names
        ]
        if extra_colours:
            raise ValueError(
                f"The following colours are not in the palette but were assigned weights:\n{extra_colours}"
            )

        return weights
