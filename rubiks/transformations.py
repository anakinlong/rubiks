"""
Rubiks image transformations.
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2
from .palette_class import Palette, PaletteWeights


class Transformation(ABC):
    """
    An abstract transformation. Lays out the general flow of actual transformations.
    """

    @classmethod
    def transform_image(cls, image: cv2.Mat, *args, **kwargs) -> cv2.Mat:
        """
        Main transform method for an image.

        :param image: an image.

        :return: a transformed image.
        """
        return cls._apply_to_all_pixels(image, *args, **kwargs)

    @classmethod
    @abstractmethod
    def _transform_pixel(cls, pixel: np.ndarray, *args, **kwargs) -> np.ndarray:
        """
        Main transform method for a pixel. Must be implemented by child classes.

        :param pixel: a pixel.

        :return: a transformed pixel.
        """
        raise NotImplementedError("_transform_pixel not implemented for base Transformation class")

    @classmethod
    def _apply_to_all_pixels(
        cls,
        image: cv2.Mat,
        *args,
        **kwargs,
    ) -> cv2.Mat:
        """
        Apply a pixel transform method to all pixels in an image.

        :param image: an image.

        :return: the transformed image.
        """
        height, width = image.shape[:2]
        transformed_image = image.copy()
        for x in range(width):
            for y in range(height):
                # Find the current pixel:
                original_pixel = image[y, x]
                # Transform it:
                transformed_image[y, x] = cls._transform_pixel(original_pixel, *args, **kwargs)

        return transformed_image


class NoneTransformation(Transformation):
    """
    Keep each pixel exactly the same colour.
    """

    @classmethod
    def _transform_pixel(cls, pixel: np.ndarray, *args, **kwargs) -> np.ndarray:

        return pixel


class RecolourClosest(Transformation):
    """
    Recolour each pixel to the colour in the palette that is closest geometrically.
    """

    @classmethod
    def transform_image(
        cls,
        image: cv2.Mat,
        palette: Palette,
        palette_weights: PaletteWeights | None = None,
    ) -> cv2.Mat:
        """
        Recolour each pixel of an image by chosing the colour in the palette which is geometrically closest.

        :param image: an image in the form of a cv2.Mat.
        :param palette: a palette of colours from which the final image will be constructed.
        :param palette_weights: a map from colour names to "weights", which will determine how big of a sphere of
        influence each colour has. Defaults to all colours having equal weights.

        :return: an image (cv2.Mat) made up of the colours in the given palette.
        """
        # Input verification for the weights (if None, creates equal weight for each colour):
        palette_weights = cls.__validate_weights(palette, palette_weights)

        return cls._apply_to_all_pixels(image, palette, palette_weights)

    # TODO implement _transform_pixel()

    @staticmethod
    def __validate_weights(palette: Palette, palette_weights: PaletteWeights | None) -> PaletteWeights:
        """
        Do input validation for the palette weights.

        :param palette: a Palette of Colours.
        :param pallete_weights: a PaletteWeights instance.

        :return: palette_weights
        """
        # If we haven't been given any weights, assign each colour a weight of 1:
        if palette_weights is None:
            palette_weights = PaletteWeights({colour_name: 1 for colour_name in palette.names})

        # Check that each colour in the palette has a weight:
        colours_with_no_weight = [
            colour_name for colour_name in palette.names if colour_name not in palette_weights.names
        ]
        if colours_with_no_weight:
            raise ValueError(f"The following colours were not assigned weights:\n{colours_with_no_weight}")

        # Check that only colours from the palette are in the weights:
        extra_colours = [colour_name for colour_name in palette_weights.names if colour_name not in palette.names]
        if extra_colours:
            raise ValueError(
                f"The following colours are not in the palette but were assigned weights:\n{extra_colours}"
            )

        return palette_weights
