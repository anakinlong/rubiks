"""
Rubiks image transformations.
"""

from abc import ABC, abstractmethod
import numpy as np
import cv2
from .palette_class import Palette, PaletteWeights
from .constants import CV2_CHANNEL_FORMAT
from .pixel_transformations import recolour_closest_weighted, recolour_closest_weighted_greyscale


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
        # Input verification for the palette:
        palette = cls._validate_palette(palette)
        # Input verification for the weights (if None, creates equal weight for each colour):
        palette_weights = cls._validate_weights(palette, palette_weights)

        return cls._apply_to_all_pixels(image, palette, palette_weights)

    @classmethod
    def _transform_pixel(cls, pixel: np.ndarray, palette: Palette, palette_weights: PaletteWeights) -> np.ndarray:
        """
        Change the colour of the pixel to the one from the palette which is geometrically closest.

        :param pixel: a pixel.
        :param palette: a palette of colours from which the final image will be constructed. All colours must have the
        same channel format as a cv2.Mat.
        :param palette_weights: a map from colour names to "weights", which will determine how big of a sphere of
        influence each colour has.

        :return: the transformed pixel.
        """

        return recolour_closest_weighted(pixel, palette, palette_weights)

    @staticmethod
    def _validate_palette(palette: Palette) -> Palette:
        """
        Do input validation for the palette.

        :param palette: a Palette of Colours.

        :return: palette
        """
        # Reformat to the same channel format as cv2.Mat has:
        palette.reformat(CV2_CHANNEL_FORMAT)

        return palette

    @staticmethod
    def _validate_weights(palette: Palette, palette_weights: PaletteWeights | None) -> PaletteWeights:
        """
        Do input validation for the palette weights.

        :param palette: a Palette of Colours.
        :param pallete_weights: a PaletteWeights instance.

        :return: palette_weights
        """
        # If we haven't been given any weights, assign each colour a weight of 1:
        if palette_weights is None:
            palette_weights = PaletteWeights({colour_name: 1 for colour_name in palette.names})

        # Check that the palette weights and palette share the same colours:
        palette_weights.validate_against_palette(palette)

        return palette_weights


class RecolourClosestGreyscale(RecolourClosest):
    """
    Recolour each pixel to the colour in the palette that is closest geometrically when converted to greyscale.
    """

    @classmethod
    def _transform_pixel(cls, pixel: np.ndarray, palette: Palette, palette_weights: PaletteWeights) -> np.ndarray:
        """
        Change the colour of the pixel to the one from the palette which is geometrically closest when converted to
        greyscale.

        :param pixel: a pixel.
        :param palette: a palette of colours from which the final image will be constructed. All colours must have the
        same channel format as a cv2.Mat.
        :param palette_weights: a map from colour names to "weights", which will determine how big of a sphere of
        influence each colour has.

        :return: the transformed pixel.
        """

        return recolour_closest_weighted_greyscale(pixel, palette, palette_weights)
