"""
Rubiks pixel transformations.
"""

import numpy as np
from .lib import pixel_to_greyscale
from .palette_class import Palette, PaletteWeights


def recolour_closest_weighted(pixel: np.ndarray, palette: Palette, palette_weights: PaletteWeights) -> np.ndarray:
    """
    Change the colour of the pixel to the one from the palette which is geometrically closest.

    :param pixel: a pixel.
    :param palette: a palette of colours from which the final image will be constructed. All colours must have the
    same channel format as a cv2.Mat.
    :param palette_weights: a map from colour names to "weights", which will determine how big of a sphere of
    influence each colour has.

    :return: the transformed pixel.
    """
    # Measure the Euclidean distance to the pixel colour of each colour in the palette:
    distances = {
        colour_name: np.linalg.norm(pixel - np.array(colour)) / palette_weights[colour_name]
        for colour_name, colour in palette.colour_dict.items()
    }

    # Find the minimum distance:
    smallest_distance = min(distances.values())
    # Create a list of all the colour names with this distance:
    colours_with_smallest_distance = [
        colour_name for colour_name, distance in distances.items() if distance == smallest_distance
    ]

    # Choose one of these colours randomly:
    closest_colour = np.random.choice(colours_with_smallest_distance)

    return np.array(palette[closest_colour])


def recolour_closest_weighted_greyscale(
    pixel: np.ndarray, palette: Palette, palette_weights: PaletteWeights
) -> np.ndarray:
    """
    Change the colour of the pixel to the one from the palette which is geometrically closest when comparing their
    greyscale values.

    :param pixel: a pixel.
    :param palette: a palette of colours from which the final image will be constructed. All colours must have the
    same channel format as a cv2.Mat.
    :param palette_weights: a map from colour names to "weights", which will determine how big of a sphere of
    influence each colour has.

    :return: the transformed pixel.
    """
    # Measure the Euclidean distance to the greyscale value of each colour in the palette:
    pixel_greyscale = pixel_to_greyscale(pixel)
    distances = {
        colour_name: abs(pixel_greyscale - colour.greyscale) / palette_weights[colour_name]
        for colour_name, colour in palette.colour_dict.items()
    }

    # Find the minimum distance:
    smallest_distance = min(distances.values())
    # Create a list of all the colour names with this distance:
    colours_with_smallest_distance = [
        colour_name for colour_name, distance in distances.items() if distance == smallest_distance
    ]

    # Choose one of these colours randomly:
    closest_colour = np.random.choice(colours_with_smallest_distance)

    return np.array(palette[closest_colour])
