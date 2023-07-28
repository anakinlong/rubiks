"""
Some public methods for rubiks images.
"""

import cv2
import math
import numpy as np


def rubiks_dimension_estimator(
    n_cubes: int,
    cube_size: int = 3,
    width: int | None = None,
    height: int | None = None,
    image: cv2.Mat | None = None,
):
    """
    Output the rubiks cube dimensions of an image.

    :param n_cubes: the number of rubiks cubes available.
    :param cube_size: the "dimension" of the rubiks cubes, ie. how many tiles across they are. Defaults to 3.
    :param width: the width of the input image in pixels.
    :param height: the height of the input image in pixels.
    :param image: a cv2.Mat.

    :return: output_dimensions the dimensions of the resulting rubiks image (in rubiks tiles).
    """
    # Verify inputs:
    # Check that we either have both width and height and NOT image, or have image and NEITHER of width and height:
    if not (
        ((width is not None and height is not None) and (image is None))
        or ((width is None and height is None) and (image is not None))
    ):
        raise ValueError("Either supply both 'width' and 'height' or just 'image'.")
    # If we have an image, extract the width and height:
    if image is not None:
        height, width = image.shape[:2]

    # Calculate aspect ratio of input image:
    input_aspect_ratio = width / height

    # From this, give some good estimates on the best width and heigh of the rubiks image:
    # sqrt(n cubes * aspect ratio) * sqrt(n cubes / aspect ratio) = n cubes, so by taking the floor of each we get
    # width * height = floor(sqrt(n cubes * aspect ratio)) * floor(sqrt(n cubes / aspect ratio))
    #                >= sqrt(n cubes * aspect ratio) * sqrt(n cubes / aspect ratio)
    #                = n cubes,
    # so we won't ever produce an estimate which uses more cubes than n cubes, and the output aspect ratio will be
    # relatively similar to the aspect ratio of the input image.
    output_width_in_cubes = math.floor(np.sqrt(n_cubes * input_aspect_ratio))
    output_height_in_cubes = math.floor(np.sqrt(n_cubes / input_aspect_ratio))

    # Multiply the width and height by the cube size to get them in terms of rubiks cube tiles:
    return cube_size * output_width_in_cubes, cube_size * output_height_in_cubes


def resize_image(
    image: cv2.Mat,
    new_width: int | None = None,
    new_height: int | None = None,
    scale: float | None = None,
    interpolation: int | None = None,
) -> cv2.Mat:
    """
    Take an image and change to a desired resolution or resize by a given scale. Supply either target width and height,
    or a scale.

    If enlarging the image, uses the "INTER_NEAREST" interpolation method from cv2 by default.
    If shrinking the image, uses the "INTER_AREA" interpolation method from cv2 by default.

    :param image: the image.
    :param new_width: the width of the resulting image.
    :param new_height: the width of the resulting image.
    :param scale: the scale by which to resize both dimensions.
    :param interpolation: the cv2 interpolation method to use.

    :return: new_image the resized image.
    """
    # Verify inputs:
    # Check that we either have both width and height and NOT scale, or have scale and NEITHER of width and height:
    if not (
        ((new_width is not None and new_height is not None) and (scale is None))
        or ((new_width is None and new_height is None) and (scale is not None))
    ):
        raise ValueError("Either supply both 'new_width' and 'new_height' or just 'scale'.")

    # If we have a scale, use it to calculate the new width and height:
    current_height, current_width = image.shape[:2]
    if scale is not None:
        new_height, new_width = scale * current_height, scale * current_width
    # Otherwise, figure out the scale from the input width and height:
    else:
        scale = (new_width * new_height) / (current_width * current_height)

    # If we have not been given an interpolation, figure out which default to use:
    if interpolation is None:
        # If image is being enlarged, use LINEAR_NEAREST:
        if scale >= 1:
            interpolation = cv2.INTER_NEAREST
        # If image is being shrunk, use INTER_AREA:
        else:
            interpolation = cv2.INTER_AREA

    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)
