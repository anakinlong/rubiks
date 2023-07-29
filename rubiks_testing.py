"""
Some initial testing for rubiks cube project.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple, List, Union, Dict, Any
import cv2

# The [B,G,R] of each rubiks cube colour:
RUBIKS_COLOURS = {
    "g": [72, 155, 0],
    "w": [255, 255, 255],
    "r": [52, 18, 183],
    "y": [0, 213, 255],
    "b": [173, 70, 0],
    "o": [0, 88, 255],
}

RUBIKS_COLOUR_WEIGHTS = {"g": 1.0, "w": 1.0, "r": 1.0, "y": 1.0, "b": 0.8, "o": 1.0}


def scale_image(image: cv2.Mat, f: int, interpolation: Union[str, None] = None) -> cv2.Mat:
    """
    Scale an image, maintaining the aspect ratio.

    :param image: an image.
    :param f: the scale factor.
    :param interpolation: the interpolation method. Defaults to cv2.INTER_NEAREST.

    :return new_img: the scaled image.
    """

    # Check that f is an integer:
    if type(f) is not int:
        raise TypeError(f"{type(f)} is not an accepted type for f, please use an int.")

    # Get the dimensions of the image:
    height, width = image.shape[:2]

    # Create the new, scaled dimensions:
    dim = (f * width, f * height)

    # Set that as the new dimensions:
    if interpolation is None:
        interpolation = cv2.INTER_NEAREST
    new_img = cv2.resize(image, dim, interpolation=interpolation)

    return new_img


def rubiks_dimension_estimator(
    width: int, height: int, n_cubes: int, d: int = 3, reporting: bool = False
) -> tuple[int]:
    """
    Output the rubiks cube dimensions of an image.

    :param width: the width of the input image in pixels.
    :param height: the height of the input image in pixels.
    :param n_cubes: the number of rubiks cubes available.
    :param d: the "dimension" of the rubiks cubes, ie. how many tiles across they are. Defaults to 3.
    :param reporting: if True, print some metrics. Defaults to False.

    :return image_dimensions_r: the dimensions of the resulting rubiks image (in rubiks tiles).
    """

    # Calculate aspect ratio of input image:
    aspect = width / height
    # From this, give some good estimates on the best width and heigh of the rubiks image:
    width_r = math.floor(np.sqrt(n_cubes * aspect))
    height_r = math.floor(np.sqrt(n_cubes / aspect))

    image_dimensions_r = (d * width_r, d * height_r)

    if reporting:
        # Some nice things to report:
        aspect_r = width_r / height_r
        n_cubes_r = width_r * height_r
        cubes_dimensions = (width_r, height_r)

        print(f"Cube pixels: {image_dimensions_r}")
        print(f"Cubes: {cubes_dimensions}")
        print(f"Initial aspect ratio: {aspect}")
        print(f"Rubiks aspect ratio: {aspect_r}")
        print(f"Max cubes: {n_cubes}")
        print(f"Used cubes: {n_cubes_r}")
        print(f"Quantitative metric: {arrangement_metric(aspect, aspect_r, n_cubes, n_cubes_r)}")

    return image_dimensions_r


def arrangement_metric(aspect_i: float, aspect_r: float, n_cubes_i: int, n_cubes_r: int) -> float:
    """
    Given an initial image with aspect ratio aspect_i, and a number of cubes n_cubes_i, a quantitative metric
    to find out how good a rubiks image with aspect ratio aspect_r and number of cubes n_cubes_r is. This metric
    is entirely made up and probably terrible.

    In theory, a lower score indicates a better arrangement of cubes, with a score of 0 being perfect.

    :param aspect_i: aspect ratio of initial image (width / height).
    :param aspect_r: aspect ratio of rubiks image.
    :param n_cubes_i: number of cubes available.
    :param n_cubes_r: number of cubes used in rubiks image.

    :return score: a measure of how good the arrangement of cubes is.
    """

    # Something like the chi-squared metric:
    def norm_dist(e: float, o: float) -> float:
        """
        The distance between two numbers e and o, normalised by e.

        :param a: a float, the expected value.
        :param b: a float, the observed value.

        :return normalised_distance: abs(e - o) / abs(e)
        """
        return abs(e - o) / abs(e)

    # Add them together so both are considered I suppose, can't multiply in case one is zero:
    score = norm_dist(aspect_i, aspect_r) + norm_dist(n_cubes_i, n_cubes_r)

    return score


def average_pixel_colour(image: cv2.Mat, position_0: Tuple[float], position_1: Tuple[float]) -> np.ndarray[float]:
    """
    Take a rectangular section of an image, and calculate the average pixel colour. The vertices of the rectangular
    aren't necessarily on the vertices of pixels, so we need to weight the average by the area of each pixel that
    is in the section in which we are interested. For a lot of this function it's easier to look at the diagram.

    :param image: an image.
    :param position_0: the coordinates of the top left corner of the rectangular section, in the format (x_0, y_0).
    :param position_1: the coordinates of the bottom right corner of the rectangular section, in the format
    (x_1, y_1).

    :return colour: the average pixel colour of the rectangular section of the image.
    """

    # Unpack the positions for convenience:
    (w_0, h_0) = position_0
    (w_1, h_1) = position_1

    # Check that _0 is the top left and _1 is the bottom right:
    if not all(v_1 >= v_0 for v_0, v_1 in zip(position_0, position_1)):
        raise ValueError(
            f"Each coordinate in position_1 must be greater than the corresponding coordinate in position_0: \n"
            f"position_0: {position_0}\n"
            f"position_1: {position_1}\n"
        )

    # Create an array of pixel vertices:
    def create_green_values(v_0: float, v_1: float) -> List[float]:
        """
        Create a list of values from v_0 to v_1 inclusive, where each element is the next integer from the last
        until we reach v_1, and in that case we finish the list with the exact value of v_1.

        :param v_0: the starting number of the list.
        :param v_1: the last number in the list. Must be greater than v_0.

        :return values: a list of integers between v_0 and v_1, bookended by v_0 and v_1.
        """

        # Check that v_1 >= v_0:
        if not v_1 >= v_0:
            raise ValueError(f"v_1 must be greater than (or equal to) v_0:\n" f"v_0: {v_0}\n" f"v_1: {v_1}\n")
        # First value is always v_0:
        values = [v_0]
        done = False
        while not done:
            # Take the previous value:
            v_i = values[-1]
            # Increment by one, unless that's greater than v_1, and in that case set it equal to v_1:
            v = min(math.floor(v_i + 1), v_1)
            # Append the value to our list:
            values.append(v)
            # If that value was v_1, we are done
            if v == v_1:
                done = True
            # Otherwise, we repeat.
        return values

    # Create the separate horizontal and vertical coordinates of the vertices:
    green_width_values = create_green_values(w_0, w_1)
    green_height_values = create_green_values(h_0, h_1)

    # A 2D list for area values:
    area = np.ones((len(green_height_values) - 1, len(green_width_values) - 1))
    # The rectangular section of the image:
    small_image = image[math.floor(h_0) : math.ceil(h_1), math.floor(w_0) : math.ceil(w_1), :]
    # Loop through each pixel and calculate how much of its area is in the rectangular section.
    # We already know that only the pixels around the edge of the rectangle are going to have an area not equal
    # to 1, so we start off with an array of ones and only modify the values around the edges:
    width_indices = len(green_width_values) - 1
    height_indices = len(green_height_values) - 1
    for i in range(width_indices):
        for j in range(height_indices):
            # We are on an edge if either index is 0 or their maximum:
            edge = (i in [0, width_indices - 1]) or (j in [0, height_indices - 1])
            if edge:
                # Take the vertex down and right of this one,
                # and calculate the side lengths of the resulting rectangle:
                width = green_width_values[i + 1] - green_width_values[i]
                height = green_height_values[j + 1] - green_height_values[j]
                # The area of that pixel that is in the rectangle is the product of the side lengths:
                area[j, i] = width * height

    # Now to average the colours of each pixel, weighted by their areas.
    # We need to do this for each of the (B, G, R) values:
    colour = np.zeros(3)
    for i in range(3):
        colour[i] = np.average(small_image[:, :, i], weights=area)

    return colour


def resize_image(image: cv2.Mat, dimensions: Tuple[int]) -> cv2.Mat:
    """
    Take an image, and change to a desired resolution.

    :param image: the image.
    :param dimensions: the dimensions of the resulting image, in the format (width, height).

    :return new_img: the resized image.
    """

    # Unpack our new dimensions for convenience:
    (width_r, height_r) = dimensions
    # Get the actual dimensions of the original image:
    height, width = image.shape[:2]

    # Create an image with our desired dimensions:
    new_img = np.zeros((height_r, width_r, 3), np.uint8)

    # Take the coordinates of each pixel in the new image and scale up to original image size:
    coords = np.zeros((height_r + 1, width_r + 1), dtype=object)
    for x in range(width_r + 1):
        for y in range(height_r + 1):
            # Make sure that they don't accidentally go off the end of the image due to numerical issues:
            coords[y, x] = (
                min(x * width / width_r, width),
                min(y * height / height_r, height),
            )

    # Loop through these new big pixels and get the average colour of that area in the original image:
    for x in range(width_r):
        for y in range(height_r):
            new_img[y, x] = average_pixel_colour(image, coords[y, x], coords[y + 1, x + 1])

    return new_img


def find_nearest_colour(colour: List[int], pallete: List[List[int]]) -> List[int]:
    """
    Take a colour, and find the colour in the pallete that it is geometrically closest to.

    :param colour: the input colour.
    :param pallete: a list of colours to compare to the input colour.

    :return closest_colour: the colour in the pallete that is closest to the input colour.
    """

    # Loop through the colours in the pallete, and measure their Euclidean distance to colour:
    distances = np.zeros(len(pallete))
    for i, c in enumerate(pallete):
        # np.linalg.norm only works with numpy arrays:
        distances[i] = np.linalg.norm(np.array(c) - np.array(colour))

    # Find the value of the minimum distance:
    min_dist = min(distances)

    # Check that there is only one minimum:
    n_minimums = np.count_nonzero(distances == min_dist)

    # If there is just one, we can just return which colour it was:
    if n_minimums == 1:
        min_index = np.argmin(distances)
    # If there are multiple, choose from one of the closest at random:
    elif n_minimums > 1:
        min_indices = [i for i, x in enumerate(distances) if x == min_dist]
        min_index = np.random.choice(min_indices)
    # Otherwise, I have no idea:
    else:
        raise ValueError("Somehow there are zero or a negative amount of minimum distances (o_0)!")

    # Get the closest colour
    closest_colour = pallete[min_index]

    return closest_colour


def recolour_nearest_colour(image: cv2.Mat, pallete: List[List[int]]) -> cv2.Mat:
    """
    Recolour an image by replacing each pixel with the colour from the pallete that is geometrically closest
    to the original colour.

    :param image: an image.
    :param pallete: a list of colours to use in the recoloured image.

    :return new_img: the recoloured image.
    """

    # Loop through each pixel and apply the method:
    height, width = image.shape[:2]
    new_img = image.copy()
    for x in range(width):
        for y in range(height):
            # Find the current colour:
            original_colour = image[y, x]
            # Set it to the closest one from the pallete:
            new_img[y, x] = find_nearest_colour(original_colour, pallete)

    return new_img


def find_nearest_colour_weighted(
    colour: List[int],
    pallete: List[List[int]],
    weights: List[float] = [1, 1, 1, 1, 1, 1],
) -> List[int]:
    """
    Take a colour, and find the colour in the pallete that it is geometrically closest to.

    :param colour: the input colour.
    :param pallete: a list of colours to compare to the input colour.
    :param weights: a list of weights, one for each colour.

    :return closest_colour: the colour in the pallete that is closest to the input colour.
    """

    # Loop through the colours in the pallete, and measure their Euclidean distance to colour:
    distances = np.zeros(len(pallete))
    for i, c in enumerate(pallete):
        # np.linalg.norm only works with numpy arrays:
        distances[i] = np.linalg.norm(np.array(c) - np.array(colour)) * weights[i]

    # Find the value of the minimum distance:
    min_dist = min(distances)

    # Check that there is only one minimum:
    n_minimums = np.count_nonzero(distances == min_dist)

    # If there is just one, we can just return which colour it was:
    if n_minimums == 1:
        min_index = np.argmin(distances)
    # If there are multiple, choose from one of the closest at random:
    elif n_minimums > 1:
        min_indices = [i for i, x in enumerate(distances) if x == min_dist]
        min_index = np.random.choice(min_indices)
    # Otherwise, I have no idea:
    else:
        raise ValueError("Somehow there are zero or a negative amount of minimum distances (o_0)!")

    # Get the closest colour
    closest_colour = pallete[min_index]

    return closest_colour


def recolour_nearest_colour_weighted(
    image: cv2.Mat,
    pallete: List[List[int]],
    weights: List[float] = list(RUBIKS_COLOUR_WEIGHTS.values()),
) -> cv2.Mat:
    """
    Recolour an image by replacing each pixel with the colour from the pallete that is geometrically closest
    to the original colour.

    :param image: an image.
    :param pallete: a list of colours to use in the recoloured image.
    :param weights: a list of weights, one for each colour.

    :return new_img: the recoloured image.
    """

    # Check that each colour has a weight:
    p, w = len(pallete), len(weights)
    if p != w:
        raise ValueError(
            f"There are {p} colours in the pallete, but {w} weights. Must have the same number of pallete colours"
            f" as weights."
        )

    # Loop through each pixel and apply the method:
    height, width = image.shape[:2]
    new_img = image.copy()
    for x in range(width):
        for y in range(height):
            # Find the current colour:
            original_colour = image[y, x]
            # Set it to the closest one from the pallete:
            new_img[y, x] = find_nearest_colour_weighted(original_colour, pallete, weights)

    return new_img


def plane_colour(
    colour: List[int],
    plane: Tuple[float] = (1, 1, 1, 255),
    origin: Tuple[float] = (0, 0, 0),
) -> List[int]:
    """
    Take a colour and project it from the origin to a plane in 3d colour space.

    :param colour: the colour.
    :param plane: the coefficients of the plane, in the format (b, g, r, c) => bB + gG + rR = c for colours B, G, R.
    Defaults to B + G + R = 255.
    :param origin: the point in 3d colour space from which we will project the colour, in the format (b, g, r).
    Defaults to (0, 0, 0).

    :return new_colour: the projected colour.
    """

    # Unpack the plane coefficients, origin, and colour:
    (a, b, c, d) = plane
    (b_0, g_0, r_0) = origin
    [b_c, g_c, r_c] = colour

    # Check that the plane is valid:
    if all(x == 0 for x in [a, b, c]):
        raise ValueError(f"{plane} is not a valid plane.")

    # Check that the colour isn't at the origin:
    if all(c == o for c, o in zip(colour, origin)):
        # If it is, just return the middle(-ish) of the plane:
        try:
            v = int(d / (a + b + c))
            return list(np.clip([v, v, v], 0, 255))
        # In the case that a + b + c = 0:
        except ZeroDivisionError:
            return [127, 127, 127]

    # Work out the t value:
    t = (d - np.sum(np.multiply([a, b, c], origin))) / np.sum(np.subtract(colour, origin))

    # Project the colour using the t value:
    new_colour = np.add(origin, np.multiply(t, np.multiply([a, b, c], colour)))
    # Make sure it is within our allowed volume:
    new_colour = np.clip(new_colour, 0, 255)
    # Make sure the values are integers:
    new_colour = [int(x) for x in new_colour]

    return new_colour


def recolour_nearest_colour_planed(
    image: cv2.Mat,
    pallete: List[List[int]],
    plane: Tuple[float] = (1, 1, 1, 255),
    origin: Tuple[float] = (0, 0, 0),
) -> cv2.Mat:
    """
    Recolour an image by replacing each pixel with the colour from the pallete that is geometrically closest
    to the original colour.

    :param image: an image.
    :param pallete: a list of colours to use in the recoloured image.
    :param plane: the coefficients of the plane, in the format (b, g, r, c) => bB + gG + rR = c for colours B, G, R.
    Defaults to B + G + R = 255.
    :param origin: the point in 3d colour space from which we will project the colour, in the format (b, g, r).
    Defaults to (0, 0, 0).

    :return new_img: the recoloured image.
    """

    # Project the pallete to the plane:
    pallete_planed = [plane_colour(c, plane, origin) for c in pallete]

    # Loop through each pixel and apply the method:
    height, width = image.shape[:2]
    new_img = image.copy()
    for x in range(width):
        for y in range(height):
            # Find the current colour:
            original_colour = image[y, x]
            # Project it to the plane:
            original_colour_planed = plane_colour(original_colour, plane, origin)
            # Find the closest one from the projected pallete:
            closest = find_nearest_colour(original_colour_planed, pallete_planed)
            # Set the colour as the corresponding pallete colour:
            p_i = pallete_planed.index(closest)
            new_img[y, x] = pallete[p_i]

    return new_img


def recolour_image(image: cv2.Mat, method: str, pallete: List[List[int]]) -> cv2.Mat:
    """
    Take an image, and recolour it according to a pallete of colours and a method.

    :param image: an image.
    :param pallete: a list of colours, each in [B, G, R] format.
    :param method: the method by which the image will be recoloured.
    :param method_kwargs: any arguments for the recolouring method, in the form of a dictionary.

    :return new_img: the recoloured image.
    """

    # A dictionary where the keys are method names and the values are the methods:
    methods = {
        "nearest colour": recolour_nearest_colour,
        "nearest colour weighted": recolour_nearest_colour_weighted,
        "nearest colour planed": recolour_nearest_colour_planed,
    }

    # Check that the method is a valid one:
    if method not in methods:
        raise ValueError(f"'{method}' is not a valid method. Please choose from:\n" f"{list(methods)}")

    # Use the recolouring method to recolour our image:
    try:
        f = methods[method]
        new_img = f(image, pallete)
    except KeyError:
        # If the method name is in the dictionary, but the function doesn't work:
        raise NotImplementedError(
            f"'{method}' has not yet been implemented. Please choose a different method from:\n" f"{list(methods)}"
        )

    return new_img


def plot_colour_list(colour_list: cv2.Mat) -> None:
    """
    Plot the (B,G,R) coordinates of each pixel of an image in 3D space.

    :param image: the image.

    :return None:
    """

    # Set up our plot:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Loop through each pixel, and plot its position:
    n_pixels = len(colour_list)
    bs, gs, rs = np.zeros(n_pixels), np.zeros(n_pixels), np.zeros(n_pixels)
    colours = np.zeros((n_pixels, 3))
    for i in range(n_pixels):
        # Get the colour:
        [b, g, r] = colour_list[i]
        colour = np.multiply((r, g, b), 1 / 255)
        bs[i], gs[i], rs[i] = b, g, r
        colours[i] = colour

    # Actually plot the pixels:
    ax.scatter(bs, gs, rs, marker=".", color=colours)

    # Cosmetics:
    ax.set_xlabel("B")
    ax.set_xlim((0, 255))
    ax.set_ylabel("G")
    ax.set_ylim((0, 255))
    ax.set_zlabel("R")
    ax.set_zlim((0, 255))
    # Set the aspect ratio to be 1:1:1:
    ax.set_box_aspect((np.ptp(bs), np.ptp(gs), np.ptp(rs)))

    plt.show()


def colour_range(image: cv2.Mat, channel: Union[int, str]) -> Tuple[int]:
    """
    Find the minimum and maximum values of a particular colour channel in an image.

    :param image: the image.
    :param channel: which colour channel we are looking at.

    :return channel_range: a tuple of the range, in the format (min, max).
    """

    # Make sure channel input is valid:
    channel_names = ("b", "g", "r")
    if channel in channel_names:
        channel = channel_names.index(channel)
    if type(channel) is not int:
        raise TypeError("cba to write this error rn")

    # Find min and max:
    channel_values = image[:, :, channel]
    channel_min = np.amin(channel_values)
    channel_max = np.amax(channel_values)

    channel_range = (channel_min, channel_max)

    return channel_range


def plot_colour_box(image: cv2.Mat) -> None:
    """
    Plot the smallest box that will fit around the 3D representation of the colours of each pixel of the image.

    :param image: the image.

    :return None:
    """

    # Set up our plot:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Find the ranges for each channel:
    ranges = {}
    for channel in ["b", "g", "r"]:
        ranges[channel] = colour_range(image, channel)

    # The vertices of the cuboid will be all the combinations of the ranges:
    bs, gs, rs = np.zeros(8), np.zeros(8), np.zeros(8)
    colours = np.zeros((8, 3))
    i = 0
    for b in ranges["b"]:
        for g in ranges["g"]:
            for r in ranges["r"]:
                bs[i], gs[i], rs[i] = b, g, r
                colours[i] = np.multiply((r, g, b), 1 / 255)
                i += 1

    # Actually plot the pixels:
    ax.scatter(bs, gs, rs, marker=".", color=colours)

    # Cosmetics:
    ax.set_xlabel("B")
    ax.set_xlim((0, 255))
    ax.set_ylabel("G")
    ax.set_ylim((0, 255))
    ax.set_zlabel("R")
    ax.set_zlim((0, 255))
    # Set the aspect ratio to be 1:1:1:
    ax.set_box_aspect((np.ptp(bs), np.ptp(gs), np.ptp(rs)))

    plt.show()


def main(
    image_file_name: str,
    n_cubes: int,
    pallete: List[List[int]],
    recolouring_method: str,
    image_scale: float = 1,
    image_folder: str = ".\\Images\\",
) -> None:
    """
    Take an image and recreate it in rubiks cube form.

    :param image_file_name: str,
    :param n_cubes: int,
    :param pallete: List[List[int]],
    :param recolouring_method: str,
    :param recolouring_kwargs: Dict[str, Any] = {},
    :param image_scale: str = 1,
    :param image_folder: str = ".\\Images\\"

    :return None:
    """

    # Read in our image:
    im = cv2.imread(image_folder + image_file_name)

    # Get its dimensions, then estimate the best dimensions for the final image:
    height, width = im.shape[:2]
    (new_height, new_width) = rubiks_dimension_estimator(width, height, n_cubes, reporting=True)

    # Create a real-colour version of the final image:
    new_im = resize_image(im, (new_height, new_width))

    # Recolour this image using the pallete and method given:
    new_new_im = recolour_image(new_im, recolouring_method, pallete)

    # Scale the image up so we can see it, and show the two pixelated images side-by-side:
    scale = math.floor(image_scale * width / new_width)
    horizontal = np.hstack((scale_image(new_im, scale), scale_image(new_new_im, scale)))
    # vertical = np.vstack((scale_image(new_im, scale), scale_image(new_new_im, scale)))
    cv2.imshow("im", horizontal)
    cv2.waitKey(0)

    cv2.imwrite("Rubiks Images\\mona lisa rbks.png", new_new_im)


if __name__ == "__main__":

    main(
        image_file_name="mona lisa.png",
        n_cubes=600,
        pallete=list(RUBIKS_COLOURS.values()),
        recolouring_method="nearest colour weighted",
        image_scale=3
        # recolouring_kwargs = {
        #     "weights": [1, 1, 1, 1, 1, 1]
        # }
    )
