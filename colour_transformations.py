"""
Colour transformations.
"""

import colour_class as c
import random_functions as rf
from typing import Tuple


def colour_average(colour_1: c.Colour, colour_2: c.Colour, weights: Tuple[float] = (1, 1)) -> c.Colour:
    """
    Take two colours and return a new colour which has the colour channel format of the first, and whose colour channel
    values are the averages of the two.

    :param colour_1: a colour.
    :param colour_2: a colour.
    :param weights: a tuple of weights, in the format (weight_1, weight_2)

    :return new_colour: the average of the two colours, with the channel format of the first.
    """

    # Check they have the same set of channel names:
    if not rf.is_permutation(colour_1.format, colour_2.format):
        raise ValueError(
            f"Both colours must have the same set of channel names:\n" f"{colour_1.format}" f"{colour_2.format}"
        )

    # Check that weights is the right format:
    rf.check_type(weights, [tuple], "weights")

    # Calculate new values:
    (w_1, w_2) = [w / sum(weights) for w in weights]
    new_values = colour_1.values
    new_format = colour_1.format
    for i, channel in enumerate(new_format):
        new_value = w_1 * colour_1.channel_value(channel) + w_2 * colour_2.channel_value(channel)
        new_values[i] = round(new_value)

    # Make it a colour:
    new_colour = c.Colour(new_values, new_format)

    return new_colour
