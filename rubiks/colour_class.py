"""
Colour class for rubiks images.
"""

from __future__ import annotations
from .lib import is_permutation, check_type
from typing import Any, Iterable

# Some defaults for Colour:
DEFAULT_COLOUR = [0, 0, 0]
DEFAULT_CHANNEL_FORMAT = ["b", "g", "r"]


class Colour(list):
    """
    A colour :^). This is basically a list with a few extra features. Currently only supports 3 colour channels.
    """

    def __init__(self, iterable: Iterable[int] | None = None, channel_format: list[str] = None) -> None:
        """
        A colour :^). This is basically a list with a few extra features. Currently only supports 3 colour channels.

        :param iterable: a list of colour channel values. Defaults to DEFAULT_COLOUR [0, 0, 0].
        :param channel_format: the format of the colour channels, in the form of a list of strings. Defaults to
        DEFAULT_CHANNEL_FORMAT ["b", "g", "r"].
        """
        # Use defaults if input(s) not given:
        if iterable is None:
            iterable = DEFAULT_COLOUR
        if channel_format is None:
            channel_format = DEFAULT_CHANNEL_FORMAT
        # Input validation:
        Colour.__validate_iterable(iterable)
        Colour.__validate_channel_format(channel_format)

        # Set the colour channel format as an attribute:
        self.format = channel_format

        # Initialise attributes as None then update them:
        self._values = None
        self._format = None
        self.__update()

    @property
    def values(self):
        """
        The list of colour channel values.
        """
        return self._values

    @property
    def format(self):
        """
        The list of colour channel names.
        """
        return self._format

    def __update(self, channel_format: list[str]) -> None:
        """
        Update various attributes.

        :param channel_format: the format of the colour channels.

        :return: None
        """
        # The channel values as a list:
        self._values = [element for element in self]
        # The channel format as a list:
        self._format = channel_format

    def __setitem__(self, index: int, value: Any) -> None:
        """
        Set self[index] to value.

        :param index: the index.
        :param value: the value.

        :return: None
        """
        # Do the regular listy stuff:
        super().__setitem__(index, value)

        # Check we are still valid:
        Colour.__validate_iterable(self)

        # Also update things that need updating:
        self.__update()

    def channel_value(self, channel: str) -> int:
        """
        Return the value of a given colour channel.

        :param channel: the channel as a string.

        :return: the channel value
        """
        # Check the channel exists:
        if channel not in self.format:
            raise ValueError(f"{channel} is not a valid channel.\n{self.format}")

        return self[self.format.index(channel)]

    def reformat(self, new_format: list[str]) -> None:
        """
        Change the order of the channel values to match a new format.

        :param new_format: a new channel format. Must be a permutation of the current format.

        :return: None
        """
        # Check the new format is a permutation of the current format:
        if not is_permutation(new_format, self.format):
            raise ValueError(
                f"New channel format must be a permutation of the current format.\n"
                f"Current: {self.format}"
                f"New: {new_format}"
            )

        # The new order of the channel indices in terms of the current format:
        new_index_order = [self.format.index(channel) for channel in new_format]
        # Rearrange the channel values to be in the new order:
        new_values = [self[index] for index in new_index_order]

        # Refresh the instance using the new values and format:
        self.__init__(new_values, new_format)

    def show(self) -> None:
        """
        Show this colour.
        :return: None
        """
        raise NotImplementedError(f"show not yet implemented for Colour.")

    @classmethod
    def colour_average(
        cls,
        first_colour: Colour,
        second_colour: Colour,
        weights: list[float | int] | None = None,
    ) -> Colour:
        """
        Find the average of two colours. Uses the channel format of the first colour.

        :param first_colour: a Colour.
        :param second_colour: a Colour.
        :param weights: how much to weight each colour by.

        :return: the average of the two colours.
        """
        # Set equal weights if not given:
        if weights is None:
            weights = [1, 1]
        # Check that weights is the right format:
        check_type(weights, [list, int], "weights")
        # Make sure the colours are both using the first colour's channel format:
        second_colour.reformat(first_colour.format)

        # Calculate normalised weights:
        [weight_1, weight_2] = [weight / sum(weights) for weight in weights]
        # Use them to find the average of the two colours
        new_values = [
            round(weight_1 * first_colour_value + weight_2 * second_colour_value, None)
            for first_colour_value, second_colour_value in zip(first_colour.values, second_colour.values)
        ]

        # Make it a colour:
        return cls(new_values, first_colour.format)

    @staticmethod
    def __validate_iterable(iterable: Iterable[int]) -> None:
        """
        Check the given iterable is acceptable.

        :param iterable: an iterable (supposedly).

        :return: None
        """
        # Check the type is correct:
        try:
            iter(iterable)
        except:
            raise TypeError(f"{type(iterable)} object is not iterable.\n{iterable}")
        # Check the length is correct:
        if len(iterable) != 3:
            raise ValueError(f"Iterable must have exactly 3 elements. Found {len(iterable)}.\n{iterable}")
        # Check the types of the elements are correct:
        for element in iterable:
            check_type(element, int, "an element of the colour iterable")

    @staticmethod
    def __validate_channel_format(channel_format: list[str]) -> None:
        """
        Check the given channel format is acceptable.

        :param channel_format: a list of channel names (supposedly).

        :return: None
        """
        # Check the type is correct:
        check_type(channel_format, list, "the channel_format")
        # Check the types of the elements are correct:
        for channel in channel_format:
            check_type(channel, str, "an element of the channel format list")
        # Check there are no duplicates:
        if len(channel_format) != len(set(channel_format)):
            raise ValueError(f"Duplicate channel name in the channel format:\n{channel_format}")
