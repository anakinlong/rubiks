"""
The colour class.
"""

# import constants (circular import problems)
import random_functions as rf
from typing import List, Iterable, Union

DEFAULT_COLOUR = [0, 0, 0]
DEFAULT_FORMAT = ["b", "g", "r"]


class Colour(list):
    def __init__(self, iterable: Union[Iterable, None] = None, channel_format: List[str] = None):
        """
        A colour :^). This is basically a list with a few extra features. Currently only supports 3 colour channels.

        :param iterable: a list of colour channel values. Defaults to [0, 0, 0].
        :param channel_format: the format of the colour channels, in the form of a list of strings.
        Defaults to ["b", "g", "r"].
        """

        # Iterable:
        if iterable == None:
            iterable = DEFAULT_COLOUR

        # Check that the iterable we have been given is the right shape:
        if len(iterable) != 3:
            raise ValueError(f"A colour must be made of a list with exactly 3 values.")

        # Check that the elements of the iterable are integers:
        for x in iterable:
            rf.check_type(x, [int], "an element of iterable")

        # Behave like a normal list mostly:
        super().__init__(iterable)

        # Format:
        if channel_format == None:
            channel_format = DEFAULT_FORMAT
        # Check there are no duplicates:
        if len(channel_format) != len(set(channel_format)):
            raise ValueError(f"Duplicate channel name in the channel format:\n" f"{channel_format}")
        self.format = channel_format

        # Also update things that need updating:
        self.update()

    def update(self):
        """
        Updates various properties.
        """

        # The channel values as a list:
        self.values = [x for x in self]

    def __setitem__(self, i, o):
        # Do the regular listy stuff:
        super().__setitem__(i, o)

        # Check that we are still the right shape:
        if len(self) != 3:
            raise ValueError(f"Colour is now an invalid shape!")

        # Check all the elements are still ints:
        for x in self:
            rf.check_type(x, [int], "a colour element")

        # Also update things that need updating:
        self.update()

    def channel_value(self, channel: str):
        """
        Return the value of a given colour channel.
        """

        # Check it's a channel that exists:
        if channel not in self.format:
            print(f"'{channel}' is not a valid channel.")
            return None

        # Return its value:
        channel_index = self.format.index(channel)

        return self[channel_index]

    def reformat(self, new_format: List[str]):
        """
        Change the order of the channel values to match a new format.
        """

        # Check that the new format is a permutation of the current:
        if not rf.is_permutation(new_format, self.format):
            raise ValueError(
                f"Invalid format specified. Please use a permutation of the current format:\n" f"{self.format}"
            )

        # Get the indices of the channels of the new format in terms of the old one:
        indices = [self.format.index(channel) for channel in new_format]

        # Rearrange the channel values according to these new indices:
        new_values = [0, 0, 0]
        for i, value in enumerate(self):
            new_values[indices[i]] = value

        # Set the new values and format:
        self.__init__(new_values, new_format)

    def show(self):

        raise NotImplementedError()

    def plot(self):

        raise NotImplementedError()


if __name__ == "__main__":

    a = [1, 2, 3]

    b = Colour(a)
    print(b)
    b.reformat(["r", "g", "b"])
    print(b)
