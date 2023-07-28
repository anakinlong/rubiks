"""
Palette class for rubiks images.
"""

from __future__ import annotations
from .colour_class import Colour
from .lib import check_type

# A simple dictionary containing black and white:
DEFAULT_COLOUR_DICT = {
    "white": Colour([255, 255, 255]),
    "black": Colour([0, 0, 0]),
}


class Palette(dict):
    """
    A colour palette :^). Basically a dictionary.
    """

    def __init__(self, colour_dict: dict[str, Colour] | None = None) -> None:
        """
        A colour palette :^). Basically a dictionary. Maps colour names to Colours.

        :param colour_dict: a dictionary where the keys are colour names and the values are colours. Defaults to
        DEFAULT_COLOUR_DICT.
        """
        # Use defaults if input(s) not given:
        if colour_dict is None:
            colour_dict = DEFAULT_COLOUR_DICT
        # Input validation:
        Palette.__validate_colour_dict(colour_dict)

        # Do regular dictionary stuff:
        super().__init__(colour_dict)

        # Initialise attributes as None then update them:
        self._names = None
        self._colours = None
        self._colour_dict = None
        self.__update()

    @property
    def names(self):
        """
        The list of colour names. Equivalent of dictionary keys.
        """
        return self._names

    @property
    def colours(self):
        """
        The list of colours. Equivalent of dictionary values.
        """
        return self._colours

    @property
    def colour_dict(self):
        """
        The dictionary mapping colour names to colours. Equivalent of dictionary.
        """
        return self._colour_dict

    def __update(self) -> None:
        """
        Updates various attributes.

        :return: None
        """
        # Colours and colour names:
        self._names = list(self.keys())
        self._colours = list(self.values())
        # Combined as a dictionary:
        self._colour_dict = dict(zip(self._names, self._colours))

    def __setitem__(self, key: str, value: Colour) -> None:
        """
        Set self[key] to value.

        :param key: a string key.
        :param value: a Colour value.

        :return: None
        """
        # Check that the key and value are valid inputs:
        Palette.__validate_key_value_pair(key, value)

        # Do normal dictionary things:
        super().__setitem__(key, value)

        # Also update things that need updating:
        self.__update()

    @classmethod
    def create_combined_palette(cls, colour_dict: dict[str, Colour]) -> Palette:
        """
        Create a "combined" palette, where combinations of colours from the colour dict are also included.

        :param colour_dict: a dictionary mapping string colour names to colours.

        :return: a palette with combinations of colours.
        """
        # Combine the colour dict:
        combined_colour_dict = Palette.__combine_colour_dict(colour_dict)

        return cls(combined_colour_dict)

    @staticmethod
    def __combine_colour_dict(colour_dict: dict[str, Colour]) -> dict[str, Colour]:
        """
        Create a "combined" colour dict, where combinations of colours from the colour dict are also included.

        :param colour_dict: a dictionary mapping string colour names to colours.

        :return: a colour dictionary with combinations of the colours.
        """
        combined_colour_dict = {}
        # Get the colour names:
        names = colour_dict.keys()
        l = len(names)

        # Loop through the combinations without repeating:
        for i in range(l):
            for j in range(l - i):
                # Get the two names:
                name_1, name_2 = names[i], names[j]
                # Combine the names:
                new_name = name_1 + name_2
                # Combine those two colours:
                new_colour = Colour.colour_average(colour_dict[name_1], colour_dict[name_2])
                # Put them in the combined colour dictionary:
                combined_colour_dict[new_name] = new_colour

        return combined_colour_dict

    @staticmethod
    def __validate_key_value_pair(key: str, value: Colour) -> None:
        """
        Check the given key, value pair is acceptable.

        :param key: a string key.
        :param value: a Colour value.

        :return: None
        """
        # Check they are the correct types:
        check_type(key, str, "a key")
        check_type(value, Colour, "a value")

    @staticmethod
    def __validate_colour_dict(colour_dict: dict[str, Colour]) -> None:
        """
        Check the given colour dict is acceptable.

        :param colour_dict: a dictionary mapping names to colours.

        :return: None
        """
        # Check colour dict is a dictionary:
        check_type(colour_dict, dict, "colour_dict")
        # Check that the keys of the colour dict are strings:
        for k in colour_dict.keys():
            check_type(k, str, "a key in colour_dict")
        # Check that all the values of the colour dict are Colours:
        for v in colour_dict.values():
            check_type(v, Colour, "a value in colour_dict")


# TODO create a CombinedPalette class. This will be composed of two palettes - the original, and the one with ONLY the
# colours formed from combinations of the original colours. Will make using it easier. Can also have a method to
# create one single palette with all those colours together.


class PaletteWeights(dict):
    """
    Colour weights. Basically a dictionary.
    """

    def __init__(self, weights_dict: dict[str, float | int]) -> None:
        """
        Palette weights. Basically a dictionary.

        :param weights_dict: a dictionary where the keys are colour names and the values are float weights.
        """
        # Input validation:
        PaletteWeights.__validate_weights_dict(weights_dict)

        # Do regular dictionary stuff:
        super().__init__(weights_dict)

        # Initialise attributes as None then update them:
        self._names = None
        self._weights = None
        self._weights_dict = None
        self.__update()

    @property
    def names(self):
        """
        The list of colour names. Equivalent of dictionary keys.
        """
        return self._names

    @property
    def weights(self):
        """
        The list of weights. Equivalent of dictionary values.
        """
        return self._weights

    @property
    def weights_dict(self):
        """
        The dictionary mapping colour names to weights. Equivalent of dictionary.
        """
        return self._weights_dict

    def __update(self) -> None:
        """
        Updates various attributes.

        :return: None
        """
        # Weights and colour names:
        self._names = list(self.keys())
        self._weights = list(self.values())
        # Combined as a dictionary:
        self._weights_dict = dict(zip(self._names, self._weights))

    def __setitem__(self, key: str, value: Colour) -> None:
        """
        Set self[key] to value.

        :param key: a string key.
        :param value: a float or int value.

        :return: None
        """
        # Check that the key and value are valid inputs:
        PaletteWeights.__validate_key_value_pair(key, value)

        # Do normal dictionary things:
        super().__setitem__(key, value)

        # Also update things that need updating:
        self.__update()

    @staticmethod
    def __validate_key_value_pair(key: str, value: float | int) -> None:
        """
        Check the given key, value pair is acceptable.

        :param key: a string key.
        :param value: a float or int value.

        :return: None
        """
        # Check they are the correct types:
        check_type(key, str, "a key")
        check_type(value, [float, int], "a value")

    @staticmethod
    def __validate_weights_dict(weights_dict: dict[str, float | int]) -> None:
        """
        Check the given weights dict is acceptable.

        :param weights_dict: a dictionary mapping colour names to weights.

        :return: None
        """
        # Check weights dict is a dictionary:
        check_type(weights_dict, dict, "weights_dict")
        # Check that the keys of the weights dict are strings:
        for k in weights_dict.keys():
            check_type(k, str, "a key in weights_dict")
        # Check that all the values of the weights dict are floats or ints:
        for v in weights_dict.values():
            check_type(v, [float, int], "a value in weights_dict")
