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
    def names(self) -> list[str]:
        """
        The list of colour names. Equivalent of dictionary keys.
        """
        return self._names

    @property
    def colours(self) -> list[Colour]:
        """
        The list of colours. Equivalent of dictionary values.
        """
        return self._colours

    @property
    def colour_dict(self) -> dict[str, Colour]:
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

    def reformat(self, new_format: list[str]) -> None:
        """
        Change the order of the channel values in all colours to match a new format.

        :param new_format: a new channel format. Must be a permutation of the current formats.

        :return: None
        """
        # Reformat each colour:
        for colour in self._colours:
            colour.reformat(new_format)
        # Update other attributes:
        self.__update()

    def to_combined_palette(self) -> Palette:
        """
        Create a combined palette using this Palette instance.

        :return: a Palette instance where the colours are combinations of the colours in this Palette.
        """
        return Palette.create_combined_palette_from_colour_dict(self.colour_dict)

    @classmethod
    def create_combined_palette_from_colour_dict(cls, colour_dict: dict[str, Colour]) -> tuple[Palette, Palette]:
        """
        Create a "combined" palette, where the colours are combinations of colours from the colour dict.

        :param colour_dict: a dictionary mapping string colour names to colours.

        :return: a palette with combinations of colours from the original palette.
        """
        # Combine the colour dict:
        combined_colour_dict = Palette.__combine_colour_dict(colour_dict)

        return cls(combined_colour_dict)

    @staticmethod
    def __combine_colour_dict(colour_dict: dict[str, Colour]) -> dict[str, Colour]:
        """
        Create a "combined" colour dict, where combinations of colours from the colour dict are also included.

        :param colour_dict: a dictionary mapping string colour names to colours.

        :return: a colour dictionary with combinations of the colours in the input colour dictionary.
        """
        combined_colour_dict = {}
        # Get the colour names:
        names = list(colour_dict.keys())
        number_of_names = len(names)

        # Loop through the combinations without repeating:
        for i in range(number_of_names):
            for j in range(i + 1, number_of_names):
                # Get the two names:
                name_1, name_2 = names[i], names[j]
                # Combine the names:
                new_name = name_1 + name_2
                # Combine those two colours:
                new_colour = Colour.colour_average(colour_dict[name_1], colour_dict[name_2])
                # Put them in the combined colour dictionary:
                combined_colour_dict[new_name] = new_colour

        # Check that these two dictionaries do not share any keys:
        if any(key in colour_dict for key in combined_colour_dict) or any(
            key in combined_colour_dict for key in colour_dict
        ):
            raise ValueError(
                "Colour dict contains names such that combinations of them already exist in the names:\n"
                f"{colour_dict.keys()}"
            )

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
    def names(self) -> list[str]:
        """
        The list of colour names. Equivalent of dictionary keys.
        """
        return self._names

    @property
    def weights(self) -> list[float | int]:
        """
        The list of weights. Equivalent of dictionary values.
        """
        return self._weights

    @property
    def weights_dict(self) -> dict[str, float | int]:
        """
        The dictionary mapping colour names to weights. Equivalent of dictionary.
        """
        return self._weights_dict

    def validate_against_palette(self, palette: Palette) -> None:
        """
        Do input validation for the palette weights against a palette. Checks that they are 1:1 in terms of colours.

        :param palette: a Palette of Colours.

        :return: None
        """
        # Check that each colour in the palette has a weight:
        colours_with_no_weight = [colour_name for colour_name in palette.names if colour_name not in self._names]
        if colours_with_no_weight:
            raise ValueError(f"The following colours were not assigned weights:\n{colours_with_no_weight}")

        # Check that only colours from the palette are in the weights:
        extra_colours = [colour_name for colour_name in self._names if colour_name not in palette.names]
        if extra_colours:
            raise ValueError(
                f"The following colours are not in the palette but were assigned weights:\n{extra_colours}"
            )

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
