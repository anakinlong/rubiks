"""
Pallete class for rubiks images.
"""

from colour_class import Colour
from lib import check_type
from typing import Any

# The rubiks cube colours as a dictionary:
RUBIKS_COLOUR_DICT = {
    "g": Colour([72, 155, 0]),
    "w": Colour([255, 255, 255]),
    "r": Colour([52, 18, 183]),
    "y": Colour([0, 213, 255]),
    "b": Colour([173, 70, 0]),
    "o": Colour([0, 88, 255]),
}


class Pallete(dict):
    """
    A colour pallete :^). Basically a dictionary.
    """

    def __init__(self, colour_dict: dict[str, Colour] | None) -> None:
        """
        A colour pallete :^). Basically a dictionary. Maps colour names to Colours.

        :param colour_dict: a dictionary where the keys are colour names and the values are colours. Defaults to
        RUBIKS_COLOUR_DICT.
        """
        # Use defaults if input(s) not given:
        if colour_dict is None:
            colour_dict = RUBIKS_COLOUR_DICT
        # Input validation:
        Pallete.__validate_colour_dict(colour_dict)

        # Do regular dictionary stuff:
        super().__init__(colour_dict)

        # Also update things that need updating:
        self.__update()

    def __update(self) -> None:
        """
        Updates various attributes.
        :return: None
        """
        # Colours and colour names:
        self.names = list(self.keys())
        self.colours = list(self.values())
        # Combined as a dictionary:
        self.colour_dict = dict(zip(self.names, self.colours))

    def __setitem__(self, key: str, value: Colour) -> None:
        """
        Set self[key] to value.
        :param key: a string key.
        :param value: a Colour value.
        :return: None
        """
        # Check that the key and value are valid inputs:
        Pallete.__validate_key_value_pair(key, value)

        # Do normal dictionary things:
        super().__setitem__(key, value)

        # Also update things that need updating:
        self.__update()

    @classmethod
    def create_combined_pallete(cls, colour_dict: dict[str, Colour]):
        """
        Create a "combined" pallete, where combinations of colours from the colour dict are also included.
        :param colour_dict: a dictionary mapping string colour names to colours.
        :return: a pallete with combinations of colours.
        """
        raise NotImplementedError("combined palletes not yet implemented")

    @staticmethod
    def __validate_key_value_pair(key: Any, value: Any) -> None:
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
    def __validate_colour_dict(colour_dict: Any) -> None:
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
