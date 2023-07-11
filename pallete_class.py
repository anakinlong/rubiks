"""
The pallete class.
"""

import colour_class as c
import colour_transformations as ct
import random_functions as rf
# import constants (circular import problems)
from typing import Dict, Tuple, Union, List


RUBIKS_PALLETE = {
    "g": c.Colour([ 72, 155,   0]),
    "w": c.Colour([255, 255, 255]),
    "r": c.Colour([ 52,  18, 183]),
    "y": c.Colour([  0, 213, 255]),
    "b": c.Colour([173,  70,   0]),
    "o": c.Colour([  0,  88, 255])
}


class Pallete(dict):

    def __init__(self, colour_dict: Dict[str, c.Colour]):
        """
        A colour pallete :^). Basically a dictionary.

        :param colour_dict: a dictionary where the keys are colour channel names and the values are colours.
        """

        # Check colour dict is a dictionary:
        rf.check_type(colour_dict, [dict], "colour_dict")

        # Check that the keys of the colour dict are strings:
        for k in colour_dict.keys():
            rf.check_type(k, [str], "a key in colour_dict")

        # Check that all the values of the colour dict are Colours:
        for v in colour_dict.values():
            rf.check_type(v, [c.Colour], "a value in colour_dict")

        # Do regular dictionary stuff:
        super().__init__(colour_dict)

        # Also update things that need updating:
        self.update()


    def update(self):
        """
        Updates various properties.
        """

        # Colours and colour names:
        self.colour_names: List[str] = list(self.keys())
        self.colours: List[c.Colour] = list(self.values())
        # Combined as a dictionary:
        self.colour_dict = dict(zip(self.colour_names, self.colours))


    def __setitem__(self, key: str, value: c.Colour) -> None:

        # Check that the key and value are valid inputs:
        rf.check_type(key, [str], "a key")
        rf.check_type(value, [c.Colour], "a value")

        # Do normal dictionary things:
        super().__setitem__(key, value)

        # Also update things that need updating:
        self.update()


    def get_combined_colour_dict(self, *args, **kwargs):
        """
        Not implemented. Will raise an error.
        """
        raise NotImplementedError


class CombinedPallete(Pallete):

    def __init__(self, pallete: Pallete):

        combined_colour_dict = self.get_combined_colour_dict(pallete)

        super().__init__(combined_colour_dict)


    def get_combined_colour_dict(self, pallete: Pallete):
        """
        Return a dictionary of combinations of colours.
        """

        combined_colour_dict = {}
        # Get the colour names:
        names = pallete.colour_names
        l = len(names)

        # Loop through the combinations without repeating:
        for i in range(l):
            for j in range(l - i):
                # Get the two names:
                name_1, name_2 = names[i], names[j]
                # Combine the names:
                new_name = name_1 + name_2
                # Combine those two colours:
                new_colour = ct.colour_average(pallete[name_1], pallete[name_2])
                # Put them in the combined colour dictionary:
                combined_colour_dict[new_name] = new_colour

        return combined_colour_dict


if __name__ == "__main__":

    b = Pallete(RUBIKS_PALLETE)
    # print(b)
    print(b.colour_names)
    # print(b.colours)

    print("-" * 50)

    c = CombinedPallete(b)
    # print(c)
    print(c.colour_names)
    print(c["og"])
