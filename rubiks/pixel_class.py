"""
Colour class for rubiks images.
"""

import numpy as np
from typing import Any, Iterable

from numpy._typing import NDArray


class Pixel(np.ndarray):
    """
    A pixel :^). This is basically a numpy.ndarray with a few extra features.
    """

    def __new__(cls, array: np.ndarray[int], coordinates: tuple[int | None] = (None, None)) -> None:
        """
        A pixel :^). This is basically a numpy.ndarray with a few extra features.

        :param iterable: an iterable with length three containing positive integer elements.
        :param coordinates: an iterable with length three containing positive integer elements.
        """
        # np.ndarrays are weird to subclass. See https://numpy.org/doc/stable/user/basics.subclassing.html
        # Input validation:
        Pixel.__validate_array(array)

        # Create an ndarray:
        obj = np.asarray(array).view(cls)

        # Store the coordinates:
        obj.x, obj.y = coordinates

        return obj

    def __array_finalize__(self, obj: np.ndarray[int]) -> None:
        """
        Set the public attributes of the new ndarray.
        """
        self.x = getattr(obj, "x", None)
        self.y = getattr(obj, "y", None)

    @staticmethod
    def __validate_array(array: np.ndarray[int]) -> None:
        """
        Check the given array is acceptable.

        :param array: an array (supposedly).

        :return: None
        """
        # Check the type is correct:
        try:
            iter(array)
        except TypeError:
            raise
        # Check the length is correct:
        if len(array) != 3:
            raise ValueError(f"Array must have exactly 3 elements. Found {len(array)}.\n{array}")
