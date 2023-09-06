"""
Colour class for rubiks images.
"""

from __future__ import annotations
import numpy as np
from typing import Any, Iterable
from .lib import check_type


class Pixel(np.ndarray):
    """
    A pixel :^). This is basically a numpy.ndarray with a few extra features.
    """

    def __init__(self, iterable: Iterable[int], coordinates: tuple[int | None] = (None, None)) -> None:
        """
        A pixel :^). This is basically a numpy.ndarray with a few extra features.

        :param iterable: an iterable with length three containing positive integer elements.
        :param coordinates: an iterable with length three containing positive integer elements.
        """
        # Input validation:
        Pixel.__validate_iterable(iterable)

        # Do regular ndarray stuff:
        super().__init__(iterable)

        # Store the coordinates:
        self._x, self._y = coordinates

    @property
    def x(self) -> int:
        """
        The 0-indexed x coordinate of the pixel in its image.
        """
        return self._x

    @property
    def y(self) -> int:
        """
        The 0-indexed y coordinate of the pixel in its image.
        """
        return self._y

    def __setitem__(self, index: int, value: Any) -> None:
        """
        Set self[index] to value.

        :param index: the index.
        :param value: the value.

        :return: None
        """
        # Do the regular ndarray __setitem__:
        super().__setitem__(index, value)

        # Check this is still a valid pixel:
        Pixel.__validate_iterable(self)

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
        except TypeError:
            raise
        # Check the length is correct:
        if len(iterable) != 3:
            raise ValueError(f"Iterable must have exactly 3 elements. Found {len(iterable)}.\n{iterable}")
        # Check the types of the elements are correct:
        for element in iterable:
            check_type(element, int, "an element of the pixel iterable")
