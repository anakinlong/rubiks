"""
Some useful public constants that aren't defined elsewhere already.
"""

from .colour_class import Colour
from .palette_class import Palette


# From Colour:
RUBIKS_GREEN = Colour([72, 155, 0])
RUBIKS_WHITE = Colour([255, 255, 255])
RUBIKS_RED = Colour([52, 18, 183])
RUBIKS_YELLOW = Colour([0, 213, 255])
RUBIKS_BLUE = Colour([173, 70, 0])
RUBIKS_ORANGE = Colour([0, 88, 255])

# From Palette:
RUBIKS_PALETTE = Palette(
    {
        "green": RUBIKS_GREEN,
        "white": RUBIKS_WHITE,
        "red": RUBIKS_RED,
        "yellow": RUBIKS_YELLOW,
        "blue": RUBIKS_BLUE,
        "orange": RUBIKS_ORANGE,
    }
)
