"""
Rubik's constants.
"""

import colour_class as c


DEFAULT_COLOUR = [0, 0, 0]

DEFAULT_FORMAT = ["b", "g", "r"]

RUBIKS_PALLETE = {
    "g": c.Colour([ 72, 155,   0]),
    "w": c.Colour([255, 255, 255]),
    "r": c.Colour([ 52,  18, 183]),
    "y": c.Colour([  0, 213, 255]),
    "b": c.Colour([173,  70,   0]),
    "o": c.Colour([  0,  88, 255])
}
