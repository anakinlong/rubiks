import unittest

from rubiks.colour_class import Colour
from rubiks.palette_class import Palette, PaletteWeights, DEFAULT_COLOUR_DICT


class PaletteTest(unittest.TestCase):
    """
    Test some use cases for the Palette class.
    """

    def setUp(self) -> None:

        self.rubiks_colour_dict = {
            "green": Colour([72, 155, 0]),
            "white": Colour([255, 255, 255]),
            "red": Colour([52, 18, 183]),
            "yellow": Colour([0, 213, 255]),
            "blue": Colour([173, 70, 0]),
            "orange": Colour([0, 88, 255]),
        }
        self.rubiks_palette = Palette(self.rubiks_colour_dict)

    def test_empty(self) -> None:
        """
        Test that not providing any arguments is equivalent to providing DEFAULT_COLOUR_DICT.
        """
        self.assertEqual(Palette(), Palette(DEFAULT_COLOUR_DICT))

    def test_names(self) -> None:
        """
        Test the names property returns the correct value.
        """
        self.assertEqual(self.rubiks_palette.names, ["green", "white", "red", "yellow", "blue", "orange"])

    def test_colours(self) -> None:
        """
        Test the colours property returns the correct value.
        """
        expected_colours = [
            Colour([72, 155, 0]),
            Colour([255, 255, 255]),
            Colour([52, 18, 183]),
            Colour([0, 213, 255]),
            Colour([173, 70, 0]),
            Colour([0, 88, 255]),
        ]
        self.assertEqual(self.rubiks_palette.colours, expected_colours)

    def test_colour_dict(self) -> None:
        """
        Test the colour_dict property returns the correct value.
        """
        self.assertEqual(self.rubiks_palette.colour_dict, self.rubiks_colour_dict)

    def test_setitem(self) -> None:
        """
        Test that dict[key] = value (__setitem__) works on Palette.
        """
        self.rubiks_palette["green"] = Colour([0, 0, 0])
        equivalent_palette = Palette(
            {
                "green": Colour([0, 0, 0]),
                "white": Colour([255, 255, 255]),
                "red": Colour([52, 18, 183]),
                "yellow": Colour([0, 213, 255]),
                "blue": Colour([173, 70, 0]),
                "orange": Colour([0, 88, 255]),
            }
        )
        self.assertEqual(self.rubiks_palette, equivalent_palette)

    def test_combine_colour_dict(self) -> None:
        """
        Test that the __combine_colour_dict method produces the expected result.
        """
        equivalent_colour_dict = {
            "white": Colour([255, 255, 255]),
            "black": Colour([0, 0, 0]),
            "whiteblack": Colour([128, 128, 128]),
        }
        self.assertEqual(Palette._Palette__combine_colour_dict(DEFAULT_COLOUR_DICT), equivalent_colour_dict)

    def test_combine_colour_dict_duplicate_keys(self) -> None:
        """
        Test that the __combine_colour_dict method produces the expected result.
        """
        with self.assertRaises(ValueError):
            weird_colour_dict = {
                "a": Colour(),
                "b": Colour(),
                "ab": Colour(),
            }
            Palette._Palette__combine_colour_dict(weird_colour_dict)

    def test_create_combined_palette(self) -> None:
        """
        Test that the create_combined_palette method produces the expected result.
        """
        equivalent_palette = Palette(
            {"white": Colour([255, 255, 255]), "black": Colour([0, 0, 0]), "whiteblack": Colour([128, 128, 128])}
        )
        self.assertEqual(Palette.create_combined_palette(DEFAULT_COLOUR_DICT), equivalent_palette)


class PaletteWeightsTest(unittest.TestCase):
    """
    Test some use cases for the PaletteWeights class.
    """

    def setUp(self) -> None:

        self.rubiks_colour_weights = {
            "green": 1.5,
            "white": 2,
            "red": 3,
            "yellow": 4.5,
            "blue": 5,
            "orange": 6,
        }
        self.rubiks_palette_weights = PaletteWeights(self.rubiks_colour_weights)

    def test_names(self) -> None:
        """
        Test the names property returns the correct value.
        """
        self.assertEqual(self.rubiks_palette_weights.names, ["green", "white", "red", "yellow", "blue", "orange"])

    def test_weights(self) -> None:
        """
        Test the weights property returns the correct value.
        """
        self.assertEqual(self.rubiks_palette_weights.weights, [1.5, 2, 3, 4.5, 5, 6])

    def test_weights_dict(self) -> None:
        """
        Test the weights_dict property returns the correct value.
        """
        self.assertEqual(self.rubiks_palette_weights.weights_dict, self.rubiks_colour_weights)

    def test_setitem(self) -> None:
        """
        Test that dict[key] = value (__setitem__) works on Palette.
        """
        self.rubiks_palette_weights["green"] = 15
        equivalent_palette_weights = PaletteWeights(
            {
                "green": 15,
                "white": 2,
                "red": 3,
                "yellow": 4.5,
                "blue": 5,
                "orange": 6,
            }
        )
        self.assertEqual(self.rubiks_palette_weights, equivalent_palette_weights)
