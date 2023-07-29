import unittest

from rubiks import Colour


class ColourTest(unittest.TestCase):
    """
    Test some use cases for the Colour class.
    """

    def setUp(self) -> None:

        self.colour = Colour([1, 2, 3], channel_format=["b", "g", "r"])

    def test_empty(self) -> None:
        """
        Test that supplying no arguments still produces a Colour.
        """
        self.assertIsInstance(Colour(), Colour)

    def test_values(self) -> None:
        """
        Test that the values property returns the expected result.
        """
        self.assertEqual(self.colour.values, [1, 2, 3])

    def test_format(self) -> None:
        """
        Test that the values property returns the expected result.
        """
        self.assertEqual(self.colour.format, ["b", "g", "r"])

    def test_setitem(self) -> None:
        """
        Test that list[index] = value (__setitem__) works on Colour.
        """
        self.colour[0] = 5
        self.assertEqual(self.colour, Colour([5, 2, 3], channel_format=["b", "g", "r"]))

    def test_channel_value(self) -> None:
        """
        Test that the channel_value method returns the expected result.
        """
        self.assertEqual(self.colour.channel_value("b"), 1)

    def test_channel_value_invalid_channel(self) -> None:
        """
        Test that the channel_value method raises a ValueError if the channel is not in the channel format.
        """
        with self.assertRaises(ValueError):
            self.colour.channel_value("x")

    def test_reformat(self) -> None:
        """
        Test that the reformat method returns the expected result.
        """
        # Reformat the current colour:
        new_channel_format = ["r", "g", "b"]
        self.colour.reformat(new_channel_format)
        # Create an equivalent colour:
        equivalent_colour = Colour([3, 2, 1], channel_format=["r", "g", "b"])
        self.assertEqual(self.colour, equivalent_colour)

    def test_reformat_invalid_format(self) -> None:
        """
        Test that the reformat method raises a ValueError if the format is not a permutation of the current format.
        """
        with self.assertRaises(ValueError):
            new_channel_format = ["x", "g", "b"]
            self.colour.reformat(new_channel_format)

    def test_show(self) -> None:
        """
        Test that the show method returns a not implemented error.
        """
        with self.assertRaises(NotImplementedError):
            self.colour.show()

    def test_colour_average_unweighted(self) -> None:
        """
        Test that the colour average method returns the expected result when not using weights.
        """
        # Calculate the average of these two colours:
        colour_1 = Colour([1, 2, 3], channel_format=["r", "g", "b"])
        colour_2 = Colour([10, 20, 30], channel_format=["g", "b", "r"])
        colour_average = Colour.colour_average(colour_1, colour_2)
        # Create an equivalent colour:
        equivalent_colour = Colour([16, 6, 12], channel_format=["r", "g", "b"])
        self.assertEqual(colour_average, equivalent_colour)

    def test_colour_average_weighted(self) -> None:
        """
        Test that the colour average method returns the expected result when using weights.
        """
        # Calculate the average of these two colours:
        colour_1 = Colour([1, 2, 3], channel_format=["r", "g", "b"])
        colour_2 = Colour([10, 20, 30], channel_format=["g", "b", "r"])
        weights = [1, 2]
        colour_average = Colour.colour_average(colour_1, colour_2, weights=weights)
        # Create an equivalent colour:
        equivalent_colour = Colour([20, 7, 14], channel_format=["r", "g", "b"])
        self.assertEqual(colour_average, equivalent_colour)

    def test_validate_iterable_iterable_type(self) -> None:
        """
        Test that the __validate_iterable method raises a TypeError when the iterable is not iterable.
        """
        with self.assertRaises(TypeError):
            Colour._Colour__validate_iterable(3)

    def test_validate_iterable_value(self) -> None:
        """
        Test that the __validate_iterable method raises a ValueError when the iterable does not have a length of 3.
        """
        with self.assertRaises(ValueError):
            Colour._Colour__validate_iterable([1, 2, 3, 4])

    def test_validate_iterable_element_type(self) -> None:
        """
        Test that the __validate_iterable method raises a TypeError when the iterable does not contain integer elements.
        """
        with self.assertRaises(TypeError):
            Colour._Colour__validate_iterable(["1", "2", "3"])

    def test_validate_channel_format_channel_format_type(self) -> None:
        """
        Test that the __validate_channel_format method raises a TypeError when the channel_format is not a list.
        """
        with self.assertRaises(TypeError):
            Colour._Colour__validate_channel_format(3)

    def test_validate_channel_format_value(self) -> None:
        """
        Test that the __validate_channel_format method raises a ValueError when the channel_format does not have a
        length of 3.
        """
        with self.assertRaises(ValueError):
            Colour._Colour__validate_channel_format(["1", "2", "3", "4"])

    def test_validate_channel_format_element_type(self) -> None:
        """
        Test that the __validate_channel_format method raises a TypeError when the channel_format does not contain
        string elements.
        """
        with self.assertRaises(TypeError):
            Colour._Colour__validate_channel_format([1, 2, 3])

    def test_validate_channel_format_element_duplicates(self) -> None:
        """
        Test that the __validate_channel_format method raises a ValueError when the channel_format does contains
        duplicates.
        """
        with self.assertRaises(ValueError):
            Colour._Colour__validate_channel_format(["1", "1", "2"])
