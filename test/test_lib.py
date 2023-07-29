import unittest

from rubiks.lib import is_permutation, check_type


class IsPermutationTest(unittest.TestCase):
    """
    Test some use cases for the is_permutation function.
    """

    def setUp(self) -> None:

        self.list_1 = [1, 2, 3, 4]
        self.list_2 = [3, 2, 4, 1]
        self.list_3 = [5, 2, 3, 4]
        self.list_4 = [1, 2, 3]

    def test_true_case(self) -> None:
        """
        Test a case which should be a permutation.
        """
        self.assertTrue(is_permutation(self.list_1, self.list_2))

    def test_false_case(self) -> None:
        """
        Test a case which should not be a permutation.
        """
        self.assertFalse(is_permutation(self.list_1, self.list_3))

    def test_different_length_case(self) -> None:
        """
        Test that providing lists of different lengths returns False.
        """
        self.assertFalse(is_permutation(self.list_1, self.list_4))


class CheckTypeTest(unittest.TestCase):
    """
    Test some use cases for the check_type function.
    """

    def setUp(self) -> None:

        self.str_object = "asdf"

    def test_iterable_equivalent_case(self) -> None:
        """
        Test that providing a single type in a list is equivalent to just providing a single type.
        """
        # Both should not raise any errors:
        check_type(self.str_object, str)
        check_type(self.str_object, [str])

    def test_true_case(self) -> None:
        """
        Test a case which should be the correct type.
        """
        check_type(self.str_object, str)

    def test_false_case(self) -> None:
        """
        Test a case which should not be the correct type.
        """
        with self.assertRaises(TypeError):
            check_type(self.str_object, int)
