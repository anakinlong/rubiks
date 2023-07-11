"""
Random useful functions.
"""

from typing import List, Any


def is_permutation(list_1: List[Any], list_2: List[Any]) -> bool:
    """
    Check if two lists are permutations of each other.

    :param list_1: a list.
    :param list_2: a list.

    :return answer: a bool of whether the lists are permutations of each other.
    """

    # We do this by sorting both of them and seeing if they produce the same sorted list:
    answer = (sorted(list_1) == sorted(list_2))
    return answer


def check_type(var: Any, types: List[type], name: str) -> None:
    """
    Checks that type(var) is one of the expected types, and raise an error if not.

    :param var: a variable.
    :param types: a list of acceptable types:
    :param name: the name of the variable that shows up in the error message.

    :return: None.
    """

    if type(var) not in types:
        raise TypeError(
            f"{type(var)} is an invalid type for {name}. Must be from {types}."
        )


if __name__ == "__main__":

    a = 10
    check_type(a, [str, list], "a")
