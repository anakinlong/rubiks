"""
Random useful internal functions.
"""

from typing import Any, Iterable


def is_permutation(list_1: list[Any], list_2: list[Any]) -> bool:
    """
    Check if two lists are permutations of each other.

    :param list_1: a list.
    :param list_2: a list.

    :return answer: a bool of whether the lists are permutations of each other.
    """
    # We do this by sorting both of them and seeing if they produce the same sorted list:
    answer = sorted(list_1) == sorted(list_2)
    return answer


def check_type(
    object: object, types: type | Iterable[type], name: str | None = None
) -> None:
    """
    Checks that the type of a variable is one of the expected types, and raise an error if not.

    :param var: a variable.
    :param types: an iterable of acceptable types:
    :param name: the name of the variable that shows up in the error message.

    :return: None.
    """
    # If only a single type given, put it in a list:
    if isinstance(types, type):
        types = [types]
    types = tuple(types)
    if not isinstance(object, types):
        if name is None:
            name = "<this object>"
        raise TypeError(
            f"{type(object)} is an invalid type for {name} ({object}). Must be from {types}."
        )


if __name__ == "__main__":

    a = 10
    check_type(a, [str, list], "a")
