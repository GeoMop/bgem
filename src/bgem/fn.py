"""
Various function programming tools.
"""

# Various functional imports
from functools import cached_property


def compose(*functions):
    """
    Return composition of functions:
    compose(A,B,C)(any args) is equivalent to A(B(C(any args))

    Useful for functional programming and dependency injection.
    """

    def composed_function(*args, **kwargs):
        # Start by applying the rightmost function with all arguments
        result = functions[-1](*args, **kwargs)
        # Then apply the rest in reverse order, each to the result of the previous
        for f in reversed(functions[:-1]):
            result = f(result)
        return result

    return composed_function
