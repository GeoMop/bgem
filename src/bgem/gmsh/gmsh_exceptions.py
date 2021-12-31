import warnings


def make_warning(cls):
    """
    Takes 'class_name' of an object and creates new type 'class_nameWarning'
    as an descendant of Warning class.
    Used for retyping error classes to warnings.
    """
    return type(cls.__name__ + "Warning", (Warning,), {})


class GmshError(Exception):
    pass


class FragmentationError(GmshError):
    pass


class BoolOperationError(GmshError):
    pass


class GetBoundaryError(GmshError):
    pass
