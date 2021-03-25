

class GmshError(Exception):
    pass

class FragmentationError(GmshError):
    pass

class BoolOperationError(GmshError):
    pass

class GetBoundaryError(GmshError):
    pass