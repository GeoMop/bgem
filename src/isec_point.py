
import numpy as np
import enum
import surface_point as SP

class Axis(enum.IntEnum):
    """
    Axis of the 2d bspline.
    """
    u = 0
    v = 1


#class SurfPoint:
#    """
#    Class to represent a point on the surface including the patch coordinates.
#    TODO: Use this to pass surface points to IsecPoint and at other places.
#    """
#    def __init__(self, surf, iuv, uv):
#        self.surf = surf
#        self.iuv = iuv
#        self.uv = uv


class IsecPoint:
    """
    Point as the result of intersection with corresponding coordinates on both surfaces

    """
    def __init__(self, surface_point_a,surface_point_b, xyz):
        """
        Todo: parameter description.
        Todo: consider introduction of SurfPoint class or NamedTuple to collect: (surf, iu, iv, uv)
        Todo: XYZ not used, can possibly be removed
        Todo: better names for flag and sum_idx, possibly use add_patches direstly not through constructor.

        TODO: replace surface_boundary_flag by a method SurfPoint.is_on_surf_boundary which use 'iuv' to check if the point is on the boundary.
        """

        self.surface_point_a = surface_point_a
        self.surface_point_b = surface_point_b


        self.tol = 1  # have to be implemented
        self.xyz = xyz
        self.connected = 0


