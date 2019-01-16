
import numpy as np
import enum
import surface_point as SP

class Axis(enum.IntEnum):
    """
    Axis of the 2d bspline.
    """
    u = 0
    v = 1

class IsecPoint:
    """
    Point as the result of intersection with corresponding coordinates on both surfaces

    """
    def __init__(self, surface_point_a, surface_point_b, xyz):
        """

        :param surface_point_a: surface point
        :param surface_point_b: surface point
        :param xyz: array of global coordinates as numpy array 3x1
        """


        self.surface_point_a = surface_point_a
        self.surface_point_b = surface_point_b
        self.tol = 1  # have to be implemented
        self.xyz = xyz
        self.connected = 0


