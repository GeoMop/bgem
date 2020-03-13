import enum
import numpy as np
from .surface_point import SurfacePoint

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
    def __init__(self, own_point, other_point, xyz):
        """
        TODO: variable paramaterers - curve point / surface point
        :param own_point: surface point
        :param other_point: surface point
        :param xyz: array of global coordinates as numpy array 3x1
        """


        self.duplicite_with = None
        # TODO: document attribute
        self.own_point: SurfacePoint = own_point
        # Position of the intersection on 'own' surface.
        self.other_point: SurfacePoint = other_point
        # Position of the intersection on 'other' surface.
        self.tol:float = 1  # have to be implemented
        self.xyz:np.array = xyz
        # Position of the intersection in 3d space.
        self.connected:bool = 0
        # True if the point is connected to a curve.


