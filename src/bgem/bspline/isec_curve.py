import numpy as np


class IsecCurve:
    """
    Represents a curve arising from intersection of two surface
    """

    def __init__(self):
        """
        Constructor of the spline basis.
        :param degree: Degree of Bezier polynomials >=0.
        :param knots: Numpy array of the knots including multiplicities.
        """
        self.point = []
        self.own_neighbours = []
        self.other_neighbours = []
        self.surf = []
        self.loop = False

    # @classmethod
    def reverse(self):
        """
        performs reverse on all the lists corresponding to the curve in order to move boundary point (and all
        corresponding data) of the curve to the first position of the lists
        TODO: Method of IntersectionCurve
        TODO: Do we need curve_own_neighbours, curve_other_neighbours, curve_surf ?
        :return:
        """
        self.point.reverse()
        self.own_neighbours.reverse()
        self.other_neighbours.reverse()
        self.surf.reverse()

    # @classmethod
    def add_point(self, point, i_surf, own_info, other_info):
        """
        connects the point to the last curve
        :param point: as isec_point
        :param i_surf: as integer, defines ID of the surface [0/1]
        :param own_info: as integer, number candidates for the next point from own_isec_points
        :param other_info: as integer, number candidates for the next point from other_isec_points
        :return:
        """

        self.point.append(point)
        self.point[-1].connected = 1
        self.own_neighbours.append(own_info)
        self.other_neighbours.append(other_info)
        self.surf.append(i_surf)

    def loop_check(self):
        """
        detects closed curves, i.e., the first and the last points (of the curve) can be found on at least one common
        patch_id (on both surfaces)
        TODO: should be method of the IntersectionCurve
        TODO: We can use IsecPoint.connected to distinguish the initial point (e.g. with value 2), so we can detect the loop when ever we
        find neighbour with connected == 2 (rather use enum with values: unconnected, connected, initial)

        :return:
        """

        first_isec_point = self.point[0]
        last_isec_point = self.point[-1]
        point1_surf = self.surf[0]
        point2_surf = self.surf[-1]

        if point1_surf == point2_surf:
            n1 = len(first_isec_point.own_point.patch_id() & last_isec_point.own_point.patch_id())
            n2 = len(first_isec_point.other_point.patch_id() & last_isec_point.other_point.patch_id())
        else:
            n1 = len(first_isec_point.own_point.patch_id() & last_isec_point.other_point.patch_id())
            n2 = len(first_isec_point.other_point.patch_id() & last_isec_point.own_point.patch_id())

        if np.logical_and(n1 > 0, n2 > 0):
            self.loop = True
