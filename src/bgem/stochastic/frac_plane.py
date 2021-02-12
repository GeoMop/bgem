


class FracPlane:
    """
    Point as the result of intersection with corresponding coordinates on both surfaces

    """

    def __init__(self, vertices,x_loc,y_loc,area):
        """
        TODO: variable paramaterers - curve point / surface point
        :param surface_point_a: surface point
        :param surface_point_b: surface point
        :param xyz: array of global coordinates as numpy array 3x1
        """

        self.vertices = vertices
        self.x_loc = x_loc
        self.y_loc = y_loc
        self.area = area
        self.isecs = []

    #@staticmethod
    def _add_intersection_point(self, isp):
        self.isecs.append(isp)


