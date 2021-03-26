from bgem.polygons import polygons as poly


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
        self.nisecs = -1
        self.pd = poly.PolygonDecomposition()


        sg_a, = self.pd.add_line(vertices[:,0], vertices[:,1])
        sg_b, = self.pd.add_line(vertices[:, 1], vertices[:, 2])
        sg_c, = self.pd.add_line(vertices[:, 2], vertices[:, 3])
        sg_d, = self.pd.add_line(vertices[:, 3], vertices[:, 0])
        in_wire = sg_a.wire[left_side]


    #@staticmethod
    def _add_intersection_point(self, isp):
        self.isecs[self.nisecs].append(isp)

    def _initialize_new_intersection(self):
        self.nisecs += 1
        self.isecs.append([])
