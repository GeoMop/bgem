class IsecFracPlanePoint:
    """
    Point as the result of intersection with corresponding coordinates on both surfaces

    """

    def __init__(self, frac_id1,frac_id2,loc_coor1,loc_coor2,coor,id):
        """
        TODO: variable paramaterers - curve point / surface point
        :param surface_point_a: surface point
        :param surface_point_b: surface point
        :param xyz: array of global coordinates as numpy array 3x1
        """

        self.frac_id1 = frac_id1
        self.frac_id2 = frac_id2
        self.loc_coor1 = loc_coor1
        self.loc_coor2 = loc_coor2
        self.coor = coor
        self.id = id
