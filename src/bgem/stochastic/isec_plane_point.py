import scipy.linalg as la


class IsecFracPlanePoint:
    """
    Point as the result of intersection with the corresponding coordinates on both surfaces

    """

    def __init__(self, id, coor):
        """
        :param id as position of the intersection in the list
        :param coor as absolute coordinate in R3 as numpy array 3x1
        """

        self.id = id
        self.coor = coor
        self.frac_id = []
        self.loc_coor = []

    def add_fracture_data(self, frac_id, loc_coor):

        already_found = 0
        for ids in self.frac_id:
            if self.frac_id == frac_id:
                already_found = 1

        if already_found == 0:
            self.frac_id.append(frac_id)
            self.loc_coor.append(loc_coor)

    def check_duplicity(self, coor, tol):
        # test or do not test fracture vertices?
        rel_tol = 2 * la.norm(self.coor - coor) / la.norm(self.coor + coor)
        if rel_tol >= tol:
            duplicity = False
        else:
            duplicity = True

        return duplicity
