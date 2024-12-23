import numpy as np


class IsecConflict:
    """
    Point as the result of intersection with the corresponding coordinates on both surfaces
    """

    def __init__(self, fracture_A, fracture_B, points_false_A, points_false_B, points_init_ind_A, points_init_ind_B):
        """
        :param vertices: vertices of the fracture as numpy array 3x4
        :param x_loc: normalized vector of local x-axis as numpy array 3x1
        :param y_loc: normalized vector of local y-axis as numpy array 3x1
        :param area: as area of the surface
        """

        self.fracture_A = fracture_A
        self.fracture_B = fracture_B
        self.points_false_A = points_false_A
        self.points_init_ind_A = points_init_ind_A
        self.points_false_B = points_false_B
        self.points_init_ind_B = points_init_ind_B

        self.angle = None  # unused

        self.AB_dist = None
        self.AB_angle = None  # unused

        self.BA_dist = None
        self.BA_angle = None  # unused

    def get_distance(self):
        self.AB_dist = self.solve_conflict(self.fracture_A, self.fracture_B, self.points_false_A,
                                           self.points_init_ind_A)
        self.BA_dist = self.solve_conflict(self.fracture_B, self.fracture_A, self.points_false_B,
                                           self.points_init_ind_B)

    def solve_conflict(self, fracture_A, fracture_B, points_false_A, points_init_ind_A):

        AB_dist = []
        for i in range(0, len(points_false_A)):
            dist = []
            A_vert = fracture_A.vertices[points_init_ind_A[i]]

            if points_false_A[i] != []:
                A_points_false = fracture_A.transform(points_false_A[i])
                loc_B_points_false = fracture_B.back_transform(A_points_false)
                ind_points_false = fracture_B.internal_point_2d(loc_B_points_false)
                if len(ind_points_false) > 0:
                    dist.append(np.linalg.norm(points_false_A[i] - A_vert))
            distx = fracture_B.dist_from_plane(A_vert)
            ort_proj = A_vert + distx * fracture_B.normal
            b_ref_point = fracture_B.back_transform(ort_proj)
            ind = fracture_B.internal_point_2d(b_ref_point)
            if len(ind) > 0:
                dist.append(np.abs(distx))
            if len(dist) > 0:
                AB_dist.append(min(dist))

        return AB_dist
