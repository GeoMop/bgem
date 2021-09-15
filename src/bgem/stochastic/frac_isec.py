from bgem.polygons import polygons as poly
import numpy as np
import scipy.linalg as la

class FracIsec:
    """
    Point as the result of intersection with corresponding coordinates on both surfaces

    """

    def __init__(self, fracture_A,fracture_B):
        """
        :param vertices: vertices of the fracture as numpy array 3x4
        :param x_loc: normalized vector of local x-axis as numpy array 3x1
        :param y_loc: normalized vector of local y-axis as numpy array 3x1
        :param area: as area of the surface
        """

        self.fracture_A = fracture_A
        self.fracture_B = fracture_B
        self.direct_C = None
        self.x_0 = None
        self.loc_x0_A = None
        self.loc_direct_C_A = None
        self.loc_x0_B = None
        self.loc_direct_C_B = None

    def _get_points(self):

        points_A = self._get_frac_isecs(self.fracture_A, self.fracture_B,self.loc_direct_C_A)
        points_B = self._get_frac_isecs(self.fracture_B, self.fracture_A,self.loc_direct_C_B)

    def _get_frac_isecs(self,fracture_A,fracture_B,loc_direct):

        self._get_isec_eqn()
        points = fracture_A.get_isec_with_line(self.x_0,loc_direct)
        points = fracture_B.internal_point(self, points)
        return points

    def _get_isec_eqn(self):

        normal_A = self.fracture_A.normal
        normal_B = self.fracture_B.normal

        a_1 = normal_A[0,0]
        a_2 = normal_A[0,1]
        a_3 = normal_A[0,2]

        b_1 = normal_B[0,0]
        b_2 = normal_B[0,1]
        b_3 = normal_B[0,2]

        self.direct_C = np.array([a_1 * b_2 - a_2 * b_1, a_3 * b_1 - a_1 * b_3, a_2 * b_3 - a_3 * b_2])
            #np.cross(normal_A, normal_B)

        a_4 = self.fracture_A.distance
        b_4 = self.fracture_B.distance

        c_1 = self.direct_C[0]
        c_2 = self.direct_C[1]
        c_3 = self.direct_C[2]

        detA = np.linalg.norm(self.direct_C)
        detAx = a_4 * b_3 * c_2 + a_2 * b_4 * c_3 - a_4 * b_2 * c_3 - a_3 * b_4 * c_2
        detAy = a_3 * b_4 * c_1 + a_4 * b_1 * c_3 - a_1 * b_4 * c_3 - a_4 * b_3 * c_1
        detAz = a_4 * b_2 * c_1 + a_1 * b_4 * c_2 - a_2 * b_4 * c_1 - a_4 * b_1 * c_2


        self.x_0 = 1/detA * np.array( [detAx, detAy, detAz])


        # az kdyz bude potvrzena existence
        self.loc_x0_A, self.loc_direct_C_A = self._transform_to_local(self.x_0,self.direct_C, self.fracture_A)
        self.loc_x0_B, self.loc_direct_C_B = self._transform_to_local(self.x_0,self.direct_C, self.fracture_B)

    def _transform_to_local(self,x0,direct,fracture):
        x0 -= fracture.centre
        a = fracture.plane_coor_system
        loc_x0 = fracture.plane_coor_system @ x0
        loc_direct = fracture.plane_coor_system @ direct
        aspect = np.array([0.5 * fracture.r, 0.5 * fracture.aspect * fracture.r, 1], dtype=float)
        loc_x0[:, :] /= aspect[None, :]
        loc_direct[:, :] /= aspect[None, :]
        return loc_x0, loc_direct

 #   def _transform_to_local(self,x_0,direct_C,fracture):
 #       loc_x0 = x_0# - fracture.centre
 #       loc_direct = direct_C# - fracture.centre
 #       #vertices = fracture.SquareShape()._points
 #       loc_x0 = fracture._plane_coor_system @ loc_x0
 #       loc_direct = fracture._plane_coor_system @ loc_direct
 #       aspect = np.array([0.5 * fracture.r, 0.5 * fracture.aspect * fracture.r, 1], dtype=float)
 #       loc_x0[:, :] /= aspect[None, :]
 #       loc_direct[:, :] /= aspect[None, :]
 #       return loc_x0, loc_direct

