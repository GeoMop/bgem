from bgem.polygons import polygons as poly
import numpy as np
import scipy.linalg as la
from bgem.stochastic import isec_conflict as IC

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

        self._get_isec_eqn()
        points_A,points_false_A, points_init_ind_A = self._get_frac_isecs(self.fracture_A, self.fracture_B,self.loc_x0_A,self.loc_direct_C_A)
        points_B,points_false_B, points_init_ind_B = self._get_frac_isecs(self.fracture_B, self.fracture_A,self.loc_x0_B,self.loc_direct_C_B)
        conflict = IC.IsecConflict(self.fracture_A,self.fracture_B, points_false_A ,points_false_B,points_init_ind_A, points_init_ind_B)

        return points_A, points_B, conflict

    def _get_frac_isecs(self,fracture_A,fracture_B,loc_x0,loc_direct):

        ind_points = []
        points, points_false, points_init_ind = fracture_A.get_isec_with_line(loc_x0,loc_direct)
        if len(points) > 0:
            A_points = fracture_A.transform(points)
            loc_B_points = fracture_B.back_transform(A_points)
            ind_points = fracture_B.internal_point_2d(loc_B_points)

        if len(ind_points) > 0:
            points = A_points[ind_points, :]  # _clear
        else:
            points = []

        #points_false, points_init


        return points, points_false, points_init_ind

    def _get_isec_eqn(self):

        normal_A = self.fracture_A.normal
        normal_B = self.fracture_B.normal

        a_1 = normal_A[0,0]
        a_2 = normal_A[0,1]
        a_3 = normal_A[0,2]

        b_1 = normal_B[0,0]
        b_2 = normal_B[0,1]
        b_3 = normal_B[0,2]

        self.direct_C = np.array([[a_2 * b_3 - a_3 * b_2, a_3 * b_1 - a_1 * b_3, a_1 * b_2 - a_2 * b_1 ]])
        self.direct_C = self.direct_C/np.linalg.norm(self.direct_C)
       # np.cross(normal_A,normal_B)#
        a_4 = self.fracture_A.distance
        b_4 = self.fracture_B.distance

        rhs = np.array([[-a_4, -b_4, 0]])

        c_1 = self.direct_C[0,0]
        c_2 = self.direct_C[0,1]
        c_3 = self.direct_C[0,2]

        x0 = a_4 * b_3 * c_2 + a_2 * b_4 * c_3 - a_4 * b_2 * c_3 - a_3 * b_4 * c_2
        y0 = a_3 * b_4 * c_1 + a_4 * b_1 * c_3 - a_1 * b_4 * c_3 - a_4 * b_3 * c_1
        z0 = a_4 * b_2 * c_1 + a_1 * b_4 * c_2 - a_2 * b_4 * c_1 - a_4 * b_1 * c_2

        mat = np.array([normal_A[0,:].T,normal_B[0,:].T,self.direct_C[0,:].T])
        dt = np.linalg.det(mat)

        self.x_0 = np.array([[x0, y0, z0]])/dt


        #testao = self.fracture_A.normal @ self.fracture_A.centre.T + self.fracture_A.distance
        #testbo = self.fracture_B.normal @  self.fracture_B.centre.T + self.fracture_B.distance
        #testa = self.fracture_A.normal @ self.x_0.T + self.fracture_A.distance
        #testb = self.fracture_B.normal @  self.x_0.T + self.fracture_B.distance


        # az kdyz bude potvrzena existence
        #self.loc_x0_A, self.loc_direct_C_A = self._transform_to_local(self.x_0,self.direct_C, self.fracture_A)
        #self.loc_x0_B, self.loc_direct_C_B = self._transform_to_local(self.x_0,self.direct_C, self.fracture_B)

        self.loc_x0_A = self.fracture_A.back_transform(self.x_0)
        self.loc_direct_C_A = self.fracture_A.back_transform_clear(self.direct_C)

        self.loc_x0_B = self.fracture_B.back_transform(self.x_0)
        self.loc_direct_C_B = self.fracture_B.back_transform_clear(self.direct_C)

    def _transform_to_local(self,x0,direct,fracture):
        x0 -= fracture.centre
        #a = fracture.plane_coor_system
        loc_x0 = fracture.plane_coor_system @ x0
        loc_direct = fracture.plane_coor_system @ direct
        #aspect = np.array([0.5 * fracture.r, 0.5 * fracture.aspect * fracture.r, 1], dtype=float)
        #loc_x0 /= aspect[0:2] # [:, :]
        #loc_direct /= aspect[0:2]
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

