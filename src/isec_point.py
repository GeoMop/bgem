
import numpy as np


class IsecPoint:
    """
    Point as the result of intersection with correspondind coordinates on both surfaces
    """
    def __init__(self, surf1, iu1, iv1, uv1, add_patches, surface_boundary_flag, flag, sum_idx, surf2, iu2, iv2, uv2, xyz):

        self.surf1 = surf1
        self.iu1 = iu1
        self.iv1 = iv1
        self.uv1 = uv1
        self.surface_boundary_flag = surface_boundary_flag

        self.surf2 = surf2
        self.iu2 = iu2
        self.iv2 = iv2
        self.uv2 = uv2

        self.tol = 1  # have to be implemented
        self.xyz = xyz
        self.connected = 0

        if add_patches == 1:
            self.add_patches(sum_idx, flag[2])

        if np.logical_or(flag[0] != -1, flag[1] != -1):
            self.extend_patches(flag[0:2])

    def belongs_to(self, iu, iv, isurf):

        belongs = 0
        if isurf == 0:
            for i in range(self.iu1.__len__):
                if np.logical_and(self.iu1[i] == iu,self.iv1[i] == iv):
                    belongs = 1
                    break

        return belongs

    def n_patches(self):

        niu1 = self.iu1.__len__
        niv1 = self.iv1.__len__
        niu2 = self.iu2.__len__
        niv2 = self.iv2.__len__

        if np.logical_and(np.logical_and(niu1 == niv1), np.logical_and(niu2 == niv2)):
            n1 = niu1
            n2 = niu2
        else:
            print("problem")  # assert

        same = 0
        if n1 == n2:
            same = n1

        return n1, n2, same

    def extend_patches(self, flag):
        # extend reference of the patches on the second surface

        if flag[0] != -1:
            direction_u = 2 * flag[0] - 1
            self.iu2.append(self.iu2[0] + direction_u)
            self.iv2.append(self.iv2[0])

        if flag[1] != -1:
            direction_v = 2 * flag[1] - 1
            self.iu2.append(self.iu2[0])
            self.iv2.append(self.iv2[0]+ direction_v)

        if np.logical_and(flag[0] != -1, flag[1] != -1):
            self.iu2.append(self.iu2[0] + direction_u)
            self.iv2.append(self.iv2[0] + direction_v)

    def add_patches(self, sum_idx, flag):
        # add neighbour patches in the case when intersection computed
        # using main threads lies on patch boundary

        n = self.iu1.__len__()

        direction = 2 * flag - 1

        if sum_idx == 0:
            for i in range(n):
                self.iu1.append(self.iu1[0]+direction)
                self.iv1.append(self.iv1[i])
        elif sum_idx == 1:
            for i in range(n):
                self.iu1.append(self.iu1[i])
                self.iv1.append(self.iv1[0]+direction)
