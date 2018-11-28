
import numpy as np
import enum


class Axis(enum.Enum):
    """
    Axis of the 2d bspline.
    """
    u = 0
    v = 1



class IsecPoint:
    """
    Point as the result of intersection with correspondind coordinates on both surfaces

    """
    def __init__(self, surf1, iu1, iv1, uv1, boundary_flag, flag, sum_idx, surf2, iu2, iv2, uv2, XYZ):
        """
        Todo: parameter description.
        Todo: consider introduction of SurfPoint class or NamedTuple to collect: (surf, iu, iv, uv)
        Todo: XYZ not used, can possibly be removed
        Todo: better names for flag and sum_idx, possibly use add_patches direstly not through constructor.

        """
        self.surf1 = surf1
        self.iu1 = iu1
        self.iv1 = iv1
        self.uv1 = uv1
        self.boundary_flag = boundary_flag


        self.surf2 = surf2
        self.iu2 = iu2
        self.iv2 = iv2
        self.uv2 = uv2


        self.tol = 1 # implement
        self.R3_coor = XYZ
        self.connected = 0


        if boundary_flag == 1:
            self.add_patches(sum_idx, flag[2])

        if np.logical_or(flag[0] != -1, flag[1] != -1):
            self.extend_patches(flag[0:2])

        # flag

    def belongs_to(self, iu, iv, isurf):
        """
        Todo: description.

        :param iu:
        :param iv:
        :param isurf:
        :return:
        """
        belongs = 0
        if isurf == 0:
            for i in range(self.iu1.__len__):
                if (np.logical_and(self.iu1[i] == iu),np.logical_and(self.iv1[i] == iv)):
                    belongs = 1
                    break

        return belongs

    #def duplicite(self,point):
    #    duplicite = 0

    #    if point.iu1.__len__






    #    return duplicite


    def connect(self):
        self.connected = 1

    def connected(self):
        return self.connected


    def extend_patches(self, flag):

        if flag[0]!=-1:
            direction_u = 2 * flag[0] - 1
            self.iu2.append(self.iu2[0] + direction_u)
            self.iv2.append(self.iv2[0])

        if flag[1]!=-1:
            direction_v = 2 * flag[1] - 1
            self.iu2.append(self.iu2[0])
            self.iv2.append(self.iv2[0]+ direction_v)

        if np.logical_and(flag[0]!=-1,flag[1]!=-1):
            self.iu2.append(self.iu2[0]+ direction_u)
            self.iv2.append(self.iv2[0]+ direction_v)


    def add_patches(self, sum_idx,flag):

        n = self.iu1.__len__()

        direction = 2*flag -1

        if sum_idx == 0:
            for i in range(n):
                self.iu1.append(self.iu1[0]+direction)
                self.iv1.append(self.iv1[i])
        elif sum_idx == 1:
            for i in range(n):
                self.iu1.append(self.iu1[i])
                self.iv1.append(self.iv1[0]+direction)
