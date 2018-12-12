
import numpy as np
import enum


class Axis(enum.IntEnum):
    """
    Axis of the 2d bspline.
    """
    u = 0
    v = 1



class IsecPoint:
    """
    Point as the result of intersection with correspondind coordinates on both surfaces

    """
    def __init__(self, surf1, iu1, iv1, uv1, add_patches, surface_boundary_flag, flag, sum_idx, surf2, iu2, iv2, uv2, xyz):
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
                if np.logical_and(self.iu1[i] == iu,self.iv1[i] == iv):
                    belongs = 1
                    break

        return belongs

    def extend_patches(self, flag):
        # extend reference of the patches on the second surface

        n = 0

        u_bound_low = np.logical_or(flag[0] != 0, self.iu2[0] != 0)
        u_bound_high = np.logical_or(flag[0] != 1, self.iu2[0] != self.surf2.u_basis.n_intervals - 1)
        u_bound_neg = np.logical_or(u_bound_low, u_bound_high)

        v_bound_low = np.logical_or(flag[1] != 0, self.iu2[0] != 0)
        v_bound_high = np.logical_or(flag[1] != 1, self.iu2[0] != self.surf2.v_basis.n_intervals - 1)
        v_bound_neg = np.logical_or(v_bound_low, v_bound_high)

        if np.logical_and(flag[0] != -1, u_bound_neg):
            direction_u = 2 * flag[0] - 1
            self.iu2.append(self.iu2[0] + direction_u)
            self.iv2.append(self.iv2[0])
            n = n + 1

        if np.logical_and(flag[0] != -1, v_bound_neg):
            direction_v = 2 * flag[1] - 1
            self.iu2.append(self.iu2[0])
            self.iv2.append(self.iv2[0] + direction_v)
            n = n + 1

        if n == 2:
            self.iu2.append(self.iu2[0] + direction_u)
            self.iv2.append(self.iv2[0] + direction_v)

    def add_patches(self, sum_idx, flag):
        # add neighbour patches in the case when intersection computed
        # using main threads lies on patch boundary

        n = len(self.iu1)

        direction = 2 * flag - 1

        if sum_idx == 0:
            for i in range(n):
                self.iu1.append(self.iu1[0]+direction)
                self.iv1.append(self.iv1[i])
        elif sum_idx == 1:
            for i in range(n):
                self.iu1.append(self.iu1[i])
                self.iv1.append(self.iv1[0]+direction)
