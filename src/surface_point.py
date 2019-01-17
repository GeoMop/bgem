import numpy as np


class SurfacePoint:
    """
    Class to represent a point on the surface including the patch coordinates.

    """
    def __init__(self, surf, iuv, uv):
        """
        :param surf: surface
        :param iuv: numpy array 2x1, corresponds to the position of the patch on the surface (defined by knot vectors )
        :param uv: numpy array 2x1 of local coordinates

        """
        self.surf = surf
        self.iuv = []
        self.iuv.append(iuv)
        self.uv = uv

        self.surface_boundary_flag = self.extend_patches()

    def extend_patches(self):
        """
        add reference to the neighboring patches (if point lies on patches interface)
        :return:
        """
        ui = self.surf.u_basis.knots[self.iuv[0][0] + 2:self.iuv[0][0] + 4]
        vi = self.surf.v_basis.knots[self.iuv[0][1] + 2:self.iuv[0][1] + 4]

        # does not work
        #ui = self.surf.u_basis.knot_interval_bounds(self.iuv[0][0])
        #vi = self.surf.u_basis.knot_interval_bounds(self.iuv[0][1])

        interface_flag = np.zeros([2], 'int')
        boundary_flag = np.zeros([2], 'int')

        interface_flag[0], boundary_flag[0] = self._curve_boundary_intersection(self.uv[0], ui)
        interface_flag[1], boundary_flag[1] = self._curve_boundary_intersection(self.uv[1], vi)

        surface_boundary_flag = np.logical_or(boundary_flag[0], boundary_flag[1])

        n = 0

        if boundary_flag[0] == 0:
            iuv = np.zeros([2], dtype=int)
            iuv[0] = self.iuv[0][0] + interface_flag[0]
            iuv[1] = self.iuv[0][1]
            self.iuv.append(iuv)
            n = n + 1

        if boundary_flag[1] == 0:
            iuv = np.zeros([2], dtype=int)
            iuv[0] = self.iuv[0][0]
            iuv[1] = self.iuv[0][1] + interface_flag[1]
            self.iuv.append(iuv)
            n = n + 1

        if n == 2:
            iuv = np.zeros([2], dtype=int)
            iuv[0] = self.iuv[0][0] + interface_flag[0]
            iuv[1] = self.iuv[0][1] + interface_flag[1]
            self.iuv.append(iuv)

        return surface_boundary_flag

    @staticmethod
    def _curve_boundary_intersection(t, ti):
        """

        :param t: parameter value as double
        :param ti: parameter interval as numpy array (2x1)
        :return:
        interface_flag as   "-1" corresponds to lower bound of the curve
                            "1" corresponds to upper bound of the curve
                            "0" corresponds to the interior points of the curve
        boundary_flag as    "1" if parameter corresponds to the surface boundary, i.e, equal to 0 or 1
                            "0" otherwise

        """

        interface_flag = 0
        boundary_flag = 0

        if abs(ti[0] - t) == 0:
            interface_flag = -1
        elif abs(ti[1] - t) == 0:
            interface_flag = 1

        if np.logical_or(ti[0] == 0, ti[1] == 1):
            boundary_flag = 1

        return interface_flag, boundary_flag




