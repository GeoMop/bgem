
class SurfacePoint:
    """
    Point belonging to the surface

    """
    def __init__(self, surf, iuv, uv):
        """
        :param surf: surface
        :param iuv: numpy array 2x1, corresponds to the position of the patch on the surface (defined by knot vectors )
        :param uv: numpy array 2x1 of local coordinates

        """
        self.surf = surf
        self.iuv = iuv
        self.uv = uv

        self.surface_boundary_flag = 0

