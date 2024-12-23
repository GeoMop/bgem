from bgem.polygons import polygons as poly
import scipy.linalg as la


class FracPlane:
    """
    Point as the result of intersection with corresponding coordinates on both surfaces

    """

    def __init__(self, vertices, vert_id, fracture):
        """
        :param vertices: vertices of the fracture as numpy array 3x4
        :param x_loc: normalized vector of local x-axis as numpy array 3x1
        :param y_loc: normalized vector of local y-axis as numpy array 3x1
        :param area: as area of the surface
        """

        self.vertices = vertices
        self.vert_id = vert_id
        self.n_vertices = vertices.shape[1]
        # self.zero =

        self.x_loc = x_loc
        self.y_loc = y_loc
        self.area = area
        self.isecs = []
        self.nisecs = -1
        self.pd = poly.PolygonDecomposition()

        # sg_a, = self.pd.add_line(np.array([0,0]), np.array([1,0]))
        # sg_b, = self.pd.add_line(np.array([1,0]), np.array([1,1]))
        # sg_c, = self.pd.add_line(np.array([1,1]), np.array([0,1]))
        # sg_d, = self.pd.add_line(np.array([0,1]), np.array([0,0]))

    #       # in_wire = sg_a.wire[left_side]

    # @staticmethod
    def _add_intersection_point(self, isp):
        self.isecs[self.nisecs].append(isp)

    def _initialize_new_intersection(self):
        self.nisecs += 1
        self.isecs.append([])

    def _check_duplicity(self, coor, tol, duplicity_with):

        if duplicity_with != -1:
            for i in range(0, 4):
                vertex_coor = self.vertices[i, :].transpose()
                rel_tol = 2 * la.norm(vertex_coor - coor) / la.norm(vertex_coor + coor)
                if rel_tol >= tol:
                    duplicity_with = self.vert_id[i]
                    break

        return duplicity_with
