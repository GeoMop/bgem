"""
Module with classes representing various B-spline and NURMS curves and surfaces.
These classes provide just basic functionality:
- storing the data
- evaluation of XYZ for UV
- derivatives
In future:
- serialization and deserialization using JSONdata - must make it an installable module
"""

"""
This module tries to approximate 2.5D array of terrain points
using B-Spline surface.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import numpy.linalg as la


__author__ = 'Jan Brezina <jan.brezina@tul.cz>, Jiri Hnidek <jiri.hnidek@tul.cz>, Jiri Kopal <jiri.kopal@tul.cz>'

    
class ParamError(Exception):
    pass

def check_matrix(mat, shape, values, idx=[]):
    '''
    Check shape and type of scalar, vector or matrix.
    :param mat: Scalar, vector, or vector of vectors (i.e. matrix). Vector may be list or other iterable.
    :param shape: List of dimensions: [] for scalar, [ n ] for vector, [n_rows, n_cols] for matrix.
    If a value in this list is None, the dimension can be arbitrary. The shape list is set fo actual dimensions
    of the matrix.
    :param values: Type or tuple of  allowed types of elements of the matrix. E.g. ( int, float )
    :param idx: Internal. Used to pass actual index in the matrix for possible error messages.
    :return:
    '''
    try:
        if len(shape) == 0:
            if not isinstance(mat, values):
                raise ParamError("Element at index {} of type {}, expected instance of {}.".format(idx, type(mat), values))
        else:
            if shape[0] is None:
                shape[0] = len(mat)
            l=None
            if not hasattr(mat, '__len__'):
                l=0
            elif len(mat) != shape[0]:
                l=len(mat)
            if not l is None:
                raise ParamError("Wrong len {} of element {}, should be  {}.".format(l, idx, shape[0]))
            for i, item in enumerate(mat):
                sub_shape = shape[1:]
                check_matrix(item, sub_shape, values, idx = [i] + idx)
                shape[1:] = sub_shape
        return shape
    except ParamError:
        raise
    except Exception as e:
        raise ParamError(e)


def check_knots(deg, knots, N):
    total_multiplicity = 0
    for knot, mult in knots:
        # This condition must hold if we assume only (0,1) interval of curve or surface parameters.
        #assert float(knot) >= 0.0 and float(knot) <= 1.0
        total_multiplicity += mult
    assert total_multiplicity == deg + N + 1


scalar_types = (int, float, np.int64)














class SplineBasis:
    """
    Represents a spline basis for a given knot vector and degree.
    Provides canonical evaluation for the bases functions and their derivatives, knot vector lookup etc.
    """

    @classmethod
    def make_equidistant(cls, degree, n_intervals, knot_range=[0.0, 1.0]):
        """
        Returns spline basis for an eqidistant knot vector
        having 'n_intervals' subintervals.
        :param degree: degree of the spline basis
        :param n_intervals: length of vector
        :param knot_range: support of the spline, min and max valid 't'
        :return: np array of knots
        """
        n = n_intervals + 2 * degree + 1
        knots = np.array((knot_range[0],) * n)
        diff = (knot_range[1] - knot_range[0]) / n_intervals
        for i in range(degree + 1, n - degree):
            knots[i] = (i - degree) * diff + knot_range[0]
        knots[-degree - 1:] = knot_range[1]
        return cls(degree, knots)

    @classmethod
    def make_from_packed_knots(cls, degree, knots):
        full_knots = [ q for q, mult in knots for i in range(mult)  ]
        return cls(degree, full_knots)


    def __init__(self, degree, knots):
        """
        Constructor of the basis.
        :param degree: Degree of Bezier polynomials >=0.
        :param knots: Numpy array of the knots including multiplicities.
        """
        assert degree >=0
        self.degree = degree

        # check free ends (and  full degree along the whole curve)
        for i in range(self.degree):
            assert knots[i] == knots[i+1]
            assert knots[-i-1] == knots[-i-2]
        self.knots = knots

        self.size = len(self.knots) - self.degree -1
        self.knots_idx_range = [self.degree, len(self.knots) - self.degree - 1]
        self.domain = self.knots[self.knots_idx_range]
        self.domain_size = self.domain[1] - self.domain[0]
        # Number of basis functions.

    def pack_knots(self):
        last, mult = self.knots[0], 0
        packed_knots = []
        for q in self.knots:
            if q == last:
                mult+=1
            else:
                packed_knots.append( (last, mult) )
                last, mult = q, 1
        return packed_knots

    def find_knot_interval(self, t):
        """
        Find the first non-empty knot interval containing the value 't'.
        i.e. knots[i] <= t <= knots[i+1], where  knots[i] < knots[i+1]
        Returns I = i  - degree, which is the index of the first basis function
        nonzero in 't'.

        :param t:  float, must be within knots limits.
        :return: I
        """
        assert self.knots[0] <= t <= self.knots[-1]

        """
        This function try to find index for given t_param in knot_vec that
        is covered by all (3) base functions.
        :param self.knots:
        :param t:
        :return:
        """

        # get range without multiplicities
        mn = self.knots_idx_range[0]
        mx = self.knots_idx_range[1]
        diff = mx - mn

        while diff > 1:
            # estimate for equidistant knots
            t_01 = (t - self.knots[mn]) / self.domain_size
            est = int(t_01 * diff + mn)
            if t > self.knots[est]:
                if mn == est :
                    break
                mn = est
            else:
                if mx == est:
                    mn = mx
                    break
                mx = est
            diff = mx - mn
        return min(mn, self.knots_idx_range[1] - 1) - self.degree

    def _basis(self, deg, idx, t):
        """
        Recursive evaluation of basis function of given degree and index.

        :param deg: Degree of the basis function
        :param idx: Index of the basis function to evaluate.
        :param t: Point of evaluation.
        :return Value of the basis function.
        """

        if deg == 0:
            t_0 = self.knots[idx]
            t_1 = self.knots[idx + 1]
            value = 1.0 if t_0 <= t < t_1 else 0.0
            return value
        else:
            t_i = self.knots[idx]
            t_ik = self.knots[idx + deg]
            top = t - t_i
            bottom = t_ik - t_i
            if bottom != 0:
                value = top / bottom * self._basis(deg-1, idx, t)
            else:
                value = 0.0

            t_ik1 = self.knots[idx + deg + 1]
            t_i1 = self.knots[idx + 1]
            top = t_ik1 - t
            bottom = t_ik1 - t_i1
            if bottom != 0:
                value += top / bottom * self._basis(deg-1, idx+1, t)

            return value

    def fn_supp(self, i_base):
        """
        Support of the base function 'i_base'.
        :param i_base:
        :return: (t_min, t_max)
        """
        return (self.knots[i_base], self.knots[i_base + self.degree + 1])

    def eval(self, i_base, t):
        """
        :param i_base: Index of base function to evaluate.
        :param t: point in which evaluate
        :return: b_i(y)
        """
        assert 0 <= i_base < self.size
        if i_base == self.size -1 and t == self.domain[1]:
            return 1.0
        return self._basis(self.degree, i_base, t)



class Curve:

    @classmethod
    def make_raw(cls, poles, knots, rational=False, degree=2):
        """
        Construct a B-spline curve.
        :param poles: List of poles (control points) ( X, Y, Z ) or weighted points (X,Y,Z, w). X,Y,Z,w are floats.
                   Weighted points are used only for rational B-splines (i.e. nurbs)
        :param knots: List of tuples (knot, multiplicity), where knot is float, t-parameter on the curve of the knot
                   and multiplicity is positive int. Total number of knots, i.e. sum of their multiplicities, must be
                   degree + N + 1, where N is number of poles.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles.
        :param degree: Non-negative int
        """
        basis = SplineBasis(degree, knots)
        return cls(basis, poles, rational)

    """
    Defines a 3D (or (dim -D) curve as B-spline. We shall work only with B-splines of degree 2.
    Corresponds to "B-spline Curve - <3D curve record 7>" from BREP format description.
    """
    def __init__(self, basis, poles, rational = False):
        self.basis = basis
        dim = len(poles[0]) - rational
        check_matrix(poles, [self.basis.size, dim + rational], scalar_types )

        self.poles=np.array(poles)  # N x D
        self.rational=rational
        if rational:
            self._weights = poles[:, dim]
            self._poles = (poles[:, 0:dim].T * self._weights ).T


    def eval(self, t):

        it = self.basis.find_knot_interval(t)
        dt = self.basis.degree + 1
        t_base_vec = np.array([self.basis.eval(jt, t) for jt in range(it, it + dt)])

        if self.rational:
            top_value = np.inner(t_base_vec, self._poles[it: it + dt, :].T)
            bot_value = np.inner(t_base_vec, self._weights[it: it + dt])
            return top_value / bot_value
        else:
            return  np.inner(t_base_vec, self.poles[it: it + dt, :].T)





class Surface:
    """
    Defines a B-spline surface.
    """

    @classmethod
    def make_raw(cls, poles, knots, rational=False, degree=(2,2)):
        """
        Construct a B-spline curve.
        :param poles: List of poles (control points) ( X, Y, Z ) or weighted points (X,Y,Z, w). X,Y,Z,w are floats.
                   Weighted points are used only for rational B-splines (i.e. nurbs)
        :param knots: List of tuples (knot, multiplicity), where knot is float, t-parameter on the curve of the knot
                   and multiplicity is positive int. Total number of knots, i.e. sum of their multiplicities, must be
                   degree + N + 1, where N is number of poles.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles.
        :param degree: Non-negative int
        """
        u_basis = SplineBasis(degree[0], knots[0])
        v_basis = SplineBasis(degree[0], knots[0])
        return cls( (u_basis, v_basis), poles, rational)


    def __init__(self, basis, poles, rational=False):
        """
        Construct a B-spline in 3d space.
        :param poles: Matrix (list of lists) of Nu times Nv poles (control points).
                      Single pole is a points ( X, Y, Z ) or weighted point (X,Y,Z, w). X,Y,Z,w are floats.
                      Weighted points are used only for rational B-splines (i.e. nurbs)
        :param knots: Tuple (u_knots, v_knots). Both u_knots and v_knots are lists of tuples
                      (knot, multiplicity), where knot is float, t-parameter on the curve of the knot
                      and multiplicity is positive int. For both U and V knot vector the total number of knots,
                      i.e. sum of their multiplicities, must be degree + N + 1, where N is number of poles.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles. BREP format have two independent flags
                      for U and V parametr, but only choices 0,0 and 1,1 have sense.
        :param degree: (u_degree, v_degree) Both positive ints.
        """

        self.u_basis, self.v_basis = basis
        self.rational = rational
        dim = len(poles[0][0]) - rational
        check_matrix(poles, [self.u_basis.size, self.v_basis.size, dim + rational], scalar_types )
        self.poles=np.array(poles)
        assert self.poles.shape == (self.u_basis.size, self.v_basis.size, dim + rational)
        if rational:
            self._weights = poles[:, :, dim]
            self._poles = (poles[:,:,0:dim].T * self._weights.T ).T

    def eval(self, u, v):
        iu = self.u_basis.find_knot_interval(u)
        iv = self.v_basis.find_knot_interval(v)
        du = self.u_basis.degree + 1
        dv = self.v_basis.degree + 1
        u_base_vec = np.array([self.u_basis.eval(ju, u) for ju in range(iu, iu + du)])
        v_base_vec = np.array([self.v_basis.eval(jv, v) for jv in range(iv, iv + dv)])

        if self.rational:
            top_value = np.inner(u_base_vec, self._poles[iu: iu + du, iv: iv + dv, :].T)
            top_value = np.inner(top_value, v_base_vec)
            bot_value = np.inner(u_base_vec, self._weights[iu: iu + du, iv: iv + dv].T)
            bot_value = np.inner(bot_value, v_base_vec)
            return top_value / bot_value
        else:
            #print("u: {} v: {} p: {}".format(u_base_vec.shape, v_base_vec.shape, self.poles[iu: iu + du, iv: iv + dv, :].shape))
            # inner product sums along the last axis of its parameters
            value = np.inner(u_base_vec, self.poles[iu: iu + du, iv: iv + dv, :].T )
            #print("val: {}".format(value.shape))
            return np.inner( value, v_base_vec)




class Z_Surface:
    """
    Simplified B-spline surface that use just linear or bilinear transform between XY  and UV.

    TODO:
    - We need conversion to full 3D surface for the BREP output
    - Optimization: simplified Bspline evaluation just for the singel coordinate
    """

    def eval_xy(self, xy_points):
        pass


























# Cache used of knot vector (computation of differences)
KVN_CACHE = {}
SB_CACHE = {}



def spline_base_vec(knot_vec, t_param, order, sparse=False):
    """
    This function compute normalized blending function aka base function of B-Spline curve or surface.
    :param knot_vec:
    :param t_param:
    :param order: (0: function value, 1: derivative function value)
    :param sparse:
    :return:
    """


    idx = find_index(knot_vec, t_param)
    n_basf = len(knot_vec) - 3

    # Create sparse matrix
    if sparse is True:
        basis_values = np.zeros(3)
    else:
        basis_values = np.zeros(n_basf)

    tk1 = knot_vec[idx + 1]
    tk2 = knot_vec[idx + 2]
    tk3 = knot_vec[idx + 3]
    tk4 = knot_vec[idx + 4]

    d31 = tk3 - tk1
    d32 = tk3 - tk2
    d42 = tk4 - tk2

    dt1 = t_param - tk1
    dt2 = t_param - tk2
    d3t = tk3 - t_param
    d4t = tk4 - t_param

    d31_d32 = d31 * d32
    d42_d32 = d42 * d32

    # basis function values
    if order == 0:
        if sparse is True:
            basis_values[0] = (d3t * d3t) / d31_d32
            basis_values[1] = ((dt1 * d3t) / d31_d32) + ((dt2 * d4t) / d42_d32)
            basis_values[2] = (dt2 * dt2) / d42_d32

        else:
            basis_values[idx] = (d3t * d3t) / d31_d32
            basis_values[idx + 1] = ((dt1 * d3t) / d31_d32) + ((dt2 * d4t) / d42_d32)
            basis_values[idx + 2] = (dt2 * dt2) / d42_d32

    # basis function derivatives
    elif order == 1:
        if sparse is True:
            basis_values[0] = -2 * d3t / d31_d32
            basis_values[1] = (d3t - dt1) / d31_d32 + (d4t - dt2) / d42_d32
            basis_values[2] = 2 * dt2 / d42_d32
        else:
            basis_values[idx] = -2*d3t / d31_d32
            basis_values[idx + 1] = (d3t - dt1) / d31_d32 + (d4t - dt2) / d42_d32
            basis_values[idx + 2] = 2 * dt2 / d42_d32

    return basis_values, idx


def test_spline_base_vec(knots=np.array((0.0, 0.0, 0.0, 1/3.0, 2/3.0, 1.0, 1.0, 1.0)), sparse=False):
    """
    Test optimized version of spline base function
    :param knots: numpy array of knots
    :param sparse: is sparse matrix used
    :return:
    """

    num = 100
    n_basf = len(knots) - 3
    y_coords = {}
    for k in range(0, n_basf):
        temp = {}
        for i in range(0, num+1):
            t_param = min(knots) + max(knots) * i / float(num)
            if sparse is True:
                temp[i] = spline_base_vec(knots, t_param, 0, sparse)[0].toarray()[0]
            else:
                temp[i] = spline_base_vec(knots, t_param, 0, sparse)[0]
        y_coords[k] = temp

    diff_x = (max(knots) - min(knots)) / num
    x_coord = [min(knots) + diff_x*i for i in range(0, num+1)]

    for temp in y_coords.values():
        plt.plot(x_coord, temp.values())
    plt.show()




def spline_base(knot_vec, basis_fnc_idx, t_param):
    """
    Evaluate a second order bases function in 't_param'.
    This function compute normalized blending function aka base function of B-Spline curve or surface.
    This function implement some optimization.
    :param knot_vec: knot vector
    :param basis_fnc_idx: index of basis function
    :param t_param: parameter t in interval <0, 1>
    :return: value of basis function
    """

    # When basis function has zero value at given interval, then return 0
    if t_param < knot_vec[basis_fnc_idx] or knot_vec[basis_fnc_idx+3] < t_param:
        return 0.0

    try:
        value = SB_CACHE[(tuple(knot_vec), basis_fnc_idx, t_param)]
    except KeyError:
        try:
            kvn = KVN_CACHE[tuple(knot_vec)]
        except KeyError:
            knot_vec_len = len(knot_vec)
            kvn = [0] * knot_vec_len
            i = 0
            while i < knot_vec_len - 1:
                if knot_vec[i] - knot_vec[i+1] != 0:
                    kvn[i] = 1.0
                i += 1
            KVN_CACHE[tuple(knot_vec)] = kvn
        tks = [knot_vec[basis_fnc_idx + k] for k in range(0, 4)]

        value = 0.0
        if knot_vec[basis_fnc_idx] <= t_param <= knot_vec[basis_fnc_idx+1] and kvn[basis_fnc_idx] != 0:
            value = (t_param - tks[0])**2 / ((tks[2] - tks[0]) * (tks[1] - tks[0]))
        elif knot_vec[basis_fnc_idx+1] <= t_param <= knot_vec[basis_fnc_idx+2] and kvn[basis_fnc_idx+1] != 0:
            value = ((t_param - tks[0]) * (tks[2] - t_param)) / ((tks[2] - tks[0]) * (tks[2] - tks[1])) + \
                   ((t_param - tks[1]) * (tks[3] - t_param)) / ((tks[3] - tks[1]) * (tks[2] - tks[1]))
        elif knot_vec[basis_fnc_idx+2] <= t_param <= knot_vec[basis_fnc_idx+3] and kvn[basis_fnc_idx+2] != 0:
            value = (tks[3] - t_param)**2 / ((tks[3] - tks[1]) * (tks[3] - tks[2]))
        SB_CACHE[(tuple(knot_vec), basis_fnc_idx, t_param)] = value

    return value


KNOT_VEC_CACHE = {}


def spline_surface(poles, u_param, v_param, u_knots, v_knots, u_mults, v_mults):
    """
    Compute coordinate of one point at B-Surface (u and v degree is 2)
    :param poles: matrix of "poles"
    :param u_param: X coordinate in range <0, 1>
    :param v_param: Y coordinate in range <0, 1>
    :param u_knots: list of u knots
    :param v_knots: list of v knots
    :param u_mults: list of u multiplicities
    :param v_mults: list of v multiplicities
    :return: tuple of (x, y, z) coordinate of B-Spline surface
    """

    # "Decompress" knot vectors using multiplicities
    # e.g
    # u_knots: (0.0, 0.5, 1.0) u_mults: (3, 1, 3) will be converted to
    # _u_knot: (0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0)
    _u_knots = []
    _v_knots = []
    try:
        _u_knots = KNOT_VEC_CACHE[(u_knots, u_mults)]
    except KeyError:
        for idx, mult in enumerate(u_mults):
            _u_knots.extend([u_knots[idx]] * mult)
        KNOT_VEC_CACHE[(u_knots, u_mults)] = _u_knots
    try:
        _v_knots = KNOT_VEC_CACHE[(v_knots, v_mults)]
    except KeyError:
        for idx, mult in enumerate(v_mults):
            _v_knots.extend([v_knots[idx]] * mult)
        KNOT_VEC_CACHE[(v_knots, v_mults)] = _v_knots

    u_n_basf = len(_u_knots) - 3
    v_n_basf = len(_v_knots) - 3

    uf_mat = [0.0] * u_n_basf
    vf_mat = [0.0] * v_n_basf

    # Pre-compute base values of functions
    for k in range(0, u_n_basf):
        uf_mat[k] = spline_base(_u_knots, k, u_param)
    for k in range(0, v_n_basf):
        vf_mat[k] = spline_base(_v_knots, k, v_param)

    x_coord, y_coord, z_coord = 0.0, 0.0, 0.0

    # Compute point at B-Spline surface
    for i in range(0, u_n_basf):
        for j in range(0, v_n_basf):
            base_i_j = uf_mat[i] * vf_mat[j]
            x_coord += poles[i][j][0] * base_i_j
            y_coord += poles[i][j][1] * base_i_j
            z_coord += poles[i][j][2] * base_i_j

    return x_coord, y_coord, z_coord





def transform_points(quad, terrain_data):
    """
    Function computes corresponding (u,v) for (x,y)
    :param terrain_data: matrix of 3D terrain data, N rows, 3 cols
    :param quad: points defining quadrangle area (array) 2 rows 4 cols, 
    :return: transformed and cropped terrain points
    """

    # if quad points are given counter clock wise, this is outer normal in XY plane
    mat_n = np.empty_like(quad)
    
    mat_n[:, 0] = quad[:, 0] - quad[:, 3]
    nt = mat_n[0, 0]
    mat_n[0, 0] = -mat_n[1, 0]
    mat_n[1, 0] = nt
    mat_n[:, 0] = mat_n[:, 0] / la.norm(mat_n[:, 0])

    for i in range(1, 4):
        mat_n[:, i] = quad[:, i] - quad[:, i-1]
        nt = mat_n[0, i]
        mat_n[0, i] = -mat_n[1, i]
        mat_n[1, i] = nt
        mat_n[:, i] = mat_n[:, i] / la.norm(mat_n[:, i])

    terrain_len = len(terrain_data)

    # Compute local coordinates and drop all points outside quadraangle
    param_terrain = np.empty_like(terrain_data)
    # indexing
    d0 = (terrain_data[:, 0:2] - quad[:, 0].transpose())
    d2 = (terrain_data[:, 0:2] - quad[:, 2].transpose())
    d3 = (terrain_data[:, 0:2] - quad[:, 3].transpose())
    d0_n0 = d0 * mat_n[:, 0]
    d0_n1 = d0 * mat_n[:, 1]
    d2_n2 = d2 * mat_n[:, 2]
    d3_n3 = d3 * mat_n[:, 3]
    u = np.divide(d0_n0, d0_n0 + d2_n2)
    v = np.divide(d0_n1, d0_n1 + d3_n3)

    h = -1
    for j in range(0,terrain_len):
        if (u[j] >= 0.0) and (u[j] <= 1.0) and (v[j] >= 0.0) and (v[j] <= 1.0):
            h += 1
            param_terrain[h, 0] = u[j]
            param_terrain[h, 1] = v[j]
            param_terrain[h, 2] = terrain_data[j, 2]

    param_terrain.resize(h+1, 3)

    uv = np.reshape(param_terrain[:, 0:2], 2*h+2).transpose()

    a = quad[:, 3] - quad[:, 2]
    b = quad[:, 0] - quad[:, 1]
    c = quad[:, 1] - quad[:, 2]
    d = quad[:, 0] - quad[:, 3]

    ldiag = np.zeros([2 * h + 1, 1])
    diag = np.zeros([2 * h + 2, 1])
    udiag = np.zeros([2 * h + 1, 1])

    # fixed point Newton iteration
    for i in range(0, 1):  # 1->5
        for j in range(0, h+1):
            mat_j = compute_jacobi(uv[2 * j, 0], uv[2 * j + 1, 0], a, b, c, d, -1)
            ldiag[2 * j, 0] = mat_j[1, 0]
            diag[2 * j, 0] = mat_j[0, 0]
            diag[2 * j + 1, 0] = mat_j[1, 1]
            udiag[2 * j, 0] = mat_j[0, 1]

        mat_jg = scipy.sparse.diags([ldiag[:, 0], diag[:, 0], udiag[:, 0]], [-1, 0, 1], format="csr")
        uv = uv - mat_jg.dot(uv)

    uv = uv.reshape([h + 1, 2])

    # Tresholding of the refined coordinates
    for j in range(0, h+1):
        if uv[j, 0] < 0:
            uv[:, 0] = 0
        elif uv[j, 0] > 1:
            uv[j, 0] = 1
        elif uv[j, 1] < 0:
            uv[:, 1] = 0
        elif uv[j, 1] > 1:
            uv[j, 1] = 1

    param_terrain[:, 0:2] = uv

    return param_terrain


