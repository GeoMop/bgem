"""
Module with classes representing various B-spline and NURMS curves and surfaces.
These classes provide just basic functionality:
- storing the data
- evaluation of XYZ for UV
In future:
- evaluation and xy<->uv functions accepting np.arrays,
- serialization and deserialization using JSONdata - must make it an installable module
- use de Boor algorithm for evaluation of curves and surfaces
- evaluation of derivatives
- implement degree increasing and knot insertion
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

    def make_linear_poles(self):
        """
        Return poles of basis functions to get a f(x) = x.
        :return:
        """
        poles= [ 0.0 ]
        for i in range(self.size-1):
            pp = poles[-1] + (self.knots[i + self.degree + 1] - self.knots[i  + 1]) / float(self.degree)
            poles.append(pp)
        return poles



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
        self.dim = len(poles[0]) - rational
        check_matrix(poles, [self.basis.size, self.dim + rational], scalar_types )

        self.poles=np.array(poles)  # N x D
        self.rational=rational
        if rational:
            self._weights = poles[:, self.dim]
            self._poles = (poles[:, 0:self.dim].T * self._weights ).T


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
        self.dim = len(poles[0][0]) - rational
        check_matrix(poles, [self.u_basis.size, self.v_basis.size, self.dim + rational], scalar_types )
        self.poles=np.array(poles)
        assert self.poles.shape == (self.u_basis.size, self.v_basis.size, self.dim + rational)
        if rational:
            self._weights = poles[:, :, self.dim]
            self._poles = (poles[:,:,0:self.dim].T * self._weights.T ).T

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
    def __init__(self, xy_quad, z_surface):
        """
        Construct a surface given by the  1d surface for the Z coordinate and XY quadrilateral
        for the bilinear UV -> XY mapping.
        :param xy_quad: np array N x 2
            Four or three points, determining bilinear or linear mapping, respectively.
            Four points giving XY coordinates for the uv corners: (0,0), (0,1), (1,0), (1,1)
            Three points giving XY coordinates for the uv corners: (0,0), (0,1), (1,0)
        :param z_surface: !D Surface object.
        """
        assert z_surface.dim == 1
        self.z_surface = z_surface
        self.u_basis = z_surface.u_basis
        self.v_basis = z_surface.v_basis
        self.dim = 3

        if len(xy_quad) == 3:
            # linear case
            self.xy_shift = xy_quad[0]
            v_vec = xy_quad[1] - xy_quad[0]
            u_vec = xy_quad[2] - xy_quad[0]
            self.mat_uv_to_xy = np.column_stack((u_vec, v_vec))
            self.mat_xy_to_uv = la.inv(self.mat_uv_to_xy)

            self.xy_to_uv = self._linear_xy_to_uv
            self.uv_to_xy = self._linear_uv_to_xy

        elif len(xy_quad) == 4:
            # bilinear case
            self.quad = xy_quad

            self.xy_to_uv = self._bilinear_xy_to_uv
            self.uv_to_xy = self._bilinear_uv_to_xy

        else:
            assert False, "Three or four points must be given."

    def make_full_surface(self):
        """
        Return representation of the surface by the 3d Surface object.
        Compute redundant XY poles.
        :return: Surface.
        """
        basis = (self.z_surface.u_basis, self.z_surface.v_basis)

        u = basis[0].make_linear_poles()
        v = basis[1].make_linear_poles()
        V, U = np.meshgrid(v,u)
        uv_poles = np.stack([U, V], axis=2)
        xy_poles = np.apply_along_axis(lambda x: self.uv_to_xy(x[0], x[1]), 2, uv_poles)
        poles = np.concatenate( (xy_poles, self.z_surface.poles), axis = 2 )

        return Surface(basis, poles)

    def _linear_uv_to_xy(self, u, v):
        #assert uv_points.shape[0] == 2, "Size: {}".format(uv_points.shape)
        uv_points = np.array([u, v])
        return self.mat_uv_to_xy.dot(uv_points) + self.xy_shift


    def _bilinear_uv_to_xy(self, u, v):
        weights = np.array([ (1-u)*(1-v), (1-u)*v, u*(1-v), u*v ])
        return self.quad.T.dot( weights )

    def _linear_xy_to_uv(self, x, y):
        # assert xy_points.shape[0] == 2
        xy_points = np.array([x,y])
        return self.mat_xy_to_uv.dot((xy_points - self.xy_shift))


    def _bilinear_xy_to_uv(self, x, y):
        assert False, "Not implemented yet."


    def eval(self, u, v):
        z = self.z_surface.eval(u, v)
        x, y = self.uv_to_xy(u, v)
        return np.array( [x, y, z] )

    def eval_xy(self, x, y):
        u, v  = self.xy_to_uv(x, y)
        z = self.z_surface.eval(u, v)
        return np.array([x, y, z])



















