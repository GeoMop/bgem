"""
Collection of functions to produce Bapline curves and surfaces as approximation of various analytical curves and surfaces.
"""

import bspline as bs
import numpy as np
import math
import time
#import numpy.linalg
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate

            
"""
Approximation methods for B/splines of degree 2.

"""
def plane_surface(vtxs, overhang=0.0):
    """
    Returns B-spline surface of a plane given by 3 points.
    We retun also list of UV coordinates of the given points.
    U direction v0 -> v1
    V direction v0 -> v2
    :param vtxs: List of tuples (X,Y,Z)
    :return: ( Surface, vtxs_uv )
    """
    assert len(vtxs) == 3, "n vtx: {}".format(len(vtxs))
    vtxs = np.array(vtxs)
    vv = vtxs[1] + vtxs[2] - vtxs[0]
    vtx4 = [ vtxs[0], vtxs[1], vv, vtxs[2]]
    return bilinear_surface(vtx4, overhang)



def bilinear_surface(vtxs, overhang=0.0):
    """
    Returns B-spline surface of a bilinear surface given by 4 corner points:
    uv coords:
    We retun also list of UV coordinates of the given points.
    :param vtxs: List of tuples (X,Y,Z)
    :return: ( Surface, vtxs_uv )
    """
    assert len(vtxs) == 4, "n vtx: {}".format(len(vtxs))
    vtxs = np.array(vtxs)
    if overhang > 0.0:
        dv = np.roll(vtxs, -1, axis=0) - vtxs
        dv *= overhang
        vtxs +=  np.roll(dv, 1, axis=0) - dv

    def mid(*idx):
        return np.mean( vtxs[list(idx)], axis=0)

    # v - direction v0 -> v2
    # u - direction v0 -> v1
    poles = [ [vtxs[0],  mid(0, 3), vtxs[3]],
                [mid(0,1), mid(0,1,2,3), mid(2,3)],
                [vtxs[1], mid(1,2), vtxs[2]]
                ]
    knots = 3 * [0.0 - overhang] + 3 * [1.0 + overhang]
    basis = bs.SplineBasis(2, knots)
    surface = bs.Surface((basis, basis), poles)
    #vtxs_uv = [ (0, 0), (1, 0), (1, 1), (0, 1) ]
    return surface




def line(vtxs, overhang = 0.0):
    '''
    Return B-spline approximation of a line from two points
    :param vtxs: [ X0, X1 ], Xn are point coordinates in arbitrary dimension D
    :return: Curve2D
    '''
    assert len(vtxs) == 2
    vtxs = np.array(vtxs)
    if overhang > 0.0:
        dv = overhang*(vtxs[1] - vtxs[0])
        vtxs[0] -= dv
        vtxs[1] += dv
    mid = np.mean(vtxs, axis=0)
    poles = [ vtxs[0],  mid, vtxs[1] ]
    knots = 3*[0.0 - overhang] + 3*[1.0 + overhang]
    basis = bs.SplineBasis(2, knots)
    return bs.Curve(basis, poles)




def surface_from_grid(grid_surface, nuv, **kwargs):
    """
    Make a Z_Surface of degree 2 as an approximation of the GridSurface.
    :param grid_surface: grid surface to approximate
    :param (nu, nv) Prescribed number of poles in u and v directions.
    :return: Z_surface object.
    """
    approx = _SurfaceApprox(grid_surface, nuv, **kwargs)
    return  approx.get_approximation()


def curve_from_grid(points, **kwargs):
    """
    Make a Curve (of degree 3) as an approximation of a sequence of points.
    :param points - N x D array, D is dimension
    :param nt Prescribed number of poles of the resulting spline.
    :return: Curve object.

    TODO:
    - Measure efficiency. Estimate how good we can be. Do it our self if we can do at leas 10 times better.
    - Find out which method is used. Hoschek (4.4.1) refers to several methods how to determine parametrization of
    the curve, i.e. Find parameters t_i to the given approximation points P_i.
    - Further on it is not clear what is the mening of the 's' parameter and how one cna influence tolerance and smoothness.
    - Some sort of adaptivity is used.


    """
    deg = kwargs.get('degree', 3)
    tol = kwargs.get('tol', 0.01)
    weights = np.ones(points.shape[0])
    weights[0] = weights[-1] = 1000.0
    tck = scipy.interpolate.splprep(points.T, k=deg, s=tol, w = weights)[0]
    knots, poles, degree  = tck
    curve_poles=np.array(poles).T
    curve_poles[0] = points[0]
    curve_poles[-1] = points[-1]
    basis = bs.SplineBasis(degree, knots)
    curve = bs.Curve(basis, curve_poles)
    return curve






class _SurfaceApprox:
    """
    TODO:
    - Check efficiency of scipy methods, compare it to our approach assuming theoretical number of operations.
    - Optimize construction of A regularization matrix.
    - Compute BtB directly during single assembly pass, local 9x9 matricies as in A matrix.
    - In contradiction to some literature (Hoschek) solution of the LS system is fast as long as the basis is local (
      this is true for B-splines).
    - Extensions to fitting X and Y as well - general Surface

    """



    def __init__(self, grid_surface, nuv=None, **kwargs):
        self.degree = np.array((2, 2))
        self.grid_surf = grid_surface
        self.nuv = nuv
        self.regularization_weight = kwargs.get('reg_weight', 0.001)

    def get_approximation(self):
        if not hasattr(self, 'z_surf'):
            self.z_surf = self.approx_chol()
        return self.z_surf


    def approx_chol(self):
        """
        This function tries to approximate terrain data with B-Spline surface patches
        using Cholesky decomposition
        :param terrain_data: matrix of 3D terrain data
        :param quad: points defining quadrangle area (array)
        :param u_knots: array of u knots
        :param v_knots: array of v knots
        :param sparse: if sparse matrix is used
        :param filter_thresh: threshold of filter
        :return: B-Spline patch
        """


        print('Transforming points to parametric space ...')
        start_time = time.time()

        points_uvz = self.grid_surf.grid_uvz.reshape(-1, 3)
        points_uv = points_uvz[:, 0:2]

        # remove points far from unit square
        eps = 1.0e-15
        cut_min = np.array( [ -eps, -eps ])
        cut_max = np.array( [ 1+eps, 1+eps ])
        in_idx = np.all(np.logical_and(cut_min < points_uv,  points_uv <= cut_max), axis=1)
        points_uv = points_uv[in_idx]
        self.points_z = points_uvz[in_idx, 2][:,None]
        print("Number of points out of the grid domain: {}".format(len(points_uv) - np.sum(in_idx)))

        # snap to unit square
        points_uv = np.maximum(points_uv, np.array([0.0, 0.0]))
        self.points_uv = np.minimum(points_uv, np.array([1.0, 1.0]))

        # determine number of knots
        assert len(self.points_uv) == len(self.points_z)
        n_points = len(self.points_uv)

        grid_shape = np.asarray(self.grid_surf.shape, dtype=int)
        if self.nuv == None:
            nuv = np.floor_divide( grid_shape / 3 )
        else:
            nuv = np.floor(np.array(self.nuv))
        nuv = np.minimum( nuv, grid_shape - self.degree - 2)
        size_u, size_v  = nuv + self.degree

        # try to make number of unknowns less then number of remaining points
        # +1 to improve determination
        if (size_u + 1) * (size_v + 1) > n_points:
            sv = np.floor(np.sqrt( n_points * size_v / size_u ))
            su = np.floor(sv * size_u / size_v)
            nuv = np.array( [su, sv] ) - self.degree
        nuv = nuv.astype(int)
        if nuv[0] < 1 or nuv[1] < 1:
            raise Exception("Two few points, {}, to make approximation, degree: {}".format(n_points, self.degree))

        print("Using {} x {} B-spline approximation.".format(nuv[0], nuv[1]))
        self.u_basis = bs.SplineBasis.make_equidistant(2, nuv[0])
        self.v_basis = bs.SplineBasis.make_equidistant(2, nuv[1])

        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))


        # Own computation of approximation
        print('Creating B matrix ...')
        start_time = time.time()
        b_mat, interval = self.build_ls_matrix()
        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))

        print('Creating A matrix ...')
        start_time = time.time()
        a_mat = self.build_sparse_reg_matrix()
        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))

        print('Computing B^T B matrix ...')
        start_time = time.time()
        bb_mat = b_mat.transpose().dot(b_mat)

        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))


        print('Computing A and B svds approximation ...')
        start_time = time.time()

        bb_norm = scipy.sparse.linalg.svds(bb_mat, k=1, ncv=10, tol=1e-4, which='LM', v0=None,
                                           maxiter=300, return_singular_vectors=False)
        a_norm = scipy.sparse.linalg.svds(a_mat, k=1, ncv=10, tol=1e-4, which='LM', v0=None,
                                          maxiter=300, return_singular_vectors=False)
        c_mat = bb_mat + self.regularization_weight * (bb_norm[0] / a_norm[0]) * a_mat

        g_vec = self.points_z[:, 0]
        b_vec = b_mat.transpose().dot( g_vec )
        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))


        print('Computing Z coordinates ...')
        start_time = time.time()
        z_vec = scipy.sparse.linalg.spsolve(c_mat, b_vec)
        assert not np.isnan(np.sum(z_vec)), "Singular matrix for approximation."

        print(type(z_vec))
        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))


        print('Computing differences ...')
        start_time = time.time()
        diff = b_mat.dot(z_vec) - g_vec
        max_diff = np.max(diff)
        print("Approximation error (max norm): {}".format(max_diff) )
        end_time = time.time()
        print('Computed in {0} seconds.'.format(end_time - start_time))

        # Construct Z-Surface
        poles_z = z_vec.reshape(self.v_basis.size, self.u_basis.size).T
        poles_z *= self.grid_surf.z_scale
        poles_z += self.grid_surf.z_shift
        surface_z = bs.Surface((self.u_basis, self.v_basis), poles_z[:,:,None])
        z_surf = bs.Z_Surface(self.grid_surf.quad[0:3], surface_z)

        return z_surf



    def build_ls_matrix(self):
        """
        Construction of the matrix (B) of the system of linear algebraic
        equations for control points of the 2th order B-spline surface
        :param u_knots:
        :param v_knots:
        :param terrain:
        :param sparse:
        :return:
        """
        u_n_basf = self.u_basis.size
        v_n_basf = self.v_basis.size
        n_points = self.points_uv.shape[0]

        n_uv_loc_nz =  (self.u_basis.degree +  1) * (self.v_basis.degree +  1)
        row = np.zeros(n_points * n_uv_loc_nz)
        col = np.zeros(n_points * n_uv_loc_nz)
        data = np.zeros(n_points * n_uv_loc_nz)

        nnz_b = 0

        interval = np.empty((n_points, 2))

        for idx in range(0, n_points):
            u, v = self.points_uv[idx, 0:2]
            iu = self.u_basis.find_knot_interval(u)
            iv = self.v_basis.find_knot_interval(v)
            u_base_vec = self.u_basis.eval_base_vector(iu, u)
            v_base_vec = self.v_basis.eval_base_vector(iv, v)
            # Hard-coded Kronecker product (problem based)
            for n in range(0, 3):
                data[nnz_b + 3 * n:nnz_b + 3 * (n + 1)] = v_base_vec[n] * u_base_vec
                for m in range(0, 3):
                    col_item = (iv + n) * u_n_basf + iu + m
                    col[nnz_b + (3 * n) + m] = col_item
            row[nnz_b:nnz_b + 9] = idx
            nnz_b += 9

            interval[idx][0] = iu
            interval[idx][1] = iv

        mat_b = scipy.sparse.csr_matrix((data, (row, col)), shape=(n_points, u_n_basf * v_n_basf))

        return mat_b, interval

    def _basis_in_q_points(self, basis):
        n_int = basis.n_intervals
        nq_points = len(self._q_points)
        q_point = np.zeros((n_int * nq_points, 1))
        point_val_outer = np.zeros((3, 3, n_int)) # "3" considers degree 2
        d_point_val_outer = np.zeros((3, 3, n_int)) # "3" considers degree 2

        #TODO: use numpy functions for quadrature points
        n = 0
        for i in range(n_int):
            us = basis.knots[i + 2]
            uil = basis.knots[i + 3] - basis.knots[i + 2]
            for j in range(nq_points):
                up = us + uil * self._q_points[j]
                q_point[n] = up
                u_base_vec = basis.eval_base_vector(i, up)
                u_base_vec_diff = basis.eval_diff_base_vector(i, up)
                point_val_outer[:, :, i] += self._weights[j] * np.outer(u_base_vec,u_base_vec)
                d_point_val_outer[:, :, i] += self._weights[j] * np.outer(u_base_vec_diff,u_base_vec_diff)
                n += 1

        return  point_val_outer, d_point_val_outer,q_point


    def build_sparse_reg_matrix(self):
        """
        Construction of the regularization matrix (A) to decrease variation of the terrain
        B z = b  ---> (B^T B + A)z = B^T b
        :param u_knots: vector of v-knots
        :param v_knots: vector of u-knots
        :param quad: points defining quadrangle area (array)
        :return: matrix

        -
        """

        #a = quad[:, 3] - quad[:, 2]
        #b = quad[:, 0] - quad[:, 1]
        #c = quad[:, 1] - quad[:, 2]
        #d = quad[:, 0] - quad[:, 3]

        u_n_basf = self.u_basis.size
        v_n_basf = self.v_basis.size
        u_n_inter = self.u_basis.n_intervals
        v_n_inter = self.v_basis.n_intervals
        n_uv_loc_nz = (self.u_basis.degree + 1) * (self.v_basis.degree + 1)

        # TODO: use Gauss quadrature from scipy
        # in fact for general degrees we should use different quadrature for u and different for v
        self._q_points =  [0, (0.5 - 1 / math.sqrt(20)), (0.5 + 1 / math.sqrt(20)), 1]
        self._weights = [1.0 / 6, 5.0 / 6, 5.0 / 6, 1.0 / 6]
        nq_points = len(self._q_points)

        u_val_outer, u_diff_val_outer, q_u_point = self._basis_in_q_points(self.u_basis)
        v_val_outer, v_diff_val_outer, q_v_point = self._basis_in_q_points(self.v_basis)

        row_m = np.zeros((v_n_inter * u_n_inter * n_uv_loc_nz * n_uv_loc_nz))
        col_m = np.zeros((v_n_inter * u_n_inter * n_uv_loc_nz * n_uv_loc_nz))
        data_m = np.zeros((v_n_inter * u_n_inter * n_uv_loc_nz * n_uv_loc_nz))

        nnz_a = 0
        linsp = np.linspace(0,self.u_basis.degree,self.u_basis.degree+1)
        llinsp = np.tile(linsp,self.u_basis.degree+1)

        print("vnint: {} unint: {} nqp: {} prod: {}".format(v_n_inter, u_n_inter, nq_points, v_n_inter* u_n_inter* nq_points*nq_points))
        for i in range(v_n_inter):
            for l in range(u_n_inter):
                jac = 1.0 / u_n_inter / v_n_inter
                idx_range = n_uv_loc_nz * n_uv_loc_nz
                v_val_outer_loc = v_val_outer[:, :, i]
                dv_val_outer_loc = v_diff_val_outer[:, : , i]
                u_val_outer_loc = u_val_outer[:, :, i]
                du_val_outer_loc = u_diff_val_outer[:, : , i]
                data_m[nnz_a:nnz_a + idx_range] = jac * ( np.kron(v_val_outer_loc, du_val_outer_loc)
                            + np.kron(dv_val_outer_loc, u_val_outer_loc) ).ravel()
                colv = np.repeat((i + linsp) * u_n_basf,self.u_basis.degree+1) + llinsp
                col_m[nnz_a:nnz_a + idx_range] = np.repeat(colv,n_uv_loc_nz)
                row_m[nnz_a:nnz_a + idx_range] = np.tile(colv,n_uv_loc_nz)
                nnz_a += idx_range
        print("Assembled")
        mat_a = scipy.sparse.coo_matrix((data_m, (row_m, col_m)),
                                        shape=(u_n_basf * v_n_basf, u_n_basf * v_n_basf)).tocsr()
        return mat_a