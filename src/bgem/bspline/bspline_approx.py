"""
Collection of functions to produce Bspline curves and
surfaces as approximation of various analytical curves and surfaces.

TODO:
- store points only withing patches
- initial assignment to patches with use of sorted points by U and V respectively
- keep assignment to patch during refinement
- keep local matrix of patch and recalculate only after refinement
- precomputed base functions and derivatives of points and quad points on patches
- quad points evaluated only on knot intervals.. using tensor product structure

- regularization distorts approximation on the boundary, since the misfit is less strong there,
  possibly decrease regularization on the boundary
- some sort of boundary regularization is necessary if the number of points in boundary patches is small
  the boundary DOFS are not determined



"""

import logging
import numpy as np
import numpy.linalg as la
import scipy.sparse
import scipy.sparse.linalg
import scipy.interpolate
from bgem.bspline import bspline as bs
from bgem.bspline.surface_point_set import SurfacePointSet
from bgem import tools

"""
Approximation methods for B/splines of degree 2.

"""


def plane_surface(vtxs, overhang=0.0):
    """
    Returns B-spline surface of a plane given by 3 points.
    We return also list of UV coordinates of the given points.
    U direction v0 -> v1
    V direction v0 -> v2
    :param vtxs: List of tuples (X,Y,Z)
    :param overhang: relative factor to enlarge the resulting parallelogram surface on all sides
    :return: ( Surface, vtxs_uv )
    """
    assert len(vtxs) == 3, "n vtx: {}".format(len(vtxs))
    vtxs = np.array(vtxs)
    vv = vtxs[1] + vtxs[2] - vtxs[0]
    vtx4 = [vtxs[0], vtxs[1], vv, vtxs[2]]
    return bilinear_surface(vtx4, overhang)


def bilinear_surface(vtxs, overhang=0.0):
    """
    Returns B-spline surface of a bilinear surface given by 4 corner points:
    uv coords:
    We return also list of UV coordinates of the given points.
    :param vtxs: List of tuples (X,Y,Z)
    :param overhang: relative factor to enlarge the resulting parallelogram surface on all sides
    :return: ( Surface, vtxs_uv )
    """
    assert len(vtxs) == 4, "n vtx: {}".format(len(vtxs))
    vtxs = np.array(vtxs)
    if overhang > 0.0:
        dv = np.roll(vtxs, -1, axis=0) - vtxs
        dv *= overhang
        vtxs += np.roll(dv, 1, axis=0) - dv

    def mid(*idx):
        return np.mean(vtxs[list(idx)], axis=0)

    # v - direction v0 -> v2
    # u - direction v0 -> v1
    poles = [[vtxs[0], mid(0, 3), vtxs[3]],
             [mid(0, 1), mid(0, 1, 2, 3), mid(2, 3)],
             [vtxs[1], mid(1, 2), vtxs[2]]
             ]
    knots = 3 * [0.0 - overhang] + 3 * [1.0 + overhang]
    basis = bs.SplineBasis(2, knots)
    surface = bs.Surface((basis, basis), poles)
    # vtxs_uv = [ (0, 0), (1, 0), (1, 1), (0, 1) ]
    return surface


def line(vtxs, overhang=0.0):
    '''
    Return B-spline approximation of a line from two points
    :param vtxs: [ X0, X1 ], Xn are point coordinates in arbitrary dimension D
    :param overhang: relative factor to enlarge the resulting line on both sides
    :return: Curve2D
    '''
    assert len(vtxs) == 2
    vtxs = np.array(vtxs)
    if overhang > 0.0:
        dv = overhang * (vtxs[1] - vtxs[0])
        vtxs[0] -= dv
        vtxs[1] += dv
    mid = np.mean(vtxs, axis=0)
    poles = [vtxs[0], mid, vtxs[1]]
    knots = 3 * [0.0 - overhang] + 3 * [1.0 + overhang]
    basis = bs.SplineBasis(2, knots)
    return bs.Curve(basis, poles)


def surface_from_grid(grid_surface, nuv):
    """
    Make a Z_Surface of degree 2 as an approximation of the GridSurface.
    :param grid_surface: grid surface to approximate
    :param (nu, nv) Prescribed number of poles in u and v directions.
    :return: Z_surface object.
    """
    approx = SurfaceApprox.approx_from_grid_surface(grid_surface)
    return approx.compute_approximation(nuv=nuv)


def curve_from_grid(points, **kwargs):
    """
    Make a Curve (of degree 3) as an approximation of a sequence of points.
    :param points - N x D array, D is dimension
    :param nt Prescribed number of poles of the resulting spline.
    :return: Curve object.

    TODO:
    - Measure efficiency. Estimate how good we can be. Do it ourselves if we can do at least 10 times better.
    - Find out which method is used. Hoschek (4.4.1) refers to several methods how to determine parametrization of
    the curve, i.e. Find parameters t_i to the given approximation points P_i.
    - Apart from that it is not clear what is the meaning of the 's' parameter and how one can influence tolerance and smoothness.
    - Some sort of adaptivity is used.
    """
    deg = kwargs.get('degree', 3)
    tol = kwargs.get('tol', 0.00001)
    weights = np.ones(points.shape[0])
    weights[0] = weights[-1] = 1000.0
    tck = scipy.interpolate.splprep(points.T, k=deg, s=tol, w=weights)[0]
    knots, poles, degree = tck
    curve_poles = np.array(poles).T
    curve_poles[0] = points[0]
    curve_poles[-1] = points[-1]
    basis = bs.SplineBasis(degree, knots)
    curve = bs.Curve(basis, curve_poles)
    return curve


# @attrs.define()
# class BS_Patch:
#     loc_uv_points : np.array # UV coordinates on the patch
#     z_points : np.array


class SurfaceApprox:
    """
    Class to compute a Bspline surface approximation from given set of XYZ points.
    TODO:
    - Check efficiency of scipy methods, compare it to our approach assuming theoretical number of operations.
    - Compute BtB directly during single assembly pass, local 9x9 matrices as in A matrix.
    - In contradiction to some literature (Hoschek) solution of the LS system is fast as long as the basis is local (
      this is true for B-splines).
    - Extensions to fitting X and Y as well - general Surface

    """

    @classmethod
    def approx_from_file(cls, filename, file_delimiter=" ", file_skip_lines=0):
        sps = SurfacePointSet.from_file(filename, file_delimiter, file_skip_lines)
        return cls(sps)

    @classmethod
    def approx_from_grid_surface(cls, grid_surface):
        sps = SurfacePointSet.from_grid_surface(grid_surface)
        return cls(sps)

    def __init__(self, surface_points: SurfacePointSet) -> None:
        """
        Initialize the approximation object with the points.
        """

        # Degree of approximation in U anv V directions, currently fixed to 2.
        self._degree = np.array((2, 2))
        self._surface_points = surface_points

        # Approximation parameters.

        # Weight of the regularizing term.
        self.regul_coef = None

        # Temporaries
        self._u_basis = None
        self._v_basis = None
        self._patches = None

        # Approximation results

        # Approximating BSSurface
        self.surface = None

        # Error of the approximation
        self.error = None

    @property
    def nuv(self):
        return (self._u_basis.n_intervals, self._v_basis.n_intervals)

    @property
    def quad(self):
        return self._surface_points.quad

    @property
    def n_points(self):
        return self._surface_points.n_active

    @property
    def xy_points(self):
        return self._surface_points.xy_points

    @property
    def z_points(self):
        return self._surface_points.z_points

    @property
    def weights(self):
        return self._surface_points.weights

    @property
    def uv_points(self):
        return self._surface_points.uv_points

    def _refine_knots(self, basis, ref_vec):
        """
        Subdivide intervals of the knot vector that are marked by the 'ref_vec[i]>0'.
        TOOD: simplify
        """

        knot = basis.knots
        knotlist = []
        for i in range(0, len(knot) - 5):
            knotlist.append([])

        for i in range(0, len(knot) - 5):
            knotlist[i].append(knot[i + 2])
            if ref_vec[i] > 0:
                knotlist[i].append((knot[i + 2] + knot[i + 3]) / 2)

        refined_knot = []
        for i in range(0, len(knotlist)):
            for j in range(0, len(knotlist[i])):
                refined_knot.append(knotlist[i][j])

        n = len(knot) + sum(ref_vec > 0)
        ref_knot = np.zeros(n)
        # print(n)

        ref_knot[2:n - 3] = np.asarray(refined_knot)
        ref_knot[n - 3:n] = 1

        return bs.SplineBasis.make_from_knots(2, ref_knot)

    def compute_approximation(self, **kwargs):
        """
        Approximate the point set (given to the constructor) by a B-spline surface.
        The knot vectors in u and V direction can be adaptively refined until a prescribed tolerance is reached
        three kinds of the adaptivity is available.
        In order to prevent overfitting we regularize by penalizing the gradients of the constructed surface.
        The regularization parameter can be automatically tuned via several methods.

           to balance the approximation error |Z - surf(b)|
           and the regularization |grad surf(b)|_L2. Alternatively the cross-validation method can be applied.

        Approximation parameters can be passed in through kwargs or set in the object before the call.
        :param nuv: (nu, nv) Set number of intervals of the resulting B-spline, in U and V direction
        :param max_iters: determines number refinement steps of the knot vectors , default (20)
        :param solver:
            'spsolve' (default) use the sparse direct solver scipy.sparse.linalg.spsolve ,
            'cg' use conjugate gradient solver scipy.sparse.linalg.cg
        :param adapt_type: None, 'l2', 'linf'
            Adaptivity type to use. Denoting 'z(x,y)' the surface value and (x_i, y_i, z_i) given points:
            'absolute' (default) refine patches where |z(x_i, y_i) - z_i|_inf > max_diff
            'std_dev' If the total L2 error is greater then 'std_dev', refine 'max_part' fraction of the rows/columns with highest L2 error contribution.
        :param tolerance: target tolerance for the refinement
        :param refinement_ratio: fraction of the raws/columns to be refined (1.0 is maximum) for the 'std_dev' refinement
        :param validation: Small number 0-1 giving the fraction of the point set used for the 'cross-validate' regularization.
        :param regularization: 'None', ...
        :param regul_coef(0.001): initial value of the regularization parameter
        :return: B-Spline surface

        Two refinement algorithms:

        absolute norm based adaptivity
         maximum norm is evaluated on every patch, if it holds: patch maximum norm > max_diff (param)
         then both of the knot vectors ("u" AND "v") are refined in corresponding intervals
         finished: number of iteration achieved max_iters (param) OR maximum norm on every patch < max_diff (param)

        standard deviation based adaptivity
         Euclidean norms of the errors are computed with respect u,v knot intervals
         even iteration: max_part (param) ratio of the "u" knot intervals involving the largest norm are refined
         odd iteration: max_part (param) ratio of the "v" knot intervals involving the largest norm are refined
         finished: number of iteration achieved max_iters (param) OR standard deviation < std_dev (param)

        TODO:
        - common method for calculation the approximation
        - can reuse previous approximation ?? how
        - only process kwargs and set methods for: adaptivity (target criteria: None, L2, Linf),
          regularization: fixed/SVD, automatic using validation subset
        """

        nuv_ = kwargs.get("nuv", None)
        self.regul_coef = kwargs.get("regul_coef", 0.01)
        self.validation_fr = kwargs.get("validation", 0.05)
        self.solver = kwargs.get("solver", "spsolve")  # cg
        self.max_iters = kwargs.get("max_iters", 20)  #
        self.adapt_type = kwargs.get("adapt_type", "linf")  # "std_dev"
        if self.adapt_type is None:
            self.tolerance = None
            self.max_iters = 1
            self._mark_refinement = self._refine_patches_none
        elif self.adapt_type == "linf":
            self.tolerance = 10
            self._mark_refinement = self._refine_patches_linf
        elif self.adapt_type == "l2":
            self.tolerance = 1.0
            self._mark_refinement = self._refine_patches_l2
        else:
            assert False, "Wrong adaptivity type."
            # different tolerances for different adaptivity norms
        # TODO: try to rescale norms to have a same meaning of the tolerances
        self.tolerance = kwargs.get("tolerance", self.tolerance)  # for absolute based adaptivity
        self.refine_part = kwargs.get("refinement_ratio", 0.2)  # for standard deviation based adaptivity

        with tools.catch_time(f"Transforming points (n={self.n_points})") as time:
            if nuv_ is None:
                nuv_ = self._compute_default_nuv()

            if self.validation_fr != 1.0:
                n = len(self.uv_points)
                lsp = np.linspace(0, n - 1, n, dtype=int)
                red_lsp = np.random.choice(lsp, int(np.ceil(n * self.validation_fr)))
                compl_lsp = np.setxor1d(lsp, red_lsp)
                self._w_quad_points_compl = np.zeros(n)
                self._w_quad_points_compl[compl_lsp] = self.weights[compl_lsp]
                self._w_quad_points_compl_mask = np.ones(n)
                self._w_quad_points_compl_mask[red_lsp] = np.zeros(len(red_lsp))
                self._w_quad_points_mask = np.ones(n)
                self._w_quad_points_mask[compl_lsp] = np.zeros(len(compl_lsp))

                self.weights[compl_lsp] = np.zeros(len(compl_lsp))

            self._u_basis = bs.SplineBasis.make_equidistant(2, nuv_[0])
            self._v_basis = bs.SplineBasis.make_equidistant(2, nuv_[1])

        iters = 0
        while True:  # Adaptivity loop

            # Approximation itself
            with tools.catch_time("Creating explicitly system of normal equations B^TBz=B^Tb"):
                self._locate_points()
                btb_mat, btwb_vec, avg_vec = self._build_system_of_normal_equations()

            with tools.catch_time("Creating A matrix"):
                a_mat = self._build_sparse_reg_matrix()

            with tools.catch_time("Computing A and B^TB svds approximation") as time:
                # if iters == 0:
                bb_norm = scipy.sparse.linalg.eigsh(btb_mat, k=1, ncv=10, tol=1e-2, which='LM',
                                                    maxiter=300, return_eigenvectors=False)
                a_norm = scipy.sparse.linalg.eigsh(a_mat, k=1, ncv=10, tol=1e-2, which='LM',
                                                   maxiter=300, return_eigenvectors=False)
                reg_coef = self.regul_coef * bb_norm[0] / a_norm[0]

                logging.info(f"Reg coef: {reg_coef} = {self.regul_coef} * {bb_norm[0]} / {a_norm[0]}")
                c_mat = btb_mat + reg_coef * a_mat

            with tools.catch_time("Solving for Z coordinates") as time:
                z_vec = self._solve_system(c_mat, btwb_vec, avg_vec)
                # fig = plt.figure()
                # plt.spy(c_mat,markersize=1)
                # plt.show()
                assert not np.isnan(np.sum(z_vec)), "Singular matrix for approximation."

            with tools.catch_time("Computing error") as time:
                # diff, diff_mat_max, err_mat_eucl2, std_dev =
                err_vec = self._compute_errors(z_vec)
                z_l2_error = z_vec @ (btb_mat @ z_vec - btwb_vec)
                z_regul = np.sqrt(0.5 * reg_coef * z_vec @ (a_mat @ z_vec))

                logging.info(f"btb err: {z_l2_error}, regul: {z_regul}")
                # self.error = max_diff = np.max(diff)
                # logging.info("Approximation error (max norm): {}".format(max_diff))
                # logging.info("Standard deviation: {}".format(std_dev))

            # if self.validation_fr != 1.0:
            #     diff_compl = diff * self._w_quad_points_compl_mask # make sense only fow w_i in set(0,1)

            ref_vec_u, ref_vec_v, err_mat = self._mark_refinement(err_vec)
            # self.plot_error(err_mat, ref_vec_u, ref_vec_v)

            #
            # # Regularization coefficient
            # print("L2 diff: ", diff.dot(diff))
            # print("A2 diff: ", z_vec.dot(a_mat.dot(z_vec)))
            # #reg_coef = diff.dot(diff) / z_vec.dot(a_mat.dot(z_vec)) # shoud be replaced by a more stable formula
            #
            # print("reg_coef =", reg_coef)
            # print("iteration =",iters)
            # print("\nL2_diff =", std_dev)
            # print("\nmax_diff =",np.max(diff))
            # print("area =", self._u_basis.n_intervals,'x',self._v_basis.n_intervals, "(n_patches =",self._u_basis.n_intervals*self._v_basis.n_intervals,")")
            # if self.validation_fr != 1.0:
            #     logging.info("Efficient points ratio: {}".format(self.validation_fr))
            #     logging.info("Ratio of the errors (efficient/complete): {}".format(np.linalg.norm(diff_compl) / np.linalg.norm(diff)))

            # if (iters % 2) == 0:
            #     if np.sum(ref_vec_u) > 0:
            # else:
            #     if np.sum(ref_vec_v) > 0:

            # prevent over-refinement
            patch_sizes = np.array([len(l) for l in self.point_loc]).reshape(
                (self._u_basis.n_intervals, self._v_basis.n_intervals))
            u_patch_min = np.min(patch_sizes, axis=1)
            v_patch_min = np.min(patch_sizes, axis=0)
            # logging.info(f"U refinement {ref_vec_u}, min patch size: {u_patch_min}")
            # logging.info(f"V refinement {ref_vec_v}, min patch size: {u_patch_min}")
            ref_vec_u = np.logical_and(ref_vec_u, u_patch_min > 1)
            ref_vec_v = np.logical_and(ref_vec_v, v_patch_min > 1)
            # logging.info(f"U refinement {ref_vec_u}")
            # logging.info(f"V refinement {ref_vec_v}")

            n_marked = sum(ref_vec_u) + sum(ref_vec_v)
            # logging.info(f"N: {n_marked}, iters: {iters}")

            if n_marked == 0 or iters >= self.max_iters:
                break

            self._u_basis = self._refine_knots(self._u_basis, ref_vec_u)
            self._v_basis = self._refine_knots(self._v_basis, ref_vec_v)
            # self.error = max_diff = np.max(diff)
            # logging.info("Approximation error (max norm): {}".format(max_diff))
            # logging.info("Standard deviation: {}".format(std_dev))
            iters += 1

        # Construct Z-Surface
        poles_z = z_vec.reshape(self._v_basis.size, self._u_basis.size).T
        # poles_z *= self.grid_surf.z_scale
        # poles_z += self.grid_surf.z_shift
        surface_z = bs.Surface((self._u_basis, self._v_basis), poles_z[:, :, None])
        self.surface = bs.Z_Surface(self.quad[0:3], surface_z)

        return self.surface

    def _solve_system(self, c_mat, btwb_vec, avg_vec):

        if self.solver == 'spsolve':
            z_vec = scipy.sparse.linalg.spsolve(c_mat, btwb_vec, use_umfpack=True)
        elif self.solver == 'cg':
            # Hegedus trick (for initial condition)
            Ax = c_mat.dot(avg_vec)
            bAx = btwb_vec.dot(Ax)
            Ax_norm = np.linalg.norm(Ax)
            ksi = bAx / (Ax_norm * Ax_norm)
            avg_vec = ksi * avg_vec
            App = scipy.sparse.diags(c_mat.diagonal())  # Jacobi preconditioner
            z_vec, info = scipy.sparse.linalg.cg(c_mat, btwb_vec, x0=ksi * avg_vec, tol=1e-8, maxiter=100, M=App,
                                                 callback=None, atol=0.0)  # None
            # z_vec = z_vec[0]

        return z_vec

    def _refine_patches_none(self, err_vec):
        ref_vec_u = np.zeros(self._u_basis.n_intervals)
        ref_vec_v = np.zeros(self._v_basis.n_intervals)
        err_mat = np.empty((self._u_basis.n_intervals, self._v_basis.n_intervals))
        return ref_vec_u, ref_vec_v, err_mat

    def _refine_patches_linf(self, err_vec):
        """
        Determines interval in knot vector that have to be refined
        :return ref_vec_u, ref_vec_v as numpy array
        """
        err_mat = np.empty(self.nuv)
        for ii, p in enumerate(self.point_loc):
            if p:
                linf_norm = np.max(np.abs(err_vec[p]))
            else:
                linf_norm = 0
            iu, iv = self.patch_id2pos(ii)
            err_mat[iu][iv] = linf_norm
        error = np.max(err_mat)
        self.error = error
        if error > self.tolerance:
            logging.info(f"Linf refinement, error: {error} ")
            ref_vec_u = np.max(err_mat, axis=1) > self.tolerance
            ref_vec_v = np.max(err_mat, axis=0) > self.tolerance
        else:
            ref_vec_u = np.zeros(self._u_basis.n_intervals)
            ref_vec_v = np.zeros(self._v_basis.n_intervals)
        return ref_vec_u, ref_vec_v, err_mat

    def _refine_patches_l2(self, err_vec):
        err_mat = np.empty(self.nuv)
        err_vec_2 = err_vec * err_vec
        for ii, p in enumerate(self.point_loc):
            l2_sum = np.sum(err_vec_2[p])
            iu, iv = self.patch_id2pos(ii)
            err_mat[iu][iv] = l2_sum
        # TODO: this assumes uniform point distribution

        error = np.sqrt(np.sum(err_mat) / self._surface_points.n_active)
        self.error = error
        tol_density_sq_error = self.tolerance * self.tolerance * self._surface_points.n_active
        if error > self.tolerance:
            ref_vec_u = np.sum(err_mat, axis=1) / self._u_basis.interval_diff_vector()
            ref_vec_v = np.sum(err_mat, axis=0) / self._v_basis.interval_diff_vector()
            logging.info(f"L2 refinement, error: {error} > {self.tolerance}")
            u_bound = np.quantile(ref_vec_u, 1 - self.refine_part)
            ref_vec_u = ref_vec_u > min(tol_density_sq_error, u_bound)
            v_bound = np.quantile(ref_vec_v, 1 - self.refine_part)
            ref_vec_v = ref_vec_v > min(tol_density_sq_error, v_bound)
        else:
            logging.info(f"L2 finalize, error: {error} ")
            ref_vec_u = np.zeros(self._u_basis.n_intervals)
            ref_vec_v = np.zeros(self._v_basis.n_intervals)
        return ref_vec_u, ref_vec_v, err_mat

    def plot_error(self, err_mat, u_ref, v_ref):
        max_val = np.empty((self._u_basis.n_intervals, self._v_basis.n_intervals))
        for iu in range(max_val.shape[0]):
            for iv in range(max_val.shape[1]):
                patch_points = self.point_loc[self.patch_pos2id(iu, iv)]
                if len(patch_points) == 0:
                    if iu > 0: patch_points.extend(self.point_loc[self.patch_pos2id(iu - 1, iv)])
                    if iu + 1 < max_val.shape[0]: patch_points.extend(self.point_loc[self.patch_pos2id(iu + 1, iv)])
                    if iv > 0: patch_points.extend(self.point_loc[self.patch_pos2id(iu, iv - 1)])
                    if iv + 1 < max_val.shape[1]: patch_points.extend(self.point_loc[self.patch_pos2id(iu, iv + 1)])

                max_val[iu][iv] = np.max(self.z_points[patch_points])
        # max_val = np.array([i for i in range(len(self.point_loc))]).reshape((self._u_basis.n_intervals, self._v_basis.n_intervals))

        import matplotlib.pyplot as plt
        from matplotlib.image import NonUniformImage

        fig = plt.figure(figsize=(20, 10))
        ax1, ax2 = fig.subplots(1, 2)
        # ax1.imshow(err_mat)
        im = NonUniformImage(ax1, interpolation='nearest', extent=(0, 0, 1, 1))
        im.set_data(self._v_basis.interval_centers(), self._u_basis.interval_centers(), err_mat)
        ax1.images.append(im)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title("Linf error o patches")
        # U refinement on X axis
        ax1.scatter(np.zeros_like(u_ref), u_ref * self._u_basis.interval_centers(), c='r')
        # V refinement on Y axis
        ax1.scatter(v_ref * self._v_basis.interval_centers(), np.zeros_like(v_ref), c='r')

        # print(np.min(mat), np.max(mat))
        normalize = plt.Normalize(vmin=np.min(err_mat), vmax=np.max(err_mat))
        scalar_mappable = plt.cm.ScalarMappable(norm=normalize)
        fig.colorbar(scalar_mappable, ax=ax1)

        # ax2.scatter(self.uv_points[:, 1], self.uv_points[:, 0], c=self.z_points, s=3, alpha=0.9)
        ax2.imshow(max_val)
        ax2.set_xlabel("V")
        ax2.set_ylabel("U")
        fig.show()

    def _compute_default_nuv(self):
        """
        Default nu and nv for given number of points inside of quad.
        :return: (nu, nv)
        """
        logging.info(f"n points: {self.n_points}")
        n_points = self.n_points

        dv = la.norm(self.quad[0, :] - self.quad[1, :])
        du = la.norm(self.quad[2, :] - self.quad[1, :])

        # try to make number of unknowns less than the number of remaining points
        # +1 to improve determination
        point_density = np.sqrt(n_points / dv / du)
        nv = dv * point_density
        nu = du * point_density
        nuv = np.array([np.floor(nu / 3), np.floor(nv / 3)]) - self._degree
        nuv = np.maximum(1, nuv)
        nuv = nuv.astype(int)
        if nuv[0] < 1 or nuv[1] < 1:
            raise Exception("Two few points, {}, to make approximation, degree: {}".format(n_points, self._degree))

        return nuv

    def _init_coo_structure(self):
        """
        Initialize coordinate structure of the system of normal equations
        """

        u_n_basf = self._u_basis.size
        u_n_int = self._u_basis.size - 2
        v_n_int = self._v_basis.size - 2

        n_uv_loc_nz = ((self._u_basis.degree + 1) * (self._v_basis.degree + 1)) ** 2
        n_nz = u_n_int * v_n_int * n_uv_loc_nz

        row = np.zeros(n_nz, dtype=int)
        col = np.zeros(n_nz, dtype=int)
        data = np.zeros(n_nz)
        linsp = np.array([0, 1, 2], dtype=int)
        linsp31 = np.repeat(linsp, 3)  # linsp_v
        linsp13 = np.tile(linsp, 3)  # linsp_u
        nnz_b = 0

        for iu in range(0, u_n_int):
            iu_shift = np.repeat(iu, 9) + linsp13
            for iv in range(0, v_n_int):
                col_item = (linsp31 + np.repeat(iv, 9)) * u_n_basf + iu_shift
                col[nnz_b: nnz_b + 81] = np.tile(col_item, 9)
                row[nnz_b: nnz_b + 81] = np.repeat(col_item, 9)
                nnz_b += 81

        return row, col, data

    def patch_pos2id(self, iu, iv):
        id = iu * self._v_basis.n_intervals + iv
        return id

    def patch_id2pos(self, patch_id):
        iu = int(np.floor(patch_id / self._v_basis.n_intervals))
        iv = int(patch_id - (iu * self._v_basis.n_intervals))

        return iu, iv

    @tools.func_timer
    def _locate_points(self):
        """
        Construction of the system B^TWBz=B^TWb
        for control points of the 2nd order B-spline surface
        """
        logging.info(f"Using {self.nuv} B-spline approximation.")

        point_loc = []
        n_points = self.uv_points.shape[0]

        for i in range(0, (self._u_basis.size - 2) * (self._v_basis.size - 2)):
            point_loc.append([])

        for idx in range(n_points):
            u, v = self.uv_points[idx, 0:2]
            iu = self._u_basis.find_knot_interval(u)
            iv = self._v_basis.find_knot_interval(v)
            idp = self.patch_pos2id(iu, iv)
            point_loc[idp].append(idx)

        self.point_loc = point_loc

    def _build_system_of_normal_equations(self):
        """
        Construction of the system B^TWBz=B^TWb
        for control points of the 2nd order B-spline surface

        Possible assembly optimization:
        for point n, with u, v coords wit nonzero basis functions:
        I,I+1,I+2 in u, and J,... in v
        nonzero points: [ size_v * (I+i) + (J+j), size_v * (I+k) + (J+l) ] with values:
        M_ijkl = b_i(u) * b_k(u) * b_j(v) * b_l(v) * w_n^2

        b_i*b_k .. can be efficiently evaluated for all u in the interval, having always long enough vectors of all points in that interval
        b_j*b_l .. the same in V basis

        moreover we can (at least in early approximation) merge close U points, very efficient for (sparse) grid point clouds
        also try to merge weights per patches

        local polynomial (Legendre) approximation can be used to find suitable refinement of knot vectors and
        also resample to regular grid

        also reorder points to be more memory local

        BtB assembly theoretically involve 91 multiplications and about 200 instruction at most for basis functions
        so 300 instructions per point millions of points should be processed in a second.
        Using CG we can possibly apply matrix free approach,but only for a (sparse) point grid
        """
        normal_matrix_size = self._u_basis.size * self._v_basis.size
        n_patches = (self._u_basis.size - 2) * (self._v_basis.size - 2)
        n_points = self.uv_points.shape[0]
        row, col, data = self._init_coo_structure()
        vec_BTb = np.zeros(normal_matrix_size)
        avg_vec = np.ones(normal_matrix_size) * np.sum(self.z_points) / n_points

        for patch_id in range(0, n_patches):
            if len(self.point_loc[patch_id]) > 0:
                patch_point_loc = self.point_loc[patch_id]
                # logging.info(f"{patch_id} : {len(patch_point_loc)}")
                u_loc_vec = self.uv_points[patch_point_loc, 0]
                v_loc_vec = self.uv_points[patch_point_loc, 1]
                w_loc_vec = self.weights[patch_point_loc]
                b_loc_vec = self.z_points[patch_point_loc]
                iu, iv = self.patch_id2pos(patch_id)
                u_loc_base_vec = self._u_basis.eval_vector(iu, u_loc_vec)
                v_loc_base_vec = self._v_basis.eval_vector(iv, v_loc_vec)
                v_kron_u = (u_loc_base_vec[None, :, :] * v_loc_base_vec[:, None, :]).reshape(9, len(
                    self.point_loc[patch_id]))
                w_mult_v_kron_u = w_loc_vec[None] * v_kron_u
                loc_norm_mat = np.sum(w_mult_v_kron_u[None, :, :] * w_mult_v_kron_u[:, None, :], axis=2).reshape(81)
                b_row = col[patch_id * 81: patch_id * 81 + 9]
                data[patch_id * 81:(patch_id + 1) * 81] = loc_norm_mat
                vec_BTb[b_row.tolist()] += np.sum((b_loc_vec * w_mult_v_kron_u), axis=1)

        with tools.catch_time("assembly"):
            mat_BTB = scipy.sparse.csr_matrix((data, (row, col)), shape=(normal_matrix_size, normal_matrix_size))
        return mat_BTB, vec_BTb, avg_vec

    def _compute_errors(self, z_vec):  # ,input_data_reduction,w_quad_points_compl):
        """
        Compute errors in approximation of the surface with respect
        differences in z-coordinate. Computation is performed individually
        on every patch in order to avoid to store whole system of the
        overdetermined matrix at one time.
        point_loc[patch_id][point_id]:  as list of the list
        z_vec: z-coordinates corresponding to the computed surface as numpy array of the size equal to n_points
        """
        n_patches = (self._u_basis.size - 2) * (self._v_basis.size - 2)
        if self.validation_fr != 1.0:
            n_points_glob = int(np.ceil(len(self.uv_points) * self.validation_fr))
            w_quad_points = self.weights + self._w_quad_points_compl
        else:
            n_points_glob = len(self.uv_points)
            w_quad_points = self.weights

        u_n_basf = self._u_basis.size
        v_n_basf = self._v_basis.size
        g_vec = self.z_points[:]
        n = g_vec.shape[0]
        err = np.zeros([n])

        linsp = np.array([0, 1, 2], dtype=int)
        linsp31 = np.repeat(linsp, 3)  # linsp_v
        linsp13 = np.tile(linsp, 3)  # linsp_u
        err_mat_max = np.zeros((self._u_basis.size - 2, self._v_basis.size - 2))
        err_mat_eucl2 = np.zeros((self._u_basis.size - 2, self._v_basis.size - 2))

        for patch_id in range(0, n_patches):
            if len(self.point_loc[patch_id]) > 0:
                patch_point_loc = self.point_loc[patch_id]
                iu, iv = self.patch_id2pos(patch_id)
                u_vec = self.uv_points[patch_point_loc, 0]
                v_vec = self.uv_points[patch_point_loc, 1]
                w_quad_points_loc = w_quad_points[patch_point_loc]
                # iu = self._u_basis.find_knot_interval(u_vec[0])
                # iv = self._v_basis.find_knot_interval(v_vec[0])
                col = (linsp31 + iv) * u_n_basf + iu + linsp13
                z_loc = z_vec[col]
                u_base_vec = self._u_basis.eval_vector(iu, u_vec)
                v_base_vec = self._v_basis.eval_vector(iv, v_vec)
                z_mat_loc = z_loc.reshape(self._v_basis.degree + 1, self._u_basis.degree + 1)
                z_u_mat = z_mat_loc @ u_base_vec
                patch_z_vec = np.sum(v_base_vec * z_u_mat, axis=0)
                patch_err = (patch_z_vec - g_vec[patch_point_loc]) * w_quad_points_loc
                err[patch_point_loc] = patch_err
                # if self.validation_fr != 1.0:
                #     patch_err = patch_err * self._w_quad_points_mask[patch_point_loc]
                # err_mat_max[iu][iv] = np.max(np.abs(patch_err))
                # err_mat_eucl2[iu][iv] = np.linalg.norm(patch_err)*np.linalg.norm(patch_err)

        # std_dev = math.sqrt(np.sum(np.sum(err_mat_eucl2, axis=0))/(n_points_glob - 1))

        return err  # , err_mat_max, err_mat_eucl2, std_dev

    def _basis_in_q_points(self, basis):
        n_int = basis.n_intervals
        nq_points = len(self._q_points)
        q_point = np.zeros((n_int * nq_points, 1))
        point_val_outer = np.zeros((3, 3, n_int))  # "3" considers degree 2
        d_point_val_outer = np.zeros((3, 3, n_int))  # "3" considers degree 2

        # TODO: use numpy functions for quadrature points
        n = 0
        for i in range(n_int):
            us = basis.knots[i + 2]
            uil = basis.knots[i + 3] - basis.knots[i + 2]
            for j in range(nq_points):
                up = us + uil * self._q_points[j]
                q_point[n] = up
                u_base_vec = basis.eval_vector(i, up)
                u_base_vec_diff = basis.eval_diff_vector(i, up)
                point_val_outer[:, :, i] += self._q_weights[j] * np.outer(u_base_vec, u_base_vec)
                d_point_val_outer[:, :, i] += self._q_weights[j] * np.outer(u_base_vec_diff, u_base_vec_diff)
                n += 1

        # #TODO: use numpy functions for quadrature points
        # n = 0
        # for i in range(n_int):
        #     a, b = basis.knot_interval_bounds(i)
        #     us = basis.knots[i + 2]
        #     uil = basis.knots[i + 3] - basis.knots[i + 2]
        #     t_q_points = a + (b-a) * self._q_points
        #     basis_vec = basis.eval_vector(i, t_q_points)
        #     diff_vec = basis.eval_diff_vector(i, t_q_points)
        #     point_val_outer[:, :, i] += (basis_vec @ self._q_weights)[:, ] @ basis_vec[* np.outer(u_base_vec, u_base_vec)
        #     d_point_val_outer[:, :, i] += self._q_weights[j] * np.outer(u_base_vec_diff, u_base_vec_diff)
        #     for j in range(nq_points):
        #         up = us + uil * self._q_points[j]
        #         q_point[n] = up
        #         u_base_vec = basis.eval_vector(i, up)
        #         u_base_vec_diff = basis.eval_diff_vector(i, up)
        #         n += 1

        return point_val_outer, d_point_val_outer, q_point

    def _build_sparse_reg_matrix(self):
        """
        Construction of the regularization matrix (A) to decrease variation of the terrain
        B z = b  ---> (B^T B + A)z = B^T b
        :param u_knots: vector of v-knots
        :param v_knots: vector of u-knots
        :param quad: points defining quadrangle area (array)
        :return: matrix

        -
        """

        # a = quad[:, 3] - quad[:, 2]
        # b = quad[:, 0] - quad[:, 1]
        # c = quad[:, 1] - quad[:, 2]
        # d = quad[:, 0] - quad[:, 3]

        u_n_basf = self._u_basis.size
        v_n_basf = self._v_basis.size
        u_n_inter = self._u_basis.n_intervals
        v_n_inter = self._v_basis.n_intervals
        n_uv_loc_nz = (self._u_basis.degree + 1) * (self._v_basis.degree + 1)

        # TODO: use Gauss quadrature from scipy
        # in fact for general degrees we should use different quadrature for u and different for v
        self._q_points = [0, (0.5 - 1 / np.sqrt(20)), (0.5 + 1 / np.sqrt(20)), 1]
        self._q_weights = [1.0 / 6, 5.0 / 6, 5.0 / 6, 1.0 / 6]
        nq_points = len(self._q_points)

        u_val_outer, u_diff_val_outer, q_u_point = self._basis_in_q_points(self._u_basis)
        v_val_outer, v_diff_val_outer, q_v_point = self._basis_in_q_points(self._v_basis)
        # xy_outer shape is (3, 3, n_inter)

        row_m = np.zeros((v_n_inter * u_n_inter * n_uv_loc_nz * n_uv_loc_nz))
        col_m = np.zeros((v_n_inter * u_n_inter * n_uv_loc_nz * n_uv_loc_nz))
        data_m = np.zeros((v_n_inter * u_n_inter * n_uv_loc_nz * n_uv_loc_nz))

        nnz_a = 0
        # linsp = np.linspace(0, self._u_basis.degree, self._u_basis.degree+1)
        # llinsp = np.tile(linsp, self._u_basis.degree+1)
        # np.repeat((iv + linsp) * u_n_basf, self._u_basis.degree + 1) + llinsp
        i_local = np.arange(self._u_basis.degree + 1, dtype=int)
        iuv_local = (u_n_basf * i_local[:, None] + i_local[None, :]).ravel()  # 0,1,2, N+[0,1,2], 2*N+[0,1,2]
        # print("vnint: {} unint: {} nqp: {} prod: {}".format(v_n_inter, u_n_inter, nq_points, v_n_inter* u_n_inter* nq_points*nq_points))
        # jac = 1.0 / u_n_inter / v_n_inter
        idx_range = n_uv_loc_nz * n_uv_loc_nz  # 9 * 9 = 81 NZ per single bspline square
        for iv in range(v_n_inter):
            v_val_outer_loc = v_val_outer[:, :, iv]
            dv_val_outer_loc = v_diff_val_outer[:, :, iv]

            for iu in range(u_n_inter):
                jac = (self._u_basis.knots[iu + 3] - self._u_basis.knots[iu + 2]) * (
                            self._v_basis.knots[iv + 3] - self._v_basis.knots[iv + 2])
                u_val_outer_loc = u_val_outer[:, :, iu]
                du_val_outer_loc = u_diff_val_outer[:, :, iu]
                # xy_outer_loc have shape 3x3

                v_du = np.kron(v_val_outer_loc, du_val_outer_loc)
                dv_u = np.kron(dv_val_outer_loc, u_val_outer_loc)
                data_m[nnz_a:nnz_a + idx_range] = jac * (v_du + dv_u).ravel()  # 9x9 values
                iuv = iu + iv * u_n_basf
                colv = iuv + iuv_local
                col_m[nnz_a:nnz_a + idx_range] = np.repeat(colv, n_uv_loc_nz)
                row_m[nnz_a:nnz_a + idx_range] = np.tile(colv, n_uv_loc_nz)
                nnz_a += idx_range
        # print("Assembled")
        mat_a = scipy.sparse.coo_matrix((data_m, (row_m, col_m)),
                                        shape=(u_n_basf * v_n_basf, u_n_basf * v_n_basf)).tocsr()
        return mat_a
