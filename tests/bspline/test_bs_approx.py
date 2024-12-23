import pytest
import logging

logging.basicConfig(filename='test_bs_approx.log', level=logging.INFO, filemode='w')
# Forcing the new setting works only since Python 3.8
# logging.basicConfig(filename='test_bs_approx.log', level=logging.INFO, force=True, filemode='w')

import os
import numpy as np
from bgem.bspline import bspline as bs, \
    bspline_plot as bs_plot, \
    bspline_approx as bs_approx
from bgem.bspline.surface_point_set import scale_relative, SurfacePointSet

# from fixtures import catch_time
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

"""
test cases:

grid on unit square (-1,1)x(-1,1)
- semi regular grid of points (regularization works for uniform distribution of the points)
- random points with uniform distribution
- points linearly transformed in order to test rectangle domain detection

- uniformly oscillation function: sin(x+y)*cos(x-y)
- function with varying gradients/curvatures: cos(r)exp(-0.1*abs(r)), r= sqrt(x^2+y^2)
- add some noise

- test resulting error compared to prescribed
- test overfitting
"""


class TestCurveApprox:

    def plot_approx_2d(self, curve, fn, vx, cy, vy):
        plt = bs_plot.Plotting()
        # orig points
        n_points = 10000
        basis = curve.basis
        t_coord = np.linspace(basis.domain[0], basis.domain[1], n_points)
        x_coord = curve.eval_array(t_coord)[:, 0]
        plt.plot_2d(x_coord, fn(x_coord), color='orange')
        plt.scatter_2d(vx, vy, color='orange')
        plt.plot_curve_poles_2d(curve)
        plt.plot_curve_2d(curve, n_points)
        plt.scatter_2d(vx, cy, color='blue')
        plt.show()

    def check_curve(self, curve, fn, tol):
        T = np.random.rand(200)
        curve_points = curve.eval_array(T)
        curve_y = curve_points[:, 1]
        vx = curve_points[:, 0]
        vy = fn(vx)
        err_y = np.linalg.norm(curve_y - vy)
        if err_y > tol:
            print(f"l2_err({err_y}) > tol({tol});")
            self.plot_approx_2d(curve, fn, vx, curve_y, vy)
            assert False, "Different curves."
        else:
            logging.info(f"l2_err({err_y}) < tol({tol};")

    def test_analytical(self):
        np.random.seed(1234)
        domain = (1.1, 3.0)
        fn = np.vectorize(lambda x: np.sin(10 * x))
        x_vec = np.linspace(*domain, 800)
        y_vec = fn(x_vec)
        points = np.stack((x_vec, y_vec), axis=1)
        tol = 0.01
        curve = bs_approx.curve_from_grid(points, tol=0.1 * tol * tol)
        self.check_curve(curve, fn, 0.01)

        # plt.plot(x_vec, y_vec, color='green')
        # plt.show()


# Functions -----------------------------
class AnalyticalSurface:
    """
    A surface given by an explicit function 'f(u,v)'
    as Z = f(X,Y).
    """

    @classmethod
    def cos_cos(cls, fx=4, fy=4):
        def func(x):
            """ x = ((x,y),...)"""
            x = np.atleast_2d(x)
            assert x.shape[1] == 2
            return np.cos(fx * x[:, 0]) * np.cos(fy * x[:, 1])

        return cls(func)

    @classmethod
    def cos_exp(cls, f_cos=6, f_exp=0.01):
        def func(x):
            """ x = ((x,y),...)"""
            x = np.atleast_2d(x)
            assert x.shape[1] == 2
            r = np.linalg.norm(x, axis=1)
            return np.cos(f_cos * r) * np.exp(- f_exp * r)

        return cls(func)

    def __init__(self, fn):
        self.fn = fn

    def eval(self, points):
        """
        np.array of function values for np.array of XY points.
        points: [(x,y), ...], shape (N, 2)
        """
        return self.fn(points)

    def eval_xyz(self, points, noise_level):
        """
        Return points XYZ with Z coordinate given by the function value f(X,Y)
        with added Gaussian noise with zero mean and standard variance equal to 'noise_level'.
        """
        # noise = noise_level * np.random.rand(len(points))
        rng = np.random.default_rng()
        noise = rng.normal(0, noise_level, len(points))
        z = self.eval(points) + noise
        return np.stack([points[:, 0], points[:, 1], z], axis=1)


# Grid points -----------------------------
class PointSet:
    """
    A set of points in a rectangle used for the approximation tests.
    Provides regular grid, perturbed grid, random points.
    S
    """

    @staticmethod
    def regular_grid(nu, nv):
        # surface on unit square
        U = np.linspace(0.0, 1.0, nu)
        V = np.linspace(0.0, 1.0, nv)
        V_grid, U_grid = np.meshgrid(V, U)

        return np.stack([U_grid.ravel(), V_grid.ravel()], axis=1)

    @staticmethod
    def irregular_grid(nu, nv):
        points = PointSet.regular_grid(nu, nv)
        dx = 0.5 / nu * np.random.random(nu * nv)
        dy = 0.5 / nv * np.random.random(nu * nv)
        points += np.stack([dx, dy], axis=1)
        points[points[:, 0] < 0, 0] = 0
        points[points[:, 0] > 1, 0] = 1
        points[points[:, 1] < 0, 1] = 0
        points[points[:, 1] > 1, 1] = 1
        return points

    @staticmethod
    def uniform_random(nu, nv):
        return np.random.random(2 * nu * nv).reshape((nu * nv, 2))


class Transform2D:
    """
    Linear transform of 2D points.
    TODO: use a sort of more general class.
    """

    def __init__(self, xy, z):
        xy = np.atleast_2d(xy)
        z = np.atleast_1d(z)
        self.xy_scale = xy[:, :2]
        self.z_scale = z[0]
        self.xyz_shift = np.array([xy[0, 2], xy[1, 2], z[1]])

    def __call__(self, X):
        _X = X.copy()
        _X[:, 0:2] = (self.xy_scale @ _X[:, 0:2].T).T
        _X[:, 2] *= self.z_scale
        return _X + self.xyz_shift


# def plot_points(points):
#     """
#     Scatter plot of the (N, 3) array of XYZ points.
#     """
#     plt = bs_plot.Plotting()
#     plt.scatter_3d(points[:, 0], points[:, 1], points[:, 2])
#     plt.show()


def grid_surf_plot_cmp(a_grid, b_grid, point_cloud):
    """
    Plot approximation surface given by `b_grid`,
    the blue validation points given by `a_grid`,
    the original `point_cloud`.
    Second plot is difference of the approximation against the validation grid.
    -
    """
    diff = (b_grid[:, :, 2] - a_grid[:, :, 2]).flatten()
    c = point_cloud.reshape(-1, 3)
    x, y, z = a_grid.reshape(-1, 3).T

    plt = bs_plot.Plotting()
    # orig points
    plt.scatter_3d(c[:, 0], c[:, 1], c[:, 2], size=2, color='orange')
    # validation grid
    plt.scatter_3d(x, y, z, size=2, color='blue')
    plt.plot_surface(b_grid[:, :, 0], b_grid[:, :, 1], b_grid[:, :, 2])

    plt.show()

    plt.scatter_3d(x, y, diff, size=2, color='blue')
    plt.show()


def vec_cmp_z(a_z, b_z, tol):
    """
    Compute and report Linf difference over the tolerance `tol`.
    """
    eps = 0.0
    n_err = 0
    for i, (za, zb) in enumerate(zip(a_z, b_z)):
        diff = np.abs(za - zb)
        eps = max(eps, diff)
        if diff > tol:
            n_err += 1
            if n_err < 10:
                print(" {} =|a({}) - b({})| > tol({}), idx: {}".format(diff, za, zb, tol, i))
            elif n_err == 10:
                print("... skipping")
    print("Max norm: ", eps, "Tol: ", tol)
    return eps


def grid_cmp(a, b, c, tol):
    """
    a - validation grid XYZ array, shape: (Nx, Ny, 3)
    b - approximation values at the same grid
    c - original point cloud, shape (N, 3)
    """
    a_z = a[:, :, 2].ravel()
    b_z = b[:, :, 2].ravel()
    eps = vec_cmp_z(a_z, b_z, tol)
    if eps > tol:
        grid_surf_plot_cmp(a, b, c)
        assert False, "Different surfaces."


def plot_cmp(a, b):
    """
    Plot approx points vs. validation points a s a scatter plot.
    """
    plt = bs_plot.Plotting()
    plt.scatter_3d(a[:, 0], a[:, 1], a[:, 2], color='green')
    plt.scatter_3d(b[:, 0], b[:, 1], b[:, 2], color='red')
    plt.show()

    diff = b - a
    # plt.scatter_3d(a[:, 0], a[:, 1], diff[:, 0])
    # plt.scatter_3d(a[:, 0], a[:, 1], diff[:, 1])
    plt.scatter_3d(a[:, 0], a[:, 1], diff[:, 2])
    plt.show()


def vec_cmp(a, b, tol):
    a_z = a[:, 2]
    b_z = b[:, 2]
    eps = vec_cmp_z(a_z, b_z, tol)
    if eps > tol:
        plot_cmp(a, b)
        assert False, f"Different surfaces: {eps} > {tol}."


class TestSurfaceApprox:

    def analytical_points(self, points_fn, z_fn, noise_level, transform):
        # generate points and transform to [-1, 1] x [-1, 1]
        xy = 2 * points_fn(50, 50) + np.array([[-1, -1]])
        # rotate clockwise pi/4 and blow up by 1.44
        # shift to (10,20)
        # z scale by 2, shift to -10
        xyz = transform(z_fn().eval_xyz(xy, noise_level))

        surf_points = SurfacePointSet(xyz)
        return surf_points

    def validation_grid(self, nuv, z_fn, transform, noise_level):
        # Boundaries are bad for random point sets, so use slightly smaller point set for validation
        validation_xy = 2 * PointSet.regular_grid(*nuv) + np.array([[-1, -1]])
        validation_xy = scale_relative(validation_xy, 0.95)
        return transform(z_fn().eval_xyz(validation_xy, noise_level))

    # @pytest.mark.skip
    @pytest.mark.parametrize("points_fn",
                             [PointSet.regular_grid, PointSet.irregular_grid, PointSet.uniform_random])
    # [Points.uniform_random])
    @pytest.mark.parametrize("z_fn",
                             [AnalyticalSurface.cos_cos, AnalyticalSurface.cos_exp])
    # [Function.cos_exp])
    @pytest.mark.parametrize("noise",
                             # [0.0, 0.001])
                             [0.0])
    @pytest.mark.parametrize("adapt_tol", [(None, 0.1, 25), ("l2", 0.018, 5), ("linf", 0.018, 5)])
    # @pytest.mark.parametrize("adapt_tol", [("linf", 0.018, 5)])
    def test_analytical(self, points_fn, z_fn, noise, adapt_tol):
        """
        Test of a point set approximation with fixed (automatically chosen) knot vectors.

        None adaptivity

        L2 adaptivity have slightly larger Linf error then prescribed L2 norm tolerance.
        Not clear if it is the only reason.

        Linf is also unable to achieve target tolerance on validation pointset. REgularzitaion needied as there are
        empty patches. Regularization should therefore be dependent on number of patch points.
        """
        np.random.seed(1234)
        transform = Transform2D(xy=[[1.0, -1.0, 10], [1.0, 1.0, 20]],
                                z=[2.0, -10])
        surf_points = self.analytical_points(points_fn, z_fn, noise, transform)
        quad = surf_points.compute_default_quad()

        # approx quad check
        ref_quad = np.array([[-2, 0], [0, -2], [2, 0], [0, 2]])
        ref_quad += np.array([10, 20])
        assert np.allclose(ref_quad, quad, atol=1e-2)
        # set quad to ref in order to contain also validation points
        surf_points.set_quad(ref_quad)

        # Boundaries are bad for random point sets, so use slightly smaller point set for validation
        validation_nuv = (101, 101)
        validation_xyz = self.validation_grid(validation_nuv, z_fn, transform, noise)

        approx = bs_approx.SurfaceApprox(surf_points)
        nuv = approx._compute_default_nuv()
        assert np.allclose(np.array([14, 14]), nuv)

        adapt_type, tol, init_n = adapt_tol
        surface = approx.compute_approximation(
            nuv=[init_n, init_n],
            adapt_type=adapt_type,
            tolerance=0.01,
            regul_coef=0.00001)
        validation_xy = validation_xyz[:, :2]
        approx_xyz = surface.eval_xy_array(validation_xy)
        print("Approx error: ", approx.error)
        # vec_cmp(validation_xyz, approx_xyz, tol)
        grid_cmp(validation_xyz.reshape((*validation_nuv, 3)),
                 approx_xyz.reshape((*validation_nuv, 3)),
                 surf_points.xyz(), tol)

    def real_data_approx(self, file, **kwargs):
        this_source_dir = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(this_source_dir, file)
        sps = SurfacePointSet.from_file(data_path, delimiter=',')
        validation_set = sps.remove_random(100)

        sa = bs_approx.SurfaceApprox(sps)
        surface = sa.compute_approximation(**kwargs)
        print("Approx error: ", sa.error)
        approx_xyz = surface.eval_xy_array(validation_set.xy_points)
        vec_cmp(validation_set.xyz(), approx_xyz, kwargs['validation_tol'])

    @pytest.mark.skip
    def test_cg(self):
        np.random.seed(1234)
        self.real_data_approx(file="grid_200_m.csv",
                              nuv=[20, 20],
                              solver='spsolve',
                              adapt_type="l2",
                              max_iter=10,
                              std_dev=None,
                              tolerance=13,  # achieve 11
                              regul_coef=0.5,
                              validation_tol=26)  # Linf norm

    def plot_aprox(self, surf_approx):
        app = surf_approx.approx
        myplot = bs_plot.Plotting((bs_plot.PlottingPlotly()))
        myplot.plot_surface_3d(surf_approx.surfz, poles=False)
        myplot.scatter_3d(app._xy_points[:, 0], app._xy_points[:, 1], app._z_points)
        myplot.show()  # view

    @pytest.mark.skip
    def test_adapt(self):
        control_points = [60, 60]
        file = "grid_200_m.csv"
        this_source_dir = os.path.dirname(os.path.realpath(__file__))
        absfile = this_source_dir + file
        solver = "spsolve"
        adapt_type = "std_dev"
        max_iters = 5
        max_diff = None
        std_dev = 1
        input_data_reduction = 1.0

        sapp = SurfApprox(control_points, absfile, solver, adapt_type, max_iters, max_diff, std_dev,
                          input_data_reduction)
        self.plot(sapp)

    @pytest.mark.skip
    def test_adapt_regul(self):
        control_points = [50, 50]
        file = "/grid_200_m.csv"
        this_source_dir = os.path.dirname(os.path.realpath(__file__))
        absfile = this_source_dir + file
        solver = "cg"
        adapt_type = "std_dev"
        max_iters = 10
        max_diff = None
        std_dev = 1
        input_data_reduction = 0.6

        sapp = SurfApprox(control_points, absfile, solver, adapt_type, max_iters, max_diff, std_dev,
                          input_data_reduction)
        self.plot(sapp)
