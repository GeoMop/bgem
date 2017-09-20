import pytest
import numpy as np
import bspline as bs
import bspline_approx as bs_approx
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import bspline_plot as bs_plot

def function_sin_cos(x):
    return math.sin(x[0] * 4) * math.cos(x[1] * 4)


def gen_uv_grid(nu, nv):
    # surface on unit square
    U = np.linspace(0.0, 1.0, nu)
    V = np.linspace(0.0, 1.0, nv)
    V_grid, U_grid = np.meshgrid(V, U)

    return np.stack([U_grid.ravel(), V_grid.ravel()], axis=1)

def eval_func_on_grid(func, xy_mat, xy_shift, z_mat, shape=(50, 50)):
    nu, nv = shape
    UV = gen_uv_grid(nu, nv)
    XY = xy_mat.dot(UV.T).T + xy_shift
    z_func_eval = np.array([z_mat[0] * func([u, v]) + z_mat[1] for u, v in UV])
    return  np.concatenate((XY, z_func_eval[:, None]), axis=1).reshape(nu, nv, 3)


def eval_z_surface_on_grid(surface, xy_mat, xy_shift, shape=(50, 50)):
    nu, nv = shape
    UV = gen_uv_grid(nu, nv)
    XY = xy_mat.dot(UV.T).T + xy_shift
    Z = surface.z_eval_xy_array(XY)
    return np.concatenate((XY, Z[:, None]), axis=1).reshape(nu, nv, 3)


def eval_surface_on_grid(surface, shape=(50, 50)):
    nu, nv = shape
    UV = gen_uv_grid(nu, nv)
    return surface.eval_array(UV).reshape(nu, nv, 3)

def plot_cmp(a_grid, b_grid):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(a_grid[:, :, 0], a_grid[:, :, 1], a_grid[:, :, 2], color='green')
    ax.plot_surface(b_grid[:, :, 0], b_grid[:, :, 1], b_grid[:, :, 2], color='red')
    plt.show()

    diff = b_grid - a_grid
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(a_grid[:, :, 0], a_grid[:, :, 1], diff[:, :, 0], color='red')
    ax.plot_surface(a_grid[:, :, 0], a_grid[:, :, 1], diff[:, :, 1], color='green')
    ax.plot_surface(a_grid[:, :, 0], a_grid[:, :, 1], diff[:, :, 2], color='blue')
    plt.show()

def grid_cmp( a, b, tol):
    a_z = a[:, :, 2].ravel()
    b_z = b[:, :, 2].ravel()
    eps = 0.0
    n_err=0
    for i, (za, zb) in enumerate(zip(a_z, b_z)):
        diff = np.abs( za - zb)
        eps = max(eps, diff)
        if diff > tol:
            n_err +=1
            if n_err < 10:
                print(" {} =|a({}) - b({})| > tol({}), idx: {}".format(diff, za, zb, tol, i) )
            elif n_err == 10:
                print("... skipping")
    print("Max norm: ", eps, "Tol: ", tol)
    if eps > tol:
        plot_cmp(a, b)

class TestSurfaceApprox:
    # todo: numerical test

    def plot_surf(self, surf):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        bs_plot.plot_surface_3d(surf, ax)
        plt.show()


    def test_approx_func(self):
        xy_mat = np.array( [ [1.0, -1.0, 10 ], [1.0, 1.0, 20 ]])    # rotate left pi/4 and blow up 1.44
        z_mat = np.array( [2.0, -10] )

        xyz_func = eval_func_on_grid(function_sin_cos, xy_mat[:2,:2], xy_mat[:, 2], z_mat)

        print("Compare: Func - GridSurface.transform")
        points = bs.make_function_grid(function_sin_cos, 40, 30)
        gs = bs.GridSurface(points.reshape(-1, 3))
        gs.transform(xy_mat, z_mat)
        xyz_grid = eval_z_surface_on_grid(gs, xy_mat[:2,:2], xy_mat[:, 2])
        #grid_cmp(xyz_func, xyz_grid, 0.02)

        print("Compare: Func - GridSurface.transform.z_surf")
        xyz_grid = eval_z_surface_on_grid(gs.z_surface, xy_mat[:2, :2], xy_mat[:, 2])
        xyz_grid[:,:,2] *= z_mat[0]
        xyz_grid[:, :, 2] += z_mat[1]
        #grid_cmp(xyz_func, xyz_grid, 0.02)

        print("Compare: Func - GridSurface.transform.Z_surf_approx")
        z_surf = bs_approx.surface_from_grid(gs, (7, 5))
        #z_surf.transform(xy_mat, z_mat)
        xyz_grid = eval_z_surface_on_grid(z_surf, xy_mat[:2,:2], xy_mat[:, 2])
        grid_cmp(xyz_func, xyz_grid, 0.02)

        print("Compare: Func - GridSurface.transform.Z_surf_approx.full_surface")
        xyz_grid = eval_surface_on_grid(z_surf.make_full_surface())
        #grid_cmp(xyz_func, xyz_grid, 0.01)






        #points = bs.make_function_grid(function_sin_cos, 20, 30)
        #points = np.dot(points, mat.T) + np.array([10, 20, -5.0])
        #gs = bs.GridSurface(points.reshape(-1, 3))
        #z_surf = bs_approx.surface_from_grid(gs, (3,4) )
        #self.plot_surf(z_surf)








    def plot_approx_transformed_grid(self):
        pass

    def plot_plane(self):
        surf = bs_approx.plane_surface([ [0.0, 0, 0], [1.0, 0, 0], [0.0, 0, 1] ], overhang=0.1)
        self.plot_surf(surf)

    def test_surface_approx(self):
        # self.plot_approx_grid()
        # self.plot_approx_transformed_grid()
        # self.plot_plane()
        pass




class TestCurveApprox:

    def plot_approx_2d(self):
        x_vec = np.linspace(1.1, 3.0, 100)
        y_vec = np.array([ np.sin(10*x) for x in x_vec ])
        points = np.stack( (x_vec, y_vec), axis=1)
        curve = bs_approx.curve_from_grid(points)

        bs_plot.plot_curve_2d(curve, 1000)
        bs_plot.plot_curve_poles_2d(curve)

        plt.show()

        #plt.plot(x_vec, y_vec, color='green')
        #plt.show()

    def test_approx_2d(self):
        # self.plot_approx_2d()
        pass