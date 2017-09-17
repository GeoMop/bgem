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


class TestSurfaceApprox:

    def test_approx_grid(self):
        points = bs.make_function_grid(function_sin_cos, 20, 30)
        gs = bs.GridSurface()
        gs.init_from_seq(points.reshape(-1, 3).T)
        z_surf = bs_approx.from_grid(gs, (3,4) )

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        bs_plot.plot_surface_3d(z_surf, ax)

        plt.show()

    def test_approx_transformed_grid(self):
        points = bs.make_function_grid(function_sin_cos, 20, 30)
        mat = np.array( [ [2.0, 1.0, 0.0 ], [1.0, 2.0, 0.0 ], [0.0, 0.0, 0.5 ]])
        points = np.dot(points, mat.T) + np.array([10, 20, -5.0])
        gs = bs.GridSurface()
        gs.init_from_seq(points.reshape(-1, 3).T)
        z_surf = bs_approx.from_grid(gs, (3,4) )

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        bs_plot.plot_surface_3d(z_surf, ax)

        plt.show()

        # todo: transform grid points to uV in Gridsurface
        # todo: keep quad in z_surf (OK)
        # check that conversion to full keeps the transformation