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
    # todo: numerical test

    def plot_surf(self, surf):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        bs_plot.plot_surface_3d(surf, ax)
        plt.show()


    def plot_approx_grid(self):
        points = bs.make_function_grid(function_sin_cos, 20, 30)
        gs = bs.GridSurface()
        gs.init_from_seq(points.reshape(-1, 3).T)
        z_surf = bs_approx.surface_from_grid(gs, (3,4) )
        self.plot_surf(z_surf)


    def plot_approx_transformed_grid(self):
        points = bs.make_function_grid(function_sin_cos, 20, 30)
        mat = np.array( [ [2.0, 1.0, 0.0 ], [1.0, 2.0, 0.0 ], [0.0, 0.0, 0.5 ]])
        points = np.dot(points, mat.T) + np.array([10, 20, -5.0])
        gs = bs.GridSurface()
        gs.init_from_seq(points.reshape(-1, 3).T)
        z_surf = bs_approx.surface_from_grid(gs, (3,4) )
        self.plot_surf(z_surf)

    def plot_plane(self):
        surf = bs_approx.plane_surface([ [0, 0, 0], [1,1,0], [0,0,1] ])
        self.plot_surf(surf)

    def test_surface_approx(self):
        #self.plot_approx_grid()
        #self.plot_approx_transformed_grid()
        self.plot_plane()
        pass




class TestCurveApprox:

    def test_approx_2d(self):
        x_vec = np.linspace(1.1, 3.0, 100)
        y_vec = np.array([ np.sin(10*x) for x in x_vec ])
        points = np.stack( (x_vec, y_vec), axis=1)
        curve = bs_approx.curve_from_grid(points)

        bs_plot.plot_curve_2d(curve, 1000)
        bs_plot.plot_curve_poles_2d(curve)


        plt.plot(x_vec, y_vec, color='green')
        plt.show()