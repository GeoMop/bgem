import pytest
import isec_surf_surf as iss
import bspline as bs
import numpy as np
import math
import bspline_plot as bs_plot

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class TestSurface:

    def plot_extrude(self):
        fig1 = plt.figure()

        ax1 = fig1.gca(projection='3d')


        # curve extruded to surface
        #poles_yz = [[0., 0.], [1.0, 0.5], [2., -2.], [3., 1.]]
        #poles_x = [0, 1, 2]
        #poles = [ [ [x] + yz for yz in poles_yz ] for x in poles_x ]

        def function(x):
            return math.sin(x[0]*4) * math.cos(x[1]*4)

        def function2(x):
            return math.cos(x[0]*4) * math.sin(x[1]*4)

        u_basis = bs.SplineBasis.make_equidistant(2, 10)
        v_basis = bs.SplineBasis.make_equidistant(2, 15)

        poles = bs.make_function_grid(function, 12, 17)
        surface_extrude = bs.Surface((u_basis, v_basis), poles)


        bs_plot.plot_surface_3d(surface_extrude, ax1, poles = True)
       # plt.show()
        ###
        #fig2 = plt.figure()
        #ax2 = fig2.gca(projection='3d')
        poles2 = bs.make_function_grid(function2, 12, 17)
        surface_extrude2 = bs.Surface((u_basis, v_basis), poles2)

        bs_plot.plot_surface_3d(surface_extrude2, ax1, poles=True)
        plt.show()

        return surface_extrude, surface_extrude2



    def plot_function(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # function surface
        def function(x):
            return math.sin(x[0]) * math.cos(x[1])

        poles = bs.make_function_grid(function, 4, 5)
        u_basis = bs.SplineBasis.make_equidistant(2, 2)
        v_basis = bs.SplineBasis.make_equidistant(2, 3)
        surface_func = bs.Surface( (u_basis, v_basis), poles)
        bs_plot.plot_surface_3d(surface_func, ax)
        bs_plot.plot_surface_poles_3d(surface_func, ax)

        plt.show()


    def boudingbox(self):
        SurfSurf = IsecSurfSurf.bounding_boxes()

    def test_isec(self):


        surf1, surf2 = self.plot_extrude()
        isec = iss.IsecSurfSurf(surf1, surf2)
        self.boudingbox(self)