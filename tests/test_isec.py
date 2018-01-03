import pytest
import isec_surf_surf as iss
import bspline as bs
import numpy as np
import math
import bspline_plot as bp


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class TestSurface:

    def plot_extrude(self):
        #fig1 = plt.figure()

        #ax1 = fig1.gca(projection='3d')



        def function(x):
            return math.sin(x[0]*4) * math.cos(x[1]*4)

        def function2(x):
            return math.cos(x[0]*4) * math.sin(x[1]*4)

        u1_int = 4
        v1_int = 4
        u2_int = 4
        v2_int = 4

        u_basis = bs.SplineBasis.make_equidistant(2, u1_int) #10
        v_basis = bs.SplineBasis.make_equidistant(2, v1_int) #15
        poles = bs.make_function_grid(function, u1_int + 2, v1_int + 2) #12, 17
        surface_extrude = bs.Surface((u_basis, v_basis), poles)

        myplot = bp.Plotting((bp.PlottingPlotly()))
        myplot.plot_surface_3d(surface_extrude, poles = False)
        poles2 = bs.make_function_grid(function2,  u2_int + 2, v2_int + 2) #12, 17
        surface_extrude2 = bs.Surface((u_basis, v_basis), poles2)
        myplot.plot_surface_3d(surface_extrude2, poles=False)

        #myplot.show() # view

        return surface_extrude, surface_extrude2



    #def plot_function(self):
    #    fig = plt.figure()
    #    ax = fig.gca(projection='3d')

        # function surface
    #    def function(x):
    #        return math.sin(x[0]) * math.cos(x[1])

        #poles = bs.make_function_grid(function, 4, 5)
        #u_basis = bs.SplineBasis.make_equidistant(2, 2)
        #v_basis = bs.SplineBasis.make_equidistant(2, 3)
        #surface_func = bs.Surface( (u_basis, v_basis), poles)
        #bs_plot.plot_surface_3d(surface_func, ax)
        #bs_plot.plot_surface_poles_3d(surface_func, ax)

        #plt.show()


    #def boudingbox(self):
    #    SurfSurf = IsecSurfSurf.bounding_boxes()

    def test_isec(self):


        surf1, surf2 = self.plot_extrude()
        isec = iss.IsecSurfSurf(surf1, surf2)
        #box1, tree1 = isec.bounding_boxes(surf1)
        #box2, tree2 = isec.bounding_boxes(surf2)
        #isec.get_intersection(surf1,surf2,tree1,tree2,box1,box2,isec.nt,isec.nit) # surf1,surf2,tree1,tree2
        isec.get_intersection()

        #print(tree1.find_box(boxes2[0]))
        #print(surf1.poles[:,:,1])
        #print(surf1.u_basis.n_intervals)
        #print(surf1.u_basis.knots)


