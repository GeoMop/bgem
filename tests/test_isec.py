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
        u2_basis = bs.SplineBasis.make_equidistant(2, u2_int) #10
        v2_basis = bs.SplineBasis.make_equidistant(2, v2_int) #15
        poles = bs.make_function_grid(function, u1_int + 2, v1_int + 2) #12, 17
        surface_extrude = bs.Surface((u_basis, v_basis), poles)

        myplot = bp.Plotting((bp.PlottingPlotly()))
        myplot.plot_surface_3d(surface_extrude, poles = False)
        poles2 = bs.make_function_grid(function2,  u2_int + 2, v2_int + 2) #12, 17
        surface_extrude2 = bs.Surface((u2_basis, v2_basis), poles2)
        myplot.plot_surface_3d(surface_extrude2, poles=False)



        return surface_extrude, surface_extrude2, myplot



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


        surf1, surf2, myplot = self.plot_extrude()
        isec = iss.IsecSurfSurf(surf1, surf2)
        #box1, tree1 = isec.bounding_boxes(surf1)
        #box2, tree2 = isec.bounding_boxes(surf2)
        #isec.get_intersection(surf1,surf2,tree1,tree2,box1,box2,isec.nt,isec.nit) # surf1,surf2,tree1,tree2
        point_list1, point_list2 = isec.get_intersection()

        m = point_list1.__len__() + point_list2.__len__()

        X= np.zeros([m])
        Y = np.zeros([m])
        Z = np.zeros([m])

        i = -1

        for point in point_list1:
            i += 1
            X[i] = point.R3_coor[0]
            Y[i]= point.R3_coor[1]
            Z[i]= point.R3_coor[2]

        for point in point_list2:
            i += 1
            X[i] = point.R3_coor[0]
            Y[i]= point.R3_coor[1]
            Z[i]= point.R3_coor[2]



        myplot.scatter_3d(X, Y, Z)


        myplot.show() # view

        #print(tree1.find_box(boxes2[0]))
        #print(surf1.poles[:,:,1])
        #print(surf1.u_basis.n_intervals)
        #print(surf1.u_basis.knots)


