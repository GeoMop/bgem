import pytest
import isec_surf_surf as iss

import bspline as bs
import bspline_approx as bsa
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

        def function3(x):
            return (-x[0]+ x[1] +4 +3+math.cos(3*x[0]))

        def function4(x):
            return (2*x[0]- x[1] +3+math.cos(3*x[0])  )

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
        #myplot.plot_surface_3d(surface_extrude, poles = False)
        poles2 = bs.make_function_grid(function2,  u2_int + 2, v2_int + 2) #12, 17
        surface_extrude2 = bs.Surface((u2_basis, v2_basis), poles2)
        #myplot.plot_surface_3d(surface_extrude2, poles=False)

        m = 100
        fc = np.zeros([m * m, 3])
        fc2 = np.empty([m * m, 3])
        a = 5
        b = 7
        #print(fc)

        for i in range(m):
            for j in range(m):
                #print([i,j])
                x = i / m * a
                y = j / m * b
                z = function3([x, y])
                z2 = function4([x, y])
                fc[i + j * m, :] = [x, y, z]
                fc2[i + j * m, :] = [x, y, z2]

        #print(fc)

        #gs = bs.GridSurface(fc.reshape(-1, 3))
        #gs.transform(xy_mat, z_mat)
        #approx = bsa.SurfaceApprox.approx_from_grid_surface(gs)



        #approx = bsa.SurfaceApprox(fc)
        approx = bsa.SurfaceApprox(fc)
        approx2 = bsa.SurfaceApprox(fc2)
         ##!!!
        surfz = approx.compute_approximation(nuv=np.array([10, 12]))
        #surfz.make_linear_poles()
        surfz2 = approx2.compute_approximation(nuv = np.array([6, 20]))

        surfzf = surfz.make_full_surface()
        surfzf2 = surfz2.make_full_surface()


        myplot.plot_surface_3d(surfzf, poles=False)
        myplot.plot_surface_3d(surfzf2, poles=False)

        #class SurfaceApprox:

        #return surface_extrude, surface_extrude2, myplot
        return surfzf, surfzf2, myplot



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
    @pytest.mark.skip
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


