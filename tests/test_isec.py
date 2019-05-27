import pytest
import isec_surf_surf as iss

import bspline as bs
import bspline_approx as bsa
import numpy as np
import math
import bspline_plot as bp


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class xTestSurface:

    def plot_extrude(self):
        #fig1 = plt.figure()

        #ax1 = fig1.gca(projection='3d')



        def function(x):
            return math.sin(x[0]*4) * math.cos(x[1] * 4)

        def function2(x):
            return math.cos(x[0]*4) * math.sin(x[1] * 4)

        def function3(x):
            return (-x[0] + x[1] + 4 + 3 + math.cos(3 * x[0]))

        def function4(x):
            return (2 * x[0] - x[1] + 3 + math.cos(3 * x[0]))

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




        approx = bsa.SurfaceApprox(fc)
        approx2 = bsa.SurfaceApprox(fc2)
        surfz = approx.compute_approximation(nuv=np.array([11, 26]))
        surfz2 = approx2.compute_approximation(nuv=np.array([20, 16]))
        #surfz = approx.compute_approximation(nuv=np.array([3, 5]))
        #surfz2 = approx2.compute_approximation(nuv=np.array([2, 4]))
        surfzf = surfz.make_full_surface()
        surfzf2 = surfz2.make_full_surface()


        myplot.plot_surface_3d(surfzf, poles=False)
        myplot.plot_surface_3d(surfzf2, poles=False)

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
    #@pytest.mark.skip
    def test_isec(self):


        surf1, surf2, myplot = self.plot_extrude()
        isec = iss.IsecSurfSurf(surf1, surf2)
        #box1, tree1 = isec.bounding_boxes(surf1)
        #box2, tree2 = isec.bounding_boxes(surf2)
        #isec.get_intersection(surf1,surf2,tree1,tree2,box1,box2,isec.nt,isec.nit) # surf1,surf2,tree1,tree2
        point_list1, point_list2 = isec.get_intersection()

        m = point_list1.__len__() + point_list2.__len__()

        X = np.zeros([m])
        Y = np.zeros([m])
        Z = np.zeros([m])

        i = -1

        for point in point_list1:
            i += 1
            X[i] = point.xyz[0]
            Y[i] = point.xyz[1]
            Z[i] = point.xyz[2]

        for point in point_list2:
            i += 1
            X[i] = point.xyz[0]
            Y[i] = point.xyz[1]
            Z[i] = point.xyz[2]



        myplot.scatter_3d(X, Y, Z)


        myplot.show() # view

        #print(tree1.find_box(boxes2[0]))
        #print(surf1.poles[:,:,1])
        #print(surf1.u_basis.n_intervals)
        #print(surf1.u_basis.knots)


class SurfApprox:

    def __init__(self, f, xr, yr, xp, yp, kx, ky, fn):
        """
         :param f: plane vector as numpy array 4x1
         :param xr: x range as double
         :param yr: y range as double
         :param xp: number of sample points in x range as double
         :param yp: number of sample points in y range as double
         :param kx: x control points as double
         :param ky: y control points as double
         :return:
         """
        self.f = f
        self.xr = xr
        self.yr = yr
        self.xp = xp
        self.yp = yp
        self.kx = kx
        self.ky = ky
        self.err = None
        self.surfz = None
        self.fn = fn

        self.err, self.surfz = self.surf_app()

    def surf_app(self):

        fc = np.zeros([self.xp * self.yp, 3])
        for i in range(self.xp):
            for j in range(self.yp):
                x = i / self.xp * self.xr
                y = j / self.yp * self.yr
                z = -(self.f[0] * x + self.f[1] * y + self.f[3]) / self.f[2] + self.fn(3 * x)
                fc[i + j * self.yp, :] = [x, y, z]

        approx = bsa.SurfaceApprox(fc)
        surfz = approx.compute_approximation(nuv=np.array([self.kx, self.ky]))
        err = approx.error
        surfzf = surfz.make_full_surface()
        return err, surfzf


class TestIsecx:

    def test(self):
        f1 = np.array([-1, 1, -1, 7])
        f2 = np.array([2, -1, -1, 3])

        def cosx(x):
            return math.cos(3 * x)

        a = 5
        b = 7
        #xd = 100
        #yd = 100
        #x1k = 11
        #y1k = 26
        #x2k = 20
        #y2k = 16


        xd = 20
        yd = 20
        x1k = 3
        y1k = 5
        x2k = 4
        y2k = 6


        sapp1 = SurfApprox(f1, a, b, xd, yd, x1k, y1k, cosx)
        sapp2 = SurfApprox(f2, a, b, xd, yd, x2k, y2k, cosx)
        ist = IntersectTest(sapp1, sapp2)
        ist.test_isec()


class IntersectTest:

    def __init__(self, surfapprox, surfapprox2):
        self.surfapprox = surfapprox
        self.surfapprox2 = surfapprox2


    def plot(self):

        myplot = bp.Plotting((bp.PlottingPlotly()))
        myplot.plot_surface_3d(self.surfapprox.surfz, poles=False)
        myplot.plot_surface_3d(self.surfapprox2.surfz, poles=False)
        #myplot.scatter_3d(X, Y, Z)
        myplot.show() # view
        #return surfzf, surfzf2, myplot, err

    def test_isec(self):


        def icurv1(x):
            return (- x[2] + 0.5 * x[0] + 5 + math.cos(3 * x[0]))

        def icurv2(x):
            return ((3 * x[0] - 2 * x[1] - 4) / math.sqrt(13))


        def pcurv(x):
            return ((3 * x[0] - 2 * x[1] - 4) / math.sqrt(13))

        #surf1, surf2, myplot, err = self.plot_extrude()

        self.plot()

        isec = iss.IsecSurfSurf(self.surfapprox.surfz, self.surfapprox2.surfz)
        point_list1, point_list2 = isec.get_intersection()

        m = len(point_list1) + len(point_list2)

        X = np.zeros([m])
        Y = np.zeros([m])
        Z = np.zeros([m])

        i = -1

        tol =  self.surfapprox.err + self.surfapprox2.err

        for point in point_list1:
            i += 1
            X[i] = point.xyz[0]
            Y[i] = point.xyz[1]
            Z[i] = point.xyz[2]
            d1 = icurv1([X[i], Y[i], Z[i]])
            d2 = icurv2([X[i], Y[i]])
          #  assert math.sqrt(d1*d1 + d2*d2) < tol, "intersection point tolerance has been exceeded."


        for point in point_list2:
            i += 1
            X[i] = point.xyz[0]
            Y[i] = point.xyz[1]
            Z[i] = point.xyz[2]
            d1 = icurv1([X[i], Y[i], Z[i]])
            d2 = icurv2([X[i], Y[i]])
          #  assert math.sqrt(d1 * d1 + d2 * d2) < tol, "intersection point tolerance has been exceeded."




