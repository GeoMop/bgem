import pytest
import bspline as bs
import numpy as np
import math
import bspline_plot as bs_plot
import matplotlib.pyplot as plt


class TestSplineBasis:
    def test_find_knot_interval(self):
        eq_basis = bs.SplineBasis.make_equidistant(2, 100)
        assert eq_basis.find_knot_interval(0.0) == 0
        assert eq_basis.find_knot_interval(0.001) == 0
        assert eq_basis.find_knot_interval(0.01) == 0
        assert eq_basis.find_knot_interval(0.011) == 1
        assert eq_basis.find_knot_interval(0.5001) == 50
        assert eq_basis.find_knot_interval(1.0 - 0.011) == 98
        assert eq_basis.find_knot_interval(1.0 - 0.01) == 98
        assert eq_basis.find_knot_interval(1.0 - 0.001) == 99
        assert eq_basis.find_knot_interval(1.0) == 99

    def plot_basis(self, degree):

        eq_basis = bs.SplineBasis.make_equidistant(degree, 4)

        n_points = 401
        dx = (eq_basis.domain[1] - eq_basis.domain[0]) / (n_points -1)
        x_coord = [eq_basis.domain[0] + dx * i for i in range(n_points)]

        for i_base in range(eq_basis.size):
            y_coord = [ eq_basis.eval(i_base, x) for x in x_coord ]
            plt.plot(x_coord, y_coord)

        plt.show()


    def test_eval(self):
        #self.plot_basis(0)
        #self.plot_basis(1)
        #self.plot_basis(2)
        #self.plot_basis(3)

        eq_basis = bs.SplineBasis.make_equidistant(0, 2)
        assert eq_basis.eval(0, 0.0) == 1.0
        assert eq_basis.eval(1, 0.0) == 0.0
        assert eq_basis.eval(0, 0.5) == 0.0
        assert eq_basis.eval(1, 0.5) == 1.0
        assert eq_basis.eval(1, 1.0) == 1.0

        eq_basis = bs.SplineBasis.make_equidistant(1, 4)
        assert eq_basis.eval(0, 0.0) == 1.0
        assert eq_basis.eval(1, 0.0) == 0.0
        assert eq_basis.eval(2, 0.0) == 0.0
        assert eq_basis.eval(3, 0.0) == 0.0
        assert eq_basis.eval(4, 0.0) == 0.0

        assert eq_basis.eval(0, 0.125) == 0.5
        assert eq_basis.eval(1, 0.125) == 0.5
        assert eq_basis.eval(2, 1.0) == 0.0


    def test_linear_poles(self):
        eq_basis = bs.SplineBasis.make_equidistant(2, 4)
        poles = eq_basis.make_linear_poles()

        t_vec = np.linspace(0.0, 1.0, 21)
        x_vec = []
        for t in t_vec:
            b_vals = np.array([ eq_basis.eval(i, t) for i in range(eq_basis.size) ])
            x = np.dot(b_vals, poles)
            assert np.abs( x - t ) < 1e-15


class TestCurve:

    def plot_4p(self):
        degree = 2
        poles = [ [0., 0.], [1.0, 0.5], [2., -2.], [3., 1.] ]
        basis = bs.SplineBasis.make_equidistant(degree, 2)
        curve = bs.Curve(basis, poles)

        bs_plot.plot_curve_2d(curve)
        bs_plot.plot_curve_poles_2d(curve)
        plt.show()

    def test_evaluate(self):
        #self.plot_4p()
        pass

    # TODO: test rational curves, e.g. circle







def make_function_grid(fn, nu, nv):
    X_grid = np.linspace(0, 1.0, nu)
    Y_grid = np.linspace(0, 1.0, nv)
    Y, X = np.meshgrid(Y_grid, X_grid)

    points_uv = np.stack([X.ravel(), Y.ravel()], 1)
    Z = np.apply_along_axis(fn, 1, points_uv)
    points = np.stack([X.ravel(), Y.ravel(), Z], 1)

    return points.reshape( (nu, nv, 3) )




class TestSurface:

    def plot_extrude(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # curve extruded to surface
        poles_yz = [[0., 0.], [1.0, 0.5], [2., -2.], [3., 1.]]
        poles_x = [0, 1, 2]
        poles = [ [ [x] + yz for yz in poles_yz ] for x in poles_x ]
        u_basis = bs.SplineBasis.make_equidistant(2, 1)
        v_basis = bs.SplineBasis.make_equidistant(2, 2)
        surface_extrude = bs.Surface( (u_basis, v_basis), poles)
        bs_plot.plot_surface_3d(surface_extrude, ax)
        bs_plot.plot_surface_poles_3d(surface_extrude, ax)
        plt.show()

    def plot_function(self):
        # function surface
        def function(x):
            return math.sin(x[0]) * math.cos(x[1])

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        poles = make_function_grid(function, 4, 5)
        u_basis = bs.SplineBasis.make_equidistant(2, 2)
        v_basis = bs.SplineBasis.make_equidistant(2, 3)
        surface_func = bs.Surface( (u_basis, v_basis), poles)
        bs_plot.plot_surface_3d(surface_func, ax)
        bs_plot.plot_surface_poles_3d(surface_func, ax)

        plt.show()

    def test_evaluate(self):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        #self.plot_extrude()
        #self.plot_function()


        # TODO: test rational surfaces, e.g. sphere



class TestZ_Surface:




    def plot_function_uv(self):
        # function surface
        def function(x):
            return math.sin(x[0]*4) * math.cos(x[1]*4)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        poles = make_function_grid(function, 4, 5)
        u_basis = bs.SplineBasis.make_equidistant(2, 2)
        v_basis = bs.SplineBasis.make_equidistant(2, 3)
        surface_func = bs.Surface( (u_basis, v_basis), poles[:,:, [2] ])

        quad = np.array( [ [0, 0], [0, 0.5], [1, 0.1],  [1.1, 1.1] ]  )
        #quad = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        z_surf = bs.Z_Surface(quad, surface_func)
        full_surf = z_surf.make_full_surface()

        bs_plot.plot_surface_3d(z_surf, ax)
        bs_plot.plot_surface_3d(full_surf, ax, color='red')
        #bs_plot.plot_surface_poles_3d(surface_func, ax)

        plt.show()

    def test_eval_uv(self):
        self.plot_function_uv()
        pass