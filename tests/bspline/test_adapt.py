from bgem.bspline import bspline_plot as bp, bspline_approx as bsa
import numpy as np
import math
import pytest


class SurfApprox:

    def __init__(self, plane_coefficients, length, samples, control_points, additive_function):
        """
         :param plane_normal_vector: plane vector as numpy array 4x1
         :param x_size: x range as double
         :param y_size: y range as double
         :param x_n_samples: number of sample points in x range as double
         :param y_n_samples: number of sample points in y range as double
         :param x_n_control_points: x control points as double
         :param y_n_control_points: y control points as double
         :return:
         """
        self.plane_coefficients = plane_coefficients
        self.x_length = length[0]
        self.y_length = length[1]
        self.x_n_samples = samples[0]
        self.y_n_samples = samples[1]
        self.x_n_control_points = control_points[0]
        self.y_n_control_points = control_points[1]
        self.err = None
        self.surfz = None
        self.additive_function = additive_function

        def surfzf(X):
            return -(self.plane_coefficients[0] * X[0] + self.plane_coefficients[1] * X[1] + self.plane_coefficients[
                3]) / self.plane_coefficients[2] + self.additive_function(3 * X[0])

        self.surfzf = surfzf

        self.err, self.surfz = self.surf_app()

    def surf_app(self):

        fc = np.zeros([self.x_n_samples * self.y_n_samples, 3])
        for i in range(self.x_n_samples):
            for j in range(self.y_n_samples):
                fc[i + j * self.y_n_samples, 0:2] = [i / self.x_n_samples * self.x_length,
                                                     j / self.y_n_samples * self.y_length]
                fc[i + j * self.y_n_samples, 2] = self.surfzf(fc[i + j * self.y_n_samples, 0:2])

        approx = bsa.SurfaceApprox(bsa.SurfacePointSet(fc))
        surfz = approx.compute_approximation(nuv=np.array([self.x_n_control_points, self.y_n_control_points]))
        err = approx.error
        surfzf = surfz.make_full_surface()
        return err, surfzf


class TestAdapt:
    """
    Intersection test for two planes shifted by a nonlinear function
    TODO:
    - document support test classes
    - add tests for more variants of control points, add functions etc. (e.g. use pytest parameters)

     @pytest.mark.parametrize( "my_param_x", [1,2,3])
     def test_fn(my_param_x) :
            pass
     - one parameter for function
     - one parameter for n_control_points quartet
     n_samples - can be probably fixed
     x_length, y_length - probably can be fixed
    - try to document which cases are covered by which parameters
    """


coefficients = [
    # (np.array([-1, 1, -1, 7])),
    (np.array([-1, 2, -1, 7])),
]

control_points = [
    # ([5, 15]),
    ([11, 10])  #
]

length = [
    # ([5, 7]),
    ([2, 3]),  #
    # ([6, 6])
]


@pytest.mark.parametrize("plane_coefficients1", coefficients)
@pytest.mark.parametrize("control_points_1", control_points)
@pytest.mark.parametrize("length1", length)
def test_surface_intersection(plane_coefficients1, control_points_1, length1):  # plane_coefficients1

    print(plane_coefficients1, control_points_1, length1)

    # plane_coefficients1 = np.array([-1, 1, -1, 7])
    plane_coefficients2 = np.array([2, -1, -1, 3])

    def cosx(x):
        return math.cos(3 * x)

    length2 = [5, 7]
    control_points_2 = [10, 11]
    samples = [200, 200]

    sapp2 = SurfApprox(plane_coefficients2, length2, samples, control_points_2, cosx)

    myplot = bp.Plotting((bp.PlottingPlotly()))
    myplot.plot_surface_3d(sapp2.surfz, poles=False)
    myplot.show()  # view
