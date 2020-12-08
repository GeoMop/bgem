from bgem.bspline import bspline as bs, isec_surf_surf as iss, bspline_plot as bp, bspline_approx as bsa
import numpy as np
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
import math
import pytest
#import statprof


class SurfApprox:

    def __init__(self, control_points):
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

        self.x_n_control_points = control_points[0]
        self.y_n_control_points = control_points[1]
        self.err = None
        self.surfz = None
        self.err, self.surfz = self.surf_app()

    def surf_app(self):
        #self.approx = bsa.SurfaceApprox.approx_from_file("/home/jiri/Github/real_data/grid_50_m.csv")
        self.approx = bsa.SurfaceApprox.approx_from_file("/home/jiri/Github/milan/zdrojova_data_plochy.txt")
        surfz = self.approx.compute_approximation(nuv=np.array([self.x_n_control_points, self.y_n_control_points]))
        err = self.approx.error
        surfzf = surfz.make_full_surface()
        return err, surfzf

class TestAdapt:

    def test_surface_intersection(self):
        return
        control_points = [100, 100]
        sapp = SurfApprox(control_points)
        app = sapp.approx
        myplot = bp.Plotting((bp.PlottingPlotly()))
        myplot.plot_surface_3d(sapp.surfz, poles=False)
        #myplot.scatter_3d(app._xy_points[:, 0], app._xy_points[:, 1], app._z_points)
        myplot.show() # view




