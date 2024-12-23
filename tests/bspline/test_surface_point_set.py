"""
Test:
- quad
- hull and default quad
- weights filter
- uv_points calculation and uv_points filter
"""
import logging

logging.basicConfig(filename='test_bs_approx.log', level=logging.INFO, force=True)

import os
import time
import pytest
import numpy as np
from bgem.bspline import bspline as bs, \
    bspline_plot as bs_plot, \
    bspline_approx as bs_approx
from bgem.bspline import surface_point_set
from fixtures import catch_time


class TestQuad:
    """
    Test detection of minimal bounding rectangle.
    Test detection of nuv.
    """

    # def __init__(self):
    #     self.points = None
    #     self.hull = None
    #     self.quad = None

    def show(self):
        plt = bs_plot.Plotting()
        plt.scatter_2d(self.points[:, 0], self.points[:, 1])
        plt.plot_2d(self.hull[:, 0], self.hull[:, 1])
        box_lines = np.concatenate((self.quad, self.quad[0:1, :]), axis=0)
        plt.plot_2d(box_lines[:, 0], box_lines[:, 1])
        plt.show()

    @pytest.mark.skip
    def test_hull_and_box(self):
        """Some real test"""
        pass

    def test_benchmark_hull_and_box(self):
        self.points = np.random.randn(1000000, 2)
        with catch_time() as t:
            for i in range(1):
                self.hull = surface_point_set.convex_hull_2d(self.points)
        print(f"\nConvex hull of 1M points: {t}")

        with catch_time() as t:
            for i in range(10):
                self.quad = surface_point_set.min_bounding_rect(self.hull)
        print(f"\nMin area bounding box: {t} s")

        # self.show()
        return


def test_surface_set():
    sps = surface_point_set.SurfacePointSet.from_file("points_simple.csv", delimiter='\\s+')
    assert (len(sps) == 8)
    assert sps.n_active == 6
    assert bool(sps.valid_points[5]) is False
    assert bool(sps.valid_points[7]) is False
    assert sps.xy_points.shape == (6, 2)
    assert np.all(sps.weights == [1, 2, 3, 4, 5, 7])
    assert np.allclose(sps.quad, [[-1, 0], [0, -1], [1, 0], [0, 1]])
    # U = quad[2] - quad[1] = [1, 1]; V = quad[0] - quad[1] = [-1, 1]
    assert np.allclose(sps.uv_points, [[1, 0], [1, 1], [0, 1], [0, 0], [0.75, 0.25], [0.75, 0.75]])

    subset = sps.remove_random(2)
    assert len(subset) == 2
    assert len(sps) == 6
