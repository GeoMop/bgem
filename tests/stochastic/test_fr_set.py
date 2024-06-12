import numpy as np
import pytest
from bgem.stochastic import fr_set as frac
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


"""
Test base shapes.
"""
def test_ellipse_shape():
    shape = frac.EllipseShape


def common_shape_test_(shape):
    """
    Use MC integration to:
    - confirm the shape has unit area
    - check it could determine interrior points (but not that this check is correct)
    - check that the corresponding primitive could be made in GMSH interface
    - confirm that aabb is correct for that primitive
    :param shape:
    :return:
    """

    aabb = shape.aabb
    assert aabb.shape == (2, 2)
    N = 100000
    points = np.random.random((N, 2)) * (aabb[1] - aabb[0]) + aabb[0]
    N_in = sum(shape.are_points_inside(points))
    aabb_area = np.prod(aabb[1] - aabb[0])
    area_estimate = N_in / N * aabb_area
    assert abs(area_estimate - 1.0) < 0.01

def test_base_shapes():
    """
    Test common shape properties.
    :return:
    """
    shapes = [frac.EllipseShape(), frac.RectangleShape(), frac.PolygonShape(6), frac.PolygonShape(8)]
    for shp in shapes:
        common_shape_test_(shp)
