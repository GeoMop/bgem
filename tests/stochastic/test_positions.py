import os
import itertools
import pytest

from bgem.polygons import polygons as poly
from bgem.stochastic import frac_plane as FP
from bgem.stochastic import frac_isec as FIC
from bgem.stochastic import isec_plane_point as IPP
#from bgem.stochastic import fracture as FRC
#from bgem.stochastic import SquareShape as SS

import scipy.linalg as la
import numpy.linalg as lan
import attr
import numpy as np
import math
import collections
# import matplotlib.pyplot as plt

# from bgem
from bgem.gmsh import gmsh
from bgem.gmsh import options as gmsh_options
from bgem.gmsh import field as gmsh_field
from bgem.stochastic import fracture
from bgem.bspline import brep_writer as bw

script_dir = os.path.dirname(os.path.realpath(__file__))


# def test_touch():
#     """
#     Create the BREP file from a list of fractures using the brep writer interface.
#     """
#
#     faces = []
#     # fracx = fracture(shape_class=fracture.SquareShape, r=5, centre= np.array([1,5,3]),normal=np.array([1,-2,1]),shape_angle=0.3,aspect=1)
#
#     eps = 0.1
#     r1 = 5.0
#     center1 = np.array([0.0, 0.0, 0.0])
#     normal1 = np.array([[0.0, 0.0, 1.0]])
#     normal1 = normal1 / np.linalg.norm(normal1)
#     angle1 = 0.0
#
#     r2 = 5.0
#     center2 = np.array([0.0, 0.0, math.sqrt(2)*r2/2 + eps])
#     normal2 = np.array([[0.0, 1.0, 0.0]])
#     normal2 = normal2 / np.linalg.norm(normal2)
#     angle2 = math.pi/4
#
#     frac_X1 = fracture.Fracture(fracture.SquareShape, r1, center1, normal1, angle1, 1.0)
#     frac_X2 = fracture.Fracture(fracture.SquareShape, r2, center2, normal2, angle2, 1.0)
#
#     X1_vert = frac_X1.transform(frac_X1.ref_vertices)
#     X2_vert = frac_X2.transform(frac_X2.ref_vertices)
#
#     faces = get_face(X1_vert, faces)
#     faces = get_face(X2_vert, faces)
#
#     # frac_isec = FIC.FracIsec(frac_X1,frac_X2)
#     # points_A, points_B = frac_isec._get_points()
#     print('here')
#
#
#
#     comp = bw.Compound(faces)
#     loc = bw.Location([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
#     with open('positions_touch_test.brep', "w") as f:
#         bw.write_model(f, comp, loc)
#
#     return

def test_angle():

    """
    Create the BREP file from a list of fractures using the brep writer interface.
    """

    faces = []
    #fracx = fracture(shape_class=fracture.SquareShape, r=5, centre= np.array([1,5,3]),normal=np.array([1,-2,1]),shape_angle=0.3,aspect=1)

    phi = math.pi/4#math.pi/4  # (0, pi/2)
    eps = 0.1
    r1 = 7.0
    offset = np.array([5.0, 5.0, 0.0])
    center1 = np.array([0.0, 0.0, 0.0])+offset
    normal1 = np.array([[0.0, 0.0, 1.0]])
    normal1 = normal1 / np.linalg.norm(normal1)
    angle1 = 0.0

    r2 = 2.0
    center2 = np.array([0.0, 0.0, r2/2 + eps])+offset
    normal2 = np.array([[0.0, math.cos(phi), math.sin(phi)]])
    normal2 = normal2 / np.linalg.norm(normal2)
    angle2 = 0.0

    frac_X1= fracture.Fracture(fracture.SquareShape,r1, center1 , normal1 ,angle1, 1.0)
    frac_X2= fracture.Fracture(fracture.SquareShape,r2, center2 , normal2 ,angle2, 1.0)

    X1_vert = frac_X1.transform(frac_X1.ref_vertices)
    X2_vert = frac_X2.transform(frac_X2.ref_vertices)

    faces = get_face(X1_vert,faces)
    faces = get_face(X2_vert,faces)


    frac_isec = FIC.FracIsec(frac_X1,frac_X2)
    points_A, points_B, conflict = frac_isec._get_points()
    conflict.get_distance()

    print('here')
    if len(points_A) ==1:
        e1 = bw.Vertex(points_A[0])
        faces.append(e1)
    if len(points_A) == 2:
        v1 = bw.Vertex(points_A[0])
        v2 = bw.Vertex(points_A[1])
        e1 = bw.Edge([v1, v2])
        faces.append(e1)
    if len(points_B) ==1:
        e1 = bw.Vertex(points_B[0])
        faces.append(e1)
    if len(points_B) == 2:
        v1 = bw.Vertex(points_B[0])
        v2 = bw.Vertex(points_B[1])
        e1 = bw.Edge([v1, v2])
        faces.append(e1)

#
#
#    faces.append(e1)
#    v3 = bw.Vertex(points_B[0])
#    v4 = bw.Vertex(points_B[1])
#    e2 = bw.Edge([v3, v4])
#    faces.append(e2)

    comp = bw.Compound(faces)
    loc = bw.Location([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    with open('positions_angle_test.brep', "w") as f:
        bw.write_model(f, comp, loc)

    return


def test_cross():

    """
    Create the BREP file from a list of fractures using the brep writer interface.
    """

    faces = []
    #fracx = fracture(shape_class=fracture.SquareShape, r=5, centre= np.array([1,5,3]),normal=np.array([1,-2,1]),shape_angle=0.3,aspect=1)

    phi = math.pi/6  # (0, pi/2)
#math.cos(phi), math.sin(phi)
    eps = 0.01
    r1 = 5.0
    center1 = np.array([0.0, 0.0, 0.0])
    normal1 = np.array([[math.sin(phi), 0.0, math.cos(phi)]])
    normal1 = normal1 / np.linalg.norm(normal1)
    angle1 = 0.0

    r2 = 8.0
    center2 = np.array([r2/2 + r1/2 + eps,0.0 , 0 ])
    normal2 = np.array([[0.0,1.0, 0.0]])  #
    normal2 = normal2 / np.linalg.norm(normal2)
    angle2 = 0.0

    frac_X1= fracture.Fracture(fracture.SquareShape,r1, center1 , normal1 ,angle1, 1.0)
    frac_X2= fracture.Fracture(fracture.SquareShape,r2, center2 , normal2 ,angle2, 1.0)

    X1_vert = frac_X1.transform(frac_X1.ref_vertices)
    X2_vert = frac_X2.transform(frac_X2.ref_vertices)

    faces = get_face(X1_vert,faces)
    faces = get_face(X2_vert,faces)


    #frac_isec = FIC.FracIsec(frac_X1,frac_X2)
    #points_A, points_B = frac_isec._get_points()
    print('here')


    comp = bw.Compound(faces)
    loc = bw.Location([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    with open('positions_cross_test.brep', "w") as f:
        bw.write_model(f, comp, loc)

    return

def get_face(frac,faces):

    v1 = bw.Vertex(frac[0, :])
    v2 = bw.Vertex(frac[1, :])
    v3 = bw.Vertex(frac[2, :])
    v4 = bw.Vertex(frac[3, :])
    e1 = bw.Edge([v1, v2])
    e2 = bw.Edge([v2, v3])
    e3 = bw.Edge([v3, v4])
    e4 = bw.Edge([v4, v1])
    f1 = bw.Face([e1, e2, e3, e4])
    faces.append(f1)

    return faces