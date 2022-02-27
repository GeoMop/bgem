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

from fixtures import sandbox_fname


#script_dir = os.path.dirname(os.path.realpath(__file__))


def test_trans():
    """
    Create the BREP file from a list of fractures using the brep writer interface.
    """

    faces = []
    #fracx = fracture(shape_class=fracture.SquareShape, r=5, centre= np.array([1,5,3]),normal=np.array([1,-2,1]),shape_angle=0.3,aspect=1)


    frac_X1= fracture.Fracture(fracture.SquareShape,1.0, np.array([1.0, 5.0, 3.0]), np.array([[1.0, -2.0, 1.0]])/np.linalg.norm(np.array([[1.0, -2.0, 1.0]])),math.pi/4, 1.0)
    frac_X2= fracture.Fracture(fracture.SquareShape,2.0, np.array([1.0, 5.0, 3.0]), np.array([[1.0, 1.0, 2.0]])/np.linalg.norm(np.array([[1.0, 1.0, 2.0]])),math.pi/3, 1.0)
    frac_X3= fracture.Fracture(fracture.SquareShape,5.0, np.array([1.0, 5.0, 3.0]), np.array([[3.0, 2.0, 1.0]])/np.linalg.norm(np.array([[3.0, 2.0, 1.0]])),math.pi/6, 1.0)

    #X1_vert = frac_X1.transform(frac_X1.shape_class._points)
    #X2_vert = frac_X2.transform(frac_X2.shape_class._points)
    #X3_vert = frac_X3.transform(frac_X3.shape_class._points)

    X1_vert = frac_X1.transform(frac_X1.ref_vertices)
    X2_vert = frac_X2.transform(frac_X2.ref_vertices)
    X3_vert = frac_X3.transform(frac_X3.ref_vertices)


#    X1_vert = frac_X1.vertices
#    X2_vert = frac_X2.vertices
#    X3_vert = frac_X3.vertices



    X1_vertr = frac_X1.back_transform(X1_vert)
    X2_vertr = frac_X2.back_transform(X2_vert)
    X3_vertr = frac_X3.back_transform(X3_vert)


    #frac = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]])
    #frac = np.array([[1, 0, -2], [1, 3, 1], [-1, 3, 1], [-1, 0, -2]])

    faces = get_face(X1_vert,faces)
    faces = get_face(X2_vert,faces)
    faces = get_face(X3_vert,faces)
    faces = get_face(X1_vertr, faces)
    faces = get_face(X2_vertr, faces)
    faces = get_face(X3_vertr, faces)

    #faces = get_face(frac_B,faces)

    frac_isec = FIC.FracIsec(frac_X1,frac_X2)
    points_A, points_B, conflict = frac_isec._get_points()
    print('here')
    #v1 = bw.Vertex(frac[2, :])
    #v2 = bw.Vertex(frac[3, :])
    #e1 = bw.Edge([v1, v2])

    comp = bw.Compound(faces)
    loc = bw.Location([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    with open(sandbox_fname('trans_test','brep'), "w") as f:
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