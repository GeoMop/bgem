import numpy.linalg as lan
import numpy as np
import math
from bgem.stochastic import fr_set
from bgem.bspline import brep_writer as bw
from bgem import Transform
from fixtures import sandbox_fname


def test_trans():
    """
    Create the BREP file from a list of fr_sets using the brep writer interface.
    """
    faces = []
    square_id = fr_set.RectangleShape().id
    frac_X1 = fr_set.Fracture(square_id, 1.0, np.array([1.0, 5.0, 3.0]),
                              np.array([[1.0, -2.0, 1.0]]) / np.linalg.norm(np.array([[1.0, -2.0, 1.0]])), math.pi / 4)
    frac_X2 = fr_set.Fracture(square_id, 2.0, np.array([1.0, 5.0, 3.0]),
                              np.array([[1.0, 1.0, 2.0]]) / np.linalg.norm(np.array([[1.0, 1.0, 2.0]])), math.pi / 3)
    frac_X3 = fr_set.Fracture(square_id, 5.0, np.array([1.0, 5.0, 3.0]),
                              np.array([[3.0, 2.0, 1.0]]) / np.linalg.norm(np.array([[3.0, 2.0, 1.0]])), math.pi / 6)

    X1_vert = frac_X1.transform(frac_X1.ref_vertices)
    X2_vert = frac_X2.transform(frac_X2.ref_vertices)
    X3_vert = frac_X3.transform(frac_X3.ref_vertices)

    X1_vertr = frac_X1.back_transform(X1_vert)
    X2_vertr = frac_X2.back_transform(X2_vert)
    X3_vertr = frac_X3.back_transform(X3_vert)

    faces = get_face(X1_vert, faces)
    faces = get_face(X2_vert, faces)
    faces = get_face(X3_vert, faces)
    faces = get_face(X1_vertr, faces)
    faces = get_face(X2_vertr, faces)
    faces = get_face(X3_vertr, faces)

    comp = bw.Compound(faces)
    loc = Transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    with open(sandbox_fname('trans_test', 'brep'), "w") as f:
        bw.write_model(f, comp, loc)

    return


def get_face(frac, faces):
    v1 = bw.Vertex(frac[0, :])
    v2 = bw.Vertex(frac[1, :])
    v3 = bw.Vertex(frac[2, :])
    v4 = bw.Vertex(frac[3, :])
    e1 = bw.Edge(v1, v2)
    e2 = bw.Edge(v2, v3)
    e3 = bw.Edge(v3, v4)
    e4 = bw.Edge(v4, v1)
    f1 = bw.Face([e1, e2, e3, e4])
    faces.append(f1)

    return faces
