import numbers
from typing import *
import numpy as np
import gmsh

import inspect
gmsh_field = gmsh.model.mesh.field

"""
TODO:
1. make application of fields to gmsh mdel lazy in order to create fields independently of the model.
2. Field objects with overloader operators and functions to make natural expressions passed to MAtHEvla field.
"""




class Par:

    @staticmethod
    def Field(x):
        x = Field.wrap(x)
        return Par(Par._set_field, x)

    @staticmethod
    def Fields(fields):
        fields = [Field.wrap(f) for f in fields]
        return Par(Par._set_fields, fields)

    @staticmethod
    def Numbers(x):
        return Par(gmsh_field.setNumbers, x)

    @staticmethod
    def Number(x):
        if isinstance(x, Par):
            return x
        elif isinstance(x, numbers.Number):
            return Par(gmsh_field.setNumber, x)
        else:
            assert False, f"Wrong field argument: {x}"

    @staticmethod
    def _set_field(id, parameter, field):
        gmsh_field.setNumber(id, parameter, field.construct())

    @staticmethod
    def _set_fields(id, parameter, fields):
        gmsh_field.setNumbers(id, parameter, [f.construct() for f in fields])


    def __init__(self, setter, value):
        self._setter = setter
        self._value = value

    def set(self, field_id, parameter):
        if self._value is not None:
            self._setter(field_id, parameter, self._value)

    def reset_id(self):
        if self._setter == self._set_field and self._value is not None:
            self._value.reset_id()
        elif self._setter == self._set_fields:
            for f in self._value:
                f.reset_id()


class Field:
    """
    A scalar or tensor field used to specify mesh step distribution.
    The Field object represents the field but particular apprication to the gmsh model is done
    through GmshOCC.set_mesh_step_field(field).
    """

    @staticmethod
    def wrap(x):
        if isinstance(x, Field):
            return x
        elif isinstance(x, numbers.Number):
            return constant(x)
        elif x is None:
            return None
        else:
            assert False, f"Can not wrap {x} as a field."


    def __init__(self, gmsh_field:str, **kwargs):
        """
        Intorduce a new gmsh field instance.
        :param gmsh_field: gmsh field kind

        Following rules simplify definition of the interface functions:
        :param kwargs:
        - keys are gmsh field options
        - values are instances of Par, with appropriate setter function and value to set
        - Par.Number is applied automatically
        """
        self._gmsh_field = gmsh_field
        self._args = [ (k, Par.Number(v)) for k, v in kwargs.items()]
        self._id = None

    def reset_id(self):
        self._id = None
        for param, arg in self._args:
            arg.reset_id()

    def construct(self):
        assert self._id is None, "Cyclic field dependency."
        self._id = gmsh_field.add(self._gmsh_field)
        for parameter, arg in self._args:
            arg.set(self._id, parameter)
        return self._id


#     """
#     Define operators.
#     """
#
#
# class FieldExpr(Field):
#     def __init__(self, expr_fmt, inputs):
#         self._expr = expr_fmt
#         self._inputs = inputs
#
#     def evaluate(self):
#         # substitute all FieldExpr and form the expression
#
#         # creata MathEval


# @field_function
# def AttractorAnisoCurve

# @field_function
# AutomaticMeshSizeField

# @field_function
# Ball

# @field_function
# BoundaryLayer


Point = Union[Tuple[float, float, float], List[float], np.array]

def box(pt_min: Point, pt_max: Point, v_in:float, v_out:float=1e300) -> Field:
    """
    The value of this field is VIn inside the box, VOut outside the box. The box is given by

    pt_a <= (x, y, z) <= pt_b

    Can be used instead of a constant field.
    """
    return Field('Box',
                 VIn = v_in,
                 VOut = v_out,
                 XMax = pt_max[0],
                 XMin = pt_min[0],
                 YMax = pt_max[1],
                 YMin = pt_min[1],
                 ZMax = pt_max[2],
                 ZMin = pt_min[2])



def constant(value):
    """
    Make a field with constant value = value.
    Emulated using a box field with box containing shpere in origin with given 'readius'.

    TODO: automatic choice of radius.
    """
    return box((-1, -1, -1), (1, 1, 1), v_in=value, v_out=value)




#@field_function
#Curvature

#@field_function
#Cylinder

def distance_nodes(nodes:List[int], coordinate_fields:Tuple[Field,Field,Field]=(None, None, None)):
    """
    Distance from a set of 'nodes' given by their tags.
    Optional coordinate_fields = ( field_x, field_y, field_z),
    gives fields used as X,Y,Z coordinates (not clear how exactly these curved coordinates are used).
    """
    fx, fy, fz = coordinate_fields
    return Field('Distance',
                 NodesList=Par.Numbers(nodes),
                 FieldX=Par.Field(fx),
                 FieldY=Par.Field(fy),
                 FieldZ=Par.Field(fz))



# def distance_edges(curves, nodes_per_edge=8, coordinate_fields=None) -> Field:
#     """
#     Distance from a set of curves given by their tags. Curves are replaced by 'node_per_edge' nodes
#     and DistanceNodes is applied.
#     Optional coordinate_fields = ( field_x, field_y, field_z),
#     gives fields used as X,Y,Z coordinates (not clear how exactly these curved coordinates are used).
#     """
#     id = gmsh_field.add('Distance')
#     gmsh_field.setNumbers(id, "EdgesList", curves)
#     gmsh_field.setNumber(id, "NNodesByEdge", nodes_per_edge)
#     if coordinate_fields:
#         fx, fy, fz = coordinate_fields
#         gmsh_field.setNumber(id, "FieldX", fx)
#         gmsh_field.setNumber(id, "FieldY", fy)
#         gmsh_field.setNumber(id, "FieldZ", fz)
#     return id

# @field_function
# def distance_surfaces(curves, nodes_per_edge=8, coordinate_fields=None) -> Field:

# @field_function
# def Frustum

# @field_function
# Gradient

# @field_function
# IntersectAniso

# Laplacian
# LonLat

# @field_function
# def math_eval(expr) -> Field:
#     """
#     Temporary solution.
#
#     Usage:
#     dist = field.distance(nodes)
#     box = field.box(...)
#     formula = f'1/F{dist} + F{box}'
#     f = field.math_eval(formula)
#
#     TODO: Use Field to catch native Python operations, provide basic functions: in the field namespace:
#     GMSH use MathEx, which supports:
#     operators:
#     unary + -
#     binary + - * / ^ % < >
#
#     functions:
#            { "abs",     fabs },
#          { "acos",    acos },
#          { "asin",    asin },
#          { "atan",    atan },
#          { "cos",     cos },
#          { "cosh",    cosh },
#          { "deg",     deg },   // added
#          { "exp",     exp },
#          { "fac",     fac },   // added, round to int, factorial for 0..170,
#          { "log",     log },
#          { "log10",   log10 },
#          // { "pow10",   pow10 } // in future, add it?
#          { "rad",     rad },   // added
#          { "round",   round }, // added
#          { "sign",    sign },
#          { "sin",     sin },
#          { "sinh",    sinh },
#          // { "sqr",     sqr }, // added
#          { "sqrt",    sqrt },
#          { "step",    step }, // (x>=0), not necessary
#          { "tan",     tan },
#          { "tanh",    tanh },
# #if !defined(WIN32)
#          { "atanh",   atanh },
# #endif
#          { "trunc",   trunc }, // added
#          { "floor",   floor }, // largest integer not grather than x
#          { "ceil",    ceil }, // smallest integer not less than x
#
#          addfunc("rand", p_rand, 0); // rand()
#
#          addfunc("sum", p_sum, UNDEFARGS);  // sum(1,2,...)
#
#          addfunc("max", p_max, UNDEFARGS);
#
#          addfunc("min", p_min, UNDEFARGS);
#
#          addfunc("med", p_med, UNDEFARGS);  // average !!
#
#     """
#     id = gmsh_field.add('MathEval')
#     gmsh_field.setString(id, 'F', expr)
#     return id

#MathEvalAniso


def max(*fields: Field) -> Field:
    """
    Field that is maximum of other fields.
    :param fields: variadic list of fields
    Automatically wrap constants as a constant field.
    """
    return Field('Max', FieldsList=Par.Fields(fields))


def min(*fields: Field) -> Field:
    """
    Field that is minimum of other fields.
    :param fields: variadic list of fields
    Automatically wrap constants as a constant field.
    """
    return Field('Min', FieldsList=Par.Fields(fields))


# MaxEigenHessian

# Mean

# MinAniso

# Octree

# Param

# @field_function
# def restrict(field:Field, *object_sets, add_boundary=False):
#     """
#     Restrict the application of a 'field' to a given list of geometrical entities: points, curves, surfaces or volumes.
#     Entities are given as object sets.
#     """
#     if not object_sets:
#         return
#
#     factory = object_sets[0].factory
#     factory.synchronize()
#     group = factory.group(*object_sets)
#     if add_boundary:
#         b_group = group.get_boundary(combined=False)
#         group = factory.group(group, b_group)
#     points, edges, faces, volumes = group.split_by_dimension()
#     id = gmsh_field.add('Restrict')
#     gmsh_field.setNumber(id, "IField", field)
#     gmsh_field.setNumbers(id, "VerticesList", points.tags)
#     gmsh_field.setNumbers(id, "EdgesList", edges.tags)
#     gmsh_field.setNumbers(id, "FacesList", faces.tags)
#     gmsh_field.setNumbers(id, "RegionsList", volumes.tags)
#     return id

# Structured



# @field_function
# def threshold(field:Field, lower_bound:Tuple[float, float], upper_bound:Tuple[float, float]=None, sigmoid:bool=False) -> Field:
#     """
#     Apply a threshold function to the 'field'.
#
#     field_min, threshold_min = lower_bound
#     field_max, threshold_max = lower_bound
#
#     Threshold function:
#     threshold = threshold_min IF field <= field_min
#     threshold = threshold_max IF field >= field_max
#     interpolation otherwise.
#
#     upper_bound = None is equivalent to field_max = infinity
#
#     For sigmoid = True, use sigmoid as a smooth approximation of the threshold function.
#
#     TODO: not clear why field_min and field_max are necessary, if it can make transition discontinuous.
#     """
#     id = gmsh_field.add('Threshold')
#
#     gmsh_field.setNumber(id, "IField", field)
#     field_min, threshold_min = lower_bound
#     gmsh_field.setNumber(id, "DistMin", field_min)
#     gmsh_field.setNumber(id, "LcMin", threshold_min)
#     if upper_bound:
#         field_max, threshold_max = lower_bound
#         gmsh_field.setNumber(id, "DistMax", field_max)
#         gmsh_field.setNumber(id, "LcMax", threshold_max)
#     else:
#         gmsh_field.setNumber(id, "StopAtDistMax", True)
#
#     if sigmoid:
#         gmsh_field.setNumber(id, "Sigmoid", True)
#     return id









