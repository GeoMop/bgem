
import numbers
import sys
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
    """
    Helper class to define new field factory functions.
    The type specification static methods: Field, Field, Number, Numbers
    are used to wrap parameters provided by user. These methods create the Par instance
    which contain a particular GMSH field settr function: _
    Usage:

    """

    @staticmethod
    def Field(x):
        """
        Make an argument for a parameter of the type Field parameter.
        Numerical values are wrapped as the 'constant' field.
        """
        x = Field.wrap(x)
        return Par(Par._set_field, x)

    @staticmethod
    def Fields(fields):
        """
        Make an argument for a parameter of the type List[Field].
        Numerical values are wrapped as the 'constant' field.
        """
        fields = [Field.wrap(f) for f in fields]
        return Par(Par._set_fields, fields)


    @staticmethod
    def Number(x):
        """
        Make an argument for a parameter of the type Number
        (i.e. any value convertible to double).
        """
        return Par(Par._set_number, float(x))

    @staticmethod
    def Numbers(x):
        """
        Make an argument for a parameter of the type List[Number]
        (Number is  any value convertible to double).
        """
        return Par(Par._set_numbers, [float(v) for v in x])

    @staticmethod
    def String(x):
        """
        Make an argument for a parameter of the type Number
        (i.e. any value convertible to double).
        """
        return Par(Par._set_string, x)


    @staticmethod
    def _set_string(field_id, parameter, x):
        """ Auxiliary setter method used in the Par.Number. """
        gmsh_field.setString(field_id, parameter, x)

    @staticmethod
    def _set_number(field_id, parameter, x):
        """ Auxiliary setter method used in the Par.Number. """
        gmsh_field.setNumber(field_id, parameter, x)

    @staticmethod
    def _set_numbers(field_id, parameter, x):
        """ Auxiliary setter method used in the Par.Number. """
        gmsh_field.setNumbers(field_id, parameter, x)

    @staticmethod
    def _set_field(field_id, parameter, field):
        """ Auxiliary setter method used in the Par.Field. """
        gmsh_field.setNumber(field_id, parameter, field.construct())

    @staticmethod
    def _set_field(field_id, parameter, field):
        """ Auxiliary setter method used in the Par.Field. """
        gmsh_field.setNumber(field_id, parameter, field.construct())

    @staticmethod
    def _set_fields(id, parameter, fields):
        """ Auxiliary setter method used in the Par.Fields. """
        gmsh_field.setNumbers(id, parameter, [f.construct() for f in fields])


    def __init__(self, setter, value):
        self._setter = setter
        """A method used to set the field argument. One of `Par._set_*`"""
        self._value = value
        """The argument value."""

    def _set_field_argument(self, field_id, parameter):
        """
        Set the `parameter` of the GMSH field with the `field_id` to
        the stored argument `self._value`.
        Skip the None values.
        """
        if self._value is not None:
            self._setter(field_id, parameter, self._value)

    def _reset_id(self):
        if self._setter == self._set_field and self._value is not None:
            self._value._reset_id()
        elif self._setter == self._set_fields:
            for f in self._value:
                f._reset_id()


class Field:
    """
    A scalar or tensor field used to specify mesh step distribution.
    The Field object represents the field but particular application to the gmsh model is done
    through GmshOCC.set_mesh_step_field(field).

    TODO: implement GmshOCC.set_boundary_layer_field(field)
    """

    # Special IDs marking state of the field.
    unset = -2
    unfinished = -1


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
        self._args = kwargs
        self._id = Field.unset
        """ 
        Field.unset : uninitialized field
        Field.unfinished : GMSH field created but arguments not set yet (for detection of cycles)
        >0   : GMSH field id 
        """

    def reset_id(self):
        """Unset whole field DAG to (-2), assuming all _ids > 0."""
        if self._id == Field.unset:
            return

        assert self._id != Field.unfinished, "Cyclic field dependency."
        self._id = Field.unfinished
        for param, arg in self._args.items():
            arg._reset_id()
        self._id = None

    def construct(self):
        """Create the GMSH field with all arguments, return its ID."""
        if self._id > 0:
            return self._id

        assert self._id != Field.unfinished, "Cyclic field dependency."
        self._id = Field.unfinished
        field_id = gmsh_field.add(self._gmsh_field)
        for parameter, arg in self._args.items():
            arg._set_field_argument(field_id, parameter)
        self._id = field_id
        return self._id

    def _expand(self):
        return f"F{self.construct()}"

#     """
#     Define operators.
#     """
#
    def __add__(self, other):
        return FieldExpr("{0}+{1}", [self, other])

    def __radd__(self, other):
        return FieldExpr("{0}+{1}", [other, self])

    def __sub__(self, other):
        return FieldExpr("{0}-{1}", [self, other])

    def __rsub__(self, other):
        return FieldExpr("{0}-{1}", [other, self])

    def __mul__(self, other):
        return FieldExpr("{0}*{1}", [self, other])

    def __rmul__(self, other):
        return FieldExpr("{0}*{1}", [other, self])

    def __truediv__(self, other):
        return FieldExpr("{0}/{1}", [self, other])

    def __rtruediv__(self, other):
        return FieldExpr("{0}/{1}", [other, self])

    def __mod__(self, other):
        return FieldExpr("{0}%{1}", [self, other])

    def __rmod__(self, other):
        return FieldExpr("{0}%{1}", [other, self])

    def __pow__(self, power):
        return FieldExpr("{0}^{1}", [self, power])

    def __rpow__(self, base):
        return FieldExpr("{0}^{1}", [base, self])

    def __neg__(self):
        return FieldExpr("-{0}", [self])

    def __pos__(self):
        return FieldExpr("+{0}", [self])

    def __lt__(self, other):
        return FieldExpr("{0}<{1}", [self, other])

    def __gt__(self, other):
        return FieldExpr("{0}>{1}", [self, other])

    def __le__(self, other):
        return FieldExpr("{0}<={1}", [self, other])

    def __ge__(self, other):
        return FieldExpr("{0}=>{1}", [self, other])

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

"""
Field functions.
"""

functions = [
"abs",
"acos",
"asin",
"atan",
"cos",
"cosh",
"deg",
"exp",
"log",
"log10",
"pow10",
"rad",
"round",
"sign",
"sin",
"sinh",
"sqrt",
"step",
"tan",
"tanh",
"atanh",
"trunc",
"floor",
"ceil"]


current_module = sys.modules[__name__]
for fn in functions:
    body = lambda x, f=fn: FieldExpr(f"{f}({{0}})", [x])
    setattr(current_module, fn, body)

variadic_functions = [
    ('min', 'min'),
    ('max', 'max'),
    ('sum', 'sum'),
    ('avg', 'med')
]

for pyfn, gmsh_fn in variadic_functions:
    body = lambda *x, f=gmsh_fn: FieldExpr(f"{f}{{variadic}}", x)
    setattr(current_module, pyfn, body)





class FieldExpr(Field):
    def __init__(self, expr_fmt, inputs):
        self._expr = expr_fmt
        self._inputs = [Field.wrap(f) for f in inputs]

    def _expand(self):
        return self._make_math_eval_expr()

    def _make_math_eval_expr(self):
        input_exprs = [in_field._expand() for in_field in self._inputs]
        variadic = "(" + ",".join(input_exprs) + ")"
        return self._expr.format(*input_exprs, variadic=variadic)
        #     [in_field._make_math_eval_expr() for in_field in self._inputs]
        # )
        # self._expr.format()

    def reset_id(self):
        """Unset whole field DAG to (-2), assuming all _ids > 0."""
        for f_in in self._inputs:
            f_in.reset_id()


    def construct(self):
        expr = self._make_math_eval_expr()
        print("MathEval expr: ", expr)
        field = Field("MathEval",
                      F=Par.String(expr))
        return field.construct()

        # substitute all FieldExpr and form the expression

        # creata MathEval

"""
Predefined coordinate fields.
Usage:
    field = field.sin(field.x) * field.cos(field.y)
"""
x = FieldExpr("x",[])
y = FieldExpr("y",[])
z = FieldExpr("z",[])

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
                 VIn = Par.Number(v_in),
                 VOut = Par.Number(v_out),
                 XMax = Par.Number(pt_max[0]),
                 XMin = Par.Number(pt_min[0]),
                 YMax = Par.Number(pt_max[1]),
                 YMin = Par.Number(pt_min[1]),
                 ZMax = Par.Number(pt_max[2]),
                 ZMin = Par.Number(pt_min[2]))



def constant(value):
    """
    Make a field with constant value = value.
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


#MathEvalAniso


def maximum(*fields: Field) -> Field:
    """
    Field that is maximum of other fields.
    :param fields: variadic list of fields
    Automatically wrap constants as a constant field.
    """
    return Field('Max', FieldsList=Par.Fields(fields))


def minimum(*fields: Field) -> Field:
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




def threshold(field:Field, lower_bound:Tuple[float, float],
              upper_bound:Tuple[float, float]=None, sigmoid:bool=False) -> Field:
    """
    Apply a threshold function to the 'field'.

    field_min, threshold_min = lower_bound
    field_max, threshold_max = lower_bound

    Threshold function:
    threshold = threshold_min IF field <= field_min
    threshold = threshold_max IF field >= field_max
    interpolation otherwise.

    upper_bound = None is equivalent to field_max = infinity

    For sigmoid = True, use sigmoid as a smooth approximation of the threshold function.

    TODO: not clear why field_min and field_max are necessary, if it can make transition discontinuous.
    """
    field_min, threshold_min = lower_bound
    if upper_bound is None:
        field_max, threshold_max = (None, None)
        stop_at_dist_max = True
    else:
        field_max, threshold_max = upper_bound
        stop_at_dist_max = False

    return Field('Threshold',
                 IField=Par.Field(field),
                 DistMin=Par.Number(field_min),
                 LcMin=Par.Number(threshold_min),
                 DistMax=Par.Number(field_max),
                 LcMax=Par.Number(threshold_max),
                 StopAtDistMax=Par.Number(stop_at_dist_max),
                 Sigmoid=Par.Number(sigmoid))









