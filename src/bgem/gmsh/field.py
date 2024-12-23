import numbers
import sys
import math
from typing import *
import numpy as np
from gmsh import model as gmsh_model
from bgem.gmsh.gmsh import ObjectSet

gmsh_field = gmsh_model.mesh.field


class Par:
    """
    Helper class to define a new field factory functions.
    The type specification static methods: Field, Fields, Number, Numbers, String
    are used to wrap parameters provided by user. These methods create the Par instance
    which contain a particular GMSH field setter function: _
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
        Make an argument for a parameter of the type String.
        (i.e. any value convertible to double). Your mesh must contain at least points.
        """
        return Par(Par._set_string, x)

    @staticmethod
    def _set_field(model, field_id, parameter, field):
        """ Auxiliary setter method used in the Par.Field. """
        gmsh_field.setNumber(field_id, parameter, field.construct(model))

    @staticmethod
    def _set_fields(model, field_id, parameter, fields):
        """ Auxiliary setter method used in the Par.Fields. """
        gmsh_field.setNumbers(field_id, parameter, [f.construct(model) for f in fields])

    @staticmethod
    def _set_number(model, field_id, parameter, x):
        """ Auxiliary setter method used in the Par.Number. """
        gmsh_field.setNumber(field_id, parameter, x)

    @staticmethod
    def _set_numbers(model, field_id, parameter, x):
        """ Auxiliary setter method used in the Par.Numbers. """
        gmsh_field.setNumbers(field_id, parameter, x)

    @staticmethod
    def _set_string(model, field_id, parameter, x):
        """ Auxiliary setter method used in the Par.String. """
        gmsh_field.setString(field_id, parameter, x)

    def __init__(self, setter, value):
        self._setter = setter
        """A method used to set the field argument. One of `Par._set_*`"""
        self._value = value
        """The argument value."""

    def _set_field_argument(self, model, field_id, parameter):
        """
        Set the `parameter` of the GMSH field with the `field_id` to
        the stored argument `self._value`.
        Skip the None values.
        """
        if self._value is not None:
            self._setter(model, field_id, parameter, self._value)

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

    def __init__(self, gmsh_field: str, **kwargs):
        """
        Introduce a new gmsh field instance.
        :param gmsh_field: gmsh field kind

        Following rules simplify definition of the interface functions:
        :param kwargs:
        - keys are gmsh field options
        - values are instances of Par, with appropriate setter function and value to set
        - Par.Number is applied automatically
        """
        self._id = Field.unset
        self._gmsh_model_id = id(None)
        self._gmsh_field = gmsh_field
        self._args = kwargs
        """ 
        Field.unset : uninitialized field
        Field.unfinished : GMSH field created but arguments not set yet (for detection of cycles)
        >0   : GMSH field id 
        """

    def is_constructed(self, model):
        return self._gmsh_model_id == id(model) and self._id >= 0

    def reset_id(self):
        """Unset whole field DAG to (-2), assuming all _ids > 0."""
        if self._id == Field.unset:
            return

        assert self._id != Field.unfinished, "Cyclic field dependency."
        self._id = Field.unfinished
        for param, arg in self._args.items():
            arg._reset_id()
        self._id = None

    def construct(self, model):
        """
        Create the GMSH field with all arguments, return its ID.
        :param model: bgem.gmsh.gmsh.GeometryOCC
        """
        if self.is_constructed(model):
            return self._id
        self._gmsh_model_id = id(model)

        assert self._id != Field.unfinished, "Cyclic field dependency."
        self._id = Field.unfinished
        field_id = gmsh_field.add(self._gmsh_field)
        for parameter, arg in self._args.items():
            arg._set_field_argument(model, field_id, parameter)
        self._id = field_id
        print("construct Field: ", self._gmsh_field, self._id)
        return self._id

    def _expand(self, model):
        return f"F{self.construct(model)}"

    #     """
    #     Define operators.
    #     """
    #
    def __add__(self, other):
        return FieldExpr("({0}+{1})", [self, other])

    def __radd__(self, other):
        return FieldExpr("({0}+{1})", [other, self])

    def __sub__(self, other):
        return FieldExpr("({0}-{1})", [self, other])

    def __rsub__(self, other):
        return FieldExpr("({0}-{1})", [other, self])

    def __mul__(self, other):
        return FieldExpr("({0}*{1})", [self, other])

    def __rmul__(self, other):
        return FieldExpr("({0}*{1})", [other, self])

    def __truediv__(self, other):
        return FieldExpr("({0}/{1})", [self, other])

    def __rtruediv__(self, other):
        return FieldExpr("({0}/{1})", [other, self])

    def __mod__(self, other):
        return FieldExpr("({0}%{1})", [self, other])

    def __rmod__(self, other):
        return FieldExpr("({0}%{1})", [other, self])

    def __pow__(self, power):
        return FieldExpr("({0}^{1})", [self, power])

    def __rpow__(self, base):
        return FieldExpr("({0}^{1})", [base, self])

    def __neg__(self):
        return FieldExpr("(-{0})", [self])

    def __pos__(self):
        return FieldExpr("(+{0})", [self])

    def __lt__(self, other):
        return FieldExpr("(1-step({0}-{1}))", [self, other])

    def __gt__(self, other):
        return FieldExpr("(1-step({1}-{0}))", [self, other])

    def __le__(self, other):
        return FieldExpr("step({1}-{0})", [self, other])

    def __ge__(self, other):
        return FieldExpr("step({0}-{1})", [self, other])


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
        self._id = Field.unset
        self._gmsh_model_id = id(None)
        self._expr = expr_fmt
        self._inputs = [Field.wrap(f) for f in inputs]

    def _expand(self, model):
        return self._make_math_eval_expr(model)

    def _make_math_eval_expr(self, model):
        input_exprs = [in_field._expand(model) for in_field in self._inputs]
        variadic = "(" + ",".join(input_exprs) + ")"
        eval_expr = self._expr.format(*input_exprs, variadic=variadic)

        # print("expr: ", eval_expr)
        return eval_expr

    def reset_id(self):
        """Unset whole field DAG to (-2), assuming all _ids > 0."""
        for f_in in self._inputs:
            f_in.reset_id()

    def construct(self, model):
        if self.is_constructed(model):
            return self._id
        self._gmsh_model_id = id(model)

        expr = self._make_math_eval_expr(model)
        print("construct MathEval expr: ", expr)
        field = Field("MathEval",
                      F=Par.String(expr))
        return field.construct(model)

        # substitute all FieldExpr and form the expression

        # create MathEval


"""
Predefined coordinate fields.
Usage:
    field = field.sin(field.x) * field.cos(field.y)
"""
x = FieldExpr("x", [])
y = FieldExpr("y", [])
z = FieldExpr("z", [])


def attractor_aniso_curve(curves: 'ObjectSet', dist_range=(0.1, 0.5), h_normal=(0.05, 0.5), h_tangent=(0.5, 0.5),
                          sampling=20):
    """
    curves : ObjectSet, its 1d tags are passed to the field.
    dist_range : (float, float), below and above this range the minimum mesh size and maximum mesh size is applied respectively
    sampling : Number of sampling points on each curve
    h_normal : (float, float), Normal direction mesh step (h_min, h_max) out of the distance range.
    h_tangent : (float, float), Tangential direction mesh step (h_min, h_max) out of the distance range.
    Requires: Mesh.Algorithm = Algorithm.BAMG; // BAMG = 7 in 2D
              Mesh.Algorithm3D = Algorithm3D.MMG3D; // MMG3D = 7 in 3D
    TODO: force automatically
    """
    print("Warning: Anisotropic mesh size fields requires Mesh.Algorithm = 7; // BAMG.")
    return Field('AttractorAnisoCurve',
                 CurvesList=Par.Numbers(curves.split_by_dimension()[1].tags),
                 DistMax=Par.Number(dist_range[1]),
                 DistMin=Par.Number(dist_range[0]),
                 Sampling=Par.Number(sampling),
                 SizeMaxNormal=Par.Number(h_normal[1]),
                 SizeMaxTangent=Par.Number(h_tangent[1]),
                 SizeMinNormal=Par.Number(h_normal[0]),
                 SizeMinTangent=Par.Number(h_tangent[0]))


# @field_function
# AutomaticMeshSizeField

# @field_function
# Ball

# @field_function
# BoundaryLayer


Point = Union[Tuple[float, float, float], List[float], np.array]


def box(pt_min: Point, pt_max: Point, v_in: float, v_out: float = 1e300) -> Field:
    """
    The value of this field is VIn inside the box, VOut outside the box. The box is given by

    pt_a <= (x, y, z) <= pt_b

    Can be used instead of a constant field.
    """
    return Field('Box',
                 VIn=Par.Number(v_in),
                 VOut=Par.Number(v_out),
                 XMax=Par.Number(pt_max[0]),
                 XMin=Par.Number(pt_min[0]),
                 YMax=Par.Number(pt_max[1]),
                 YMin=Par.Number(pt_min[1]),
                 ZMax=Par.Number(pt_max[2]),
                 ZMin=Par.Number(pt_min[2]))


def constant(value):
    """
    Make a field with constant value = value.
    """
    return box((-1, -1, -1), (1, 1, 1), v_in=value, v_out=value)


# @field_function
# Curvature

# @field_function
# Cylinder

def distance(entity_group: Union["ObjectSet", List["ObjectSet"]], sampling=20):
    """
     Distance from a set of geometric entities 'entity_group'. Curves and surfaces are internally replaced by the sets of nodes
     of size 'sampling' per entity. The field is evaluated with respect to these points.
     Approximately linear complexity with respect to the total number of sampling points.
     This is quite limiting as for detailed mesh we need lot of points, which significantly increase the meshing time.
     TODO:
     - scale the sampling by the entity size (e.g. given by a sort of bounding box)
     - modify GMSH to use bounding boxes tree to get log(N) complexity
    """
    entity_group = ObjectSet.group(entity_group)
    dim_entities = [[], [], []]
    for dim, tag in entity_group.dim_tags:
        dim_entities[dim].append(tag)
    return _distance_field(dim_entities, sampling)


def _distance_field(dim_entities, sampling):
    return Field('Distance',
                 PointsList=Par.Numbers(dim_entities[0]),
                 CurvesList=Par.Numbers(dim_entities[1]),
                 SurfacesList=Par.Numbers(dim_entities[2]),
                 Sampling=Par.Number(sampling),
                 )


def distance_nodes(nodes: List[int]) -> Field:
    """
     Distance from a set of nodes given by their tags.
    """
    # nodes = [(0, tag) for tag in nodes]
    return _distance_field([nodes, [], []], 0)


def distance_edges(curves, sampling=20) -> Field:
    """
    Distance from a set of curves given by their tags. Curves are internally replaced by a set of nodes
    of size 'sampling' and DistanceNodes is applied.
    """
    # curves = [(1, tag) for tag in curves]
    return _distance_field([[], curves, []], sampling)


def distance_surfaces(surfaces, sampling=20) -> Field:
    """
     Distance from a set of surfaces given by their tags. Surfaces are internally replaced by a set of  nodes
     of size 'sampling' and DistanceNodes is applied.
    """
    # surfaces = [(2, tag) for tag in surfaces]
    return _distance_field([[], [], surfaces], sampling)


# @field_function
# def Frustum

# @field_function
# Gradient

# @field_function
# IntersectAniso

# Laplacian
# LonLat


# MathEvalAniso


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


def geometric(field: Field, a: Tuple[float, float], b: Tuple[float, float]) -> Field:
    """
    Geometric interpolation between points a = (x0, y0) and b = (x1, y1).
    """
    x = field
    x0, y0 = a
    x1, y1 = b
    l0, l1 = math.log(y0), math.log(y1)
    a = (l1 - l0) / (x1 - x0)
    x_avg = (l1 * x0 - l0 * x1) / (l1 - l0)
    return current_module.exp(a * (x - x_avg))


def linear(field: Field, a: Tuple[float, float], b: Tuple[float, float]) -> Field:
    """
    Linear transformation of the scalar field given by two points.
    Prescribed mapping a[0] to a[1], and b[0] to b[1].
    """
    x = field
    x0, y0 = a
    x1, y1 = b
    a = (y1 - y0) / (x1 - x0)
    x_avg = (y1 * x0 - y0 * x1) / (y1 - y0)
    return a * (x - x_avg)


def polynomial(field: Field, a: Tuple[float, float], b: Tuple[float, float], q: float = 1) -> Field:
    """
    Power interpolation, given by two points as for linear, but
    apply power with exponent 'q'.
    """
    x = field
    x0, y0 = a
    x1, y1 = b
    l0, l1 = y0 ** (1 / q), y1 ** (1 / q)
    a = (l1 - l0) / (x1 - x0)
    x_avg = (l1 * x0 - l0 * x1) / (l1 - l0)
    return (a * (x - x_avg)) ** q


def threshold(field: Field, lower_bound: Tuple[float, float],
              upper_bound: Tuple[float, float] = None, sigmoid: bool = False) -> Field:
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


"MathEvalAniso"
