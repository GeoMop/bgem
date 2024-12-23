import enum
import numpy as np
from typing import *
from . import bspline
from bgem import Transform, check_matrix, ParamError

'''
TODO:
- For Solid make auto conversion from Faces similar to Face from Edges
- For Solid make test that Shells are closed.
- Implement closed test for shells (similar to wire)
- Improve test of closed (Wire, Shell) to check also orientation of (Edges, Faces). ?? May be both if holes are allowed.
- Rename attributes and methods to more_words_are_separated_by_underscore.
- document public methods
- Plane and Line support
- support of implicit attachment of vertices and edges to surfaces (projection), 
  has to be done in Approx
- test various attachments
- test composed location of composed locations (possibly need to do expansion before output)

??? Jak fungují location v ShapeRef?
    Chceme to podporovat? Lze definovat treba podstavu a pak ji jen posunout pomoci location?
    Když to posunu, tak jak to může sedět s plochama?
    Asi by bylo bezpečnější to v našem rozhraní nepodporovat a nechat tam vždy Identity.
'''


class BREPGroup(enum.IntEnum):
    locations = 0
    curves_3d = 1
    curves_2d = 2
    surfaces = 3
    shapes = 4


class BREPObject:
    """
    Basic class of the BREP objects, define common methods necessary for the
    file output. Objects form a tree (or possibly DAG) and can be processed
    be a graph search without maintaining global structures.
    """

    def __init__(self, group: BREPGroup, id_in_postvisit=True) -> None:
        self._brep_group: BREPGroup = group
        self._brep_id: Optional[int] = None

        # Set ID and append to the BREP group in either DFS previsit or DFS postvisit.
        if id_in_postvisit:
            self._dfs_previsit = self._group_pass
            self._dfs_postvisit = self._group_append
        else:
            self._dfs_previsit = self._group_append
            self._dfs_postvisit = self._group_pass

    @property
    def brep_id(self):
        assert self._brep_id is not None, str(self)
        # if self._brep_id is  None:
        #    print("    None ID:", str(self))
        return self._brep_id

    def children(self, recursive=False, of_type=None, _visited=None):
        """
        Generator of the child BREPObjets for the DFS.
        :param of_type: one or tuple of BREPObject subclasses to generate
        """
        if _visited is None:
            _visited = set()

        if of_type is None:
            of_type = BREPObject

        if id(self) not in _visited:
            _visited.add(id(self))
        if isinstance(self, of_type):
            yield self
        for ch in self._children_impl():
            for cc in ch.children(recursive, of_type, _visited):
                yield cc

    def _children_impl(self):
        """
        Generator of the child BREPObjets for the DFS.
        Default no children.
        """
        return []

    @staticmethod
    def gather_groups(objs):
        # DFS thorough the BREP object fromm the `self` as a root.
        # Assign BREP IDs and collect BREP objects into groups.
        visited = set()
        group_size = max(BREPGroup) + 1
        groups = [[] for _ in range(group_size)]
        for obj in objs:
            obj._dfs_gather_groups(groups, visited)
        return groups

    def _dfs_gather_groups(self, groups: List[List['BREPObject']], visited: Set[int]):
        # DFS recursive function.
        if id(self) in visited:
            return
        # print(f"visited: {self}")
        visited.add(id(self))

        self._dfs_previsit(groups)
        for ch in self._children_impl():
            assert isinstance(ch, BREPObject), f"{type(self)}._children_impl produced child: {type(ch)}"
            ch._dfs_gather_groups(groups, visited)
        self._dfs_postvisit(groups)

    def _dfs_finish(self, visited: Set[int] = None):
        # DFS recursive function.
        if visited is None:
            visited = set()
        if id(self) in visited:
            return
        visited.add(id(self))
        for ch in self._children_impl():
            assert isinstance(ch, BREPObject), f"{type(self)}._children_impl produced child: {type(ch)}"
            ch._dfs_finish(visited)

    def _group_append(self, groups):
        # print(f"append: {id(self):x} {self} ")
        group = groups[self._brep_group]
        group.append(self)
        self._brep_id = len(group)

    def _group_pass(self, groups):
        pass


LocationPower = Tuple[Transform, int]


class Location(BREPObject):
    """
    Location defines an affine transformation in 3D space. Corresponds to the <location data 1> in the BREP file.
    BREP format allows to use different transformations for individual shapes.
    Location are numbered from 1. Zero index means identity location.

    BGEM provides construction of location from bgem.Transform.

    TODO:
    - unfortunately we can not do location compression without _used_location references.
    - Possibly we can have a temporary wrapper around Transform and create Location and CompositeLocation,
      just during write_model using Wrapper as a proxy to final Locations and their brep_ids.
    """

    @staticmethod
    def make(transform: Transform) -> Union['Location', 'CompositeLocation']:
        assert isinstance(transform, Transform)
        if transform.is_composed():
            return CompositeLocation(transform)
        elif transform.is_identity():
            return _IdentityLocation.instance()
        else:
            return Location(transform)

    def __init__(self, transform: Transform):
        super().__init__(group=BREPGroup.locations)
        self.transform = transform
        self._used_location = self

    def __call__(self, location):
        return Location.make(self.transform @ location.transform)

    def compress(self, transform_map):
        self._used_location = transform_map.setdefault(id(self.transform), self)

    @property
    def brep_id(self):
        p = self
        while p._used_location is not p:
            p = p._used_location
        return p._used_location._brep_id

    def _brep_output(self, stream):
        assert self.transform.matrix is not None
        stream.write("1\n")
        for row in self.transform.matrix:
            for number in row:
                stream.write(" {}".format(number))
            stream.write("\n")


Identity = Transform()


class _IdentityLocation(Location):

    @classmethod
    def instance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = _IdentityLocation(Identity)
        return cls._instance

    def _brep_output(self, stream):
        pass


class CompositeLocation(Location):
    @staticmethod
    def expand_transform(t: Transform, power=1) -> List[LocationPower]:
        """
        Return composition list: List[]
        """
        composition = []
        if t.is_composed():
            for t, p in reversed(t._composition):
                composition.extend(CompositeLocation.expand_transform(t, power * p))
        elif not t.is_identity():
            composition.append((t, power))
        return composition

    def __init__(self, transform):
        super(Location, self).__init__(group=BREPGroup.locations)
        self.transform = transform
        self.composition = self.expand_transform(transform)
        self._used_location = self

    def compress(self, transform_map):
        """
        Reuse simple location objects for same instances of simple Transforms.
        """

        def get_basic_loc(t):
            if t.is_identity():
                return Identity
            else:
                try:
                    return transform_map[id(t)]
                except KeyError:
                    new_loc = transform_map[id(t)] = Location(t)
                    return new_loc

        self.composition = [(get_basic_loc(t), p) for t, p in self.composition]

    def _brep_output(self, stream):
        stream.write("2 ")
        for loc, pow in self.composition:
            stream.write("{} {} ".format(loc.brep_id, pow))
        stream.write("0\n")


def check_knots(deg, knots, N):
    total_multiplicity = 0
    for knot, mult in knots:
        # This condition must hold if we assume only (0,1) interval of curve or surface parameters.
        # assert float(knot) >= 0.0 and float(knot) <= 1.0
        total_multiplicity += mult
    assert total_multiplicity == deg + N + 1


# TODO: perform explicit conversion to np.float64 in order to avoid problems on different arch
# should be unified in bspline as well, convert  to np.arrays as soon as possible
scalar_types = (int, float, np.int32, np.int64, np.float32, np.float64)


def curve_from_bs(curve):
    """
    Make BREP writer Curve (2d or 3d) from bspline curve.
    :param curve: bs.Curve object
    :return:
    """
    dim = curve.dim
    if dim == 2:
        curve_dim = Curve2D
    elif dim == 3:
        curve_dim = Curve3D
    else:
        assert False
    c = curve_dim(curve.poles, curve.basis.pack_knots(), curve.rational, curve.basis.degree)
    c._bs_curve = curve
    return c


class Line3D(BREPObject):
    def _brep_output(self, stream):
        """
        Output of BREP format: 3d curve : Line record
        <3D curve record 1> = "1" <_> <3D point> <_> <3D direction> <_\n>;
        Parametric format.
        """
        # writes plane surface
        stream.write(f"1 {self.origin} {self.direction}\n")


class Curve3D(BREPObject):
    """
    Defines a 3D curve as B-spline. We shall work only with B-splines of degree 2.
    Corresponds to "B-spline Curve - <3D curve record 7>" from BREP format description.
    """

    def __init__(self, poles, knots, rational=False, degree=2):
        """
        Construct a B-spline in 3d space.
        :param poles: List of poles (control points) ( X, Y, Z ) or weighted points (X,Y,Z, w). X,Y,Z,w are floats.
                      Weighted points are used only for rational B-splines (i.e. nurbs)
        :param knots: List of tuples (knot, multiplicity), where knot is float, t-parameter on the curve of the knot
                      and multiplicity is positive int. Total number of knots, i.e. sum of their multiplicities, must be
                      degree + N + 1, where N is number of poles.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles.
        :param degree: Positive int.
        """

        if rational:
            check_matrix(poles, [None, 4], scalar_types)
        else:
            check_matrix(poles, [None, 3], scalar_types)
        N = len(poles)
        check_knots(degree, knots, N)

        super().__init__(group=BREPGroup.curves_3d)
        self.poles = poles
        self.knots = knots
        self.rational = rational
        self.degree = degree

    def _eval_check(self, t, point):
        if hasattr(self, '_bs_curve'):
            repr_pt = self._bs_curve.eval(t)
            if not np.allclose(np.array(point), repr_pt, rtol=1.0e-3):
                raise Exception("Point: {} far from curve repr: {}".format(point, repr_pt))

    def _brep_output(self, stream):
        # writes b-spline curve
        stream.write("7 {} 0  {} {} {} ".format(int(self.rational), self.degree, len(self.poles), len(self.knots)))
        for pole in self.poles:
            for value in pole:
                stream.write(" {}".format(value))
            stream.write(" ")
        for knot in self.knots:
            for value in knot:
                stream.write(" {}".format(value))
            stream.write(" ")
        stream.write("\n")


class Line2D(BREPObject):
    def _brep_output(self, stream):
        """
        Output of BREP format: 3d curve : Line record
        <2D curve record 1> = "1" <_> <2D point> <_> <2D direction> <_\n>;
        Parametric format.
        """
        # writes plane surface
        stream.write(f"1 {self.origin} {self.direction}\n")


class Curve2D(BREPObject):
    """
    Defines a 2D curve as B-spline. We shall work only with B-splines of degree 2.
    Corresponds to "B-spline Curve - <2D curve record 7>" from BREP format description.
    """

    def __init__(self, poles, knots, rational=False, degree=2):
        """
        Construct a B-spline in 2d space.
        :param poles: List of points ( X, Y ) or weighted points (X,Y, w). X,Y,w are floats.
                      Weighted points are used only for rational B-splines (i.e. nurbs)
        :param knots: List of tuples (knot, multiplicity), where knot is float, t-parameter on the curve of the knot
                      and multiplicity is positive int. Total number of knots, i.e. sum of their multiplicities, must be
                      degree + N + 1, where N is number of poles.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles.
        :param degree: Positive int.
        """

        N = len(poles)
        if rational:
            check_matrix(poles, [N, 3], scalar_types)
        else:
            check_matrix(poles, [N, 2], scalar_types)
        check_knots(degree, knots, N)

        super().__init__(group=BREPGroup.curves_2d)

        self.poles = poles
        self.knots = knots
        self.rational = rational
        self.degree = degree

    def _eval_check(self, t, surface, point):
        if hasattr(self, '_bs_curve'):
            u, v = self._bs_curve.eval(t)
            surface._eval_check(u, v, point)

    def _brep_output(self, stream):
        # writes b-spline curve
        stream.write("7 {} 0  {} {} {} ".format(int(self.rational), self.degree, len(self.poles), len(self.knots)))
        for pole in self.poles:
            for value in pole:
                stream.write(" {}".format(value))
            stream.write(" ")
        for knot in self.knots:
            for value in knot:
                stream.write(" {}".format(value))
            stream.write(" ")
        stream.write("\n")


def surface_from_bs(surf):
    """
    Make BREP writer Surface from bspline surface.
    :param surf: bs.Surface object
    :return:
    """
    s = Surface(surf.poles, (surf.u_basis.pack_knots(), surf.v_basis.pack_knots()),
                (surf.u_basis.degree, surf.v_basis.degree), surf.rational)
    s._bs_surface = surf
    return s


class Plane(BREPObject):

    def _brep_output(self, stream):
        """
        Output of BREP format: Plane surface record
        <surface record 1> = "1" <_> <3D point> (<_> <3D direction>) ^ 3 <_\n>;
        Three orthogonal direction vectors: normal, U-vector, V-vector.
        """
        # writes plane surface
        stream.write(f"1 {self.origin} {self.normal} {self.u_vector} {self.v_vector}\n")


class Surface(BREPObject):
    """
    Defines a B-spline surface in 3d space. We shall work only with B-splines of degree 2.
    Corresponds to "B-spline Surface - < surface record 9 >" from BREP format description.
    """

    def __init__(self, poles, knots, degree=(2, 2), rational=False):
        """
        Construct a B-spline in 3d space.
        :param poles: Matrix (list of lists) of Nu times Nv poles (control points).
                      Single pole is a points ( X, Y, Z ) or weighted point (X,Y,Z, w). X,Y,Z,w are floats.
                      Weighted points are used only for rational B-splines (i.e. nurbs)
        :param knots: Tuple (u_knots, v_knots). Both u_knots and v_knots are lists of tuples
                      (knot, multiplicity), where knot is float, t-parameter on the curve of the knot
                      and multiplicity is positive int. For both U and V knot vector the total number of knots,
                      i.e. sum of their multiplicities, must be degree + N + 1, where N is number of poles.
        :param degree: (u_degree, v_degree) Both positive ints.
        :param rational: True for rational B-spline, i.e. NURB. Use weighted poles. BREP format have two independent flags
                      for U and V parameter, but only choices 0,0 and 1,1 have sense.
        """

        if rational:
            check_matrix(poles, [None, None, 4], scalar_types)
        else:
            check_matrix(poles, [None, None, 3], scalar_types)

        assert len(poles) > 0
        assert len(poles[0]) > 0
        self.Nu = len(poles)
        self.Nv = len(poles[0])
        for row in poles:
            assert len(row) == self.Nv

        assert (not rational and len(poles[0][0]) == 3) or (rational and len(poles[0][0]) == 4)

        (u_knots, v_knots) = knots
        check_knots(degree[0], u_knots, self.Nu)
        check_knots(degree[1], v_knots, self.Nv)

        super().__init__(group=BREPGroup.surfaces)
        self.poles = poles
        self.knots = knots
        self.rational = rational
        self.degree = degree

    def _eval_check(self, u, v, point):
        if hasattr(self, '_bs_surface'):
            repr_pt = self._bs_surface.eval(u, v)
            if not np.allclose(np.array(point), repr_pt, rtol=1.0e-3):
                raise Exception("Point: {} far from curve repr: {}".format(point, repr_pt))

    def _brep_output(self, stream):
        # writes b-spline surface
        stream.write("9 {} {} 0 0 ".format(int(self.rational),
                                           int(self.rational)))  # prints B-spline surface u or v rational flag - both same
        for i in self.degree:  # prints <B-spline surface u degree> <_>  <B-spline surface v degree>
            stream.write(" {}".format(i))
        (u_knots, v_knots) = self.knots
        stream.write(" {} {}  {} {} ".format(self.Nu, self.Nv, len(u_knots), len(v_knots)))
        # prints  <B-spline surface u pole count> <_> <B-spline surface v pole count> <_> <B-spline surface u multiplicity knot count> <_>  <B-spline surface v multiplicity knot count> <B-spline surface v multiplicity knot count>
        #        stream.write(" {}".format(self.poles)) #TODO: tohle smaz, koukam na format poles a chci: B-spline surface weight poles
        for pole in self.poles:  # TODO: check, takovy pokus o poles
            for vector in pole:
                for value in vector:
                    stream.write(" {}".format(value))
                stream.write(" ")
            stream.write(" ")
        for knot in u_knots:  # prints B-spline surface u multiplicity knots
            for value in knot:
                stream.write(" {}".format(value))
            stream.write(" ")
        for knot in v_knots:  # prints B-spline surface v multiplicity knots
            for value in knot:
                stream.write(" {}".format(value))
            stream.write(" ")
        stream.write("\n")


class Approx:
    """
    Approximation methods for B/splines of degree 2.

    """

    @classmethod
    def plane(cls, vtxs):
        """
        Returns B-spline surface of a plane given by 3 points.
        We return also list of UV coordinates of the given points.
        :param vtxs: List of tuples (X,Y,Z)
        :return: ( Surface, vtxs_uv )
        """
        assert len(vtxs) == 3, "n vtx: {}".format(len(vtxs))
        vtxs.append((0, 0, 0))
        vtxs = np.array(vtxs)
        vv = vtxs[1] + vtxs[2] - vtxs[0]
        vtx4 = [vtxs[0], vtxs[1], vv, vtxs[2]]
        (surf, vtxs_uv) = cls.bilinear(vtx4)
        return (surf, [vtxs_uv[0], vtxs_uv[1], vtxs_uv[3]])

    @classmethod
    def bilinear(cls, vtxs):
        """
        Returns B-spline surface of a bilinear surface given by 4 corner points.
        We retun also list of UV coordinates of the given points.
        :param vtxs: List of tuples (X,Y,Z)
        :return: ( Surface, vtxs_uv )
        """
        assert len(vtxs) == 4, "n vtx: {}".format(len(vtxs))
        vtxs = np.array(vtxs)

        def mid(*idx):
            return np.mean(vtxs[list(idx)], axis=0)

        # v - direction v0 -> v2
        # u - direction v0 -> v1
        poles = [[vtxs[0], mid(0, 3), vtxs[3]],
                 [mid(0, 1), mid(0, 1, 2, 3), mid(2, 3)],
                 [vtxs[1], mid(1, 2), vtxs[2]]
                 ]
        knots = [(0.0, 3), (1.0, 3)]
        bs_surface = bspline.Surface.make_raw(poles, (knots, knots), degree=(2, 2), rational=False)
        surface = surface_from_bs(bs_surface)
        vtxs_uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
        return (surface, vtxs_uv)

    @classmethod
    def _line(cls, vtxs, overhang=0.0):
        '''
        :param vtxs: List of tuples (X,Y) or (X,Y,Z)
        :return:
        '''
        assert len(vtxs) == 2
        vtxs = np.array(vtxs)
        mid = np.mean(vtxs, axis=0)
        poles = [vtxs[0], mid, vtxs[1]]
        knots = [(0.0 + overhang, 3), (1.0 - overhang, 3)]
        return (poles, knots)

    @classmethod
    def line_2d(cls, vtxs):
        """
        Return B-spline approximation of line from two 2d points
        :param vtxs: [ (X0, Y0), (X1, Y1) ]
        :return: Curve2D
        """
        return Curve2D(*cls._line(vtxs))

    @classmethod
    def line_3d(cls, vtxs):
        """
        Return B-spline approximation of line from two 3d points
        :param vtxs: [ (X0, Y0, Z0), (X1, Y1, Z0) ]
        :return: Curve2D
        """
        return Curve3D(*cls._line(vtxs))


class Orient(enum.IntEnum):
    Forward = 0
    Reversed = 1
    Internal = 2
    External = 3


class ShapeRef:
    """
    Auxiliary data class to store an object with its orientation
    and possibly location. Meaning of location in this context is not clear yet.
    Identity location (0) in all BREPs produced by OCC.
    All methods accept the tuple (shape, orient, location) and
    construct the ShapeRef object automatically.
    """

    orient_chars = ['+', '-', 'i', 'e']

    def __init__(self, shape, orient=Orient.Forward, transform=Identity):
        """
        :param shape: referenced shape
        :param orient: orientation of the shape, value is enum Orient
        :param transform: A Location object. Default is None = identity location.
        """
        if not issubclass(type(shape), Shape):
            raise ParamError("Expected Shape, get: {}.".format(shape))
        self.shape = shape
        # referenced shape
        # assert isinstance(orient, Orient)
        self.orientation = orient
        # orientation of the shape
        assert isinstance(transform, Transform)
        self.location = Location.make(transform)
        # (affine) transformation of the shape

    def _brep_output(self, stream):
        stream.write(
            "{}{} {} ".format(self.orient_chars[self.orientation], self.shape.brep_id, self.location.brep_id))

    def __repr__(self):
        return "{}{} ".format(self.orient_chars[self.orientation], self.shape.brep_id)


class ShapeFlag(dict):
    """
    Auxiliary data class representing the shape flag word of BREP shapes.
    All methods set the flags automatically, but it can be overwritten.

    Free - Seems to indicate a top level shapes.
    Modified - ??
    Checked - for format version 2 may indicate that shape topology is already checked
    Orientable - ??
    Closed - used to indicate closed Wires and Shells
    Infinite - ?? may indicate shapes extending to infinite, not our case
    Convex - ?? may indicate convexity of the shape, not clear how this is combined with geometry
    """
    flag_names = ['free', 'modified', 'checked', 'orientable', 'closed', 'infinite', 'convex']

    def __init__(self, *args):
        for k, f in zip(self.flag_names, args):
            assert f in [0, 1]
            self[k] = f

    def set(self, key, value=1):
        if value:
            value = 1
        else:
            value = 0
        self[key] = value

    def unset(self, key):
        self[key] = 0

    def _brep_output(self, stream):
        for k in self.flag_names:
            stream.write(str(self[k]))


class Shape(BREPObject):
    def __init__(self, children, dim):
        """
        Construct base Shape object.
        Examples:
            Wire([ edge_1, edge_2.m(), edge_3])     # recommended
            Wire(edge_1, ShapeRef(edge_2, Orient.Reversed, some_location), edge_3)
            ... not recommended since it is bad idea to reference same shape with different Locations.

        :param children: List of ShapeRefs or child objects.
        """

        # self.subtypes - List of allowed types of children.
        assert hasattr(self, 'sub_types'), self

        # convert list of shape reference tuples to ShapeRef objects
        # automatically wrap naked shapes into tuple.
        self._subshape_refs = []
        for child in children:
            self.append(child)  # append convert to ShapeRef

        self._dim = dim
        # Shape dimensionality.

        # These flags are usually produced by OCC for all other shapes safe vertices.
        self.flags = ShapeFlag(0, 1, 0, 1, 0, 0, 0)

        super().__init__(group=BREPGroup.shapes)
        assert hasattr(self, 'brep_shpname'), self
        # Name of particular shape in BREP format, defined in children.
        assert hasattr(self, 'sub_types')
        # Valid types of the shape children.

    def dimension(self):
        return self._dim

    def _children_impl(self):
        for sub_ref in self._subshape_refs:
            yield sub_ref.location
            yield sub_ref.shape

    """
    Methods to simplify creation of oriented references.
    """

    def p(self):
        # orientation plus (default)
        return ShapeRef(self, Orient.Forward)

    def m(self):
        # orientation minus, e.g. reversed edge
        return ShapeRef(self, Orient.Reversed)

    def i(self):
        # ? BREP doc
        return ShapeRef(self, Orient.Internal)

    def e(self):
        # ? BREP doc
        return ShapeRef(self, Orient.External)

    def subrefs(self):
        """
        List of subreferences of the shape (reference to subshapes).
        """
        return self._subshape_refs

    def subshapes(self):
        # Return list of subshapes stored in child ShapeRefs.
        return [chld.shape for chld in self._subshape_refs]

    def append(self, shape_ref):
        """
        Append a reference to a child
        :param shape_ref: Either ShapeRef or child shape.
        :return: None
        """
        if isinstance(shape_ref, Shape):
            shape_ref = ShapeRef(shape_ref)
        if not isinstance(shape_ref, ShapeRef):
            raise ParamError("Wrong child type: {}, allowed: Shape or ShapRef".format(type(shape_ref)))
        if not isinstance(shape_ref.shape, tuple(self.sub_types)):
            raise ParamError("Wrong child type: {}, allowed: {}".format(type(shape_ref.shape), self.sub_types))
        self._subshape_refs.append(shape_ref)

    # def _convert_to_shaperefs(self, childs):

    def set_flags(self, flags):
        """
        Set flags given as tuple.
        :param flags: Tuple of 7 flags.
        :return:
        """
        self.flags = ShapeFlag(*flags)

    def is_closed(self):
        return self.flags['closed']

    def _brep_output(self, stream):
        stream.write("{}\n".format(self.brep_shpname))
        self._subrecordoutput(stream)
        self.flags._brep_output(stream)
        stream.write("\n")
        #        stream.write("{}".format(self.childs))
        for subref in self._subshape_refs:
            subref._brep_output(stream)
        stream.write("*\n")
        # subshape, tj. childs

    def _subrecordoutput(self, stream):
        stream.write("\n")

    def _head(self):
        return f"{id(self):x} {self.brep_shpname} {str(self.brep_id)} "

    def __repr__(self):
        # if not hasattr(self, 'id'):
        #    self.index_all()
        repr = self._head()
        # if len(self.childs)==0:
        #    return ""
        repr += " : ["
        for child in self._subshape_refs:
            repr += child.shape._head()
        repr += "]"
        repr += "\n"
        return repr


"""
Shapes with no special parameters, only flags and subshapes.
Writer can be generic implemented in bas class Shape.
"""


class Compound(Shape):
    def __init__(self, shapes=None):
        if shapes is None:
            shapes = []
        if isinstance(shapes, (Shape, ShapeRef)):
            shapes = [shapes]
        self.sub_types = [Compound, CompoundSolid, Solid, Shell, Wire, Face, Edge, Vertex]
        self.brep_shpname = 'Co'
        super().__init__(shapes, None)
        # flags: free, modified, IGNORED, orientable, closed, infinite, convex
        self.set_flags((1, 1, 0, 0, 0, 0, 0))  # free, modified

    def dimension(self):
        return max(self.subshapes().dimension())

    def set_free_shapes(self):
        """
        Set 'free' attributes to all shapes of the compound.
        :return:
        """
        for shape in self.subshapes():
            shape.flags.set('free', True)


class CompoundSolid(Shape):
    def __init__(self, solids=None):
        self.sub_types = [Solid]
        self.brep_shpname = 'Cs'
        super().__init__(solids, dim=3)


class Solid(Shape):
    def __init__(self, shells=None):
        self.sub_types = [Shell]
        self.brep_shpname = 'So'
        super().__init__(shells, dim=3)
        self.set_flags((0, 1, 0, 0, 0, 0, 0))  # modified


class Shell(Shape):
    def __init__(self, faces=None):
        self.sub_types = [Face]
        self.brep_shpname = 'Sh'
        super().__init__(faces, dim=2)
        self.set_flags((0, 1, 0, 1, 0, 0, 0))  # modified, orientable


class Wire(Shape):
    def __init__(self, edges=None):
        self.sub_types = [Edge]
        self.brep_shpname = 'Wi'
        super().__init__(edges, dim=1)
        self.set_flags((0, 1, 0, 1, 0, 0, 0))  # modified, orientable
        self._set_closed()

    def _set_closed(self):
        '''
        Return true for the even parity of vertices.
        :return: REtrun true if wire is closed.
        '''
        vtx_set = {}
        for edge in self.subshapes():
            for vtx in edge.subshapes():
                vtx_set[vtx] = 0
                vtx.n_edges += 1
        closed = True
        for vtx in vtx_set.keys():
            if vtx.n_edges % 2 != 0:
                closed = False
            vtx.n_edges = 0
        self.flags.set('closed', closed)


"""
Shapes with special parameters.
Specific writers are necessary.
"""


class Face(Shape):
    """
    Face class.
    Like vertex and edge have some additional parameters in the BREP format.
    """

    def __init__(self, wires, surface=None, transform=Identity, tolerance=1.0e-3):
        """
        :param wires: List of wires, or list of edges, or list of ShapeRef tuples of Edges to construct a Wire.
        :param surface: Representation of the face, surface on which face lies.
        :param location: Location of the surface.
        :param tolerance: Tolerance of the representation.
        """
        self.sub_types = [Wire, Edge]
        self.tol = tolerance
        self.restriction_flag = 0
        self.brep_shpname = 'Fa'

        if type(wires) != list:
            wires = [wires]
        assert (len(wires) > 0)
        super().__init__(wires, dim=2)

        # auto convert list of edges into wire
        shape_type = type(self._subshape_refs[0].shape)
        for shape in self.subshapes():
            assert type(shape) == shape_type
        if shape_type == Edge:
            wire = Wire(self._subshape_refs)
            self._subshape_refs = []
            self.append(wire)

        # check that wires are closed
        for wire in self.subshapes():
            if not wire.is_closed():
                raise Exception("Trying to make face from non-closed wire.")

        if surface is None:
            self.repr = []
        else:
            assert type(surface) == Surface
            location = Location.make(transform)
            self.repr = [(surface, location)]

    def _children_impl(self):
        # Finalize the shape.
        for repr, loc in self.repr:
            yield repr
            yield loc

        yield from super()._children_impl()

    def _dfs_finish(self, visited):
        if not self.repr:
            self.implicit_surface()

        super(Face, self)._dfs_finish(visited)

    def edge_refs(self):
        """
        Generator of edge references of all wires, orientation and location
        combined.
        TODO: test location propagation
        """
        for w in self.subrefs():
            for e in w.shape.subrefs():
                t = w.location(e.location).transform
                yield ShapeRef(e.shape,
                               orient=(e.orientation * w.orientation) % 2,
                               transform=t)

    def implicit_surface(self):
        """
        Construct a surface if surface is None. Works only for
        3 and 4 vertices (plane or bilinear surface)
        Should be called in _dfs just after all child shapes are passed.
        :return: None

        TODO: simplify
        """
        edges = {}
        vtxs = []
        for wire in self.subshapes():
            for edge in wire.subrefs():
                edges[id(edge.shape)] = edge.shape
                e_vtxs = edge.shape.subshapes()
                if edge.orientation == Orient.Reversed:
                    e_vtxs.reverse()
                for vtx in e_vtxs:
                    vtxs.append((id(vtx), vtx.point))
        vtxs = vtxs[1:] + vtxs[:1]
        odd_vtx = vtxs[1::2]
        even_vtx = vtxs[0::2]
        assert odd_vtx == even_vtx, "odd: {} even: {}".format(odd_vtx, even_vtx)
        vtxs = odd_vtx
        if len(vtxs) == 3:
            constructor = Approx.plane
        elif len(vtxs) == 4:
            constructor = Approx.bilinear
        else:
            raise Exception("Too many vertices {} for implicit surface construction.".format(len(vtxs)))
        (ids, points) = zip(*vtxs)
        (surface, vtxs_uv) = constructor(list(points))
        self.repr = [(surface, Location.make(Identity))]

        # set representation of edges
        assert len(ids) == len(vtxs_uv)
        id_to_uv = dict(zip(ids, vtxs_uv))
        for edge in edges.values():
            e_vtxs = edge.subshapes()
            v0_uv = id_to_uv[id(e_vtxs[0])]
            v1_uv = id_to_uv[id(e_vtxs[1])]
            edge.attach_to_surface(surface, v0_uv, v1_uv)

        # TODO: Possibly more general attachment of edges to 2D curves for general surfaces, but it depends
        # on organisation of intersection curves.
        return self

    def _subrecordoutput(self, stream):
        assert len(self.repr) == 1
        surf, loc = self.repr[0]
        stream.write("{} {} {} {}\n\n".format(self.restriction_flag, self.tol, surf.brep_id, loc.brep_id))


class Edge(Shape):
    """
    Edge class. Special edge flags have unclear meaning.
    Allow setting representations of the edge, this is crucial for good mash generation.
    """

    class Repr(enum.IntEnum):
        Curve3d = 1
        Curve2d = 2
        # Continuous2d=3

    def __init__(self, a, b, tolerance=1.0e-3):
        """
        :param vertices: List of shape reference tuples, see ShapeRef class.
        :param tolerance: Tolerance of the representation.
        """
        self.sub_types = [Vertex]
        self.brep_shpname = 'Ed'
        self.tol = tolerance
        self.repr = []
        self.edge_flags = (1, 1, 0)  # this is usual value

        super().__init__([a, b], dim=1)
        # Overwrite vertex orientation
        self._subshape_refs[0].orientation = Orient.Forward
        self._subshape_refs[1].orientation = Orient.Reversed

    def set_edge_flags(self, same_parameter, same_range, degenerated):
        """
        Edge flags with unclear meaning.
        :param same_parameter:
        :param same_range:
        :param degenerated:
        :return:
        """
        self.edge_flags = (same_parameter, same_range, degenerated)

    def points(self):
        '''
        :return: List of coordinates of the edge vertices.
        '''
        return [vtx.point for vtx in self.subshapes()]

    def attach(self, repr, transform):
        """
        Apply transformed representation (from other vertex).
        """
        l_repr = list(repr)
        l_repr[-1] = Location.make(transform)(l_repr[-1])
        self.repr.append(tuple(l_repr))

    def attach_to_3d_curve(self, t_range, curve, transform=Identity):
        """
        Add vertex representation on a 3D curve.
        :param t_range: Tuple (t_min, t_max).
        :param curve: 3D curve object (Curve3d)
        :param location: Location object. Default is None = identity location.
        :return: None
        """
        assert type(curve) == Curve3D
        location = Location.make(transform)
        curve._eval_check(t_range[0], self.points()[0])
        curve._eval_check(t_range[1], self.points()[1])
        self.repr.append((self.Repr.Curve3d, t_range, curve, location))
        return self

    def attach_to_2d_curve(self, t_range, curve, surface, transform=Identity):
        """
        Add vertex representation on a 2D curve.
        :param t_range: Tuple (t_min, t_max).
        :param curve: 2D curve object (Curve2d)
        :param surface: Surface on which the curve lies.
        :param location: Location object. Default is None = identity location.
        :return: None
        """
        # print(f"attach: {self} {curve}")
        assert type(surface) == Surface
        assert type(curve) == Curve2D
        location = Location.make(transform)
        curve._eval_check(t_range[0], surface, self.points()[0])
        curve._eval_check(t_range[1], surface, self.points()[1])
        self.repr.append((self.Repr.Curve2d, t_range, curve, surface, location))
        return self

    def _vtx_surface_uv(self, i, surface, uv_vtx=None):
        if uv_vtx is not None:
            return uv_vtx
        return self._subshape_refs[i].shape._surface_uv(surface)

    def attach_to_surface(self, surface, v0=None, v1=None):
        """
        Construct and attach 2D line in UV space of the 'surface'
        :param surface: A Surface object.
        :param v0: UV coordinate of the first edge point
        :param v1: UV coordinate of the second edge point
        Try to get UV coordinates from the end points.

        :return:
        """
        assert type(surface) == Surface
        v0 = self._vtx_surface_uv(0, surface, v0)
        v1 = self._vtx_surface_uv(1, surface, v1)
        self.attach_to_2d_curve((0.0, 1.0), Approx.line_2d([v0, v1]), surface)
        return self

    def implicit_curve(self):
        """
        Construct a line 3d curve if there is no 3D representation.
        Should be called in _dfs.
        :return:
        """
        vtx_points = self.points()
        self.attach_to_3d_curve((0.0, 1.0), Approx.line_3d(vtx_points))
        return self

    def _dfs_finish(self, visited):
        if all((r[0] != self.Repr.Curve3d for r in self.repr)):
            self.implicit_curve()
        # No need to finish Vertex

    def _children_impl(self):
        # finalize
        assert len(self.repr) > 0

        for repr in self.repr:
            if repr[0] == self.Repr.Curve2d:
                yield repr[2]
                yield repr[3]
                yield repr[4]
            elif repr[0] == self.Repr.Curve3d:
                yield repr[2]
                yield repr[3]
        yield from super()._children_impl()

    def _subrecordoutput(self, stream):
        # print(f"subrecord: {self} {id(self):x}")
        assert len(self.repr) > 0
        stream.write(" {} {} {} {}\n".format(self.tol, self.edge_flags[0], self.edge_flags[1], self.edge_flags[2]))
        for i, repr in enumerate(self.repr):
            if repr[0] == self.Repr.Curve2d:
                curve_type, t_range, curve, surface, location = repr
                # print(f"E2: {location.brep_id}")
                stream.write("2 {} {} {} {} {}\n".format(
                    curve.brep_id, surface.brep_id, location.brep_id, t_range[0], t_range[1]))

            elif repr[0] == self.Repr.Curve3d:
                curve_type, t_range, curve, location = repr
                # print(f"E1: {location.brep_id}")
                stream.write("1 {} {} {} {}\n".format(curve.brep_id, location.brep_id, t_range[0], t_range[1]))
        stream.write("0\n")


class Vertex(Shape):
    """
    Vertex class.
    Allow setting representations of the vertex but seems it is not used in BREPs produced by OCC.
    """

    class Repr(enum.IntEnum):
        Curve3d = 1
        Curve2d = 2
        Surface = 3

    @staticmethod
    def on_surface(u, v, surface, transform=Identity):
        point = surface._bs_surface.eval(u, v)
        return Vertex(point).attach_to_surface(u, v, surface, transform)

    @staticmethod
    def on_curve_2d(t, curve, surface, transform=Identity):
        uv = curve._bs_curve.eval(t)
        point = surface._bs_surface.eval(*uv)
        return Vertex(point).attach_to_2d_curve(t, curve, surface, transform)

    @staticmethod
    def on_curve_3d(t, curve, transform=Identity):
        point = curve._bs_curve.eval(t)
        return Vertex(point).attach_to_3d_curve(t, curve, transform)

    def __init__(self, point, tolerance=1.0e-3):
        """
        :param point: 3d point (X,Y,Z)
        :param tolerance: Tolerance of the representation.
        """
        check_matrix(point, [3], scalar_types)

        # These flags are produced by OCC for vertices.
        self.flags = ShapeFlag(0, 1, 0, 1, 1, 0, 1)
        # Coordinates in the 3D space. [X, Y, Z]
        self.point = np.array(point)
        # tolerance of representations.
        self.tolerance = tolerance
        # List of geometrical representations of the vertex. Possibly not necessary for meshing.
        self.repr = []
        # Number of edges in which vertex is used. Used internally to check closed wires.
        self.n_edges = 0
        self.brep_shpname = 'Ve'
        self.sub_types = []

        super().__init__(children=[], dim=0)

    def attach(self, repr, transform):
        """
        Apply transformed representation (from other vertex).
        """
        l_repr = list(repr)
        l_repr[-1] = Location.make(transform)(l_repr[-1])
        self.repr.append(tuple(l_repr))

    def attach_to_3d_curve(self, t, curve, transform=Identity):
        """
        Add vertex representation on a 3D curve.
        :param t: Parameter of the point on the curve.
        :param curve: 3D curve object (Curve3d)
        :param location: Location object. Default is None = identity location.
        :return: None
        """
        curve._eval_check(t, self.point)
        location = Location.make(transform)
        self.repr.append((self.Repr.Curve3d, t, curve, location))
        return self

    def attach_to_2d_curve(self, t, curve, surface, transform=Identity):
        """
        Add vertex representation on a 2D curve on a surface.
        :param t: Parameter of the point on the curve.
        :param curve: 2D curve object (Curve2d)
        :param surface: Surface on which the curve lies.
        :param location: Location object. Default is None = identity location.
        :return: None
        """
        curve._eval_check(t, surface, self.point)
        location = Location.make(transform)
        self.repr.append((self.Repr.Curve2d, t, curve, surface, location))
        return self

    def attach_to_surface(self, u, v, surface, transform=Identity):
        """
        Add vertex representation on a 3D curve.
        :param u,v: Parameters u,v  of the point on the surface.
        :param surface: Surface object.
        :param location: Location object. Default is None = identity location.
        :return: None
        """
        surface._eval_check(u, v, self.point)
        location = Location.make(transform)
        self.repr.append((self.Repr.Surface, u, v, surface, location))
        return self

    def _children_impl(self):
        for repr in self.repr:
            if repr[0] == self.Repr.Surface:
                yield repr[3]  # surface
                yield repr[4]  # location
            if repr[0] == self.Repr.Curve2d:
                yield repr[2]  # curve
                yield repr[3]  # surface
                yield repr[4]  # location
            elif repr[0] == self.Repr.Curve3d:
                yield repr[2]  # curve
                yield repr[3]  # location
        yield from super()._children_impl()

    def _subrecordoutput(self, stream):  # prints vertex data
        stream.write("{}\n".format(self.tolerance))
        for i in self.point:
            stream.write("{} ".format(i))
        stream.write("\n")

        # <vertex data representation>
        for i, repr in enumerate(self.repr):
            if repr[0] == self.Repr.Surface:
                _, u, v, surface, location = repr
                # print(f"V3: {location.brep_id}")
                stream.write("{} 3 {} {} {}\n".format(
                    u, v, surface.brep_id, location.brep_id))
            if repr[0] == self.Repr.Curve2d:
                _, t, curve, surface, location = repr
                # print(f"V2: {location.brep_id}")
                stream.write("{} 2 {} {} {}\n".format(
                    t, curve.brep_id, surface.brep_id, location.brep_id))
            elif repr[0] == self.Repr.Curve3d:
                _, t, curve, location = repr
                # print(f"V1: {location.brep_id}")
                stream.write("{} 1 {} {}\n".format(
                    t, curve.brep_id, location.brep_id))

        stream.write("\n0 0\n\n")

    def _surface_uv(self, surface):
        for r in self.repr:
            if r[0] == self.Repr.Surface and r[3] is surface:
                return (r[1], r[2])
        raise KeyError("Vertex not attached to the surface.")

    # def _curve_t(self, curve):
    #     for r in self.repr:
    #         if r[0] == self.Repr.Surface and r[3] is surface:
    #             return (r[1], r[2])
    #     raise KeyError("Vertex not attached to the surface.")


def make_locations(locations):
    """
    Expand composed Transforms.
    Create Locations for basic transforms.
    Compress (reuse basic transforms).
    """
    simple_loc_map = {}
    composed_loc_list = []
    for loc in locations:
        loc.compress(simple_loc_map)
        if isinstance(loc, CompositeLocation):
            composed_loc_list.append(loc)
    # renumber locations
    new_locations = list(simple_loc_map.values())
    new_locations.extend(composed_loc_list)
    for i, loc in enumerate(new_locations):
        loc._brep_id = i
    return new_locations


def write_model(stream, compound, transform=Identity):
    """
    Write a BREP representation of the model 'compound' transformed to the 'location'
    to the 'stream'.
    """
    assert isinstance(compound, Compound)
    compound._dfs_finish()
    location = Location.make(transform)
    groups = BREPObject.gather_groups([Location.make(Identity), compound, location])
    locations = groups[BREPGroup.locations]
    curves_3d = groups[BREPGroup.curves_3d]
    curves_2d = groups[BREPGroup.curves_2d]
    surfaces = groups[BREPGroup.surfaces]
    shapes = groups[BREPGroup.shapes]

    locations = make_locations(locations)
    n_shapes = len(shapes) + 1
    for shape in shapes:
        shape._brep_id = n_shapes - shape.brep_id

    stream.write("DBRep_DrawableShape\n\n")
    stream.write("CASCADE Topology V1, (c) Matra-Datavision\n")
    stream.write("Locations {}\n".format(len(locations) - 1))
    for loc in locations[1:]:
        loc._brep_output(stream)

    stream.write("Curve2ds {}\n".format(len(curves_2d)))
    for curve in curves_2d:
        curve._brep_output(stream)

    stream.write("Curves {}\n".format(len(curves_3d)))
    for curve in curves_3d:
        curve._brep_output(stream)

    stream.write("Polygon3D 0\n")

    stream.write("PolygonOnTriangulations 0\n")

    stream.write("Surfaces {}\n".format(len(surfaces)))
    for surface in surfaces:
        surface._brep_output(stream)

    stream.write("Triangulations 0\n")

    stream.write("\nTShapes {}\n".format(len(shapes)))
    for shape in shapes:
        shape._brep_output(stream)
    stream.write(f"\n+1 {location.brep_id}")


class Factory:
    """
    Preliminary factory functions for creating basic shapes.
    """

    @staticmethod
    def polygon(points_2d: List['Point']):
        """
        :param points: list of 2d points, or  array of shape (N,2)
        """
        surface, _ = Approx.plane([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        vtxs = [Vertex.on_surface(*uv, surface) for uv in points_2d]
        # vtxs = [Vertex([*p, 0]) for p in points_2d]
        vtxs.append(vtxs[0])
        # edges = [Edge(*vtx).attach_to_surface(surface, *uv) for vtx, uv in zip(zip(vtxs[:-1], vtxs[1:]), points_2d)]
        edges = [Edge(*vtx).attach_to_surface(surface) for vtx in zip(vtxs[:-1], vtxs[1:])]
        # edges = [Edge(a, b) for a, b in zip(vtxs[:-1], vtxs[1:])]
        face = Face(edges, surface=surface)
        # face = Face(edges)
        return face

    @staticmethod
    def prism(points_2d, height: float):
        """
        :param points: np.array, shape (2, N)
        """
        base_face = Factory.polygon(points_2d)
        prism_raw = Factory.extrude(base_face, [0, 0, height])
        # centered

        return Compound(ShapeRef(prism_raw, transform=Transform.Translate([0, 0, -height / 2])))

    @staticmethod
    def extrude(shape, vector):
        """
        Extrude 'shape' which should be at most of dimension 2 (not Shell or Solid)
        Composed should not be excluded.
        TOOD: through testing
        """
        assert shape.dimension() <= 2
        vector = np.array(vector, dtype=float)
        shift_transform = Transform().translate(vector)
        top_map = {}
        extrusion_map = {}

        # new vertices
        for v_bot in shape.children(recursive=True, of_type=Vertex):
            v_top = Vertex(v_bot.point + vector)
            for r in v_bot.repr:
                v_top.attach(r, transform=shift_transform)
            top_map[id(v_bot)] = v_top
            extrusion_map[id(v_bot)] = Edge(v_bot, v_top)
        # new edges
        for e_bot in shape.children(recursive=True, of_type=Edge):
            vtxs_top = [top_map[id(v)] for v in e_bot.subshapes()]
            assert len(vtxs_top) == 2
            e_top = Edge(*vtxs_top)
            for r in e_bot.repr:
                e_top.attach(r, transform=shift_transform)
            # print(f"{id(e_bot)} : {e_bot}")
            top_map[id(e_bot)] = e_top

            edges_extr = [extrusion_map[id(v)] for v in e_bot.subshapes()]

            edges = [e_bot, edges_extr[1], e_top.m(), edges_extr[0].m()]
            # print(f"{id(e_bot)} : {e_bot}")
            extrusion_map[id(e_bot)] = Face(edges)

        # new faces

        def map_shape_ref(shape_ref, map):
            return ShapeRef(map[id(shape_ref.shape)], shape_ref.orientation, shape_ref.location.transform)

        def top_wire(bot_wire):
            top_edges = [map_shape_ref(edge, top_map) for edge in bot_wire.subrefs()]
            return Wire(top_edges)

        solids = []
        for f_bot in shape.children(recursive=True, of_type=Face):
            top_wires = [top_wire(bot_wire.shape) for bot_wire in f_bot.subrefs()]
            surf, loc = f_bot.repr[0]
            f_top = Face(top_wires, surface=surf, transform=shift_transform @ loc.transform)
            top_map[id(f_bot)] = f_top

            faces_extr_refs = [map_shape_ref(edge, extrusion_map) for edge in f_bot.edge_refs()]
            shell = Shell([f_bot, f_top.m(), *faces_extr_refs])
            solids.append(Solid([shell]))

        return CompoundSolid(solids)

    @staticmethod
    def box(dimensions=[1, 1, 1], center=[0, 0, 0]):
        raw_box = Factory.prism([[-1, -1], [1, -1], [1, 1], [-1, 1]], height=1)
        loc = Transform().scale(np.array(dimensions) / 2).translate(center)
        return Compound(ShapeRef(raw_box, transform=loc))
