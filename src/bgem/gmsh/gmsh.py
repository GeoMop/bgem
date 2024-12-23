import itertools
from typing import *
from collections import defaultdict
import enum
import attrs
import numpy as np
import gmsh
import re
import warnings

from bgem import Transform
from bgem.gmsh import gmsh_exceptions
from bgem.gmsh import options as gmsh_options
from bgem.gmsh import gmsh_io

"""
Structure:
gmsh
gmsh.option
gmsh.model
gmsh.model.mesh
gmsh.model.mesh.field
gmsh.model.geo
gmsh.model.geo,mesh
gmsh.model.occ
gmsh.view
gmsh.plugin
gmsh.graphics
gmsh.fltk
gmsh.onelab
gmsh.logger

gmsh_api, issues:
- terrible interface to fields (resolved by field.py)

- get_boundary not part of geometry model (occ/geo), need lot of synchronizations
  (resolved by automatic synchronizations)
- all existing dimtags are meshed not only those with assigned physical groups
  (resolved by a step before meshing that removes all objects without associated physical group)
- physical groups are not assigned to the objects, but groups are formed from objects, 
  possible error having single object in more physical groups
  (Not sure what happens in the case of two groups for single object, but in general the concept is a valid option.)
- Mesh.Format option - doc do not support gmsh 2.0 format
  gmsh.write function seems to ignore the format and use extensions which are not documented
  (Resolved. Version can be set by different option,)
- gmsh.model.occ.setMeshSize - seems have no effect, in particular in combination with getBoundary
  (Confirmed, replaced by similar function in other module)
- no constant field
  (Resolved by the sphere field)
- gmsh.model.occ.removeAllDuplicates ... doesn't work
  (No sure, it works at least partially.) 
- seems that occ.copy() doesn't preserve boundaries, so boundary dim tags are copied twice
  (It does exactly what it is asked for just copy the given shapes)
(Problem resolved by introduction of select_by_intersection)

Proposed functional apporach:

Rationale:
- ObjectSet keeps set of dimtags, which may become invalid, ObjectSets could become invalid because of  operation of larger ObjectSet that includes it.
  
  Resoluion: We would rather refactor ObjectSets into handles to the particular geometry part referenced by the complete tree of operations leading to it.
   The whole geometry would be first constructed in terms of a DAG of operations and then applied to GMSH just before meshing. 
   The application would treat each operation as global, properly mapping dimtags to the new ons and mange properties associated with the 
   dimtags.
- Getting boundary is not associative, bool opertations provides mapping only to dimtags explicitely provided as arguments. 
  Probably no way to map boundaries. Need to treat boundary operations in lazy way apply them more times using orig
    

"""



@attrs.define(auto_attribs=True, frozen=False)
class Region:
    dim: Optional[int]
    id: int
    name: str
    _boundary_region: 'Region' = None
    _gmsh_id: int = None
    # Physical domain ID in GMSH mesh
    _max_reg_id = 99999


    @classmethod
    def get_region_id(cls):
        cls._max_reg_id += 1
        return cls._max_reg_id

    @classmethod
    def get(cls, name, dim=None):
        """
        Return a unique possibly uncompleted region.
        """
        return Region(dim, cls.get_region_id(), name)

    def complete(self, dim):
        """
        Check dimension match and complete the region.
        """
        if self.dim is None:
            self.dim = dim
        else:
            assert self.dim == dim, (self.dim, dim)
        return self

    def set_unique_name(self, idx):
        self.name = "{}_{}".format(self.name, idx)


# Initialize class attribute
Region.default_region = [Region.get("default_{}d".format(dim), dim) for dim in range(4)]


class MeshFormat(enum.IntEnum):
    msh = 1
    unv = 2
    msh2 = 3  # only for extension, code unknown
    auto = 10
    vtk = 16
    vrml = 19
    mail = 21
    pos_stat = 26
    stl = 27
    p3d = 28
    mesh = 30
    bdf = 31
    cgns = 32
    med = 33
    diff = 34
    ir3 = 38
    inp = 39
    ply2 = 40
    celum = 41
    su2 = 42
    tochnog = 47
    neu = 49
    matlab = 50


DimTag = Tuple[int, int]




class Mesh:
    """
    Interface to the GMSH mesh object with access to suitable mesh processing for
    Flow123d simulation.
    - mesh healing (with allowed small corruption of internal interfaces)
    - region/physical domain  association based on element entities
    - region/physical domain association assigned after meshing, elements need not to respect interfaces
    # - reading and writing field data on the mesh

    GMSH model use global variables and could not be simply
    interfaced using a functional approach.
    Thus we rather want to copy mesh data to our own structures.


    """
    pass
  
  
class GeometryOCC:
    """
    User friendly and mesh consistent interface to gmsh_api (gmsh_sdk package).
    Only single instance is allowed (due to limitation of gmsh_sdk and OCC).
    TODO: use a singleton pattern
    TODO: add remaining creation methods
    TODO: add documentation
    """
    _have_instance = False

    # def addPoint(self, x, y, z, size):
    #     return self.object(0, self.model.addPoint(x, y, z, size))
    #
    # def addLine(self, start, end):
    #     return self.object(1, self.model.addLine(start.point(), end.point()))
    #
    # def addCircleArc(self, start, center, end):
    #     return self.object(1, self.model.addCircleArc(start.point(), center.point(), end.point()))
    #
    # def addEllipseArc(self, start, center, end):
    #     return self.object(1, self.model.addEllipseArc(start.point(), center.point(), end.point()))
    #
    # def addSpline(self, points):
    #     points = [pt_obj.point() for pt_obj in points]
    #     return self.object(1, self.model.addSpline(points))
    #
    # def addBSpline(self, points):
    #     points = [pt_obj.point() for pt_obj in points]
    #     return self.object(1, self.model.addBSpline(points))
    #
    # def addBezier(self, points):
    #     points = [pt_obj.point() for pt_obj in points]
    #     return self.object(1, self.model.addBezier(points))
    #
    # def addCurveLoop(self):
    #     pass
    #
    # def addPlaneSurface(self):
    #     pass
    #
    # addSurfaceFilling
    #
    # addSurfaceLoop
    #
    # addVolume
    #
    # extrude
    #
    # revolve
    #
    #
    # twist
    #
    # translate
    #
    # rotate
    # dilate
    #
    # symmetrize
    #
    # copy
    #
    # remove
    #
    # removeAllDuplicates
    #
    # synchronize

    def __init__(self, model_name, model_str='occ', **kwargs):
        """

        Args:
            model_name: Name of the geometry model, used to name resulting files.
            model_str:
                'occ' - use geometry model based on OCC
                'geo' - use own GMSH geometry model, no support for boolean operations
            **kwargs:
                'verbose' - force GMSH output to stdout
                'gmsh_exceptions' - if True (default), then re-raise GMSH exceptions
                                  - otherwise log GMSH exceptions as warnings
        """
        if model_str == 'occ':
            self.model = gmsh.model.occ
        elif model_str == 'geo':
            self.model = gmsh.model.geo
        else:
            raise ValueError

        if self._have_instance:
            raise Exception("Only single instance of GMSHFactory is allowed.")
        else:
            self._have_instance = False

        self.model_name = model_name
        gmsh.initialize()
        gmsh.model.add(model_name)
        print("GMSH initialized")

        self._region_names = {}
        self._need_synchronize = False
        self.mesh_options = gmsh_options.Mesh()
        self.geom_options = gmsh_options.Geometry()
        gmsh.option.setNumber("General.Terminal", kwargs.get('verbose', False))
        self.gmsh_exceptions = kwargs.get('gmsh_exceptions', True)

    @staticmethod
    def get_logger():
        """
        See: https://gmsh.info/dev/doc/texinfo/gmsh.html#Namespace-gmsh_002flogger
        """
        return gmsh.logger

    def _raise_gmsh_exception(self, gmsh_err, msg):
        if self.gmsh_exceptions:
            # raise gmsh_err.with_traceback(err.__traceback__) from err
            raise gmsh_err(msg)
        else:
            warn_cls = gmsh_exceptions.make_warning(gmsh_err)
            warnings.warn(message="[GMSH]: " + msg, category=warn_cls, stacklevel=3)

    def reinit(self):
        """
        Clear whole geometry model.
        TODO: need tests
        Returns:
        """
        gmsh.clear()

    def get_region_name(self, name: str) -> Region:
        """
        Return the 'region' object by given 'name'

        Create a new region of that name if it doesn't exist yet.
        """
        region = self._region_names.get(name, Region.get(name))
        self._region_names[name] = region
        return region

    def object(self, dim: int, tag: int) -> 'ObjectSet':
        """
        Create new object set from a dimtag.
        """
        return ObjectSet(self, [(dim, tag)], [Region.default_region[dim]])

    def make_simplex(self, dim=3):
        """
        Make reference simplex of dimension 'dim' [0,1,2,3].
        Vertices are in origin and in be base vectors.
        TODO: use own methods for construction of geometries (combine with BSplines lib.)

        return: Object set with a single dimtag.
        """
        points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        lines = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]
        faces = [(0, 1, 2), (0, 3, 4), (2, 3, 5), (1, 4, 5)]
        if dim == 0:
            res = self.model.addPoint(*points[0])
        elif dim == 1:
            point_ids = [self.model.addPoint(*p) for p in points[:2]]
            res = self.model.addLine(*point_ids)
        elif dim == 2:
            point_ids = [self.model.addPoint(*p) for p in points[:3]]
            line_ids = [self.model.addLine(*[point_ids[p] for p in l]) for l in lines[:3]]
            loop = self.model.addCurveLoop(line_ids)
            res = self.model.addPlaneSurface([loop])
        elif dim == 3:
            point_ids = [self.model.addPoint(*p) for p in points[:4]]
            line_ids = [self.model.addLine(*[point_ids[p] for p in l]) for l in lines[:6]]
            loop_ids = [self.model.addCurveLoop([line_ids[l] for l in f]) for f in faces[:4]]
            face_ids = [self.model.addPlaneSurface([loop]) for loop in loop_ids]
            surf_loop = self.model.addSurfaceLoop(face_ids)
            res = self.model.addVolume([surf_loop])
        self._need_synchronize = True
        return self.object(dim, res)

    def point(self, coord=[0, 0, 0]):
        """
        Add a geometrical point.
        """
        point_tag = self.model.addPoint(*coord)
        self._need_synchronize = True
        return self.object(0, point_tag)

    def _get_point(self, a):
        """
        If 'a' is a point dim tag, i.e. (0, tag), return the dim tag.
        If 'a' is a ObjectSet containing a point return its dimtag.
        If 'a' are coordi
        Take either a point dimtag or point coordi
        a: tuple(float, float, float)
        Add a point with given coordinates or

        a: ObjectSet containing a single point.
        return: the point (dim, id)
        """
        if isinstance(a, (tuple, list)):
            a = self.point(a)
        assert isinstance(a, ObjectSet) and len(a.dim_tags) == 1, a
        dim, tag = a.dim_tags[0]
        assert dim == 0
        return tag

    def line(self, a, b):
        """
        Make line between points a,b.
        return: Object set with a single dimtag.
        """
        point_ids = [self._get_point(p) for p in [a, b]]
        # point_ids = [self.model.addPoint(*p) for p in [a, b]]
        res = self.model.addLine(*point_ids)
        self._need_synchronize = True
        return self.object(1, res)

    def rectangle(self, xy_sides=[1, 1], center=[0, 0, 0]):
        """
        TODO: Better match GMSH API, possibly use origin as the default left corner.
        Support for round points?? Does it work in OCC?

        Add a rectangle with lower left corner at (`x', `y', `z') and upper right
        corner at (`x' + `dx', `y' + `dy', `z'). If `tag' is positive, set the tag
        explicitly; otherwise a new tag is selected automatically. Round the
        corners if `roundedRadius' is nonzero. Return the tag of the rectangle.

        Return an integer value.
        """
        corner = np.array(center) - np.array([*xy_sides, 0]) / 2
        rec_tag = self.model.addRectangle(*corner.tolist(), *xy_sides)
        self._need_synchronize = True
        return self.object(2, rec_tag)

    def circle(self, radius, center=[0, 0, 0]):
        """
        Creates circle.
        Note that OCC model has a direct function for it.
        GEO model has to build the circle from circle arcs
        (at least 3 due to arcs have to have angle strictly smaller than pi).
        """
        circ_arcs = []
        if self.model is gmsh.model.occ:
            circ_arcs.append(self.model.addCircle(*center, radius))
        elif self.model is gmsh.model.geo:
            a = center + radius * np.array([1, 0, 0])
            b = center + radius * np.array([-1, 0, 0])
            c = center + radius * np.array([0, 1, 0])
            d = center + radius * np.array([0, -1, 0])

            ap = self.model.addPoint(*a)
            bp = self.model.addPoint(*b)
            cp = self.model.addPoint(*c)
            dp = self.model.addPoint(*d)
            centp = self.model.addPoint(*center)

            circ_arcs.append(self.model.addCircleArc(ap, centp, cp))
            circ_arcs.append(self.model.addCircleArc(cp, centp, bp))
            circ_arcs.append(self.model.addCircleArc(bp, centp, dp))
            circ_arcs.append(self.model.addCircleArc(dp, centp, ap))

        circ_loop = self.model.addCurveLoop(circ_arcs)
        self._need_synchronize = True
        return self.object(1, circ_loop)

    def disc(self, center=[0, 0, 0], rx=1, ry=1):
        """
        Add a disk with `center` and radius `rx' along the x-axis
        and `ry' along the y-axis.

        Return an ObjectSet with the created disc.
        """
        if self.model is gmsh.model.geo:
            #     circ = self.circle(radius, center)
            #     surface = self.model.addPlaneSurface([*circ.tags])
            #     self._need_synchronize = True
            #     return self.object(2, surface)
            return None
        elif self.model is gmsh.model.occ:
            disc = self.model.addDisk(*center, rx, ry)
            self._need_synchronize = True
            return self.object(2, disc)

    def box(self, sides, center=[0, 0, 0]):
        """
        TODO: see addRectangle.
        Add a parallelepipedic box defined by a point (`x', `y', `z') and the
        extents along the x-, y- and z-axes.

        Return an integer value.
        """
        corner = np.array(center) - np.array(sides) / 2
        box_tag = self.model.addBox(*corner, *sides)
        self._need_synchronize = True
        return self.object(3, box_tag)

    def cylinder(self, r=1, axis=[0, 0, 1], center=[0, 0, 0]):
        """
        Add a cylinder, defined by the 'center' of its first circular
        face, by its 'axis' vector between centers of the faces
        and its radius `r'.
        TODO: The optional `angle' argument defines the angular
        opening (from 0 to 2*Pi).

        Return the resulting ObjectSet, containing single 3d dimtag.
        """
        cylinder_tag = self.model.addCylinder(*center, *axis, r)
        self._need_synchronize = True
        return self.object(3, cylinder_tag)

    def disc_discrete(self, radius=1, center=[0, 0, 0], n_points=6, axis=[0, 0, 1]):
        """
        Create a regular polygon with n_points vertices.
        :param radius: Radius of
        :param center:
        :param n_points:
        :param axis:
        :return:
        """
        points = []
        v = [1, 0, 0]  # take a random vector
        # test if v and axis are coplanar
        n = np.abs(np.dot(axis / np.linalg.norm(axis), v) - 1)
        if n < 5e-15:
            v = [0, 0, 1]

        v = np.cross(v, axis)  # directional vector in disc plane
        v = v / np.linalg.norm(v)  # normalize
        dphi = 2 * np.pi / n_points  # differential angle between circ points
        for i in range(0, n_points):
            points.append(center + radius * v)
            v = np.dot(rotation_matrix(axis, dphi), v)

        self._need_synchronize = True
        return self.make_polygon(points)

    def cylinder_discrete(self, radius=1, axis=[0, 0, 1], center=[0, 0, 0], n_points=6):
        base_center = center - axis / 2
        base = self.disc_discrete(radius, base_center, n_points, axis)
        base_extrude = base.extrude(axis)
        cylinder = base_extrude[3]

        self._need_synchronize = True
        return cylinder

    def make_polygon(self, points, mesh_step=None):
        """
        Add a polygon given by vertices. Assume thay are coplanar.
        :param mesh_step: scalar or array (N,)
        :param points: Array (N, 3) of points.
        """
        if mesh_step is None:
            mesh_step = 0.0
        if type(mesh_step) is float:
            mesh_step = np.full(len(points), mesh_step)
        vertices = [self.model.addPoint(*p, step, tag=-1) for p, step in zip(points, mesh_step)]
        vertices.append(vertices[0])
        lines = [self.model.addLine(a, b) for a, b in zip(vertices[0:-1], vertices[1:])]
        loop = self.model.addCurveLoop(lines, tag=-1)
        surface = self.model.addPlaneSurface([loop], tag=-1)
        return self.object(2, surface)

    def import_shapes(self, fileName, highestDimOnly=True):
        """
        Import BREP, STEP or IGES shapes from the file fileName in the OpenCASCADE CAD representation.
        :param fileName:
        :param highestDimOnly:

        """
        shapes = self.model.importShapes(fileName, highestDimOnly=highestDimOnly)
        self._need_synchronize = True
        return ObjectSet(self, shapes, [Region.default_region[dim] for dim, _ in shapes])

    def synchronize(self):
        """
        Not clear which actions requires synchronization. Seems that it should be called after calculation of
        new shapes and before new shapes are added explicitly.
        """
        if self._need_synchronize:
            self.model.synchronize()
            self._need_synchronize = False

    def make_rectangle(self, scale) -> int:
        # Vertices of the rectangle
        shifts = np.array([(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)])
        corners = shifts[:, :] * scale[None, :]
        point_tags = [self.model.addPoint(*corner) for corner in corners]
        lines = [self.model.addLine(point_tags[i - 1], point_tags[i]) for i in range(4)]
        cl = self.model.addCurveLoop(lines)
        self._need_synchronize = True
        return self.model.addPlaneSurface([cl])

    def make_fractures(self, fractures, base_shape: 'ObjectSet'):
        # From given fracture date list 'fractures'.
        # transform the base_shape to fracture objects
        # fragment fractures by their intersections
        # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
        shapes = []
        for i, fr in enumerate(fractures):
            shape = base_shape.copy()
            print("fr: ", i, "tag: ", shape.dim_tags)
            shape = shape.transfrom(fr.transform_mat) \
                .translate(fr.center) \
                .set_region(fr.region)

            shapes.append(shape)

        fracture_fragments = self.fragment(*shapes)
        return fracture_fragments

    def fragment(self, *object_sets: 'ObjectSet') -> List['ObjectSet']:
        """
        Fragment given objects mutually return list of fragmented objects.
        First perform copy of the intput dimtags.
        :param object_sets:
        :return:
        """

        cumulsizes = list(itertools.accumulate((o.size for o in object_sets)))
        all_dimtags = list(itertools.chain(*[o.dim_tags for o in object_sets]))
        # copy_all_dimtags = ObjectSet(self, all_dimtags).copy()
        if len(all_dimtags) == 1:
            new_tags, tags_map = all_dimtags, [all_dimtags]
        else:
            try:
                new_tags, tags_map = self.model.fragment(all_dimtags, [], removeObject=True, removeTool=True)
                # list of new dimtags, list of
            except ValueError as err:
                message = "Fragmentation failed!\nall dimtags: {}, ...".format(str(all_dimtags[:20]))
                self._raise_gmsh_exception(gmsh_exceptions.BoolOperationError, message)

        # assign regions
        new_sets = []
        begin = 0
        assert cumulsizes[-1] == len(tags_map), str(tags_map[cumulsizes[-1]:])
        for o, end in zip(object_sets, cumulsizes):
            dim_tag_map = tags_map[begin:end]
            object_list = []
            for reg, step, new_subtags in zip(o.regions, o.mesh_step_size, dim_tag_map):
                newobj = ObjectSet(self, new_subtags, [reg])
                newobj.mesh_step(step)
                object_list.append(newobj)
            newset = self.group(*object_list)
            new_sets.append(newset)
            begin = end
            o.invalidate()

        self._need_synchronize = True
        return new_sets

    def _assign_physical_groups(self, obj):
        self.synchronize()
        reg_to_tags = {}
        reg_names = defaultdict(set)

        # collect tags of regions
        for dimtag, reg in obj.dimtagreg():
            dim, tag = dimtag
            reg.complete(dim)
            reg_to_tags.setdefault(reg.id, (reg, []))
            reg_to_tags[reg.id][1].append(tag)
            reg_names[reg.name].add(reg.id)

        # make used region names unique
        for id_set in reg_names.values():
            if len(id_set) > 1:
                for i, id in enumerate(sorted(id_set)):
                    reg_to_tags[id][0].set_unique_name(i)

        # set physical groups
        for reg, tags in reg_to_tags.values():
            reg._gmsh_id = gmsh.model.addPhysicalGroup(reg.dim, tags, tag=-1)
            gmsh.model.setPhysicalName(reg.dim, reg._gmsh_id, reg.name)

    def _set_mesh_step(self, obj: 'ObjectSet'):
        self.synchronize()
        step_to_dimtags = {}

        # create map step -> dimtags
        for dimtag, step in zip(obj.dim_tags, obj.mesh_step_size):
            if step == obj.default_mesh_step:
                continue
            step_to_dimtags.setdefault(step, [])
            step_to_dimtags[step].append(dimtag)

        # sort from the largest to the smallest step
        step_to_dimtags_sorted = sorted(step_to_dimtags.items(), key=lambda item: item[0], reverse=True)
        for step, dimtags in step_to_dimtags_sorted:
            self._set_size_recursive(dimtags, step)
            # this does not actually work to set mesh step size
            # self.model.mesh.setSize(dimtags, step)

    def _set_size_recursive(self, dimtags, step):
        # Workaround for the non-functional occ.setSize.
        # This method retrieves boundary nodes recursively
        try:
            b_dimtags = gmsh.model.getBoundary(dimtags, combined=False, oriented=False, recursive=True)
        except ValueError as err:
            message = "Set size recursively failed!\nobj dimtags: {} ...".format(str(dimtags[:10]))
            self._raise_gmsh_exception(gmsh_exceptions.GetBoundaryError, message)

        nodes = [(dim, tag) for dim, tag in b_dimtags if dim == 0]
        gmsh.model.mesh.setSize(nodes, step)

    def set_mesh_step_field(self, field: 'Field') -> None:
        field.reset_id()
        id = field.construct(self)
        gmsh.model.mesh.field.setAsBackgroundMesh(id)

    def make_mesh(self, objects: List['ObjectSet'], dim=3, eliminate=True) -> None:
        """
        Generate mesh for given objects.
        0. set the mesh step.
        1. set physical groups from objects regions.
        2. OPTIONAL remove other shapes then specified
        3. call meshing
        4. remove duplicate nodes using tolerance Geometry.Tolerance.

        TODO: change parameters, to 8objects instead of  the list
        :param dim: Set highest dimension to mesh.
        :param eliminate:
        """

        group = self.group(*objects)
        self._assign_physical_groups(group)
        self._set_mesh_step(group)

        if eliminate:
            self.keep_only(group)
        self.synchronize()
        gmsh.model.mesh.generate(dim)
        gmsh.model.mesh.removeDuplicateNodes()
        bad_entities = gmsh.model.mesh.getLastEntityError()
        if bad_entities:
            print("Bad entities:", bad_entities)

    def write_brep(self, filename=None):
        self.synchronize()
        if filename is None:
            filename = self.model_name
        gmsh.write(filename + '.brep')

    def write_mesh(self, filename: Optional[str] = None, format: MeshFormat = MeshFormat.auto) -> None:
        """
        Write a mesh generated by 'make_mesh' to the file 'filename'.
        Format is given by extension (see MeshFormat for supported formats)
        If 'filename' is not provided it is determined by the model name given in constructor.
        In such case the format is given by the 'format' parameter.
        TODO: check that mesh is created.
        """
        if filename is None:
            gmsh.option.setNumber('Mesh.Format', format)
            extension = format.name
            filename = "{}.{}".format(self.model_name, extension)
        gmsh.write(filename)

    def remove_duplicate_entities(self):
        self.synchronize()
        try:
            self.model.removeAllDuplicates()
        except Exception as err:
            msg = "Remove duplicate entities failed!"
            self._raise_gmsh_exception(gmsh_exceptions.FragmentationError, msg)

        self._need_synchronize = True

    def all(self):
        """
        Return all model entities as a single ObjectSet.
        """
        all_dimtags = gmsh.model.getEntities()
        regions = [Region.default_region[dim] for dim, tag in all_dimtags]
        return ObjectSet(self, all_dimtags, regions)

    def keep_only(self, *object_sets):
        self.synchronize()
        if object_sets:
            group_dimtags = self.group(*object_sets).dim_tags
        else:
            group_dimtags = []
        all_dimtags = set(gmsh.model.getEntities())
        remove_dimtags = all_dimtags.difference(set(group_dimtags))
        remove_dimtags = list(remove_dimtags)
        # Recursive removal may lead to duplicate removal of dimtags.
        # We sort the list by dimension avoid this.
        remove_dimtags.sort()
        while remove_dimtags:
            try:
                self.model.remove(remove_dimtags, recursive=True)
            except Exception as e:
                msg = str(e)
                res = re.match('Unknown OpenCASCADE entity of dimension (\\d*) with tag (\\d*)', msg)
                if res:
                    dim, tag = int(res[1]), int(res[2])
                    idx = remove_dimtags.index((dim, tag))
                    remove_dimtags = remove_dimtags[idx + 1:]
                else:
                    raise e
            else:
                remove_dimtags = []

    def all_entities(self):
        self.synchronize()
        return gmsh.model.getEntities()

    def show(self):
        gmsh.fltk.run()

    def __del__(self):
        gmsh_io.gmsh_finalize()

    def group(self, *obj_list: Union['ObjectSet', List['ObjectSet']]) -> 'ObjectSet':
        if obj_list:
            return ObjectSet.group(*obj_list)
        else:
            return ObjectSet(self, [], [])


class ObjectSet:
    default_mesh_step = 0

    @staticmethod
    def group(*obj_list: 'ObjectSet') -> 'ObjectSet':
        """
        Group any number of ObjectSets into a single one.
        :param obj_list:
        :return:
        """
        assert obj_list
        new_obj_list = []
        model = set()
        for item in obj_list:
            if isinstance(item, ObjectSet):
                new_obj_list.append(item)
                model.add(item.factory)
            else:
                raise Exception(f"group: Wrong argument of type {type(item)}, expecting ObjectSet..")

        assert len(model) == 1
        model = list(model)[0]

        # for obj in new_obj_list:
        #     if iobj

        # Wrap dim_tags
        # for obj in new_obj_list
        # elif isinstance(item, tuple):
        # assert len(item) == 2
        # new_obj_list.append(ObjectSet([item]))

        if len(new_obj_list) == 1:
            return new_obj_list[0]
        all_dim_tags = [dim_tag
                        for obj in new_obj_list
                        for dim_tag in obj.dim_tags]
        regions = [reg
                   for obj in new_obj_list
                   for reg in obj.regions]
        mesh_step_size = [step
                          for obj in new_obj_list
                          for step in obj.mesh_step_size]

        g = ObjectSet(model, all_dim_tags, regions)
        g.mesh_step_size = mesh_step_size
        return g

    def __init__(self, factory: 'GeometryOCC', dim_tags: List[DimTag], regions: List[Region]) -> None:
        self.factory = factory
        self.dim_tags = dim_tags
        # TODO: allow no region to be assigned, but that makes problem later in region operations
        # if regions is None:
        #    regions = [None]
        if len(regions) == 1:
            self.regions = [regions[0] for _ in dim_tags]
        else:
            assert (len(regions) == len(dim_tags))
            self.regions = regions
        self.mesh_step_size = [self.default_mesh_step for _ in dim_tags]

    def __repr__(self):
        return f"ObjectSet[{self.dim_tags}]"

    @property
    def tags(self):
        return [tag for dim, tag in self.dim_tags]

    @property
    def size(self):
        return len(self.dim_tags)

    def max_dim(self):
        dims = [d for d, t in self.dim_tags]
        return max(dims, default=1)

    def set_region(self, region):
        """
        Set given region to all self.dimtags.
        Create a new region if just a string is given.
        :return: self
        """
        if isinstance(region, str):
            region = self.factory.get_region_name(region)
        self.regions = [region.complete(dim) for dim, tag in self.dim_tags]
        return self

    def modify_regions(self, format: str):
        """
        For every of object's regions create a new region with a name given by the 'format'
        and original region name.
        : param format: a string format with single placeholder.
        E.g. to prefix all region names by 'XY_' use format "XY_{}".

        TODO: allow to include: dim, tag, entity type into the format string through named placehodders
        """
        regions = []
        for region in self.regions:
            new_name = format.format(region.name)
            new_region = self.factory.get_region_name(new_name)
            regions.append(new_region)
        self.regions = regions
        return self

    def translate(self, vector):
        self.factory.model.translate(self.dim_tags, *vector)
        self.factory._need_synchronize = True
        return self

    def transform(self, transform:Transform):
        self.factory.model.affineTransform(self.dim_tags, transform.full_affine_matrix)

    def rotate(self, axis, angle, center=[0, 0, 0]):
        self.factory.model.rotate(self.dim_tags, *center, *axis, angle)
        self.factory._need_synchronize = True
        return self

    def scale(self, scale_vector, center=[0, 0, 0]):
        self.factory.model.dilate(self.dim_tags, *center, *scale_vector)
        self.factory._need_synchronize = True
        return self

    def extrude(self, vector, numElements=[], heights=[], recombine=False) -> List['ObjectSet']:
        """
        Extrudes the self object in the direction of 'vector'.
        Self object is NOT destroyed.
        Returns list ObjecSet of length 4, each corresponds to its dimension.

        Parameters numElements, heights, recombine have not been investigated yet.
        """
        try:
            outDimTags = self.factory.model.extrude(self.dim_tags, *vector, numElements, heights, recombine)
        except ValueError as err:
            message = "\nExtrusion failed!\ndimtags: {}".format(str(self.dim_tags[:10]))
            gerr = gmsh_exceptions.BoolOperationError(message)
            self._raise_gmsh_exception(gerr, err)

        regions = [Region.default_region[dim] for dim, tag in outDimTags]
        all_obj = ObjectSet(self.factory, outDimTags, regions)

        self.factory._need_synchronize = True
        # split the Objectset by dimtags
        return all_obj.split_by_dimension()

    def revolve(self, center, axis, angle, numElements=[], heights=[], recombine=False) -> List['ObjectSet']:
        """
        Extrudes the self object by revolving it around the axis given by 'center' and 'axis'.
        Self object is NOT destroyed.
        Returns list ObjecSet of length 4, each corresponds to its dimension.

        Parameters numElements, heights, recombine have not been investigated yet.
        """
        try:
            outDimTags = self.factory.model.revolve(self.dim_tags, *center, *axis, angle, numElements, heights,
                                                    recombine)
        except ValueError as err:
            message = "\nRevolving failed!\ndimtags: {}".format(str(self.dim_tags[:10]))
            gerr = gmsh_exceptions.BoolOperationError(message)
            self._raise_gmsh_exception(gerr, err)

        regions = [Region.default_region[dim] for dim, tag in outDimTags]
        all_obj = ObjectSet(self.factory, outDimTags, regions)

        self.factory._need_synchronize = True
        # split the Objectset by dimtags
        return all_obj.split_by_dimension()

    def copy(self) -> 'ObjectSet':
        """
        Problem: gmsh.model.occ.copy fails to copy boundary dimtags.
        """
        copy_tags = self.factory.model.copy(self.dim_tags)
        self.factory._need_synchronize = True
        copy_obj = ObjectSet(self.factory, copy_tags, self.regions)
        copy_obj.mesh_step_size = self.mesh_step_size.copy()
        return copy_obj

    def get_boundary(self, combined=False):
        """
        Get the boundary of the model entities dimTags.
        Return in outDimTags the boundary of the individual entities
        (if combined is false) or the boundary of the combined geometrical shape
        formed by all input entities (if combined is true).
        Return tags multiplied by the sign of the boundary entity if oriented is true.
        Apply the boundary operator recursively down to dimension 0 (i.e. to points) if recursive is true.

        derive_regions - if combined True, make derived boundary regions, other wise default regions are used
        combined=True ... omit fracture intersetions (boundary of combined object)
        combined=False ... give also intersections (boundary of indiviual objects)

        TODO: some support for oriented=True (returns signed tag according to its orientation)
              recursive=True (seems to provide boundary nodes)
        """
        self.factory.synchronize()
        try:
            dimtags = gmsh.model.getBoundary(self.dim_tags, combined=combined, oriented=False)
        except ValueError as err:
            message = "\nGetting boundary failed!\nobj dimtags: {}".format(str(self.dim_tags[:10]))
            gerr = gmsh_exceptions.GetBoundaryError(message)
            self._raise_gmsh_exception(gerr, err)
        regions = [Region.default_region[dim] for dim, tag in dimtags]
        return ObjectSet(self.factory, dimtags, regions)

    def split_by_region(self):
        """
        Split objects in ObjectSet into ObjectSets one per region.
        :return: list of ObjectSets
        TODO: Return Group
        """
        reg_to_tags = {}
        # collect tags of regions
        for dimtag, reg in self.dimtagreg():
            dim, tag = dimtag
            reg.complete(dim)
            reg_to_tags.setdefault(reg.id, (reg, []))
            reg_to_tags[reg.id][1].append(dimtag)
        reg_sets = [ObjectSet(self.factory, dimtags, [reg]) for reg, dimtags in reg_to_tags.values()]
        return reg_sets

    def split_by_dimension(self):
        """
        Split objects in ObjectSet into ObjectSets of same dimension.
        :return: list of ObjectSets
        TODO: add parameter dim: Union[Int,List[Int]] specify dimension to get
        Rename to 'get_dim'.
        """
        dimtags = [[], [], [], []]
        regions = [[], [], [], []]
        for dimtag, reg in self.dimtagreg():
            dim, tag = dimtag
            reg.complete(dim)
            dimtags[dim].append(dimtag)
            regions[dim].append(reg)
        sets = [ObjectSet(self.factory, dimtags, regs) for regs, dimtags in zip(regions, dimtags)]
        return sets

    def get_boundary_per_region(self, format=".{}"):
        """
        Split object by regions, call get_boundary for individual region subobjects and assign
        related boundary regions.
        :return:
        TODO: Return Group
        """

        reg_sets = self.split_by_region()
        b_sets = []
        for rset in reg_sets:
            reg = rset.regions[0]
            b_reg_name = format.format(reg.name)
            b_reg = Region.get(b_reg_name, dim=reg.dim - 1)
            # self.factory.get_region_name()
            boundary = rset.get_boundary(combined=True).set_region(b_reg)
            b_sets.append(boundary)

        return b_sets

    def have_common_dim(self, dim_tags=None):
        if dim_tags is None:
            dim_tags = self.dim_tags
        assert dim_tags
        dim = dim_tags[0][0]
        for d, tag in dim_tags:
            if d != dim:
                return None
        return dim

    def dimtagreg(self):
        assert len(self.regions) == len(self.dim_tags)
        return zip(self.dim_tags, self.regions)

    def mesh_step(self, step):
        """
        Saves the mesh step for all dimtags in this ObjectSet.
        The values are applied later when making mesh.

        Returns self.

        At the end, it will sort the dimtags by the mesh step size
        and set the mesh step from the largest to smallest.
        This resolves the problem with the recursion of the gmsh setSize function
        and  puts priority on the smaller mesh step.
        """
        self.mesh_step_size = [step for _ in self.dim_tags]
        return self

    def mesh_step_direct(self, step):
        """
        Set mesh step 'step' IMMEDIATELY to all nodes recursively to all dimtags in the ObjectSet.

        Use it carefully, only if you fully understand how this works.
        Otherwise use mesh_step().

        Returns self.

        TODO: be resistant to nonexistent dimtags
        """
        # Get boundary recursive to obtain nodes
        self.factory.synchronize()
        try:
            dimtags = gmsh.model.getBoundary(self.dim_tags, combined=False, oriented=False, recursive=True)
        except ValueError as err:
            message = "\nobj dimtags: {} ...".format(str(self.dim_tags[:10]))
            gerr = gmsh_exceptions.GetBoundaryError(message)
            self._raise_gmsh_exception(gerr, err)
        nodes = [(dim, tag) for dim, tag in dimtags if dim == 0]
        gmsh.model.mesh.setSize(nodes, step)
        return self

    def select_by_intersect(self, *tool_objects: 'ObjectSet') -> 'ObjectSet':
        """
        Make intersection with copy of the object
        :param tool_objects:
        :return:
        """
        sc = self.copy()
        tool = self.factory.group(*tool_objects).copy()
        objs, map = self.factory.model.intersect(sc.dim_tags, tool.dim_tags)
        tool.invalidate()
        sc.invalidate()

        isec = []
        for dimtag_map, dimtagreg in zip(map, self.dimtagreg()):
            if len(dimtag_map) > 1:
                message = "\nCannot select by intersect, insufficient fragmentation:\n{}".format(self.dim_tags)
                gerr = gmsh_exceptions.BoolOperationError(message)
                self._raise_gmsh_exception(gerr, None)
            if len(dimtag_map) == 1:
                isec.append(dimtagreg)
        if isec:
            dimtags, regs = zip(*isec)
        else:
            return ObjectSet(self.factory, [], [])
        return ObjectSet(self.factory, dimtags, regs)

    def split_by_cut(self, *tool_objects: 'ObjectSet') -> Tuple['ObjectSet', 'ObjectSet', 'ObjectSet', 'ObjectSet']:
        """
        Cut self object and return both cut object and the remainder object.
        Doesn't work properly for boundaries due to a bug i OCC.

        :param tool_objects: any number of ObjectSet
        :return: cut objectset, intersection objectset, tool remainder objectset
        TODO: Return Group
        """
        factory = self.factory
        tool_objects = self.factory.group(*tool_objects)
        new_obj, new_tool = self.factory.fragment(self, tool_objects)
        dict_obj = dict(new_obj.dimtagreg())
        dict_tool = dict(new_tool.dimtagreg())
        cut_obj = {k: dict_obj[k] for k in set(dict_obj) - set(dict_tool)}
        cut_tool = {k: dict_tool[k] for k in set(dict_tool) - set(dict_obj)}
        isec_set = set(dict_obj) & set(dict_tool)
        isec_obj = {k: dict_obj[k] for k in isec_set}
        isec_tool = {k: dict_tool[k] for k in isec_set}

        out_objs = [ObjectSet(factory, list(d.keys()), list(d.values()))
                    for d in [cut_obj, cut_tool, isec_obj, isec_tool]]
        return out_objs

    def set_region_from_dimtag(self):
        """
        Mainly for debugging purposes. Set new regions for every dimtag.
        :return:
        """
        regions = []
        for dim, tag in self.dim_tags:
            name = "{}_{}".format(dim, tag)
            regions.append(self.factory.get_region_name(name))
        self.regions = regions

    def _apply_operation(self, tool_objects, operation):
        tool_objects = self.factory.group(*tool_objects).copy()
        try:
            new_tags, old_tags_map = operation(self.dim_tags, tool_objects.dim_tags, removeObject=True, removeTool=True)
        except ValueError as err:
            message = "\nobj dimtags: {}\ntool dimtags: {}".format(str(self.dim_tags[:10]),
                                                                   str(tool_objects.dim_tags[:10]))
            gerr = gmsh_exceptions.BoolOperationError(message)
            self._raise_gmsh_exception(gerr, err)

        # assign regions
        assert len(self.regions) == len(self.dim_tags), (len(self.regions), len(self.dim_tags))
        old_tags_objects = []
        for reg, step, new_subtags in zip(self.regions, self.mesh_step_size, old_tags_map[:len(self.dim_tags)]):
            newobj = ObjectSet(self.factory, new_subtags, [reg])
            newobj.mesh_step(step)
            old_tags_objects.append(newobj)
        new_obj = self.factory.group(*old_tags_objects)

        # store auxiliary information
        # TODO: remove, should not be necessary
        # new_obj._previous_obj = self
        # new_obj._previous_dim_tags = self.dim_tags
        # new_obj._previous_map = old_tags_map

        # invalidate original objects
        self.factory._need_synchronize = True
        self.invalidate()
        tool_objects.invalidate()
        return new_obj

    def cut(self, *tool_objects) -> 'ObjectSet':
        """
        Cut self object with 'tool_objects'.
        Returns the cut object, self is destroyed, tool_objects are preserved (we use their copy).
        Regions set on self are transferred to the result.
        """
        return self._apply_operation(tool_objects, self.factory.model.cut)

    def intersect(self, *tool_objects) -> 'ObjectSet':
        """
        Intersect self object with 'tool_objects'.
        Returns the intersected object, self is destroyed, tool_objects are preserved (we use their copy).
        Regions set on self are transferred to the result.
        """
        return self._apply_operation(tool_objects, self.factory.model.intersect)

    def fragment(self, *tool_objects) -> 'ObjectSet':
        """
        Fragment self object with 'tool_objects'.
        Returns the fragmented object, self is destroyed, tool_objects are preserved (we use their copy).
        Regions set on self are transferred to the result.
        """
        return self._apply_operation(tool_objects, self.factory.model.fragment)

    def fuse(self, *tool_objects) -> 'ObjectSet':
        """
        Fuse self object with 'tool_objects'.
        Returns the fused object, self is destroyed, tool_objects are destroyed.
        Default regions are prescribed to all resulting dimtags.
        """
        # return self._apply_operation(tool_objects, self.factory.model.fuse)
        # tool_objects = self.factory.group(*tool_objects).copy()
        tool_objects = self.factory.group(*tool_objects)
        try:
            new_tags, old_tags_map = self.factory.model.fuse(self.dim_tags, tool_objects.dim_tags, removeObject=True,
                                                             removeTool=True)
        except ValueError as err:
            message = "Fusion failed!\nobj dimtags: {}\ntool dimtags: {}".format(str(self.dim_tags[:10]),
                                                                                 str(tool_objects.dim_tags[:10]))
            gerr = gmsh_exceptions.BoolOperationError(message)
            self._raise_gmsh_exception(gerr, err)

        # assign regions
        assert len(self.regions) == len(self.dim_tags), (len(self.regions), len(self.dim_tags))

        regions = [Region.default_region[dim] for dim, tag in new_tags]
        new_obj = ObjectSet(self.factory, new_tags, regions)
        self.factory._need_synchronize = True
        self.invalidate()
        tool_objects.invalidate()
        return new_obj

    def invalidate(self):
        self.factory = None
        self.dim_tags = None
        self.regions = None

    def mass(self):
        return sum((self.factory.model.getMass(*dimtag) for dimtag in self.dim_tags))

    def center_of_mass(self):
        center = np.zeros(3)
        mass_total = 0
        for dimtag in self.dim_tags:
            mass = self.factory.model.getMass(*dimtag)
            center += mass * np.array(self.factory.model.getCenterOfMass(*dimtag))
            mass_total += mass
        if mass_total > 0.0:
            return center / mass_total, mass_total
        else:
            return 0, 0

    def remove_small_mass(self, mass_limit):
        """
        Remove objects with the mass under the limit.
        """
        masses = [self.factory.model.getMass(*dt) for dt in self.dim_tags]
        # for dt, mass in zip(self.dim_tags, masses):
        #    print(dt, mass)

        dimtags = [dt for dt, m in zip(self.dim_tags, masses) if m > mass_limit]
        regions = [r for r, m in zip(self.regions, masses) if m > mass_limit]
        self.dim_tags = dimtags
        self.regions = regions
        return self


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
