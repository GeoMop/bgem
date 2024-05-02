"""
Module for representation and processing of a set of fractures, i.e. a single DFN sample.

Should include:
- Baseclasses for fracture shapes: EllipseShape, RectangleShape, PolygonShape
- Representation of the single discrete fracture network sample: FractureSet
  Use vectorized storage, but could extract single fractures for backward compatibility.
- FractureField

Class dedicated to fractire intersections, simplification, meshing.
- creation of the BREP model from the list of fractures (deal with intersections)
- fracture network manipulations and simplifications to allow good meshing

TODO:
1. Just collection of fractures.
2. Fracture properties - conductivity model on a single fracture
   - conductivity = alpha * r ** beta
   - apperture = 12 * sqrtp(alpha * r**beta)

   ... possible extensions to heterogenous models
.. TO Be Done
"""
from typing import *
from nptyping import NDArray, Shape, Float, Int32

import attrs
import numpy as np

from bgem import fn
from .fracture import Fracture, SquareShape

"""
Reference fracture shapes. 
Placed in XY plane and with isotropic shape.

Different shapes should have the surface area same as the unit disc 
in order to be comparable in density (not necesarily in the connectivity).
"""
class BaseShape:
    """
    Abstract class.
    """

    pass


class LineShape(BaseShape):
    pass

class EllipseShape(BaseShape):
    """
    Reference fracture shape  - unit disc.
    """
    def is_point_inside(self, x, y):
        return x**2 + y**2 <= 1

    def are_points_inside(self, points):
        sq = points ** 2
        return sq[:, 0] + sq[:, 1]  <= 1

    def aabb(self):
        """
        Size of the bounding box for any rotation of the reference shape.
        For an isotropic reference shape we have a bounding box (-D,+D) x (-D,+D)
        :return: D - half of the box size
        """
        return 1.0

class RectangleShape(BaseShape):
    """
    Reference square shape.
    """
    def __init__(self):
        """
        Initializes a RegularPolygon instance for an N-sided polygon.

        Args:
        - N: Number of sides of the regular polygon.
        """
        # Square with area of unit disc.
        self.half_side = np.sqrt(np.pi) / 2

    def is_point_inside(self, x, y):
        """
        Tests if a point (x, y) is inside the regular N-sided polygon.

        Args:
        - x, y: Coordinates of the point to test.

        Returns:
        - True if the point is inside the polygon, False otherwise.
        """
        return (math.abs(x) < self.half_side) and (math.abs(y) < self.half_side)

    def are_points_inside(self, points):
        """
        Tests if points in a NumPy array are inside the regular N-sided polygon.
        Args:
        - points: A 2D NumPy array of shape (M, 2), where M is the number of points
          and each row represents a point (x, y).
        Returns:
        - A boolean NumPy array where each element indicates whether the respective
          point is inside the polygon.
        """
        return np.max(np.abs(points), axis=1) < self.half_side

    def aabb(self):
        """
        Size of the bounding box for any rotation of the reference shape.
        For an isotropic reference shape we have a bounding box (-D,+D) x (-D,+D)
        :return: D - half of the box size
        """
        return np.sqrt(2.0)

class PolygonShape(BaseShape):

    @classmethod
    def disc_approx(cls, x_scale, y_scale, step=1.0):
        n_sides = np.pi * min(x_scale, y_scale) / step
        n_sides = max(4, n_sides)
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        points = np.stack(np.cos(angles) * x_scale, np.sin(angles) * y_scale, np.ones_like(angles))
        return points

    def __init__(self, N):
        """
        Initializes a RegularPolygon instance for an N-sided polygon.

        Args:
        - N: Number of sides of the regular polygon.
        """
        self.N = N
        self.theta_segment = 2 * math.pi / N  # Angle of each segment
        self.R_inscribed = math.cos(self.theta_segment / 2)  # Radius of inscribed circle for R=1

    def is_point_inside(self, x, y):
        """
        Tests if a point (x, y) is inside the regular N-sided polygon.

        Args:
        - x, y: Coordinates of the point to test.

        Returns:
        - True if the point is inside the polygon, False otherwise.
        """
        r = math.sqrt(x**2 + y**2)  # Convert point to polar coordinates (radius)
        theta = math.atan2(y, x)  # Angle in polar coordinates

        # Compute the reminder of the angle and the x coordinate of the reminder point
        theta_reminder = theta % self.theta_segment
        x_reminder = math.cos(theta_reminder) * r

        # Check if the x coordinate of the reminder point is less than
        # the radius of the inscribed circle (for R=1)
        return x_reminder <= self.R_inscribed

    def are_points_inside(self, points):
        """
        Tests if points in a NumPy array are inside the regular N-sided polygon.
        Args:
        - points: A 2D NumPy array of shape (M, 2), where M is the number of points
          and each row represents a point (x, y).
        Returns:
        - A boolean NumPy array where each element indicates whether the respective
          point is inside the polygon.
        """
        r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        theta = np.arctan2(points[:, 1], points[:, 0])
        theta_reminder = theta % self.theta_segment
        x_reminder = np.cos(theta_reminder) * r
        return x_reminder <= self.R_inscribed

    def aabb(self):
        """
        Size of the bounding box for any rotation of the reference shape.
        For an isotropic reference shape we have a bounding box (-D,+D) x (-D,+D)
        :return: D - half of the box size
        TODO: more precise estimate, improve for radius scaling according to
        same surface ar the Ellipse case.
        """
        return np.sqrt(2.0)



__base_shapes = [LineShape, EllipseShape, RectangleShape, PolygonShape]
__shape_ids = {shape:i for i, shape in enumerate(__base_shapes)}

# class LineShape:
#     """
#     Class represents the reference line 2D fracture shape.
#
#     The polymorphic `make_approx` method is used to create polygon (approximation in case of disc) of the
#     actual fracture.
#     """
#     _points = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
#
#     @classmethod
#     def make_approx(cls, x_scale, y_scale, step=None):
#         xy_scale = np.array([x_scale, y_scale, 1.0])
#         return cls._points[:, :] * xy_scale[None, :]


# class SquareShape(LineShape):
#     """
#     Class represents the square fracture shape.
#     """
#     _points = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])


# class DiscShape:
#     """
#     Class represents the square fracture shape.
#     """
#
#     @classmethod
#     def make_approx(cls, x_scale, y_scale, step=1.0):
#         n_sides = np.pi * min(x_scale, y_scale) / step
#         n_sides = max(4, n_sides)
#         angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
#         points = np.stack(np.cos(angles) * x_scale, np.sin(angles) * y_scale, np.ones_like(angles))
#         return points








@attrs.define
class Fracture:
    """
    Single fracture sample.
    TODO: modify to the acessor into the FrSet objects.
    """
    shape_class: Any
    # Basic fracture shape.
    r: float
    # Fracture diameter, laying in XY plane
    center: np.array
    # location of the barycentre of the fracture
    normal: np.array
    # fracture normal
    shape_angle: float
    # angle to rotate the unit shape around z-axis; rotate anti-clockwise
    region_id: int # Union[str, int] = "fracture"
    # name or ID of the physical group
    family: 'FrFamily' = None
    # Original family, None if created manually (in tests)
    aspect: float = 1
    # aspect ratio of the fracture =  y_length / x_length where  x_length == r
    id: Any = None
    # any value associated with the fracture (DEPRECATED should be replaced by
    # FrValue class and fr_mesh code

    _rotation_axis: np.array = attrs.field(init=False, default=None)
    # axis of rotation
    _rotation_angle: float = attrs.field(init=False, default=None)
    # angle of rotation around the axis (?? counterclockwise with axis pointing up)
    _distance: float = attrs.field(init=False, default=None)
    # absolute term in plane equation
    _plane_coor_system: np.array = attrs.field(init=False, default=None)
    # local coordinate system
    _vertices: np.array = attrs.field(init=False, default=None)
    # coordinates of the vertices
    _ref_vertices: np.array = attrs.field(init=False, default=None)
    # local coordinates of the vertices (xy - plane)


    @property
    def vertices(self):
        if self._vertices is None:
            _vertices = self.transform(self.shape_class._points)
        return _vertices

    @property
    def ref_vertices(self):
        if self._ref_vertices is None:
            _ref_vertices = self.shape_class._points
        return _ref_vertices

    @property
    def rx(self):
        return self.r

    @property
    def ry(self):
        return self.r * self.aspect

    @property
    def scale(self):
        return [self.r, self.r * self.aspect]

    @property
    def rotation_angle(self):
        if self._rotation_angle is None:
            _rotation_axis, _rotation_angle = self.axis_angle()
        return _rotation_angle

    @property
    def rotation_axis(self):
        if self._rotation_axis is None:
            _rotation_axis, _rotation_angle = self.axis_angle()
        return _rotation_axis

    def axis_angle(self):
        axis, angle = normal_to_axis_angle(self.normal)
        return axis, angle

    def axis_angles(self):
        axis_angle = normals_to_axis_angles([self.normal])[0,:]
        _rotation_axis = axis_angle[0:3]
        _rotation_angle = axis_angle[3]
        return _rotation_axis, _rotation_angle

    @property
    def distance(self):
        if self._distance is None:
            _distance = -np.dot(self.center, self.normal[0, :])
        return _distance

    @property
    def plane_coor_system(self):
        if self._plane_coor_system is None:
            _plane_coor_system = self.transform(np.array([[1.0, 0, 0], [0, 1.0 ,0]]))
        return _plane_coor_system

    def get_angle_with_respect_normal(self,vec):

        dot = self.normal[0] * vec[0] + self.normal[1] * vec[1] + self.normal[2] * vec[2]
        angle = np.arccos((dot)/np.linalg(vec))

        return angle

    def internal_point_2d(self, points):
        """
        Determines the interior points of the fracture.
        :param points: array (3,n)
        :return:
        polygon_points as list of int: indices od of the interior points in points
        """
        polygon_points = []
        for i in range(0,points.shape[0]):
            #eps = abs(self.normal[0,0] * points[i,0] + self.normal[0,1] * points[i,1] + self.normal[0,2] * points[i,2] - self.distance)\
            #      / math.sqrt(np.linalg.norm(self.normal)**2 + self.distance**2 )
            #if eps < 1e-15:
            #    continue

            dot = np.zeros((self.ref_vertices.shape[0]))
            for j in range(-1, self.ref_vertices.shape[0]-1):
                bound_vec = self.ref_vertices[j+1] - self.ref_vertices[j]
                sec_vec = self.ref_vertices[j+1] - points[i,:]
                dot[j+1] = bound_vec[0]*sec_vec[0] + bound_vec[1]*sec_vec[1] + bound_vec[2]*sec_vec[2]

            if np.sum(dot>0) == self.ref_vertices.shape[0] or np.sum(dot<0) == self.ref_vertices.shape[0]:
                polygon_points.append(i)

        #if polygon_points == []:
        #     polygon_points = None

        return polygon_points

    def dist_from_plane(self,point):
     """
        Computes distance from plane
        :param point: array (3,)
        :return: distance as double
        """

     dist = self.normal[0,0] * point[0] + self.normal[0,1] * point[1] + self.normal[0,2] * point[2] + self.distance
     return dist


    def get_isec_with_line(self, x_0, loc_direct):
        """
        Computes intersection of the fracture and line x0 + t*loc_direct (in local coordinates).
        :param x0: array (3,)
        :param loc_direct: array (3,)
        :return:
        x_isec as list of array (3,): real intersection points
        x_isec_false as list of array (3,): intersection points outside the edge of the fracture
        x_isec_start_vert_ind as list of int: index of the nearest initial point of the false intersection point
        """

        x_isec = []
        x_isec_false = []
        x_isec_start_vert_ind = []

        bound_vec = np.zeros(self.shape_class._points.shape)
        x_0_b = np.zeros(self.shape_class._points.shape)

        aspect = np.array([self.r, self.aspect * self.r, 1], dtype=float) # 0.5 *
        points = self.shape_class._points #* aspect[None, :]  # self.shape_class._points



        col2 = loc_direct[0]
        for i in range(0, self.shape_class._points.shape[0]-1):
            col1 = points[i]  - points[i-1]
            rhs = (x_0 - points[i-1])[0]
            det = col1[0] * col2[1] - col1[1] * col2[0]
            det_x1 = rhs[0] * col2[1] - rhs[1] * col2[0]
            #colinear intersections (joins) should be solved in a different way
            if abs(det) > 0:
                t = det_x1/det
                if (t >= 0.0) and (t <= 1.0):
                    x_isec.append(x_0_b[i] + col1 * t)
                else:
                    x_isec_false.append(x_0_b[i] + col1 * t)
                if (t < 0.0):
                    x_isec_start_vert_ind.append(i-1)
                elif (t > 1.0):
                    x_isec_start_vert_ind.append(i)
            else:
                if (i - 1) not in x_isec_start_vert_ind:
                    x_isec_start_vert_ind.append(i-1)
                    x_isec_false.append([])
                if (i) not in x_isec_start_vert_ind:
                    x_isec_start_vert_ind.append(i)
                    x_isec_false.append([])

        return x_isec, x_isec_false, x_isec_start_vert_ind

    def transform(self, points):
        """
        Map local points on the fracture to the 3d scene.
        :param points: array (n, 3)
        :return: transformed points
        """
        aspect = np.array([self.r, self.aspect * self.r, 1], dtype=float)
        t_points= points  * aspect[None, :] #[:, :]
        #points[:, :] *= aspect[:,None]
        t_points = rotate(t_points, np.array([0, 0, 1]), self.shape_angle)
        t_points = rotate(t_points, self.rotation_axis, self.rotation_angle)
        t_points += self.center[None, :]
        return t_points

    def back_transform(self, points):
        """
        Map points from 3d scene into local coordinate system.
        :param points: array (n, 3)
        :return: transformed points
        """
        aspect = np.array([self.r, self.aspect * self.r, 1], dtype=float)
        t_points = points - self.center[None, :]
        t_points = rotate(t_points, self.rotation_axis, -self.rotation_angle)
        t_points = rotate(t_points, np.array([0, 0, 1]), -self.shape_angle)
        t_points /= aspect[None, :]
        return t_points


    def transform_clear(self, points):
        """
        Map local points on the fracture to the 3d scene.
        :param points: array (n, 3)
        :return: transformed points
        """
        aspect = np.array([self.r, self.aspect * self.r, 1], dtype=float)
        t_points= points  * aspect[None, :] #[:, :]
        #points[:, :] *= aspect[:,None]
        t_points = rotate(t_points, np.array([0, 0, 1]), self.shape_angle)
        t_points = rotate(t_points, self.rotation_axis, self.rotation_angle)
        #t_points += self.centre[None, :]
        return t_points

    def back_transform_clear(self, points):
        """
        Map points from 3d scene into local coordinate system.
        :param points: array (n, 3)
        :return: transformed points
        """
        aspect = np.array([self.r, self.aspect * self.r, 1], dtype=float)
        #t_points = points - self.centre[None, :]
        t_points = rotate(points, self.rotation_axis, -self.rotation_angle)
        t_points = rotate(t_points, np.array([0, 0, 1]), -self.shape_angle)
        t_points /= aspect[None, :]
        return t_points






def array_attr(shape, dtype=Float, default=[]):
    return attrs.field(
        type=NDArray[Shape[shape], dtype],
        converter=np.array,
        default=default)


@attrs.define
class FractureSet:
    """
    Interface to the array based storage for the fractures.
    Has given outer box domain.

    The 1D fractures in 2D are treated as 2D fractures in 3D but:
    - centers have z=0
    - normals have z=0
    - shape_angle = 0
    - shape is two point line segment
    - r[0,:] = r[1,:]
    """

    #domain: NDArray[Shape['3'], Float]            # Box given by (3,) shape array
    #base_shapes : List[Any]                 # Unique reference shape classes

    shape_idx = array_attr((attrs.field(type=NDArray[Any, Int32], converter=np.array)          # Base shape type index into 'base_shapes' list.
    radius = attrs.field(type=NDArray[Any, Float], converter=np.array)             # shape (2, n_fractures), X and Y scaling of the reference shape.
    center = attrs.field(type=NDArray[Any, Float], converter=np.array)              # center (3, n_fractures); translation of the reference shape to actual position
    normal = attrs.field(type=NDArray[Any, Float], converter=np.array)              # fracture unit normal vectors.

    shape_axis = attrs.field(type=NDArray[Any, Float], converter=np.array)     # X reference unit vector in XY plane (2, n_fractures)
    family: NDArray[Any, Int32] = None         # index of the fracture family

    population: 'Population' = None         # Generating population. Gives meaning to fr family indices.

    @property
    def base_shapes(self):
        return self.__module__.__base_shapes

    @classmethod
    def parallel_plates(cls, box, normal, shift=0):
        """
        Construct parallel fractures  covering the box given normal and separation given by the normal length.
        The optional shift parameter provides shift po the fracture grid with respect to the origin.
        :param box:
        :param normal:

        :return:
        """
        box = np.array(box)
        normal = np.array(normal)
        diag = np.linalg.norm(box)
        separation = np.linalg.norm(normal)
        n_fr = int(diag / separation) + 1
        shift = shift % separation
        plates = [
            Fracture(
                SquareShape,
                r=diag,
                center=(i - n_fr // 2) * normal + shift * normal + box / 2.0,
                normal=normal / separation,
                shape_angle=0,
                region_id=0)
            for i in range(n_fr)
        ]
        return cls.from_list(plates)

    @classmethod
    def from_list(cls, fr_list: List[Fracture]):
        """
        Construct a fracture set from a list of 'Fracture' objects.
        :param fr_list:
        :return:
        """
        fr_attribute = lambda attr : [getattr(fr, attr) for fr in fr_list]
        shape_class_list = fr_attribute('shape_class')
        #base_shape = {}
        shape_idx = [cls.shape_ids[sc] for sc in shape_class_list]
        shape_angle = np.array(fr_attribute('shape_angle'))
        shape_axis = np.stack((np.cos(shape_angle), np.sin(shape_angle)))

        return cls(
            shape_idx,
            radius=fr_attribute('r'),
            center=fr_attribute('center'),
            normal=fr_attribute('normal'),
            shape_axis=shape_axis,
            family=fr_attribute('region_id')
        )


    @classmethod
    def merge(cls, fr_sets: List['FractureSet']) -> 'FractureSet':
        return cls(
            base_shape,
            shape_idx,
            radius=fr_attribute('r'),
            center=fr_attribute('center'),
            normal=fr_attribute('normal'),
            shape_axis=shape_axis,
            family=fr_attribute('region_id')
        )



    @property
    def size(self):
        return len(self.normal)

    @fn.cached_property
    def AABB(self):
        """
        Axis Aligned Bounding Box for each fracture.
        :return:
        """
        max_radii = np.max
        min_corner = np.min(self.center, axis=0)
        max_corner = np.max(self.center, axis=0)


    @fn.cached_property
    def transform_mat(self):
        """
        Rotate and scale matrices for the fractures. The full transform involves 'self.center' as well:
        ambient_space_points = self.center + self.transform_mat @ local_fr_points[:, None, :]
        :return: Shape (N, 3, 3).
        """
        x_vec = self.shape_axis
        z_vec = self.normal
        y_vec = np.cross(z_vec, x_vec, axis=1)
        return np.stack((x_vec, y_vec, z_vec), axis=1)

@attrs.define
class FractureValues:
    """
    A quantities on the fracture set, one value for each fracture, constant on the fracture.
    """
    fractures: FractureSet
    values: NDArray[Any, Float]

"""
Some Fracture Values Operations
"""
def fr_values_permeability():
        pass

class Fractures:
    """
    Stub of the class for fracture network simplification.
    New approach should be:
    - 2D meshing by GMSH
    - Healing with specific processing to deal properties of merged fractures.
    """
    # regularization of 2d fractures
    def __init__(self, fractures, epsilon):
        self.epsilon = epsilon
        self.fractures = fractures
        self.points = []
        self.lines = []
        self.pt_boxes = []
        self.line_boxes = []
        self.pt_bih = None
        self.line_bih = None
        self.fracture_ids = []
        # Maps line to its fracture.

        self.make_lines()
        self.make_bihs()

    def make_lines(self):
        # sort from large to small fractures
        self.fractures.sort(key=lambda fr:fr.rx, reverse=True)
        base_line = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
        for i_fr, fr in enumerate(self.fractures):
            line = FisherOrientation.rotate(base_line * fr.rx, np.array([0, 0, 1]), fr.shape_angle)
            line += fr.center
            i_pt = len(self.points)
            self.points.append(line[0])
            self.points.append(line[1])
            self.lines.append((i_pt, i_pt+1))
            self.fracture_ids.append(i_fr)

    def get_lines(self, fr_range):
        lines = {}
        fr_min, fr_max = fr_range
        for i, (line, fr) in enumerate(zip(self.lines, self.fractures)):
            if fr_min <= fr.rx < fr_max:
                lines[i] = [self.points[p][:2] for p in line]
        return lines

    def make_bihs(self):
        import bih
        shift = np.array([self.epsilon, self.epsilon, 0])
        for line in self.lines:
            pt0, pt1 = self.points[line[0]], self.points[line[1]]
            b0 = [(pt0 - shift).tolist(), (pt0 + shift).tolist()]
            b1 = [(pt1 - shift).tolist(), (pt1 + shift).tolist()]
            box_pt0 = bih.AABB(b0)
            box_pt1 = bih.AABB(b1)
            line_box = bih.AABB(b0 + b1)
            self.pt_boxes.extend([box_pt0, box_pt1])
            self.line_boxes.append(line_box)
        self.pt_bih = bih.BIH()
        self.pt_bih.add_boxes(self.pt_boxes)
        self.line_bih = bih.BIH()
        self.line_bih.add_boxes(self.line_boxes)
        self.pt_bih.construct()
        self.line_bih.construct()

    def find_root(self, i_pt):
        i = i_pt
        while self.pt_map[i] != i:
            i = self.pt_map[i]
        root = i
        i = i_pt
        while self.pt_map[i] != i:
            j = self.pt_map[i]
            self.pt_map[i] = root
            i = j
        return root

    def snap_to_line(self, pt, pt0, pt1):
        v = pt1 - pt0
        v /= np.linalg.norm(v)
        t = v @ (pt - pt0)
        if 0 < t < 1:
            projected = pt0 + t * v
            if np.linalg.norm(projected - pt) < self.epsilon:
                return projected
        return pt



    def simplify(self):
        """
        Kruskal algorithm is somehow used to avoid loops in line createion.
        :return:
        """
        self.pt_map = list(range(len(self.points)))
        for i_pt, point in enumerate(self.points):
            pt = point.tolist()
            for j_pt_box in  self.pt_bih.find_point(pt):
                if i_pt != j_pt_box and j_pt_box == self.pt_map[j_pt_box] and self.pt_boxes[j_pt_box].contains_point(pt):
                    self.pt_map[i_pt] = self.find_root(j_pt_box)
                    break
        new_lines = []
        new_fr_ids = []
        for i_ln, ln in enumerate(self.lines):
            pt0, pt1 = ln
            pt0, pt1 = self.find_root(pt0), self.find_root(pt1)
            if pt0 != pt1:
                new_lines.append((pt0, pt1))
                new_fr_ids.append(self.fracture_ids[i_ln])
        self.lines = new_lines
        self.fracture_ids = new_fr_ids

        for i_pt, point in enumerate(self.points):
            if self.pt_map[i_pt] == i_pt:
                pt = point.tolist()
                for j_line in self.line_bih.find_point(pt):
                    line = self.lines[j_line]
                    if i_pt != line[0] and i_pt != line[1] and self.line_boxes[j_line].contains_point(pt):
                        pt0, pt1 = self.points[line[0]], self.points[line[1]]
                        self.points[i_pt] = self.snap_to_line(point, pt0, pt1)
                        break

    def line_fragment(self, i_ln, j_ln):
        """
        Compute intersection of the two lines and if its position is well in interior
        of both lines, benote it as the fragmen point for both lines.
        """
        pt0i, pt1i = (self.points[ipt] for ipt in self.lines[i_ln])
        pt0j, pt1j = (self.points[ipt] for ipt in self.lines[j_ln])
        A = np.stack([pt1i - pt0i, -pt1j + pt0j], axis=1)
        b = -pt0i + pt0j
        ti, tj = np.linalg.solve(A, b)
        if self.epsilon <= ti <= 1 - self.epsilon and self.epsilon <= tj <= 1 - self.epsilon:
            X = pt0i + ti * (pt1i - pt0i)
            ix = len(self.points)
            self.points.append(X)
            self._fragment_points[i_ln].append((ti, ix))
            self._fragment_points[j_ln].append((tj, ix))

    def fragment(self):
        """
        Fragment fracture lines, update map from new line IDs to original fracture IDs.
        :return:
        """
        new_lines = []
        new_fracture_ids = []
        self._fragment_points = [[] for l in self.lines]
        for i_ln, line in enumerate(self.lines):
            for j_ln in self.line_bih.find_box(self.line_boxes[i_ln]):
                if j_ln > i_ln:
                    self.line_fragment(i_ln, j_ln)
            # i_ln line is complete, we can fragment it
            last_pt = self.lines[i_ln][0]
            fr_id = self.fracture_ids[i_ln]
            for t, ix in sorted(self._fragment_points[i_ln]):
                new_lines.append(last_pt, ix)
                new_fracture_ids.append(fr_id)
                last_pt = ix
            new_lines.append(last_pt, self.lines[i_ln][1])
            new_fracture_ids.append(fr_id)
        self.lines = new_lines
        self.fracture_ids = new_fracture_ids





    # def compute_transformed_shapes(self):
    #     n_frac = len(self.fractures)
    #
    #     unit_square = unit_square_vtxs()
    #     z_axis = np.array([0, 0, 1])
    #     squares = np.tile(unit_square[None, :, :], (n_frac, 1, 1))
    #     center = np.empty((n_frac, 3))
    #     trans_matrix = np.empty((n_frac, 3, 3))
    #     for i, fr in enumerate(self.fractures):
    #         vtxs = squares[i, :, :]
    #         vtxs[:, 1] *= fr.aspect
    #         vtxs[:, :] *= fr.r
    #         vtxs = FisherOrientation.rotate(vtxs, z_axis, fr.shape_angle)
    #         vtxs = FisherOrientation.rotate(vtxs, fr.rotation_axis, fr.rotation_angle)
    #         vtxs += fr.centre
    #         squares[i, :, :] = vtxs
    #
    #         center[i, :] = fr.centre
    #         u_vec = vtxs[1] - vtxs[0]
    #         u_vec /= (u_vec @ u_vec)
    #         v_vec = vtxs[2] - vtxs[0]
    #         u_vec /= (v_vec @ v_vec)
    #         w_vec = FisherOrientation.rotate(z_axis, fr.rotation_axis, fr.rotation_angle)
    #         trans_matrix[i, :, 0] = u_vec
    #         trans_matrix[i, :, 1] = v_vec
    #         trans_matrix[i, :, 2] = w_vec
    #     self.squares = squares
    #     self.center = center
    #     self.trans_matrix = trans_matrix
    #
    # def snap_vertices_and_edges(self):
    #     n_frac = len(self.fractures)
    #     epsilon = 0.05  # relaitve to the fracture
    #     min_unit_fr = np.array([0 - epsilon, 0 - epsilon, 0 - epsilon])
    #     max_unit_fr = np.array([1 + epsilon, 1 + epsilon, 0 + epsilon])
    #     cos_limit = 1 / np.sqrt(1 + (epsilon / 2) ** 2)
    #
    #     all_points = self.squares.reshape(-1, 3)
    #
    #     isec_condidates = []
    #     wrong_angle = np.zeros(n_frac)
    #     for i, fr in enumerate(self.fractures):
    #         if wrong_angle[i] > 0:
    #             isec_condidates.append(None)
    #             continue
    #         projected = all_points - self.center[i, :][None, :]
    #         projected = np.reshape(projected @ self.trans_matrix[i, :, :], (-1, 4, 3))
    #
    #         # get bounding boxes in the loc system
    #         min_projected = np.min(projected, axis=1)  # shape (N, 3)
    #         max_projected = np.max(projected, axis=1)
    #         # flag fractures that are out of the box
    #         flag = np.any(np.logical_or(min_projected > max_unit_fr[None, :], max_projected < min_unit_fr[None, :]),
    #                       axis=1)
    #         flag[i] = 1  # omit self
    #         candidates = np.nonzero(flag == 0)[0]  # indices of fractures close to 'fr'
    #         isec_condidates.append(candidates)
    #         # print("fr: ", i, candidates)
    #         for i_fr in candidates:
    #             if i_fr > i:
    #                 cos_angle_of_normals = self.trans_matrix[i, :, 2] @ self.trans_matrix[i_fr, :, 2]
    #                 if cos_angle_of_normals > cos_limit:
    #                     wrong_angle[i_fr] = 1----
    #                     print("wrong_angle: ", i, i_fr)
    #
    #                 # atract vertices
    #                 fr = projected[i_fr]
    #                 flag = np.any(np.logical_or(fr > max_unit_fr[None, :], fr < min_unit_fr[None, :]), axis=1)
    #                 print(np.nonzero(flag == 0))


def fr_intersect(fractures):
    """
    1. create fracture shape vertices (rotated, translated) square
        - create vertices of the unit shape
        - use FisherOrientation.rotate
    2. intersection of a line with plane/square
    3. intersection of two squares:
        - length of the intersection
        - angle
        -
    :param fractures:
    :return:
    """

    # project all points to all fractures (getting local coordinates on the fracture system)
    # fracture system axis:
    # u_vec = vtxs[1] - vtxs[0]
    # v_vec = vtxs[2] - vtxs[0]
    # w_vec ... unit normal
    # fractures with angle that their max distance in the case of intersection
    # is not greater the 'epsilon'



