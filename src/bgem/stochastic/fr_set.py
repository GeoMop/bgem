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
import pathlib
from typing import *

import attrs
import math
import numpy as np
import numpy.typing as npt

from bgem import fn
from bgem.core import array_attr
from bgem.bspline import brep_writer as bw

def embed_to_3d(points_2d):
    points_3d = np.concatenate((
        points_2d,
        np.zeros(len(points_2d))[:, None]
    ), axis=1)
    return points_3d

"""
Reference fracture shapes. 
Placed in XY plane and with isotropic shape.

Different shapes should have the surface area same as the unit disc 
in order to be comparable in density (not necesarily in the connectivity).
"""
class BaseShape:
    """
    Abstract class.
    All subclasses should represent shpapes with the area equal to 1.0.
    TODO: check that aabb and unit area asre satisfied for individual shapes.
    """

    @staticmethod
    def shape_for_id(id: int):
        if id == 0:
            return EllipseShape()
        elif id == 1 or id == 4:
            return RectangleShape()
        elif id == 2:
            return LineShape()
        else:
            return PolygonShape(id)


    @property
    def aabb(self):
        """
        Size of the bounding box for any rotation of the reference shape.
        For an isotropic reference shape we have a bounding box (-D,+D) x (-D,+D)
        This method provides conservative generic AABB. Shape specific implementation
        could be provided.
        :return: D - half of the box size
        """
        half_size = np.ones(2) * self.R
        return np.stack([-half_size, half_size])

    @staticmethod
    def ellipses_aabb(self, centers, a_vectors, b_vectors):
        """
        Compute AABBs for ellipses given by `centers` and semiaxis vectors `a_vectors`, `b_vectors`.
        Inputs shape: (n_fractures, 3)
        Based on finding extremes of the parametric equations.
        :return: AABBs array , shape (n_fracutres, 2, 3)
        AABB of fracture i has min corner AABB[i, 0, :] and max corner AABB[i, 1, :]
        """
        # Number of fractures
        n_fractures = centers.shape[0]

        # Calculate theta values for critical points
        theta = np.arctan2(b_vectors, a_vectors)

        # Compute cos(theta) and sin(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Parametric equations evaluated at critical points
        points_1 = cos_theta * a_vectors + sin_theta * b_vectors,
        points_2 = -cos_theta * a_vectors - sin_theta * b_vectors
        extrem_points = np.stack((
            np.minimum(points_1, points_2),
            np.maximum(points_1, points_2)),
            axis=1
        )

        return extrem_points + centers[:, None, :]


    def are_points_inside(self, points):
        """
        Virtual method.
        :param points: shape (N, 2)
        :return: logical array (N,)
        """
        pass

    @staticmethod
    def disc_approx(n_sides=8, scale=(1.0, 1.0)):
        """
        Return (n_sides, 3) array with coordinates of regular polygon
        inscribed to the circle with radius 'scale'.
        Z coordinate set to 0.
        :param n_sides:
        :param scale: radius of circle or two radii (r_x, r_y) of the ellipse
        :return: array of (n_sides, 3) shape
        """
        scale = scale * np.ones(2)
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        points = np.stack((
            np.cos(angles) * scale[0],
            np.sin(angles) * scale[1],
            np.zeros_like(angles)
        ), axis=1)
        return points

    def vertices(self, n_sides=None):
        """
        Return 3d coordinates of the fracture polygon or its approximation for ellipse.
        :param n_sides: only used by ellipse implementation,
        :return: ndarray (n_sides, 3)
        """
        points_3d = self.disc_approx(self.n_sides, scale=self.R)

        return points_3d

class LineShape(BaseShape):
    """
    Does not fit to 3D conceptually. Introduce carefully once 3D case API is properly designed and tested.
    """
    id = 2


    def gmsh_base_shape(self, gmsh_geom: 'GeometryOCC'):
        return gmsh_geom.line([-0.5, 0, 0], [0.5, 0, 0])

class EllipseShape(BaseShape):
    """
    Disc base fracture shape.
    """

    id = 0

    def __init__(self):
        self.n_sides = np.inf
        self.R = 1 / math.sqrt(math.pi)
        self.r = self.R
        # Radius of the reference disc of unit area
        self._scale_sqr = 1 / math.pi
        # Faster identification of inside points.

    def is_point_inside(self, x, y):
        return x**2 + y**2 <= self._scale_sqr

    def are_points_inside(self, points):
        sq = points ** 2
        return sq[:, 0] + sq[:, 1]  <= self._scale_sqr

    def gmsh_base_shape(self, gmsh_geom: 'GeometryOCC'):
        return gmsh_geom.disc(rx=self.scale, ry=self.scale)

    def vertices(self, n_sides=8):
        """
        Approximate `n_sides` polygon of the unit area.
        :return: ndarray (n_sides, 3)
        """
        return PolygonShape(n_sides).vertices()


class RectangleShape(BaseShape):
    """
    Reference square shape with area 1.0 and center at origin.
    """
    id = 1

    def __init__(self):
        """
        Initializes a RegularPolygon instance for an N-sided polygon.

        Args:
        - N: Number of sides of the regular polygon.
        """
        # Square with area of unit disc.
        self.R = 1/math.sqrt(2)
        # Radius of circumcircle for square of unit area.
        # S = 4 * sin(45)* cos(45) * R^2 = 4 * 1/2 * 1/sqrt(2)^2 = 1
        self.r = 0.5
        # Radius of inscribed circle for square of unit area.
        self.n_sides = 4

    def is_point_inside(self, x, y):
        """
        Tests if a point (x, y) is inside the regular N-sided polygon.

        Args:
        - x, y: Coordinates of the point to test.

        Returns:
        - True if the point is inside the polygon, False otherwise.
        """
        return (abs(x) < self.r) and (abs(y) < self.r)

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
        return np.max(np.abs(points), axis=1) < self.r

    def gmsh_base_shape(self, gmsh_geom: 'GeometryOCC'):
        return gmsh_geom.rectangle()

    def vertices(self, n_sides=8):
        """
        PolygonShape(4) provides square in diamond position.
        We need edges parallel with axis to provide fast interior indicator.
        :return: ndarray (n_sides, 3)
        """
        return self.r * np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]] )


class PolygonShape(BaseShape):

    def __init__(self, N):
        """
        Initializes a RegularPolygon instance for an N-sided polygon.

        Area S = N * sin th/2 * cos th/2 * R^2 = 0.5 * N * sin th * R^2
        Args:
        - N: Number of sides of the regular polygon.
        """
        assert N > 4

        self.n_sides = N
        self.theta_segment_half = math.pi / N       # half angle of each segment
        self.cos_theta = math.cos(self.theta_segment_half)
        self.R = 1 / math.sqrt(0.5 * N * math.sin(2 * self.theta_segment_half))
        # Radius of circumcircle. For polygon of the unit area.
        self.r = self.cos_theta * self.R
        # Radius of inscribed circle for R=1

    @property
    def id(self):
        return self.n_sides


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
        theta_reminder = theta % self.theta_segment_half
        x_reminder = math.cos(theta_reminder) * r

        # Check if the x coordinate of the reminder point is less than
        # the radius of the inscribed circle (for R=1)
        return x_reminder <= self.r

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
        theta_reminder = theta % self.theta_segment_half
        x_reminder = np.cos(theta_reminder) * r
        return x_reminder <= self.r

    def gmsh_base_shape(self, gmsh_geom: 'GeometryOCC'):
        """
        Base shape for an N side polygon.
        :param gmsh_geom:
        :return:
        """
        points = self.disc_approx(n_sides=self.n_sides, scale=self.R)
        return gmsh_geom.make_polygon(points)


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




def normal_to_axis_angle(normal): ## todo
    """

    """
    z_axis = np.array([0, 0, 1], dtype=float)
    norms = normal / np.linalg.norm(normal)
    cos_angle = np.dot(norms, z_axis)
    angle = np.arccos(cos_angle)
    # sin_angle = np.sqrt(1-cos_angle**2)

    axis = np.cross(z_axis, norms)
    ax_norm = np.linalg.norm(axis)
    if ax_norm < 1e-13:
        axis = np.array([1, 0, 0])
    else:
        axis = axis / ax_norm
    #return axes, angles
    return axis, angle

def normals_to_axis_angles(normals): ## todo
    """

    """
    z_axis = np.array([0, 0, 1], dtype=float)
    norms = normals / np.linalg.norm(normals, axis=1)[:, None]
    cos_angle = norms @ z_axis
    angles = np.arccos(cos_angle)
    # sin_angle = np.sqrt(1-cos_angle**2)

    axes = np.cross(z_axis, norms, axisb=1)
    ax_norm = np.maximum(np.linalg.norm(axes, axis=1), 1e-200)
    axes = axes / ax_norm[:, None]
    #return axes, angles
    return np.concatenate([axes, angles[:, None]], axis=1)


def rotate(vectors, axis=None, angle=0.0, axis_angle=None):
    """
    Rotate given vector around given 'axis' by the 'angle'.
    :param vectors: array of 3d vectors, shape (n, 3)
    :param axis_angle: pass both as array (4,)
    :return: shape (n, 3)
    """
    if axis_angle is not None:
        axis, angle = axis_angle[:3], axis_angle[3]
    if angle == 0:
        return vectors
    vectors = np.atleast_2d(vectors)
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    rotated = vectors * cos_angle \
              + np.cross(axis, vectors, axisb=1) * sin_angle \
              + axis[None, :] * (vectors @ axis)[:, None] * (1 - cos_angle)
    # Rodrigues formula for rotation of vectors around axis by an angle
    return rotated








@attrs.define
class Fracture:
    """
    Single fracture sample.
    TODO: modify to the acessor into the FrSet objects.
    """
    shape_idx: int
    # Basic fracture shape idx.
    radius: Tuple[float, float]
    # Fracture diameter, laying in XY plane
    center: np.array
    # location of the barycentre of the fracture
    normal: np.array
    # fracture normal
    shape_axis: np.array = np.array([1, 0])
    # angle to rotate the unit shape around z-axis; rotate anti-clockwise
    #region_id: int # Union[str, int] = "fracture"
    # Family index in population. Could be used to identify group of fractures even for population = None
    family: int = None
    # Original family, None if created manually (in tests)
    #aspect: float = 1
    # aspect ratio of the fracture =  y_length / x_length where  x_length == r
    #id: Any = None
    # any value associated with the fracture (DEPRECATED should be replaced by
    # FrValue class and fr_mesh code
    population: 'Population' = None


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
    def shape(self):
        return BaseShape.shape_for_id(self.shape_idx)

    @property
    def r(self):
        return self.radius[0]

    @property
    def aspect(self):
        return self.radius[1] / self.radius[0]

    @property
    def shape_angle(self):
        angle = np.arctan2(self.shape_axis[1], self.shape_axis[0])
        return angle

    @property
    def vertices(self):
        if self._vertices is None:
            _vertices = self.transform(self.shape._points)
        return _vertices

    @property
    def ref_vertices(self):
        if self._ref_vertices is None:
            _ref_vertices = self.shape._points
        return _ref_vertices

    @property
    def rx(self):
        return self.radius[0]

    @property
    def ry(self):
        return self.radius[1]

    @property
    def scale(self):
        return np.array([self.r, self.r * self.aspect])

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

        bound_vec = np.zeros(self.shape._points.shape)
        x_0_b = np.zeros(self.shape._points.shape)

        aspect = np.array([self.r, self.aspect * self.r, 1], dtype=float) # 0.5 *
        points = self.shape._points #* aspect[None, :]  # self.shape_class._points



        col2 = loc_direct[0]
        for i in range(0, self.shape._points.shape[0] - 1):
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








@attrs.define
class FractureSet:
    """
    Interface to the array based storage for the fractures.
    Has given outer box domain.

    The 1D fractures in 2D are treated as 2D fractures in 3D but:
    - centers have z=0
    - normals have z=0
    - shape_angle = 0
    - shape is two point line segment= array_attr(shape=(-1, 2), dtype=np.double)
    - r[0,:] = r[1,:]
    """

    #domain: NDArray[Shape['3'], Float]            # Box given by (3,) shape array
    #base_shapes : List[Any]                 # Unique reference shape classes

    #shape_idx = array_attr(shape=(-1,), dtype=np.int32)           # Base shape type index into 'base_shapes' list.
    base_shape_idx = attrs.field(type=int)                             # keep fracture sets of common shape, that is far enough for practical applications
    radius = array_attr(shape=(-1, 2), dtype=np.double)           # shape (n_fractures, 2), X and Y scaling of the reference shape.
    center = array_attr(shape=(-1, 3), dtype=np.double)           # center (n_fractures, 3); translation of the reference shape to actual position
    normal = array_attr(shape=(-1, 3), dtype=np.double)           # fracture unit normal vectors.

    shape_axis = array_attr(shape=(-1, 2), dtype=np.double)       # X reference unit vector in XY plane (n_fractures, 2)
    family = array_attr(shape=(-1,), dtype=np.int32)                # index of the fracture family within population

    population = attrs.field(type='Population', default=None)         # Generating population. Gives meaning to fr family indices.

    def __len__(self):
        return self.radius.shape[0]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @property
    def base_shapes(self):
        return self.__module__.__base_shapes

    @property
    def base_shape(self):
        return BaseShape.shape_for_id(self.base_shape_idx)

    @property
    def _base_shape_area(self):
        """
        Array of areas of the unit/base shapes.
        """
        return np.array([shp.area for shp in self.base_shapes])

    @fn.cached_property
    def area(self):
        """
        Array of fracture areas
        """
        shape_factor = self._base_shape_area[self.base_shape_idx]
        return shape_factor * np.prod(self.radius, axis=1)

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
                RectangleShape.id,
                radius=diag,
                center=(i - n_fr // 2) * normal + shift * normal + box / 2.0,
                normal=normal / separation,
                shape_angle=0,
                region_id=0)
            for i in range(n_fr)
        ]
        return cls.from_list(plates)

    @classmethod
    def area_sorted(cls, other: 'FractureSet') -> 'FractureSet':
        area_order = np.argsort(other.area)
        return cls(
            base_shape_idx=other.base_shape_idx,
            radius=other.radius[area_order],
            center=other.center[area_order],
            normal=other.normal[area_order],
            shape_axis=other.shape_axis[area_order],
            family=other.family[area_order],
            population=other.population
        )

    @classmethod
    def from_list(cls, fr_list: List[Fracture]) -> 'FractureSet':
        """
        Construct a fracture set from a list of 'Fracture' objects.
        :param fr_list:
        :return:
        TODO: deal with shape_idx
        TODO: fix shape_axis
        """
        fr_attribute = lambda attr : [getattr(fr, attr) for fr in fr_list]
        shape_idx_list = fr_attribute('shape_idx')
        #base_shape = {}
        #shape_idx_set = {si for si in shape_class_list}
        assert len(set(shape_idx_list)) == 1
        shape_idx = shape_idx_list[0]

        shape_axis = fr_attribute('shape_axis')
        shape_axis = np.array([[1, 0] if sa is None else sa for sa in shape_axis])
        shape_axis = np.stack(shape_axis, axis=0)

        family_idx = np.full(len(fr_list), 0)

        return cls(
            shape_idx,
            radius=fr_attribute('radius'),
            center=fr_attribute('center'),
            normal=fr_attribute('normal'),
            shape_axis=shape_axis,
            family=family_idx
        )

    @staticmethod
    def _concat_attr(fr_sets, attr, size=0):
        con = np.concatenate([getattr(fs, attr) for fs in fr_sets])
        if size:
            assert con.shape[0] == size
        return con

    @classmethod
    def merge(cls, fr_sets: List['FractureSet'], population=None) -> 'FractureSet':
        shape_idx_set = {fs.base_shape_idx for fs in fr_sets}
        assert len(shape_idx_set) == 1
        shape_idx = list(shape_idx_set)[0]
        ccat = lambda attr : cls._concat_attr(fr_sets, attr)
        return cls(
            shape_idx,
            radius=ccat('radius'),
            center=ccat('center'),
            normal=ccat('normal'),
            shape_axis=ccat('shape_axis'),
            family=ccat('family'),
            population = population
        )


    @fn.cached_property
    def AABB(self):
        """
        Axis Aligned Bounding Box for each fracture.

        See this blog: https://iquilezles.org/articles/diskbbox/
        AABB of an ellipse with axes U, V with center at origin:
        AABB[ax] = +/-sqrt( U[ax]**2 + V[ax]**2)

        The For given normal the formula is:
        AABB[ax] = +/- R * sqrt(1-N[ax]**2)

        That is AABB of the cicumflex disc.
        Tight AABB for a non 1:1 aspect ratio fracture
        would be much more complicated.

        :return: AABB corners as the array of shape: (N, 2, 3)
        min corner: AABB[:, 0, :], max corner AABB[:, 1, :]

        TODO: Do better! AABB is bounding, but not tight, gap is about 16%.
        """
        max_radii = np.max(self.radius, axis=1) * self.base_shape.R
        corners = self.center[:,None, :] + np.stack([-max_radii, +max_radii], axis=1)[:, :, None]
        return corners


    @fn.cached_property
    def rotation_mat(self):
        """
        Rotate and scale matrices for the fractures. The full transform involves 'self.center' as well:
        ambient_space_points = self.center + self.transform_mat @ local_fr_points[:, None, :]

        The Z- local axis is transformed to normal N (assumed unit).
        The shape_axis S = [sx, sy, 0] must be rotated  to SS by the rotation that rotates Z -> N.
        Then SS is transformation of the local X axis.
        The transformation of the Y axis is then computed by the corss product.

        Let's compute S':
        1. Z -> N rotation unit axis K = [-Ny, Nx, 0] / Nxy
        2. K . S = Sx Ny - Sy Nx
        3. follow Rodriguez formula proof, split S into part parallel (p) with K and ortogonal (o) to K
           Sp = (K.S) K
           So = S - Sp
        4. In the plane perpendicular to K,
           we have vertical component giving: cos(th) = Nz
           and horizontal component giving sin(th) = Nxy = sqrt(Nx^2 + Ny^2)
        5. We rotate So by angle th:
           SSo[z] = -|So| Nxy *sng(Nx)
           SSo[x,y] = (So [x,y]) Nz
        6. SSp = Sp
        7. Sum:
           SS = SSo + SSp :
           SSx = Spx + (Nz)Sox
           SSy = Spy + (Nz)Soy
           SSz = -|So| Nxy *sng(Nx)
        Finally, the third vector of the rotated bases is   cross(N, S')
        TODO: find a better vector representation of the rotations, allowing faster construction of the transfrom matrix
        :return: Shape (N, 3, 3).
        """
        N = self.normal
        assert np.allclose(np.linalg.norm(N, axis=1), 1)
        Nxy = np.stack([-N[:, 1], N[:, 0]], axis=1)
        norm_Nxy = np.linalg.norm(Nxy, axis=1)
        K = Nxy / norm_Nxy[:, None]
        arg_small = np.argwhere(norm_Nxy < 1e-13)[:, 0]
        K[arg_small, :] = np.array([1, 0], dtype=float)
        K_dot_S = self.shape_axis[:, None, :] @ K[:, :, None]
        S_p = K_dot_S[:, 0, :] * K
        S_o = self.shape_axis - S_p
        cos_th = N[:, 2:3]
        SS_xy = S_p + cos_th * S_o
        # ?? th is in (0, pi) -> sin(th) always positive
        #pos_nx = np.logical_xor(N[:, 0] > 0, N[:, 2] < 0)
        pos_nx = (N[:, None, 0:2] @ self.shape_axis[:, :, None])[:, 0, 0] > 0
        sin_th = norm_Nxy
        sin_th[pos_nx] = - sin_th[pos_nx]
        SS_z = np.linalg.norm(S_o, axis=1) * sin_th

        # Construct the rotated X axis SS vector, shape (N, 3)
        SS = np.concatenate([
            SS_xy,
            SS_z[:, None]
        ], axis=1)
        scaled_trans_x = SS     #* self.radius[:, 0:1]
        scaled_trans_y = np.cross(N, SS, axis=1)    #* self.radius[:, 1:2]
        trans_z = N
        rot_mat = np.stack([scaled_trans_x, scaled_trans_y, trans_z], axis=2)
        return rot_mat


    # @fn.cached_property
    # def rotation_mat(self):
    #     """
    #     Rotate and scale matrices for the fractures. The full transform involves 'self.center' as well:
    #     ambient_space_points = self.center + self.transform_mat @ local_fr_points[:, None, :]
    #
    #     The Z- local axis is transformed to normal N (assumed unit).
    #     The shape_axis S = [sx, sy, 0] must be rotated  to SS by the rotation that rotates Z -> N.
    #     Then SS is transformation of the local X axis.
    #     The transformation of the Y axis is then computed by the corss product.
    #
    #     Let's compute S':
    #     1. Z -> N rotation unit axis K = [-Ny, Nx, 0] / Nxy
    #     2. K . S = Sx Ny - Sy Nx
    #     3. follow Rodriguez formula proof, split S into part parallel (p) with K and ortogonal (o) to K
    #        Sp = (K.S) K
    #        So = S - Sp
    #     4. In the plane perpendicular to K,
    #        we have vertical component giving: cos(th) = Nz
    #        and horizontal component giving sin(th) = Nxy = sqrt(Nx^2 + Ny^2)
    #     5. We rotate So by angle th:
    #        SSo[z] = -|So| Nxy *sng(Nx)
    #        SSo[x,y] = (So [x,y]) Nz
    #     6. SSp = Sp
    #     7. Sum:
    #        SS = SSo + SSp :
    #        SSx = Spx + (Nz)Sox
    #        SSy = Spy + (Nz)Soy
    #        SSz = -|So| Nxy *sng(Nx)
    #     Finally, the third vector of the rotated bases is   cross(N, S')
    #     TODO: find a better vector representation of the rotations, allowing faster construction of the transfrom matrix
    #     :return: Shape (N, 3, 3).
    #     """
    #     N = self.normal
    #     assert np.allclose(np.linalg.norm(N, axis=1), 1)
    #     Nxy = N[:, 0:2]
    #     norm_Nxy = np.linalg.norm(Nxy, axis=1)
    #     Nxy_norm = Nxy / norm_Nxy
    #     arg_small = np.argwhere(Nxy_norm < 1e-13)[:, 0]
    #     Nxy_norm[arg_small, :] = np.array([1, 0], dtype=float)
    #
    #     #Nxy_ort = np.stack([-N[:, 1], N[:, 0]], axis=1)
    #     #norm_Nxy_ort = np.linalg.norm(Nxy_ort, axis=1)
    #     # axis of rotation (0, 0, 1) -> normal
    #     #K = Nxy_ort / norm_Nxy_ort[:, None]
    #     #arg_small = np.argwhere(norm_Nxy_ort < 1e-13)[:, 0]
    #     #K[arg_small, :] = np.array([1, 0], dtype=float)
    #     #K_dot_S = self.shape_axis[:, None, :] @ K[:, :, None]
    #
    #     S_dot_Nxy = (self.shape_axis[:, None, :] @ Nxy_norm[:, :, None])[:, 0, :]
    #
    #     # component of shape_axis parallel with axis L
    #     #S_p = K_dot_S[:, 0, :] * K
    #     # component of shape_axis ortogonal to axis
    #     #S_o = self.shape_axis - S_p
    #
    #     S_o_ = S_dot_Nxy * Nxy_norm
    #     S_p_ = self.shape_axis - S_o_
    #
    #     # We compute sin and cos of the rotation angle theta
    #     cos_th = N[:, 2:3]
    #     SS_xy = S_p_ + cos_th * S_o_
    #     # S_o is in direction of Nxy, but sign may differ
    #     # sin_th is negative iff sign(S_o) == Nxy
    #     # sin_th is positive iff sign(S_o) != Nxy
    #     ## ?? th is in (0, pi) -> sin(th) always positive
    #     ## pos_nx = np.logical_xor(N[:, 0] > 0, N[:, 2] < 0)
    #
    #     neg_sin = S_dot_Nxy[:, 0] > 0
    #     sin_th = norm_Nxy
    #     sin_th[neg_sin] = - sin_th[neg_sin]
    #     SS_z = np.linalg.norm(S_o_, axis=1) * sin_th
    #
    #     # Construct the rotated X axis SS vector, shape (N, 3)
    #     SS = np.concatenate([
    #         SS_xy,
    #         SS_z[:, None]
    #     ], axis=1)
    #     scaled_trans_x = SS     #* self.radius[:, 0:1]
    #     scaled_trans_y = np.cross(N, SS, axis=1)    #* self.radius[:, 1:2]
    #     trans_z = N
    #     rot_mat = np.stack([scaled_trans_x, scaled_trans_y, trans_z], axis=2)
    #     return rot_mat

    @fn.cached_property
    def transform_mat(self):
        """
        Rotate and scale matrices for the fractures. The full transform involves 'self.center' as well:
        ambient_space_points = self.center + self.transform_mat @ local_fr_points[:, None, :]

        The Z- local axis is transformed to normal N (assumed unit).
        The shape_axis S = [sx, sy, 0] must be rotated  to SS by the rotation that rotates Z -> N.
        Then SS is transformation of the local X axis.
        The transformation of the Y axis is then computed by the corss product.

        Let's compute S':
        1. Z -> N rotation unit axis K = [-Ny, Nx, 0] / Nxy
        2. K . S = Sx Ny - Sy Nx
        3. follow Rodriguez formula proof, split S into part parallel (p) with K and ortogonal (o) to K
           Sp = (K.S) K
           So = S - Sp
        4. In the plane perpendicular to K,
           we have vertical component giving: cos(th) = Nz
           and horizontal component giving sin(th) = Nxy = sqrt(Nx^2 + Ny^2)
        5. We rotate So by angle th:
           SSo[z] = -|So| Nxy *sng(Nx)
           SSo[x,y] = (So [x,y]) Nz
        6. SSp = Sp
        7. Sum:
           SS = SSo + SSp :
           SSx = Spx + (Nz)Sox
           SSy = Spy + (Nz)Soy
           SSz = -|So| Nxy *sng(Nx)
        Finally, the third vector of the rotated bases is   cross(N, S')
        TODO: find a better vector representation of the rotations, allowing faster construction of the transfrom matrix
        :return: Shape (N, 3, 3).
        """
        trans_mat = (self.rotation_mat[:, :, :]).copy()
        trans_mat[:, :, 0] *=  (self.radius[:, 0])[:, None]
        trans_mat[:, :, 1] *=  (self.radius[:, 1])[:, None]
        return trans_mat

    @fn.cached_property
    def inv_transform_mat(self):
        """ TODO transpose of just rot mat, deal with scaling as well."""
        inv_trans_mat = (self.rotation_mat[:, :, :].transpose((0, 2, 1))).copy()
        inv_trans_mat[:, 0, :] /=  (self.radius[:, 0])[:, None]
        inv_trans_mat[:, 1, :] /=  (self.radius[:, 1])[:, None]
        return inv_trans_mat

    def make_fractures_gmsh(self, gmsh_geom: 'GeometryOCC', transform=None):
        """

        :param gmsh_geom:
        :param fractures:
        :param base_shape:
        :param shift:
        :return:
        """
        # From given fracture date list 'fractures'.
        # transform the base_shape to fracture objects
        # fragment fractures by their intersections
        # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
        if len(self) == 0:
            return []
        base_shape = self.base_shape.gmsh_base_shape(gmsh_geom)
        shapes = []
        region_map = {}
        for i, fr in enumerate(self):
            shape = base_shape.copy()
            #print("fr: ", i, "tag: ", shape.dim_tags)
            region_name = f"fam_{fr.family}_{i:03d}"
            shape = shape.scale([fr.rx, fr.ry, 1]) \
                .rotate(axis=[0, 0, 1], angle=fr.shape_angle) \
                .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
                .translate(fr.center) \
                .set_region(region_name)
            region_map[region_name] = i
            shapes.append(shape)

        fracture_fragments = gmsh_geom.fragment(*shapes)
        return fracture_fragments, region_map

    def make_fractures_brep(self, brep_name: Union[str, pathlib.Path]):
        """
        Create the BREP file from a list of fractures using the brep writer interface.
        """
        # fracture_mesh_step = geometry_dict['fracture_mesh_step']
        # dimensions = geometry_dict["box_dimensions"]

        #print("n fractures:", len(self))
        if isinstance(brep_name, str):
            brep_name = pathlib.Path(brep_name)
        brep_name = brep_name.with_suffix(".brep")
        faces = []
        base_vertices = self.base_shape.vertices(8)

        # Legacy transform
        fr_vtxs = lambda fr : fr.transform(base_vertices) # fr.center
        fractures_vertices = np.array([fr_vtxs(fr) for fr in self])

        #fractures_vertices = self.transform_mat @ (base_vertices.T)[None, :, :]   # (n_fr, 3, 3) @ (1, 3, n_points) -> (n_fr, 3, n_points)
        #fractures_vertices = fractures_vertices.transpose((0, 2, 1))
        #fractures_vertices = fractures_vertices + self.center[:, None, :] # (n_fr, 3, n_points) -> (n_fr, n_points, 3)


        for i, fr_vertices in enumerate(fractures_vertices):
            vtxs = [bw.Vertex(p) for p in fr_vertices]
            edges = [bw.Edge(a, b) for a, b in zip(vtxs[:-1], vtxs[1:])]
            edges.append(bw.Edge(vtxs[-1], vtxs[0]))
            face = bw.Face(edges)
            faces.append(face)

        comp = bw.Compound(faces)
        with open(brep_name, "w") as f:
            bw.write_model(f, comp)
        return brep_name


    def __getitem__(self, item):
        family_idx = self.family[item]
        return Fracture(
            shape_idx=self.base_shape_idx,
            radius=self.radius[item, :],
            center=self.center[item, :],
            normal=self.normal[item, :],
            shape_axis=self.shape_axis[item, :],
            family=family_idx,
            population=self.population
        )

def fr_conductivity_cubic(dfn:FractureSet, tensor:bool=False):
    scalar




@attrs.define
class FractureValues:
    """
    A quantities on the fracture set, one value for each fracture, constant on the fracture.
    """
    fractures: FractureSet
    values: npt.NDArray[np.double]

"""
Some Fracture Values Operations
"""
def fr_values_permeability():
        pass


class FractureMesh:
    """
    GMSH specific class for fracture mesh generation.
    Mesh generation procedure:
    1. Bulk GMSH geometry, functional approach allowing association of some data with parts of geometry.
       Functional approach: refer to the shapes through Entities: dim_id set + dict property  -> PropertyValues;
       ProperyValues: [value], {dim_id: value_idx}, works as dim_id -> value map (Shold exsits as a CompressedDict or SparseDict)

       Operations: merge properties dicts, parent has priority Intersection - get A * B parent properties, optionaly get disable 'other' properties
       cut - get parent properties
       union - keep properties of A, B, intersection properties on A * B
    2. Create fractures and apply them to the bulk geometry. Can apply to given bulk shape or to the whole geometry.
       Cut fractures at shape boundary. => Intersection map: fracture -> Entity(dim_id set)
    3. FractureMesh - various simplification and modification operations
       FractureMap - attach entities and properties to mesh elements through gmsh_shape_ids (internaly)
       Field ... could be defined with respect to fracture properties, access to element attached properties

    Temporary solution until the functional GMSH approach would be implemented:
    1. Assign fracture region IDs to the geometry fracture shapes after fragmentation -> return from fracture application
    2. Create FractureMesh with element -> fracture properties (r, ...), should work as Fields having value only on fracture elements
       FractureMehs should provide getters to fracture and bulk property fields:
       bulk:
       - center
       - el_volume
       - el_transform
       - custom bulk declared properties
       fracture (in adition):
       - r
       - normal
       - fr_transform (for anisotropic properties on the reference fracture shape)
       - fr_center
       Fields could be constructed by first evaluate the bulk elements and then the fracture elements using fracture fields.
    """

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



