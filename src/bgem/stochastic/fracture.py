"""
Module for statistical description of the fracture networks.
It provides appropriate statistical models as well as practical sampling methods.
"""

from typing import *
import numpy as np
#import attr
import attrs
import math
import json


class LineShape:
    """
    Class represents the line fracture shape.
    The polymorphic `make_approx` method is used to create polygon (approximation in case of disc) of the
    actual fracture.
    """
    _points = np.array([[-0.5, 0, 0], [0.5, 0, 0]])

    @classmethod
    def make_approx(cls, x_scale, y_scale, step=None):
        xy_scale = np.array([x_scale, y_scale, 1.0])
        return cls._points[:, :] * xy_scale[None, :]


class SquareShape(LineShape):
    """
    Class represents the square fracture shape.
    """
    _points = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])


class DiscShape:
    """
    Class represents the square fracture shape.
    """

    @classmethod
    def make_approx(cls, x_scale, y_scale, step=1.0):
        n_sides = np.pi * min(x_scale, y_scale) / step
        n_sides = max(4, n_sides)
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        points = np.stack(np.cos(angles) * x_scale, np.sin(angles) * y_scale, np.ones_like(angles))
        return points

class ConvexPolygon:
    """
    Class represents the convex polygon shape.
    """

    @classmethod
    def make_approx(cls, x_scale, y_scale, step=1.0):
        n_sides = np.pi * min(x_scale, y_scale) / step
        n_sides = max(4, n_sides)
        angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
        points = np.stack(np.cos(angles) * x_scale, np.sin(angles) * y_scale, np.ones_like(angles))
        return points


@attrs.define
class Fracture:
    """
    Single fracture sample.
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
    i_family: int
    # index of the group
    aspect: float = 1
    # aspect ratio of the fracture =  y_length / x_length where  x_length == r
    region: Any = None
    # auxiliary, TODO: Separate generation of fractures and fracture shapes.
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
        axis_angle = normal_to_axis_angle(self.normal)[0,:]
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

def normal_to_axis_angle(normal): ## todo
    """

    """
    z_axis = np.array([0, 0, 1], dtype=float)
    norms = normal / np.linalg.norm(normal, axis=1)[:, None]
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
class VonMisesOrientation:
    """
    Distribution for random orientation in 2d.
    X = east, Y = north
    """

    trend: float = 0
    # azimuth (0, 360) of the fractures normal
    concentration: float = 0
    # concentration parameter, 0 = uniformely dispersed, 1 = exect orientation

    def sample_axis_angle(self, size=1):
        """
        Sample fracture orientation angles.
        :param size: Number of samples
        :return: shape (n, 4), every row: unit axis vector and angle
        """
        axis_angle = np.tile(np.array([0, 0, 1, 0], dtype=float), size).reshape((size, 4))
        axis_angle[:, 3] = self.sample_angle(size)
        return axis_angle


    def sample_angle(self, size=1):
        trend = np.radians(self.trend)
        if self.concentration > np.log(np.finfo(float).max):
            return trend + np.zeros(size)
        else:
            if self.concentration == 0:
                return np.random.uniform(size=size) * 2 * np.pi
            else:
                return np.random.vonmises(mu=trend, kappa=self.concentration, size=size)

    def sample_normal(self, size=1):
        """
        Draw samples for the fracture normals.
        :param size: number of samples
        :return: array (n, 3)
        """
        angle = self.sample_angle(size)
        return np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=1)


@attrs.define
class FisherOrientation:
    """
    Distribution for random orientation in 3d.

    Coordinate system: X - east, Y - north, Z - up

    strike, dip - used for the orientation of the planar geological features
    trend, plunge - used for the orientation of the line geological features

    As the distribution is considerd as distribution of the fracture normal vectors we use
    trend, plunge as the primal parameters.
    """

    trend: float
    # mean fracture normal (pointing down = negative Z)
    # azimuth (0, 360) of the normal's projection to the horizontal plane
    # related term is the strike =  trend - 90; that is azimuth of the strike line
    # - the intersection of the fracture with the horizontal plane
    plunge: float
    # mean fracture normal (pointing down = = negative Z)
    # angle (0, 90) between the normal and the horizontal plane
    # related term is the dip = 90 - plunge; that is the angle between the fracture and the horizontal plane
    #
    # strike and dip can by understood as the first two Eulerian angles.
    concentration: float
    # the concentration parameter; 0 = uniform dispersion, infty - no dispersion

    @staticmethod
    def strike_dip(strike, dip, concentration):
        """
        Initialize from (strike, dip, concentration)
        """
        return FisherOrientation(strike + 90, 90 - dip, concentration)

    def _sample_standard_fisher(self, n) -> np.array:
        """
        Normal vector of random fractures with mean direction (0,0,1).
        :param n:
        :return: array of normals (n, 3)
        """
        if self.concentration > np.log(np.finfo(float).max):
            normals = np.zeros((n, 3))
            normals[:, 2] = 1.0
        else:
            unif = np.random.uniform(size=n)
            psi = 2 * np.pi * np.random.uniform(size=n)
            cos_psi = np.cos(psi)
            sin_psi = np.sin(psi)
            if self.concentration == 0:
                cos_theta = 1 - 2 * unif
            else:
                exp_k = np.exp(self.concentration)
                exp_ik = 1 / exp_k
                cos_theta = np.log(exp_k - unif * (exp_k - exp_ik)) / self.concentration
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            # theta = 0 for the up direction, theta = pi  for the down direction
            normals = np.stack((sin_psi * sin_theta, cos_psi * sin_theta, cos_theta), axis=1)
        return normals

    def sample_normal(self, size=1):
        """
        Draw samples for the fracture normals.
        :param size: number of samples
        :return: array (n, 3)
        """
        raw_normals = self._sample_standard_fisher(size)
        mean_norm = self._mean_normal()
        axis_angle = normal_to_axis_angle(mean_norm[None, :])
        return rotate(raw_normals, axis_angle=axis_angle[0])


    #def sample_axis_angle(self, size=1):
    #    """
    #    Sample fracture orientation angles.
    #    :param size: Number of samples
    #    :return: shape (n, 4), every row: unit axis vector and angle
    #    """
    #    normals = self._sample_normal(size)
    #    return self.normal_to_axis_angle(normals[:])


    def _mean_normal(self):
        trend = np.radians(self.trend)
        plunge = np.radians(self.plunge)
        normal = np.array([np.sin(trend) * np.cos(plunge),
                           np.cos(trend) * np.cos(plunge),
                           -np.sin(plunge)])

        # assert np.isclose(np.linalg.norm(normal), 1, atol=1e-15)
        return normal

    # def normal_2_trend_plunge(self, normal):
    #
    #     plunge = round(degrees(-np.arcsin(normal[2])))
    #     if normal[1] > 0:
    #         trend = round(degrees(np.arctan(normal[0] / normal[1]))) + 360
    #     else:
    #         trend = round(degrees(np.arctan(normal[0] / normal[1]))) + 270
    #
    #     if trend > 360:
    #         trend = trend - 360
    #
    #     assert trend == self.trend
    #     assert plunge == self.plunge


# class Position:
#     def __init__(self):




@attrs.define
class PowerLawSize:
    """
    Truncated Power Law distribution for the fracture size 'r'.
    The density function:

    f(r) = f_0 r ** (-power - 1)

    for 'r' in [size_min, size_max], zero elsewhere.

    The class allows to set a different (usually reduced) sampling range for the fracture sizes,
    one can either use `set_sample_range` to directly set the sampling range or just increase the lower bound to meet
    prescribed fracture intensity via the `set_range_by_intansity` method.

    """
    power: float
    # power of th power law
    diam_range: (float, float)
    # lower and upper bound of the power law for the fracture diameter (size), values for which the intensity is given
    intensity: float
    # number of fractures with size in the size_range per unit volume (denoted as P30 in SKB reports)

    sample_range: (float, float) = attrs.field(init=False)
    # range used for sampling., not part of the statistical description
    # default initiaizer:
    @sample_range.default
    def copy_full_range(self):
        return list(self.diam_range).copy()  # need copy to preserve original range

    @classmethod
    def from_mean_area(cls, power, diam_range, p32, p32_power=None):
        """
        Construct the distribution using the mean arrea (P32) instead of intensity.
        :param p32: mean area of the fractures in given `diam_range`.
        :param p32_power: if the mean area is given for different power parameter.
        :return: PowerLawSize instance.
        """
        if p32_power is None:
            p32_power = power
        return cls(power, diam_range, cls.intensity_for_mean_area(p32, power, diam_range, p32_exp=p32_power))

    def cdf(self, x, range):
        """
        Power law distribution function for the given support interval (min, max).
        """
        min, max = range
        pmin = min ** (-self.power)
        pmax = max ** (-self.power)
        return (pmin - x ** (-self.power)) / (pmin - pmax)

    def ppf(self, x, range):
        """
        Power law quantile (inverse distribution) function for the given support interval (min, max).
        """
        min, max = range
        pmin = min ** (-self.power)
        pmax = max ** (-self.power)
        scaled = pmin - x * (pmin - pmax)
        return scaled ** (-1 / self.power)

    def range_intensity(self, range):
        """
        Computes the fracture intensity (P30) for different given fracture size range.
        :param range: (min, max) - new fracture size range
        """
        a, b = self.diam_range
        c, d = range
        k = self.power
        return self.intensity * (c ** (-k) - d ** (-k)) / (a ** (-k) - b ** (-k))

    def set_sample_range(self, sample_range=None):
        """
        Set the range for the fracture sampling.
        :param sample_range: (min, max), None to reset to the full range.
        """
        if sample_range is None:
            sample_range = self.diam_range
        self.sample_range = list(sample_range).copy()

    def set_lower_bound_by_intensity(self, intensity):
        """
        Increase lower fracture size bound of the sample range in order to achieve target fracture intensity.
        """
        a, b = self.diam_range
        c, d = self.sample_range
        k = self.power
        lower_bound = (intensity * (a ** (-k) - b ** (-k)) / self.intensity + d ** (-k)) ** (-1 / k)
        self.sample_range[0] = lower_bound

    def set_upper_bound_by_intensity(self, intensity):
        """
        Increase lower fracture size bound of the sample range in order to achieve target fracture intensity.
        """
        a, b = self.diam_range
        c, d = self.sample_range
        k = self.power
        upper_bound = (c ** (-k) - intensity * (a ** (-k) - b ** (-k)) / self.intensity ) ** (-1 / k)
        self.sample_range[1] = upper_bound


    def mean_size(self, volume=1.0):
        """
        :return: Mean number of fractures for given volume
        """
        sample_intensity = self.range_intensity(self.sample_range)
        return sample_intensity * volume

    def sample(self, volume, size=None, force_nonempty=False):
        """
        Sample the fracture diameters.
        :param volume: By default the volume and fracture sample intensity is used to determine actual number of the fractures.
        :param size: ... alternatively the prescribed number of fractures can be generated.
        :param force_nonempty: If True at leas one fracture is generated.
        :return: Array of fracture sizes.
        """
        if size is None:
            size = np.random.poisson(lam=self.mean_size(volume), size=1)
            if force_nonempty:
                size = max(1, size)
        #print("PowerLaw sample: ", force_nonempty, size)
        U = np.random.uniform(0, 1, int(size))
        return self.ppf(U, self.sample_range)

    def mean_area(self, volume=1.0, shape_area=1.0):
        """
        Compute mean fracture surface area from current sample range intensity.
        :param shape_area: Area of the unit fracture shape (1 for square, 'pi/4' for disc)
        :return:
        """
        sample_intensity = volume * self.range_intensity(self.sample_range)
        a, b = self.sample_range
        exp = self.power
        integral_area = (b ** (2 - exp) - a ** (2 - exp)) / (2 - exp)
        integral_intensity = (b ** (-exp) - a ** (-exp)) / -exp
        p_32 = sample_intensity / integral_intensity * integral_area * shape_area
        return p_32

    @staticmethod
    def intensity_for_mean_area(p_32, exp, size_range, shape_area=1.0, p32_exp=None):
        """
        Compute fracture intensity from the mean fracture surface area per unit volume.
        :param p_32: mean fracture surface area
        :param exp: power law exponent
        :param size_range: fracture size range
        :param shape_area: Area of the unit fracture shape (1 for square, 'pi/4' for disc)
        :param p32_exp: possibly different value of the power parameter for which p_32 mean area is given
        :return: p30 - fracture intensity

        TODO: modify to general recalculation for two different powers and introduce separate wrapper functions
        for p32 to p30, p32 to p20, etc. Need to design suitable construction methods.
        """
        if p32_exp is None:
            p32_exp = exp
        a, b = size_range
        integral_area = (b ** (2 - p32_exp) - a ** (2 - p32_exp)) / (2 - p32_exp)
        integral_intensity = (b ** (-exp) - a ** (-exp)) / -exp
        return p_32 / integral_area / shape_area * integral_intensity


# @attr.s(auto_attribs=True)
# class PoissonIntensity:
#     p32: float
#     # number of fractures
#     size_min: float
#     #
#     size_max:
#     def sample(self, box_min, box_max):

@attrs.define
class UniformBoxPosition:
    dimensions: List[float]
    center: List[float] = [0, 0, 0]

    def sample(self):
        # size = 1
        # pos = np.empty((size, 3), dtype=float)
        # for i in range(3):
        #    pos[:, i] =  np.random.uniform(self.center[i] - self.dimensions[i]/2, self.center[i] + self.dimensions[i]/2, size)
        pos = np.empty(3, dtype=float)
        for i in range(3):
            pos[i] = np.random.uniform(self.center[i] - self.dimensions[i] / 2, self.center[i] + self.dimensions[i] / 2,
                                       size=1)
        return pos


@attrs.define
class ConnectedPosition:
    """
    Generate a fracture positions in such way, that all fractures are connected to some of the initial surfaces.
    Sampling algorithm:
    0. sampling position of the i-th fracture:
    1. select random surface using theoretical frequencies of the fractures:
        f_k = N_k / (N_f - k), with N_k ~ S_k, S_k is the area of k-th surface
       ... this is done by taking a random number from (0, sum f_k) and determining 'k'
           by search in the array of cumulative frequencies (use dynarray package).
    2. one point of the N_k points in k-th surface
    3. center of the new fracture such, that it contains the selected point

    N_k is obtained as:
    1. generate N_p * S_i points
    2. remove points that are close to some existing points on other fractures

    Possible improvements:
    Instead of grouping points according to fractures, make groups of points according to some volume cells.
    This way one can obtain more uniform distribution over given volume.
    """

    confining_box: List[float]
    # dimensions of the confining box (center in origin)
    point_density: float
    # number of points per unit square

    # List of fractures, fracture is the transformation matrix (4,3) to transform from the local UVW coordinates to the global coordinates XYZ.
    # Fracture in UvW: U=(-1,1), V=(-1,1), W=0.

    all_points: List[np.array] = []
    # all points on surfaces
    surf_points: List[int] = []
    # len = n surfaces + 1 - start of fracture's points in all_points, last entry is number of all points
    surf_cum_freq: List[float] = []

    # len = n surfaces + 1 - cumulative mean frequencies for surfaces; total_freq - the last entry is surf_cum_freq
    # used for efficient sampling of the parent fracture index

    @classmethod
    def init_surfaces(cls, confining_box, n_fractures, point_density, points):
        """
        :param confinign_box: dimensions of axis aligned box, points out of this box are excluded.
        :param point_density: number of points per unit square
        :param points: List of 3d points on the virtual initial surface.
        :return:
        """
        np = len(points)
        freq = np / (n_fractures - 0)
        return cls(confining_box, point_density, points.copy(), [0, np], [0, freq])

    # TODO continue
    def sample(self, diameter, axis, angle, shape_angle):
        """
        Sample position of the fracture with given shape and orientation.
        :return:
        sampling position of the i-th fracture:
        1. select random surface using theoretical frequencies of the fractures:
            f_k = N_k / (N_f - k), with N_k ~ S_k, S_k is the area of k-th surface
            ... this is done by taking a random number from (0, sum f_k) and determining 'k'
                by search in the array of cumulative frequencies (use dynarray package).
        2. one point of the N_k points in k-th surface
        3. center of the new fracture such, that it contains the selected point

        N_k is obtained as:
            1. generate N_p * S_i points
            2. remove points that are close to some existing points on other fractures

        """

        if len(self.fractures) == 0:
            self.confining_box = np.array(self.confining_box)
            # fill by box sides
            self.points = np.empty((0, 3))
            for fr_mat in self.boxes_to_fractures(self.init_boxes):
                self.add_fracture(fr_mat)
        # assert len(self.fractures) == len(self.surfaces)

        q = np.random.uniform(-1, 1, size=3)
        q[2] = 0
        uvq_vec = np.array([[1, 0, 0], [0, 1, 0], q])
        uvq_vec *= diameter / 2
        uvq_vec = FisherOrientation.rotate(uvq_vec, np.array([0, 0, 1]), shape_angle)
        uvq_vec = FisherOrientation.rotate(uvq_vec, axis, angle)

        # choose the fracture to prolongate
        i_point = np.random.randint(0, len(self.points), size=1)[0]
        center = self.points[i_point] + uvq_vec[2, :]
        self.add_fracture(self.make_fracture(center, uvq_vec[0, :], uvq_vec[1, :]))
        return center

    def add_fracture(self, fr_mat):
        i_fr = len(self.fractures)
        self.fractures.append(fr_mat)
        surf = np.linalg.norm(fr_mat[:, 2])

        points_density = 0.01
        # mean number of points per unit square meter
        points_mean_dist = 1 / np.sqrt(points_density)
        n_points = np.random.poisson(lam=surf * points_density, size=1)
        uv = np.random.uniform(-1, 1, size=(2, n_points[0]))
        fr_points = fr_mat[:, 0:2] @ uv + fr_mat[:, 3][:, None]
        fr_points = fr_points.T
        new_points = []

        for pt in fr_points:
            # if len(self.points) >0:
            dists_short = np.linalg.norm(self.points[:, :] - pt[None, :], axis=1) < points_mean_dist
            # else:
            #    dists_short = []
            if np.any(dists_short):
                # substitute current point for a choosed close points
                i_short = np.random.choice(np.arange(len(dists_short))[dists_short])
                self.points[i_short] = pt
                # self.point_fracture = i_fr
            else:
                # add new points that are in the confining box
                if np.all((pt - self.confining_box / 2) < self.confining_box):
                    new_points.append(pt)
                # self.point_fracture.append(i_fr)
        if new_points:
            self.points = np.concatenate((self.points, new_points), axis=0)

    @classmethod
    def boxes_to_fractures(cls, boxes):
        fractures = []
        for box in boxes:
            box = np.array(box)
            ax, ay, az, bx, by, bz = range(6)
            sides = [[ax, ay, az, bx, ay, az, ax, ay, bz],
                     [ax, ay, az, ax, by, az, bx, ay, az],
                     [ax, ay, az, ax, ay, bz, ax, by, az],
                     [bx, by, bz, ax, by, bz, bx, by, az],
                     [bx, by, bz, bx, ay, bz, ax, by, bz],
                     [bx, by, bz, bx, by, az, bx, ay, bz]]
            for side in sides:
                v0 = box[side[0:3]]
                v1 = box[side[3:6]]
                v2 = box[side[6:9]]
                fractures.append(cls.make_fracture(v0, v1 / 2, v2 / 2))
        return fractures

    @classmethod
    def make_fracture(cls, center, u_vec, v_vec):
        """
        Construct transformation matrix from one square cornerthree square corners,
        """
        w_vec = np.cross(u_vec, v_vec)
        return np.stack((u_vec, v_vec, w_vec, center), axis=1)


@attrs.define
class FrFamily:
    """
    Describes a single fracture family with defined orientation and shape distributions.
    """
    name: str
    orientation: FisherOrientation
    size: PowerLawSize
    shape_angle: VonMisesOrientation

    @staticmethod
    def from_cfg_3d(family):
        fisher_orientation = FisherOrientation(family["trend"], family["plunge"], family["concentration"])
        size_range = (family["r_min"], family["r_max"])
        power_law_size = PowerLawSize.from_mean_area(family["power"], size_range, family["p_32"])
        assert np.isclose(family["p_32"], power_law_size.mean_area())
        shape_angle = VonMisesOrientation(trend=0, concentration=0)
        return FrFamily(family["name"], fisher_orientation, power_law_size, shape_angle)

    @staticmethod
    def from_cfg_2d(family):
        orientation = FisherOrientation(0, 90, np.inf)
        size_range = (family["r_min"], family["r_max"])
        power_law_size = PowerLawSize.from_mean_area(family["power"], size_range, family["p_32"])
        assert np.isclose(family["p_32"], power_law_size.mean_area())
        shape_angle = VonMisesOrientation(family["trend"], family["concentration"])
        return FrFamily(family["name"], orientation, power_law_size, shape_angle)



@attrs.define
class Population:
    """
    Data class to describe whole population of fractures, several families.
    Supports sampling across the families.
    """
    # Attributes
    domain: Tuple[float, float, float]
    # dimensions of the box domain, one dimension is 0 for 2d population
    families: List[FrFamily]
    #
    shape_class: LineShape = SquareShape

    @property
    def volume(self):
        return np.product([l if l>0 else 1.0 for l in self.domain])

    @classmethod
    def initialize_3d(cls, families: List[Dict[str, Any]], box):
        """
        Load families from a list of dict, with keywords: [ name, trend, plunge, concentration, power, r_min, r_max, p_32 ]
        Assuming fixed statistical model: Fischer, Uniform, PowerLaw Poisson
        """
        families = [FrFamily.from_cfg_3d(family) for family in families]
        return cls(box, families)

    @classmethod
    def initialize_2d(cls, families: List[Dict[str, Any]], box):
        """
        Load families from a list of dict, with keywords: [ name, trend, plunge, concentration, power, r_min, r_max, p_32 ]
        Assuming fixed statistical model: Fischer, Uniform, PowerLaw Poisson
        :param families json_file: JSON file with families data
        """
        families = [FrFamily.from_cfg_2d(family) for family in families]
        assert len(box) == 3 and sum((l > 0 for l in box)) == 2
        return cls(box, families)

    def init_from_json(self, json_file):
        """
        Load families from a JSON file. Assuming fixed statistical model: Fischer, Uniform, PowerLaw Poisson
        :param json_file: JSON file with families data
        """
        with open(json_file) as f:
            self.initialize(json.load(f))

    def init_from_yaml(self, yaml_file):
        """
        Load families from a YAML file. Assuming fixed statistical model: Fischer, Uniform, PowerLaw Poisson
        :param json_file: YAML file with families data
        """
        with open(yaml_file) as f:
            self.initialize(json.load(f))





    def mean_size(self):
        sizes = [family.size.mean_size(self.volume) for family in self.families]
        return sum(sizes)

    def set_sample_range(self, sample_range, sample_size=None):
        """
        Set sample range for fracture diameter.
        :param sample_range: (min_bound, max_bound) - one of these can be None if 'sample_size' is provided
                                                      this bound is set to match mean number of fractures
        :param sample_size: If provided, the None bound is changed to achieve given mean number of fractures.
                            If neither of the bounds is None, the lower one is reset.
        :return:
        TODO: split fracture population (stochastic model) and fracture generator with
        particular sample size and domain. Make both objects frozen.
        """
        min_size, max_size = sample_range
        for f in self.families:
            r_min, r_max = f.size.diam_range
            if min_size is not None:
                r_min = min_size
            if max_size is not None:
                r_max = max_size
            f.size.set_sample_range((r_min, r_max))
        if sample_size is not None:
            family_sizes = [family.size.mean_size(self.volume) for family in self.families]
            total_size = np.sum(family_sizes)

            if max_size is None:
                for f, size in zip(self.families, family_sizes):
                    family_intensity = size / total_size * sample_size / self.volume
                    f.size.set_upper_bound_by_intensity(family_intensity)
            else:
                for f, size in zip(self.families, family_sizes):
                    family_intensity = size / total_size * sample_size / self.volume
                    f.size.set_lower_bound_by_intensity(family_intensity)


    def sample(self, pos_distr=None, keep_nonempty=False) -> List[Fracture]:
        """
        Provide a single fracture set  sample from the population.
        :param pos_distr: Fracture position distribution, common to all families.
        An object with method .sample(size) returning array of positions (size, 3).
        :return: List of FractureShapes.
        """
        if pos_distr is None:
            size = np.cbrt(self.volume)
            pos_distr = UniformBoxPosition([size, size, size])

        fractures = []
        for ifam, f in enumerate(self.families):
            name = f.name
            diams = f.size.sample(self.volume)
            fr_normals = f.orientation.sample_normal(size=len(diams))
            #fr_axis_angle = f.orientation.sample_axis_angle(size=len(diams))
            shape_angle = f.shape_angle.sample_angle(len(diams))
                #np.random.uniform(0, 2 * np.pi, len(diams))
            for r, normal, sa in zip(diams, fr_normals, shape_angle):
                #axis, angle = aa[:3], aa[3]
                center = pos_distr.sample()
                fractures.append(Fracture(self.shape_class, r, center, normal[None,:], sa, name, ifam, 1))
        return fractures


def plotly_fractures(fr_set, fr_points):
    """
    Plot generated fractures.
    :param fr_set: List[FractureShape]
    :param fr_set: List[np.array(n, 2)] local point coordinates on fractures
    :return:
    """
    import plotly.offline as pl
    import plotly.graph_objs as go
    # import plotly.graph_objects as go
    for ifr, (fr, points) in enumerate(zip(fr_set, fr_points)):
        n_side = 5
        boundary = np.empty((4, n_side, 3))
        corners = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0]])
        for s in range(4):
            start, end = corners[s, :], corners[(s + 1) % 4, :]
            boundary[s, :, :] = start[None, :] + (end - start)[None, :] * np.linspace(0, 1, n_side, endpoint=False)[:,
                                                                          None]
        boundary = boundary.reshape((-1, 3))
        boundary = fr.transform(boundary)
        points = fr.transform(points)

        fig = go.Figure(data=[
                go.Scatter3d(x=boundary[:, 0], y=boundary[:, 1], z=boundary[:, 2],
                             marker=dict(size=1, color='blue')),
                go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                             marker=dict(size=1.5, color='red'))
                ])
        fig.update_layout(
            scene=dict(
                # xaxis=dict(range=[-2, 2]),
                # yaxis=dict(range=[-2, 2]),
                # zaxis=dict(range=[-1, 1]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1)

            ),
        )
        pl.plot(fig, filename='fractures.html')


#
# class FractureGenerator:
#     def __init__(self, frac_type):
#         self.frac_type = frac_type
#
#     def generate_fractures(self, min_distance, min_radius, max_radius):
#         fractures = []
#
#         for i in range(self.frac_type.n_fractures):
#             x = uniform(2 * min_distance, 1 - 2 * min_distance)
#             y = uniform(2 * min_distance, 1 - 2 * min_distance)
#             z = uniform(2 * min_distance, 1 - 2 * min_distance)
#
#             tpl = TPL(self.frac_type.kappa, self.frac_type.r_min, self.frac_type.r_max, self.frac_type.r_0)
#             r = tpl.rnd_number()
#
#             orient = Orientation(self.frac_type.trend, self.frac_type.plunge, self.frac_type.k)
#             axis, angle = orient.compute_axis_angle()
#
#             fd = FractureData(x, y, z, r, axis[0], axis[1], axis[2], angle, i * 100)
#
#             fractures.append(fd)
#
#         return fractures
#
#     def write_fractures(self, fracture_data, file_name):
#         with open(file_name, "w") as writer:
#             for d in fracture_data:
#                 writer.write("%f %f %f %f %f %f %f %f %d\n" % (d.centre[0], d.centre[1], d.centre[2], d.r, d.rotation_axis[0],
#                                                         d.rotation_axis[1], d.rotation_axis[2], d.rotation_angle, d.tag))
#
#     def read_fractures(self, file_name):
#         data = []
#         with open(file_name, "r") as reader:
#             for l in reader.readlines():
#                 x, y, z, r, axis_0, axis_1, axis_2, angle = [float(i) for i in l.split(' ')[:-1]]
#                 tag = int(l.split(' ')[-1])
#                 d = FractureData(x, y, z, r, axis_0, axis_1, axis_2, angle, tag)
#                 data.append(d)
#
#         return data
#


def unit_square_vtxs():
    return np.array([
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.5, 0.5, 0],
        [-0.5, 0.5, 0]])




class Fractures:
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



# class Quat:
#     """
#     Simple quaternion class as numerically more stable alternative to the Orientation methods.
#     TODO: finish, test, substitute
#     """
#
#     def __init__(self, q):
#         self.q = q
#
#     def __matmul__(self, other: 'Quat') -> 'Quat':
#         """
#         Composition of rotations. Quaternion multiplication.
#         """
#         w1, x1, y1, z1 = self.q
#         w2, x2, y2, z2 = other.q
#         w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
#         x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
#         y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
#         z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
#         return Quat((w, x, y, z))
#
#     @staticmethod
#     def from_euler(a: float, b: float, c: float) -> 'Quat':
#         """
#         X-Y-Z Euler angles to quaternion
#         :param a: angle to rotate around Z
#         :param b: angle to rotate around X
#         :param c: angle to rotate around Z
#         :return: Quaterion for composed rotation.
#         """
#         return Quat([np.cos(a / 2), 0, 0, np.sin(a / 2)]) @ \
#                Quat([np.cos(b / 2), 0, np.sin(b / 2), 0]) @ \
#                Quat([np.cos(c / 2), np.sin(c / 2), 0, 0])
#
#     def axisangle_to_q(self, v, theta):
#         # convert rotation given by axis 'v' and angle 'theta' to quaternion representation
#         v = v / np.linalg.norm(v)
#         x, y, z = v
#         theta /= 2
#         w = np.cos(theta)
#         x = x * np.sin(theta)
#         y = y * np.sin(theta)
#         z = z * np.sin(theta)
#         return w, x, y, z
#
#     def q_to_axisangle(self, q):
#         # convert from quaternion to rotation given by axis and angle
#         w, v = q[0], q[1:]
#         theta = np.acos(w) * 2.0
#         return v / np.linalg.norm(v), theta
