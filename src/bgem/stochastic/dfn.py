"""
Module for statistical description of the fracture networks.
It provides appropriate statistical models as well as practical sampling methods.

TODO:
- move pos_distr into Population configuration as well
- shape modification as separate fn,, or part of other population reconfiguration functions (common range)
"""

from typing import *
from pathlib import Path
import numpy as np
import attrs
import json
import yaml
from bgem.stochastic import fr_set

"""
Auxiliary normal manipulation functions.
TODO: tr to remove, incorporate into FractureSet if needed.
"""


@attrs.define
class VonMisesOrientation:
    """
    Distribution for random orientation in 2d.
    X = east, Y = north
    """

    trend: float = 0
    # azimuth (0, 360) of the fractures normal
    concentration: float = 0

    # concentration parameter, 0 = uniformly dispersed, 1 = exact orientation

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

    As the distribution is considered as distribution of the fracture normal vectors we use
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

    # the concentration parameter; 0 = uniform dispersion, infinity - no dispersion

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
        axis_angle = fr_set.normals_to_axis_angles(mean_norm[None, :])
        return fr_set.rotate(raw_normals, axis_angle=axis_angle[0])

    # def sample_axis_angle(self, size=1):
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


Interval = Tuple[float, float]


@attrs.define
class PowerLawSize:
    """
    Truncated Power Law distribution for the fracture size 'r'.
    The density function:

    f(r) = f_0 r ** (-power - 1)

    for 'r' in [size_min, size_max], zero elsewhere.

    The class allows to set a different (usually reduced) sampling range for the fracture sizes,
    one can either use `set_sample_range` to directly set the sampling range or just increase the lower bound to meet
    prescribed fracture intensity via the `set_range_by_intensity` method.

    """
    power = attrs.field(type=float)
    # power of th power law
    diam_range = attrs.field(type=Interval)
    # lower and upper bound of the power law for the fracture diameter (size), values for which the intensity is given
    intensity = attrs.field(type=float)
    # number of fractures with size in the size_range per unit volume (denoted as P30 in SKB reports)

    sample_range = attrs.field(type=Interval)

    # range used for sampling., not part of the statistical description

    # default attrs initiaizer:
    @sample_range.default
    def copy_full_range(self):
        return list(self.diam_range).copy()  # need copy to preserve original range

    @classmethod
    def from_mean_area(cls, power, diam_range, p32, p32_power=None):
        """
        Construct the distribution using the mean arrea (P32) instead of intensity.
        :param power: power law exponent
        :param dim_range: size range for which p32 mean area is given
        :param p32: mean area of the fractures in given `diam_range`.
        :param p32_power: if the mean area is given for different power parameter.
        :return: PowerLawSize instance.
        """
        if p32_power is None:
            p32_power = power
        intensity = cls.intensity_for_mean_area(p32, power, diam_range, p32_exp=p32_power)
        return cls(power, diam_range, intensity)

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
        DEPRECATED Use extract_range for the functional API.
        """
        if sample_range is None:
            sample_range = self.diam_range
        self.sample_range = list(sample_range).copy()

    def extract_range(self, sample_range):
        return PowerLawSize(
            self.power,
            self.diam_range,
            self.intensity,
            sample_range=sample_range)

    def _range_for_intensity(self, intensity, i_bound=0):
        a, b = self.diam_range
        c, d = self.sample_range
        k = self.power
        if i_bound == 0:
            lower_bound = (intensity * (a ** (-k) - b ** (-k)) / self.intensity + d ** (-k)) ** (-1 / k)
            return (lower_bound, self.sample_range[1])
        else:
            upper_bound = (c ** (-k) - intensity * (a ** (-k) - b ** (-k)) / self.intensity) ** (-1 / k)
            return (self.sample_range[0], upper_bound)

    def set_lower_bound_by_intensity(self, intensity):
        """
        Increase lower fracture size bound of the sample range in order to achieve target fracture intensity.
        DEPRECATED
        """
        self.sample_range = self._range_for_intensity(intensity, i_bound=0)

    def set_upper_bound_by_intensity(self, intensity):
        """
        Increase lower fracture size bound of the sample range in order to achieve target fracture intensity.
        DEPRECATED
        """
        self.sample_range = self._range_for_intensity(intensity, i_bound=1)

    def mean_size(self, volume=1.0):
        """
        :return: Mean number of fractures for given volume
        """
        sample_intensity = self.range_intensity(self.sample_range)
        return sample_intensity * volume

    def sample(self, volume, size=None, force_nonempty=False):
        """
        Sample the fracture diameters.
        :param volume: By default, the volume and fracture sample intensity is used to determine actual number of the fractures.
        :param size: ... alternatively the prescribed number of fractures can be generated.
        :param force_nonempty: If True at leas one fracture is generated.
        :return: Array of fracture sizes.
        """
        if size is None:
            size = np.random.poisson(lam=self.mean_size(volume), size=1)
            if force_nonempty:
                size = max(1, size)
        # print("PowerLaw sample: ", force_nonempty, size)
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
    dimensions = attrs.field(type=List[float], converter=np.array)
    center = attrs.field(type=List[float], converter=np.array, default=np.zeros(3))

    # TODO: default center should be dimensions / 2 !! see DIFF

    def sample(self, size=1):
        # size = 1
        # pos = np.empty((size, 3), dtype=float)
        # for i in range(3):
        #    pos[:, i] =  np.random.uniform(self.center[i] - self.dimensions[i]/2, self.center[i] + self.dimensions[i]/2, size)
        pos = np.empty(3, dtype=float)
        return (np.random.random([size, 3]) - 0.5) * self.dimensions[None, :] + self.center[None, :]

    @property
    def volume(self):
        return np.prod(self.dimensions)


@attrs.define
class ConnectedPosition:
    """
    Generate a fracture positions in such way, that all the fractures are connected to some initial surfaces.
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
        :param confining_box: dimensions of axis aligned box, points out of this box are excluded.
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
                # substitute current point for chosen close points
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
        Construct transformation matrix from one square corner three square corners,
        """
        w_vec = np.cross(u_vec, v_vec)
        return np.stack((u_vec, v_vec, w_vec, center), axis=1)


FamilyCfg = Dict[str, Union[str, float, int]]
PopulationDict = Dict[str, FamilyCfg]
PopulationList = Dict[str, FamilyCfg]  # Deprecated, list of Family cfg with "name" attribute
PopulationCfg = Union[PopulationDict, Path, str]


# Population configuration dict/list, or YAML or JASON input file


@attrs.define
class FrFamily:
    """
    Describes a single fracture family with defined distribution of:
     - normal orientation
     - shape orientation
     - size orientation
     - position distribution
     - more complex correlation structure,
     e.g. large fractures with independent orientations smaller with correlated orientations
     needs more general sampling paradigm
    """
    orientation: FisherOrientation
    size: PowerLawSize
    shape_angle: VonMisesOrientation

    name: Optional[str] = None

    # position:
    # correlation: None

    @classmethod
    def from_cfg(cls, family: FamilyCfg, name='') -> 'FrFamily':
        trend = family.get("trend", None)
        plunge = family.get("plunge", None)
        if trend is None or plunge is None:
            # use strike & dip instead
            try:
                trend = family.get("strike") + 90
                plunge = 90 - family.get("dip")
            except KeyError as e:
                print("Incomplete fracture family configuration. Use trend+plunge or strike+dip keys.")
                raise e

        fisher_orientation = FisherOrientation(trend, plunge, family["concentration"])
        size_range = (family["r_min"], family["r_max"])
        if "p_32" in family:
            power_law_size = PowerLawSize.from_mean_area(family["power"], size_range, family["p_32"])
        elif "p_30" in family:
            power_law_size = PowerLawSize(family["power"], size_range, family["p_30"])
        else:
            raise KeyError("Missing p_32 or p_30 key in FrFamily config dictionary.")
        assert np.isclose(family["p_32"], power_law_size.mean_area())
        shape_angle = VonMisesOrientation(trend=0, concentration=0)
        return cls(fisher_orientation, power_law_size, shape_angle, name=name)

    @staticmethod
    def project_cfg(family: FamilyCfg, plane_normal=[0, 0, 1]):
        """

        :param family:
        :param plane_normal:
        :return:
        """
        assert False, "Not implemented yet"
        # Idea is to have specific dict key "2d_angle" that allows to differentiate
        # 3d and 2d configurations.

        orientation = FisherOrientation(0, 90, np.inf)
        size_range = (family["r_min"], family["r_max"])
        power_law_size = PowerLawSize.from_mean_area(family["power"], size_range, family["p_32"])
        assert np.isclose(family["p_32"], power_law_size.mean_area())
        shape_angle = VonMisesOrientation(family["trend"], family["concentration"])
        return FrFamily(family["name"], orientation, power_law_size, shape_angle)

    def with_size_range(self, size_range):
        """
        Copy of the family with modified fracture size range.
        :param size_range:
        :return:
        """
        return FrFamily(self.orientation, self.size.extract_range(size_range), self.shape_angle, self.name)

    def sample(self, position_distribution, shape=fr_set.RectangleShape(), i_fam=0, force_size: int = None):
        """
        Generate FractureSet sample from the FrFamily.
        :param position_distribution:
        :param shape:
        :param i_fam:
        :return:
        TODO: include position distribution into FrFamily, apply different domains as with change in sample_size
        but rather keep both separated from distributions and keep them common to all families at the population level.
        Pass them down when sampling of computing size estimates.
        TODO: add distribution of aspect (log normal with mean 1 and given log_10 sigma)
        """
        radii = self.size.sample(position_distribution.volume, size=force_size)
        aspect = 1.0
        radii = np.stack((radii, aspect * radii), axis=1)
        n_fractures = len(radii)
        shape_angle = self.shape_angle.sample_angle(size=n_fractures)
        shape_axis = np.stack((np.cos(shape_angle), np.sin(shape_angle)), axis=1)
        return fr_set.FractureSet(
            base_shape_idx=shape.id,
            radius=radii,
            normal=self.orientation.sample_normal(size=n_fractures),
            center=position_distribution.sample(size=n_fractures),
            shape_axis=shape_axis,
            family=np.full(n_fractures, i_fam)
        )


@attrs.define
class Population:
    """
    Data class to describe whole population of fractures, several families.
    Supports sampling across the families.
    """
    # Attributes
    families: List[FrFamily]
    # families list
    domain: Tuple[float, float, float]
    # dimensions of the box domain, the Z dimension is = 0 for 2d population
    shape: fr_set.BaseShape = fr_set.RectangleShape()
    # Reference Shape of generated fractures

    __loaders = {
        '.json': json.load,
        '.yaml': yaml.safe_load
    }

    @property
    def volume(self):
        return np.product([l if l > 0 else 1.0 for l in self.domain])

    @staticmethod
    def project_list_to_2d(families: PopulationDict, plane_normal=[0, 0, 1]):
        """
        Convert families as dicts into 2d.
        :return:
        """
        return {k: FrFamily.project_cfg(v, plane_normal) for k, v in families.items()}

    @classmethod
    def from_cfg(cls, families: PopulationDict, box, shape=fr_set.RectangleShape):
        """
        Load families from a list of dict, with keywords: [ name, trend, plunge, concentration, power, r_min, r_max, p_32 ]
        Assuming fixed statistical model: Fischer, Uniform, PowerLaw Poisson
        """
        if isinstance(families, (str, Path)):
            # Load from file.
            path = Path(families)
            with open(path) as f:
                fam_cfg = cls.__loaders[path.suffix](f)
        else:
            fam_cfg = families
        if isinstance(fam_cfg, dict):
            families = [FrFamily.from_cfg(family, name=family_key) for family_key, family in fam_cfg.items()]
        elif isinstance(fam_cfg, list):
            families = [FrFamily.from_cfg(family, name=family['name']) for family in fam_cfg]
        else:
            raise TypeError(
                "Families (possibly loaded from the provided file path) must be either a dictionary or a list of dictionaries with the 'name' item.")

        return cls(families, box, shape)

    # @classmethod
    # def initialize_2d(cls, families: List[Dict[str, Any]], box):
    #     """
    #     Load families from a list of dict, with keywords: [ name, trend, plunge, concentration, power, r_min, r_max, p_32 ]
    #     Assuming fixed statistical model: Fischer, Uniform, PowerLaw Poisson
    #     :param families json_file: JSON file with families data
    #     """
    #     families = [FrFamily.from_cfg_2d(family) for family in families]
    #     assert len(box) == 3 and sum((l > 0 for l in box)) == 2
    #     return cls(box, families)

    @classmethod
    def from_json(cls, json_file, box) -> 'Population':
        """
        Load families from a JSON file. Assuming fixed statistical model: Fischer, Uniform, PowerLaw Poisson
        :param json_file: JSON file with families data
        DEPRECATED use from_cfg
        """
        return cls.from_cfg(json_file, box)

    @classmethod
    def init_from_yaml(cls, yaml_file: str, box) -> 'Population':
        """
        Load families from a YAML file. Assuming fixed statistical model: Fischer, Uniform, PowerLaw Poisson
        :param json_file: YAML file with families data
        DEPRECATED use from _cfg
        """
        return cls.from_cfg(yaml_file, box)

    def mean_size(self):
        """
        Mean number of fractures for the set sample range.
        :return:

        """
        sizes = [family.size.mean_size(self.volume) for family in self.families]
        return sum(sizes)

    def set_range_from_size(self, sample_size):
        """
        :param sample_size:
        :return: Population with new common fracture range.
        """
        return self.set_sample_range(self.common_range_for_sample_size(sample_size))

    def set_sample_range(self, sample_range):
        """
        Set sample range for fracture diameter.
        :param sample_range: (min_bound, max_bound) - one of these can be None if 'sample_size' is provided
                                                      this bound is set to match mean number of fractures
        #:param sample_size: If provided, the None bound is changed to achieve given mean number of fractures.
        #                    If neither of the bounds is None, the lower one is reset.
        #                    DEPRECATED. Use self.set_sample_range(self.common_range_for_sample_size(target_size)
        :return: Population with new common fracture range.
        """
        families = [fam.with_size_range(sample_range) for fam in self.families]
        return Population(families, self.domain, self.shape)
        # min_size, max_size = sample_range
        # for f in self.families:
        #     r_min, r_max = f.size.diam_range
        #     if min_size is not None:
        #         r_min = min_size
        #     if max_size is not None:
        #         r_max = max_size
        #     f.size.set_sample_range((r_min, r_max))
        # if sample_size is not None:
        #     family_sizes = [family.size.mean_size(self.volume) for family in self.families]
        #     total_size = np.sum(family_sizes)
        #
        #     if max_size is None:
        #         for f, size in zip(self.families, family_sizes):
        #             family_intensity = size / total_size * sample_size / self.volume
        #             f.size.set_upper_bound_by_intensity(family_intensity)
        #     else:
        #         for f, size in zip(self.families, family_sizes):
        #             family_intensity = size / total_size * sample_size / self.volume
        #             f.size.set_lower_bound_by_intensity(family_intensity)

    def common_range_for_sample_size(self, sample_size=None, free_bound=0, initial_range=None) -> Interval:
        """
        Compute common size range across families for given mean sample size.
        :param sample_size: Target mean number of fractures in the population. Sum of mean sample sizes over families.
            If None, current mean size is used, so we only compute common size range that preserve same mean sample size.

            Setting the common family bound to obtain prescribed sample size is a nonlinear
            yet monotone problem. Therefore, we apply simple iterating strategy to find correct bound.
            1. split the total sample_size according to current range family intensities
            2. compute common new bound for each family, set the common bound as the median of these
            3. continue to 1. if the estimated number of samples match prescribed sample size with error up to 1.
        :param free_bound index of bound (0-lower, 1-upper) to adapt.
        :param initial_range - initial range of the iterative algorithm; median of family sample ranges by default
        """
        if sample_size is None:
            sample_size = self.mean_size()
        target_total_intenzity = sample_size / self.volume
        if initial_range is None:
            fam_ranges = np.array([f.size.sample_range for f in self.families])
            initial_range = np.median(fam_ranges, axis=0)
        common_range = initial_range

        fn_fam_intensities = lambda range: [f.size.range_intensity(range) for f in self.families]

        def fn_update_ranges(intensities):
            rel_total_intensity = target_total_intenzity / sum(intensities)
            return [f.size._range_for_intensity(intensity * rel_total_intensity, i_bound=free_bound)
                    for f, intensity in zip(self.families, intensities)]

        intensities = fn_fam_intensities(common_range)
        while (sum(intensities) - target_total_intenzity) * self.volume > 1:
            update_ranges = fn_update_ranges(intensities)
            common_range = np.median(update_ranges, axis=0)
            intensities = fn_fam_intensities(common_range)
        return common_range

    def extract_size_range(self, range: Interval) -> 'Population':
        """
        Copy of population with modified size distribution set to prescribed sample range.
        :return: 
        """
        families = [
            FrFamily(fam.orientation, fam.size.extract_range(range), fam.shape_angle, fam.name)
            for fam in self.families
        ]
        return Population(
            families=families,
            domain=self.domain,
            shape=self.shape
        )

    def sample(self, pos_distr=None, keep_nonempty=False) -> fr_set.FractureSet:
        """
        Provide a single fracture set  sample from the population.
        :param pos_distr: Fracture position distribution, common to all families.
        An object with method .sample(size) returning array of positions (size, 3).
        :return: List of FractureShapes.
        TODO: move position distribution into FrFamily for consistency
        TODO: set sample size and seed here, both optional
        """
        if pos_distr is None:
            pos_distr = UniformBoxPosition(self.domain)
        fr_fam_sets = [fam.sample(pos_distr, self.shape, i_fam=i_fam) for i_fam, fam in enumerate(self.families)]
        fracture_set = fr_set.FractureSet.merge(fr_fam_sets, population=self)
        if keep_nonempty and len(fracture_set) == 0:
            fam_probs = [fam.size.mean_size(1.0) for fam in self.families]
            fam_probs = np.array(fam_probs) / np.sum(fam_probs)
            sample = np.random.multinomial(1, fam_probs, size=1)[0]  # Take the single sample.
            i_family = np.argmax(sample)
            fracture_set = self.families[i_family].sample(pos_distr, self.shape, i_fam=i_family, force_size=1)
            fracture_set = fr_set.FractureSet.merge([fracture_set], population=self)
        return fracture_set
        #
        # for ifam, fam in enumerate(self.families):
        #     fr_set.FractureSet(
        #         base_shapes=[self.shape],
        #         shape_idx=0,
        #         radius=fam.size.sample(self.volume),
        #         normal=fam.orientation.sample_normal(size=len(diams)),
        #         center=pos_distr.center(size=len(diams)),
        #         shape_axis=fam.shape_angle.sample_angle(len(diams))
        #     )
        #     #name = fam.name
        #     diams =
        #     fr_normals =
        #     #fr_axis_angle = f.orientation.sample_axis_angle(size=len(diams))
        #     shape_angle =
        #         #np.random.uniform(0, 2 * np.pi, len(diams))
        #     center =
        #
        #     for r, normal, sa in zip(diams, fr_normals, shape_angle):
        #         #axis, angle = aa[:3], aa[3]
        #         center = pos_distr.sample()
        #         fractures.append(Fracture(
        #             shape_class=self.shape,
        #             r=r,
        #             center=center,
        #             normal=normal[None, :],
        #             shape_angle=sa,
        #             family=fam,
        #             aspect=1,
        #             id=name))
        # return fractures


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
