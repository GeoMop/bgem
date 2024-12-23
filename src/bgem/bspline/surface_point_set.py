"""
TODO:
- implement 3D convex hull and 3D identification of UVZ coordinate system
  first plane fit, then current approach, full 3D transform of points into UVW,
  advantage of having W as Z scaled to unit cube as well
"""
import numpy as np
import pandas as pd
import logging


def convex_hull_2d(sample):
    """
    Args:
        sample: Points in plane as array of shape (N,2)
    Returns: List of points forming the convex hull.
    """
    link = lambda a, b: np.concatenate((a, b[1:]))

    def dome(sample, base):
        """
        Return convex hull of the points on the right side from the base.
        :param sample: Nx2 numpy array of points
        :param base: A segment, np array  [[x0,y0], [x1,y1]]
        :return: np array of points Nx2 on forming the convex hull
        """
        # print("sample: ", len(sample))
        # End points of line.
        h, t = base
        normal = np.dot(((0, -1), (1, 0)), (t - h))
        # Distances from the line.
        dists = np.dot(sample - h, normal)

        outer = sample[dists > np.finfo(float).eps, :]  # extract points on the positive half-plane
        n_outer = len(outer)
        if n_outer == 0:
            return base
        elif n_outer == 1:
            # prevents infinite recursion due to rounding errors
            return [h, outer[0], t]
        else:
            # at least two outer point -> pivot exists
            pivot = sample[np.argmax(dists)]
            return link(dome(outer, [h, pivot]),
                        dome(outer, [pivot, t]))

    if len(sample) > 2:
        x_coords = sample[:, 0]
        # Get left most and right most points.
        base = [sample[np.argmin(x_coords)], sample[np.argmax(x_coords)]]  # extreme points in X coord

        return link(dome(sample, base), dome(sample, base[::-1]))
    else:
        return sample


def min_bounding_rect(hull):
    """
    Compute minimal area bounding box from a convex hull.
    Quadratic algorithm with respect to number of hull points is used, anyway calculation a convex hull
    takes longer since number of hull points is about sqrt of all points.
    :param hull: Nx2 numpy array of the convex hull points. First and last must be the same.
    :return: Corners of the rectangle.
    """
    # Compute edges (x2-x1,y2-y1)
    edges = hull[1:, :] - hull[:-1, :]

    # Calculate edge angles   atan2(y/x)
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])

    # Check for angles in 1st quadrant
    edge_angles = np.abs(edge_angles % (np.pi / 2))

    # Remove duplicate angles
    edge_angles = np.unique(edge_angles)

    # Test each angle to find bounding box with the smallest area
    min_bbox = (0, float("inf"), 0, 0, 0, 0, 0, 0)  # rot_angle, area, width, height, min_x, max_x, min_y, max_y
    for i in range(len(edge_angles)):

        # Create rotation matrix to shift points to baseline
        # R = [ cos(theta)      , cos(theta-PI/2)
        #       cos(theta+PI/2) , cos(theta)     ]
        angle = edge_angles[i]
        R = np.array([[np.cos(angle), np.cos(angle - (np.pi / 2))],
                      [np.cos(angle + (np.pi / 2)), np.cos(angle)]])

        # Apply this rotation to convex hull points
        rot_points = np.dot(R, np.transpose(hull))  # 2x2 * 2xn

        # Find min/max x,y points
        min_x = np.nanmin(rot_points[0], axis=0)
        max_x = np.nanmax(rot_points[0], axis=0)
        min_y = np.nanmin(rot_points[1], axis=0)
        max_y = np.nanmax(rot_points[1], axis=0)

        # Calculate height/width/area of this bounding rectangle
        area = (max_x - min_x) * (max_y - min_y)

        # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
        if (area < min_bbox[1]):
            min_bbox = (edge_angles[i], area, min_x, max_x, min_y, max_y)

    # Re-create rotation matrix for smallest rect
    angle = min_bbox[0]
    R = np.array([[np.cos(angle), np.cos(angle - (np.pi / 2))],
                  [np.cos(angle + (np.pi / 2)), np.cos(angle)]])

    # min/max x,y points are against baseline
    min_x = min_bbox[2]
    max_x = min_bbox[3]
    min_y = min_bbox[4]
    max_y = min_bbox[5]

    # Calculate corner points and project onto rotated frame
    corner_points = np.zeros((4, 2))  # empty 2 column array
    corner_points[0] = np.dot([min_x, max_y], R)
    corner_points[1] = np.dot([min_x, min_y], R)
    corner_points[2] = np.dot([max_x, min_y], R)
    corner_points[3] = np.dot([max_x, max_y], R)

    return corner_points


def scale_relative(points, factor):
    """
    Scale D dim points relative to their barycenter.
    :param points: N x D,
    """
    barycenter = np.mean(points, axis=0)
    return barycenter + factor * (points[:, :] - barycenter[None, :])


class SurfacePointSet:
    @classmethod
    def from_file(cls, filename, delimiter=" ", skip_rows=0):
        """
        Load a sequence of XYZ points on a surface to be approximated.
        Optionally points may have weights (i.e. four values per line: XYZW)
        :param filename: Path to the input text file.
        :return: The approximation object.
        TODO: checkout datatable, should be fastest and without unnecessary dependencies

        """
        # with open(filename, 'r') as f:
        #     point_seq = np.array([l for l in csv.reader(f, delimiter=' ')], dtype=float)
        # point_seq = np.array([l for l in csv.reader(f, delimiter=' ')], dtype=float)

        # too slow: alternatives: loadtxt (16s), csv.reader (1.6s), pandas. read_csv (0.6s)
        # point_seq = np.loadtxt(filename)

        raw_df = pd.read_csv(filename, header=None, sep=delimiter, skiprows=skip_rows, index_col=False,
                             engine="python")
        # raw_df = pd.read_csv(filename, header=None, sep=delimiter, skiprows=skip_rows, index_col=False,
        #                     engine="python")
        point_seq = np.array(raw_df)

        return cls(point_seq)

    @classmethod
    def from_grid_surface(cls, grid_surface):
        """
        Approximation from a GridSurface object. Use grid of Z coords in
        XY positions of poles.
        :param grid_surface: GridSurface.
        :return:
        """
        u_basis, v_basis = grid_surface.u_basis, grid_surface.v_basis

        u_coord = u_basis.make_linear_poles()
        v_coord = v_basis.make_linear_poles()

        U, V = np.meshgrid(u_coord, v_coord)
        uv_points = np.stack([U.ravel(), V.ravel()], axis=1)

        xyz = grid_surface.eval_array(uv_points)
        approx = cls(xyz)
        approx.set_quad(grid_surface.quad)
        return approx

    def __init__(self, points):
        """
        :param points: Nx3 (XYZ) or Nx4 (XYZW - points with weights)
        weights (if given) represents standard deviations of the z coordinate
        """
        assert points.shape[1] >= 3
        self._valid_points = None
        """
        Valid points, necessary to filter zero weights and points out of the prescribed quad, etc.
        Possibly a bit slower then indices but little concern as one can still make a copy if the subset is significantly smaller.
        That also could be possible optimization in the properties.
        """
        # XYZ points
        self._xy_points = points[:, 0:2]
        self._z_points = points[:, 2]
        self._weights = None

        # Bounding quadrilateral of the approximation (currently only parallelograms are supported).
        # Only first three points P0, P1, P2 are considered. V direction is P0 - P1, U direction is P2 - P1.
        # I.e. points are sorted counter-clockwise.
        self._quad = None

        self._uv_points = None
        """ UV coordinates of the XY points."""

        self._u_sorted = None  # indices of points sorted by U coord
        self._v_sorted = None  # indices of points sorted by V coord

        if points.shape[1] > 3:
            self.set_weights(points[:, 3])
        else:
            self._weights = self._weights = np.ones_like(self._z_points)

    def __len__(self):
        return len(self._xy_points)

    @property
    def n_active(self):
        return np.sum(self.valid_points)

    @property
    def xy_points(self):
        return self._xy_points[self.valid_points]

    @property
    def z_points(self):
        return self._z_points[self.valid_points]

    @property
    def weights(self):
        assert self._weights is not None
        return self._weights[self.valid_points]

    @property
    def valid_points(self):
        if self._valid_points is None:
            self._valid_points = np.full_like(self._z_points, True, dtype=bool)
        return self._valid_points

    @property
    def quad(self):
        if self._quad is None:
            self.set_quad(self.compute_default_quad())
        return self._quad

    @property
    def uv_points(self):
        if self._uv_points is None:
            self._update_uv_points()
        return self._uv_points

    @property
    def u_sorted(self):
        # indices of valid points sorted by U coord
        if self._u_sorted is None:
            self._u_sorted = np.argsort(self.uv_points[:, 0])
        return self._u_sorted

    @property
    def v_sorted(self):
        # indices of valid points sorted by U coord
        if self._v_sorted is None:
            self._v_sorted = np.argsort(self.uv_points[:, 1])
        return self._v_sorted

    def _invalidate(self):
        self._valid_points = None
        self._uv_points = None
        self._u_sorted = None
        self._v_sorted = None

    def update_valid_points(self, mask):
        vp = self.valid_points
        self._invalidate()
        self._valid_points = np.logical_and(vp, mask)

    def set_weights(self, weights):
        """
        Explicitly set the weights of the given points.
        """
        self._invalidate()
        if weights is None:
            weights = np.ones_like(self._z_points)
        self._weights = np.atleast_1d(weights)
        mask = self._weights > 0
        self.update_valid_points(mask)

    def set_quad(self, quad, overhang=0.0):
        """
        Set quadrilateral to specify parametric domain (UV square).
        In fact only linear transform is used so the last point can be omitted.
        :param quad: V direction, origin, U direction, UV corner (not used)
        :param overhang: quad enlarged by given factor along all its sides.
        """
        quad = np.atleast_2d(quad)
        assert 3 <= quad.shape[0] <= 4 and quad.shape[1] == 2
        enlarged_quad = scale_relative(quad, 1 + overhang)
        if np.any(enlarged_quad != self._quad):
            self._quad = enlarged_quad
            # reset valid_points from weights
            self.set_weights(self._weights)

    def compute_default_quad(self):
        """
        Compute and set boundary quad as a minimum area bounding box of the input XY point set.
        :return: The quadrilateral vertices.
        """
        hull = convex_hull_2d(self.xy_points)
        return min_bounding_rect(hull)

    def _update_uv_points(self):
        """
        Map XY points to quad, remove points out of quad.
        Results: self._uv_quad_points, self._z_quad_points, self._w_quad_points
        TODO:
        - use valid_points consistently, see what should be the interface provided to SurfaceApprox
        :return:
        """
        xy_shift = self.quad[1, :]
        v_vec = self.quad[0, :] - self.quad[1, :]
        u_vec = self.quad[2, :] - self.quad[1, :]
        mat_uv_to_xy = np.column_stack((u_vec, v_vec))
        mat_xy_to_uv = np.linalg.inv(mat_uv_to_xy)
        points_uv = np.dot((self._xy_points - xy_shift), mat_xy_to_uv.T)

        # remove points far from unit square
        eps = 1.0e-15
        cut_min = np.array([-eps, -eps])
        cut_max = np.array([1 + eps, 1 + eps])
        in_quad_mask = np.all(np.logical_and(cut_min < points_uv, points_uv <= cut_max), axis=1)
        self.valid_points[np.logical_not(in_quad_mask)] = False

        n_out = len(self._xy_points) - np.sum(in_quad_mask)
        logging.warning(f"#{n_out} points out of the surface quad: {self.quad}")

        # snap to unit square
        points_uv = np.maximum(points_uv, np.array([0.0, 0.0]))
        points_uv = np.minimum(points_uv, np.array([1.0, 1.0]))
        self._uv_points = points_uv[self.valid_points]

    def xyz(self):
        new_data = np.empty((self.n_active, 3))
        new_data[:, :2] = self.xy_points
        new_data[:, 2] = self._z_points
        return new_data

    def remove_random(self, n):
        """
        Remove n random points from the set and return them as a new set.
        """
        subset = np.random.choice(np.arange(len(self)), n, replace=False)
        mask = np.ones_like(self._z_points, dtype=bool)
        mask[subset] = False
        new_data = np.empty((n, 4))
        new_data[:, :2] = self._xy_points[subset]
        new_data[:, 2] = self._z_points[subset]
        new_data[:, 3] = self._weights[subset]

        self._invalidate()
        self._xy_points = self._xy_points[mask]
        self._z_points = self._z_points[mask]
        self._valid_points = np.full_like(self._z_points, True)
        self.set_weights(self._weights[mask])
        return SurfacePointSet(new_data)
