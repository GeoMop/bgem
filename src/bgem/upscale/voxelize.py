from typing import *
import math

import attrs
from functools import cached_property
from bgem.stochastic import Fracture
import numpy as np
"""
Voxelization of fracture network.
Task description:
Input: List[Fracutres], DFN sample, fractures just as gemoetric objects.
Output: Intersection arrays: cell_idx, fracture_idx, intersection volume estimate
(taking fr aperture and intersectoin area into account)

Possible approaches:

- for each fracture -> AABB -> loop over its cells -> intersection test -> area calculation
- Monte Carlo : random points on each fracture (N ~ r**2), count point numbers in each cell, weights -> area/volume estimate
"""



"""
TODO:
1. Port decovalex solution as the basic building block 
    - i.e. contributions to full tensor on cells intersectiong the fracture.
    - Conservative homogenization, slow.
   
   1. AABB grid -> centers of cells within AABB of fracture
   2. Fast selection of active cells: centers_distance < tol
   3. For active cells detect intersection with plane.
      - active cells corners -> projection to fr plane
      - detect nodes in ellipse, local system 
      - alternative function to detect nodes within n-polygon
   4. output: interacting cells, cell cords in local system, optiaonly -> estimate of intersection surface
   5. rasterized full tenzor:
      - add to all interacting cells
      - add multiplied by surface dependent coefficient
       
   
2. Direct homogenization test, with at least 2 cells across fracture.
   Test flow with rasterized fractures, compare different homogenization routines
   ENOUGH FOR SURAO
 
3. For cells in AABB compute distance in parallel, simple fabric tenzor homogenization. 
   Possibly faster due to vectorization, possibly more precise for thin fractures.
4. Comparison of conservative homo + direct upscaling on fine grid with fabric homogenization.

5. Improved determination of candidate cells and distance, differential algorithm.    

"""

class EllipseShape:
    def is_point_inside(self, x, y):
        return x**2 + y**2 <= 1

    def are_points_inside(self, points):
        sq = points ** 2
        return sq[:, 0] + sq[:, 1]  <= 1

def test_EllipseShape():
    ellipse = EllipseShape(6)  # Create a hexagon for testing

    # Points to test
    inside_point = (0.5, 0.5)
    edge_point = (np.cos(np.pi / 6), np.sin(np.pi / 6))  # A point exactly on the edge
    outside_point = (1.5, 1.5)

    # Test for a point inside
    assert ellipse.is_point_inside(*inside_point), "The point should be inside the ellipse"

    # Test for a point on the edge
    # Depending on your implementation's edge handling, this could be either True or False
    # Here we assume points on edges are considered inside
    assert ellipse.is_point_inside(*edge_point), "The point should be considered inside the ellipse"

    # Test for a point outside
    assert not ellipse.is_point_inside(*outside_point), "The point should be outside the ellipse"

    # Test points: inside, on edge, outside
    points = np.array([
        [0.5, 0.5],  # Inside
        [-0.5, -0.5],  # Inside
        [0.86, 0.5],  # On edge (approximately, considering rounding)
        [1, 1],  # Outside
        [0, 0]  # Centrally inside
    ])

    expected_results = np.array([
        True,  # Inside
        True,  # Inside
        False,  # On edge, but due to precision, might be considered outside
        False,  # Outside
        True  # Centrally inside
    ])

    actual_results = ellipse.are_points_inside(points)
    np.testing.assert_array_equal(actual_results, expected_results,
                                  "The are_points_inside method failed to accurately determine if points are inside the ellipse.")


class PolygonShape:
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

def test_PolyShape():
    polygon = PolygonShape(6)  # Create a hexagon for testing

    # Points to test
    inside_point = (0.5, 0.5)
    edge_point = (np.cos(np.pi / 6), np.sin(np.pi / 6))  # A point exactly on the edge
    outside_point = (1.5, 1.5)

    # Test for a point inside
    assert polygon.is_point_inside(*inside_point), "The point should be inside the polygon"

    # Test for a point on the edge
    # Depending on your implementation's edge handling, this could be either True or False
    # Here we assume points on edges are considered inside
    assert polygon.is_point_inside(*edge_point), "The point should be considered inside the polygon"

    # Test for a point outside
    assert not polygon.is_point_inside(*outside_point), "The point should be outside the polygon"

    # Test points: inside, on edge, outside
    points = np.array([
        [0.5, 0.5],  # Inside
        [-0.5, -0.5],  # Inside
        [0.86, 0.5],  # On edge (approximately, considering rounding)
        [1, 1],  # Outside
        [0, 0]  # Centrally inside
    ])

    expected_results = np.array([
        True,  # Inside
        True,  # Inside
        False,  # On edge, but due to precision, might be considered outside
        False,  # Outside
        True  # Centrally inside
    ])

    actual_results = polygon.are_points_inside(points)
    np.testing.assert_array_equal(actual_results, expected_results,
                                  "The are_points_inside method failed to accurately determine if points are inside the polygon.")


@attrs.define
class FractureVoxelize:
    """
    Auxiliary class with intersection of fractures with a (structured, rectangular) grid.
    The class itslef could be used for any types of elements, but the supported voxelization algorithms
    are specific for the uniform rectangular grid, allowing different step for each of X, Y, Z directions.

    The intersections could be understood as a sparse matrix for computing cell scalar property as:
    i - grid index, j - fracture index
    grid_property[i] = (1 - sum_j intersection[i, j]) * bulk_property[i] + sum_j intersection[i, j] * fr_property[j]

    The sparse matrix 'intersection' is formed in  terms of the triplex lists: cell_id, fracture_id, volume.
    It actualy is intersection_volume[i,j] / cell_volume[i] , the cell_volume is minimum of the volume of the i-th cell
    and sum of volumes of the intersectiong fracutres.

    The cached properties for the bulk weight vector and fracture interpolation sparse matrix for efficient multiplication
    are provided.
    """
    grid: 'Grid'            # Any grid composed of numbered cells.
    cell_ids: List[int]     # For each intersection the cell id.
    fr_ids: List[int]       # For each intersection the fracture id.
    volume: List[float]       # For each intersection the intersection fracture volume estimate.



    @cached_property
    def cell_fr_sums(self):
        cell_sums = np.zeros(, dtype=np.float64)


    def project_property(self, fr_property, bulk_property):


class FractureBoundaries3d:
    @staticmethod
    def build(polygons):
        n_fractures, n_points, dim = polygons
        assert dim == 3
        assert n_points % 2 == 0

        # Get AABB and sort coordinates from largest to smallest
        aabb_min = polygons.min(axis=1)
        aabb_max = polygons.max(axis=1)
        aabb_ptp = aabb_max - aabb_min
        axes_sort = np.argsort(-aabb_ptp, axis=1)
        aabb_min_sort = aabb_min[:, axes_sort]
        aabb_max_sort = aabb_max[:, axes_sort]
        polygons_sort = polygons[:, :, axes_sort]
        # for evary fracture get sequence of points from >=X_min to <X_max
        # half of the points, we could not be sure if we get lower or upper arc
        argmin_X = np.argmin(polygons_sort[:, :, 2], axis=1)
        #flag_upper_Y = polygons[:, :, 1] > (aabb_min_sort[:, 1] + aabb_min_sort[:, 1]) / 2

        # half of points + 1 to get the end point as well.
        # We get other half by central symmetry.
        selected_indices = (argmin_X[:, None] + np.arange(n_points // 2 + 1)[None, :]) % n_points

        o_grid = np.ogrid[:n_fractures, :3]
        all_fractures = np.arange(n_fractures)[:, None, None]
        all_dims = np.arange(3)[None, None, :]
        half_arc = polygons[all_fractures, selected_indices[:, :, None], all_dims]
        """
        1. Use half arc to generate Y ranges in the X range. 
        This produces variable size arrays and could not be implemented in Numpy efficeintly.
        Use classical loop over fractures and over lines. Generate list of XY celles, compute estimate of XY projection,
        interior cells and list of boundary cells. 
        Interior cells - use normal, Z distance from center, and fracture aperture
        to determine tensor contribution, multiply by XY projection for the boundary cells.
        """


def form_table():
    pass


def unit_area_tab(x, y, z_slack):
    """
    Assume 1 > x > y > 0.
    1 > z_slack > 0
    :return: approx area of intersection of fracture plane in distance z_slack from origin

    """


def tensor_contribution(normal, slack, slack_axis, aperture):
    """
    Compute contribution to the cell equivalent/fabric tensor.
    We assume aperture et most 1/10 of min cell dimension.

    normal - fracture normal vector
    slack - vector from cel center to fracture with single nonzero component, the minimum one.
            should be relatively close to normal (up to orientation)
            angle to normal on unit cube at most 50degs
    aperture of the fracture
    :return: 3d fabric tensor

    1. scale to unit cell
    2. approx surface of intersection on unit cell
    3. scale surface back
    3. tn = surf * apperture / 1 * (n otimes n)
    4. scale back

    ===
    - We will scale whole coord system to have unit cells, possibly scaling back individual equivalent tensors
      or fracture eq. tensors.
      This will guarantee that slack direction is that max. component of the normal.
    """
    normal_reminder = np.abs(np.delete(normal, slack_axis)) / normal[slack_axis]
    normal_rel_max = np.max(normal_reminder)
    normal_rel_min = np.min(normal_reminder)
    area = unit_area_tab(normal_rel_max, normal_rel_min, slack)
    rel_area = aperture * area  / np.dot(normal, cell)
    tn = rel_area * normal[:, None] * normal[None, :]
    return tn



__rel_corner = np.array([[0, 0, 0], [1, 0, 0],
                         [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1],
                         [1, 1, 1], [0, 1, 1]])

def intersect_cell(loc_corners: np.array, ellipse: Fracture) -> bool:
    """
    loc_corners - shape (3, 8)
    """
    # check if cell center is inside radius of fracture
    center = np.mean(loc_corners, axis=1)
    if np.sum(np.square(center[0:2] / ellipse.radius)) >= 1:
        return False

    # cell center is in ellipse
    # find z of cell corners in xyz of fracture

    if np.min(loc_corners[2, :]) >= 0. or np.max(loc_corners[2, :]) <= 0.:
        # All coreners on one side => no intersection.
        return False

    return True

def fracture_for_ellipse(grid: Grid, i_ellipse: int, ellipse: Ellipse) -> Fracture:
    # calculate rotation matrix for use later in rotating coordinates of nearby cells
    direction = np.cross([0,0,1], ellipse.normal)
    #cosa = np.dot([0,0,1],normal)/(np.linalg.norm([0,0,1])*np.linalg.norm(normal)) #frobenius norm = length
    #above evaluates to normal[2], so:
    angle = np.arccos(ellipse.normal[2]) # everything is in radians
    mat_to_local = tr.rotation_matrix(angle, direction)[:3, :3].T
    #find fracture in domain coordinates so can look for nearby cells
    b_box_min = ellipse.translation - np.max(ellipse.radius)
    b_box_max = ellipse.translation + np.max(ellipse.radius)

    i_box_min = grid.cell_coord(b_box_min)
    i_box_max = grid.cell_coord(b_box_max) + 1
    axis_ranges = [range(max(0, a), min(b, n)) for a, b, n in zip(i_box_min, i_box_max, grid.cell_dimensions)]

    grid_cumul_prod = np.array([1, grid.cell_dimensions[0], grid.cell_dimensions[0] * grid.cell_dimensions[1]])
    cells = []
    # X fastest running
    for ijk in itertools.product(*reversed(axis_ranges)):
        # make X the first coordinate
        ijk = np.flip(np.array(ijk))
        corners = grid.origin[None, :] + (ijk[None, :] + __rel_corner[:, :]) * grid.step[None, :]
        loc_corners = mat_to_local @ (corners - ellipse.translation).T
        if intersect_cell(loc_corners, ellipse):
            logging.log(logging.DEBUG, f"       cell {ijk}")
            cell_index = ijk @ grid_cumul_prod
            cells.append(cell_index)
    if len(cells) > 0:
        logging.log(logging.INFO, f"   #{i_ellipse} fr, {len(cells)} cell intersections")
    return Fracture(ellipse, cells)

def map_dfn(grid, ellipses):
    '''Identify intersecting fractures for each cell of the ECPM domain.
     Extent of ECPM domain is determined by nx,ny,nz, and d (see below).
     ECPM domain can be smaller than the DFN domain.
     Return numpy array (fracture) that contains for each cell:
     number of intersecting fractures followed by each intersecting fracture id.

     ellipses = list of dictionaries containing normal, translation, xrad,
                and yrad for each fracture
     origin = [x,y,z] float coordinates of lower left front corner of DFN domain
     nx = int number of cells in x in ECPM domain
     ny = int number of cells in y in ECPM domain
     nz = int number of cells in z in ECPM domain
     step = [float, float, float] discretization length in ECPM domain

     JB TODO: allow smaller ECPM domain
    '''
    logging.log(logging.INFO, f"Callculating Fracture - Cell intersections ...")
    return [fracture_for_ellipse(grid, ie, ellipse) for ie, ellipse in enumerate(ellipses)]


