from typing import *
from pathlib import Path
import csv
import math

import attrs
from functools import cached_property
from bgem.stochastic import Fracture
from bgem.upscale import Grid
from bgem.stochastic import FractureSet, EllipseShape, PolygonShape
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
1. 1. AABB grid -> centers of cells within AABB of fracture
   2. Fast selection of active cells: centers_distance < tol
   3. For active cells detect intersection with plane.
      - active cells corners -> projection to fr plane
      - detect nodes in ellipse, local system 
      - alternative function to detect nodes within n-polygon
   4. output: tripples: i_fracture, i_cell, cell_distance for each intersection
      to get consistent conductivity along  voxelized fracture, we must modify cells within normal*grid.step band 
   5. rasterized full tenzor:
      - add to all interacting cells
      - add multiplied by distance dependent coefficient
       
   
2. Direct homogenization test, with at least 2 cells across fracture.
   Test flow with rasterized fractures, compare different homogenization routines
   ENOUGH FOR SURAO
 
3. For cells in AABB compute distance in parallel, simple fabric tenzor homogenization. 
   Possibly faster due to vectorization, possibly more precise for thin fractures.
4. Comparison of conservative homo + direct upscaling on fine grid with fabric homogenization.

5. Improved determination of candidate cells and distance, differential algorithm.    

"""


@attrs.define
class FracturedMedia:
    dfn: FractureSet                #
    fr_cross_section: np.ndarray    # shape (n_fractures,)
    fr_conductivity: np.ndarray     # shape (n_fractures,)
    conductivity: float


    @staticmethod
    def fracture_cond_params(dfn :FractureSet, unit_cross_section, bulk_conductivity):
        # unit_cross_section = 1e-4
        viscosity = 1e-3
        gravity_accel = 10
        density = 1000
        permeability_factor = 1 / 12
        permeability_to_conductivity = gravity_accel * density / viscosity
        # fr cond r=100 ~ 80
        # fr cond r=10 ~ 0.8
        fr_r = np.array([fr.r for fr in dfn])
        fr_cross_section = unit_cross_section * fr_r
        fr_cond = permeability_to_conductivity * permeability_factor * fr_r ** 2
        fr_cond = np.full_like(fr_r, 10)
        return FracturedMedia(dfn, fr_cross_section, fr_cond, bulk_conductivity)

    @classmethod
    def _read_dfn_file(cls, f_path):
        with open(f_path, 'r') as file:
            rdr = csv.reader(filter(lambda row: row[0] != '#', file), delimiter=' ', skipinitialspace=True)
            return [row for row in rdr]

    @classmethod
    def from_dfn_works(cls, input_dir: Union[Path, str], bulk_conductivity):
        '''
        Read dfnWorks-Version2.0 output files:
        normal_vectors.dat - three values per line, normal vectors
        translations.dat - three values per line, fracture centers,
                        'R' marks isolated fracture, currently ignored
        radii.dat - three values per line: (major_r, minor_r, shape_family)
                shape_family: -1 = RectangleShape, 0 = EllipseShape, >0 fracture family index
                (unfortunate format as it mixes two different attributes ahape and fracture statistical family, which are independent)
        perm.dat - 6 values per line; 4th is permitivity
        aperture.dat - 4 values per line; 4th is apperture
        polygons.dat - not used, DFN triangulation

        :param source_dir: directory with the files
        :param bulk_conductivity: background / bulk conductivity
            (constant only)
        :return: FracturedMedia
        '''
        __radiifile = 'radii.dat'
        __normalfile = 'normal_vectors.dat'
        __transfile = 'translations.dat'
        __permfile = 'perm.dat'
        __aperturefile = 'aperture.dat'
        workdir = Path(input_dir)

        radii = np.array(cls._read_dfn_file(workdir / __radiifile), dtype=float)
        n_frac = radii.shape[0]
        shape_family = radii[:, 2]
        radii = radii[:, 0:2]
        assert radii.shape[1] == 2
        normals = np.array(cls._read_dfn_file(workdir / __normalfile), dtype=float)
        assert normals.shape == (n_frac, 3)
        translations = np.array([t for t in cls._read_dfn_file(workdir / __transfile) if t[-1] != 'R'], dtype=float)
        assert translations.shape == (n_frac, 3)
        permeability = np.array(cls._read_dfn_file(workdir / __permfile), dtype=float)[:, 3]
        apperture = np.array(cls._read_dfn_file(workdir / __aperturefile), dtype=float)[:, 3]
        shape_idx = np.zeros(n_frac)
        dfn = FractureSet([EllipseShape], shape_idx, radius, center, normal)
        return cls(normals, centers, radii, )

        ellipses = [Ellipse(np.array(n), np.array(t), np.array(r)) for n, t, r in zip(normals, translations, radii)]
        return ellipses


@attrs.define
class Intersection:
    grid: Grid
    fractures: List[Fracture]
    i_fr_cell: np.ndarray
    factor: np.ndarray = None


def intersections_centers(grid: Grid, fractures: List[Fracture]):
    """
    Estimate intersections between grid cells and fractures

    1. for all fractures compute what could be computed in vector fassion
    2. for each fracture determine cell centers close enough
    3. compute XY local coords and if in the Shape
    """
    fr_normal = np.array([fr.normal for fr in fractures])
    fr_center = np.array([fr.center for fr in fractures])
    import decovalex_dfnmap as dmap

    ellipses = [dmap.Ellipse(fr.normal, fr.center, fr.scale) for fr in fractures]
    d_grid = dmap.Grid.make_grid(grid.origin, grid.step, grid.dimensions)
    d_fractures = dmap.map_dfn(d_grid, ellipses)
    i_fr_cell = np.stack([(i_fr, i_cell)  for i_fr, fr in enumerate(d_fractures) for i_cell in fr.cells])
    #fr, cell = zip([(i_fr, i_cell)  for i_fr, fr in enumerate(fractures) for i_cell in fr.cells])
    return Intersection(grid, fractures, i_fr_cell, None)


def intersections_decovalex(grid: Grid, fractures: List[Fracture]):
    """
    Estimate intersections between grid cells and fractures

    Temporary interface to original map_dfn code inorder to perform one to one test.
    """
    import decovalex_dfnmap as dmap

    ellipses = [dmap.Ellipse(fr.normal, fr.center, fr.scale) for fr in fractures]
    d_grid = dmap.Grid.make_grid(grid.origin, grid.step, grid.dimensions)
    d_fractures = dmap.map_dfn(d_grid, ellipses)
    i_fr_cell = np.stack([(i_fr, i_cell)  for i_fr, fr in enumerate(d_fractures) for i_cell in fr.cells])
    #fr, cell = zip([(i_fr, i_cell)  for i_fr, fr in enumerate(fractures) for i_cell in fr.cells])
    return Intersection(grid, fractures, i_fr_cell, None)

def perm_aniso_fr_values(fractures, fr_transmisivity: np.array, grid_step) -> np.ndarray:
    '''Calculate anisotropic permeability tensor for each cell of ECPM
       intersected by one or more fractures. Discard off-diagonal components
       of the tensor. Assign background permeability to cells not intersected
       by fractures.
       Return numpy array of anisotropic permeability (3 components) for each
       cell in the ECPM.

       fracture = numpy array containing number of fractures in each cell, list of fracture numbers in each cell
       ellipses = [{}] containing normal and translation vectors for each fracture
       T = [] containing intrinsic transmissivity for each fracture
       d = length of cell sides
       k_background = float background permeability for cells with no fractures in them
    '''
    assert len(fractures) == len(fr_transmisivity)
    # Construc array of fracture tensors
    def full_tensor(n, fr_cond):
        normal = np.array(n)
        normal_axis_step = grid_step[np.argmax(np.abs(n))]
        return fr_cond * (np.eye(3) - normal[:, None] * normal[None, :]) / normal_axis_step

    return np.array([full_tensor(fr.normal, fr_cond)  for fr, fr_cond in zip(fractures, fr_transmisivity)])


def perm_iso_fr_values(fractures, fr_transmisivity: np.array, grid_step) -> np.ndarray:
    '''Calculate isotropic permeability for each cell of ECPM intersected by
     one or more fractures. Sums fracture transmissivities and divides by
     cell length (d) to calculate cell permeability.
     Assign background permeability to cells not intersected by fractures.
     Returns numpy array of isotropic permeability for each cell in the ECPM.

     fracture = numpy array containing number of fractures in each cell, list of fracture numbers in each cell
     T = [] containing intrinsic transmissivity for each fracture
     d = length of cell sides
     k_background = float background permeability for cells with no fractures in them
    '''
    assert len(fractures) == len(fr_transmisivity)
    fr_norm = np.array([fr.normal for fr in fractures])
    normalised_transmissivity = fr_transmisivity / grid_step[np.argmax(np.abs(fr_norm), axis=1)]
    return normalised_transmissivity

def _conductivity_decovalex(fr_media: FracturedMedia, grid: Grid, fr_values_fn):
    isec  = intersections_decovalex(grid, fr_media.dfn)
    fr_transmissivity = fr_media.fr_conductivity * fr_media.fr_cross_section
    fr_values = fr_values_fn(isec.fractures, fr_transmissivity, isec.grid.step)
    # accumulate tensors in cells
    ncells = isec.grid.n_elements
    k_aniso = np.full((ncells, *fr_values.shape[1:]), fr_media.conductivity, dtype=np.float64)
    np.add.at(k_aniso, isec.i_fr_cell[:,1], fr_values[isec.i_fr_cell[:,0]])
    return k_aniso #arange_for_hdf5(grid, k_iso).flatten()

def permeability_aniso_decovalex(fr_media: FracturedMedia, grid: Grid):
    return _conductivity_decovalex(fr_media, grid, perm_aniso_fr_values)

def permeability_iso_decovalex(fr_media: FracturedMedia, grid: Grid):
    return _conductivity_decovalex(fr_media, grid, perm_iso_fr_values)





def aniso_lump(tn_array):
    """
    Convert array of full anisotropic tensors to the array of diagonal
    tensors by lumping (summing) tensor rows to the diagonal.
    :param tn_array: shape (n, k, k)
    """
    assert len(tn_array.shape) == 3
    assert tn_array.shape[1] == tn_array.shape[2]
    return np.sum(tn_array, axis=-1)[:, None, :] * np.eye(3)

def aniso_diag(tn_array):
    """
    Convert array of full anisotropic tensors to the array of diagonal
    tensors by extraction only diagonal elements.
    :param tn_array: shape (n, k, k)
    """
    assert len(tn_array.shape) == 3
    assert tn_array.shape[1] == tn_array.shape[2]
    return tn_array * np.eye(3)[None, :, :]




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



    # @cached_property
    # def cell_fr_sums(self):
    #     cell_sums = np.zeros(, dtype=np.float64)
    #

    def project_property(self, fr_property, bulk_property):
        pass

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

# def fracture_for_ellipse(grid: Grid, i_ellipse: int, ellipse: Ellipse) -> Fracture:
#     # calculate rotation matrix for use later in rotating coordinates of nearby cells
#     direction = np.cross([0,0,1], ellipse.normal)
#     #cosa = np.dot([0,0,1],normal)/(np.linalg.norm([0,0,1])*np.linalg.norm(normal)) #frobenius norm = length
#     #above evaluates to normal[2], so:
#     angle = np.arccos(ellipse.normal[2]) # everything is in radians
#     mat_to_local = tr.rotation_matrix(angle, direction)[:3, :3].T
#     #find fracture in domain coordinates so can look for nearby cells
#     b_box_min = ellipse.translation - np.max(ellipse.radius)
#     b_box_max = ellipse.translation + np.max(ellipse.radius)
#
#     i_box_min = grid.cell_coord(b_box_min)
#     i_box_max = grid.cell_coord(b_box_max) + 1
#     axis_ranges = [range(max(0, a), min(b, n)) for a, b, n in zip(i_box_min, i_box_max, grid.cell_dimensions)]
#
#     grid_cumul_prod = np.array([1, grid.cell_dimensions[0], grid.cell_dimensions[0] * grid.cell_dimensions[1]])
#     cells = []
#     # X fastest running
#     for ijk in itertools.product(*reversed(axis_ranges)):
#         # make X the first coordinate
#         ijk = np.flip(np.array(ijk))
#         corners = grid.origin[None, :] + (ijk[None, :] + __rel_corner[:, :]) * grid.step[None, :]
#         loc_corners = mat_to_local @ (corners - ellipse.translation).T
#         if intersect_cell(loc_corners, ellipse):
#             logging.log(logging.DEBUG, f"       cell {ijk}")
#             cell_index = ijk @ grid_cumul_prod
#             cells.append(cell_index)
#     if len(cells) > 0:
#         logging.log(logging.INFO, f"   #{i_ellipse} fr, {len(cells)} cell intersections")
#     return Fracture(ellipse, cells)
#
# def map_dfn(grid, ellipses):
#     '''Identify intersecting fractures for each cell of the ECPM domain.
#      Extent of ECPM domain is determined by nx,ny,nz, and d (see below).
#      ECPM domain can be smaller than the DFN domain.
#      Return numpy array (fracture) that contains for each cell:
#      number of intersecting fractures followed by each intersecting fracture id.
#
#      ellipses = list of dictionaries containing normal, translation, xrad,
#                 and yrad for each fracture
#      origin = [x,y,z] float coordinates of lower left front corner of DFN domain
#      nx = int number of cells in x in ECPM domain
#      ny = int number of cells in y in ECPM domain
#      nz = int number of cells in z in ECPM domain
#      step = [float, float, float] discretization length in ECPM domain
#
#      JB TODO: allow smaller ECPM domain
#     '''
#     logging.log(logging.INFO, f"Callculating Fracture - Cell intersections ...")
#     return [fracture_for_ellipse(grid, ie, ellipse) for ie, ellipse in enumerate(ellipses)]


