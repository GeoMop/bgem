from typing import *
from pathlib import Path
import csv
import itertools
import attrs
from functools import cached_property
from bgem.stochastic import Fracture
from bgem.core import array_attr
from bgem.upscale import Grid
from bgem.stochastic import FractureSet, EllipseShape
import numpy as np

"""
Voxelization of fracture network.
Task description:
Input: List[Fractures], DFN sample, fractures just as geometric objects.
Output: Intersection arrays: cell_idx, fracture_id  x, intersection volume estimate
(taking fr aperture and intersection area into account)
That could be rather encoded into a sparse matrix or two for interpolation 
of bulk and fracture values to the field on the target domain. 
While the separate matrix for bulk and for fracture values may be better in some cases,
all use cases are covered by a single interpolation matrix multiplying a vector composed 
of both fracture and bulk values.
Covered cases:
- input fields on a VTK/GMSH mesh (suitable internal format for the mesh that ideally separates array for triangles, and the array for tetrahedrons
- bulk on separate grid, fracture fields constant on fractures or their fragments
- bulk on the trager grid, ...
- bulk on any points first interpolated to the target grid

TODO:
1. More restricted API.
   a) function to project bulk fields between grids, first implement using SCIPY interpolation.
   b) computing intersection matrices to combine bulk field on the target grid and fracture field values
      into the target grid field.
   c) function to apply intersection object to particular bulk and fracture fields (support of fields of arbitrary shape: scalar, vector, tensor valued fields)
       
2. Future API using connected bulk and fracture field vectors and common sparse matrix.
   Design after experience with restricted API. 

Possible intersection approaches:

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
   4. output: triples: i_fracture, i_cell, cell_distance for each intersection
      to get consistent conductivity along  voxelized fracture, we must modify cells within normal*grid.step band 
   5. rasterized full tensor:
      - add to all interacting cells
      - add multiplied by distance dependent coefficient
       
   
2. Direct homogenization test, with at least 2 cells across fracture.
   Test flow with rasterized fractures, compare different homogenization routines
   ENOUGH FOR SURAO
 
3. For cells in AABB compute distance in parallel, simple fabric tensor homogenization. 
   Possibly faster due to vectorization, possibly more precise for thin fractures.
4. Comparison of conservative homo + direct upscaling on fine grid with fabric homogenization.

5. Improved determination of candidate cells and distance, differential algorithm.    

"""


def base_shape_interior_grid(shape, step: float) -> np.ndarray:
    """
    Return points of a grid that are inside the reference shape.
    The grid is with `step` resolution extending from origin
    in the XY reference plane of the shape.
    Return shape (N, 2)
    """
    aabb = shape.aabb
    points = Grid.from_step(aabb[1] - aabb[0], step, origin=aabb[0]).barycenters()
    are_inside = shape.are_points_inside(points)
    selected_pts = points[are_inside]
    return selected_pts


@attrs.define
class FracturedDomain:
    """
    Structured bulk grid + unstructured fracture set.
    The input of the voxelization procedure.

    Specification of the fracture - target grid geometry.
    This is in priciple enough to construct basic voxelization case.
    other cases may need information about source grid/mesh.
    """
    dfn: FractureSet  #
    fr_cross_section: np.ndarray  # cross_sections of fractures
    grid: Grid  # target homogenization grid


@attrs.define
class Intersection:
    """
    Intersection of fractures with the grid.
    That is a sparse matrix for contribution of the fractures,
    1 - rowsum  is scaling factor of the underlying bulk array.

    First we will proceed with this design moving to actual sparse matrix implementation later on.
    The interpolation would be:

    result_grid_field[i_cell] = (1-rowsum[i_cell])/cell_volume[i_cell] * bulk_grid_field[i_cell]
                                + volume[k] * (cells[k] == i_cell) * (fracture[k] == i_fr) * fr_field[i_fr]
    The sparse
    TODO: refine design, try to use sparse matrix for interpolation of a connected bulk-fracture values vector
    """
    domain = attrs.field(type=FracturedDomain)  # The source object.
    i_cell = array_attr(shape=(-1,), dtype=int)  # sparse matrix rows, cell idx of intersection
    i_fr = array_attr(shape=(-1,), dtype=int)  # sparse matrix columns
    isec = array_attr(shape=(-1,), dtype=float)  # effective volume of the intersection

    # bulk_scale: np.ndarray    #
    # used to scale bulk field

    @classmethod
    def const_isec(cls, domain, i_cell, i_fr, isec):
        assert len(i_cell) == len(i_fr)
        isec = np.broadcast_to([isec], (len(i_cell),))
        return cls(domain, i_cell, i_fr, isec)

    @property
    def grid(self):
        return self.domain.grid

    @cached_property
    def bulk_scale(self):
        scale = np.ones(self.grid.n_elements, dtype=float)
        scale[self.i_cell[:]] -= self.isec[:]  # !! have to broadcast isec, use convertor from fr_set
        return scale

    def cell_field(self):
        field = np.zeros(self.grid.n_elements)
        field[self.i_cell] = 1.0
        return field

    def count_fr_cells(self):
        """
        Array of count of intersecting cells for every fracture.
        :return:
        """
        fr_counts = np.zeros(len(self.domain.dfn))
        unique_values, counts = np.unique(self.i_fr, return_counts=True)
        fr_counts[unique_values] = counts
        return fr_counts

    def interpolate(self, bulk_field, fr_field, source_grid=None):
        """
        Rasterize bulk and fracture fields to the target grid, i.e. self.grid.
        If source_Grid is given the bulk filed is first resampled to the target grid
        using linear interpolation and scipy.


        :param bulk_field:
        :param fr_field:
        :param source_grid:
        :return:
        """
        # if source_grid is not None:
        #     assert np.allclose(np.array(source_grid.origin), np.array(self.grid.origin))
        #     assert np.allclose(np.array(source_grid.dimensions), np.array(self.grid.dimensions))
        #     grid_points = source_grid.axes_cell_coords()
        #     target_points = self.grid.barycenters()
        #     # !! Interpolation problem, we have piecewise values at input, but want to interpolate them linearly to the output grid
        #     # finner output grid points are out of the range of the input grid.
        #     bulk_field = interpolate.interpn(grid_points,
        #                                      bulk_field.reshape(*source_grid.shape, *bulk_field.shape[1:]),
        #                                      target_points, method='linear')

        assert len(bulk_field) == self.domain.grid.n_elements
        assert len(fr_field) == len(self.domain.dfn)
        len_value_shape = len(bulk_field.shape) - 1
        scalar_shape = (-1, *(len_value_shape * [1]))
        combined = bulk_field * self.bulk_scale.reshape(scalar_shape)
        combined[self.i_cell[:]] += self.isec[:].reshape(scalar_shape) * fr_field[self.i_fr[:]]
        return combined

    def fr_tensor_2(self, fr_cond_scalar):
        """

        :return:
        """
        dfn = self.domain.dfn
        # normal_axis_step = grid_step[np.argmax(np.abs(n))]
        return fr_cond_scalar[:, None, None] * (
                    np.eye(3) - dfn.normal[:, :, None] * dfn.normal[:, None, :])  # / normal_axis_step

    def perm_aniso_fr_values(fractures, fr_transmisivity: np.array, grid_step) -> np.ndarray:
        '''Calculate anisotrop
            assert source_grid.origin == self.originic permeability tensor for each cell of ECPM
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

        return np.array([full_tensor(fr.normal, fr_cond) for fr, fr_cond in zip(fractures, fr_transmisivity)])

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


def intersection_decovalex(dfn: FractureSet, grid: Grid) -> 'Intersection':
    """
    Based on DFN map / decovalex 2023 approach. Support for different fracture shapes,
    vectorization.
    Steps:
    1. for fractures compute arrays that could be computed by vector operations:
        - fracture normal
        - fracture transform matrix
        - fracture angle (??)
        - bounding box (depend on shape)
    2. estimate set of candidate cells:
        - bounding box
        - future: axis closes to normal for each point in AABB projection
          determine fracture intersection (regular pattern), add 4/8 neighbor cells
          Only do for larger fractures
    3. project nodes of candidate cells, need node_i_coord to i_node map,
       use 2D matrix of the AABB projection
    4. Fast identification of celles within distance range from the fracture
    5. Shape matching of the cells.
    6. cell to fracture distance estimate
        what could be computed in vector fassion
    2. for each fracture determine cell centers close enough
    3. compute XY local coords and if in the Shape

    :param domain:
    :return:
    """
    """
    Estimate intersections between grid cells and fractures

    Temporary interface to original map_dfn code inorder to perform one to one test.
    """
    import bgem.upscale.decovalex_dfnmap as dmap
    assert dfn.base_shape_idx == EllipseShape.id

    domain = FracturedDomain(dfn, np.ones(len(dfn)), grid)
    ellipses = [dmap.Ellipse(fr.normal, fr.center, fr.scale * fr.shape.R) for fr in dfn]
    d_grid = dmap.Grid.make_grid(domain.grid.origin, domain.grid.step, grid.dimensions)
    d_fractures = dmap.map_dfn(d_grid, ellipses)
    i_pairs = [(i_c, i_f) for i_f, fr in enumerate(d_fractures) for i_c in fr.cells]
    if i_pairs:
        i_cell, i_fr = zip(*i_pairs)
    else:
        i_cell = []
        i_fr = []
    # fr, cell = zip([(i_fr, i_cell)  for i_fr, fr in enumerate(fractures) for i_cell in fr.cells])
    return Intersection.const_isec(domain, i_cell, i_fr, 1.0)


__rel_corner = np.array([[0, 0, 0], [1, 0, 0],
                         [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1],
                         [1, 1, 1], [0, 1, 1]])


def intersect_cell(loc_corners: np.array, shape) -> bool:
    """
    loc_corners - shape (3, 8)
    """
    # check if cell center is inside radius of fracture
    center = np.mean(loc_corners, axis=1)
    if not shape.is_point_inside(*center[:2]):
        return False

    # cell center is in ellipse
    # find z of cell corners in xyz of fracture

    if np.min(loc_corners[2, :]) >= 0. or np.max(loc_corners[2, :]) < 0.:  # fracture lies in z=0 plane
        # fracture intersects that cell
        return False

    return True


def intersection_cell_corners(dfn: FractureSet, grid: Grid) -> 'Intersection':
    domain = FracturedDomain(dfn, np.ones(len(dfn)), grid)

    i_cell = []
    i_fr = []
    for i in range(len(dfn)):
        i_box_min, i_box_max = grid.coord_aabb(dfn.AABB[i])
        axis_ranges = [range(max(0, a), min(b, n)) for a, b, n in zip(i_box_min, i_box_max, grid.shape)]

        grid_cumul_prod = np.array([1, grid.shape[0], grid.shape[0] * grid.shape[1]])
        # X fastest running
        for kji in itertools.product(*reversed(axis_ranges)):
            # make X the first coordinate
            ijk = np.flip(np.array(kji))
            corners = grid.origin[None, :] + (ijk[None, :] + __rel_corner[:, :]) * grid.step[None, :]
            loc_corners = dfn.inv_transform_mat[i] @ (corners - dfn.center[i]).T
            if intersect_cell(loc_corners, dfn.base_shape):
                # logging.log(logging.DEBUG, f"       cell {ijk}")
                cell_index = ijk @ grid_cumul_prod
                i_cell.append(cell_index)
                i_fr.append(i)

    return Intersection.const_isec(domain, i_cell, i_fr, 1.0)


def intersection_interpolation(domain: FracturedDomain) -> 'Intersection':
    """
    Approximate fast intersection for small number of fractures.
    1. fractures are encoded as decreasing 2 powers: 2**(-i_fr), assume fractures sorted from large down
    2. place points on the fractures with their values
    3. project all points to ambient space by transform matrices (use advanced indexing)
    4. summ to the cells
    5. get few largest fractures in each cell
    :param domain:
    :return:
    """


def intersection_band_antialias(domain: FracturedDomain) -> 'Intersection':
    """
    This approach interprets fractures as bands of given cross-section,
    the interpolation is based on a fast approximation of the volume of the band-cell
    intersection. The band-cell intersection candidates are determined by modified decovalex algorithm.
    """
    # logging.log(logging.INFO, f"Calculating Fracture - Cell intersections ...")
    # dfn = domain.dfn
    # grid = domain.grid
    # for fr, aabb, trans_mat in zip(dfn, dfn.AABB, dfn.inv_transform_mat):
    #     min_corner_cell, max_corner_cell = grid.project_points(aabb.reshape(2, 3))
    #     axis_ranges = [range(max(0, a), min(b, n))
    #                    for a, b, n in zip(min_corner_cell, max_corner_cell, grid.shape)]
    #     itertools.product(*reversed(axis_ranges))
    #     grid.
    # return [fracture_for_ellipse(grid, ie, ellipse) for ie, ellipse in enumerate(ellipses)]
    pass


def fr_conductivity(dfn: FractureSet, cross_section_factor=1e-4, perm_factor=1.0):
    """

    :param dfn:
    :param cross_section_factor: scalar = cross_section / fracture mean radius
    :return:
    """
    rho = 1000
    g = 9.81
    viscosity = 8.9e-4
    perm_to_cond = rho * g / viscosity
    cross_section = cross_section_factor * np.sqrt(np.prod(dfn.radius, axis=1))
    perm = perm_factor * cross_section * cross_section / 12
    conductivity = perm_to_cond * perm
    cond_tn = conductivity[:, None, None] * (np.eye(3) - dfn.normal[:, :, None] * dfn.normal[:, None, :])

    return cross_section, cond_tn


# ============ DEPRECATED

@attrs.define
class FracturedMedia:
    """
    Representation of the fractured media sample.
    Geometry:
    dfn + grid or dfn + arbitrary bulk points
    Fields, should rather be separated for different type of quantities.
    scalar (porosity): scalars (fields in future) on fractures, bulk scalar field on grid or at points
    vector (velocity): vectors on fractures, vector bulk field
    2d-tensor (conductivity): tensors on fractures, tensor bulk field
        scalars on fractures -> imply scalar * (n \otimes n) tensor
    4d-cauchy tensor: ?
    4d-dispersion tensor: ? it should describe second order, i.e. variance of the velocity field
    Seems reasonable to assume that all quantities are homogenized as weighted avarages of fracture and bulk values.

    1. DFN imply a box
    2. If we add a grid step we can specify bulk values on that grid
    3. voxelization grid could be independent. Make interpolation in each axis independently.

    Deprecated design. We should separate interpolation matrix from the value arrays.
    TODO: use FracturedMedia instead separate arrays.
    """
    dfn: FractureSet  #
    fr_cross_section: np.ndarray  # shape (n_fractures,)
    fr_conductivity: np.ndarray  # shape (n_fractures,)
    conductivity: float

    @staticmethod
    def fracture_cond_params(dfn: FractureSet, unit_cross_section, bulk_conductivity):
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
        perm.dat - 6 values per line; 4th is permittivity
        aperture.dat - 4 values per line; 4th is aperture
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
        radii = radii[:, 0:2]
        assert radii.shape[1] == 2
        normals = np.array(cls._read_dfn_file(workdir / __normalfile), dtype=float)
        assert normals.shape == (n_frac, 3)
        translations = np.array([t for t in cls._read_dfn_file(workdir / __transfile) if t[-1] != 'R'], dtype=float)
        assert translations.shape == (n_frac, 3)
        # permeability = np.array(cls._read_dfn_file(workdir / __permfile), dtype=float)[:, 3]
        # apperture = np.array(cls._read_dfn_file(workdir / __aperturefile), dtype=float)[:, 3]
        shape_axis = np.repeat(n_frac, np.array([1, 0]), axis=0)
        shape_idx = EllipseShape().id
        dfn = FractureSet(shape_idx, radii, translations, normals, shape_axis)
        return cls(dfn, None, None)


def intersections_centers(grid: Grid, fractures: List[Fracture]):
    """
    Estimate intersections between grid cells and fractures

    1. for all fractures compute what could be computed in vector fashion
    2. for each fracture determine cell centers close enough
    3. compute XY local coords and if in the Shape
    """
    fr_normal = np.array([fr.normal for fr in fractures])
    fr_center = np.array([fr.center for fr in fractures])
    import decovalex_dfnmap as dmap

    ellipses = [dmap.Ellipse(fr.normal, fr.center, fr.scale) for fr in fractures]
    d_grid = dmap.Grid.make_grid(grid.origin, grid.step, grid.dimensions)
    d_fractures = dmap.map_dfn(d_grid, ellipses)
    i_fr_cell = np.stack([(i_fr, i_cell) for i_fr, fr in enumerate(d_fractures) for i_cell in fr.cells])
    # fr, cell = zip([(i_fr, i_cell)  for i_fr, fr in enumerate(fractures) for i_cell in fr.cells])
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
    i_fr_cell = np.stack([(i_fr, i_cell) for i_fr, fr in enumerate(d_fractures) for i_cell in fr.cells])
    # fr, cell = zip([(i_fr, i_cell)  for i_fr, fr in enumerate(fractures) for i_cell in fr.cells])
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

    return np.array([full_tensor(fr.normal, fr_cond) for fr, fr_cond in zip(fractures, fr_transmisivity)])


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
    isec = intersections_decovalex(grid, fr_media.dfn)
    fr_transmissivity = fr_media.fr_conductivity * fr_media.fr_cross_section
    fr_values = fr_values_fn(isec.fractures, fr_transmissivity, isec.grid.step)
    # accumulate tensors in cells
    ncells = isec.grid.n_elements
    k_aniso = np.full((ncells, *fr_values.shape[1:]), fr_media.conductivity, dtype=np.float64)
    np.add.at(k_aniso, isec.i_fr_cell[:, 1], fr_values[isec.i_fr_cell[:, 0]])
    return k_aniso  # arange_for_hdf5(grid, k_iso).flatten()


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
    and sum of volumes of the intersecting fracutres.

    The cached properties for the bulk weight vector and fracture interpolation sparse matrix for efficient multiplication
    are provided.
    DEPRECATED DESIGN.
    - The interpolation should be provided by the sparse interpolation matrix
    - Input of the interpolation should be a connected vector of both bulk and fracture values
    """
    grid: 'Grid'  # Any grid composed of numbered cells.
    cell_ids: List[int]  # For each intersection the cell id.
    fr_ids: List[int]  # For each intersection the fracture id.
    volume: List[float]  # For each intersection the intersection fracture volume estimate.

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
        # flag_upper_Y = polygons[:, :, 1] > (aabb_min_sort[:, 1] + aabb_min_sort[:, 1]) / 2

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
    rel_area = aperture * area / np.dot(normal, cell)
    tn = rel_area * normal[:, None] * normal[None, :]
    return tn


# =================
# Main interface functions
# Usage example:
# bulk_grid ... grid of the input bulk values
# homo_grid ... output grid of homogenization
# bulk_on_homo_field = bulk_interpolate(bulk_geometry, bulk_field, homo_grid)
# # bulk_geometry is either grid or array of 3d points where bulk_field is given
# A, B = voxelize_xyz(dfn, homo_grid)
# homo_field = A * bulk_on_homo_field + B @ fracture_field
# # or
# homo_field = A * bulk_on_homo_field + B @ normal_field(dfn, scalar_fracture_field)
# or a function
# homogenize(voxel_obj(dfn, A, B, homo_grid), bulk_on_homo_field, scalar_fracture_field)
# =================

@attrs.define
class Homogenize:
    """
    Class representing intersection of the fractures with a regular grid,
    capable to perform average homogenization of individual fields.
    dfn: FractureSet, ):

    Several methods supported:
    - source mixed mesh -> regular gird, using projection of Gauss points to the regular grid,
      adaptive refinement relative to the target mesh
    - decovalex voxelization  for bulk on a grid pre´fractures any
    - eter mesh

    """
    domain: FracturedDomain
    bulk_scaling: np.ndarray  # (N_homo_grid_cells, field_shape)
    fracture_interpolation: Any  # sparse matrix (N_homo_grid_cells, n_fractures, field_shape)

    @staticmethod
    def mesh(mesh_path, grid: Grid):
        grid_cell_volume = np.prod(grid.step) / 27

        ref_el_2d = np.array([(0, 0), (1, 0), (0, 1)])
        ref_el_3d = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])

        pvd_content = pv.get_reader(flow_out.hydro.spatial_file.path)
        pvd_content.set_active_time_point(0)
        dataset = pvd_content.read()[0]  # Take first block of the Multiblock dataset

        velocities = dataset.cell_data['velocity_p0']
        cross_section = dataset.cell_data['cross_section']

        p_dataset = dataset.cell_data_to_point_data()
        p_dataset.point_data['velocity_magnitude'] = np.linalg.norm(p_dataset.point_data['velocity_p0'], axis=1)
        plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1))
        cut_dataset = p_dataset.clip_surface(plane)

        plotter = pv.Plotter()
        plotter.add_mesh(p_dataset, color='white', opacity=0.3, label='Original Dataset')
        plotter.add_mesh(cut_dataset, scalars='velocity_magnitude', cmap='viridis', label='Velocity Magnitude')

        # Add legend and show the plot
        plotter.add_scalar_bar(title='Velocity Magnitude')
        plotter.add_legend()
        plotter.show()

        # num_cells = dataset.n_cells
        # shifts = np.zeros((num_cells, 3))
        # transform_matrices = np.zeros((num_cells, 3, 3))
        # volumes = np.zeros(num_cells)

        weights_sum = np.zeros((grid.n_elements,))
        grid_velocities = np.zeros((grid.n_elements, 3))
        levels = np.zeros(dataset.n_cells, dtype=np.int32)
        # Loop through each cell
        for i in range(dataset.n_cells):
            cell = dataset.extract_cells(i)
            points = cell.points

            if len(points) < 3:
                continue  # Skip cells with less than 3 vertices

            # Shift: the first vertex of the cell
            shift = points[0]
            # shifts[i] = shift

            transform_matrix = points[1:] - shift
            if len(points) == 4:  # Tetrahedron
                # For a tetrahedron, we use all three vectors formed from the first vertex
                # transform_matrices[i] = transform_matrix[:3].T
                # Volume calculation for a tetrahedron:
                volume = np.abs(np.linalg.det(transform_matrix[:3])) / 6
                ref_el = ref_el_3d
            elif len(points) == 3:  # Triangle
                # For a triangle, we use only two vectors
                # transform_matrices[i, :2] = transform_matrix.T
                # Area calculation for a triangle:
                volume = 0.5 * np.linalg.norm(np.cross(transform_matrix[0], transform_matrix[1])) * cross_section[i]
                ref_el = ref_el_2d
            level = max(int(np.log2(volume / grid_cell_volume) / 3.0), 0)
            levels[i] = level
            ref_barycenters = refine_barycenters(ref_el[None, :, :], level)
            barycenters = shift[None, :] + ref_barycenters @ transform_matrix
            grid_indices = grid.project_points(barycenters)
            weights_sum[grid_indices] += volume
            #
            # grid_velocities[grid_indices] += volume * velocities[i]

            values.extend(len(grid_indices) * [volume])
            rows.extend(grid_indices)
            cols.extend(len(grid_indices) * [i])
        # print(np.bincount(levels))
        # grid_velocities = grid_velocities / weights_sum[:, None]

        values[:] /= weights_sum[rows[:]]

        sp.csr_matrix((vals, (rows, cols)), shape=(grid.n_elements, dataset.n_cells))
        return grid_velocities

    # @staticmethod
    # def (dfn: FractureSet, fr_cross_section: np.ndarray, grid: Grid):
    #     """
    #     Create the grid - fracture set intersection object using particular voxelization algorithm.
    #     :return:
    #     """
    #     return

    def __call__(self, bulk_field, fracture_field):
        assert bulk_field.shape[1:] == fracture_field.shape[1:]
        field_shape = bulk_field.shape[1:]
        assert bulk_field.shape[0] == self.domain.grid.n_elements
        n_bulk = bulk_field.shape[0]
        assert fracture_field.shape[0] == len(self.domain.dfn)
        n_frac = fracture_field.shape[0]
        bulk_f = bulk_field.reshape(n_bulk, -1)
        frac_f = fracture_field.reshape(n_frac, -1)
        result_f = self.bulk_scaling[:, None] * bulk_f + self.fracture_interpolation @ frac_f
        return result_f.reshape(n_bulk, *field_shape)

    def interpolate_grid(self, bulk_grid, bulk_field, fracture_field):
        """
        TODO: Interpolate bulk_filed given at bulk_grid to the domain.grid,
        then return result of call: `self(interpolated_field, fracture_field)`

        :param bulk_points: (n_points, 3)
        :param bulk_field: (n_points, field_shape)
        :param fracture_field:
        :return:
        """
        pass

    def interpolate_points(self, bulk_points, bulk_field, fracture_field):
        """
        TODO: Interpolate bulk_filed given at bulk_points to the domain.grid,
        then return result of call: `self(interpolated_field, fracture_field)`

        :param bulk_points: (n_points, 3)
        :param bulk_field: (n_points, field_shape)
        :param fracture_field:
        :return:
        """
        pass


def voxelize(dfn, bulk_grid):
    """
    Compute a sparse matrices for average homogenization:
    homo_field = bulk_values_on_grid + A @ fracture_values
    :param dfn:
    :param bulk_grid:
    :return:
    """
