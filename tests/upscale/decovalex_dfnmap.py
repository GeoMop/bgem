import csv
import itertools
import logging
import time
from pathlib import Path
from typing import *

import attrs
import numpy as np

#from . import decovalec_transform as tr


def float_array(x: List[float]) -> np.array:
    return np.array(x, dtype=float)


def int_array(x: List[float]) -> np.array:
    return np.array(x, dtype=int)


@attrs.define
class StructuredGrid:
    """
    General tensor product  grid.
    Steps can vary in ecach dimension.
    """
    origin: np.array  # 3d point
    steps: np.array  # list of step arrays in each axis, array [steps_x, steps_y, steps_z] [m]

    @property
    def cell_dimensions(self):
        # [nx, ny, nz], int, number of cells in XYZ
        return np.array([len(axis_steps) for axis_steps in self.steps])


@attrs.define
class Grid:
    @classmethod
    def make_grid(cls, origin, step, dimensions):
        s = np.array(step)
        d = np.array(dimensions)
        cell_dim = np.round(d / s).astype(int)
        dim = s * cell_dim
        return cls(np.array(origin), s, d, cell_dim)

    origin: np.array  # 3d point
    step: np.array  # [step_x, step_y, step_z] [m]
    dimensions: np.array  # [lx, ly, lz] [m]
    cell_dimensions: np.array  # [nx, ny, nz], int, number of cells in XYZ

    def cell_coord(self, x: np.array) -> np.array:
        """
        Cell multiindex containing point 'x'.
        or None if does not exist.
        """
        idx = (x - self.origin) / self.step
        return idx.astype(int)

    def trim(self, x):
        """
        Project point ot of the grid to its surface.
        """
        x = np.maximum(x, self.origin + self.step / 2)
        outer_corner = self.origin + self.dimensions - self.step / 2
        x = np.minimum(x, outer_corner)
        return x

    def make_xyz_field(self):
        return [o + s * np.arange(0, n + 1) for o, n, s in zip(self.origin, self.cell_dimensions, self.step)]


@attrs.define
class Ellipse:
    normal: np.array = attrs.field(converter=float_array)
    # 3D normal vector
    translation: np.array = attrs.field(converter=float_array)
    # 3D translation vector
    radius: np.array = attrs.field(converter=float_array)
    # (x_radius, y_ radius) before transformation


@attrs.define
class Fracture:
    ellipse: Ellipse
    cells: List[int]


__radiifile = 'radii.dat'
__normalfile = 'normal_vectors.dat'
__transfile = 'translations.dat'
__permfile = 'perm.dat'
__aperturefile = 'aperture.dat'


def read_dfn_file(f_path):
    with open(f_path, 'r') as file:
        rdr = csv.reader(filter(lambda row: row[0] != '#', file), delimiter=' ', skipinitialspace=True)
        return [row for row in rdr]



def readEllipse(workdir: Path, ):
    '''Read dfnWorks-Version2.0 output files describing radius, orientation, and
       location of fractures in space.
       Subsequent methods assume elliptical (in fact circular) fractures.
       Return a list of dictionaries describing each ellipse.
    '''

    radii = np.array(read_dfn_file(workdir / __radiifile), dtype=float)
    shape_family = radii[:, 2]
    radii = radii[:, 0:2]
    assert radii.shape[1] == 2
    # with open(radiifile,'r') as f:
    #   radii = f.readlines() #two lines to get rid of at top of file
    # radii.pop(0)
    # radii.pop(0)
    normals = np.array(read_dfn_file(workdir / __normalfile), dtype=float)
    assert normals.shape[1] == 3
    # with open(normalfile, 'r') as f:
    #   normals = f.readlines() #no lines to get rid of at top of file
    translations = np.array([t for t in read_dfn_file(workdir / __transfile) if t[-1] != 'R'], dtype=float)
    assert translations.shape[1] == 3
    # with open(transfile, 'r') as f:
    #   temp = f.readlines() #one line to get rid of at top of file
    # temp.pop(0)
    # translations = []
    # for line in temp:
    #   if line.split()[-1] != 'R':
    #     translations.append(line)
    ellipses = [Ellipse(np.array(n), np.array(t), np.array(r)) for n, t, r in zip(normals, translations, radii)]
    return ellipses


def fr_transmissivity_apperture(workdir):
    '''Read dfnWorks-Version2.0 output files describing fracture aperture and
     permeability.
     Return list containing transmissivity for each fracture.
    '''

    # from both files use just third column
    permeability = np.array(read_dfn_file(workdir / __permfile), dtype=float)[:, 3]
    apperture = np.array(read_dfn_file(workdir / __aperturefile), dtype=float)[:, 3]
    return permeability * apperture, apperture


__rel_corner = np.array([[0, 0, 0], [1, 0, 0],
                         [1, 1, 0], [0, 1, 0],
                         [0, 0, 1], [1, 0, 1],
                         [1, 1, 1], [0, 1, 1]])


def intersect_cell(loc_corners: np.array, ellipse: Ellipse) -> bool:
    """
    loc_corners - shape (3, 8)
    """
    # check if cell center is inside radius of fracture
    center = np.mean(loc_corners, axis=1)
    if np.sum(np.square(center[0:2] / ellipse.radius)) >= 1:
        return False

    # cell center is in ellipse
    # find z of cell corners in xyz of fracture

    if np.min(loc_corners[2, :]) >= 0. or np.max(loc_corners[2, :]) <= 0.:  # fracture lies in z=0 plane
        # fracture intersects that cell
        return False

    return True


def fracture_for_ellipse(grid: Grid, i_ellipse: int, ellipse: Ellipse) -> Fracture:
    # calculate rotation matrix for use later in rotating coordinates of nearby cells
    normal = ellipse.normal /np.linalg.norm(ellipse.normal)
    ellipse.normal = normal
    direction = np.cross([0, 0, 1], normal)
    # cosa = np.dot([0,0,1],normal)/(np.linalg.norm([0,0,1])*np.linalg.norm(normal)) #frobenius norm = length
    # above evaluates to normal[2], so:
    angle = np.arccos(normal[2])  # everything is in radians

    from bgem.transform import Transform
    # rotation is from ez to normal, local system to ambient system
    local_to_ambient = Transform().rotate(direction, angle).matrix[:3, :3]
    mat_to_local = local_to_ambient.T
    #mat_to_local = tr.rotation_matrix(angle, direction)[:3, :3].T
    # find fracture in domain coordinates so can look for nearby cells
    b_box_min = ellipse.translation - np.max(ellipse.radius)
    b_box_max = ellipse.translation + np.max(ellipse.radius)

    i_box_min = grid.cell_coord(b_box_min)
    i_box_max = grid.cell_coord(b_box_max) + 1
    axis_ranges = [range(max(0, a), min(b, n)) for a, b, n in zip(i_box_min, i_box_max, grid.cell_dimensions)]

    grid_cumul_prod = np.array([1, grid.cell_dimensions[0], grid.cell_dimensions[0] * grid.cell_dimensions[1]])
    cells = []
    # X fastest running
    for kji in itertools.product(*reversed(axis_ranges)):
        # make X the first coordinate
        ijk = np.flip(np.array(kji))
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


def arange_for_hdf5(grid: Grid, a: np.array) -> np.array:
    """
    grid.cell_dimension: (nx, ny, nz)
    a - array for cells indexed by: `intersect_cell`  with X coordinate the fastest running (i.e. the last one)
    return - array with Z coordinate be the fastest running
    """
    return a.reshape(grid.cell_dimensions[::-1]).transpose([2, 1, 0])


def porosity_mean(grid: Grid, fractures: List[Fracture], fr_apperture: np.array, bulk_por: float) -> np.array:
    '''Calculate fracture porosity for each cell of ECPM intersected by
       one or more fractures. Simplifying assumptions:
       1. each fracture crosses the cell parallel to cell faces,
       2. each fracture completely crosses the cell.
       3. fracture volume has porosity 1, matrix has porosity `bulk_por`

       fr_apperture: array of appertures of the `fractures`
    '''
    porosity = np.zeros(grid.cell_dimensions.prod(), dtype=float)
    for fr, a in zip(fractures, fr_apperture):
        if fr.cells:
            normal_axis = np.argmax(np.abs(fr.ellipse.normal))
            porosity[fr.cells] += a / grid.step[normal_axis]
    porosity = bulk_por + porosity * (1 - bulk_por)
    return arange_for_hdf5(grid, porosity)


def porosity_min(grid: Grid, fractures: List[Fracture], fr_apperture: np.array, bulk_por: float) -> np.array:
    '''Calculate fracture porosity to match the fast transport by the pore velocity field.
       one or more fractures. Simplifying assumptions:
       1. each fracture crosses the cell parallel to cell faces,
       2. each fracture completely crosses the cell.
       3. cells containing fractures have porosity given by the fractures only (not realistic, but lead to correct break-through times for advection only simulations.)
          cells without fractures have matrix porosity `bulk_por`

       fr_apperture: array of appertures of the `fractures`
    '''
    porosity = np.zeros(grid.cell_dimensions.prod(), dtype=float)
    for fr, a in zip(fractures, fr_apperture):
        if fr.cells:
            normal_axis = np.argmax(np.abs(fr.ellipse.normal))
            porosity[fr.cells] += a / grid.step[normal_axis]

    porosity[porosity == 0] = bulk_por
    return arange_for_hdf5(grid, porosity)


def permIso(grid: Grid, fractures: List[Fracture], fr_transmisivity: np.array, k_background: float) -> np.array:
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
    k_iso = np.full(grid.cell_dimensions.prod(), k_background, dtype=float)
    for fr, a in zip(fractures, fr_transmisivity):
        normal_axis = np.argmax(np.abs(fr.ellipse.normal))
        k_iso[fr.cells] += a / grid.step[normal_axis]
    return k_iso #arange_for_hdf5(grid, k_iso).flatten()



def permAnisoRaw(grid: Grid, fractures: List[Fracture], fr_transmisivity: np.array, k_background: float):
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
    # quick error check
    #nfrac = len(ellipses)
    #if nfrac != len(T):
    #    print('ellipses and transmissivity contain different numbers of fractures')
    #    return
    full_tensor = lambda n, fr_cond : fr_cond * (np.eye(3) - n[:, None] * n[None, :])
    fr_tensor = np.array([full_tensor(fr.ellipse.normal, fr_cond)  for fr, fr_cond in zip(fractures, fr_transmisivity)])
    #ellipseT = np.zeros((nfrac, 3), '=f8')
    #fullTensor = []
    #T_local = np.zeros((3, 3), dtype=np.float)
    #t0 = time.time()
    # calculate transmissivity tensor in domain coordinates for each ellipse
    #for f in range(nfrac):
    #    normal = ellipses[f]['normal']
    #    direction = np.cross(normal, [0, 0, 1])
    #    angle = np.arccos(normal[2])
    #    M = tr.rotation_matrix(angle, direction)
    #    Transpose = np.transpose(M[:3, :3])
    #    T_local[0, 0] = T[f]
    #    T_local[1, 1] = T[f]
        # permeability = 0 in local z direction of fracture
    #    T_domain = np.dot(np.dot(M[:3, :3], T_local), Transpose)

    #    ellipseT[f][0:3] = [T_domain[0, 0], T_domain[1, 1], T_domain[2, 2]]
    #    fullTensor.append(T_domain)


    #t1 = time.time()
    #print('time spent calculating fracture transmissivity %f' % (t1 - t0))

    # in case you were wondering what those off-diagonal terms are:
    #t0 = time.time()
    #fout = open('Ttensor.txt', 'w')
    #for f in range(len(fullTensor)):
    #    fout.write(str(fullTensor[f]))
    #    fout.write('\n\n')
    #fout.close()
    #t1 = time.time()
    #print('time spent writing fracture transmissivity %f' % (t1 - t0))

    # calculate cell effective permeability by adding fracture k to background k
    #t0 = time.time()
    ncells = grid.cell_dimensions.prod()
    k_aniso = np.full((ncells, 3, 3), k_background, dtype=np.float64)
    for fr, tn in zip(fractures, fr_tensor):
        normal_axis = np.argmax(np.abs(fr.ellipse.normal))
        k_aniso[fr.cells] += tn / grid.step[normal_axis]
    return k_aniso #arange_for_hdf5(grid, k_iso).flatten()

    #
    # for i in range(ncells):
    #     if fracture[i][0] != 0:
    #         for j in range(1, fracture[i][0] + 1):
    #             fracnum = fracture[i][j]
    #             if LUMP:  # lump off diagonal terms
    #                 # because symmetrical doesn't matter if direction of summing is correct, phew!
    #                 k_aniso[i][0] += np.sum(fullTensor[fracnum - 1][0, :3]) / d
    #                 k_aniso[i][1] += np.sum(fullTensor[fracnum - 1][1, :3]) / d
    #                 k_aniso[i][2] += np.sum(fullTensor[fracnum - 1][2, :3]) / d
    #             else:  # discard off diagonal terms (default)
    #                 k_aniso[i][0] += ellipseT[fracnum - 1][0] / d  # ellipseT is 0 indexed, fracture numbers are 1 indexed
    #                 k_aniso[i][1] += ellipseT[fracnum - 1][1] / d
    #                 k_aniso[i][2] += ellipseT[fracnum - 1][2] / d
    #
    # #t1 = time.time()
    # #print('time spent summing cell permeabilities %f' % (t1 - t0))
    #
    # return k_aniso



def count_stuff(filename='mapELLIPSES.txt'):
    '''Mapping DFN to ECPM can result in false connections when non-intersecting
       fractures happen to intersect the same cell of the ECPM.
       Make sweeping assumption that cells with 3 or more fractures in them
       are more likely than cells with 2 fractures in them to contain a false
       connection and count them as such.
       Return counts of 1) total number of cells, 2) number of (active) cells
       containing fractures, 3) number of cells containing 3 or more fractures.

       (This method could more efficiently use the fracture array returned by
        map_dfn().)
    '''
    fin = file(filename, 'r')
    cellcount = 0
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    morethan4 = 0
    for line in fin:
        if line.startswith('#'):
            continue
        elif int(line.split()[3]) == 0:
            cellcount += 1
            count0 += 1
        elif int(line.split()[3]) == 1:
            cellcount += 1
            count1 += 1
        elif int(line.split()[3]) == 2:
            cellcount += 1
            count2 += 1
        elif int(line.split()[3]) == 3:
            cellcount += 1
            count3 += 1
        elif int(line.split()[3]) >= 4:
            cellcount += 1
            morethan4 += 1
    print('\n')
    print('Information for %s ' % filename)
    print('Total number of cells in grid %i' % cellcount)
    print('Number of cells containing fractures %i' % (cellcount - count0))
    print('Percent active cells %.1f' % (100. * (float(cellcount) - float(count0)) / float(cellcount)))
    print('Number of cells containing 1 fracture %i' % (count1))
    print('Number of cells containing 2 fractures %i' % (count2))
    print('Number of cells containing 3 fractures %i' % (count3))
    print('Number of cells containing 4 or more fractures %i' % (morethan4))
    print('Possible false connections %i (cells containing >= 3 fractures)' % (count3 + morethan4))
    fin.close()
    count = {'cells': cellcount, 'active': (cellcount - count0), 'false': (count3 + morethan4)}

    return count

