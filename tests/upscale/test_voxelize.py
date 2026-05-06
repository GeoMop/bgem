"""
Test of homogenization algorithms from voxelize.py
- Homogenization of bulk constant conductivity + discreate fractures with size dependent conductivity.
  Reference is decovalex slow solution modified for anisotropic regular grid.
  This assigns The same conductivity to all intersection cells.

  In order to develop more precises homogenization techniques, we must use two-scale test problems.
"""
import pytest
import fixtures

from typing import *
import yaml
import shutil
from pathlib import Path
from scipy import integrate

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


import numpy as np
import attrs
import pyvista as pv

from bgem import stochastic
from bgem import fn
from bgem.gmsh import gmsh, options
from mesh_class import Mesh
from bgem.core import call_flow, dotdict, workdir as workdir_mng
from bgem.upscale import *
from bgem.upscale import fem_plot
from bgem.upscale.voxelize import base_shape_interior_grid

script_dir = Path(__file__).absolute().parent
workdir = script_dir / "sandbox"
from joblib import Memory
memory = Memory(workdir, verbose=0)


def bulk_sphere_field(grid: Grid, out_sphere_value, in_sphere_value):
    r = np.min(grid.dimensions) * 0.8 / 2
    center = grid.grid_center()
    field = np.full(grid.shape, out_sphere_value).flatten()
    in_sphere = np.linalg.norm(grid.barycenters() - center, axis=1) < r
    field[in_sphere] = in_sphere_value
    return field

def bulk_perm_field(grid: Grid):
    """
    This bulk field has high permeability out of a sphere that should transfer
    the outer BC to the inner sphere of lower conductivity. There fore the
    equivalent tensor should be rotation invariant, i.e. if we rotate fractures in the sphere
    the equivalent tensor T of the whole domain should match the Q.T @T' @Q, if T' is
    equivalent tensor of fractures rotated by matrix Q.T.
    This allows to test correct rasterization with respect to different rotations.
    :param grid:
    :return:
    """
    return bulk_sphere_field(grid, 1, 1e-10)[:, None, None] * np.eye(3)


def drop_tuple_item(x, i) :
    return (*x[:i], *x[i+1:])

def insert_tuple_item(x, i, item):
    return (*x[:i], item, *x[i:])


def product_concatenate(a, b, axis=0):
    assert len(a.shape) == len(b.shape)
    a_shape = drop_tuple_item(a.shape, axis)
    b_shape = drop_tuple_item(b.shape, axis)
    common_shape = np.broadcast_shapes(a_shape, b_shape)
    a_new_shape = insert_tuple_item(common_shape, axis, a.shape[axis])
    b_new_shape = insert_tuple_item(common_shape, axis, b.shape[axis])
    result = np.concatenate((
        np.broadcast_to(a, a_new_shape),
        np.broadcast_to(b, b_new_shape)
    ), axis=axis)
    return result


def probe_fr_intersection(fr_set: stochastic.FractureSet, grid: Grid):
    domain = FracturedDomain(fr_set, np.ones(len(fr_set)), grid)
    i_cell = []
    i_fracture = []
    min_grid_step = min(grid.step)

    for i in range(len(fr_set)):
        radius = np.max(fr_set.radius[i])
        step = 0.5 * min_grid_step / radius
        ref_points_xy = base_shape_interior_grid(fr_set.base_shape, step)
        #z_coord = np.array([-min_grid_step/2.1, 0, min_grid_step/2.1])
        n_z = np.abs(fr_set.normal[i, 2])
        n_xy = np.linalg.norm(fr_set.normal[i, :2])
        n_max = max(n_z, n_xy)
        n_min = min(n_z, n_xy)

        z1 = min_grid_step * n_min / n_max
        z_coord = np.array([-z1/8, 0, z1/8])
        ref_points_xyz = product_concatenate(ref_points_xy[:, None, :], z_coord[None, :, None], axis=2).reshape(-1, 3)

        actual_points = (fr_set.transform_mat[i] @ ref_points_xyz[:, :, None])[:, :, 0] + fr_set.center[i]
        cell_indices = np.unique(grid.project_points(actual_points))
        i_cell.extend(cell_indices.tolist())
        i_fracture.extend(len(cell_indices)*[i])
    return Intersection.const_isec(domain, i_cell, i_fr, 1.0)

def plot_isec_fields(intersections: List[Intersection], names: List[str], outpath: Path):
    """
    Assume common grid
    :param intersections:
    :return:
    """
    grid = intersections[0].grid
    cell_fields = {n: isec.cell_field() for n, isec in zip(names, intersections)}

    pv_grid = fem_plot.grid_fields_vtk(grid, cell_fields, vtk_path=outpath)

    #plotter = fem_plot.create_plotter()  # off_screen=True, window_size=(1024, 768))
    #plotter.add_mesh(pv_grid, scalars='cell_field')
    #plotter.show()

def plot_isec_fields2(isec: Intersection, in_field, out_field, outpath: Path):
    """
    Assume common grid
    :param intersections:
    :return:
    """
    grid = isec.grid
    cell_fields = {
        'cell_field': isec.cell_field(),
        'in_field' : in_field,
        'out_field' : out_field}

    pv_grid = fem_plot.grid_fields_vtk(grid, cell_fields, vtk_path=outpath)

    #plotter = fem_plot.create_plotter()  # off_screen=True, window_size=(1024, 768))
    #plotter.add_mesh(pv_grid, scalars='cell_field')
    #plotter.show()



def compare_intersections(isec, isec_ref, fname):
    count_ref = isec_ref.count_fr_cells()
    count_isec = isec.count_fr_cells()
    count_diff = count_isec - count_ref
    rel_error = np.abs(count_diff) / np.maximum(count_isec, count_ref)
    if np.max(rel_error) > 0.1:
        print("Large error fractures:\n")
        for i in range(len(count_ref)):
            print(f"fr #{i}, ref: {count_ref[i]}, isec: {count_isec[i]}, err: {rel_error[i]}")
        plot_isec_fields([isec, isec_ref], ['isec', 'isec_ref'], workdir / (fname + '.vtk'))
        assert False

def isec_decovalex_case(fr_list: List[stochastic.Fracture], grid: Grid):
    """
    Test detected cell-fracture intersections produced by the DFN Works algorithm.
    We just generate a fine grid in XY reference plane of the fracture and map
    them to the grid coordinates to find how much
    :param fr_set:
    :return:
    """
    fr_set = stochastic.FractureSet.from_list(fr_list)
    isec = intersection_decovalex(fr_set, grid)
    isec_corners = intersection_cell_corners(fr_set, grid)
    isec_probe = probe_fr_intersection(fr_set, grid)
    compare_intersections(isec, isec_probe, "compare_decovalex")
    compare_intersections(isec, isec_corners, "compare_corners")

@pytest.mark.skip
def test_intersection_decovalex():
    """
    Test correct set of intersection cells for each fracture.
    :return:
    """
    steps = 3* [41]
    grid = Grid(3*[100], steps, origin=3*[-50]) # test grid with center in (0,0,0)
    shape = stochastic.EllipseShape

    fr = lambda r, c, n : stochastic.Fracture(shape.id, r, c, n/np.linalg.norm(n))
    fr_list = [fr(45, [0, 0.7, 0], [0, 0, 1]),]
    isec_decovalex_case(fr_list, grid)

    fr_list = [fr(50, [0.7, 0, 0], [0, 1, 0]),]
    isec_decovalex_case(fr_list, grid)

    fr_list = [fr(50, [0, 0, 0.7], [1, 0, 0]),]
    isec_decovalex_case(fr_list, grid)

    fr_list = [fr(50, [0, 0, 0], [0, 1, 1]),]
    isec_decovalex_case(fr_list, grid)

    fr_list = [fr(50, [0, 0, -0.7], [0, 1, 3]),]
    isec_decovalex_case(fr_list, grid)

    fr_list = [fr(60, [0, 5, -5], [0, 1, 3]),
               fr(40, [10, 0, -10], [1, 0, 0]),
               fr(40, [10, 0, 0], [3, 1, 0]),
               fr(60, [-5, 0, 0], [-2, 1, 3]),
               ]
    isec_decovalex_case(fr_list, grid)

def isec_corners_case(fr_list: List[stochastic.Fracture], grid: Grid):
    """
    Test detected cell-fracture intersections produced by the DFN Works algorithm.
    We just generate a fine grid in XY reference plane of the fracture and map
    them to the grid coordinates to find how much
    :param fr_set:
    :return:
    """
    fr_set = stochastic.FractureSet.from_list(fr_list)
    #isec = intersection_decovalex(fr_set, grid)
    isec_corners = intersection_cell_corners(fr_set, grid)
    isec_probe = probe_fr_intersection(fr_set, grid)
    compare_intersections(isec_corners, isec_probe, "compare_corners_rect")

@pytest.mark.skip
def test_intersection_corners_rectangle():
    """
    Test correct set of intersection cells for each fracture.
    :return:
    """
    steps = 3* [41]
    grid = Grid(3*[100], steps, origin=3*[-50]) # test grid with center in (0,0,0)
    shape = stochastic.RectangleShape

    fr = lambda r, c, n, ax=[1,0] : stochastic.Fracture(shape.id, r, c, n/np.linalg.norm(n), ax/np.linalg.norm(ax))
    fr_list = [fr(45, [0, 0.7, 0], [0, 0, 1]),]
    isec_corners_case(fr_list, grid)

    fr_list = [fr(50, [0.7, 0, 0], [0, 1, 0]),]
    isec_corners_case(fr_list, grid)

    fr_list = [fr(50, [0, 0, 0.7], [1, 0, 0]),]
    isec_corners_case(fr_list, grid)

    fr_list = [fr(50, [0, 0, 0], [0, 1, 1]),]
    isec_corners_case(fr_list, grid)

    fr_list = [fr(50, [0, 0, -0.7], [0, 1, 3], ax=[1,1])]
    isec_corners_case(fr_list, grid)

    fr_list = [fr(30, [0, 10, -10], [0, 1, 3], ax=[1,1] ),
               fr(60, [10, 0, -10], [1, 0, 0], ax=[-2,1]),
               fr(30, [10, 0, 0], [3, 1, 0], ax=[-1,-2]),
               fr(30, [-10, 0, 0], [-2, 1, 3], ax=[2,-1]),
               ]
    isec_corners_case(fr_list, grid)

def test_rasterized_field():
    """
    Test whole rasterization process using intersection_cell_corners
    :return:
    """
    source_grid = Grid(3*[100], 3*[41], origin=3*[-50])
    bulk_source_conductivity = bulk_sphere_field(source_grid, 1.0, 1e-11)
    bulk_tn = bulk_source_conductivity[:, None, None] * np.eye(3)[None, :, :]

    steps = 3* [41]
    target_grid = Grid(3*[100], steps, origin=3*[-50]) # test grid with center in (0,0,0)

    shape = stochastic.RectangleShape
    fr = lambda r, c, n, ax=[1,0] : stochastic.Fracture(shape.id, r, c, n/np.linalg.norm(n), ax/np.linalg.norm(ax))
    fr_list = [fr(30, [0, 5, -5], [0, 1, 3], ax=[1,1] ),
               fr(30, [5, 0, -5], [1, 0, 0], ax=[-2,1]),
               fr(30, [5, 0, 0], [3, 1, 0], ax=[-1,-2]),
               fr(30, [-5, 0, 0], [-2, 1, 3], ax=[2,-1]),
               ]
    fr_set = stochastic.FractureSet.from_list(fr_list)
    isec_corners = intersection_cell_corners(fr_set, target_grid)
    #isec_probe = probe_fr_intersection(fr_set, target_grid)
    cross_section, fr_cond = fr_conductivity(fr_set)
    rasterized = isec_corners.interpolate(bulk_tn, fr_cond, source_grid=source_grid)
    plot_isec_fields2(isec_corners, bulk_tn, rasterized, workdir / "raster_field.vtk")
    for i_ax in range(3):
        assert np.all(bulk_tn[:, i_ax, i_ax] <= rasterized[:, i_ax, i_ax])
    for i_ax in range(3):
        assert np.all(rasterized[:, i_ax, i_ax].max() <= fr_conductivity[:, i_ax, i_ax].max())




# def dfn_4_fractures():
#     return voxelize.FracturedMedia.from_dfn_works(script_dir / "4_fractures", 0.01)

# def test_load_dfnworks():
#     dfn = dfn_4_fractures()
#     assert dfn.dfn.size == 4


def tst_fracture_set(R, shape):
    fr = lambda c, n : stochastic.Fracture(shape.id, R, c, n, 0.0, 123, 1)
    return [
        #fr([0, 0, 0.7], [0, 0, 1]),
        #fr([0, 0.7, 0], [0, 1, 0]),
        #fr([0.7, 0, 0], [1, 0, 0]),
        #fr([0, 0, 0], [0.5, 0, 1]),
        fr([0, 0, 0.7], [0, 0.5, 1]),
        #fr([0, 0, 0], [0.1, 1, 1]),
        #fr([0, 0, 0], [0.3, 1, 1]),
        #fr([0, 0, -0.7], [0.5, 1, 1]),
        fr([0, 0, -0.5], [1, 1, 1])
           ]



def homo_decovalex(fr_media: FracturedMedia, grid:Grid, perm_fn):
    """
    Homogenize fr_media to the conductivity tensor field on grid.
    :return: conductivity_field, np.array, shape (n_elements, n_voight)
    """
    ellipses = [dmap.Ellipse(fr.normal, fr.center, fr.scale) for fr in fr_media.dfn]
    d_grid = dmap.Grid.make_grid(grid.origin, grid.step, grid.dimensions)
    fractures = dmap.map_dfn(d_grid, ellipses)
    fr_transmissivity = fr_media.fr_conductivity * fr_media.fr_cross_section
    return perm_fn(d_grid, fractures, fr_transmissivity, fr_media.conductivity)

def homo_decovalex_iso(fr_media: FracturedMedia, grid:Grid):
    perm_fn = lambda *args : dmap.permIso(*args)[:, None, None] * np.eye(3)
    return homo_decovalex(fr_media, grid, perm_fn)

def homo_decovalex_aniso_raw(fr_media: FracturedMedia, grid: Grid):
    perm_fn = lambda *args : dmap.permAnisoRaw(*args)
    return homo_decovalex(fr_media, grid, perm_fn)

def homo_decovalex_aniso_diag(fr_media: FracturedMedia, grid: Grid):
    perm_fn = lambda *args : dmap.aniso_diag(dmap.permAnisoRaw(*args))
    return homo_decovalex(fr_media, grid, perm_fn)

def homo_decovalex_aniso_lump(fr_media: FracturedMedia, grid: Grid):
    perm_fn = lambda *args : dmap.aniso_lump(dmap.permAnisoRaw(*args))
    return homo_decovalex(fr_media, grid, perm_fn)

def rasterize_dfn(fr_set):
    # Fracture set
    domain_size = 100

    # Coarse Problem
    steps = (10, 12, 14)
    grid = Grid(3*[domain_size], steps, origin=-domain_size / 2)

    dfn = tst_fracture_set(grid.dimensions)
    fr_media = FracturedMedia.fracture_cond_params(dfn, 0.1, 1)

    xyz_range = [ np.linspace(grid.origin[ax], grid.origin[ax] + grid.dimensions[ax], grid.shape[ax] + 1, dtype=np.float32)
                  for ax in [0, 1, 2]
                ]

    x, y, z = np.meshgrid(*xyz_range, indexing='ij')
    pv_grid = pv.StructuredGrid(x, y, z)
    #points = grid.nodes()
    for name, homo_fn in homo_fns.items():

        grid_permitivity = homo_fn(fr_media, grid)
        if len(grid_permitivity.shape) > 1:
            # anisotropic case
            assert grid_permitivity.shape[1:] == (3, 3)
            grid_permitivity = grid_permitivity.reshape(-1, 9)
        pv_grid.cell_data[name] = grid_permitivity
    pv_grid.save(str(workdir / "test_resterize.vtk"))



def test_reasterize():
    homo_fns=dict(
        k_deco_iso=voxelize.permeability_iso_decovalex,
        k_deco_aniso_raw=voxelize.permeability_aniso_decovalex,
        k_deco_aniso_diag=fn.compose(voxelize.aniso_diag, voxelize.permeability_aniso_decovalex),
        k_deco_aniso_lump=fn.compose(voxelize.aniso_lump, voxelize.permeability_aniso_decovalex)
    )
    rasterize_dfn(homo_fns)


# def fracture_band_field(grid, fr_set:stochastic.FractureSet):
#     """
#     1. define distance field on a grid in the fracture coordinates, grid has step:
#         1/k *  [norm(grid.step)/rx, norm(grid.step)/ry, band_width]
#         band_width = grid.step @ np.abs(normal)
#         Use base_shape slightly enlarged aabb as a source domain extent.
#         Distance field is linear only in Z axis, but has jump on the shape border.
#     2. evaluate in cell centers transformed  into fracture system use:
#         scipy.interpolate.interpn
#         values on fracture = cross_section /
#     :param grid:
#     :param fr_set:
#     :return:
#     """
#     N = 1e4
#     aabb = fr_set.base_shape.aabb
#     shape_grid = Grid.from_aabb(aabb, n_steps)oints = np.random.random((N, 2)) * (aabb[1] - aabb[0]) + aabb[0]
#     inside_vector = fr_set.base_shape.are_points_inside(points)
#     points = points[inside_vector]
#
#     i_cells = grid.project_points(fr_set.transform_mat[0] @ points.T + fr_set.center[0])
#     field = np.bincount(i_cells, minlength=grid.n_elements) / len(points)
#     return field
#
# def compare_voxelization(grid, fractures:stochastic.FractureSet):
#     """
#     For given grid compare all available voxelization functions
#     with respect to the sampling the fracture band by points
#     :return:
#     """
#     cross_section = 1e-4 * fractures.radius_norm
#     domain = FracturedDomain(fractures, cross_section, grid)
#     isec_band = Intersection.band_fractures(domain)
#     for fr in fractures:
#
#
# def test_voxelize_single_fracture():
#     size = 100
#     domain = Grid(size, 16)
#     fractures = fixtures.get_dfn_sample(size, 123)
#     for fr in fractures:
#         fr_set = stochastic.FractureSet.from_list([fr])
#         compare_voxelization(grid, fr_set)