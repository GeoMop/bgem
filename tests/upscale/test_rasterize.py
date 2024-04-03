"""
Test of homogenization techniquest within a two-scale problem.
- reference solution is evaluated by Flow123d, using direct rasterization of full DFN sample
  The pressure field is projected to the nodes of the rectangular grid,
  velocity filed is averaged over rectangular cells.

- two-scale solution involves:
  1. homogenization of DFN to the rectangular grid ; general permeability tensor field
  2. custom 3d solver for the rectangular grid is used to solve the coarse problem

- various homogenization techniques could be used, homogenization time is evaluated and compared.
"""
from typing import *
import yaml
import shutil
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


import numpy as np
import attrs
import pyvista as pv

from bgem import stochastic
from bgem.gmsh import gmsh, options
from mesh_class import Mesh
from bgem.core import call_flow, dotdict, workdir as workdir_mng
from bgem.upscale import Grid, Fe, voigt_to_tn, tn_to_voigt
import decovalex_dfnmap as dmap

script_dir = Path(__file__).absolute().parent
workdir = script_dir / "sandbox"
from joblib import Memory
memory = Memory(workdir, verbose=0)


@attrs.define
class FracturedMedia:
    dfn: List[stochastic.Fracture]
    fr_cross_section: np.ndarray    # shape (n_fractures,)
    fr_conductivity: np.ndarray     # shape (n_fractures,)
    conductivity: float

    @staticmethod
    def constant_fr(dfn, fr_conductivity, fr_cross_section, bulk_conductivity):
        fr_cond = np.full([len(dfn)], fr_conductivity)
        fr_cross = np.full([len(dfn)], fr_cross_section)
        return FracturedMedia(dfn, fr_cross, fr_cond, bulk_conductivity)

    @staticmethod
    def cubic_law_fr(dfn, unit_cross_section, bulk_conductivity):
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

def tst_fracture_set(domain):
    R = 2*np.max(domain)
    fr = lambda c, n : stochastic.Fracture(stochastic.SquareShape, R, c, n, 0.0, 123, 1)
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


def rasterize_decovalex(dfn, grid:Grid):
    ellipses = [dmap.Ellipse(fr.normal, fr.center, fr.scale) for fr in dfn]
    d_grid = dmap.Grid.make_grid(grid.origin, grid.step, grid.dimensions)
    return dmap.map_dfn(d_grid, ellipses)

def homo_decovalex_(fr_media: FracturedMedia, grid:Grid):
    """
    Homogenize fr_media to the conductivity tensor field on grid.
    :return: conductivity_field, np.array, shape (n_elements, n_voight)
    """
    ellipses = [dmap.Ellipse(fr.normal, fr.center, fr.scale) for fr in fr_media.dfn]
    d_grid = dmap.Grid.make_grid(grid.origin, grid.step, grid.dimensions)
    fractures = dmap.map_dfn(d_grid, ellipses)
    fr_transmissivity = fr_media.fr_conductivity * fr_media.fr_cross_section
    k_iso = dmap.permIso(d_grid, fractures, fr_transmissivity, fr_media.conductivity)
    dmap.permAniso(d_grid, fractures, fr_transmissivity, fr_media.conductivity)
    k_voigt = k_iso[:, None] * np.array([1, 1, 1, 0, 0, 0])[None, :]
    return k_voigt

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

def rasterize_dfn(homo_fns):
    # Fracture set
    domain_size = 100

    # Coarse Problem
    steps = (10, 12, 14)
    grid = Grid(domain_size, steps, Fe.Q1(dim=3), origin=-domain_size / 2)
    dfn = tst_fracture_set(grid.dimensions)
    fr_media = FracturedMedia.constant_fr(dfn, 10, 1, 0.01)

    xyz_range = [ np.linspace(grid.origin[ax], grid.origin[ax] + grid.dimensions[ax], grid.n_steps[ax] + 1, dtype=np.float32)
                  for ax in [0, 1, 2]
                ]

    x, y, z = np.meshgrid(*xyz_range, indexing='ij')
    pv_grid = pv.StructuredGrid(x, y, z)
    #points = grid.nodes()
    for name, homo_fn in homo_fns.items():
        grid_permitivity = homo_fn(fr_media, grid)
        pv_grid.cell_data[name] = grid_permitivity.reshape(-1, 9)
    pv_grid.save(str(workdir / "test_resterize.vtk"))


def test_reasterize():
    homo_fns=dict(
        k_deco_iso=homo_decovalex_iso,
        k_deco_aniso_raw=homo_decovalex_aniso_raw,
        k_deco_aniso_diag=homo_decovalex_aniso_diag,
        k_deco_aniso_lump=homo_decovalex_aniso_lump
    )
    rasterize_dfn(homo_fns)
