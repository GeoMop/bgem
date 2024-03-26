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
from bgem import stochastic
from bgem.gmsh import gmsh, options
from mesh_class import Mesh
from bgem.core import call_flow, dotdict, workdir as workdir_mng
from bgem.upscale import Grid, Fe
import pyvista as pv

script_dir = Path(__file__).absolute().parent
workdir = script_dir / "sandbox"
from joblib import Memory
memory = Memory(workdir, verbose=0)


@attrs.define
class FracturedMedia:
    dfn: List[stochastic.Fracture]
    fr_cross_section: np.ndarray    # shape (n_fractures,)
    fr_conductivity: np.ndarray     # shape (n_fractures,)
    conductvity: float

    @staticmethod
    def fracture_cond_params(dfn, unit_cross_section, bulk_conductivity):
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
        return FracturedMedia(dfn, fr_cross_section, fr_cond, bulk_conductivity)

def fracture_set(seed, size_range, max_frac = None):
    rmin, rmax = size_range
    box_dimensions = (rmax, rmax, rmax)
    with open(script_dir/"fractures_conf.yaml") as f:
        pop_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    fr_pop = stochastic.Population.initialize_3d(pop_cfg, box_dimensions)
    fr_pop.set_sample_range([rmin, rmax], sample_size=max_frac)
    print(f"fr set range: {[rmin, rmax]}, fr_lim: {max_frac}, mean population size: {fr_pop.mean_size()}")
    pos_gen = stochastic.UniformBoxPosition(fr_pop.domain)
    np.random.seed(seed)
    fractures = fr_pop.sample(pos_distr=pos_gen, keep_nonempty=True)
    for fr in fractures:
        fr.region = gmsh.Region.get("fr", 2)
    return fractures

def create_fractures_rectangles(gmsh_geom, fractures, base_shape: gmsh.ObjectSet,
                                shift = np.array([0,0,0])):
    # From given fracture date list 'fractures'.
    # transform the base_shape to fracture objects
    # fragment fractures by their intersections
    # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
    if len(fractures) == 0:
        return []

    shapes = []
    for i, fr in enumerate(fractures):
        shape = base_shape.copy()
        print("fr: ", i, "tag: ", shape.dim_tags)
        shape = shape.scale([fr.rx, fr.ry, 1]) \
            .rotate(axis=[0,0,1], angle=fr.shape_angle) \
            .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
            .translate(fr.center + shift) \
            .set_region(fr.region)

        shapes.append(shape)

    fracture_fragments = gmsh_geom.fragment(*shapes)
    return fracture_fragments


def ref_solution_mesh(work_dir, domain_dimensions, fractures, fr_step, bulk_step):
    factory = gmsh.GeometryOCC("homo_cube", verbose=True)
    gopt = options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001
    box = factory.box(domain_dimensions)

    fractures = create_fractures_rectangles(factory, fractures, factory.rectangle())
    fractures_group = factory.group(*fractures).intersect(box)
    box_fr, fractures_fr = factory.fragment(box, fractures_group)
    fractures_fr.mesh_step(fr_step) #.set_region("fractures")
    objects = [box_fr, fractures_fr]
    factory.write_brep(str(work_dir / factory.model_name) )
    #factory.mesh_options.CharacteristicLengthMin = cfg.get("min_mesh_step", cfg.boreholes_mesh_step)
    factory.mesh_options.CharacteristicLengthMax = bulk_step
    #factory.mesh_options.Algorithm = options.Algorithm3d.MMG3D

    # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
    # mesh.Algorithm = options.Algorithm2d.Delaunay
    # mesh.Algorithm = options.Algorithm2d.FrontalDelaunay

    factory.mesh_options.Algorithm = options.Algorithm3d.Delaunay
    #mesh.ToleranceInitialDelaunay = 0.01
    # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
    #mesh.CharacteristicLengthFromPoints = True
    #factory.mesh_options.CharacteristicLengthFromCurvature = False
    #factory.mesh_options.CharacteristicLengthExtendFromBoundary = 2  # co se stane if 1
    #mesh.CharacteristicLengthMin = min_el_size
    #mesh.CharacteristicLengthMax = max_el_size

    #factory.keep_only(*objects)
    #factory.remove_duplicate_entities()
    factory.make_mesh(objects, dim=3)
    #factory.write_mesh(me gmsh.MeshFormat.msh2) # unfortunately GMSH only write in version 2 format for the extension 'msh2'
    f_name = work_dir / (factory.model_name + ".msh2")
    factory.write_mesh(str(f_name), format=gmsh.MeshFormat.msh2)
    return f_name

def fr_cross_section(fractures, cross_to_r):
    return [cross_to_r * fr.r for fr in fractures]


def fr_field(mesh, fractures, fr_values, bulk_value):
    fr_map = mesh.fr_map(fractures) # np.array of fracture indices of elements, n_frac for nonfracture elements
    fr_values_ = np.concatenate((
        np.array(fr_values),
        np.atleast_1d(bulk_value)))
    return fr_values_[fr_map]




# def velocity_p0(grid_step, min_corner, max_corner, mesh, values):
#     """
#     Pressure P1 field projection
#     - P0 pressure in barycenters
#     - use interpolation to construct P1 Structured grid
#     Interpolate: discrete points -> field, using RBF placed at points
#     Sample: field -> nodal data, evaluate nodes in source field
#
#     Velocity P0 projection
#     1. get array of barycenters and element volumes
#     2. ijk cell coords of each source point
#     3. weights = el_volume / np.add.at(cell_vol_sum, ijk, el_volume)[ijk[:]]
#     4. np.add.at(cell_velocities, ijk, weights * velocities)
#     :return:
#     """
#     pass

@memory.cache
def reference_solution(fr_media, target_grid, bc_gradient):
    domain_dimensions = target_grid.size
    dfn = fr_media.dfn
    bulk_conductivity = fr_media.conductivity

    workdir = script_dir / "sandbox"
    workdir.mkdir(parents=True, exist_ok=True)

    # Input crssection and conductivity
    mesh_file = ref_solution_mesh(workdir, domain_dimensions, dfn, fr_step=10, bulk_step=10)
    full_mesh = Mesh.load_mesh(mesh_file, heal_tol = 0.001) # gamma
    fields = dict(
        conductivity=fr_field(full_mesh, dfn, fr_media.fr_conductivity, bulk_conductivity),
        cross_section=fr_field(full_mesh, dfn, fr_media.fr_cross_section, 1.0)
    )
    cond_file = full_mesh.write_fields(str(workdir / "input_fields.msh2"), fields)
    cond_file = Path(cond_file)
    cond_file =  cond_file.rename(cond_file.with_suffix(".msh"))
    # solution
    flow_cfg = dotdict(
        flow_executable=[
         "/home/jb/workspace/flow123d/bin/fterm",
         "--no-term",
#        - flow123d/endorse_ci:a785dd
#        - flow123d/ci-gnu:4.0.0a_d61969
         "dbg",
         "run",
         "--profiler_path",
         "profile"
        ],
        mesh_file=cond_file,
        pressure_grad=bc_gradient
    )
    f_template = "flow_upscale_templ.yaml"
    shutil.copy( (script_dir / f_template), workdir)
    with workdir_mng(workdir):
        flow_out = call_flow(flow_cfg, f_template, flow_cfg)

    # Project to target grid
    print(flow_out)
    #vel_p0 = velocity_p0(target_grid, flow_out)
    # projection of fields
    return flow_out

def project_ref_solution(flow_out, grid: Grid):
    #     Velocity P0 projection
    #     1. get array of barycenters (source points) and element volumes of the fine mesh
    #     2. ijk cell coords of each source point
    #     3. weights = el_volume / np.add.at(cell_vol_sum, ijk, el_volume)[ijk[:]]
    #     4. np.add.at(cell_velocities, ijk, weights * velocities)
    #     :return:
    pvd_content = pv.get_reader(flow_out.hydro.spatial_file.path)
    pvd_content.set_active_time_point(0)
    dataset = pvd_content.read()[0] # Take first block of the Multiblock dataset
    cell_centers_coords = dataset.cell_centers().points
    grid_min_corner = -grid.size / 2
    centers_ijk_grid = (cell_centers_coords - grid_min_corner) // grid.step[None, :]
    centers_ijk_grid = centers_ijk_grid.astype(np.int32)
    assert np.alltrue(centers_ijk_grid < grid.n_steps[None, :])

    grid_cell_idx = centers_ijk_grid[:, 0] + grid.n_steps[0] * (centers_ijk_grid[:, 1] + grid.n_steps[1]*centers_ijk_grid[:, 2])
    sized = dataset.compute_cell_sizes()
    cell_volume = np.abs(sized.cell_data["Volume"])
    grid_sum_cell_volume = np.zeros(grid.n_elements)
    np.add.at(grid_sum_cell_volume, grid_cell_idx, cell_volume)
    weights = cell_volume[:] / grid_sum_cell_volume[grid_cell_idx[:]]
    velocities = dataset.cell_data['velocity_p0']
    grid_velocities = np.zeros((grid.n_elements, 3))
    wv = weights[:, None] * velocities
    for ax in [0, 1, 2]:
        np.add.at(grid_velocities[:, ax], grid_cell_idx, wv[:, ax])

    return grid_velocities.reshape( (*grid.n_steps, 3))


def homogenize(fr_media: FracturedMedia, grid:Grid):
    """
    Homogenize fr_media to the conductivity tensor field on grid.
    :return: conductivity_field, np.array, shape (n_elements, n_voight)
    """



def test_two_scale():
    # Fracture set
    domain_size = 100
    fr_range = (30, domain_size)
    dfn = fracture_set(123, fr_range)
    fr_media = FracturedMedia.fracture_cond_params(dfn, 1e-4, 0.1)


    # Coarse Problem
    steps = (10, 2, 5)
    grid = Grid(domain_size, steps, Fe.Q1(dim=3))
    bc_pressure_gradient = [1, 0, 0]

    flow_out = reference_solution(fr_media, grid, bc_pressure_gradient)
    ref_velocity_grid = project_ref_solution(flow_out, grid)

    grid_permitivity = homogenize(fr_media, grid)
    pressure_field = grid.solve_system(grid_permitivity, bc_pressure_gradient)