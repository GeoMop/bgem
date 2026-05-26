"""
Test of homogenization techniquest within a two-scale problem.
- reference solution is evaluated by Flow123d, using direct rasterization of full DFN sample
  The pressure field is projected to the nodes of the rectangular grid,
  velocity filed is averaged over rectangular cells.

- two-scale solution involves:
  1. homogenization of DFN to the rectangular grid ; general permeability tensor field
  2. custom 3d solver for the rectangular grid is used to solve the coarse problem

- various homogenization techniques could be used, homogenization time is evaluated and compared.

TODO:
- use large fractures in first test in order to have no difference between ellipse and rectangle shapes
- set parameters to have signifficant difference between fracture and bulk velocities
- fix ref velocity projection
- homogenization test
- implement ellipse shape meshing for flow
- implement rasterization (decovalex based) for rectangles and polygons
"""

import pytest
import shutil
from pathlib import Path

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


import numpy as np
import pyvista as pv

from bgem import stochastic
from bgem.gmsh import gmsh, options
from mesh_class import Mesh
from bgem.core import call_flow, dotdict, workdir as workdir_mng
from bgem.upscale import fem_plot, fem, voigt_to_tn, FracturedMedia, decovalex_dfnmap as dmap
from scipy.interpolate import LinearNDInterpolator

script_dir = Path(__file__).absolute().parent
workdir = script_dir / "sandbox"
from joblib import Memory
memory = Memory(workdir, verbose=0)



def fracture_random_set(seed, size_range, max_frac = 1e21):
    rmin, rmax = size_range
    box_dimensions = (rmax, rmax, rmax)
    fr_cfg_path = script_dir/"fractures_conf.yaml"
    #with open() as f:
    #    pop_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    fr_pop = stochastic.Population.from_cfg(fr_cfg_path, box_dimensions, shape=stochastic.EllipseShape())
    if fr_pop.mean_size() > max_frac:
        common_range = fr_pop.common_range_for_sample_size(sample_size=max_frac)
        fr_pop = fr_pop.set_sample_range(common_range)
    print(f"fr set range: {[rmin, rmax]}, fr_lim: {max_frac}, mean population size: {fr_pop.mean_size()}")
    pos_gen = stochastic.UniformBoxPosition(fr_pop.domain)
    np.random.seed(seed)
    fractures = fr_pop.sample(pos_distr=pos_gen, keep_nonempty=True)
    #for fr in fractures:
    #    fr.region = gmsh.Region.get("fr", 2)
    return fractures

def fracture_fixed_set():
    rmin, rmax = size_range
    box_dimensions = (rmax, rmax, rmax)
    fr_cfg_path = script_dir/"fractures_conf.yaml"
    #with open() as f:
    #    pop_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    fr_pop = stochastic.Population.from_cfg(fr_cfg_path, box_dimensions, shape=stochastic.EllipseShape())
    if fr_pop.mean_size() > max_frac:
        common_range = fr_pop.common_range_for_sample_size(sample_size=max_frac)
        fr_pop = fr_pop.set_sample_range(common_range)
    print(f"fr set range: {[rmin, rmax]}, fr_lim: {max_frac}, mean population size: {fr_pop.mean_size()}")
    pos_gen = stochastic.UniformBoxPosition(fr_pop.domain)
    np.random.seed(seed)
    stochastic.FractureSet.from_list()
    fractures = fr_pop.sample(pos_distr=pos_gen, keep_nonempty=True)
    #for fr in fractures:
    #    fr.region = gmsh.Region.get("fr", 2)
    return fractures




def ref_solution_mesh(work_dir, domain_dimensions, fractures, fr_step, bulk_step):
    factory = gmsh.GeometryOCC("homo_cube", verbose=True)
    gopt = options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001
    box = factory.box(domain_dimensions)

    # TODO: use shape from fractures
    fractures, fr_region_map = create_fractures_rectangles(factory, fractures, factory.rectangle())
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
    return f_name, fr_region_map

def fr_cross_section(fractures, cross_to_r):
    return [cross_to_r * fr.r for fr in fractures]


def fr_field(mesh, dfn, reg_id_to_fr, fr_values, bulk_value):
    """
    Provide impoicit fields on fractures as input.
    :param mesh:
    :param fractures:
    :param fr_values:
    :param bulk_value:
    :return:
    """
    fr_map = mesh.fr_map(dfn, reg_id_to_fr) # np.array of fracture indices of elements, n_frac for nonfracture elements
    fr_values_ = np.concatenate((
        np.array(fr_values),
        np.atleast_1d(bulk_value)))
    field_vals = fr_values_[fr_map]
    return field_vals




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
def reference_solution(fr_media: FracturedMedia, dimensions, bc_gradient):
    dfn = fr_media.dfn
    bulk_conductivity = fr_media.conductivity

    workdir = script_dir / "sandbox"
    workdir.mkdir(parents=True, exist_ok=True)

    # Input crssection and conductivity
    mesh_file, fr_region_map = ref_solution_mesh(workdir, dimensions, dfn, fr_step=7, bulk_step=7)
    full_mesh = Mesh.load_mesh(mesh_file, heal_tol = 0.001) # gamma
    fields = dict(
        conductivity=fr_field(full_mesh, dfn, fr_region_map, fr_media.fr_conductivity, bulk_conductivity),
        cross_section=fr_field(full_mesh, dfn, fr_region_map, fr_media.fr_cross_section, 1.0)
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

def project_ref_solution_(flow_out, grid: fem.Grid):
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
    grid_min_corner = -grid.dimensions / 2
    centers_ijk_grid = (cell_centers_coords - grid_min_corner) // grid.step[None, :]
    centers_ijk_grid = centers_ijk_grid.astype(np.int32)
    assert np.alltrue(centers_ijk_grid < grid.shape[None, :])

    grid_cell_idx = centers_ijk_grid[:, 0] + grid.shape[0] * (centers_ijk_grid[:, 1] + grid.shape[1] * centers_ijk_grid[:, 2])
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

    return grid_velocities.reshape((*grid.shape, 3))

def det33(mat):
    """
    mat: (N, 3, 3)
    :param mat:
    :return: (N, )
    """
    return sum(
        np.prod(mat[:, [(col, (row+step)%3) for col in range(3)]])
        for row in [0, 1, 2] for step in [1,2]
    )

@memory.cache
def refine_barycenters(element, level):
    """
    Produce refinement of given element (triangle or tetrahedra), shape (N, n_vertices, 3)
    and return barycenters of refined subelements.
    """
    return np.mean(refine_element(element, level), axis=1)

@memory.cache
def project_adaptive_source_quad(flow_out, grid: fem.Grid):
    grid_cell_volume = np.prod(grid.step)/27

    ref_el_2d = np.array([(0, 0), (1, 0), (0, 1)])
    ref_el_3d = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])

    pvd_content = pv.get_reader(flow_out.hydro.spatial_file.path)
    pvd_content.set_active_time_point(0)
    dataset = pvd_content.read()[0] # Take first block of the Multiblock dataset

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


    #num_cells = dataset.n_cells
    #shifts = np.zeros((num_cells, 3))
    #transform_matrices = np.zeros((num_cells, 3, 3))
    #volumes = np.zeros(num_cells)

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
        #shifts[i] = shift

        transform_matrix = points[1:] - shift
        if len(points) == 4:  # Tetrahedron
            # For a tetrahedron, we use all three vectors formed from the first vertex
            #transform_matrices[i] = transform_matrix[:3].T
            # Volume calculation for a tetrahedron:
            volume = np.abs(np.linalg.det(transform_matrix[:3])) / 6
            ref_el = ref_el_3d
        elif len(points) == 3:  # Triangle
            # For a triangle, we use only two vectors
            #transform_matrices[i, :2] = transform_matrix.T
            # Area calculation for a triangle:
            volume = 0.5 * np.linalg.norm(np.cross(transform_matrix[0], transform_matrix[1])) * cross_section[i]
            ref_el = ref_el_2d
        level = max(int(np.log2(volume/grid_cell_volume) / 3.0), 0)
        levels[i] = level
        ref_barycenters = refine_barycenters(ref_el[None, :, :],level)
        barycenters = shift[None, :] + ref_barycenters @ transform_matrix
        grid_indices = grid.project_points(barycenters)
        weights_sum[grid_indices] += volume
        grid_velocities[grid_indices] += volume * velocities[i]
    print(np.bincount(levels))
    grid_velocities = grid_velocities / weights_sum[:, None]
    return grid_velocities


# Define transformation matrices and index mappings for 2D and 3D refinements
_transformation_matrices = {
    3: np.array([
        [1, 0, 0],  # Vertex 0
        [0, 1, 0],  # Vertex 1
        [0, 0, 1],  # Vertex 2
        [0.5, 0.5, 0],  # Midpoint between vertices 0 and 1
        [0, 0.5, 0.5],  # Midpoint between vertices 1 and 2
        [0.5, 0, 0.5],  # Midpoint between vertices 0 and 2
    ]),
    4: np.array([
        [1, 0, 0, 0],  # Vertex 0
        [0, 1, 0, 0],  # Vertex 1
        [0, 0, 1, 0],  # Vertex 2
        [0, 0, 0, 1],  # Vertex 3
        [0.5, 0.5, 0, 0],  # Midpoint between vertices 0 and 1
        [0.5, 0, 0.5, 0],  # Midpoint between vertices 0 and 2
        [0.5, 0, 0, 0.5],  # Midpoint between vertices 0 and 3
        [0, 0.5, 0.5, 0],  # Midpoint between vertices 1 and 2
        [0, 0.5, 0, 0.5],  # Midpoint between vertices 1 and 3
        [0, 0, 0.5, 0.5],  # Midpoint between vertices 2 and 3
    ])
}

_index_maps = {
    3: np.array([
        [0, 3, 5],  # Triangle 1
        [3, 1, 4],  # Triangle 2
        [3, 4, 5],  # Triangle 3
        [5, 4, 2]  # Triangle 4
    ]),
    4: np.array([
        [0, 4, 5, 6],  # Tetrahedron 1
        [1, 4, 7, 8],  # Tetrahedron 2
        [2, 5, 7, 9],  # Tetrahedron 3
        [3, 6, 8, 9],  # Tetrahedron 4
        [4, 5, 6, 7],  # Center tetrahedron 1
        [4, 7, 8, 6],  # Center tetrahedron 2
        [5, 7, 9, 6],  # Center tetrahedron 3
        [6, 8, 9, 7],  # Center tetrahedron 4
    ])
}


def refine_element(element, level):
    """
    Recursively refines an element (triangle or tetrahedron) in space using matrix multiplication.

    :param element: A numpy array of shape (1, N, M), where N is the number of vertices (3 or 4).
    :param level: Integer, the level of refinement.
    :return: A numpy array containing the vertices of all refined elements.
    """
    if level == 0:
        return element
    n_tria, num_vertices, dim = element.shape
    assert n_tria == 1
    assert num_vertices == dim + 1
    transformation_matrix = _transformation_matrices[num_vertices]
    index_map = _index_maps[num_vertices]
    # Generate all nodes by applying the transformation matrix to the original vertices
    nodes = np.dot(transformation_matrix, element[0])
    # Construct new elements using advanced indexing
    new_elements = nodes[index_map]
    # Recursively refine each smaller element
    result = np.vstack([
        refine_element(new_elem[None, :, :], level - 1) for new_elem in new_elements
    ])
    return result


def plot_triangles(triangles):
    """
    Plots a series of refined triangles.

    :param triangles: A numpy array of shape (N, 3, 2) containing the vertices of all triangles.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Flatten the array for plotting
    triangles_flat = triangles.reshape(-1, 2)
    tri_indices = np.arange(len(triangles_flat)).reshape(-1, 3)

    # Create a Triangulation object
    triangulation = tri.Triangulation(triangles_flat[:, 0], triangles_flat[:, 1], tri_indices)

    # Plot the triangulation
    ax.triplot(triangulation, 'ko-')

    # Setting the aspect ratio to be equal to ensure the triangle is not distorted
    ax.set_aspect('equal')

    # Turn off the grid
    ax.grid(False)

    # Setting the limits to get a better view
    ax.set_xlim(triangles_flat[:, 0].min() - 0.1, triangles_flat[:, 0].max() + 0.1)
    ax.set_ylim(triangles_flat[:, 1].min() - 0.1, triangles_flat[:, 1].max() + 0.1)

    # Add a title
    plt.title('Refined Triangles Visualization')
    plt.show()

@pytest.mark.skip
def test_refine_triangle():
    # Example usage
    initial_triangle = np.array([[[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]]])
    L = 2  # Set the desired level of refinement

    # Refine the triangle
    refined_triangles = refine_element(initial_triangle, L)
    print(f"Refined Triangles at Level {L}:")
    print(refined_triangles)
    print("Total triangles:", len(refined_triangles))

    plot_triangles(refined_triangles)



def plot_tetrahedra(tetrahedra):
    """
    Plots a series of refined tetrahedra in 3D.

    :param tetrahedra: A numpy array of shape (N, 4, 3) containing the vertices of all tetrahedra.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Flatten the array for plotting
    for tet in tetrahedra:
        vtx = tet.reshape(-1, 3)
        tri = [[vtx[0], vtx[1], vtx[2]],
               [vtx[0], vtx[1], vtx[3]],
               [vtx[0], vtx[2], vtx[3]],
               [vtx[1], vtx[2], vtx[3]]]
        for s in tri:
            poly = Poly3DCollection([s], edgecolor='k', alpha=0.2, facecolor=np.random.rand(3, ))
            ax.add_collection3d(poly)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    plt.title('Refined Tetrahedra Visualization')
    plt.show()

@pytest.mark.skip
def test_refine_tetra():
    initial_tetrahedron = np.array(
        [[[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3]]])
    L = 1  # Set the desired level of refinement

    # Refine the tetrahedron
    refined_tetrahedra = refine_element(initial_tetrahedron, L)
    plot_tetrahedra(refined_tetrahedra)


"""
Projection using aditional quad points on the source mesh.
We need to generate baricenters of an L-order refinement of a simplex.
Vertices of first reinement of triangle:
vertices coords relative to V0, V1, V2 (bary coords)
T0: (1, 0, 0), (1/2, 1/2, 0), (1/2, 0, 1/2) 
T1: (1/2, 1/2, 0), (0, 1, 0),  (0, 1/2, 1/2) 
T2: (1/2, 1/2, 0), (0, 1/2, 1/2), (0, 0, 1), 
T3: (1/2, 1/2, 0), (1/2, 1/2, 0), (1/2, 0, 1/2) 

ON size 2 grid:
T0: (1, 0, 0), (1/2, 1/2, 0), (1/2, 0, 1/2) 
T1: (1/2, 1/2, 0), (0, 1, 0),  (0, 1/2, 1/2) 
T2: (1/2, 1/2, 0), (0, 1/2, 1/2), (0, 0, 1), 
T3: (1/2, 1/2, 0), (1/2, 1/2, 0), (1/2, 0, 1/2) 

... tensor (4, 3, 3) ... n_childes, n_source_veritices, n_result verites (same as source)
source_vertices ... shape (n_vertices, coords_3d)
.... T[:n_childs, :n_vertices, :, None] * source_vertices[None, :, None, :] 

Iterative aplication of the tensor + finaly barycenters.
"""

def project_ref_solution(flow_out, grid: fem.Grid):
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
    velocities = dataset.cell_data['velocity_p0']
    interpolator = LinearNDInterpolator(cell_centers_coords, velocities, 0.0)
    grid_velocities = interpolator(grid.barycenters())
    return grid_velocities


def homo_decovalex(fr_media: FracturedMedia, grid:fem.Grid):
    """
    Homogenize fr_media to the conductivity tensor field on grid.
    :return: conductivity_field, np.array, shape (n_elements, n_voight)
    """
    ellipses = [dmap.Ellipse(fr.normal, fr.center, fr.scale) for fr in fr_media.dfn]
    d_grid = dmap.Grid.make_grid(grid.origin, grid.step, grid.dimensions)
    fractures = dmap.map_dfn(d_grid, ellipses)
    fr_transmissivity = fr_media.fr_conductivity * fr_media.fr_cross_section
    k_iso_zyx = dmap.permIso(d_grid, fractures, fr_transmissivity, fr_media.conductivity)
    k_iso_xyz = grid.cell_field_C_like(k_iso_zyx)
    k_voigt = k_iso_xyz[:, None] * np.array([1, 1, 1, 0, 0, 0])[None, :]
    return k_voigt

#@pytest.mark.skip
def test_two_scale():
    # Fracture set
    domain_size = 100
    #fr_range = (30, domain_size)
    fr_range = (50, domain_size)

    # Random fractures
    # dfn = fracture_random_set(123, fr_range, max_frac=10)

    # Fixed fracture set
    shape_id = stochastic.EllipseShape.id
    fr  = lambda c, n : stochastic.Fracture(shape_id, [100, 100], c, n)
    fractures = [
        #fr([30,-10, 10], [0, 1, 0]),
        fr([0,0,0], [1, 1, 0]),
        #fr([-30,10,-10], [0, 1, 0])
    ]
    dfn = stochastic.FractureSet.from_list(fractures)
    # Cubic law transmissvity
    fr_media = FracturedMedia.fracture_cond_params(dfn, 1e-4, 0.001)

    # Fractures and properties from DFNWorks
    #fr_media = FracturedMedia.from_dfn_works("", bulk_conductivity)


    # Coarse Problem
    #steps = (50, 60, 70)
    steps = (9, 10, 11)
    #steps = (50, 60, 70)
    #steps = (3, 4, 5)
    fem_grid = fem.fem_grid(domain_size, steps, fem.Fe.Q(dim=3), origin=-domain_size / 2)
    bc_pressure_gradient = [1, 0, 0]
    grid_cond = homo_decovalex(fr_media, fem_grid.grid)
    #grid_cond = np.ones(grid.n_elements)[:, None] * np.array([1, 1, 1, 0, 0, 0])[None, :]
    pressure = fem_grid.solve_sparse(grid_cond, np.array(bc_pressure_gradient)[None, :])
    assert not np.any(np.isnan(pressure))

    flow_out = reference_solution(fr_media, fem_grid.grid.dimensions, bc_pressure_gradient)
    project_fn = project_adaptive_source_quad
    #project_fn = project_ref_solution
    #ref_velocity_grid = grid.cell_field_F_like(project_fn(flow_out, grid).reshape((-1, 3)))
    ref_velocity_grid = project_fn(flow_out, fem_grid.grid).reshape((-1, 3))

    grad_pressure = fem_grid.field_grad(pressure)   # (n_vectors, n_els, dim)
    grad_pressure = grad_pressure[0, :, :][:, :, None]  # (n_els, dim, 1)
    velocity = -voigt_to_tn(grid_cond) @ grad_pressure    # (n_els, dim, 1)
    #velocity = grad_pressure    # (n_els, dim, 1)
    velocity = velocity[:, :, 0] # transpose
    velocity_zyx = fem_grid.grid.cell_field_F_like(velocity)  #.reshape(*grid.n_steps, -1).transpose(2,1,0,3).reshape((-1, 3))
    # Comparison
    # origin = [0, 0, 0]

    #pv_grid = pv.StructuredGrid()
    x, y, z = np.meshgrid(*fem_grid.grid.axes_linspace(), indexing='ij')
    pv_grid = pv.StructuredGrid(x, y, z)
    #points = grid.nodes()
    pv_grid_centers = pv_grid.cell_centers().points
    print(fem_grid.grid.barycenters())
    print(pv_grid_centers)

    cell_fields = dict(
        ref_velocity = ref_velocity_grid,
        homo_velocity = velocity_zyx,
        diff = velocity_zyx - ref_velocity_grid,
        homo_cond = fem_grid.grid.cell_field_F_like(grid_cond),
    )
    point_fields = dict(
        homo_pressure=pressure[0]
    )
    pv_grid = fem_plot.grid_fields_vtk(fem_grid.grid, cell_fields, vtk_path=workdir/'test_result.vtk')
    plotter = fem_plot.create_plotter() #off_screen=True, window_size=(1024, 768))
    plotter.add_mesh(pv_grid, scalars='ref_velocity')
    plotter.show()
    #pv_grid.dimensions = grid.n_steps + 1
    #pv_grid.cell_data['ref_velocity'] = np.arange(22*20*18).reshape((18,20,22)).transpose((2, 1, 0)).reshape((-1,))

    #pv_grid.save(str(workdir / "test_result.vtk"))

