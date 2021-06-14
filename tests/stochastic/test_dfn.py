"""

"""

import os
import itertools
import pytest

from bgem.polygons import polygons as poly
from bgem.stochastic import frac_plane as FP
from bgem.stochastic import isec_plane_point as IPP

import scipy.linalg as la
import numpy.linalg as lan
import attr
import numpy as np
import collections
# import matplotlib.pyplot as plt

# from bgem
from bgem.gmsh import gmsh
from bgem.gmsh import options as gmsh_options
from bgem.gmsh import field as gmsh_field
from bgem.stochastic import fracture
from bgem.bspline import brep_writer as bw

script_dir = os.path.dirname(os.path.realpath(__file__))

geometry_dict = {
    'box_dimensions': [600, 600, 600],
    'center_depth': 5000,
    'fracture_mesh_step': 15,
    'n_frac_limit': 200,
    'well_distance': 200,
    'well_effective_radius': 10,
    'well_openning': [-50, 50]}

fracture_stats = [
    {'concentration': 17.8,
     'name': 'NS',
     'p_32': 0.094,
     'plunge': 1,
     'power': 2.5,
     'r_max': 564,
     'r_min': 0.038,
     'trend': 292},
    {'concentration': 14.3,
     'name': 'NE',
     'p_32': 0.163,
     'plunge': 2,
     'power': 2.7,
     'r_max': 564,
     'r_min': 0.038,
     'trend': 326},
    {'concentration': 12.9,
     'name': 'NW',
     'p_32': 0.098,
     'plunge': 6,
     'power': 3.1,
     'r_max': 564,
     'r_min': 0.038,
     'trend': 60},
    {'concentration': 14.0,
     'name': 'EW',
     'p_32': 0.039,
     'plunge': 2,
     'power': 3.1,
     'r_max': 564,
     'r_min': 0.038,
     'trend': 15},
    {'concentration': 15.2,
     'name': 'HZ',
     'p_32': 0.141,
     'plunge': 86,
     'power': 2.38,
     'r_max': 564,
     'r_min': 0.038,
     'trend': 5}]


# TODO:
# - enforce creation of empty physical groups, or creation of empty regions in the flow input
# - speedup mechanics

@attr.s(auto_attribs=True)
class ValueDescription:
    time: float
    position: str
    quantity: str
    unit: str


def to_polar(x, y, z):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    if z > 0:
        phi += np.pi
    return (phi, rho)


def plot_fr_orientation(fractures):
    family_dict = collections.defaultdict(list)
    for fr in fractures:
        x, y, z = \
        fracture.FisherOrientation.rotate(np.array([0, 0, 1]), axis=fr.rotation_axis, angle=fr.rotation_angle)[0]
        family_dict[fr.region].append([
            to_polar(z, y, x),
            to_polar(z, x, -y),
            to_polar(y, x, z)
        ])

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
    for name, data in family_dict.items():
        # data shape = (N, 3, 2)
        data = np.array(data)
        for i, ax in enumerate(axes):
            phi = data[:, i, 0]
            r = data[:, i, 1]
            c = ax.scatter(phi, r, cmap='hsv', alpha=0.75, label=name)
    axes[0].set_title("X-view, Z-north")
    axes[1].set_title("Y-view, Z-north")
    axes[2].set_title("Z-view, Y-north")
    for ax in axes:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
    fig.legend(loc=1)
    fig.savefig("fracture_orientation.pdf")
    plt.close(fig)
    # plt.show()


def generate_fractures(geometry_dict, statistics):
    dimensions = geometry_dict["box_dimensions"]
    well_z0, well_z1 = geometry_dict["well_openning"]
    well_length = well_z1 - well_z0
    well_r = geometry_dict["well_effective_radius"]
    well_dist = geometry_dict["well_distance"]

    # generate fracture set
    fracture_box = [1.5 * well_dist, 1.5 * well_length, 1.5 * well_length]
    volume = np.product(fracture_box)
    pop = fracture.Population(volume)
    pop.initialize(statistics)
    pop.set_sample_range([1, well_dist], sample_size=geometry_dict["n_frac_limit"])
    print("total mean size: ", pop.mean_size())
    connected_position = geometry_dict.get('connected_position_distr', False)
    if connected_position:
        # Not yet supported.
        eps = well_r / 2
        left_well_box = [-well_dist / 2 - eps, -eps, well_z0, -well_dist / 2 + eps, +eps, well_z1]
        right_well_box = [well_dist / 2 - eps, -eps, well_z0, well_dist / 2 + eps, +eps, well_z1]
        pos_gen = fracture.ConnectedPosition(
            confining_box=fracture_box,
            init_boxes=[left_well_box, right_well_box])
    else:
        pos_gen = fracture.UniformBoxPosition(fracture_box)
    fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=True)
    # fracture.fr_intersect(fractures)

    for fr in fractures:
        fr.region = "fr"
    used_families = set((f.region for f in fractures))
    # config_fracture_regions(used_families)
    return fractures


# def config_fracture_regions(used_families):
#     for model in ["hm_params", "th_params", "th_params_ref"]:
#         model_dict = config_dict[model]
#         model_dict["fracture_regions"] = list(used_families)
#         model_dict["left_well_fracture_regions"] = [".{}_left_well".format(f) for f in used_families]
#         model_dict["right_well_fracture_regions"] = [".{}_right_well".format(f) for f in used_families]


def create_fractures_rectangles(gmsh_geom, fractures, base_shape: 'ObjectSet'):
    # From given fracture date list 'fractures'.
    # transform the base_shape to fracture objects
    # fragment fractures by their intersections
    # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
    shapes = []
    for i, fr in enumerate(fractures):
        shape = base_shape.copy()
        print("fr: ", i, "tag: ", shape.dim_tags)
        shape = shape.scale([fr.rx, fr.ry, 1]) \
            .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
            .translate(fr.centre) \
            .set_region(fr.region)

        shapes.append(shape)

    fracture_fragments = gmsh_geom.fragment(*shapes)
    return fracture_fragments


def create_fractures_polygons(gmsh_geom, fractures):
    # From given fracture date list 'fractures'.
    # transform the base_shape to fracture objects
    # fragment fractures by their intersections
    # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
    frac_obj = fracture.Fractures(fractures)
    frac_obj.snap_vertices_and_edges()
    shapes = []
    for fr, square in zip(fractures, frac_obj.squares):
        shape = gmsh_geom.make_polygon(square).set_region(fr.region)
        shapes.append(shape)

    fracture_fragments = gmsh_geom.fragment(*shapes)
    return fracture_fragments


def make_mesh(geometry_dict, fractures: fracture.Fracture, mesh_name: str):
    """
    Create the GMSH mesh from a list of fractures using the bgem.gmsh interface.
    """
    fracture_mesh_step = geometry_dict['fracture_mesh_step']
    dimensions = geometry_dict["box_dimensions"]
    well_z0, well_z1 = geometry_dict["well_openning"]
    well_r = geometry_dict["well_effective_radius"]
    well_dist = geometry_dict["well_distance"]

    factory = gmsh.GeometryOCC(mesh_name, verbose=True)
    gopt = gmsh_options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001
    # gopt.MatchMeshTolerance = 1e-1

    # Main box
    box = factory.box(dimensions).set_region("box")
    side_z = factory.rectangle([dimensions[0], dimensions[1]])
    side_y = factory.rectangle([dimensions[0], dimensions[2]])
    side_x = factory.rectangle([dimensions[2], dimensions[1]])
    sides = dict(
        side_z0=side_z.copy().translate([0, 0, -dimensions[2] / 2]),
        side_z1=side_z.copy().translate([0, 0, +dimensions[2] / 2]),
        side_y0=side_y.copy().translate([0, 0, -dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_y1=side_y.copy().translate([0, 0, +dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_x0=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        side_x1=side_x.copy().translate([0, 0, +dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.modify_regions(name)

    b_box = box.get_boundary().copy()

    # two vertical cut-off wells, just permeable part
    left_center = [-well_dist / 2, 0, 0]
    right_center = [+well_dist / 2, 0, 0]
    left_well = factory.cylinder(well_r, axis=[0, 0, well_z1 - well_z0]) \
        .translate([0, 0, well_z0]).translate(left_center)
    right_well = factory.cylinder(well_r, axis=[0, 0, well_z1 - well_z0]) \
        .translate([0, 0, well_z0]).translate(right_center)
    b_right_well = right_well.get_boundary()
    b_left_well = left_well.get_boundary()

    print("n fractures:", len(fractures))
    fractures = create_fractures_rectangles(factory, fractures, factory.rectangle())
    # fractures = create_fractures_polygons(factory, fractures)
    fractures_group = factory.group(*fractures)
    # fractures_group = fractures_group.remove_small_mass(fracture_mesh_step * fracture_mesh_step / 10)

    # drilled box and its boundary
    box_drilled = box.cut(left_well, right_well)

    # fractures, fragmented, fractures boundary
    print("cut fractures by box without wells")
    fractures_group = fractures_group.intersect(box_drilled.copy())
    print("fragment fractures")
    box_fr, fractures_fr = factory.fragment(box_drilled, fractures_group)
    print("finish geometry")
    b_box_fr = box_fr.get_boundary()
    b_left_r = b_box_fr.select_by_intersect(b_left_well).set_region(".left_well")
    b_right_r = b_box_fr.select_by_intersect(b_right_well).set_region(".right_well")

    box_all = []
    for name, side_tool in sides.items():
        isec = b_box_fr.select_by_intersect(side_tool)
        box_all.append(isec.modify_regions("." + name))
    box_all.extend([box_fr, b_left_r, b_right_r])

    b_fractures = factory.group(*fractures_fr.get_boundary_per_region())
    b_fractures_box = b_fractures.select_by_intersect(b_box).modify_regions("{}_box")
    b_fr_left_well = b_fractures.select_by_intersect(b_left_well).modify_regions("{}_left_well")
    b_fr_right_well = b_fractures.select_by_intersect(b_right_well).modify_regions("{}_right_well")
    b_fractures = factory.group(b_fr_left_well, b_fr_right_well, b_fractures_box)
    mesh_groups = [*box_all, fractures_fr, b_fractures]

    print(fracture_mesh_step)
    # fractures_fr.set_mesh_step(fracture_mesh_step)

    factory.keep_only(*mesh_groups)
    factory.remove_duplicate_entities()
    factory.write_brep()

    min_el_size = fracture_mesh_step / 10
    fracture_el_size = np.max(dimensions) / 20
    max_el_size = np.max(dimensions) / 8

    fracture_el_size = gmsh_field.constant(fracture_mesh_step, 10000)
    frac_el_size_only = gmsh_field.restrict(fracture_el_size, fractures_fr, add_boundary=True)
    gmsh_field.set_mesh_step_field(frac_el_size_only)

    mesh = gmsh_options.Mesh()
    # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
    # mesh.Algorithm = options.Algorithm2d.Delaunay
    # mesh.Algorithm = options.Algorithm2d.FrontalDelaunay
    # mesh.Algorithm3D = options.Algorithm3d.Frontal
    # mesh.Algorithm3D = options.Algorithm3d.Delaunay
    mesh.ToleranceInitialDelaunay = 0.01
    # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
    mesh.CharacteristicLengthFromPoints = True
    mesh.CharacteristicLengthFromCurvature = True
    mesh.CharacteristicLengthExtendFromBoundary = 2
    mesh.CharacteristicLengthMin = min_el_size
    mesh.CharacteristicLengthMax = max_el_size
    mesh.MinimumCirclePoints = 6
    mesh.MinimumCurvePoints = 2

    # factory.make_mesh(mesh_groups, dim=2)
    factory.make_mesh(mesh_groups)
    factory.write_mesh(format=gmsh.MeshFormat.msh2)
    os.rename(mesh_name + ".msh2", mesh_name + ".msh")
    factory.show()


# def find_fracture_neigh(mesh, fract_regions, n_levels=1):
#     """
#     Find neighboring elements in the bulk rock in the vicinity of the fractures.
#     Creates several levels of neighbors.
#     :param mesh: GmshIO mesh object
#     :param fract_regions: list of physical names of the fracture regions
#     :param n_levels: number of layers of elements from the fractures
#     :return:
#     """
#
#     # make node -> element map
#     node_els = collections.defaultdict(set)
#     max_ele_id = 0
#     for eid, e in mesh.elements.items():
#         max_ele_id = max(max_ele_id, eid)
#         type, tags, node_ids = e
#         for n in node_ids:
#             node_els[n].add(eid)
#
#     print("max_ele_id = %d" % max_ele_id)
#
#     # select ids of fracture regions
#     fr_regs = fract_regions
#     # fr_regs = []
#     # for fr in fract_regions:
#     #     rid, dim = mesh.physical['fr']
#     #     assert dim == 2
#     #     fr_regs.append(rid)
#
#     # for n in node_els:
#     #     if len(node_els[n]) > 1:
#     #         print(node_els[n])
#
#     visited_elements = np.zeros(shape=(max_ele_id+1, 1), dtype=int)
#     fracture_neighbors = []
#
#     def find_neighbors(mesh, element, level, fracture_neighbors, visited_elements):
#         """
#         Auxiliary function which finds bulk neighbor elements to 'element' and
#         saves them to list 'fracture_neighbors'.
#         'visited_elements' keeps track of already investigated elements
#         'level' is number of layer from the fractures in which we search
#         """
#         type, tags, node_ids = element
#         ngh_elements = common_elements(node_ids, mesh, node_els, True)
#         for ngh_eid in ngh_elements:
#             if visited_elements[ngh_eid] > 0:
#                 continue
#             ngh_ele = mesh.elements[ngh_eid]
#             ngh_type, ngh_tags, ngh_node_ids = ngh_ele
#             if ngh_type == 4:  # if they are bulk elements and not already added
#                 visited_elements[ngh_eid] = 1
#                 fracture_neighbors.append((ngh_eid, level))  # add them
#
#     # ele type: 1 - line, 2-triangle, 4-tetrahedron, 15-node
#     # find the first layer of elements neighboring to fractures
#     for eid, e in mesh.elements.items():
#         type, tags, node_ids = e
#         if type == 2: # fracture elements
#             visited_elements[eid] = 1
#             if tags[0] not in fr_regs:  # is element in fracture region ?
#                 continue
#             find_neighbors(mesh, element=e, level=0, fracture_neighbors=fracture_neighbors,
#                            visited_elements=visited_elements)
#
#     # find next layers of elements from the first layer
#     for i in range(1, n_levels):
#         for eid, lev in fracture_neighbors:
#              if lev < i:
#                  e = mesh.elements[eid]
#                  find_neighbors(mesh, element=e, level=i, fracture_neighbors=fracture_neighbors,
#                                 visited_elements=visited_elements)
#
#     return fracture_neighbors
#
#
# def common_elements(node_ids, mesh, node_els, subset=False, max=1000):
#     """
#     Finds elements common to the given nodes.
#     :param node_ids: Ids of the nodes for which we look for common elements.
#     :param mesh:
#     :param node_els: node -> element map
#     :param subset: if true, it returns all the elements that are adjacent to at least one of the nodes
#                    if false, it returns all the elements adjacent to all the nodes
#     :param max:
#     :return:
#     """
#     # Generates active elements common to given nodes.
#     node_sets = [node_els[n] for n in node_ids]
#     if subset:
#         elements = list(set(itertools.chain.from_iterable(node_sets)))  # remove duplicities
#     else:
#         elements = set.intersection(*node_sets)
#
#     if len(elements) > max:
#         print("Too many connected elements:", len(elements), " > ", max)
#         for eid in elements:
#             type, tags, node_ids = mesh.elements[eid]
#             print("  eid: ", eid, node_ids)
#     # return elements
#     return active(mesh, elements)
#
#
# def active(mesh, element_iterable):
#     for eid in element_iterable:
#         if eid in mesh.elements:
#             yield eid

# def test_fracture_neighbors(config_dict):
#     """
#     Function that tests finding fracture neighbors.
#     It outputs mesh data - level per element.
#     :param config_dict:
#     :return:
#     """
#     setup_dir(config_dict, clean=True)
#     mesh_repo = config_dict.get('mesh_repository', None)
#     if mesh_repo:
#         healed_mesh = sample_mesh_repository(mesh_repo)
#         config_fracture_regions(config_dict["fracture_regions"])
#     else:
#         fractures = generate_fractures(config_dict)
#         # plot_fr_orientation(fractures)
#         healed_mesh = prepare_mesh(config_dict, fractures)
#         print("Created mesh: " + os.path.basename(healed_mesh))
#
#     mesh = gmsh_io.GmshIO(healed_mesh)
#     fracture_neighbors = find_fracture_neigh(mesh, ["fr"], n_levels=3)
#
#     ele_ids = np.array(list(mesh.elements.keys()), dtype=float)
#     ele_ids_map = dict()
#     for i in range(len(ele_ids)):
#         ele_ids_map[ele_ids[i]] = i
#
#     data = -1 * np.ones(shape=(len(ele_ids), 1))
#
#     for eid, lev in fracture_neighbors:
#         data[ele_ids_map[eid]] = lev
#
#     # Separate base from extension
#     mesh_name, extension = os.path.splitext(healed_mesh)
#     # Initial new name
#     new_mesh_name = os.path.join(os.curdir, mesh_name + "_data" + extension)
#
#     with open(new_mesh_name, "w") as fout:
#         mesh.write_ascii(fout)
#         mesh.write_element_data(fout, ele_ids, 'data', data)


# def test_gmsh_dfn():
#    np.random.seed()
#    fractures = generate_fractures(geometry_dict, fracture_stats)
#    factory, mesh = make_mesh(geometry_dict, fractures, "geothermal_dnf")


# @pytest.mark.skip
def test_brep_dfn():
    np.random.seed()
    fractures = generate_fractures(geometry_dict, fracture_stats)
    fr_points = make_brep(geometry_dict, fractures, "_test_fractures.brep")
   # ipps = compute_intersections(fr_points, fractures)
    #resolve_fractures_intersection(ipss)

    print('brep_test_done')

    # TODO:
    # dfn = dfn.DFN(fractures)
    # dfn_simplified = dfn.simplify()
    # brep = dfn_simplified.make_brep()


#def resolve_fractures_intersection(ipss):



def make_brep(geometry_dict, fractures: fracture.Fracture, brep_name: str):
    """
    Create the BREP file from a list of fractures using the brep writer interface.
    """
    fracture_mesh_step = geometry_dict['fracture_mesh_step']
    dimensions = geometry_dict["box_dimensions"]

    print("n fractures:", len(fractures))

    faces = []
    fr_points = []
    for i, fr in enumerate(fractures):
        ref_fr_points = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]) # polovina
        # ref_frac = fracture.SquareShape()
        frac_points = fr.transform(ref_fr_points)
        fr_points.append(frac_points.transpose())  #
        #print(frac_points.shape)
        v1 = bw.Vertex(frac_points[0,:])
        v2 = bw.Vertex(frac_points[1,:])
        v3 = bw.Vertex(frac_points[2,:])
        v4 = bw.Vertex(frac_points[3,:])
        e1 = bw.Edge([v1, v2])
        e2 = bw.Edge([v2, v3])
        e3 = bw.Edge([v3, v4])
        e4 = bw.Edge([v4, v1])
        f1 = bw.Face([e1, e2, e3, e4])
        faces.append(f1)

    comp = bw.Compound(faces)
    loc = bw.Location([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    with open(brep_name, "w") as f:
        bw.write_model(f, comp, loc)

    return fr_points

#ipps = make_vertex_list(fr_points)


def compute_intersections(fr_points,fractures: fracture.Fracture):
    surface = []
    fracs = []
    vertices = []
    #fracture_vertices = []
    vertex_id = -1
    tol = 1e-8
    n_fr = len(fr_points)
    for k in range(0,len(fractures)):
        fr_vertices = fr_points[k]
#    for fracture in fr_points:
        vert_id = []
        for i in range(0, fracture.shape[1]):
            vertex_id += 1
            vertices.append(fr_vertices[:, i])
            vert_id.append(vertex_id)
        frac_plane = FP.FracPlane(vertices,vert_id,fractures[k])
        fracs.append(frac_plane)

    n_orig_vertices = vertex_id

    p = np.array(surface).argsort()

    #ipps = []
    id = vertex_id
    for i in p:
        #for j in p: # may be reduced to relevant adepts
        #    if i == j:
        #        continue
        for j in p[i + 1:n_fr]:  # may be reduced to relevant adepts
            fi = fracs[i]
            fj = fracs[j]
            vi_x = fi.x_loc
            vi_y = fi.y_loc
            vj_x = fj.x_loc
            vj_y = fj.y_loc

            rhs1 = np.array([fj.vertices[:, 0] - fi.vertices[:, 0], fj.vertices[:, 3] - fi.vertices[:, 0]]).transpose()
            rhs2 = np.array([fj.vertices[:, 0] - fi.vertices[:, 0], fj.vertices[:, 1] - fi.vertices[:, 0]]).transpose()
            linsys1 = np.array([vi_x, vi_y, -vj_x])
            linsys2 = np.array([vi_x, vi_y, -vj_y])

            linsys = np.zeros([3, 3, 2])
            rhs = np.zeros([3, 2, 2])
            linsys[:, :, 0] = linsys1
            linsys[:, :, 1] = linsys2
            rhs[:, :, 0] = rhs1
            rhs[:, :, 1] = rhs2

            init = 0
            for i  in range(0,2):
                if lan.matrix_rank(linsys[:,:,i]) == 3:
                    x = la.solve(linsys[:, :, i], rhs[:, :, i])
                    for k in range(0, 2):
                        if ((x[:, k] >= 0).all() and (x[:, k] <= 1).all()):
                            coor = vi_x * x[0, k] + vi_y * x[1, k]
                            lcoor = np.zeros([2])
                            lcoor[1-k] = np.double(k)
                            lcoor[k] = x[2, k]

                            ids = check_duplicities(fi,fj,coor,vertices,tol)


                            if init == 0:
                                fi._initialize_new_intersection()
                                fj._initialize_new_intersection()
                                init = 1

                            if ids == -1:
                                id += 1
                                ipp = IPP.IsecFracPlanePoint(id, coor)
                                ipp.add_fracture_data(i, x[0:2, k])
                                ipp.add_fracture_data(j, lcoor)
                                vertices.append(ipp)
                                fi._add_intersection_point(id)
                                fj._add_intersection_point(id)
                            else:
                                ipp = vertices[ids]
                                ipp.add_fracture_data(i, x[0:2, k])
                                ipp.add_fracture_data(j, lcoor)
                                fi._add_intersection_point(ids)
                                fj._add_intersection_point(ids)

    return vertices, n_orig_vertices,

def check_duplicities(fi,fj,coor,vertices,tol):

    duplicity_with = -1
    duplicity_with = fi._check_duplicity(coor,tol,duplicity_with)
    duplicity_with = fj._check_duplicity(coor, tol,duplicity_with)

    for fracs in fi.isecs:
        if duplicity_with == -1:
            for ids in fracs:
                if vertices[ids].check_duplicity(coor, tol) == True:
                   duplicity_with = ids
                   break