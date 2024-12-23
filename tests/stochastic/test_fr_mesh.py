import os
import numpy as np
from bgem.stochastic import fr_set
from bgem.gmsh import gmsh, options as gmsh_options, field as gmsh_field


def make_mesh(geometry_dict, fractures: fr_set.Fracture, mesh_name: str):
    """
    Create the GMSH mesh from a list of fractures using the bgem.gmsh interface.
    """
    fracture_mesh_step = geometry_dict['fracture_mesh_step']
    dimensions = geometry_dict["box_dimensions"]
    well_z0, well_z1 = geometry_dict["well_opening"]
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
    np.random.seed(123)
    fractures = generate_uniform(fracture_stats, n_frac_limit=50)
    for i, f in enumerate(fractures):
        f.id = i
    make_brep(geometry_dict, fractures, sandbox_fname("test_dfn", "brep"))

    ipps = compute_intersections(fractures)
    # resolve_fractures_intersection(ipss)

    print('brep_test_done')

    # TODO:
    # dfn = dfn.DFN(fractures)
    # dfn_simplified = dfn.simplify()
    # brep = dfn_simplified.make_brep()

# def resolve_fractures_intersection(ipss):
