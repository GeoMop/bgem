import numpy as np
import math
import os

from bgem.gmsh import gmsh
from bgem.gmsh import options
from bgem.gmsh import field


def shift(radius, width):
    return math.sqrt(radius * radius - 0.25 * width * width) - radius / 2


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


def make_mesh(config_dict, fractures, mesh_name, mesh_file):
    geom = config_dict["geometry"]
    fracture_mesh_step = geom['fracture_mesh_step']
    dimensions = geom["box_dimensions"]
    boundary_mesh_step = geom['boundary_mesh_step']
    boreholes_mesh_step = geom['boreholes_mesh_step']
    main_tunnel_mesh_step = geom['main_tunnel_mesh_step']

    mtr = geom["main_tunnel_radius"]
    mtw = geom["main_tunnel_width"]
    mtl = geom["main_tunnel_length"]

    st_r = geom["lateral_tunnel_radius"]
    stw = geom["lateral_tunnel_width"]
    stl = geom["lateral_tunnel_length"]

    br = geom["borehole_radius"]
    bl = geom["borehole_length"]
    bd = geom["borehole_distance"]

    factory = gmsh.GeometryOCC(mesh_name, verbose=True)
    gopt = options.Geometry()
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

    # b_box = box.get_boundary().copy()

    ###########################################################################################################

    # borehole
    b_1 = factory.cylinder(br, axis=[bl, 0, 0]).translate([stl - bl / 2, 0, 0])
    b_2 = b_1.copy().translate([0, bd, 0])
    b_3 = b_1.copy().translate([0, -bd, 0])
    borehole = factory.group(b_1, b_2, b_3)

    # edz
    edz = geom["edz_thickness"]
    b1_bigger = b_1.copy().scale((1, edz, edz))
    b2_bigger = b1_bigger.copy().translate([0, bd, 0])
    b3_bigger = b1_bigger.copy().translate([0, -bd, 0])
    bigger = factory.group(b1_bigger, b2_bigger, b3_bigger).cut(borehole.copy())
    # box = box.fragment(bigger)

    # main_tunnel = longitudinal + lateral parts
    main_tunnel_block = factory.box([mtw, mtl, mtr])
    y_shift = math.sqrt(mtr * mtr - 0.25 * mtw * mtw) - mtr / 2
    main_tunnel_block_tmp = main_tunnel_block.copy().translate([0, 0, mtr])
    main_tunnel_cylinder_tmp = factory.cylinder(mtr, axis=[0, mtl, 0]).translate([0, -mtl / 2, -y_shift])
    main_tunnel_cylinder = main_tunnel_block_tmp.intersect(main_tunnel_cylinder_tmp)

    y_shift_2 = math.sqrt(st_r * st_r - 0.25 * stw * stw) - st_r / 2
    small_tunnel_block = factory.box([stl, stw, st_r])
    small_tunnel_block_tmp = small_tunnel_block.copy().translate([0, 0, st_r])
    small_tunnel_cylinder_tmp = factory.cylinder(st_r, axis=[stl, 0, 0])\
        .translate([-stl / 2, 0, -y_shift_2])
    small_tunnel_cylinder = small_tunnel_block_tmp.intersect(small_tunnel_cylinder_tmp)

    lateral_tunnel_part_1 = factory.group(small_tunnel_block, small_tunnel_cylinder) \
        .translate([stl / 2, 0, 0])
    lateral_tunnel_part_2 = lateral_tunnel_part_1.copy().translate([0, bd, 0])
    lateral_tunnel_part_3 = lateral_tunnel_part_1.copy().translate([0, -bd, 0])

    block = main_tunnel_block.fuse(main_tunnel_cylinder, lateral_tunnel_part_1, lateral_tunnel_part_2,
                                   lateral_tunnel_part_3).translate([-bl/2, 0, 0])

    ###########################################################################################################

    b_borehole = borehole.get_boundary()
    b_block = block.get_boundary()

    fractures = create_fractures_rectangles(factory, fractures, factory.rectangle())
    fractures_group = factory.group(*fractures)

    bigger_copy = bigger.copy()

    box_drilled = box.cut(block.copy(), bigger.copy(), borehole.copy())
    fractures_group = fractures_group.intersect(box_drilled.copy())

    edz_drilled = bigger.cut(borehole.copy())

    box_fr, fractures_fr, edz_fr = factory.fragment(box_drilled, fractures_group, edz_drilled)

    b_box_fr = box_fr.get_boundary()
    b_edz_fr = edz_fr.get_boundary()
    b_borehole_fr = b_edz_fr.select_by_intersect(b_borehole).set_region(".borehole")
    b_block_fr = b_box_fr.select_by_intersect(b_block).set_region(".main_tunnel")
    edz = edz_fr.select_by_intersect(bigger_copy).set_region("edz")
    # sides2 = b_box_fr.cut(b_block_fr.copy(), b_edz_fr.copy(), b_borehole_fr.copy()).set_region(".sides")

    bb = borehole.get_boundary()
    sides2 = b_box_fr.cut(b_block_fr.copy(), b_edz_fr.copy(), bb).set_region(".sides")

    box_all = []
    # for name, side_tool in sides.items():
    #     isec = b_box_fr.select_by_intersect(side_tool)
    #     box_all.append(isec.modify_regions("." + name))

    box_all.extend([box_fr, b_borehole_fr, b_block_fr, edz, sides2])

    mesh_groups = [*box_all, fractures_fr]
    factory.keep_only(*mesh_groups)
    factory.remove_duplicate_entities()
    factory.write_brep()

    fractures_fr.mesh_step(fracture_mesh_step)
    b_borehole_fr.mesh_step(boreholes_mesh_step)
    edz.mesh_step(boreholes_mesh_step*2)
    # b_box_fr.mesh_step(boundary_mesh_step)
    sides2.mesh_step(boundary_mesh_step)
    b_block_fr.mesh_step(main_tunnel_mesh_step)

    min_el_size = 0.5
    max_el_size = 100

    mesh = options.Mesh()
    # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
    # mesh.Algorithm = options.Algorithm2d.Delaunay
    # mesh.Algorithm = options.Algorithm2d.FrontalDelaunay
    # mesh.Algorithm3D = options.Algorithm3d.Frontal
    # mesh.Algorithm3D = options.Algorithm3d.Delaunay
    mesh.ToleranceInitialDelaunay = 0.01
    # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
    mesh.CharacteristicLengthFromPoints = True
    mesh.CharacteristicLengthFromCurvature = True
    mesh.CharacteristicLengthExtendFromBoundary = 2  # co se stane if 1
    mesh.CharacteristicLengthMin = min_el_size
    mesh.CharacteristicLengthMax = max_el_size
    mesh.MinimumCirclePoints = 6
    mesh.MinimumCurvePoints = 2

    # factory.make_mesh(mesh_groups, dim=2)
    factory.make_mesh(mesh_groups)
    factory.write_mesh(format=gmsh.MeshFormat.msh2)
    os.rename(mesh_name + ".msh2", mesh_file)
    # factory.show()