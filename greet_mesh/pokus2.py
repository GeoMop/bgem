import sys
import os
import itertools

import shutil
import subprocess
import yaml
import attr
import numpy as np
import collections

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../src/bgem/gmsh'))
# sys.path.append(os.path.join(script_dir, '../../dfn/src'))


from bgem.gmsh import gmsh
from bgem.gmsh import options
from bgem.gmsh import heal_mesh
# from bgem.gmsh.gmsh import gmsh
# import gmsh_io
# import fracture


def create_box(gmsh_occ, box_geom):
    box = gmsh_occ.box(box_geom["size"])

    rot_x = float(box_geom["rot_x"])
    rot_y = float(box_geom["rot_y"])
    rot_z = float(box_geom["rot_z"])
    if rot_x != 0:
        box.rotate([1,0,0], rot_x)
    if rot_y != 0:
        box.rotate([0,1,0], rot_y)
    if rot_z != 0:
        box.rotate([0,0,1], rot_z)

    box.translate(box_geom["center"])
    # box.set_region(box_geom["name"])
    return box

def create_plane(gmsh_occ, plane_geom):
    # points = np.array(fr_geom["nodes"])

    plane = gmsh_occ.make_polygon(plane_geom["nodes"])

    return plane


def create_cylinder(gmsh_occ, cyl_geom, stretch_factor=0.005):

    radius = float(cyl_geom["radius"])
    start = np.array((cyl_geom["start"]))
    end = np.array((cyl_geom["end"]))

    dc = stretch_factor
    u = end-start
    start_t = start - dc*u
    end_t = end + dc*u

    middle = (start+end)/2
    box_lz = np.abs(end_t[2] - start_t[2]) + 2* radius
    box_lx = np.abs(end_t[0] - start_t[0]) + 2* radius
    box_ly = 0.99*np.abs(end[1] - start[1])
    box = gmsh_occ.box([box_lx, box_ly, box_lz], middle)

    cylinder = gmsh_occ.cylinder(radius, end_t-start_t, start_t)
    # cylinder_cut = cylinder.intersect(box)
    #
    # rot_x = float(cyl_geom["rot_x"])
    # rot_y = float(cyl_geom["rot_y"])
    # rot_z = float(cyl_geom["rot_z"])
    # if rot_x != 0:
    #     cylinder_cut.rotate([1, 0, 0], rot_x)
    # if rot_y != 0:
    #     cylinder_cut.rotate([0, 1, 0], rot_y)
    # if rot_z != 0:
    #     cylinder_cut.rotate([0, 0, 1], rot_z)
    #
    # cylinder_cut.translate(cyl_geom["center"])
    # cylinder_cut.set_region(cyl_geom["name"])
    return cylinder, box


def generate_mesh(geometry_dict):
    # options.Geometry.Tolerance = 0.0001
    # options.Geometry.ToleranceBoolean = 0.001
    # options.Geometry.AutoCoherence = 2
    # options.Geometry.OCCFixSmallEdges = True
    # options.Geometry.OCCFixSmallFaces = True
    # options.Geometry.OCCFixDegenerated = True
    # options.Geometry.OCCSewFaces = True
    # # #
    # options.Mesh.ToleranceInitialDelaunay = 0.01
    # options.Mesh.CharacteristicLengthMin = 0.5
    # # options.Mesh.CharacteristicLengthMax = 20
    # options.Mesh.AngleToleranceFacetOverlap = 0.8

    gen = gmsh.GeometryOCC("greet_mesh_tunnel")

    with open(os.path.join(script_dir, "geometry.yaml"), "r") as f:
        geometry_dict = yaml.safe_load(f)

    # create inner box
    box_inner = create_box(gen, geometry_dict['inner_box'])

    # create tunnel
    box_size = np.array(geometry_dict['outer_box']["size"])
    tunnel_start = np.array(geometry_dict['tunnel_1']["start"])
    tunnel_mid = np.array(geometry_dict['tunnel_1']["end"])
    tunnel_end = np.array(geometry_dict['tunnel_2']["end"])
    side_y = gen.rectangle([box_size[0], box_size[2]]).rotate([-1, 0, 0], np.pi / 2)
    tunnel_split = dict(
        start=side_y.copy().translate([0, tunnel_start[1], 0]),
        mid=side_y.copy().translate([0, tunnel_mid[1], 0]),
        end=side_y.copy().translate([0, tunnel_end[1], 0])
    )

    tunnel_1_c, tunnel_box_1 = create_cylinder(gen, geometry_dict['tunnel_1'], 0.2)
    tunnel_1_x = tunnel_1_c.copy().intersect(tunnel_box_1)

    tunnel_2_c, tunnel_box_2 = create_cylinder(gen, geometry_dict['tunnel_2'], 0.2)
    tunnel_2_x = tunnel_2_c.copy().intersect(tunnel_box_2)

    tunnel_1_s, tunnel_2_s, s1, s3 = gen.fragment(tunnel_1_c.copy(), tunnel_2_c.copy(),
                                                  tunnel_split["start"].copy(),
                                                  # tunnel_split["mid"].copy(),
                                                  tunnel_split["end"].copy())

    tunnel_1_f = tunnel_1_s.select_by_intersect(tunnel_1_x)
    tunnel_2_f = tunnel_2_s.select_by_intersect(tunnel_2_x)
    tunnel_f = [tunnel_1_f, tunnel_2_f]

    # box_inner_cut = box_inner.copy().cut(*tunnel_x)
    # box_inner_cut = box_inner.cut(tunnel)
    # box_inner_reg = box_inner.cut(tunnel)
    # box_inner_reg = box_inner.cut(tunnel_1, tunnel_2)

    # splits = []
    # split_boxes = []
    # splits.append(tunnel_split["start"])
    # t1_length_y = np.abs(tunnel_mid[1] - tunnel_start[1])
    # n_parts = 5  # number of parts of a single tunnel section
    # part_y = t1_length_y / n_parts  # y dimension of a single part
    #
    # y_split = tunnel_start[1] - part_y
    # box = gen.box([box_size[0], part_y, box_size[2]])
    # for i in range(1, n_parts):
    #     split = side_y.copy().translate([0, y_split, 0])
    #     splits.append(split)
    #     box_part = box.copy().translate([0, y_split + part_y/2, 0])
    #     split_boxes.append(box_part)
    #     y_split = y_split - part_y  # move split to the next one
    # splits.append(tunnel_split["mid"])
    # split_boxes.append(box.copy().translate([0, y_split + part_y / 2, 0]))

    # t2_length_y = np.abs(tunnel_end[1] - tunnel_mid[1])
    # part_y = t2_length_y / n_parts  # y dimension of a single part

    # y_split = tunnel_mid[1] - part_y
    # box.invalidate()
    # box = gen.box([box_size[0], part_y, box_size[2]])
    # for i in range(1, n_parts):
    #     split = side_y.copy().translate([0, y_split, 0])
    #     splits.append(split)
    #     box_part = box.copy().translate([0, y_split + part_y / 2, 0])
    #     split_boxes.append(box_part)
    #     y_split = y_split - part_y  # move split to the next one
    # splits.append(tunnel_split["end"])
    # split_boxes.append(box.copy().translate([0, y_split + part_y / 2, 0]))
    #
    # # last box outside tunnel
    # box.invalidate()
    # part_y = np.abs(tunnel_end[1] - box_size[1]/2)
    # box = gen.box([box_size[0], part_y, box_size[2]])
    # split_boxes.append(box.copy().translate([0, tunnel_end[1] - part_y / 2, 0]))

    # box_inner_cut = box_inner.copy().cut(*tunnel_x)
    # box_inner_cut = box_inner.copy().cut(*tunnel_f)
    # frag = gen.fragment(box_inner.copy(), tunnel_1_c, tunnel_2_c, *splits)
    # frag = gen.fragment(box_inner.copy(), tunnel_x, *splits)
    # frag = gen.fragment(box_inner.copy(), tunnel.copy(), *splits)
    # frag = gen.fragment(box_inner_cut, *splits)
    # frag = gen.fragment(box_inner.copy(), tunnel_1_c.copy(), tunnel_2_c.copy(), *splits)
    # frag = gen.fragment(box_inner_cut.copy(), *splits)
    # box_inner_f = frag[0]
    # tunnel = frag[1]
    # tunnel_1_f = frag[1]
    # tunnel_2_f = frag[2]

    # split the fragmented box to regions
    # box_inner_f = frag[0]
    # box_inner_reg = box_inner_f.select_by_intersect(box_inner_cut)
    # box_inner_reg = box_inner.copy().cut(*tunnel_f)
    box_inner_reg = box_inner.cut(*tunnel_f)

    # tunnel_1 = box_inner_f.select_by_intersect(tunnel_1_x)
    # tunnel_2 = box_inner_f.select_by_intersect(tunnel_2_x)
    # tunnel = [tunnel_1, tunnel_2]


    # frag = gen.fragment(box_inner.copy(), *splits)
    # box_inner_f = frag[0]

    # split the fragmented box to regions
    # box_inner_reg = box_inner_f.select_by_intersect(box_inner)
    # box_inner_reg = box_inner_f
    # box_inner_reg.set_region(geometry_dict['inner_box']["name"])


    # # make boundaries
    # print("Making boundaries...")
    # box_size = np.array(geometry_dict['outer_box']["size"])
    # side_z = gen.rectangle([box_size[0], box_size[1]])
    # side_y = gen.rectangle([box_size[0], box_size[2]])
    # side_x = gen.rectangle([box_size[2], box_size[1]])
    # sides = dict(
    #     bottom=side_z.copy().translate([0, 0, -box_size[2] / 2]),
    #     top=side_z.copy().translate([0, 0, +box_size[2] / 2]),
    #     back=side_y.copy().translate([0, 0, -box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
    #     front=side_y.copy().translate([0, 0, +box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
    #     right=side_x.copy().translate([0, 0, -box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2),
    #     left=side_x.copy().translate([0, 0, +box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    # )
    # for name, side in sides.items():
    #     side.modify_regions(name)
    #
    b_box_inner = box_inner_reg.get_boundary()
    box_all = []
    # for name, side_tool in sides.items():
    #     isec_inner = b_box_inner.select_by_intersect(side_tool)
    #     box_all.append(isec_inner.modify_regions("." + geometry_dict['inner_box']["name"] + "_" + name))
    #
    # # b_tunnel = b_box_inner.select_by_intersect(*tunnel_x)
    b_tunnel = b_box_inner.select_by_intersect(*tunnel_f)
    box_all.append(b_tunnel.modify_regions("." + geometry_dict['inner_box']["name"] + "_tunnel"))

    box_all.extend([box_inner_reg])

    # for i in range(len(split_boxes)):
    #     box_reg = box_inner_reg.select_by_intersect(split_boxes[i])
    #     box_all.append(box_reg.modify_regions(geometry_dict['inner_box']["name"] + "_" + str(i)))

    print("Making boundaries...[finished]")


    # from the outermost to innermost
    box_inner_reg.set_mesh_step(4.0)
    b_tunnel.set_mesh_step(1.0)
    # tunnel_1_f.set_mesh_step(0.5)
    # tunnel_2_f.set_mesh_step(0.5)

    # gen.synchronize()
    mesh_all = [*box_all]
    # mesh_all = [tunnel]
    # mesh_all = [*box_all, tunnel_1_f, tunnel_2_f]
    # gen.remove_duplicate_entities()

    print("Generating mesh...")
    gen.make_mesh(mesh_all)
    print("Generating mesh...[finished]")
    print("Writing mesh...")
    mesh_name = "greet_mesh.msh2"
    gen.write_mesh(mesh_name, gmsh.MeshFormat.msh2)
    print("Writing mesh...[finished]")
    # gen.show()

    # healer = heal_mesh.HealMesh.read_mesh(mesh_name)
    # healer.heal_mesh()
    # healer.write()


def generate_mesh_cut_tunnel(geometry_dict):
    # options.Geometry.Tolerance = 0.0001
    # options.Geometry.ToleranceBoolean = 0.001
    # options.Geometry.AutoCoherence = 2
    # options.Geometry.OCCFixSmallEdges = True
    # options.Geometry.OCCFixSmallFaces = True
    # options.Geometry.OCCFixDegenerated = True
    # options.Geometry.OCCSewFaces = True
    # # #
    # options.Mesh.ToleranceInitialDelaunay = 0.01
    # options.Mesh.CharacteristicLengthMin = 0.5
    # # options.Mesh.CharacteristicLengthMax = 20
    # options.Mesh.AngleToleranceFacetOverlap = 0.8

    gen = gmsh.GeometryOCC("greet_mesh_tunnel")

    with open(os.path.join(script_dir, "geometry.yaml"), "r") as f:
        geometry_dict = yaml.safe_load(f)

    # create inner box
    box_inner = create_box(gen, geometry_dict['inner_box'])

    # create tunnel
    box_size = np.array(geometry_dict['outer_box']["size"])
    tunnel_start = np.array(geometry_dict['tunnel_1']["start"])
    tunnel_mid = np.array(geometry_dict['tunnel_1']["end"])
    tunnel_end = np.array(geometry_dict['tunnel_2']["end"])
    side_y = gen.rectangle([box_size[0], box_size[2]]).rotate([-1, 0, 0], np.pi / 2)
    tunnel_split = dict(
        start=side_y.copy().translate([0, tunnel_start[1], 0]),
        mid=side_y.copy().translate([0, tunnel_mid[1], 0]),
        end=side_y.copy().translate([0, tunnel_end[1], 0])
    )

    tunnel_1_c, tunnel_box_1 = create_cylinder(gen, geometry_dict['tunnel_1'], 0.2)
    tunnel_1_x = tunnel_1_c.copy().intersect(tunnel_box_1)

    tunnel_2_c, tunnel_box_2 = create_cylinder(gen, geometry_dict['tunnel_2'], 0.2)
    tunnel_2_x = tunnel_2_c.copy().intersect(tunnel_box_2)

    tunnel_1_s, tunnel_2_s, s1, s3 = gen.fragment(tunnel_1_c.copy(), tunnel_2_c.copy(),
                                                  tunnel_split["start"].copy(),
                                                  # tunnel_split["mid"].copy(),
                                                  tunnel_split["end"].copy())


    tunnel_1 = tunnel_1_s.select_by_intersect(tunnel_1_x)
    tunnel_2 = tunnel_2_s.select_by_intersect(tunnel_2_x)

    # box_inner_cut = box_inner.copy().cut(*tunnel_x)
    # box_inner_cut = box_inner.cut(tunnel)
    # box_inner_reg = box_inner.cut(tunnel)
    box_inner_reg = box_inner.cut(tunnel_1, tunnel_2)

    # make boundaries
    print("Making boundaries...")

    b_box_inner = box_inner_reg.get_boundary()
    box_all = []
    # b_tunnel = b_box_inner.select_by_intersect(*tunnel_x)
    b_tunnel = b_box_inner.select_by_intersect(tunnel_1, tunnel_2)
    box_all.append(b_tunnel.modify_regions("." + geometry_dict['inner_box']["name"] + "_tunnel"))
    box_all.append(box_inner_reg)

    print("Making boundaries...[finished]")


    # from the outermost to innermost
    box_inner_reg.set_mesh_step(4)
    b_tunnel.set_mesh_step(1.5)
    # tunnel_1_s.set_mesh_step(1.5)
    # tunnel_2_s.set_mesh_step(1.5)
    #
    # # gen.synchronize()
    mesh_all = [*box_all]
    # mesh_all = [tunnel_1_s, tunnel_2_s]
    # mesh_all = [*box_all, tunnel_1_f, tunnel_2_f]
    # gen.remove_duplicate_entities()

    print("Generating mesh...")
    gen.make_mesh(mesh_all)
    print("Generating mesh...[finished]")
    print("Writing mesh...")
    mesh_name = "greet_mesh.msh2"
    gen.write_mesh(mesh_name, gmsh.MeshFormat.msh2)
    print("Writing mesh...[finished]")
    # gen.show()

    healer = heal_mesh.HealMesh.read_mesh(mesh_name)
    healer.heal_mesh()
    healer.write()



if __name__ == "__main__":
    sample_dir = sys.argv[1]
    with open(os.path.join(script_dir, "geometry.yaml"), "r") as f:
        geometry_dict = yaml.safe_load(f)

    os.chdir(sample_dir)
    print("generate mesh")
    generate_mesh(geometry_dict)
    # generate_mesh_cut_tunnel(geometry_dict)
    print("finished")