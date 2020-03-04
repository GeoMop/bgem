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


# def create_volume(nodes):
#     """
#     Creates box with lower and upper base and sides.
#     :param nodes: Nodes must be ordered: [lower_base_nodes, upper_base_nodes]
#     :return:
#     """
#     n = int(len(nodes)/2)
#     p_tags = []
#     for p in nodes:
#         p_tags.append(gmsh.model.occ.addPoint(p[0], p[1], p[2]))
#
#     l_tags = []
#     for i in range(0, n-1):
#         l_tags.append(gmsh.model.occ.addLine(p_tags[i], p_tags[i+1]))
#     l_tags.append(gmsh.model.occ.addLine(p_tags[n-1], p_tags[0]))
#     for i in range(n, 2*n-1):
#         l_tags.append(gmsh.model.occ.addLine(p_tags[i], p_tags[i+1]))
#     l_tags.append(gmsh.model.occ.addLine(p_tags[2*n-1], p_tags[n]))
#     for i in range(0, n):
#         l_tags.append(gmsh.model.occ.addLine(p_tags[i], p_tags[n+i]))
#
#     b = [l_tags[i] for i in range(0, n)]
#     loop_down = gmsh.model.occ.addCurveLoop(b)
#     base_down = gmsh.model.occ.addPlaneSurface([loop_down])
#
#     b = [l_tags[i] for i in range(n, 2*n)]
#     loop_up = gmsh.model.occ.addCurveLoop(b)
#     base_up = gmsh.model.occ.addPlaneSurface([loop_up])
#     sides = []
#     for i in range(0, n-1):
#         lines = [l_tags[i], l_tags[2*n+i+1], -l_tags[n+i], -l_tags[2*n+i]]
#         loop_side = gmsh.model.occ.addCurveLoop(lines)
#         sides.append(gmsh.model.occ.addPlaneSurface([loop_side]))
#     lines = [l_tags[n-1], l_tags[2*n], -l_tags[2*n-1], -l_tags[3*n-1]]
#     loop_side = gmsh.model.occ.addCurveLoop(lines)
#     sides.append(gmsh.model.occ.addPlaneSurface([loop_side]))
#
#     plane_loop = gmsh.model.occ.addSurfaceLoop([base_down, base_up, *sides])
#     volume = gmsh.model.occ.addVolume([plane_loop])
#     return 3, volume

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
    box_ly = np.abs(end[1] - start[1])
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

    options.Geometry.Tolerance = 0.001
    options.Geometry.ToleranceBoolean = 0.01
    options.Geometry.AutoCoherence = 2
    options.Geometry.OCCFixSmallEdges = True
    options.Geometry.OCCFixSmallFaces = True
    options.Geometry.OCCFixDegenerated = True
    options.Geometry.OCCSewFaces = True
    #
    options.Mesh.ToleranceInitialDelaunay = 0.01
    options.Mesh.CharacteristicLengthMin = 0.5
    options.Mesh.CharacteristicLengthMax = 20
    options.Mesh.AngleToleranceFacetOverlap = 0.1

    gen = gmsh.GeometryOCC("greet_mesh")

    with open(os.path.join(script_dir, "geometry.yaml"), "r") as f:
        geometry_dict = yaml.safe_load(f)

    # compute barycenter of the given points to translate the box
    outer_box_points = np.array(geometry_dict['outer_box']["nodes"])
    barycenter = np.array([0,0,0])
    # barycenter = [np.average(outer_box_points[:, 0]), np.average(outer_box_points[:, 1]), np.average(outer_box_points[:, 2])]

    # create outer box
    geometry_dict['outer_box']["center"] = barycenter
    box_outer = create_box(gen, geometry_dict['outer_box'])

    # create inner box
    geometry_dict['inner_box']["center"] = barycenter
    box_inner = create_box(gen, geometry_dict['inner_box'])

    # create cut outer_box object for setting the correct region
    box_outer_cut = box_outer.copy().cut(box_inner)

    # # create fracture cut box
    # geometry_dict['cut_fracture_box']["center"] = barycenter
    # cut_fracture_box = create_box(gen, geometry_dict['cut_fracture_box'])

    # create tunnel
    geometry_dict['tunnel_1']["center"] = barycenter
    tunnel_1_c, tunnel_box_1 = create_cylinder(gen, geometry_dict['tunnel_1'])
    tunnel_1_x = tunnel_1_c.cut(tunnel_box_1)

    geometry_dict['tunnel_2']["center"] = barycenter
    tunnel_2_c, tunnel_box_2 = create_cylinder(gen, geometry_dict['tunnel_2'])
    tunnel_2_x = tunnel_2_c.cut(tunnel_box_2)

    # tunnel = [tunnel_1]
    # tunnel = tunnel_1
    # tunnel = tunnel_1.fuse(tunnel_2)
    tunnel_1_t, tunnel_2_t, b1, b2 = gen.fragment(tunnel_1_x.copy(), tunnel_2_x.copy(), tunnel_box_1.copy(), tunnel_box_2.copy())
    tunnel_1_t.remove_small_mass(0.01)
    tunnel_2_t.remove_small_mass(0.01)

    tunnel = [tunnel_1_t, tunnel_2_t]

    # tunnel.remove_small_mass(0.01)

    # create cut inner_box object for setting the correct region
    box_inner_cut = box_inner.copy().cut(*tunnel)
    # box_inner_cut.invalidate()

    # create fractures
    # fractures = []
    # for f in geometry_dict['fractures']:
    #     fract = create_plane(gen, f)
    #     fract.set_region(f["name"])
    #     fractures.append(fract)

    gen.synchronize()

    # cut fractures
    # fractures_cut = []
    # for f in fractures:
    #     fractures_cut.append(f.intersect(cut_fracture_box))

    # connect tunnels and split them to sections


    # gen.synchronize()

    # tunnel_1.set_region("tunnel_1")
    # tunnel_2.set_region("tunnel_2")

    # do fragmentation
    # all = gen.group([fract1, box_outer])
    # all_obj = [box_outer, box_inner]
    all_obj = [box_outer, box_inner, *tunnel]
    frag_all = gen.fragment(*all_obj)
    # box_outer_f, box_inner_f = frag_all
    box_outer_f, box_inner_f, tunnel_1_f, tunnel_2_f = frag_all
    # box_outer_f, box_inner_f, tunnel_f = frag_all

    # split the fragmented box to regions
    box_inner_reg = box_outer_f.select_by_intersect(box_inner_cut)
    box_inner_reg.set_region(geometry_dict['inner_box']["name"])
    box_outer_reg = box_outer_f.select_by_intersect(box_outer_cut)
    box_outer_reg.set_region(geometry_dict['outer_box']["name"])
    box_inner_f.invalidate()

    # make boundaries
    print("Making boundaries...")
    box_size = np.array(geometry_dict['outer_box']["size"])
    side_z = gen.rectangle([box_size[0], box_size[1]])
    side_y = gen.rectangle([box_size[0], box_size[2]])
    side_x = gen.rectangle([box_size[2], box_size[1]])
    sides = dict(
        bottom=side_z.copy().translate([0, 0, -box_size[2] / 2]),
        top=side_z.copy().translate([0, 0, +box_size[2] / 2]),
        back=side_y.copy().translate([0, 0, -box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        front=side_y.copy().translate([0, 0, +box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        right=side_x.copy().translate([0, 0, -box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        left=side_x.copy().translate([0, 0, +box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.modify_regions(name)

    b_box_outer = box_outer_reg.get_boundary()
    b_box_inner = box_inner_reg.get_boundary()
    box_all = []
    for name, side_tool in sides.items():
        isec_outer = b_box_outer.select_by_intersect(side_tool)
        isec_inner = b_box_inner.select_by_intersect(side_tool)
        box_all.append(isec_outer.modify_regions("." + geometry_dict['outer_box']["name"] + "_" + name))
        box_all.append(isec_inner.modify_regions("." + geometry_dict['inner_box']["name"] + "_" + name))

    b_tunnel = b_box_inner.select_by_intersect(tunnel_1_f, tunnel_2_f)
    box_all.append(b_tunnel.modify_regions("." + geometry_dict['inner_box']["name"] + "_tunnel"))

    box_all.extend([box_outer_reg, box_inner_reg])
    print("Making boundaries...[finished]")

    # from the outermost to innermost
    box_outer_reg.set_mesh_step(12)
    box_inner_reg.set_mesh_step(4)
    tunnel_1_f.set_mesh_step(1.5)
    tunnel_2_f.set_mesh_step(1.5)
    # for f in frag_all:
    #     f.set_mesh_step(4)

    # gen.synchronize()
    mesh_all = [*box_all]
    # mesh_all = [*box_all, tunnel_1_f, tunnel_2_f]
    # mesh_all = [box_outer_f, *box_all, box_inner_f, tunnel_1_f, tunnel_2_f]
    # tunnel_1_f.set_region("tunnel_1")
    # box_inner_f.set_region("rock_inner")
    # box_outer_f.set_region("rock_outer")

    # gen.remove_duplicate_entities()

    print("Generating mesh...")
    gen.make_mesh(mesh_all)
    print("Generating mesh...[finished]")
    print("Writing mesh...")
    gen.write_mesh("greet_mesh.msh2", gmsh.MeshFormat.msh2)
    print("Writing mesh...[finished]")
    # gen.show()


def generate_mesh_2(geometry_dict):

    gen = gmsh.GeometryOCC("greet_mesh")

    with open(os.path.join(script_dir, "geometry.yaml"), "r") as f:
        geometry_dict = yaml.safe_load(f)

    # compute barycenter of the given points to translate the box
    outer_box_points = np.array(geometry_dict['outer_box']["nodes"])
    barycenter = np.array([0,0,0])
    # barycenter = [np.average(outer_box_points[:, 0]), np.average(outer_box_points[:, 1]), np.average(outer_box_points[:, 2])]

    # create inner box
    geometry_dict['inner_box']["center"] = barycenter
    box_inner = create_box(gen, geometry_dict['inner_box'])

    # create outer box
    geometry_dict['outer_box']["center"] = barycenter
    box_outer = create_box(gen, geometry_dict['outer_box'])

    # create cut outer_box object for setting the correct region
    box_outer_cp = box_outer.copy()
    box_outer_cut = box_outer_cp.cut(box_inner)

    gen.synchronize()

    # do fragmentation
    # all = gen.group([fract1, box_outer])
    all_obj = [box_outer, box_inner]
    # all = [*fractures_cut, tunnel_1, box_outer, box_inner]
    # all = [*fractures_cut, tunnel_1_cut, box_outer, box_inner]
    frag_all = gen.fragment(*all_obj)
    box_outer_f, box_inner_f = frag_all

    # split the fragmented box to regions
    box_inner_reg = box_outer_f.select_by_intersect(box_inner_f)
    box_inner_reg.set_region(geometry_dict['inner_box']["name"])
    box_outer_reg = box_outer_f.select_by_intersect(box_outer_cut)
    box_outer_reg.set_region(geometry_dict['outer_box']["name"])
    box_inner_f.invalidate()

    # make boundaries
    box_size = np.array(geometry_dict['outer_box']["size"])
    side_z = gen.rectangle([box_size[0], box_size[1]])
    side_y = gen.rectangle([box_size[0], box_size[2]])
    side_x = gen.rectangle([box_size[2], box_size[1]])
    sides = dict(
        bottom=side_z.copy().translate([0, 0, -box_size[2] / 2]),
        top=side_z.copy().translate([0, 0, +box_size[2] / 2]),
        back=side_y.copy().translate([0, 0, -box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        front=side_y.copy().translate([0, 0, +box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        right=side_x.copy().translate([0, 0, -box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        left=side_x.copy().translate([0, 0, +box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.modify_regions(name)

    b_box_outer = box_outer_reg.get_boundary()
    b_box_inner = box_inner_reg.get_boundary()
    box_all = []
    for name, side_tool in sides.items():
        isec_outer = b_box_outer.select_by_intersect(side_tool)
        isec_inner = b_box_inner.select_by_intersect(side_tool)
        box_all.append(isec_outer.modify_regions("." + geometry_dict['outer_box']["name"] + "_" + name))
        box_all.append(isec_inner.modify_regions("." + geometry_dict['inner_box']["name"] + "_" + name))

    box_all.extend([box_outer_reg, box_inner_reg])

    # from the outermost to innermost
    # box_outer_f.set_mesh_step(12)
    # box_inner_f.set_mesh_step(4)
    # for f in frag_all:
    #     f.set_mesh_step(4)

    # gen.synchronize()
    # mesh_all = [*frag_all]
    mesh_all = [*box_all]
    # mesh_all = [box_outer_reg, box_inner_reg]
    # mesh_all = [box_outer_f, box_inner_f]
    # mesh_all = [box_outer_f, *box_all, box_inner_f, tunnel_1_f, tunnel_2_f]
    # tunnel_1_f.set_region("tunnel_1")
    # box_inner_f.set_region("rock_inner")
    # box_outer_f.set_region("rock_outer")

    # gen.make_mesh([tunnel_1], 3)
    gen.make_mesh(mesh_all)
    gen.write_mesh("greet_mesh.msh2", gmsh.MeshFormat.msh2)
    # gen.show()



def generate_mesh_tunnel(geometry_dict):
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

    # create tunnel
    tunnel_1_c, tunnel_box_1 = create_cylinder(gen, geometry_dict['tunnel_1'], 0.2)
    tunnel_1_x = tunnel_1_c.copy().intersect(tunnel_box_1)

    tunnel_2_c, tunnel_box_2 = create_cylinder(gen, geometry_dict['tunnel_2'], 0.00)
    tunnel_2_x = tunnel_2_c.copy().intersect(tunnel_box_2)

    # tunnel = [tunnel_1_x, tunnel_2_x]

    # tunnel = [tunnel_1_x]
    # tunnel = tunnel_1
    # tunnel = tunnel_1_x.fuse(tunnel_2_x)

    # tunnel_fuse = tunnel_1_c.copy().fuse(tunnel_2_c)
    tunnel_fuse = tunnel_1_x.copy().fuse(tunnel_2_c)
    # tunnel_f, tunnel_1_t, b1, b2 = gen.fragment(tunnel_fuse.copy(), tunnel_1_x.copy(), tunnel_box_1.copy(), tunnel_box_2.copy())
    # tunnel_1_t.remove_small_mass(0.01)
    # tunnel_2_t.remove_small_mass(0.01)
    # tunnel = [tunnel_1_t, tunnel_2_t]

    # tunnel = tunnel_fuse.intersect(tunnel_1_x,tunnel_2_x)
    tunnel = tunnel_fuse
    # tunnel_i = tunnel_fuse.intersect(tunnel_1_x, tunnel_2_x)
    # tunnel_xx = gen.healShapes(tunnel_i)
    # tunnel = tunnel_xx[0]
    # tunnel.set_region("tunnel")
    tunnel.set_mesh_step(1.0)
    # tunnel.remove_small_mass(0.1)

    # gen.synchronize()
    mesh_all = [tunnel]
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

def generate_mesh_tunnel_2(geometry_dict):
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
    tunnel_1_c, tunnel_box_1 = create_cylinder(gen, geometry_dict['tunnel_1'], 0.2)
    tunnel_1_x = tunnel_1_c.copy().intersect(tunnel_box_1)

    tunnel_2_c, tunnel_box_2 = create_cylinder(gen, geometry_dict['tunnel_2'], 0.0000)
    # tunnel_2_x = tunnel_2_c.copy().intersect(tunnel_box_2)

    tunnel = [tunnel_1_x, tunnel_2_c]

    # tunnel = [tunnel_1_x]
    # tunnel = tunnel_1
    # tunnel = tunnel_1_x.fuse(tunnel_2_x)

    # tunnel_fuse = tunnel_1_c.copy().fuse(tunnel_2_c)
    # tunnel_fuse = tunnel_1_x.copy().fuse(tunnel_2_c)
    # tunnel_f, tunnel_1_t, b1, b2 = gen.fragment(tunnel_fuse.copy(), tunnel_1_x.copy(), tunnel_box_1.copy(), tunnel_box_2.copy())
    # tunnel_1_t.remove_small_mass(0.01)
    # tunnel_2_t.remove_small_mass(0.01)
    # tunnel = [tunnel_1_t, tunnel_2_t]

    # tunnel = tunnel_fuse.intersect(tunnel_1_x,tunnel_2_x)
    # tunnel = tunnel_fuse
    # tunnel_i = tunnel_fuse.intersect(tunnel_1_x, tunnel_2_x)
    # tunnel_xx = gen.healShapes(tunnel_i)
    # tunnel = tunnel_xx[0]
    # tunnel.set_region("tunnel")
    # tunnel.set_mesh_step(0.5)
    # tunnel.remove_small_mass(0.1)

    # create cut inner_box object for setting the correct region
    # box_inner_cut = box_inner.cut(*tunnel)
    # box_inner_cut.invalidate()

    # gen.synchronize()

    # split the fragmented box to regions
    # box_inner_reg = box_inner.select_by_intersect(box_inner_cut)

    box_inner_reg = box_inner.cut(*tunnel)
    box_inner_reg.set_region(geometry_dict['inner_box']["name"])

    # make boundaries
    print("Making boundaries...")
    box_size = np.array(geometry_dict['outer_box']["size"])
    side_z = gen.rectangle([box_size[0], box_size[1]])
    side_y = gen.rectangle([box_size[0], box_size[2]])
    side_x = gen.rectangle([box_size[2], box_size[1]])
    sides = dict(
        bottom=side_z.copy().translate([0, 0, -box_size[2] / 2]),
        top=side_z.copy().translate([0, 0, +box_size[2] / 2]),
        back=side_y.copy().translate([0, 0, -box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        front=side_y.copy().translate([0, 0, +box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        right=side_x.copy().translate([0, 0, -box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        left=side_x.copy().translate([0, 0, +box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.modify_regions(name)

    b_box_inner = box_inner_reg.get_boundary()
    box_all = []
    for name, side_tool in sides.items():
        isec_inner = b_box_inner.select_by_intersect(side_tool)
        box_all.append(isec_inner.modify_regions("." + geometry_dict['inner_box']["name"] + "_" + name))

    b_tunnel = b_box_inner.select_by_intersect(*tunnel)
    box_all.append(b_tunnel.modify_regions("." + geometry_dict['inner_box']["name"] + "_tunnel"))

    box_all.extend([box_inner_reg])
    print("Making boundaries...[finished]")


    # from the outermost to innermost
    box_inner_reg.set_mesh_step(4)
    b_tunnel.set_mesh_step(1.0)
    # tunnel_1_x.set_mesh_step(1.5)
    # tunnel_2_x.set_mesh_step(1.5)

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

    healer = heal_mesh.HealMesh.read_mesh(mesh_name)
    healer.heal_mesh()
    healer.write()

def generate_mesh_tunnel_3(geometry_dict):
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
    side_y = gen.rectangle([box_size[0], box_size[2]])
    tunnel_split = dict(
        start=side_y.copy().translate([0, 0, tunnel_start[1]]).rotate([-1, 0, 0], np.pi / 2),
        mid=side_y.copy().translate([0, 0, tunnel_mid[1]]).rotate([-1, 0, 0], np.pi / 2),
        end=side_y.copy().translate([0, 0, tunnel_end[1]]).rotate([-1, 0, 0], np.pi / 2)
    )

    tunnel_1_c, tunnel_box_1 = create_cylinder(gen, geometry_dict['tunnel_1'], 0.2)
    tunnel_1_x = tunnel_1_c.copy().intersect(tunnel_box_1)
    tunnel_1_s, s1, s2 = gen.fragment(tunnel_1_c, tunnel_split["start"].copy(), tunnel_split["mid"].copy())
    tunnel_1 = tunnel_1_s.select_by_intersect(tunnel_1_x)

    tunnel_2_c, tunnel_box_2 = create_cylinder(gen, geometry_dict['tunnel_2'], 0.2)
    tunnel_2_x = tunnel_2_c.copy().intersect(tunnel_box_2)
    tunnel_2_s, s1, s2 = gen.fragment(tunnel_2_c, tunnel_split["mid"].copy(), tunnel_split["end"].copy())
    tunnel_2 = tunnel_2_s.select_by_intersect(tunnel_2_x)

    tunnel = [tunnel_1, tunnel_2]

    # tunnel = [tunnel_1_x]
    # tunnel = tunnel_1
    # tunnel = tunnel_1_x.fuse(tunnel_2_x)

    # tunnel_fuse = tunnel_1_c.copy().fuse(tunnel_2_c)
    # tunnel_fuse = tunnel_1_x.copy().fuse(tunnel_2_c)
    # tunnel_f, tunnel_1_t, b1, b2 = gen.fragment(tunnel_fuse.copy(), tunnel_1_x.copy(), tunnel_box_1.copy(), tunnel_box_2.copy())
    # tunnel_1_t.remove_small_mass(0.01)
    # tunnel_2_t.remove_small_mass(0.01)
    # tunnel = [tunnel_1_t, tunnel_2_t]

    # tunnel = tunnel_fuse.intersect(tunnel_1_x,tunnel_2_x)
    # tunnel = tunnel_fuse
    # tunnel_i = tunnel_fuse.intersect(tunnel_1_x, tunnel_2_x)
    # tunnel_xx = gen.healShapes(tunnel_i)
    # tunnel = tunnel_xx[0]
    # tunnel.set_region("tunnel")
    # tunnel.set_mesh_step(0.5)
    # tunnel.remove_small_mass(0.1)

    # create cut inner_box object for setting the correct region
    # box_inner_cut = box_inner.cut(*tunnel)
    # box_inner_cut.invalidate()

    # gen.synchronize()

    # split the fragmented box to regions
    # box_inner_reg = box_inner.select_by_intersect(box_inner_cut)

    box_inner_reg = box_inner.cut(*tunnel)
    box_inner_reg.set_region(geometry_dict['inner_box']["name"])

    # make boundaries
    print("Making boundaries...")
    box_size = np.array(geometry_dict['outer_box']["size"])
    side_z = gen.rectangle([box_size[0], box_size[1]])
    side_y = gen.rectangle([box_size[0], box_size[2]])
    side_x = gen.rectangle([box_size[2], box_size[1]])
    sides = dict(
        bottom=side_z.copy().translate([0, 0, -box_size[2] / 2]),
        top=side_z.copy().translate([0, 0, +box_size[2] / 2]),
        back=side_y.copy().translate([0, 0, -box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        front=side_y.copy().translate([0, 0, +box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        right=side_x.copy().translate([0, 0, -box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        left=side_x.copy().translate([0, 0, +box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.modify_regions(name)

    b_box_inner = box_inner_reg.get_boundary()
    box_all = []
    for name, side_tool in sides.items():
        isec_inner = b_box_inner.select_by_intersect(side_tool)
        box_all.append(isec_inner.modify_regions("." + geometry_dict['inner_box']["name"] + "_" + name))

    b_tunnel = b_box_inner.select_by_intersect(*tunnel)
    box_all.append(b_tunnel.modify_regions("." + geometry_dict['inner_box']["name"] + "_tunnel"))

    box_all.extend([box_inner_reg])
    print("Making boundaries...[finished]")


    # from the outermost to innermost
    box_inner_reg.set_mesh_step(4)
    b_tunnel.set_mesh_step(1.0)
    # tunnel_1.set_mesh_step(1.5)
    # tunnel_2.set_mesh_step(1.5)

    # gen.synchronize()
    mesh_all = [*box_all]
    # mesh_all = [*tunnel]
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
    healer.heal_mesh(gamma_tol=0.001)
    healer.write()

def generate_mesh_tunnel_4(geometry_dict):
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
    tunnel_1_s, s1, s2 = gen.fragment(tunnel_1_c.copy(), tunnel_split["start"].copy(), tunnel_split["mid"].copy())
    tunnel_1 = tunnel_1_s.select_by_intersect(tunnel_1_x)

    tunnel_2_c, tunnel_box_2 = create_cylinder(gen, geometry_dict['tunnel_2'], 0.2)
    tunnel_2_x = tunnel_2_c.copy().intersect(tunnel_box_2)
    tunnel_2_s, s1, s2 = gen.fragment(tunnel_2_c.copy(), tunnel_split["mid"].copy(), tunnel_split["end"].copy())
    tunnel_2 = tunnel_2_s.select_by_intersect(tunnel_2_x)

    tunnel_x = [tunnel_1, tunnel_2]

    splits = []
    split_boxes = []
    splits.append(tunnel_split["start"])
    t1_length_y = np.abs(tunnel_mid[1] - tunnel_start[1])
    n_parts = 5  # number of parts of a single tunnel section
    part_y = t1_length_y / n_parts  # y dimension of a single part

    y_split = tunnel_start[1] - part_y
    box = gen.box([box_size[0], part_y, box_size[2]])
    for i in range(1, n_parts):
        split = side_y.copy().translate([0, y_split, 0])
        splits.append(split)
        box_part = box.copy().translate([0, y_split + part_y/2, 0])
        split_boxes.append(box_part)
        y_split = y_split - part_y  # move split to the next one
    splits.append(tunnel_split["mid"])
    split_boxes.append(box.copy().translate([0, y_split + part_y / 2, 0]))

    t2_length_y = np.abs(tunnel_end[1] - tunnel_mid[1])
    part_y = t2_length_y / n_parts  # y dimension of a single part

    y_split = tunnel_mid[1] - part_y
    box.invalidate()
    box = gen.box([box_size[0], part_y, box_size[2]])
    for i in range(1, n_parts):
        split = side_y.copy().translate([0, y_split, 0])
        splits.append(split)
        box_part = box.copy().translate([0, y_split + part_y / 2, 0])
        split_boxes.append(box_part)
        y_split = y_split - part_y  # move split to the next one
    splits.append(tunnel_split["end"])
    split_boxes.append(box.copy().translate([0, y_split + part_y / 2, 0]))

    # last box outside tunnel
    box.invalidate()
    part_y = np.abs(tunnel_end[1] - box_size[1]/2)
    box = gen.box([box_size[0], part_y, box_size[2]])
    split_boxes.append(box.copy().translate([0, tunnel_end[1] - part_y / 2, 0]))

    box_inner_cut = box_inner.copy().cut(*tunnel_x)
    # frag = gen.fragment(box_inner.copy(), tunnel_1_c, tunnel_2_c, *splits)
    # frag = gen.fragment(box_inner.copy(), tunnel_x, *splits)
    frag = gen.fragment(box_inner.copy().cut(*tunnel_x), tunnel_1.copy(), tunnel_2.copy(), *splits)
    # frag = gen.fragment(box_inner_cut.copy(), *splits)
    box_inner_f = frag[0]
    # tunnel = frag[1]
    # tunnel_1_f = frag[1]
    # tunnel_2_f = frag[2]

    # split the fragmented box to regions
    # box_inner_cut = box_inner.cut(tunnel_1_x, tunnel_2_x)
    # box_inner_reg = box_inner_f.select_by_intersect(box_inner_cut)
    box_inner_reg = box_inner_f
    # tunnel_1 = box_inner_f.select_by_intersect(tunnel_1_x)
    # tunnel_2 = box_inner_f.select_by_intersect(tunnel_2_x)

    # frag = gen.fragment(box_inner.copy(), *splits)
    # box_inner_f = frag[0]

    # split the fragmented box to regions
    # box_inner_reg = box_inner_f.select_by_intersect(box_inner)
    # box_inner_reg = box_inner_f
    # box_inner_reg.set_region(geometry_dict['inner_box']["name"])


    # make boundaries
    print("Making boundaries...")
    box_size = np.array(geometry_dict['outer_box']["size"])
    side_z = gen.rectangle([box_size[0], box_size[1]])
    side_y = gen.rectangle([box_size[0], box_size[2]])
    side_x = gen.rectangle([box_size[2], box_size[1]])
    sides = dict(
        bottom=side_z.copy().translate([0, 0, -box_size[2] / 2]),
        top=side_z.copy().translate([0, 0, +box_size[2] / 2]),
        back=side_y.copy().translate([0, 0, -box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        front=side_y.copy().translate([0, 0, +box_size[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        right=side_x.copy().translate([0, 0, -box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        left=side_x.copy().translate([0, 0, +box_size[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.modify_regions(name)

    b_box_inner = box_inner_reg.get_boundary()
    box_all = []
    for name, side_tool in sides.items():
        isec_inner = b_box_inner.select_by_intersect(side_tool)
        box_all.append(isec_inner.modify_regions("." + geometry_dict['inner_box']["name"] + "_" + name))

    b_tunnel = b_box_inner.select_by_intersect(*tunnel_x)
    box_all.append(b_tunnel.modify_regions("." + geometry_dict['inner_box']["name"] + "_tunnel"))

    # box_all.extend([box_inner_reg])

    for i in range(len(split_boxes)):
        box_reg = box_inner_reg.select_by_intersect(split_boxes[i])
        box_all.append(box_reg.modify_regions(geometry_dict['inner_box']["name"] + "_" + str(i)))

    print("Making boundaries...[finished]")


    # from the outermost to innermost
    box_inner_reg.set_mesh_step(4)
    b_tunnel.set_mesh_step(1.5)
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

    healer = heal_mesh.HealMesh.read_mesh(mesh_name)
    healer.heal_mesh()
    healer.write()

def generate_mesh_fuse(geometry_dict):
    gen = gmsh.GeometryOCC("greet_mesh")

    # create inner box
    box_1 = gen.box([20, 20, 20])
    box_2 = gen.box([10, 10, 40])

    # all_obj = gen.fuse(box_1, box_2)
    # all_obj = gen.fragment(box_1, box_2)
    # all_obj = [box_1, box_2]

    # box_fused = box_1.fragment(box_2)
    # box_fused = box_1.fuse(box_2)
    box_fused = box_2.fuse(box_1)
    # box_1.invalidate()
    # box_2.invalidate()
    box_fused.set_region("box")
    # box_fused.set_mesh_step(1)
    all_obj = [box_fused]

    gen.synchronize()

    # gen.synchronize()
    # mesh_all = [*frag_all]
    mesh_all = [*all_obj]

    gen.make_mesh(mesh_all)
    gen.write_mesh("greet_mesh.msh2", gmsh.MeshFormat.msh2)
    # gen.show()

def to_polar(x, y, z):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    if z > 0:
        phi += np.pi
    return phi, rho


# def plot_fr_orientation(fractures):
#     family_dict = collections.defaultdict(list)
#     for fr in fractures:
#         x, y, z = fracture.FisherOrientation.rotate(np.array([0,0,1]), axis=fr.rotation_axis, angle=fr.rotation_angle)[0]
#         family_dict[fr.region].append([
#             to_polar(z, y, x),
#             to_polar(z, x, -y),
#             to_polar(y, x, z)
#             ])
#
#     import matplotlib.pyplot as plt
#     fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
#     for name, data in family_dict.items():
#         # data shape = (N, 3, 2)
#         data = np.array(data)
#         for i, ax in enumerate(axes):
#             phi = data[:, i, 0]
#             r = data[:, i, 1]
#             c = ax.scatter(phi, r, cmap='hsv', alpha=0.75, label=name)
#     axes[0].set_title("X-view, Z-north")
#     axes[1].set_title("Y-view, Z-north")
#     axes[2].set_title("Z-view, Y-north")
#     for ax in axes:
#         ax.set_theta_zero_location("N")
#         ax.set_theta_direction(-1)
#         ax.set_ylim(0, 1)
#     fig.legend(loc = 1)
#     fig.savefig("fracture_orientation.pdf")
#     plt.close(fig)
#     #plt.show()
#
#
# def generate_fractures(config_dict):
#     geom = config_dict["geometry"]
#     dimensions = geom["box_dimensions"]
#     well_z0, well_z1 = geom["well_openning"]
#     well_length = well_z1 - well_z0
#     well_r = geom["well_effective_radius"]
#     well_dist = geom["well_distance"]
#
#     # generate fracture set
#     fracture_box = [1.5 * well_dist, 1.5 * well_length, 1.5 * well_length]
#     volume = np.product(fracture_box)
#     pop = fracture.Population(volume)
#     pop.initialize(geom["fracture_stats"])
#     pop.set_sample_range([1, well_dist], max_sample_size=geom["n_frac_limit"])
#     print("total mean size: ", pop.mean_size())
#     connected_position = geom.get('connected_position_distr', False)
#     if connected_position:
#         eps = well_r / 2
#         left_well_box = [-well_dist/2-eps, -eps, well_z0, -well_dist/2+eps, +eps, well_z1]
#         right_well_box = [well_dist/2-eps, -eps, well_z0, well_dist/2+eps, +eps, well_z1]
#         pos_gen = fracture.ConnectedPosition(
#             confining_box=fracture_box,
#             init_boxes=[left_well_box, right_well_box])
#     else:
#         pos_gen = fracture.UniformBoxPosition(fracture_box)
#     fractures = pop.sample(pos_distr=pos_gen, keep_nonempty=True)
#     #fracture.fr_intersect(fractures)
#
#     for fr in fractures:
#         fr.region = "fr"
#     used_families = set((f.region for f in fractures))
#     for model in ["hm_params", "th_params", "th_params_ref"]:
#         model_dict = config_dict[model]
#         model_dict["fracture_regions"] = list(used_families)
#         model_dict["left_well_fracture_regions"] = [".{}_left_well".format(f) for f in used_families]
#         model_dict["right_well_fracture_regions"] = [".{}_right_well".format(f) for f in used_families]
#     return fractures
#
#
# def create_fractures_rectangles(gmsh_geom, fractures, base_shape: 'ObjectSet'):
#     # From given fracture date list 'fractures'.
#     # transform the base_shape to fracture objects
#     # fragment fractures by their intersections
#     # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
#     shapes = []
#     for i, fr in enumerate(fractures):
#         shape = base_shape.copy()
#         print("fr: ", i, "tag: ", shape.dim_tags)
#         shape = shape.scale([fr.rx, fr.ry, 1]) \
#             .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
#             .translate(fr.centre) \
#             .set_region(fr.region)
#
#         shapes.append(shape)
#
#     fracture_fragments = gmsh_geom.fragment(*shapes)
#     return fracture_fragments
#
#
# def create_fractures_polygons(gmsh_geom, fractures):
#     # From given fracture date list 'fractures'.
#     # transform the base_shape to fracture objects
#     # fragment fractures by their intersections
#     # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
#     frac_obj = fracture.Fractures(fractures)
#     frac_obj.snap_vertices_and_edges()
#     shapes = []
#     for fr, square in zip(fractures, frac_obj.squares):
#         shape = gmsh_geom.make_polygon(square).set_region(fr.region)
#         shapes.append(shape)
#
#     fracture_fragments = gmsh_geom.fragment(*shapes)
#     return fracture_fragments


# def make_mesh(config_dict, fractures, mesh_name, mesh_file):
#     geom = config_dict["geometry"]
#     fracture_mesh_step = geom['fracture_mesh_step']
#     dimensions = geom["box_dimensions"]
#     well_z0, well_z1 = geom["well_openning"]
#     well_length = well_z1 - well_z0
#     well_r = geom["well_effective_radius"]
#     well_dist = geom["well_distance"]
#     print("load gmsh api")
#
#     from gmsh_api import gmsh
#     from gmsh_api import options
#     from gmsh_api import field
#
#     factory = gmsh.GeometryOCC(mesh_name, verbose=True)
#     gopt = options.Geometry()
#     gopt.Tolerance = 0.0001
#     gopt.ToleranceBoolean = 0.001
#     # gopt.MatchMeshTolerance = 1e-1
#
#     # Main box
#     box = factory.box(dimensions).set_region("box")
#     side_z = factory.rectangle([dimensions[0], dimensions[1]])
#     side_y = factory.rectangle([dimensions[0], dimensions[2]])
#     side_x = factory.rectangle([dimensions[2], dimensions[1]])
#     sides = dict(
#         side_z0=side_z.copy().translate([0, 0, -dimensions[2] / 2]),
#         side_z1=side_z.copy().translate([0, 0, +dimensions[2] / 2]),
#         side_y0=side_y.copy().translate([0, 0, -dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
#         side_y1=side_y.copy().translate([0, 0, +dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
#         side_x0=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2),
#         side_x1=side_x.copy().translate([0, 0, +dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2)
#     )
#     for name, side in sides.items():
#         side.modify_regions(name)
#
#     b_box = box.get_boundary().copy()
#
#     # two vertical cut-off wells, just permeable part
#     left_center = [-well_dist/2, 0, 0]
#     right_center = [+well_dist/2, 0, 0]
#     left_well = factory.cylinder(well_r, axis=[0, 0, well_z1 - well_z0]) \
#         .translate([0, 0, well_z0]).translate(left_center)
#     right_well = factory.cylinder(well_r, axis=[0, 0, well_z1 - well_z0]) \
#         .translate([0, 0, well_z0]).translate(right_center)
#     b_right_well = right_well.get_boundary()
#     b_left_well = left_well.get_boundary()
#
#     print("n fractures:", len(fractures))
#     fractures = create_fractures_rectangles(factory, fractures, factory.rectangle())
#     #fractures = create_fractures_polygons(factory, fractures)
#     fractures_group = factory.group(*fractures)
#     #fractures_group = fractures_group.remove_small_mass(fracture_mesh_step * fracture_mesh_step / 10)
#
#     # drilled box and its boundary
#     box_drilled = box.cut(left_well, right_well)
#
#     # fractures, fragmented, fractures boundary
#     print("cut fractures by box without wells")
#     fractures_group = fractures_group.intersect(box_drilled.copy())
#     print("fragment fractures")
#     box_fr, fractures_fr = factory.fragment(box_drilled, fractures_group)
#     print("finish geometry")
#     b_box_fr = box_fr.get_boundary()
#     b_left_r = b_box_fr.select_by_intersect(b_left_well).set_region(".left_well")
#     b_right_r = b_box_fr.select_by_intersect(b_right_well).set_region(".right_well")
#
#     box_all = []
#     for name, side_tool in sides.items():
#         isec = b_box_fr.select_by_intersect(side_tool)
#         box_all.append(isec.modify_regions("." + name))
#     box_all.extend([box_fr, b_left_r, b_right_r])
#
#     b_fractures = factory.group(*fractures_fr.get_boundary_per_region())
#     b_fractures_box = b_fractures.select_by_intersect(b_box).modify_regions("{}_box")
#     b_fr_left_well = b_fractures.select_by_intersect(b_left_well).modify_regions("{}_left_well")
#     b_fr_right_well = b_fractures.select_by_intersect(b_right_well).modify_regions("{}_right_well")
#     b_fractures = factory.group(b_fr_left_well, b_fr_right_well, b_fractures_box)
#     mesh_groups = [*box_all, fractures_fr, b_fractures]
#
#     print(fracture_mesh_step)
#     #fractures_fr.set_mesh_step(fracture_mesh_step)
#
#     factory.keep_only(*mesh_groups)
#     factory.remove_duplicate_entities()
#     factory.write_brep()
#
#     min_el_size = fracture_mesh_step / 10
#     fracture_el_size = np.max(dimensions) / 20
#     max_el_size = np.max(dimensions) / 8
#
#
#     fracture_el_size = field.constant(fracture_mesh_step, 10000)
#     frac_el_size_only = field.restrict(fracture_el_size, fractures_fr, add_boundary=True)
#     field.set_mesh_step_field(frac_el_size_only)
#
#     mesh = options.Mesh()
#     #mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
#     #mesh.Algorithm = options.Algorithm2d.Delaunay
#     #mesh.Algorithm = options.Algorithm2d.FrontalDelaunay
#     #mesh.Algorithm3D = options.Algorithm3d.Frontal
#     #mesh.Algorithm3D = options.Algorithm3d.Delaunay
#     mesh.ToleranceInitialDelaunay = 0.01
#     #mesh.ToleranceEdgeLength = fracture_mesh_step / 5
#     mesh.CharacteristicLengthFromPoints = True
#     mesh.CharacteristicLengthFromCurvature = True
#     mesh.CharacteristicLengthExtendFromBoundary = 2
#     mesh.CharacteristicLengthMin = min_el_size
#     mesh.CharacteristicLengthMax = max_el_size
#     mesh.MinimumCirclePoints = 6
#     mesh.MinimumCurvePoints = 2
#
#
#     #factory.make_mesh(mesh_groups, dim=2)
#     factory.make_mesh(mesh_groups)
#     factory.write_mesh(format=gmsh.MeshFormat.msh2)
#     os.rename(mesh_name + ".msh2", mesh_file)
#     #factory.show()
#
#
# def prepare_mesh(config_dict, fractures):
#     mesh_name = config_dict["mesh_name"]
#     mesh_file = mesh_name + ".msh"
#     if not os.path.isfile(mesh_file):
#         make_mesh(config_dict, fractures, mesh_name, mesh_file)
#
#     mesh_healed = mesh_name + "_healed.msh"
#     if not os.path.isfile(mesh_healed):
#         import heal_mesh
#         hm = heal_mesh.HealMesh.read_mesh(mesh_file, node_tol=1e-4)
#         hm.heal_mesh(gamma_tol=0.01)
#         hm.stats_to_yaml(mesh_name + "_heal_stats.yaml")
#         hm.write()
#         assert hm.healed_mesh_name == mesh_healed
#     return mesh_healed



if __name__ == "__main__":
    sample_dir = sys.argv[1]
    with open(os.path.join(script_dir, "geometry.yaml"), "r") as f:
        geometry_dict = yaml.safe_load(f)

    os.chdir(sample_dir)
    print("generate mesh")
    # generate_mesh(geometry_dict)
    # generate_mesh_tunnel(geometry_dict)
    # generate_mesh_tunnel_2(geometry_dict)
    # generate_mesh_tunnel_3(geometry_dict)
    generate_mesh_tunnel_4(geometry_dict)
    # generate_mesh_2(geometry_dict)
    # generate_mesh_fuse(geometry_dict)
    print("finished")

    # prepare_th_input(config_dict)
    # np.random.seed()
    # sample(config_dict)