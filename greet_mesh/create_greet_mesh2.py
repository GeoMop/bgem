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
# from bgem.gmsh import heal_mesh


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

    # middle = (start+end)/2
    # box_lz = np.abs(end_t[2] - start_t[2]) + 2* radius
    # box_lx = np.abs(end_t[0] - start_t[0]) + 2* radius
    # # box_ly = 0.99*np.abs(end[1] - start[1])
    # box_ly = np.abs(end[1] - start[1])
    # box = gmsh_occ.box([box_lx, box_ly, box_lz], middle)

    cylinder = gmsh_occ.cylinder(radius, end_t-start_t, start_t)
    # cylinder = gmsh_occ.cylinder_discrete(radius, end_t - start_t, middle, 24)
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
    return cylinder #, box


def fuse_tunnels(gmsh_occ, geom):
    tunnel_start = np.array(geom['tunnel_1']["start"])
    tunnel_mid = np.array(geom['tunnel_1']["end"])
    tunnel_end = np.array(geom['tunnel_2']["end"])

    tunnel_1_c = create_cylinder(gmsh_occ, geom['tunnel_1'], 0.2)
    tunnel_2_c = create_cylinder(gmsh_occ, geom['tunnel_2'], 0.2)

    # cutting box
    box_s = 200
    box = gmsh_occ.box([box_s, box_s, box_s])
    # cutting plane between the cylinders
    plane = gmsh_occ.rectangle([box_s, box_s]).rotate([1, 0, 0], np.pi / 2)

    # directional vectors of the cylinders
    # supposing equal radius
    v_t1 = tunnel_mid - tunnel_start
    v_t1 = v_t1 / np.linalg.norm(v_t1)  # normalize
    v_t2 = tunnel_mid - tunnel_end
    v_t2 = v_t2 / np.linalg.norm(v_t2)  # normalize

    # directional vectors of the cutting plane
    u = v_t1 + v_t2
    v = np.cross(v_t1, v_t2)
    # normal of the cutting plane
    n = np.cross(u, v)
    n = n / np.linalg.norm(n)  # normalize

    # angle between cutting plane and y-axis
    angle = np.arccos(np.dot([0, 0, 1], n)) - np.pi / 2
    # rotate cutting box and plane
    box.rotate(axis=[1, 0, 0], angle=angle)
    plane.rotate(axis=[1, 0, 0], angle=angle)

    # move the cutting plane into the connecting point
    plane.translate(tunnel_mid)

    # move box and cut the first cylinder
    box.translate(tunnel_mid)
    box.translate(-box_s / 2 * n)
    tunnel_1_x = tunnel_1_c.intersect(box)

    # move box and cut the second cylinder
    box.translate(+box_s * n)
    tunnel_2_x = tunnel_2_c.intersect(box)

    box2 = gmsh_occ.box([box_s, box_s, box_s]).translate([0, tunnel_end[1] + box_s/2, 0])
    tunnel_2_xx = tunnel_2_x.intersect(box2)

    tunnel_fuse = tunnel_1_x.fuse(tunnel_2_xx)

    splits = []
    box_size = np.array(geometry_dict['outer_box']["size"])
    split_plane = gmsh_occ.rectangle([box_size[0], box_size[2]]).rotate([-1, 0, 0], np.pi / 2)
    t1_length_y = np.abs(tunnel_mid[1] - tunnel_start[1])
    n_parts = 5  # number of parts of a single tunnel section
    part_y = t1_length_y / n_parts  # y dimension of a single part

    y_split = tunnel_start[1] - part_y
    for i in range(n_parts-1):
        split = split_plane.copy().translate([0, y_split, 0])
        splits.append(split)
        y_split = y_split - part_y  # move split to the next one
    splits.append(plane)    # add cutting plane in the middle

    t2_length_y = np.abs(tunnel_end[1] - tunnel_mid[1])
    part_y = t2_length_y / n_parts  # y dimension of a single part

    y_split = tunnel_mid[1] - part_y
    for i in range(n_parts-1):
        split = split_plane.copy().translate([0, y_split, 0])
        splits.append(split)
        y_split = y_split - part_y  # move split to the next one

    # splits.append(split_plane.copy().translate([0, tunnel_end[1], 0]))

    # tunnel_f = gmsh_occ.fragment(tunnel_fuse, *splits)
    tunnel_f = tunnel_fuse.fragment(*splits)

    # split fragmented ObjectSet into list of ObjectSets by dimtags
    tunnel = []
    for dimtag, reg in tunnel_f.dimtagreg():
        tunnel.append(gmsh.ObjectSet(gmsh_occ, [dimtag], [gmsh.Region.default_region[3]]))

    def center_comparison(obj):
        center, mass = obj.center_of_mass()
        return center[1]

    # for t in tunnel:
    #     print(center_comparison(t))

    tunnel.sort(reverse=True, key=center_comparison)

    return tunnel

def generate_mesh(geom):
    # options.Geometry.Tolerance = 0.001
    # options.Geometry.ToleranceBoolean = 1e-2
    # options.Geometry.AutoCoherence = 2
    # options.Geometry.OCCFixSmallEdges = True
    # options.Geometry.OCCFixSmallFaces = True
    # options.Geometry.OCCFixDegenerated = True
    # options.Geometry.OCCSewFaces = True
    # # #
    # options.Mesh.ToleranceInitialDelaunay = 0.01
    # options.Mesh.CharacteristicLengthMin = 0.1
    # options.Mesh.CharacteristicLengthMax = 2.0
    # options.Mesh.AngleToleranceFacetOverlap = 0.8

    gen = gmsh.GeometryOCC("greet_mesh_tunnel", verbose=True)

    with open(os.path.join(script_dir, "geometry.yaml"), "r") as f:
        geom = yaml.safe_load(f)

    # create tunnel
    # box_size = np.array(geometry_dict['outer_box']["size"])
    tunnel_start = np.array(geom['tunnel_1']["start"])
    tunnel_mid = np.array(geom['tunnel_1']["end"])
    tunnel_end = np.array(geom['tunnel_2']["end"])

    tunnel = fuse_tunnels(gen, geom)

    # create inner box
    box_inner = create_box(gen, geometry_dict['inner_box'])
    box_inner_cut = box_inner.cut(*tunnel)
    box_inner_cut.set_region(geometry_dict['inner_box']["name"])

    # print("fragment start")
    # frag = gen.fragment(box_inner, *splits)
    # frag = gen.fragment(box_inner_cut, *splits)
    # box_inner_f = box_inner_cut.fragment(*splits)
    # print("fragment end")

    box_all = [box_inner_cut]
    b_box_inner = box_inner_cut.get_boundary()
    # # for name, side_tool in sides.items():
    # #     isec_inner = b_box_inner.select_by_intersect(side_tool)
    # #     box_all.append(isec_inner.modify_regions("." + geometry_dict['inner_box']["name"] + "_" + name))
    # #

    mesh_step_dict = dict()
    mesh_step_dict[box_inner_cut] = 4.0
    # reg_names = dict()
    # reg_name = "." + geometry_dict['inner_box']["name"] + "_tunnel"
    # b_tunnel.set_region(reg_name)
    # reg_names["tunnel"] = reg_name
    # box_all.append(b_tunnel)
    # box_all.append(b_tunnel.modify_regions("." + geometry_dict['inner_box']["name"] + "_tunnel"))

    # reg_names["tunnel"] = []
    for i in range(len(tunnel)):
        b_tunnel_part = b_box_inner.select_by_intersect(tunnel[i])
        reg_name = ".tunnel" + "_" + str(i)
        # reg_names["tunnel"].append(reg_name)
        mesh_step_dict[b_tunnel_part] = 1.5
        b_tunnel_part.set_region(reg_name)
        box_all.append(b_tunnel_part)


    # the tunnel must be removed before meshing | we do not know the reason
    for t in tunnel:
        gen.model.remove(t.dim_tags)
    # for sb in split_boxes:
    #     gen.model.remove(sb.dim_tags)

    i=0
    for b in box_all:
        print("i={0} ".format(i))
        for r in b.regions:
            print("{0} ({1}) ".format(r.name, r.id))
        for dt in b.dim_tags:
            print(dt)
        i = i+1

    # mesh_all = [*tunnel]
    mesh_all = [*box_all]
    # mesh_all = [*split_boxes]

    print("Generating mesh...")
    gen.keep_only(*mesh_all)
    gen.write_brep()

    for obj in box_all:
        obj.set_mesh_step(mesh_step_dict[obj])
    # for b in box_all:
    #     if b.regions[0].name in reg_names["tunnel"]:
    #         b.set_mesh_step(1.5)
    #         continue
    #
    #     if b.regions[0].name in reg_names["boxes"]:
    #         b.set_mesh_step(2)
    #         continue

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

if __name__ == "__main__":
    sample_dir = sys.argv[1]
    with open(os.path.join(script_dir, "geometry.yaml"), "r") as f:
        geometry_dict = yaml.safe_load(f)

    os.chdir(sample_dir)
    print("generate mesh")

    generate_mesh(geometry_dict)

    print("finished")