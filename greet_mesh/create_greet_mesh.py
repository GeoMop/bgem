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

    tunnel = tunnel_1_x.fuse(tunnel_2_x)
    return tunnel, plane, box

def generate_mesh(geom):
    options.Geometry.Tolerance = 0.001
    options.Geometry.ToleranceBoolean = 1e-2
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

    gen = gmsh.GeometryOCC("greet_mesh_tunnel", verbose=True)

    with open(os.path.join(script_dir, "geometry.yaml"), "r") as f:
        geom = yaml.safe_load(f)

    # create inner box
    # box_inner = create_box(gen, geometry_dict['inner_box'])

    # create tunnel
    # box_size = np.array(geometry_dict['outer_box']["size"])
    tunnel_start = np.array(geom['tunnel_1']["start"])
    tunnel_mid = np.array(geom['tunnel_1']["end"])
    tunnel_end = np.array(geom['tunnel_2']["end"])
    # side_y = gen.rectangle([box_size[0], box_size[2]]).rotate([-1, 0, 0], np.pi / 2)
    # tunnel_split = dict(
    #     start=side_y.copy().translate([0, tunnel_start[1], 0]),
    #     mid=side_y.copy().translate([0, tunnel_mid[1], 0]),
    #     end=side_y.copy().translate([0, tunnel_end[1], 0])
    # )

    tunnel, cutting_plane, cutting_box = fuse_tunnels(gen, geom)
    tunnel.set_region("tunnel")
    tunnel.set_mesh_step(1.5)

    # tunnel_1_x.set_region("t1")
    # tunnel_2_x.set_region("t2")
    # tunnel_1_x.set_mesh_step(1.5)
    # tunnel_2_x.set_mesh_step(1.5)

    mesh_all = [tunnel]

    print("Generating mesh...")
    gen.keep_only(*mesh_all)
    gen.write_brep()
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