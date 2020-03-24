from bgem.gmsh import gmsh
import numpy as np
import pytest


def test_fuse_boxes():
    """
    Test of fusion function. It makes union of two intersection boxes.
    """
    mesh_name = "box_fuse"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    # create inner box
    box_1 = gen.box([20, 20, 20])
    box_2 = gen.box([10, 10, 40])

    box_fused = box_2.fuse(box_1)
    box_fused.set_region("box")
    # box_fused.set_mesh_step(1)
    all_obj = [box_fused]

    mesh_all = [*all_obj]

    gen.make_mesh(mesh_all)
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)


def test_fuse_boxes2():
    """
    Test of fusion function. It makes union of three face-adjacent boxes.
    Possibly it can make union of two non-intersecting boxes.
    """
    mesh_name = "box_fuse_2"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    # create inner box
    box_1 = gen.box([20, 20, 20])
    box_2 = gen.box([20, 20, 20])
    box_3 = gen.box([20, 20, 20])

    box_2.translate([0, 20, 0])
    box_3.translate([0, 40, 0])

    # make union of two non-intersecting boxes
    # box_fused = box_1.fuse(box_3)

    box_fused = box_1.fuse(box_2, box_3)
    assert box_fused.regions[0] == gmsh.Region.default_region[3]
    box_fused.set_region("box")
    # box_fused.set_mesh_step(1)
    all_obj = [box_fused]

    mesh_all = [*all_obj]

    gen.make_mesh(mesh_all)
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)


def test_splitting():
    """
    In this test, we split the object (cylinder) into chosen number of parts.
    We should provide a function for this.
    Question is how to set the splitting plane, how to provide the axis in which we split the object.

    TODO:
    The main point is in the end, where we transform ObjectSet into list of ObjectSet,
    taking advantage of the simple problem.. We will have to use the symetric fragmentation and then select
    properly the parts...
    We should think of creating method for half-space defined by a plane..
    """
    mesh_name = "splitting"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    # tunnel_start = np.array([-50, -50, 10])
    tunnel_start = np.array([0, 0, 0])
    tunnel_end = np.array([50, 50, 10])
    radius = 5

    tunnel = gen.cylinder(radius, tunnel_end-tunnel_start, tunnel_start)

    # cutting box
    box_s = 50
    # cutting plane between the cylinders
    split_plane = gen.rectangle([box_s, box_s])
    # normal is in z-axis
    z = [0, 0, 1]

    # directional vector of the cylinder
    u_t = tunnel_end - tunnel_start
    u_t = u_t / np.linalg.norm(u_t)  # normalize

    # axis of rotation
    axis = np.cross(z, u_t)
    axis = axis / np.linalg.norm(axis)  # normalize

    angle = np.arccos(np.dot(u_t, z))
    split_plane.rotate(axis=axis, angle=angle, center=[0, 0, 0])

    splits = []
    length_t = np.linalg.norm(tunnel_end-tunnel_start)
    n_parts = 5  # number of parts
    length_part = length_t / n_parts  # length of a single part

    split_pos = tunnel_start + length_part*u_t
    for i in range(n_parts-1):
        split = split_plane.copy().translate(split_pos)
        splits.append(split)
        split_pos = split_pos + length_part*u_t

    # tunnel_f = tunnel.fragment(*splits)
    tunnel_f = tunnel.fragment(*[s.copy() for s in splits])

    # split fragmented ObjectSet into list of ObjectSets by dimtags
    tunnel_parts = []
    for dimtag, reg in tunnel_f.dimtagreg():
        tunnel_parts.append(gmsh.ObjectSet(gen, [dimtag], [gmsh.Region.default_region[3]]))

    def center_comparison(obj):
        center, mass = obj.center_of_mass()
        return np.linalg.norm(center-tunnel_start)

    # for t in tunnel:
    #     print(center_comparison(t))

    tunnel_parts.sort(reverse=False, key=center_comparison)

    for t in tunnel_parts:
        print(gen.model.getMass(*(t.dim_tags[0])))
        t.set_mesh_step(2.0)

    gen.make_mesh([*tunnel_parts, *splits])
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)