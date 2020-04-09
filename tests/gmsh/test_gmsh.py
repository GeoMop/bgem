from bgem.gmsh import gmsh
import numpy as np
import pytest


def test_revolve_square():
    """
    Test revolving a square.
    """
    mesh_name = "revolve_square_mesh"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    square = gen.rectangle([2, 2], [5,0,0])
    axis = []
    center = [5, 10, 0]
    square_revolved = square.revolve(center=[5, 10, 0], axis=[1, 0, 0], angle=np.pi*3/4)

    obj = square_revolved[3].mesh_step(0.5)

    mesh_all = [obj]

    # gen.write_brep(mesh_name)
    gen.make_mesh(mesh_all)
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)


def test_cylinder_discrete():
    """
    Test creating discrete cylinder (prism with regular n-point base), extrusion.
    """
    mesh_name = "cylinder_discrete_mesh"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    r = 2.5
    start = np.array([-10, -5, -15])
    end = np.array([5, 15, 10])
    axis = end-start
    center = (end+start)/2
    cyl = gen.cylinder_discrete(r,axis,center=center, n_points=12)
    cyl.mesh_step(1.0)

    mesh_all = [cyl]

    # gen.write_brep(mesh_name)
    gen.make_mesh(mesh_all)
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)


def test_extrude_circle():
    """
    Test extrusion of an circle.
    """
    mesh_name = "extrude_circle"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    circ = gen.disc(center=[2,5,1], rx=3, ry=3)
    circ_extrude = circ.extrude([2, 2, 2])

    tube = circ_extrude[3]
    tube.set_region("tube").mesh_step(0.5)

    mesh_all = [tube]

    # gen.write_brep(mesh_name)
    gen.make_mesh(mesh_all)
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)


def test_extrude_rect():
    """
    Test extrusion of an rectangle.
    """
    mesh_name = "extrude_rect"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    rect = gen.rectangle([2,5])
    prism_extrude = rect.extrude([1, 3, 4])

    prism = prism_extrude[3]
    prism.set_region("prism").mesh_step(0.5)

    mesh_all = [prism]

    # gen.write_brep(mesh_name)
    gen.make_mesh(mesh_all)
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)


def test_extrude_polygon():
    """
    Test extrusion of an polygon.
    """
    mesh_name = "extrude_polygon"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    # plane directional vectors vector
    u = np.array([1, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.array([0, 1, 1])
    v = v / np.linalg.norm(v)

    #normal
    n = np.cross(u,v)
    n = n / np.linalg.norm(n)

    # add some points in the plane
    points= []
    points.append(u)
    points.append(2 * u + 1 * v)
    points.append(5 * u + -2 * v)
    points.append(5 * u + 3 * v)
    points.append(4 * u + 5 * v)
    points.append(-2 * u + 3*v)
    points.append(v)

    # create polygon
    polygon = gen.make_polygon(points)
    # trying to set mesh step directly to nodes
    # polygon = gen.make_polygon(points, 0.2)
    prism_extrude = polygon.extrude(3*n)

    prism = prism_extrude[3]
    prism.set_region("prism").mesh_step_direct(0.5)

    mesh_all = [prism]

    # gen.write_brep(mesh_name)
    gen.make_mesh(mesh_all)
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)


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
    # box_fused.mesh_step(1)
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
    # box_fused.mesh_step(1)
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

    # test setting mesh step size
    i = 0
    delta = 0.4
    for t in tunnel_parts:
        print(gen.model.getMass(*(t.dim_tags[0])))
        t.mesh_step(0.6+i*delta)
        i = i+1

    gen.make_mesh([*tunnel_parts, *splits])
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)