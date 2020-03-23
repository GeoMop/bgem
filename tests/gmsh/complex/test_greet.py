from bgem.gmsh import gmsh
import numpy as np
import pytest


def geometry():
    return dict(
        outer_box=dict(
            name="rock_outer",
            size=[100, 150, 100],  # length in x, y, z
            center=[0, 0, 0]
        ),
        inner_box=dict(
            name="rock_inner",
            size=[30, 150, 30],  # length in x, y, z
            center=[0, 0, 0]
        ),
        tunnel_1=dict(
            name="tunnel_1",
            radius=2.5,
            start=[0, 75, 0],
            end=[0, 17.180586073, -3]
        ),
        tunnel_2=dict(
            name="tunnel_2",
            radius=2.5,
            start=[0, 17.180586073, -3],
            end=[0, -29.319723669, -3]
        )
    )


def create_cylinder(gmsh_occ, cyl_geom, stretch_factor=0.005):

    radius = float(cyl_geom["radius"])
    start = np.array((cyl_geom["start"]))
    end = np.array((cyl_geom["end"]))

    dc = stretch_factor
    u = end-start
    start_t = start - dc*u
    end_t = end + dc*u

    middle = (start+end)/2
    box_lz = np.abs(end_t[2] - start_t[2]) + 2 * radius
    box_lx = np.abs(end_t[0] - start_t[0]) + 2 * radius
    box_ly = 0.99*np.abs(end[1] - start[1])
    # box_ly = np.abs(end[1] - start[1])
    box = gmsh_occ.box([box_lx, box_ly, box_lz], middle)

    cylinder = gmsh_occ.cylinder(radius, end_t-start_t, start_t)
    return cylinder, box


def test_empty_mesh():
    """
    Problem: Even though gmsh reports errors, it creates mesh with no elements.
    See seg fault problem below...

    The fragmentation of the tunnels is a dead end, however, we do not understand the behaviour above.
    """
    mesh_name = "greet_empty_mesh"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)
    geom = geometry()

    # create inner box
    # box_inner = create_box(gen, geometry_dict['inner_box'])

    # create tunnel
    box_size = np.array(geom['outer_box']["size"])
    tunnel_start = np.array(geom['tunnel_1']["start"])
    tunnel_mid = np.array(geom['tunnel_1']["end"])
    tunnel_end = np.array(geom['tunnel_2']["end"])
    side_y = gen.rectangle([box_size[0], box_size[2]]).rotate([-1, 0, 0], np.pi / 2)
    tunnel_split = dict(
        start=side_y.copy().translate([0, tunnel_start[1], 0]),
        mid=side_y.copy().translate([0, tunnel_mid[1], 0]),
        end=side_y.copy().translate([0, tunnel_end[1], 0])
    )

    tunnel_1_c, tunnel_box_1 = create_cylinder(gen, geom['tunnel_1'], 0.2)
    tunnel_1_x = tunnel_1_c.copy().intersect(tunnel_box_1)

    tunnel_2_c, tunnel_box_2 = create_cylinder(gen, geom['tunnel_2'], 0.2)
    tunnel_2_x = tunnel_2_c.copy().intersect(tunnel_box_2)

    tunnel_1_s, tunnel_2_s, s1, s3 = gen.fragment(tunnel_1_c.copy(), tunnel_2_c.copy(),
                                                  tunnel_split["start"].copy(),
                                                  # tunnel_split["mid"].copy(),
                                                  tunnel_split["end"].copy())

    tunnel_1 = tunnel_1_s.select_by_intersect(tunnel_1_x)
    tunnel_2 = tunnel_2_s.select_by_intersect(tunnel_2_x)
    tunnel_1.set_region("t1")
    tunnel_2.set_region("t2")

    # box_inner_reg = box_inner.cut(*tunnel)

    # ENDS WITH SEG FAULT !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # tunnel_1.set_mesh_step(1.5)
    # tunnel_2.set_mesh_step(1.5)

    # ends normally, but with empty mesh !!!!!!!!!!!!!!!!!!!!!!!!!!!
    tunnel_1_s.set_mesh_step(1.5)
    tunnel_2_s.set_mesh_step(1.5)

    mesh_all = [tunnel_1, tunnel_2]

    # gen.write_brep()
    print("Generating mesh...")
    gen.make_mesh(mesh_all)
    print("Generating mesh...[finished]")
    print("Writing mesh...")
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)
    print("Writing mesh...[finished]")
    # gen.show()


@pytest.mark.skip
def test_greet_no_volume():
    """
    Problem: Does not generate volume mesh.
    select_by_intersect probably fails, in comparison to simple cut(*tunnel_f)

    The cutting of the tunnels is a dead end, however, we do not understand the behaviour above.
    """
    mesh_name = "greet_no_volume"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)
    geom = geometry()

    # create inner box
    box_inner = gen.box(geom['inner_box']["size"])
    # create tunnel
    box_size = np.array(geom['outer_box']["size"])
    tunnel_start = np.array(geom['tunnel_1']["start"])
    tunnel_mid = np.array(geom['tunnel_1']["end"])
    tunnel_end = np.array(geom['tunnel_2']["end"])
    side_y = gen.rectangle([box_size[0], box_size[2]]).rotate([-1, 0, 0], np.pi / 2)
    tunnel_split = dict(
        start=side_y.copy().translate([0, tunnel_start[1], 0]),
        mid=side_y.copy().translate([0, tunnel_mid[1], 0]),
        end=side_y.copy().translate([0, tunnel_end[1], 0])
    )

    tunnel_1_c, tunnel_box_1 = create_cylinder(gen, geom['tunnel_1'], 0.2)
    tunnel_1_x = tunnel_1_c.copy().intersect(tunnel_box_1)

    tunnel_2_c, tunnel_box_2 = create_cylinder(gen, geom['tunnel_2'], 0.2)
    tunnel_2_x = tunnel_2_c.copy().intersect(tunnel_box_2)

    splits = [tunnel_split["start"].copy(), tunnel_split["end"].copy()]
    frag = gen.fragment(box_inner.copy(), tunnel_1_c, tunnel_2_c, *splits)

    tunnel_1_f = frag[1].select_by_intersect(tunnel_1_x)
    tunnel_2_f = frag[2].select_by_intersect(tunnel_2_x)
    tunnel_f = [tunnel_1_f, tunnel_2_f]

    box_inner_cut = box_inner.cut(*tunnel_f)
    box_inner_reg = frag[0].select_by_intersect(box_inner_cut)
    # box_inner_reg = box_inner.cut(tunnel_box_1)
    box_inner_reg.set_mesh_step(4.0)

    box_all = []
    b_box_inner = box_inner_reg.get_boundary()
    b_tunnel = b_box_inner.select_by_intersect(*tunnel_f)
    box_all.append(b_tunnel.modify_regions("." + geom['inner_box']["name"] + "_tunnel"))

    box_all.extend([box_inner_reg])

    box_inner_reg.set_mesh_step(4.0)
    b_tunnel.set_mesh_step(1.0)

    mesh_all = [*box_all]

    print("Generating mesh...")
    gen.make_mesh(mesh_all)
    print("Generating mesh...[finished]")
    print("Writing mesh...")
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)
    print("Writing mesh...[finished]")
    # gen.show()


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