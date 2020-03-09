from bgem.gmsh import gmsh
import numpy as np


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


def test_greet_no_volume():
    """
    Problem: Does not generate volume mesh.
    select_by_intersect probably fails, in comparison to simple cut(*tunnel_f)
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