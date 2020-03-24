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
    """
    Auxiliary function, creates prolongated cylinder.
    """
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



def test_fuse_tunnel():
    """
    Test shows, how to fuse two cylinders cross-secting each in a common plane under given angle.
    It prolongates the cylinders, creates the common face in the middle, cuts the original cylinders
    and then fuses them into final object.
    """
    mesh_name = "tunnel"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)
    geom = geometry()

    # create tunnel
    tunnel_start = np.array(geom['tunnel_1']["start"])
    tunnel_mid = np.array(geom['tunnel_1']["end"])
    tunnel_end = np.array(geom['tunnel_2']["end"])

    tunnel_1_c = create_cylinder(gen, geom['tunnel_1'], 0.2)
    tunnel_2_c = create_cylinder(gen, geom['tunnel_2'], 0.2)

    # cutting box
    box_s = 200
    box = gen.box([box_s, box_s, box_s])
    # cutting plane between the cylinders
    plane = gen.rectangle([box_s, box_s]).rotate([1, 0, 0], np.pi / 2)

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

    print("fuse...")
    tunnel = tunnel_1_x.fuse(tunnel_2_x)
    # tunnel = tunnel_2_x.fuse(tunnel_1_x)
    tunnel.set_region("tunnel")
    tunnel.set_mesh_step(1.5)

    mesh_all = [tunnel]

    print("Generating mesh...")
    gen.keep_only(*mesh_all)
    # gen.write_brep()
    gen.make_mesh(mesh_all)
    print("Generating mesh...[finished]")
    print("Writing mesh...")
    gen.write_mesh(mesh_name, gmsh.MeshFormat.msh2)
    print("Writing mesh...[finished]")
    # gen.show()