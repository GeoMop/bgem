from bgem.gmsh import gmsh
import numpy as np
import pytest
import os

this_source_dir = os.path.dirname(os.path.realpath(__file__))


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
            # end=[0, -29.319723669, -3] -> increase the angle fort better visualization
            end=[0, -29.319723669, -50]
        )
    )


def create_cylinder(gmsh_occ, cyl_geom, stretch_factor=0.005):
    """
    Auxiliary function, creates prolongated cylinder by given `stretch_factor`.
    returns: cylinder, its bounding box
    """
    radius = float(cyl_geom["radius"])
    start = np.array((cyl_geom["start"]))
    end = np.array((cyl_geom["end"]))

    dc = stretch_factor
    u = end - start
    start_t = start - dc * u
    end_t = end + dc * u

    middle = (start + end) / 2
    box_lz = np.abs(end_t[2] - start_t[2]) + 2 * radius
    box_lx = np.abs(end_t[0] - start_t[0]) + 2 * radius
    box_ly = 0.99 * np.abs(end[1] - start[1])
    # box_ly = np.abs(end[1] - start[1])
    box = gmsh_occ.box([box_lx, box_ly, box_lz], middle)

    cylinder = gmsh_occ.cylinder(radius, end_t - start_t, start_t)
    return cylinder, box


@pytest.mark.skip
def test_empty_mesh():
    """
    Problem: Even though gmsh reports errors, it creates mesh with no elements.
    See seg fault problem below...

    The fragmentation of the tunnels is a dead end, however, we do not understand the behaviour above.

    JB: I can not reproduce seg faults but the meshing problem comes from trying to mesh a rounded tip
    of the two tunnels intersection, that may lead to creating overlapping surface elements on different
    surfaces.
    """
    os.chdir(this_source_dir)
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
    # tunnel_1.mesh_step(1.5)
    # tunnel_2.mesh_step(1.5)

    # ends normally, but with empty mesh !!!!!!!!!!!!!!!!!!!!!!!!!!!
    tunnel_1_s.mesh_step(1.5)
    tunnel_2_s.mesh_step(1.5)

    mesh_all = [tunnel_1, tunnel_2]

    gen.write_brep()
    print("Generating mesh...")

    # Occasionally ends with error:
    # Exception: Invalid boundary mesh (overlapping facets) on surface 39 surface 40
    #   ...
    #   Info    : Tetrahedrizing 744 nodes...
    #   Info    : Done tetrahedrizing 752 nodes (Wall 0.0108547s, CPU 0.010854s)
    #   Info    : Reconstructing mesh...
    #   Info    :  - Creating surface mesh
    #   Info    : Found two overlapping facets.
    #   Info    :   1st: [26, 25, 64] #39
    #   Info    :   2nd: [26, 25, 64] #40
    #   ----------------------------- Captured stderr call -----------------------------
    #   Warning : 16 elements remain invalid in surface 11
    #   Warning : 18 elements remain invalid in surface 25
    #   Warning : 12 elements remain invalid in surface 41
    #   Warning : 18 elements remain invalid in surface 45
    #   Warning : 66 elements remain invalid in surface 25
    #   Warning : 32 elements remain invalid in surface 45
    #   Error   : Invalid boundary mesh (overlapping facets) on surface 39 surface 40

    # Successful test run:
    # Info    : Found volume 6
    # Info    : It. 0 - 0 nodes created - the worst tet radius 1.14137 (nodes removed 0 0)
    # Info    : 3D refinement terminated (2567 nodes total):
    # Info    :  - 0 Delaunay cavities modified for star shapeness
    # Info    :  - 0 nodes could not be inserted
    # Info    :  - 657 tetrahedra created in 0.000683932 sec. (960621 tets/s)
    # Info    : Tetrahedrizing 585 nodes...
    # Info    : Done tetrahedrizing 593 nodes (Wall 0.00824587s, CPU 0.008245s)
    # Info    : Reconstructing mesh...
    # Info    :  - Creating surface mesh
    # Info    :  - Identifying boundary edges
    # Info    :  - Recovering boundary
    # Info    : Done reconstructing mesh (Wall 0.0159928s, CPU 0.015992s)

    # local TOX run:
    # ...
    # Info    : Found volume 6
    # Info    : It. 0 - 0 nodes created - worst tet radius 1.87415 (nodes removed 0 0)
    # Info    : 3D refinement terminated (2553 nodes total):
    # Info    :  - 0 Delaunay cavities modified for star shapeness
    # Info    :  - 7 nodes could not be inserted
    # Info    :  - 64 tetrahedra created in 4.7456e-05 sec. (1348617 tets/s)
    # Info    : Tetrahedrizing 1297 nodes...
    # Info    : Done tetrahedrizing 1305 nodes (Wall 0.020723s, CPU 0.020724s)
    # Info    : Reconstructing mesh...
    # Info    :  - Creating surface mesh
    # Info    : Found two overlapping facets.
    # Info    :   1st: [114, 113, 64] #39
    # Info    :   2nd: [114, 113, 64] #40
    # ---------------------------------------------------------------------------------------------- Captured stderr call -----------------------------------------------------------------------------------------------
    # Warning : 16 elements remain invalid in surface 11
    # Warning : 18 elements remain invalid in surface 25
    # Warning : 4 elements remain invalid in surface 41
    # Warning : 24 elements remain invalid in surface 45
    # Warning : 34 elements remain invalid in surface 25
    # Warning : 6 elements remain invalid in surface 41
    # Warning : 20 elements remain invalid in surface 45
    # Error   : Invalid boundary mesh (overlapping facets) on surface 39 surface 40
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
    box_inner_reg.mesh_step(4.0)

    box_all = []
    b_box_inner = box_inner_reg.get_boundary()
    b_tunnel = b_box_inner.select_by_intersect(*tunnel_f)
    box_all.append(b_tunnel.modify_regions("." + geom['inner_box']["name"] + "_tunnel"))

    box_all.extend([box_inner_reg])

    b_tunnel.mesh_step(1.0)

    mesh_all = [*box_all]

    print("Generating brep...")
    gen.keep_only(*mesh_all)
    gen.write_brep()
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
    os.chdir(this_source_dir)
    mesh_name = "fuse_tunnel"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)
    geom = geometry()

    # create tunnel
    tunnel_start = np.array(geom['tunnel_1']["start"])
    tunnel_mid = np.array(geom['tunnel_1']["end"])
    tunnel_end = np.array(geom['tunnel_2']["end"])

    tunnel_1_c = create_cylinder(gen, geom['tunnel_1'], 0.2)[0]
    tunnel_2_c = create_cylinder(gen, geom['tunnel_2'], 0.2)[0]

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
    tunnel.set_region("tunnel").mesh_step(1.5)

    mesh_all = [tunnel]

    print("Generating mesh...")
    gen.keep_only(*mesh_all)
    gen.write_brep()
    gen.make_mesh(mesh_all)
    print("Generating mesh...[finished]")
    print("Writing mesh...")
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)
    print("Writing mesh...[finished]")
    # gen.show()


def test_fuse_tunnel_2():
    """
    Test shows, how to fuse two cylinders cross-secting each in a common plane under given angle.
    In contrast to previous test, we rotate the second tunnel around extremal point EP of the intersection ellipse
    and then fuses both into final object.

    -> RESULT: not a good approach - the point EP is found correctly, however the cylinders stick out
    which can be seen both in brep and even better in resulting mesh with finer mesh step.
    """
    os.chdir(this_source_dir)
    mesh_name = "fuse_tunnel_2"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)
    geom = geometry()

    # create tunnel
    tunnel_start = np.array(geom['tunnel_1']["start"])
    tunnel_mid = np.array(geom['tunnel_1']["end"])
    tunnel_end = np.array(geom['tunnel_2']["end"])

    radius = float(geom['tunnel_1']["radius"])
    assert radius == float(geom['tunnel_2']["radius"])

    # # cutting box
    box_s = 20
    # box = gen.box([box_s, box_s, box_s])
    # # cutting plane between the cylinders
    plane = gen.rectangle([box_s, box_s]).rotate([1, 0, 0], np.pi / 2)

    # directional vectors of the cylinders
    # supposing equal radius
    v_t1 = tunnel_mid - tunnel_start
    v_t1 = v_t1 / np.linalg.norm(v_t1)  # normalize
    v_t2 = tunnel_mid - tunnel_end
    v_t2 = v_t2 / np.linalg.norm(v_t2)  # normalize

    # directional vectors of the cutting plane
    angle_12 = np.pi - np.arccos(np.dot(v_t1, v_t2))
    d = radius / np.sin(angle_12)
    u = d * (v_t1 + v_t2)  # points from the center to the extremal point of the ellipse E

    # distance we have to move both cylinder, so they meet at E with their corners
    delta = np.sqrt(np.dot(u, u) - radius * radius)

    # create prolongated tunnels for fusion
    tunnel_mid_1 = tunnel_mid + delta * v_t1
    tunnel_mid_2 = tunnel_mid + delta * v_t2
    tunnel_1 = gen.cylinder(radius, tunnel_mid_1 - tunnel_start, tunnel_start)
    tunnel_2 = gen.cylinder(radius, tunnel_mid_2 - tunnel_end, tunnel_end)

    # if needed, make the cutting ellipse plane
    v = np.cross(v_t1, v_t2)
    # normal of the cutting plane
    n = np.cross(u, v)
    n = n / np.linalg.norm(n)  # normalize

    # angle between cutting plane and y-axis
    angle = np.arccos(np.dot([0, 0, 1], n)) - np.pi / 2
    # rotate cutting box and plane
    plane.rotate(axis=[1, 0, 0], angle=angle)

    # move the cutting plane into the connecting point
    plane.translate(tunnel_mid)

    print("fuse...")
    # tunnel = tunnel_1.copy().fuse(tunnel_2.copy())
    tunnel = tunnel_1.fuse(tunnel_2)
    # tunnel = tunnel_2_x.fuse(tunnel_1_x)
    mesh_step = 1.0
    tunnel.set_region("tunnel").mesh_step(mesh_step)

    # tunnel_3 = gen.cylinder(radius, tunnel_mid - tunnel_start, tunnel_start)
    # tunnel_4 = gen.cylinder(radius, tunnel_mid - tunnel_end, tunnel_end)
    # tunnel_1.set_region("tunnel_1").mesh_step(mesh_step)
    # tunnel_2.set_region("tunnel_2").mesh_step(mesh_step)
    # tunnel_3.set_region("tunnel_3").mesh_step(mesh_step)
    # tunnel_4.set_region("tunnel_4").mesh_step(mesh_step)

    # mesh_all = [tunnel]

    # uncomment one of the following (and the corresponding above) to see auxiliary objects
    mesh_all = [tunnel, plane]
    # mesh_all = [plane, tunnel_1, tunnel_2]
    # mesh_all = [tunnel_1, tunnel_2, tunnel, plane, tunnel_3, tunnel_4]

    print("Generating mesh...")
    gen.keep_only(*mesh_all)
    gen.write_brep()
    gen.make_mesh(mesh_all)
    print("Generating mesh...[finished]")
    print("Writing mesh...")
    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)
    print("Writing mesh...[finished]")
    # gen.show()
