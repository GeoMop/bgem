from gmsh import model as gmsh_model
import pytest
import sys
import numpy as np
from bgem.gmsh import gmsh
from fixtures import sandbox_fname
from bgem.gmsh import gmsh_exceptions


def test_line():
    mesh_name = "square_mesh"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True, gmsh_exceptions=True)
    l = gen.line([0, 0, 0], [1, 1, 1])
    print(l)


@pytest.mark.skip
def test_exceptions():
    """
    Test exceptions.
    Using the broken remove_duplicate_entities() function for testing.
    """
    mesh_name = "square_mesh"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True, gmsh_exceptions=True)

    # square = gen.rectangle([2, 2], [5, 0, 0])
    # with pytest.raises(gmsh_exceptions.FragmentationError, match=r".* duplicate .*"):
    #    gen.remove_duplicate_entities()

    # gen.gmsh_exceptions = False
    # we cannot check warning type due to inline creation of the warning type in gmsh_exceptions.make_warning
    # with pytest.warns(Warning, match=r".* duplicate .*"):
    #    gen.remove_duplicate_entities()


def test_revolve_square():
    """
    Test revolving a square.
    """
    mesh_name = "revolve_square_mesh"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    square = gen.rectangle([2, 2], [5, 0, 0])
    axis = []
    center = [5, 10, 0]
    square_revolved = square.revolve(center=[5, 10, 0], axis=[1, 0, 0], angle=np.pi * 3 / 4)

    obj = square_revolved[3].mesh_step(0.5)

    mesh_all = [obj]

    # gen.write_brep(mesh_name)
    gen.make_mesh(mesh_all)
    gen.write_mesh(sandbox_fname(mesh_name, "msh2"), gmsh.MeshFormat.msh2)


def test_cylinder_discrete():
    """
    Test creating discrete cylinder (prism with regular n-point base), extrusion.
    """
    mesh_name = "cylinder_discrete_mesh"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    r = 2.5
    start = np.array([-10, -5, -15])
    end = np.array([5, 15, 10])
    axis = end - start
    center = (end + start) / 2
    cyl = gen.cylinder_discrete(r, axis, center=center, n_points=12)
    cyl.mesh_step(1.0)

    mesh_all = [cyl]

    # gen.write_brep(mesh_name)
    gen.make_mesh(mesh_all)
    gen.write_mesh(sandbox_fname(mesh_name, "msh2"), gmsh.MeshFormat.msh2)


def test_extrude_circle():
    """
    Test extrusion of a circle.
    """
    mesh_name = "extrude_circle"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    circ = gen.disc(center=[2, 5, 1], rx=3, ry=3)
    circ_extrude = circ.extrude([2, 2, 2])

    tube = circ_extrude[3]
    tube.set_region("tube").mesh_step(0.5)

    mesh_all = [tube]

    # gen.write_brep(mesh_name)
    gen.make_mesh(mesh_all)
    gen.write_mesh(sandbox_fname(mesh_name, "msh2"), gmsh.MeshFormat.msh2)


def test_extrude_rect():
    """
    Test extrusion of a rectangle.
    """
    mesh_name = "extrude_rect"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    rect = gen.rectangle([2, 5])
    prism_extrude = rect.extrude([1, 3, 4])

    prism = prism_extrude[3]
    prism.set_region("prism").mesh_step(0.5)

    mesh_all = [prism]

    # gen.write_brep(mesh_name)
    gen.make_mesh(mesh_all)
    gen.write_mesh(sandbox_fname(mesh_name, "msh2"), gmsh.MeshFormat.msh2)


def test_extrude_polygon():
    """
    Test extrusion of a polygon.
    """
    mesh_name = "extrude_polygon"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    # plane directional vectors vector
    u = np.array([1, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.array([0, 1, 1])
    v = v / np.linalg.norm(v)

    # normal
    n = np.cross(u, v)
    n = n / np.linalg.norm(n)

    # add some points in the plane
    points = []
    points.append(u)
    points.append(2 * u + 1 * v)
    points.append(5 * u + -2 * v)
    points.append(5 * u + 3 * v)
    points.append(4 * u + 5 * v)
    points.append(-2 * u + 3 * v)
    points.append(v)

    # create polygon
    polygon = gen.make_polygon(points)
    # trying to set mesh step directly to nodes
    # polygon = gen.make_polygon(points, 0.2)
    prism_extrude = polygon.extrude(3 * n)

    prism = prism_extrude[3]
    prism.set_region("prism").mesh_step_direct(0.5)

    mesh_all = [prism]

    # gen.write_brep(mesh_name)
    gen.make_mesh(mesh_all)
    gen.write_mesh(sandbox_fname(mesh_name, "msh2"), gmsh.MeshFormat.msh2)


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
    gen.write_mesh(sandbox_fname(mesh_name, "msh2"), gmsh.MeshFormat.msh2)


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
    gen.write_mesh(sandbox_fname(mesh_name, "msh2"), gmsh.MeshFormat.msh2)


def test_splitting():
    """
    In this test, we split the object (cylinder) into chosen number of parts.
    We should provide a function for this.
    Question is how to set the splitting plane, how to provide the axis in which we split the object.

    TODO:
    The main point is in the end, where we transform ObjectSet into list of ObjectSet,
    taking advantage of the simple problem.. We will have to use the symmetric fragmentation and then select
    properly the parts...
    We should think of creating method for half-space defined by a plane..
    """
    mesh_name = "splitting"
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)

    # tunnel_start = np.array([-50, -50, 10])
    tunnel_start = np.array([0, 0, 0])
    tunnel_end = np.array([50, 50, 10])
    radius = 5

    tunnel = gen.cylinder(radius, tunnel_end - tunnel_start, tunnel_start)

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
    length_t = np.linalg.norm(tunnel_end - tunnel_start)
    n_parts = 5  # number of parts
    length_part = length_t / n_parts  # length of a single part

    split_pos = tunnel_start + length_part * u_t
    for i in range(n_parts - 1):
        split = split_plane.copy().translate(split_pos)
        splits.append(split)
        split_pos = split_pos + length_part * u_t

    # tunnel_f = tunnel.fragment(*splits)
    tunnel_f = tunnel.fragment(*[s.copy() for s in splits])

    # split fragmented ObjectSet into list of ObjectSets by dimtags
    tunnel_parts = []
    for dimtag, reg in tunnel_f.dimtagreg():
        tunnel_parts.append(gmsh.ObjectSet(gen, [dimtag], [gmsh.Region.default_region[3]]))

    def center_comparison(obj):
        center, mass = obj.center_of_mass()
        return np.linalg.norm(center - tunnel_start)

    # for t in tunnel:
    #     print(center_comparison(t))

    tunnel_parts.sort(reverse=False, key=center_comparison)

    # test setting mesh step size
    i = 0
    delta = 0.4
    for t in tunnel_parts:
        print(gen.model.getMass(*(t.dim_tags[0])))
        t.mesh_step(0.6 + i * delta)
        i = i + 1

    gen.make_mesh([*tunnel_parts, *splits])
    gen.write_mesh(sandbox_fname(mesh_name, "msh2"), gmsh.MeshFormat.msh2)


def test_2D_tunnel_cut():
    """
    Generates a square mesh with an elliptic hole in its center.
    Test:
    - square and ellipse creation
    - cut and fragment functions
    - adding physical boundaries (select_by_intersect)
    - setting mesh_step to the tunnel boundary
    - reading the gmsh logger
    """
    mesh_name = "2d_tunnel_cut"

    tunnel_mesh_step = 0.5
    dimensions = [100, 100]
    tunnel_dims = np.array([4.375, 3.5]) / 2
    tunnel_center = [0, 0, 0]

    # test gmsh loggger
    gen = gmsh.GeometryOCC(mesh_name, verbose=True)
    gmsh_logger = gen.get_logger()
    gmsh_logger.start()

    # Main box
    box = gen.rectangle(dimensions).set_region("box")
    side = gen.line([-dimensions[0] / 2, 0, 0], [dimensions[0] / 2, 0, 0])
    sides = dict(
        bottom=side.copy().translate([0, -dimensions[1] / 2, 0]),
        top=side.copy().translate([0, +dimensions[1] / 2, 0]),
        left=side.copy().translate([0, +dimensions[0] / 2, 0]).rotate([0, 0, 1], np.pi / 2),
        right=side.copy().translate([0, -dimensions[0] / 2, 0]).rotate([0, 0, 1], np.pi / 2)
    )

    # ellipse of the tunnel cross-section
    tunnel_disc = gen.disc(tunnel_center, *tunnel_dims)
    tunnel_select = tunnel_disc.copy()

    box_drilled = box.cut(tunnel_disc)
    box_fr, tunnel_fr = gen.fragment(box_drilled, tunnel_disc)

    box_all = []

    b_box_fr = box_fr.get_boundary()
    for name, side_tool in sides.items():
        isec = b_box_fr.select_by_intersect(side_tool)
        box_all.append(isec.modify_regions("." + name))

    b_tunnel_select = tunnel_select.get_boundary()
    b_tunnel = b_box_fr.select_by_intersect(b_tunnel_select)
    b_tunnel.modify_regions(".tunnel").mesh_step(tunnel_mesh_step)
    box_all.extend([box_fr, b_tunnel])

    mesh_groups = [*box_all]
    gen.keep_only(*mesh_groups)
    # gen.write_brep()

    min_el_size = tunnel_mesh_step / 2
    max_el_size = np.max(dimensions) / 10

    gen.make_mesh(mesh_groups)

    gmsh_log_msgs = gmsh_logger.get()
    gmsh_logger.stop()

    def check_gmsh_log(lines):
        """
        Search for "No elements in volume" message -> could not mesh the volume -> empty mesh.
        # PLC Error:  A segment and a facet intersect at point (-119.217,65.5762,-40.8908).
        #   Segment: [70,2070] #-1 (243)
        #   Facet:   [3147,9829,13819] #482
        # Info    : failed to recover constrained lines/triangles
        # Info    : function failed
        # Info    : function failed
        # Error   : HXT 3D mesh failed
        # Error   : No elements in volume 1
        # Info    : Done meshing 3D (Wall 0.257168s, CPU 0.256s)
        # Info    : 13958 nodes 34061 elements
        # Error   : ------------------------------
        # Error   : Mesh generation error summary
        # Error   :     0 warnings
        # Error   :     2 errors
        # Error   : Check the full log for details
        # Error   : ------------------------------
        """
        empty_volume_error = "No elements in volume"
        res = [line for line in lines if empty_volume_error in line]
        if len(res) != 0:
            raise Exception("GMSH error - No elements in volume")

    check_gmsh_log(gmsh_log_msgs)

    gen.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)

    # estimate number of the smallest elements around the tunnel
    tunnel_circumference = np.pi * np.sqrt(2 * (tunnel_dims[0] ** 2 + tunnel_dims[1] ** 2))
    n_expected = np.round(tunnel_circumference / tunnel_mesh_step)

    # get number of the smallest elements
    n_match = check_min_mesh_step(dim=2, step_size=tunnel_mesh_step, tolerance=0.05)
    assert n_match > n_expected


def check_min_mesh_step(dim, step_size, tolerance):
    """
    Return number of elements that approximately match the given element size.
    """
    ref_shape_edges = {
        1: [(0, 1)],
        2: [(0, 1), (0, 2), (1, 2)],
        3: [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    }

    node_tags, coords, param_coords = gmsh_model.mesh.getNodes(dim=-1, returnParametricCoord=False)
    coords = np.reshape(coords, (-1, 3))
    node_indices = {tag: idx for idx, tag in enumerate(node_tags)}
    assert coords.shape[0] == len(node_tags)
    ele_types, ele_tags, ele_node_tags = gmsh_model.mesh.getElements(dim=dim)
    assert len(ele_types) == 1 and len(ele_tags) == 1 and len(ele_node_tags) == 1
    ele_tags = ele_tags[0]
    ele_node_tags = np.reshape(ele_node_tags[0], (-1, dim + 1))

    n_match = 0
    max_rel_error = 0
    for ele_tag, ele_nodes in zip(ele_tags, ele_node_tags):
        i_nodes = [node_indices[n_tag] for n_tag in ele_nodes]
        vertices = coords[i_nodes, :]
        edges = [vertices[i, :] - vertices[j, :] for i, j in ref_shape_edges[dim]]
        ele_size = np.max(np.linalg.norm(edges, axis=1))
        barycenter = np.average(vertices, axis=0)
        rel_error = abs(ele_size - step_size) / step_size
        max_rel_error = max(max_rel_error, rel_error)
        # print(f"ele {ele_tag}, size: {ele_size}, ref size: {ref_ele_size}, {rel_error}")
        if rel_error <= tolerance:
            print(f"Size mismatch, ele {ele_tag}, size: {ele_size}, ref size: {step_size}, rel_err: {rel_error}")
            n_match += 1
    return n_match


def test_copy():
    """
    Test ObjectSet.copy of dimtags.
    """
    dimensions = [10, 20, 30]

    # test gmsh loggger
    gen = gmsh.GeometryOCC("test_copy", verbose=True)
    gmsh_logger = gen.get_logger()
    gmsh_logger.start()

    # Main box
    box = gen.box(dimensions).set_region("box")
    dimtags = box.dim_tags
    boundary = box.get_boundary()
    boundary_copy = boundary.copy()
    print(boundary)
    print(boundary_copy)
