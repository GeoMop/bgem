import pytest
from gmsh import model as gmsh_model
from bgem.gmsh import gmsh, field, options
import numpy as np
import os
import math
import yaml


def create_fractures_rectangles(gmsh_geom, fractures, base_shape: 'ObjectSet'):
    # From given fracture date list 'fractures'.
    # transform the base_shape to fracture objects
    # fragment fractures by their intersections
    # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
    shapes = []
    for i, fr in enumerate(fractures):
        shape = base_shape.copy()
        print("fr: ", i, "tag: ", shape.dim_tags)
        shape = shape.scale([fr.rx, fr.ry, 1]) \
            .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
            .translate(fr.centre) \
            .set_region(fr.region)

        shapes.append(shape)

    fracture_fragments = gmsh_geom.fragment(*shapes)
    return fracture_fragments

#def apply_field(field1, field2, reference_fn, dim=2, tolerance=0.15, max_mismatch=5, mesh_name="field_mesh"):
def apply_field(dim, tolerance=0.15, max_mismatch=5, mesh_name="field_mesh"):
    """
    Create a mesh of dimension dim on a unit cube and
    compare element sizes to given reference function of coordinates.
    """

    sample_dir = "output"
    os.chdir(sample_dir)

    model = gmsh.GeometryOCC(mesh_name)

    #circ = model.circle(10).set_region("circle")

    rec3 = model.rectangle([220, 20]).set_region("square3")
    rec = model.rectangle([300, 100]).cut(rec3.copy()).set_region("square")
    #rec = model.rectangle([500, 500]).cut(rec3.copy(), rec4.copy()).set_region("square")
    #rec = model.group(rec3, rec2)

    boundaries = gmsh.ObjectSet.get_boundary(rec3, False)
    boundaries2 = gmsh.ObjectSet.get_boundary(rec, False)

    #rec3 = model.rectangle([50, 10]).translate([100,50,0])

    #tagy = rec1.tags

    a = 3
    b = 0.2
    c = 10
    d = 10

    field3 = field.Field('MathEval')
    # field3.F = "x/100"

    uzly = []
    for i in range(boundaries.size):
        uzly.append(boundaries.dim_tags[i][1])

    #for j in range(boundaries2.size):
    #   uzly.append(boundaries2.dim_tags[i][1])


    #field1 = field.threshold(field.distance_nodes([boundaries.dim_tags[0][1], boundaries.dim_tags[1][1]]), lower_bound=(a, b), upper_bound=(c, d))
    f = field.threshold(field.distance_edges(uzly, 30), lower_bound=(a, b), upper_bound=(c, d))

    """
    pta = [0, 0, 0]
    ptb = [100, 100, 0]

    field2 = field.box(pta, ptb, 1, 20)

    f_1 = field.constant(1)
    #f1 = (field.abs(field.y)- 10)/10
    f1 = (field.abs(field.y) - 10)
    f2 = (field.abs(field.x) - 50)/10

    f1 = (1 - field.sign(field.abs(field.y) - 10)) + (1 + field.sign(field.abs(field.y) - 10)) * field.abs(field.abs(field.y) - 10)/8
    f2 = (1 - field.sign(field.abs(field.x) - 50)) + (1 + field.sign(field.abs(field.x) - 50)) * field.abs(
        field.abs(field.x) - 50) / 8

    f1 = 5 - field.y * field.y / 100 - field.x * field.x / 100
    f2 = field.y * field.y / 500 + field.x * field.x / 500

    #f2 = (field.abs(field.y) +10) / 10

    f = field.maximum(f1, f2)
    f = field1
    """

    ####################################################
    """
    model = gmsh.GeometryOCC(mesh_name)

    rec3 = model.box([200, 200, 200]).set_region("square3")
    rec4 = model.box([50, 50, 50]).set_region("square4")
    rec = rec3.cut(rec4.copy()).set_region("square")

    boundaries = gmsh.ObjectSet.get_boundary(rec4, False)

    uzly = []
    for i in range(boundaries.size):
        uzly.append(boundaries.dim_tags[i][1])

    f = field.threshold(field.distance_surfaces(uzly, 8), lower_bound=(a, b), upper_bound=(c, d))
    """
    ####################################################

    model.set_mesh_step_field(f)

    model.write_brep()
    model.mesh_options.CharacteristicLengthMin = 0.1
    model.mesh_options.CharacteristicLengthMax = 100
    model.make_mesh([rec], dim=dim)
    model.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)

    del model

def apply_field3(dim, tolerance=0.15, max_mismatch=5, mesh_name="field_mesh"):
    """
    Create a mesh of dimension dim on a unit cube and
    compare element sizes to given reference function of coordinates.
    """

    sample_dir = "output"
    os.chdir(sample_dir)

    model = gmsh.GeometryOCC(mesh_name)

    rec = model.rectangle([20, 20]).set_region("square2")
    boundaries = gmsh.ObjectSet.get_boundary(rec, True)

    f1 = 3 - field.y * field.y / 5 - field.x * field.x / 5
    f2 = field.y * field.y / 50 + field.x * field.x / 50


    #f2 = (field.abs(field.y) +10) / 10

    f = field.maximum(f1, f2)
    #f = f2

    model.set_mesh_step_field(f)
    # model.set_mesh_step_field(field2)
    # rec1.mesh_step(1)
    # rec.mesh_step(0.5)

    model.write_brep()
    model.mesh_options.CharacteristicLengthMin = 0.5
    model.mesh_options.CharacteristicLengthMax = 10
    model.make_mesh([rec], dim=dim)
    model.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)

    del model

def test_eval_expr():
    # f_1 = field.constant(1)
    # f_3 = field.constant(3)
    # f = f_1*f_3 + f_3/f_1 + 1 - f_1
    f = field.constant(10)
    def ref_size(x):
        return 6
    apply_field(f, ref_size, dim=2, tolerance=0.38, max_mismatch=0)

#@pytest.mark.skip
def test_eval_sin_cos():
    f_1 = field.constant(1)
    f = 11 + 10 * field.sin(field.x/50*np.pi) * field.cos(field.y/25*np.pi) + field.log(f_1)
    def ref_size(x):
        return 11 + 10 * np.sin(x[0]/50*np.pi) * np.cos(x[1]/25*np.pi)
    apply_field(f, ref_size, dim=2, tolerance=0.60, max_mismatch=50, mesh_name="sin_cos")

#@pytest.mark.skip
def test_eval_max():
    f = 0.2*field.max(field.y, -field.y, 20)

    def ref_size(x):
        z = 0.2*(max(x[1], -x[1], 20))
        return z
    apply_field(f, ref_size, dim=2, tolerance=0.30, max_mismatch=20, mesh_name="variadic_max")

#@pytest.mark.skip
def test_eval_sum():
    f = 0.2*field.sqrt(field.sum(field.y*field.y, field.x*field.x, 100))

    def ref_size(x):
        z = 0.2*np.sqrt((sum([x[0]*x[0], x[1]*x[1], 100])))
        return z
    apply_field(f, ref_size, dim=2, tolerance=0.30, max_mismatch=11, mesh_name="variadic_sum")


def test_eval():
    f_3 = field.constant(3)
    def ref_size(x):
        return 6
    f = f_3 + f_3
    apply_field(f, ref_size, dim=2, tolerance=0.38, max_mismatch=0)
    f = (f_3 > 2) * 6
    apply_field(f, ref_size, dim=2, tolerance=0.38, max_mismatch=0)
    f = (2 < f_3) * 6
    apply_field(f, ref_size, dim=2, tolerance=0.38, max_mismatch=0)
    f = (3 >= f_3) * 6
    apply_field(f, ref_size, dim=2, tolerance=0.38, max_mismatch=0)
    f = (f_3 <= 3) * 6
    apply_field(f, ref_size, dim=2, tolerance=0.38, max_mismatch=0)


#@pytest.mark.skip
def test_constant_2d():
    f_const = field.constant(3)
    def ref_size(x):
        return 7
    apply_field(f_const, ref_size, dim=2, tolerance=0.3, max_mismatch=0)

#@pytest.mark.skip
def test_box_2d():
    f_box = field.box([-20,-20, 0], [20,20, 0], 7, 11)
    def ref_size(x):
        return 11 if np.max(np.abs(x)) > 20 else 7
    apply_field(f_box, ref_size, dim=2, tolerance=0.3, max_mismatch=15)

#@pytest.mark.skip
def test_distance_nodes_2d():
    """
    For some reason the mesh size is actually close to half of the distance
    and naturally the error is quite high, up to 70%
    """
    f_distance = field.distance_nodes([1, 3])  # points: (-50, -50), (50, 50)
    def ref_size(x):
        dist = min(np.linalg.norm(np.array(x) - np.array([-50,-50,0])),
                   np.linalg.norm(np.array(x) - np.array([50,50,0])))
        return max(0.5 * dist, 0.5)
    apply_field(f_distance, ref_size, dim=2, tolerance=0.7, max_mismatch=10, mesh_name="dist_2d")


def test_minmax_nodes_2d(a, b):
    f_distance = field.min(a, field.max(b, field.distance_nodes([1, 3]))) # points: (-50, -50), (50, 50)
    def ref_size(x):
        dist = min(np.linalg.norm(np.array(x) - np.array([-50,-50,0])),
                   np.linalg.norm(np.array(x) - np.array([50,50,0])))
        return min(a, max(b, dist))
    apply_field(f_distance, ref_size, dim=2, tolerance=0.4, max_mismatch=10)


def test_threshold_2d(a,b,c,d, c1, c2):
    f_distance = field.threshold(field.distance_nodes([1, 3]),
                                 lower_bound=(a,b), upper_bound=(c, d))

    def thold(x,minx,miny,maxx,maxy):
        if x < minx:
            return miny
        elif x > maxx:
            return maxy
        else:
            return (x - minx)/(maxx - minx) * (maxy - miny) + miny

    def ref_size(x):
        dist = min(np.linalg.norm(np.array(x) - np.array([-c1,-c2,0])),
                   np.linalg.norm(np.array(x) - np.array([c1,c2,0])))
        # dist = np.linalg.norm(np.array(x) - np.array([0, 0, 0]))

        return thold(dist, a, b, c, d)

    apply_field(f_distance, ref_size, dim=2, tolerance=0.3, max_mismatch=10, mesh_name = "thd")


def test_threshold_sigmoid_2d(a,b,c,d):
    f_distance = field.threshold(field.distance_nodes([1, 3]),
                                 lower_bound=(a,b), upper_bound=(c,d), sigmoid=True)

    def thold(x,minx,miny,maxx,maxy):
        scaled_X = (x - minx)/(maxx - minx)
        y = 1/(1 + np.exp(-scaled_X))
        return  y * (maxy - miny) + miny

    def ref_size(x):
        dist = min(np.linalg.norm(np.array(x) - np.array([-50,-50,0])),
                   np.linalg.norm(np.array(x) - np.array([50,50,0])))
        return thold(dist, a, b, c, d)
    apply_field(f_distance, ref_size, dim=2, tolerance=0.4, max_mismatch=8, mesh_name = "thds")

def test_threshold_2d_moje(a,b,c,d, c1, c2):
    f_distance1 = field.threshold(field.distance_nodes([1, 2, 3, 4]),
                                 lower_bound=(a,b), upper_bound=(c, d))

    f_distance2 = field.threshold(field.distance_nodes([3, 4]),
                                 lower_bound=(a, b/2), upper_bound=(c, d))

    def thold(x,minx,miny,maxx,maxy):
        if x < minx:
            return miny
        elif x > maxx:
            return maxy
        else:
            return (x - minx)/(maxx - minx) * (maxy - miny) + miny

    def ref_size(x):
        # dist = min(np.linalg.norm(np.array(x) - np.array([-c1,-c2,0])),
        #            np.linalg.norm(np.array(x) - np.array([c1,c2,0])))
        dist = np.linalg.norm(np.array(x))

        return thold(dist, a, b, c, d)

    apply_field(f_distance1, f_distance2, ref_size, dim=2, tolerance=0.3, max_mismatch=10, mesh_name = "thd")

def apply_field4(config_dict, fractures, mesh_name, mesh_file):

    #sample_dir = "output"
    #os.chdir(sample_dir)

    model = gmsh.GeometryOCC(mesh_name)

    geom = config_dict["geometry"]
    fracture_mesh_step = geom['fracture_mesh_step']
    dimensions = geom["box_dimensions"]
    boundary_mesh_step = geom['boundary_mesh_step']
    boreholes_mesh_step = geom['boreholes_mesh_step']
    main_tunnel_mesh_step = geom['main_tunnel_mesh_step']

    mtr = geom["main_tunnel_radius"]
    mtw = geom["main_tunnel_width"]
    mtl = geom["main_tunnel_length"]

    st_r = geom["lateral_tunnel_radius"]
    stw = geom["lateral_tunnel_width"]
    stl = geom["lateral_tunnel_length"]

    br = geom["borehole_radius"]
    bl = geom["borehole_length"]
    bd = geom["borehole_distance"]

    #factory = gmsh.GeometryOCC(mesh_name, verbose=True)
    gopt = options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001

    # Main box
    box = model.box(dimensions).set_region("box")
    side_z = model.rectangle([dimensions[0], dimensions[1]])
    side_y = model.rectangle([dimensions[0], dimensions[2]])
    side_x = model.rectangle([dimensions[2], dimensions[1]])
    sides = dict(
        side_z0=side_z.copy().translate([0, 0, -dimensions[2] / 2]),
        side_z1=side_z.copy().translate([0, 0, +dimensions[2] / 2]),
        side_y0=side_y.copy().translate([0, 0, -dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_y1=side_y.copy().translate([0, 0, +dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_x0=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        side_x1=side_x.copy().translate([0, 0, +dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.modify_regions(name)

    ###########################################################################################################

    # borehole
    b_1 = model.cylinder(br, axis=[bl, 0, 0]).translate([stl - bl / 2, 0, 0])
    b_2 = b_1.copy().translate([0, bd, 0])
    b_3 = b_1.copy().translate([0, -bd, 0])
    borehole = model.group(b_1, b_2, b_3)

    # main_tunnel = longitudinal + lateral parts
    main_tunnel_block = model.box([mtw, mtl, mtr])
    y_shift = math.sqrt(mtr * mtr - 0.25 * mtw * mtw) - mtr / 2
    main_tunnel_block_tmp = main_tunnel_block.copy().translate([0, 0, mtr])
    main_tunnel_cylinder_tmp = model.cylinder(mtr, axis=[0, mtl, 0]).translate([0, -mtl / 2, -y_shift])
    main_tunnel_cylinder = main_tunnel_block_tmp.intersect(main_tunnel_cylinder_tmp)

    y_shift_2 = math.sqrt(st_r * st_r - 0.25 * stw * stw) - st_r / 2
    small_tunnel_block = model.box([stl, stw, st_r])
    small_tunnel_block_tmp = small_tunnel_block.copy().translate([0, 0, st_r])
    small_tunnel_cylinder_tmp = model.cylinder(st_r, axis=[stl, 0, 0]) \
            .translate([-stl / 2, 0, -y_shift_2])
    small_tunnel_cylinder = small_tunnel_block_tmp.intersect(small_tunnel_cylinder_tmp)

    lateral_tunnel_part_1 = model.group(small_tunnel_block, small_tunnel_cylinder) \
            .translate([stl / 2, 0, 0])
    lateral_tunnel_part_2 = lateral_tunnel_part_1.copy().translate([0, bd, 0])
    lateral_tunnel_part_3 = lateral_tunnel_part_1.copy().translate([0, -bd, 0])

    block = main_tunnel_block.fuse(main_tunnel_cylinder, lateral_tunnel_part_1, lateral_tunnel_part_2,
                                       lateral_tunnel_part_3).translate([-bl / 2, 0, 0])

    ###########################################################################################################

    b_borehole = borehole.get_boundary()
    b_block = block.get_boundary()

    fractures = create_fractures_rectangles(model, fractures, model.rectangle())
    fractures_group = model.group(*fractures)

    sides2 = box.get_boundary().set_region(".sides")
    box_drilled = box.cut(block.copy(), borehole.copy())

    fractures_group = fractures_group.intersect(box_drilled.copy())

    box_fr, fractures_fr, borehole_fr = model.fragment(box_drilled, fractures_group, borehole)

    b_box_fr = box_fr.get_boundary()

    b_borehole_fr = b_box_fr.select_by_intersect(b_borehole).set_region(".borehole")
    b_block_fr = b_box_fr.select_by_intersect(b_block).set_region(".main_tunnel")
    sides2 = b_box_fr.cut(b_block_fr.copy(), b_borehole_fr.copy()).set_region(".sides")

    box_all = []
    # for name, side_tool in sides.items():
    #     isec = b_box_fr.select_by_intersect(side_tool)
    #     box_all.append(isec.modify_regions("." + name))

    box_all.extend([box_fr, b_borehole_fr, b_block_fr, sides2])

    """
    a = 2
    b = 1
    c = 10
    d = 5
    boundaries = gmsh.ObjectSet.get_boundary(borehole, False)
    uzly = []
    for i in range(boundaries.size):
        uzly.append(boundaries.dim_tags[i][1])
    f = field.threshold(field.distance_surfaces(uzly, 8), lower_bound=(a, b), upper_bound=(c, d))
    model.set_mesh_step_field(f)
    """

    fractures_fr.mesh_step(fracture_mesh_step)
    b_borehole_fr.mesh_step(boreholes_mesh_step)
    # b_box_fr.mesh_step(boundary_mesh_step)
    sides2.mesh_step(boundary_mesh_step)
    b_block_fr.mesh_step(main_tunnel_mesh_step)

    #sides2.mesh_step(30)

    #b_block_fr.mesh_step(main_tunnel_mesh_step)

    mesh_groups = [*box_all]
    model.keep_only(*mesh_groups)
    #model.remove_duplicate_entities()

    min_el_size = 0.1
    max_el_size = 30

    mesh = options.Mesh()
    # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
    # mesh.Algorithm = options.Algorithm2d.Delaunay
    # mesh.Algorithm = options.Algorithm2d.FrontalDelaunay
    # mesh.Algorithm3D = options.Algorithm3d.Frontal
    # mesh.Algorithm3D = options.Algorithm3d.Delaunay
    mesh.ToleranceInitialDelaunay = 0.01
    # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
    mesh.CharacteristicLengthFromPoints = True
    mesh.CharacteristicLengthFromCurvature = True
    mesh.CharacteristicLengthExtendFromBoundary = 2  # co se stane if 1
    mesh.CharacteristicLengthMin = min_el_size
    mesh.CharacteristicLengthMax = max_el_size
    mesh.MinimumCirclePoints = 6
    mesh.MinimumCurvePoints = 2

    model.write_brep()
    model.make_mesh(mesh_groups, dim=dim)
    model.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)

    del model

def apply_field5(config_dict, dim=3, tolerance=0.15, max_mismatch=5, mesh_name="field_mesh"):

    model = gmsh.GeometryOCC(mesh_name)

    geom = config_dict["geometry"]
    fracture_mesh_step = geom['fracture_mesh_step']
    dimensions = geom["box_dimensions"]
    boundary_mesh_step = geom['boundary_mesh_step']
    boreholes_mesh_step = geom['boreholes_mesh_step']
    main_tunnel_mesh_step = geom['main_tunnel_mesh_step']

    mtr = geom["main_tunnel_radius"]
    mtw = geom["main_tunnel_width"]
    mtl = geom["main_tunnel_length"]

    st_r = geom["lateral_tunnel_radius"]
    stw = geom["lateral_tunnel_width"]
    stl = geom["lateral_tunnel_length"]

    br = geom["borehole_radius"]
    bl = geom["borehole_length"]
    bd = geom["borehole_distance"]

    # factory = gmsh.GeometryOCC(mesh_name, verbose=True)
    gopt = options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001

    borehole2, borehole3, block = True, True, True
    shift = bd
    dimensions = [100, 100, 20]
    stl, bl, bd = 20, 25, 20

    #################################################################################

    box = model.box(dimensions).set_region("box")

    b_1 = model.cylinder(br, axis=[bl, 0, 0]).translate([stl - bl / 2, 1.5*bd, 0])

    if borehole2: b_2 = b_1.copy().translate([0, -1.5*bd, 0])
    if borehole3: b_3 = b_1.copy().translate([0, -3*bd, 0])
    if block: block1 = model.box([mtr, bl, mtr]).translate([bd, 0, 0])

    #################################################################################

    if borehole2 and borehole3:
        if block: rec = box.cut(b_1, b_2, b_3, block1)
        else: rec = box.cut(b_1, b_2, b_3)
    else:
        if borehole2: rec = box.cut(b_1, b_2)
        else: rec = box.cut(b_1)

    rec.set_region("box")
    b_rec = rec.get_boundary().set_region(".sides")

    #################################################################################

    b_tmp = b_1.get_boundary()
    bb_1 = b_rec.select_by_intersect(b_tmp).set_region(".inner_sides_1")

    a, b, c, d = 2, 0.4, 5, 10
    f = field.threshold(field.distance_surfaces(bb_1.tags, 2 * bl / (2 * b)), lower_bound=(a, b), upper_bound=(c, d))

    #################################################################################

    if borehole2:
        b_tmp = b_2.get_boundary()
        bb_2 = b_rec.select_by_intersect(b_tmp).set_region(".inner_sides_2")

        a, b, c, d = 1, 0.8, 5, 10
        f_rest = field.threshold(field.distance_surfaces(bb_2.tags, 2 * bl / (2 * b)), lower_bound=(a, b), upper_bound=(c, d))

    ####################################################

    if borehole3:
        b_tmp = b_3.get_boundary()
        bb_3 = b_rec.select_by_intersect(b_tmp).set_region(".inner_sides_3")

        a, b, c, d = 2, 1.5, 5, 10
        f_b_3b = field.threshold(field.distance_surfaces(bb_3.tags, 2 * bl / (2 * b)), lower_bound=(a, b),
                                 upper_bound=(c, d))

    ####################################################

    if block:
        b_tmp = block1.get_boundary()
        b_block = b_rec.select_by_intersect(b_tmp).set_region(".inner_sides_block")

        a, b, c, d = 1, 1, 5, 5
        f_block = field.threshold(field.distance_surfaces(b_block.tags, 2 * bl / (2 * b)), lower_bound=(a, b),
                                 upper_bound=(c, d))

    ####################################################

    if borehole2 and borehole3:
        if block:
            ff = field.minimum(f_rest, f, f_b_3b, f_block)
        else:
            ff = field.minimum(f_rest, f, f_b_3b)
    else:
        if borehole2:
            ff = field.minimum(f_rest, f)
        else:
            ff = f

    if borehole2 and borehole3:
        if block: sides2 = b_rec.cut(bb_1, bb_2, bb_3, b_block)
        else: sides2 = b_rec.cut(bb_1, bb_2, bb_3)
    else:
        if borehole2: sides2 = b_rec.cut(bb_1, bb_2)
        else: sides2 = b_rec.cut(bb_1)

    sides2.set_region(".sides")

    model.set_mesh_step_field(ff)

    # model.remove_duplicate_entities()
    model.write_brep()
    model.mesh_options.CharacteristicLengthMin = 0.1
    model.mesh_options.CharacteristicLengthMax = 30
    #model.make_mesh([rec, boundaries, bb, sides2], dim=dim)

    if borehole2 and borehole3:
        if block:
            model.make_mesh([rec, bb_1, bb_2, bb_3, b_block, sides2], dim=dim)
        else:
            model.make_mesh([rec, bb_1, bb_2, bb_3, sides2], dim=dim)
    else:
        if borehole2:
            model.make_mesh([rec, bb_1, bb_2, sides2], dim=dim)
        else:
            model.make_mesh([rec, bb_1, sides2], dim=dim)

    model.write_mesh("prd.msh2", gmsh.MeshFormat.msh2)

    del model

def apply_field6(config_dict, dim, tolerance, max_mismatch, mesh_name="thd6d"):

    #sample_dir = "output"
    #os.chdir(sample_dir)

    model = gmsh.GeometryOCC(mesh_name)

    geom = config_dict["geometry"]
    fracture_mesh_step = geom['fracture_mesh_step']
    dimensions = geom["box_dimensions"]
    boundary_mesh_step = geom['boundary_mesh_step']
    boreholes_mesh_step = geom['boreholes_mesh_step']
    main_tunnel_mesh_step = geom['main_tunnel_mesh_step']

    mtr = geom["main_tunnel_radius"]
    mtw = geom["main_tunnel_width"]
    mtl = geom["main_tunnel_length"]

    st_r = geom["lateral_tunnel_radius"]
    stw = geom["lateral_tunnel_width"]
    stl = geom["lateral_tunnel_length"]

    br = geom["borehole_radius"]
    bl = geom["borehole_length"]
    bd = geom["borehole_distance"]

    #factory = gmsh.GeometryOCC(mesh_name, verbose=True)
    gopt = options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001

    # Main box
    box = model.box(dimensions).set_region("box")
    side_z = model.rectangle([dimensions[0], dimensions[1]])
    side_y = model.rectangle([dimensions[0], dimensions[2]])
    side_x = model.rectangle([dimensions[2], dimensions[1]])
    sides = dict(
        side_z0=side_z.copy().translate([0, 0, -dimensions[2] / 2]),
        side_z1=side_z.copy().translate([0, 0, +dimensions[2] / 2]),
        side_y0=side_y.copy().translate([0, 0, -dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_y1=side_y.copy().translate([0, 0, +dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_x0=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        side_x1=side_x.copy().translate([0, 0, +dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.modify_regions(name)

    ###########################################################################################################

    # borehole
    b_1 = model.cylinder(br, axis=[bl, 0, 0]).translate([stl - bl / 2, 0, 0])
    b_2 = b_1.copy().translate([0, bd, 0])
    b_3 = b_1.copy().translate([0, -bd, 0])
    borehole = model.group(b_1, b_2, b_3)

    # main_tunnel = longitudinal + lateral parts
    main_tunnel_block = model.box([mtw, mtl, mtr])
    y_shift = math.sqrt(mtr * mtr - 0.25 * mtw * mtw) - mtr / 2
    main_tunnel_block_tmp = main_tunnel_block.copy().translate([0, 0, mtr])
    main_tunnel_cylinder_tmp = model.cylinder(mtr, axis=[0, mtl, 0]).translate([0, -mtl / 2, -y_shift])
    main_tunnel_cylinder = main_tunnel_block_tmp.intersect(main_tunnel_cylinder_tmp)

    y_shift_2 = math.sqrt(st_r * st_r - 0.25 * stw * stw) - st_r / 2
    small_tunnel_block = model.box([stl, stw, st_r])
    small_tunnel_block_tmp = small_tunnel_block.copy().translate([0, 0, st_r])
    small_tunnel_cylinder_tmp = model.cylinder(st_r, axis=[stl, 0, 0]) \
            .translate([-stl / 2, 0, -y_shift_2])
    small_tunnel_cylinder = small_tunnel_block_tmp.intersect(small_tunnel_cylinder_tmp)

    lateral_tunnel_part_1 = model.group(small_tunnel_block, small_tunnel_cylinder) \
            .translate([stl / 2, 0, 0])
    lateral_tunnel_part_2 = lateral_tunnel_part_1.copy().translate([0, bd, 0])
    lateral_tunnel_part_3 = lateral_tunnel_part_1.copy().translate([0, -bd, 0])

    block = main_tunnel_block.fuse(main_tunnel_cylinder, lateral_tunnel_part_1, lateral_tunnel_part_2,
                                       lateral_tunnel_part_3).translate([-bl / 2, 0, 0])

    ###########################################################################################################

    #b_borehole = borehole.get_boundary().set_region(".borehole")
    #b_block = block.get_boundary().set_region(".main_tunnel")

    ##fractures = create_fractures_rectangles(model, fractures, model.rectangle())
    ##fractures_group = model.group(*fractures)

    sides2 = box.get_boundary().set_region(".sides")
    box_drilled = box.cut(borehole.copy()).set_region(".box")

    ##fractures_group = fractures_group.intersect(box_drilled.copy())

    ##box_fr, fractures_fr, borehole_fr = model.fragment(box_drilled, fractures_group, borehole)

    ##b_box_fr = box_fr.get_boundary()

    ##b_borehole_fr = b_box_fr.select_by_intersect(b_borehole).set_region(".borehole")
    ##b_block_fr = b_box_fr.select_by_intersect(b_block).set_region(".main_tunnel")
    ##sides2 = b_box_fr.cut(b_block_fr.copy(), b_borehole_fr.copy()).set_region(".sides")

    #box_all = []
    ##box_all.extend([box_fr, b_borehole_fr, b_block_fr, sides2])
    #box_all.extend([box, b_borehole, b_block, sides2])

    a = 2
    b = 1
    c = 10
    d = 5

    boundaries = gmsh.ObjectSet.get_boundary(borehole, False)

    uzly = []
    for i in range(boundaries.size):
        uzly.append(boundaries.dim_tags[i][1])
    f = field.threshold(field.distance_surfaces(uzly, 8), lower_bound=(a, b), upper_bound=(c, d))

    model.set_mesh_step_field(f)

    """fractures_fr.mesh_step(fracture_mesh_step)
    b_borehole_fr.mesh_step(boreholes_mesh_step)
    # b_box_fr.mesh_step(boundary_mesh_step)
    sides2.mesh_step(boundary_mesh_step)
    b_block_fr.mesh_step(main_tunnel_mesh_step)"""

    #mesh_groups = [*box_all]
    ###mesh_groups = [box, b_borehole, b_block, sides2]
    ##model.keep_only(*mesh_groups)
    #model.remove_duplicate_entities()

    """mesh = options.Mesh()
    # mesh.Algorithm = options.Algorithm2d.MeshAdapt # produce some degenerated 2d elements on fracture boundaries ??
    # mesh.Algorithm = options.Algorithm2d.Delaunay
    # mesh.Algorithm = options.Algorithm2d.FrontalDelaunay
    # mesh.Algorithm3D = options.Algorithm3d.Frontal
    # mesh.Algorithm3D = options.Algorithm3d.Delaunay
    mesh.ToleranceInitialDelaunay = 0.01
    # mesh.ToleranceEdgeLength = fracture_mesh_step / 5
    mesh.CharacteristicLengthFromPoints = True
    mesh.CharacteristicLengthFromCurvature = True
    mesh.CharacteristicLengthExtendFromBoundary = 2  # co se stane if 1
    mesh.CharacteristicLengthMin = min_el_size
    mesh.CharacteristicLengthMax = max_el_size
    mesh.MinimumCirclePoints = 6
    mesh.MinimumCurvePoints = 2

    model.write_brep()
    model.make_mesh([box_drilled], dim=dim)
    model.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)"""

    model.write_brep()
    model.mesh_options.CharacteristicLengthMin = 1
    model.mesh_options.CharacteristicLengthMax = 100
    model.make_mesh([box_drilled, boundaries], dim=dim)
    model.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)

    del model

def apply_field7(config_dict, dim=3, tolerance=0.15, max_mismatch=5, mesh_name="field_mesh"):

    model = gmsh.GeometryOCC(mesh_name)

    geom = config_dict["geometry"]
    fracture_mesh_step = geom['fracture_mesh_step']
    dimensions = geom["box_dimensions"]
    boundary_mesh_step = geom['boundary_mesh_step']
    boreholes_mesh_step = geom['boreholes_mesh_step']
    main_tunnel_mesh_step = geom['main_tunnel_mesh_step']

    mtr = geom["main_tunnel_radius"]
    mtw = geom["main_tunnel_width"]
    mtl = geom["main_tunnel_length"]

    st_r = geom["lateral_tunnel_radius"]
    stw = geom["lateral_tunnel_width"]
    stl = geom["lateral_tunnel_length"]

    br = geom["borehole_radius"]
    bl = geom["borehole_length"]
    bd = geom["borehole_distance"]

    # factory = gmsh.GeometryOCC(mesh_name, verbose=True)
    gopt = options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001

    # Main box
    box = model.box(dimensions).set_region("box")
    side_z = model.rectangle([dimensions[0], dimensions[1]])
    side_y = model.rectangle([dimensions[0], dimensions[2]])
    side_x = model.rectangle([dimensions[2], dimensions[1]])
    sides = dict(
        side_z0=side_z.copy().translate([0, 0, -dimensions[2] / 2]),
        side_z1=side_z.copy().translate([0, 0, +dimensions[2] / 2]),
        side_y0=side_y.copy().translate([0, 0, -dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_y1=side_y.copy().translate([0, 0, +dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_x0=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        side_x1=side_x.copy().translate([0, 0, +dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.modify_regions(name)

    #####################################################################################################

        # borehole
        b_1 = model.cylinder(br, axis=[bl, 0, 0]).translate([stl - bl / 2, 0, 0])
        b_2 = b_1.copy().translate([0, bd, 0])
        b_3 = b_1.copy().translate([0, -bd, 0])
        borehole = model.group(b_1, b_2, b_3)

        # main_tunnel = longitudinal + lateral parts
        main_tunnel_block = model.box([mtw, mtl, mtr])
        y_shift = math.sqrt(mtr * mtr - 0.25 * mtw * mtw) - mtr / 2
        main_tunnel_block_tmp = main_tunnel_block.copy().translate([0, 0, mtr])
        main_tunnel_cylinder_tmp = model.cylinder(mtr, axis=[0, mtl, 0]).translate([0, -mtl / 2, -y_shift])
        main_tunnel_cylinder = main_tunnel_block_tmp.intersect(main_tunnel_cylinder_tmp)

        y_shift_2 = math.sqrt(st_r * st_r - 0.25 * stw * stw) - st_r / 2
        small_tunnel_block = model.box([stl, stw, st_r])
        small_tunnel_block_tmp = small_tunnel_block.copy().translate([0, 0, st_r])
        small_tunnel_cylinder_tmp = model.cylinder(st_r, axis=[stl, 0, 0]) \
            .translate([-stl / 2, 0, -y_shift_2])
        small_tunnel_cylinder = small_tunnel_block_tmp.intersect(small_tunnel_cylinder_tmp)

        lateral_tunnel_part_1 = model.group(small_tunnel_block, small_tunnel_cylinder) \
            .translate([stl / 2, 0, 0])
        lateral_tunnel_part_2 = lateral_tunnel_part_1.copy().translate([0, bd, 0])
        lateral_tunnel_part_3 = lateral_tunnel_part_1.copy().translate([0, -bd, 0])

        block = main_tunnel_block.fuse(main_tunnel_cylinder, lateral_tunnel_part_1, lateral_tunnel_part_2,
                                       lateral_tunnel_part_3).translate([-bl / 2, 0, 0])

    ###########################################################################################################

    rec = box.cut(borehole, block)
    rec.set_region("box")
    b_rec = rec.get_boundary().set_region(".sides")

    ###########################################################################################################

    b_tmp = borehole.get_boundary()
    b_borehole = b_rec.select_by_intersect(b_tmp).set_region(".inner_sides")

    a, b, c, d = 1, 0.5, 5, 15
    f_borehole = field.threshold(field.distance_surfaces(b_borehole.tags, 2 * bl / (2 * b)), lower_bound=(a, b), upper_bound=(c, d))

    #################################################################################

    b_tmp = block.get_boundary()
    b_block = b_rec.select_by_intersect(b_tmp).set_region(".tunnel_sides")

    a, b, c, d = 2, 3, 5, 15
    f_block = field.threshold(field.distance_surfaces(b_block.tags, 2 * bl / (2 * b)), lower_bound=(a, b),
                        upper_bound=(c, d))

    #################################################################################

    ff = field.minimum(f_borehole, f_block)

    sides2 = b_rec.cut(b_borehole, b_block).set_region(".sides")

    model.set_mesh_step_field(ff)

    # model.remove_duplicate_entities()
    model.write_brep()
    model.mesh_options.CharacteristicLengthMin = 0.1
    model.mesh_options.CharacteristicLengthMax = 30
    #model.make_mesh([rec, boundaries, bb, sides2], dim=dim)

    model.make_mesh([rec, b_borehole, b_block, sides2], dim=dim)

    model.write_mesh("geom_komplet.msh2", gmsh.MeshFormat.msh2)

    del model

def apply_field_pukliny(config_dict, fractures, dim=3, tolerance=0.15, max_mismatch=5, mesh_name="field_mesh"):

    model = gmsh.GeometryOCC(mesh_name)

    geom = config_dict["geometry"]
    fracture_mesh_step = geom['fracture_mesh_step']
    dimensions = geom["box_dimensions"]
    boundary_mesh_step = geom['boundary_mesh_step']
    boreholes_mesh_step = geom['boreholes_mesh_step']
    main_tunnel_mesh_step = geom['main_tunnel_mesh_step']

    mtr = geom["main_tunnel_radius"]
    mtw = geom["main_tunnel_width"]
    mtl = geom["main_tunnel_length"]

    st_r = geom["lateral_tunnel_radius"]
    stw = geom["lateral_tunnel_width"]
    stl = geom["lateral_tunnel_length"]

    br = geom["borehole_radius"]
    bl = geom["borehole_length"]
    bd = geom["borehole_distance"]

    # factory = gmsh.GeometryOCC(mesh_name, verbose=True)
    gopt = options.Geometry()
    gopt.Tolerance = 0.0001
    gopt.ToleranceBoolean = 0.001

    # Main box
    box = model.box(dimensions).set_region("box")
    side_z = model.rectangle([dimensions[0], dimensions[1]])
    side_y = model.rectangle([dimensions[0], dimensions[2]])
    side_x = model.rectangle([dimensions[2], dimensions[1]])
    sides = dict(
        side_z0=side_z.copy().translate([0, 0, -dimensions[2] / 2]),
        side_z1=side_z.copy().translate([0, 0, +dimensions[2] / 2]),
        side_y0=side_y.copy().translate([0, 0, -dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_y1=side_y.copy().translate([0, 0, +dimensions[1] / 2]).rotate([-1, 0, 0], np.pi / 2),
        side_x0=side_x.copy().translate([0, 0, -dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2),
        side_x1=side_x.copy().translate([0, 0, +dimensions[0] / 2]).rotate([0, 1, 0], np.pi / 2)
    )
    for name, side in sides.items():
        side.modify_regions(name)

    #####################################################################################################

        # borehole
        b_1 = model.cylinder(br, axis=[bl, 0, 0]).translate([stl - bl / 2, 0, 0])
        b_2 = b_1.copy().translate([0, bd, 0])
        b_3 = b_1.copy().translate([0, -bd, 0])
        borehole = model.group(b_1, b_2, b_3)

        # main_tunnel = longitudinal + lateral parts
        main_tunnel_block = model.box([mtw, mtl, mtr])
        y_shift = math.sqrt(mtr * mtr - 0.25 * mtw * mtw) - mtr / 2
        main_tunnel_block_tmp = main_tunnel_block.copy().translate([0, 0, mtr])
        main_tunnel_cylinder_tmp = model.cylinder(mtr, axis=[0, mtl, 0]).translate([0, -mtl / 2, -y_shift])
        main_tunnel_cylinder = main_tunnel_block_tmp.intersect(main_tunnel_cylinder_tmp)

        y_shift_2 = math.sqrt(st_r * st_r - 0.25 * stw * stw) - st_r / 2
        small_tunnel_block = model.box([stl, stw, st_r])
        small_tunnel_block_tmp = small_tunnel_block.copy().translate([0, 0, st_r])
        small_tunnel_cylinder_tmp = model.cylinder(st_r, axis=[stl, 0, 0]) \
            .translate([-stl / 2, 0, -y_shift_2])
        small_tunnel_cylinder = small_tunnel_block_tmp.intersect(small_tunnel_cylinder_tmp)

        lateral_tunnel_part_1 = model.group(small_tunnel_block, small_tunnel_cylinder) \
            .translate([stl / 2, 0, 0])
        lateral_tunnel_part_2 = lateral_tunnel_part_1.copy().translate([0, bd, 0])
        lateral_tunnel_part_3 = lateral_tunnel_part_1.copy().translate([0, -bd, 0])

        block = main_tunnel_block.fuse(main_tunnel_cylinder, lateral_tunnel_part_1, lateral_tunnel_part_2,
                                       lateral_tunnel_part_3).translate([-bl / 2, 0, 0])

    ###########################################################################################################

    fractures = create_fractures_rectangles(model, fractures, model.rectangle())
    fractures_group = model.group(*fractures)

    rec = box.cut(borehole, block)
    rec.set_region("box")
    b_rec = rec.get_boundary()#.set_region(".sides")

    fractures_group = fractures_group.intersect(rec)
    rec_fr, fractures_fr = model.fragment(rec, fractures_group)

    b_rec_fr = rec_fr.get_boundary()

    ###########################################################################################################

    #borehole_fr = borehole.fragment(fractures_group)
    b_tmp = borehole.get_boundary()
    b_borehole_fr = b_rec_fr.select_by_intersect(b_tmp).set_region(".borehole")

    a, b, c, d = 2, 3, 5, 15
    f_borehole = field.threshold(field.distance_surfaces(b_borehole_fr.tags, 2 * bl / (2 * b)), lower_bound=(a, b), upper_bound=(c, d))
    #f_borehole = field.threshold(field.distance_surfaces(b_rec_fr.tags, 2 * bl / (2 * b)), lower_bound=(a, b), upper_bound=(c, d))


    #################################################################################
    '''
    #block_fr = block.fragment(fractures_group)
    b_tmp = block.get_boundary()
    b_block_fr = b_rec_fr.select_by_intersect(b_tmp).set_region(".main_tunnel")

    a, b, c, d = 3, 5, 5, 15
    f_block = field.threshold(field.distance_surfaces(b_block_fr.tags, 2 * bl / (2 * b)), lower_bound=(a, b),
                        upper_bound=(c, d))

    #################################################################################

    ff = field.minimum(f_borehole, f_block)
    '''

    #sides2 = b_rec_fr.cut(b_borehole_fr, b_block_fr).set_region(".sides")
    sides2 = b_rec_fr.cut(b_borehole_fr).set_region(".sides")

    #ff = f_borehole
    ff = field.constant(10)

    model.set_mesh_step_field(ff)

    # model.remove_duplicate_entities()
    model.write_brep()
    model.mesh_options.CharacteristicLengthMin = 0.1
    model.mesh_options.CharacteristicLengthMax = 30
    #model.make_mesh([rec, boundaries, bb, sides2], dim=dim)

    #model.make_mesh([rec_fr, b_borehole_fr, b_block_fr, sides2], dim=dim)
    model.make_mesh([rec_fr, sides2, fractures_fr, b_borehole_fr], dim=dim)

    model.write_mesh("geom_pukliny.msh2", gmsh.MeshFormat.msh2)

    del model
