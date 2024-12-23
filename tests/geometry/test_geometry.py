from bgem.geometry.geometry import *
from bgem.polygons.polygons import PolygonDecomposition
from fixtures import sandbox_fname


def test_simple():
    # layer geometry object
    lg = LayerGeometry()

    # create interfaces
    top_iface = lg.add_interface(transform_z=(1.0, 0.0), elevation=0.0)
    bot_iface = lg.add_interface(transform_z=(1.0, 0.0), elevation=-1.0)

    # add region to layer geometry
    reg = lg.add_region(name="region_name", dim=RegionDim.bulk)

    # create decomposition
    decomp = PolygonDecomposition()
    pt1 = decomp.add_point((0.0, 0.0))
    pt2 = decomp.add_point((10.0, 0.0))
    pt3 = decomp.add_point((10.0, 10.0))
    pt4 = decomp.add_point((0.0, 10.0))
    decomp.add_line_for_points(pt1, pt2)
    decomp.add_line_for_points(pt2, pt3)
    decomp.add_line_for_points(pt3, pt4)
    decomp.add_line_for_points(pt4, pt1)
    res = decomp.get_last_polygon_changes()
    # assert res[0] == PolygonChange.add
    polygon = decomp.polygons[res[2]]
    decomp.set_attr(polygon, reg)

    # add layer to layer geometry
    lg.add_stratum_layer(decomp, top_iface, decomp, bot_iface)

    # generate mesh file
    lg.filename_base = sandbox_fname("mesh_simple", "")
    lg.init()

    lg.construct_brep_geometry()
    lg.make_gmsh_shape_dict()
    lg.distribute_mesh_step()

    lg.call_gmsh(mesh_step=0.0)
    lg.modify_mesh()


def test_fracture():
    # layer geometry object
    lg = LayerGeometry()

    # create interfaces
    top_iface = lg.add_interface(transform_z=(1.0, 0.0), elevation=0.0)
    fr_iface = lg.add_interface(transform_z=(1.0, 0.0), elevation=-1.0)
    bot_iface = lg.add_interface(transform_z=(1.0, 0.0), elevation=-2.0)

    # add region to layer geometry
    reg = lg.add_region(name="region_name", dim=RegionDim.bulk)

    # create decomposition
    decomp = PolygonDecomposition()
    pt1 = decomp.add_point((0.0, 0.0))
    pt2 = decomp.add_point((10.0, 0.0))
    pt3 = decomp.add_point((10.0, 10.0))
    pt4 = decomp.add_point((0.0, 10.0))
    decomp.add_line_for_points(pt1, pt2)
    decomp.add_line_for_points(pt2, pt3)
    decomp.add_line_for_points(pt3, pt4)
    decomp.add_line_for_points(pt4, pt1)
    res = decomp.get_last_polygon_changes()
    # assert res[0] == PolygonChange.add
    polygon = decomp.polygons[res[2]]
    decomp.set_attr(polygon, reg)

    # add region to layer geometry
    fr_reg = lg.add_region(name="fracture_region_name", dim=RegionDim.fracture)

    # create fracture decomposition
    fr_decomp = PolygonDecomposition()
    pt1 = fr_decomp.add_point((0.0, 0.0))
    pt2 = fr_decomp.add_point((10.0, 0.0))
    pt3 = fr_decomp.add_point((10.0, 10.0))
    fr_decomp.add_line_for_points(pt1, pt2)
    fr_decomp.add_line_for_points(pt2, pt3)
    fr_decomp.add_line_for_points(pt3, pt1)
    res = fr_decomp.get_last_polygon_changes()
    # assert res[0] == PolygonChange.add
    polygon = fr_decomp.polygons[res[2]]
    fr_decomp.set_attr(polygon, fr_reg)

    # add layer to layer geometry
    lg.add_stratum_layer(decomp, top_iface, decomp, fr_iface)
    lg.add_fracture_layer(fr_decomp, fr_iface)
    lg.add_stratum_layer(decomp, fr_iface, decomp, bot_iface)

    # generate mesh file
    lg.filename_base = sandbox_fname("mesh_fracture", "")
    lg.init()

    lg.construct_brep_geometry()
    lg.make_gmsh_shape_dict()
    lg.distribute_mesh_step()

    lg.call_gmsh(mesh_step=0.0)
    lg.modify_mesh()
