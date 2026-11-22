"""
Fracture set meshing support, should provide functions to crate a fractures shapes
using GMSH or BrepWriter.
- fracture set regularizations
- cration of GMSH geometry entities using gmsh api
- cration of BrepWriter entities
"""
import pathlib
import numpy as np
from typing import Union
from bgem.bspline import brep_writer as bw
from bgem.gmsh import gmsh


def create_fractures_rectangles(gmsh_geom, fractures, base_shape: 'ObjectSet'):
    """
    DEPRECATED, use geometry_gmsh instead.
    # From given fracture date list 'fractures'.
    # transform the base_shape to fracture objects
    # fragment fractures by their intersections
    # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
    """
    assert False, "DEPRECATED, use geometry_gmsh(gmsh_geom, frectures) instead."
    return None

    # shapes = []
    # for i, fr in enumerate(fractures):
    #     shape = base_shape.deepcopy()
    #     print("fr: ", i, "tag: ", shape.dim_tags)
    #     shape = shape.scale([fr.rx, fr.ry, 1]) \
    #         .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
    #         .translate(fr.center) \
    #         .set_region(fr.region)
    #
    #     shapes.append(shape)
    #
    # fracture_fragments = gmsh_geom.fragment(*shapes)
    # return fracture_fragments


# def create_fractures_polygons(gmsh_geom, fractures):
#     # From given fracture date list 'fractures'.
#     # transform the base_shape to fracture objects
#     # fragment fractures by their intersections
#     # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
#     frac_obj = fracture.Fractures(fractures)
#     frac_obj.snap_vertices_and_edges()
#     shapes = []
#     for fr, square in zip(fractures, frac_obj.squares):
#         shape = gmsh_geom.make_polygon(square).set_region(fr.region)
#         shapes.append(shape)
#
#     fracture_fragments = gmsh_geom.fragment(*shapes)
#     return fracture_fragments


def geometry_gmsh(fr_set, gmsh_geom: 'GeometryOCC'):
    """

    :param gmsh_geom:
    :param fractures:
    :param base_shape:
    :param shift:
    :return:
    """
    # From given fracture date list 'fractures'.
    # transform the base_shape to fracture objects
    # fragment fractures by their intersections
    # return dict: fracture.region -> GMSHobject with corresponding fracture fragments
    if len(fr_set) == 0:
        return []
    base_shape = fr_set.base_shape.gmsh_base_shape(gmsh_geom)
    shapes = []
    region_map = {}
    for i, fr in enumerate(fr_set):
        shape = base_shape.deepcopy()
        #print("fr: ", i, "tag: ", shape.dim_tags)
        region_name = f"fam_{fr.family}_{i:03d}"
        shape = shape.scale([fr.rx, fr.ry, 1]) \
            .rotate(axis=[0, 0, 1], angle=fr.shape_angle) \
            .rotate(axis=fr.rotation_axis, angle=fr.rotation_angle) \
            .translate(fr.center) \
            .set_region(region_name)
        region_map[region_name] = i
        shapes.append(shape)

    #fracture_fragments = gmsh_geom.fragment(*shapes)
    fr_shapes = gmsh_geom.group(*shapes)
    return fr_shapes, region_map


def geometry_brep_writer(fr_set, brep_name: Union[str, pathlib.Path]):
    """
    Create the BREP file from a list of fractures using the brep writer interface.

    Currently works only for 2D .
    """
    # fracture_mesh_step = geometry_dict['fracture_mesh_step']
    # dimensions = geometry_dict["box_dimensions"]

    #print("n fractures:", len(self))
    if isinstance(brep_name, str):
        brep_name = pathlib.Path(brep_name)
    brep_name = brep_name.with_suffix(".brep")
    faces = []
    base_vertices = fr_set.base_shape.vertices(8)

    # Legacy transform
    fr_vtxs = lambda fr : fr.transform(base_vertices) # fr.center
    fractures_vertices = np.array([fr_vtxs(fr) for fr in fr_set])

    #fractures_vertices = self.transform_mat @ (base_vertices.T)[None, :, :]   # (n_fr, 3, 3) @ (1, 3, n_points) -> (n_fr, 3, n_points)
    #fractures_vertices = fractures_vertices.transpose((0, 2, 1))
    #fractures_vertices = fractures_vertices + self.center[:, None, :] # (n_fr, 3, n_points) -> (n_fr, n_points, 3)


    for i, fr_vertices in enumerate(fractures_vertices):
        vtxs = [bw.Vertex(p) for p in fr_vertices]
        edges = [bw.Edge(a, b) for a, b in zip(vtxs[:-1], vtxs[1:])]
        edges.append(bw.Edge(vtxs[-1], vtxs[0]))
        face = bw.Face(edges)
        faces.append(face)

    comp = bw.Compound(faces)
    with open(brep_name, "w") as f:
        bw.write_model(f, comp)
    return brep_name
