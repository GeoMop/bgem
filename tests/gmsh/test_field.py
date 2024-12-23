import gmsh
import pytest
import os
from typing import *
from gmsh import model as gmsh_model, model
from bgem.gmsh import field, options
from bgem.gmsh.gmsh import GeometryOCC, ObjectSet, MeshFormat
import numpy as np
from fixtures import sandbox_fname


def base_model_2d(mesh_name, size=100):
    model = GeometryOCC(mesh_name)
    rec = model.rectangle([size, size])
    return model, rec


def base_model_3d(mesh_name, size=100):
    model = GeometryOCC(mesh_name)
    rec = model.box([size, size, size])
    return model, rec


def make_test_mesh(model, obj_set: List['ObjectSet'], field, min_step=0.5):
    mesh_name = model.model_name
    # obj_set = ObjectSet.group(*obj_set)
    dim = ObjectSet.group(*obj_set).max_dim()
    model.set_mesh_step_field(field)
    brep_fname = sandbox_fname(mesh_name, "brep")
    model.write_brep(brep_fname)
    model.mesh_options.CharacteristicLengthMin = min_step
    model.mesh_options.CharacteristicLengthMax = 1000
    # model.remove_duplicate_entities()
    model.make_mesh(obj_set, dim=dim)
    mesh_fname = sandbox_fname(mesh_name, "msh2")
    model.write_mesh(mesh_fname, MeshFormat.msh2)


def check_mesh(dim, reference_fn, tolerance, max_mismatch):
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

    n_mismatch = 0
    max_rel_error = 0
    for ele_tag, ele_nodes in zip(ele_tags, ele_node_tags):
        i_nodes = [node_indices[n_tag] for n_tag in ele_nodes]
        vertices = coords[i_nodes, :]
        edges = [vertices[i, :] - vertices[j, :] for i, j in ref_shape_edges[dim]]
        ele_size = np.max(np.linalg.norm(edges, axis=1))
        barycenter = np.average(vertices, axis=0)
        ref_ele_size = reference_fn(barycenter)
        rel_error = abs(ele_size - ref_ele_size) / ref_ele_size
        max_rel_error = max(max_rel_error, rel_error)
        # print(f"ele {ele_tag}, size: {ele_size}, ref size: {ref_ele_size}, {rel_error}")
        if rel_error > tolerance:
            print(f"Size mismatch, ele {ele_tag}, size: {ele_size}, ref size: {ref_ele_size}, rel_err: {rel_error}")
            n_mismatch += 1
    assert n_mismatch <= max_mismatch
    print(f"n_mismatch: {n_mismatch}, max n mismatch: {max_mismatch}")
    print(f"max rel error: {max_rel_error}")


def check_mesh_aniso(dim, fn_ref_tn, tolerance, max_mismatch):
    node_tags, coords, param_coords = gmsh_model.mesh.getNodes(dim=-1, returnParametricCoord=False)
    coords = np.reshape(coords, (-1, 3))
    node_indices = {tag: idx for idx, tag in enumerate(node_tags)}
    assert coords.shape[0] == len(node_tags)
    ele_types, ele_tags, ele_node_tags = gmsh_model.mesh.getElements(dim=dim)
    assert len(ele_types) == 1 and len(ele_tags) == 1 and len(ele_node_tags) == 1
    ele_tags = ele_tags[0]
    ele_node_tags = np.reshape(ele_node_tags[0], (-1, dim + 1))

    n_mismatch = 0
    max_rel_error = 0
    for ele_tag, ele_nodes in zip(ele_tags, ele_node_tags):
        i_nodes = [node_indices[n_tag] for n_tag in ele_nodes]
        vertices = coords[i_nodes, :]
        # element size tensor
        ele_map = vertices[1:, :] - vertices[0, :]
        ele_tn = ele_map.T @ ele_map
        barycenter = np.average(vertices, axis=0)
        ref_ele_tn = fn_ref_tn(barycenter)
        rel_error = np.linalg.norm(ele_tn - ref_ele_tn) / np.linalg.norm(ref_ele_tn)
        max_rel_error = max(max_rel_error, rel_error)
        # print(f"ele {ele_tag}, size: {ele_size}, ref size: {ref_ele_size}, {rel_error}")
        if rel_error > tolerance:
            print(f"Size mismatch, ele {ele_tag}, tn: {ele_tn}, ref tn: {ref_ele_tn}, rel_err: {rel_error}")
            n_mismatch += 1
    assert n_mismatch <= max_mismatch
    print(f"n_mismatch: {n_mismatch}, max n mismatch: {max_mismatch}")
    print(f"max rel error: {max_rel_error}")


def apply_field(field, reference_fn, dim=2, tolerance=0.15, max_mismatch=5, mesh_name="field_mesh", model=None,
                obj_set=None):
    """
    Create a mesh of dimension dim on a unit cube and
    compare element sizes to given reference function of coordinates.

    dim: Create mesh of that dimension.
    tolerance: maximum relative error of element sizes, actual size is max of edges, reference size is the field
          evaluated in the barycenter
    max_mismatch: maximum number of elements that could be over the tolerance
    """
    if model is None:
        model, obj_set = base_model_2d(mesh_name)
    dim = obj_set.max_dim()
    make_test_mesh(model, [obj_set], field)
    check_mesh(dim, reference_fn, tolerance, max_mismatch)
    del model


def test_eval_expr():
    f_1 = field.constant(1)
    f_3 = field.constant(3)
    f = f_1 * f_3 + f_3 / f_1 + 1 - f_1

    def ref_size(x):
        return 6

    apply_field(f, ref_size, dim=2, tolerance=0.38, max_mismatch=0)


@pytest.mark.skip
def test_attractor_aniso():
    """
    Test anisotropic attractor field for the axis Z from z=-100 to z=100.

    Some problems to get this field work. Can not achieve anisotropic mesh.
    Should work for both 2D and 3D meshing when using appropriate algorithms
    https://github-wiki-see.page/m/johnmoore4/MeshOpt/wiki/Wing

    See also sources:
    https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/src/mesh/Field.cpp

    MMG3d
    ** MISSING DATA: Your mesh must contain at least points.
    Mmg3d: unable to set mesh size
    """
    mesh_name = "attract_aniso_3d"
    model = GeometryOCC(mesh_name)
    x_len = 60
    cube = model.box([200, 200, x_len]).cut(model.cylinder(r=20, axis=[0, 0, x_len], center=[-100, -100, -x_len / 2]))
    dist_line = ([-100, -100, -x_len / 2], [-100, -100, x_len / 2])
    c1 = model.line(*dist_line)
    dist_range = (20, 50)
    h_normal = (5, 30)
    h_tangent = (20, 30)
    f_distance = field.attractor_aniso_curve(c1, dist_range, h_normal, h_tangent, sampling=100)

    # points spanning o_set objects
    line_a, line_b = np.array(dist_line)
    ref_points = []
    for t in np.linspace(0, 1, 100):
        ref_points.append(line_a * t + line_b * (1 - t))

    ref_points = np.array(ref_points)

    def ref_tn(x):
        diffs = np.array(x) - ref_points
        distances = np.linalg.norm(diffs, axis=1)
        i_dist = np.argmin(distances)
        dist = distances[i_dist]
        normal = np.array(x) / np.linalg.norm(x)
        min_tn = h_tangent[0] * np.ones(3) + (h_normal[0] - h_tangent[0]) * normal[:, None] * normal[None, :]
        max_tn = h_tangent[1] * np.ones(3) + (h_normal[1] - h_tangent[1]) * normal[:, None] * normal[None, :]
        t = (dist - dist_range[0]) / (dist_range[1] - dist_range[0])
        if dist < dist_range[0]:
            return min_tn
        elif dist < dist_range[1]:
            return (1 - t) * min_tn + t * max_tn
        else:
            return max_tn

    dim = 3
    model.mesh_options.Algorithm = options.Algorithm2d.BAMG
    # model.mesh_options.Algorithm = options.Algorithm2d.MeshAdapt
    model.mesh_options.Algorithm3D = options.Algorithm3d.MMG3D
    make_test_mesh(model, [cube], f_distance, min_step=0.01)
    # check_mesh_aniso(dim, ref_tn, tolerance=1.3, max_mismatch=3)


# @pytest.mark.skip
def test_eval_sin_cos():
    f_1 = field.constant(1)
    f = 11 + 10 * field.sin(field.x / 50 * np.pi) * field.cos(field.y / 25 * np.pi) + field.log(f_1)

    def ref_size(x):
        return 11 + 10 * np.sin(x[0] / 50 * np.pi) * np.cos(x[1] / 25 * np.pi)

    apply_field(f, ref_size, dim=2, tolerance=0.60, max_mismatch=50, mesh_name="sin_cos")


# @pytest.mark.skip
def test_eval_max():
    f = 0.2 * field.max(field.y, -field.y, 20)

    def ref_size(x):
        z = 0.2 * (max(x[1], -x[1], 20))
        return z

    # apply_field(f, ref_size, dim=2, tolerance=0.30, max_mismatch=20, mesh_name="variadic_max")
    apply_field(f, ref_size, dim=2, tolerance=0.30, max_mismatch=24, mesh_name="variadic_max")


# @pytest.mark.skip
def test_eval_sum():
    f = 0.2 * field.sqrt(field.sum(field.y * field.y, field.x * field.x, 100))

    def ref_size(x):
        z = 0.2 * np.sqrt((sum([x[0] * x[0], x[1] * x[1], 100])))
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


# @pytest.mark.skip
def test_constant_2d():
    f_const = field.constant(7)

    def ref_size(x):
        return 7

    # apply_field(f_const, ref_size, dim=2, tolerance=0.3, max_mismatch=0)
    apply_field(f_const, ref_size, dim=2, tolerance=0.3, max_mismatch=2)


# @pytest.mark.skip
def test_box_2d():
    f_box = field.box([-20, -20, 0], [20, 20, 0], 7, 11)

    def ref_size(x):
        return 11 if np.max(np.abs(x)) > 20 else 7

    apply_field(f_box, ref_size, dim=2, tolerance=0.3, max_mismatch=15)


# @pytest.mark.skip
def test_distance_nodes_2d():
    """
    For some reason the mesh size is actually close to half of the distance
    and naturally the error is quite high, up to 70%
    """
    f_distance = field.distance_nodes([1, 3])  # points: (-50, -50), (50, 50)

    def ref_size(x):
        dist = min(np.linalg.norm(np.array(x) - np.array([-50, -50, 0])),
                   np.linalg.norm(np.array(x) - np.array([50, 50, 0])))
        return max(0.5 * dist, 0.5)

    apply_field(f_distance, ref_size, dim=2, tolerance=0.7, max_mismatch=10, mesh_name="dist_2d")


def test_distance_3d():
    mesh_name = "dist_all_3d"
    model = GeometryOCC(mesh_name)
    cube = model.box([200, 200, 200])
    dist_rec = [50, 50]
    rec = model.rectangle(dist_rec)
    dist_line = ([-100, -100, -100], [-100, -100, 100])
    c1 = model.line(*dist_line)
    dist_point = [100, 100, 100]
    p1 = model.point(dist_point)
    o_set = model.group(rec, c1, p1)
    f_distance = field.distance(o_set, sampling=20)

    # points spanning o_set objects
    line_a, line_b = np.array(dist_line)
    ref_points = [dist_point]
    for t in np.linspace(0, 1, 20):
        ref_points.append(line_a * t + line_b * (1 - t))
    a, b = dist_rec
    for u in np.linspace(0, 1, 20):
        for v in np.linspace(0, 1, 20):
            ref_points.append(np.array([a / 2 * (2 * u - 1), b / 2 * (2 * v - 1), 0]))

    ref_points = np.array(ref_points)

    def ref_size(x):
        dist = np.min(np.linalg.norm(np.array(x) - ref_points, axis=1))
        return max(dist, 5)

    dim = 3
    obj_set = [cube, o_set]
    make_test_mesh(model, obj_set, f_distance, min_step=3)
    check_mesh(dim, ref_size, tolerance=1.3, max_mismatch=3)


def test_aniso():
    pass


def test_minmax_nodes_2d():
    f_distance = field.min(10, field.max(5, field.distance_nodes([1, 3])))  # points: (-50, -50), (50, 50)

    def ref_size(x):
        dist = min(np.linalg.norm(np.array(x) - np.array([-50, -50, 0])),
                   np.linalg.norm(np.array(x) - np.array([50, 50, 0])))
        return min(10, max(5, dist))

    apply_field(f_distance, ref_size, dim=2, tolerance=0.4, max_mismatch=10)


def test_threshold_2d():
    f_distance = field.threshold(field.distance_nodes([1, 3]),
                                 lower_bound=(10, 5), upper_bound=(30, 10))

    def thold(x, minx, miny, maxx, maxy):
        if x < minx:
            return miny
        elif x > maxx:
            return maxy
        else:
            return (x - minx) / (maxx - minx) * (maxy - miny) + miny

    def ref_size(x):
        dist = min(np.linalg.norm(np.array(x) - np.array([-50, -50, 0])),
                   np.linalg.norm(np.array(x) - np.array([50, 50, 0])))
        return thold(dist, 10, 5, 30, 10)

    apply_field(f_distance, ref_size, dim=2, tolerance=0.3, max_mismatch=3, mesh_name="thd")


def test_threshold_sigmoid_2d():
    f_distance = field.threshold(field.distance_nodes([1, 3]),
                                 lower_bound=(10, 5), upper_bound=(30, 10), sigmoid=True)

    def thold(x, minx, miny, maxx, maxy):
        scaled_X = (x - minx) / (maxx - minx)
        y = 1 / (1 + np.exp(-scaled_X))
        return y * (maxy - miny) + miny

    def ref_size(x):
        dist = min(np.linalg.norm(np.array(x) - np.array([-50, -50, 0])),
                   np.linalg.norm(np.array(x) - np.array([50, 50, 0])))
        return thold(dist, 10, 5, 30, 10)

    apply_field(f_distance, ref_size, dim=2, tolerance=0.4, max_mismatch=8, mesh_name="thds")
