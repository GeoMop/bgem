import pytest
import os
from gmsh import model as gmsh_model
from bgem.gmsh import gmsh, field, options
import numpy as np
from fixtures import sandbox_fname




def apply_field(field, reference_fn, dim=2, tolerance=0.15, max_mismatch=5, mesh_name="field_mesh"):
    """
    Create a mesh of dimension dim on a unit cube and
    compare element sizes to given reference funcion of coordinates.

    dim: Create mesh of that dimension.
    tolerance: maximum relative error of element sizes, actual size is max of edges, reference size is the field
          evaluated in the barycenter
    max_mismatch: maximum number of elements that could be over the tolerance
    """
    model = gmsh.GeometryOCC(mesh_name)
    rec = model.rectangle([100, 100])
    model.set_mesh_step_field(field)
    brep_fname = sandbox_fname(mesh_name, "brep")
    model.write_brep(brep_fname)
    model.mesh_options.CharacteristicLengthMin = 0.5
    model.mesh_options.CharacteristicLengthMax = 100
    model.make_mesh([rec], dim=dim)
    mesh_fname = sandbox_fname(mesh_name, "msh2")
    model.write_mesh(mesh_fname, gmsh.MeshFormat.msh2)

    # check
    ref_shape_edges = {
        1: [(0,1)],
        2: [(0,1), (0, 2), (1,2)],
        3: [(0, 1), (0, 2), (0,3), (1,2), (1,3), (2,3)],
    }

    node_tags, coords, param_coords = gmsh_model.mesh.getNodes(dim=-1, returnParametricCoord=False)
    coords = np.reshape(coords, (-1, 3))
    node_indices = {tag:idx for idx, tag in enumerate(node_tags)}
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
        edges = [vertices[i,:] - vertices[j,:] for i,j in ref_shape_edges[dim]]
        ele_size = np.max(np.linalg.norm(edges, axis=1))
        barycenter = np.average(vertices, axis=0)
        ref_ele_size = reference_fn(barycenter)
        rel_error = abs(ele_size - ref_ele_size) / ref_ele_size
        max_rel_error = max(max_rel_error, rel_error)
        #print(f"ele {ele_tag}, size: {ele_size}, ref size: {ref_ele_size}, {rel_error}")
        if rel_error > tolerance:
            print(f"Size mismatch, ele {ele_tag}, size: {ele_size}, ref size: {ref_ele_size}, rel_err: {rel_error}")
            n_mismatch += 1
    assert n_mismatch <= max_mismatch
    print(f"n_mismatch: {n_mismatch}, max n mismatch: {max_mismatch}")
    print(f"max rel error: {max_rel_error}")
    del model

def test_eval_expr():
    f_1 = field.constant(1)
    f_3 = field.constant(3)
    f = f_1*f_3 + f_3/f_1 + 1 - f_1
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
    f_const = field.constant(7)
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


def test_minmax_nodes_2d():
    f_distance = field.min(10, field.max(5, field.distance_nodes([1, 3]))) # points: (-50, -50), (50, 50)
    def ref_size(x):
        dist = min(np.linalg.norm(np.array(x) - np.array([-50,-50,0])),
                   np.linalg.norm(np.array(x) - np.array([50,50,0])))
        return min(10, max(5, dist))
    apply_field(f_distance, ref_size, dim=2, tolerance=0.4, max_mismatch=10)


def test_threshold_2d():
    f_distance = field.threshold(field.distance_nodes([1, 3]),
                                 lower_bound=(10,5), upper_bound=(30, 10))

    def thold(x,minx,miny,maxx,maxy):
        if x < minx:
            return miny
        elif x > maxx:
            return maxy
        else:
            return (x - minx)/(maxx - minx) * (maxy - miny) + miny

    def ref_size(x):
        dist = min(np.linalg.norm(np.array(x) - np.array([-50,-50,0])),
                   np.linalg.norm(np.array(x) - np.array([50,50,0])))
        return thold(dist, 10, 5, 30, 10)
    apply_field(f_distance, ref_size, dim=2, tolerance=0.3, max_mismatch=3, mesh_name = "thd")


def test_threshold_sigmoid_2d():
    f_distance = field.threshold(field.distance_nodes([1, 3]),
                                 lower_bound=(10,5), upper_bound=(30, 10), sigmoid=True)

    def thold(x,minx,miny,maxx,maxy):
        scaled_X = (x - minx)/(maxx - minx)
        y = 1/(1 + np.exp(-scaled_X))
        return  y * (maxy - miny) + miny

    def ref_size(x):
        dist = min(np.linalg.norm(np.array(x) - np.array([-50,-50,0])),
                   np.linalg.norm(np.array(x) - np.array([50,50,0])))
        return thold(dist, 10, 5, 30, 10)
    apply_field(f_distance, ref_size, dim=2, tolerance=0.4, max_mismatch=8, mesh_name = "thds")


