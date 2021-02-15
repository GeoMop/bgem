import pytest
from gmsh import model as gmsh_model
from bgem.gmsh import gmsh, field, options
import numpy as np



def apply_field(field, reference_fn, dim=2, tolerance=0.15, max_mismatch=5):
    """
    Create a mesh of dimension dim on a unit cube and
    compare element sizes to given reference funcion of coordinates.
    """
    mesh_name = "test_mesh"
    model = gmsh.GeometryOCC(mesh_name)
    rec = model.rectangle([100, 100])
    model.set_mesh_step_field(field)
    model.write_brep()
    model.mesh_options.CharacteristicLengthMin = 0.1
    model.mesh_options.CharacteristicLengthMax = 100
    model.make_mesh([rec], dim=dim)
    model.write_mesh(mesh_name + ".msh2", gmsh.MeshFormat.msh2)

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
        return max(0.5 * dist, 0.1)
    apply_field(f_distance, ref_size, dim=2, tolerance=0.7, max_mismatch=10)


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
    apply_field(f_distance, ref_size, dim=2, tolerance=0.3, max_mismatch=3)
